#!/usr/bin/env python3
"""Evaluate UIU-BEC ionic piezo predictor on JARVIS DFPT JSONL records."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers as ase_atomic_numbers
from ase.optimize import FIRE
from mace.calculators import MACECalculator

from build_uiu_bec_ionic_piezo import (
    COMPONENTS,
    CONV_E_PER_A2_TO_C_PER_M2,
    MIN_VOLUME,
    STRAIN_MODES,
    apply_voigt_strain,
    compute_bec,
    corrcoef,
    internal_displacement,
    metrics_for_vectors,
    rankdata,
    scalar_stats,
    tensor_to_row,
    write_tsv,
)


warnings.filterwarnings(
    "ignore",
    message=".*The TorchScript type system doesn't support instance-level annotations on empty non-base types.*",
)
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--input",
        default="results/jarvis_dfpt_reextracted_clean20/splits/clean20_test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/uiu_bec_ionic_piezo/jarvis_clean20_test_uiu_bec_ionic.tsv",
    )
    parser.add_argument("--summary", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--default_dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--strain", type=float, default=0.01)
    parser.add_argument("--max_structures", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--relax_base", action="store_true")
    parser.add_argument("--no_relax_strained", action="store_true")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--relax_steps_base", type=int, default=200)
    parser.add_argument("--relax_steps_strained", type=int, default=20)
    parser.add_argument("--parity", type=float, default=1.0, choices=[-1.0, 1.0])
    parser.add_argument("--bec_scale", type=float, default=1.0)
    parser.add_argument("--transpose_bec", action="store_true")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def relax_structure(atoms, calc, fmax, steps):
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)
    atoms.calc = None


def atoms_from_record(record):
    structure = record["structure"]
    cell = np.asarray(structure["lattice"]["matrix"], dtype=float)
    symbols = []
    positions = []
    for site in structure["sites"]:
        species = site["species"]
        if len(species) != 1:
            raise ValueError("mixed-occupancy JARVIS site is not supported")
        symbols.append(species[0]["element"])
        positions.append(site["xyz"])
    atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float), cell=cell, pbc=True)
    atoms.info["jid"] = record.get("jid", "")
    atoms.info["source_dir"] = record.get("source_dir", "")
    return atoms


def tensor_from_record(record, key):
    values = record.get(key)
    if values is None:
        raise ValueError(f"missing {key}")
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3, 6):
        raise ValueError(f"{key} shape {arr.shape}; expected (3, 6)")
    return arr


def load_records(path, start, max_structures):
    records = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if max_structures is not None and len(records) >= max_structures:
                break
            record = json.loads(line)
            record["_index"] = index
            records.append(record)
    return records


def compute_row(record, calc, device, args):
    atoms = atoms_from_record(record)
    max_z = max(ase_atomic_numbers[s] for s in atoms.get_chemical_symbols())
    if max_z >= 95:
        raise ValueError(f"contains unsupported element Z={max_z}")

    base_atoms = atoms.copy()
    if args.relax_base:
        relax_structure(base_atoms, calc, args.fmax, args.relax_steps_base)

    volume = float(base_atoms.get_volume())
    if not np.isfinite(volume) or volume <= MIN_VOLUME:
        raise ValueError(f"invalid volume: {volume}")

    bec, alpha_abs_mean = compute_bec(
        base_atoms,
        calc,
        device,
        parity=args.parity,
        bec_scale=args.bec_scale,
        transpose_bec=args.transpose_bec,
    )

    pred_ionic = np.zeros((3, 6), dtype=float)
    row = {
        "index": record["_index"],
        "jid": record.get("jid", ""),
        "formula": atoms.get_chemical_formula(),
        "natoms": len(base_atoms),
        "volume": volume,
        "parity": args.parity,
        "bec_scale": args.bec_scale,
        "transpose_bec": int(args.transpose_bec),
        "alpha_abs_mean": alpha_abs_mean,
    }
    row.update(scalar_stats("bec", bec))

    relax_strained = not args.no_relax_strained
    for mode_idx, (mode_name, _, _) in enumerate(STRAIN_MODES):
        plus_affine = apply_voigt_strain(base_atoms, mode_idx, args.strain)
        plus_relaxed = plus_affine.copy()
        if relax_strained:
            relax_structure(plus_relaxed, calc, args.fmax, args.relax_steps_strained)

        minus_affine = apply_voigt_strain(base_atoms, mode_idx, -args.strain)
        minus_relaxed = minus_affine.copy()
        if relax_strained:
            relax_structure(minus_relaxed, calc, args.fmax, args.relax_steps_strained)

        u_plus = internal_displacement(plus_affine, plus_relaxed)
        u_minus = internal_displacement(minus_affine, minus_relaxed)
        du_deta = (u_plus - u_minus) / (2.0 * args.strain)

        pred_ionic[:, mode_idx] = (
            CONV_E_PER_A2_TO_C_PER_M2
            / volume
            * np.einsum("aik,ak->i", bec, du_deta)
        )

        row[f"{mode_name}_u_plus_rms"] = float(np.sqrt(np.mean(u_plus**2)))
        row[f"{mode_name}_u_minus_rms"] = float(np.sqrt(np.mean(u_minus**2)))
        row[f"{mode_name}_du_deta_rms"] = float(np.sqrt(np.mean(du_deta**2)))
        row[f"{mode_name}_du_deta_abs_max"] = float(np.max(np.abs(du_deta)))

    row.update(tensor_to_row("uiu_bec_ionic", pred_ionic))
    row.update(tensor_to_row("ionic", tensor_from_record(record, "piezoelectric_tensor_ionic_Cm2")))
    row.update(tensor_to_row("static", tensor_from_record(record, "piezoelectric_tensor_clamped_Cm2")))
    row.update(tensor_to_row("total", tensor_from_record(record, "piezoelectric_tensor_Cm2")))

    base_atoms.calc = None
    if not all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
        raise ValueError("non-finite value in output row")
    return row


def summarize(rows):
    pred = np.asarray(
        [[row[f"uiu_bec_ionic_{comp}"] for comp in COMPONENTS] for row in rows],
        dtype=float,
    )
    refs = {
        "ionic_tensor": np.asarray(
            [[row[f"ionic_{comp}"] for comp in COMPONENTS] for row in rows], dtype=float
        ),
        "static_tensor": np.asarray(
            [[row[f"static_{comp}"] for comp in COMPONENTS] for row in rows], dtype=float
        ),
        "total_tensor": np.asarray(
            [[row[f"total_{comp}"] for comp in COMPONENTS] for row in rows], dtype=float
        ),
    }
    metric_rows = []
    for name, ref in refs.items():
        metric_rows.append({"target": name, "variant": "raw", **metrics_for_vectors(ref, pred)})
        metric_rows.append({"target": name, "variant": "negated", **metrics_for_vectors(ref, -pred)})
        flat_ref = ref.reshape(-1)
        flat_pred = pred.reshape(-1)
        denom = float(np.dot(flat_pred, flat_pred))
        scale = float(np.dot(flat_ref, flat_pred) / denom) if denom > 0 else math.nan
        scaled = pred * scale if np.isfinite(scale) else pred * math.nan
        metric_rows.append(
            {"target": name, "variant": f"best_scalar={scale:.6g}", **metrics_for_vectors(ref, scaled)}
        )

    for name, ref in refs.items():
        label = name.replace("_tensor", "_eijmax_abs")
        metric_rows.append(
            {
                "target": label,
                "variant": "absmax",
                **metrics_for_vectors(np.max(np.abs(ref), axis=1), np.max(np.abs(pred), axis=1)),
            }
        )
        label = name.replace("_tensor", "_norm")
        metric_rows.append(
            {
                "target": label,
                "variant": "frobenius",
                **metrics_for_vectors(np.linalg.norm(ref, axis=1), np.linalg.norm(pred, axis=1)),
            }
        )
    return metric_rows


def write_metrics(path, metric_rows, args, n_ok, n_error):
    columns = [
        "target",
        "variant",
        "n",
        "mae",
        "rmse",
        "bias",
        "pearson",
        "spearman",
        "ref_abs_mean",
        "pred_abs_mean",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write(f"# model\t{args.model}\n")
        handle.write(f"# input\t{args.input}\n")
        handle.write(f"# n_ok\t{n_ok}\n")
        handle.write(f"# n_error\t{n_error}\n")
        handle.write(f"# strain\t{args.strain}\n")
        handle.write(f"# relax_base\t{int(args.relax_base)}\n")
        handle.write(f"# relax_strained\t{int(not args.no_relax_strained)}\n")
        handle.write(f"# parity\t{args.parity}\n")
        handle.write(f"# bec_scale\t{args.bec_scale}\n")
        handle.write(f"# transpose_bec\t{int(args.transpose_bec)}\n")
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(metric_rows)


def main():
    args = parse_args()
    if args.default_dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    device = resolve_device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".metrics.tsv")

    records = load_records(args.input, args.start, args.max_structures)
    calc = MACECalculator(model_paths=args.model, device=device, default_dtype=args.default_dtype)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    iterator = tqdm(records, desc="JARVIS UIU BEC ionic piezo") if tqdm else records

    rows = []
    errors = []
    for record in iterator:
        try:
            rows.append(compute_row(record, calc, device, args))
        except Exception as exc:
            errors.append({"index": record.get("_index"), "jid": record.get("jid", ""), "error": str(exc)})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    write_tsv(output_path, rows)
    metric_rows = summarize(rows) if rows else []
    write_metrics(summary_path, metric_rows, args, len(rows), len(errors))
    if errors:
        error_path = output_path.with_suffix(".errors.tsv")
        with error_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "jid", "error"], delimiter="\t")
            writer.writeheader()
            writer.writerows(errors)

    print(f"Wrote rows: {output_path}")
    print(f"Wrote metrics: {summary_path}")
    if errors:
        print(f"Wrote errors: {output_path.with_suffix('.errors.tsv')}")


if __name__ == "__main__":
    main()
