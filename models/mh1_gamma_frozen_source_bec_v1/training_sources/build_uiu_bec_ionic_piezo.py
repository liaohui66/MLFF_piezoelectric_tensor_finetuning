#!/usr/bin/env python3
"""Estimate ionic piezo tensors from MACELES-uiu BEC outputs.

This script intentionally uses the model's ``compute_bec=True`` path instead of
manually assembling a polarization from latent charges and dipoles. The target
for this diagnostic is ``REF_piezoelectric_ionic`` because Born effective
charges combined with internal strain response correspond to the ionic part of
the piezo tensor, not the clamped-ion/static part.
"""

import argparse
import csv
import gc
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from ase.data import atomic_numbers as ase_atomic_numbers
from ase.io import read
from ase.optimize import FIRE
from mace.calculators import MACECalculator
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


warnings.filterwarnings(
    "ignore",
    message=".*The TorchScript type system doesn't support instance-level annotations on empty non-base types.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
)

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

DEFAULT_MODEL = (
    "/home/liaohui/桌面/MACELES-MPtrj-uiu/"
    "checkpoints/MACELES-MPtrj-uiu-r5.5_run-123_stagetwo.model"
)
DEFAULT_INPUT = (
    "results/piezo_component_labels/"
    "piezo_nextgen_cif_with_static_ionic_total.xyz"
)
DEFAULT_OUTPUT = "results/uiu_bec_ionic_piezo/nextgen_uiu_bec_ionic.tsv"

CONV_E_PER_A2_TO_C_PER_M2 = 16.0217663
MIN_VOLUME = 1.0e-6
AXES = ("x", "y", "z")
COMPONENTS = [f"e{i}{j}" for i in range(1, 4) for j in range(1, 7)]
STRAIN_MODES = (
    ("xx", (0, 0), False),
    ("yy", (1, 1), False),
    ("zz", (2, 2), False),
    ("yz", (1, 2), True),
    ("xz", (0, 2), True),
    ("xy", (0, 1), True),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use MACELES-uiu compute_bec=True output to estimate ionic piezo tensors."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--default_dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--strain", type=float, default=0.01)
    parser.add_argument("--max_structures", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--relax_base", action="store_true")
    parser.add_argument("--no_relax_strained", action="store_true")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--relax_steps_base", type=int, default=200)
    parser.add_argument("--relax_steps_strained", type=int, default=100)
    parser.add_argument("--parity", type=float, default=1.0, choices=[-1.0, 1.0])
    parser.add_argument("--bec_scale", type=float, default=1.0)
    parser.add_argument("--transpose_bec", action="store_true")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def apply_voigt_strain(atoms, mode_idx, strain):
    strained = atoms.copy()
    _, (a, b), is_shear = STRAIN_MODES[mode_idx]
    deformation = np.eye(3)
    if is_shear:
        deformation[a, b] += 0.5 * strain
        deformation[b, a] += 0.5 * strain
    else:
        deformation[a, a] += strain
    strained.set_cell(strained.get_cell() @ deformation.T, scale_atoms=False)
    strained.set_positions(strained.get_positions() @ deformation.T)
    return strained


def relax_structure(atoms, calc, fmax, steps):
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)
    atoms.calc = None


def tensor_label(atoms, key):
    values = atoms.info.get(key)
    if values is None:
        raise ValueError(f"missing {key}")
    return np.asarray(values, dtype=float).reshape(3, 6)


def material_id_from_atoms(atoms):
    return (
        atoms.info.get("mp_id")
        or atoms.info.get("material_id")
        or atoms.info.get("config_id")
        or atoms.info.get("id")
        or ""
    )


def normalize_bec_shape(raw_bec, n_atoms):
    arr = np.asarray(raw_bec, dtype=float)
    if arr.ndim == 2 and arr.shape == (n_atoms, 9):
        arr = arr.reshape(n_atoms, 3, 3)
    elif arr.ndim == 3 and arr.shape == (n_atoms, 3, 3):
        pass
    elif arr.ndim == 4 and arr.shape[0] == n_atoms and arr.shape[-2:] == (3, 3):
        arr = arr.reshape(n_atoms, -1, 3, 3).sum(axis=1)
    elif arr.ndim == 4 and arr.shape[1] == n_atoms and arr.shape[-2:] == (3, 3):
        arr = arr.reshape(-1, n_atoms, 3, 3).sum(axis=0)
    else:
        raise ValueError(f"unexpected BEC output shape {arr.shape}")

    if arr.shape != (n_atoms, 3, 3):
        raise ValueError(f"normalized BEC shape {arr.shape}; expected ({n_atoms}, 3, 3)")
    return arr


def compute_bec(atoms, calc, device, parity, bec_scale, transpose_bec):
    model = calc.models[0]
    config = config_from_atoms(atoms)
    data = AtomicData.from_config(config, z_table=calc.z_table, cutoff=calc.r_max)
    loader = torch_geometric.dataloader.DataLoader(dataset=[data], batch_size=1)
    batch = next(iter(loader)).to(device)

    # compute_bec uses model-side derivatives; keep gradients enabled.
    output = model(
        batch.to_dict(),
        training=False,
        compute_force=False,
        compute_bec=True,
    )
    if output.get("BEC") is None:
        raise ValueError("model output does not contain BEC; check compute_bec support")

    bec = normalize_bec_shape(output["BEC"].detach().cpu().numpy(), len(atoms))
    if transpose_bec:
        bec = np.transpose(bec, (0, 2, 1))
    bec = bec * parity * bec_scale

    alpha_abs_mean = math.nan
    if output.get("latent_alphas") is not None:
        alpha = output["latent_alphas"].detach().cpu().numpy()
        alpha_abs_mean = float(np.mean(np.abs(alpha)))

    del output, batch, data, config, loader
    return bec, alpha_abs_mean


def internal_displacement(affine_atoms, relaxed_atoms):
    affine_scaled = affine_atoms.get_scaled_positions(wrap=False)
    relaxed_scaled = relaxed_atoms.get_scaled_positions(wrap=False)
    delta_scaled = relaxed_scaled - affine_scaled
    delta_scaled -= np.round(delta_scaled)
    return delta_scaled @ np.asarray(relaxed_atoms.get_cell())


def tensor_to_row(prefix, tensor):
    return {f"{prefix}_{comp}": float(value) for comp, value in zip(COMPONENTS, tensor.reshape(-1))}


def scalar_stats(prefix, values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(arr))),
        f"{prefix}_abs_max": float(np.max(np.abs(arr))),
    }


def compute_row(idx, atoms, calc, device, args):
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
        "index": idx,
        "material_id": material_id_from_atoms(atoms),
        "formula": atoms.info.get("pretty_formula", atoms.get_chemical_formula()),
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
    row.update(tensor_to_row("ionic", tensor_label(atoms, "REF_piezoelectric_ionic")))
    row.update(tensor_to_row("static", tensor_label(atoms, "REF_piezoelectric_static")))
    row.update(tensor_to_row("total", tensor_label(atoms, "REF_piezoelectric_total")))

    base_atoms.calc = None
    if not all(np.isfinite(v) for k, v in row.items() if isinstance(v, (int, float))):
        raise ValueError("non-finite value in output row")
    return row


def rankdata(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    unique, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    starts = np.cumsum(np.r_[0, counts[:-1]])
    ranks_unique = starts + (counts - 1) / 2.0 + 1.0
    return ranks_unique[inverse]


def corrcoef(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def metrics_for_vectors(ref, pred):
    ref = np.asarray(ref, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    err = pred - ref
    return {
        "n": int(ref.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "pearson": corrcoef(ref, pred),
        "spearman": corrcoef(rankdata(ref), rankdata(pred)),
        "ref_abs_mean": float(np.mean(np.abs(ref))),
        "pred_abs_mean": float(np.mean(np.abs(pred))),
    }


def summarize(rows, args):
    pred = np.asarray(
        [[row[f"uiu_bec_ionic_{comp}"] for comp in COMPONENTS] for row in rows],
        dtype=float,
    )
    ref_ionic = np.asarray(
        [[row[f"ionic_{comp}"] for comp in COMPONENTS] for row in rows],
        dtype=float,
    )
    ref_total = np.asarray(
        [[row[f"total_{comp}"] for comp in COMPONENTS] for row in rows],
        dtype=float,
    )

    metric_rows = []
    for name, ref in (("ionic_tensor", ref_ionic), ("total_tensor", ref_total)):
        metric = metrics_for_vectors(ref, pred)
        metric_rows.append({"target": name, "variant": "raw", **metric})

        metric = metrics_for_vectors(ref, -pred)
        metric_rows.append({"target": name, "variant": "negated", **metric})

        flat_ref = ref.reshape(-1)
        flat_pred = pred.reshape(-1)
        denom = float(np.dot(flat_pred, flat_pred))
        scale = float(np.dot(flat_ref, flat_pred) / denom) if denom > 0 else math.nan
        scaled_pred = pred * scale if np.isfinite(scale) else pred * math.nan
        metric = metrics_for_vectors(ref, scaled_pred)
        metric_rows.append({"target": name, "variant": f"best_scalar={scale:.6g}", **metric})

    for name, ref in (("ionic_eijmax_abs", ref_ionic), ("total_eijmax_abs", ref_total)):
        ref_max = np.max(np.abs(ref), axis=1)
        pred_max = np.max(np.abs(pred), axis=1)
        metric_rows.append({"target": name, "variant": "absmax", **metrics_for_vectors(ref_max, pred_max)})

    for name, ref in (("ionic_norm", ref_ionic), ("total_norm", ref_total)):
        ref_norm = np.linalg.norm(ref, axis=1)
        pred_norm = np.linalg.norm(pred, axis=1)
        metric_rows.append({"target": name, "variant": "frobenius", **metrics_for_vectors(ref_norm, pred_norm)})

    return metric_rows


def ordered_columns(rows):
    fixed = [
        "index",
        "material_id",
        "formula",
        "natoms",
        "volume",
        "parity",
        "bec_scale",
        "transpose_bec",
        "alpha_abs_mean",
    ]
    tensor_cols = []
    for prefix in ("uiu_bec_ionic", "ionic", "static", "total"):
        tensor_cols.extend([f"{prefix}_{comp}" for comp in COMPONENTS])
    keys = set()
    for row in rows:
        keys.update(row)
    middle = sorted(k for k in keys if k not in set(fixed) and k not in set(tensor_cols))
    return fixed + middle + tensor_cols


def write_tsv(path, rows):
    if not rows:
        return
    columns = ordered_columns(rows)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
    with open(path, "w", newline="") as handle:
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
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metric_rows)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".metrics.tsv")

    atoms_list = read(args.input, index=":")
    if args.start:
        atoms_list = atoms_list[args.start :]
    if args.max_structures is not None:
        atoms_list = atoms_list[: args.max_structures]

    calc = MACECalculator(
        model_paths=args.model,
        device=device,
        default_dtype=args.default_dtype,
    )

    rows = []
    errors = []
    iterator = enumerate(atoms_list, start=args.start)
    if tqdm:
        iterator = tqdm(list(iterator), desc="UIU BEC ionic piezo")

    for idx, atoms in iterator:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rows.append(compute_row(idx, atoms, calc, device, args))
        except Exception as exc:
            errors.append({"index": idx, "message": str(exc)})
            print(f"ERROR {idx}: {exc}")

    write_tsv(output_path, rows)
    metric_rows = summarize(rows, args) if rows else []
    write_metrics(summary_path, metric_rows, args, n_ok=len(rows), n_error=len(errors))

    if errors:
        error_path = output_path.with_suffix(".errors.tsv")
        with open(error_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "message"], delimiter="\t")
            writer.writeheader()
            writer.writerows(errors)

    print(f"Wrote rows: {output_path}")
    print(f"Wrote metrics: {summary_path}")
    if errors:
        print(f"Wrote errors: {output_path.with_suffix('.errors.tsv')}")


if __name__ == "__main__":
    main()
