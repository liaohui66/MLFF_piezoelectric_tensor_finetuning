#!/usr/bin/env python3
"""Train one dipole potential whose derivatives yield BEC and static piezo."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mace.calculators import MACECalculator
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric
from scipy.stats import pearsonr, spearmanr

from build_uiu_bec_ionic_piezo_jarvis import atoms_from_record
from uiu_differentiable_polarization import (
    DifferentiablePolarizationHead,
    differentiate_polarization_responses,
    ionic_piezo_from_bec_displacement,
)


SPLIT_FILES = {
    "train": "safe5_train.jsonl",
    "valid": "safe5_valid.jsonl",
    "test": "safe5_test.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=(
            "results/uiu_bec_jarvis_safe5_omat24_epoch21_finetune/"
            "MACELES-uiu-jarvis-bec-ft-best.model"
        ),
    )
    parser.add_argument(
        "--split-dir",
        default=(
            "results/jarvis_dfpt_reextracted_clean20_voigt_corrected/"
            "splits_piezo_safe5"
        ),
    )
    parser.add_argument(
        "--dfpt-cache",
        default="results/jarvis_dfpt_internal_response/jarvis_dfpt_internal_response.pt",
    )
    parser.add_argument(
        "--quality-summary",
        default="results/jarvis_dfpt_internal_response/quality_summary.tsv",
    )
    parser.add_argument(
        "--outdir",
        default="results/uiu_physical_polarization_response_safe5_v1",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--scalar-channels", type=int, default=32)
    parser.add_argument("--vector-channels", type=int, default=16)
    parser.add_argument("--tensor-channels", type=int, default=16)
    parser.add_argument("--charge-scale", type=float, default=0.1)
    parser.add_argument("--dipole-scale", type=float, default=0.1)
    parser.add_argument("--huber-beta", type=float, default=0.2)
    parser.add_argument("--lambda-bec", type=float, default=1.0)
    parser.add_argument("--lambda-static", type=float, default=1.0)
    parser.add_argument("--lambda-ionic", type=float, default=0.5)
    parser.add_argument("--lambda-total", type=float, default=1.0)
    parser.add_argument("--lambda-bec-cosine", type=float, default=0.1)
    parser.add_argument("--max-reconstruction-mae", type=float, default=0.01)
    parser.add_argument("--allow-unstable-hessian", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume head, optimizer, scheduler, best state, and history.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize a compatible physical polarization head checkpoint.",
    )
    parser.add_argument(
        "--initial-validation-candidate",
        action="store_true",
        help="Allow the initialized epoch-0 head to remain the best checkpoint.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_tensor_dict(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_split(
    split: str,
    args: argparse.Namespace,
    calc: MACECalculator,
    dfpt_map: dict[str, dict],
    quality_map: dict[str, dict],
) -> tuple[list[AtomicData], list[dict], dict[str, int]]:
    records = read_records(Path(args.split_dir) / SPLIT_FILES[split])
    limit = getattr(args, f"max_{split}")
    graphs: list[AtomicData] = []
    metadata: list[dict] = []
    rejected = {
        "missing_dfpt": 0,
        "quality": 0,
        "atom_mismatch": 0,
        "missing_label": 0,
    }
    for record in records:
        jid = record["jid"]
        if jid not in dfpt_map or jid not in quality_map:
            rejected["missing_dfpt"] += 1
            continue
        quality = quality_map[jid]
        stable = float(quality["reconstruction_mae"]) <= args.max_reconstruction_mae
        if not args.allow_unstable_hessian:
            stable = stable and int(quality["negative_mode_count"]) == 0
        if not stable:
            rejected["quality"] += 1
            continue
        required = (
            "born_effective_charge",
            "piezoelectric_tensor_clamped_Cm2",
            "piezoelectric_tensor_ionic_Cm2",
            "piezoelectric_tensor_Cm2",
        )
        if any(key not in record for key in required):
            rejected["missing_label"] += 1
            continue
        atoms = atoms_from_record(record)
        dfpt = dfpt_map[jid]
        bec = torch.as_tensor(record["born_effective_charge"], dtype=torch.float32)
        displacement = dfpt["internal_displacement_dstrain"].float()
        if bec.shape != (len(atoms), 3, 3) or displacement.shape != (len(atoms), 3, 6):
            rejected["atom_mismatch"] += 1
            continue
        bec = bec - bec.mean(dim=0, keepdim=True)
        static = torch.as_tensor(
            record["piezoelectric_tensor_clamped_Cm2"], dtype=torch.float32
        ).reshape(1, 3, 6)
        ionic = torch.as_tensor(
            record["piezoelectric_tensor_ionic_Cm2"], dtype=torch.float32
        ).reshape(1, 3, 6)
        total = torch.as_tensor(
            record["piezoelectric_tensor_Cm2"], dtype=torch.float32
        ).reshape(1, 3, 6)

        graph = AtomicData.from_config(
            config_from_atoms(atoms),
            z_table=calc.z_table,
            cutoff=calc.r_max,
        )
        graph.target_bec = bec
        graph.target_displacement = displacement
        graph.target_static = static
        graph.target_ionic = ionic
        graph.target_total = total
        # PyG automatically offsets attributes whose names contain "index".
        # Keep this graph-level identifier under a neutral name.
        graph.sample_id = torch.tensor([len(metadata)], dtype=torch.long)
        graphs.append(graph)
        metadata.append(
            {
                "jid": jid,
                "formula": atoms.get_chemical_formula(),
                "volume": float(atoms.get_volume()),
            }
        )
        if limit is not None and len(graphs) >= limit:
            break
    return graphs, metadata, rejected


def make_loader(graphs, batch_size, shuffle, workers, pin_memory):
    options = {}
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return torch_geometric.dataloader.DataLoader(
        dataset=graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        **options,
    )


def fit_scales(graphs: list[AtomicData]) -> dict[str, float]:
    values = {
        "bec": torch.cat([graph.target_bec.reshape(-1) for graph in graphs]),
        "static": torch.cat([graph.target_static.reshape(-1) for graph in graphs]),
        "ionic": torch.cat([graph.target_ionic.reshape(-1) for graph in graphs]),
        "total": torch.cat([graph.target_total.reshape(-1) for graph in graphs]),
    }
    return {
        key: float(torch.sqrt(torch.mean(value.square())).clamp_min(0.05))
        for key, value in values.items()
    }


def normalized_huber(prediction, target, scale, beta):
    return F.smooth_l1_loss(prediction / scale, target / scale, beta=beta)


def model_batch(base_model, head, batch, scales, args, create_graph):
    data = clone_tensor_dict(batch.to_dict())
    base_positions = data["positions"]
    cell = data["cell"].view(-1, 3, 3)
    volumes = torch.linalg.det(cell).abs()
    num_graphs = int(data["ptr"].numel() - 1)

    output = base_model(
        data,
        training=False,
        compute_force=False,
        compute_virials=False,
        compute_stress=False,
        compute_displacement=True,
        compute_bec=False,
    )
    prepared_positions = data["positions"]
    dipole_output = head(
        output["node_feats"],
        prepared_positions,
        data["batch"],
        num_graphs,
    )
    response = differentiate_polarization_responses(
        dipole_output["dipole_eA"],
        base_positions,
        output["displacement"],
        data["batch"],
        volumes,
        head.voigt_basis,
        create_graph=create_graph,
    )
    ionic = ionic_piezo_from_bec_displacement(
        response["bec"],
        data["target_displacement"],
        data["batch"],
        volumes,
        num_graphs,
    )
    total = response["static_piezo"] + ionic

    target_bec = data["target_bec"]
    target_static = data["target_static"].reshape(num_graphs, 3, 6)
    target_ionic = data["target_ionic"].reshape(num_graphs, 3, 6)
    target_total = data["target_total"].reshape(num_graphs, 3, 6)
    bec_loss = normalized_huber(
        response["bec"], target_bec, scales["bec"], args.huber_beta
    )
    static_loss = normalized_huber(
        response["static_piezo"], target_static, scales["static"], args.huber_beta
    )
    ionic_loss = normalized_huber(
        ionic, target_ionic, scales["ionic"], args.huber_beta
    )
    total_loss = normalized_huber(
        total, target_total, scales["total"], args.huber_beta
    )
    target_norm = torch.linalg.vector_norm(target_bec.reshape(len(target_bec), -1), dim=1)
    active = target_norm >= torch.quantile(target_norm.detach(), 0.5)
    bec_cosine = (
        1.0
        - F.cosine_similarity(
            response["bec"][active].reshape(active.sum(), -1),
            target_bec[active].reshape(active.sum(), -1),
            dim=1,
            eps=1.0e-8,
        )
    ).mean()
    loss = (
        args.lambda_bec * bec_loss
        + args.lambda_static * static_loss
        + args.lambda_ionic * ionic_loss
        + args.lambda_total * total_loss
        + args.lambda_bec_cosine * bec_cosine
    )
    return loss, {
        "loss": loss,
        "bec_loss": bec_loss,
        "static_loss": static_loss,
        "ionic_loss": ionic_loss,
        "total_loss": total_loss,
        "bec_cosine_loss": bec_cosine,
        "asr_max_before": response["bec_asr_before"].abs().max(),
        "pred_bec": response["bec"],
        "pred_static": response["static_piezo"],
        "pred_ionic": ionic,
        "pred_total": total,
        "pred_polarization": response["polarization_Cm2"],
        "target_bec": target_bec,
        "target_static": target_static,
        "target_ionic": target_ionic,
        "target_total": target_total,
        "sample_id": data["sample_id"].reshape(-1),
    }


def finite_corr(function, reference, prediction):
    reference = np.asarray(reference, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    mask = np.isfinite(reference) & np.isfinite(prediction)
    if mask.sum() < 2 or np.std(reference[mask]) == 0 or np.std(prediction[mask]) == 0:
        return float("nan")
    return float(function(reference[mask], prediction[mask]).statistic)


def tensor_metrics(reference, prediction):
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - reference
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson": finite_corr(pearsonr, reference, prediction),
        "spearman": finite_corr(spearmanr, reference, prediction),
    }


def evaluate(base_model, head, loader, scales, args, metadata, collect=False):
    head.eval()
    sums = {key: 0.0 for key in (
        "loss", "bec_loss", "static_loss", "ionic_loss", "total_loss",
        "bec_cosine_loss",
    )}
    count = 0
    collected = {key: [] for key in (
        "pred_bec", "pred_static", "pred_ionic", "pred_total",
        "pred_polarization", "target_bec", "target_static", "target_ionic",
        "target_total", "sample_id",
    )}
    max_asr = 0.0
    with torch.enable_grad():
        for batch in loader:
            batch = batch.to(args.device)
            loss, result = model_batch(
                base_model, head, batch, scales, args, create_graph=False
            )
            batch_count = int(result["pred_total"].shape[0])
            for key in sums:
                sums[key] += float(result[key].detach()) * batch_count
            max_asr = max(max_asr, float(result["asr_max_before"].detach()))
            count += batch_count
            if collect:
                for key in collected:
                    collected[key].append(result[key].detach().cpu())
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics["bec_asr_max_before_projection"] = max_asr
    if not collect:
        return metrics, None
    payload = {key: torch.cat(value) for key, value in collected.items()}
    metrics.update({f"bec_{key}": value for key, value in tensor_metrics(
        payload["target_bec"], payload["pred_bec"]
    ).items()})
    for target in ("static", "ionic", "total"):
        metrics.update({f"{target}_{key}": value for key, value in tensor_metrics(
            payload[f"target_{target}"], payload[f"pred_{target}"]
        ).items()})
        reference_emax = payload[f"target_{target}"].abs().amax(dim=(1, 2))
        prediction_emax = payload[f"pred_{target}"].abs().amax(dim=(1, 2))
        metrics.update({f"{target}_eijmax_{key}": value for key, value in tensor_metrics(
            reference_emax, prediction_emax
        ).items()})
    payload["metadata"] = [metadata[int(index)] for index in payload["sample_id"]]
    return metrics, payload


def write_metrics(path: Path, rows: list[dict]) -> None:
    keys = ["split"] + sorted({key for row in rows for key in row if key != "split"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loading frozen MACELES model: {args.model}", flush=True)
    calc = MACECalculator(
        model_paths=args.model,
        device=args.device,
        default_dtype="float32",
    )
    base_model = calc.models[0]
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    dfpt_payload = torch.load(args.dfpt_cache, map_location="cpu", weights_only=False)
    dfpt_map = {sample["jid"]: sample for sample in dfpt_payload["samples"]}
    quality_frame = pd.read_csv(args.quality_summary, sep="\t")
    quality_map = {row["jid"]: row for row in quality_frame.to_dict("records")}

    split_graphs = {}
    split_metadata = {}
    split_rejected = {}
    for split in ("train", "valid", "test"):
        graphs, metadata, rejected = build_split(
            split, args, calc, dfpt_map, quality_map
        )
        split_graphs[split] = graphs
        split_metadata[split] = metadata
        split_rejected[split] = rejected
        print(
            f"{split}: usable={len(graphs)} rejected={rejected}",
            flush=True,
        )
    if not split_graphs["train"] or not split_graphs["valid"]:
        raise RuntimeError("empty train or validation split after quality filtering")

    pin_memory = args.device.startswith("cuda")
    loaders = {
        split: make_loader(
            graphs,
            args.batch_size,
            split == "train",
            args.loader_workers,
            pin_memory,
        )
        for split, graphs in split_graphs.items()
        if graphs
    }
    scales = fit_scales(split_graphs["train"])
    print(f"target RMS scales: {scales}", flush=True)
    head = DifferentiablePolarizationHead(
        scalar_channels=args.scalar_channels,
        vector_channels=args.vector_channels,
        tensor_channels=args.tensor_channels,
        charge_scale=args.charge_scale,
        dipole_scale=args.dipole_scale,
    ).to(args.device)
    if args.init_checkpoint is not None and args.resume is None:
        initialization = torch.load(
            args.init_checkpoint, map_location=args.device, weights_only=False
        )
        state_dict = initialization.get(
            "model_state", initialization.get("model_state_dict")
        )
        if state_dict is None:
            raise KeyError("initial checkpoint has no model state")
        head.load_state_dict(state_dict)
        print(f"Initialized physical head from {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, args.patience // 4)
    )
    trainable = sum(parameter.numel() for parameter in head.parameters())
    print(f"trainable polarization parameters={trainable}", flush=True)

    history = []
    best_state = None
    best_epoch = 0
    best_valid = math.inf
    stale = 0
    start_epoch = 1
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location=args.device, weights_only=False)
        head.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        best_state = state["best_state"]
        best_epoch = int(state["best_epoch"])
        best_valid = float(state["best_valid_loss"])
        stale = int(state["stale"])
        start_epoch = int(state["epoch"]) + 1
        print(
            f"Resuming from epoch {state['epoch']} with best epoch {best_epoch}",
            flush=True,
        )
    elif args.initial_validation_candidate:
        initial_metrics, _ = evaluate(
            base_model,
            head,
            loaders["valid"],
            scales,
            args,
            split_metadata["valid"],
            collect=False,
        )
        best_valid = float(initial_metrics["loss"])
        best_state = copy.deepcopy(head.state_dict())
        history.append(
            {
                "epoch": 0,
                "train_loss": float("nan"),
                "valid_loss": best_valid,
                "valid_bec_loss": initial_metrics["bec_loss"],
                "valid_static_loss": initial_metrics["static_loss"],
                "valid_ionic_loss": initial_metrics["ionic_loss"],
                "valid_total_loss": initial_metrics["total_loss"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch 0000 valid={best_valid:.6f} retained as initial candidate",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        head.train()
        running = 0.0
        seen = 0
        for batch in loaders["train"]:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, result = model_batch(
                base_model, head, batch, scales, args, create_graph=True
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            optimizer.step()
            batch_count = int(result["pred_total"].shape[0])
            running += float(loss.detach()) * batch_count
            seen += batch_count
        train_loss = running / max(seen, 1)
        valid_metrics, _ = evaluate(
            base_model,
            head,
            loaders["valid"],
            scales,
            args,
            split_metadata["valid"],
            collect=False,
        )
        valid_loss = valid_metrics["loss"]
        scheduler.step(valid_loss)
        improved = valid_loss < best_valid - 1.0e-6
        if improved:
            best_valid = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_bec_loss": valid_metrics["bec_loss"],
            "valid_static_loss": valid_metrics["static_loss"],
            "valid_ionic_loss": valid_metrics["ionic_loss"],
            "valid_total_loss": valid_metrics["total_loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        torch.save(
            {
                "epoch": epoch,
                "model_state": head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_state": best_state,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid,
                "stale": stale,
                "history": history,
                "args": vars(args),
                "scales": scales,
            },
            outdir / "training_state.pt",
        )
        print(
            f"epoch {epoch:04d} train={train_loss:.6f} "
            f"valid={valid_loss:.6f} static={valid_metrics['static_loss']:.6f} "
            f"ionic={valid_metrics['ionic_loss']:.6f} "
            f"total={valid_metrics['total_loss']:.6f} best={best_epoch}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    head.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "args": vars(args),
            "scales": scales,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid,
        },
        outdir / "physical_polarization_head.pt",
    )
    pd.DataFrame(history).to_csv(outdir / "history.tsv", sep="\t", index=False)

    metric_rows = []
    evaluation_splits = ["train", "valid"]
    if not args.skip_test:
        evaluation_splits.append("test")
    for split in evaluation_splits:
        metrics, payload = evaluate(
            base_model,
            head,
            loaders[split],
            scales,
            args,
            split_metadata[split],
            collect=True,
        )
        metric_rows.append({"split": split, **metrics})
        torch.save(payload, outdir / f"{split}_predictions.pt")
    write_metrics(outdir / "metrics.tsv", metric_rows)

    manifest = {
        "objective": "single differentiable dipole -> BEC + proper static; ionic=BEC@du/dstrain; total=static+ionic",
        "model": args.model,
        "split_dir": args.split_dir,
        "dfpt_cache": args.dfpt_cache,
        "quality_summary": args.quality_summary,
        "splits": {key: len(value) for key, value in split_graphs.items()},
        "rejected": split_rejected,
        "target_scales": scales,
        "trainable_parameters": trainable,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid,
        "proper_piezo": True,
        "voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
        "ionic_displacement": "DFT internal_displacement_dstrain in phase-1 training/evaluation",
        "args": vars(args),
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
