#!/usr/bin/env python3
"""Train the shared displacement head on cached foundation representations."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from e3nn import o3

from evaluate_displacement_increment_locked_test import load_model as load_response_head
from finetune_uiu_backbone_displacement import (
    DynamicGraphFactory,
    MACEBackboneFeatureExtractor,
    materialize_dynamic_samples,
)
from train_uiu_internal_strain_head import (
    configure_response_emphasis,
    evaluate,
    fit_atomic_response_scales,
    fit_hvp_projected_scale,
    fit_soft_mode_scale,
    fit_target_scales,
    fit_work_scale,
    load_unified_bec_map,
    make_loader,
    move_batch,
    prepare_split,
)
from uiu_equivariant_piezo_model import DEFAULT_NODE_IRREPS
from uiu_internal_strain_model import AtomicInternalStrainHead


DEFAULT_ROOT = Path("results/foundation_displacement_benchmark_v1")
DATA_ROOT = Path("results/pbesol_finish_displacement_increment_v2")
UNIFIED_DIR = Path("results/uiu_regularized_modal_solver_grouped_resplit_v1/physical_polarization")
RESPONSE_MANIFEST = Path("results/uiu_displacement_response_three_way_v1/bec_visible_inverse_response/manifest.json")
RESPONSE_CHECKPOINT = Path("results/uiu_displacement_response_three_way_v1/bec_visible_inverse_response/displacement_response_head.pt")
REF41_BACKBONE = Path("results/uiu_backbone_lora_displacement_safe5_v1/MACELES-uiu-displacement-lastblock-lora-best.model")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loss_args(device: str, unified_dir: Path = UNIFIED_DIR):
    with RESPONSE_MANIFEST.open() as handle:
        args = SimpleNamespace(**json.load(handle)["arguments"])
    args.device = device
    args.node_cache_dir = str(DATA_ROOT / "cache/node")
    args.dfpt_cache = str(DATA_ROOT / "combined_internal_response.pt")
    args.quality_summary = str(DATA_ROOT / "combined_quality_summary.tsv")
    args.atomic_response_dir = str(DATA_ROOT / "cache/atomic_response")
    args.model_hessian_dir = None
    args.unified_prediction_dir = str(unified_dir)
    args.hvp_labels = None
    args.max_train = args.max_valid = None
    args.symmetry_projection = False
    return args


def load_fixed_samples(device: str):
    args = make_loss_args(device)
    train_records = DATA_ROOT / "splits/safe5_train.jsonl"
    valid_records = DATA_ROOT / "splits/safe5_valid.jsonl"
    requested = set()
    for path in (train_records, valid_records):
        with path.open(encoding="utf-8") as handle:
            requested.update(json.loads(line)["jid"] for line in handle if line.strip())
    dfpt_payload = torch.load(args.dfpt_cache, map_location="cpu", weights_only=False)
    dfpt_map = {
        sample["jid"]: sample
        for sample in dfpt_payload["samples"]
        if sample["jid"] in requested
    }
    quality = pd.read_csv(args.quality_summary, sep="\t")
    quality_map = {
        row["jid"]: row
        for row in quality.to_dict("records")
        if row["jid"] in requested
    }
    train_bec = load_unified_bec_map(args.unified_prediction_dir, "train", dfpt_map)
    valid_bec = load_unified_bec_map(args.unified_prediction_dir, "valid", dfpt_map)
    train, train_rejected = prepare_split(
        "train", args, dfpt_map, quality_map, train_bec, require_unified_bec=False
    )
    valid, valid_rejected = prepare_split("valid", args, dfpt_map, quality_map, valid_bec)
    configure_response_emphasis(train, valid, args)
    if len(train) != 1864 or len(valid) != 189:
        raise RuntimeError(
            f"Fixed safe5 filter changed: train={len(train)} valid={len(valid)}; "
            f"rejected train={train_rejected} valid={valid_rejected}"
        )
    return args, train, valid, {"train": train_rejected, "valid": valid_rejected}


def ref41(args, train, valid, outdir: Path, batch_size: int):
    """Re-evaluate §41 with its actual LoRA backbone and frozen response head."""
    from mace.calculators import MACECalculator
    from build_uiu_bec_ionic_piezo_jarvis import atoms_from_record
    from finetune_uiu_backbone_displacement import read_jsonl_map

    outdir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl_map(DATA_ROOT / "splits/safe5_valid.jsonl")
    calc = MACECalculator(
        model_paths=str(REF41_BACKBONE), device=args.device, default_dtype="float32"
    )
    extractor = MACEBackboneFeatureExtractor(calc.models[0]).to(args.device).eval()
    factory = DynamicGraphFactory(records, calc.z_table, calc.r_max)
    dynamic_valid = materialize_dynamic_samples(
        valid, extractor, factory, args.device, batch_size
    )
    response = torch.load(RESPONSE_CHECKPOINT, map_location="cpu", weights_only=False)
    head = load_response_head(response, args.device)
    loader = make_loader(
        dynamic_valid, batch_size, False, 0, args.device.startswith("cuda")
    )
    metrics, rows, predictions = evaluate(
        head,
        loader,
        response["target_block_scales"].to(args.device),
        float(response["active_threshold"]),
        torch.tensor(float(response.get("ionic_scale", 1.0)), device=args.device),
        torch.tensor(float(response.get("soft_mode_scale", 1.0)), device=args.device),
        args,
        collect=True,
        work_scale=torch.tensor(float(response.get("work_scale", 1.0)), device=args.device),
        force_scale=torch.tensor(float(response.get("force_scale", 1.0)), device=args.device),
        hvp_scale=torch.tensor(float(response.get("hvp_scale", 1.0)), device=args.device),
    )
    torch.save({"metrics": metrics, "rows": rows, "predictions": predictions}, outdir / "ref41_valid.pt")
    (outdir / "ref41_valid_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(outdir / "ref41_valid_predictions.tsv", sep="\t", index=False)
    print("REF41 validation:", json.dumps({k: metrics[k] for k in metrics if "mae" in k or "pearson" in k}, indent=2), flush=True)


def parse_irrep_blocks(metadata: dict) -> list[o3.Irreps]:
    blocks = [o3.Irreps(value) for value in metadata["source_irreps_per_block"]]
    if len(blocks) != 2:
        raise ValueError("Each model cache must contain two hidden feature blocks")
    return blocks


def fit_irrep_scales(samples: list[dict], key: str, blocks: list[o3.Irreps], is_response: bool) -> list[torch.Tensor]:
    """RMS per irrep copy, sharing scale across a vector/tensor's components."""
    result = []
    for layer_index, irreps in enumerate(blocks):
        scale_parts = []
        for irrep_slice, (multiplicity, irrep) in zip(irreps.slices(), irreps):
            sums = torch.zeros(multiplicity, dtype=torch.float64)
            count = 0
            for sample in samples:
                tensor = sample[key]
                if is_response:
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                offset = sum(block.dim for block in blocks[:layer_index])
                part = tensor[..., offset + irrep_slice.start : offset + irrep_slice.stop].double()
                part = part.reshape(-1, multiplicity, irrep.dim)
                sums += part.square().sum(dim=(0, 2))
                count += part.shape[0] * irrep.dim
            channel_scale = torch.sqrt(sums / max(count, 1)).clamp_min(1.0e-3)
            scale_parts.append(channel_scale[:, None].expand(multiplicity, irrep.dim).reshape(-1))
        result.append(torch.cat(scale_parts).float())
    return result


class FeatureProjection(torch.nn.Module):
    def __init__(
        self,
        source_blocks,
        feature_scales,
        response_scales,
        channels=128,
        native_input=False,
        input_rms_normalization=None,
        identity_init=False,
    ):
        super().__init__()
        self.source_blocks = source_blocks
        self.channels = None if native_input else int(channels)
        self.native_input = bool(native_input)
        self.identity_init = bool(identity_init)
        if input_rms_normalization is None:
            input_rms_normalization = not self.native_input
        self.input_rms_normalization = bool(input_rms_normalization)
        if self.native_input and self.identity_init:
            raise ValueError("native_input and identity_init are mutually exclusive")
        target_blocks = source_blocks if native_input else [
            o3.Irreps([(self.channels, ir) for _, ir in source])
            for source in source_blocks
        ]
        if self.identity_init:
            for source, target in zip(source_blocks, target_blocks):
                if source != target:
                    raise ValueError("Identity initialization requires equal source/target irreps")
        self.target_blocks = target_blocks
        self.public_irreps = o3.Irreps([])
        for block in target_blocks:
            self.public_irreps += block
        linears = []
        for source, target in zip(source_blocks, target_blocks):
            if native_input:
                linear = torch.nn.Identity()
            elif self.identity_init:
                # Keep only the like-for-like irrep-channel paths.  With
                # e3nn's path normalization, sqrt(multiplicity) * I gives an
                # exact identity map at initialization.
                instructions = [(index, index) for index in range(len(source))]
                linear = o3.Linear(source, target, instructions=instructions)
                with torch.no_grad():
                    offset = 0
                    for instruction in linear.instructions:
                        multiplicity = instruction.path_shape[0]
                        matrix = linear.weight[offset : offset + multiplicity * multiplicity]
                        matrix.view(multiplicity, multiplicity).copy_(
                            torch.eye(multiplicity, dtype=matrix.dtype, device=matrix.device)
                            * (multiplicity ** 0.5)
                        )
                        offset += multiplicity * multiplicity
            else:
                linear = o3.Linear(source, target)
            linears.append(linear)
        self.linears = torch.nn.ModuleList(linears)
        if not self.input_rms_normalization:
            feature_scales = [torch.ones(block.dim) for block in source_blocks]
            response_scales = [torch.ones(block.dim) for block in source_blocks]
        self.register_buffer("feature_scales", torch.cat(feature_scales))
        self.register_buffer("response_scales", torch.cat(response_scales))
        self.source_dims = [block.dim for block in source_blocks]

    def _project(self, values: torch.Tensor, scales: torch.Tensor):
        projected = []
        offset = 0
        scale_offset = 0
        for block, linear in zip(self.source_blocks, self.linears):
            stop = offset + block.dim
            scale_stop = scale_offset + block.dim
            part = values[..., offset:stop]
            if self.input_rms_normalization:
                part = part / scales[scale_offset:scale_stop]
            projected.append(linear(part))
            offset = stop
            scale_offset = scale_stop
        return torch.cat(projected, dim=-1)

class ProjectedResponseModel(torch.nn.Module):
    def __init__(self, projection: FeatureProjection, head: AtomicInternalStrainHead):
        super().__init__()
        self.projection = projection
        self.head = head

    def forward(self, node_features, batch, atomic_response=None, num_graphs=None):
        projected_nodes = self.projection._project(
            node_features, self.projection.feature_scales
        )
        projected_response = None
        if atomic_response is not None:
            shape = atomic_response.shape
            projected_response = self.projection._project(
                atomic_response.reshape(-1, shape[-1]), self.projection.response_scales
            ).reshape(shape[0], shape[1], -1)
        return self.head(projected_nodes, batch, projected_response, num_graphs)


def attach_cache(samples, cache_path: Path):
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache = payload["samples"]
    missing = [sample["jid"] for sample in samples if sample["jid"] not in cache]
    if missing:
        raise KeyError(f"Feature cache missing {len(missing)} samples; first={missing[0]}")
    for sample in samples:
        item = cache[sample["jid"]]
        if item["node_features"].shape[0] != sample["target_irreps"].shape[0]:
            raise ValueError(f"atom count mismatch in cache for {sample['jid']}")
        sample["node_features"] = item["node_features"]
        sample["atomic_response"] = item["atomic_response"]
    return payload["metadata"]


def projected_initial_scales(model, train_samples, device):
    # Response normalization is fit after the randomly initialized equivariant
    # input projection, using training data only.
    temporary = []
    with torch.no_grad():
        for sample in train_samples:
            r = sample["atomic_response"].to(device)
            sample_copy = {"atomic_response": model.projection._project(
                r.reshape(-1, r.shape[-1]), model.projection.response_scales
            ).reshape(r.shape[0], r.shape[1], -1).cpu()}
            temporary.append(sample_copy)
    return fit_atomic_response_scales(temporary, model.projection.public_irreps)


def ref_score(metrics: dict, ref: dict) -> float:
    pairs = (
        ("ionic_component_mae", "ionic_component_mae"),
        ("ionic_eijmax_mae", "ionic_eijmax_mae"),
        ("unified_ionic_component_mae", "unified_ionic_component_mae"),
        ("unified_ionic_eijmax_mae", "unified_ionic_eijmax_mae"),
    )
    values = []
    for key, refkey in pairs:
        denominator = float(ref[refkey])
        if key not in metrics or denominator <= 0:
            raise KeyError(f"Selection score needs metric {key} and baseline {refkey}")
        values.append(float(metrics[key]) / denominator)
    return float(np.mean(values))


def train_frozen(
    model_id: str,
    root: Path,
    device: str,
    epochs: int,
    batch_size: int,
    valid_batch_size: int,
    smoke_structures: int | None = None,
    cache_model_id: str | None = None,
    projection_channels: int = 128,
    native_input: bool = False,
    input_rms_normalization: bool | None = None,
    identity_init: bool = False,
    seed: int = 123,
    loss_overrides: dict[str, float] | None = None,
    patience_checks: int = 6,
    output_subdir: str = "frozen_probe",
):
    set_seed(seed)
    outdir = (root / "compatibility" / "smoke_fit" / model_id
              if smoke_structures is not None else root / output_subdir / model_id)
    outdir.mkdir(parents=True, exist_ok=True)
    args, train, valid, rejected = load_fixed_samples(device)
    if loss_overrides:
        allowed = {
            "lambda_bec_visible", "lambda_bec_null", "lambda_ionic",
            "lambda_weighted_dft_ionic", "lambda_dft_eijmax",
            "lambda_unified_ionic", "lambda_unified_eijmax",
            "lambda_cosine", "high_response_weight",
        }
        unknown = set(loss_overrides) - allowed
        if unknown:
            raise ValueError(f"Unsupported loss override(s): {sorted(unknown)}")
        for key, value in loss_overrides.items():
            setattr(args, key, float(value))
    cache_model_id = cache_model_id or model_id
    metadata = attach_cache(train, root / "cache" / cache_model_id / "train_features.pt")
    valid_metadata = attach_cache(valid, root / "cache" / cache_model_id / "valid_features.pt")
    if smoke_structures is not None:
        train = train[:smoke_structures]
        if len(train) != smoke_structures:
            raise ValueError(f"Requested {smoke_structures} smoke structures, have {len(train)}")
    if metadata["model_sha256"] != valid_metadata["model_sha256"]:
        raise ValueError("Train/validation feature caches came from different weights")
    blocks = parse_irrep_blocks(metadata)
    feature_scales = fit_irrep_scales(train, "node_features", blocks, False)
    response_scales = fit_irrep_scales(train, "atomic_response", blocks, True)
    projection = FeatureProjection(
        blocks, feature_scales, response_scales,
        channels=projection_channels,
        native_input=native_input,
        input_rms_normalization=input_rms_normalization,
        identity_init=identity_init,
    ).to(device)
    # A candidate-specific projector has a candidate-specific parameter count;
    # reset the RNG so the common displacement head starts identically in every arm.
    torch.manual_seed(seed + 1000)
    target_scales = fit_target_scales(train).to(device)
    # Fit shared head response normalization using its initial projection.
    provisional = AtomicInternalStrainHead(
        node_irreps=projection.public_irreps,
        scalar_channels=32,
        vector_channels=16,
        tensor_channels=16,
        atomic_response=True,
        atomic_response_feature_scales=torch.ones(projection.public_irreps.dim),
        atomic_response_mix_init=0.1,
        output_block_scales=target_scales,
    )
    model = ProjectedResponseModel(projection, provisional).to(device)
    response_target_scales = projected_initial_scales(model, train, device)
    model.head.atomic_response_feature_scales.copy_(response_target_scales.to(device))

    ref_path = root / "ref41_valid_metrics.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"Run --reference-only first: {ref_path}")
    ref = json.loads(ref_path.read_text())
    train_loader = make_loader(
        train,
        batch_size,
        True,
        0,
        device.startswith("cuda"),
        generator=torch.Generator().manual_seed(seed),
    )
    valid_loader = make_loader(valid, valid_batch_size, False, 0, device.startswith("cuda"))
    ionic_scale = torch.sqrt(torch.mean(torch.stack([s["reference_ionic"] for s in train]).square())).clamp_min(0.05).to(device)
    soft_mode_scale = fit_soft_mode_scale(train).to(device)
    work_scale = fit_work_scale(train).to(device)
    force_values = torch.cat([s["force_response"].reshape(-1) for s in train]).float()
    force_scale = torch.quantile(force_values.abs(), 0.75).clamp_min(1.0e-3).to(device)
    hvp_scale = fit_hvp_projected_scale(train).to(device)
    train_norms = torch.cat([torch.linalg.vector_norm(s["target_irreps"], dim=1) for s in train])
    active_threshold = float(torch.quantile(train_norms, args.active_quantile))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.0)
    best_score = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    initial_metrics, _, _ = evaluate(
        model, valid_loader, target_scales, active_threshold, ionic_scale,
        soft_mode_scale, args, work_scale=work_scale, force_scale=force_scale,
        hvp_scale=hvp_scale,
    )
    print(f"{model_id}: 1864 train / 189 valid; initial validation score={ref_score(initial_metrics, ref):.5f}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for cpu_batch in train_loader:
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            from train_uiu_internal_strain_head import batch_loss
            loss, _, _, _ = batch_loss(
                model, batch, target_scales, active_threshold, ionic_scale,
                soft_mode_scale, args, work_scale, force_scale, hvp_scale,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 5 != 0 and epoch != epochs:
            continue
        metrics, _, _ = evaluate(
            model, valid_loader, target_scales, active_threshold, ionic_scale,
            soft_mode_scale, args, work_scale=work_scale, force_scale=force_scale,
            hvp_scale=hvp_scale,
        )
        score = ref_score(metrics, ref)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_score": score, **metrics})
        print(
            f"{model_id} epoch {epoch:03d}: score={score:.5f} "
            f"DFT emax MAE={metrics['ionic_eijmax_mae']:.4f} "
            f"model-BEC emax MAE={metrics['unified_ionic_eijmax_mae']:.4f} "
            f"model-BEC r={metrics['unified_ionic_eijmax_pearson']:.4f}",
            flush=True,
        )
        if score < best_score - 1.0e-8:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_id": model_id,
                    "model_metadata": metadata,
                    "source_irreps": [str(block) for block in blocks],
                    "public_irreps": str(projection.public_irreps),
                    "projection_channels": projection_channels,
                    "native_input": native_input,
                    "input_rms_normalization": projection.input_rms_normalization,
                    "identity_init": identity_init,
                    "feature_stage": metadata.get("feature_stage", "pre_product"),
                    "source_feature_scales": feature_scales,
                    "source_response_scales": response_scales,
                    "target_block_scales": target_scales.cpu(),
                    "atomic_response_feature_scales": response_target_scales,
                    "active_threshold": active_threshold,
                    "ionic_scale": float(ionic_scale),
                    "soft_mode_scale": float(soft_mode_scale),
                    "work_scale": float(work_scale),
                    "force_scale": float(force_scale),
                    "hvp_scale": float(hvp_scale),
                    "best_epoch": best_epoch,
                    "best_selection_score": best_score,
                },
                outdir / "best_checkpoint.pt",
            )
        else:
            stale += 1
        if stale >= patience_checks:
            print(f"{model_id}: early stopping after {patience_checks} validation checks", flush=True)
            break
    if best_state is None:
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(best_state)
    final_metrics, rows, predictions = evaluate(
        model, valid_loader, target_scales, active_threshold, ionic_scale,
        soft_mode_scale, args, collect=True, work_scale=work_scale,
        force_scale=force_scale, hvp_scale=hvp_scale,
    )
    pd.DataFrame(history).to_csv(outdir / "history.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(outdir / "valid_predictions.tsv", sep="\t", index=False)
    torch.save({"split": "valid", "samples": predictions}, outdir / "valid_displacement_predictions.pt")
    manifest = {
        "model_id": model_id,
        "model_sha256": metadata["model_sha256"],
        "n_train": len(train), "n_valid": len(valid),
        "rejected": rejected,
        "source_irreps": [str(block) for block in blocks],
        "public_irreps": str(projection.public_irreps),
        "projection_channels": projection_channels,
        "native_input": native_input,
        "input_rms_normalization": projection.input_rms_normalization,
        "identity_init": identity_init,
        "feature_stage": metadata.get("feature_stage", "pre_product"),
        "projection_parameters": sum(p.numel() for p in model.projection.parameters()),
        "head_parameters": sum(p.numel() for p in model.head.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "selection_score": best_score,
        "initial_valid_metrics": initial_metrics,
        "final_valid_metrics": final_metrics,
        "training": {"seed": seed, "max_epochs": epochs, "valid_every": 5, "patience_checks": patience_checks, "batch_size": batch_size, "lr": 1e-4},
        "smoke_fit_structures": smoke_structures,
        "loss_overrides": loss_overrides or {},
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{model_id} final validation:", json.dumps({k: final_metrics[k] for k in final_metrics if "mae" in k or "pearson" in k}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience-checks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--valid-batch-size", type=int, default=16)
    parser.add_argument("--materialize-batch-size", type=int, default=4)
    parser.add_argument("--smoke-structures", type=int)
    parser.add_argument("--cache-model-id")
    parser.add_argument("--projection-channels", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--native-input", action="store_true")
    parser.add_argument("--input-rms-normalization", dest="input_rms_normalization", action="store_true")
    parser.add_argument("--no-input-rms-normalization", dest="input_rms_normalization", action="store_false")
    parser.set_defaults(input_rms_normalization=None)
    parser.add_argument("--identity-init-projection", action="store_true")
    parser.add_argument(
        "--loss-override", action="append", default=[], metavar="NAME=VALUE",
        help="Override one existing response-loss coefficient; may be repeated.",
    )
    parser.add_argument("--output-subdir", default="frozen_probe")
    args = parser.parse_args()
    if args.reference_only:
        loss_args, train, valid, _ = load_fixed_samples(args.device)
        ref41(loss_args, train, valid, args.root, args.materialize_batch_size)
    elif args.model_id:
        loss_overrides = {}
        for item in args.loss_override:
            if "=" not in item:
                parser.error(f"invalid --loss-override {item!r}; expected NAME=VALUE")
            name, value = item.split("=", 1)
            if name in loss_overrides:
                parser.error(f"duplicate --loss-override for {name}")
            try:
                loss_overrides[name] = float(value)
            except ValueError:
                parser.error(f"invalid numeric value in --loss-override {item!r}")
        train_frozen(
            args.model_id,
            args.root,
            args.device,
            args.epochs,
            args.batch_size,
            args.valid_batch_size,
            args.smoke_structures,
            cache_model_id=args.cache_model_id,
            projection_channels=args.projection_channels,
            native_input=args.native_input,
            input_rms_normalization=args.input_rms_normalization,
            identity_init=args.identity_init_projection,
            seed=args.seed,
            loss_overrides=loss_overrides,
            patience_checks=args.patience_checks,
            output_subdir=args.output_subdir,
        )
    else:
        parser.error("choose --reference-only or --model-id")


if __name__ == "__main__":
    main()
