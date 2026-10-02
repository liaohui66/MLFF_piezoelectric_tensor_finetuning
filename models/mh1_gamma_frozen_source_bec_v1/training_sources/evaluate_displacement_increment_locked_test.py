#!/usr/bin/env python3
"""Compare displacement checkpoints on one unchanged grouped test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from train_uiu_internal_strain_head import (
    evaluate,
    load_unified_bec_map,
    make_loader,
    prepare_split,
)
from uiu_internal_strain_model import AtomicInternalStrainHead
from uiu_collective_attention_displacement import build_displacement_head
from uiu_internal_strain_feature_adapter import (
    EquivariantFeatureTangentAdapter,
    FeatureAdaptedInternalStrainHead,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-checkpoint",
        type=Path,
        default=Path(
            "results/uiu_regularized_modal_solver_grouped_resplit_v1/"
            "displacement_base/displacement_response_head.pt"
        ),
    )
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--unified-dir",
        type=Path,
        default=Path(
            "results/uiu_regularized_modal_solver_grouped_resplit_v1/physical_polarization"
        ),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_model(
    checkpoint: dict,
    device: str,
    *,
    load_weights: bool = True,
) -> torch.nn.Module:
    config = checkpoint["model_config"]
    state_dict = checkpoint.get("model_state_dict") if load_weights else None
    wrapped_historical = (
        config.get("feature_adapter", False)
        and config.get("head_architecture", "historical") == "historical"
    )
    head = build_displacement_head(
        config,
        checkpoint["atomic_response_feature_scales"],
        checkpoint["target_block_scales"],
        local_geometry_feature_scales=checkpoint.get("local_geometry_feature_scales"),
        load_weights=load_weights and not wrapped_historical,
        model_state_dict=state_dict if load_weights and not wrapped_historical else None,
    )
    model = head
    if wrapped_historical:
        if not load_weights:
            raise ValueError("random initialization for wrapped feature-adapter heads is unsupported")
        adapter = EquivariantFeatureTangentAdapter(
            node_irreps=config["node_irreps"],
            scalar_channels=config["adapter_scalar_channels"],
            vector_channels=config["adapter_vector_channels"],
            tensor_channels=config["adapter_tensor_channels"],
            feature_scales=checkpoint["node_feature_scales"],
            tangent_step=config["adapter_tangent_step"],
        )
        model = FeatureAdaptedInternalStrainHead(head, adapter)
        model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    return model


def main() -> None:
    cli = parse_args()
    cli.outdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(cli.candidate_manifest.read_text())
    args = SimpleNamespace(**manifest["arguments"])
    args.device = cli.device
    args.max_valid = None
    args.max_test = None
    args.unified_prediction_dir = str(cli.unified_dir)
    dfpt_payload = torch.load(args.dfpt_cache, map_location="cpu", weights_only=False)
    dfpt_map = {sample["jid"]: sample for sample in dfpt_payload["samples"]}
    quality = pd.read_csv(args.quality_summary, sep="\t")
    quality_map = {row["jid"]: row for row in quality.to_dict("records")}
    unified = load_unified_bec_map(str(cli.unified_dir), "test", dfpt_map)
    samples, rejected = prepare_split("test", args, dfpt_map, quality_map, unified)
    loader = make_loader(samples, cli.batch_size, False, 0, cli.device.startswith("cuda"))

    metric_rows = []
    for name, path in (
        ("historical_displacement", cli.historical_checkpoint),
        ("pbesol_increment_displacement", cli.candidate_checkpoint),
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = load_model(checkpoint, cli.device)
        evaluation_args = SimpleNamespace(**vars(args))
        if name == "historical_displacement":
            # The historical checkpoint is the unprojected control.  The
            # candidate is evaluated with the exact symmetry configuration
            # recorded in its manifest.
            evaluation_args.symmetry_projection = False
            evaluation_args.lambda_work = 0.0
            evaluation_args.lambda_reciprocity = 0.0
        metrics, rows, predictions = evaluate(
            model,
            loader,
            checkpoint["target_block_scales"].to(cli.device),
            float(checkpoint["active_threshold"]),
            torch.tensor(1.0, device=cli.device),
            torch.tensor(float(checkpoint.get("soft_mode_scale", 1.0)), device=cli.device),
            evaluation_args,
            collect=True,
            work_scale=torch.tensor(
                float(checkpoint.get("work_scale", 1.0)),
                device=cli.device,
            ),
            force_scale=torch.tensor(
                float(checkpoint.get("force_scale", 1.0)),
                device=cli.device,
            ),
        )
        metric_rows.append({"model": name, **metrics})
        pd.DataFrame(rows).to_csv(cli.outdir / f"{name}_test_predictions.tsv", sep="\t", index=False)
        torch.save(
            {"split": "test", "samples": predictions},
            cli.outdir / f"{name}_test_displacement_predictions.pt",
        )
    frame = pd.DataFrame(metric_rows)
    frame.to_csv(cli.outdir / "locked_test_metrics.tsv", sep="\t", index=False)
    keys = [
        "model",
        "lambda_component_mae",
        "lambda_component_pearson",
        "ionic_component_mae",
        "ionic_component_pearson",
        "ionic_eijmax_mae",
        "ionic_eijmax_pearson",
        "unified_ionic_component_mae",
        "unified_ionic_component_pearson",
        "unified_ionic_eijmax_mae",
        "unified_ionic_eijmax_pearson",
    ]
    print(frame[[key for key in keys if key in frame]].to_string(index=False), flush=True)
    (cli.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "n_test": len(samples),
                "rejected": rejected,
                "historical_checkpoint": str(cli.historical_checkpoint),
                "candidate_checkpoint": str(cli.candidate_checkpoint),
                "test_split_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
