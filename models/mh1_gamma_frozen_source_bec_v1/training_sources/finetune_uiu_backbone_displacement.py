#!/usr/bin/env python3
"""Fine-tune the final MACE LoRA block for internal displacement response."""

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
import torch.nn.functional as F
from mace.calculators import MACECalculator
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.utils import prepare_graph
from mace.tools import torch_geometric

from build_uiu_bec_ionic_piezo import apply_voigt_strain
from build_uiu_bec_ionic_piezo_jarvis import atoms_from_record
from evaluate_displacement_increment_locked_test import load_model as load_response_head
from train_uiu_internal_strain_head import (
    batch_loss,
    configure_response_emphasis,
    evaluate,
    fit_node_feature_scales,
    load_unified_bec_map,
    make_loader,
    move_batch,
    prepare_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("results/pbesol_finish_displacement_increment_v2"),
    )
    parser.add_argument(
        "--mace-model",
        type=Path,
        default=Path(
            "results/uiu_bec_jarvis_safe5_omat24_epoch21_finetune/"
            "MACELES-uiu-jarvis-bec-ft-best.model"
        ),
    )
    parser.add_argument(
        "--response-checkpoint",
        type=Path,
        default=Path(
            "results/uiu_displacement_response_three_way_v1/"
            "bec_visible_inverse_response/displacement_response_head.pt"
        ),
    )
    parser.add_argument(
        "--response-manifest",
        type=Path,
        default=Path(
            "results/uiu_displacement_response_three_way_v1/"
            "bec_visible_inverse_response/manifest.json"
        ),
    )
    parser.add_argument(
        "--unified-dir",
        type=Path,
        default=Path(
            "results/uiu_regularized_modal_solver_grouped_resplit_v1/"
            "physical_polarization"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/uiu_backbone_lora_displacement_safe5_v1"),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--valid-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lambda-feature-anchor", type=float, default=2.0e-3)
    parser.add_argument("--lambda-response-anchor", type=float, default=2.0e-3)
    parser.add_argument("--lambda-parameter-anchor", type=float, default=1.0e-4)
    parser.add_argument("--lambda-head-parameter-anchor", type=float, default=1.0e-4)
    parser.add_argument("--head-lr", type=float, default=1.0e-5)
    parser.add_argument("--lambda-ionic-parallel-amplitude", type=float, default=0.0)
    parser.add_argument("--lambda-ionic-perpendicular", type=float, default=0.0)
    parser.add_argument("--lambda-unified-ionic", type=float, default=0.0)
    parser.add_argument("--lambda-unified-eijmax", type=float, default=0.0)
    parser.add_argument(
        "--loss-profile",
        choices=("dft_bec", "model_bec"),
        default="dft_bec",
        help="Use the existing DFT-BEC ionic contraction objective or replace it with model-BEC terms.",
    )
    parser.add_argument(
        "--skip-locked-test",
        action="store_true",
        help="Do not read or evaluate test data; save validation-only metrics.",
    )
    parser.add_argument("--ionic-direction-threshold", type=float, default=0.05)
    parser.add_argument(
        "--variant",
        choices=(
            "single_lora",
            "full_lora",
            "head_readout",
            "full_joint",
            "six_mode_amplitude",
            "full_head",
            "full_joint_head",
            "last_block_full_head",
        ),
        default="single_lora",
        help=(
            "single_lora: existing one-mode LoRA control; full_lora: all six "
            "strain modes with last-block LoRA; head_readout: fixed epoch-120 "
            "backbone with trainable readouts; full_joint: all six modes with "
            "last-block LoRA plus readouts; six_mode_amplitude: all six "
            "strain modes with amplitude/direction and fixed-model-BEC losses; "
            "full_head: frozen backbone with every displacement-head parameter "
            "trainable; full_joint_head: last-block LoRA plus full head; "
            "last_block_full_head: all final interaction/product parameters "
            "plus full head."
        ),
    )
    parser.add_argument(
        "--fixed-backbone-model",
        type=Path,
        default=None,
        help="Optional MACE model used for fixed feature materialization in head_readout.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps; effective batch is batch-size times this value.",
    )
    parser.add_argument("--validation-interval", type=int, default=6)
    parser.add_argument(
        "--checkpoint-metric",
        choices=("loss", "ionic_eijmax_mae", "unified_ionic_eijmax_mae"),
        default="loss",
        help="Validation metric used for checkpoint selection.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many validation checks without improvement; 0 disables early stopping.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume LoRA parameters, optimizer, history, and best state.",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl_map(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["jid"]] = record
    return records


class MACEBackboneFeatureExtractor(torch.nn.Module):
    """Run MACE only through interaction/product blocks and return node_feats."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        context = prepare_graph(
            data,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            lammps_mliap=False,
        )
        node_feats = self.model.node_embedding(data["node_attrs"])
        edge_attrs = self.model.spherical_harmonics(context.vectors)
        edge_feats, cutoff = self.model.radial_embedding(
            context.lengths,
            data["node_attrs"],
            data["edge_index"],
            self.model.atomic_numbers,
        )
        if hasattr(self.model, "joint_embedding"):
            embedding_features = {
                name: data[name] for name, _ in self.model.embedding_specs.items()
            }
            node_feats = node_feats + self.model.joint_embedding(
                data["batch"], embedding_features
            )

        node_feats_list = []
        for index, (interaction, product) in enumerate(
            zip(self.model.interactions, self.model.products)
        ):
            node_feats, skip = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(index == 0),
                lammps_class=context.interaction_kwargs.lammps_class,
                lammps_natoms=context.interaction_kwargs.lammps_natoms,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=skip,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
        return torch.cat(node_feats_list, dim=-1)


class DynamicGraphFactory:
    """Build MACE graphs while caching only unstrained structures."""

    def __init__(self, records: dict[str, dict], z_table, cutoff: float) -> None:
        self.records = records
        self.z_table = z_table
        self.cutoff = cutoff
        self.atoms = {}
        self.base_graphs = {}

    def base_atoms(self, jid: str):
        if jid not in self.atoms:
            self.atoms[jid] = atoms_from_record(self.records[jid])
        return self.atoms[jid]

    def graph(self, jid: str, mode: int | None = None, sign: int = 0):
        if mode is None:
            if jid not in self.base_graphs:
                atoms = self.base_atoms(jid)
                self.base_graphs[jid] = AtomicData.from_config(
                    config_from_atoms(atoms),
                    z_table=self.z_table,
                    cutoff=self.cutoff,
                )
            return self.base_graphs[jid]
        atoms = apply_voigt_strain(
            self.base_atoms(jid),
            mode,
            float(sign) * 0.005,
        )
        return AtomicData.from_config(
            config_from_atoms(atoms),
            z_table=self.z_table,
            cutoff=self.cutoff,
        )


def dynamic_features(
    extractor: MACEBackboneFeatureExtractor,
    factory: DynamicGraphFactory,
    jids: list[str],
    cached_response: torch.Tensor,
    atom_slices: list[tuple[int, int]],
    device: str,
    modes: list[int] | None,
    mode_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dynamic base features and one-mode or full six-mode tangents."""
    full_response = modes is None and mode_indices is None
    graphs = []
    per_sample_modes = []
    for sample_index, jid in enumerate(jids):
        if mode_indices is not None:
            selected = [int(mode) for mode in mode_indices]
        elif full_response:
            selected = list(range(6))
        else:
            selected = [int(modes[sample_index])]
        per_sample_modes.append(selected)
        graphs.append(factory.graph(jid))
        for mode in selected:
            graphs.append(factory.graph(jid, mode, 1))
            graphs.append(factory.graph(jid, mode, -1))

    graph_batch = torch_geometric.Batch.from_data_list(graphs).to(device)
    features = extractor(graph_batch.to_dict())
    pointers = graph_batch.ptr.detach().cpu().tolist()
    base_features = []
    responses = []
    graph_index = 0
    for sample_index, selected in enumerate(per_sample_modes):
        start, stop = atom_slices[sample_index]
        count = stop - start
        base = features[pointers[graph_index] : pointers[graph_index + 1]]
        graph_index += 1
        if len(base) != count:
            raise ValueError(f"{jids[sample_index]} dynamic base atom mismatch")
        response = cached_response[start:stop].clone()
        for mode in selected:
            plus = features[pointers[graph_index] : pointers[graph_index + 1]]
            minus = features[pointers[graph_index + 1] : pointers[graph_index + 2]]
            graph_index += 2
            if len(plus) != count or len(minus) != count:
                raise ValueError(f"{jids[sample_index]} strained atom mismatch")
            response[:, mode, :] = (plus - minus) / 0.01
        base_features.append(base)
        responses.append(response)
    return torch.cat(base_features), torch.cat(responses)


def materialize_dynamic_samples(
    samples: list[dict],
    extractor: MACEBackboneFeatureExtractor,
    factory: DynamicGraphFactory,
    device: str,
    batch_size: int,
    progress_interval: int = 0,
    local_geometry: bool = False,
    geometry_radial_channels: int = 4,
) -> list[dict]:
    materialized = []
    extractor.eval()
    with torch.no_grad():
        for offset in range(0, len(samples), batch_size):
            chunk = samples[offset : offset + batch_size]
            cpu_batch = next(iter(make_loader(chunk, len(chunk), False, 0, False)))
            cached_response = cpu_batch["atomic_response"].to(device)
            node, response = dynamic_features(
                extractor,
                factory,
                cpu_batch["jid"],
                cached_response,
                cpu_batch["atom_slices"],
                device,
                None,
            )
            local_geometry_features = None
            local_geometry_ptr = None
            if local_geometry:
                from uiu_collective_attention_displacement import (
                    local_geometry_features_from_graph,
                )

                base_graph_batch = torch_geometric.Batch.from_data_list(
                    [factory.graph(jid) for jid in cpu_batch["jid"]]
                ).to(device)
                local_geometry_features = local_geometry_features_from_graph(
                    base_graph_batch.to_dict(),
                    factory.cutoff,
                    geometry_radial_channels,
                )
                local_geometry_ptr = base_graph_batch.ptr.detach().cpu().tolist()
            for sample_index, (sample, (start, stop)) in enumerate(
                zip(chunk, cpu_batch["atom_slices"])
            ):
                updated = dict(sample)
                updated["node_features"] = node[start:stop].detach().cpu()
                updated["atomic_response"] = response[start:stop].detach().cpu()
                if local_geometry_features is not None:
                    geometry_start = local_geometry_ptr[sample_index]
                    geometry_stop = local_geometry_ptr[sample_index + 1]
                    if geometry_stop - geometry_start != stop - start:
                        raise ValueError(
                            f"{sample['jid']} local geometry atom mismatch"
                        )
                    updated["local_geometry"] = local_geometry_features[
                        geometry_start:geometry_stop
                    ].detach().cpu()
                materialized.append(updated)
            completed = min(offset + batch_size, len(samples))
            if (
                offset == 0
                or completed >= len(samples)
                or (progress_interval > 0 and completed % progress_interval < batch_size)
            ):
                print(
                    f"dynamic materialization {completed}/"
                    f"{len(samples)}",
                    flush=True,
                )
    return materialized


def lora_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def trainable_state(
    module: torch.nn.Module,
    prefixes: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
        and (prefixes is None or name.startswith(prefixes))
    }


def load_lora_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameter_map = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameter_map[name].copy_(value.to(parameter_map[name]))


def load_trainable_state(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    parameter_map = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if name not in parameter_map:
                raise KeyError(f"missing trainable parameter in module: {name}")
            parameter_map[name].copy_(value.to(parameter_map[name]))


def configure_last_block_lora(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in model.named_parameters():
        if name.startswith(("interactions.1", "products.1")) and "LoRA_weight" in name:
            parameter.requires_grad_(True)
            trainable.append((name, parameter))
    if not trainable:
        raise RuntimeError("no final-block LoRA parameters were found")
    return trainable


def configure_variant(
    mace_model: torch.nn.Module,
    head: torch.nn.Module,
    variant: str,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    """Freeze everything except the explicitly requested MACE/head parameters."""
    for parameter in mace_model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    mace_trainable = []
    if variant in {
        "single_lora",
        "full_lora",
        "full_joint",
        "six_mode_amplitude",
        "full_joint_head",
    }:
        for name, parameter in mace_model.named_parameters():
            if name.startswith(("interactions.1", "products.1")) and "LoRA_weight" in name:
                parameter.requires_grad_(True)
                mace_trainable.append((name, parameter))
    elif variant == "last_block_full_head":
        for name, parameter in mace_model.named_parameters():
            if name.startswith(("interactions.1.", "products.1.")):
                parameter.requires_grad_(True)
                mace_trainable.append((name, parameter))
    if variant == "single_lora" and not mace_trainable:
        raise RuntimeError("no final-block LoRA parameters were found")

    head_trainable = []
    if variant in {"head_readout", "full_joint"}:
        for name, parameter in head.named_parameters():
            if name.startswith(("context_readout.", "atomic_response_readout.")):
                parameter.requires_grad_(True)
                head_trainable.append((name, parameter))
        if getattr(head, "atomic_response_mix", None) is not None:
            head.atomic_response_mix.requires_grad_(True)
            head_trainable.append(("atomic_response_mix", head.atomic_response_mix))
        if not head_trainable:
            raise RuntimeError("no response-head readout parameters were found")
    elif variant in {"full_head", "full_joint_head", "last_block_full_head"}:
        for name, parameter in head.named_parameters():
            parameter.requires_grad_(True)
            head_trainable.append((name, parameter))
        if not head_trainable:
            raise RuntimeError("the displacement head has no trainable parameters")
    return mace_trainable, head_trainable


def parameter_anchor_loss(
    trainable: list[tuple[str, torch.nn.Parameter]],
    initial: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for name, parameter in trainable:
        reference = initial[name].to(parameter)
        scale = reference.square().mean().sqrt().clamp_min(1.0e-3)
        losses.append((parameter - reference).square().mean() / scale.square())
    if not losses:
        raise ValueError("parameter_anchor_loss received no trainable parameters")
    return torch.stack(losses).mean()


def source_loss_args(cli: argparse.Namespace) -> SimpleNamespace:
    payload = json.loads(cli.response_manifest.read_text())
    args = SimpleNamespace(**payload["arguments"])
    args.device = cli.device
    args.node_cache_dir = str(cli.data_root / "cache/node")
    args.dfpt_cache = str(cli.data_root / "combined_internal_response.pt")
    args.quality_summary = str(cli.data_root / "combined_quality_summary.tsv")
    args.atomic_response_dir = str(cli.data_root / "cache/atomic_response")
    args.model_hessian_dir = None
    args.unified_prediction_dir = None
    args.hvp_labels = None
    args.max_train = cli.max_train
    args.max_valid = cli.max_valid
    args.feature_adapter = False
    args.lambda_feature_adapter = 0.0
    args.lambda_response_adapter = 0.0
    args.lambda_hvp_projected = 0.0
    args.lambda_ionic_parallel_amplitude = cli.lambda_ionic_parallel_amplitude
    args.lambda_ionic_perpendicular = cli.lambda_ionic_perpendicular
    args.lambda_unified_ionic = cli.lambda_unified_ionic
    args.lambda_unified_eijmax = cli.lambda_unified_eijmax
    args.ionic_direction_threshold = cli.ionic_direction_threshold
    if cli.loss_profile == "model_bec":
        args.lambda_ionic = 0.0
        args.lambda_weighted_dft_ionic = 0.0
        args.lambda_dft_eijmax = 0.0
    return args


def evaluate_samples(
    head,
    samples,
    checkpoint,
    loss_args,
    ionic_scale,
    collect=False,
    batch_size=64,
):
    loader = make_loader(samples, batch_size, False, 0, loss_args.device.startswith("cuda"))
    return evaluate(
        head,
        loader,
        checkpoint["target_block_scales"].to(loss_args.device),
        float(checkpoint["active_threshold"]),
        ionic_scale,
        torch.tensor(float(checkpoint.get("soft_mode_scale", 1.0)), device=loss_args.device),
        loss_args,
        collect=collect,
        work_scale=torch.tensor(float(checkpoint.get("work_scale", 1.0)), device=loss_args.device),
        force_scale=torch.tensor(float(checkpoint.get("force_scale", 1.0)), device=loss_args.device),
        hvp_scale=torch.tensor(float(checkpoint.get("hvp_scale", 1.0)), device=loss_args.device),
    )


def main() -> None:
    cli = parse_args()
    set_seed(cli.seed)
    cli.outdir.mkdir(parents=True, exist_ok=True)
    loss_args = source_loss_args(cli)
    required_splits = ("train", "valid") if cli.skip_locked_test else ("train", "valid", "test")
    record_maps = {
        split: read_jsonl_map(cli.data_root / "splits" / f"safe5_{split}.jsonl")
        for split in required_splits
    }
    requested_jids = {jid for records in record_maps.values() for jid in records}
    dfpt_payload = torch.load(loss_args.dfpt_cache, map_location="cpu", weights_only=False)
    dfpt_map = {
        sample["jid"]: sample
        for sample in dfpt_payload["samples"]
        if sample["jid"] in requested_jids
    }
    quality = pd.read_csv(loss_args.quality_summary, sep="\t")
    quality_map = {
        row["jid"]: row
        for row in quality.to_dict("records")
        if row["jid"] in requested_jids
    }
    train_unified = load_unified_bec_map(str(cli.unified_dir), "train", dfpt_map)
    valid_unified = load_unified_bec_map(str(cli.unified_dir), "valid", dfpt_map)
    train_samples, train_rejected = prepare_split(
        "train",
        loss_args,
        dfpt_map,
        quality_map,
        train_unified,
        require_unified_bec=False,
    )
    valid_samples, valid_rejected = prepare_split(
        "valid", loss_args, dfpt_map, quality_map, valid_unified
    )
    if cli.max_train is not None:
        train_samples = train_samples[: cli.max_train]
    if cli.max_valid is not None:
        valid_samples = valid_samples[: cli.max_valid]
    configure_response_emphasis(train_samples, valid_samples, loss_args)

    records = {}
    for mapping in record_maps.values():
        records.update(mapping)
    missing = [
        sample["jid"]
        for sample in train_samples + valid_samples
        if sample["jid"] not in records
    ]
    if missing:
        raise KeyError(f"missing structures for {len(missing)} samples; first={missing[0]}")

    model_path = cli.mace_model
    if cli.variant in {"head_readout", "full_head"}:
        model_path = cli.fixed_backbone_model or Path(
            "results/uiu_backbone_lora_displacement_safe5_v1/"
            "MACELES-uiu-displacement-lastblock-lora-best.model"
        )
    calc = MACECalculator(
        model_paths=str(model_path),
        device=cli.device,
        default_dtype="float32",
    )
    mace_model = calc.models[0]
    initial_mace_full = {
        name: value.detach().clone()
        for name, value in mace_model.state_dict().items()
    }
    extractor = MACEBackboneFeatureExtractor(mace_model).to(cli.device)
    factory = DynamicGraphFactory(records, calc.z_table, calc.r_max)

    response_checkpoint = torch.load(
        cli.response_checkpoint, map_location="cpu", weights_only=False
    )
    head = load_response_head(response_checkpoint, cli.device)
    initial_head_full = copy.deepcopy(head.state_dict())
    mace_trainable, head_trainable = configure_variant(
        mace_model, head, cli.variant
    )
    trainable = [
        (f"mace::{name}", parameter) for name, parameter in mace_trainable
    ] + [
        (f"head::{name}", parameter) for name, parameter in head_trainable
    ]
    if not trainable:
        raise RuntimeError(f"variant {cli.variant} has no trainable parameters")
    initial_mace = trainable_state(mace_model)
    initial_head = trainable_state(head)
    initial_trainable = {
        **{f"mace::{name}": value for name, value in initial_mace.items()},
        **{f"head::{name}": value for name, value in initial_head.items()},
    }
    head.train(bool(head_trainable))

    node_scales = fit_node_feature_scales(train_samples).to(cli.device)
    response_scales = head.atomic_response_feature_scales.to(cli.device)
    ionic_scale = torch.tensor(
        float(json.loads(cli.response_manifest.read_text())["ionic_scale"]),
        device=cli.device,
    )
    target_scales = response_checkpoint["target_block_scales"].to(cli.device)
    active_threshold = float(response_checkpoint["active_threshold"])
    soft_mode_scale = torch.tensor(
        float(response_checkpoint.get("soft_mode_scale", 1.0)), device=cli.device
    )
    work_scale = torch.tensor(
        float(response_checkpoint.get("work_scale", 1.0)), device=cli.device
    )
    force_scale = torch.tensor(
        float(response_checkpoint.get("force_scale", 1.0)), device=cli.device
    )
    hvp_scale = torch.tensor(
        float(response_checkpoint.get("hvp_scale", 1.0)), device=cli.device
    )

    # Frozen-backbone head runs reuse features from the selected starting model.
    if cli.variant in {"head_readout", "full_head"}:
        train_samples = materialize_dynamic_samples(
            train_samples, extractor, factory, cli.device, cli.valid_batch_size
        )
        valid_samples = materialize_dynamic_samples(
            valid_samples, extractor, factory, cli.device, cli.valid_batch_size
        )
        initial_dynamic_valid = valid_samples
    else:
        initial_dynamic_valid = materialize_dynamic_samples(
            valid_samples, extractor, factory, cli.device, cli.valid_batch_size
        )
    cached_metrics, _, _ = evaluate_samples(
        head, valid_samples, response_checkpoint, loss_args, ionic_scale
    )
    initial_metrics, initial_valid_rows, _ = evaluate_samples(
        head,
        initial_dynamic_valid,
        response_checkpoint,
        loss_args,
        ionic_scale,
        collect=True,
        batch_size=cli.valid_batch_size,
    )
    print(
        "epoch 0000 "
        f"cached_ionic_mae={cached_metrics['ionic_component_mae']:.6f} "
        f"dynamic_ionic_mae={initial_metrics['ionic_component_mae']:.6f} "
        f"dynamic_eijmax_mae={initial_metrics['ionic_eijmax_mae']:.6f}",
        flush=True,
    )
    if cli.checkpoint_metric not in initial_metrics:
        raise ValueError(
            f"checkpoint metric {cli.checkpoint_metric!r} is unavailable; "
            f"metrics={sorted(initial_metrics)}"
        )
    if cli.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be non-negative")

    parameter_groups = []
    if mace_trainable:
        parameter_groups.append(
            {"params": [parameter for _, parameter in mace_trainable], "lr": cli.lr}
        )
    if head_trainable:
        parameter_groups.append(
            {"params": [parameter for _, parameter in head_trainable], "lr": cli.head_lr}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
    train_loader = make_loader(
        train_samples,
        cli.batch_size,
        True,
        0,
        False,
        seed=cli.seed,
    )
    history = [
        {
            "epoch": 0,
            "mode": "all",
            "valid_loss": initial_metrics["loss"],
            "valid_ionic_component_mae": initial_metrics["ionic_component_mae"],
            "valid_ionic_component_pearson": initial_metrics["ionic_component_pearson"],
            "valid_ionic_eijmax_mae": initial_metrics["ionic_eijmax_mae"],
            "valid_ionic_eijmax_pearson": initial_metrics["ionic_eijmax_pearson"],
        }
    ]
    best_score = float(initial_metrics[cli.checkpoint_metric])
    best_epoch = 0
    validation_checks_since_improvement = 0
    best_mace = copy.deepcopy(initial_mace)
    best_head = copy.deepcopy(initial_head)
    start_epoch = 1
    optimizer_step_count = 0
    if cli.resume is not None:
        state = torch.load(cli.resume, map_location="cpu", weights_only=False)
        if "current_mace_state" in state:
            load_trainable_state(mace_model, state["current_mace_state"])
            load_trainable_state(head, state["current_head_state"])
            best_mace = state["best_mace_state"]
            best_head = state["best_head_state"]
        else:
            load_lora_state(mace_model, state["current_lora_state"])
            best_mace = state["best_lora_state"]
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = list(state["history"])
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        optimizer_step_count = int(state.get("optimizer_step_count", 0))
        start_epoch = int(state["epoch"]) + 1
        print(
            f"Resuming from epoch {state['epoch']} with best={best_epoch}",
            flush=True,
        )

    if cli.grad_accum_steps < 1:
        raise ValueError("grad-accum-steps must be positive")

    for epoch in range(start_epoch, cli.epochs + 1):
        extractor.train(bool(mace_trainable))
        head.train(bool(head_trainable))
        mode = (epoch - 1) % 6
        full_dynamic = cli.variant in {
            "full_lora",
            "full_joint",
            "six_mode_amplitude",
            "full_joint_head",
            "last_block_full_head",
        }
        totals = {
            "loss": 0.0,
            "task_loss": 0.0,
            "feature_anchor": 0.0,
            "response_anchor": 0.0,
            "parameter_anchor": 0.0,
            "head_parameter_anchor": 0.0,
        }
        count = 0
        optimizer.zero_grad(set_to_none=True)
        batches = train_loader
        num_batches = len(train_loader)
        for batch_index, cpu_batch in enumerate(batches):
            accumulation_start = (batch_index // cli.grad_accum_steps) * cli.grad_accum_steps
            accumulation_group_size = min(
                cli.grad_accum_steps,
                num_batches - accumulation_start,
            )
            mode_groups = (
                [None]
                if cli.variant in {"head_readout", "full_head"}
                else ([list(range(6))] if full_dynamic else [[mode]])
            )
            for active_modes in mode_groups:
                batch = move_batch(cpu_batch, cli.device)
                if cli.variant in {"head_readout", "full_head"}:
                    feature_anchor = torch.zeros((), device=cli.device)
                    response_anchor = torch.zeros((), device=cli.device)
                else:
                    dynamic_node, dynamic_response = dynamic_features(
                        extractor,
                        factory,
                        batch["jid"],
                        batch["atomic_response"],
                        batch["atom_slices"],
                        cli.device,
                        None if full_dynamic else [active_modes[0]] * len(batch["jid"]),
                        mode_indices=active_modes if full_dynamic else None,
                    )
                    feature_anchor = F.smooth_l1_loss(
                        (dynamic_node - batch["node_features"]) / node_scales[None, :],
                        torch.zeros_like(dynamic_node),
                        beta=0.2,
                    )
                    response_delta = (
                        dynamic_response[:, active_modes, :]
                        - batch["atomic_response"][:, active_modes, :]
                    ) / response_scales[None, None, :]
                    response_anchor = F.smooth_l1_loss(
                        response_delta,
                        torch.zeros_like(response_delta),
                        beta=0.2,
                    )
                    batch["node_features"] = dynamic_node
                    batch["atomic_response"] = dynamic_response
                task_loss, _, _, _ = batch_loss(
                    head,
                    batch,
                    target_scales,
                    active_threshold,
                    ionic_scale,
                    soft_mode_scale,
                    loss_args,
                    work_scale,
                    force_scale,
                    hvp_scale,
                )
                mace_anchor = (
                    parameter_anchor_loss(mace_trainable, initial_mace)
                    if mace_trainable
                    else torch.zeros((), device=cli.device)
                )
                head_anchor = (
                    parameter_anchor_loss(head_trainable, initial_head)
                    if head_trainable
                    else torch.zeros((), device=cli.device)
                )
                raw_loss = (
                    task_loss
                    + cli.lambda_feature_anchor * feature_anchor
                    + cli.lambda_response_anchor * response_anchor
                    + cli.lambda_parameter_anchor * mace_anchor
                    + cli.lambda_head_parameter_anchor * head_anchor
                )
                (raw_loss / (accumulation_group_size * len(mode_groups))).backward()
                graph_count = len(batch["jid"])
                totals["loss"] += float(raw_loss.detach()) * graph_count
                totals["task_loss"] += float(task_loss.detach()) * graph_count
                totals["feature_anchor"] += float(feature_anchor.detach()) * graph_count
                totals["response_anchor"] += float(response_anchor.detach()) * graph_count
                totals["parameter_anchor"] += float(mace_anchor.detach()) * graph_count
                totals["head_parameter_anchor"] += float(head_anchor.detach()) * graph_count
                count += graph_count
            last_batch = batch_index + 1 == num_batches
            update_step = (
                (batch_index + 1) % cli.grad_accum_steps == 0
                or last_batch
            )
            if update_step:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in trainable], cli.gradient_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1
        row = {
            "epoch": epoch,
            "mode": "all"
            if full_dynamic or cli.variant in {"head_readout", "full_head"}
            else mode,
            "lr": optimizer.param_groups[0]["lr"],
            "mace_lr": cli.lr if mace_trainable else 0.0,
            "head_lr": cli.head_lr if head_trainable else 0.0,
            **{f"train_{key}": value / count for key, value in totals.items()},
        }

        if epoch % cli.validation_interval == 0 or epoch == cli.epochs:
            if cli.variant in {"head_readout", "full_head"}:
                dynamic_valid = valid_samples
            else:
                dynamic_valid = materialize_dynamic_samples(
                    valid_samples, extractor, factory, cli.device, cli.valid_batch_size
                )
            valid_metrics, _, _ = evaluate_samples(
                head,
                dynamic_valid,
                response_checkpoint,
                loss_args,
                ionic_scale,
            )
            row.update({f"valid_{key}": value for key, value in valid_metrics.items()})
            selection_value = float(valid_metrics[cli.checkpoint_metric])
            improved = np.isfinite(selection_value) and selection_value < best_score
            if improved:
                best_score = selection_value
                best_epoch = epoch
                best_mace = trainable_state(mace_model)
                best_head = trainable_state(head)
                validation_checks_since_improvement = 0
            else:
                validation_checks_since_improvement += 1
            print(
                f"epoch {epoch:04d} mode={'all' if full_dynamic else mode} train={row['train_loss']:.6f} "
                f"valid={valid_metrics['loss']:.6f} "
                f"ionic_mae={valid_metrics['ionic_component_mae']:.6f} "
                f"eijmax_mae={valid_metrics['ionic_eijmax_mae']:.6f} "
                f"eijmax_r={valid_metrics['ionic_eijmax_pearson']:.6f} "
                f"select_{cli.checkpoint_metric}={selection_value:.6f} "
                f"best={best_epoch}",
                flush=True,
            )
        else:
            print(
                f"epoch {epoch:04d} mode={'all' if full_dynamic else mode} train={row['train_loss']:.6f}",
                flush=True,
            )
        history.append(row)
        torch.save(
            {
                "epoch": epoch,
                "optimizer_step_count": optimizer_step_count,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "checkpoint_metric": cli.checkpoint_metric,
                "variant": cli.variant,
                "current_mace_state": trainable_state(mace_model),
                "best_mace_state": best_mace,
                "current_head_state": trainable_state(head),
                "best_head_state": best_head,
                # Keep the old names so the original single_lora run remains
                # readable by older analysis tools.
                "current_lora_state": trainable_state(mace_model),
                "best_lora_state": best_mace,
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            cli.outdir / "training_state.pt",
        )
        if (
            cli.early_stopping_patience > 0
            and epoch % cli.validation_interval == 0
            and validation_checks_since_improvement >= cli.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: "
                f"{cli.checkpoint_metric} did not improve for "
                f"{validation_checks_since_improvement} validation checks; "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            break

    load_trainable_state(mace_model, best_mace)
    load_trainable_state(head, best_head)
    pd.DataFrame(history).to_csv(cli.outdir / "history.tsv", sep="\t", index=False)
    torch.save(
        {
            "base_mace_model": str(model_path),
            "response_checkpoint": str(cli.response_checkpoint),
            "trainable_state_dict": best_mace,
            "trainable_parameter_names": [name for name, _ in mace_trainable],
            "best_epoch": best_epoch,
            "best_valid_loss": best_score,
            "checkpoint_metric": cli.checkpoint_metric,
            "early_stopping_patience": cli.early_stopping_patience,
            "variant": cli.variant,
        },
        cli.outdir / "mace_trainable_state.pt",
    )
    if cli.variant in {
        "single_lora",
        "full_lora",
        "full_joint",
        "six_mode_amplitude",
        "full_joint_head",
    }:
        torch.save(
            {
                "base_mace_model": str(model_path),
                "response_checkpoint": str(cli.response_checkpoint),
                "lora_state_dict": best_mace,
                "trainable_parameter_names": [name for name, _ in mace_trainable],
                "best_epoch": best_epoch,
                "best_valid_loss": best_score,
                "checkpoint_metric": cli.checkpoint_metric,
                "early_stopping_patience": cli.early_stopping_patience,
                "variant": cli.variant,
            },
            cli.outdir / "mace_backbone_lora.pt",
        )
    for parameter in mace_model.parameters():
        parameter.requires_grad_(False)
    torch.save(mace_model, cli.outdir / "MACELES-uiu-displacement-scope-best.model")
    head_checkpoint = dict(response_checkpoint)
    head_checkpoint["model_state_dict"] = {
        name: value.detach().cpu().clone() for name, value in head.state_dict().items()
    }
    head_checkpoint["variant"] = cli.variant
    torch.save(head_checkpoint, cli.outdir / "response_head_best.pt")

    selected_valid_samples = valid_samples
    if cli.variant not in {"head_readout", "full_head"}:
        selected_valid_samples = materialize_dynamic_samples(
            valid_samples, extractor, factory, cli.device, cli.valid_batch_size
        )
    selected_valid_metrics, valid_rows, _ = evaluate_samples(
        head,
        selected_valid_samples,
        response_checkpoint,
        loss_args,
        ionic_scale,
        collect=True,
        batch_size=cli.valid_batch_size,
    )
    valid_metric_frame = pd.DataFrame(
        [
            {"model": "source_starting_model", **initial_metrics},
            {"model": cli.variant, **selected_valid_metrics},
        ]
    )
    valid_metric_frame.to_csv(cli.outdir / "valid_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(initial_valid_rows).to_csv(
        cli.outdir / "source_valid_predictions.tsv", sep="\t", index=False
    )
    pd.DataFrame(valid_rows).to_csv(
        cli.outdir / "candidate_valid_predictions.tsv", sep="\t", index=False
    )

    test_samples = []
    test_rejected = None
    source_metrics = None
    candidate_metrics = None
    metric_frame = valid_metric_frame
    if not cli.skip_locked_test:
        loss_args.max_valid = cli.max_test
        unified = load_unified_bec_map(str(cli.unified_dir), "test", dfpt_map)
        test_samples, test_rejected = prepare_split(
            "test", loss_args, dfpt_map, quality_map, unified
        )
        if cli.max_test is not None:
            test_samples = test_samples[: cli.max_test]
        mace_model.load_state_dict(initial_mace_full)
        head.load_state_dict(initial_head_full)
        source_dynamic_test = materialize_dynamic_samples(
            test_samples, extractor, factory, cli.device, cli.valid_batch_size
        )
        source_metrics, source_rows, _ = evaluate_samples(
            head,
            source_dynamic_test,
            response_checkpoint,
            loss_args,
            ionic_scale,
            collect=True,
            batch_size=cli.valid_batch_size,
        )
        load_trainable_state(mace_model, best_mace)
        load_trainable_state(head, best_head)
        dynamic_test = materialize_dynamic_samples(
            test_samples, extractor, factory, cli.device, cli.valid_batch_size
        )
        candidate_metrics, candidate_rows, _ = evaluate_samples(
            head,
            dynamic_test,
            response_checkpoint,
            loss_args,
            ionic_scale,
            collect=True,
            batch_size=cli.valid_batch_size,
        )
        metric_frame = pd.DataFrame(
            [
                {"model": "source_starting_model", **source_metrics},
                {"model": cli.variant, **candidate_metrics},
            ]
        )
        metric_frame.to_csv(cli.outdir / "locked_test_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(source_rows).to_csv(
            cli.outdir / "source_test_predictions.tsv", sep="\t", index=False
        )
        pd.DataFrame(candidate_rows).to_csv(
            cli.outdir / "candidate_test_predictions.tsv", sep="\t", index=False
        )
    manifest = {
        "objective": "controlled MACE backbone/readout fine-tuning for displacement response",
        "variant": cli.variant,
        "loss_profile": cli.loss_profile,
        "test_used": not cli.skip_locked_test,
        "locked_test_evaluated": not cli.skip_locked_test,
        "combined_dfpt_cache_deserialized": True,
        "combined_dfpt_cache_may_contain_locked_test_records": True,
        "locked_test_rows_selected": not cli.skip_locked_test,
        "locked_test_used_for_training_or_selection": False,
        "active_loss_weights": {
            "lambda_ionic_dft_bec": loss_args.lambda_ionic,
            "lambda_weighted_dft_ionic": loss_args.lambda_weighted_dft_ionic,
            "lambda_dft_eijmax": loss_args.lambda_dft_eijmax,
            "lambda_unified_ionic_model_bec": loss_args.lambda_unified_ionic,
            "lambda_unified_eijmax_model_bec": loss_args.lambda_unified_eijmax,
            "lambda_tensor": 1.0,
        },
        "base_mace_model": str(model_path),
        "frozen_response_checkpoint": str(cli.response_checkpoint),
        "fixed_model_bec_source": json.loads(
            (cli.unified_dir / "manifest.json").read_text(encoding="utf-8")
        ).get("model"),
        "six_mode_training": {
            "modes": ["xx", "yy", "zz", "yz", "xz", "xy"],
            "central_difference_strain": 0.005,
            "all_modes_in_each_optimizer_update": cli.variant
            in {
                "full_lora",
                "full_joint",
                "six_mode_amplitude",
                "full_joint_head",
                "last_block_full_head",
            }
        },
        "added_loss_weights": {
            "ionic_parallel_amplitude": cli.lambda_ionic_parallel_amplitude,
            "ionic_perpendicular": cli.lambda_ionic_perpendicular,
            "fixed_model_bec_ionic": cli.lambda_unified_ionic,
            "fixed_model_bec_eijmax": cli.lambda_unified_eijmax,
            "ionic_direction_threshold_Cm2": cli.ionic_direction_threshold,
        },
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "trainable_tensor_count": len(trainable),
        "mace_trainable_parameter_count": sum(parameter.numel() for _, parameter in mace_trainable),
        "head_trainable_parameter_count": sum(parameter.numel() for _, parameter in head_trainable),
        "mace_trainable_parameter_names": [name for name, _ in mace_trainable],
        "head_trainable_parameter_names": [name for name, _ in head_trainable],
        "optimizer_step_count": optimizer_step_count,
        "starting_backbone_model": str(model_path),
        "starting_backbone_state_restored_for_source_test": not cli.skip_locked_test,
        "n_train": len(train_samples),
        "fixed_model_bec_train_coverage": {
            "available": sum(
                bool(sample.get("unified_bec_available", False))
                for sample in train_samples
            ),
            "total": len(train_samples),
            "missing_or_atom_mismatched": sum(
                not bool(sample.get("unified_bec_available", False))
                for sample in train_samples
            ),
        },
        "fixed_model_bec_valid_coverage": {
            "available": sum(
                bool(sample.get("unified_bec_available", False))
                for sample in valid_samples
            ),
            "total": len(valid_samples),
        },
        "fixed_model_bec_test_coverage": None if cli.skip_locked_test else {
            "available": sum(
                bool(sample.get("unified_bec_available", False))
                for sample in test_samples
            ),
            "total": len(test_samples),
        },
        "n_valid": len(valid_samples),
        "n_test": None if cli.skip_locked_test else len(test_samples),
        "train_rejected": train_rejected,
        "valid_rejected": valid_rejected,
        "test_rejected": test_rejected,
        "best_epoch": best_epoch,
        "best_valid_loss": best_score if cli.checkpoint_metric == "loss" else None,
        "best_valid_metric": best_score,
        "epoch0_cached_valid": cached_metrics,
        "epoch0_dynamic_valid": initial_metrics,
        "selected_checkpoint_valid": selected_valid_metrics,
        "source_locked_test": source_metrics,
        "candidate_locked_test": candidate_metrics,
        "effective_batch_size": cli.batch_size * cli.grad_accum_steps,
        "optimizer_step_count": optimizer_step_count,
        "optimizer_steps_per_epoch": int(np.ceil(len(train_loader) / cli.grad_accum_steps)),
        "arguments": vars(cli),
    }
    manifest["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in manifest["arguments"].items()
    }
    (cli.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(("VALIDATION METRICS" if cli.skip_locked_test else "LOCKED TEST METRICS"), flush=True)
    print(metric_frame[[
        "model",
        "ionic_component_mae",
        "ionic_component_pearson",
        "ionic_eijmax_mae",
        "ionic_eijmax_pearson",
        "unified_ionic_component_mae",
        "unified_ionic_eijmax_mae",
        "unified_ionic_eijmax_pearson",
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
