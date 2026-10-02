#!/usr/bin/env python3
"""Run matched C0/C1/C2 task adaptation from a frozen-probe checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from e3nn import o3

from foundation_displacement_extract import MACEInteractionFeatures, NequIPInteractionFeatures
from foundation_displacement_train import (
    DEFAULT_ROOT,
    ProjectedResponseModel,
    FeatureProjection,
    attach_cache,
    fit_irrep_scales,
    load_fixed_samples,
    ref_score,
)
from build_uiu_bec_ionic_piezo import apply_voigt_strain
from build_uiu_bec_ionic_piezo_jarvis import atoms_from_record
from train_uiu_internal_strain_head import (
    batch_loss,
    evaluate,
    fit_hvp_projected_scale,
    fit_soft_mode_scale,
    fit_work_scale,
    make_loader,
    move_batch,
)
from uiu_equivariant_piezo_model import DEFAULT_NODE_IRREPS
from uiu_internal_strain_model import AtomicInternalStrainHead


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_records(data_root: Path, split: str):
    result = {}
    with (data_root / "splits" / f"safe5_{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                result[item["jid"]] = item
    return result


class LiveBackend:
    def __init__(self, backend, model_path: Path, head: str | None, device: str,
                 feature_stage: str = "pre_product", lora_rank: int = 8,
                 apply_lora: bool = False):
        self.backend_name = backend
        self.device = device
        if backend == "mace":
            from mace.calculators import MACECalculator

            self.AtomicData = __import__("mace.data", fromlist=["AtomicData"]).AtomicData
            self.config_from_atoms = __import__("mace.data.utils", fromlist=["config_from_atoms"]).config_from_atoms
            self.torch_geometric = __import__("mace.tools", fromlist=["torch_geometric"]).torch_geometric
            self.calc = MACECalculator(
                model_paths=str(model_path), device=device, default_dtype="float32", head=head
            )
            self.extractor = MACEInteractionFeatures(
                self.calc.models[0], feature_stage=feature_stage
            ).to(device)
            self.model = self.calc.models[0]
            self.lora_adapters = []
            if apply_lora:
                self.lora_adapters = install_equivariant_lora(
                    self.model.interactions[-1], rank=lora_rank
                )
            self.source_irreps = [o3.Irreps(value) for value in self.extractor.source_irreps]
        else:
            self.extractor = NequIPInteractionFeatures(model_path, device)
            self.model = self.extractor.model
            self.source_irreps = self.extractor.irreps

    def graph(self, atoms):
        if self.backend_name == "mace":
            config = self.config_from_atoms(atoms, head_name=self.calc.head)
            return self.AtomicData.from_config(
                config, z_table=self.calc.z_table, cutoff=self.calc.r_max
            )
        return self.extractor.graph(atoms)

    def forward(self, graphs):
        if self.backend_name == "mace":
            batch = self.torch_geometric.Batch.from_data_list(graphs).to(self.device)
            return self.extractor(batch.to_dict())
        return self.extractor.forward(graphs)

    def trainable_parameters(self, phase):
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        selected = []
        if self.backend_name == "mace":
            modules = []
            if phase == "LoRA":
                for adapter in self.lora_adapters:
                    for parameter in list(adapter.lora_a) + list(adapter.lora_b):
                        parameter.requires_grad_(True)
                        selected.append(parameter)
            elif phase == "C1":
                modules = [self.model.interactions[-1]]
                if self.extractor.feature_stage == "post_product":
                    modules.append(self.model.products[-1])
            elif phase == "C2":
                modules = [self.model.node_embedding]
                if hasattr(self.model, "radial_embedding"):
                    modules.append(self.model.radial_embedding)
                modules += list(self.model.interactions)
                modules += list(self.model.products if self.extractor.feature_stage == "post_product"
                                 else self.model.products[:-1])
                if hasattr(self.model, "joint_embedding"):
                    modules.append(self.model.joint_embedding)
            else:
                return []
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
                    selected.append(parameter)
        else:
            if phase == "C1":
                modules = [self.extractor.modules[-1][1]]
            elif phase == "C2":
                modules = []
                for name, module in self.extractor.func.named_children():
                    if name == "type_embed" or name.startswith("layer") and name.endswith("_convnet"):
                        modules.append(module)
            else:
                return []
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
                    selected.append(parameter)
        # De-duplicate parameters if nested module lists overlap.
        unique = []
        seen = set()
        for parameter in selected:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                unique.append(parameter)
        return unique


class EquivariantLinearLoRA(torch.nn.Module):
    """Low-rank channel updates for each allowed e3nn Linear irrep path."""

    def __init__(self, base: torch.nn.Module, rank: int = 8):
        super().__init__()
        self.base = base
        self.irreps_in = o3.Irreps(base.irreps_in)
        self.irreps_out = o3.Irreps(base.irreps_out)
        self.instructions = [
            ins for ins in base.instructions
            if ins.i_in >= 0 and ins.i_out >= 0 and all(size > 0 for size in ins.path_shape)
        ]
        self.input_slices = self.irreps_in.slices()
        self.output_slices = self.irreps_out.slices()
        self.ranks = []
        self.lora_a = torch.nn.ParameterList()
        self.lora_b = torch.nn.ParameterList()
        for instruction in self.instructions:
            mul_in, mul_out = instruction.path_shape
            local_rank = min(rank, mul_in, mul_out)
            self.ranks.append(local_rank)
            a = torch.nn.Parameter(torch.empty(mul_in, local_rank))
            b = torch.nn.Parameter(torch.zeros(local_rank, mul_out))
            torch.nn.init.kaiming_uniform_(a, a=5**0.5)
            self.lora_a.append(a)
            self.lora_b.append(b)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        reference = getattr(base, "weight", None)
        if isinstance(reference, torch.Tensor) and reference.numel() > 0:
            self.to(device=reference.device, dtype=reference.dtype)

    def forward(self, features, *args, **kwargs):
        output = self.base(features, *args, **kwargs)
        if not self.instructions:
            return output
        updates = {}
        for instruction, rank, a, b in zip(
            self.instructions, self.ranks, self.lora_a, self.lora_b
        ):
            _mul_in, ir_in = self.irreps_in[instruction.i_in]
            _mul_out, ir_out = self.irreps_out[instruction.i_out]
            if ir_in != ir_out:
                raise RuntimeError("LoRA path changes irrep type and would break equivariance")
            in_slice = self.input_slices[instruction.i_in]
            out_slice = self.output_slices[instruction.i_out]
            block = features[..., in_slice].reshape(
                *features.shape[:-1], instruction.path_shape[0], ir_in.dim
            )
            delta = torch.einsum("...md,mr,ro->...od", block, a, b)
            updates[instruction.i_out] = (
                delta if instruction.i_out not in updates
                else updates[instruction.i_out] + delta
            )
        result = []
        for out_index, (multiplicity, irrep) in enumerate(self.irreps_out):
            base_block = output[..., self.output_slices[out_index]].reshape(
                *output.shape[:-1], multiplicity, irrep.dim
            )
            result.append(
                (base_block + updates[out_index]).flatten(start_dim=-2)
                if out_index in updates else base_block.flatten(start_dim=-2)
            )
        return torch.cat(result, dim=-1)


def install_equivariant_lora(module: torch.nn.Module, rank: int = 8):
    """Wrap e3nn Linear descendants with zero-initialized equivariant LoRA paths."""
    adapters = []

    def visit(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, o3.Linear):
                adapter = EquivariantLinearLoRA(child, rank=rank)
                setattr(parent, name, adapter)
                adapters.append(adapter)
            else:
                visit(child)

    visit(module)
    if not adapters:
        raise RuntimeError("no e3nn Linear modules found for equivariant LoRA")
    return adapters


def graph_cache(backend: LiveBackend, samples, records, delta=0.005):
    cache = {}
    for sample in samples:
        jid = sample["jid"]
        atoms = atoms_from_record(records[jid])
        graphs = [backend.graph(atoms)]
        for mode in range(6):
            graphs.append(backend.graph(apply_voigt_strain(atoms, mode, delta)))
            graphs.append(backend.graph(apply_voigt_strain(atoms, mode, -delta)))
        cache[jid] = graphs
    return cache


def dynamic_features(backend: LiveBackend, jids, graphs_by_jid):
    graphs = []
    counts = []
    for jid in jids:
        graphs.extend(graphs_by_jid[jid])
        counts.append(len(graphs_by_jid[jid][0]["positions"]) if backend.backend_name == "mace" else len(graphs_by_jid[jid][0]["pos"]))
    features = backend.forward(graphs)
    pointers = [0]
    for count in counts:
        pointers.append(pointers[-1] + count * 13)
    bases, responses = [], []
    atom_offset = 0
    for local, count in enumerate(counts):
        start = pointers[local]
        base = features[start : start + count]
        bases.append(base)
        response = []
        for mode in range(6):
            plus_idx = start + count * (1 + 2 * mode)
            minus_idx = plus_idx + count
            response.append((features[plus_idx : plus_idx + count] - features[minus_idx : minus_idx + count]) / 0.01)
        responses.append(torch.stack(response, dim=1))
        atom_offset += count
    return torch.cat(bases), torch.cat(responses)


def load_probe_model(model_id, root, train, metadata, device, probe_subdir="frozen_probe"):
    checkpoint = torch.load(
        root / probe_subdir / model_id / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    blocks = [o3.Irreps(item) for item in checkpoint["source_irreps"]]
    feature_scales = checkpoint["source_feature_scales"]
    response_scales = checkpoint["source_response_scales"]
    projection = FeatureProjection(
        blocks,
        feature_scales,
        response_scales,
        channels=checkpoint.get("projection_channels", 128),
        native_input=checkpoint.get("native_input", False),
        input_rms_normalization=checkpoint.get("input_rms_normalization"),
        identity_init=checkpoint.get("identity_init", False),
    ).to(device)
    head = AtomicInternalStrainHead(
        node_irreps=projection.public_irreps,
        scalar_channels=32,
        vector_channels=16,
        tensor_channels=16,
        atomic_response=True,
        atomic_response_feature_scales=checkpoint["atomic_response_feature_scales"],
        atomic_response_mix_init=0.1,
        output_block_scales=checkpoint["target_block_scales"],
    )
    model = ProjectedResponseModel(projection, head).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def fit_dynamic_samples(backend, samples, graphs_by_jid, structure_batch=8):
    result = []
    if hasattr(backend.extractor, "eval"):
        backend.extractor.eval()
    else:
        backend.model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), structure_batch):
            chunk = samples[start : start + structure_batch]
            node, response = dynamic_features(
                backend, [sample["jid"] for sample in chunk], graphs_by_jid
            )
            offset = 0
            for sample in chunk:
                count = sample["target_irreps"].shape[0]
                updated = dict(sample)
                updated["node_features"] = node[offset : offset + count].detach().cpu()
                updated["atomic_response"] = response[offset : offset + count].detach().cpu()
                result.append(updated)
                offset += count
    return result


def normalized_anchor(current, reference, scales):
    residual = (current - reference) / scales.clamp_min(1.0e-3)
    return F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=0.2)


def parameter_anchor(parameters, initial):
    terms = []
    for parameter, reference in zip(parameters, initial):
        reference = reference.to(parameter)
        scale = reference.square().mean().sqrt().clamp_min(1.0e-3)
        terms.append((parameter - reference).square().mean() / scale.square())
    return torch.stack(terms).mean() if terms else torch.tensor(0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--run-id",
        help="Unique directory key for this seed/probe/phase family; defaults to model-id.",
    )
    parser.add_argument("--probe-model-id")
    parser.add_argument(
        "--probe-subdir", default="frozen_probe",
        help="Subdirectory containing the frozen-probe checkpoint for this backbone.",
    )
    parser.add_argument("--cache-model-id")
    parser.add_argument("--backend", choices=("mace", "nequip"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--head", default="default")
    parser.add_argument("--phase", choices=("C0", "C1", "C2", "LoRA"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("results/pbesol_finish_displacement_increment_v2"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument(
        "--match-microbatch-control",
        action="store_true",
        help="Use the configured micro-batch for C0 too, matching C1's per-structure loss context.",
    )
    parser.add_argument("--valid-batch-size", type=int, default=16)
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from this run's best checkpoint; optimizer state is restored when available.",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    run_id = args.run_id or args.model_id
    outdir = args.root / "task_finetune" / run_id / args.phase
    outdir.mkdir(parents=True, exist_ok=True)

    loss_args, train, valid, rejected = load_fixed_samples(args.device)
    loss_args.response_weight_mean = float(
        torch.tensor([float(sample["response_weight"]) for sample in train]).mean()
    )
    cache_model_id = args.cache_model_id or args.model_id
    probe_model_id = args.probe_model_id or args.model_id
    probe_checkpoint_path = args.root / args.probe_subdir / probe_model_id / "best_checkpoint.pt"
    probe_checkpoint_sha256 = hashlib.sha256(probe_checkpoint_path.read_bytes()).hexdigest()
    metadata = attach_cache(train, args.root / "cache" / cache_model_id / "train_features.pt")
    valid_metadata = attach_cache(valid, args.root / "cache" / cache_model_id / "valid_features.pt")
    if metadata["model_sha256"] != valid_metadata["model_sha256"]:
        raise ValueError("Train and validation caches use different backbone weights")
    model, probe = load_probe_model(
        probe_model_id, args.root, train, metadata, args.device,
        probe_subdir=args.probe_subdir,
    )
    target_scales = probe["target_block_scales"].to(args.device)
    active_threshold = float(probe["active_threshold"])
    ionic_scale = torch.tensor(float(probe["ionic_scale"]), device=args.device)
    soft_mode_scale = torch.tensor(float(probe["soft_mode_scale"]), device=args.device)
    work_scale = torch.tensor(float(probe["work_scale"]), device=args.device)
    force_scale = torch.tensor(float(probe["force_scale"]), device=args.device)
    hvp_scale = torch.tensor(float(probe["hvp_scale"]), device=args.device)
    ref = json.loads((args.root / "ref41_valid_metrics.json").read_text())

    backend = None
    graphs_train = graphs_valid = None
    backbone_parameters = []
    if args.phase in {"C1", "C2", "LoRA"}:
        backend = LiveBackend(
            args.backend,
            args.model_path,
            args.head,
            args.device,
            feature_stage=probe.get("feature_stage", metadata.get("feature_stage", "pre_product")),
            lora_rank=args.lora_rank,
            apply_lora=args.phase == "LoRA",
        )
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        backbone_parameters = backend.trainable_parameters(args.phase)
        if not backbone_parameters:
            raise RuntimeError(f"No backbone parameters selected for {args.phase}")
        records = {}
        records.update(read_records(args.data_root, "train"))
        records.update(read_records(args.data_root, "valid"))
        graphs_train = graph_cache(backend, train, records)
        graphs_valid = graph_cache(backend, valid, records)
        # Verify the live graph path against the frozen cache before updating.
        with torch.no_grad():
            node, response = dynamic_features(backend, [train[0]["jid"]], graphs_train)
        n = train[0]["node_features"].shape[0]
        node_error = float((node.cpu() - train[0]["node_features"]).abs().max())
        response_error = float((response.cpu() - train[0]["atomic_response"]).abs().max())
        if node_error > 2.0e-4 or response_error > 5.0e-3:
            raise RuntimeError(
                f"live/frozen feature mismatch: node={node_error:.3g}, response={response_error:.3g}"
            )
        print(f"live extractor parity: node max abs={node_error:.3g}; tangent max abs={response_error:.3g}", flush=True)

    initial_backbone = None
    if backbone_parameters:
        initial_backbone = [p.detach().clone() for p in backbone_parameters]
    projection_head_parameters = list(model.projection.parameters()) + list(model.head.parameters())
    for parameter in projection_head_parameters:
        parameter.requires_grad_(True)
    initial_projection_head = [p.detach().clone() for p in projection_head_parameters]

    if args.phase == "C0" and not args.match_microbatch_control:
        batch_size = args.effective_batch_size
    else:
        batch_size = args.micro_batch_size
    accumulation = max(1, args.effective_batch_size // batch_size)
    train_loader = make_loader(
        train,
        batch_size,
        True,
        0,
        args.device.startswith("cuda"),
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = make_loader(valid, args.valid_batch_size, False, 0, args.device.startswith("cuda"))
    parameter_groups = [{"params": projection_head_parameters, "lr": 1.0e-5}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": 2.0e-6})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
    resume_checkpoint = None
    start_epoch = 1
    best_score, best_epoch, best_state, best_backbone_state, stale = float("inf"), -1, None, None, 0
    history = []
    if args.resume_from_checkpoint:
        checkpoint_path = outdir / "best_checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("model_id") != args.model_id or resume_checkpoint.get("phase") != args.phase:
            raise ValueError("Resume checkpoint model/phase does not match this run")
        if resume_checkpoint.get("source_sha256") != metadata["model_sha256"]:
            raise ValueError("Resume checkpoint was created from a different backbone weight")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        if backend is not None:
            backbone_state = resume_checkpoint.get("backbone_state_dict")
            if backbone_state is None:
                raise ValueError("Backbone-phase resume checkpoint has no backbone state")
            backend.model.load_state_dict(backbone_state, strict=True)
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in resume_checkpoint["model_state_dict"].items()
        }
        best_backbone_state = (
            {
                key: value.detach().cpu().clone()
                for key, value in resume_checkpoint["backbone_state_dict"].items()
            }
            if resume_checkpoint.get("backbone_state_dict") is not None
            else None
        )
        best_epoch = int(resume_checkpoint["best_epoch"])
        best_score = float(resume_checkpoint["best_selection_score"])
        stale = int(resume_checkpoint.get("stale", 0))
        start_epoch = best_epoch + 1
        optimizer_state_resumed = "optimizer_state_dict" in resume_checkpoint
        if optimizer_state_resumed:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        history_path = outdir / "history.tsv"
        if history_path.is_file():
            history = pd.read_csv(history_path, sep="\t").to_dict("records")
            history = [row for row in history if int(row["epoch"]) <= best_epoch]
        else:
            history = list(resume_checkpoint.get("history", []))
        print(
            f"resuming {args.model_id} {args.phase} from best epoch {best_epoch} "
            f"(score={best_score:.5f}, optimizer state restored={optimizer_state_resumed})",
            flush=True,
        )
    else:
        optimizer_state_resumed = False
    backbone_gradient_norm = None
    if backend is not None:
        # Prove that the finite-difference feature path remains differentiable
        # through the selected native block before spending a full run.
        audit_loader = make_loader(
            train,
            min(args.micro_batch_size, len(train)),
            False,
            0,
            False,
            generator=torch.Generator().manual_seed(args.seed + 20260928),
        )
        audit_batch = move_batch(next(iter(audit_loader)), args.device)
        backend.model.train()
        audit_node, audit_response = dynamic_features(
            backend, audit_batch["jid"], graphs_train
        )
        audit_batch["node_features"] = audit_node
        audit_batch["atomic_response"] = audit_response
        audit_loss, _, _, _ = batch_loss(
            model,
            audit_batch,
            target_scales,
            active_threshold,
            ionic_scale,
            soft_mode_scale,
            loss_args,
            work_scale,
            force_scale,
            hvp_scale,
        )
        audit_loss.backward()
        squared_gradients = [
            parameter.grad.detach().float().square().sum()
            for parameter in backbone_parameters
            if parameter.grad is not None
        ]
        backbone_gradient_norm = (
            float(torch.stack(squared_gradients).sum().sqrt())
            if squared_gradients
            else 0.0
        )
        optimizer.zero_grad(set_to_none=True)
        if not np.isfinite(backbone_gradient_norm) or backbone_gradient_norm <= 0.0:
            raise RuntimeError(
                f"No finite nonzero response-loss gradient reached the {args.phase} backbone"
            )
        print(
            f"backbone gradient audit: ||grad||={backbone_gradient_norm:.6g}",
            flush=True,
        )
    baseline_valid = valid
    if backend is not None:
        baseline_valid = fit_dynamic_samples(backend, valid, graphs_valid, structure_batch=8)
    initial_metrics, _, _ = evaluate(
        model,
        make_loader(baseline_valid, args.valid_batch_size, False, 0, args.device.startswith("cuda")),
        target_scales,
        active_threshold,
        ionic_scale,
        soft_mode_scale,
        loss_args,
        work_scale=work_scale,
        force_scale=force_scale,
        hvp_scale=hvp_scale,
    )
    initial_score = ref_score(initial_metrics, ref)
    print(f"{args.model_id} {args.phase} initial score={initial_score:.5f}", flush=True)
    if resume_checkpoint is None:
        # Treat the frozen probe as an explicit epoch-0 candidate. Fine-tuning
        # should never force selection of a checkpoint worse than its start.
        best_score, best_epoch, stale = initial_score, 0, 0
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_backbone_state = (
            {
                key: value.detach().cpu().clone()
                for key, value in backend.model.state_dict().items()
            }
            if backend is not None else None
        )
        initial_checkpoint = {
            "model_state_dict": best_state,
            "backbone_state_dict": best_backbone_state,
            "model_id": args.model_id,
            "phase": args.phase,
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "source_sha256": metadata["model_sha256"],
            "optimizer_state_dict": optimizer.state_dict(),
            "stale": stale,
            "history": history,
        }
        torch.save(initial_checkpoint, outdir / "best_checkpoint.pt")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if backend is not None:
            backend.model.train()
        running = 0.0
        batches_in_group = 0
        optimizer.zero_grad(set_to_none=True)
        for cpu_batch in train_loader:
            batch = move_batch(cpu_batch, args.device)
            loss, _, _, _ = batch_loss(
                model,
                batch,
                target_scales,
                active_threshold,
                ionic_scale,
                soft_mode_scale,
                loss_args,
                work_scale,
                force_scale,
                hvp_scale,
            ) if backend is None else (None, None, None, None)
            if backend is not None:
                node, response = dynamic_features(backend, batch["jid"], graphs_train)
                reference_node = batch["node_features"]
                reference_response = batch["atomic_response"]
                batch["node_features"] = node
                batch["atomic_response"] = response
                loss, _, _, _ = batch_loss(
                    model,
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
                feature_anchor = normalized_anchor(
                    node, reference_node, model.projection.feature_scales
                )
                response_anchor = normalized_anchor(
                    response, reference_response, model.projection.response_scales
                )
                loss = loss + 2.0e-3 * feature_anchor + 2.0e-3 * response_anchor
            else:
                feature_anchor = loss.new_zeros(())
                response_anchor = loss.new_zeros(())
            current_trainables = projection_head_parameters + backbone_parameters
            references = initial_projection_head + (initial_backbone or [])
            loss = loss + 1.0e-4 * parameter_anchor(current_trainables, references)
            (loss / accumulation).backward()
            running += float(loss.detach())
            batches_in_group += 1
            if batches_in_group == accumulation:
                torch.nn.utils.clip_grad_norm_(current_trainables, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                batches_in_group = 0
        if batches_in_group:
            # The final accumulation group contains the 8 remaining training
            # structures, so average it over the micro-batches it actually has.
            partial_group_scale = accumulation / batches_in_group
            for parameter in current_trainables:
                if parameter.grad is not None:
                    parameter.grad.mul_(partial_group_scale)
            torch.nn.utils.clip_grad_norm_(current_trainables, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if epoch % 5 != 0 and epoch != args.epochs:
            continue
        if backend is not None:
            valid_for_epoch = fit_dynamic_samples(backend, valid, graphs_valid, structure_batch=8)
        else:
            valid_for_epoch = valid
        metrics, _, _ = evaluate(
            model,
            make_loader(valid_for_epoch, args.valid_batch_size, False, 0, args.device.startswith("cuda")),
            target_scales,
            active_threshold,
            ionic_scale,
            soft_mode_scale,
            loss_args,
            work_scale=work_scale,
            force_scale=force_scale,
            hvp_scale=hvp_scale,
        )
        score = ref_score(metrics, ref)
        history.append({"epoch": epoch, "train_loss": running / max(1, len(train_loader)), "selection_score": score, **metrics})
        print(
            f"{args.model_id} {args.phase} epoch {epoch:03d}: score={score:.5f} "
            f"DFT emax MAE={metrics['ionic_eijmax_mae']:.4f} "
            f"model-BEC emax MAE={metrics['unified_ionic_eijmax_mae']:.4f} "
            f"model-BEC r={metrics['unified_ionic_eijmax_pearson']:.4f}",
            flush=True,
        )
        if score < best_score - 1.0e-8:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_backbone_state = (
                {
                    key: value.detach().cpu().clone()
                    for key, value in backend.model.state_dict().items()
                }
                if backend is not None
                else None
            )
            checkpoint_state = {
                    "model_state_dict": best_state,
                    "backbone_state_dict": best_backbone_state,
                    "model_id": args.model_id,
                    "phase": args.phase,
                    "best_epoch": best_epoch,
                    "best_selection_score": best_score,
                    "source_sha256": metadata["model_sha256"],
                    "optimizer_state_dict": optimizer.state_dict(),
                    "stale": stale,
                    "history": history,
                }
            checkpoint_tmp = outdir / "best_checkpoint.pt.tmp"
            torch.save(checkpoint_state, checkpoint_tmp)
            checkpoint_tmp.replace(outdir / "best_checkpoint.pt")
        else:
            stale += 1
        history_tmp = outdir / "history.tsv.tmp"
        pd.DataFrame(history).to_csv(history_tmp, sep="\t", index=False)
        history_tmp.replace(outdir / "history.tsv")
        if stale >= 6:
            print(f"{args.model_id} {args.phase}: early stopping after six validation checks", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No fine-tuning checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    if backend is not None:
        backend.model.load_state_dict(best_backbone_state, strict=True)
        valid_for_epoch = fit_dynamic_samples(backend, valid, graphs_valid, structure_batch=8)
    else:
        valid_for_epoch = valid
    final_metrics, rows, predictions = evaluate(
        model,
        make_loader(valid_for_epoch, args.valid_batch_size, False, 0, args.device.startswith("cuda")),
        target_scales,
        active_threshold,
        ionic_scale,
        soft_mode_scale,
        loss_args,
        collect=True,
        work_scale=work_scale,
        force_scale=force_scale,
        hvp_scale=hvp_scale,
    )
    history_tmp = outdir / "history.tsv.tmp"
    pd.DataFrame(history).to_csv(history_tmp, sep="\t", index=False)
    history_tmp.replace(outdir / "history.tsv")
    pd.DataFrame(rows).to_csv(outdir / "valid_predictions.tsv", sep="\t", index=False)
    torch.save({"split": "valid", "samples": predictions}, outdir / "valid_displacement_predictions.pt")
    manifest = {
        "model_id": args.model_id,
        "run_id": run_id,
        "probe_model_id": probe_model_id,
        "probe_subdir": args.probe_subdir,
        "probe_checkpoint_sha256": probe_checkpoint_sha256,
        "cache_model_id": cache_model_id,
        "phase": args.phase,
        "lora_rank": args.lora_rank if args.phase == "LoRA" else None,
        "backbone": args.backend,
        "head": args.head,
        "source_sha256": metadata["model_sha256"],
        "n_train": len(train),
        "n_valid": len(valid),
        "rejected": rejected,
        "backbone_parameters": sum(p.numel() for p in backbone_parameters),
        "backbone_gradient_norm_audit": backbone_gradient_norm,
        "projection_head_parameters": sum(p.numel() for p in projection_head_parameters),
        "effective_batch_size": args.effective_batch_size,
        "micro_batch_size": batch_size,
        "match_microbatch_control": args.match_microbatch_control,
        "response_weight_mean": loss_args.response_weight_mean,
        "backbone_learning_rate": 2.0e-6 if backbone_parameters else 0.0,
        "projection_head_learning_rate": 1.0e-5,
        "epochs": args.epochs,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_score": best_score,
        "resumed_from_checkpoint": args.resume_from_checkpoint,
        "resume_checkpoint_epoch": (
            int(resume_checkpoint["best_epoch"]) if resume_checkpoint is not None else None
        ),
        "optimizer_state_resumed": optimizer_state_resumed,
        "training_status": "complete",
        "initial_valid_metrics": initial_metrics,
        "final_valid_metrics": final_metrics,
        "feature_response_anchor": 2.0e-3 if backend is not None else 0.0,
        "parameter_anchor": 1.0e-4,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.model_id} {args.phase} final:", json.dumps({k: final_metrics[k] for k in final_metrics if "mae" in k or "pearson" in k}, indent=2), flush=True)


if __name__ == "__main__":
    main()
