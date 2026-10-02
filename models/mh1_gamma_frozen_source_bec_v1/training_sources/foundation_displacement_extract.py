#!/usr/bin/env python3
"""Cache atomwise foundation-model features and six strain tangents.

The locked test split is accepted only for inference after model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from build_uiu_bec_ionic_piezo import apply_voigt_strain
from build_uiu_bec_ionic_piezo_jarvis import atoms_from_record


def load_records(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                result[item["jid"]] = item
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MACEInteractionFeatures(torch.nn.Module):
    """Read l=0/l=1 channels before or after the last two product blocks."""

    def __init__(self, model, feature_stage: str = "pre_product"):
        super().__init__()
        from e3nn import o3
        from mace.modules.utils import prepare_graph

        if feature_stage not in {"pre_product", "post_product"}:
            raise ValueError(f"Unknown MACE feature stage: {feature_stage}")
        self.model = model
        self.feature_stage = feature_stage
        self.prepare_graph = prepare_graph
        self.block_slices = []
        self.source_irreps = []
        self.selected_layers = []
        source_modules = list(model.interactions if feature_stage == "pre_product" else model.products)[-2:]
        for module in source_modules:
            irreps = o3.Irreps(
                module.irreps_out if feature_stage == "pre_product" else module.linear.irreps_out
            )
            selected = []
            packed_offset = 0
            flat_offset = 0
            for mul, ir in irreps:
                packed = slice(packed_offset, packed_offset + ir.dim)
                flat = slice(flat_offset, flat_offset + mul * ir.dim)
                if ir.l == 0:
                    selected.append((packed, flat, ir))
                elif ir.l == 1:
                    selected.append((packed, flat, ir))
                packed_offset += ir.dim
                flat_offset += mul * ir.dim
            if not selected:
                raise ValueError(f"Cannot identify scalar/vector channels in {irreps}")
            self.block_slices.append(selected)
            self.selected_layers.append(str(irreps))
            self.source_irreps.append(
                "+".join(f"{mul}x{ir}" for mul, ir in irreps if ir.l in (0, 1))
            )

    def forward(self, data):
        context = self.prepare_graph(
            data,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            lammps_mliap=False,
        )
        model = self.model
        node_feats = model.node_embedding(data["node_attrs"])
        edge_attrs = model.spherical_harmonics(context.vectors)
        edge_feats, cutoff = model.radial_embedding(
            context.lengths,
            data["node_attrs"],
            data["edge_index"],
            model.atomic_numbers,
        )
        if hasattr(model, "joint_embedding"):
            embedding_features = {
                name: data[name] for name, _ in model.embedding_specs.items()
            }
            node_feats = node_feats + model.joint_embedding(
                data["batch"], embedding_features
            )

        kept = []
        start = len(model.interactions) - 2
        for index, (interaction, product) in enumerate(
            zip(model.interactions, model.products)
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
            if index >= start and self.feature_stage == "pre_product":
                kept.append(self._select(node_feats, self.block_slices[index - start]))
            node_feats = product(
                node_feats=node_feats,
                sc=skip,
                node_attrs=data["node_attrs"],
            )
            if index >= start and self.feature_stage == "post_product":
                kept.append(self._select(node_feats, self.block_slices[index - start]))
        return torch.cat(kept, dim=-1)

    @staticmethod
    def _select(node_feats, blocks):
        return torch.cat(
            [
                node_feats[..., flat_slice] if node_feats.ndim == 2
                else node_feats[..., component_slice].flatten(start_dim=-2)
                for component_slice, flat_slice, _ in blocks
            ],
            dim=-1,
        )


class NequIPInteractionFeatures:
    """Read the final two non-scalar hidden layers of an OAM package."""

    ELEMENTS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Ac Th Pa U Np Pu".split()

    def __init__(self, package_path: Path, device: str):
        from nequip.data import AtomicDataDict
        from nequip.data.transforms import (
            ChemicalSpeciesToAtomTypeMapper,
            NeighborListTransform,
        )
        from nequip.model import ModelFromPackage
        from e3nn import o3

        self.AtomicDataDict = AtomicDataDict
        self.ChemicalSpeciesToAtomTypeMapper = ChemicalSpeciesToAtomTypeMapper
        self.NeighborListTransform = NeighborListTransform
        self.mapper = ChemicalSpeciesToAtomTypeMapper(chemical_symbols=self.ELEMENTS)
        self.model = ModelFromPackage(package_path=str(package_path))["sole_model"].to(device).eval()
        self.func = self.model.model.func
        modules = []
        for name, module in self.func.named_modules():
            irreps = getattr(getattr(module, "equivariant_nonlin", None), "irreps_out", None)
            if irreps is None:
                continue
            irreps = o3.Irreps(irreps)
            if any(ir.l == 1 for _, ir in irreps) and any(ir.l == 0 for _, ir in irreps):
                modules.append((name, module, irreps))
        if len(modules) < 2:
            raise ValueError("OAM package has fewer than two equivariant feature blocks")
        self.modules = modules[-2:]
        self.irreps = [
            o3.Irreps([(mul, ir) for mul, ir in item[2] if ir.l in (0, 1)])
            for item in self.modules
        ]
        self._captured = {}
        for name, module, _ in self.modules:
            module.register_forward_hook(self._hook(name))
        self.device = device
        self.r_max = float(self.model.metadata["r_max"])
        self._nl = NeighborListTransform(r_max=self.r_max)

    def _hook(self, name):
        def capture(_module, _inputs, output):
            self._captured[name] = output["node_features"]
        return capture

    def graph(self, atoms):
        from nequip.data.ase import from_ase

        data = from_ase(atoms)
        data = self.mapper(data)
        return self._nl(data)

    def forward(self, graphs):
        batch = self.AtomicDataDict.batched_from_list(graphs)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        self._captured.clear()
        # Bypass energy/force/stress differentiation; hidden features are
        # produced by the packaged graph network itself.
        output = self.func(batch)
        del output
        kept = []
        from e3nn import o3
        for (name, _, full_irreps), selected_irreps in zip(self.modules, self.irreps):
            hidden = self._captured[name]
            slices = full_irreps.slices()
            scalar = [s for s, (_, ir) in zip(slices, full_irreps) if ir.l == 0]
            vector = [s for s, (_, ir) in zip(slices, full_irreps) if ir.l == 1]
            if len(scalar) != 1 or len(vector) != 1:
                raise ValueError(f"Expected one l=0 and l=1 block in {irreps}")
            kept.append(torch.cat((hidden[:, scalar[0]], hidden[:, vector[0]]), dim=-1))
        return torch.cat(kept, dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mace", "nequip"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--head", default="default")
    parser.add_argument("--feature-stage", choices=("pre_product", "post_product"), default="pre_product")
    parser.add_argument("--data-root", type=Path, default=Path("results/pbesol_finish_displacement_increment_v2"))
    parser.add_argument("--split", choices=("train", "valid", "test", "both"), required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/foundation_displacement_benchmark_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-structures", type=int, default=4)
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--delta", type=float, default=0.005)
    args = parser.parse_args()

    if args.split == "both":
        original = sys.argv[1:]
        for split_name in ("train", "valid"):
            child = []
            index = 0
            while index < len(original):
                if original[index] == "--split":
                    child.extend(("--split", split_name))
                    index += 2
                else:
                    child.append(original[index])
                    index += 1
            subprocess.run([sys.executable, __file__, *child], check=True)
        return

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    record_path = args.data_root / "splits" / f"safe5_{args.split}.jsonl"
    records = load_records(record_path)
    # The accepted train/validation lists are generated from these fixed split
    # records; extra rejected quality rows are filtered by the training loader.
    if args.backend == "mace":
        from mace.calculators import MACECalculator
        from mace.data import AtomicData
        from mace.data.utils import config_from_atoms
        from mace.tools import torch_geometric

        calc = MACECalculator(
            model_paths=str(args.model_path),
            device=str(device),
            default_dtype="float32",
            head=args.head,
        )
        model = calc.models[0]
        extractor = MACEInteractionFeatures(model, args.feature_stage).to(device).eval()
        source_irreps = extractor.source_irreps

        def build_graph(atoms):
            conf = config_from_atoms(atoms, head_name=calc.head)
            return AtomicData.from_config(conf, z_table=calc.z_table, cutoff=calc.r_max)

        def forward_graphs(graphs):
            batch = torch_geometric.Batch.from_data_list(graphs).to(device)
            return extractor(batch.to_dict()), batch.ptr.detach().cpu().tolist()

        model_info = {
            "backend": "mace",
            "model_class": type(model).__name__,
            "r_max": float(calc.r_max),
            "head": calc.head,
            "feature_stage": args.feature_stage,
            "selected_layers": extractor.selected_layers,
            "source_irreps_per_block": [str(ir) for ir in source_irreps],
        }
    else:
        extractor = NequIPInteractionFeatures(args.model_path, str(device))
        source_irreps = [str(ir) for ir in extractor.irreps]

        def build_graph(atoms):
            return extractor.graph(atoms)

        def forward_graphs(graphs):
            features = extractor.forward(graphs)
            counts = [len(g["pos"]) for g in graphs]
            pointers = [0]
            for n in counts:
                pointers.append(pointers[-1] + int(n))
            return features, pointers

        model_info = {
            "backend": "nequip",
            "model_class": type(extractor.model).__name__,
            "r_max": extractor.r_max,
            "source_irreps_per_block": source_irreps,
            "selected_layers": [name for name, _, _ in extractor.modules],
        }

    jids = list(records)
    if args.max_structures is not None:
        jids = jids[: args.max_structures]
    result = {}
    chunk_size = max(1, args.batch_structures)
    for offset in range(0, len(jids), chunk_size):
        chunk = jids[offset : offset + chunk_size]
        structures, ordered_graphs = [], []
        for jid in chunk:
            atoms = atoms_from_record(records[jid])
            structures.append(atoms)
            ordered_graphs.append(build_graph(atoms))
            for mode in range(6):
                ordered_graphs.append(build_graph(apply_voigt_strain(atoms, mode, args.delta)))
                ordered_graphs.append(build_graph(apply_voigt_strain(atoms, mode, -args.delta)))
        features, pointers = forward_graphs(ordered_graphs)
        for local, jid in enumerate(chunk):
            start = pointers[13 * local]
            stop = pointers[13 * local + 1]
            base = features[start:stop].float().cpu().contiguous()
            response = torch.empty((stop - start, 6, features.shape[-1]), dtype=torch.float32)
            # Each of six modes contributes + and - graphs after the base graph.
            for mode in range(6):
                pidx = 13 * local + 1 + 2 * mode
                p0, p1 = pointers[pidx], pointers[pidx + 1]
                m0, m1 = pointers[pidx + 1], pointers[pidx + 2]
                if p1 - p0 != stop - start or m1 - m0 != stop - start:
                    raise ValueError(f"atom-order/size mismatch for {jid} strain mode {mode}")
                response[:, mode] = (
                    features[p0:p1] - features[m0:m1]
                ).float().cpu() / (2.0 * args.delta)
            result[jid] = {"node_features": base, "atomic_response": response.contiguous()}
        done = min(offset + len(chunk), len(jids))
        print(f"[{done}/{len(jids)}] cached {args.model_id} {args.split}", flush=True)

    model_info.update(
        {
            "model_id": args.model_id,
            "model_path": str(args.model_path),
            "model_sha256": sha256(args.model_path),
            "split": args.split,
            "n_structures": len(result),
            "delta": args.delta,
            "feature_dim": int(next(iter(result.values()))["node_features"].shape[-1]) if result else 0,
        }
    )
    target = args.outdir / "cache" / args.model_id
    target.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": model_info, "samples": result}, target / f"{args.split}_features.pt")
    print(json.dumps(model_info, indent=2), flush=True)


if __name__ == "__main__":
    main()
