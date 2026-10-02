#!/usr/bin/env python3
"""Train an atom-resolved equivariant internal-strain tensor head."""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from e3nn import o3
from scipy.stats import pearsonr, spearmanr
try:
    from pymatgen.core import Composition
except ImportError:  # Keep the fixed safe5 head experiment runnable without pymatgen.
    from ase.formula import Formula

    class Composition:  # type: ignore[no-redef]
        def __init__(self, formula: str):
            reduced, _ = Formula(formula).reduce()
            self.reduced_formula = reduced.format()
try:
    import spglib
except ImportError:
    spglib = None
from ase.io import read as ase_read

from piezo_irreps import (
    PIEZO_IRREPS,
    cartesian_to_voigt,
    irreps_to_voigt,
    voigt_to_cartesian,
    voigt_to_irreps,
)
from strain_response_irreps import rotate_strain_response
from uiu_equivariant_piezo_model import DEFAULT_NODE_IRREPS
from uiu_internal_strain_model import (
    PIEZO_BLOCK_SLICES,
    AtomicInternalStrainHead,
)
from uiu_internal_strain_feature_adapter import (
    EquivariantFeatureTangentAdapter,
    FeatureAdaptedInternalStrainHead,
)


warnings.filterwarnings(
    "ignore",
    message=".*The TorchScript type system doesn't support instance-level annotations.*",
)


CONV = 16.0217663
COMPONENTS = tuple(f"{axis}{mode}" for axis in "xyz" for mode in range(1, 7))
LOSS_KEYS = (
    "loss",
    "tensor_loss",
    "cosine_loss",
    "ionic_loss",
    "weighted_dft_ionic_loss",
    "dft_eijmax_loss",
    "unified_ionic_loss",
    "unified_eijmax_loss",
    "ionic_parallel_amplitude_loss",
    "ionic_perpendicular_loss",
    "block_amplitude_loss",
    "block_direction_loss",
    "soft_amplitude_loss",
    "soft_direction_loss",
    "soft_piezo_amplitude_loss",
    "soft_piezo_direction_loss",
    "graph_amplitude_loss",
    "graph_direction_loss",
    "bec_visible_loss",
    "bec_null_loss",
    "work_loss",
    "reciprocity_loss",
    "hessian_consistency_loss",
    "hvp_projected_loss",
    "feature_adapter_loss",
    "response_adapter_loss",
    "gamma_asinh_loss",
    "hurdle_gate_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node-cache-dir",
        default="results/uiu_equivariant_piezo_cache_safe5_voigt_corrected",
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
        "--atomic-response-dir",
        default=None,
        help="Enable the six-strain atom-feature branch using this cache directory.",
    )
    parser.add_argument(
        "--model-hessian-dir",
        default=None,
        help=(
            "Optional frozen model-Hessian cache directory containing "
            "{split}_model_hessian.pt for Phi Gamma ~= Xi supervision."
        ),
    )
    parser.add_argument(
        "--unified-prediction-dir",
        default=None,
        help=(
            "Optional physical-polarization prediction directory containing "
            "{split}_predictions.pt. Its ASR-projected model BEC is used for a "
            "second downstream ionic-piezo supervision chain."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="results/uiu_internal_strain_head_context_safe5_v1",
    )
    parser.add_argument(
        "--target-kind",
        choices=("internal_strain", "displacement"),
        default="internal_strain",
        help=(
            "Predict dF/dstrain (internal_strain) or the complete inverse-Hessian "
            "response du/dstrain (displacement)."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--fixed-epochs",
        action="store_true",
        help="Use the final epoch without validation-driven scheduling or checkpoint selection.",
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--scalar-channels", type=int, default=32)
    parser.add_argument("--vector-channels", type=int, default=16)
    parser.add_argument("--tensor-channels", type=int, default=16)
    parser.add_argument("--atomic-response-mix-init", type=float, default=0.1)
    parser.add_argument("--huber-beta", type=float, default=0.2)
    parser.add_argument("--active-quantile", type=float, default=0.8)
    parser.add_argument("--high-response-weight", type=float, default=0.5)
    parser.add_argument("--lambda-cosine", type=float, default=0.2)
    parser.add_argument("--lambda-ionic", type=float, default=0.25)
    parser.add_argument(
        "--lambda-bec-visible",
        type=float,
        default=0.0,
        help="Additional loss on the displacement row-space visible through DFT BEC.",
    )
    parser.add_argument(
        "--lambda-bec-null",
        type=float,
        default=0.0,
        help="Small auxiliary loss on the BEC-null-space displacement.",
    )
    parser.add_argument(
        "--bec-projection-rcond",
        type=float,
        default=1.0e-8,
        help="Relative ridge used to build the differentiable BEC row-space projector.",
    )
    parser.add_argument("--lambda-weighted-dft-ionic", type=float, default=0.0)
    parser.add_argument("--lambda-dft-eijmax", type=float, default=0.0)
    parser.add_argument("--lambda-unified-ionic", type=float, default=0.0)
    parser.add_argument("--lambda-unified-eijmax", type=float, default=0.0)
    parser.add_argument("--lambda-block-amplitude", type=float, default=0.0)
    parser.add_argument("--lambda-block-direction", type=float, default=0.0)
    parser.add_argument(
        "--response-loss-alpha",
        type=float,
        default=0.0,
        help="Capped high-response multiplier used by optional graph losses.",
    )
    parser.add_argument("--response-weight-gamma", type=float, default=0.5)
    parser.add_argument("--response-weight-cap", type=float, default=3.0)
    parser.add_argument(
        "--response-bin-balance-power",
        type=float,
        default=0.0,
        help="Inverse response-bin frequency exponent for WeightedRandomSampler.",
    )
    parser.add_argument(
        "--response-bin-target-fractions",
        default=None,
        help=(
            "Comma-separated target sampling fractions for zero, low, active, "
            "and high response bins."
        ),
    )
    parser.add_argument(
        "--source-target-fractions",
        default=None,
        help="Comma-separated target fractions for historical and new source domains.",
    )
    parser.add_argument(
        "--formula-balance-power",
        type=float,
        default=0.0,
        help="Downweight repeated reduced-formula groups by count to this power.",
    )
    parser.add_argument("--lambda-soft-amplitude", type=float, default=0.0)
    parser.add_argument("--lambda-soft-direction", type=float, default=0.0)
    parser.add_argument("--lambda-soft-piezo-amplitude", type=float, default=0.0)
    parser.add_argument("--lambda-soft-piezo-direction", type=float, default=0.0)
    parser.add_argument("--lambda-graph-amplitude", type=float, default=0.0)
    parser.add_argument("--lambda-graph-direction", type=float, default=0.0)
    parser.add_argument("--soft-candidate-modes", type=int, default=12)
    parser.add_argument("--soft-top-modes", type=int, default=6)
    parser.add_argument(
        "--soft-mode-ranking",
        choices=("displacement", "piezo"),
        default="displacement",
    )
    parser.add_argument("--max-reconstruction-mae", type=float, default=0.01)
    parser.add_argument("--require-stable-hessian", action="store_true", default=True)
    parser.add_argument("--allow-unstable-hessian", action="store_false", dest="require_stable_hessian")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume model, optimizer, scheduler, best state, and history from this file.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model weights from a completed compatible response-head checkpoint.",
    )
    parser.add_argument(
        "--initial-validation-candidate",
        action="store_true",
        help="Keep the initialized epoch-0 model unless fine-tuning improves validation loss.",
    )
    parser.add_argument(
        "--feature-adapter",
        action="store_true",
        help=(
            "Insert a zero-initialized nonlinear equivariant adapter in the "
            "frozen UIU node-feature space and propagate it to strain tangents."
        ),
    )
    parser.add_argument("--adapter-scalar-channels", type=int, default=32)
    parser.add_argument("--adapter-vector-channels", type=int, default=16)
    parser.add_argument("--adapter-tensor-channels", type=int, default=16)
    parser.add_argument(
        "--adapter-tangent-step",
        type=float,
        default=0.005,
        help="Feature-space central-difference step used for the adapter tangent.",
    )
    parser.add_argument("--adapter-lr", type=float, default=2.0e-4)
    parser.add_argument(
        "--freeze-base-head",
        action="store_true",
        help="Train only the representation adapter and keep the initialized head fixed.",
    )
    parser.add_argument("--lambda-feature-adapter", type=float, default=1.0e-3)
    parser.add_argument("--lambda-response-adapter", type=float, default=1.0e-3)
    parser.add_argument(
        "--symmetry-projection",
        action="store_true",
        help=(
            "Project atom-resolved response tensors with the crystal space-group "
            "operations and atom mappings before the loss/readout."
        ),
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=1.0e-3,
        help="spglib position tolerance used for the atom-level space-group projector.",
    )
    parser.add_argument(
        "--lambda-work",
        type=float,
        default=0.0,
        help="Weight of Lambda^T Gamma nuclear-relaxation work supervision.",
    )
    parser.add_argument(
        "--lambda-reciprocity",
        type=float,
        default=0.0,
        help="Weight of the antisymmetric Lambda^T Gamma reciprocity penalty.",
    )
    parser.add_argument(
        "--lambda-hessian-consistency",
        type=float,
        default=0.0,
        help="Weight of the frozen-Hessian consistency loss Phi Gamma ~= Xi.",
    )
    parser.add_argument(
        "--hessian-consistency-beta",
        type=float,
        default=0.2,
        help="Huber beta used for the Hessian/internal-strain consistency loss.",
    )
    parser.add_argument(
        "--hvp-labels",
        default=None,
        help=(
            "Optional torch label package containing selected BEC-visible HVP "
            "targets. The labels are used only as an auxiliary projected-response loss."
        ),
    )
    parser.add_argument(
        "--lambda-hvp-projected",
        type=float,
        default=0.0,
        help=(
            "Weight of the BEC-visible HVP-projected displacement loss, "
            "(H q)^T Gamma ~= (H q)^T Gamma_DFT."
        ),
    )
    parser.add_argument(
        "--hvp-beta",
        type=float,
        default=0.2,
        help="Huber beta after robust asinh scaling for the projected HVP loss.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def finite_correlation(function, reference, prediction) -> float:
    reference = np.asarray(reference, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    mask = np.isfinite(reference) & np.isfinite(prediction)
    if mask.sum() < 2:
        return float("nan")
    if np.std(reference[mask]) == 0 or np.std(prediction[mask]) == 0:
        return float("nan")
    return float(function(reference[mask], prediction[mask]).statistic)


def translation_projector(n_atoms: int) -> np.ndarray:
    translations = np.zeros((3 * n_atoms, 3), dtype=np.float64)
    for atom in range(n_atoms):
        translations[3 * atom : 3 * atom + 3] = np.eye(3)
    translations /= math.sqrt(n_atoms)
    return np.eye(3 * n_atoms) - translations @ translations.T


def load_space_group_projector(
    source_dir: str,
    n_atoms: int,
    symprec: float,
) -> tuple[torch.Tensor, torch.Tensor, dict] | None:
    """Build Cartesian rotations and atom maps for an atom-level space-group projector.

    For each operation g, ``source_for_target[g, a]`` identifies the atom b
    that is mapped to target atom a.  The projected tensor is then the average
    of R tensor(source) over all operations.  Returning None keeps data
    preparation robust for records without a readable POSCAR/CONTCAR.
    """
    directory = Path(source_dir)
    if spglib is None:
        return None
    structure_path = next(
        (directory / name for name in ("CONTCAR", "POSCAR") if (directory / name).exists()),
        None,
    )
    if structure_path is None:
        return None
    try:
        atoms = ase_read(str(structure_path), format="vasp")
        if len(atoms) != n_atoms or not atoms.pbc.all():
            return None
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        scaled = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=np.float64)
        numbers = np.asarray(atoms.numbers, dtype=np.int32)
        symmetry = spglib.get_symmetry(
            (cell, scaled, numbers),
            symprec=float(symprec),
            angle_tolerance=-1.0,
        )
        if symmetry is None:
            return None
        fractional_rotations = np.asarray(symmetry["rotations"], dtype=np.float64)
        translations = np.asarray(symmetry["translations"], dtype=np.float64)
        cartesian_basis = cell.T
        inverse_basis = np.linalg.inv(cartesian_basis)
        rotations = []
        source_maps = []
        for fractional_rotation, translation in zip(
            fractional_rotations, translations
        ):
            transformed = (
                scaled @ fractional_rotation.T + translation[None, :]
            ) % 1.0
            source_to_target = np.full(n_atoms, -1, dtype=np.int64)
            for source_index, position in enumerate(transformed):
                delta = (scaled - position[None, :] + 0.5) % 1.0 - 0.5
                distances = np.linalg.norm(delta @ cell, axis=1)
                candidates = np.flatnonzero(
                    (numbers == numbers[source_index])
                    & (distances <= max(float(symprec) * 5.0, 1.0e-5))
                )
                if len(candidates) != 1:
                    return None
                source_to_target[source_index] = int(candidates[0])
            if len(np.unique(source_to_target)) != n_atoms:
                return None
            target_to_source = np.argsort(source_to_target)
            cartesian_rotation = (
                cartesian_basis
                @ fractional_rotation
                @ inverse_basis
            )
            rotations.append(cartesian_rotation)
            source_maps.append(target_to_source)
        metadata = {
            "space_group_operations": len(rotations),
            "symprec": float(symprec),
            "structure_path": str(structure_path),
        }
        return (
            torch.from_numpy(np.asarray(rotations, dtype=np.float32)),
            torch.from_numpy(np.asarray(source_maps, dtype=np.int64)),
            metadata,
        )
    except Exception:
        return None


def project_atom_voigt(
    tensor: torch.Tensor,
    rotations: torch.Tensor,
    source_for_target: torch.Tensor,
) -> torch.Tensor:
    """Project one atom-resolved 3x6 polar-strain tensor by space-group symmetry."""
    cartesian = voigt_to_cartesian(tensor)
    source = cartesian[source_for_target]
    transformed = torch.einsum(
        "gip,gjq,gkr,gapqr->gaijk",
        rotations,
        rotations,
        rotations,
        source,
    )
    return cartesian_to_voigt(transformed.mean(dim=0))


def project_batch_atom_voigt(
    tensor: torch.Tensor,
    batch: dict,
) -> torch.Tensor:
    """Apply variable-size space-group projectors to concatenated atom tensors."""
    projected = tensor.clone()
    for rotations, source_for_target, (start, stop) in zip(
        batch["symmetry_rotations"],
        batch["symmetry_source_for_target"],
        batch["atom_slices"],
    ):
        projected[start:stop] = project_atom_voigt(
            tensor[start:stop], rotations, source_for_target
        )
    return projected


def ionic_operator_from_bec(
    sample: dict,
    bec: np.ndarray | torch.Tensor,
    target_kind: str,
) -> torch.Tensor:
    """Return the linear map from a BEC tensor and supervised response to piezo."""
    bec = np.asarray(bec, dtype=np.float64)
    n_atoms = len(bec)
    bec = bec - np.mean(bec, axis=0, keepdims=True)
    bec_matrix = np.transpose(bec, (1, 0, 2)).reshape(3, 3 * n_atoms)
    if target_kind == "internal_strain":
        hessian = np.asarray(sample["force_constants_eVA2"], dtype=np.float64)
        projector = translation_projector(n_atoms)
        hessian = projector @ (0.5 * (hessian + hessian.T)) @ projector
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        tolerance = max(float(np.max(np.abs(eigenvalues))) * 1.0e-8, 1.0e-8)
        inverse = np.zeros_like(eigenvalues)
        active = eigenvalues > tolerance
        inverse[active] = 1.0 / eigenvalues[active]
        hessian_inverse = eigenvectors @ (inverse[:, None] * eigenvectors.T)
        hessian_inverse = projector @ hessian_inverse @ projector
        operator = (
            CONV
            / max(float(sample["volume"]), 1.0e-12)
            * bec_matrix
            @ hessian_inverse
        )
    elif target_kind == "displacement":
        operator = CONV / max(float(sample["volume"]), 1.0e-12) * bec_matrix
    else:
        raise ValueError(target_kind)
    return torch.from_numpy(operator.astype(np.float32))


def ionic_operator(sample: dict, target_kind: str) -> torch.Tensor:
    """Return the DFT-BEC map from the supervised response to ionic piezo."""
    return ionic_operator_from_bec(
        sample,
        sample["born_effective_charge"],
        target_kind,
    )


def fit_atomic_response_scales(
    samples: list[dict], node_irreps: o3.Irreps = DEFAULT_NODE_IRREPS
) -> torch.Tensor:
    scale_blocks = []
    for irrep_slice, (multiplicity, irrep) in zip(
        node_irreps.slices(),
        node_irreps,
    ):
        sum_squares = torch.zeros(multiplicity, dtype=torch.float64)
        count = 0
        for sample in samples:
            response = sample["atomic_response"][..., irrep_slice].double()
            response = response.reshape(-1, multiplicity, irrep.dim)
            sum_squares += torch.sum(response.square(), dim=(0, 2))
            count += response.shape[0] * irrep.dim
        copy_scales = torch.sqrt(sum_squares / max(1, count)).clamp_min(1.0e-3)
        scale_blocks.append(
            copy_scales[:, None].expand(multiplicity, irrep.dim).reshape(-1)
        )
    return torch.cat(scale_blocks).float()


def fit_node_feature_scales(samples: list[dict]) -> torch.Tensor:
    """Fit symmetry-compatible RMS scales for each irrep copy."""
    scale_blocks = []
    for irrep_slice, (multiplicity, irrep) in zip(
        DEFAULT_NODE_IRREPS.slices(),
        DEFAULT_NODE_IRREPS,
    ):
        sum_squares = torch.zeros(multiplicity, dtype=torch.float64)
        count = 0
        for sample in samples:
            features = sample["node_features"][..., irrep_slice].double()
            features = features.reshape(-1, multiplicity, irrep.dim)
            sum_squares += torch.sum(features.square(), dim=(0, 2))
            count += features.shape[0] * irrep.dim
        copy_scales = torch.sqrt(sum_squares / max(1, count)).clamp_min(1.0e-3)
        scale_blocks.append(
            copy_scales[:, None].expand(multiplicity, irrep.dim).reshape(-1)
        )
    return torch.cat(scale_blocks).float()


def fit_target_scales(samples: list[dict]) -> torch.Tensor:
    targets = torch.cat([sample["target_irreps"] for sample in samples], dim=0)
    return torch.stack(
        [
            torch.sqrt(torch.mean(targets[:, block].square())).clamp_min(1.0e-3)
            for block in PIEZO_BLOCK_SLICES
        ]
    ).float()


def fit_work_scale(samples: list[dict]) -> torch.Tensor:
    """Fit one robust scale for the 6x6 Lambda^T Gamma work matrix."""
    values = torch.cat(
        [sample["work_target"].reshape(-1) for sample in samples], dim=0
    ).float()
    # Work has a much heavier tail than the displacement components.  A RMS
    # scale would let a few pathological entries dominate the auxiliary loss.
    return torch.quantile(values.abs(), 0.75).clamp_min(1.0e-3)


def fit_hvp_projected_scale(samples: list[dict]) -> torch.Tensor:
    """Fit a robust scale for (H q)^T Gamma in the selected HVP replay set."""
    values = []
    for sample in samples:
        if not sample.get("hvp_available", False):
            continue
        hvp = sample["target_hvp"].reshape(-1, 3)
        target = sample["target_voigt"].reshape(-1, 6)
        values.append((hvp.T @ target).reshape(-1))
    if not values:
        return torch.tensor(1.0)
    values = torch.cat(values).float()
    return torch.quantile(values.abs(), 0.75).clamp_min(1.0e-3)


def soft_mode_targets(
    sample: dict,
    target: torch.Tensor,
    candidate_modes: int,
    top_modes: int,
    ranking: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the response-dominant modes among the lowest positive modes."""
    n_atoms = target.shape[0]
    hessian = np.asarray(sample["force_constants_eVA2"], dtype=np.float64)
    projector = translation_projector(n_atoms)
    hessian = projector @ (0.5 * (hessian + hessian.T)) @ projector
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    tolerance = max(float(np.max(np.abs(eigenvalues))) * 1.0e-8, 1.0e-8)
    positive = np.flatnonzero(eigenvalues > tolerance)
    candidates = positive[: min(candidate_modes, len(positive))]
    if len(candidates) == 0:
        return (
            torch.zeros((3 * n_atoms, 0), dtype=torch.float32),
            torch.zeros((0, 6), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.float32),
        )
    vectors = eigenvectors[:, candidates]
    target_matrix = target.numpy().reshape(3 * n_atoms, 6)
    coefficients = vectors.T @ target_matrix
    bec = np.asarray(sample["born_effective_charge"], dtype=np.float64)
    bec -= np.mean(bec, axis=0, keepdims=True)
    bec_matrix = np.transpose(bec, (1, 0, 2)).reshape(3, 3 * n_atoms)
    mode_piezo_factor = (
        CONV
        / max(float(sample["volume"]), 1.0e-12)
        * (bec_matrix @ vectors).T
    )
    if ranking == "displacement":
        response_scores = np.linalg.norm(coefficients, axis=1)
    elif ranking == "piezo":
        contributions = mode_piezo_factor[:, :, None] * coefficients[:, None, :]
        response_scores = np.linalg.norm(contributions.reshape(len(candidates), -1), axis=1)
    else:
        raise ValueError(ranking)
    selected_local = np.argsort(response_scores)[-min(top_modes, len(candidates)) :]
    selected_local = selected_local[np.argsort(eigenvalues[candidates[selected_local]])]
    return (
        torch.from_numpy(vectors[:, selected_local].astype(np.float32)),
        torch.from_numpy(coefficients[selected_local].astype(np.float32)),
        torch.from_numpy(eigenvalues[candidates[selected_local]].astype(np.float32)),
        torch.from_numpy(mode_piezo_factor[selected_local].astype(np.float32)),
    )


def fit_soft_mode_scale(samples: list[dict]) -> torch.Tensor:
    coefficients = [
        sample["soft_mode_target"].abs().reshape(-1)
        for sample in samples
        if sample.get("soft_mode_target") is not None
        and sample["soft_mode_target"].numel() > 0
    ]
    if not coefficients:
        return torch.tensor(1.0)
    values = torch.cat(coefficients)
    nonzero = values[values > 1.0e-10]
    if not len(nonzero):
        return torch.tensor(1.0)
    return torch.quantile(nonzero, 0.75).clamp_min(1.0e-3).float()


def load_response_map(directory: str | None, split: str) -> dict[str, torch.Tensor]:
    if directory is None:
        return {}
    path = Path(directory) / f"{split}_atomic_response_all.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    allowlist_path = Path(directory) / f"{split}_jids.txt"
    allowlist = (
        set(allowlist_path.read_text(encoding="utf-8").splitlines())
        if allowlist_path.exists()
        else None
    )
    return {
        sample["jid"]: sample["atomic_response"].float()
        for sample in payload["samples"]
        if allowlist is None or sample["jid"] in allowlist
    }


def load_model_hessian_map(
    directory: str | None, split: str
) -> dict[str, torch.Tensor]:
    if directory is None:
        return {}
    path = Path(directory) / f"{split}_model_hessian.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = {}
    for jid, record in payload.items():
        hessian = record["hessian"] if isinstance(record, dict) else record
        result[jid] = torch.as_tensor(hessian, dtype=torch.float32)
    return result


def load_unified_bec_map(
    directory: str | None,
    split: str,
    dfpt_map: dict[str, dict],
) -> dict[str, torch.Tensor]:
    if directory is None:
        return {}
    path = Path(directory) / f"{split}_predictions.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    allowlist_path = Path(directory) / f"{split}_jids.txt"
    allowlist = (
        set(allowlist_path.read_text(encoding="utf-8").splitlines())
        if allowlist_path.exists()
        else None
    )
    predictions = payload["pred_bec"].float()
    result = {}
    offset = 0
    for metadata in payload["metadata"]:
        jid = metadata["jid"]
        if jid not in dfpt_map:
            raise KeyError(f"unified BEC metadata contains unknown JID {jid}")
        natoms = int(dfpt_map[jid]["natoms"])
        bec = predictions[offset : offset + natoms].clone()
        offset += natoms
        if allowlist is None or jid in allowlist:
            result[jid] = bec - bec.mean(dim=0, keepdim=True)
    if offset != len(predictions):
        raise ValueError(
            f"BEC atom count mismatch: consumed {offset}, payload has {len(predictions)}"
        )
    return result


def load_hvp_label_map(path: str | None) -> dict[str, dict]:
    """Load the optional selected-structure HVP labels keyed by JID."""
    if path is None:
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    entries = payload.get("entries", payload.get("samples"))
    if entries is None:
        raise KeyError(f"HVP label package has no entries/samples field: {path}")
    result = {}
    for entry in entries:
        jid = str(entry["jid"])
        result[jid] = entry
    return result


def prepare_split(
    split: str,
    args: argparse.Namespace,
    dfpt_map: dict[str, dict],
    quality_map: dict[str, dict],
    unified_bec_map: dict[str, torch.Tensor] | None = None,
    hvp_map: dict[str, dict] | None = None,
    require_unified_bec: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    node_path = Path(args.node_cache_dir) / f"{split}_node_features_all.pt"
    node_payload = torch.load(node_path, map_location="cpu", weights_only=False)
    allowlist_path = Path(args.node_cache_dir) / f"{split}_jids.txt"
    allowlist = (
        set(allowlist_path.read_text(encoding="utf-8").splitlines())
        if allowlist_path.exists()
        else None
    )
    response_map = load_response_map(args.atomic_response_dir, split)
    hessian_map = load_model_hessian_map(args.model_hessian_dir, split)
    samples = []
    rejected = {
        "missing_dfpt": 0,
        "quality": 0,
        "atom_mismatch": 0,
        "missing_response": 0,
        "missing_model_hessian": 0,
        "missing_unified_bec": 0,
        "unified_bec_unavailable": 0,
        "unified_bec_atom_mismatch": 0,
        "missing_symmetry": 0,
        "symmetry_fallback_identity": 0,
        "hvp_labels": 0,
        "hvp_label_mismatch": 0,
    }
    for node_sample in node_payload["samples"]:
        jid = node_sample["jid"]
        if allowlist is not None and jid not in allowlist:
            continue
        if jid not in dfpt_map or jid not in quality_map:
            rejected["missing_dfpt"] += 1
            continue
        if args.model_hessian_dir is not None and jid not in hessian_map:
            rejected["missing_model_hessian"] += 1
            continue
        quality = quality_map[jid]
        stable = (
            float(quality["reconstruction_mae"]) <= args.max_reconstruction_mae
            and (
                not args.require_stable_hessian
                or int(quality["negative_mode_count"]) == 0
            )
        )
        if not stable:
            rejected["quality"] += 1
            continue
        dfpt = dfpt_map[jid]
        target_key = (
            "internal_strain_eVA"
            if args.target_kind == "internal_strain"
            else "internal_displacement_dstrain"
        )
        target = dfpt[target_key].float()
        if target.shape[0] != node_sample["node_features"].shape[0]:
            rejected["atom_mismatch"] += 1
            continue
        target = target - target.mean(dim=0, keepdim=True)
        force_response = torch.as_tensor(
            dfpt["internal_strain_eVA"], dtype=torch.float32
        )
        if force_response.shape != target.shape:
            rejected["atom_mismatch"] += 1
            continue
        force_response = force_response - force_response.mean(dim=0, keepdim=True)
        model_hessian = None
        if args.model_hessian_dir is not None:
            model_hessian = hessian_map[jid]
            expected_size = 3 * int(target.shape[0])
            if model_hessian.shape != (expected_size, expected_size):
                rejected["atom_mismatch"] += 1
                continue
        target_matrix = target.reshape(-1, 6)
        force_matrix = force_response.reshape(-1, 6)
        work_target = force_matrix.T @ target_matrix
        work_target = 0.5 * (work_target + work_target.T)
        sample = {
            "jid": jid,
            "formula": node_sample.get("formula", ""),
            "reduced_formula": node_sample.get("formula", ""),
            "node_features": node_sample["node_features"].float(),
            "target_voigt": target,
            "target_irreps": voigt_to_irreps(target),
            "force_response": force_response,
            "work_target": work_target,
            "ionic_operator": ionic_operator(dfpt, args.target_kind),
            "reference_ionic": dfpt["reference_ionic_Cm2"].float(),
            "reconstructed_ionic": dfpt["reconstructed_ionic_Cm2"].float(),
            "quality": quality,
        }
        if hvp_map is not None:
            # Keep fixed-shape zero placeholders so ordinary structures and
            # the sparse HVP replay structures can share every minibatch.
            num_modes = 3
            hvp_modes = torch.zeros(
                (target.shape[0], num_modes, 3), dtype=torch.float32
            )
            target_hvp = torch.zeros_like(hvp_modes)
            hvp_available = False
            entry = hvp_map.get(jid)
            if entry is not None:
                entry_modes = torch.as_tensor(
                    entry["bec_visible_directions"], dtype=torch.float32
                )
                entry_hvp = torch.as_tensor(
                    entry["target_hvp_eVA2"], dtype=torch.float32
                )
                if (
                    entry_modes.shape == hvp_modes.shape
                    and entry_hvp.shape == target_hvp.shape
                ):
                    hvp_modes = entry_modes
                    target_hvp = entry_hvp
                    hvp_available = True
                    rejected["hvp_labels"] += 1
                else:
                    rejected["hvp_label_mismatch"] += 1
            sample["hvp_modes"] = hvp_modes
            sample["target_hvp"] = target_hvp
            sample["hvp_available"] = hvp_available
        if model_hessian is not None:
            sample["model_hessian"] = 0.5 * (model_hessian + model_hessian.T)
        try:
            sample["reduced_formula"] = Composition(
                sample["formula"]
            ).reduced_formula
        except Exception:
            pass
        source_dir = str(dfpt.get("source_dir", ""))
        source_domain = dfpt.get("source_domain")
        if source_domain is None:
            if jid.startswith("PBESOL::"):
                source_domain = "pbesol_finish"
            elif source_dir.startswith("official_figshare_archive"):
                source_domain = "new_official"
            else:
                source_domain = "historical_local"
        sample["source_domain"] = str(source_domain)
        if unified_bec_map is not None:
            model_bec = unified_bec_map.get(jid)
            if model_bec is None:
                rejected["unified_bec_unavailable"] += 1
                sample["unified_bec_available"] = False
                if require_unified_bec:
                    rejected["missing_unified_bec"] += 1
                    continue
            elif model_bec.shape[0] != target.shape[0]:
                rejected["unified_bec_atom_mismatch"] += 1
                sample["unified_bec_available"] = False
                if require_unified_bec:
                    rejected["missing_unified_bec"] += 1
                    continue
            else:
                sample["unified_bec_operator"] = ionic_operator_from_bec(
                    dfpt,
                    model_bec,
                    args.target_kind,
                )
                sample["unified_bec_available"] = True
        if args.target_kind == "displacement":
            vectors, coefficients, eigenvalues, piezo_factor = soft_mode_targets(
                dfpt,
                target,
                args.soft_candidate_modes,
                args.soft_top_modes,
                args.soft_mode_ranking,
            )
            sample["soft_mode_vectors"] = vectors
            sample["soft_mode_target"] = coefficients
            sample["soft_mode_eigenvalues"] = eigenvalues
            sample["soft_mode_piezo_factor"] = piezo_factor
        if args.atomic_response_dir is not None:
            response = response_map.get(jid)
            if response is None:
                rejected["missing_response"] += 1
                continue
            if response.shape[0] != target.shape[0]:
                rejected["atom_mismatch"] += 1
                continue
            sample["atomic_response"] = response
        if args.symmetry_projection:
            symmetry_payload = load_space_group_projector(
                str(dfpt["source_dir"]),
                target.shape[0],
                args.symprec,
            )
            if symmetry_payload is None:
                # Preserve the fixed split when a source structure is not
                # readable: identity is a valid no-op projector, and the
                # fallback count is recorded for auditability.
                symmetry_payload = (
                    torch.eye(3, dtype=torch.float32).unsqueeze(0),
                    torch.arange(target.shape[0], dtype=torch.long).unsqueeze(0),
                    {
                        "space_group_operations": 1,
                        "symprec": float(args.symprec),
                        "fallback_identity": True,
                    },
                )
                rejected["symmetry_fallback_identity"] += 1
            (
                sample["symmetry_rotations"],
                sample["symmetry_source_for_target"],
                sample["symmetry_metadata"],
            ) = symmetry_payload
        samples.append(sample)
        limit = args.max_train if split == "train" else args.max_valid
        if limit is not None and len(samples) >= limit:
            break
    limit = args.max_train if split == "train" else args.max_valid
    if limit is not None:
        samples = samples[:limit]
    return samples, rejected


class InternalStrainDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def collate(samples: list[dict]) -> dict:
    features = []
    targets = []
    batches = []
    atom_slices = []
    start = 0
    for graph_index, sample in enumerate(samples):
        count = len(sample["node_features"])
        features.append(sample["node_features"])
        targets.append(sample["target_irreps"])
        batches.append(torch.full((count,), graph_index, dtype=torch.long))
        atom_slices.append((start, start + count))
        start += count
    result = {
        "node_features": torch.cat(features),
        "target_irreps": torch.cat(targets),
        "batch": torch.cat(batches),
        "atom_slices": atom_slices,
        "operators": [sample["ionic_operator"] for sample in samples],
        "reference_ionic": torch.stack([sample["reference_ionic"] for sample in samples]),
        "reconstructed_ionic": torch.stack([sample["reconstructed_ionic"] for sample in samples]),
        "response_weight": torch.tensor(
            [sample.get("response_weight", 1.0) for sample in samples],
            dtype=torch.float32,
        ),
        "jid": [sample["jid"] for sample in samples],
        "formula": [sample["formula"] for sample in samples],
        "force_response": torch.cat(
            [sample["force_response"] for sample in samples]
        ),
        "work_target": torch.stack(
            [sample["work_target"] for sample in samples]
        ),
    }
    if "model_hessian" in samples[0]:
        result["model_hessian"] = [sample["model_hessian"] for sample in samples]
    if "target_hvp" in samples[0]:
        result["hvp_modes"] = torch.cat([sample["hvp_modes"] for sample in samples])
        result["target_hvp"] = torch.cat([sample["target_hvp"] for sample in samples])
        result["hvp_available"] = torch.tensor(
            [sample["hvp_available"] for sample in samples], dtype=torch.bool
        )
    if "atomic_response" in samples[0]:
        result["atomic_response"] = torch.cat(
            [sample["atomic_response"] for sample in samples]
        )
    if "local_geometry" in samples[0]:
        result["local_geometry"] = torch.cat(
            [sample["local_geometry"] for sample in samples]
        )
    if "structure_features" in samples[0]:
        result["structure_features"] = torch.stack(
            [sample["structure_features"] for sample in samples]
        )
    if "response_is_strong" in samples[0]:
        result["response_is_strong"] = torch.tensor(
            [sample["response_is_strong"] for sample in samples],
            dtype=torch.float32,
        )
    if any("unified_bec_available" in sample for sample in samples):
        result["unified_available"] = torch.tensor(
            [sample.get("unified_bec_available", False) for sample in samples],
            dtype=torch.bool,
        )
        result["unified_operators"] = [
            sample.get(
                "unified_bec_operator",
                torch.zeros((3, 3 * len(sample["node_features"])), dtype=torch.float32),
            )
            for sample in samples
        ]
    if "soft_mode_vectors" in samples[0]:
        result["soft_mode_vectors"] = [sample["soft_mode_vectors"] for sample in samples]
        result["soft_mode_target"] = [sample["soft_mode_target"] for sample in samples]
        result["soft_mode_eigenvalues"] = [
            sample["soft_mode_eigenvalues"] for sample in samples
        ]
        result["soft_mode_piezo_factor"] = [
            sample["soft_mode_piezo_factor"] for sample in samples
        ]
    if "symmetry_rotations" in samples[0]:
        result["symmetry_rotations"] = [
            sample["symmetry_rotations"] for sample in samples
        ]
        result["symmetry_source_for_target"] = [
            sample["symmetry_source_for_target"] for sample in samples
        ]
    return result


def make_loader(
    samples,
    batch_size,
    shuffle,
    workers,
    pin_memory,
    sample_weights=None,
    seed=123,
    generator=None,
):
    options = {}
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    sampler = None
    if sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(samples),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle = False
    return torch.utils.data.DataLoader(
        InternalStrainDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        generator=generator,
        **options,
    )


def move_batch(batch: dict, device: str) -> dict:
    moved = dict(batch)
    for key in (
        "node_features",
        "target_irreps",
        "batch",
        "reference_ionic",
        "reconstructed_ionic",
        "response_weight",
        "atomic_response",
        "local_geometry",
        "force_response",
        "work_target",
        "hvp_modes",
        "target_hvp",
        "hvp_available",
        "unified_available",
        "structure_features",
        "response_is_strong",
    ):
        if key in moved:
            moved[key] = moved[key].to(device, non_blocking=True)
    if "model_hessian" in moved:
        moved["model_hessian"] = [
            value.to(device, non_blocking=True) for value in moved["model_hessian"]
        ]
    moved["operators"] = [value.to(device, non_blocking=True) for value in batch["operators"]]
    if "unified_operators" in batch:
        moved["unified_operators"] = [
            value.to(device, non_blocking=True)
            for value in batch["unified_operators"]
        ]
    for key in (
        "soft_mode_vectors",
        "soft_mode_target",
        "soft_mode_eigenvalues",
        "soft_mode_piezo_factor",
    ):
        if key in moved:
            moved[key] = [value.to(device, non_blocking=True) for value in batch[key]]
    for key in ("symmetry_rotations", "symmetry_source_for_target"):
        if key in moved:
            moved[key] = [value.to(device, non_blocking=True) for value in batch[key]]
    return moved


def downstream_ionic(
    prediction: torch.Tensor,
    batch: dict,
    operator_key: str = "operators",
) -> torch.Tensor:
    tensors = []
    prediction_voigt = irreps_to_voigt(prediction)
    for operator, (start, stop) in zip(batch[operator_key], batch["atom_slices"]):
        coupling = prediction_voigt[start:stop].reshape(-1, 6)
        tensors.append(operator @ coupling)
    return torch.stack(tensors)


def bec_visible_null_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: dict,
    scale: torch.Tensor,
    beta: float,
    rcond: float,
    weight_normalizer: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split displacement into BEC-visible and BEC-null components.

    For each strain column, the BEC operator A maps atom displacements to the
    ionic piezo tensor.  P=A^T(AA^T)^+A is the displacement row-space
    projector, so A(Pu)=Au and A((I-P)u)=0 up to numerical tolerance.
    """
    prediction_voigt = irreps_to_voigt(prediction)
    target_voigt = irreps_to_voigt(target)
    visible_losses = []
    null_losses = []
    for operator, (start, stop), graph_weight in zip(
        batch["operators"], batch["atom_slices"], batch["response_weight"]
    ):
        matrix = operator
        gram = matrix @ matrix.T
        ridge = rcond * torch.trace(gram).div(max(gram.shape[0], 1)).clamp_min(1.0)
        regularized = gram + ridge * torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )
        size = stop - start
        pred_matrix = prediction_voigt[start:stop].reshape(3 * size, 6)
        target_matrix = target_voigt[start:stop].reshape(3 * size, 6)
        # Apply P=A^T(AA^T)^+A without materializing the dense 3N x 3N
        # projector.  This keeps the row/null decomposition linear in N.
        solve_pred = torch.linalg.solve(regularized, matrix @ pred_matrix)
        solve_target = torch.linalg.solve(regularized, matrix @ target_matrix)
        pred_visible = matrix.T @ solve_pred
        target_visible = matrix.T @ solve_target
        pred_null = pred_matrix - pred_visible
        target_null = target_matrix - target_visible
        visible_losses.append(
            F.smooth_l1_loss(
                pred_visible / scale,
                target_visible / scale,
                beta=beta,
            ) * graph_weight
        )
        null_losses.append(
            F.smooth_l1_loss(
                pred_null / scale,
                target_null / scale,
                beta=beta,
            ) * graph_weight
        )
    if not visible_losses:
        zero = prediction.new_zeros(())
        return zero, zero
    weights = batch["response_weight"].sum().clamp_min(1.0e-12)
    if weight_normalizer is not None:
        weights = prediction.new_tensor(float(weight_normalizer)) * len(visible_losses)
    return torch.stack(visible_losses).sum() / weights, torch.stack(null_losses).sum() / weights


def configure_response_emphasis(
    train_samples: list[dict],
    valid_samples: list[dict],
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict]:
    """Fit capped high-response weights and optional response-bin sampling."""
    train_response = np.asarray(
        [float(sample["reference_ionic"].abs().max()) for sample in train_samples],
        dtype=float,
    )
    scale = max(float(np.quantile(train_response, 0.75)), 1.0e-6)

    def loss_weight(value: float) -> float:
        ratio = max(value, 0.0) / scale
        bonus = min(ratio**args.response_weight_gamma, args.response_weight_cap)
        return 1.0 + args.response_loss_alpha * bonus

    for sample in train_samples + valid_samples:
        response = float(sample["reference_ionic"].abs().max())
        sample["response_weight"] = loss_weight(response)

    edges = np.asarray([1.0e-12, 0.05, 1.0], dtype=float)
    bins = np.digitize(train_response, edges, right=False)
    counts = np.bincount(bins, minlength=len(edges) + 1)
    sampler_weights = None
    target_fractions = None
    if args.response_bin_target_fractions:
        target_fractions = np.asarray(
            [float(value) for value in args.response_bin_target_fractions.split(",")],
            dtype=float,
        )
        if len(target_fractions) != len(counts) or np.any(target_fractions < 0.0):
            raise ValueError(
                "response-bin-target-fractions must contain four non-negative values"
            )
        if not np.isclose(target_fractions.sum(), 1.0):
            raise ValueError("response-bin-target-fractions must sum to one")

    source_values = np.asarray(
        [sample.get("source_domain", "unknown") for sample in train_samples],
        dtype=object,
    )
    source_names = sorted(set(source_values.tolist()))
    source_counts = {
        name: int(np.sum(source_values == name)) for name in source_names
    }
    source_fractions = None
    if args.source_target_fractions:
        source_fractions = np.asarray(
            [float(value) for value in args.source_target_fractions.split(",")],
            dtype=float,
        )
        if len(source_fractions) != len(source_names) or np.any(source_fractions < 0.0):
            raise ValueError(
                "source-target-fractions must match the sorted source domains"
            )
        if not np.isclose(source_fractions.sum(), 1.0):
            raise ValueError("source-target-fractions must sum to one")

    formula_values = np.asarray(
        [sample.get("reduced_formula", sample.get("formula", "")) for sample in train_samples],
        dtype=object,
    )
    formula_counts = {
        name: int(np.sum(formula_values == name))
        for name in set(formula_values.tolist())
    }
    if (
        target_fractions is not None
        or source_fractions is not None
        or args.formula_balance_power > 0.0
    ):
        base = np.asarray(
            [
                max(float(sample["response_weight"]), 1.0e-12)
                * max(float(formula_counts[formula]), 1.0) ** (-args.formula_balance_power)
                for sample, formula in zip(train_samples, formula_values)
            ],
            dtype=float,
        )
        sampler_weights = np.zeros(len(train_samples), dtype=float)
        if target_fractions is not None and source_fractions is not None:
            source_fraction_map = dict(zip(source_names, source_fractions))
            for bin_index in range(len(counts)):
                for source_name in source_names:
                    mask = (bins == bin_index) & (source_values == source_name)
                    mass = float(target_fractions[bin_index] * source_fraction_map[source_name])
                    denominator = float(base[mask].sum())
                    if mask.any() and denominator > 0.0:
                        sampler_weights[mask] = mass * base[mask] / denominator
        elif target_fractions is not None:
            for bin_index, mass in enumerate(target_fractions):
                mask = bins == bin_index
                denominator = float(base[mask].sum())
                if mask.any() and denominator > 0.0:
                    sampler_weights[mask] = float(mass) * base[mask] / denominator
        elif source_fractions is not None:
            source_fraction_map = dict(zip(source_names, source_fractions))
            for source_name, mass in source_fraction_map.items():
                mask = source_values == source_name
                denominator = float(base[mask].sum())
                if mask.any() and denominator > 0.0:
                    sampler_weights[mask] = float(mass) * base[mask] / denominator
        else:
            sampler_weights = base
        sampler_weights /= max(float(np.mean(sampler_weights)), 1.0e-12)
    diagnostics = {
        "response_q75": scale,
        "response_bin_edges": edges.tolist(),
        "response_bin_counts": counts.tolist(),
        "response_bin_labels": ["zero", "low_<0.05", "active_0.05-1", "high_>=1"],
        "target_response_bin_fractions": (
            target_fractions.tolist() if target_fractions is not None else None
        ),
        "source_counts": source_counts,
        "target_source_fractions": (
            dict(zip(source_names, source_fractions.tolist()))
            if source_fractions is not None
            else None
        ),
        "formula_group_count": len(formula_counts),
        "formula_balance_power": args.formula_balance_power,
        "sampler_enabled": sampler_weights is not None,
        "sampler_weight_min": float(np.min(sampler_weights))
        if sampler_weights is not None
        else 1.0,
        "sampler_weight_max": float(np.max(sampler_weights))
        if sampler_weights is not None
        else 1.0,
    }
    return sampler_weights, diagnostics


def normalized_block_loss(prediction, target, scales, beta, high_response_weight) -> torch.Tensor:
    losses = []
    atom_norm = torch.linalg.vector_norm(target, dim=1)
    norm_scale = torch.median(atom_norm).clamp_min(1.0e-3)
    weights = 1.0 + high_response_weight * torch.clamp(
        torch.log1p(atom_norm / norm_scale), max=3.0
    )
    for index, block in enumerate(PIEZO_BLOCK_SLICES):
        error = F.smooth_l1_loss(
            prediction[:, block] / scales[index],
            target[:, block] / scales[index],
            beta=beta,
            reduction="none",
        ).mean(dim=1)
        losses.append(torch.sum(error * weights) / torch.sum(weights))
    return torch.stack(losses).mean()


def weighted_graph_component_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    beta: float,
    weight_normalizer: float | None = None,
) -> torch.Tensor:
    per_graph = F.smooth_l1_loss(
        prediction / scale,
        target / scale,
        beta=beta,
        reduction="none",
    ).mean(dim=(1, 2))
    denominator = torch.sum(weights).clamp_min(1.0e-12)
    if weight_normalizer is not None:
        denominator = per_graph.new_tensor(float(weight_normalizer)) * len(per_graph)
    return torch.sum(per_graph * weights) / denominator


def eijmax_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    beta: float,
    weight_normalizer: float | None = None,
) -> torch.Tensor:
    prediction_emax = prediction.abs().amax(dim=(1, 2))
    target_emax = target.abs().amax(dim=(1, 2))
    per_graph = F.smooth_l1_loss(
        torch.log1p(prediction_emax / scale),
        torch.log1p(target_emax / scale),
        beta=beta,
        reduction="none",
    )
    denominator = torch.sum(weights).clamp_min(1.0e-12)
    if weight_normalizer is not None:
        denominator = per_graph.new_tensor(float(weight_normalizer)) * len(per_graph)
    return torch.sum(per_graph * weights) / denominator


def ionic_direction_amplitude_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    threshold: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize amplitude error along the reference tensor and orthogonal error."""
    prediction_flat = prediction.reshape(len(prediction), -1)
    target_flat = target.reshape(len(target), -1)
    target_norm = torch.linalg.vector_norm(target_flat, dim=1)
    active = target.abs().amax(dim=(1, 2)) >= threshold
    if not torch.any(active):
        zero = prediction.new_zeros(())
        return zero, zero

    direction = target_flat[active] / target_norm[active, None].clamp_min(1.0e-8)
    predicted_parallel = torch.sum(prediction_flat[active] * direction, dim=1)
    sample_weights = weights[active]
    denominator = sample_weights.sum().clamp_min(1.0e-12)
    parallel_loss = torch.sum(
        F.smooth_l1_loss(
            predicted_parallel / scale,
            target_norm[active] / scale,
            beta=beta,
            reduction="none",
        )
        * sample_weights
    ) / denominator

    perpendicular = prediction_flat[active] - predicted_parallel[:, None] * direction
    perpendicular_loss = torch.sum(
        F.smooth_l1_loss(
            perpendicular / scale,
            torch.zeros_like(perpendicular),
            beta=beta,
            reduction="none",
        ).mean(dim=1)
        * sample_weights
    ) / denominator
    return parallel_loss, perpendicular_loss


def irreps_block_response_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scales: torch.Tensor,
    atom_weights: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate amplitude and direction supervision for each irreducible block."""
    amplitude_losses = []
    direction_losses = []
    for index, block in enumerate(PIEZO_BLOCK_SLICES):
        block_scale = scales[index] * math.sqrt(block.stop - block.start)
        predicted_norm = torch.linalg.vector_norm(prediction[:, block], dim=1)
        target_norm = torch.linalg.vector_norm(target[:, block], dim=1)
        amplitude = F.smooth_l1_loss(
            torch.asinh(predicted_norm / block_scale),
            torch.asinh(target_norm / block_scale),
            beta=beta,
            reduction="none",
        )
        amplitude_losses.append(
            torch.sum(amplitude * atom_weights)
            / torch.sum(atom_weights).clamp_min(1.0e-12)
        )
        active = target_norm > 0.05 * block_scale
        if torch.any(active):
            direction = 1.0 - F.cosine_similarity(
                prediction[active, block],
                target[active, block],
                dim=1,
                eps=1.0e-8,
            )
            active_weights = atom_weights[active] * (
                1.0 + torch.clamp(torch.log1p(target_norm[active] / block_scale), max=3.0)
            )
            direction_losses.append(
                torch.sum(direction * active_weights)
                / torch.sum(active_weights).clamp_min(1.0e-12)
            )
    zero = prediction.new_zeros(())
    return (
        torch.stack(amplitude_losses).mean() if amplitude_losses else zero,
        torch.stack(direction_losses).mean() if direction_losses else zero,
    )


def soft_response_losses(
    prediction_irreps: torch.Tensor,
    target_irreps: torch.Tensor,
    batch: dict,
    soft_mode_scale: torch.Tensor,
    ionic_scale: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    zero = prediction_irreps.new_zeros(())
    if "soft_mode_vectors" not in batch:
        return {
            "soft_amplitude_loss": zero,
            "soft_direction_loss": zero,
            "soft_piezo_amplitude_loss": zero,
            "soft_piezo_direction_loss": zero,
            "graph_amplitude_loss": zero,
            "graph_direction_loss": zero,
        }
    prediction_voigt = irreps_to_voigt(prediction_irreps)
    target_voigt = irreps_to_voigt(target_irreps)
    soft_amplitudes = []
    soft_directions = []
    soft_piezo_amplitudes = []
    soft_piezo_directions = []
    graph_amplitudes = []
    graph_directions = []
    for vectors, target_coefficients, piezo_factor, (start, stop) in zip(
        batch["soft_mode_vectors"],
        batch["soft_mode_target"],
        batch["soft_mode_piezo_factor"],
        batch["atom_slices"],
    ):
        prediction_matrix = prediction_voigt[start:stop].reshape(-1, 6)
        target_matrix = target_voigt[start:stop].reshape(-1, 6)
        if vectors.shape[1] > 0:
            predicted_coefficients = vectors.T @ prediction_matrix
            transformed_prediction = torch.asinh(
                predicted_coefficients / soft_mode_scale
            )
            transformed_target = torch.asinh(
                target_coefficients / soft_mode_scale
            )
            soft_amplitudes.append(
                F.smooth_l1_loss(
                    transformed_prediction,
                    transformed_target,
                    beta=beta,
                )
            )
            target_norm = torch.linalg.vector_norm(target_coefficients, dim=1)
            active = target_norm > 0.05 * soft_mode_scale
            if torch.any(active):
                mode_weights = 1.0 + torch.clamp(
                    torch.log1p(target_norm[active] / soft_mode_scale), max=3.0
                )
                mode_cosine = 1.0 - F.cosine_similarity(
                    predicted_coefficients[active],
                    target_coefficients[active],
                    dim=1,
                    eps=1.0e-8,
                )
                soft_directions.append(
                    torch.sum(mode_weights * mode_cosine) / torch.sum(mode_weights)
                )
            predicted_contribution = (
                piezo_factor[:, :, None] * predicted_coefficients[:, None, :]
            )
            target_contribution = (
                piezo_factor[:, :, None] * target_coefficients[:, None, :]
            )
            soft_piezo_amplitudes.append(
                F.smooth_l1_loss(
                    torch.asinh(predicted_contribution / ionic_scale),
                    torch.asinh(target_contribution / ionic_scale),
                    beta=beta,
                )
            )
            contribution_norm = torch.linalg.vector_norm(
                target_contribution.reshape(target_contribution.shape[0], -1), dim=1
            )
            piezo_active = contribution_norm > 0.05 * ionic_scale
            if torch.any(piezo_active):
                contribution_weights = 1.0 + torch.clamp(
                    torch.log1p(contribution_norm[piezo_active] / ionic_scale), max=3.0
                )
                contribution_cosine = 1.0 - F.cosine_similarity(
                    predicted_contribution[piezo_active].flatten(1),
                    target_contribution[piezo_active].flatten(1),
                    dim=1,
                    eps=1.0e-8,
                )
                soft_piezo_directions.append(
                    torch.sum(contribution_weights * contribution_cosine)
                    / torch.sum(contribution_weights)
                )
        predicted_norm = torch.linalg.vector_norm(prediction_matrix)
        target_norm = torch.linalg.vector_norm(target_matrix)
        graph_amplitudes.append(
            F.smooth_l1_loss(
                torch.log1p(predicted_norm / soft_mode_scale),
                torch.log1p(target_norm / soft_mode_scale),
                beta=beta,
            )
        )
        if target_norm > 1.0e-8:
            graph_directions.append(
                1.0
                - F.cosine_similarity(
                    prediction_matrix.reshape(1, -1),
                    target_matrix.reshape(1, -1),
                    dim=1,
                    eps=1.0e-8,
                ).mean()
            )
    return {
        "soft_amplitude_loss": torch.stack(soft_amplitudes).mean()
        if soft_amplitudes
        else zero,
        "soft_direction_loss": torch.stack(soft_directions).mean()
        if soft_directions
        else zero,
        "soft_piezo_amplitude_loss": torch.stack(soft_piezo_amplitudes).mean()
        if soft_piezo_amplitudes
        else zero,
        "soft_piezo_direction_loss": torch.stack(soft_piezo_directions).mean()
        if soft_piezo_directions
        else zero,
        "graph_amplitude_loss": torch.stack(graph_amplitudes).mean()
        if graph_amplitudes
        else zero,
        "graph_direction_loss": torch.stack(graph_directions).mean()
        if graph_directions
        else zero,
    }


def work_reciprocity_losses(
    prediction_irreps: torch.Tensor,
    batch: dict,
    work_scale: torch.Tensor,
    beta: float,
    weight_normalizer: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise the nuclear-relaxation work and its reciprocity symmetry.

    With the project's force-response convention, Lambda^T Gamma is the
    six-strain compliance/work matrix.  Its symmetric part is supervised by
    the DFT target and its antisymmetric part is penalized independently.
    """
    if "force_response" not in batch or "work_target" not in batch:
        zero = prediction_irreps.new_zeros(())
        return zero, zero
    prediction_voigt = irreps_to_voigt(prediction_irreps)
    work_losses = []
    reciprocity_losses = []
    scale = work_scale.clamp_min(1.0e-3)
    for graph_index, (start, stop) in enumerate(batch["atom_slices"]):
        force_matrix = batch["force_response"][start:stop].reshape(-1, 6)
        predicted_matrix = prediction_voigt[start:stop].reshape(-1, 6)
        predicted_work = force_matrix.T @ predicted_matrix
        target_work = batch["work_target"][graph_index]
        predicted_symmetric = 0.5 * (
            predicted_work + predicted_work.T
        )
        work_losses.append(
            F.smooth_l1_loss(
                torch.asinh(predicted_symmetric / scale),
                torch.asinh(target_work / scale),
                beta=beta,
            )
        )
        antisymmetric = predicted_work - predicted_work.T
        reciprocity_losses.append(
            torch.mean(torch.asinh(antisymmetric / scale).square())
        )
    if not work_losses:
        zero = prediction_irreps.new_zeros(())
        return zero, zero
    graph_weights = batch["response_weight"]
    denominator = graph_weights.sum().clamp_min(1.0e-12)
    if weight_normalizer is not None:
        denominator = graph_weights.new_tensor(float(weight_normalizer)) * len(work_losses)
    weights = graph_weights / denominator
    return (
        torch.sum(torch.stack(work_losses) * weights),
        torch.sum(torch.stack(reciprocity_losses) * weights),
    )


def hessian_consistency_loss(
    prediction_irreps: torch.Tensor,
    batch: dict,
    force_scale: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Enforce the linear-response identity Phi Gamma ~= Xi.

    The Hessian is frozen and used only as an auxiliary operator.  Translation
    modes are projected out before applying it, matching the DFPT response
    reconstruction used when the displacement targets were created.
    """
    if "model_hessian" not in batch:
        return prediction_irreps.new_zeros(())
    prediction_voigt = irreps_to_voigt(prediction_irreps)
    losses = []
    scale = force_scale.clamp_min(1.0e-3)
    for hessian, (start, stop) in zip(
        batch["model_hessian"], batch["atom_slices"]
    ):
        size = stop - start
        projector = torch.as_tensor(
            translation_projector(size),
            device=hessian.device,
            dtype=hessian.dtype,
        )
        hessian = projector @ (0.5 * (hessian + hessian.T)) @ projector
        displacement = prediction_voigt[start:stop].reshape(3 * size, 6)
        force_target = batch["force_response"][start:stop].reshape(3 * size, 6)
        force_prediction = hessian @ (projector @ displacement)
        losses.append(
            F.smooth_l1_loss(
                force_prediction / scale,
                force_target / scale,
                beta=beta,
            )
        )
    return torch.stack(losses).mean() if losses else prediction_irreps.new_zeros(())


def hvp_projected_loss(
    prediction_irreps: torch.Tensor,
    batch: dict,
    scale: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Use selected DFT HVPs to constrain BEC-visible displacement response.

    For a BEC-visible mode q, the DFT label h=Hq gives the projected response
    h^T Gamma.  This keeps the auxiliary target linear in the predicted
    displacement and therefore sends its gradient directly into the direct
    displacement head.  asinh is important because a few soft/high-response
    structures have very large HVP magnitudes.
    """
    if "target_hvp" not in batch or "hvp_available" not in batch:
        return prediction_irreps.new_zeros(())
    prediction_voigt = irreps_to_voigt(prediction_irreps)
    target_voigt = irreps_to_voigt(batch["target_irreps"])
    losses = []
    robust_scale = scale.to(
        device=prediction_irreps.device,
        dtype=prediction_irreps.dtype,
    ).clamp_min(1.0e-3)
    for graph_index, (start, stop) in enumerate(batch["atom_slices"]):
        if not bool(batch["hvp_available"][graph_index]):
            continue
        hvp = batch["target_hvp"][start:stop].reshape(-1, 3)
        predicted = prediction_voigt[start:stop].reshape(-1, 6)
        target = target_voigt[start:stop].reshape(-1, 6)
        predicted_projected = hvp.T @ predicted
        target_projected = hvp.T @ target
        losses.append(
            F.smooth_l1_loss(
                torch.asinh(predicted_projected / robust_scale),
                torch.asinh(target_projected / robust_scale),
                beta=beta,
            )
        )
    return torch.stack(losses).mean() if losses else prediction_irreps.new_zeros(())


def batch_loss(
    model,
    batch,
    scales,
    active_threshold,
    ionic_scale,
    soft_mode_scale,
    args,
    work_scale=None,
    force_scale=None,
    hvp_scale=None,
):
    if work_scale is None:
        work_scale = torch.tensor(1.0, device=batch["node_features"].device)
    if force_scale is None:
        force_scale = torch.tensor(1.0, device=batch["node_features"].device)
    if hvp_scale is None:
        hvp_scale = torch.tensor(1.0, device=batch["node_features"].device)
    model_inputs = (
        batch["node_features"],
        batch["batch"],
        batch.get("atomic_response"),
        len(batch["jid"]),
    )
    if getattr(model, "expects_structure_features", False):
        output = model(
            *model_inputs,
            batch.get("local_geometry"),
            structure_features=batch["structure_features"],
        )
    elif getattr(model, "local_geometry", False):
        output = model(*model_inputs, batch.get("local_geometry"))
    else:
        output = model(*model_inputs)
    prediction = output["lambda_irreps"]
    if getattr(args, "symmetry_projection", False):
        projected_voigt = project_batch_atom_voigt(
            irreps_to_voigt(prediction), batch
        )
        projected_irreps = voigt_to_irreps(projected_voigt)
        prediction, asr_residual = AtomicInternalStrainHead.enforce_acoustic_sum_rule(
            projected_irreps,
            batch["batch"],
            len(batch["jid"]),
        )
        output["lambda_irreps"] = prediction
        output["lambda_voigt"] = irreps_to_voigt(prediction)
        output["asr_residual_irreps"] = asr_residual
    target = batch["target_irreps"]
    tensor_loss = normalized_block_loss(
        prediction,
        target,
        scales,
        args.huber_beta,
        args.high_response_weight,
    )
    gamma_asinh_loss = prediction.new_zeros(())
    if getattr(args, "lambda_gamma_asinh", 0.0):
        # Apply a radial transform to the complete O(3) tensor. The invariant
        # norm supplies one scalar per atom, so rotations only affect direction.
        target_scale = torch.as_tensor(
            getattr(args, "gamma_asinh_scale", scales.mean()),
            device=prediction.device,
            dtype=prediction.dtype,
        ).clamp_min(1.0e-6)
        pred_norm = torch.linalg.vector_norm(prediction, dim=-1, keepdim=True)
        target_norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
        pred_factor = torch.asinh(pred_norm / target_scale) / pred_norm.clamp_min(1.0e-8)
        target_factor = torch.asinh(target_norm / target_scale) / target_norm.clamp_min(1.0e-8)
        pred_transformed = prediction * pred_factor
        target_transformed = target * target_factor
        gamma_asinh_loss = F.smooth_l1_loss(
            pred_transformed / target_scale,
            target_transformed / target_scale,
            beta=args.huber_beta,
        )
    hurdle_gate_loss = prediction.new_zeros(())
    gate = output.get("hurdle_gate")
    if gate is not None and getattr(args, "lambda_hurdle_gate", 0.0):
        pos_weight = torch.as_tensor(
            getattr(args, "hurdle_pos_weight", 1.0),
            device=gate.device,
            dtype=gate.dtype,
        )
        gate_target = batch["response_is_strong"].to(dtype=gate.dtype)
        per_sample = F.binary_cross_entropy(
            gate.reshape(-1).clamp(1.0e-6, 1.0 - 1.0e-6),
            gate_target,
            reduction="none",
        )
        class_weight = torch.where(gate_target > 0.5, pos_weight, 1.0)
        hurdle_gate_loss = (per_sample * class_weight).mean()
    target_norm = torch.linalg.vector_norm(target, dim=1)
    active = target_norm >= active_threshold
    if torch.any(active):
        cosine_loss = (
            1.0
            - F.cosine_similarity(prediction[active], target[active], dim=1, eps=1.0e-8)
        ).mean()
    else:
        cosine_loss = prediction.new_zeros(())
    ionic_prediction = downstream_ionic(prediction, batch)
    response_weights = batch["response_weight"]
    weight_normalizer = getattr(args, "response_weight_mean", None)
    if (
        getattr(args, "lambda_bec_visible", 0.0) != 0.0
        or getattr(args, "lambda_bec_null", 0.0) != 0.0
    ):
        bec_visible_loss, bec_null_loss = bec_visible_null_losses(
            prediction,
            target,
            batch,
            scales.mean().clamp_min(1.0e-3),
            args.huber_beta,
            getattr(args, "bec_projection_rcond", 1.0e-8),
            weight_normalizer,
        )
    else:
        bec_visible_loss = prediction.new_zeros(())
        bec_null_loss = prediction.new_zeros(())
    ionic_loss = F.smooth_l1_loss(
        ionic_prediction / ionic_scale,
        batch["reference_ionic"] / ionic_scale,
        beta=args.huber_beta,
    )
    weighted_dft_ionic_loss = weighted_graph_component_loss(
        ionic_prediction,
        batch["reference_ionic"],
        ionic_scale,
        response_weights,
        args.huber_beta,
        weight_normalizer,
    )
    dft_eijmax_loss = eijmax_loss(
        ionic_prediction,
        batch["reference_ionic"],
        ionic_scale,
        response_weights,
        args.huber_beta,
        weight_normalizer,
    )
    unified_prediction = None
    zero = prediction.new_zeros(())
    unified_ionic_loss = zero
    unified_eijmax_loss = zero
    ionic_parallel_amplitude_loss = zero
    ionic_perpendicular_loss = zero
    if "unified_operators" in batch:
        unified_prediction = downstream_ionic(
            prediction,
            batch,
            "unified_operators",
        )
        unified_available = batch.get(
            "unified_available",
            torch.ones(len(unified_prediction), device=unified_prediction.device, dtype=torch.bool),
        )
        if torch.any(unified_available):
            unified_ionic_loss = weighted_graph_component_loss(
                unified_prediction[unified_available],
                batch["reference_ionic"][unified_available],
                ionic_scale,
                response_weights[unified_available],
                args.huber_beta,
                weight_normalizer,
            )
            unified_eijmax_loss = eijmax_loss(
                unified_prediction[unified_available],
                batch["reference_ionic"][unified_available],
                ionic_scale,
                response_weights[unified_available],
                args.huber_beta,
                weight_normalizer,
            )
    lambda_parallel = getattr(args, "lambda_ionic_parallel_amplitude", 0.0)
    lambda_perpendicular = getattr(args, "lambda_ionic_perpendicular", 0.0)
    if lambda_parallel or lambda_perpendicular:
        ionic_parallel_amplitude_loss, ionic_perpendicular_loss = (
            ionic_direction_amplitude_losses(
                ionic_prediction,
                batch["reference_ionic"],
                ionic_scale,
                response_weights,
                getattr(args, "ionic_direction_threshold", 0.05),
                args.huber_beta,
            )
        )
    atom_weights = response_weights[batch["batch"]]
    block_amplitude_loss, block_direction_loss = irreps_block_response_losses(
        prediction,
        target,
        scales,
        atom_weights,
        args.huber_beta,
    )
    soft_losses = soft_response_losses(
        prediction,
        target,
        batch,
        soft_mode_scale,
        ionic_scale,
        args.huber_beta,
    )
    work_loss, reciprocity_loss = work_reciprocity_losses(
        prediction,
        batch,
        work_scale,
        args.huber_beta,
        weight_normalizer,
    )
    hessian_loss = hessian_consistency_loss(
        prediction,
        batch,
        force_scale,
        getattr(args, "hessian_consistency_beta", args.huber_beta),
    )
    hvp_loss = hvp_projected_loss(
        prediction,
        batch,
        hvp_scale,
        getattr(args, "hvp_beta", args.huber_beta),
    )
    feature_adapter_loss = prediction.new_zeros(())
    response_adapter_loss = prediction.new_zeros(())
    if "feature_adapter_residual_normalized" in output:
        normalized_residual = output["feature_adapter_residual_normalized"]
        feature_adapter_loss = F.smooth_l1_loss(
            normalized_residual,
            torch.zeros_like(normalized_residual),
            beta=args.huber_beta,
        )
    if "response_adapter_residual_normalized" in output:
        normalized_response_residual = output[
            "response_adapter_residual_normalized"
        ]
        response_adapter_loss = F.smooth_l1_loss(
            normalized_response_residual,
            torch.zeros_like(normalized_response_residual),
            beta=args.huber_beta,
        )
    total = tensor_loss + args.lambda_cosine * cosine_loss + args.lambda_ionic * ionic_loss
    total = (
        total
        + getattr(args, "lambda_bec_visible", 0.0) * bec_visible_loss
        + getattr(args, "lambda_bec_null", 0.0) * bec_null_loss
        + getattr(args, "lambda_work", 0.0) * work_loss
        + getattr(args, "lambda_reciprocity", 0.0) * reciprocity_loss
        + getattr(args, "lambda_hessian_consistency", 0.0) * hessian_loss
        + getattr(args, "lambda_hvp_projected", 0.0) * hvp_loss
        + getattr(args, "lambda_feature_adapter", 0.0) * feature_adapter_loss
        + getattr(args, "lambda_response_adapter", 0.0) * response_adapter_loss
        + getattr(args, "lambda_gamma_asinh", 0.0) * gamma_asinh_loss
        + getattr(args, "lambda_hurdle_gate", 0.0) * hurdle_gate_loss
    )
    total = (
        total
        + args.lambda_soft_amplitude * soft_losses["soft_amplitude_loss"]
        + args.lambda_soft_direction * soft_losses["soft_direction_loss"]
        + args.lambda_soft_piezo_amplitude
        * soft_losses["soft_piezo_amplitude_loss"]
        + args.lambda_soft_piezo_direction
        * soft_losses["soft_piezo_direction_loss"]
        + args.lambda_graph_amplitude * soft_losses["graph_amplitude_loss"]
        + args.lambda_graph_direction * soft_losses["graph_direction_loss"]
    )
    if args.lambda_weighted_dft_ionic:
        total = total + args.lambda_weighted_dft_ionic * weighted_dft_ionic_loss
    if args.lambda_dft_eijmax:
        total = total + args.lambda_dft_eijmax * dft_eijmax_loss
    if args.lambda_unified_ionic:
        total = total + args.lambda_unified_ionic * unified_ionic_loss
    if args.lambda_unified_eijmax:
        total = total + args.lambda_unified_eijmax * unified_eijmax_loss
    total = (
        total
        + lambda_parallel * ionic_parallel_amplitude_loss
        + lambda_perpendicular * ionic_perpendicular_loss
    )
    if args.lambda_block_amplitude:
        total = total + args.lambda_block_amplitude * block_amplitude_loss
    if args.lambda_block_direction:
        total = total + args.lambda_block_direction * block_direction_loss
    if unified_prediction is not None:
        output["unified_ionic_prediction"] = unified_prediction
    return total, {
        "loss": total,
        "tensor_loss": tensor_loss,
        "cosine_loss": cosine_loss,
        "ionic_loss": ionic_loss,
        "weighted_dft_ionic_loss": weighted_dft_ionic_loss,
        "dft_eijmax_loss": dft_eijmax_loss,
        "unified_ionic_loss": unified_ionic_loss,
        "unified_eijmax_loss": unified_eijmax_loss,
        "ionic_parallel_amplitude_loss": ionic_parallel_amplitude_loss,
        "ionic_perpendicular_loss": ionic_perpendicular_loss,
        "block_amplitude_loss": block_amplitude_loss,
        "block_direction_loss": block_direction_loss,
        "bec_visible_loss": bec_visible_loss,
        "bec_null_loss": bec_null_loss,
        "work_loss": work_loss,
        "reciprocity_loss": reciprocity_loss,
        "hessian_consistency_loss": hessian_loss,
        "hvp_projected_loss": hvp_loss,
        "feature_adapter_loss": feature_adapter_loss,
        "response_adapter_loss": response_adapter_loss,
        "gamma_asinh_loss": gamma_asinh_loss,
        "hurdle_gate_loss": hurdle_gate_loss,
        **soft_losses,
        "asr_max": output["asr_residual_irreps"].abs().max(),
    }, output, ionic_prediction


def metrics(reference: np.ndarray, prediction: np.ndarray, prefix: str = "") -> dict:
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - reference
    return {
        f"{prefix}mae": float(np.mean(np.abs(error))),
        f"{prefix}rmse": float(np.sqrt(np.mean(error**2))),
        f"{prefix}pearson": finite_correlation(pearsonr, reference, prediction),
        f"{prefix}spearman": finite_correlation(spearmanr, reference, prediction),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    scales,
    active_threshold,
    ionic_scale,
    soft_mode_scale,
    args,
    collect=False,
    work_scale=None,
    force_scale=None,
    hvp_scale=None,
):
    if work_scale is None:
        work_scale = torch.tensor(1.0, device=args.device)
    model.eval()
    totals = {key: 0.0 for key in LOSS_KEYS}
    count = 0
    target_atoms, prediction_atoms = [], []
    reference_ionic, prediction_ionic = [], []
    reference_unified, prediction_unified = [], []
    rows = []
    prediction_cache = []
    max_asr = 0.0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, args.device)
        total, parts, output, ionic = batch_loss(
            model,
            batch,
            scales,
            active_threshold,
            ionic_scale,
            soft_mode_scale,
            args,
            work_scale=work_scale,
            force_scale=force_scale,
            hvp_scale=hvp_scale,
        )
        graph_count = len(batch["jid"])
        for key in totals:
            totals[key] += float(parts[key]) * graph_count
        count += graph_count
        max_asr = max(max_asr, float(parts["asr_max"]))
        target_np = batch["target_irreps"].detach().cpu().numpy()
        prediction_np = output["lambda_irreps"].detach().cpu().numpy()
        target_atoms.append(target_np)
        prediction_atoms.append(prediction_np)
        reference_np = batch["reference_ionic"].detach().cpu().numpy()
        ionic_np = ionic.detach().cpu().numpy()
        reference_ionic.append(reference_np)
        prediction_ionic.append(ionic_np)
        unified = output.get("unified_ionic_prediction")
        unified_np = unified.detach().cpu().numpy() if unified is not None else None
        if unified_np is not None:
            unified_available = batch.get(
                "unified_available",
                torch.ones(len(unified_np), device=unified.device, dtype=torch.bool),
            ).detach().cpu().numpy().astype(bool)
            if np.any(unified_available):
                prediction_unified.append(unified_np[unified_available])
                reference_unified.append(reference_np[unified_available])
        if collect:
            prediction_voigt = output["lambda_voigt"].detach().cpu()
            target_voigt = irreps_to_voigt(batch["target_irreps"]).detach().cpu()
            for graph_index, (jid, formula, atom_slice) in enumerate(
                zip(batch["jid"], batch["formula"], batch["atom_slices"])
            ):
                start, stop = atom_slice
                ref = reference_np[graph_index]
                pred = ionic_np[graph_index]
                row = {"jid": jid, "formula": formula, "natoms": stop - start}
                row.update({f"ref_ionic_{name}": value for name, value in zip(COMPONENTS, ref.reshape(-1))})
                row.update({f"pred_ionic_{name}": value for name, value in zip(COMPONENTS, pred.reshape(-1))})
                row["ionic_component_mae"] = float(np.mean(np.abs(pred - ref)))
                row["ref_ionic_eijmax"] = float(np.max(np.abs(ref)))
                row["pred_ionic_eijmax"] = float(np.max(np.abs(pred)))
                if unified_np is not None and unified_available[graph_index]:
                    unified_pred = unified_np[graph_index]
                    row.update(
                        {
                            f"pred_unified_ionic_{name}": value
                            for name, value in zip(COMPONENTS, unified_pred.reshape(-1))
                        }
                    )
                    row["unified_ionic_component_mae"] = float(
                        np.mean(np.abs(unified_pred - ref))
                    )
                    row["pred_unified_ionic_eijmax"] = float(
                        np.max(np.abs(unified_pred))
                    )
                rows.append(row)
                prediction_cache.append(
                    {
                        "jid": jid,
                        (
                            "target_internal_strain_eVA"
                            if args.target_kind == "internal_strain"
                            else "target_internal_displacement_dstrain"
                        ): target_voigt[start:stop].clone(),
                        (
                            "predicted_internal_strain_eVA"
                            if args.target_kind == "internal_strain"
                            else "predicted_internal_displacement_dstrain"
                        ): prediction_voigt[start:stop].clone(),
                    }
                )
    target_atoms = np.concatenate(target_atoms)
    prediction_atoms = np.concatenate(prediction_atoms)
    reference_ionic = np.concatenate(reference_ionic)
    prediction_ionic = np.concatenate(prediction_ionic)
    target_norm = np.linalg.norm(target_atoms, axis=1)
    prediction_norm = np.linalg.norm(prediction_atoms, axis=1)
    cosine = np.sum(target_atoms * prediction_atoms, axis=1) / np.maximum(
        target_norm * prediction_norm, 1.0e-12
    )
    active = target_norm >= active_threshold
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result.update(metrics(target_atoms, prediction_atoms, "lambda_component_"))
    result.update(metrics(reference_ionic, prediction_ionic, "ionic_component_"))
    result.update(
        {
            "lambda_cosine_mean": float(np.mean(cosine)),
            "lambda_active_cosine_mean": float(np.mean(cosine[active])) if np.any(active) else float("nan"),
            "lambda_norm_ratio_median": float(np.median(prediction_norm / np.maximum(target_norm, 1.0e-12))),
            "asr_max_abs": max_asr,
            "n_structures": count,
            "n_atoms": len(target_atoms),
        }
    )
    ref_emax = np.max(np.abs(reference_ionic), axis=(1, 2))
    pred_emax = np.max(np.abs(prediction_ionic), axis=(1, 2))
    result.update(metrics(ref_emax, pred_emax, "ionic_eijmax_"))
    if prediction_unified:
        prediction_unified_array = np.concatenate(prediction_unified)
        reference_unified_array = np.concatenate(reference_unified)
        result.update(
            metrics(
                reference_unified_array,
                prediction_unified_array,
                "unified_ionic_component_",
            )
        )
        unified_ref_emax = np.max(np.abs(reference_unified_array), axis=(1, 2))
        unified_emax = np.max(np.abs(prediction_unified_array), axis=(1, 2))
        result.update(metrics(unified_ref_emax, unified_emax, "unified_ionic_eijmax_"))
    return result, rows, prediction_cache


def equivariance_diagnostic(model, sample, device, atomic_response_enabled):
    model.eval()
    generator = torch.Generator().manual_seed(20260803)
    matrix = torch.randn((3, 3), generator=generator)
    q, r = torch.linalg.qr(matrix)
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    rotation = q.to(device)
    node = sample["node_features"].to(device)
    batch = torch.zeros(len(node), dtype=torch.long, device=device)
    response = sample.get("atomic_response")
    response_device = response.to(device) if response is not None else None
    with torch.no_grad():
        original = model(node, batch, response_device, 1)["lambda_irreps"]
        node_transform = DEFAULT_NODE_IRREPS.D_from_matrix(rotation)
        rotated_node = node @ node_transform.T
        rotated_response = None
        if atomic_response_enabled:
            rotated_response = rotate_strain_response(
                response_device,
                DEFAULT_NODE_IRREPS,
                rotation,
            )
        rotated = model(rotated_node, batch, rotated_response, 1)["lambda_irreps"]
        output_transform = PIEZO_IRREPS.D_from_matrix(rotation)
        expected = original @ output_transform.T
    difference = rotated - expected
    return {
        "max_abs_error": float(difference.abs().max().cpu()),
        "relative_l2_error": float(
            (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected).clamp_min(1.0e-12)).cpu()
        ),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dfpt_payload = torch.load(args.dfpt_cache, map_location="cpu", weights_only=False)
    dfpt_map = {sample["jid"]: sample for sample in dfpt_payload["samples"]}
    quality_frame = pd.read_csv(args.quality_summary, sep="\t")
    quality_map = {row["jid"]: row for row in quality_frame.to_dict(orient="records")}
    hvp_map = load_hvp_label_map(args.hvp_labels)
    if (
        args.lambda_unified_ionic > 0.0 or args.lambda_unified_eijmax > 0.0
    ) and args.unified_prediction_dir is None:
        raise ValueError(
            "unified ionic losses require --unified-prediction-dir"
        )
    train_unified_bec = load_unified_bec_map(
        args.unified_prediction_dir,
        "train",
        dfpt_map,
    )
    valid_unified_bec = load_unified_bec_map(
        args.unified_prediction_dir,
        "valid",
        dfpt_map,
    )
    train_samples, train_rejected = prepare_split(
        "train",
        args,
        dfpt_map,
        quality_map,
        train_unified_bec if args.unified_prediction_dir is not None else None,
        hvp_map if args.hvp_labels is not None else None,
    )
    valid_samples, valid_rejected = prepare_split(
        "valid",
        args,
        dfpt_map,
        quality_map,
        valid_unified_bec if args.unified_prediction_dir is not None else None,
        hvp_map if args.hvp_labels is not None else None,
    )
    if not train_samples or not valid_samples:
        raise RuntimeError("no usable train/valid structures after quality filtering")
    print(
        f"usable train={len(train_samples)} valid={len(valid_samples)} "
        f"rejected_train={train_rejected} rejected_valid={valid_rejected}",
        flush=True,
    )

    target_scales = fit_target_scales(train_samples)
    train_norms = torch.cat(
        [torch.linalg.vector_norm(sample["target_irreps"], dim=1) for sample in train_samples]
    )
    active_threshold = float(torch.quantile(train_norms, args.active_quantile))
    ionic_scale = torch.sqrt(
        torch.mean(torch.stack([sample["reference_ionic"] for sample in train_samples]).square())
    ).clamp_min(0.05)
    response_scales = (
        fit_atomic_response_scales(train_samples)
        if args.atomic_response_dir is not None
        else None
    )
    node_feature_scales = (
        fit_node_feature_scales(train_samples) if args.feature_adapter else None
    )
    soft_mode_scale = fit_soft_mode_scale(train_samples)
    work_scale = fit_work_scale(train_samples).to(args.device)
    force_values = torch.cat(
        [sample["force_response"].reshape(-1) for sample in train_samples]
    )
    force_scale = torch.quantile(force_values.abs(), 0.75).clamp_min(1.0e-3).to(args.device)
    hvp_scale = fit_hvp_projected_scale(train_samples).to(args.device)
    sampler_weights, response_diagnostics = configure_response_emphasis(
        train_samples,
        valid_samples,
        args,
    )
    base_head = AtomicInternalStrainHead(
        scalar_channels=args.scalar_channels,
        vector_channels=args.vector_channels,
        tensor_channels=args.tensor_channels,
        atomic_response=args.atomic_response_dir is not None,
        atomic_response_feature_scales=response_scales,
        atomic_response_mix_init=args.atomic_response_mix_init,
        output_block_scales=target_scales,
    )
    model = base_head
    if args.feature_adapter:
        adapter = EquivariantFeatureTangentAdapter(
            scalar_channels=args.adapter_scalar_channels,
            vector_channels=args.adapter_vector_channels,
            tensor_channels=args.adapter_tensor_channels,
            feature_scales=node_feature_scales,
            tangent_step=args.adapter_tangent_step,
        )
        model = FeatureAdaptedInternalStrainHead(base_head, adapter)
        if args.freeze_base_head:
            for parameter in model.head.parameters():
                parameter.requires_grad_(False)
    model = model.to(args.device)
    if args.init_checkpoint is not None and args.resume is None:
        initialization = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        initialized_adapter = bool(
            initialization.get("model_config", {}).get("feature_adapter", False)
        )
        if args.feature_adapter and not initialized_adapter:
            model.head.load_state_dict(initialization["model_state_dict"])
        else:
            model.load_state_dict(initialization["model_state_dict"])
        print(f"Initialized model from {args.init_checkpoint}", flush=True)
    target_scales = target_scales.to(args.device)
    ionic_scale = ionic_scale.to(args.device)
    soft_mode_scale = soft_mode_scale.to(args.device)
    pin_memory = args.device.startswith("cuda")
    train_loader = make_loader(
        train_samples,
        args.batch_size,
        True,
        args.loader_workers,
        pin_memory,
        sampler_weights,
        args.seed,
    )
    valid_loader = make_loader(
        valid_samples, args.batch_size, False, args.loader_workers, pin_memory
    )
    if args.feature_adapter:
        parameter_groups = [
            {
                "params": [
                    parameter
                    for parameter in model.adapter.parameters()
                    if parameter.requires_grad
                ],
                "lr": args.adapter_lr,
                "name": "feature_adapter",
            }
        ]
        head_parameters = [
            parameter for parameter in model.head.parameters() if parameter.requires_grad
        ]
        if head_parameters:
            parameter_groups.append(
                {"params": head_parameters, "lr": args.lr, "name": "base_head"}
            )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, args.patience // 4), min_lr=1.0e-5
    )
    history = []
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    start_epoch = 1
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("target_kind", "internal_strain") != args.target_kind:
            raise ValueError("resume target kind does not match the current run")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        best_state = state["best_state_dict"]
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        stale = int(state["stale"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        print(
            f"Resuming from epoch {state['epoch']} with best_epoch={best_epoch}",
            flush=True,
        )
    elif args.initial_validation_candidate:
        initial_metrics, _, _ = evaluate(
            model,
            valid_loader,
            target_scales,
            active_threshold,
            ionic_scale,
            soft_mode_scale,
            args,
            work_scale=work_scale,
            force_scale=force_scale,
            hvp_scale=hvp_scale,
        )
        best_score = float(initial_metrics["loss"])
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        history.append(
            {
                "epoch": 0,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"valid_{key}": value for key, value in initial_metrics.items()},
            }
        )
        print(
            f"epoch 0000 valid={initial_metrics['loss']:.5f} "
            f"ionic_mae={initial_metrics['ionic_component_mae']:.4f} "
            f"ionic_r={initial_metrics['ionic_component_pearson']:.4f} best=0",
            flush=True,
        )
    training_state_path = outdir / "training_state.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_totals = {key: 0.0 for key in LOSS_KEYS}
        train_count = 0
        for cpu_batch in train_loader:
            batch = move_batch(cpu_batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts, _, _ = batch_loss(
                model,
                batch,
                target_scales,
                active_threshold,
                ionic_scale,
                soft_mode_scale,
                args,
                work_scale,
                force_scale,
                hvp_scale,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            graph_count = len(batch["jid"])
            for key in train_totals:
                train_totals[key] += float(parts[key].detach()) * graph_count
            train_count += graph_count
        valid_metrics, _, _ = evaluate(
            model,
            valid_loader,
            target_scales,
            active_threshold,
            ionic_scale,
            soft_mode_scale,
            args,
            work_scale=work_scale,
            force_scale=force_scale,
            hvp_scale=hvp_scale,
        )
        train_metrics = {f"train_{key}": value / train_count for key, value in train_totals.items()}
        score = valid_metrics["loss"]
        if not args.fixed_epochs:
            scheduler.step(score)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        history.append(row)
        if args.fixed_epochs:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        elif score < best_score - 1.0e-6:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        atomic_torch_save(
            {
                "epoch": epoch,
                "target_kind": args.target_kind,
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_state_dict": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale": stale,
                "history": history,
            },
            training_state_path,
        )
        if epoch == 1 or epoch % 10 == 0 or stale == 0:
            print(
                f"epoch {epoch:04d} train={train_metrics['train_loss']:.5f} "
                f"valid={valid_metrics['loss']:.5f} "
                f"lambda_r={valid_metrics['lambda_component_pearson']:.4f} "
                f"ionic_mae={valid_metrics['ionic_component_mae']:.4f} "
                f"ionic_r={valid_metrics['ionic_component_pearson']:.4f} "
                f"best={best_epoch}",
                flush=True,
            )
        if not args.fixed_epochs and stale >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_metrics, rows, prediction_cache = evaluate(
        model,
        valid_loader,
        target_scales,
        active_threshold,
        ionic_scale,
        soft_mode_scale,
        args,
        collect=True,
        work_scale=work_scale,
        force_scale=force_scale,
        hvp_scale=hvp_scale,
    )
    equivariance = equivariance_diagnostic(
        model, valid_samples[0], args.device, args.atomic_response_dir is not None
    )
    checkpoint = {
        "model_state_dict": best_state,
        "model_config": {
            "node_irreps": str(DEFAULT_NODE_IRREPS),
            "scalar_channels": args.scalar_channels,
            "vector_channels": args.vector_channels,
            "tensor_channels": args.tensor_channels,
            "atomic_response": args.atomic_response_dir is not None,
            "atomic_response_mix_init": args.atomic_response_mix_init,
            "feature_adapter": args.feature_adapter,
            "adapter_scalar_channels": args.adapter_scalar_channels,
            "adapter_vector_channels": args.adapter_vector_channels,
            "adapter_tensor_channels": args.adapter_tensor_channels,
            "adapter_tangent_step": args.adapter_tangent_step,
        },
        "target_block_scales": target_scales.detach().cpu(),
        "atomic_response_feature_scales": response_scales,
        "node_feature_scales": node_feature_scales,
        "active_threshold": active_threshold,
        "soft_mode_scale": float(soft_mode_scale.detach().cpu()),
        "work_scale": float(work_scale.detach().cpu()),
        "force_scale": float(force_scale.detach().cpu()),
        "hvp_scale": float(hvp_scale.detach().cpu()),
        "best_epoch": best_epoch,
        "best_valid_loss": best_score,
    }
    checkpoint_name = (
        "internal_strain_head.pt"
        if args.target_kind == "internal_strain"
        else "displacement_response_head.pt"
    )
    prediction_name = (
        "valid_internal_strain_predictions.pt"
        if args.target_kind == "internal_strain"
        else "valid_displacement_predictions.pt"
    )
    torch.save(checkpoint, outdir / checkpoint_name)
    torch.save(
        {"split": "valid", "samples": prediction_cache},
        outdir / prediction_name,
    )
    pd.DataFrame(history).to_csv(outdir / "history.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(outdir / "valid_ionic_predictions.tsv", sep="\t", index=False)
    pd.DataFrame([{"split": "valid", **final_metrics}]).to_csv(
        outdir / "metrics.tsv", sep="\t", index=False
    )
    manifest = {
        "objective": (
            "atom-resolved equivariant internal-strain Lambda head"
            if args.target_kind == "internal_strain"
            else "atom-resolved equivariant inverse-Hessian displacement-response head"
        ),
        "target_kind": args.target_kind,
        "test_evaluated": False,
        "n_train": len(train_samples),
        "n_valid": len(valid_samples),
        "train_rejected": train_rejected,
        "valid_rejected": valid_rejected,
        "best_epoch": best_epoch,
        "best_valid_loss": best_score,
        "atomic_response": args.atomic_response_dir is not None,
        "atomic_response_dir": args.atomic_response_dir,
        "feature_adapter": args.feature_adapter,
        "freeze_base_head": args.freeze_base_head,
        "adapter_lr": args.adapter_lr if args.feature_adapter else None,
        "lambda_feature_adapter": args.lambda_feature_adapter,
        "lambda_response_adapter": args.lambda_response_adapter,
        "symmetry_projection": args.symmetry_projection,
        "symprec": args.symprec,
        "lambda_work": args.lambda_work,
        "lambda_reciprocity": args.lambda_reciprocity,
        "hvp_labels": args.hvp_labels,
        "hvp_label_count_train": int(sum(sample.get("hvp_available", False) for sample in train_samples)),
        "hvp_label_count_valid": int(sum(sample.get("hvp_available", False) for sample in valid_samples)),
        "hvp_projected_scale": float(hvp_scale.detach().cpu()),
        "lambda_hvp_projected": args.lambda_hvp_projected,
        "hvp_beta": args.hvp_beta,
        "active_threshold": active_threshold,
        "target_block_scales": target_scales.detach().cpu().tolist(),
        "ionic_scale": float(ionic_scale.detach().cpu()),
        "response_emphasis": response_diagnostics,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "equivariance": equivariance,
        "final_valid_metrics": final_metrics,
        "arguments": vars(args),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
