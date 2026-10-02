#!/usr/bin/env python3
"""O(3)-equivariant representations for a 3x6 piezoelectric tensor."""

from __future__ import annotations

import torch
from e3nn import o3


VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
PIEZO_RTP = o3.ReducedTensorProducts(
    "ijk=ikj",
    i="1o",
    j="1o",
    k="1o",
)
PIEZO_IRREPS = PIEZO_RTP.irreps_out
PIEZO_CHANGE_OF_BASIS = PIEZO_RTP.change_of_basis.detach().clone()


def voigt_to_cartesian(tensor: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3, 6) Voigt storage to (..., 3, 3, 3)."""
    if tensor.shape[-2:] != (3, 6):
        raise ValueError(f"expected (..., 3, 6), got {tuple(tensor.shape)}")
    full = tensor.new_zeros(tensor.shape[:-2] + (3, 3, 3))
    for voigt_index, (j, k) in enumerate(VOIGT_PAIRS):
        full[..., :, j, k] = tensor[..., :, voigt_index]
        full[..., :, k, j] = tensor[..., :, voigt_index]
    return full


def cartesian_to_voigt(tensor: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3, 3, 3) to (..., 3, 6), symmetrizing the last axes."""
    if tensor.shape[-3:] != (3, 3, 3):
        raise ValueError(f"expected (..., 3, 3, 3), got {tuple(tensor.shape)}")
    columns = [
        0.5 * (tensor[..., :, j, k] + tensor[..., :, k, j])
        for j, k in VOIGT_PAIRS
    ]
    return torch.stack(columns, dim=-1)


def cartesian_to_irreps(tensor: torch.Tensor) -> torch.Tensor:
    """Project a symmetric rank-3 Cartesian tensor onto 2x1o+2o+3o."""
    change = PIEZO_CHANGE_OF_BASIS.to(device=tensor.device, dtype=tensor.dtype)
    return torch.einsum("aijk,...ijk->...a", change, tensor)


def irreps_to_cartesian(features: torch.Tensor) -> torch.Tensor:
    """Reconstruct a symmetric rank-3 Cartesian tensor from its 18 irreps."""
    if features.shape[-1] != PIEZO_IRREPS.dim:
        raise ValueError(
            f"expected last dimension {PIEZO_IRREPS.dim}, got {features.shape[-1]}"
        )
    change = PIEZO_CHANGE_OF_BASIS.to(
        device=features.device,
        dtype=features.dtype,
    )
    return torch.einsum("aijk,...a->...ijk", change, features)


def voigt_to_irreps(tensor: torch.Tensor) -> torch.Tensor:
    return cartesian_to_irreps(voigt_to_cartesian(tensor))


def irreps_to_voigt(features: torch.Tensor) -> torch.Tensor:
    return cartesian_to_voigt(irreps_to_cartesian(features))


def rotate_cartesian_tensor(
    tensor: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Apply T'_ijk = R_ia R_jb R_kc T_abc."""
    return torch.einsum(
        "...ia,...jb,...kc,...abc->...ijk",
        rotation,
        rotation,
        rotation,
        tensor,
    )


def project_cartesian_tensor(
    tensor: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Project one tensor onto the invariant subspace of point-group rotations."""
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(
            f"expected rotations with shape (n, 3, 3), got {tuple(rotations.shape)}"
        )
    transformed = torch.einsum(
        "gia,gjb,gkc,abc->gijk",
        rotations,
        rotations,
        rotations,
        tensor,
    )
    return transformed.mean(dim=0)


def project_irreps(
    features: torch.Tensor,
    rotations: list[torch.Tensor],
) -> torch.Tensor:
    """Point-group project a batch of piezo irreps with variable-size groups."""
    if features.ndim != 2 or len(features) != len(rotations):
        raise ValueError("features and rotations must describe the same batch")
    cartesian = irreps_to_cartesian(features)
    projected = [
        project_cartesian_tensor(cartesian[index], group)
        for index, group in enumerate(rotations)
    ]
    return cartesian_to_irreps(torch.stack(projected, dim=0))
