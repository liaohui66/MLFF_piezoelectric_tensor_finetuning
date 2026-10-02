#!/usr/bin/env python3
"""O(3) representation helpers for atom-resolved strain derivatives."""

from __future__ import annotations

import torch
from e3nn import o3


VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
STRAIN_RTP = o3.ReducedTensorProducts(
    "ij=ji",
    i="1o",
    j="1o",
)
STRAIN_IRREPS = STRAIN_RTP.irreps_out
STRAIN_CHANGE_OF_BASIS = STRAIN_RTP.change_of_basis.detach().clone()


def voigt_response_to_strain_irreps(response: torch.Tensor) -> torch.Tensor:
    """Convert (..., 6, feature) derivatives to (..., 0e+2e, feature)."""
    if response.ndim < 2 or response.shape[-2] != 6:
        raise ValueError(
            f"expected response with shape (..., 6, feature), got {tuple(response.shape)}"
        )
    cartesian = response.new_zeros(response.shape[:-2] + (response.shape[-1], 3, 3))
    for mode_index, (j, k) in enumerate(VOIGT_PAIRS):
        cartesian[..., :, j, k] = response[..., mode_index, :]
        cartesian[..., :, k, j] = response[..., mode_index, :]
    change = STRAIN_CHANGE_OF_BASIS.to(
        device=response.device,
        dtype=response.dtype,
    )
    return torch.einsum("aij,...dij->...ad", change, cartesian)


def strain_irreps_to_voigt_response(response: torch.Tensor) -> torch.Tensor:
    """Convert (..., 0e+2e, feature) derivatives to (..., 6, feature)."""
    if response.ndim < 2 or response.shape[-2] != STRAIN_IRREPS.dim:
        raise ValueError(
            "expected response with shape "
            f"(..., {STRAIN_IRREPS.dim}, feature), got {tuple(response.shape)}"
        )
    change = STRAIN_CHANGE_OF_BASIS.to(
        device=response.device,
        dtype=response.dtype,
    )
    cartesian = torch.einsum("aij,...ad->...dij", change, response)
    columns = [cartesian[..., :, j, k] for j, k in VOIGT_PAIRS]
    return torch.stack(columns, dim=-2)


def rotate_strain_response(
    response: torch.Tensor,
    node_irreps: o3.Irreps,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Rotate both the node-feature and symmetric-strain indices."""
    strain_features = voigt_response_to_strain_irreps(response)
    strain_transform = STRAIN_IRREPS.D_from_matrix(rotation)
    node_transform = o3.Irreps(node_irreps).D_from_matrix(rotation)
    rotated = torch.einsum(
        "as,...sh,bh->...ab",
        strain_transform,
        strain_features,
        node_transform,
    )
    return strain_irreps_to_voigt_response(rotated)
