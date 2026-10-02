#!/usr/bin/env python3
"""Nonlinear equivariant adaptation of frozen UIU atomic representations."""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate

from uiu_equivariant_piezo_model import DEFAULT_NODE_IRREPS
from uiu_internal_strain_model import AtomicInternalStrainHead


class EquivariantFeatureTangentAdapter(nn.Module):
    """Adapt node features and their six strain tangents consistently.

    The frozen cache contains ``h`` and ``dh / d strain_J``.  A nonlinear
    equivariant residual ``f(h)`` defines the adapted representation
    ``h' = h + f(h)``.  Its strain tangent is evaluated with the same central
    strain step used to construct the cache:

        dh'/de_J = dh/de_J + [f(h+s dh/de_J)-f(h-s dh/de_J)]/(2s).

    The final equivariant linear map is zero initialized, so the complete
    wrapped model exactly reproduces the source checkpoint at epoch zero.
    """

    def __init__(
        self,
        node_irreps: o3.Irreps = DEFAULT_NODE_IRREPS,
        scalar_channels: int = 32,
        vector_channels: int = 16,
        tensor_channels: int = 16,
        feature_scales: torch.Tensor | None = None,
        tangent_step: float = 0.005,
    ) -> None:
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.tangent_step = float(tangent_step)
        if self.tangent_step <= 0.0:
            raise ValueError("tangent_step must be positive")

        self.pre_irreps = o3.Irreps(
            f"{scalar_channels}x0e + {vector_channels}x1o"
        )
        gate_scalars = o3.Irreps(f"{scalar_channels}x0e")
        gate_gates = o3.Irreps(f"{vector_channels + tensor_channels}x0e")
        gate_tensors = o3.Irreps(
            f"{vector_channels}x1o + {tensor_channels}x2e"
        )
        gate_input_irreps = gate_scalars + gate_gates + gate_tensors

        self.pre = o3.Linear(self.node_irreps, self.pre_irreps)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.pre_irreps,
            self.pre_irreps,
            gate_input_irreps,
        )
        self.gate = Gate(
            gate_scalars,
            [torch.nn.functional.silu],
            gate_gates,
            [torch.sigmoid],
            gate_tensors,
        )
        self.post = o3.Linear(self.gate.irreps_out, self.node_irreps)
        self._zero_initialize_post()

        if feature_scales is None:
            feature_scales = torch.ones(
                self.node_irreps.dim,
                dtype=torch.get_default_dtype(),
            )
        scales = torch.as_tensor(feature_scales, dtype=torch.get_default_dtype())
        if scales.shape != (self.node_irreps.dim,):
            raise ValueError(
                f"feature scales must have shape ({self.node_irreps.dim},), "
                f"got {tuple(scales.shape)}"
            )
        self.register_buffer("feature_scales", scales.clamp_min(1.0e-8))

    def _zero_initialize_post(self) -> None:
        with torch.no_grad():
            for parameter in self.post.parameters():
                parameter.zero_()

    def residual(self, node_features: torch.Tensor) -> torch.Tensor:
        normalized = node_features / self.feature_scales[None, :]
        hidden = self.pre(normalized)
        hidden = self.gate(self.tensor_product(hidden, hidden))
        return self.post(hidden) * self.feature_scales[None, :]

    def forward(
        self,
        node_features: torch.Tensor,
        atomic_response: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        feature_residual = self.residual(node_features)
        adapted_features = node_features + feature_residual
        if atomic_response is None:
            return adapted_features, None, feature_residual, None
        if atomic_response.shape[-2:] != (6, self.node_irreps.dim):
            raise ValueError(
                "expected atomic response shape (n_atoms, 6, "
                f"{self.node_irreps.dim}), got {tuple(atomic_response.shape)}"
            )

        step = self.tangent_step
        shifted = step * atomic_response
        plus = (node_features[:, None, :] + shifted).reshape(
            -1, self.node_irreps.dim
        )
        minus = (node_features[:, None, :] - shifted).reshape(
            -1, self.node_irreps.dim
        )
        response_residual = (
            (self.residual(plus) - self.residual(minus)) / (2.0 * step)
        ).reshape_as(atomic_response)
        adapted_response = atomic_response + response_residual
        return (
            adapted_features,
            adapted_response,
            feature_residual,
            response_residual,
        )


class FeatureAdaptedInternalStrainHead(nn.Module):
    """Apply a nonlinear representation adapter before a proven response head."""

    def __init__(
        self,
        head: AtomicInternalStrainHead,
        adapter: EquivariantFeatureTangentAdapter,
    ) -> None:
        super().__init__()
        self.head = head
        self.adapter = adapter

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        atomic_response: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> dict[str, torch.Tensor]:
        (
            adapted_features,
            adapted_response,
            feature_residual,
            response_residual,
        ) = self.adapter(node_features, atomic_response)
        output = self.head(
            adapted_features,
            batch,
            adapted_response,
            num_graphs,
        )
        output["feature_adapter_residual"] = feature_residual
        output["feature_adapter_residual_normalized"] = (
            feature_residual / self.adapter.feature_scales[None, :]
        )
        if response_residual is not None:
            output["response_adapter_residual"] = response_residual
            output["response_adapter_residual_normalized"] = (
                response_residual
                / self.head.atomic_response_feature_scales[None, None, :]
            )
        return output
