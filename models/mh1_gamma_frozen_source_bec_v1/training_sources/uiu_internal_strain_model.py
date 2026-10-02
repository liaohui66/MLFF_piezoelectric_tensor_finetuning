#!/usr/bin/env python3
"""Atom-resolved O(3)-equivariant internal-strain (Lambda) readout."""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate

from piezo_irreps import PIEZO_IRREPS, irreps_to_voigt
from strain_response_irreps import STRAIN_IRREPS, voigt_response_to_strain_irreps
from uiu_equivariant_piezo_model import DEFAULT_NODE_IRREPS


PIEZO_BLOCK_SLICES = (
    slice(0, 3),
    slice(3, 6),
    slice(6, 11),
    slice(11, 18),
)


class AtomicInternalStrainHead(nn.Module):
    """Predict one 3x6 dF/dstrain tensor per atom with exact graph ASR."""

    def __init__(
        self,
        node_irreps: o3.Irreps = DEFAULT_NODE_IRREPS,
        scalar_channels: int = 32,
        vector_channels: int = 16,
        tensor_channels: int = 16,
        atomic_response: bool = False,
        atomic_response_feature_scales: torch.Tensor | None = None,
        atomic_response_mix_init: float = 0.1,
        output_block_scales: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
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
        self.context_readout = o3.FullyConnectedTensorProduct(
            self.gate.irreps_out,
            self.pre_irreps,
            PIEZO_IRREPS,
        )

        self.atomic_response_enabled = bool(atomic_response)
        if self.atomic_response_enabled:
            self.atomic_response_readout = o3.FullyConnectedTensorProduct(
                self.node_irreps,
                STRAIN_IRREPS,
                PIEZO_IRREPS,
            )
            self.atomic_response_mix = nn.Parameter(
                torch.tensor(
                    float(atomic_response_mix_init),
                    dtype=torch.get_default_dtype(),
                )
            )
        else:
            self.atomic_response_readout = None
            self.register_parameter("atomic_response_mix", None)

        if atomic_response_feature_scales is None:
            atomic_response_feature_scales = torch.ones(
                self.node_irreps.dim,
                dtype=torch.get_default_dtype(),
            )
        scales = torch.as_tensor(
            atomic_response_feature_scales,
            dtype=torch.get_default_dtype(),
        )
        if scales.shape != (self.node_irreps.dim,):
            raise ValueError(
                "atomic response scales must have shape "
                f"({self.node_irreps.dim},), got {tuple(scales.shape)}"
            )
        self.register_buffer(
            "atomic_response_feature_scales",
            scales.clamp_min(1.0e-8),
        )
        if output_block_scales is None:
            output_block_scales = torch.ones(
                len(PIEZO_BLOCK_SLICES),
                dtype=torch.get_default_dtype(),
            )
        output_scales = torch.as_tensor(
            output_block_scales,
            dtype=torch.get_default_dtype(),
        )
        if output_scales.shape != (len(PIEZO_BLOCK_SLICES),):
            raise ValueError(
                "output block scales must have shape "
                f"({len(PIEZO_BLOCK_SLICES)},), got {tuple(output_scales.shape)}"
            )
        self.register_buffer(
            "output_block_scales",
            output_scales.clamp_min(1.0e-8),
        )

    @staticmethod
    def enforce_acoustic_sum_rule(
        atom_irreps: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project atom tensors onto sum_a Lambda_a=0 for every graph."""
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if len(batch) else 0
        sums = atom_irreps.new_zeros((num_graphs, atom_irreps.shape[-1]))
        sums.index_add_(0, batch, atom_irreps)
        counts = atom_irreps.new_zeros((num_graphs, 1))
        counts.index_add_(
            0,
            batch,
            atom_irreps.new_ones((atom_irreps.shape[0], 1)),
        )
        means = sums / counts.clamp_min(1.0)
        projected = atom_irreps - means[batch]
        residual = projected.new_zeros((num_graphs, projected.shape[-1]))
        residual.index_add_(0, batch, projected)
        return projected, residual

    def response_irreps(self, atomic_response: torch.Tensor) -> torch.Tensor:
        if atomic_response.shape[-2:] != (6, self.node_irreps.dim):
            raise ValueError(
                "expected atomic response shape (n_atoms, 6, "
                f"{self.node_irreps.dim}), got {tuple(atomic_response.shape)}"
            )
        normalized = (
            atomic_response
            / self.atomic_response_feature_scales[None, None, :]
        )
        response = voigt_response_to_strain_irreps(normalized)
        num_atoms, strain_dim, node_dim = response.shape
        response_nodes = response.reshape(num_atoms * strain_dim, node_dim)
        strain_basis = torch.eye(
            strain_dim,
            device=response.device,
            dtype=response.dtype,
        )
        strain_basis = (
            strain_basis[None]
            .expand(num_atoms, -1, -1)
            .reshape(num_atoms * strain_dim, strain_dim)
        )
        return self.atomic_response_readout(
            response_nodes,
            strain_basis,
        ).reshape(num_atoms, strain_dim, PIEZO_IRREPS.dim).sum(dim=1)

    def forward_irreps(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        atomic_response: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> dict[str, torch.Tensor]:
        pre = self.pre(node_features)
        context = self.gate(self.tensor_product(pre, pre))
        raw = self.context_readout(context, pre)
        response = raw.new_zeros(raw.shape)
        if self.atomic_response_enabled:
            if atomic_response is None:
                raise ValueError("atomic-response head requires atomic_response")
            response = self.response_irreps(atomic_response)
            raw = raw + self.atomic_response_mix * response
        raw = torch.cat(
            [
                raw[:, block_slice] * self.output_block_scales[index]
                for index, block_slice in enumerate(PIEZO_BLOCK_SLICES)
            ],
            dim=1,
        )
        projected, asr_residual = self.enforce_acoustic_sum_rule(
            raw,
            batch,
            num_graphs,
        )
        return {
            "raw_irreps": raw,
            "lambda_irreps": projected,
            "response_irreps": response,
            "asr_residual_irreps": asr_residual,
        }

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        atomic_response: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.forward_irreps(
            node_features,
            batch,
            atomic_response,
            num_graphs,
        )
        output["lambda_voigt"] = irreps_to_voigt(output["lambda_irreps"])
        return output
