#!/usr/bin/env python3
"""Atom-to-graph O(3)-equivariant piezoelectric tensor readout."""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate

from piezo_irreps import PIEZO_IRREPS, irreps_to_voigt, project_irreps
from strain_response_irreps import (
    STRAIN_IRREPS,
    voigt_response_to_strain_irreps,
)


DEFAULT_NODE_IRREPS = o3.Irreps(
    "128x0e + 128x1o + 128x0e + 128x1o"
)
PIEZO_BLOCK_SLICES = (
    slice(0, 3),
    slice(3, 6),
    slice(6, 11),
    slice(11, 18),
)


class EquivariantPiezoHead(nn.Module):
    """Predict static and ionic piezo irreps from frozen MACELES atom features."""

    def __init__(
        self,
        node_irreps: o3.Irreps = DEFAULT_NODE_IRREPS,
        scalar_channels: int = 32,
        vector_channels: int = 16,
        tensor_channels: int = 16,
        volume_scale: float = 20.0,
        output_scales: tuple[float, float] = (1.0, 1.0),
        symmetry_mode: str = "none",
        amplitude_input_dim: int = 0,
        amplitude_hidden: tuple[int, ...] = (128, 64),
        amplitude_log_gain_clip: float = 3.0,
        amplitude_mode: str = "global",
        total_eijmax_amplitude: bool = False,
        total_eijmax_gain_clip: float = 3.0,
        atomic_response: bool = False,
        atomic_response_feature_scales: torch.Tensor | None = None,
        atomic_response_mix_init: float = 0.1,
        standard_amplitude_scaling: bool = True,
        pooled_amplitude_features: bool = False,
        amplitude_dropout: float = 0.0,
        direct_total_irrep_amplitude: bool = False,
        direct_total_irrep_amplitude_init: torch.Tensor | None = None,
        direct_total_irrep_gain_clip: float = 100.0,
        tail_ionic_amplitude: bool = False,
        tail_ionic_mode: str = "irrep",
        tail_ionic_gain_clip: float = 1.5,
        ionic_direction_residual: bool = False,
        ionic_contextual_direction_residual: bool = False,
        direction_context_scalar_channels: int = 16,
        direction_context_vector_channels: int = 8,
        direction_context_tensor_channels: int = 8,
        direction_response_scalar_channels: int = 16,
        direction_response_vector_channels: int = 8,
        direction_fused_vector_channels: int = 12,
        direction_fused_tensor_channels: int = 8,
        direction_fused_octupole_channels: int = 4,
    ):
        super().__init__()
        if symmetry_mode not in {"none", "soft", "hard"}:
            raise ValueError(f"unknown symmetry mode: {symmetry_mode}")

        self.node_irreps = o3.Irreps(node_irreps)
        self.pre_irreps = o3.Irreps(
            f"{scalar_channels}x0e + {vector_channels}x1o"
        )
        gate_scalars = o3.Irreps(f"{scalar_channels}x0e")
        gate_gates = o3.Irreps(f"{vector_channels + tensor_channels}x0e")
        gate_tensors = o3.Irreps(
            f"{vector_channels}x1o + {tensor_channels}x2e"
        )
        self.gate_input_irreps = gate_scalars + gate_gates + gate_tensors

        self.pre = o3.Linear(self.node_irreps, self.pre_irreps)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.pre_irreps,
            self.pre_irreps,
            self.gate_input_irreps,
        )
        self.gate = Gate(
            gate_scalars,
            [torch.nn.functional.silu],
            gate_gates,
            [torch.sigmoid],
            gate_tensors,
        )
        self.static_readout = o3.FullyConnectedTensorProduct(
            self.gate.irreps_out,
            self.pre_irreps,
            PIEZO_IRREPS,
        )
        self.ionic_readout = o3.FullyConnectedTensorProduct(
            self.gate.irreps_out,
            self.pre_irreps,
            PIEZO_IRREPS,
        )
        self.atomic_response_enabled = bool(atomic_response)
        if self.atomic_response_enabled:
            self.static_atomic_response_readout = o3.FullyConnectedTensorProduct(
                self.node_irreps,
                STRAIN_IRREPS,
                PIEZO_IRREPS,
            )
            self.ionic_atomic_response_readout = o3.FullyConnectedTensorProduct(
                self.node_irreps,
                STRAIN_IRREPS,
                PIEZO_IRREPS,
            )
            self.atomic_response_mix = nn.Parameter(
                torch.full(
                    (2,),
                    float(atomic_response_mix_init),
                    dtype=torch.get_default_dtype(),
                )
            )
        else:
            self.static_atomic_response_readout = None
            self.ionic_atomic_response_readout = None
            self.register_parameter("atomic_response_mix", None)
        if ionic_direction_residual and ionic_contextual_direction_residual:
            raise ValueError(
                "linear and contextual ionic direction residuals are mutually exclusive"
            )
        self.ionic_contextual_direction_residual_enabled = bool(
            ionic_contextual_direction_residual
        )
        self.ionic_direction_residual_enabled = bool(
            ionic_direction_residual
            or self.ionic_contextual_direction_residual_enabled
        )
        if self.ionic_direction_residual_enabled:
            if not self.atomic_response_enabled:
                raise ValueError(
                    "ionic direction residual requires atomic strain response"
                )
        if ionic_direction_residual:
            self.ionic_direction_residual_readout = (
                o3.FullyConnectedTensorProduct(
                    self.node_irreps,
                    STRAIN_IRREPS,
                    PIEZO_IRREPS,
                )
            )
            for parameter in self.ionic_direction_residual_readout.parameters():
                nn.init.zeros_(parameter)
        else:
            self.ionic_direction_residual_readout = None

        if self.ionic_contextual_direction_residual_enabled:
            self.direction_context_pre_irreps = o3.Irreps(
                f"{direction_context_scalar_channels}x0e + "
                f"{direction_context_vector_channels}x1o"
            )
            direction_gate_scalars = o3.Irreps(
                f"{direction_context_scalar_channels}x0e"
            )
            direction_gate_gates = o3.Irreps(
                f"{direction_context_vector_channels + direction_context_tensor_channels}x0e"
            )
            direction_gate_tensors = o3.Irreps(
                f"{direction_context_vector_channels}x1o + "
                f"{direction_context_tensor_channels}x2e"
            )
            direction_gate_input_irreps = (
                direction_gate_scalars
                + direction_gate_gates
                + direction_gate_tensors
            )
            self.direction_context_pre = o3.Linear(
                self.node_irreps,
                self.direction_context_pre_irreps,
            )
            self.direction_context_tensor_product = (
                o3.FullyConnectedTensorProduct(
                    self.direction_context_pre_irreps,
                    self.direction_context_pre_irreps,
                    direction_gate_input_irreps,
                )
            )
            self.direction_context_gate = Gate(
                direction_gate_scalars,
                [torch.nn.functional.silu],
                direction_gate_gates,
                [torch.sigmoid],
                direction_gate_tensors,
            )
            self.direction_response_irreps = o3.Irreps(
                f"{direction_response_scalar_channels}x0e + "
                f"{direction_response_vector_channels}x1o"
            )
            self.direction_response_projection = o3.Linear(
                self.node_irreps,
                self.direction_response_irreps,
            )
            self.direction_fused_response_irreps = o3.Irreps(
                f"{direction_fused_vector_channels}x1o + "
                f"{direction_fused_tensor_channels}x2o + "
                f"{direction_fused_octupole_channels}x3o"
            )
            self.direction_context_response_product = (
                o3.FullyConnectedTensorProduct(
                    self.direction_context_gate.irreps_out,
                    self.direction_response_irreps,
                    self.direction_fused_response_irreps,
                )
            )
            self.ionic_contextual_direction_readout = (
                o3.FullyConnectedTensorProduct(
                    self.direction_fused_response_irreps,
                    STRAIN_IRREPS,
                    PIEZO_IRREPS,
                )
            )
            # Preserve the initialized model exactly; contextual layers begin
            # receiving gradients after the readout takes its first step.
            for parameter in self.ionic_contextual_direction_readout.parameters():
                nn.init.zeros_(parameter)
        else:
            self.direction_context_pre_irreps = None
            self.direction_context_pre = None
            self.direction_context_tensor_product = None
            self.direction_context_gate = None
            self.direction_response_irreps = None
            self.direction_response_projection = None
            self.direction_fused_response_irreps = None
            self.direction_context_response_product = None
            self.ionic_contextual_direction_readout = None

        if atomic_response_feature_scales is None:
            atomic_response_feature_scales = torch.ones(
                self.node_irreps.dim,
                dtype=torch.get_default_dtype(),
            )
        atomic_response_feature_scales = torch.as_tensor(
            atomic_response_feature_scales,
            dtype=torch.get_default_dtype(),
        )
        if atomic_response_feature_scales.shape != (self.node_irreps.dim,):
            raise ValueError(
                "atomic response scales must have shape "
                f"({self.node_irreps.dim},), got "
                f"{tuple(atomic_response_feature_scales.shape)}"
            )
        self.register_buffer(
            "atomic_response_feature_scales",
            atomic_response_feature_scales.clamp_min(1e-8),
        )

        self.volume_scale = float(volume_scale)
        self.symmetry_mode = symmetry_mode
        self.amplitude_input_dim = int(amplitude_input_dim)
        self.pooled_amplitude_features = bool(pooled_amplitude_features)
        self.standard_amplitude_scaling = bool(standard_amplitude_scaling)
        self.direct_total_irrep_amplitude_enabled = bool(
            direct_total_irrep_amplitude
        )
        self.direct_total_irrep_gain_clip = float(direct_total_irrep_gain_clip)
        self.tail_ionic_amplitude_enabled = bool(tail_ionic_amplitude)
        if tail_ionic_mode not in {"global", "irrep"}:
            raise ValueError(f"unknown tail ionic mode: {tail_ionic_mode}")
        self.tail_ionic_mode = tail_ionic_mode
        self.tail_ionic_gain_clip = float(tail_ionic_gain_clip)
        self.pooled_amplitude_dim = 0
        if self.pooled_amplitude_features:
            self.amplitude_pool_irreps = o3.Irreps("16x0e + 8x1o")
            self.amplitude_pool = o3.Linear(
                self.node_irreps,
                self.amplitude_pool_irreps,
            )
            # scalar mean/rms, vector-norm mean/rms, and invariant response mean/rms
            self.pooled_amplitude_dim = 16 * 2 + 8 * 2 + 2
        else:
            self.amplitude_pool_irreps = None
            self.amplitude_pool = None
        self.effective_amplitude_input_dim = (
            self.amplitude_input_dim + self.pooled_amplitude_dim
        )
        self.amplitude_log_gain_clip = float(amplitude_log_gain_clip)
        if amplitude_mode not in {"global", "irrep"}:
            raise ValueError(f"unknown amplitude mode: {amplitude_mode}")
        self.amplitude_mode = amplitude_mode
        self.total_eijmax_amplitude_enabled = bool(total_eijmax_amplitude)
        self.total_eijmax_gain_clip = float(total_eijmax_gain_clip)
        if self.standard_amplitude_scaling and self.effective_amplitude_input_dim > 0:
            amplitude_layers = []
            previous = self.effective_amplitude_input_dim
            for hidden in amplitude_hidden:
                amplitude_layers.extend(
                    [
                        nn.Linear(previous, hidden),
                        nn.SiLU(),
                        nn.LayerNorm(hidden),
                    ]
                )
                if amplitude_dropout > 0:
                    amplitude_layers.append(nn.Dropout(amplitude_dropout))
                previous = hidden
            amplitude_outputs = 2 if amplitude_mode == "global" else 8
            final = nn.Linear(previous, amplitude_outputs)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            amplitude_layers.append(final)
            self.amplitude_mlp = nn.Sequential(*amplitude_layers)
        else:
            self.amplitude_mlp = None
        if self.total_eijmax_amplitude_enabled:
            if self.effective_amplitude_input_dim <= 0:
                raise ValueError(
                    "total_eijmax_amplitude requires invariant response features"
                )
            total_layers = []
            previous = self.effective_amplitude_input_dim
            for hidden in amplitude_hidden:
                total_layers.extend(
                    [
                        nn.Linear(previous, hidden),
                        nn.SiLU(),
                        nn.LayerNorm(hidden),
                    ]
                )
                if amplitude_dropout > 0:
                    total_layers.append(nn.Dropout(amplitude_dropout))
                previous = hidden
            final = nn.Linear(previous, 1)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            total_layers.append(final)
            self.total_eijmax_mlp = nn.Sequential(*total_layers)
        else:
            self.total_eijmax_mlp = None
        if self.direct_total_irrep_amplitude_enabled:
            if self.effective_amplitude_input_dim <= 0:
                raise ValueError(
                    "direct total-irrep amplitude requires invariant features"
                )
            direct_layers = []
            previous = self.effective_amplitude_input_dim
            for hidden in amplitude_hidden:
                direct_layers.extend(
                    [
                        nn.Linear(previous, hidden),
                        nn.SiLU(),
                        nn.LayerNorm(hidden),
                    ]
                )
                if amplitude_dropout > 0:
                    direct_layers.append(nn.Dropout(amplitude_dropout))
                previous = hidden
            final = nn.Linear(previous, len(PIEZO_BLOCK_SLICES))
            nn.init.zeros_(final.weight)
            if direct_total_irrep_amplitude_init is None:
                initial_amplitudes = torch.ones(len(PIEZO_BLOCK_SLICES))
            else:
                initial_amplitudes = torch.as_tensor(
                    direct_total_irrep_amplitude_init,
                    dtype=final.bias.dtype,
                )
            initial_amplitudes = initial_amplitudes.clamp_min(1e-4)
            with torch.no_grad():
                final.bias.copy_(torch.log(torch.expm1(initial_amplitudes)))
            direct_layers.append(final)
            self.direct_total_irrep_mlp = nn.Sequential(*direct_layers)
        else:
            self.direct_total_irrep_mlp = None
        if self.tail_ionic_amplitude_enabled:
            if self.effective_amplitude_input_dim <= 0:
                raise ValueError(
                    "tail ionic amplitude requires invariant response features"
                )
            tail_layers = []
            previous = self.effective_amplitude_input_dim
            for hidden in amplitude_hidden:
                tail_layers.extend(
                    [
                        nn.Linear(previous, hidden),
                        nn.SiLU(),
                        nn.LayerNorm(hidden),
                    ]
                )
                if amplitude_dropout > 0:
                    tail_layers.append(nn.Dropout(amplitude_dropout))
                previous = hidden
            gain_outputs = 1 if tail_ionic_mode == "global" else len(PIEZO_BLOCK_SLICES)
            final = nn.Linear(previous, 1 + gain_outputs)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            tail_layers.append(final)
            self.tail_ionic_mlp = nn.Sequential(*tail_layers)
        else:
            self.tail_ionic_mlp = None
        self.register_buffer(
            "output_scales",
            torch.as_tensor(output_scales, dtype=torch.get_default_dtype()),
        )

    def set_output_scales(self, scales: torch.Tensor) -> None:
        if scales.shape != (2,):
            raise ValueError(f"expected two output scales, got {tuple(scales.shape)}")
        self.output_scales.copy_(
            scales.to(device=self.output_scales.device, dtype=self.output_scales.dtype)
        )

    def forward_irreps(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        symmetry_rotations: list[torch.Tensor] | None = None,
        amplitude_features: torch.Tensor | None = None,
        atomic_response: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        pre = self.pre(node_features)
        hidden = self.gate(self.tensor_product(pre, pre))
        atom_static = self.static_readout(hidden, pre)
        atom_ionic = self.ionic_readout(hidden, pre)
        normalized_response = None
        strain_response = None
        atom_direction_residual = None
        if self.atomic_response_enabled:
            if atomic_response is None:
                raise ValueError(
                    "atomic response branch requires atom-resolved six-strain features"
                )
            if atomic_response.shape != (
                node_features.shape[0],
                6,
                self.node_irreps.dim,
            ):
                raise ValueError(
                    "expected atomic response shape "
                    f"({node_features.shape[0]}, 6, {self.node_irreps.dim}), got "
                    f"{tuple(atomic_response.shape)}"
                )
            normalized_response = (
                atomic_response
                / self.atomic_response_feature_scales[None, None, :]
            )
            strain_response = voigt_response_to_strain_irreps(
                normalized_response
            )
            num_atoms = strain_response.shape[0]
            strain_dim = STRAIN_IRREPS.dim
            response_nodes = strain_response.reshape(
                num_atoms * strain_dim,
                self.node_irreps.dim,
            )
            strain_basis = torch.eye(
                strain_dim,
                device=node_features.device,
                dtype=node_features.dtype,
            )
            strain_basis = (
                strain_basis[None, :, :]
                .expand(num_atoms, -1, -1)
                .reshape(num_atoms * strain_dim, strain_dim)
            )
            response_static = self.static_atomic_response_readout(
                response_nodes,
                strain_basis,
            ).reshape(num_atoms, strain_dim, PIEZO_IRREPS.dim).sum(dim=1)
            response_ionic = self.ionic_atomic_response_readout(
                response_nodes,
                strain_basis,
            ).reshape(num_atoms, strain_dim, PIEZO_IRREPS.dim).sum(dim=1)
            if self.ionic_direction_residual_readout is not None:
                atom_direction_residual = self.ionic_direction_residual_readout(
                    response_nodes,
                    strain_basis,
                ).reshape(
                    num_atoms,
                    strain_dim,
                    PIEZO_IRREPS.dim,
                ).sum(dim=1)
            elif self.ionic_contextual_direction_readout is not None:
                context_pre = self.direction_context_pre(node_features)
                context = self.direction_context_gate(
                    self.direction_context_tensor_product(
                        context_pre,
                        context_pre,
                    )
                )
                response_projected = self.direction_response_projection(
                    response_nodes
                )
                repeated_context = (
                    context[:, None, :]
                    .expand(-1, strain_dim, -1)
                    .reshape(num_atoms * strain_dim, context.shape[-1])
                )
                contextual_response = self.direction_context_response_product(
                    repeated_context,
                    response_projected,
                )
                atom_direction_residual = (
                    self.ionic_contextual_direction_readout(
                        contextual_response,
                        strain_basis,
                    )
                    .reshape(num_atoms, strain_dim, PIEZO_IRREPS.dim)
                    .sum(dim=1)
                )
            atom_static = (
                atom_static
                + self.atomic_response_mix[0] * response_static
            )
            atom_ionic = (
                atom_ionic
                + self.atomic_response_mix[1] * response_ionic
            )

        num_graphs = int(volumes.shape[0])
        graph_static = atom_static.new_zeros((num_graphs, PIEZO_IRREPS.dim))
        graph_ionic = atom_ionic.new_zeros((num_graphs, PIEZO_IRREPS.dim))
        graph_static.index_add_(0, batch, atom_static)
        graph_ionic.index_add_(0, batch, atom_ionic)
        normalization = self.volume_scale / volumes.clamp_min(1e-8)
        graph_static = graph_static * normalization[:, None] * self.output_scales[0]
        graph_ionic = graph_ionic * normalization[:, None] * self.output_scales[1]
        graph_direction_residual = graph_ionic.new_zeros(
            (num_graphs, PIEZO_IRREPS.dim)
        )
        if atom_direction_residual is not None:
            graph_direction_residual.index_add_(
                0,
                batch,
                atom_direction_residual,
            )
            graph_direction_residual = (
                graph_direction_residual
                * normalization[:, None]
                * self.output_scales[1]
            )
        raw = torch.stack([graph_static, graph_ionic], dim=1)
        augmented_amplitude_features = amplitude_features
        if self.pooled_amplitude_features:
            pooled = self.amplitude_pool(node_features)
            scalar_features = pooled[:, :16]
            vector_features = pooled[:, 16:].reshape(-1, 8, 3)
            vector_norms = torch.linalg.vector_norm(vector_features, dim=-1)
            counts = pooled.new_zeros((num_graphs, 1))
            counts.index_add_(
                0,
                batch,
                pooled.new_ones((pooled.shape[0], 1)),
            )
            counts = counts.clamp_min(1.0)

            def graph_mean(values):
                result = values.new_zeros((num_graphs, values.shape[1]))
                result.index_add_(0, batch, values)
                return result / counts

            scalar_mean = graph_mean(scalar_features)
            scalar_rms = torch.sqrt(
                graph_mean(torch.square(scalar_features)).clamp_min(1e-12)
            )
            vector_mean = graph_mean(vector_norms)
            vector_rms = torch.sqrt(
                graph_mean(torch.square(vector_norms)).clamp_min(1e-12)
            )
            if strain_response is None:
                response_mean = pooled.new_zeros((num_graphs, 1))
                response_rms = pooled.new_zeros((num_graphs, 1))
            else:
                # Use the orthonormal 0e+2e strain basis before taking a norm.
                response_norms = torch.linalg.vector_norm(
                    strain_response.reshape(strain_response.shape[0], -1),
                    dim=1,
                    keepdim=True,
                )
                response_mean = graph_mean(response_norms)
                response_rms = torch.sqrt(
                    graph_mean(torch.square(response_norms)).clamp_min(1e-12)
                )
            learned_invariants = torch.cat(
                [
                    scalar_mean,
                    scalar_rms,
                    vector_mean,
                    vector_rms,
                    response_mean,
                    response_rms,
                ],
                dim=1,
            )
            if augmented_amplitude_features is None:
                augmented_amplitude_features = learned_invariants
            else:
                augmented_amplitude_features = torch.cat(
                    [augmented_amplitude_features, learned_invariants],
                    dim=1,
                )
        gains = raw.new_ones((num_graphs, 2, 1))
        total_eijmax_gain = raw.new_ones((num_graphs, 1))
        direct_total_irrep_amplitudes = raw.new_zeros(
            (num_graphs, len(PIEZO_BLOCK_SLICES))
        )
        tail_ionic_gate_logits = raw.new_zeros((num_graphs, 1))
        tail_ionic_gains = raw.new_ones(
            (
                num_graphs,
                1 if self.tail_ionic_mode == "global" else len(PIEZO_BLOCK_SLICES),
            )
        )
        if self.amplitude_mlp is not None:
            if augmented_amplitude_features is None:
                raise ValueError("amplitude branch requires invariant response features")
            log_gains = self.amplitude_mlp(augmented_amplitude_features)
            log_gains = torch.clamp(
                log_gains,
                min=-self.amplitude_log_gain_clip,
                max=self.amplitude_log_gain_clip,
            )
            if self.amplitude_mode == "global":
                gains = torch.exp(log_gains).reshape(num_graphs, 2, 1)
                raw = raw * gains
            else:
                gains = torch.exp(log_gains).reshape(num_graphs, 2, 4)
                scaled_blocks = [
                    raw[:, :, block_slice] * gains[:, :, block_index, None]
                    for block_index, block_slice in enumerate(PIEZO_BLOCK_SLICES)
                ]
                raw = torch.cat(scaled_blocks, dim=-1)

        if self.total_eijmax_mlp is not None:
            if augmented_amplitude_features is None:
                raise ValueError(
                    "total-eijmax amplitude branch requires invariant response features"
                )
            log_gain = self.total_eijmax_mlp(augmented_amplitude_features)
            log_gain = torch.clamp(
                log_gain,
                min=-self.total_eijmax_gain_clip,
                max=self.total_eijmax_gain_clip,
            )
            total_eijmax_gain = torch.exp(log_gain)
            raw = raw * total_eijmax_gain[:, None, :]

        pre_tail_raw = raw
        if self.tail_ionic_mlp is not None:
            if augmented_amplitude_features is None:
                raise ValueError(
                    "tail ionic amplitude branch requires invariant response features"
                )
            tail_output = self.tail_ionic_mlp(augmented_amplitude_features)
            tail_ionic_gate_logits = tail_output[:, :1]
            residual_log_gains = torch.clamp(
                tail_output[:, 1:],
                min=-self.tail_ionic_gain_clip,
                max=self.tail_ionic_gain_clip,
            )
            gated_log_gains = torch.sigmoid(tail_ionic_gate_logits) * residual_log_gains
            tail_ionic_gains = torch.exp(gated_log_gains)
            if self.tail_ionic_mode == "global":
                corrected_ionic = raw[:, 1] * tail_ionic_gains
            else:
                corrected_ionic = torch.cat(
                    [
                        raw[:, 1, block_slice] * tail_ionic_gains[:, block_index, None]
                        for block_index, block_slice in enumerate(PIEZO_BLOCK_SLICES)
                    ],
                    dim=-1,
                )
            raw = torch.stack([raw[:, 0], corrected_ionic], dim=1)

        pre_direction_raw = raw
        if atom_direction_residual is not None:
            raw = torch.stack(
                [
                    raw[:, 0],
                    raw[:, 1] + graph_direction_residual,
                ],
                dim=1,
            )

        if self.direct_total_irrep_mlp is not None:
            direct_total_irrep_amplitudes = torch.nn.functional.softplus(
                self.direct_total_irrep_mlp(augmented_amplitude_features)
            )
            raw_total = raw[:, 0] + raw[:, 1]
            scaled_blocks = []
            for block_index, block_slice in enumerate(PIEZO_BLOCK_SLICES):
                current_norm = torch.linalg.vector_norm(
                    raw_total[:, block_slice],
                    dim=1,
                ).clamp_min(1e-6)
                block_gain = (
                    direct_total_irrep_amplitudes[:, block_index] / current_norm
                ).clamp(max=self.direct_total_irrep_gain_clip)
                scaled_blocks.append(
                    raw[:, :, block_slice] * block_gain[:, None, None]
                )
            raw = torch.cat(scaled_blocks, dim=-1)

        if self.symmetry_mode == "hard":
            if symmetry_rotations is None:
                raise ValueError("hard symmetry mode requires point-group rotations")
            static = project_irreps(raw[:, 0], symmetry_rotations)
            ionic = project_irreps(raw[:, 1], symmetry_rotations)
            raw = torch.stack([static, ionic], dim=1)
        total = raw[:, 0] + raw[:, 1]
        pre_tail_total = pre_tail_raw[:, 0] + pre_tail_raw[:, 1]
        pre_direction_total = pre_direction_raw[:, 0] + pre_direction_raw[:, 1]
        return (
            torch.cat([raw, total[:, None]], dim=1),
            torch.cat([pre_tail_raw, pre_tail_total[:, None]], dim=1),
            torch.cat(
                [pre_direction_raw, pre_direction_total[:, None]],
                dim=1,
            ),
            graph_direction_residual,
            hidden,
            gains,
            total_eijmax_gain,
            direct_total_irrep_amplitudes,
            tail_ionic_gate_logits,
            tail_ionic_gains,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        symmetry_rotations: list[torch.Tensor] | None = None,
        amplitude_features: torch.Tensor | None = None,
        atomic_response: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        (
            irreps,
            pre_tail_irreps,
            pre_direction_irreps,
            ionic_direction_residual,
            hidden,
            gains,
            total_eijmax_gain,
            direct_amplitudes,
            tail_gate_logits,
            tail_ionic_gains,
        ) = self.forward_irreps(
            node_features,
            batch,
            volumes,
            symmetry_rotations,
            amplitude_features,
            atomic_response,
        )
        return {
            "irreps": irreps,
            "voigt": irreps_to_voigt(irreps),
            "pre_tail_irreps": pre_tail_irreps,
            "pre_tail_voigt": irreps_to_voigt(pre_tail_irreps),
            "pre_direction_irreps": pre_direction_irreps,
            "pre_direction_voigt": irreps_to_voigt(pre_direction_irreps),
            "ionic_direction_residual_irreps": ionic_direction_residual,
            "ionic_direction_residual_voigt": irreps_to_voigt(
                ionic_direction_residual
            ),
            "hidden": hidden,
            "amplitude_gains": gains,
            "total_eijmax_gain": total_eijmax_gain,
            "direct_total_irrep_amplitudes": direct_amplitudes,
            "tail_ionic_gate_logits": tail_gate_logits,
            "tail_ionic_gains": tail_ionic_gains,
            "atomic_response_mix": self.atomic_response_mix,
        }

    def symmetry_residual(
        self,
        irreps: torch.Tensor,
        symmetry_rotations: list[torch.Tensor],
    ) -> torch.Tensor:
        projected = torch.stack(
            [
                project_irreps(irreps[:, index], symmetry_rotations)
                for index in range(irreps.shape[1])
            ],
            dim=1,
        )
        return torch.mean((irreps - projected) ** 2)
