"""Equivariant displacement-response heads with optional intra-cell attention."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from e3nn import o3
from e3nn.nn import Gate

from piezo_irreps import PIEZO_IRREPS, irreps_to_voigt
from strain_response_irreps import STRAIN_IRREPS, voigt_response_to_strain_irreps
from uiu_internal_strain_model import AtomicInternalStrainHead, PIEZO_BLOCK_SLICES


ARCHITECTURE_VERSION = 1
DEFAULT_SCALAR_CHANNELS = 32
DEFAULT_VECTOR_CHANNELS = 16
DEFAULT_TENSOR_CHANNELS = 16
DEFAULT_RESPONSE_CHANNELS = 4
DEFAULT_ATTENTION_LAYERS = 2
DEFAULT_ATTENTION_SCALAR_CHANNELS = 32
DEFAULT_ATTENTION_HEADS = 1
DEFAULT_GEOMETRY_RADIAL_CHANNELS = 4


def _validate_copy_scales(scales: torch.Tensor, irreps: o3.Irreps) -> None:
    if scales.shape != (irreps.dim,):
        raise ValueError(
            f"feature scales must have shape ({irreps.dim},), got {tuple(scales.shape)}"
        )
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise ValueError("feature scales must be finite and strictly positive")
    for block, (multiplicity, irrep) in zip(irreps.slices(), irreps):
        copies = scales[block].reshape(multiplicity, irrep.dim)
        spread = copies.amax(dim=-1) - copies.amin(dim=-1)
        if torch.any(spread > 1.0e-6 * copies.mean(dim=-1).clamp_min(1.0)):
            raise ValueError(
                "feature scales must be identical across each irrep copy's m components"
            )


def channel_invariants(features: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """One rotation-invariant squared norm per irrep copy."""
    if features.shape[-1] != irreps.dim:
        raise ValueError(f"expected {irreps.dim} channels, got {features.shape[-1]}")
    return torch.cat(
        [
            torch.log1p(
                features[:, section]
                .reshape(-1, multiplicity, irrep.dim)
                .square()
                .mean(dim=-1)
            )
            for section, (multiplicity, irrep) in zip(irreps.slices(), irreps)
        ],
        dim=-1,
    )


def gate_irrep_copies(
    features: torch.Tensor,
    gates: torch.Tensor,
    irreps: o3.Irreps,
) -> torch.Tensor:
    """Apply one scalar gate per irrep copy, shared over magnetic components."""
    if gates.shape[-1] != irreps.num_irreps:
        raise ValueError(
            f"expected {irreps.num_irreps} irrep-copy gates, got {gates.shape[-1]}"
        )
    blocks = []
    gate_offset = 0
    for section, (multiplicity, irrep) in zip(irreps.slices(), irreps):
        block = features[:, section].reshape(-1, multiplicity, irrep.dim)
        gate = gates[:, gate_offset : gate_offset + multiplicity]
        blocks.append((block * gate[..., None]).reshape(len(features), -1))
        gate_offset += multiplicity
    return torch.cat(blocks, dim=-1)


def route_attention_values(
    weights: torch.Tensor,
    values: torch.Tensor,
    irreps: o3.Irreps,
    mode: str,
) -> torch.Tensor:
    """Route head attention through values while preserving complete irreps.

    ``shared_mean`` is the original implementation: every head sees the full
    value tensor and the head messages are averaged. ``irrep_copy_split``
    assigns whole copies of each irrep to heads and restores the original
    irrep-copy ordering after aggregation.
    """
    if weights.ndim != 3:
        raise ValueError(f"attention weights must have shape (heads, atoms, atoms), got {tuple(weights.shape)}")
    if values.ndim != 2 or values.shape[0] != weights.shape[1]:
        raise ValueError("attention values and weights have incompatible shapes")
    if weights.shape[1] != weights.shape[2]:
        raise ValueError("attention weights must be square within one structure")
    if mode == "shared_mean":
        return torch.einsum("hab,bd->had", weights, values).mean(dim=0)
    if mode != "irrep_copy_split":
        raise ValueError(f"unknown attention value routing mode: {mode}")

    heads = int(weights.shape[0])
    routed_blocks = []
    for section, (multiplicity, irrep) in zip(irreps.slices(), irreps):
        if multiplicity % heads != 0:
            raise ValueError(
                f"irrep multiplicity {multiplicity} is not divisible by {heads} heads"
            )
        block = values[:, section].reshape(values.shape[0], multiplicity, irrep.dim)
        block = block.reshape(values.shape[0], heads, multiplicity // heads, irrep.dim)
        routed = torch.einsum("hab,bhcm->ahcm", weights, block)
        routed_blocks.append(routed.reshape(values.shape[0], -1))
    return torch.cat(routed_blocks, dim=-1)


def local_geometry_irreps(radial_channels: int) -> o3.Irreps:
    """Irreps of radial-shell-resolved local spherical-harmonic moments."""
    if int(radial_channels) < 1:
        raise ValueError("radial_channels must be positive")
    radial_channels = int(radial_channels)
    return o3.Irreps(
        f"{radial_channels}x0e + {radial_channels}x1o + "
        f"{radial_channels}x2e + {radial_channels}x3o"
    )


def local_geometry_features_from_graph(
    data: dict[str, torch.Tensor],
    cutoff: float,
    radial_channels: int = DEFAULT_GEOMETRY_RADIAL_CHANNELS,
) -> torch.Tensor:
    """Build periodic, atom-resolved equivariant neighbor geometry features.

    Each atom receives radial-shell-weighted sums of spherical harmonics through
    l=3. The receiver index follows MACE's edge convention, and the vectors are
    obtained from the graph's periodic unit shifts, so no Cartesian coordinate
    origin enters the feature.
    """
    from mace.modules.utils import prepare_graph

    graph = {key: value for key, value in data.items()}
    context = prepare_graph(
        graph,
        compute_virials=False,
        compute_stress=False,
        compute_displacement=False,
        lammps_mliap=False,
    )
    vectors = context.vectors
    lengths = context.lengths.reshape(-1)
    if vectors.ndim != 2 or vectors.shape[-1] != 3:
        raise ValueError("graph vectors must have shape (edges, 3)")
    if len(vectors) == 0:
        return vectors.new_zeros((data["positions"].shape[0], local_geometry_irreps(radial_channels).dim))
    if not torch.isfinite(vectors).all() or not torch.isfinite(lengths).all():
        raise ValueError("non-finite periodic graph geometry")

    angular_irreps = o3.Irreps("0e + 1o + 2e + 3o")
    angular = o3.spherical_harmonics(
        angular_irreps,
        vectors,
        normalize=True,
        normalization="component",
    )
    centers = torch.linspace(
        0.5,
        max(float(cutoff) - 0.25, 0.5),
        int(radial_channels),
        device=lengths.device,
        dtype=lengths.dtype,
    )
    width = max(float(cutoff) / max(int(radial_channels), 1), 0.25)
    radial = torch.exp(-((lengths[:, None] - centers[None, :]) / width) ** 2)
    envelope = 0.5 * (
        1.0 + torch.cos(math.pi * lengths.clamp_min(0.0) / max(float(cutoff), 1.0e-6))
    )
    radial = radial * envelope.clamp_min(0.0)[:, None]
    receiver = data["edge_index"][1]
    n_atoms = int(data["positions"].shape[0])
    blocks = []
    for section in angular_irreps.slices():
        angular_block = angular[:, section]
        aggregated = angular_block.new_zeros((n_atoms, int(radial_channels), angular_block.shape[-1]))
        for shell in range(int(radial_channels)):
            aggregated[:, shell].index_add_(
                0,
                receiver,
                angular_block * radial[:, shell, None],
            )
        blocks.append(aggregated.reshape(n_atoms, -1))
    result = torch.cat(blocks, dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("non-finite local geometry features")
    return result


class CollectiveAttentionBlock(nn.Module):
    """Within-structure attention whose scores and updates preserve O(3)."""

    def __init__(
        self,
        irreps: o3.Irreps,
        scalar_channels: int,
        scalar_state_channels: int,
        attention_heads: int = DEFAULT_ATTENTION_HEADS,
        attention_value_mode: str = "shared_mean",
    ) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.query_key_dim = int(scalar_channels)
        self.attention_heads = int(attention_heads)
        self.attention_value_mode = str(attention_value_mode)
        if self.attention_value_mode not in {"shared_mean", "irrep_copy_split"}:
            raise ValueError(
                "attention_value_mode must be 'shared_mean' or 'irrep_copy_split'"
            )
        if self.attention_heads < 1 or self.query_key_dim % self.attention_heads != 0:
            raise ValueError(
                "attention_heads must be positive and divide attention scalar channels"
            )
        if self.attention_value_mode == "irrep_copy_split":
            for multiplicity, _ in self.irreps:
                if multiplicity % self.attention_heads != 0:
                    raise ValueError(
                        "irrep_copy_split requires every irrep multiplicity to be "
                        "divisible by attention_heads"
                    )
        self.query_head_dim = self.query_key_dim // self.attention_heads
        self.invariant_state = nn.Sequential(
            nn.Linear(scalar_state_channels + self.irreps.num_irreps, scalar_channels),
            nn.LayerNorm(scalar_channels),
            nn.SiLU(),
        )
        self.query = nn.Linear(scalar_channels, scalar_channels, bias=False)
        self.key = nn.Linear(scalar_channels, scalar_channels, bias=False)
        self.value = o3.Linear(self.irreps, self.irreps)
        self.message = o3.Linear(self.irreps, self.irreps)
        self.copy_gates = nn.Linear(scalar_channels, self.irreps.num_irreps)
        if self.attention_heads == 1:
            # Keep the scalar parameter and exact H=1 arithmetic compatible with
            # the original A2 checkpoint.
            self.coherence = nn.Parameter(torch.tensor(0.1))
        else:
            self.coherence = nn.Parameter(torch.full((self.attention_heads,), 0.1))

    def forward(
        self,
        features: torch.Tensor,
        scalar_state: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        invariants = channel_invariants(features, self.irreps)
        state = self.invariant_state(torch.cat([scalar_state, invariants], dim=-1))
        query = self.query(state)
        key = self.key(state)
        value = self.value(features)
        normalized = F.normalize(features, dim=-1, eps=1.0e-8)
        messages = torch.zeros_like(value)

        for graph_index in range(num_graphs):
            indices = torch.where(batch == graph_index)[0]
            if indices.numel() == 0:
                continue
            graph_query = query.index_select(0, indices)
            graph_key = key.index_select(0, indices)
            graph_value = value.index_select(0, indices)
            graph_features = normalized.index_select(0, indices)
            if self.attention_heads == 1:
                logits = graph_query @ graph_key.transpose(0, 1)
                logits = logits / math.sqrt(self.query_key_dim)
                logits = logits + self.coherence.tanh() * (
                    graph_features @ graph_features.transpose(0, 1)
                )
                weights = torch.softmax(logits, dim=-1)
                graph_message = weights @ graph_value
            else:
                head_query = graph_query.reshape(
                    graph_query.shape[0], self.attention_heads, self.query_head_dim
                )
                head_key = graph_key.reshape(
                    graph_key.shape[0], self.attention_heads, self.query_head_dim
                )
                logits = torch.einsum("ahd,bhd->hab", head_query, head_key)
                logits = logits / math.sqrt(self.query_head_dim)
                pairwise = graph_features @ graph_features.transpose(0, 1)
                logits = logits + self.coherence.tanh()[:, None, None] * pairwise[None]
                weights = torch.softmax(logits, dim=-1)
                graph_message = route_attention_values(
                    weights,
                    graph_value,
                    self.irreps,
                    self.attention_value_mode,
                )
                if self.attention_value_mode == "shared_mean":
                    # Keep the output scale comparable to H=1 while retaining
                    # independent atom-association patterns per head.
                    pass
            messages = messages.index_copy(0, indices, graph_message)

        updated = (features + self.message(messages)) / math.sqrt(2.0)
        gates = 2.0 * torch.sigmoid(self.copy_gates(state))
        updated = gate_irrep_copies(updated, gates, self.irreps)
        return updated, (scalar_state + state) / math.sqrt(2.0)


class CollectiveAttentionDisplacementHead(nn.Module):
    """Predict atom-resolved Gamma, retaining strain-response irreps until readout.

    The response representation contains both even (0e/2e) and odd
    (1o/2o/3o) channels. Environmental tensor products convert even response
    information into legal odd displacement-response outputs when permitted by
    O(3) coupling rules.
    """

    architecture_version = ARCHITECTURE_VERSION
    target_kind = "displacement"
    voigt_order = ("xx", "yy", "zz", "yz", "xz", "xy")
    strain_amplitude = 0.005

    def __init__(
        self,
        node_irreps: str | o3.Irreps,
        scalar_channels: int = DEFAULT_SCALAR_CHANNELS,
        vector_channels: int = DEFAULT_VECTOR_CHANNELS,
        tensor_channels: int = DEFAULT_TENSOR_CHANNELS,
        response_channels: int = DEFAULT_RESPONSE_CHANNELS,
        attention_layers: int = DEFAULT_ATTENTION_LAYERS,
        attention_scalar_channels: int = DEFAULT_ATTENTION_SCALAR_CHANNELS,
        attention_heads: int = DEFAULT_ATTENTION_HEADS,
        attention_value_mode: str = "shared_mean",
        atomic_response_feature_scales: torch.Tensor | None = None,
        output_block_scales: torch.Tensor | None = None,
        local_geometry: bool = False,
        geometry_radial_channels: int = DEFAULT_GEOMETRY_RADIAL_CHANNELS,
        local_geometry_feature_scales: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if response_channels < 1 or attention_layers < 0:
            raise ValueError("response_channels must be positive; attention_layers nonnegative")
        if min(scalar_channels, vector_channels, tensor_channels, attention_scalar_channels) < 1:
            raise ValueError("all channel counts must be positive")

        self.node_irreps = o3.Irreps(node_irreps)
        self.scalar_channels = int(scalar_channels)
        self.vector_channels = int(vector_channels)
        self.tensor_channels = int(tensor_channels)
        self.response_channels = int(response_channels)
        self.attention_layers = int(attention_layers)
        self.attention_scalar_channels = int(attention_scalar_channels)
        self.attention_heads = int(attention_heads)
        self.attention_value_mode = str(attention_value_mode)
        self.local_geometry = bool(local_geometry)
        self.geometry_radial_channels = int(geometry_radial_channels)
        if self.attention_value_mode not in {"shared_mean", "irrep_copy_split"}:
            raise ValueError(
                "attention_value_mode must be 'shared_mean' or 'irrep_copy_split'"
            )
        if self.attention_scalar_channels % self.attention_heads != 0:
            raise ValueError("attention_scalar_channels must be divisible by attention_heads")

        k = self.response_channels
        self.response_latent_irreps = o3.Irreps(
            f"{k}x0e + {k}x2e + {2*k}x1o + {k}x2o + {k}x3o"
        )
        self.hidden_irreps = o3.Irreps(f"{2*k}x1o + {k}x2o + {k}x3o")
        if self.attention_value_mode == "irrep_copy_split":
            for multiplicity, _ in self.hidden_irreps:
                if multiplicity % self.attention_heads != 0:
                    raise ValueError(
                        "irrep_copy_split requires every hidden irrep multiplicity "
                        "to be divisible by attention_heads"
                    )
        self.pre_irreps = o3.Irreps(
            f"{self.scalar_channels}x0e + {self.vector_channels}x1o"
        )
        gate_scalars = o3.Irreps(f"{self.scalar_channels}x0e")
        gate_irreps = o3.Irreps(
            f"{self.vector_channels + self.tensor_channels}x0e"
        )
        gate_tensors = o3.Irreps(
            f"{self.vector_channels}x1o + {self.tensor_channels}x2e"
        )
        gate_inputs = gate_scalars + gate_irreps + gate_tensors

        self.pre = o3.Linear(self.node_irreps, self.pre_irreps)
        self.environment_product = o3.FullyConnectedTensorProduct(
            self.pre_irreps, self.pre_irreps, gate_inputs
        )
        self.environment_gate = Gate(
            gate_scalars,
            [F.silu],
            gate_irreps,
            [torch.sigmoid],
            gate_tensors,
        )
        self.environment_to_hidden = o3.FullyConnectedTensorProduct(
            self.environment_gate.irreps_out,
            self.pre_irreps,
            self.hidden_irreps,
        )

        self.response_to_latent = o3.FullyConnectedTensorProduct(
            self.node_irreps,
            STRAIN_IRREPS,
            self.response_latent_irreps,
        )
        self.response_direct_to_hidden = o3.Linear(
            self.response_latent_irreps, self.hidden_irreps
        )
        self.response_environment_product = o3.FullyConnectedTensorProduct(
            self.response_latent_irreps,
            self.environment_gate.irreps_out,
            self.hidden_irreps,
        )
        if self.local_geometry:
            self.local_geometry_irreps = local_geometry_irreps(
                self.geometry_radial_channels
            )
            # The pre-projected node features condition the explicit neighbor
            # geometry on the central atom's chemical environment.
            self.local_geometry_product = o3.FullyConnectedTensorProduct(
                self.pre_irreps,
                self.local_geometry_irreps,
                self.hidden_irreps,
            )
            self.local_geometry_mix = nn.Parameter(torch.tensor(0.1))
        else:
            self.local_geometry_irreps = None
            self.local_geometry_product = None
            self.local_geometry_mix = None

        self.scalar_readout = o3.Linear(self.node_irreps, f"{self.scalar_channels}x0e")
        self.scalar_norm = nn.LayerNorm(self.scalar_channels)
        self.fusion_gates = nn.Sequential(
            nn.Linear(
                self.scalar_channels + self.response_latent_irreps.num_irreps,
                self.scalar_channels,
            ),
            nn.SiLU(),
            nn.Linear(self.scalar_channels, self.hidden_irreps.num_irreps),
            nn.Sigmoid(),
        )
        self.attention = nn.ModuleList(
            [
                CollectiveAttentionBlock(
                    self.hidden_irreps,
                    self.attention_scalar_channels,
                    self.scalar_channels,
                    self.attention_heads,
                    self.attention_value_mode,
                )
                for _ in range(self.attention_layers)
            ]
        )
        self.output_readout = o3.Linear(self.hidden_irreps, PIEZO_IRREPS)

        if atomic_response_feature_scales is None:
            atomic_response_feature_scales = torch.ones(self.node_irreps.dim)
        scales = torch.as_tensor(atomic_response_feature_scales).detach().clone().float()
        _validate_copy_scales(scales, self.node_irreps)
        if output_block_scales is None:
            output_block_scales = torch.ones(len(PIEZO_BLOCK_SLICES))
        output_scales = torch.as_tensor(output_block_scales).detach().clone().float()
        if output_scales.shape != (len(PIEZO_BLOCK_SLICES),):
            raise ValueError(
                f"output scales must have shape ({len(PIEZO_BLOCK_SLICES)},)"
            )
        if not torch.isfinite(output_scales).all() or torch.any(output_scales <= 0):
            raise ValueError("output block scales must be finite and strictly positive")
        self.register_buffer("atomic_response_feature_scales", scales)
        self.register_buffer("output_block_scales", output_scales)
        if self.local_geometry:
            if local_geometry_feature_scales is None:
                local_geometry_feature_scales = torch.ones(
                    self.local_geometry_irreps.dim
                )
            geometry_scales = torch.as_tensor(
                local_geometry_feature_scales
            ).detach().clone().float()
            _validate_copy_scales(geometry_scales, self.local_geometry_irreps)
            self.register_buffer("local_geometry_feature_scales", geometry_scales)
        else:
            self.register_buffer(
                "local_geometry_feature_scales", torch.ones(1), persistent=False
            )

    @property
    def model_config(self) -> dict[str, Any]:
        return {
            "head_architecture": "collective_attention",
            "architecture_version": self.architecture_version,
            "node_irreps": str(self.node_irreps),
            "response_latent_irreps": str(self.response_latent_irreps),
            "hidden_irreps": str(self.hidden_irreps),
            "scalar_channels": self.scalar_channels,
            "vector_channels": self.vector_channels,
            "tensor_channels": self.tensor_channels,
            "response_channels": self.response_channels,
            "attention_layers": self.attention_layers,
            "attention_scalar_channels": self.attention_scalar_channels,
            "attention_heads": self.attention_heads,
            "attention_value_mode": self.attention[0].attention_value_mode
            if self.attention
            else "shared_mean",
            "local_geometry": self.local_geometry,
            "geometry_radial_channels": self.geometry_radial_channels,
            "local_geometry_irreps": str(self.local_geometry_irreps)
            if self.local_geometry_irreps is not None
            else None,
            "target_kind": self.target_kind,
            "voigt_order": list(self.voigt_order),
            "strain_amplitude": self.strain_amplitude,
        }

    def response_features(self, atomic_response: torch.Tensor) -> torch.Tensor:
        if atomic_response.ndim != 3 or atomic_response.shape[1:] != (
            6,
            self.node_irreps.dim,
        ):
            raise ValueError(
                "atomic_response must have shape "
                f"(atoms, 6, {self.node_irreps.dim}), got {tuple(atomic_response.shape)}"
            )
        if not torch.isfinite(atomic_response).all():
            raise ValueError("atomic_response contains non-finite values")
        normalized = atomic_response / self.atomic_response_feature_scales.to(
            dtype=atomic_response.dtype
        )[None, None, :]
        strain_features = voigt_response_to_strain_irreps(normalized)
        atom_count, strain_dim, _ = strain_features.shape
        strain_basis = torch.eye(
            strain_dim, device=atomic_response.device, dtype=atomic_response.dtype
        )
        strain_basis = strain_basis[None].expand(atom_count, -1, -1).reshape(
            atom_count * strain_dim, strain_dim
        )
        response = self.response_to_latent(
            strain_features.reshape(atom_count * strain_dim, self.node_irreps.dim),
            strain_basis,
        )
        return response.reshape(atom_count, strain_dim, -1).sum(dim=1)

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        atomic_response: torch.Tensor | None = None,
        num_graphs: int | None = None,
        local_geometry: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if node_features.ndim != 2 or node_features.shape[-1] != self.node_irreps.dim:
            raise ValueError(
                f"node_features must have shape (atoms, {self.node_irreps.dim})"
            )
        if node_features.shape[0] == 0:
            raise ValueError("cannot evaluate a displacement head on an empty atom set")
        if batch.ndim != 1 or batch.shape[0] != node_features.shape[0]:
            raise ValueError("batch must contain one graph index per atom")
        if batch.dtype != torch.long or batch.device != node_features.device:
            raise ValueError("batch must be int64 and colocated with node_features")
        if torch.any(batch < 0):
            raise ValueError("batch graph indices must be nonnegative")
        if not torch.isfinite(node_features).all():
            raise ValueError("node_features contains non-finite values")
        if self.local_geometry:
            if local_geometry is None:
                raise ValueError("local-geometry head requires local geometry features")
            if local_geometry.ndim != 2 or local_geometry.shape[0] != node_features.shape[0]:
                raise ValueError("local_geometry must have one feature row per atom")
            if local_geometry.shape[1] != self.local_geometry_irreps.dim:
                raise ValueError(
                    f"local_geometry must have {self.local_geometry_irreps.dim} channels"
                )
            if local_geometry.device != node_features.device or local_geometry.dtype != node_features.dtype:
                raise ValueError("local_geometry and node_features must share device and dtype")
            if not torch.isfinite(local_geometry).all():
                raise ValueError("local_geometry contains non-finite values")
        elif local_geometry is not None:
            raise ValueError("local geometry was supplied to a head that does not use it")
        if atomic_response is None:
            raise ValueError("collective displacement head requires six-mode response features")
        if atomic_response.shape[0] != node_features.shape[0]:
            raise ValueError("node_features and atomic_response atom counts differ")
        if atomic_response.device != node_features.device:
            raise ValueError("node_features and atomic_response must share a device")
        if atomic_response.dtype != node_features.dtype:
            raise ValueError("node_features and atomic_response must share a dtype")
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1
        if num_graphs < 1 or torch.any(batch >= num_graphs):
            raise ValueError("num_graphs is inconsistent with batch graph indices")

        pre = self.pre(node_features)
        environment = self.environment_gate(self.environment_product(pre, pre))
        contextual = self.environment_to_hidden(environment, pre)
        response = self.response_features(atomic_response)
        coupled = self.response_environment_product(response, environment)
        direct = self.response_direct_to_hidden(response)
        scalar_state = self.scalar_norm(self.scalar_readout(node_features))
        response_invariants = channel_invariants(response, self.response_latent_irreps)
        gates = 2.0 * self.fusion_gates(
            torch.cat([scalar_state, response_invariants], dim=-1)
        )
        hidden = (
            contextual + direct + gate_irrep_copies(coupled, gates, self.hidden_irreps)
        ) / math.sqrt(3.0)
        if self.local_geometry:
            geometry = local_geometry / self.local_geometry_feature_scales.to(
                device=local_geometry.device, dtype=local_geometry.dtype
            )[None, :]
            geometry_hidden = self.local_geometry_product(pre, geometry)
            hidden = hidden + torch.tanh(self.local_geometry_mix) * geometry_hidden
        for attention in self.attention:
            hidden, scalar_state = attention(hidden, scalar_state, batch, num_graphs)

        raw = self.output_readout(hidden)
        raw = torch.cat(
            [
                raw[:, section] * self.output_block_scales[index].to(raw.dtype)
                for index, section in enumerate(PIEZO_BLOCK_SLICES)
            ],
            dim=-1,
        )
        projected, residual = AtomicInternalStrainHead.enforce_acoustic_sum_rule(
            raw, batch, num_graphs
        )
        return {
            "raw_irreps": raw,
            "lambda_irreps": projected,
            "lambda_voigt": irreps_to_voigt(projected),
            "response_latent_irreps": response,
            "asr_residual_irreps": residual,
        }


class ChannelLocalDisplacementHead(CollectiveAttentionDisplacementHead):
    """Same multi-channel local response fusion, with inter-atom attention off."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["attention_layers"] = 0
        super().__init__(*args, **kwargs)

    @property
    def model_config(self) -> dict[str, Any]:
        config = super().model_config
        config["head_architecture"] = "channel_local"
        return config


class StructureConditionedCoupledDisplacementHead(CollectiveAttentionDisplacementHead):
    """Add a zero-initialized, structure-conditioned equivariant correction.

    The frozen collective-attention head supplies the base atom-resolved Gamma.
    A graph-pooled ``0e + 2e`` context conditions an equivariant tensor product
    that corrects the base PIEZO irreps.  The product weights are initialized
    to zero, so the new model is exactly the source model at step zero.
    """

    def __init__(
        self,
        *args,
        coupling_scalar_channels: int = 16,
        coupling_tensor_channels: int = 8,
        coupling_scale: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if coupling_scalar_channels < 1 or coupling_tensor_channels < 1:
            raise ValueError("coupling channel counts must be positive")
        if not math.isfinite(float(coupling_scale)) or float(coupling_scale) <= 0:
            raise ValueError("coupling_scale must be finite and positive")
        self.coupling_scalar_channels = int(coupling_scalar_channels)
        self.coupling_tensor_channels = int(coupling_tensor_channels)
        self.coupling_scale = float(coupling_scale)
        self.coupling_context_irreps = o3.Irreps(
            f"{self.coupling_scalar_channels}x0e + "
            f"{self.coupling_tensor_channels}x2e"
        )
        self.coupling_context = o3.Linear(
            self.node_irreps, self.coupling_context_irreps
        )
        self.coupling_product = o3.FullyConnectedTensorProduct(
            PIEZO_IRREPS,
            self.coupling_context_irreps,
            PIEZO_IRREPS,
        )
        # Exact source-model initialization.  The context branch remains
        # trainable, but cannot change the output until the product learns.
        with torch.no_grad():
            self.coupling_product.weight.zero_()

    @property
    def model_config(self) -> dict[str, Any]:
        config = super().model_config
        config.update(
            {
                "head_architecture": "collective_attention_coupled",
                "coupling_scalar_channels": self.coupling_scalar_channels,
                "coupling_tensor_channels": self.coupling_tensor_channels,
                "coupling_scale": self.coupling_scale,
                "coupling_context_irreps": str(self.coupling_context_irreps),
            }
        )
        return config

    def _structure_context(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        context = self.coupling_context(node_features)
        pooled = context.new_zeros((num_graphs, context.shape[-1]))
        pooled.index_add_(0, batch, context)
        counts = torch.bincount(batch, minlength=num_graphs).to(
            device=batch.device, dtype=context.dtype
        )
        pooled = pooled / counts.clamp_min(1.0).sqrt()[:, None]
        return pooled.index_select(0, batch)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        node_features = args[0] if args else kwargs["node_features"]
        batch = args[1] if len(args) > 1 else kwargs["batch"]
        base = super().forward(*args, **kwargs)
        num_graphs = int(batch.max().item()) + 1
        context = self._structure_context(node_features, batch, num_graphs)
        correction = self.coupling_product(base["lambda_irreps"], context)
        raw = base["lambda_irreps"] + self.coupling_scale * correction
        projected, residual = AtomicInternalStrainHead.enforce_acoustic_sum_rule(
            raw, batch, num_graphs
        )
        base.update(
            {
                "coupling_context_irreps": context,
                "coupling_correction_irreps": correction,
                "raw_irreps": raw,
                "lambda_irreps": projected,
                "lambda_voigt": irreps_to_voigt(projected),
                "asr_residual_irreps": residual,
            }
        )
        return base


class StructureConditionedCoupledDisplacementHeadV2(CollectiveAttentionDisplacementHead):
    """Structure-conditioned residual with a genuine nonzero ``2e`` context.

    The source node representation contains ``1o`` channels but no ``2e``
    channels.  V2 therefore builds the anisotropic context explicitly with
    ``1o x 1o -> 2e`` and uses arithmetic mean pooling, which is invariant to
    repeating a periodic cell.
    """

    def __init__(
        self,
        *args,
        coupling_context_mode: str = "scalar_2e",
        coupling_scalar_channels: int = 16,
        coupling_vector_channels: int = 8,
        coupling_tensor_channels: int = 8,
        coupling_scale: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if coupling_context_mode not in {"scalar", "scalar_2e"}:
            raise ValueError("coupling_context_mode must be 'scalar' or 'scalar_2e'")
        if min(coupling_scalar_channels, coupling_vector_channels, coupling_tensor_channels) < 1:
            raise ValueError("coupling channel counts must be positive")
        if not math.isfinite(float(coupling_scale)) or float(coupling_scale) <= 0:
            raise ValueError("coupling_scale must be finite and positive")
        self.coupling_context_mode = coupling_context_mode
        self.coupling_scalar_channels = int(coupling_scalar_channels)
        self.coupling_vector_channels = int(coupling_vector_channels)
        self.coupling_tensor_channels = int(coupling_tensor_channels)
        self.coupling_scale = float(coupling_scale)
        self.coupling_scalar_irreps = o3.Irreps(f"{self.coupling_scalar_channels}x0e")
        self.coupling_scalar_context = o3.Linear(
            self.node_irreps, self.coupling_scalar_irreps
        )
        if coupling_context_mode == "scalar_2e":
            self.coupling_vector_irreps = o3.Irreps(f"{self.coupling_vector_channels}x1o")
            self.coupling_tensor_irreps = o3.Irreps(f"{self.coupling_tensor_channels}x2e")
            self.coupling_vector_context = o3.Linear(
                self.node_irreps, self.coupling_vector_irreps
            )
            self.coupling_tensor_product = o3.FullyConnectedTensorProduct(
                self.coupling_vector_irreps,
                self.coupling_vector_irreps,
                self.coupling_tensor_irreps,
            )
            context_irreps = self.coupling_scalar_irreps + self.coupling_tensor_irreps
        else:
            self.coupling_vector_irreps = None
            self.coupling_tensor_irreps = None
            self.coupling_vector_context = None
            self.coupling_tensor_product = None
            context_irreps = self.coupling_scalar_irreps
        self.coupling_context_irreps = o3.Irreps(context_irreps)
        self.coupling_product = o3.FullyConnectedTensorProduct(
            PIEZO_IRREPS, self.coupling_context_irreps, PIEZO_IRREPS
        )
        with torch.no_grad():
            self.coupling_product.weight.zero_()

    @property
    def model_config(self) -> dict[str, Any]:
        config = super().model_config
        config.update(
            {
                "head_architecture": "collective_attention_coupled_v2",
                "coupling_context_mode": self.coupling_context_mode,
                "coupling_scalar_channels": self.coupling_scalar_channels,
                "coupling_vector_channels": self.coupling_vector_channels,
                "coupling_tensor_channels": self.coupling_tensor_channels,
                "coupling_scale": self.coupling_scale,
                "coupling_context_irreps": str(self.coupling_context_irreps),
            }
        )
        return config

    @staticmethod
    def _mean_pool(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        pooled = values.new_zeros((num_graphs, values.shape[-1]))
        pooled.index_add_(0, batch, values)
        counts = torch.bincount(batch, minlength=num_graphs).to(
            device=batch.device, dtype=values.dtype
        )
        return pooled / counts.clamp_min(1.0)[:, None]

    def _structure_context(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        scalar = self._mean_pool(
            self.coupling_scalar_context(node_features), batch, num_graphs
        )
        if self.coupling_context_mode == "scalar":
            return scalar.index_select(0, batch)
        vectors = self.coupling_vector_context(node_features)
        tensors = self.coupling_tensor_product(vectors, vectors)
        tensor = self._mean_pool(tensors, batch, num_graphs)
        pooled = torch.cat([scalar, tensor], dim=-1)
        return pooled.index_select(0, batch)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        node_features = args[0] if args else kwargs["node_features"]
        batch = args[1] if len(args) > 1 else kwargs["batch"]
        base = super().forward(*args, **kwargs)
        num_graphs = int(batch.max().item()) + 1
        context = self._structure_context(node_features, batch, num_graphs)
        correction = self.coupling_product(base["lambda_irreps"], context)
        raw = base["lambda_irreps"] + self.coupling_scale * correction
        projected, residual = AtomicInternalStrainHead.enforce_acoustic_sum_rule(
            raw, batch, num_graphs
        )
        base.update(
            {
                "coupling_context_irreps": context,
                "coupling_correction_irreps": correction,
                "raw_irreps": raw,
                "lambda_irreps": projected,
                "lambda_voigt": irreps_to_voigt(projected),
                "asr_residual_irreps": residual,
            }
        )
        return base


class StructureConditionedAdditiveDisplacementHeadV3(
    StructureConditionedCoupledDisplacementHeadV2
):
    """Independent additive equivariant correction conditioned by graph context.

    V2 modulates the already predicted ``lambda`` through a tensor product.
    V3 instead maps the response latent itself together with the pooled graph
    context to an additive PIEZO correction.  The correction is zero
    initialized and the final atom-resolved tensor is projected with ASR.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.coupling_product = o3.FullyConnectedTensorProduct(
            self.response_latent_irreps,
            self.coupling_context_irreps,
            PIEZO_IRREPS,
        )
        with torch.no_grad():
            self.coupling_product.weight.zero_()

    @property
    def model_config(self) -> dict[str, Any]:
        config = super().model_config
        config.update(
            {
                "head_architecture": "collective_attention_additive_v3",
                "additive_correction": True,
                "coupling_product_input": str(self.response_latent_irreps),
            }
        )
        return config

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        node_features = args[0] if args else kwargs["node_features"]
        batch = args[1] if len(args) > 1 else kwargs["batch"]
        base = CollectiveAttentionDisplacementHead.forward(self, *args, **kwargs)
        num_graphs = int(batch.max().item()) + 1
        context = self._structure_context(node_features, batch, num_graphs)
        correction = self.coupling_product(
            base["response_latent_irreps"], context
        )
        raw = base["lambda_irreps"] + self.coupling_scale * correction
        projected, residual = AtomicInternalStrainHead.enforce_acoustic_sum_rule(
            raw, batch, num_graphs
        )
        base.update(
            {
                "coupling_context_irreps": context,
                "coupling_correction_irreps": correction,
                "raw_irreps": raw,
                "lambda_irreps": projected,
                "lambda_voigt": irreps_to_voigt(projected),
                "asr_residual_irreps": residual,
            }
        )
        return base


def build_displacement_head(
    model_config: dict[str, Any],
    atomic_response_feature_scales: torch.Tensor,
    output_block_scales: torch.Tensor,
    *,
    local_geometry_feature_scales: torch.Tensor | None = None,
    load_weights: bool = True,
    model_state_dict: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Construct a supported displacement head; strict loading prevents mismatch."""
    architecture = model_config.get("head_architecture", "historical")
    if architecture == "historical":
        head = AtomicInternalStrainHead(
            node_irreps=model_config["node_irreps"],
            scalar_channels=model_config["scalar_channels"],
            vector_channels=model_config["vector_channels"],
            tensor_channels=model_config["tensor_channels"],
            atomic_response=model_config["atomic_response"],
            atomic_response_feature_scales=atomic_response_feature_scales,
            atomic_response_mix_init=model_config.get("atomic_response_mix_init", 0.1),
            output_block_scales=output_block_scales,
        )
    elif architecture in {
        "channel_local",
        "collective_attention",
        "collective_attention_coupled",
        "collective_attention_coupled_v2",
        "collective_attention_additive_v3",
    }:
        version = int(model_config.get("architecture_version", ARCHITECTURE_VERSION))
        if version != ARCHITECTURE_VERSION:
            raise ValueError(f"unsupported displacement-head architecture version {version}")
        if architecture == "channel_local":
            cls = ChannelLocalDisplacementHead
        elif architecture == "collective_attention":
            cls = CollectiveAttentionDisplacementHead
        elif architecture == "collective_attention_coupled_v2":
            cls = StructureConditionedCoupledDisplacementHeadV2
        elif architecture == "collective_attention_additive_v3":
            cls = StructureConditionedAdditiveDisplacementHeadV3
        else:
            cls = StructureConditionedCoupledDisplacementHead
        if architecture == "channel_local" and int(model_config.get("attention_layers", 0)) != 0:
            raise ValueError("channel_local checkpoints must have attention_layers=0")
        extra = {}
        if architecture == "collective_attention_coupled":
            extra = {
                "coupling_scalar_channels": int(
                    model_config.get("coupling_scalar_channels", 16)
                ),
                "coupling_tensor_channels": int(
                    model_config.get("coupling_tensor_channels", 8)
                ),
                "coupling_scale": float(model_config.get("coupling_scale", 0.1)),
            }
        elif architecture == "collective_attention_coupled_v2":
            extra = {
                "coupling_context_mode": str(
                    model_config.get("coupling_context_mode", "scalar_2e")
                ),
                "coupling_scalar_channels": int(
                    model_config.get("coupling_scalar_channels", 16)
                ),
                "coupling_vector_channels": int(
                    model_config.get("coupling_vector_channels", 8)
                ),
                "coupling_tensor_channels": int(
                    model_config.get("coupling_tensor_channels", 8)
                ),
                "coupling_scale": float(model_config.get("coupling_scale", 0.1)),
            }
        elif architecture == "collective_attention_additive_v3":
            extra = {
                "coupling_context_mode": str(
                    model_config.get("coupling_context_mode", "scalar_2e")
                ),
                "coupling_scalar_channels": int(
                    model_config.get("coupling_scalar_channels", 16)
                ),
                "coupling_vector_channels": int(
                    model_config.get("coupling_vector_channels", 8)
                ),
                "coupling_tensor_channels": int(
                    model_config.get("coupling_tensor_channels", 8)
                ),
                "coupling_scale": float(model_config.get("coupling_scale", 0.1)),
            }
        head = cls(
            node_irreps=model_config["node_irreps"],
            scalar_channels=int(model_config.get("scalar_channels", DEFAULT_SCALAR_CHANNELS)),
            vector_channels=int(model_config.get("vector_channels", DEFAULT_VECTOR_CHANNELS)),
            tensor_channels=int(model_config.get("tensor_channels", DEFAULT_TENSOR_CHANNELS)),
            response_channels=int(model_config.get("response_channels", DEFAULT_RESPONSE_CHANNELS)),
            attention_layers=int(model_config.get("attention_layers", DEFAULT_ATTENTION_LAYERS)),
            attention_scalar_channels=int(
                model_config.get("attention_scalar_channels", DEFAULT_ATTENTION_SCALAR_CHANNELS)
            ),
            attention_heads=int(
                model_config.get("attention_heads", DEFAULT_ATTENTION_HEADS)
            ),
            attention_value_mode=str(
                model_config.get("attention_value_mode", "shared_mean")
            ),
            atomic_response_feature_scales=atomic_response_feature_scales,
            output_block_scales=output_block_scales,
            local_geometry=bool(model_config.get("local_geometry", False)),
            geometry_radial_channels=int(
                model_config.get(
                    "geometry_radial_channels", DEFAULT_GEOMETRY_RADIAL_CHANNELS
                )
            ),
            local_geometry_feature_scales=local_geometry_feature_scales
            if model_config.get("local_geometry", False)
            else None,
            **extra,
        )
        for key in ("response_latent_irreps", "hidden_irreps"):
            expected = str(getattr(head, key))
            supplied = model_config.get(key)
            if supplied is not None and str(o3.Irreps(supplied)) != expected:
                raise ValueError(f"checkpoint {key}={supplied!r} conflicts with constructed {expected!r}")
    else:
        raise ValueError(f"unknown displacement-head architecture: {architecture}")

    if load_weights:
        if model_state_dict is None:
            raise ValueError("model_state_dict is required when load_weights=True")
        head.load_state_dict(model_state_dict, strict=True)
    elif model_state_dict is not None:
        raise ValueError("refusing to ignore model_state_dict when load_weights=False")
    return head


def make_model_config(
    architecture: str,
    node_irreps: str | o3.Irreps,
    scalar_channels: int = DEFAULT_SCALAR_CHANNELS,
    vector_channels: int = DEFAULT_VECTOR_CHANNELS,
    tensor_channels: int = DEFAULT_TENSOR_CHANNELS,
    response_channels: int = DEFAULT_RESPONSE_CHANNELS,
    attention_layers: int = DEFAULT_ATTENTION_LAYERS,
    attention_scalar_channels: int = DEFAULT_ATTENTION_SCALAR_CHANNELS,
    attention_heads: int = DEFAULT_ATTENTION_HEADS,
    attention_value_mode: str = "shared_mean",
    local_geometry: bool = False,
    geometry_radial_channels: int = DEFAULT_GEOMETRY_RADIAL_CHANNELS,
) -> dict[str, Any]:
    if architecture == "historical":
        return {
            "head_architecture": "historical",
            "architecture_version": 0,
            "node_irreps": str(o3.Irreps(node_irreps)),
            "scalar_channels": int(scalar_channels),
            "vector_channels": int(vector_channels),
            "tensor_channels": int(tensor_channels),
            "atomic_response": True,
            "atomic_response_mix_init": 0.1,
            "target_kind": "displacement",
            "voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
            "strain_amplitude": 0.005,
        }
    if architecture not in {
        "channel_local",
        "collective_attention",
        "collective_attention_coupled",
        "collective_attention_coupled_v2",
        "collective_attention_additive_v3",
    }:
        raise ValueError(f"unknown architecture: {architecture}")
    k = int(response_channels)
    if k < 1:
        raise ValueError("response_channels must be positive")
    layers = 0 if architecture == "channel_local" else int(attention_layers)
    if layers < 0:
        raise ValueError("attention_layers must be nonnegative")
    return {
        "head_architecture": architecture,
        "architecture_version": ARCHITECTURE_VERSION,
        "node_irreps": str(o3.Irreps(node_irreps)),
        "response_latent_irreps": str(
            o3.Irreps(f"{k}x0e + {k}x2e + {2*k}x1o + {k}x2o + {k}x3o")
        ),
        "hidden_irreps": str(o3.Irreps(f"{2*k}x1o + {k}x2o + {k}x3o")),
        "scalar_channels": int(scalar_channels),
        "vector_channels": int(vector_channels),
        "tensor_channels": int(tensor_channels),
        "response_channels": k,
        "attention_layers": layers,
        "attention_scalar_channels": int(attention_scalar_channels),
        "attention_heads": int(attention_heads),
        "attention_value_mode": str(attention_value_mode),
        "local_geometry": bool(local_geometry),
        "geometry_radial_channels": int(geometry_radial_channels),
        "local_geometry_irreps": str(local_geometry_irreps(geometry_radial_channels))
        if local_geometry
        else None,
        "target_kind": "displacement",
        "voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
        "strain_amplitude": 0.005,
    }
