#!/usr/bin/env python3
"""Differentiable equivariant dipole model for physical piezo responses."""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate

from uiu_equivariant_piezo_model import DEFAULT_NODE_IRREPS


CONV = 16.0217663


def _voigt_deformation_basis() -> torch.Tensor:
    """Return dF/deta for xx, yy, zz, yz, xz, xy engineering strains."""
    basis = torch.zeros((6, 3, 3), dtype=torch.get_default_dtype())
    basis[0, 0, 0] = 1.0
    basis[1, 1, 1] = 1.0
    basis[2, 2, 2] = 1.0
    basis[3, 1, 2] = basis[3, 2, 1] = 0.5
    basis[4, 0, 2] = basis[4, 2, 0] = 0.5
    basis[5, 0, 1] = basis[5, 1, 0] = 0.5
    return basis


def graph_sum(
    values: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    result = values.new_zeros((num_graphs,) + values.shape[1:])
    result.index_add_(0, batch, values)
    return result


def graph_mean(
    values: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    summed = graph_sum(values, batch, num_graphs)
    counts = graph_sum(values.new_ones((len(values), 1)), batch, num_graphs)
    view_shape = (num_graphs,) + (1,) * (values.ndim - 1)
    return summed / counts.reshape(view_shape).clamp_min(1.0)


class DifferentiablePolarizationHead(nn.Module):
    """Map frozen MACELES atom features to one smooth graph dipole moment.

    The graph dipole contains a neutral charge-position contribution and a
    local equivariant dipole contribution. BEC and clamped-ion piezo tensors
    are not independent readouts; they are derivatives of this dipole.
    """

    def __init__(
        self,
        node_irreps: o3.Irreps = DEFAULT_NODE_IRREPS,
        scalar_channels: int = 32,
        vector_channels: int = 16,
        tensor_channels: int = 16,
        charge_scale: float = 0.1,
        dipole_scale: float = 0.1,
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
        gate_input = gate_scalars + gate_gates + gate_tensors

        self.pre = o3.Linear(self.node_irreps, self.pre_irreps)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.pre_irreps,
            self.pre_irreps,
            gate_input,
        )
        self.gate = Gate(
            gate_scalars,
            [torch.nn.functional.silu],
            gate_gates,
            [torch.sigmoid],
            gate_tensors,
        )
        self.charge_readout = o3.Linear(self.gate.irreps_out, o3.Irreps("1x0e"))
        self.dipole_readout = o3.FullyConnectedTensorProduct(
            self.gate.irreps_out,
            self.pre_irreps,
            o3.Irreps("1x1o"),
        )
        self.log_charge_scale = nn.Parameter(
            torch.tensor(float(charge_scale)).abs().clamp_min(1.0e-4).log()
        )
        self.log_dipole_scale = nn.Parameter(
            torch.tensor(float(dipole_scale)).abs().clamp_min(1.0e-4).log()
        )
        self.register_buffer("voigt_basis", _voigt_deformation_basis())

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> dict[str, torch.Tensor]:
        pre = self.pre(node_features)
        hidden = self.gate(self.tensor_product(pre, pre))

        charge = self.charge_readout(hidden).squeeze(-1)
        charge = charge * self.log_charge_scale.clamp(-5.0, 5.0).exp()
        charge = charge - graph_mean(charge[:, None], batch, num_graphs)[
            batch, 0
        ]

        local_dipole = self.dipole_readout(hidden, pre)
        local_dipole = (
            local_dipole * self.log_dipole_scale.clamp(-5.0, 5.0).exp()
        )
        centers = graph_mean(positions, batch, num_graphs)
        centered_positions = positions - centers[batch]
        charge_dipole = graph_sum(
            charge[:, None] * centered_positions,
            batch,
            num_graphs,
        )
        permanent_dipole = graph_sum(local_dipole, batch, num_graphs)
        dipole = charge_dipole + permanent_dipole
        return {
            "dipole_eA": dipole,
            "atomic_charge": charge,
            "atomic_dipole_eA": local_dipole,
            "charge_dipole_eA": charge_dipole,
            "permanent_dipole_eA": permanent_dipole,
        }


def enforce_bec_asr(
    bec: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = graph_sum(bec, batch, num_graphs)
    projected = bec - graph_mean(bec, batch, num_graphs)[batch]
    return projected, residual


def differentiate_polarization_responses(
    dipole_eA: torch.Tensor,
    base_positions: torch.Tensor,
    displacement: torch.Tensor,
    batch: torch.Tensor,
    volumes_A3: torch.Tensor,
    voigt_basis: torch.Tensor,
    create_graph: bool,
) -> dict[str, torch.Tensor]:
    """Differentiate one dipole function into BEC and proper static piezo.

    MACE's ``displacement`` variable deforms both positions and cell with a
    symmetric infinitesimal strain. The proper-piezo correction subtracts the
    affine transformation of the Cartesian dipole, D_J @ M, from dM/deta_J.
    """
    position_rows = []
    strain_rows = []
    for component in range(3):
        position_grad, strain_grad = torch.autograd.grad(
            dipole_eA[:, component].sum(),
            (base_positions, displacement),
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=False,
        )
        position_rows.append(position_grad)
        strain_rows.append(strain_grad)

    bec_raw = torch.stack(position_rows, dim=1)
    num_graphs = dipole_eA.shape[0]
    bec, bec_asr_before = enforce_bec_asr(bec_raw, batch, num_graphs)

    # [graph, polarization_i, strain_row, strain_col]
    dipole_strain_cartesian = torch.stack(strain_rows, dim=1)
    dipole_strain_voigt = torch.einsum(
        "bikl,jkl->bij", dipole_strain_cartesian, voigt_basis
    )
    affine_dipole = torch.einsum("jik,bk->bij", voigt_basis, dipole_eA)
    static_piezo = (
        CONV
        / volumes_A3[:, None, None].clamp_min(1.0e-12)
        * (dipole_strain_voigt - affine_dipole)
    )
    polarization = (
        CONV / volumes_A3[:, None].clamp_min(1.0e-12) * dipole_eA
    )
    return {
        "bec": bec,
        "bec_raw": bec_raw,
        "bec_asr_before": bec_asr_before,
        "dipole_strain_cartesian": dipole_strain_cartesian,
        "dipole_strain_voigt": dipole_strain_voigt,
        "static_piezo": static_piezo,
        "polarization_Cm2": polarization,
    }


def ionic_piezo_from_bec_displacement(
    bec: torch.Tensor,
    displacement_dstrain: torch.Tensor,
    batch: torch.Tensor,
    volumes_A3: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    atom_contributions = torch.einsum(
        "aik,akj->aij", bec, displacement_dstrain
    )
    graph_contributions = graph_sum(atom_contributions, batch, num_graphs)
    return (
        CONV
        / volumes_A3[:, None, None].clamp_min(1.0e-12)
        * graph_contributions
    )
