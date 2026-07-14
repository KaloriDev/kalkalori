# KalKalori - Heat Exchanger Open Engine
# Copyright (C) 2025  KalKalori Project Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# NOTE ON UNITS
# -------------
# All calculations in the KalKalori core engine are performed using
# plain Python floats in consistent SI base units:
#   T [K], p [Pa], rho [kg/m3], mu [Pa*s], k [W/(m*K)],
#   cp [J/(kg*K)], m_dot [kg/s], alfa [W/(m2*K)].

"""
v0.5.1 - Rating (overdesign / surface margin).

Design-practice terminology
----------------------------
Given known geometry and a *closed* heat balance (duty, both temperature
programs, both flow rates -- via ``core.models.heat_balance``), Rating reports
how much surface margin / overdesign the geometry provides. This is the
inverse question to Simulation (``core.models.simulation``), which starts
from known inlets and computes achievable outlets.

Over-surface definition
------------------------
``A_o`` is the actual outer heat-transfer area from geometry (``HXResult.A_o``).
``A_required`` is the area that would be exactly required for the closed
balance's duty at the working-condition ``U``:

    overdesign_factor = A_o / A_required - 1   (positive = margin, negative = shortfall)

``ua_margin = UA_actual / UA_required - 1`` is reported alongside "for free".

Assumption: ``U`` is evaluated once at the closed balance's mean bulk state
(known, since the balance is closed) and held constant in the area-margin
arithmetic -- this is standard over-surface practice. A property-iteration
refinement of ``U`` itself is out of scope for v0.5.1.

Algorithm
---------
1. Balance is already closed (``ClosedBalance``): duty ``Q``, both ``T_in``,
   both ``T_out``, ``C_min``, ``C_max``.
2. Evaluate ``U_mean`` at the closed balance's mean bulk properties, actual
   geometry and actual flows via one ``BareTubeHeatExchanger.solve()`` pass
   (same shape as the pass used inside Simulation's ``_evaluate``).
3. ``Q_max = C_min*(T_hot_in - T_cold_in)``, ``eps_req = Q/Q_max``,
   ``NTU_req = ntu_from_effectiveness(eps_req, ...)``, ``UA_req = NTU_req*C_min``,
   ``A_required = UA_req/U_mean``.
4. ``overdesign_factor = A_o/A_required - 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.heat_transfer.streams import SensibleHeatStream
from core.heat_transfer.ntu import ntu_from_effectiveness
from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props
from core.common.warnings import ModelWarning

from core.models.heat_balance import ClosedBalance

if TYPE_CHECKING:
    from core.models.bare_tube import BareTubeHeatExchanger
    from core.models.simulation import HXSimulationResult


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HXRatingResult:
    """Result of a heat-exchanger rating (overdesign / surface margin)."""

    overdesign_factor: float   # [-] A_o/A_required - 1
    ua_margin: float           # [-] UA_actual/UA_required - 1

    A_o: float                 # [m^2] actual outer area from geometry
    A_required: float          # [m^2] area required for the closed balance's duty
    UA_required: float         # [W/K]
    UA_actual: float           # [W/K] at the closed balance's working conditions
    U_mean: float              # [W/(m2*K)] referenced to A_o

    Q_required: float          # [W] duty of the closed balance
    Q_achievable: float | None  # [W] from an optional comparison simulate() run

    closed_balance: ClosedBalance
    simulation: "HXSimulationResult | None"

    warnings: list[ModelWarning] | None = None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_rating(
    hx: "BareTubeHeatExchanger",
    closed_balance: ClosedBalance,
    *,
    flow_arrangement: str | None = None,
    K_inlet: float = 0.5,
    K_outlet: float = 1.0,
    K_turn: float = 1.5,
    euler_provider: str = "zukauskas",
    include_simulation: bool = False,
) -> HXRatingResult:
    """Rate a bare-tube exchanger against a closed heat balance.

    Backing implementation of ``BareTubeHeatExchanger.rate``. See the module
    docstring for the algorithm.
    """
    if flow_arrangement is None:
        flow_arrangement = hx.bundle.flow_arrangement

    inside = closed_balance.inside
    outside = closed_balance.outside
    hot_is_inside = closed_balance.hot_is_inside

    T_mean_inside = 0.5 * (inside.T_in + inside.T_out)
    T_mean_outside = 0.5 * (outside.T_in + outside.T_out)
    props_in = inside.provider.at(T=T_mean_inside, p=inside.p)
    props_out = outside.provider.at(T=T_mean_outside, p=outside.p)

    if hot_is_inside:
        hot_stream = SensibleHeatStream(C=inside.C, T_in=inside.T_in)
        cold_stream = SensibleHeatStream(C=outside.C, T_in=outside.T_in)
    else:
        hot_stream = SensibleHeatStream(C=outside.C, T_in=outside.T_in)
        cold_stream = SensibleHeatStream(C=inside.C, T_in=inside.T_in)

    solve_result = hx.solve(
        hot_stream=hot_stream,
        cold_stream=cold_stream,
        m_dot_tube_side=inside.m_dot,
        tube_side_props=to_internal_fluid_props(props_in),
        m_dot_outside=outside.m_dot,
        outside_props=to_outside_fluid_props(props_out),
        K_inlet=K_inlet,
        K_outlet=K_outlet,
        K_turn=K_turn,
        flow_arrangement=flow_arrangement,
        euler_provider=euler_provider,
    )

    A_o = solve_result.A_o
    UA_actual = solve_result.UA
    U_mean = UA_actual / A_o if A_o > 0.0 else math.nan

    C_hot = hot_stream.capacity_rate()
    C_cold = cold_stream.capacity_rate()
    C_min = min(C_hot, C_cold)

    eps_req = closed_balance.Q / closed_balance.Q_max
    NTU_req = ntu_from_effectiveness(
        eps_req,
        C_hot,
        C_cold,
        flow_arrangement=flow_arrangement,
        C_inside=inside.C,
        C_outside=outside.C,
    )
    UA_req = NTU_req * C_min
    A_req = UA_req / U_mean

    overdesign_factor = A_o / A_req - 1.0
    ua_margin = UA_actual / UA_req - 1.0

    warnings_list: list[ModelWarning] = list(closed_balance.warnings or []) + list(solve_result.warnings or [])

    simulation_result = None
    Q_achievable = None
    if include_simulation:
        from core.models.simulation import run_simulation

        simulation_result = run_simulation(
            hx,
            inside.to_hx_side_input(),
            outside.to_hx_side_input(),
            flow_arrangement=flow_arrangement,
            K_inlet=K_inlet,
            K_outlet=K_outlet,
            K_turn=K_turn,
            euler_provider=euler_provider,
        )
        Q_achievable = simulation_result.q

    return HXRatingResult(
        overdesign_factor=overdesign_factor,
        ua_margin=ua_margin,
        A_o=A_o,
        A_required=A_req,
        UA_required=UA_req,
        UA_actual=UA_actual,
        U_mean=U_mean,
        Q_required=closed_balance.Q,
        Q_achievable=Q_achievable,
        closed_balance=closed_balance,
        simulation=simulation_result,
        warnings=warnings_list if warnings_list else None,
    )
