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
v0.5.3 - Rating (overdesign / surface margin).

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

``U``/``UA_actual`` source (v0.5.3)
-------------------------------------
``U_mean``/``UA_actual`` (and the new ``alfa_i``/``alfa_o`` fields) are now the
converged, wall-temperature- and finite-length-corrected values from
``core.heat_transfer.thermal_iteration.solve_iterative_thermal_state`` --
evaluated at the closed balance's actual inlet temperatures and flows, self-
consistently with its own epsilon-NTU-implied mean bulk/wall state (see that
module's docstring for the heat-duty convention and algorithm). This
supersedes the v0.5.1/v0.5.2 behavior, which evaluated ``U`` from a single
uncorrected ``BareTubeHeatExchanger.solve()`` pass at the closed balance's
*specified* mean bulk temperature and never applied the wall-temperature or
finite-length corrections -- i.e. the iterative thermal state was computed
elsewhere (``BareTubeHeatExchanger.solve_thermal_state``) but never consumed
by Rating. ``run_rating`` also calls ``BareTubeHeatExchanger.solve()`` once to
obtain the areas and the final provider-based three-state tube-bundle
hydraulic result; its thermal coefficients are deliberately not propagated to
``HXRatingResult`` because the thermal-state values are authoritative.

Algorithm
---------
1. Balance is already closed (``ClosedBalance``): duty ``Q``, both ``T_in``,
   both ``T_out``, ``C_min``, ``C_max``.
2. Evaluate the converged iterative thermal state (wall/length-corrected
   ``alfa_i``, ``alfa_o``, ``U_mean``, ``UA_actual``) via
   ``solve_iterative_thermal_state`` at the closed balance's inlet
   temperatures, flows and providers.
3. ``Q_max = C_min*(T_hot_in - T_cold_in)``, ``eps_req = Q/Q_max``,
   ``NTU_req = ntu_from_effectiveness(eps_req, ...)``, ``UA_req = NTU_req*C_min``,
   ``A_required = UA_req/U_mean``.
4. ``overdesign_factor = A_o/A_required - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from core.heat_transfer.streams import SensibleHeatStream
from core.heat_transfer.ntu import ntu_from_effectiveness
from core.heat_transfer.thermal_iteration import (
    IterativeThermalState,
    WallTemperatureEnvelope,
    estimate_wall_temperature_envelope,
    solve_iterative_thermal_state,
)
from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props
from core.common.warnings import ModelWarning

from core.models.heat_balance import ClosedBalance
from core.phase_change.types import PhaseChangeResult, WaterSteamPhaseChangeResult

if TYPE_CHECKING:
    from core.models.bare_tube import BareTubeHeatExchanger, HXResult
    from core.models.simulation import HXSimulationResult


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HXRatingResult:
    """Result of a heat-exchanger rating (overdesign / surface margin).

    ``inside_properties_inlet/midpoint/outlet`` and the corresponding
    ``outside_*`` accessors expose the existing hydraulic point states from
    ``final_result``. They are distinct from the representative 0D properties
    on ``thermal_state``. For active wet-gas condensation, the applicable
    side's outlet point uses final ``W_out`` composition and gas mass flow.
    """

    overdesign_factor: float   # [-] A_o/A_required - 1
    ua_margin: float           # [-] UA_actual/UA_required - 1

    A_o: float                 # [m^2] actual outer area from geometry
    A_required: float          # [m^2] area required for the closed balance's duty
    UA_required: float         # [W/K]
    UA_actual: float           # [W/K] at the closed balance's working conditions (== thermal_state.UA)
    U_mean: float              # [W/(m2*K)] referenced to A_o (== thermal_state.U)
    EMTD: float                # [K] effective mean temperature difference, Q_required/UA_required

    # Multi-zone steam reports the resistance-consistent equivalent HTC here;
    # all other models retain their established corrected-HTC semantics.
    alfa_i: float              # [W/(m2*K)] wall/length-corrected (== thermal_state.alfa_i)
    alfa_o: float              # [W/(m2*K)] wall-corrected (== thermal_state.alfa_o)

    Q_required: float          # [W] duty of the closed balance
    Q_achievable: float | None  # [W] from an optional comparison simulate() run

    closed_balance: ClosedBalance
    final_result: "HXResult"
    simulation: "HXSimulationResult | None"

    # Converged iterative thermal state (wall temperatures, per-side
    # correction/Nu diagnostics, convergence/iterations/residual). This is
    # the SAME state that UA_actual/U_mean/alfa_i/alfa_o above are read from
    # -- it is never overwritten by the separate (uncorrected)
    # ``BareTubeHeatExchanger.solve()`` pass used only for area/regime/
    # hydraulic diagnostics elsewhere in this module.
    thermal_state: IterativeThermalState
    wall_temperature_envelope: WallTemperatureEnvelope

    warnings: list[ModelWarning] | None = None

    # Phase-change results; see HXSimulationResult for the field
    # semantics -- same meaning here. ``None`` only if this HXRatingResult
    # was constructed directly by ``run_rating`` (the sensible-only driver)
    # rather than through ``BareTubeHeatExchanger.rate``.
    inside_phase_change: "PhaseChangeResult | WaterSteamPhaseChangeResult | None" = None
    outside_phase_change: "PhaseChangeResult | None" = None

    @property
    def phase_change_active(self) -> bool:
        return any(
            result is not None and result.active
            for result in (self.inside_phase_change, self.outside_phase_change)
        )

    @property
    def inside_condensate_mass_flow(self) -> float:
        return self.inside_phase_change.m_dot_condensate if self.inside_phase_change is not None else 0.0

    @property
    def inside_water_ratio_in(self) -> float | None:
        return None if self.inside_phase_change is None else self.inside_phase_change.W_in

    @property
    def inside_water_ratio_out(self) -> float | None:
        return None if self.inside_phase_change is None else self.inside_phase_change.W_out

    @property
    def inside_sensible_duty(self) -> float:
        return self.inside_phase_change.Q_sensible if self.inside_phase_change is not None else self.Q_required

    @property
    def inside_latent_duty(self) -> float:
        return self.inside_phase_change.Q_latent if self.inside_phase_change is not None else 0.0

    @property
    def outside_condensate_mass_flow(self) -> float:
        return self.outside_phase_change.m_dot_condensate if self.outside_phase_change is not None else 0.0

    @property
    def outside_water_ratio_in(self) -> float | None:
        return None if self.outside_phase_change is None else self.outside_phase_change.W_in

    @property
    def outside_water_ratio_out(self) -> float | None:
        return None if self.outside_phase_change is None else self.outside_phase_change.W_out

    @property
    def outside_Q_sensible(self) -> float:
        return self.outside_phase_change.Q_sensible if self.outside_phase_change is not None else self.Q_required

    @property
    def outside_Q_latent(self) -> float:
        return self.outside_phase_change.Q_latent if self.outside_phase_change is not None else 0.0

    @property
    def inside_wall_temperature_mean(self) -> float:
        return self.thermal_state.inside_wall_temperature

    @property
    def outside_wall_temperature_mean(self) -> float:
        return self.thermal_state.outside_wall_temperature

    @property
    def inside_wall_temperature_min_estimate(self) -> float:
        return self.wall_temperature_envelope.inside_min

    @property
    def inside_wall_temperature_max_estimate(self) -> float:
        return self.wall_temperature_envelope.inside_max

    @property
    def outside_wall_temperature_min_estimate(self) -> float:
        return self.wall_temperature_envelope.outside_min

    @property
    def outside_wall_temperature_max_estimate(self) -> float:
        return self.wall_temperature_envelope.outside_max

    @property
    def tube_side_hydraulic(self):
        """Nested straight tube-bundle hydraulic result."""
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return None
        return self.final_result.tube_side_hydraulic

    @property
    def inside_properties_inlet(self):
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult):
            return self.inside_phase_change.state_in
        return self.final_result.inside_properties_inlet

    @property
    def inside_properties_midpoint(self):
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult):
            return self.inside_phase_change.state_midpoint
        return self.final_result.inside_properties_midpoint

    @property
    def inside_properties_outlet(self):
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult):
            return self.inside_phase_change.state_out
        return self.final_result.inside_properties_outlet

    @property
    def outside_properties_inlet(self):
        return self.final_result.outside_properties_inlet

    @property
    def outside_properties_midpoint(self):
        return self.final_result.outside_properties_midpoint

    @property
    def outside_properties_outlet(self):
        return self.final_result.outside_properties_outlet

    @property
    def inside_dp_friction(self) -> float:
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return math.nan
        return self.final_result.inside_dp_friction

    @property
    def inside_dp_acceleration(self) -> float:
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return math.nan
        return self.final_result.inside_dp_acceleration

    @property
    def inside_dp_tube_bundle(self) -> float:
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return math.nan
        return self.final_result.inside_dp_tube_bundle

    @property
    def inside_dp_total(self) -> float:
        return self.inside_dp_tube_bundle

    @property
    def outside_tube_bank_hydraulic(self):
        """Nested three-state outside tube-bank hydraulic result."""
        return self.final_result.outside_tube_bank_hydraulic

    @property
    def outside_dp_drag(self) -> float:
        return self.final_result.outside_dp_drag

    @property
    def outside_dp_acceleration(self) -> float:
        return self.final_result.outside_dp_acceleration

    @property
    def outside_dp_total(self) -> float:
        return self.final_result.outside_dp_total

    @property
    def outside_dp(self) -> float:
        return self.outside_dp_total

    @property
    def outside_pressure_drop(self) -> float:
        return self.outside_dp_total

    # -- Pressure-drop flow-path aggregation (v0.5.6) ----------------------

    @property
    def tube_side_pressure_drop(self):
        """Complete tube-side pressure-drop result (core + local + total)."""
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return None
        return self.final_result.tube_side_pressure_drop

    @property
    def outside_side_pressure_drop(self):
        """Complete outside-side pressure-drop result (core + local + total)."""
        return self.final_result.outside_side_pressure_drop

    @property
    def inside_dp_local(self) -> float:
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return math.nan
        return self.final_result.inside_dp_local

    @property
    def outside_dp_local(self) -> float:
        return self.final_result.outside_dp_local

    # -- Tube-sheet entrance/exit pressure drop (v0.5.6) --------------------

    @property
    def inside_dp_straight_tube_friction(self) -> float:
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return math.nan
        return self.final_result.inside_dp_straight_tube_friction

    @property
    def inside_dp_straight_tube_acceleration(self) -> float:
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return math.nan
        return self.final_result.inside_dp_straight_tube_acceleration

    @property
    def inside_dp_straight_tubes(self) -> float:
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return math.nan
        return self.final_result.inside_dp_straight_tubes

    @property
    def inside_dp_tube_entrances(self) -> float:
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return math.nan
        return self.final_result.inside_dp_tube_entrances

    @property
    def inside_dp_tube_exits(self) -> float:
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return math.nan
        return self.final_result.inside_dp_tube_exits

    @property
    def tube_path_type(self):
        return self.final_result.tube_path_type

    @property
    def entrance_count(self) -> int:
        return self.final_result.entrance_count

    @property
    def exit_count(self) -> int:
        return self.final_result.exit_count

    @property
    def pass_boundary_method(self) -> str:
        return self.final_result.pass_boundary_method

    @property
    def pass_boundary_states(self):
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return ()
        return self.final_result.pass_boundary_states

    @property
    def entrance_results(self):
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return ()
        return self.final_result.entrance_results

    @property
    def exit_results(self):
        if isinstance(self.inside_phase_change, WaterSteamPhaseChangeResult) and self.inside_phase_change.active:
            return ()
        return self.final_result.exit_results


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
    max_iterations: int = 25,
    wall_temperature_tolerance_K: float = 0.05,
    relative_alfa_tolerance: float = 1e-3,
    relaxation_factor: float = 0.5,
) -> HXRatingResult:
    """Rate a bare-tube exchanger against a closed heat balance.

    Backing implementation of ``BareTubeHeatExchanger.rate``. See the module
    docstring for the algorithm and the ``U``/``UA_actual`` source note.

    ``max_iterations``/``wall_temperature_tolerance_K``/
    ``relative_alfa_tolerance``/``relaxation_factor`` are forwarded to
    ``solve_iterative_thermal_state`` (same defaults and meaning as
    ``BareTubeHeatExchanger.solve_thermal_state``).
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

    # Legacy single-pass solve: kept for the area/regime/hydraulic
    # diagnostics it produces (A_o, applicability warnings), but its own
    # (uncorrected) alfa_i/alfa_o/UA are NOT propagated below -- the
    # converged iterative thermal state computed next is authoritative for
    # those. See the module docstring's "U/UA_actual source" note.
    solve_result = hx.solve(
        hot_stream=hot_stream,
        cold_stream=cold_stream,
        m_dot_tube_side=inside.m_dot,
        tube_side_props=to_internal_fluid_props(props_in),
        tube_side_provider=inside.provider,
        tube_side_temperature_in=inside.T_in,
        tube_side_temperature_out=inside.T_out,
        tube_side_pressure=inside.p,
        m_dot_outside=outside.m_dot,
        outside_props=to_outside_fluid_props(props_out),
        outside_provider=outside.provider,
        outside_temperature_in=outside.T_in,
        outside_temperature_out=outside.T_out,
        outside_pressure=outside.p,
        K_inlet=K_inlet,
        K_outlet=K_outlet,
        K_turn=K_turn,
        flow_arrangement=flow_arrangement,
        euler_provider=euler_provider,
    )

    A_o = solve_result.A_o

    # Converged, wall-temperature- and finite-length-corrected thermal state
    # at the closed balance's actual inlet temperatures/flows/providers.
    # This is the authoritative source for alfa_i/alfa_o/U_mean/UA_actual.
    thermal_state = solve_iterative_thermal_state(
        hx,
        m_dot_inside=inside.m_dot,
        m_dot_outside=outside.m_dot,
        inside_provider=inside.provider,
        outside_provider=outside.provider,
        T_in_inside=inside.T_in,
        T_in_outside=outside.T_in,
        p_inside=inside.p,
        p_outside=outside.p,
        flow_arrangement=flow_arrangement,
        euler_provider=euler_provider,
        max_iterations=max_iterations,
        wall_temperature_tolerance_K=wall_temperature_tolerance_K,
        relative_alfa_tolerance=relative_alfa_tolerance,
        relaxation_factor=relaxation_factor,
    )
    wall_temperature_envelope = estimate_wall_temperature_envelope(
        hx,
        m_dot_inside=inside.m_dot,
        m_dot_outside=outside.m_dot,
        inside_provider=inside.provider,
        outside_provider=outside.provider,
        inside_inlet_temperature=inside.T_in,
        inside_outlet_temperature=inside.T_out,
        outside_inlet_temperature=outside.T_in,
        outside_outlet_temperature=outside.T_out,
        p_inside=inside.p,
        p_outside=outside.p,
        euler_provider=euler_provider,
        max_iterations=max_iterations,
        wall_temperature_tolerance_K=wall_temperature_tolerance_K,
        relative_alfa_tolerance=relative_alfa_tolerance,
        relaxation_factor=relaxation_factor,
    )

    UA_actual = thermal_state.UA
    U_mean = thermal_state.U

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

    warnings_list: list[ModelWarning] = (
        list(closed_balance.warnings or [])
        + list(solve_result.warnings or [])
        + list(thermal_state.warnings)
        + list(wall_temperature_envelope.warnings)
    )

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
        EMTD=closed_balance.Q / UA_req,
        alfa_i=thermal_state.alfa_i,
        alfa_o=thermal_state.alfa_o,
        Q_required=closed_balance.Q,
        Q_achievable=Q_achievable,
        closed_balance=closed_balance,
        final_result=solve_result,
        simulation=simulation_result,
        thermal_state=thermal_state,
        wall_temperature_envelope=wall_temperature_envelope,
        warnings=warnings_list if warnings_list else None,
    )
