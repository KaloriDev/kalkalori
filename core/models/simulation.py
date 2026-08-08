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
v0.5.5 - Iterative Thermal-State Heat Exchanger Simulation.

Motivation
----------
Thermal coefficients continue to use the existing converged mean bulk state.
Hydraulics are separate from the thermal convergence: the production result
evaluates inlet, midpoint, and outlet bulk properties on both the tube side
and the outside tube bank. Tube-side friction integrates ``f/rho``; outside
bank drag integrates the provider-contract ``Eu/rho`` quantity and adds the
signed face-reference acceleration term. The hydraulic representative
pressure is held constant for those property evaluations; this patch does not
add pressure-distribution iteration.

Authoritative thermal state (v0.5.3)
--------------------------------------
The default (``iterate=True``) path now delegates the *entire* mean-bulk and
wall-temperature convergence to
``core.heat_transfer.thermal_iteration.solve_iterative_thermal_state`` -- the
same function used by ``core.models.rating`` -- instead of re-implementing a
bespoke outer mean-property loop around uncorrected ``BareTubeHeatExchanger.
solve()`` passes. Concretely:

    1. ``solve_iterative_thermal_state(...)`` converges the mean bulk
       temperatures, both tube-wall surface temperatures, and the
       wall/length-corrected ``alfa_i``, ``alfa_o``, ``U``, ``UA``.
    2. Achievable duty and outlet temperatures are computed from that
       converged ``UA`` via the existing epsilon-NTU relations
       (``effectiveness_ntu``/``heat_duty_from_effectiveness``) -- reused, not
       duplicated.
    3. ``surface_margin`` (if requested) derates that same ``UA`` exactly as
       before.
    4. One final ``BareTubeHeatExchanger.solve()`` snapshot is assembled after
       the actual outlet temperatures are known.  Its nested hydraulic result
       is the common three-state straight-bundle calculation; its thermal
       coefficients remain diagnostics while ``inside_alfa_mean``/
       ``outside_alfa_mean``/``U_mean``/``UA`` come from the converged thermal
       state.

This supersedes the v0.5.1/v0.5.2 behavior, where Simulation deliberately
never computed a wall temperature at all (documented as out of scope) and
``inside_alfa_mean``/``UA`` were always the plain constant-property Gnielinski/
Zukauskas result.

Design-practice terminology
----------------------------
Given known geometry, both inlet temperatures, and both flow rates, this is
a **Simulation**: the result is the achievable outlet temperatures (and duty).
This is distinct from **Rating** (``core.models.rating``), which starts from a
*closed* heat balance (known outlet temperatures too) and reports how much
surface margin / overdesign the geometry provides.

Default entry point and the ``iterate=False`` escape hatch
---------------------------------------------------------------
The intended default entry point is ``BareTubeHeatExchanger.simulate(...)``,
which runs the corrected iterative thermal state described above -- for
*any* property provider, including ``ConstantPropertyProvider``: even with
constant bulk properties, the wall-temperature correction is not known a
priori and still requires iteration, so a ``ConstantPropertyProvider`` no
longer collapses this path to a single pass (unlike v0.5.2).

The caller may still request an explicit, fast, *uncorrected* single pass at
the raw inlet state via ``iterate=False``: this bypasses
``solve_iterative_thermal_state`` entirely (a single ``BareTubeHeatExchanger.
solve()`` call, ``converged=True``, ``iterations=1``, no wall-temperature
correction, ``thermal_state=None`` on the result) -- an intentional
approximation, not a bug, for callers that want a cheap estimate.

Surface margin (derating)
--------------------------
``surface_margin`` is an *input* to Simulation (``0.0`` by default, meaning
"on the nose" / no margin). When ``surface_margin > 0``, the converged
``UA`` (from the thermal state, or from the single ``solve()`` pass when
``iterate=False``) is derated before the epsilon-NTU balance is redone:

    UA_eff = UA_full / (1 + surface_margin)
    eps_eff = effectiveness_ntu(C_hot, C_cold, UA_eff, flow_arrangement)
    Q_eff, T_hot_out_eff, T_cold_out_eff = heat_duty_from_effectiveness(eps_eff, ...)

The derated duty and outlet temperatures (not the full-UA ones) are what the
result reports as ``q``/``T_out_*``. ``surface_margin=0.0`` skips this
recomputation entirely, so results are bit-for-bit identical to not having
the parameter at all.

Scope (v0.5.3)
--------------------
In scope: sensible heat transfer; dry air, gas mixtures, water/steam where
single-phase applies; gas-gas and gas-liquid; cases with a large property
change; convergence diagnostics; wall-temperature/finite-length correction
(via the shared iterative thermal state).

Out of scope (deliberately not handled here): condensation, wet surface,
latent heat, water removal from a gas composition, acid dew point, full wet
economizer, row-by-row / segmented models.

Conventions
-----------
- This is a *driver*: it does not duplicate any correlation. The corrected
  path is one call to ``solve_iterative_thermal_state`` plus one call to
  ``BareTubeHeatExchanger.solve`` (diagnostics only, see above); the
  ``iterate=False`` path is one call to ``BareTubeHeatExchanger.solve``.
- "inside" always maps to the tube side; "outside" maps to the outside/bundle
  side. Which side is thermally hot vs cold is decided from the inlet
  temperatures, independently of the tube/outside geometry role.
- ``U_mean`` is referenced to the outer heat-transfer area: ``U_mean = UA / A_o``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from core.properties.common import FluidTransportProperties
from core.properties.fluids import PropertyProvider
from core.properties.averaging import mean_temperature
from core.properties.adapters import (
    to_internal_fluid_props,
    to_outside_fluid_props,
)

from core.heat_transfer.streams import SensibleHeatStream
from core.heat_transfer.ntu import effectiveness_ntu, heat_duty_from_effectiveness
from core.heat_transfer.thermal_iteration import (
    IterativeThermalState,
    WallTemperatureEnvelope,
    estimate_wall_temperature_envelope,
    solve_iterative_thermal_state,
)

from core.models.bare_tube import BareTubeHeatExchanger, HXResult
from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
from core.pressure_drop.flow_path import (
    build_tube_side_pressure_drop_result,
    build_outside_pressure_drop_result,
)
from core.heat_transfer.outside_flow import calculate_outside_tube_bank_hydraulics

from core.common.warnings import ModelWarning, make_warning


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HXSideInput:
    """Per-side input for a heat-exchanger simulation.

    A "side" is one fluid stream flowing through the exchanger. Its transport
    properties are recomputed at the mean bulk temperature on every iteration
    via ``provider.at(T, p)``.

    Also used as the bridge type from a closed heat balance
    (``core.models.heat_balance.ClosedBalanceSide.to_hx_side_input()``) so a
    Rating's closed balance can be re-run through Simulation for comparison.

    Attributes:
        provider: Point-property provider exposing ``at(T, p)`` and returning a
            ``FluidTransportProperties`` (rho, mu, k, cp). Any provider works:
            ``ConstantPropertyProvider`` (forced/averaged properties),
            ``DryAirPropertyProvider``, ``GasMixturePropertyProvider``,
            ``IAPWS97WaterSteamProvider``, etc.
        m_dot: Total mass flow through this side [kg/s].
        T_in: Inlet bulk temperature [K].
        p: Bulk pressure used for property evaluation [Pa]. Constant along the
            side in this 0D simulation (pressure drop does not feed back into
            properties here).
    """

    provider: PropertyProvider
    m_dot: float
    T_in: float
    p: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.m_dot) or self.m_dot <= 0.0:
            raise ValueError("m_dot must be a positive finite value [kg/s].")
        if not math.isfinite(self.T_in) or self.T_in <= 0.0:
            raise ValueError("T_in must be a positive finite value [K].")
        if not math.isfinite(self.p) or self.p <= 0.0:
            raise ValueError("p must be a positive finite value [Pa].")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HXSimulationResult:
    """Result of a heat-exchanger simulation (mean-property or single-pass).

    All ``*_mean`` diagnostics are reported at the converged mean bulk state of
    the respective side. Naming follows the tube/outside geometry role:
    ``inside`` is the tube side, ``outside`` is the bundle side.

    For the ``iterate=False`` single-pass escape hatch, ``converged`` is True
    and ``iterations`` is 1, ``thermal_state`` is ``None``, and
    ``inside_alfa_mean``/``outside_alfa_mean``/``U_mean``/``UA`` are the plain
    uncorrected coefficients from that single pass; the reported ``T_mean_*``
    are the bulk means of the computed outlet temperatures. Otherwise (the
    default, corrected path), ``inside_alfa_mean``/``outside_alfa_mean``/
    ``U_mean``/``UA`` are read from the converged ``thermal_state`` (wall/
    length-corrected) and are never later overwritten by the separate
    (uncorrected) ``final_result`` solve pass, which is retained only for its
    area/hydraulic/regime diagnostics.

    Simulation does not rate the exchanger against a required duty, so its
    output ``overdesign_factor`` is defined as ``0.0``.  This keeps result
    reporting consistent with Rating without conflating the output with the
    input ``surface_margin`` derating.
    """

    # Convergence diagnostics
    converged: bool
    iterations: int
    residual_q_rel: float
    residual_T_inside_K: float
    residual_T_outside_K: float

    # Converged mean bulk temperatures [K]
    T_mean_inside: float
    T_mean_outside: float

    # Transport properties used in the final pass (bulk-mean; see
    # thermal_state.inside_wall_props/outside_wall_props for wall-state
    # properties, kept separate and never overwriting these).
    inside_props_mean: FluidTransportProperties
    outside_props_mean: FluidTransportProperties

    # Flow / dimensionless numbers at the final pass
    inside_velocity_mean: float
    outside_velocity_mean: float
    inside_Re_mean: float
    outside_Re_mean: float
    inside_Pr_mean: float
    outside_Pr_mean: float
    inside_alfa_mean: float    # == thermal_state.alfa_i when thermal_state is not None
    outside_alfa_mean: float   # == thermal_state.alfa_o when thermal_state is not None

    # Overall performance (of the real, undegraded geometry)
    U_mean: float          # [W/(m2*K)] referenced to outer area A_o; == thermal_state.U
    UA: float              # [W/K]; == thermal_state.UA

    # Achieved duty and outlet temperatures (after surface_margin derating)
    q: float                # [W]
    T_out_inside: float     # [K]
    T_out_outside: float    # [K]

    # Surface margin (input, echoed) and duty transparency
    surface_margin: float   # [-] 0.0 = "on the nose"; input, not an output
    overdesign_factor: float  # [-] always 0.0 for Simulation; Rating computes it
    Q_full: float            # [W] duty at the real geometry's full UA
    Q_derated: float         # [W] duty after surface_margin derating (== q)

    # Full snapshot of the final MVP solve (areas, hydraulics, per-side
    # blocks); alfa/UA fields on this snapshot are NOT authoritative -- see
    # thermal_state below.
    final_result: HXResult

    # Converged iterative thermal state (wall temperatures, per-side
    # correction/Nu diagnostics, convergence/iterations/residual). ``None``
    # only for the ``iterate=False`` single-pass escape hatch. This is the
    # single source of truth for inside_alfa_mean/outside_alfa_mean/U_mean/UA
    # above.
    thermal_state: IterativeThermalState | None

    # Independent four-point 0D endpoint estimate. It is diagnostic only and
    # does not feed back into the mean-state solution above.
    wall_temperature_envelope: WallTemperatureEnvelope

    # Diagnostics
    warnings: list[ModelWarning] | None = None

    @property
    def inside_wall_temperature_mean(self) -> float | None:
        return None if self.thermal_state is None else self.thermal_state.inside_wall_temperature

    @property
    def outside_wall_temperature_mean(self) -> float | None:
        return None if self.thermal_state is None else self.thermal_state.outside_wall_temperature

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
        return self.final_result.tube_side_hydraulic

    @property
    def inside_dp_friction(self) -> float:
        return self.final_result.inside_dp_friction

    @property
    def inside_dp_acceleration(self) -> float:
        return self.final_result.inside_dp_acceleration

    @property
    def inside_dp_tube_bundle(self) -> float:
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
        return self.final_result.tube_side_pressure_drop

    @property
    def outside_side_pressure_drop(self):
        """Complete outside-side pressure-drop result (core + local + total)."""
        return self.final_result.outside_side_pressure_drop

    @property
    def inside_dp_local(self) -> float:
        return self.final_result.inside_dp_local

    @property
    def outside_dp_local(self) -> float:
        return self.final_result.outside_dp_local

    # -- Tube-sheet entrance/exit pressure drop (v0.5.6) --------------------

    @property
    def inside_dp_straight_tube_friction(self) -> float:
        return self.final_result.inside_dp_straight_tube_friction

    @property
    def inside_dp_straight_tube_acceleration(self) -> float:
        return self.final_result.inside_dp_straight_tube_acceleration

    @property
    def inside_dp_straight_tubes(self) -> float:
        return self.final_result.inside_dp_straight_tubes

    @property
    def inside_dp_tube_entrances(self) -> float:
        return self.final_result.inside_dp_tube_entrances

    @property
    def inside_dp_tube_exits(self) -> float:
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
        return self.final_result.pass_boundary_states

    @property
    def entrance_results(self):
        return self.final_result.entrance_results

    @property
    def exit_results(self):
        return self.final_result.exit_results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_simulation(
    hx: BareTubeHeatExchanger,
    inside: HXSideInput,
    outside: HXSideInput,
    *,
    surface_margin: float = 0.0,
    iterate: bool = True,
    flow_arrangement: str | None = None,
    # Tube-side pressure-drop loss coefficients (forwarded to the MVP solver):
    K_inlet: float = 0.5,
    K_outlet: float = 1.0,
    K_turn: float = 1.5,
    # Outside pressure-drop provider selection:
    euler_provider: str = "zukauskas",
    # Convergence controls (defaults per v0.5.x spec; forwarded to
    # solve_iterative_thermal_state as max_iterations/wall_temperature_
    # tolerance_K/relaxation_factor -- same meaning/defaults as
    # BareTubeHeatExchanger.solve_thermal_state / .rate()):
    max_iter: int = 30,
    temperature_tolerance_K: float = 0.05,
    relative_duty_tolerance: float = 1e-4,
    relaxation_factor: float = 0.5,
    relative_alfa_tolerance: float = 1e-3,
) -> HXSimulationResult:
    """Simulate a bare-tube exchanger. Backing implementation of ``.simulate``.

    The default (``iterate=True``, for any property provider) calls
    ``solve_iterative_thermal_state`` to converge the mean-bulk and
    wall-temperature state and reports its corrected coefficients; see the
    module docstring for the full call path. ``iterate=False`` bypasses this
    entirely for a single, uncorrected ``solve()`` pass at the inlet state.

    ``relative_duty_tolerance`` is accepted for backward compatibility but is
    not used by the default (corrected) path, which converges on wall
    temperature/alfa (``temperature_tolerance_K``/``relative_alfa_tolerance``)
    exactly like ``solve_iterative_thermal_state``.

    ``surface_margin`` (default ``0.0``) derates the full-geometry ``UA``
    before computing duty/outlet temperatures; see the module docstring.
    """
    if flow_arrangement is None:
        flow_arrangement = hx.bundle.flow_arrangement

    if not (0.0 < relaxation_factor <= 1.0):
        raise ValueError("relaxation_factor must be in (0, 1].")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if not math.isfinite(surface_margin) or surface_margin < 0.0:
        raise ValueError("surface_margin must be a non-negative finite value.")

    hot_is_inside = inside.T_in >= outside.T_in

    def _evaluate(T_out_inside: float, T_out_outside: float):
        """One solver pass at the mean state implied by the given outlet temps.

        Returns:
            (result, T_out_inside_calc, T_out_outside_calc,
             props_inside, props_outside, Q_eff, Q_full)
        """
        T_mean_inside = mean_temperature(inside.T_in, T_out_inside)
        T_mean_outside = mean_temperature(outside.T_in, T_out_outside)

        props_in = inside.provider.at(T=T_mean_inside, p=inside.p)
        props_out = outside.provider.at(T=T_mean_outside, p=outside.p)

        C_inside = inside.m_dot * props_in.cp
        C_outside = outside.m_dot * props_out.cp

        if hot_is_inside:
            hot_stream = SensibleHeatStream(C=C_inside, T_in=inside.T_in)
            cold_stream = SensibleHeatStream(C=C_outside, T_in=outside.T_in)
        else:
            hot_stream = SensibleHeatStream(C=C_outside, T_in=outside.T_in)
            cold_stream = SensibleHeatStream(C=C_inside, T_in=inside.T_in)

        result = hx.solve(
            hot_stream=hot_stream,
            cold_stream=cold_stream,
            m_dot_tube_side=inside.m_dot,
            tube_side_props=to_internal_fluid_props(props_in),
            tube_side_provider=inside.provider,
            tube_side_temperature_in=inside.T_in,
            tube_side_temperature_out=T_out_inside,
            tube_side_pressure=inside.p,
            m_dot_outside=outside.m_dot,
            outside_props=to_outside_fluid_props(props_out),
            outside_provider=outside.provider,
            outside_temperature_in=outside.T_in,
            outside_temperature_out=T_out_outside,
            outside_pressure=outside.p,
            K_inlet=K_inlet,
            K_outlet=K_outlet,
            K_turn=K_turn,
            flow_arrangement=flow_arrangement,
            euler_provider=euler_provider,
        )

        Q_full = result.Q

        if surface_margin > 0.0:
            UA_eff = result.UA / (1.0 + surface_margin)
            eps_eff = effectiveness_ntu(
                C_hot=hot_stream.capacity_rate(),
                C_cold=cold_stream.capacity_rate(),
                UA=UA_eff,
                flow_arrangement=flow_arrangement,
                C_inside=C_inside,
                C_outside=C_outside,
            )
            Q_eff, T_hot_out_eff, T_cold_out_eff = heat_duty_from_effectiveness(
                eps=eps_eff,
                hot_stream=hot_stream,
                cold_stream=cold_stream,
            )
        else:
            Q_eff, T_hot_out_eff, T_cold_out_eff = (
                result.Q,
                result.T_hot_out,
                result.T_cold_out,
            )

        if hot_is_inside:
            T_out_inside_calc = T_hot_out_eff
            T_out_outside_calc = T_cold_out_eff
        else:
            T_out_inside_calc = T_cold_out_eff
            T_out_outside_calc = T_hot_out_eff

        # The thermal pass determines the actual outlet temperatures after the
        # hydraulic snapshot was first evaluated. Refresh both common
        # three-state hydraulic results so exposed states use those outlets.
        bundle_hydraulic = calculate_tube_bundle_hydraulics(
            m_dot=inside.m_dot,
            flow_area_per_pass=hx.bundle.internal_flow_area_per_pass,
            hydraulic_diameter=hx.bundle.internal_hydraulic_diameter,
            hydraulic_length_total=hx.bundle.internal_length_total,
            n_tube_passes=hx.bundle.n_passes_tube,
            tube_path_type=hx.bundle.tube_path_type,
            roughness_inner=getattr(hx.bundle.tube, "roughness_inner", None),
            provider=inside.provider,
            temperature_in=inside.T_in,
            temperature_out=T_out_inside_calc,
            pressure=inside.p,
        )
        tube_hydraulic = replace(
            result.tube_side_hydraulic,
            tube_bundle=bundle_hydraulic,
        )
        outside_bank_hydraulic = calculate_outside_tube_bank_hydraulics(
            m_dot=outside.m_dot,
            face_area=hx.bundle.frontal_flow_area,
            tube_outer_diameter=float(getattr(hx.bundle.tube, "D_o")),
            tube_pitch_transverse=hx.bundle.pitch_transverse,
            tube_pitch_longitudinal=hx.bundle.pitch_longitudinal,
            layout=hx.bundle.layout,
            n_rows=hx.bundle.n_rows,
            n_tubes_per_row=hx.bundle.n_tubes_per_row,
            provider=outside.provider,
            temperature_in=outside.T_in,
            temperature_out=T_out_outside_calc,
            pressure=outside.p,
            euler_provider=euler_provider,
        )
        outside_hydraulic = replace(
            result.outside_side_hydraulic,
            dp_total=outside_bank_hydraulic.dp_total,
            Re=outside_bank_hydraulic.midpoint.reynolds,
            v=outside_bank_hydraulic.midpoint.face_velocity,
            tube_bank=outside_bank_hydraulic,
        )
        # Refresh the pressure-drop flow-path aggregation to match the
        # refreshed hydraulic snapshots. The path result decomposes each
        # legacy signed core pressure difference into irreversible loss and
        # dynamic-pressure change. No specified local-loss path is threaded
        # through Simulation in this commit, so this is always core-only
        # aggregation (dp_local=0).
        tube_side_pressure_drop = replace(
            result.tube_side_pressure_drop,
            tube_bundle=bundle_hydraulic,
            flow_path=build_tube_side_pressure_drop_result(
                bundle_hydraulic,
                n_tube_passes=hx.bundle.n_passes_tube,
            ),
        )
        outside_side_pressure_drop = replace(
            result.outside_side_pressure_drop,
            tube_bank=outside_bank_hydraulic,
            flow_path=build_outside_pressure_drop_result(outside_bank_hydraulic),
        )
        result_warnings = [
            warning
            for warning in (result.warnings or [])
            if warning.source not in {
                "tube_bundle_hydraulics",
                "outside_bank_hydraulics",
            }
        ]
        result_warnings.extend(bundle_hydraulic.warnings)
        result_warnings.extend(outside_bank_hydraulic.warnings)
        result = replace(
            result,
            tube_side_hydraulic=tube_hydraulic,
            outside_side_hydraulic=outside_hydraulic,
            tube_side_pressure_drop=tube_side_pressure_drop,
            outside_side_pressure_drop=outside_side_pressure_drop,
            warnings=result_warnings if result_warnings else None,
        )

        return result, T_out_inside_calc, T_out_outside_calc, props_in, props_out, Q_eff, Q_full

    def _build_result(
        *,
        final_result: HXResult,
        thermal_state: IterativeThermalState | None,
        props_in: FluidTransportProperties,
        props_out: FluidTransportProperties,
        T_out_inside: float,
        T_out_outside: float,
        q: float,
        Q_full: float,
        converged: bool,
        iterations: int,
        residual_q_rel: float,
        residual_T_inside_K: float,
        residual_T_outside_K: float,
    ) -> HXSimulationResult:
        # Authoritative source for alfa_i/alfa_o/U/UA: the converged thermal
        # state when available; the explicit iterate=False path uses the
        # single uncorrected thermal snapshot by design. Hydraulic fields on
        # final_result always come from the three-state tube-bundle function.
        if thermal_state is not None:
            inside_alfa_mean = thermal_state.alfa_i
            outside_alfa_mean = thermal_state.alfa_o
            U_mean = thermal_state.U
            UA = thermal_state.UA
            thermal_warnings = list(thermal_state.warnings)
        else:
            A_o = final_result.A_o
            inside_alfa_mean = final_result.tube_side_thermal.alfa
            outside_alfa_mean = final_result.outside_side_thermal.alfa
            U_mean = final_result.UA / A_o if A_o > 0.0 else math.nan
            UA = final_result.UA
            thermal_warnings = []

        envelope = estimate_wall_temperature_envelope(
            hx,
            m_dot_inside=inside.m_dot,
            m_dot_outside=outside.m_dot,
            inside_provider=inside.provider,
            outside_provider=outside.provider,
            inside_inlet_temperature=inside.T_in,
            inside_outlet_temperature=T_out_inside,
            outside_inlet_temperature=outside.T_in,
            outside_outlet_temperature=T_out_outside,
            p_inside=inside.p,
            p_outside=outside.p,
            euler_provider=euler_provider,
            max_iterations=max_iter,
            wall_temperature_tolerance_K=temperature_tolerance_K,
            relative_alfa_tolerance=relative_alfa_tolerance,
            relaxation_factor=relaxation_factor,
        )
        warnings_list: list[ModelWarning] = (
            list(final_result.warnings or []) + thermal_warnings + list(envelope.warnings)
        )
        if not converged:
            warnings_list.append(
                make_warning(
                    code="simulation_not_converged",
                    message=(
                        "simulation: iterative mean-property simulation did "
                        f"not converge within max_iter={iterations}. Last "
                        f"residuals: duty_rel={residual_q_rel:.3e}, "
                        f"dT_inside={residual_T_inside_K:.3e} K, "
                        f"dT_outside={residual_T_outside_K:.3e} K. "
                        "Returning the last iterate."
                    ),
                    source="simulation",
                    severity="warning",
                )
            )

        return HXSimulationResult(
            converged=converged,
            iterations=iterations,
            residual_q_rel=residual_q_rel,
            residual_T_inside_K=residual_T_inside_K,
            residual_T_outside_K=residual_T_outside_K,
            T_mean_inside=mean_temperature(inside.T_in, T_out_inside),
            T_mean_outside=mean_temperature(outside.T_in, T_out_outside),
            inside_props_mean=props_in,
            outside_props_mean=props_out,
            inside_velocity_mean=final_result.tube_side_thermal.v,
            outside_velocity_mean=final_result.outside_side_thermal.v,
            inside_Re_mean=final_result.tube_side_thermal.Re,
            outside_Re_mean=final_result.outside_side_thermal.Re,
            inside_Pr_mean=final_result.tube_side_thermal.Pr,
            outside_Pr_mean=final_result.outside_side_thermal.Pr,
            inside_alfa_mean=inside_alfa_mean,
            outside_alfa_mean=outside_alfa_mean,
            U_mean=U_mean,
            UA=UA,
            q=q,
            T_out_inside=T_out_inside,
            T_out_outside=T_out_outside,
            surface_margin=surface_margin,
            overdesign_factor=0.0,
            Q_full=Q_full,
            Q_derated=q,
            final_result=final_result,
            thermal_state=thermal_state,
            wall_temperature_envelope=envelope,
            warnings=warnings_list if warnings_list else None,
        )

    # --- iterate=False: explicit uncorrected single-pass escape hatch -------
    if not iterate:
        # Properties are treated as already averaged (the inlet state); no
        # wall-temperature correction is applied. See module docstring.
        result, T_out_inside_calc, T_out_outside_calc, props_in, props_out, Q_eff, Q_full = _evaluate(
            inside.T_in, outside.T_in
        )
        return _build_result(
            final_result=result,
            thermal_state=None,
            props_in=props_in,
            props_out=props_out,
            T_out_inside=T_out_inside_calc,
            T_out_outside=T_out_outside_calc,
            q=Q_eff,
            Q_full=Q_full,
            converged=True,
            iterations=1,
            residual_q_rel=0.0,
            residual_T_inside_K=0.0,
            residual_T_outside_K=0.0,
        )

    # --- Default path: converged, wall/length-corrected thermal state -------
    # Delegates the entire mean-bulk + wall-temperature convergence to
    # solve_iterative_thermal_state (the same function core.models.rating
    # uses), rather than re-running a bespoke, uncorrected outer loop here.
    # This applies for ANY property provider, including
    # ConstantPropertyProvider: unlike v0.5.2, constant bulk properties no
    # longer imply a trivial single pass, because the wall-temperature
    # correction still needs to converge.
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
        max_iterations=max_iter,
        wall_temperature_tolerance_K=temperature_tolerance_K,
        relative_alfa_tolerance=relative_alfa_tolerance,
        relaxation_factor=relaxation_factor,
    )

    # Build the final snapshot after thermal convergence.  The tube-side
    # hydraulic state is evaluated from the actual inlet/outlet temperatures;
    # no one-state hydraulic result is propagated or used here.
    if hot_is_inside:
        hot_stream = SensibleHeatStream(C=inside.m_dot * thermal_state.inside_bulk_props.cp, T_in=inside.T_in)
        cold_stream = SensibleHeatStream(C=outside.m_dot * thermal_state.outside_bulk_props.cp, T_in=outside.T_in)
    else:
        hot_stream = SensibleHeatStream(C=outside.m_dot * thermal_state.outside_bulk_props.cp, T_in=outside.T_in)
        cold_stream = SensibleHeatStream(C=inside.m_dot * thermal_state.inside_bulk_props.cp, T_in=inside.T_in)

    # Achievable duty/outlet temperatures from the CORRECTED UA (reuses the
    # existing epsilon-NTU relations; no correlation is duplicated here).
    C_inside = inside.m_dot * thermal_state.inside_bulk_props.cp
    C_outside = outside.m_dot * thermal_state.outside_bulk_props.cp
    eps_full = effectiveness_ntu(
        C_hot=hot_stream.capacity_rate(),
        C_cold=cold_stream.capacity_rate(),
        UA=thermal_state.UA,
        flow_arrangement=flow_arrangement,
        C_inside=C_inside,
        C_outside=C_outside,
    )
    Q_full, T_hot_out_full, T_cold_out_full = heat_duty_from_effectiveness(
        eps=eps_full, hot_stream=hot_stream, cold_stream=cold_stream,
    )

    if surface_margin > 0.0:
        UA_eff = thermal_state.UA / (1.0 + surface_margin)
        eps_eff = effectiveness_ntu(
            C_hot=hot_stream.capacity_rate(),
            C_cold=cold_stream.capacity_rate(),
            UA=UA_eff,
            flow_arrangement=flow_arrangement,
            C_inside=C_inside,
            C_outside=C_outside,
        )
        Q_eff, T_hot_out_eff, T_cold_out_eff = heat_duty_from_effectiveness(
            eps=eps_eff, hot_stream=hot_stream, cold_stream=cold_stream,
        )
    else:
        Q_eff, T_hot_out_eff, T_cold_out_eff = Q_full, T_hot_out_full, T_cold_out_full

    if hot_is_inside:
        T_out_inside_final, T_out_outside_final = T_hot_out_eff, T_cold_out_eff
    else:
        T_out_outside_final, T_out_inside_final = T_hot_out_eff, T_cold_out_eff

    # Build the final snapshot after the corrected thermal state has supplied
    # the actual tube-side outlet temperature.  The common hydraulic function
    # therefore evaluates inlet, midpoint, and outlet bulk properties on the
    # production result itself.
    final_result = hx.solve(
        hot_stream=hot_stream,
        cold_stream=cold_stream,
        m_dot_tube_side=inside.m_dot,
        tube_side_props=to_internal_fluid_props(thermal_state.inside_bulk_props),
        tube_side_provider=inside.provider,
        tube_side_temperature_in=inside.T_in,
        tube_side_temperature_out=T_out_inside_final,
        tube_side_pressure=inside.p,
        m_dot_outside=outside.m_dot,
        outside_props=to_outside_fluid_props(thermal_state.outside_bulk_props),
        outside_provider=outside.provider,
        outside_temperature_in=outside.T_in,
        outside_temperature_out=T_out_outside_final,
        outside_pressure=outside.p,
        K_inlet=K_inlet,
        K_outlet=K_outlet,
        K_turn=K_turn,
        flow_arrangement=flow_arrangement,
        euler_provider=euler_provider,
    )

    # thermal_state.residual is a single wall-temperature residual [K] shared
    # across both sides (see thermal_iteration.py); there is no separate
    # relative-duty residual in the unified path, so residual_q_rel is
    # reported as 0.0 (not tracked) rather than a stale/synthetic value.
    return _build_result(
        final_result=final_result,
        thermal_state=thermal_state,
        props_in=thermal_state.inside_bulk_props,
        props_out=thermal_state.outside_bulk_props,
        T_out_inside=T_out_inside_final,
        T_out_outside=T_out_outside_final,
        q=Q_eff,
        Q_full=Q_full,
        converged=thermal_state.converged,
        iterations=thermal_state.iterations,
        residual_q_rel=0.0,
        residual_T_inside_K=thermal_state.residual,
        residual_T_outside_K=thermal_state.residual,
    )
