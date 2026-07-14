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
v0.5.x - Iterative Mean-Property Heat Exchanger Simulation.

Motivation
----------
The MVP ``BareTubeHeatExchanger.solve`` evaluates every transport property,
velocity, Re, Pr, ``alfa`` and ``U`` at a *single* externally supplied state
(in practice the inlet state). For water heaters with a moderate temperature
change this is acceptable, because properties barely move the result. For
gas-gas duty with a large temperature span (e.g. 30 degC vs 400 degC) the
inlet-only assumption is not adequate: rho, mu, k, cp, Pr, Re, velocity, alfa
and U all change substantially between inlet and outlet.

This module keeps the physics/correlations of ``BareTubeHeatExchanger``
untouched and adds an *outer* iteration that:

    1. guesses outlet temperatures on both sides,
    2. evaluates each side's transport properties at its *mean bulk*
       temperature ``T_mean = 0.5 * (T_in + T_out)`` using a property provider,
    3. rebuilds the sensible capacity rates ``C = m_dot * cp(T_mean)``,
    4. calls the existing MVP solver with the mean-state properties,
    5. relaxes the new outlet temperatures,
    6. repeats until duty and both outlet temperatures converge.

Design-practice terminology
----------------------------
Given known geometry, both inlet temperatures, and both flow rates, this is
a **Simulation**: the result is the achievable outlet temperatures (and duty).
This is distinct from **Rating** (``core.models.rating``), which starts from a
*closed* heat balance (known outlet temperatures too) and reports how much
surface margin / overdesign the geometry provides.

Default entry point and the "forced averaged properties" case
---------------------------------------------------------------
The intended default entry point is ``BareTubeHeatExchanger.simulate(...)``,
which runs the mean-property iteration. There is one deliberate exception:

    If both sides use a ``ConstantPropertyProvider`` (or the caller passes
    ``iterate=False``), the supplied properties are taken as *already averaged*
    and the simulation collapses to a single ``solve`` pass.

This is not a heuristic shortcut: with constant properties ``UA`` and the
capacity rates ``C`` do not depend on the guessed outlet temperatures, so the
epsilon-NTU balance is exact in one pass and iterating would only walk the
relaxation on the outlet temperatures for no physical gain. The single pass
returns ``converged=True`` and ``iterations=1``.

Surface margin (derating)
--------------------------
``surface_margin`` is an *input* to Simulation (``0.0`` by default, meaning
"on the nose" / no margin). When ``surface_margin > 0``, the full-geometry
``UA`` from ``solve()`` is derated before the epsilon-NTU balance is redone:

    UA_eff = UA_full / (1 + surface_margin)
    eps_eff = effectiveness_ntu(C_hot, C_cold, UA_eff, flow_arrangement)
    Q_eff, T_hot_out_eff, T_cold_out_eff = heat_duty_from_effectiveness(eps_eff, ...)

The derated duty and outlet temperatures (not the full-UA ones) are what the
outer relaxation loop tracks and what the result reports as ``q``/``T_out_*``.
``surface_margin=0.0`` skips this recomputation entirely, so results are
bit-for-bit identical to not having the parameter at all.

Scope (first v0.5.x)
--------------------
In scope: sensible heat transfer; dry air, gas mixtures, water/steam where
single-phase applies; gas-gas and gas-liquid; cases with a large property
change; convergence diagnostics.

Out of scope (deliberately not handled here): condensation, wet surface,
latent heat, water removal from a gas composition, acid dew point, full wet
economizer, row-by-row / segmented models, wall-temperature iteration.

Conventions
-----------
- This is a *driver*: it does not duplicate any correlation. Each iteration is
  one call to ``BareTubeHeatExchanger.solve``.
- "inside" always maps to the tube side; "outside" maps to the outside/bundle
  side. Which side is thermally hot vs cold is decided from the inlet
  temperatures, independently of the tube/outside geometry role.
- ``U_mean`` is referenced to the outer heat-transfer area: ``U_mean = UA / A_o``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.properties.common import FluidTransportProperties
from core.properties.fluids import PropertyProvider, ConstantPropertyProvider
from core.properties.averaging import mean_temperature
from core.properties.adapters import (
    to_internal_fluid_props,
    to_outside_fluid_props,
)

from core.heat_transfer.streams import SensibleHeatStream
from core.heat_transfer.ntu import effectiveness_ntu, heat_duty_from_effectiveness

from core.models.bare_tube import BareTubeHeatExchanger, HXResult

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

    For a single-pass simulation (forced constant properties, or
    ``iterate=False``) ``converged`` is True and ``iterations`` is 1; the
    reported ``T_mean_*`` are the bulk means of the computed outlet
    temperatures.

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

    # Transport properties used in the final pass
    inside_props_mean: FluidTransportProperties
    outside_props_mean: FluidTransportProperties

    # Flow / dimensionless numbers at the final pass
    inside_velocity_mean: float
    outside_velocity_mean: float
    inside_Re_mean: float
    outside_Re_mean: float
    inside_Pr_mean: float
    outside_Pr_mean: float
    inside_alfa_mean: float
    outside_alfa_mean: float

    # Overall performance (of the real, undegraded geometry)
    U_mean: float          # [W/(m2*K)] referenced to outer area A_o
    UA: float              # [W/K]

    # Achieved duty and outlet temperatures (after surface_margin derating)
    q: float                # [W]
    T_out_inside: float     # [K]
    T_out_outside: float    # [K]

    # Surface margin (input, echoed) and duty transparency
    surface_margin: float   # [-] 0.0 = "on the nose"; input, not an output
    overdesign_factor: float  # [-] always 0.0 for Simulation; Rating computes it
    Q_full: float            # [W] duty at the real geometry's full UA
    Q_derated: float         # [W] duty after surface_margin derating (== q)

    # Full snapshot of the final MVP solve (areas, hydraulics, per-side blocks)
    final_result: HXResult

    # Diagnostics
    warnings: list[ModelWarning] | None = None


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
    # Convergence controls (defaults per v0.5.x spec):
    max_iter: int = 30,
    temperature_tolerance_K: float = 0.05,
    relative_duty_tolerance: float = 1e-4,
    relaxation_factor: float = 0.5,
) -> HXSimulationResult:
    """Simulate a bare-tube exchanger. Backing implementation of ``.simulate``.

    The default (``iterate=True`` with temperature-dependent providers) runs the
    mean-property iteration described in the module docstring. When both sides
    use a ``ConstantPropertyProvider`` or ``iterate=False`` is passed, a single
    ``solve`` pass is performed and the supplied properties are treated as the
    mean-bulk properties.

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
            m_dot_outside=outside.m_dot,
            outside_props=to_outside_fluid_props(props_out),
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

        return result, T_out_inside_calc, T_out_outside_calc, props_in, props_out, Q_eff, Q_full

    def _build_result(
        *,
        final_result: HXResult,
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
        A_o = final_result.A_o
        U_mean = final_result.UA / A_o if A_o > 0.0 else math.nan

        warnings_list: list[ModelWarning] = list(final_result.warnings or [])
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
            inside_alfa_mean=final_result.tube_side_thermal.alfa,
            outside_alfa_mean=final_result.outside_side_thermal.alfa,
            U_mean=U_mean,
            UA=final_result.UA,
            q=q,
            T_out_inside=T_out_inside,
            T_out_outside=T_out_outside,
            surface_margin=surface_margin,
            overdesign_factor=0.0,
            Q_full=Q_full,
            Q_derated=q,
            final_result=final_result,
            warnings=warnings_list if warnings_list else None,
        )

    # --- Single-pass path: forced averaged properties or iterate=False -------
    forced_constant = (
        isinstance(inside.provider, ConstantPropertyProvider)
        and isinstance(outside.provider, ConstantPropertyProvider)
    )
    if forced_constant or not iterate:
        # Properties are treated as already averaged. With constant providers the
        # outlet temperatures from a single pass are exact; with a variable
        # provider and iterate=False the properties are the inlet properties.
        result, T_out_inside_calc, T_out_outside_calc, props_in, props_out, Q_eff, Q_full = _evaluate(
            inside.T_in, outside.T_in
        )
        return _build_result(
            final_result=result,
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

    # --- Iterative mean-property path ---------------------------------------
    # Initial guess: outlet == inlet (first pass evaluates at the inlet state).
    T_out_inside = inside.T_in
    T_out_outside = outside.T_in

    Q_prev: float | None = None
    residual_q_rel = math.inf
    residual_T_inside_K = math.inf
    residual_T_outside_K = math.inf
    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration

        _result, T_out_inside_calc, T_out_outside_calc, _pin, _pout, Q_eff, _Q_full = _evaluate(
            T_out_inside, T_out_outside
        )

        # Under-relaxed outlet-temperature update.
        T_out_inside_new = T_out_inside + relaxation_factor * (T_out_inside_calc - T_out_inside)
        T_out_outside_new = T_out_outside + relaxation_factor * (T_out_outside_calc - T_out_outside)

        residual_T_inside_K = abs(T_out_inside_new - T_out_inside)
        residual_T_outside_K = abs(T_out_outside_new - T_out_outside)
        residual_q_rel = (
            abs(Q_eff - Q_prev) / max(abs(Q_eff), 1e-12) if Q_prev is not None else math.inf
        )

        T_out_inside = T_out_inside_new
        T_out_outside = T_out_outside_new
        Q_prev = Q_eff

        if (
            iteration >= 2
            and residual_q_rel < relative_duty_tolerance
            and residual_T_inside_K < temperature_tolerance_K
            and residual_T_outside_K < temperature_tolerance_K
        ):
            converged = True
            break

    # Final self-consistent evaluation at the converged mean state, so that all
    # reported means (velocity, Re, Pr, alfa, U, UA, q, T_out) are mutually
    # consistent with the converged outlet temperatures.
    final_result, T_out_inside_final, T_out_outside_final, props_in, props_out, Q_eff_final, Q_full_final = _evaluate(
        T_out_inside, T_out_outside
    )

    return _build_result(
        final_result=final_result,
        props_in=props_in,
        props_out=props_out,
        T_out_inside=T_out_inside_final,
        T_out_outside=T_out_outside_final,
        q=Q_eff_final,
        Q_full=Q_full_final,
        converged=converged,
        iterations=iterations,
        residual_q_rel=residual_q_rel,
        residual_T_inside_K=residual_T_inside_K,
        residual_T_outside_K=residual_T_outside_K,
    )
