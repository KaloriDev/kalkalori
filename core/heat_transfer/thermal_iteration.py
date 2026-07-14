# KalKalori — Heat Exchanger Open Engine
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
v0.5.2 - Iterative mean-property and wall-temperature thermal state.

This module is a thin orchestration layer over existing components:
- internal/outside correlations (``core.heat_transfer.internal_flow`` /
  ``core.heat_transfer.outside_flow``),
- property providers (``core.properties``),
- geometry / wall-resistance (``core.geometry.bundle``,
  ``BareTubeHeatExchanger.tube_wall_resistance``),
- the epsilon-NTU relations (``core.heat_transfer.ntu``).

It does not duplicate any correlation or property calculation; it only
coordinates repeated evaluation of them until the mean bulk state and both
tube-wall surface temperatures are self-consistent.

Iteration sequence
-------------------
1. Estimate mean bulk temperatures for inside and outside streams
   (``T_mean = 0.5*(T_in + T_out)``; ``T_out`` starts at the previous
   iterate, or ``T_in`` on the first pass).
2. Evaluate bulk properties for both sides via the supplied providers.
3. Calculate alfa_i and alfa_o at the bulk state (uncorrected on the first
   pass; wall-corrected from the second pass onward, using the previous
   iteration's wall temperatures).
4. Calculate thermal resistances, U and UA from alfa_i, alfa_o and the
   existing cylindrical wall-resistance model.
5. Calculate the current heat duty via epsilon-NTU (``core.heat_transfer.ntu``).
6. Determine inside and outside tube-wall surface temperatures from the
   mean bulk temperatures and the resistance network (see sign convention
   below).
7. Evaluate properties at both wall-surface temperatures.
8. Recalculate alfa_i (turbulent-gas wall correction, ``internal_flow.
   gas_wall_temperature_correction``) and alfa_o (outside wall Prandtl
   number ``Pr_s``, ``outside_flow.nusselt_zukauskas``) using the
   *separate* wall properties from step 7.
9. Recalculate U, UA and both wall temperatures.
10. Repeat (relaxed) until wall temperatures and alfa's converge, or
    ``max_iterations`` is reached.

Signed heat-flow convention
----------------------------
``q_inside_to_outside = UA * (T_bulk_inside - T_bulk_outside)`` is positive
when the inside (tube) stream runs hotter on a mean-bulk basis, and negative
for the reverse -- so both wall temperatures are derived from a
resistance-weighted split of the mean bulk temperature difference:

    T_wall_inside  = T_bulk_inside  - q_inside_to_outside * R_i
    T_wall_outside = T_bulk_outside + q_inside_to_outside * R_o

This uses the *mean-bulk-consistent* heat rate (``UA`` times the mean bulk
temperature difference), not the actual duty ``Q`` from epsilon-NTU on the
inlet temperatures, specifically so that the wall-temperature split is exact
regardless of the (generally different) inlet-driven duty: both wall
temperatures always lie within the bulk-temperature interval, and they
collapse together as the wall resistance ``R_w -> 0``. It is direction
-agnostic and holds whether the inside or the outside stream is hotter, so
callers do not need to know in advance which side is hot.

Scope
-----
0D model only (no axial/finned/segmented refinement). This module does not
compute pressure drop; see ``BareTubeHeatExchanger.solve`` for the combined
thermal + hydraulic single-pass kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.properties.common import FluidTransportProperties
from core.properties.fluids import PropertyProvider
from core.properties.averaging import mean_temperature

from core.heat_transfer.internal_flow import heat_transfer_coefficient_internal
from core.heat_transfer.outside_flow import (
    outside_flow_from_mass_flow,
    prandtl_number as outside_prandtl_number,
)
from core.heat_transfer.ntu import effectiveness_ntu, heat_duty_from_effectiveness
from core.heat_transfer.streams import SensibleHeatStream

from core.common.warnings import ModelWarning, make_warning

if TYPE_CHECKING:
    from core.models.bare_tube import BareTubeHeatExchanger


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IterativeThermalState:
    """Converged (or last-iterate) mean-property / wall-temperature state."""

    inside_bulk_temperature: float    # [K]
    outside_bulk_temperature: float   # [K]

    inside_wall_temperature: float    # [K] tube inner-surface temperature
    outside_wall_temperature: float   # [K] tube outer-surface temperature

    inside_bulk_props: FluidTransportProperties
    inside_wall_props: FluidTransportProperties

    outside_bulk_props: FluidTransportProperties
    outside_wall_props: FluidTransportProperties

    alfa_i: float   # [W/(m2*K)] wall-corrected
    alfa_o: float   # [W/(m2*K)] wall-corrected

    U: float    # [W/(m2*K)] referenced to outer area A_o
    UA: float   # [W/K]

    iterations: int
    converged: bool
    residual: float   # [K] last wall-temperature residual

    warnings: tuple[ModelWarning, ...] = ()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def solve_iterative_thermal_state(
    hx: "BareTubeHeatExchanger",
    *,
    m_dot_inside: float,
    m_dot_outside: float,
    inside_provider: PropertyProvider,
    outside_provider: PropertyProvider,
    T_in_inside: float,
    T_in_outside: float,
    p_inside: float,
    p_outside: float,
    flow_arrangement: str | None = None,
    euler_provider: str = "zukauskas",
    max_iterations: int = 25,
    wall_temperature_tolerance_K: float = 0.05,
    relative_alfa_tolerance: float = 1e-3,
    relaxation_factor: float = 0.5,
) -> IterativeThermalState:
    """Iteratively resolve mean bulk state and both wall temperatures.

    See the module docstring for the algorithm and sign convention. This
    function coordinates existing correlations/geometry/NTU components; it
    does not duplicate any of them.
    """
    # Deferred import: core.properties.adapters is itself one of the modules
    # that pulls in core.heat_transfer (see adapters.py), so importing it at
    # module level here would risk a circular-import deadlock depending on
    # which module a caller imports first. Deferring to call time (the same
    # pattern already used elsewhere in this codebase, e.g.
    # BareTubeHeatExchanger.simulate/.rate) avoids that entirely.
    from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props

    for name, value in (
        ("m_dot_inside", m_dot_inside),
        ("m_dot_outside", m_dot_outside),
        ("T_in_inside", T_in_inside),
        ("T_in_outside", T_in_outside),
        ("p_inside", p_inside),
        ("p_outside", p_outside),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")
    if not (0.0 < relaxation_factor <= 1.0):
        raise ValueError("relaxation_factor must be in (0, 1].")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")
    if wall_temperature_tolerance_K <= 0.0:
        raise ValueError("wall_temperature_tolerance_K must be positive.")
    if relative_alfa_tolerance <= 0.0:
        raise ValueError("relative_alfa_tolerance must be positive.")

    if flow_arrangement is None:
        flow_arrangement = hx.bundle.flow_arrangement

    bundle = hx.bundle
    tube = bundle.tube
    A_i = bundle.total_inner_area
    A_o = bundle.total_outer_area
    R_w = hx.tube_wall_resistance()
    flow_area_pass = bundle.internal_flow_area_per_pass
    D_h = bundle.internal_hydraulic_diameter
    frontal_area = bundle.frontal_flow_area
    D_o = float(getattr(tube, "D_o"))

    hot_is_inside = T_in_inside >= T_in_outside

    def _evaluate(
        T_out_inside: float,
        T_out_outside: float,
        T_wall_inside_prev: float | None,
        T_wall_outside_prev: float | None,
    ):
        T_mean_inside = mean_temperature(T_in_inside, T_out_inside)
        T_mean_outside = mean_temperature(T_in_outside, T_out_outside)

        props_bulk_inside = inside_provider.at(T=T_mean_inside, p=p_inside)
        props_bulk_outside = outside_provider.at(T=T_mean_outside, p=p_outside)

        props_wall_inside = (
            inside_provider.at(T=T_wall_inside_prev, p=p_inside)
            if T_wall_inside_prev is not None
            else None
        )
        props_wall_outside = (
            outside_provider.at(T=T_wall_outside_prev, p=p_outside)
            if T_wall_outside_prev is not None
            else None
        )

        C_inside = m_dot_inside * props_bulk_inside.cp
        C_outside = m_dot_outside * props_bulk_outside.cp

        step_warnings: list[ModelWarning] = []

        v_i, Re_i, Pr_i, alfa_i, internal_warnings = heat_transfer_coefficient_internal(
            m_dot=m_dot_inside,
            tube_inner_diameter=D_h,
            flow_area=flow_area_pass,
            props=to_internal_fluid_props(props_bulk_inside),
            T_bulk=T_mean_inside,
            T_wall=T_wall_inside_prev,
        )
        step_warnings.extend(internal_warnings)

        Pr_s = None
        if props_wall_outside is not None:
            Pr_s = outside_prandtl_number(
                props_wall_outside.cp, props_wall_outside.mu, props_wall_outside.k
            )

        v_o, Re_o, Pr_o, alfa_o, dp_o, outside_warnings, _euler = outside_flow_from_mass_flow(
            m_dot=m_dot_outside,
            frontal_area=frontal_area,
            tube_outer_diameter=D_o,
            tube_pitch_transverse=bundle.pitch_transverse,
            tube_pitch_longitudinal=bundle.pitch_longitudinal,
            layout=bundle.layout,
            n_rows=bundle.n_rows,
            n_tubes_per_row=bundle.n_tubes_per_row,
            props=to_outside_fluid_props(props_bulk_outside),
            Pr_s=Pr_s,
            euler_provider=euler_provider,
        )
        step_warnings.extend(outside_warnings)

        R_i = 1.0 / (alfa_i * A_i)
        R_o = 1.0 / (alfa_o * A_o)
        R_tot = R_i + R_w + R_o
        if R_tot <= 0.0 or not math.isfinite(R_tot):
            raise ValueError("thermal_iteration: invalid total thermal resistance.")
        UA = 1.0 / R_tot

        if hot_is_inside:
            hot_stream = SensibleHeatStream(C=C_inside, T_in=T_in_inside)
            cold_stream = SensibleHeatStream(C=C_outside, T_in=T_in_outside)
        else:
            hot_stream = SensibleHeatStream(C=C_outside, T_in=T_in_outside)
            cold_stream = SensibleHeatStream(C=C_inside, T_in=T_in_inside)

        eps = effectiveness_ntu(
            C_hot=hot_stream.capacity_rate(),
            C_cold=cold_stream.capacity_rate(),
            UA=UA,
            flow_arrangement=flow_arrangement,
            C_inside=C_inside,
            C_outside=C_outside,
        )
        Q, T_hot_out, T_cold_out = heat_duty_from_effectiveness(eps, hot_stream, cold_stream)

        if hot_is_inside:
            T_out_inside_calc, T_out_outside_calc = T_hot_out, T_cold_out
        else:
            T_out_outside_calc, T_out_inside_calc = T_hot_out, T_cold_out

        # Wall-temperature split: use the *mean-bulk-consistent* heat rate
        # UA*(T_mean_inside - T_mean_outside) rather than the actual duty Q
        # (from eps-NTU on the inlet temperatures) to apportion the resistance
        # network. This is a resistance-weighted split of the mean bulk
        # temperature difference across R_i/R_w/R_o and guarantees:
        #   - both wall temperatures always lie within [T_mean_inside,
        #     T_mean_outside] (whichever order), for any positive resistances,
        #   - the two wall-surface temperatures collapse together as the wall
        #     resistance R_w -> 0 (no wall conduction resistance),
        # independent of whichever side happens to run hotter.
        q_inside_to_outside = UA * (T_mean_inside - T_mean_outside)

        T_wall_inside_calc = T_mean_inside - q_inside_to_outside * R_i
        T_wall_outside_calc = T_mean_outside + q_inside_to_outside * R_o

        return {
            "T_mean_inside": T_mean_inside,
            "T_mean_outside": T_mean_outside,
            "props_bulk_inside": props_bulk_inside,
            "props_bulk_outside": props_bulk_outside,
            "props_wall_inside": props_wall_inside,
            "props_wall_outside": props_wall_outside,
            "alfa_i": alfa_i,
            "alfa_o": alfa_o,
            "UA": UA,
            "T_out_inside_calc": T_out_inside_calc,
            "T_out_outside_calc": T_out_outside_calc,
            "T_wall_inside_calc": T_wall_inside_calc,
            "T_wall_outside_calc": T_wall_outside_calc,
            "warnings": step_warnings,
        }

    # --- Iteration ------------------------------------------------------
    T_out_inside = T_in_inside
    T_out_outside = T_in_outside
    T_wall_inside: float | None = None
    T_wall_outside: float | None = None
    alfa_i_prev: float | None = None
    alfa_o_prev: float | None = None

    converged = False
    iterations = 0
    residual = math.inf
    all_warnings: list[ModelWarning] = []

    for iteration in range(1, max_iterations + 1):
        iterations = iteration

        step = _evaluate(T_out_inside, T_out_outside, T_wall_inside, T_wall_outside)
        all_warnings.extend(step["warnings"])

        T_out_inside_new = T_out_inside + relaxation_factor * (
            step["T_out_inside_calc"] - T_out_inside
        )
        T_out_outside_new = T_out_outside + relaxation_factor * (
            step["T_out_outside_calc"] - T_out_outside
        )

        if T_wall_inside is None:
            T_wall_inside_new = step["T_wall_inside_calc"]
            T_wall_outside_new = step["T_wall_outside_calc"]
            wall_residual = math.inf
        else:
            T_wall_inside_new = T_wall_inside + relaxation_factor * (
                step["T_wall_inside_calc"] - T_wall_inside
            )
            T_wall_outside_new = T_wall_outside + relaxation_factor * (
                step["T_wall_outside_calc"] - T_wall_outside
            )
            wall_residual = max(
                abs(T_wall_inside_new - T_wall_inside),
                abs(T_wall_outside_new - T_wall_outside),
            )

        alfa_residual = 0.0
        if alfa_i_prev is not None and alfa_o_prev is not None:
            alfa_residual = max(
                abs(step["alfa_i"] - alfa_i_prev) / max(abs(alfa_i_prev), 1e-9),
                abs(step["alfa_o"] - alfa_o_prev) / max(abs(alfa_o_prev), 1e-9),
            )

        T_out_inside, T_out_outside = T_out_inside_new, T_out_outside_new
        T_wall_inside, T_wall_outside = T_wall_inside_new, T_wall_outside_new
        alfa_i_prev, alfa_o_prev = step["alfa_i"], step["alfa_o"]
        residual = wall_residual

        if (
            iteration >= 2
            and wall_residual < wall_temperature_tolerance_K
            and alfa_residual < relative_alfa_tolerance
        ):
            converged = True
            break

    # Final self-consistent evaluation at the converged (or last) state.
    final = _evaluate(T_out_inside, T_out_outside, T_wall_inside, T_wall_outside)
    all_warnings.extend(final["warnings"])

    if not converged:
        all_warnings.append(
            make_warning(
                code="thermal_iteration_not_converged",
                message=(
                    "thermal_iteration: did not converge within "
                    f"max_iterations={iterations}. Last wall-temperature "
                    f"residual={residual:.3e} K. Returning the last iterate."
                ),
                source="thermal_iteration",
                severity="warning",
            )
        )

    T_lo = min(final["T_mean_inside"], final["T_mean_outside"])
    T_hi = max(final["T_mean_inside"], final["T_mean_outside"])
    tol = 1e-6 * max(abs(T_hi), 1.0)
    if not (T_lo - tol <= T_wall_inside <= T_hi + tol) or not (
        T_lo - tol <= T_wall_outside <= T_hi + tol
    ):
        all_warnings.append(
            make_warning(
                code="thermal_iteration_wall_temperature_out_of_bulk_range",
                message=(
                    "thermal_iteration: a wall temperature fell outside the "
                    f"[{T_lo:.3g}, {T_hi:.3g}] K bulk-temperature interval; "
                    "unexpected for an ordinary single-phase dry case -- "
                    "verify inputs."
                ),
                source="thermal_iteration",
                severity="warning",
            )
        )

    U = final["UA"] / A_o if A_o > 0.0 else math.nan

    return IterativeThermalState(
        inside_bulk_temperature=final["T_mean_inside"],
        outside_bulk_temperature=final["T_mean_outside"],
        inside_wall_temperature=T_wall_inside,
        outside_wall_temperature=T_wall_outside,
        inside_bulk_props=final["props_bulk_inside"],
        inside_wall_props=final["props_wall_inside"],
        outside_bulk_props=final["props_bulk_outside"],
        outside_wall_props=final["props_wall_outside"],
        alfa_i=final["alfa_i"],
        alfa_o=final["alfa_o"],
        U=U,
        UA=final["UA"],
        iterations=iterations,
        converged=converged,
        residual=residual,
        warnings=tuple(all_warnings),
    )
