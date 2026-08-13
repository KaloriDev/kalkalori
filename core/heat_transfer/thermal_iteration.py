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
v0.5.3 - Iterative mean-property and wall-temperature thermal state.

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
8. Recalculate alfa_i (turbulent-gas wall-temperature correction,
   ``internal_flow.gas_wall_temperature_correction``, plus the finite
   heated-length correction, ``internal_flow.internal_length_correction``,
   applied once each) and alfa_o (outside wall Prandtl number ``Pr_s``,
   ``outside_flow.nusselt_zukauskas``) using the *separate* wall properties
   from step 7.
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

Heat-duty basis (v0.5.3)
-------------------------
This is the ONLY heat rate used anywhere in this module: it drives
``q_inside = UA*(T_mean_inside-T_mean_outside)/A_i``, ``q_outside`` (same
numerator over ``A_o``), and both wall temperatures via the cylindrical
resistance network above, consistently. It is deliberately *not* the
inlet-driven epsilon-NTU duty ``Q`` (also computed here, to advance the
outer T_out relaxation -- see ``heat_duty_from_effectiveness`` above -- but
never used for wall temperatures), and it is not a caller-supplied "required"
duty. Every caller of ``solve_iterative_thermal_state`` (currently
``BareTubeHeatExchanger.solve_thermal_state`` and ``core.models.rating``)
gets this same convention because they all funnel through this one
function; there is no separate wall-temperature calculation elsewhere.

Call path (v0.5.3)
------------------
Both ``core.models.rating`` and the default ``core.models.simulation`` path
call this function for their authoritative mean thermal state. The independent
endpoint-envelope probes reuse the same lower-level local correlation and
resistance evaluation, without rerunning either complete exchanger driver.

Scope
-----
0D model only (no axial/finned/segmented refinement). This module does not
compute pressure drop; see ``BareTubeHeatExchanger.solve`` for the combined
thermal + hydraulic single-pass kernel.

Future work (not implemented in v0.5.3)
----------------------------------------
A later patch could evaluate an optional multi-point / quadrature-based
averaging of tube-side properties and the wall correction along the heated
tube path (rather than a single mean-bulk/mean-wall evaluation), to better
approximate axial property variation without a full segmented solver. The
``ThermalIterationDiagnostics`` fields below (Nu_base/Nu_corrected/individual
correction factors) are reported per side precisely so such a refinement can
plug in additional quadrature points later without redesigning this result
structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.properties.common import FluidTransportProperties
from core.properties.fluids import PropertyProvider
from core.properties.averaging import mean_temperature

from core.heat_transfer.internal_flow import heat_transfer_coefficient_internal_diagnostics
from core.heat_transfer.outside_flow import (
    outside_flow_from_mass_flow,
    nusselt_zukauskas,
    prandtl_number as outside_prandtl_number,
)
from core.heat_transfer.ntu import effectiveness_ntu, heat_duty_from_effectiveness
from core.heat_transfer.streams import SensibleHeatStream

from core.common.warnings import ModelWarning, deduplicate_warnings, make_warning

if TYPE_CHECKING:
    from core.models.bare_tube import BareTubeHeatExchanger


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThermalIterationDiagnostics:
    """Raw converged correction/Nu/alfa breakdown for both sides.

    These are the *raw* converged values at the final self-consistent
    evaluation (see ``solve_iterative_thermal_state``'s final pass) -- not
    relaxed intermediates from the outer wall-temperature iteration.

    Internal (tube) side
    ---------------------
    ``inside_combined_correction == inside_length_correction *
    inside_wall_temperature_correction`` (see
    ``core.heat_transfer.internal_flow.InternalHeatTransferDiagnostics`` for
    the exact combination formula and references); ``inside_Nu_corrected ==
    inside_Nu_base * inside_combined_correction``; ``inside_alfa_corrected ==
    inside_Nu_corrected * inside_bulk_props.k / D_i``.

    Outside (bundle) side
    ----------------------
    ``outside_wall_property_correction`` is the Zukauskas ``(Pr/Pr_s)^0.25``
    wall-Prandtl correction (see ``core.heat_transfer.outside_flow.
    nusselt_zukauskas``); ``outside_Nu_corrected == outside_Nu_base *
    outside_wall_property_correction``. There is no finite-length correction
    on the outside (crossflow) side in the current model.

    The multi-zone pure-steam adapter uses the inside alfa fields for its
    resistance-consistent equivalent HTC. If its inside Nu fields are not
    ``None``, they are equivalent reporting diagnostics formed with the
    representative conductivity, not a local Shah or single-zone Nusselt
    number.
    """

    inside_Nu_base: float | None
    inside_Nu_corrected: float | None
    inside_length_correction: float | None
    inside_wall_temperature_correction: float | None
    inside_combined_correction: float | None
    inside_alfa_base: float
    inside_alfa_corrected: float

    outside_Nu_base: float
    outside_Nu_corrected: float
    outside_wall_property_correction: float


@dataclass(frozen=True)
class IterativeThermalState:
    """Converged (or last-iterate) mean-property / wall-temperature state.

    For multi-zone pure steam, ``alfa_i`` is the resistance-consistent
    equivalent HTC used by the 0D wall-temperature approximation.
    """

    inside_bulk_temperature: float    # [K]
    outside_bulk_temperature: float   # [K]

    inside_wall_temperature: float    # [K] tube inner-surface temperature
    outside_wall_temperature: float   # [K] tube outer-surface temperature

    inside_bulk_props: FluidTransportProperties | None
    inside_wall_props: FluidTransportProperties | None

    outside_bulk_props: FluidTransportProperties
    outside_wall_props: FluidTransportProperties | None

    alfa_i: float   # [W/(m2*K)] wall/length-corrected
    alfa_o: float   # [W/(m2*K)] wall-corrected

    U: float    # [W/(m2*K)] referenced to outer area A_o
    UA: float   # [W/K]

    iterations: int
    converged: bool
    residual: float   # [K] last wall-temperature residual

    diagnostics: ThermalIterationDiagnostics

    # Property-provider identity, for engineering audit (which provider
    # implementation produced inside_bulk_props/outside_bulk_props).
    inside_provider_name: str = ""
    outside_provider_name: str = ""

    warnings: tuple[ModelWarning, ...] = ()


@dataclass(frozen=True)
class WallTemperatureProbe:
    """Independent 0D wall-temperature calculation at one bulk-state pair.

    A multi-zone steam probe carries the whole-exchanger equivalent ``alfa_i``;
    it is not a local coefficient for any one phase zone.
    """

    inside_bulk_temperature: float
    outside_bulk_temperature: float
    inside_wall_temperature: float
    outside_wall_temperature: float
    alfa_i: float
    alfa_o: float
    converged: bool
    iterations: int
    warnings: tuple[ModelWarning, ...] = ()
    inside_bulk_props: FluidTransportProperties | None = None
    inside_wall_props: FluidTransportProperties | None = None
    outside_bulk_props: FluidTransportProperties | None = None
    outside_wall_props: FluidTransportProperties | None = None
    inside_nusselt: float | None = None
    outside_nusselt: float = math.nan
    heat_rate_probe: float = math.nan
    residual: float = math.inf


@dataclass(frozen=True)
class WallTemperatureEnvelope:
    """Estimated wall-temperature extrema from four independent 0D probes."""

    inside_min: float
    inside_max: float
    outside_min: float
    outside_max: float
    probes: tuple[WallTemperatureProbe, ...]
    method: str = "0d_endpoint_envelope"
    warnings: tuple[ModelWarning, ...] = ()
    inside_mean: float = math.nan
    outside_mean: float = math.nan


@dataclass(frozen=True)
class _LocalWallEvaluation:
    inside_bulk_props: FluidTransportProperties
    inside_wall_props: FluidTransportProperties | None
    outside_bulk_props: FluidTransportProperties
    outside_wall_props: FluidTransportProperties | None
    alfa_i: float
    alfa_o: float
    inside_nusselt: float
    outside_nusselt: float
    inside_wall_temperature: float
    outside_wall_temperature: float
    heat_rate: float
    internal_diagnostics: object
    outside_nusselt_base: float
    outside_wall_property_correction: float
    warnings: tuple[ModelWarning, ...]


def _evaluate_local_wall_state(
    hx: "BareTubeHeatExchanger",
    *,
    m_dot_inside: float,
    m_dot_outside: float,
    inside_provider: PropertyProvider,
    outside_provider: PropertyProvider,
    inside_bulk_temperature: float,
    outside_bulk_temperature: float,
    p_inside: float,
    p_outside: float,
    inside_wall_temperature: float | None,
    outside_wall_temperature: float | None,
    euler_provider: str,
) -> _LocalWallEvaluation:
    """Evaluate correlations and the resistance split at one fixed bulk pair."""
    from core.geometry.finned_tube import CircularFinnedTube

    if isinstance(hx.bundle.tube, CircularFinnedTube):
        return _evaluate_local_wall_state_finned(
            hx,
            m_dot_inside=m_dot_inside,
            m_dot_outside=m_dot_outside,
            inside_provider=inside_provider,
            outside_provider=outside_provider,
            inside_bulk_temperature=inside_bulk_temperature,
            outside_bulk_temperature=outside_bulk_temperature,
            p_inside=p_inside,
            p_outside=p_outside,
            inside_wall_temperature=inside_wall_temperature,
            euler_provider=euler_provider,
        )

    from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props

    bundle = hx.bundle
    tube = bundle.tube
    A_i = bundle.total_inner_area
    A_o = bundle.total_outer_area
    D_h = bundle.internal_hydraulic_diameter
    D_o = float(getattr(tube, "D_o"))

    bulk_i = inside_provider.at(T=inside_bulk_temperature, p=p_inside)
    bulk_o = outside_provider.at(T=outside_bulk_temperature, p=p_outside)
    warnings: list[ModelWarning] = []

    def wall_props(provider, temperature, pressure, side):
        if temperature is None:
            return None
        if not math.isfinite(temperature) or temperature <= 0.0:
            warnings.append(make_warning(
                code="thermal_iteration_invalid_wall_temperature",
                message=(f"thermal_iteration: computed {side} wall temperature "
                         f"{temperature!r} is not a finite, positive absolute temperature [K]."),
                source="thermal_iteration",
            ))
            return None
        return provider.at(T=temperature, p=pressure)

    wall_i = wall_props(inside_provider, inside_wall_temperature, p_inside, "inside")
    wall_o = wall_props(outside_provider, outside_wall_temperature, p_outside, "outside")

    internal = heat_transfer_coefficient_internal_diagnostics(
        m_dot=m_dot_inside,
        tube_inner_diameter=D_h,
        flow_area=bundle.internal_flow_area_per_pass,
        props=to_internal_fluid_props(bulk_i),
        T_bulk=inside_bulk_temperature,
        T_wall=inside_wall_temperature if wall_i is not None else None,
        L_heated=float(getattr(tube, "length_effective")),
    )
    warnings.extend(internal.warnings)

    Pr_s = None
    if wall_o is not None:
        Pr_s = outside_prandtl_number(wall_o.cp, wall_o.mu, wall_o.k)

    _, Re_o, Pr_o, alfa_o, _, outside_warnings, _ = outside_flow_from_mass_flow(
        m_dot=m_dot_outside,
        frontal_area=bundle.frontal_flow_area,
        tube_outer_diameter=D_o,
        tube_pitch_transverse=bundle.pitch_transverse,
        tube_pitch_longitudinal=bundle.pitch_longitudinal,
        layout=bundle.layout,
        n_rows=bundle.n_rows,
        n_tubes_per_row=bundle.n_tubes_per_row,
        props=to_outside_fluid_props(bulk_o),
        Pr_s=Pr_s,
        euler_provider=euler_provider,
    )
    warnings.extend(outside_warnings)
    Nu_o_base = nusselt_zukauskas(Re_o, Pr_o, bundle.n_rows, Pr_s=None)
    Nu_o = nusselt_zukauskas(Re_o, Pr_o, bundle.n_rows, Pr_s=Pr_s) if Pr_s is not None else Nu_o_base

    alfa_i = internal.alfa_corrected
    R_i = 1.0 / (alfa_i * A_i)
    R_o = 1.0 / (alfa_o * A_o)
    R_total = R_i + hx.tube_wall_resistance() + R_o
    if not math.isfinite(R_total) or R_total <= 0.0:
        raise ValueError("thermal_iteration: invalid total thermal resistance.")
    heat_rate = (inside_bulk_temperature - outside_bulk_temperature) / R_total

    return _LocalWallEvaluation(
        inside_bulk_props=bulk_i,
        inside_wall_props=wall_i,
        outside_bulk_props=bulk_o,
        outside_wall_props=wall_o,
        alfa_i=alfa_i,
        alfa_o=alfa_o,
        inside_nusselt=internal.Nu_corrected,
        outside_nusselt=Nu_o,
        inside_wall_temperature=inside_bulk_temperature - heat_rate * R_i,
        outside_wall_temperature=outside_bulk_temperature + heat_rate * R_o,
        heat_rate=heat_rate,
        internal_diagnostics=internal,
        outside_nusselt_base=Nu_o_base,
        outside_wall_property_correction=Nu_o / Nu_o_base if Nu_o_base != 0.0 else 1.0,
        warnings=tuple(warnings),
    )


def _evaluate_local_wall_state_finned(
    hx: "BareTubeHeatExchanger",
    *,
    m_dot_inside: float,
    m_dot_outside: float,
    inside_provider: PropertyProvider,
    outside_provider: PropertyProvider,
    inside_bulk_temperature: float,
    outside_bulk_temperature: float,
    p_inside: float,
    p_outside: float,
    inside_wall_temperature: float | None,
    euler_provider: str,
) -> _LocalWallEvaluation:
    """Circular-finned-tube counterpart of ``_evaluate_local_wall_state``.

    Tube-side (internal) convection is identical to the bare-tube path
    (the core tube's internal geometry/hydraulics are unchanged by fins).
    Outside convection uses Briggs & Young (1963)
    (``core.heat_transfer.outside_flow_finned``); the physical HTC is
    then combined with fin efficiency and the root/contact conduction
    path via ``core.heat_transfer.finned_tube_resistance``, and reported
    back as a single equivalent ``alfa_o`` on the gross outside-area
    basis so the surrounding resistance/UA/wall-temperature formulas
    below (identical to the bare-tube path) are correct without any
    further finned-specific change. There is no documented outside
    wall-Prandtl correction for Briggs & Young (1963), so (unlike the
    Zukauskas path) no outside wall-property correction is applied here.
    """
    from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props
    from core.heat_transfer.outside_flow_finned import (
        finned_outside_flow_from_mass_flow,
        resolve_finned_euler_provider,
    )
    from core.heat_transfer.finned_tube_resistance import build_finned_tube_resistance_network

    bundle = hx.bundle
    tube = bundle.tube
    A_i = bundle.total_inner_area
    A_o = bundle.total_outer_area

    bulk_i = inside_provider.at(T=inside_bulk_temperature, p=p_inside)
    bulk_o = outside_provider.at(T=outside_bulk_temperature, p=p_outside)
    warnings: list[ModelWarning] = []

    internal = heat_transfer_coefficient_internal_diagnostics(
        m_dot=m_dot_inside,
        tube_inner_diameter=bundle.internal_hydraulic_diameter,
        flow_area=bundle.internal_flow_area_per_pass,
        props=to_internal_fluid_props(bulk_i),
        T_bulk=inside_bulk_temperature,
        T_wall=inside_wall_temperature,
        L_heated=float(getattr(tube, "length_effective")),
    )
    warnings.extend(internal.warnings)

    finned_result = finned_outside_flow_from_mass_flow(
        m_dot=m_dot_outside,
        frontal_area=bundle.frontal_flow_area,
        tube=tube,
        tube_pitch_transverse=bundle.pitch_transverse,
        tube_pitch_longitudinal=bundle.pitch_longitudinal,
        layout=bundle.layout,
        n_rows=bundle.n_rows,
        n_tubes_per_row=bundle.n_tubes_per_row,
        props=to_outside_fluid_props(bulk_o),
        euler_provider=resolve_finned_euler_provider(euler_provider),
        calculate_pressure_drop=False,
    )
    warnings.extend(finned_result.warnings)

    alfa_i = internal.alfa_corrected
    network, network_warnings = build_finned_tube_resistance_network(
        tube,
        n_tubes=bundle.n_tubes_total,
        alfa_i=alfa_i,
        alfa_o_physical=finned_result.alfa_o_physical,
    )
    warnings.extend(network_warnings)

    alfa_o = network.alfa_o_gross_basis
    R_i = 1.0 / (alfa_i * A_i)
    R_o = 1.0 / (alfa_o * A_o)
    R_total = R_i + hx.tube_wall_resistance() + R_o
    if not math.isfinite(R_total) or R_total <= 0.0:
        raise ValueError("thermal_iteration: invalid total thermal resistance.")
    heat_rate = (inside_bulk_temperature - outside_bulk_temperature) / R_total

    return _LocalWallEvaluation(
        inside_bulk_props=bulk_i,
        inside_wall_props=None,
        outside_bulk_props=bulk_o,
        outside_wall_props=None,
        alfa_i=alfa_i,
        alfa_o=alfa_o,
        inside_nusselt=internal.Nu_corrected,
        outside_nusselt=finned_result.Nu,
        inside_wall_temperature=inside_bulk_temperature - heat_rate * R_i,
        outside_wall_temperature=outside_bulk_temperature + heat_rate * R_o,
        heat_rate=heat_rate,
        internal_diagnostics=internal,
        outside_nusselt_base=finned_result.Nu,
        outside_wall_property_correction=1.0,
        warnings=tuple(warnings),
    )


def _solve_wall_temperature_probe(
    hx: "BareTubeHeatExchanger",
    *,
    m_dot_inside: float,
    m_dot_outside: float,
    inside_provider: PropertyProvider,
    outside_provider: PropertyProvider,
    inside_bulk_temperature: float,
    outside_bulk_temperature: float,
    p_inside: float,
    p_outside: float,
    euler_provider: str = "zukauskas",
    max_iterations: int = 25,
    wall_temperature_tolerance_K: float = 0.05,
    relative_alfa_tolerance: float = 1e-3,
    relaxation_factor: float = 0.5,
) -> WallTemperatureProbe:
    wall_i = wall_o = None
    alfa_i_prev = alfa_o_prev = None
    all_warnings: list[ModelWarning] = []
    residual = math.inf
    converged = False
    evaluation = None

    for iteration in range(1, max_iterations + 1):
        evaluation = _evaluate_local_wall_state(
            hx,
            m_dot_inside=m_dot_inside, m_dot_outside=m_dot_outside,
            inside_provider=inside_provider, outside_provider=outside_provider,
            inside_bulk_temperature=inside_bulk_temperature,
            outside_bulk_temperature=outside_bulk_temperature,
            p_inside=p_inside, p_outside=p_outside,
            inside_wall_temperature=wall_i, outside_wall_temperature=wall_o,
            euler_provider=euler_provider,
        )
        all_warnings.extend(evaluation.warnings)
        target_i = evaluation.inside_wall_temperature
        target_o = evaluation.outside_wall_temperature
        if wall_i is None:
            new_i, new_o = target_i, target_o
            wall_residual = math.inf
        else:
            new_i = wall_i + relaxation_factor * (target_i - wall_i)
            new_o = wall_o + relaxation_factor * (target_o - wall_o)
            wall_residual = max(abs(new_i - wall_i), abs(new_o - wall_o))
        alfa_residual = math.inf
        if alfa_i_prev is not None:
            alfa_residual = max(
                abs(evaluation.alfa_i - alfa_i_prev) / max(abs(alfa_i_prev), 1e-9),
                abs(evaluation.alfa_o - alfa_o_prev) / max(abs(alfa_o_prev), 1e-9),
            )
        wall_i, wall_o = new_i, new_o
        alfa_i_prev, alfa_o_prev = evaluation.alfa_i, evaluation.alfa_o
        residual = wall_residual
        if iteration >= 2 and wall_residual < wall_temperature_tolerance_K and alfa_residual < relative_alfa_tolerance:
            converged = True
            break

    assert evaluation is not None
    final = _evaluate_local_wall_state(
        hx,
        m_dot_inside=m_dot_inside, m_dot_outside=m_dot_outside,
        inside_provider=inside_provider, outside_provider=outside_provider,
        inside_bulk_temperature=inside_bulk_temperature,
        outside_bulk_temperature=outside_bulk_temperature,
        p_inside=p_inside, p_outside=p_outside,
        inside_wall_temperature=wall_i, outside_wall_temperature=wall_o,
        euler_provider=euler_provider,
    )
    all_warnings.extend(final.warnings)
    wall_i, wall_o = final.inside_wall_temperature, final.outside_wall_temperature

    finite = math.isfinite(wall_i) and math.isfinite(wall_o)
    tol = 1e-6 * max(abs(inside_bulk_temperature), abs(outside_bulk_temperature), 1.0)
    if final.heat_rate >= 0.0:
        ordered = inside_bulk_temperature + tol >= wall_i >= wall_o - tol >= outside_bulk_temperature - 2 * tol
    else:
        ordered = outside_bulk_temperature + tol >= wall_o >= wall_i - tol >= inside_bulk_temperature - 2 * tol
    if not finite or not ordered:
        converged = False
        all_warnings.append(make_warning(
            code="wall_temperature_probe_physical_ordering_violation",
            message="A wall-temperature probe produced non-finite or physically misordered surface temperatures.",
            source="thermal_iteration",
        ))
    if not converged:
        all_warnings.append(make_warning(
            code="wall_temperature_probe_not_converged",
            message=(f"The 0D wall-temperature probe at inside={inside_bulk_temperature:.6g} K, "
                     f"outside={outside_bulk_temperature:.6g} K did not converge."),
            source="thermal_iteration",
        ))

    return WallTemperatureProbe(
        inside_bulk_temperature=inside_bulk_temperature,
        outside_bulk_temperature=outside_bulk_temperature,
        inside_wall_temperature=wall_i,
        outside_wall_temperature=wall_o,
        alfa_i=final.alfa_i,
        alfa_o=final.alfa_o,
        converged=converged,
        iterations=iteration,
        warnings=tuple(all_warnings),
        inside_bulk_props=final.inside_bulk_props,
        inside_wall_props=final.inside_wall_props,
        outside_bulk_props=final.outside_bulk_props,
        outside_wall_props=final.outside_wall_props,
        inside_nusselt=final.inside_nusselt,
        outside_nusselt=final.outside_nusselt,
        heat_rate_probe=final.heat_rate,
        residual=residual,
    )


def estimate_wall_temperature_envelope(
    hx: "BareTubeHeatExchanger",
    *,
    m_dot_inside: float,
    m_dot_outside: float,
    inside_provider: PropertyProvider,
    outside_provider: PropertyProvider,
    inside_inlet_temperature: float,
    inside_outlet_temperature: float,
    outside_inlet_temperature: float,
    outside_outlet_temperature: float,
    p_inside: float,
    p_outside: float,
    euler_provider: str = "zukauskas",
    max_iterations: int = 25,
    wall_temperature_tolerance_K: float = 0.05,
    relative_alfa_tolerance: float = 1e-3,
    relaxation_factor: float = 0.5,
) -> WallTemperatureEnvelope:
    """Estimate a conservative 0D endpoint envelope; no spatial coupling is implied."""
    pairs = (
        (inside_inlet_temperature, outside_inlet_temperature),
        (inside_inlet_temperature, outside_outlet_temperature),
        (inside_outlet_temperature, outside_inlet_temperature),
        (inside_outlet_temperature, outside_outlet_temperature),
    )
    probes_list: list[WallTemperatureProbe] = []
    for T_i, T_o in pairs:
        try:
            probe = _solve_wall_temperature_probe(
                hx,
                m_dot_inside=m_dot_inside, m_dot_outside=m_dot_outside,
                inside_provider=inside_provider, outside_provider=outside_provider,
                inside_bulk_temperature=T_i, outside_bulk_temperature=T_o,
                p_inside=p_inside, p_outside=p_outside,
                euler_provider=euler_provider, max_iterations=max_iterations,
                wall_temperature_tolerance_K=wall_temperature_tolerance_K,
                relative_alfa_tolerance=relative_alfa_tolerance,
                relaxation_factor=relaxation_factor,
            )
        except Exception as exc:
            warning = make_warning(
                code="wall_temperature_probe_not_converged",
                message=(f"The 0D wall-temperature probe at inside={T_i:.6g} K, "
                         f"outside={T_o:.6g} K failed: {exc}"),
                source="thermal_iteration",
            )
            probe = WallTemperatureProbe(
                inside_bulk_temperature=T_i,
                outside_bulk_temperature=T_o,
                inside_wall_temperature=math.nan,
                outside_wall_temperature=math.nan,
                alfa_i=math.nan,
                alfa_o=math.nan,
                converged=False,
                iterations=0,
                warnings=(warning,),
            )
        probes_list.append(probe)
    probes = tuple(probes_list)
    valid = tuple(p for p in probes if p.converged)
    warnings = [make_warning(
        code="wall_temperature_envelope_0d_estimate",
        message=("Wall-temperature minimum and maximum are estimated from a 0D inlet/outlet "
                 "endpoint envelope. They are not local extrema from a spatially segmented exchanger model."),
        source="thermal_iteration",
        severity="info",
    )]
    for probe in probes:
        warnings.extend(probe.warnings)
    if len(valid) < 4:
        warnings.append(make_warning(
            code="wall_temperature_envelope_incomplete",
            message=f"Only {len(valid)} of 4 wall-temperature endpoint probes converged.",
            source="thermal_iteration",
        ))
    inside_min = min((p.inside_wall_temperature for p in valid), default=math.nan)
    inside_max = max((p.inside_wall_temperature for p in valid), default=math.nan)
    outside_min = min((p.outside_wall_temperature for p in valid), default=math.nan)
    outside_max = max((p.outside_wall_temperature for p in valid), default=math.nan)
    return WallTemperatureEnvelope(
        inside_min=inside_min,
        inside_max=inside_max,
        outside_min=outside_min,
        outside_max=outside_max,
        probes=probes,
        warnings=tuple(warnings),
        inside_mean=0.5 * (inside_min + inside_max),
        outside_mean=0.5 * (outside_min + outside_max),
    )


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
    A_i = bundle.total_inner_area
    A_o = bundle.total_outer_area
    R_w = hx.tube_wall_resistance()
    hot_is_inside = T_in_inside >= T_in_outside

    def _evaluate(
        T_out_inside: float,
        T_out_outside: float,
        T_wall_inside_prev: float | None,
        T_wall_outside_prev: float | None,
    ):
        T_mean_inside = mean_temperature(T_in_inside, T_out_inside)
        T_mean_outside = mean_temperature(T_in_outside, T_out_outside)

        local = _evaluate_local_wall_state(
            hx,
            m_dot_inside=m_dot_inside, m_dot_outside=m_dot_outside,
            inside_provider=inside_provider, outside_provider=outside_provider,
            inside_bulk_temperature=T_mean_inside,
            outside_bulk_temperature=T_mean_outside,
            p_inside=p_inside, p_outside=p_outside,
            inside_wall_temperature=T_wall_inside_prev,
            outside_wall_temperature=T_wall_outside_prev,
            euler_provider=euler_provider,
        )
        props_bulk_inside = local.inside_bulk_props
        props_bulk_outside = local.outside_bulk_props
        C_inside = m_dot_inside * props_bulk_inside.cp
        C_outside = m_dot_outside * props_bulk_outside.cp
        UA = 1.0 / (
            1.0 / (local.alfa_i * A_i) + R_w + 1.0 / (local.alfa_o * A_o)
        )

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

        return {
            "T_mean_inside": T_mean_inside,
            "T_mean_outside": T_mean_outside,
            "props_bulk_inside": props_bulk_inside,
            "props_bulk_outside": props_bulk_outside,
            "props_wall_inside": local.inside_wall_props,
            "props_wall_outside": local.outside_wall_props,
            "alfa_i": local.alfa_i,
            "alfa_o": local.alfa_o,
            "UA": UA,
            "T_out_inside_calc": T_out_inside_calc,
            "T_out_outside_calc": T_out_outside_calc,
            "T_wall_inside_calc": local.inside_wall_temperature,
            "T_wall_outside_calc": local.outside_wall_temperature,
            "internal_diag": local.internal_diagnostics,
            "outside_Nu_base": local.outside_nusselt_base,
            "outside_Nu_corrected": local.outside_nusselt,
            "outside_wall_property_correction": local.outside_wall_property_correction,
            "warnings": list(local.warnings),
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

    # Final resistance-reconstruction self-check: UA must equal
    # 1/(R_i + R_w + R_o) rebuilt from the final alfa_i/alfa_o -- this holds
    # by construction (see R_tot above) but is re-verified here as a
    # standing diagnostic against future refactors silently decoupling the
    # two calculations.
    R_i_final = 1.0 / (final["alfa_i"] * A_i)
    R_o_final = 1.0 / (final["alfa_o"] * A_o)
    UA_reconstructed = 1.0 / (R_i_final + R_w + R_o_final)
    if not math.isclose(UA_reconstructed, final["UA"], rel_tol=1e-9, abs_tol=1e-9):
        all_warnings.append(
            make_warning(
                code="thermal_iteration_ua_reconstruction_mismatch",
                message=(
                    "thermal_iteration: UA reconstructed from final alfa_i/"
                    f"alfa_o/R_wall ({UA_reconstructed:.6g} W/K) does not "
                    f"match the reported UA ({final['UA']:.6g} W/K)."
                ),
                source="thermal_iteration",
                severity="critical",
            )
        )

    internal_diag = final["internal_diag"]
    diagnostics = ThermalIterationDiagnostics(
        inside_Nu_base=internal_diag.Nu_base,
        inside_Nu_corrected=internal_diag.Nu_corrected,
        inside_length_correction=internal_diag.length_correction,
        inside_wall_temperature_correction=internal_diag.wall_temperature_correction,
        inside_combined_correction=internal_diag.combined_correction,
        inside_alfa_base=internal_diag.alfa_base,
        inside_alfa_corrected=internal_diag.alfa_corrected,
        outside_Nu_base=final["outside_Nu_base"],
        outside_Nu_corrected=final["outside_Nu_corrected"],
        outside_wall_property_correction=final["outside_wall_property_correction"],
    )

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
        diagnostics=diagnostics,
        inside_provider_name=type(inside_provider).__name__,
        outside_provider_name=type(outside_provider).__name__,
        warnings=deduplicate_warnings(all_warnings),
    )
