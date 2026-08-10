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
# All calculations use SI units:
# - m_dot [kg/s], L [m], D [m], A [m^2]
# - rho [kg/m^3], mu [Pa*s]
# - dp [Pa]

"""
Tube-side (internal) pressure drop correlations for smooth circular tubes.

Production methodology (v0.5.5)
--------------------------------
The production tube-bundle result evaluates bulk properties at inlet,
midpoint, and outlet.  Distributed friction uses three-point Simpson
integration of ``f/rho`` at constant mass flux, and the signed acceleration
term follows the one-dimensional momentum balance.  The result covers only
the straight tube bundle; local nozzle, chamber, tube-sheet, return, bend,
header, and collector losses are intentionally excluded.

Literature references
---------------------
- Darcy–Weisbach equation and friction factor:
  White, F. M., "Fluid Mechanics"
  Incropera et al., "Fundamentals of Heat and Mass Transfer" (hydrodynamics basics)

- Minor losses (K coefficients) methodology:
  Idelchik, I. E., "Handbook of Hydraulic Resistance"
  Crane Co., "Flow of Fluids Through Valves, Fittings, and Pipe" (TP-410)

Notes
-----
- Smooth circular tube friction uses the existing Darcy friction-factor
  implementation.
- The compatibility component helper at the end of this module is not used
  by the exchanger production path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from core.common.warnings import ModelWarning, make_warning
from core.geometry.bundle import TubePathType
from core.pressure_drop.screens import (
    TubeSheetEntranceType,
    TubeSheetExitType,
    calculate_tube_sheet_entrance_loss,
    calculate_tube_sheet_exit_loss,
)
from core.pressure_drop.straight_sections import (
    darcy_friction_factor,
    darcy_friction_factor_method,
    friction_factor_smooth,
)

@dataclass(frozen=True)
class FluidProps:
    rho: float  # [kg/m^3]
    mu: float   # [Pa*s]


HydraulicPosition = Literal["inlet", "midpoint", "outlet"]


@dataclass(frozen=True)
class TubeSideHydraulicPoint:
    """Bulk hydraulic state at one of the three tube-side evaluation points.

    The point is also the public fluid-property diagnostic for its position.
    ``T``/``p`` and ``rho``/``cp``/``mu``/``k`` are convenience aliases for
    the already stored temperature, pressure, and transport-property object;
    no second property evaluation or duplicate state is created.

    ``friction_factor`` is the Darcy friction factor from
    ``core.pressure_drop.straight_sections.darcy_friction_factor``: the
    existing smooth-tube (Petukhov) value when no positive relative
    roughness is specified, or the Colebrook-White rough-tube value
    otherwise (v0.5.6). ``friction_factor_method`` is a stable diagnostic
    identifier for which branch was used: ``"laminar_64_over_re"``,
    ``"petukhov_smooth"``, or ``"colebrook_white"``.
    """

    position: HydraulicPosition
    temperature: float
    pressure: float
    props: FluidTransportProperties
    enthalpy: float | None
    mass_flux: float
    velocity: float
    reynolds: float
    friction_factor: float
    friction_factor_method: str
    dynamic_pressure: float
    prandtl: float

    @property
    def Pr(self) -> float:
        return self.prandtl

    @property
    def T(self) -> float:
        return self.temperature

    @property
    def p(self) -> float:
        return self.pressure

    @property
    def rho(self) -> float:
        return self.props.rho

    @property
    def cp(self) -> float:
        return self.props.cp

    @property
    def mu(self) -> float:
        return self.props.mu

    @property
    def k(self) -> float:
        return self.props.k

    @property
    def darcy_friction_factor(self) -> float:
        return self.friction_factor

    @property
    def h(self) -> float | None:
        return self.enthalpy


@dataclass(frozen=True)
class TubePassBoundaryHydraulicState:
    """Hydraulic state at one tube-side pass boundary (v0.5.6).

    For ``n_tube_passes = n``, there are ``n + 1`` boundary states, indexed
    ``0..n``: boundary 0 is the exchanger tube-side inlet (entrance to pass
    1); boundary ``n`` is the exchanger tube-side outlet (exit from the
    final pass); for straight tube paths, boundary ``j`` (``0 < j < n``) is
    simultaneously pass ``j``'s exit and pass ``j + 1``'s entrance. For
    U-tube paths, only boundaries 0 and ``n`` produce tube-sheet entrance/
    exit losses; intermediate boundaries remain available for a future
    U-bend calculation but are not used for entrance/exit losses.
    """

    boundary_index: int

    temperature: float
    pressure: float
    props: FluidTransportProperties

    flow_area_per_pass: float
    mass_flux: float
    velocity: float
    dynamic_pressure: float
    reynolds: float


@dataclass(frozen=True)
class TubeEndPressureDropResult:
    """A single tube-sheet entrance or exit pressure-drop component (v0.5.6).

    ``pass_index`` is 1-based for straight tube paths; it is ``None`` for
    U-tube paths, where a single entrance/exit spans the complete
    continuous tube path (``component_id`` values ``tube_bundle_entrance``/
    ``tube_bundle_exit``).
    """

    component_id: str
    component_type: str

    pass_index: int | None
    boundary_index: int

    loss_coefficient: float
    reference_velocity: float
    reference_dynamic_pressure: float

    pressure_drop: float
    method: str

    warnings: tuple[ModelWarning, ...]


@dataclass(frozen=True)
class TubeBundleHydraulicResult:
    """Complete tube-bundle pressure-drop result (v0.5.6).

    ``dp_tube_bundle`` is the sum of distributed straight-tube friction,
    signed tube-side acceleration, and tube-sheet entrance/exit losses:

        dp_straight_tubes = dp_straight_tube_friction + dp_straight_tube_acceleration
        dp_tube_bundle = dp_straight_tubes + dp_tube_entrances + dp_tube_exits

    ``dp_friction``/``dp_acceleration`` remain as compatibility aliases for
    ``dp_straight_tube_friction``/``dp_straight_tube_acceleration``.
    ``dp_straight_tubes`` is the pre-v0.5.6 scope of ``dp_tube_bundle``
    (friction + acceleration only, excluding tube-sheet entrance/exit).

    Inlet nozzle, outlet nozzle, inlet/outlet/return chambers, headers,
    collectors, external pipework, and direction-change (bend/elbow)
    losses remain outside this result; they belong to the optional,
    explicitly invoked pressure-drop path architecture (see
    ``core.pressure_drop.flow_path``).
    """

    inlet: TubeSideHydraulicPoint
    midpoint: TubeSideHydraulicPoint
    outlet: TubeSideHydraulicPoint
    midpoint_method: str
    flow_area_per_pass: float
    mass_flux: float
    hydraulic_diameter: float
    hydraulic_length_total: float
    mean_f_over_rho: float

    roughness_inner: float | None
    relative_roughness_inner: float | None

    dp_straight_tube_friction: float
    dp_straight_tube_acceleration: float

    dp_tube_entrances: float
    dp_tube_exits: float

    dp_tube_bundle: float

    tube_path_type: TubePathType
    entrance_count: int
    exit_count: int
    pass_boundary_method: str
    pass_boundary_states: tuple[TubePassBoundaryHydraulicState, ...]
    entrance_results: tuple[TubeEndPressureDropResult, ...]
    exit_results: tuple[TubeEndPressureDropResult, ...]

    warnings: tuple[ModelWarning, ...]

    @property
    def dp_straight_tubes(self) -> float:
        """Diagnostic/compatibility total for the pre-v0.5.6 scope of
        ``dp_tube_bundle``: straight-tube friction plus acceleration only,
        excluding tube-sheet entrance/exit losses."""
        return self.dp_straight_tube_friction + self.dp_straight_tube_acceleration

    @property
    def dp_friction(self) -> float:
        """Compatibility alias for dp_straight_tube_friction."""
        return self.dp_straight_tube_friction

    @property
    def dp_acceleration(self) -> float:
        """Compatibility alias for dp_straight_tube_acceleration."""
        return self.dp_straight_tube_acceleration

    @property
    def dp_total(self) -> float:
        """Compatibility alias for the complete tube-bundle pressure change.

        Since v0.5.6 this includes tube-sheet entrance/exit losses; see
        ``dp_straight_tubes`` for the pre-v0.5.6 (friction + acceleration
        only) scope.
        """
        return self.dp_tube_bundle

    @property
    def rho_in(self) -> float:
        return self.inlet.props.rho

    @property
    def rho_mid(self) -> float:
        return self.midpoint.props.rho

    @property
    def rho_out(self) -> float:
        return self.outlet.props.rho

    @property
    def mu_in(self) -> float:
        return self.inlet.props.mu

    @property
    def mu_mid(self) -> float:
        return self.midpoint.props.mu

    @property
    def mu_out(self) -> float:
        return self.outlet.props.mu

    @property
    def velocity_in(self) -> float:
        return self.inlet.velocity

    @property
    def velocity_mid(self) -> float:
        return self.midpoint.velocity

    @property
    def velocity_out(self) -> float:
        return self.outlet.velocity

    @property
    def dynamic_pressure_in(self) -> float:
        return self.inlet.dynamic_pressure

    @property
    def dynamic_pressure_mid(self) -> float:
        return self.midpoint.dynamic_pressure

    @property
    def dynamic_pressure_out(self) -> float:
        return self.outlet.dynamic_pressure

    @property
    def Re_in(self) -> float:
        return self.inlet.reynolds

    @property
    def Re_mid(self) -> float:
        return self.midpoint.reynolds

    @property
    def Re_out(self) -> float:
        return self.outlet.reynolds

    @property
    def friction_factor_in(self) -> float:
        return self.inlet.friction_factor

    @property
    def friction_factor_mid(self) -> float:
        return self.midpoint.friction_factor

    @property
    def friction_factor_out(self) -> float:
        return self.outlet.friction_factor

    @property
    def friction_factor_method_in(self) -> str:
        return self.inlet.friction_factor_method

    @property
    def friction_factor_method_mid(self) -> str:
        return self.midpoint.friction_factor_method

    @property
    def friction_factor_method_out(self) -> str:
        return self.outlet.friction_factor_method

    @property
    def Pr_in(self) -> float:
        return self.inlet.prandtl

    @property
    def Pr_mid(self) -> float:
        return self.midpoint.prandtl

    @property
    def Pr_out(self) -> float:
        return self.outlet.prandtl


def calculate_tube_bundle_hydraulics(
    *,
    m_dot: float,
    flow_area_per_pass: float,
    hydraulic_diameter: float,
    hydraulic_length_total: float,
    n_tube_passes: int,
    tube_path_type: TubePathType = TubePathType.STRAIGHT,
    entrance_type: TubeSheetEntranceType = TubeSheetEntranceType.SHARP_EDGED,
    exit_type: TubeSheetExitType = TubeSheetExitType.NORMAL,
    roughness_inner: float | None = None,
    provider: Any | None = None,
    temperature_in: float | None = None,
    temperature_out: float | None = None,
    pressure: float | None = None,
    inlet_props: FluidTransportProperties | None = None,
    midpoint_props: FluidTransportProperties | None = None,
    outlet_props: FluidTransportProperties | None = None,
    m_dot_inlet: float | None = None,
    m_dot_midpoint: float | None = None,
    m_dot_outlet: float | None = None,
) -> TubeBundleHydraulicResult:
    """Calculate universal tube-bundle hydraulics (v0.5.6).

    Distributed straight-tube friction and signed acceleration continue to
    use the existing three-state (inlet/midpoint/outlet) Simpson
    integration. Tube-sheet entrance and exit losses are evaluated at
    ``n_tube_passes + 1`` pass-boundary states and applied according to
    ``tube_path_type``: one entrance and one exit per pass for
    ``STRAIGHT``, or a single entrance (boundary 0) and single exit
    (final boundary) for ``U_TUBE``. See ``core.pressure_drop.screens``
    for the entrance/exit correlations themselves.

    A provider is evaluated at inlet and outlet bulk temperatures.  When
    complete enthalpy data and a provider ``temperature_from_h_p`` capability
    are available, the midpoint and pass-boundary states are enthalpy-based.
    Otherwise the same arithmetic-temperature fallback is used for every
    provider.

    ``roughness_inner`` (absolute, [m]) is optional: ``None`` (default) or
    ``0.0`` preserve the existing hydraulically smooth friction-factor
    calculation exactly; a positive value is converted to
    ``relative_roughness = roughness_inner / hydraulic_diameter`` and used
    identically at the inlet, midpoint, and outlet points via
    ``core.pressure_drop.straight_sections.darcy_friction_factor``
    (Colebrook-White for turbulent flow; ``64/Re`` for laminar flow,
    independent of roughness). Only distributed straight-tube friction
    responds to roughness: acceleration, tube-sheet entrance/exit losses,
    and pass-boundary states are unaffected.

    The point-property arguments are a deterministic test/compatibility hook.
    Production exchanger paths pass a provider and temperatures.
    """
    _validate_hydraulic_input(
        m_dot=m_dot,
        flow_area_per_pass=flow_area_per_pass,
        hydraulic_diameter=hydraulic_diameter,
        hydraulic_length_total=hydraulic_length_total,
    )
    if n_tube_passes <= 0:
        raise ValueError(
            "tube_bundle_hydraulics_invalid_pass_count: n_tube_passes must be a positive integer."
        )
    if tube_path_type == TubePathType.U_TUBE and n_tube_passes < 2:
        raise ValueError(
            "tube_bundle_hydraulics_invalid_u_tube_pass_count: "
            "tube_path_type=U_TUBE requires at least two tube passes."
        )
    if roughness_inner is not None and (not math.isfinite(roughness_inner) or roughness_inner < 0.0):
        raise ValueError(
            "tube_bundle_hydraulics_invalid_roughness: roughness_inner must be "
            "None or a finite, non-negative value."
        )
    relative_roughness = (
        None if roughness_inner is None else roughness_inner / hydraulic_diameter
    )

    if provider is None:
        if inlet_props is None:
            raise ValueError(
                "provider or inlet_props must be supplied for tube-bundle hydraulics."
            )
        # Direct solve() compatibility still uses the new three-state engine;
        # a constant property object simply makes all three states identical.
        props_in, h_in = _transport_and_enthalpy(inlet_props)
        props_mid, h_mid = _transport_and_enthalpy(midpoint_props or inlet_props)
        props_out, h_out = _transport_and_enthalpy(outlet_props or inlet_props)
        T_in = 300.0 if temperature_in is None else float(temperature_in)
        T_out = T_in if temperature_out is None else float(temperature_out)
        T_mid = 0.5 * (T_in + T_out)
        p = 101325.0 if pressure is None else float(pressure)
        midpoint_method = "arithmetic_temperature"
        warnings: list[ModelWarning] = []
    else:
        T_in = _require_temperature(temperature_in, "temperature_in")
        T_out = _require_temperature(temperature_out, "temperature_out")
        p = _require_pressure(pressure)
        props_in, h_in = _evaluate_provider_state(provider, T_in, p)
        props_out, h_out = _evaluate_provider_state(provider, T_out, p)
        warnings = []

        T_mid = 0.5 * (T_in + T_out)
        midpoint_method = "arithmetic_temperature"
        if _finite_enthalpy(h_in) and _finite_enthalpy(h_out):
            h_mid_target = (float(h_in) + float(h_out)) / 2.0
            T_mid_enthalpy = _try_temperature_from_enthalpy(
                provider,
                h=h_mid_target,
                p=p,
            )
            if T_mid_enthalpy is not None:
                T_mid = T_mid_enthalpy
                midpoint_method = "enthalpy"
            else:
                warnings.append(_midpoint_fallback_warning("inversion is unavailable or failed"))
        else:
            warnings.append(_midpoint_fallback_warning("complete inlet/outlet enthalpy data are unavailable"))

        if midpoint_props is not None:
            props_mid, h_mid = _transport_and_enthalpy(midpoint_props)
        else:
            props_mid, h_mid = _evaluate_provider_state(provider, T_mid, p)

    # Optional gas-phase point flows are used by active inside wet-gas
    # condensation.  The legacy ``m_dot`` remains the representative
    # midpoint/reference flow and the unset path is numerically unchanged.
    m_dot_in = m_dot if m_dot_inlet is None else float(m_dot_inlet)
    m_dot_mid = m_dot if m_dot_midpoint is None else float(m_dot_midpoint)
    m_dot_out = m_dot if m_dot_outlet is None else float(m_dot_outlet)
    for name, value in (
        ("m_dot_inlet", m_dot_in),
        ("m_dot_midpoint", m_dot_mid),
        ("m_dot_outlet", m_dot_out),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite gas-phase mass flow [kg/s].")

    mass_flux = m_dot / flow_area_per_pass
    mass_flux_in = m_dot_in / flow_area_per_pass
    mass_flux_mid = m_dot_mid / flow_area_per_pass
    mass_flux_out = m_dot_out / flow_area_per_pass
    points = (
        _make_hydraulic_point(
            position="inlet", temperature=T_in, pressure=p, props=props_in,
            enthalpy=h_in, mass_flux=mass_flux_in, hydraulic_diameter=hydraulic_diameter,
            relative_roughness=relative_roughness,
        ),
        _make_hydraulic_point(
            position="midpoint", temperature=T_mid, pressure=p, props=props_mid,
            enthalpy=h_mid, mass_flux=mass_flux_mid, hydraulic_diameter=hydraulic_diameter,
            relative_roughness=relative_roughness,
        ),
        _make_hydraulic_point(
            position="outlet", temperature=T_out, pressure=p, props=props_out,
            enthalpy=h_out, mass_flux=mass_flux_out, hydraulic_diameter=hydraulic_diameter,
            relative_roughness=relative_roughness,
        ),
    )

    for point in points:
        if point.reynolds <= 4000.0:
            warnings.append(
                make_warning(
                    code="tube_bundle_hydraulics_reynolds_outside_range",
                    message=(
                        "tube_bundle_hydraulics: Reynolds number at the "
                        f"{point.position} state is {point.reynolds:.6g}; verify "
                        "the applicable friction-factor regime."
                    ),
                    source="tube_bundle_hydraulics",
                    severity="info",
                )
            )

    mean_f_over_rho = (
        points[0].friction_factor / points[0].props.rho
        + 4.0 * points[1].friction_factor / points[1].props.rho
        + points[2].friction_factor / points[2].props.rho
    ) / 6.0

    # Three-state 0D+ approximation of the distributed pressure-gradient
    # integral, not a spatially segmented solver.
    if mass_flux_in == mass_flux_mid == mass_flux_out == mass_flux:
        # Preserve the sensible-only/reference-flow calculation exactly.
        dp_straight_tube_friction = (
            (hydraulic_length_total / hydraulic_diameter)
            * (mass_flux**2 / 2.0)
            * mean_f_over_rho
        )
    else:
        mean_f_G2_over_rho = (
            points[0].friction_factor * mass_flux_in**2 / points[0].props.rho
            + 4.0 * points[1].friction_factor * mass_flux_mid**2 / points[1].props.rho
            + points[2].friction_factor * mass_flux_out**2 / points[2].props.rho
        ) / 6.0
        dp_straight_tube_friction = (
            hydraulic_length_total / hydraulic_diameter
        ) * mean_f_G2_over_rho / 2.0

    # One-dimensional momentum balance.  Pressure loss is positive, so a
    # density decrease gives a positive acceleration term and a density
    # increase gives pressure recovery.  Refs: White, Fluid Mechanics;
    # Idelchik, Handbook of Hydraulic Resistance; Crane TP-410.
    dp_straight_tube_acceleration = (
        mass_flux_out**2 / points[2].props.rho
        - mass_flux_in**2 / points[0].props.rho
    )
    dp_straight_tubes = dp_straight_tube_friction + dp_straight_tube_acceleration

    if not math.isfinite(dp_straight_tube_friction) or dp_straight_tube_friction < 0.0:
        raise ValueError(
            "tube_bundle_hydraulics_invalid_friction: distributed friction must be finite and non-negative."
        )
    if not math.isfinite(dp_straight_tube_acceleration) or not math.isfinite(dp_straight_tubes):
        raise ValueError("tube_bundle_hydraulics_nonfinite_state: pressure change is not finite.")
    if dp_straight_tubes < 0.0:
        warnings.append(
            make_warning(
                code="tube_bundle_hydraulics_negative_total_pressure_change",
                message=(
                    "tube_bundle_hydraulics: signed straight tube-bundle pressure "
                    f"change is negative ({dp_straight_tubes:.6g} Pa), representing "
                    "net pressure recovery after friction."
                ),
                source="tube_bundle_hydraulics",
                severity="warning",
            )
        )

    # --------------------------------------------------------------
    # Tube-sheet entrance and exit losses (v0.5.6).  Pass-boundary states
    # reuse the same provider-evaluation helpers as the inlet/midpoint/
    # outlet points above; entrance/exit calculations themselves are
    # delegated to core.pressure_drop.screens (stateless K*dynamic_pressure
    # models), applied here according to tube_path_type/n_tube_passes.
    # --------------------------------------------------------------
    boundary_states, pass_boundary_method, boundary_warnings = _build_tube_pass_boundary_states(
        n_tube_passes=n_tube_passes,
        mass_flux=mass_flux,
        mass_flux_inlet=mass_flux_in,
        mass_flux_outlet=mass_flux_out,
        variable_gas_flow=not (
            mass_flux_in == mass_flux_mid == mass_flux_out == mass_flux
        ),
        flow_area_per_pass=flow_area_per_pass,
        hydraulic_diameter=hydraulic_diameter,
        T_in=T_in, T_out=T_out, p=p,
        provider=provider,
        props_in=props_in, h_in=h_in,
        props_mid=props_mid,
        props_out=props_out, h_out=h_out,
    )
    warnings.extend(boundary_warnings)

    entrance_results, exit_results, dp_tube_entrances, dp_tube_exits = _apply_tube_sheet_entrances_and_exits(
        tube_path_type=tube_path_type,
        n_tube_passes=n_tube_passes,
        boundary_states=boundary_states,
        entrance_type=entrance_type,
        exit_type=exit_type,
    )

    dp_tube_bundle = dp_straight_tubes + dp_tube_entrances + dp_tube_exits
    if not math.isfinite(dp_tube_entrances) or dp_tube_entrances < 0.0:
        raise ValueError(
            "tube_bundle_hydraulics_invalid_entrance_loss: dp_tube_entrances must be finite and non-negative."
        )
    if not math.isfinite(dp_tube_exits) or dp_tube_exits < 0.0:
        raise ValueError(
            "tube_bundle_hydraulics_invalid_exit_loss: dp_tube_exits must be finite and non-negative."
        )

    return TubeBundleHydraulicResult(
        inlet=points[0], midpoint=points[1], outlet=points[2],
        midpoint_method=midpoint_method,
        flow_area_per_pass=float(flow_area_per_pass), mass_flux=mass_flux,
        hydraulic_diameter=float(hydraulic_diameter),
        hydraulic_length_total=float(hydraulic_length_total),
        mean_f_over_rho=mean_f_over_rho,
        roughness_inner=roughness_inner,
        relative_roughness_inner=relative_roughness,
        dp_straight_tube_friction=dp_straight_tube_friction,
        dp_straight_tube_acceleration=dp_straight_tube_acceleration,
        dp_tube_entrances=dp_tube_entrances,
        dp_tube_exits=dp_tube_exits,
        dp_tube_bundle=dp_tube_bundle,
        tube_path_type=tube_path_type,
        entrance_count=len(entrance_results),
        exit_count=len(exit_results),
        pass_boundary_method=pass_boundary_method,
        pass_boundary_states=boundary_states,
        entrance_results=entrance_results,
        exit_results=exit_results,
        warnings=_deduplicate_warnings(warnings),
    )


# Short public alias matching the terminology used by the exchanger result.
tube_bundle_hydraulics = calculate_tube_bundle_hydraulics


def _validate_hydraulic_input(
    *, m_dot: float, flow_area_per_pass: float,
    hydraulic_diameter: float, hydraulic_length_total: float,
) -> None:
    checks = (
        ("mass flow", m_dot, "tube_bundle_hydraulics_invalid_mass_flux", True),
        ("flow area", flow_area_per_pass, "tube_bundle_hydraulics_invalid_flow_area", True),
        ("hydraulic diameter", hydraulic_diameter, "tube_bundle_hydraulics_invalid_hydraulic_diameter", True),
        ("hydraulic length", hydraulic_length_total, "tube_bundle_hydraulics_invalid_length", False),
    )
    for label, value, code, strictly_positive in checks:
        valid = math.isfinite(value) and (value > 0.0 if strictly_positive else value >= 0.0)
        if not valid:
            raise ValueError(f"{code}: {label} must be finite and valid.")


def _require_temperature(value: float | None, name: str) -> float:
    if value is None or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"tube_bundle_hydraulics_nonfinite_state: {name} must be finite and positive.")
    return float(value)


def _require_pressure(value: float | None) -> float:
    if value is None or not math.isfinite(value) or value <= 0.0:
        raise ValueError("tube_bundle_hydraulics_nonfinite_state: pressure must be finite and positive.")
    return float(value)


def _transport_and_enthalpy(raw: Any) -> tuple[FluidTransportProperties, float | None]:
    from core.properties.common import FluidTransportProperties

    transport = getattr(raw, "transport", raw)
    if isinstance(transport, FluidTransportProperties):
        props = transport
    else:
        props = FluidTransportProperties(
            rho=float(getattr(transport, "rho")), mu=float(getattr(transport, "mu")),
            k=float(getattr(transport, "k")), cp=float(getattr(transport, "cp")),
        )
    enthalpy = getattr(raw, "h", getattr(raw, "enthalpy", None))
    return props, _finite_float_or_none(enthalpy)


def _evaluate_provider_state(provider: Any, temperature: float, pressure: float) -> tuple[FluidTransportProperties, float | None]:
    full_at = getattr(provider, "full_at", None)
    raw = full_at(T=temperature, p=pressure) if callable(full_at) else provider.at(T=temperature, p=pressure)
    return _transport_and_enthalpy(raw)


def _try_temperature_from_enthalpy(provider: Any, *, h: float, p: float) -> float | None:
    for method_name in ("temperature_from_h_p", "T_from_h_p"):
        method = getattr(provider, method_name, None)
        if not callable(method):
            continue
        try:
            raw = method(h=h, p=p)
        except Exception:
            return None
        if isinstance(raw, (int, float)):
            value = float(raw)
        else:
            value = _finite_float_or_none(getattr(raw, "T", None))
        return value if value is not None and value > 0.0 else None
    return None


def _make_hydraulic_point(
    *, position: HydraulicPosition, temperature: float, pressure: float,
    props: FluidTransportProperties, enthalpy: float | None,
    mass_flux: float, hydraulic_diameter: float,
    relative_roughness: float | None = None,
) -> TubeSideHydraulicPoint:
    if not math.isfinite(props.rho) or props.rho <= 0.0:
        raise ValueError("tube_bundle_hydraulics_invalid_density: density must be finite and positive.")
    if not math.isfinite(props.mu) or props.mu <= 0.0:
        raise ValueError("tube_bundle_hydraulics_invalid_viscosity: viscosity must be finite and positive.")
    velocity = mass_flux / props.rho
    reynolds = mass_flux * hydraulic_diameter / props.mu
    if not math.isfinite(reynolds) or reynolds <= 0.0:
        raise ValueError("tube_bundle_hydraulics_nonfinite_state: Reynolds number must be finite and positive.")
    friction_factor = darcy_friction_factor(reynolds, relative_roughness)
    friction_factor_method = darcy_friction_factor_method(reynolds, relative_roughness)
    if not math.isfinite(friction_factor) or friction_factor <= 0.0:
        raise ValueError("tube_bundle_hydraulics_invalid_friction_factor: friction factor must be finite and positive.")
    dynamic_pressure_value = mass_flux**2 / (2.0 * props.rho)
    prandtl = props.mu * props.cp / props.k
    if not all(math.isfinite(value) for value in (temperature, pressure, velocity, dynamic_pressure_value, prandtl)):
        raise ValueError("tube_bundle_hydraulics_nonfinite_state: hydraulic point is not finite.")
    return TubeSideHydraulicPoint(
        position=position, temperature=float(temperature), pressure=float(pressure),
        props=props, enthalpy=enthalpy, mass_flux=float(mass_flux),
        velocity=velocity, reynolds=reynolds, friction_factor=friction_factor,
        friction_factor_method=friction_factor_method,
        dynamic_pressure=dynamic_pressure_value, prandtl=prandtl,
    )


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_enthalpy(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _midpoint_fallback_warning(reason: str) -> ModelWarning:
    return make_warning(
        code="tube_bundle_hydraulics_midpoint_temperature_fallback",
        message=(
            "tube_bundle_hydraulics: " + reason + "; using arithmetic "
            "temperature midpoint."
        ),
        source="tube_bundle_hydraulics",
        severity="info",
    )


def _deduplicate_warnings(warnings: list[ModelWarning]) -> tuple[ModelWarning, ...]:
    unique: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        identity = (warning.source, warning.code)
        if identity not in seen:
            seen.add(identity)
            unique.append(warning)
    return tuple(unique)


def _pass_boundary_fallback_warning(reason: str) -> ModelWarning:
    return make_warning(
        code="tube_bundle_hydraulics_pass_boundary_temperature_fallback",
        message=(
            "tube_bundle_hydraulics: " + reason + "; using linear "
            "temperature interpolation for tube-pass boundary states."
        ),
        source="tube_bundle_hydraulics",
        severity="info",
    )


def _make_pass_boundary_state(
    *, boundary_index: int, temperature: float, pressure: float,
    props: FluidTransportProperties, flow_area_per_pass: float,
    mass_flux: float, hydraulic_diameter: float,
) -> TubePassBoundaryHydraulicState:
    if not math.isfinite(props.rho) or props.rho <= 0.0:
        raise ValueError("tube_bundle_hydraulics_invalid_density: density must be finite and positive.")
    if not math.isfinite(props.mu) or props.mu <= 0.0:
        raise ValueError("tube_bundle_hydraulics_invalid_viscosity: viscosity must be finite and positive.")
    velocity = mass_flux / props.rho
    dynamic_pressure_value = mass_flux**2 / (2.0 * props.rho)
    reynolds = mass_flux * hydraulic_diameter / props.mu
    if not all(math.isfinite(value) for value in (temperature, pressure, velocity, dynamic_pressure_value, reynolds)):
        raise ValueError("tube_bundle_hydraulics_nonfinite_state: pass-boundary state is not finite.")
    return TubePassBoundaryHydraulicState(
        boundary_index=boundary_index,
        temperature=float(temperature), pressure=float(pressure), props=props,
        flow_area_per_pass=float(flow_area_per_pass), mass_flux=float(mass_flux),
        velocity=velocity, dynamic_pressure=dynamic_pressure_value, reynolds=reynolds,
    )


def _build_tube_pass_boundary_states(
    *,
    n_tube_passes: int,
    mass_flux: float,
    mass_flux_inlet: float,
    mass_flux_outlet: float,
    variable_gas_flow: bool,
    flow_area_per_pass: float,
    hydraulic_diameter: float,
    T_in: float,
    T_out: float,
    p: float,
    provider: Any | None,
    props_in: FluidTransportProperties,
    h_in: float | None,
    props_mid: FluidTransportProperties,
    props_out: FluidTransportProperties,
    h_out: float | None,
) -> tuple[tuple[TubePassBoundaryHydraulicState, ...], str, list[ModelWarning]]:
    """Build the ``n_tube_passes + 1`` tube-side pass-boundary states.

    Boundary 0 is the tube-side inlet (``props_in``); boundary
    ``n_tube_passes`` is the tube-side outlet (``props_out``); reuses the
    same enthalpy-interpolation-with-linear-temperature-fallback strategy
    as the existing three-state midpoint, generalized to ``n - 1``
    intermediate boundaries and applied uniformly (one method for the
    whole boundary set, not mixed per boundary).
    """
    n = n_tube_passes
    warnings: list[ModelWarning] = []
    method = "linear_temperature"
    temperatures = [T_in + (j / n) * (T_out - T_in) for j in range(n + 1)]

    if provider is None and not variable_gas_flow:
        # Compatibility/no-provider path: uniform constant properties at
        # every boundary, matching calculate_tube_bundle_hydraulics'
        # existing inlet/midpoint/outlet compatibility behaviour.
        props_list = [props_in] * (n + 1)
    elif provider is None:
        # Active wet-gas path: exact inlet/midpoint/outlet transport states
        # are supplied by the phase-change solver.  Interpolate only any
        # additional pass boundaries; for the common two-pass bundle this
        # uses the three supplied states exactly.
        def interpolate_props(left, right, fraction):
            return type(left)(
                rho=left.rho + fraction * (right.rho - left.rho),
                mu=left.mu + fraction * (right.mu - left.mu),
                k=left.k + fraction * (right.k - left.k),
                cp=left.cp + fraction * (right.cp - left.cp),
            )

        props_list = []
        for j in range(n + 1):
            position = j / n
            if position <= 0.5:
                props_list.append(interpolate_props(props_in, props_mid, 2.0 * position))
            else:
                props_list.append(interpolate_props(props_mid, props_out, 2.0 * position - 1.0))
    else:
        if _finite_enthalpy(h_in) and _finite_enthalpy(h_out):
            h_in_f, h_out_f = float(h_in), float(h_out)
            candidate_temperatures = [T_in]
            inversion_ok = True
            for j in range(1, n):
                h_j = h_in_f + (j / n) * (h_out_f - h_in_f)
                T_j = _try_temperature_from_enthalpy(provider, h=h_j, p=p)
                if T_j is None:
                    inversion_ok = False
                    break
                candidate_temperatures.append(T_j)
            if inversion_ok:
                candidate_temperatures.append(T_out)
                temperatures = candidate_temperatures
                method = "enthalpy"
            else:
                warnings.append(_pass_boundary_fallback_warning("inversion is unavailable or failed"))
        else:
            warnings.append(_pass_boundary_fallback_warning("complete inlet/outlet enthalpy data are unavailable"))

        props_list = [props_in]
        for j in range(1, n):
            props_j, _ = _evaluate_provider_state(provider, temperatures[j], p)
            props_list.append(props_j)
        props_list.append(props_out)

    boundary_mass_fluxes = (
        [
            mass_flux_inlet + (j / n) * (mass_flux_outlet - mass_flux_inlet)
            for j in range(n + 1)
        ]
        if variable_gas_flow
        else [mass_flux] * (n + 1)
    )
    states = tuple(
        _make_pass_boundary_state(
            boundary_index=j, temperature=temperatures[j], pressure=p,
            props=props_list[j], flow_area_per_pass=flow_area_per_pass,
            mass_flux=boundary_mass_fluxes[j], hydraulic_diameter=hydraulic_diameter,
        )
        for j in range(n + 1)
    )
    return states, method, warnings


def _build_entrance_result(
    *, component_id: str, pass_index: int | None,
    state: TubePassBoundaryHydraulicState, entrance_type: TubeSheetEntranceType,
) -> TubeEndPressureDropResult:
    K, dp = calculate_tube_sheet_entrance_loss(
        dynamic_pressure=state.dynamic_pressure, entrance_type=entrance_type,
    )
    return TubeEndPressureDropResult(
        component_id=component_id, component_type="tube_sheet_entrance",
        pass_index=pass_index, boundary_index=state.boundary_index,
        loss_coefficient=K, reference_velocity=state.velocity,
        reference_dynamic_pressure=state.dynamic_pressure,
        pressure_drop=dp, method=f"tube_sheet_entrance_{entrance_type.value}",
        warnings=(),
    )


def _build_exit_result(
    *, component_id: str, pass_index: int | None,
    state: TubePassBoundaryHydraulicState, exit_type: TubeSheetExitType,
) -> TubeEndPressureDropResult:
    K, dp = calculate_tube_sheet_exit_loss(
        dynamic_pressure=state.dynamic_pressure, exit_type=exit_type,
    )
    return TubeEndPressureDropResult(
        component_id=component_id, component_type="tube_sheet_exit",
        pass_index=pass_index, boundary_index=state.boundary_index,
        loss_coefficient=K, reference_velocity=state.velocity,
        reference_dynamic_pressure=state.dynamic_pressure,
        pressure_drop=dp, method=f"tube_sheet_exit_{exit_type.value}",
        warnings=(),
    )


def _apply_tube_sheet_entrances_and_exits(
    *,
    tube_path_type: TubePathType,
    n_tube_passes: int,
    boundary_states: tuple[TubePassBoundaryHydraulicState, ...],
    entrance_type: TubeSheetEntranceType,
    exit_type: TubeSheetExitType,
) -> tuple[tuple[TubeEndPressureDropResult, ...], tuple[TubeEndPressureDropResult, ...], float, float]:
    """Apply tube-sheet entrance/exit losses per ``tube_path_type``.

    ``STRAIGHT``: one entrance and one exit per pass (entrance at boundary
    ``pass_index - 1``, exit at boundary ``pass_index``); ``entrance_count
    == exit_count == n_tube_passes``. ``U_TUBE``: a single entrance at
    boundary 0 and a single exit at the final boundary only;
    ``entrance_count == exit_count == 1``. No loss is applied at
    intermediate U-bend boundaries, and no U-bend/direction-change loss is
    calculated here.
    """
    entrance_results: list[TubeEndPressureDropResult] = []
    exit_results: list[TubeEndPressureDropResult] = []

    if tube_path_type == TubePathType.STRAIGHT:
        for pass_index in range(1, n_tube_passes + 1):
            entrance_results.append(
                _build_entrance_result(
                    component_id=f"pass_{pass_index}_entrance", pass_index=pass_index,
                    state=boundary_states[pass_index - 1], entrance_type=entrance_type,
                )
            )
            exit_results.append(
                _build_exit_result(
                    component_id=f"pass_{pass_index}_exit", pass_index=pass_index,
                    state=boundary_states[pass_index], exit_type=exit_type,
                )
            )
    else:
        entrance_results.append(
            _build_entrance_result(
                component_id="tube_bundle_entrance", pass_index=None,
                state=boundary_states[0], entrance_type=entrance_type,
            )
        )
        exit_results.append(
            _build_exit_result(
                component_id="tube_bundle_exit", pass_index=None,
                state=boundary_states[-1], exit_type=exit_type,
            )
        )

    dp_tube_entrances = sum(result.pressure_drop for result in entrance_results)
    dp_tube_exits = sum(result.pressure_drop for result in exit_results)
    return tuple(entrance_results), tuple(exit_results), dp_tube_entrances, dp_tube_exits


def mean_velocity(m_dot: float, rho: float, flow_area: float) -> float:
    """v = m_dot / (rho * A)"""
    if m_dot <= 0.0:
        raise ValueError("m_dot must be positive.")
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if flow_area <= 0.0:
        raise ValueError("flow_area must be positive.")
    return m_dot / (rho * flow_area)


def reynolds_number(rho: float, v: float, D: float, mu: float) -> float:
    """Re = rho * v * D / mu"""
    if rho <= 0.0 or mu <= 0.0 or D <= 0.0 or v <= 0.0:
        raise ValueError("rho, mu, D, v must be positive.")
    return rho * v * D / mu


# NOTE: friction_factor_smooth is no longer defined here. It is imported
# above from core.pressure_drop.straight_sections (the single canonical
# pressure-drop friction-factor implementation, now also covering rough
# tubes via darcy_friction_factor) and re-exported under this module's
# namespace so that `from core.pressure_drop.internal_pressure_drop import
# friction_factor_smooth` keeps resolving to the same object as before.


def dynamic_pressure(rho: float, v: float) -> float:
    """q = rho*v^2/2"""
    if rho <= 0.0 or v <= 0.0:
        raise ValueError("rho and v must be positive.")
    return rho * v * v / 2.0


# -----------------------------
# Component pressure drop terms
# -----------------------------

def pressure_drop_tubes(f: float, L: float, D: float, rho: float, v: float) -> float:
    """
    Frictional losses along tube length (Darcy–Weisbach).

    dp = f * (L/D) * (rho*v^2/2)

    Ref: White; Incropera (fluid flow basics).
    """
    if f <= 0.0:
        raise ValueError("f must be positive.")
    if L <= 0.0 or D <= 0.0:
        raise ValueError("L and D must be positive.")
    return f * (L / D) * dynamic_pressure(rho, v)


def pressure_drop_inlet(rho: float, v: float, K_in: float = 0.5) -> float:
    """
    Inlet minor loss.

    dp = K_in * (rho*v^2/2)

    MVP default K_in=0.5 is a common starting point for abrupt/sharp-edged entrance.
    Ref: Idelchik; Crane TP-410.
    """
    if K_in < 0.0:
        raise ValueError("K_in must be non-negative.")
    return K_in * dynamic_pressure(rho, v)


def pressure_drop_outlet(rho: float, v: float, K_out: float = 1.0) -> float:
    """
    Outlet minor loss.

    dp = K_out * (rho*v^2/2)

    MVP default K_out=1.0 is a common starting point for discharge into a plenum/header.
    Ref: Idelchik; Crane TP-410.
    """
    if K_out < 0.0:
        raise ValueError("K_out must be non-negative.")
    return K_out * dynamic_pressure(rho, v)


def pressure_drop_turns(rho: float, v: float, n_turns: int, K_turn: float = 1.5) -> float:
    """
    Return/turn losses (e.g. 180° turns between passes).

    dp = n_turns * K_turn * (rho*v^2/2)

    MVP default K_turn=1.5 is a typical order-of-magnitude placeholder.
    Real K_turn depends strongly on header geometry, radius, and flow distribution.
    Ref: Idelchik; Crane TP-410.
    """
    if n_turns < 0:
        raise ValueError("n_turns must be non-negative.")
    if K_turn < 0.0:
        raise ValueError("K_turn must be non-negative.")
    return float(n_turns) * K_turn * dynamic_pressure(rho, v)


def pressure_drop_internal_total(
    m_dot: float,
    flow_area: float,
    hydraulic_diameter: float,
    flow_length: float,
    props: FluidProps,
    *,
    n_turns: int = 0,
    K_in: float = 0.5,
    K_out: float = 1.0,
    K_turn: float = 1.5,
) -> tuple[float, float, float, float, float, float]:
    """
    Compute component-based tube-side pressure drop.

    Returns
    -------
    dp_total : float
    dp_tubes : float
    dp_inlet : float
    dp_outlet : float
    dp_turns : float
    Re : float

    Also computes f internally (returned separately below).

    Notes
    -----
    - Defaults for K coefficients are allowed for MVP only.
    - Callers may override them explicitly for engineering calibration.
    """
    if flow_area <= 0.0 or hydraulic_diameter <= 0.0 or flow_length <= 0.0:
        raise ValueError("flow_area, hydraulic_diameter, flow_length must be positive.")

    v = mean_velocity(m_dot, props.rho, flow_area)
    Re = reynolds_number(props.rho, v, hydraulic_diameter, props.mu)
    f = friction_factor_smooth(Re)

    dp_t = pressure_drop_tubes(f, flow_length, hydraulic_diameter, props.rho, v)
    dp_in = pressure_drop_inlet(props.rho, v, K_in=K_in)
    dp_out = pressure_drop_outlet(props.rho, v, K_out=K_out)
    dp_turn = pressure_drop_turns(props.rho, v, n_turns=n_turns, K_turn=K_turn)

    dp_total = dp_t + dp_in + dp_out + dp_turn

    return dp_total, dp_t, dp_in, dp_out, dp_turn, Re, f, v
