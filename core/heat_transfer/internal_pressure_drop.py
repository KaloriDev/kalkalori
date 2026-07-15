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
from typing import TYPE_CHECKING, Any, Literal

from core.common.warnings import ModelWarning, make_warning

if TYPE_CHECKING:
    from core.properties.common import FluidTransportProperties


@dataclass(frozen=True)
class FluidProps:
    rho: float  # [kg/m^3]
    mu: float   # [Pa*s]


HydraulicPosition = Literal["inlet", "midpoint", "outlet"]


@dataclass(frozen=True)
class TubeSideHydraulicPoint:
    """Bulk hydraulic state at one of the three tube-side evaluation points.

    ``friction_factor`` is the Darcy friction factor from the existing smooth
    tube implementation.
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
    dynamic_pressure: float
    prandtl: float

    @property
    def Pr(self) -> float:
        return self.prandtl

    @property
    def darcy_friction_factor(self) -> float:
        return self.friction_factor

    @property
    def h(self) -> float | None:
        return self.enthalpy


@dataclass(frozen=True)
class TubeBundleHydraulicResult:
    """Straight tube-bundle friction and signed acceleration result."""

    inlet: TubeSideHydraulicPoint
    midpoint: TubeSideHydraulicPoint
    outlet: TubeSideHydraulicPoint
    midpoint_method: str
    flow_area_per_pass: float
    mass_flux: float
    hydraulic_diameter: float
    hydraulic_length_total: float
    mean_f_over_rho: float
    dp_friction: float
    dp_acceleration: float
    dp_tube_bundle: float
    warnings: tuple[ModelWarning, ...]

    @property
    def dp_total(self) -> float:
        """Compatibility alias for the straight tube-bundle pressure change."""
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
    provider: Any | None = None,
    temperature_in: float | None = None,
    temperature_out: float | None = None,
    pressure: float | None = None,
    inlet_props: FluidTransportProperties | None = None,
    midpoint_props: FluidTransportProperties | None = None,
    outlet_props: FluidTransportProperties | None = None,
) -> TubeBundleHydraulicResult:
    """Calculate universal three-state straight tube-bundle hydraulics.

    A provider is evaluated at inlet and outlet bulk temperatures.  When
    complete enthalpy data and a provider ``temperature_from_h_p`` capability
    are available, the midpoint is enthalpy-based.  Otherwise the same
    arithmetic-temperature fallback is used for every provider.

    The point-property arguments are a deterministic test/compatibility hook.
    Production exchanger paths pass a provider and temperatures.
    """
    _validate_hydraulic_input(
        m_dot=m_dot,
        flow_area_per_pass=flow_area_per_pass,
        hydraulic_diameter=hydraulic_diameter,
        hydraulic_length_total=hydraulic_length_total,
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

    mass_flux = m_dot / flow_area_per_pass
    points = (
        _make_hydraulic_point(
            position="inlet", temperature=T_in, pressure=p, props=props_in,
            enthalpy=h_in, mass_flux=mass_flux, hydraulic_diameter=hydraulic_diameter,
        ),
        _make_hydraulic_point(
            position="midpoint", temperature=T_mid, pressure=p, props=props_mid,
            enthalpy=h_mid, mass_flux=mass_flux, hydraulic_diameter=hydraulic_diameter,
        ),
        _make_hydraulic_point(
            position="outlet", temperature=T_out, pressure=p, props=props_out,
            enthalpy=h_out, mass_flux=mass_flux, hydraulic_diameter=hydraulic_diameter,
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
    dp_friction = (hydraulic_length_total / hydraulic_diameter) * (mass_flux**2 / 2.0) * mean_f_over_rho

    # One-dimensional momentum balance.  Pressure loss is positive, so a
    # density decrease gives a positive acceleration term and a density
    # increase gives pressure recovery.  Refs: White, Fluid Mechanics;
    # Idelchik, Handbook of Hydraulic Resistance; Crane TP-410.
    dp_acceleration = mass_flux**2 * (1.0 / points[2].props.rho - 1.0 / points[0].props.rho)
    dp_tube_bundle = dp_friction + dp_acceleration

    if not math.isfinite(dp_friction) or dp_friction < 0.0:
        raise ValueError(
            "tube_bundle_hydraulics_invalid_friction: distributed friction must be finite and non-negative."
        )
    if not math.isfinite(dp_acceleration) or not math.isfinite(dp_tube_bundle):
        raise ValueError("tube_bundle_hydraulics_nonfinite_state: pressure change is not finite.")
    if dp_tube_bundle < 0.0:
        warnings.append(
            make_warning(
                code="tube_bundle_hydraulics_negative_total_pressure_change",
                message=(
                    "tube_bundle_hydraulics: signed straight tube-bundle pressure "
                    f"change is negative ({dp_tube_bundle:.6g} Pa), representing "
                    "net pressure recovery after friction."
                ),
                source="tube_bundle_hydraulics",
                severity="warning",
            )
        )

    return TubeBundleHydraulicResult(
        inlet=points[0], midpoint=points[1], outlet=points[2],
        midpoint_method=midpoint_method,
        flow_area_per_pass=float(flow_area_per_pass), mass_flux=mass_flux,
        hydraulic_diameter=float(hydraulic_diameter),
        hydraulic_length_total=float(hydraulic_length_total),
        mean_f_over_rho=mean_f_over_rho,
        dp_friction=dp_friction, dp_acceleration=dp_acceleration,
        dp_tube_bundle=dp_tube_bundle,
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
) -> TubeSideHydraulicPoint:
    if not math.isfinite(props.rho) or props.rho <= 0.0:
        raise ValueError("tube_bundle_hydraulics_invalid_density: density must be finite and positive.")
    if not math.isfinite(props.mu) or props.mu <= 0.0:
        raise ValueError("tube_bundle_hydraulics_invalid_viscosity: viscosity must be finite and positive.")
    velocity = mass_flux / props.rho
    reynolds = mass_flux * hydraulic_diameter / props.mu
    if not math.isfinite(reynolds) or reynolds <= 0.0:
        raise ValueError("tube_bundle_hydraulics_nonfinite_state: Reynolds number must be finite and positive.")
    friction_factor = friction_factor_smooth(reynolds)
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


def friction_factor_smooth(Re: float) -> float:
    """
    Darcy friction factor for smooth tubes.

    - Laminar (Re < 2300): f = 64 / Re
    - Turbulent: Petukhov explicit approximation (smooth pipe)

    Ref: White (smooth pipe friction), widely used in engineering practice.
    """
    if Re <= 0.0:
        raise ValueError("Re must be positive.")
    if Re < 2300.0:
        return 64.0 / Re
    return 1.0 / (0.79 * math.log(Re) - 1.64) ** 2


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
