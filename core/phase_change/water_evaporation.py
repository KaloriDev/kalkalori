# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""IAPWS adapter and quality integration for in-tube water evaporation.

The transport equation remains in
``core.heat_transfer.evaporation_inside_shah1982``. This layer supplies one
cached saturation snapshot, converts its endpoint transport properties to
the explicit SI boundary of that equation, and integrates local resistance
over increasing thermodynamic quality. It has no heat-exchanger or public
Simulation/Rating dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from core.common.warnings import ModelWarning
from core.geometry.tube import TubeOrientation
from core.heat_transfer.evaporation_inside_shah1982 import (
    SHAH_1982_BOILING_CORRELATION,
    Shah1982BoilingResult,
    shah1982_boiling_alpha_local,
)
from core.properties.water import (
    WATER_CRITICAL_PRESSURE_PA,
    WaterSteamSaturationProperties,
    water_saturation_snapshot,
)


@dataclass(frozen=True)
class WaterEvaporationZoneResult:
    zone_alpha_evaporation: float
    correlation: str
    regimes: tuple[str, ...]
    orientation: TubeOrientation
    quality_in: float
    quality_out: float
    mass_flow_total: float
    mass_flux: float
    m_dot_evaporated: float
    Q_evaporation: float
    hfg: float
    heat_flux_inner: float
    quadrature_order: int
    local_results: tuple[Shah1982BoilingResult, ...]
    two_phase_pressure_drop_supported: bool
    warnings: tuple[ModelWarning, ...]
    assumptions: tuple[str, ...]


# Eight-point Gauss-Legendre nodes lie strictly within each quality interval;
# neither singular endpoint x=0 nor x=1 is ever sent to the local correlation.
_GL8_NODES = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
)
_GL8_WEIGHTS = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
)


def water_mass_flux(mass_flow_total: float, flow_area_per_pass: float) -> float:
    """Return total pure-water mass flux G [kg/(m2*s)]."""
    _positive_finite(mass_flow_total, "mass_flow_total")
    _positive_finite(flow_area_per_pass, "flow_area_per_pass")
    return mass_flow_total / flow_area_per_pass


def local_water_evaporation_alpha(
    *,
    p: float,
    mass_flux: float,
    tube_inner_diameter: float,
    quality: float,
    heat_flux_inner: float,
    orientation: TubeOrientation,
    saturation: WaterSteamSaturationProperties | None = None,
) -> Shah1982BoilingResult:
    """Evaluate a local, endpoint-safe Shah (1982) water-boiling HTC."""
    _positive_finite(p, "p")
    _positive_finite(mass_flux, "mass_flux")
    _positive_finite(tube_inner_diameter, "tube_inner_diameter")
    _positive_finite(heat_flux_inner, "heat_flux_inner")
    if p >= WATER_CRITICAL_PRESSURE_PA:
        raise ValueError("Water evaporation is unsupported at or above critical pressure.")
    if not isinstance(orientation, TubeOrientation):
        raise ValueError("orientation must be an explicit TubeOrientation value.")
    if not math.isfinite(quality) or not 0.0 < quality < 1.0:
        raise ValueError("Local evaporation quality must satisfy 0 < x < 1.")

    sat = saturation or water_saturation_snapshot(p)
    if not math.isclose(sat.p, p, rel_tol=0.0, abs_tol=max(1e-6, 1e-12 * p)):
        raise ValueError("The supplied saturation snapshot does not match pressure p.")
    liquid = sat.saturated_liquid.transport
    vapor = sat.saturated_vapor.transport
    if liquid is None or vapor is None:
        raise ValueError("Saturation endpoint transport states are inconsistent.")
    return shah1982_boiling_alpha_local(
        p=p,
        pcritical=WATER_CRITICAL_PRESSURE_PA,
        tube_inner_diameter=tube_inner_diameter,
        mass_flux=mass_flux,
        quality=quality,
        heat_flux_inner=heat_flux_inner,
        orientation=orientation,
        liquid_density=liquid.rho,
        vapor_density=vapor.rho,
        liquid_viscosity=liquid.mu,
        liquid_conductivity=liquid.k,
        liquid_prandtl=liquid.cp * liquid.mu / liquid.k,
        latent_heat=sat.hfg,
    )


def solve_water_evaporation_zone(
    *,
    p: float,
    mass_flow_total: float,
    flow_area_per_pass: float,
    tube_inner_diameter: float,
    quality_in: float,
    quality_out: float,
    heat_flux_inner: float,
    orientation: TubeOrientation,
    saturation: WaterSteamSaturationProperties | None = None,
) -> WaterEvaporationZoneResult:
    """Integrate increasing quality using an area-consistent harmonic HTC."""
    _positive_finite(mass_flow_total, "mass_flow_total")
    if not all(math.isfinite(value) for value in (quality_in, quality_out)):
        raise ValueError("Zone qualities must be finite.")
    if not 0.0 <= quality_in < quality_out <= 1.0:
        raise ValueError("Evaporation requires 0 <= quality_in < quality_out <= 1.")
    sat = saturation or water_saturation_snapshot(p)
    G = water_mass_flux(mass_flow_total, flow_area_per_pass)
    midpoint = 0.5 * (quality_in + quality_out)
    half_width = 0.5 * (quality_out - quality_in)
    local_results = tuple(
        local_water_evaporation_alpha(
            p=p,
            mass_flux=G,
            tube_inner_diameter=tube_inner_diameter,
            quality=midpoint + half_width * node,
            heat_flux_inner=heat_flux_inner,
            orientation=orientation,
            saturation=sat,
        )
        for node in _GL8_NODES
    )

    # At constant Tsat and zone-scale deltaT, dA=dQ/(alpha*dT). Since
    # dQ=m*hfg*dx, the area-consistent effective coefficient is the
    # quality-weighted harmonic mean, not an arithmetic alpha.
    inverse_alpha_integral = half_width * sum(
        weight / result.alpha
        for weight, result in zip(_GL8_WEIGHTS, local_results)
    )
    zone_alpha = (quality_out - quality_in) / inverse_alpha_integral
    warnings = _deduplicate(
        warning for result in local_results for warning in result.warnings
    )
    evaporated = mass_flow_total * (quality_out - quality_in)
    return WaterEvaporationZoneResult(
        zone_alpha_evaporation=zone_alpha,
        correlation=SHAH_1982_BOILING_CORRELATION,
        regimes=tuple(sorted({result.regime for result in local_results})),
        orientation=orientation,
        quality_in=quality_in,
        quality_out=quality_out,
        mass_flow_total=mass_flow_total,
        mass_flux=G,
        m_dot_evaporated=evaporated,
        Q_evaporation=evaporated * sat.hfg,
        hfg=sat.hfg,
        heat_flux_inner=heat_flux_inner,
        quadrature_order=len(_GL8_NODES),
        local_results=local_results,
        two_phase_pressure_drop_supported=False,
        warnings=tuple(warnings),
        assumptions=(
            "constant_nominal_pressure",
            "thermodynamic_equilibrium_quality",
            "inside_area_heat_flux",
            "quality_weighted_harmonic_alpha",
            "pre_dryout_subcritical_heat_flux",
            "two_phase_pressure_drop_not_supported",
        ),
    )


def _deduplicate(warnings) -> list[ModelWarning]:
    result: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        key = (warning.code, warning.message)
        if key not in seen:
            seen.add(key)
            result.append(warning)
    return result


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
