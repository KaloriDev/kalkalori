# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Pure-water condensation heat transfer inside plain tubes.

This module is deliberately separate from wet-gas condensation.  It uses
pure-fluid p-h equilibrium and vapor quality; it has no dry-carrier water
ratio, dew point, Lewis number, Chilton-Colburn analogy, or wet-surface
fraction.

Production correlation
----------------------
M. M. Shah (2009), "An Improved and Extended General Correlation for Heat
Transfer During Condensation in Plain Tubes", HVAC&R Research 15(5),
889-913, DOI 10.1080/10789669.2009.10390871, equations 4-12.

The implementation retains Shah's explicit gravity/film term, so the
coefficient does not collapse to liquid-only forced convection at low mass
flux.  Orientation is mandatory because the published regime boundaries are
different for horizontal and downward vertical/inclined tubes. Upward flow
is outside the source correlation and is not accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from core.common.warnings import ModelWarning
from core.geometry.tube import TubeOrientation
from core.heat_transfer.condensation_inside_shah2009 import (
    SHAH_2009_CORRELATION,
    SHAH_2009_DOI,
    Shah2009LocalResult,
    shah2009_condensation_alpha_local,
)
from core.properties.water import (
    WATER_CRITICAL_PRESSURE_PA,
    WaterSteamProperties,
    WaterSteamSaturationProperties,
    water_saturation_snapshot,
)


# Compatibility names remain importable from the phase-change adapter while
# their authoritative definitions live at the transport/geometry layers.
SteamTubeOrientation = TubeOrientation
SteamCondensationLocalResult = Shah2009LocalResult


@dataclass(frozen=True)
class SteamCondensationZoneResult:
    zone_alpha_condensation: float
    correlation: str
    regimes: tuple[str, ...]
    orientation: TubeOrientation
    quality_in: float
    quality_out: float
    mass_flow_total: float
    mass_flux: float
    mass_vapor_in: float
    mass_vapor_out: float
    mass_liquid_in: float
    mass_liquid_out: float
    Q_condensation: float
    hfg: float
    quadrature_order: int
    local_results: tuple[SteamCondensationLocalResult, ...]
    two_phase_pressure_drop_supported: bool
    warnings: tuple[ModelWarning, ...]
    assumptions: tuple[str, ...]


# Eight-point Gauss-Legendre quadrature. Nodes are strictly inside (-1, 1),
# so local formulas are never evaluated at singular x=0 or x=1 endpoints.
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


def steam_mass_flux(mass_flow_total: float, flow_area_per_pass: float) -> float:
    """Return total H2O mass flux G [kg/(m2*s)] for one tube pass."""
    _positive_finite(mass_flow_total, "mass_flow_total")
    _positive_finite(flow_area_per_pass, "flow_area_per_pass")
    return mass_flow_total / flow_area_per_pass


def phase_enthalpy_flow(
    phase_mass_flow: float,
    phase_state: WaterSteamProperties | None,
    *,
    mass_tolerance: float = 1.0e-12,
) -> float:
    """Return m*h while enforcing the optional-phase-state invariant.

    Zero phase mass produces zero *enthalpy flow* without assigning a fake
    specific enthalpy. Positive phase mass requires an actual phase state.
    """
    if not math.isfinite(phase_mass_flow) or phase_mass_flow < 0.0:
        raise ValueError("phase_mass_flow must be finite and non-negative.")
    if phase_mass_flow <= mass_tolerance:
        return 0.0
    if phase_state is None:
        raise ValueError("Positive phase mass requires a thermodynamic phase state.")
    return phase_mass_flow * phase_state.h


def local_steam_condensation_alpha(
    *,
    p: float,
    mass_flux: float,
    tube_inner_diameter: float,
    quality: float,
    orientation: TubeOrientation,
    saturation: WaterSteamSaturationProperties | None = None,
) -> SteamCondensationLocalResult:
    """Evaluate local Shah (2009) pure-fluid condensation HTC.

    ``quality`` must be strictly inside the saturation dome. Zone callers
    integrate with interior Gaussian nodes when an endpoint is x=0 or x=1.
    """
    _positive_finite(p, "p")
    _positive_finite(mass_flux, "mass_flux")
    _positive_finite(tube_inner_diameter, "tube_inner_diameter")
    if not isinstance(orientation, TubeOrientation):
        raise ValueError("orientation must be an explicit TubeOrientation value.")
    if not math.isfinite(quality) or not 0.0 < quality < 1.0:
        raise ValueError("Local condensation quality must satisfy 0 < x < 1.")
    if p >= WATER_CRITICAL_PRESSURE_PA:
        raise ValueError("Steam condensation is unsupported at or above critical pressure.")

    sat = saturation or water_saturation_snapshot(p)
    if not math.isclose(sat.p, p, rel_tol=0.0, abs_tol=max(1.0e-6, 1.0e-12 * p)):
        raise ValueError("The supplied saturation snapshot does not match pressure p.")
    liquid = sat.saturated_liquid.transport
    vapor = sat.saturated_vapor.transport
    if liquid is None or vapor is None:
        raise ValueError("Saturation endpoint transport states are internally inconsistent.")

    return shah2009_condensation_alpha_local(
        p=p,
        pcritical=WATER_CRITICAL_PRESSURE_PA,
        tube_inner_diameter=tube_inner_diameter,
        mass_flux=mass_flux,
        quality=quality,
        orientation=orientation,
        liquid_density=liquid.rho,
        vapor_density=vapor.rho,
        liquid_viscosity=liquid.mu,
        vapor_viscosity=vapor.mu,
        liquid_conductivity=liquid.k,
        liquid_specific_heat=liquid.cp,
    )


def solve_steam_condensation_zone(
    *,
    p: float,
    mass_flow_total: float,
    flow_area_per_pass: float,
    tube_inner_diameter: float,
    quality_in: float,
    quality_out: float,
    orientation: TubeOrientation,
    saturation: WaterSteamSaturationProperties | None = None,
) -> SteamCondensationZoneResult:
    """Integrate a condensation zone over quality with stable quadrature."""
    _positive_finite(mass_flow_total, "mass_flow_total")
    if not all(math.isfinite(value) for value in (quality_in, quality_out)):
        raise ValueError("Zone qualities must be finite.")
    if not 0.0 <= quality_out < quality_in <= 1.0:
        raise ValueError("Condensation requires 0 <= quality_out < quality_in <= 1.")

    sat = saturation or water_saturation_snapshot(p)
    G = steam_mass_flux(mass_flow_total, flow_area_per_pass)
    midpoint = 0.5 * (quality_in + quality_out)
    half_width = 0.5 * (quality_in - quality_out)
    local_results = tuple(
        local_steam_condensation_alpha(
            p=p,
            mass_flux=G,
            tube_inner_diameter=tube_inner_diameter,
            quality=midpoint + half_width * node,
            orientation=orientation,
            saturation=sat,
        )
        for node in _GL8_NODES
    )

    # With constant Tsat and a shared zone-scale temperature difference,
    # dA=dQ/(alpha*dT), making the area-consistent zone coefficient the
    # quality-weighted harmonic mean of local alpha. A future distributed
    # model may instead integrate the full local 1/U resistance; that is an
    # intentional follow-up, not an arithmetic-alpha substitution here.
    inverse_alpha_integral = half_width * sum(
        weight / result.alpha
        for weight, result in zip(_GL8_WEIGHTS, local_results)
    )
    zone_alpha = (quality_in - quality_out) / inverse_alpha_integral

    warnings = _deduplicate_warnings(
        warning for result in local_results for warning in result.warnings
    )
    Q_condensation = mass_flow_total * sat.hfg * (quality_in - quality_out)
    return SteamCondensationZoneResult(
        zone_alpha_condensation=zone_alpha,
        correlation=SHAH_2009_CORRELATION,
        regimes=tuple(sorted({result.regime for result in local_results})),
        orientation=orientation,
        quality_in=quality_in,
        quality_out=quality_out,
        mass_flow_total=mass_flow_total,
        mass_flux=G,
        mass_vapor_in=quality_in * mass_flow_total,
        mass_vapor_out=quality_out * mass_flow_total,
        mass_liquid_in=(1.0 - quality_in) * mass_flow_total,
        mass_liquid_out=(1.0 - quality_out) * mass_flow_total,
        Q_condensation=Q_condensation,
        hfg=sat.hfg,
        quadrature_order=len(_GL8_NODES),
        local_results=local_results,
        two_phase_pressure_drop_supported=False,
        warnings=tuple(warnings),
        assumptions=(
            "constant_nominal_pressure_across_condensation_zone",
            "quality_integrated_0d_zone_coefficient",
            "two_phase_pressure_drop_not_supported",
        ),
    )


def _deduplicate_warnings(warnings: Sequence[ModelWarning] | object) -> list[ModelWarning]:
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
