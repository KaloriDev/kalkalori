# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Transport-only Shah (1982) saturated flow-boiling correlation.

This module has no property-backend, heat-exchanger, or phase-orchestration
dependency. It implements equations 1--14 of the primary publication for
plain circular tubes. The caller supplies a local thermodynamic quality and
an average heat flux referenced to the *inside wetted area* of the tube.

Reference: M. M. Shah (1982), "Chart Correlation for Saturated Boiling Heat
Transfer: Equations and Further Study", ASHRAE Transactions 88(1), 165--196,
paper 2673. No DOI was assigned in the publication.

The paper explicitly limits the correlation to subcritical heat flux and to
qualities below dryout. It does not predict CHF, dryout quality, or
post-dryout heat transfer; this implementation does not add such models.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from core.common.warnings import ModelWarning, make_warning
from core.geometry.tube import TubeOrientation


SHAH_1982_BOILING_CORRELATION = "Shah 1982 saturated flow boiling"
SHAH_1982_REFERENCE = "ASHRAE Transactions 88(1), paper 2673, 165-196"
GRAVITY = 9.80665


@dataclass(frozen=True)
class Shah1982BoilingResult:
    """Local HTC plus the equation-level diagnostics used to obtain it."""

    alpha: float
    correlation: str
    regime: str
    orientation: TubeOrientation
    quality: float
    mass_flux: float
    heat_flux_inner: float
    reduced_pressure: float
    boiling_number: float
    convection_number: float
    liquid_froude_number: float
    liquid_reynolds: float
    liquid_prandtl: float
    N: float
    F: float
    alpha_liquid: float
    psi: float
    psi_nucleate: float | None
    psi_bubble_suppression: float | None
    psi_convective: float
    warnings: tuple[ModelWarning, ...]

    @property
    def Bo(self) -> float:
        return self.boiling_number

    @property
    def Co(self) -> float:
        return self.convection_number

    @property
    def Fr_L(self) -> float:
        return self.liquid_froude_number

    @property
    def Re_L(self) -> float:
        return self.liquid_reynolds


def shah1982_boiling_alpha_local(
    *,
    p: float,
    pcritical: float,
    tube_inner_diameter: float,
    mass_flux: float,
    quality: float,
    heat_flux_inner: float,
    orientation: TubeOrientation,
    liquid_density: float,
    vapor_density: float,
    liquid_viscosity: float,
    liquid_conductivity: float,
    liquid_prandtl: float,
    latent_heat: float,
) -> Shah1982BoilingResult:
    """Evaluate local Shah (1982) saturated flow-boiling HTC in SI units.

    ``heat_flux_inner`` is the average boiling-zone duty divided by the
    tube's inside wetted area. The function deliberately does not infer or
    iterate that value; the later area solver owns heat-flux coupling.
    """
    for name, value in (
        ("p", p),
        ("pcritical", pcritical),
        ("tube_inner_diameter", tube_inner_diameter),
        ("mass_flux", mass_flux),
        ("heat_flux_inner", heat_flux_inner),
        ("liquid_density", liquid_density),
        ("vapor_density", vapor_density),
        ("liquid_viscosity", liquid_viscosity),
        ("liquid_conductivity", liquid_conductivity),
        ("liquid_prandtl", liquid_prandtl),
        ("latent_heat", latent_heat),
    ):
        _positive_finite(value, name)
    if p >= pcritical:
        raise ValueError("Flow boiling is unsupported at or above critical pressure.")
    if liquid_density <= vapor_density:
        raise ValueError("liquid_density must be greater than vapor_density.")
    if not math.isfinite(quality) or not 0.0 < quality < 1.0:
        raise ValueError("Local boiling quality must satisfy 0 < x < 1.")
    if not isinstance(orientation, TubeOrientation):
        raise ValueError("orientation must be an explicit TubeOrientation value.")
    if orientation not in {
        TubeOrientation.HORIZONTAL,
        TubeOrientation.VERTICAL_UPWARD,
    }:
        raise ValueError(
            "Shah 1982 flow boiling supports horizontal or vertical-upward "
            "tube orientation only."
        )

    G = mass_flux
    D = tube_inner_diameter
    x = quality
    rho_l = liquid_density
    rho_v = vapor_density
    Bo = heat_flux_inner / (G * latent_heat)
    Co = ((1.0 / x) - 1.0) ** 0.8 * math.sqrt(rho_v / rho_l)
    Fr_L = G**2 / (rho_l**2 * GRAVITY * D)
    Re_L = G * (1.0 - x) * D / liquid_viscosity
    alpha_l = (
        0.023
        * Re_L**0.8
        * liquid_prandtl**0.4
        * liquid_conductivity
        / D
    )

    if orientation is TubeOrientation.HORIZONTAL and Fr_L <= 0.04:
        N = 0.38 * Fr_L ** (-0.3) * Co
    else:
        N = Co

    psi_convective = 1.8 / N**0.8
    F = 14.7 if Bo >= 11.0e-4 else 15.43
    psi_nucleate: float | None = None
    psi_bubble: float | None = None

    if N > 1.0:
        if Bo > 0.3e-4:
            psi_nucleate = 230.0 * math.sqrt(Bo)
        else:
            psi_nucleate = 1.0 + 46.0 * math.sqrt(Bo)
        if psi_nucleate >= psi_convective:
            regime = "nucleate_boiling"
            psi = psi_nucleate
        else:
            regime = "convective_boiling"
            psi = psi_convective
    else:
        exponent = 2.74 * N ** (-0.1) if N > 0.1 else 2.47 * N ** (-0.15)
        psi_bubble = F * math.sqrt(Bo) * math.exp(exponent)
        if psi_bubble >= psi_convective:
            regime = "bubble_suppression"
            psi = psi_bubble
        else:
            regime = "convective_boiling"
            psi = psi_convective

    alpha = psi * alpha_l
    if not all(math.isfinite(value) and value > 0.0 for value in (N, alpha_l, psi, alpha)):
        raise ValueError("Shah 1982 produced a non-finite or non-positive result.")

    warnings = _applicability_warnings(
        p_r=p / pcritical,
        D=D,
        Re_L=Re_L,
        Bo=Bo,
        Fr_L=Fr_L,
        orientation=orientation,
    )
    return Shah1982BoilingResult(
        alpha=alpha,
        correlation=SHAH_1982_BOILING_CORRELATION,
        regime=regime,
        orientation=orientation,
        quality=x,
        mass_flux=G,
        heat_flux_inner=heat_flux_inner,
        reduced_pressure=p / pcritical,
        boiling_number=Bo,
        convection_number=Co,
        liquid_froude_number=Fr_L,
        liquid_reynolds=Re_L,
        liquid_prandtl=liquid_prandtl,
        N=N,
        F=F,
        alpha_liquid=alpha_l,
        psi=psi,
        psi_nucleate=psi_nucleate,
        psi_bubble_suppression=psi_bubble,
        psi_convective=psi_convective,
        warnings=tuple(warnings),
    )


def _applicability_warnings(
    *,
    p_r: float,
    D: float,
    Re_L: float,
    Bo: float,
    Fr_L: float,
    orientation: TubeOrientation,
) -> list[ModelWarning]:
    warnings: list[ModelWarning] = []
    if p_r > 0.89:
        warnings.append(
            _warning(
                "WATER_BOILING_SHAH_1982_OUTSIDE_RANGE",
                f"reduced_pressure={p_r:.6g} exceeds the published verification limit 0.89; the value was not clipped.",
            )
        )
    if D > 0.041:
        warnings.append(
            _warning(
                "WATER_BOILING_SHAH_1982_OUTSIDE_RANGE",
                f"tube_inner_diameter={D:.6g} m exceeds the published 0.041 m verification limit; the value was not clipped.",
            )
        )
    if Re_L < 10_000.0:
        warnings.append(
            _warning(
                "WATER_BOILING_SHAH_1982_LOW_LIQUID_REYNOLDS",
                f"liquid-only Reynolds number {Re_L:.6g} is below the turbulent Dittus-Boelter range used by Shah; this is an unclipped low-G/endpoint extrapolation.",
            )
        )
    if orientation is TubeOrientation.HORIZONTAL and Fr_L <= 0.04 and Bo < 1.0e-4:
        warnings.append(
            _warning(
                "WATER_BOILING_SHAH_1982_HORIZONTAL_LOW_FR_LOW_BO",
                "Shah recommends the horizontal Fr_L <= 0.04 equations only for Bo >= 1e-4; the low-Bo value was not clipped.",
            )
        )
    warnings.append(
        _warning(
            "WATER_BOILING_DRYOUT_CHF_NOT_MODELLED",
            "Shah 1982 applies below dryout at subcritical heat flux; dryout quality, CHF, and post-dryout heat transfer are not predicted.",
            severity="info",
        )
    )
    return warnings


def _warning(code: str, message: str, *, severity: str = "warning") -> ModelWarning:
    return make_warning(
        code=code,
        message=message,
        source="evaporation_inside_shah1982",
        severity=severity,
    )


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
