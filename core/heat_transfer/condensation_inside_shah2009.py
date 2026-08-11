# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Transport-only Shah (2009) condensation correlation for plain tubes.

The function in this module is deliberately independent of property
backends, saturation solvers, heat-exchanger geometry orchestration and
phase-zone integration. Callers provide pressure, critical pressure, tube
diameter, mass flux, quality and saturated endpoint transport properties in
plain SI units.

Reference: M. M. Shah (2009), *An Improved and Extended General Correlation
for Heat Transfer During Condensation in Plain Tubes*, HVAC&R Research 15(5),
889–913, equations 4–12, DOI 10.1080/10789669.2009.10390871.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from core.common.warnings import ModelWarning, make_warning
from core.geometry.tube import TubeOrientation


SHAH_2009_CORRELATION = "Shah 2009 improved/extended in-tube condensation"
SHAH_2009_DOI = "10.1080/10789669.2009.10390871"
GRAVITY = 9.80665


@dataclass(frozen=True)
class Shah2009LocalResult:
    """Local heat-transfer result and dimensionless diagnostics."""

    alpha: float
    correlation: str
    regime: str
    orientation: TubeOrientation
    quality: float
    mass_flux: float
    reduced_pressure: float
    Re_LT: float
    Re_GT: float
    Re_LS: float
    Pr_liquid: float
    J_g: float
    Z: float
    h_forced: float
    h_gravity: float
    exponent_n: float
    warnings: tuple[ModelWarning, ...]

    @property
    def p_r(self) -> float:
        """Shah reduced-pressure symbol, as an alias for diagnostics."""
        return self.reduced_pressure

    @property
    def h_I(self) -> float:
        """Shah Regime-I forced-convection term."""
        return self.h_forced

    @property
    def h_Nu(self) -> float:
        """Shah gravity-film (Nusselt) term."""
        return self.h_gravity

    @property
    def n(self) -> float:
        """Pressure-dependent exponent used in the Regime-I term."""
        return self.exponent_n


def shah2009_condensation_alpha_local(
    *,
    p: float,
    pcritical: float,
    tube_inner_diameter: float,
    mass_flux: float,
    quality: float,
    orientation: TubeOrientation,
    liquid_density: float,
    vapor_density: float,
    liquid_viscosity: float,
    vapor_viscosity: float,
    liquid_conductivity: float,
    liquid_specific_heat: float,
) -> Shah2009LocalResult:
    """Evaluate local Shah (2009) condensation HTC from plain SI inputs."""
    for name, value in (
        ("p", p),
        ("pcritical", pcritical),
        ("tube_inner_diameter", tube_inner_diameter),
        ("mass_flux", mass_flux),
        ("liquid_density", liquid_density),
        ("vapor_density", vapor_density),
        ("liquid_viscosity", liquid_viscosity),
        ("vapor_viscosity", vapor_viscosity),
        ("liquid_conductivity", liquid_conductivity),
        ("liquid_specific_heat", liquid_specific_heat),
    ):
        _positive_finite(value, name)
    if p >= pcritical:
        raise ValueError("Condensation is unsupported at or above critical pressure.")
    if liquid_density <= vapor_density:
        raise ValueError("liquid_density must be greater than vapor_density.")
    if not isinstance(orientation, TubeOrientation):
        raise ValueError("orientation must be an explicit TubeOrientation value.")
    if not math.isfinite(quality) or not 0.0 < quality < 1.0:
        raise ValueError("Local condensation quality must satisfy 0 < x < 1.")

    G = mass_flux
    D = tube_inner_diameter
    x = quality
    p_r = p / pcritical
    Re_LT = G * D / liquid_viscosity
    Re_GT = G * D / vapor_viscosity
    Re_LS = G * (1.0 - x) * D / liquid_viscosity
    Pr_l = liquid_specific_heat * liquid_viscosity / liquid_conductivity
    Z = ((1.0 / x) - 1.0) ** 0.8 * p_r**0.4
    J_g = x * G / math.sqrt(
        GRAVITY * D * vapor_density * (liquid_density - vapor_density)
    )

    # Shah (2009), equations 8a/8b. h_LT uses total mass as liquid.
    h_LT = 0.023 * Re_LT**0.8 * Pr_l**0.4 * liquid_conductivity / D
    exponent_n = 0.0058 + 0.557 * p_r
    h_forced = h_LT * (liquid_viscosity / (14.0 * vapor_viscosity)) ** exponent_n * (
        (1.0 - x) ** 0.8
        + 3.8 * x**0.76 * (1.0 - x) ** 0.04 / p_r**0.38
    )

    # Shah (2009), equation 9: McAdams-adjusted Nusselt laminar film term.
    h_gravity = 1.32 * Re_LS ** (-1.0 / 3.0) * (
        liquid_density
        * (liquid_density - vapor_density)
        * GRAVITY
        * liquid_conductivity**3
        / liquid_viscosity**2
    ) ** (1.0 / 3.0)

    if orientation is TubeOrientation.HORIZONTAL:
        regime_i_boundary = 0.98 * (Z + 0.263) ** (-0.62)
        if J_g >= regime_i_boundary:
            regime = "I"
            alpha = h_forced
        else:
            regime = "II"
            alpha = h_forced + h_gravity
    else:
        regime_i_boundary = 1.0 / (2.4 * Z + 0.73)
        regime_iii_boundary = 0.89 - 0.93 * math.exp(-0.087 * Z ** (-1.17))
        if J_g >= regime_i_boundary:
            regime = "I"
            alpha = h_forced
        elif J_g <= regime_iii_boundary:
            regime = "III"
            alpha = h_gravity
        else:
            regime = "II"
            alpha = h_forced + h_gravity

    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("Shah 2009 produced a non-finite or non-positive HTC.")

    warnings = _applicability_warnings(
        orientation=orientation,
        D=D,
        p_r=p_r,
        G=G,
        Pr_l=Pr_l,
        Re_LT=Re_LT,
        Re_GT=Re_GT,
        x=x,
        Z=Z,
        J_g=J_g,
        regime=regime,
    )
    return Shah2009LocalResult(
        alpha=alpha,
        correlation=SHAH_2009_CORRELATION,
        regime=regime,
        orientation=orientation,
        quality=x,
        mass_flux=G,
        reduced_pressure=p_r,
        Re_LT=Re_LT,
        Re_GT=Re_GT,
        Re_LS=Re_LS,
        Pr_liquid=Pr_l,
        J_g=J_g,
        Z=Z,
        h_forced=h_forced,
        h_gravity=h_gravity,
        exponent_n=exponent_n,
        warnings=tuple(warnings),
    )


def _applicability_warnings(
    *,
    orientation: TubeOrientation,
    D: float,
    p_r: float,
    G: float,
    Pr_l: float,
    Re_LT: float,
    Re_GT: float,
    x: float,
    Z: float,
    J_g: float,
    regime: str,
) -> list[ModelWarning]:
    warnings: list[ModelWarning] = []
    ranges = (
        ("tube_inner_diameter", D, 0.002, 0.049, "m"),
        ("reduced_pressure", p_r, 0.0008, 0.905, "-"),
        ("mass_flux", G, 4.0, 820.0, "kg/(m2*s)"),
        ("liquid_Prandtl", Pr_l, 1.0, 18.0, "-"),
        ("Re_LT", Re_LT, 68.0, 84_827.0, "-"),
        ("Re_GT", Re_GT, 9_534.0, 523_317.0, "-"),
        ("quality", x, 0.01, 0.99, "-"),
        ("Z", Z, 0.005, 20.0, "-"),
        ("J_g", J_g, 0.06, 20.0, "-"),
    )
    for name, value, lower, upper, unit in ranges:
        if value < lower or value > upper:
            warnings.append(
                make_warning(
                    code="STEAM_CONDENSATION_SHAH_2009_OUTSIDE_RANGE",
                    message=(
                        f"Shah 2009 applicability: {name}={value:.6g} {unit} "
                        f"is outside the published [{lower:g}, {upper:g}] range; "
                        "the value was not clipped."
                    ),
                    source="condensation_inside_shah2009",
                    severity="warning",
                )
            )
    if orientation is TubeOrientation.HORIZONTAL and regime == "II" and Re_GT <= 35_000.0:
        warnings.append(
            make_warning(
                code="STEAM_CONDENSATION_HORIZONTAL_LOW_RE_GT_UNVERIFIED",
                message=(
                    "Shah 2009 recommends the horizontal Regime-II sum only "
                    "for Re_GT > 35000; this low-flow horizontal result retains "
                    "the gravity term but is an unverified extrapolation."
                ),
                source="condensation_inside_shah2009",
                severity="warning",
            )
        )
    if orientation is TubeOrientation.DOWNWARD_INCLINED_15_PLUS:
        warnings.append(
            make_warning(
                code="STEAM_CONDENSATION_INCLINED_TREATED_AS_VERTICAL",
                message=(
                    "Shah 2009 recommends treating downward inclinations of "
                    "15 degrees or greater with the vertical regime map, subject "
                    "to limited validation data."
                ),
                source="condensation_inside_shah2009",
                severity="info",
            )
        )
    return warnings


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
