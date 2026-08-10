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
# plain Python floats in consistent SI units.
#
# Expected SI inputs:
# - p [Pa], G [kg/(m^2*s)], D_i [m]
# - mu_L [Pa*s], k_L [W/(m*K)], cp_L [J/(kg*K)]
# Outputs:
# - alpha [W/(m^2*K)]

"""
In-tube forced-convective condensation heat transfer (v0.6.2).

This is a dedicated, standalone condensation heat-transfer-coefficient
model for a condensing quality-mixture flowing inside a tube. It is
deliberately independent of the v0.6.1 wet-gas condensation solver
(``core.phase_change.inside_condensation_solver``): that model solves a
mass-diffusion problem (dew point, humidity ratio W, Chilton-Colburn/Lewis
analogy) for water vapor condensing out of a non-condensable carrier gas.
Pure-water/steam condensation inside a tube has no carrier gas -- it is a
direct vapor-liquid phase-equilibrium problem parameterized by vapor
quality x -- so none of that mass-transfer machinery applies here.

Correlation: Shah (1979), a fluid-general forced-convective in-tube
condensation correlation (verified against water and several refrigerant/
organic fluids). The published coefficients are reproduced exactly, with
no calibration factors.

Ref: Shah, M.M. (1979). "A general correlation for heat transfer during
film condensation inside pipes." International Journal of Heat and Mass
Transfer, 22(4), 547-556.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.common.warnings import ModelWarning, make_warning

SOURCE = "heat_transfer.condensation_inside"

# Applicability range documented for the original Shah (1979) correlation:
# reduced pressures 0.002-0.44, saturation temperatures 21-310 degC, vapor
# qualities 0-1, mass flux ~10.8-210.6 kg/(m2*s) (39,000-758,000 kg/(m2*h)
# in the original units), all-liquid Reynolds numbers 100-63,000, liquid
# Prandtl numbers 1-13, tube diameters 7-40 mm; verified against water,
# R-11, R-12, R-22, R-113, methanol, ethanol, benzene, toluene and
# trichloroethylene in horizontal, vertical and inclined tubes.
SHAH_1979_REDUCED_PRESSURE_RANGE = (0.002, 0.44)
SHAH_1979_LIQUID_REYNOLDS_RANGE = (100.0, 63_000.0)
SHAH_1979_LIQUID_PRANDTL_RANGE = (1.0, 13.0)
SHAH_1979_MASS_FLUX_RANGE_KG_M2S = (10.8, 210.6)
SHAH_1979_TUBE_DIAMETER_RANGE_M = (0.007, 0.040)

CORRELATION_NAME = "shah_1979"


@dataclass(frozen=True)
class CondensationHTCResult:
    """Local Shah (1979) in-tube condensation heat-transfer result.

    Attributes:
        alpha: Local condensation heat transfer coefficient [W/(m2*K)].
        alpha_LO: All-liquid (single-phase) reference coefficient at the
            same total mass flux [W/(m2*K)] -- Shah's h_LO.
        Re_LO: All-liquid Reynolds number, G*D_i/mu_L [-].
        Pr_L: Saturated-liquid Prandtl number [-].
        p_r: Reduced pressure, p / p_critical [-].
        warnings: Applicability warnings (empty when fully inside the
            documented range).
    """

    alpha: float
    alpha_LO: float
    Re_LO: float
    Pr_L: float
    p_r: float
    warnings: list[ModelWarning]


def shah_condensation_alpha_local(
    x: float,
    *,
    p: float,
    p_critical: float,
    G: float,
    D_i: float,
    mu_L: float,
    k_L: float,
    cp_L: float,
) -> CondensationHTCResult:
    """Local in-tube condensation heat transfer coefficient (Shah, 1979).

    h_LO / (k_L/D_i) = 0.023 * Re_LO^0.8 * Pr_L^0.4
    h_TP / h_LO = (1-x)^0.8 + 3.8 * x^0.76 * (1-x)^0.04 / p_r^0.38

    This is a local, point-quality correlation. Shah's formula collapses
    to zero at x=1 (no film has yet formed) and, less obviously, is only
    validated as a *local* result away from the two-phase boundaries --
    see ``condensation_zone_alpha_effective`` for the zone-averaged value
    a 0D condensing-zone duty should actually use.

    Args:
        x: Local vapor quality [-], strictly inside (0, 1).
        p: Pressure [Pa].
        p_critical: Critical pressure of the condensing fluid [Pa].
        G: Total (vapor + liquid) mass flux [kg/(m2*s)]. Condensation does
            not change the total tube-side mass flow, only its vapor/
            liquid split, so `G` is constant along a condensing zone.
        D_i: Tube inner diameter [m].
        mu_L: Saturated-liquid dynamic viscosity at `p` [Pa*s].
        k_L: Saturated-liquid thermal conductivity at `p` [W/(m*K)].
        cp_L: Saturated-liquid specific heat at `p` [J/(kg*K)].

    Returns:
        CondensationHTCResult.

    Ref: Shah, M.M. (1979), Int. J. Heat Mass Transfer, 22(4), 547-556.
    """
    if not (0.0 < x < 1.0):
        raise ValueError(
            "shah_condensation_alpha_local is a local, point-quality "
            "correlation defined only for 0 < x < 1; it must not be "
            f"evaluated exactly at x=0 or x=1 (got x={x}). Integrate over "
            "the quality range instead, e.g. via "
            "condensation_zone_alpha_effective."
        )
    for name, value in (
        ("p", p),
        ("p_critical", p_critical),
        ("G", G),
        ("D_i", D_i),
        ("mu_L", mu_L),
        ("k_L", k_L),
        ("cp_L", cp_L),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")

    Re_LO = G * D_i / mu_L
    Pr_L = cp_L * mu_L / k_L
    p_r = p / p_critical

    alpha_LO = 0.023 * Re_LO**0.8 * Pr_L**0.4 * (k_L / D_i)
    factor = (1.0 - x) ** 0.8 + 3.8 * x**0.76 * (1.0 - x) ** 0.04 / p_r**0.38
    alpha = alpha_LO * factor

    warnings = _applicability_warnings(Re_LO=Re_LO, Pr_L=Pr_L, p_r=p_r, G=G, D_i=D_i)

    return CondensationHTCResult(
        alpha=alpha,
        alpha_LO=alpha_LO,
        Re_LO=Re_LO,
        Pr_L=Pr_L,
        p_r=p_r,
        warnings=warnings,
    )


@dataclass(frozen=True)
class CondensationZoneHTCResult:
    """Quality-averaged effective condensation HTC over a zone [x_out, x_in]."""

    alpha_effective: float
    x_in: float
    x_out: float
    warnings: list[ModelWarning]


def condensation_zone_alpha_effective(
    *,
    x_in: float,
    x_out: float,
    p: float,
    p_critical: float,
    G: float,
    D_i: float,
    mu_L: float,
    k_L: float,
    cp_L: float,
) -> CondensationZoneHTCResult:
    """Quality-averaged Shah (1979) condensation HTC over ``[x_out, x_in]``.

    The zone-effective coefficient is the mean of the local correlation
    over the quality range actually condensed in this zone, evaluated by
    fixed 5-point Gauss-Legendre quadrature. All quadrature nodes are
    strictly interior to ``(x_out, x_in)``, so the local correlation is
    never evaluated exactly at x=0 or x=1 even when those are the zone's
    own boundaries (e.g. a zone spanning the full saturated-vapor-to-
    saturated-liquid range).

    Args:
        x_in: Zone inlet vapor quality [-], the larger of the two.
        x_out: Zone outlet vapor quality [-], `0 <= x_out < x_in <= 1`.
        p, p_critical, G, D_i, mu_L, k_L, cp_L: see
            `shah_condensation_alpha_local`.

    Returns:
        CondensationZoneHTCResult.
    """
    if not (0.0 <= x_out < x_in <= 1.0):
        raise ValueError(
            "condensation_zone_alpha_effective requires 0 <= x_out < x_in <= 1 "
            f"(got x_in={x_in}, x_out={x_out})."
        )

    collected_warnings: list[ModelWarning] = []

    def local_alpha(x: float) -> float:
        result = shah_condensation_alpha_local(
            x, p=p, p_critical=p_critical, G=G, D_i=D_i, mu_L=mu_L, k_L=k_L, cp_L=cp_L
        )
        collected_warnings.extend(result.warnings)
        return result.alpha

    alpha_effective = _gauss_legendre_5_mean(local_alpha, x_out, x_in)

    return CondensationZoneHTCResult(
        alpha_effective=alpha_effective,
        x_in=x_in,
        x_out=x_out,
        warnings=_deduplicate_warnings(collected_warnings),
    )


def _applicability_warnings(
    *, Re_LO: float, Pr_L: float, p_r: float, G: float, D_i: float
) -> list[ModelWarning]:
    warnings: list[ModelWarning] = []

    def _check(name: str, value: float, bounds: tuple[float, float], units: str) -> None:
        lower, upper = bounds
        if value < lower or value > upper:
            warnings.append(
                make_warning(
                    code="SHAH_CONDENSATION_OUT_OF_RANGE",
                    message=(
                        f"{name} = {value:.6g} {units} is outside the Shah (1979) "
                        f"correlation's documented applicability range "
                        f"[{lower:.6g}, {upper:.6g}] {units}; this result is an "
                        "extrapolation."
                    ),
                    source=SOURCE,
                    severity="warning",
                )
            )

    _check("Re_LO", Re_LO, SHAH_1979_LIQUID_REYNOLDS_RANGE, "-")
    _check("Pr_L", Pr_L, SHAH_1979_LIQUID_PRANDTL_RANGE, "-")
    _check("p_r", p_r, SHAH_1979_REDUCED_PRESSURE_RANGE, "-")
    _check("G", G, SHAH_1979_MASS_FLUX_RANGE_KG_M2S, "kg/(m2*s)")
    _check("D_i", D_i, SHAH_1979_TUBE_DIAMETER_RANGE_M, "m")

    return warnings


def _deduplicate_warnings(warnings: list[ModelWarning]) -> list[ModelWarning]:
    seen: dict[tuple[str, str], ModelWarning] = {}
    for warning in warnings:
        seen.setdefault((warning.code, warning.message), warning)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Fixed 5-point Gauss-Legendre quadrature (no external numerical
# dependency). Nodes/weights are the standard tabulated values on [-1, 1].
# ---------------------------------------------------------------------------
_GAUSS_LEGENDRE_5_NODES = (
    -0.906179845938664,
    -0.5384693101056831,
    0.0,
    0.5384693101056831,
    0.906179845938664,
)
_GAUSS_LEGENDRE_5_WEIGHTS = (
    0.2369268850561891,
    0.4786286704993665,
    0.5688888888888889,
    0.4786286704993665,
    0.2369268850561891,
)


def _gauss_legendre_5_mean(f, a: float, b: float) -> float:
    """Mean value of ``f`` over ``[a, b]`` via 5-point Gauss-Legendre quadrature.

    Every node is strictly inside ``(a, b)``, which is what makes this
    quadrature safe for integrands with endpoint degeneracies (such as the
    Shah correlation at x=0/x=1): the integrand is never evaluated exactly
    on the boundary.
    """
    half_width = 0.5 * (b - a)
    midpoint = 0.5 * (a + b)
    integral = 0.0
    for node, weight in zip(_GAUSS_LEGENDRE_5_NODES, _GAUSS_LEGENDRE_5_WEIGHTS):
        integral += weight * f(midpoint + half_width * node)
    integral *= half_width
    return integral / (b - a)
