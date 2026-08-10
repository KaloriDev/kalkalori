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
# - p [Pa], G [kg/(m^2*s)], D_i [m], g [m/s^2]
# - mu_L, mu_G [Pa*s], k_L [W/(m*K)], cp_L [J/(kg*K)]
# - rho_L, rho_G [kg/m^3]
# Outputs:
# - alpha [W/(m^2*K)]

"""
In-tube condensation heat transfer -- Shah (2009) (v0.6.2 patch).

Production default for pure-steam in-tube condensation, replacing the
original Shah (1979) correlation (``core.heat_transfer.condensation_inside``,
kept as a legacy/reference implementation, still directly unit-tested and
still importable) as the model actually used by
``core.phase_change.inside_pure_steam_condensation``.

Why the 1979 correlation is not enough on its own
---------------------------------------------------
Shah (1979) is a single, purely forced-convective correlation: it scales
the local coefficient off an all-liquid Dittus-Boelter-type reference
``h_LO = 0.023 Re_LO^0.8 Pr_L^0.4 (k_L/D)``. At low mass flux, ``Re_LO``
falls into the laminar/transitional range, and a turbulent single-phase
correlation extrapolated there systematically underpredicts -- forced
convective shear is genuinely weak at low G, but condensation itself does
not stop: gravity-driven film drainage keeps transferring heat. Shah
(1979) has no gravity-film branch, so it cannot represent that regime; the
symptom is a condensation HTC that is unphysically small (confirmed by
reconstructing the full 1979 calculation chain against a real low-G
steam-heater case, see the v0.6.2 patch report).

The improved correlation
-------------------------
Shah (2009) is an explicit extension of the same author's 1979
correlation to the full range down to the laminar/Nusselt limit. It adds a
second, gravity-driven contribution (``h_Nu``, a Nusselt-type laminar
film-condensation estimate) and defines three heat-transfer regimes
(forced-convective only / combined / gravity-only) selected from two
dimensionless groups (``Jg``, the dimensionless vapor velocity, and ``Z``,
Shah's own correlating parameter, reused unchanged from 1979). Tube
orientation matters here: vertical (downward-flowing) and inclined
(>= 15 degrees downward) tubes get all three regimes (including the
pure-gravity Regime III, since Nusselt's laminar-film solution is only
analytically exact for a draining vertical/inclined film); horizontal
tubes only have Regimes I and II -- the paper explicitly states data for a
horizontal Regime III were not available ("A third regime is expected at
very low flow rates. Analyzable data were not available for such
conditions."), so this module does not invent one.

Equations 1-3 and their applicability limits are unchanged from 1979; this
module reproduces only Shah's *new* 2009 equations 4-12, using the
published coefficients exactly, with no calibration factors.

Ref: Shah, M.M. (2009). "An Improved and Extended General Correlation for
Heat Transfer During Condensation in Plain Tubes." HVAC&R Research,
15(5), 889-913. DOI 10.1080/10789669.2009.10390871.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.common.warnings import ModelWarning, make_warning
from core.geometry.tube import TubeOrientation
from core.heat_transfer.condensation_inside import _gauss_legendre_5_mean

SOURCE = "heat_transfer.condensation_inside_shah2009"

GRAVITY_M_S2 = 9.80665

CORRELATION_NAME = "shah_2009"

# Table 4 (Shah 2009): "Complete Range of Parameters in the Data Showing
# Satisfactory Agreement with the Present Correlation" -- common to both
# horizontal and vertical/inclined tubes.
SHAH_2009_TUBE_DIAMETER_RANGE_M = (0.002, 0.049)
SHAH_2009_REDUCED_PRESSURE_RANGE = (0.0008, 0.905)
SHAH_2009_MASS_FLUX_RANGE_KG_M2S = (4.0, 820.0)
SHAH_2009_LIQUID_PRANDTL_RANGE = (1.0, 18.0)
SHAH_2009_LIQUID_REYNOLDS_RANGE = (68.0, 84_827.0)
SHAH_2009_QUALITY_RANGE = (0.01, 0.99)
SHAH_2009_Z_RANGE = (0.005, 20.0)
SHAH_2009_JG_RANGE = (0.06, 20.0)

# Concluding Remarks #1 and #3 (Shah 2009): "The present correlation has
# been shown to be applicable to vertical tubes at all flow rates and to
# horizontal tubes down to ReGT >= 16,000 ... Further research is needed
# for validating/extending this correlation to horizontal and slightly
# inclined tubes at ReGT < 16,000. Analyzable data from earlier studies
# are not available." (consistent with the body text's own empirical
# finding, "Analyzable data for horizontal tubes were available only for
# ReGT >= 15,800"; 16,000 is the paper's own rounded headline figure, used
# here). Vertical/inclined tubes have no such lower bound -- Regime III
# (the Nusselt gravity limit) covers arbitrarily low flow there.
SHAH_2009_HORIZONTAL_MIN_VAPOR_REYNOLDS = 16_000.0


@dataclass(frozen=True)
class Shah2009HTCResult:
    """Local Shah (2009) in-tube condensation heat-transfer result.

    Attributes:
        alpha: Local condensation heat transfer coefficient [W/(m2*K)].
        regime: Selected heat-transfer regime, ``"I"`` (forced-convective
            only, Eq. 10), ``"II"`` (combined forced + gravity, Eq. 11) or
            ``"III"`` (gravity-only/Nusselt, Eq. 12, vertical/inclined
            tubes only).
        h_I: Forced-convective contribution [W/(m2*K)] (Eq. 8a).
        h_Nu: Gravity-film (Nusselt-type) contribution [W/(m2*K)]
            (Eq. 9) -- also usable on its own as an independent
            gravity-film order-of-magnitude sanity check.
        Jg: Dimensionless vapor velocity [-] (Eq. 6).
        Z: Shah's correlating parameter [-], ``((1-x)/x)^0.8 * p_r^0.4``.
        Re_LT: All-liquid Reynolds number, ``G*D_i/mu_L`` [-].
        Re_LS: Liquid-alone Reynolds number, ``G*(1-x)*D_i/mu_L`` [-].
        Re_GT: All-vapor Reynolds number, ``G*D_i/mu_G`` [-].
        Pr_L: Saturated-liquid Prandtl number [-].
        p_r: Reduced pressure, ``p/p_critical`` [-].
        n: Viscosity-ratio exponent [-] (Eq. 8b), ``0.0058 + 0.557*p_r``.
        warnings: Applicability warnings (empty when fully inside the
            documented range).
    """

    alpha: float
    regime: str
    h_I: float
    h_Nu: float
    Jg: float
    Z: float
    Re_LT: float
    Re_LS: float
    Re_GT: float
    Pr_L: float
    p_r: float
    n: float
    warnings: list[ModelWarning]


def shah2009_condensation_alpha_local(
    x: float,
    *,
    p: float,
    p_critical: float,
    G: float,
    D_i: float,
    orientation: TubeOrientation,
    mu_L: float,
    mu_G: float,
    k_L: float,
    cp_L: float,
    rho_L: float,
    rho_G: float,
) -> Shah2009HTCResult:
    """Local in-tube condensation heat transfer coefficient (Shah, 2009).

    Z = ((1-x)/x)^0.8 * p_r^0.4                                     (Shah's parameter, 1979)
    Jg = x*G / sqrt(g*D_i*rho_G*(rho_L-rho_G))                      (6)
    h_LT = 0.023 * Re_LT^0.8 * Pr_L^0.4 * (k_L/D_i)                 (Eq. 2, ReLT for ReLS)
    n = 0.0058 + 0.557*p_r                                          (8b)
    h_I = h_LT * (mu_L/(14*mu_G))^n
          * [(1-x)^0.8 + 3.8*x^0.76*(1-x)^0.04/p_r^0.38]            (8a)
    h_Nu = 1.32 * Re_LS^(-1/3)
           * [rho_L*(rho_L-rho_G)*g*k_L^3/mu_L^2]^(1/3)             (9)

    Regime selection (vertical downflow / inclined downward >= 15 deg):
        Jg >= 1/(2.4*Z+0.73)                        -> Regime I,  h_TP = h_I           (4, 10)
        Jg <= 0.89-0.93*exp(-0.087*Z^-1.17)          -> Regime III, h_TP = h_Nu         (5, 12)
        otherwise                                    -> Regime II, h_TP = h_I + h_Nu    (11)

    Regime selection (horizontal):
        Jg >= 0.98*(Z+0.263)^-0.62                   -> Regime I,  h_TP = h_I           (7, 10)
        otherwise                                     -> Regime II, h_TP = h_I + h_Nu    (11)
        (Regime III is not defined for horizontal tubes -- see module docstring.)

    This is a local, point-quality correlation; see
    ``condensation_zone_alpha_effective_2009`` for the zone-averaged value
    a 0D condensing-zone duty should actually use.

    Args:
        x: Local vapor quality [-], strictly inside (0, 1).
        p: Pressure [Pa].
        p_critical: Critical pressure of the condensing fluid [Pa].
        G: Total (vapor + liquid) mass flux [kg/(m2*s)].
        D_i: Tube inner diameter [m].
        orientation: Tube orientation (required -- see
            ``core.geometry.tube.TubeOrientation``; upward flow is out of
            scope for Shah 2009, see module docstring).
        mu_L, mu_G: Saturated-liquid/-vapor dynamic viscosity at `p` [Pa*s].
        k_L: Saturated-liquid thermal conductivity at `p` [W/(m*K)].
        cp_L: Saturated-liquid specific heat at `p` [J/(kg*K)].
        rho_L, rho_G: Saturated-liquid/-vapor density at `p` [kg/m3].

    Returns:
        Shah2009HTCResult.

    Ref: Shah, M.M. (2009), HVAC&R Research, 15(5), 889-913.
    """
    if not (0.0 < x < 1.0):
        raise ValueError(
            "shah2009_condensation_alpha_local is a local, point-quality "
            "correlation defined only for 0 < x < 1; it must not be "
            f"evaluated exactly at x=0 or x=1 (got x={x}). Integrate over "
            "the quality range instead, e.g. via "
            "condensation_zone_alpha_effective_2009."
        )
    for name, value in (
        ("p", p),
        ("p_critical", p_critical),
        ("G", G),
        ("D_i", D_i),
        ("mu_L", mu_L),
        ("mu_G", mu_G),
        ("k_L", k_L),
        ("cp_L", cp_L),
        ("rho_L", rho_L),
        ("rho_G", rho_G),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")
    if rho_L <= rho_G:
        raise ValueError("rho_L must exceed rho_G (liquid denser than vapor).")

    Re_LT = G * D_i / mu_L
    Re_LS = G * (1.0 - x) * D_i / mu_L
    Re_GT = G * D_i / mu_G
    Pr_L = cp_L * mu_L / k_L
    p_r = p / p_critical
    Z = ((1.0 - x) / x) ** 0.8 * p_r**0.4
    Jg = (x * G) / math.sqrt(GRAVITY_M_S2 * D_i * rho_G * (rho_L - rho_G))

    h_LT = 0.023 * Re_LT**0.8 * Pr_L**0.4 * (k_L / D_i)
    n = 0.0058 + 0.557 * p_r
    h_I = (
        h_LT
        * (mu_L / (14.0 * mu_G)) ** n
        * ((1.0 - x) ** 0.8 + 3.8 * x**0.76 * (1.0 - x) ** 0.04 / p_r**0.38)
    )
    h_Nu = 1.32 * Re_LS ** (-1.0 / 3.0) * (
        rho_L * (rho_L - rho_G) * GRAVITY_M_S2 * k_L**3 / mu_L**2
    ) ** (1.0 / 3.0)

    if orientation is TubeOrientation.HORIZONTAL:
        regime_I_boundary = 0.98 * (Z + 0.263) ** (-0.62)
        if Jg >= regime_I_boundary:
            alpha, regime = h_I, "I"
        else:
            alpha, regime = h_I + h_Nu, "II"
    else:
        regime_I_boundary = 1.0 / (2.4 * Z + 0.73)
        regime_III_boundary = 0.89 - 0.93 * math.exp(-0.087 * Z ** (-1.17))
        if Jg >= regime_I_boundary:
            alpha, regime = h_I, "I"
        elif Jg <= regime_III_boundary:
            alpha, regime = h_Nu, "III"
        else:
            alpha, regime = h_I + h_Nu, "II"

    warnings = _applicability_warnings(
        Re_LT=Re_LT, Pr_L=Pr_L, p_r=p_r, G=G, D_i=D_i, x=x, Z=Z, Jg=Jg,
        Re_GT=Re_GT, orientation=orientation,
    )

    return Shah2009HTCResult(
        alpha=alpha,
        regime=regime,
        h_I=h_I,
        h_Nu=h_Nu,
        Jg=Jg,
        Z=Z,
        Re_LT=Re_LT,
        Re_LS=Re_LS,
        Re_GT=Re_GT,
        Pr_L=Pr_L,
        p_r=p_r,
        n=n,
        warnings=warnings,
    )


@dataclass(frozen=True)
class CondensationZoneHTCResult2009:
    """Quality-averaged effective Shah (2009) condensation HTC over a zone."""

    alpha_effective: float
    x_in: float
    x_out: float
    warnings: list[ModelWarning]


def condensation_zone_alpha_effective_2009(
    *,
    x_in: float,
    x_out: float,
    p: float,
    p_critical: float,
    G: float,
    D_i: float,
    orientation: TubeOrientation,
    mu_L: float,
    mu_G: float,
    k_L: float,
    cp_L: float,
    rho_L: float,
    rho_G: float,
) -> CondensationZoneHTCResult2009:
    """Quality-averaged Shah (2009) condensation HTC over ``[x_out, x_in]``.

    Shah (2009) is, like 1979, a local (point-quality) correlation with no
    separate "mean over a quality range" formula of its own -- the paper's
    own validation against mean-HTC data uses the arithmetic mean quality
    as a stand-in local evaluation point for that purpose ("Such data were
    analyzed by using the arithmetic average quality in calculations.").
    This module instead integrates the local correlation itself (fixed
    5-point Gauss-Legendre quadrature, matching the existing 1979 zone
    treatment in ``core.heat_transfer.condensation_inside``): more
    accurate than a single arithmetic-mean-quality evaluation whenever the
    regime, and therefore the functional form of ``h_TP(x)``, changes
    partway across the zone, and consistent with how the rest of KalKalori
    already treats zone-averaged local coefficients. All quadrature nodes
    are strictly interior to ``(x_out, x_in)``, so the local correlation
    is never evaluated exactly at x=0 or x=1.

    Args:
        x_in: Zone inlet vapor quality [-], the larger of the two.
        x_out: Zone outlet vapor quality [-], `0 <= x_out < x_in <= 1`.
        Other args: see `shah2009_condensation_alpha_local`.

    Returns:
        CondensationZoneHTCResult2009.
    """
    if not (0.0 <= x_out < x_in <= 1.0):
        raise ValueError(
            "condensation_zone_alpha_effective_2009 requires "
            f"0 <= x_out < x_in <= 1 (got x_in={x_in}, x_out={x_out})."
        )

    collected_warnings: list[ModelWarning] = []

    def local_alpha(x: float) -> float:
        result = shah2009_condensation_alpha_local(
            x, p=p, p_critical=p_critical, G=G, D_i=D_i, orientation=orientation,
            mu_L=mu_L, mu_G=mu_G, k_L=k_L, cp_L=cp_L, rho_L=rho_L, rho_G=rho_G,
        )
        collected_warnings.extend(result.warnings)
        return result.alpha

    alpha_effective = _gauss_legendre_5_mean(local_alpha, x_out, x_in)

    return CondensationZoneHTCResult2009(
        alpha_effective=alpha_effective,
        x_in=x_in,
        x_out=x_out,
        warnings=_deduplicate_warnings(collected_warnings),
    )


def _applicability_warnings(
    *,
    Re_LT: float,
    Pr_L: float,
    p_r: float,
    G: float,
    D_i: float,
    x: float,
    Z: float,
    Jg: float,
    Re_GT: float,
    orientation: TubeOrientation,
) -> list[ModelWarning]:
    warnings: list[ModelWarning] = []

    def _check(name: str, value: float, bounds: tuple[float, float], units: str) -> None:
        lower, upper = bounds
        if value < lower or value > upper:
            warnings.append(
                make_warning(
                    code="SHAH_2009_CONDENSATION_OUT_OF_RANGE",
                    message=(
                        f"{name} = {value:.6g} {units} is outside the Shah (2009) "
                        f"correlation's documented applicability range "
                        f"[{lower:.6g}, {upper:.6g}] {units}; this result is an "
                        "extrapolation."
                    ),
                    source=SOURCE,
                    severity="warning",
                )
            )

    _check("Re_LT", Re_LT, SHAH_2009_LIQUID_REYNOLDS_RANGE, "-")
    _check("Pr_L", Pr_L, SHAH_2009_LIQUID_PRANDTL_RANGE, "-")
    _check("p_r", p_r, SHAH_2009_REDUCED_PRESSURE_RANGE, "-")
    _check("G", G, SHAH_2009_MASS_FLUX_RANGE_KG_M2S, "kg/(m2*s)")
    _check("D_i", D_i, SHAH_2009_TUBE_DIAMETER_RANGE_M, "m")
    _check("x", x, SHAH_2009_QUALITY_RANGE, "-")
    _check("Z", Z, SHAH_2009_Z_RANGE, "-")
    _check("Jg", Jg, SHAH_2009_JG_RANGE, "-")

    if orientation is TubeOrientation.HORIZONTAL and Re_GT < SHAH_2009_HORIZONTAL_MIN_VAPOR_REYNOLDS:
        warnings.append(
            make_warning(
                code="SHAH_2009_HORIZONTAL_LOW_REGT_EXTRAPOLATION",
                message=(
                    f"Re_GT = {Re_GT:.6g} is below the horizontal-tube lower "
                    f"bound ({SHAH_2009_HORIZONTAL_MIN_VAPOR_REYNOLDS:.6g}) Shah "
                    "(2009) validated data against ('Concluding Remarks': "
                    "'applicable ... to horizontal tubes down to ReGT >= "
                    "16,000 ... Further research is needed for ... horizontal "
                    "... tubes at ReGT < 16,000. Analyzable data from earlier "
                    "studies are not available.'). Regime I/II are still "
                    "evaluated and returned (no third, gravity-only regime is "
                    "defined for horizontal tubes at all), but this result is "
                    "an extrapolation beyond the correlation's own validated "
                    "range; consider a vertical/inclined-downward tube "
                    "orientation, for which Shah (2009) is validated at all "
                    "flow rates via Regime III."
                ),
                source=SOURCE,
                severity="warning",
            )
        )

    return warnings


def _deduplicate_warnings(warnings: list[ModelWarning]) -> list[ModelWarning]:
    seen: dict[tuple[str, str], ModelWarning] = {}
    for warning in warnings:
        seen.setdefault((warning.code, warning.message), warning)
    return list(seen.values())
