# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
#
# -------------------------------------------------------------------------
# OUTSIDE FLOW – CROSSFLOW OVER TUBE BANK (0D CORE MODEL)
# -------------------------------------------------------------------------
#
# This module implements engineering correlations for:
#   • Heat transfer coefficient: Zukauskas-type Nu(Re,Pr) with:
#       - Re-range constants (C,m)
#       - finite-row correction C2(N_L)
#       - optional (Pr/Pr_s)^0.25 correction
#   • Geometry-consistent velocity definitions:
#       V_inf (approach) and V_max based on minimum free-flow area concept
#   • Pressure-drop integration through outside_pressure_drop dispatcher:
#       selectable euler_provider = "zukauskas" | "kern" | "esdu" | custom provider
#
# All correlations used here are taken from OPEN LITERATURE sources:
#
# Heat transfer (tube banks in crossflow):
#   - Zukauskas, A. (1972), "Heat Transfer from Tubes in Crossflow"
#   - Incropera et al., Fundamentals of Heat and Mass Transfer
#   - VDI Heat Atlas (Tube Banks in Crossflow)
#   - Khan, W.A. (2004), PhD Thesis, Univ. Waterloo (finite row correction)
#
# Pressure drop architecture:
#   - delegated to outside_pressure_drop.py
#
# NOTE:
#   External closed-data providers may be attached only through the
#   euler_provider interface, without importing proprietary code here.
#
# -------------------------------------------------------------------------
# UNITS (SI ONLY)
# -------------------------------------------------------------------------
# m_dot [kg/s]
# frontal_area [m^2]
# D, tube pitches [m]
# rho [kg/m^3]
# mu [Pa*s]
# k [W/(m*K)]
# cp [J/(kg*K)]
# v [m/s]
# alfa [W/(m^2*K)]
# dp [Pa]
# -------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import math

from core.common.warnings import ModelWarning, make_warning
from .outside_pressure_drop import (
    EulerProvider,
    EulerRequest,
    EulerResult,
    check_outside_dp_applicability,
    evaluate_euler,
    pressure_drop_from_euler,
)


# -------------------------------------------------------------------------
# Fluid properties container
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class FluidProps:
    rho: float  # [kg/m^3]
    mu: float   # [Pa*s]
    k: float    # [W/(m*K)]
    cp: float   # [J/(kg*K)]


# -------------------------------------------------------------------------
# Dimensionless numbers
# -------------------------------------------------------------------------

def reynolds_number(rho: float, v: float, D: float, mu: float) -> float:
    """
    Reynolds number definition (external flow over cylinder):

        Re_D = rho * v * D / mu

    Reference:
        Incropera et al., Fundamentals of Heat and Mass Transfer
    """
    if rho <= 0.0 or mu <= 0.0 or D <= 0.0 or v <= 0.0:
        raise ValueError("rho, mu, D, v must be positive.")
    return rho * v * D / mu


def prandtl_number(cp: float, mu: float, k: float) -> float:
    """
    Prandtl number:

        Pr = cp * mu / k

    Reference:
        Standard thermophysical definition (all heat transfer textbooks)
    """
    if cp <= 0.0 or mu <= 0.0 or k <= 0.0:
        raise ValueError("cp, mu, k must be positive.")
    return cp * mu / k


# -------------------------------------------------------------------------
# Geometry-consistent velocities (minimum free-flow area concept)
# -------------------------------------------------------------------------

def vmax_ratio_min_freeflow(
    tube_outer_diameter: float,
    S_T: float,
    S_L: float,
    layout: str,
) -> float:
    """
    Returns ratio (V_max / V_inf) based on minimum free-flow area concept.

    Inline:
        V_max / V_inf = S_T / (S_T - D)

    Staggered:
        Consider two candidate "gaps" per common tube-bank treatments:
          A_T = (S_T - D)             transverse gap
          A_D = 2*(S_D - D)           diagonal gap, where S_D = sqrt(S_L^2 + (S_T/2)^2)
        Use the minimum gap to represent minimum free-flow area (per unit depth),
        leading to:
          V_max / V_inf = S_T / min(A_T, A_D)

    NOTE:
      This is a 0D geometric velocity model. For higher-fidelity layouts,
      additional corrections may apply.
    """
    if tube_outer_diameter <= 0.0 or S_T <= 0.0 or S_L <= 0.0:
        raise ValueError("tube_outer_diameter, S_T, S_L must be positive.")
    if S_T <= tube_outer_diameter:
        raise ValueError("S_T must be > D to have a valid transverse flow gap.")

    A_T = S_T - tube_outer_diameter

    if layout == "inline":
        A_min = A_T
    elif layout == "staggered":
        S_D = math.sqrt(S_L * S_L + (S_T * 0.5) ** 2)
        if S_D <= tube_outer_diameter:
            raise ValueError("Invalid geometry: diagonal pitch S_D must be > D.")
        A_D = 2.0 * (S_D - tube_outer_diameter)
        A_min = min(A_T, A_D)
    else:
        raise ValueError("layout must be 'inline' or 'staggered'.")

    return S_T / A_min


# -------------------------------------------------------------------------
# Finite row correction (Zukauskas method)
# -------------------------------------------------------------------------

def _interp_1d(x: float, xs: list[float], ys: list[float]) -> float:
    """Clamped linear interpolation."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("xs and ys must have same length >= 2.")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def finite_row_correction_c2(n_rows: int) -> float:
    """
    Finite row correction factor C2(N_L).

    Tabulated anchor values based on:
        Khan (2004), PhD Thesis, Univ. Waterloo
        Zukauskas method summaries in literature

    For N_L >= 20 → C2 ≈ 1.0
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    xs = [1, 2, 3, 4, 5, 7, 10, 13, 16, 20]
    ys = [0.64, 0.76, 0.84, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00]

    return _interp_1d(float(n_rows), [float(v) for v in xs], ys)


# -------------------------------------------------------------------------
# Zukauskas heat transfer correlation
# -------------------------------------------------------------------------

def nusselt_zukauskas(
    Re: float,
    Pr: float,
    n_rows: int,
    *,
    Pr_s: float | None = None,
    apply_finite_row_correction: bool = True,
) -> float:
    """
    Zukauskas-type correlation for tube banks in crossflow:

        Nu = C * Re^m * Pr^0.36

    with:
        - Re-range dependent C, m
        - optional finite-row correction C2(N_L)
        - optional (Pr/Pr_s)^0.25 correction

    References:
        Zukauskas (1972)
        Incropera et al.
        VDI Heat Atlas
    """
    if Re <= 0.0 or Pr <= 0.0:
        raise ValueError("Re and Pr must be positive.")
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    if Re < 1e2:
        C, m = 0.90, 0.40
    elif Re < 1e3:
        C, m = 0.52, 0.50
    elif Re < 2e5:
        C, m = 0.27, 0.63
    else:
        C, m = 0.021, 0.84

    Nu = C * (Re ** m) * (Pr ** 0.36)

    if apply_finite_row_correction:
        Nu *= finite_row_correction_c2(n_rows)

    if Pr_s is not None:
        if Pr_s <= 0.0:
            raise ValueError("Pr_s must be positive.")
        Nu *= (Pr / Pr_s) ** 0.25

    return Nu


# -------------------------------------------------------------------------
# Heat-transfer applicability
# -------------------------------------------------------------------------

def check_outside_ht_applicability(
    Re: float,
    Pr: float,
    tube_outer_diameter: float,
    tube_pitch_transverse: float,
    tube_pitch_longitudinal: float,
    layout: str,
    n_rows: int,
    *,
    use_vmax_for_ht: bool = True,
) -> list[ModelWarning]:
    """
    Applicability / diagnostic checks for the outside heat-transfer model.
    """
    warnings: list[ModelWarning] = []

    if Re <= 0.0:
        warnings.append(
            make_warning(
                code="outside_ht_re_nonpositive",
                message="outside_ht: Reynolds number must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )
    if Pr <= 0.0:
        warnings.append(
            make_warning(
                code="outside_ht_pr_nonpositive",
                message="outside_ht: Prandtl number must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )
    if tube_outer_diameter <= 0.0:
        warnings.append(
            make_warning(
                code="outside_ht_diameter_nonpositive",
                message="outside_ht: tube_outer_diameter must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )
    if tube_pitch_transverse <= 0.0:
        warnings.append(
            make_warning(
                code="outside_ht_pitch_transverse_nonpositive",
                message="outside_ht: tube_pitch_transverse must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )
    if tube_pitch_longitudinal <= 0.0:
        warnings.append(
            make_warning(
                code="outside_ht_pitch_longitudinal_nonpositive",
                message="outside_ht: tube_pitch_longitudinal must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )
    if n_rows <= 0:
        warnings.append(
            make_warning(
                code="outside_ht_n_rows_nonpositive",
                message="outside_ht: n_rows must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )

    if (
        tube_outer_diameter <= 0.0
        or tube_pitch_transverse <= 0.0
        or tube_pitch_longitudinal <= 0.0
    ):
        return warnings

    ST_over_D = tube_pitch_transverse / tube_outer_diameter
    SL_over_D = tube_pitch_longitudinal / tube_outer_diameter

    if ST_over_D <= 1.0:
        warnings.append(
            make_warning(
                code="outside_ht_st_over_d_invalid",
                message="outside_ht: ST/D <= 1.0 is geometrically invalid for crossflow tube banks.",
                source="outside_ht",
                severity="critical",
            )
        )
    elif ST_over_D < 1.1:
        warnings.append(
            make_warning(
                code="outside_ht_st_over_d_near_blockage",
                message="outside_ht: ST/D is very close to blockage limit; results may be highly sensitive.",
                source="outside_ht",
                severity="warning",
            )
        )

    if SL_over_D <= 1.0:
        warnings.append(
            make_warning(
                code="outside_ht_sl_over_d_invalid",
                message="outside_ht: SL/D <= 1.0 is outside the intended geometry range.",
                source="outside_ht",
                severity="critical",
            )
        )
    elif SL_over_D < 1.1:
        warnings.append(
            make_warning(
                code="outside_ht_sl_over_d_small",
                message="outside_ht: SL/D is very small; wake interaction may be strong and correlation confidence is reduced.",
                source="outside_ht",
                severity="warning",
            )
        )

    if ST_over_D > 4.0:
        warnings.append(
            make_warning(
                code="outside_ht_st_over_d_large",
                message="outside_ht: ST/D is unusually large for the current 0D tube-bank model; verify applicability.",
                source="outside_ht",
                severity="warning",
            )
        )

    if SL_over_D > 4.0:
        warnings.append(
            make_warning(
                code="outside_ht_sl_over_d_large",
                message="outside_ht: SL/D is unusually large for the current 0D tube-bank model; verify applicability.",
                source="outside_ht",
                severity="warning",
            )
        )

    if Re > 0.0:
        if Re < 30.0:
            warnings.append(
                make_warning(
                    code="outside_ht_re_extremely_low",
                    message="outside_ht: Re is extremely low for tube-bank crossflow; current outside HT correlation is low-confidence.",
                    source="outside_ht",
                    severity="critical",
                )
            )
        elif Re < 1.0e2:
            warnings.append(
                make_warning(
                    code="outside_ht_re_low",
                    message="outside_ht: Re is in a very low range; confirm that the selected outside HT correlation is appropriate.",
                    source="outside_ht",
                    severity="warning",
                )
            )
        elif Re > 2.0e5:
            warnings.append(
                make_warning(
                    code="outside_ht_re_high",
                    message="outside_ht: Re exceeds the main mid-range branch of the current Zukauskas-style implementation; verify high-Re applicability.",
                    source="outside_ht",
                    severity="warning",
                )
            )

    if Pr > 0.0:
        if Pr < 0.6:
            warnings.append(
                make_warning(
                    code="outside_ht_pr_low",
                    message="outside_ht: very low Pr may be outside the intended use of the current gas-side correlation.",
                    source="outside_ht",
                    severity="warning",
                )
            )
        elif Pr > 500.0:
            warnings.append(
                make_warning(
                    code="outside_ht_pr_high",
                    message="outside_ht: very high Pr is outside the usual gas-side engineering range; verify applicability.",
                    source="outside_ht",
                    severity="warning",
                )
            )

    if n_rows == 1:
        warnings.append(
            make_warning(
                code="outside_ht_single_row",
                message="outside_ht: single-row bank; finite-row effects dominate and uncertainty is elevated.",
                source="outside_ht",
                severity="warning",
            )
        )
    elif n_rows < 5:
        warnings.append(
            make_warning(
                code="outside_ht_few_rows",
                message="outside_ht: very small number of rows; finite-row correction has strong influence on the result.",
                source="outside_ht",
                severity="warning",
            )
        )

    if not use_vmax_for_ht:
        warnings.append(
            make_warning(
                code="outside_ht_velocity_reference_nonstandard",
                message="outside_ht: HT is not using V_max as reference velocity; literature tube-bank correlations are commonly referenced to maximum gap velocity.",
                source="outside_ht",
                severity="info",
            )
        )

    if layout == "staggered" and SL_over_D < 1.2:
        warnings.append(
            make_warning(
                code="outside_ht_staggered_tight_sl",
                message="outside_ht: staggered layout with very tight longitudinal pitch may require more geometry-specific validation.",
                source="outside_ht",
                severity="warning",
            )
        )

    return warnings


# -------------------------------------------------------------------------
# Main 0D solver
# -------------------------------------------------------------------------

def outside_flow_from_mass_flow(
    m_dot: float,
    frontal_area: float,
    tube_outer_diameter: float,
    tube_pitch_transverse: float,
    tube_pitch_longitudinal: float,
    layout: str,
    n_rows: int,
    n_tubes_per_row: int,
    props: FluidProps,
    *,
    Pr_s: float | None = None,
    apply_finite_row_correction: bool = True,
    euler_provider: str | EulerProvider = "zukauskas",
    use_vmax_for_ht: bool = True,
    use_vmax_for_dp: bool = True,
    is_finned: bool = False,
    pressure_drop_geometry_meta: dict | None = None,
) -> tuple[float, float, float, float, float, list[ModelWarning], EulerResult]:
    """
    Outside forced convection for tube bank (0D core).

    Parameters
    ----------
    euler_provider:
        Either:
          - built-in provider name: "zukauskas", "kern", "esdu"
          - custom provider object implementing EulerProvider

    Returns
    -------
    tuple
        (v, Re, Pr, alfa_o, dp_o, warnings_list, euler_result)
        - v: approach velocity [m/s]
        - Re: heat-transfer Reynolds number [-]
        - Pr: Prandtl number [-]
        - alfa_o: outside heat transfer coefficient [W/(m^2*K)]
        - dp_o: pressure drop [Pa]
        - warnings_list: structured applicability warnings
        - euler_result: metadata about selected pressure-drop backend
    """

    if m_dot <= 0.0:
        raise ValueError("m_dot must be positive.")
    if frontal_area <= 0.0:
        raise ValueError("frontal_area must be positive.")
    if tube_outer_diameter <= 0.0:
        raise ValueError("tube_outer_diameter must be positive.")
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    if n_tubes_per_row <= 0:
        raise ValueError("n_tubes_per_row must be positive.")

    # Mass flow per tube in row
    m_dot_tube = m_dot / float(n_tubes_per_row)

    # Frontal area per tube in row
    frontal_area_per_tube = frontal_area / float(n_tubes_per_row)

    # Approach velocity (per tube)
    v = m_dot_tube / (props.rho * frontal_area_per_tube)

    # Geometry-consistent Vmax
    ratio = vmax_ratio_min_freeflow(
        tube_outer_diameter,
        tube_pitch_transverse,
        tube_pitch_longitudinal,
        layout,
    )
    V_max = v * ratio

    # Choose reference velocities
    V_ref_ht = V_max if use_vmax_for_ht else v
    V_ref_dp = V_max if use_vmax_for_dp else v

    # Heat transfer
    Re = reynolds_number(props.rho, V_ref_ht, tube_outer_diameter, props.mu)
    Pr = prandtl_number(props.cp, props.mu, props.k)

    Nu = nusselt_zukauskas(
        Re,
        Pr,
        n_rows,
        Pr_s=Pr_s,
        apply_finite_row_correction=apply_finite_row_correction,
    )
    alfa_o = Nu * props.k / tube_outer_diameter

    # Pressure drop
    ST_over_D = tube_pitch_transverse / tube_outer_diameter
    SL_over_D = tube_pitch_longitudinal / tube_outer_diameter
    Re_dp = reynolds_number(props.rho, V_ref_dp, tube_outer_diameter, props.mu)

    request = EulerRequest(
        Re=Re_dp,
        ST_over_D=ST_over_D,
        SL_over_D=SL_over_D,
        layout=layout,
        n_rows=n_rows,
        is_finned=is_finned,
        geometry_meta=pressure_drop_geometry_meta,
    )

    euler_result = evaluate_euler(
        request,
        euler_provider=euler_provider,
    )
    dp_o = pressure_drop_from_euler(props.rho, V_ref_dp, euler_result.Eu)

    # Applicability warnings
    warnings_list = check_outside_ht_applicability(
        Re,
        Pr,
        tube_outer_diameter,
        tube_pitch_transverse,
        tube_pitch_longitudinal,
        layout,
        n_rows,
        use_vmax_for_ht=use_vmax_for_ht,
    )

    warnings_list.extend(
        check_outside_dp_applicability(
            request,
            euler_provider=euler_provider,
            use_vmax_for_dp=use_vmax_for_dp,
        )
    )

    return v, Re, Pr, alfa_o, dp_o, warnings_list, euler_result