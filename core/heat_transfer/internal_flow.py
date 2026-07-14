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
# - m_dot [kg/s], D [m], A [m^2]
# - rho [kg/m^3], mu [Pa*s], k [W/(m*K)], cp [J/(kg*K)]
# Outputs:
# - Re [-], Pr [-], Nu [-], alfa [W/(m^2*K)]

"""
Internal (tube-side) convective heat transfer and flow regime helpers
for smooth circular tubes.

This module provides:
- Reynolds and Prandtl numbers,
- Darcy friction factor correlations for smooth tubes,
- Nusselt number correlations for laminar and turbulent regimes,
- a convenience function returning alfa (heat transfer coefficient).

Theory references
-----------------
1. Incropera, F. P., DeWitt, D. P., Bergman, T. L., Lavine, A. S.
   Fundamentals of Heat and Mass Transfer, Wiley.

2. Gnielinski, V. (1976).
   New equations for heat and mass transfer in turbulent pipe and channel flow.
   International Chemical Engineering, 16(2), 359–368.

3. Petukhov, B. S. (1970).
   Heat transfer and friction in turbulent pipe flow with variable physical properties.
   Advances in Heat Transfer, Vol. 6. Also reported in Kays, W.M., Crawford, M.E.,
   Weigand, B., Convective Heat and Mass Transfer, and in VDI Heat Atlas,
   Section G1 (gas property-variation correction).

Notes
-----
- Laminar Nusselt: Nu = 3.66 corresponds to fully developed laminar flow
  in a circular tube with constant wall temperature.
- Turbulent Nusselt: Gnielinski correlation is used with a smooth-tube
  friction factor.
- Transitional regime is handled by linear blending in Re between 2300 and 4000.
- Optional turbulent-gas wall-property correction (Petukhov 1970, see
  ``gas_wall_temperature_correction``) is applied only when the caller
  supplies bulk/wall temperatures; it is independent of the outside
  Zukauskas (Pr/Pr_s) wall correction in ``outside_flow.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.common.warnings import ModelWarning, make_warning


@dataclass(frozen=True)
class FluidProps:
    """
    Minimal set of thermophysical properties required for tube-side correlations.
    """
    rho: float  # [kg/m^3]
    mu: float   # [Pa*s]
    k: float    # [W/(m*K)]
    cp: float   # [J/(kg*K)]


def reynolds_number(rho: float, v: float, D: float, mu: float) -> float:
    """
    Reynolds number Re = rho * v * D / mu.
    """
    if rho <= 0.0 or mu <= 0.0 or D <= 0.0:
        raise ValueError("rho, mu and D must be positive.")
    return rho * v * D / mu


def prandtl_number(cp: float, mu: float, k: float) -> float:
    """
    Prandtl number Pr = cp * mu / k.
    """
    if cp <= 0.0 or mu <= 0.0 or k <= 0.0:
        raise ValueError("cp, mu and k must be positive.")
    return cp * mu / k


def mean_velocity(m_dot: float, rho: float, flow_area: float) -> float:
    """
    Mean velocity v = m_dot / (rho * A).
    """
    if m_dot <= 0.0 or rho <= 0.0 or flow_area <= 0.0:
        raise ValueError("m_dot, rho and flow_area must be positive.")
    return m_dot / (rho * flow_area)


def friction_factor_smooth(Re: float) -> float:
    """
    Darcy friction factor for smooth tubes.

    - Laminar (Re < 2300): f = 64 / Re
    - Turbulent (Re >= 2300): Petukhov-type explicit approximation
      used in conjunction with Gnielinski.

    Returns
    -------
    f : float
        Darcy friction factor [-]
    """
    if Re <= 0.0:
        raise ValueError("Re must be positive.")

    if Re < 2300.0:
        return 64.0 / Re

    # Petukhov (explicit form) for smooth tubes; commonly used with Gnielinski:
    # f = [0.79*ln(Re) - 1.64]^-2  (valid roughly for 3e3 < Re < 5e6)
    return 1.0 / (0.79 * math.log(Re) - 1.64) ** 2


def nusselt_laminar_fully_developed_const_wall_temp() -> float:
    """
    Fully developed laminar flow in a circular tube, constant wall temperature.
    """
    return 3.66


# Note:
# This internal-flow Gnielinski Nusselt correlation is independent of
# the Gaddis–Gnielinski Euler-number correlation used for outside
# tube-bank pressure drop (see GaddisGnielinskiEulerProvider in
# outside_pressure_drop.py). Both are named after Gnielinski but are
# separate, independent correlations for different physical quantities.
def nusselt_gnielinski(Re: float, Pr: float) -> float:
    """
    Gnielinski correlation for turbulent flow in smooth tubes (internal
    tube-side heat transfer).

    Validity (typical):
    - 3000 < Re < 5e6
    - 0.5 < Pr < 2000

    Nu = (f/8)*(Re-1000)*Pr / [1 + 12.7*sqrt(f/8)*(Pr^(2/3)-1)]
    """
    if Re <= 0.0 or Pr <= 0.0:
        raise ValueError("Re and Pr must be positive.")

    f = friction_factor_smooth(Re)

    numerator = (f / 8.0) * (Re - 1000.0) * Pr
    denom = 1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0)

    return numerator / denom


def nusselt_internal(Re: float, Pr: float) -> float:
    """
    Nusselt number for internal flow in a smooth circular tube.

    - Laminar: Nu = 3.66
    - Turbulent: Gnielinski
    - Transitional: linear blend between Re=2300 and Re=4000

    Returns
    -------
    Nu : float
        Nusselt number [-]
    """
    if Re < 2300.0:
        return nusselt_laminar_fully_developed_const_wall_temp()

    if Re > 4000.0:
        return nusselt_gnielinski(Re, Pr)

    # Transitional blend (simple, robust MVP approach)
    Nu_lam = nusselt_laminar_fully_developed_const_wall_temp()
    Nu_turb = nusselt_gnielinski(4000.0, Pr)
    w = (Re - 2300.0) / (4000.0 - 2300.0)
    return (1.0 - w) * Nu_lam + w * Nu_turb


# Applicability bounds for the turbulent-gas wall-property correction below.
_GAS_WALL_CORR_TW_TB_LOWER_GUARD = 0.5  # gas-cooling extrapolation guard
_GAS_WALL_CORR_TW_TB_UPPER = 2.4        # gas-heating upper bound (Petukhov 1970)

# Regime above which the Gnielinski correlation (and this wall correction,
# which is only documented for turbulent flow) applies.
_GAS_WALL_CORR_RE_MIN = 4000.0


def gas_wall_temperature_correction(
    T_bulk: float,
    T_wall: float,
) -> tuple[float, list[ModelWarning]]:
    """
    Variable-property (bulk-to-wall temperature) correction factor for
    turbulent internal *gas* flow, to be applied on top of the
    constant-property Gnielinski Nusselt number.

        Nu / Nu_cp = (T_bulk / T_wall)^n      (T in absolute units, K)

            n = 0                              for T_wall/T_bulk < 1 (gas cooling)
            n = -log10(T_wall/T_bulk)/4 + 0.3   for 1 <= T_wall/T_bulk <= 2.4 (gas heating)

    Reference:
        Petukhov, B. S. (1970). Heat transfer and friction in turbulent pipe
        flow with variable physical properties. Advances in Heat Transfer,
        Vol. 6. Also reported in Kays, Crawford, Weigand, Convective Heat
        and Mass Transfer, and VDI Heat Atlas, Section G1.

    This is independent of the outside crossflow (Pr/Pr_s)^0.25 wall
    correction used in ``outside_flow.nusselt_zukauskas`` -- that correction
    uses a Prandtl-number ratio appropriate for the Zukauskas tube-bank
    correlation, while this one uses the absolute-temperature ratio
    appropriate for internal turbulent gas flow.

    Returns
    -------
    (factor, warnings) : tuple[float, list[ModelWarning]]
        Multiplicative correction factor for Nu, and any applicability
        warnings (empty list when within the documented range).
    """
    warnings_list: list[ModelWarning] = []

    if (
        not math.isfinite(T_bulk)
        or not math.isfinite(T_wall)
        or T_bulk <= 0.0
        or T_wall <= 0.0
    ):
        warnings_list.append(
            make_warning(
                code="tube_ht_gas_wall_correction_invalid_temperature",
                message=(
                    "tube_ht: T_bulk/T_wall must be finite, positive absolute "
                    "temperatures [K]; gas wall-property correction skipped "
                    "(factor=1)."
                ),
                source="tube_ht",
                severity="warning",
            )
        )
        return 1.0, warnings_list

    ratio = T_wall / T_bulk

    if ratio < 1.0:
        # Gas cooling (wall colder than bulk): Petukhov (1970) reports no
        # reduction is needed for this side.
        n = 0.0
        if ratio < _GAS_WALL_CORR_TW_TB_LOWER_GUARD:
            warnings_list.append(
                make_warning(
                    code="tube_ht_gas_wall_correction_applicability_exceeded",
                    message=(
                        f"tube_ht: T_wall/T_bulk={ratio:.3g} indicates an "
                        "unusually large gas-cooling temperature difference "
                        "for the current 0D turbulent-gas wall correction; "
                        "verify applicability."
                    ),
                    source="tube_ht",
                    severity="warning",
                )
            )
    else:
        # Gas heating (wall hotter than bulk).
        n = -math.log10(ratio) / 4.0 + 0.3
        if ratio > _GAS_WALL_CORR_TW_TB_UPPER:
            warnings_list.append(
                make_warning(
                    code="tube_ht_gas_wall_correction_applicability_exceeded",
                    message=(
                        f"tube_ht: T_wall/T_bulk={ratio:.3g} exceeds the "
                        f"Petukhov (1970) gas-heating correction's reported "
                        f"range (<= {_GAS_WALL_CORR_TW_TB_UPPER:g}); result is "
                        "an extrapolation."
                    ),
                    source="tube_ht",
                    severity="warning",
                )
            )

    factor = (T_bulk / T_wall) ** n
    return factor, warnings_list


def heat_transfer_coefficient_internal(
    m_dot: float,
    tube_inner_diameter: float,
    flow_area: float,
    props: FluidProps,
    *,
    T_bulk: float | None = None,
    T_wall: float | None = None,
) -> tuple[float, float, float, float, list[ModelWarning]]:
    """
    Convenience function returning tube-side alfa and key dimensionless groups.

    Parameters
    ----------
    m_dot : float
        Mass flow rate [kg/s].
    tube_inner_diameter : float
        Inner diameter D_i [m].
    flow_area : float
        Flow cross-sectional area [m^2] (e.g., N_tubes * pi*D_i^2/4).
    props : FluidProps
        Thermophysical properties at representative bulk conditions.
    T_bulk, T_wall : float, optional
        Bulk and tube-wall (inside surface) temperatures [K]. When both are
        supplied and the regime is turbulent (Re > 4000), the turbulent-gas
        wall-property correction (``gas_wall_temperature_correction``) is
        applied to Nu. When omitted (the default), the base Gnielinski
        result is returned unchanged -- this preserves backward
        compatibility for callers that do not supply wall state.

    Returns
    -------
    v : float
        Mean velocity [m/s]
    Re : float
        Reynolds number [-]
    Pr : float
        Prandtl number [-]
    alfa : float
        Internal convective heat transfer coefficient [W/(m^2*K)]
    warnings : list[ModelWarning]
        Applicability warnings for the wall-property correction (empty when
        no wall state is supplied or the correction is within range).
    """
    if tube_inner_diameter <= 0.0:
        raise ValueError("tube_inner_diameter must be positive.")

    v = mean_velocity(m_dot, props.rho, flow_area)
    Re = reynolds_number(props.rho, v, tube_inner_diameter, props.mu)
    Pr = prandtl_number(props.cp, props.mu, props.k)

    Nu = nusselt_internal(Re, Pr)

    warnings_list: list[ModelWarning] = []

    if T_wall is not None:
        if Re <= _GAS_WALL_CORR_RE_MIN:
            warnings_list.append(
                make_warning(
                    code="tube_ht_gas_wall_correction_not_applicable_regime",
                    message=(
                        "tube_ht: turbulent-gas wall-property correction is "
                        f"only defined for Re > {_GAS_WALL_CORR_RE_MIN:g}; "
                        "current regime is laminar/transitional, so the "
                        "correction is skipped."
                    ),
                    source="tube_ht",
                    severity="info",
                )
            )
        elif T_bulk is None:
            warnings_list.append(
                make_warning(
                    code="tube_ht_gas_wall_correction_unavailable",
                    message=(
                        "tube_ht: T_wall was supplied without T_bulk; "
                        "gas wall-property correction requires both and is "
                        "skipped (factor=1)."
                    ),
                    source="tube_ht",
                    severity="info",
                )
            )
        else:
            factor, corr_warnings = gas_wall_temperature_correction(T_bulk, T_wall)
            Nu = Nu * factor
            warnings_list.extend(corr_warnings)

    alfa = Nu * props.k / tube_inner_diameter

    return v, Re, Pr, alfa, warnings_list
