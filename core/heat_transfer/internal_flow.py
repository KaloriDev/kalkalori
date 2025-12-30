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
# - Re [-], Pr [-], Nu [-], h [W/(m^2*K)]

"""
Internal (tube-side) convective heat transfer and flow regime helpers
for smooth circular tubes.

This module provides:
- Reynolds and Prandtl numbers,
- Darcy friction factor correlations for smooth tubes,
- Nusselt number correlations for laminar and turbulent regimes,
- a convenience function returning h (heat transfer coefficient).

Theory references
-----------------
1. Incropera, F. P., DeWitt, D. P., Bergman, T. L., Lavine, A. S.
   Fundamentals of Heat and Mass Transfer, Wiley.

2. Gnielinski, V. (1976).
   New equations for heat and mass transfer in turbulent pipe and channel flow.
   International Chemical Engineering, 16(2), 359–368.

3. Petukhov, B. S. (1970).
   Heat transfer and friction in turbulent pipe flow with variable physical properties.
   Advances in Heat Transfer.

Notes
-----
- Laminar Nusselt: Nu = 3.66 corresponds to fully developed laminar flow
  in a circular tube with constant wall temperature.
- Turbulent Nusselt: Gnielinski correlation is used with a smooth-tube
  friction factor.
- Transitional regime is handled by linear blending in Re between 2300 and 4000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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


def nusselt_gnielinski(Re: float, Pr: float) -> float:
    """
    Gnielinski correlation for turbulent flow in smooth tubes.

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


def heat_transfer_coefficient_internal(
    m_dot: float,
    tube_inner_diameter: float,
    flow_area: float,
    props: FluidProps,
) -> tuple[float, float, float, float]:
    """
    Convenience function returning tube-side h and key dimensionless groups.

    Parameters
    ----------
    m_dot : float
        Mass flow rate [kg/s].
    tube_inner_diameter : float
        Inner diameter D_i [m].
    flow_area : float
        Flow cross-sectional area [m^2] (e.g., N_tubes * pi*D_i^2/4).
    props : FluidProps
        Thermophysical properties at representative conditions.

    Returns
    -------
    v : float
        Mean velocity [m/s]
    Re : float
        Reynolds number [-]
    Pr : float
        Prandtl number [-]
    h : float
        Internal convective heat transfer coefficient [W/(m^2*K)]
    """
    if tube_inner_diameter <= 0.0:
        raise ValueError("tube_inner_diameter must be positive.")

    v = mean_velocity(m_dot, props.rho, flow_area)
    Re = reynolds_number(props.rho, v, tube_inner_diameter, props.mu)
    Pr = prandtl_number(props.cp, props.mu, props.k)

    Nu = nusselt_internal(Re, Pr)
    h = Nu * props.k / tube_inner_diameter

    return v, Re, Pr, h
