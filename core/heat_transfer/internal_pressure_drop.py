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

Methodology (MVP)
-----------------
Total pressure drop is decomposed into components:

    dp_total = dp_tubes + dp_inlet + dp_outlet + dp_turns

Each component is computed by a dedicated function to allow refinement later.

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
- Single-phase flow, smooth tubes.
- Turn losses are modeled via a lumped K_turn per 180° return (U-bend/return header).
  This is a practical MVP placeholder that can be replaced with geometry-specific models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FluidProps:
    rho: float  # [kg/m^3]
    mu: float   # [Pa*s]


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
