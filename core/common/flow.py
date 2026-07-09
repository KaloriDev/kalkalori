"""
Common flow-rate helpers.

This module contains small, model-independent flow conversions used across
property tests, heat-transfer calculations, and pressure-drop calculations.

KalKalori core units:
- mass flow: kg/s
- volume flow: m3/s
- density: kg/m3

These helpers intentionally do not know anything about a specific fluid model.
They only convert between mass flow and actual volume flow using density at the
state point of interest.
"""

from __future__ import annotations

import math


def volume_flow_from_mass_flow(m_dot: float, rho: float) -> float:
    """Return actual volume flow [m3/s].

    Args:
        m_dot:
            Mass flow [kg/s].
        rho:
            Density [kg/m3] at the same state point.

    Returns:
        Actual volume flow [m3/s].

    Examples:
        >>> volume_flow_from_mass_flow(m_dot=1.2, rho=1.2)
        1.0
    """
    _validate_positive_finite("m_dot", m_dot)
    _validate_positive_finite("rho", rho)

    return m_dot / rho


def mass_flow_from_volume_flow(V_dot: float, rho: float) -> float:
    """Return mass flow [kg/s].

    Args:
        V_dot:
            Actual volume flow [m3/s].
        rho:
            Density [kg/m3] at the same state point.

    Returns:
        Mass flow [kg/s].
    """
    _validate_positive_finite("V_dot", V_dot)
    _validate_positive_finite("rho", rho)

    return V_dot * rho


def volume_flow_m3_h_from_mass_flow(m_dot: float, rho: float) -> float:
    """Return actual volume flow [m3/h] from mass flow [kg/s]."""
    return volume_flow_from_mass_flow(m_dot=m_dot, rho=rho) * 3600.0


def mass_flow_kg_h_from_volume_flow(V_dot: float, rho: float) -> float:
    """Return mass flow [kg/h] from actual volume flow [m3/s]."""
    return mass_flow_from_volume_flow(V_dot=V_dot, rho=rho) * 3600.0


def _validate_positive_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
