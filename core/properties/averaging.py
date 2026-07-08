"""
Property averaging helpers.

This module provides simple explicit helpers for evaluating properties at a
mean temperature. It intentionally does not perform numerical integration or
state-path averaging.

The MVP rule is:

    T_mean = 0.5 * (T1 + T2)
    props = provider.at(T_mean, p)

This is suitable as a clear first integration step for 0D calculations.
More advanced averaging can be introduced later without changing solver logic.
"""

from __future__ import annotations

import math

from core.properties.common import FluidTransportProperties
from core.properties.fluids import PropertyProvider


def mean_temperature(T1: float, T2: float) -> float:
    """Return arithmetic mean temperature.

    Args:
        T1: First temperature [K].
        T2: Second temperature [K].

    Returns:
        Mean temperature [K].
    """
    _validate_temperature(T1, "T1")
    _validate_temperature(T2, "T2")

    return 0.5 * (T1 + T2)


def mean_transport_props(
    provider: PropertyProvider,
    T1: float,
    T2: float,
    p: float,
) -> FluidTransportProperties:
    """Evaluate transport properties at arithmetic mean temperature.

    Args:
        provider: Property provider exposing `at(T, p)`.
        T1: First temperature [K].
        T2: Second temperature [K].
        p: Pressure [Pa].

    Returns:
        FluidTransportProperties evaluated at `0.5 * (T1 + T2)`.

    Notes:
        This helper intentionally evaluates properties at the arithmetic mean
        temperature only. It does not integrate properties along a temperature
        path.
    """
    _validate_pressure(p)

    T_mean = mean_temperature(T1, T2)
    return provider.at(T=T_mean, p=p)


def _validate_temperature(T: float, name: str) -> None:
    if not math.isfinite(T):
        raise ValueError(f"{name} must be finite [K].")
    if T <= 0.0:
        raise ValueError(f"{name} must be above absolute zero [K].")


def _validate_pressure(p: float) -> None:
    if not math.isfinite(p):
        raise ValueError("Pressure must be finite [Pa].")
    if p <= 0.0:
        raise ValueError("Pressure must be positive [Pa].")