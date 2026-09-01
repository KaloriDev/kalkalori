"""
Tabulated single-phase liquid property provider.

This module provides a manual-entry property path for single-phase liquids
whose properties are known at one or more temperature points (e.g. from a
datasheet), without requiring CoolProp or another external backend.

KalKalori core uses SI floats:
- temperature: K
- density: kg/m3
- dynamic viscosity: Pa*s
- thermal conductivity: W/(m*K)
- specific heat: J/(kg*K)
- specific enthalpy: J/kg

With one supplied point, properties are constant at every positive
temperature. With two or more points, rho/cp/k interpolate linearly and mu
interpolates log-linearly versus T; specific enthalpy is obtained by exactly
integrating the resulting piecewise-linear cp(T), with h = 0 at the lowest
supplied temperature. Querying a temperature outside the supplied table
raises ValueError -- this provider never extrapolates. Pressure is accepted
only for interface compatibility with other property providers; properties
here do not depend on it.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Sequence

from core.properties.common import FluidTransportProperties
from core.properties.coolprop_backend import CoolPropProperties


@dataclass(frozen=True)
class LiquidPropertyPoint:
    """One manually supplied single-phase liquid property point.

    Attributes:
        T: Temperature [K].
        rho: Density [kg/m3].
        cp: Specific heat capacity [J/(kg*K)].
        mu: Dynamic viscosity [Pa*s].
        k: Thermal conductivity [W/(m*K)].

    All fields must be finite and strictly positive.
    """

    T: float
    rho: float
    cp: float
    mu: float
    k: float

    def __post_init__(self) -> None:
        for name in ("T", "rho", "cp", "mu", "k"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"LiquidPropertyPoint.{name} must be finite.")
            if value <= 0.0:
                raise ValueError(f"LiquidPropertyPoint.{name} must be positive.")


@dataclass(frozen=True)
class TabulatedLiquidProvider:
    """Property provider for a manually supplied single-phase liquid.

    One point gives constant properties at every positive temperature. Two
    or more points interpolate rho/cp/k linearly and mu log-linearly versus
    T, and derive specific enthalpy by exact piecewise-linear-cp integration
    with h = 0 at the lowest supplied temperature. Pressure is accepted for
    interface compatibility only; properties are pressure-independent.

    Attributes:
        name: Fluid name used for reporting (e.g. in `full_at().fluid`).
        points: One or more `LiquidPropertyPoint`. Order does not matter --
            points are sorted by T internally. Duplicate temperatures are
            invalid.

    Notes:
        Temperatures outside the supplied table raise ValueError for a
        multi-point provider; this provider never silently extrapolates.
        A single-point provider has no such range restriction -- its
        constant properties apply at every positive temperature.
    """

    name: str
    points: Sequence[LiquidPropertyPoint]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("TabulatedLiquidProvider name must be a non-empty string.")

        points = list(self.points)
        if not points:
            raise ValueError("TabulatedLiquidProvider requires at least one point.")

        points.sort(key=lambda point: point.T)
        for previous, current in zip(points, points[1:]):
            if previous.T == current.T:
                raise ValueError(
                    f"Duplicate temperature in TabulatedLiquidProvider "
                    f"{self.name!r} points: T={current.T} K."
                )

        object.__setattr__(self, "points", tuple(points))
        object.__setattr__(self, "_T", tuple(point.T for point in points))
        object.__setattr__(self, "_cumulative_h", _cumulative_enthalpy(points))

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return transport properties at T [K] and p [Pa].

        Pressure is validated for interface compatibility but does not
        affect the returned properties.
        """
        _validate_temperature(T)
        _validate_pressure(p)

        if len(self.points) == 1:
            point = self.points[0]
            return FluidTransportProperties(rho=point.rho, mu=point.mu, k=point.k, cp=point.cp)

        lo, hi, frac = self._locate(T)
        point_lo, point_hi = self.points[lo], self.points[hi]
        if lo == hi:
            return FluidTransportProperties(
                rho=point_lo.rho, mu=point_lo.mu, k=point_lo.k, cp=point_lo.cp
            )

        rho = point_lo.rho + (point_hi.rho - point_lo.rho) * frac
        cp = point_lo.cp + (point_hi.cp - point_lo.cp) * frac
        k = point_lo.k + (point_hi.k - point_lo.k) * frac
        ln_mu_lo = math.log(point_lo.mu)
        ln_mu_hi = math.log(point_hi.mu)
        mu = math.exp(ln_mu_lo + (ln_mu_hi - ln_mu_lo) * frac)
        return FluidTransportProperties(rho=rho, mu=mu, k=k, cp=cp)

    def full_at(self, T: float, p: float) -> CoolPropProperties:
        """Return transport properties, enthalpy and phase at T [K], p [Pa].

        Relative enthalpy is zero at the reference temperature (the single
        point's T for one point, otherwise the lowest supplied T) and is
        obtained by integrating the interpolated cp(T). `phase` is always
        "liquid" and `fluid` is the supplied provider name.
        """
        transport = self.at(T, p)
        h = self._enthalpy(T)
        return CoolPropProperties(
            transport=transport,
            h=h,
            phase="liquid",
            fluid=self.name,
            warnings=[],
        )

    def _enthalpy(self, T: float) -> float:
        if len(self.points) == 1:
            point = self.points[0]
            return point.cp * (T - point.T)

        lo, hi, frac = self._locate(T)
        if lo == hi:
            return self._cumulative_h[lo]

        point_lo, point_hi = self.points[lo], self.points[hi]
        cp_at_T = point_lo.cp + (point_hi.cp - point_lo.cp) * frac
        return self._cumulative_h[lo] + 0.5 * (point_lo.cp + cp_at_T) * (T - point_lo.T)

    def _locate(self, T: float) -> tuple[int, int, float]:
        T_min, T_max = self._T[0], self._T[-1]
        if T < T_min or T > T_max:
            raise ValueError(
                f"Temperature T={T} K is outside the supplied TabulatedLiquidProvider "
                f"range [{T_min}, {T_max}] K for {self.name!r}. Extrapolation is not supported."
            )

        idx = bisect.bisect_left(self._T, T)
        if idx < len(self._T) and self._T[idx] == T:
            return idx, idx, 0.0

        lo, hi = idx - 1, idx
        Ta, Tb = self._T[lo], self._T[hi]
        frac = (T - Ta) / (Tb - Ta)
        return lo, hi, frac


def _cumulative_enthalpy(points: Sequence[LiquidPropertyPoint]) -> tuple[float, ...]:
    """Return cumulative enthalpy at each sorted knot, with h = 0 at points[0].T.

    Each interval is integrated exactly for piecewise-linear cp using the
    trapezoidal expression, which is exact (not approximate) for a linear
    integrand.
    """
    cumulative = [0.0]
    for previous, current in zip(points, points[1:]):
        segment = 0.5 * (previous.cp + current.cp) * (current.T - previous.T)
        cumulative.append(cumulative[-1] + segment)
    return tuple(cumulative)


def _validate_temperature(T: float) -> None:
    if not math.isfinite(T):
        raise ValueError("Temperature must be finite [K].")
    if T <= 0.0:
        raise ValueError("Temperature must be above absolute zero [K].")


def _validate_pressure(p: float) -> None:
    if not math.isfinite(p):
        raise ValueError("Pressure must be finite [Pa].")
    if p <= 0.0:
        raise ValueError("Pressure must be positive [Pa].")
