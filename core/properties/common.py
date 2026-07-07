"""
Common property data structures.

KalKalori core uses SI floats:
- density: kg/m3
- dynamic viscosity: Pa*s
- thermal conductivity: W/(m*K)
- specific heat: J/(kg*K)

This module intentionally contains only simple point properties.
No unit-conversion or external property backend is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FluidTransportProperties:
    """Single-phase fluid transport properties at one thermodynamic state.

    Attributes:
        rho: Density [kg/m3].
        mu: Dynamic viscosity [Pa*s].
        k: Thermal conductivity [W/(m*K)].
        cp: Specific heat capacity [J/(kg*K)].
    """

    rho: float
    mu: float
    k: float
    cp: float

    def __post_init__(self) -> None:
        values = {
            "rho": self.rho,
            "mu": self.mu,
            "k": self.k,
            "cp": self.cp,
        }

        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")