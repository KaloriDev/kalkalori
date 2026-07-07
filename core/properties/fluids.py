"""
Minimal property-provider interfaces for single-phase fluids.

This is intentionally small. The goal is to provide a stable point-property
interface for future wet economizer and tube-side calculations without adding
a full property framework yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.properties.common import FluidTransportProperties


class PropertyProvider(Protocol):
    """Protocol for point-property providers.

    Implementations return transport properties at a given temperature and
    pressure.

    Args:
        T: Temperature [K].
        p: Pressure [Pa].
    """

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return fluid properties at T and p."""
        ...


@dataclass(frozen=True)
class ConstantPropertyProvider:
    """Constant property provider.

    Useful for MVP calculations, tests, and cases where properties are supplied
    externally by the caller.

    This provider deliberately ignores T and p.
    """

    props: FluidTransportProperties

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return constant properties, independent of T and p."""
        return self.props