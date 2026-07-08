"""
Adapters between the common property layer and existing heat-transfer modules.

The property layer uses `FluidTransportProperties` as the common transport
property container. Existing heat-transfer modules still define their own
small `FluidProps` dataclasses.

This module provides explicit conversion helpers without refactoring existing
heat-transfer correlations.

Core SI units:
- density: kg/m3
- dynamic viscosity: Pa*s
- thermal conductivity: W/(m*K)
- specific heat: J/(kg*K)
"""

from __future__ import annotations

from typing import Any

from core.properties.common import FluidTransportProperties

from core.heat_transfer.internal_flow import FluidProps as InternalFluidProps
from core.heat_transfer.internal_pressure_drop import FluidProps as InternalPressureDropFluidProps
from core.heat_transfer.outside_flow import FluidProps as OutsideFluidProps


def to_internal_fluid_props(props: FluidTransportProperties) -> InternalFluidProps:
    """Convert common transport properties to internal-flow FluidProps.

    Args:
        props: Common fluid transport properties.

    Returns:
        `core.heat_transfer.internal_flow.FluidProps`.
    """
    return InternalFluidProps(
        rho=props.rho,
        mu=props.mu,
        k=props.k,
        cp=props.cp,
    )


def to_internal_pressure_drop_fluid_props(
    props: FluidTransportProperties,
) -> InternalPressureDropFluidProps:
    """Convert common transport properties to internal-pressure-drop FluidProps.

    Args:
        props: Common fluid transport properties.

    Returns:
        `core.heat_transfer.internal_pressure_drop.FluidProps`.
    """
    return InternalPressureDropFluidProps(
        rho=props.rho,
        mu=props.mu,
    )


def to_outside_fluid_props(props: FluidTransportProperties) -> OutsideFluidProps:
    """Convert common transport properties to outside-flow FluidProps.

    Args:
        props: Common fluid transport properties.

    Returns:
        `core.heat_transfer.outside_flow.FluidProps`.
    """
    return OutsideFluidProps(
        rho=props.rho,
        mu=props.mu,
        k=props.k,
        cp=props.cp,
    )


def from_internal_fluid_props(props: InternalFluidProps) -> FluidTransportProperties:
    """Convert internal-flow FluidProps to common transport properties."""
    return _from_fluid_props_like(props)


def from_internal_pressure_drop_fluid_props(
    props: InternalPressureDropFluidProps,
) -> FluidTransportProperties:
    """Convert internal-pressure-drop FluidProps to common transport properties.

    Note: `InternalPressureDropFluidProps` only carries `rho` and `mu`.
    The returned `FluidTransportProperties` will have `k` and `cp` set to NaN
    because those values are not stored in the pressure-drop props.
    """
    import math
    return FluidTransportProperties(
        rho=float(props.rho),
        mu=float(props.mu),
        k=math.nan,
        cp=math.nan,
    )


def from_outside_fluid_props(props: OutsideFluidProps) -> FluidTransportProperties:
    """Convert outside-flow FluidProps to common transport properties."""
    return _from_fluid_props_like(props)


def _from_fluid_props_like(props: Any) -> FluidTransportProperties:
    """Convert any object with rho/mu/k/cp fields to FluidTransportProperties."""
    return FluidTransportProperties(
        rho=float(props.rho),
        mu=float(props.mu),
        k=float(props.k),
        cp=float(props.cp),
    )