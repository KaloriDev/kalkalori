# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Heat transfer correlations and helper functions.

This package contains low-level, reusable building blocks.
"""

from .ntu import effectiveness_ntu, heat_duty_from_effectiveness

from .streams import (
    EnergyStream,
    SensibleHeatStream,
    CondensingSteamStream,
    MoistAirStream,
)

from .internal_flow import (
    FluidProps as InternalFlowFluidProps,
    heat_transfer_coefficient_internal,
)

from .internal_pressure_drop import (
    FluidProps as InternalPressureDropFluidProps,
    pressure_drop_internal_total,
    pressure_drop_tubes,
    pressure_drop_inlet,
    pressure_drop_outlet,
    pressure_drop_turns,
)

from .outside_flow import (
    FluidProps as OutsideFlowFluidProps,
    outside_flow_from_mass_flow,
)

__all__ = [
    # NTU
    "effectiveness_ntu",
    "heat_duty_from_effectiveness",

    # Streams
    "EnergyStream",
    "SensibleHeatStream",
    "CondensingSteamStream",
    "MoistAirStream",

    # Internal flow
    "InternalFlowFluidProps",
    "heat_transfer_coefficient_internal",

    # Internal pressure drop (component-based)
    "InternalPressureDropFluidProps",
    "pressure_drop_internal_total",
    "pressure_drop_tubes",
    "pressure_drop_inlet",
    "pressure_drop_outlet",
    "pressure_drop_turns",

    # Outside flow (mass-flow driven)
    "OutsideFlowFluidProps",
    "outside_flow_from_mass_flow",
]
