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
    heat_transfer_coefficient_internal_diagnostics,
    InternalHeatTransferDiagnostics,
    gas_wall_temperature_correction,
    internal_length_correction,
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

from .outside_pressure_drop import (
    EulerProvider,
    EulerRequest,
    EulerResult,
    evaluate_euler,
    pressure_drop_from_euler,
    check_outside_dp_applicability,
)

from .outside_pressure_drop_external import (
    ExternalCliEulerProvider,
)

from .thermal_iteration import (
    IterativeThermalState,
    WallTemperatureProbe,
    WallTemperatureEnvelope,
    estimate_wall_temperature_envelope,
    solve_iterative_thermal_state,
)

from core.common.warnings import (
    ApplicabilityRange,
    ModelWarning,
    WarningSeverity,
    check_range,
    make_warning,
    has_critical_warnings,
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
    "heat_transfer_coefficient_internal_diagnostics",
    "InternalHeatTransferDiagnostics",
    "gas_wall_temperature_correction",
    "internal_length_correction",

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

    # Outside pressure drop
    "EulerProvider",
    "EulerRequest",
    "EulerResult",
    "evaluate_euler",
    "pressure_drop_from_euler",
    "check_outside_dp_applicability",

    # Outside pressure drop external adapter
    "ExternalCliEulerProvider",

    # Iterative wall-corrected thermal state
    "IterativeThermalState",
    "WallTemperatureProbe",
    "WallTemperatureEnvelope",
    "estimate_wall_temperature_envelope",
    "solve_iterative_thermal_state",

    # Warnings and applicability checks
    "ApplicabilityRange",
    "ModelWarning",
    "WarningSeverity",
    "check_range",
    "make_warning",
    "has_critical_warnings",
]
