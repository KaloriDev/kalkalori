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

from core.pressure_drop.internal_pressure_drop import (
    FluidProps as InternalPressureDropFluidProps,
    TubeSideHydraulicPoint,
    TubePassBoundaryHydraulicState,
    TubeEndPressureDropResult,
    TubeBundleHydraulicResult,
    calculate_tube_bundle_hydraulics,
    tube_bundle_hydraulics,
    pressure_drop_internal_total,
    pressure_drop_tubes,
    pressure_drop_inlet,
    pressure_drop_outlet,
    pressure_drop_turns,
)

from .outside_flow import (
    FluidProps as OutsideFlowFluidProps,
    OutsideHydraulicPoint,
    OutsideTubeBankHydraulicResult,
    calculate_outside_tube_bank_hydraulics,
    outside_tube_bank_hydraulics,
    outside_flow_from_mass_flow,
)

from core.pressure_drop.outside_pressure_drop import (
    EulerProvider,
    EulerRequest,
    EulerResult,
    evaluate_euler,
    pressure_drop_from_euler,
    check_outside_dp_applicability,
)

from core.pressure_drop.outside_pressure_drop_external import (
    ExternalCliEulerProvider,
)

from .thermal_iteration import (
    IterativeThermalState,
    WallTemperatureProbe,
    WallTemperatureEnvelope,
    estimate_wall_temperature_envelope,
    solve_iterative_thermal_state,
)

from .fin_efficiency import (
    DEFAULT_FIN_RADIAL_CELLS,
    FinEfficiencyResult,
    annular_fin_efficiency,
    calculate_fin_efficiency,
    fin_efficiency,
    overall_surface_efficiency,
    effective_outside_area,
)

from .finned_tube_outside import (
    FinnedTubeLayout,
    FinnedTubeHeatTransferMetadata,
    FinnedTubeHeatTransferRequest,
    FinnedTubeHeatTransferResult,
    FinnedTubeHeatTransferProvider,
    BriggsYoung1963Provider,
    BRIGGS_YOUNG_1963_APPLICABILITY,
    BRIGGS_YOUNG_1963_METADATA,
    evaluate_finned_tube_heat_transfer,
    calculate_finned_tube_outside_heat_transfer,
    finned_tube_outside_heat_transfer,
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
    "TubeSideHydraulicPoint",
    "TubePassBoundaryHydraulicState",
    "TubeEndPressureDropResult",
    "TubeBundleHydraulicResult",
    "calculate_tube_bundle_hydraulics",
    "tube_bundle_hydraulics",
    "pressure_drop_internal_total",
    "pressure_drop_tubes",
    "pressure_drop_inlet",
    "pressure_drop_outlet",
    "pressure_drop_turns",

    # Outside flow (mass-flow driven)
    "OutsideFlowFluidProps",
    "OutsideHydraulicPoint",
    "OutsideTubeBankHydraulicResult",
    "calculate_outside_tube_bank_hydraulics",
    "outside_tube_bank_hydraulics",
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

    # Circular annular-fin efficiency
    "DEFAULT_FIN_RADIAL_CELLS",
    "FinEfficiencyResult",
    "annular_fin_efficiency",
    "calculate_fin_efficiency",
    "fin_efficiency",
    "overall_surface_efficiency",
    "effective_outside_area",

    # Outside heat transfer for circular-finned tube banks
    "FinnedTubeLayout",
    "FinnedTubeHeatTransferMetadata",
    "FinnedTubeHeatTransferRequest",
    "FinnedTubeHeatTransferResult",
    "FinnedTubeHeatTransferProvider",
    "BriggsYoung1963Provider",
    "BRIGGS_YOUNG_1963_APPLICABILITY",
    "BRIGGS_YOUNG_1963_METADATA",
    "evaluate_finned_tube_heat_transfer",
    "calculate_finned_tube_outside_heat_transfer",
    "finned_tube_outside_heat_transfer",

    # Warnings and applicability checks
    "ApplicabilityRange",
    "ModelWarning",
    "WarningSeverity",
    "check_range",
    "make_warning",
    "has_critical_warnings",
]
