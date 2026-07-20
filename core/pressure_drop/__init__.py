# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Pressure-drop package (v0.5.6 architecture).

Canonical home for tube-side and outside-side pressure-drop code:

- ``internal_pressure_drop``: tube-side (internal) straight-tube-bundle
  hydraulics.
- ``outside_pressure_drop``: outside crossflow tube-bank Euler-number
  providers and dispatcher.
- ``outside_pressure_drop_external``: subprocess/JSON adapter for an
  external Euler-number provider.
- ``flow_path``: generic stage/group/path result structures and the
  functions that aggregate the existing tube-bundle/tube-bank results with
  explicitly invoked local-loss stages into a complete flow-path result.

The previous import locations under ``core.heat_transfer`` remain available
as compatibility re-exports; this package is the single source of truth.
"""

from __future__ import annotations

from .internal_pressure_drop import (
    FluidProps as InternalPressureDropFluidProps,
    HydraulicPosition,
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

from .screens import (
    TubeSheetEntranceType,
    TubeSheetExitType,
    tube_sheet_entrance_loss_coefficient,
    tube_sheet_exit_loss_coefficient,
    calculate_tube_sheet_entrance_loss,
    calculate_tube_sheet_exit_loss,
    FlatObstructionType,
    FlatObstructionGeometry,
    calculate_flat_obstruction_pressure_drop,
    calculate_screen_pressure_drop,
    calculate_user_defined_pressure_drop,
)

from .straight_sections import (
    friction_factor_smooth,
    darcy_friction_factor,
    darcy_friction_factor_method,
    calculate_straight_section_pressure_drop,
)

from .area_changes import (
    calculate_area_change_pressure_drop,
)

from .direction_changes import (
    ElbowCrossSection,
    ElbowConstruction,
    RectangularTurnPlane,
    DirectionChangeMethod,
    CircularElbowGeometry,
    RectangularElbowGeometry,
    calculate_circular_elbow_pressure_drop,
    calculate_rectangular_elbow_pressure_drop,
)

from .outside_pressure_drop import (
    Layout,
    EulerProvider,
    EulerRequest,
    EulerResult,
    ZukauskasEulerProvider,
    EsduEulerProvider,
    GaddisGnielinskiEulerProvider,
    evaluate_euler,
    pressure_drop_from_euler,
    check_outside_dp_applicability,
)

from .outside_pressure_drop_external import (
    ExternalCliEulerProvider,
)

from .flow_path import (
    PressureDropFlowState,
    SectionFlowResult,
    evaluate_section_flow,
    PressureDropStageStatus,
    PressureDropStageResult,
    PressureDropStageGroupResult,
    PressureDropPathResult,
    build_tube_side_pressure_drop_result,
    build_outside_pressure_drop_result,
    calculate_pressure_drop_assembly,
    calculate_tube_side_pressure_drop_path,
    calculate_outside_pressure_drop_path,
)

__all__ = [
    # Internal (tube-side) pressure drop
    "InternalPressureDropFluidProps",
    "HydraulicPosition",
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

    # Tube-sheet entrance/exit pressure drop (screens.py)
    "TubeSheetEntranceType",
    "TubeSheetExitType",
    "tube_sheet_entrance_loss_coefficient",
    "tube_sheet_exit_loss_coefficient",
    "calculate_tube_sheet_entrance_loss",
    "calculate_tube_sheet_exit_loss",

    # Flat planar obstructions, general screens, and user-defined local
    # losses (screens.py; v0.5.6 local pressure-drop paths)
    "FlatObstructionType",
    "FlatObstructionGeometry",
    "calculate_flat_obstruction_pressure_drop",
    "calculate_screen_pressure_drop",
    "calculate_user_defined_pressure_drop",

    # Straight circular-section friction factor and local pressure drop
    # (straight_sections.py)
    "friction_factor_smooth",
    "darcy_friction_factor",
    "darcy_friction_factor_method",
    "calculate_straight_section_pressure_drop",

    # Area-change (expansion/contraction) local pressure drop (area_changes.py)
    "calculate_area_change_pressure_drop",

    # Elbow/bend local pressure drop (direction_changes.py)
    "ElbowCrossSection",
    "ElbowConstruction",
    "RectangularTurnPlane",
    "DirectionChangeMethod",
    "CircularElbowGeometry",
    "RectangularElbowGeometry",
    "calculate_circular_elbow_pressure_drop",
    "calculate_rectangular_elbow_pressure_drop",

    # Outside pressure drop
    "Layout",
    "EulerProvider",
    "EulerRequest",
    "EulerResult",
    "ZukauskasEulerProvider",
    "EsduEulerProvider",
    "GaddisGnielinskiEulerProvider",
    "evaluate_euler",
    "pressure_drop_from_euler",
    "check_outside_dp_applicability",

    # Outside pressure drop external adapter
    "ExternalCliEulerProvider",

    # Explicit pressure-drop flow state and reusable section-flow helpers
    # (flow_path.py; v0.5.6 local pressure-drop paths)
    "PressureDropFlowState",
    "SectionFlowResult",
    "evaluate_section_flow",

    # Flow-path stage/group/path result structures and aggregation
    "PressureDropStageStatus",
    "PressureDropStageResult",
    "PressureDropStageGroupResult",
    "PressureDropPathResult",
    "build_tube_side_pressure_drop_result",
    "build_outside_pressure_drop_result",

    # Generic and side-specific explicit local-loss calculation (v0.5.6
    # local pressure-drop paths; never invoked by the standard solver)
    "calculate_pressure_drop_assembly",
    "calculate_tube_side_pressure_drop_path",
    "calculate_outside_pressure_drop_path",
]
