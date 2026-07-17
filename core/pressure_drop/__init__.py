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
  (currently unimplemented) local-loss stages into a complete flow-path
  result.

The previous import locations under ``core.heat_transfer`` remain available
as compatibility re-exports; this package is the single source of truth.
"""

from __future__ import annotations

from .internal_pressure_drop import (
    FluidProps as InternalPressureDropFluidProps,
    HydraulicPosition,
    TubeSideHydraulicPoint,
    TubeBundleHydraulicResult,
    calculate_tube_bundle_hydraulics,
    tube_bundle_hydraulics,
    pressure_drop_internal_total,
    pressure_drop_tubes,
    pressure_drop_inlet,
    pressure_drop_outlet,
    pressure_drop_turns,
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
    PressureDropStageStatus,
    PressureDropStageResult,
    PressureDropStageGroupResult,
    PressureDropPathResult,
    build_tube_side_pressure_drop_result,
    build_outside_pressure_drop_result,
)

__all__ = [
    # Internal (tube-side) pressure drop
    "InternalPressureDropFluidProps",
    "HydraulicPosition",
    "TubeSideHydraulicPoint",
    "TubeBundleHydraulicResult",
    "calculate_tube_bundle_hydraulics",
    "tube_bundle_hydraulics",
    "pressure_drop_internal_total",
    "pressure_drop_tubes",
    "pressure_drop_inlet",
    "pressure_drop_outlet",
    "pressure_drop_turns",

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

    # Flow-path stage/group/path result structures and aggregation
    "PressureDropStageStatus",
    "PressureDropStageResult",
    "PressureDropStageGroupResult",
    "PressureDropPathResult",
    "build_tube_side_pressure_drop_result",
    "build_outside_pressure_drop_result",
]
