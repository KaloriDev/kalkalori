# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Geometric definitions for heat exchanger components.
"""

from .tube import BaseTube, BareTube, TubeOrientation, TubeSurfaceType
from .finned_tube import CircularFinnedTube
from .finned_flow_geometry import (
    FinnedTubeGeometryOverlapError,
    finned_blocked_equivalent_diameter,
    finned_min_free_flow_area_per_length,
    finned_vmax_ratio_min_freeflow,
    validate_no_fin_row_overlap,
)
from .bundle import TubeBundle, TubePathType

from .pressure_drop_stages import (
    FlowSectionShape,
    AreaChangeType,
    ScreenType,
    DirectionChangeType,
    CircularFlowSection,
    RectangularFlowSection,
    CustomFlowSection,
    FlowSectionGeometry,
    StraightSectionGeometry,
    AreaChangeGeometry,
    ScreenGeometry,
    DirectionChangeGeometry,
    ChamberGeometry,
    UserDefinedPressureDropGeometry,
    PressureDropStageGeometry,
    PressureDropAssemblyGeometry,
)
from .tube_side_pressure_drop_path import (
    SpecifiedTubeSidePressureDropPath,
    TubeSidePressureDropDesignRequest,
    validate_specified_tube_side_path,
)
from .outside_pressure_drop_path import (
    SpecifiedOutsidePressureDropPath,
    OutsidePressureDropDesignRequest,
)

__all__ = [
    "BaseTube",
    "BareTube",
    "TubeOrientation",
    "TubeSurfaceType",
    "CircularFinnedTube",
    "FinnedTubeGeometryOverlapError",
    "finned_blocked_equivalent_diameter",
    "finned_min_free_flow_area_per_length",
    "finned_vmax_ratio_min_freeflow",
    "validate_no_fin_row_overlap",
    "TubeBundle",
    "TubePathType",

    # Common pressure-drop stage geometry
    "FlowSectionShape",
    "AreaChangeType",
    "ScreenType",
    "DirectionChangeType",
    "CircularFlowSection",
    "RectangularFlowSection",
    "CustomFlowSection",
    "FlowSectionGeometry",
    "StraightSectionGeometry",
    "AreaChangeGeometry",
    "ScreenGeometry",
    "DirectionChangeGeometry",
    "ChamberGeometry",
    "UserDefinedPressureDropGeometry",
    "PressureDropStageGeometry",
    "PressureDropAssemblyGeometry",

    # Tube-side specified pressure-drop path
    "SpecifiedTubeSidePressureDropPath",
    "TubeSidePressureDropDesignRequest",
    "validate_specified_tube_side_path",

    # Outside-side specified pressure-drop path
    "SpecifiedOutsidePressureDropPath",
    "OutsidePressureDropDesignRequest",
]
