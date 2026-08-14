# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Geometric definitions for heat exchanger components.
"""

from .tube import (
    BaseTube,
    BareTube,
    TubeOrientation,
    TubeSurfaceType,
)
from .finned_tube import (
    CircularFinnedTube,
    EXTERNAL_AREA_OVERRIDE_RELATIVE_WARNING_THRESHOLD,
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
    "CircularFinnedTube",
    "TubeOrientation",
    "TubeSurfaceType",
    "EXTERNAL_AREA_OVERRIDE_RELATIVE_WARNING_THRESHOLD",
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
