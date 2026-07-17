# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Common pressure-drop stage geometry (v0.5.6 architecture).

This module defines the physical geometry of individual pressure-drop
stages that may occur along a tube-side or outside-side flow path: straight
sections, area changes, screens/opening arrays, direction changes, chambers,
and user-defined stages.

A geometry type here does not know which exchanger side it belongs to, or
whether it plays the role of an inlet, outlet, or return stage. Role and
side are determined entirely by where a geometry object is placed within a
``PressureDropAssemblyGeometry`` / specified flow path (see
``core.geometry.tube_side_pressure_drop_path`` and
``core.geometry.outside_pressure_drop_path``).

This module defines geometry only. No pressure-drop calculation is
implemented here; see ``core.pressure_drop`` for the (currently
not-implemented) local-loss calculation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FlowSectionShape(str, Enum):
    """Cross-sectional shape family for a straight flow section."""

    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"
    ANNULAR = "annular"
    CUSTOM = "custom"


class AreaChangeType(str, Enum):
    """Geometric family of an area-change (expansion/contraction) stage."""

    SUDDEN = "sudden"
    CONICAL = "conical"
    PYRAMIDAL = "pyramidal"
    CUSTOM = "custom"


class ScreenType(str, Enum):
    """Geometric family of a screen / opening-array stage."""

    SCREEN = "screen"
    MESH = "mesh"
    PERFORATED_PLATE = "perforated_plate"
    FLOW_STRAIGHTENER = "flow_straightener"
    TUBE_SHEET_OPENING_ARRAY = "tube_sheet_opening_array"
    CUSTOM = "custom"


class DirectionChangeType(str, Enum):
    """Geometric family of a direction-change stage.

    ``ELBOW``, ``MITER_BEND``, ``ELBOW_180``, and ``U_BEND`` represent a
    single continuous flow path turning through an angle. ``RETURN_CHAMBER``
    is architecturally distinct: it represents multiple parallel tube
    streams mixing in a chamber, turning, and re-entering another tube pass.
    Both families share this geometry type but will use different
    calculation methods once implemented.
    """

    ELBOW = "elbow"
    MITER_BEND = "miter_bend"
    ELBOW_180 = "elbow_180"
    U_BEND = "u_bend"
    RETURN_CHAMBER = "return_chamber"
    CUSTOM = "custom"


@dataclass(frozen=True)
class StraightSectionGeometry:
    """Straight flow-section geometry (nozzle, duct, header, pipe, ...).

    Pressure-drop calculation for this geometry is not implemented in this
    commit; see ``core.pressure_drop``.
    """

    flow_area: float
    hydraulic_diameter: float
    length: float
    roughness: float = 0.0
    section_shape: FlowSectionShape = FlowSectionShape.CUSTOM


@dataclass(frozen=True)
class AreaChangeGeometry:
    """Area-change (expansion/contraction) geometry.

    Direction is inferred from geometry, not declared explicitly:
    ``downstream_area > upstream_area`` is an expansion/diffuser;
    ``downstream_area < upstream_area`` is a contraction/confuser. The same
    geometry represents tube-side nozzle/chamber transitions and
    outside-side duct/plenum transitions alike.

    Pressure-drop calculation for this geometry is not implemented in this
    commit; see ``core.pressure_drop``.
    """

    change_type: AreaChangeType
    upstream_area: float
    downstream_area: float
    length: float | None = None
    included_angle_deg: float | None = None

    @property
    def is_expansion(self) -> bool:
        return self.downstream_area > self.upstream_area

    @property
    def is_contraction(self) -> bool:
        return self.downstream_area < self.upstream_area


@dataclass(frozen=True)
class ScreenGeometry:
    """Screen / opening-array geometry (screen, mesh, perforated plate,
    flow straightener, tube-sheet opening array, parallel tube entrance or
    exit).

    Kept distinct from ``AreaChangeGeometry`` even though both may change
    flow area, because screens/opening arrays require different geometric
    descriptors (opening count/diameter, plate thickness, edge treatment).

    Pressure-drop calculation for this geometry is not implemented in this
    commit; see ``core.pressure_drop``.
    """

    screen_type: ScreenType
    upstream_area: float
    open_flow_area: float

    opening_count: int | None = None
    opening_diameter: float | None = None
    plate_thickness: float | None = None
    edge_radius: float | None = None
    bevel_angle_deg: float | None = None


@dataclass(frozen=True)
class DirectionChangeGeometry:
    """Direction-change geometry (elbow, miter bend, 180-degree elbow,
    U-bend, or a multi-tube return chamber).

    ``chamber_area``/``chamber_depth`` are only meaningful for
    ``DirectionChangeType.RETURN_CHAMBER``; a plain bend/elbow instead uses
    ``flow_area``/``hydraulic_diameter``/``bend_radius``.

    Pressure-drop calculation for this geometry is not implemented in this
    commit; see ``core.pressure_drop``.
    """

    change_type: DirectionChangeType
    angle_deg: float

    flow_area: float | None = None
    hydraulic_diameter: float | None = None
    bend_radius: float | None = None

    chamber_area: float | None = None
    chamber_depth: float | None = None


@dataclass(frozen=True)
class ChamberGeometry:
    """General chamber/plenum geometry.

    A single geometry type covers inlet chambers, outlet chambers, return
    chambers, and outside-side plenums alike; those are roles assigned by
    placement in a flow path, not distinct geometry types.

    Pressure-drop calculation for this geometry is not implemented in this
    commit; see ``core.pressure_drop``.
    """

    flow_area: float
    hydraulic_diameter: float | None = None
    length: float | None = None
    volume: float | None = None


@dataclass(frozen=True)
class UserDefinedPressureDropGeometry:
    """User-supplied pressure-drop stage.

    ``pressure_drop`` (if supplied) is a direct fixed pressure drop and may
    be reported as a ``user_defined`` calculated stage. ``loss_coefficient``
    is retained for a future K-to-dp conversion; that conversion is not
    implemented in this commit, so a stage that only supplies
    ``loss_coefficient`` (no ``pressure_drop``) is reported as
    ``not_implemented``.
    """

    pressure_drop: float | None = None
    loss_coefficient: float | None = None
    reference_area: float | None = None
    description: str = ""


PressureDropStageGeometry = (
    StraightSectionGeometry
    | AreaChangeGeometry
    | ScreenGeometry
    | DirectionChangeGeometry
    | ChamberGeometry
    | UserDefinedPressureDropGeometry
)
"""Union of the supported common pressure-drop stage geometries."""


@dataclass(frozen=True)
class PressureDropAssemblyGeometry:
    """An ordered sequence of pressure-drop stage geometries.

    Order is preserved exactly as supplied. A missing physical component is
    represented by its absence from ``stages`` -- never by a disabled/zeroed
    placeholder stage.
    """

    stages: tuple[PressureDropStageGeometry, ...]
