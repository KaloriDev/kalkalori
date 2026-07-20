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

This module defines geometry only. Pressure-drop calculations live in
``core.pressure_drop`` and are invoked explicitly by the local-path API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class FlowSectionShape(str, Enum):
    """Cross-sectional shape family for a straight flow section."""

    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"
    ANNULAR = "annular"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Reusable cross-section geometry (v0.5.6 local pressure-drop paths)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CircularFlowSection:
    """Circular flow cross-section.

    Exposes ``flow_area``, ``hydraulic_diameter``,
    ``equivalent_circular_diameter`` and ``section_shape`` consistently with
    ``RectangularFlowSection``/``CustomFlowSection`` so callers never need to
    compute area or hydraulic diameter outside the geometry layer.
    """

    diameter: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.diameter) or self.diameter <= 0.0:
            raise ValueError("CircularFlowSection.diameter must be a positive, finite value.")

    @property
    def flow_area(self) -> float:
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def hydraulic_diameter(self) -> float:
        return self.diameter

    @property
    def equivalent_circular_diameter(self) -> float:
        return self.diameter

    @property
    def section_shape(self) -> FlowSectionShape:
        return FlowSectionShape.CIRCULAR


@dataclass(frozen=True)
class RectangularFlowSection:
    """Rectangular flow cross-section (e.g. duct, plenum opening)."""

    width: float
    height: float

    def __post_init__(self) -> None:
        for name, value in (("width", self.width), ("height", self.height)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"RectangularFlowSection.{name} must be a positive, finite value.")

    @property
    def flow_area(self) -> float:
        return self.width * self.height

    @property
    def hydraulic_diameter(self) -> float:
        return 2.0 * self.width * self.height / (self.width + self.height)

    @property
    def equivalent_circular_diameter(self) -> float:
        return math.sqrt(4.0 * self.flow_area / math.pi)

    @property
    def section_shape(self) -> FlowSectionShape:
        return FlowSectionShape.RECTANGULAR


@dataclass(frozen=True)
class CustomFlowSection:
    """Explicit, directly-supplied area and hydraulic diameter.

    For sections whose exact shape is not circular or rectangular (e.g. an
    annulus, or a duct with a known hydraulic diameter from vendor data).
    """

    area: float
    hydraulic_diameter: float

    def __post_init__(self) -> None:
        for name, value in (("area", self.area), ("hydraulic_diameter", self.hydraulic_diameter)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"CustomFlowSection.{name} must be a positive, finite value.")

    @property
    def flow_area(self) -> float:
        return self.area

    @property
    def equivalent_circular_diameter(self) -> float:
        return math.sqrt(4.0 * self.area / math.pi)

    @property
    def section_shape(self) -> FlowSectionShape:
        return FlowSectionShape.CUSTOM


FlowSectionGeometry = CircularFlowSection | RectangularFlowSection | CustomFlowSection
"""Union of the supported reusable flow cross-section geometries."""


class AreaChangeType(str, Enum):
    """Geometric family of an area-change (expansion/contraction) stage.

    ``SUDDEN`` uses an implicit 180-degree included angle. ``GRADUAL``
    covers both conical (circular) and pyramidal (rectangular) transitions:
    the physical shape distinction does not change the Gibson/Crane
    correlation used, only the (possibly approximated) included angle -- see
    ``core.pressure_drop.area_changes``.
    """

    SUDDEN = "sudden"
    GRADUAL = "gradual"


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


class PressureDropStageGeometry:
    """Marker base for geometry accepted in a pressure-drop assembly.

    Correlation-specific geometry classes may live in ``core.pressure_drop``
    modules, but inherit this common marker so all supported stage types can
    be carried by ``PressureDropAssemblyGeometry`` without a closed union or
    a geometry-to-correlation import cycle.
    """


@dataclass(frozen=True)
class StraightSectionGeometry(PressureDropStageGeometry):
    """Straight flow-section geometry (nozzle, duct, header, pipe, ...).

    Explicit local-path pressure-drop calculation is implemented in
    ``core.pressure_drop.straight_sections``.
    """

    flow_area: float
    hydraulic_diameter: float
    length: float
    roughness: float = 0.0
    section_shape: FlowSectionShape = FlowSectionShape.CUSTOM


@dataclass(frozen=True)
class AreaChangeGeometry(PressureDropStageGeometry):
    """Area-change (expansion/contraction) geometry.

    Direction is inferred from geometry, not declared explicitly:
    ``downstream_section.flow_area > upstream_section.flow_area`` is an
    expansion/diffuser; smaller is a contraction/confuser. The same geometry
    represents tube-side nozzle/chamber transitions and outside-side
    duct/plenum transitions alike, and either section may be circular,
    rectangular, or a custom area/hydraulic-diameter pair -- e.g. a circular
    duct expanding into a rectangular plenum.

    Real pressure-drop calculation for this geometry is implemented in
    ``core.pressure_drop.area_changes.calculate_area_change_pressure_drop``
    (Gibson/Crane gradual and sudden expansion/contraction correlations).
    """

    upstream_section: FlowSectionGeometry
    downstream_section: FlowSectionGeometry
    change_type: AreaChangeType
    length: float | None = None
    included_angle_deg: float | None = None

    def __post_init__(self) -> None:
        if self.upstream_section.flow_area == self.downstream_section.flow_area:
            raise ValueError(
                "area_change_equal_areas: upstream_section and downstream_section "
                "have equal flow_area; an area change requires a real area "
                "difference (use a straight section for an equal-area "
                "transition)."
            )
        if self.length is not None and (not math.isfinite(self.length) or self.length <= 0.0):
            raise ValueError("AreaChangeGeometry.length must be None or a positive, finite value.")
        if self.included_angle_deg is not None and (
            not math.isfinite(self.included_angle_deg)
            or not (0.0 < self.included_angle_deg <= 180.0)
        ):
            raise ValueError(
                "AreaChangeGeometry.included_angle_deg must be None or within (0, 180]."
            )

    @property
    def is_expansion(self) -> bool:
        return self.downstream_section.flow_area > self.upstream_section.flow_area

    @property
    def is_contraction(self) -> bool:
        return self.downstream_section.flow_area < self.upstream_section.flow_area


@dataclass(frozen=True)
class ScreenGeometry(PressureDropStageGeometry):
    """Screen / opening-array geometry (screen, mesh, perforated plate,
    flow straightener, tube-sheet opening array, parallel tube entrance or
    exit).

    Kept distinct from ``AreaChangeGeometry`` even though both may change
    flow area, because screens/opening arrays require different geometric
    descriptors (opening count/diameter, plate thickness, edge treatment).

    A geometry-based calculation for this stage is not implemented in this
    commit. When ``loss_coefficient`` is explicitly supplied, though, it is
    calculated (not a placeholder): ``dp = loss_coefficient * q``, referenced
    to the velocity through ``open_flow_area``; see
    ``core.pressure_drop.screens.calculate_screen_pressure_drop``. A screen
    with no ``loss_coefficient`` remains ``not_implemented``. For a
    blockage-only high-Re obstruction (bird net, wire mesh, grille, louver),
    use ``FlatObstructionGeometry`` instead.
    """

    screen_type: ScreenType
    upstream_area: float
    open_flow_area: float

    opening_count: int | None = None
    opening_diameter: float | None = None
    plate_thickness: float | None = None
    edge_radius: float | None = None
    bevel_angle_deg: float | None = None

    loss_coefficient: float | None = None


@dataclass(frozen=True)
class DirectionChangeGeometry(PressureDropStageGeometry):
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
class ChamberGeometry(PressureDropStageGeometry):
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
class UserDefinedPressureDropGeometry(PressureDropStageGeometry):
    """User-supplied pressure-drop stage.

    Exactly one of the two following supply modes must be used:

    - ``pressure_drop``: a direct fixed pressure drop, reported verbatim as
      a ``user_defined`` calculated stage (``dp_irreversible=pressure_drop``,
      ``delta_dynamic_pressure=0`` and ``dp_static=pressure_drop``).
    - ``loss_coefficient`` + ``reference_area``: ``dp_irreversible =
      loss_coefficient * rho * V_ref**2 / 2`` with ``V_ref = mass_flow /
      (rho * reference_area)`` taken from the calling
      ``PressureDropFlowState``; see
      ``core.pressure_drop.screens.calculate_user_defined_pressure_drop``.

    Construction itself stays permissive (a geometry with only
    ``loss_coefficient`` and no ``reference_area`` may still be built --
    e.g. while a path is being assembled incrementally); the supply-mode
    validation above is enforced by
    ``calculate_user_defined_pressure_drop`` when the stage is actually
    calculated, not at construction time. A stage supplying neither
    ``pressure_drop`` nor ``loss_coefficient`` is reported as
    ``not_implemented`` rather than raising.
    """

    pressure_drop: float | None = None
    loss_coefficient: float | None = None
    reference_area: float | None = None
    description: str = ""


@dataclass(frozen=True)
class PressureDropAssemblyGeometry:
    """An ordered sequence of pressure-drop stage geometries.

    Order is preserved exactly as supplied. A missing physical component is
    represented by its absence from ``stages`` -- never by a disabled/zeroed
    placeholder stage.
    """

    stages: tuple[PressureDropStageGeometry, ...]
