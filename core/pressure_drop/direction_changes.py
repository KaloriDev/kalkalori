# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

# NOTE ON UNITS
# -------------
# All calculations use SI units: rho [kg/m^3], V [m/s], dp [Pa], angles in
# degrees.

"""
Direction-change (elbow/bend) local pressure-drop calculations
(v0.5.6 local pressure-drop paths).

Covers the practical elbow matrix: circular and rectangular cross-section,
smooth-radius and segmented (mitered) construction, at any angle in
``(0, 180]`` degrees -- 45-degree, 90-degree and 180-degree/U-bend elbows
are all the same physical direction-change category (``angle_deg``), not
distinct geometry or calculation types.

This module is local-pressure-drop-path-only: it is never invoked by the
standard exchanger solver, only by explicitly invoked local-loss
calculations (see ``core.pressure_drop.flow_path``).

Scope of this commit
---------------------
``DirectionChangeMethod.USER_DEFINED_K`` (``dp = K * rho * V**2 / 2``) and
``EQUIVALENT_LENGTH`` (``K = f_D * Le/Dh`` via the canonical
``core.pressure_drop.straight_sections.darcy_friction_factor``) are fully
implemented for every elbow family (circular/rectangular,
smooth-radius/segmented, any angle).

``GEOMETRY_CORRELATION`` (automatic K purely from cross-section,
construction, angle and radius/segment-count) is architecturally supported
(the enum member and dispatch exist) but not populated with an automatic
correlation in this commit for any elbow family: published elbow
loss-coefficient correlations are curve/table-based (e.g. Miller, Idelchik)
and reproducing one exactly, without a verifiable source at hand, risks
silently invented coefficients used for real equipment sizing -- see the
"Do not invent coefficients" / "Do not copy proprietary ASHRAE tables"
constraints. A ``GEOMETRY_CORRELATION`` request therefore always reports
``not_implemented`` with an informational warning directing the caller to
``USER_DEFINED_K`` or ``EQUIVALENT_LENGTH``, exactly like the tube-sheet
entrance/exit module's not-yet-implemented geometries.

Literature references
----------------------
- Idelchik, I. E., "Handbook of Hydraulic Resistance".
- Crane Co., "Flow of Fluids Through Valves, Fittings, and Pipe" (TP-410).
- Miller, D. S., "Internal Flow Systems".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from core.common.warnings import make_warning
from core.geometry.pressure_drop_stages import PressureDropStageGeometry

if TYPE_CHECKING:
    from core.pressure_drop.flow_path import PressureDropFlowState, PressureDropStageResult


class ElbowCrossSection(str, Enum):
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"


class ElbowConstruction(str, Enum):
    SMOOTH_RADIUS = "smooth_radius"
    SEGMENTED = "segmented"


class RectangularTurnPlane(str, Enum):
    WIDTH = "width"
    HEIGHT = "height"


class DirectionChangeMethod(str, Enum):
    GEOMETRY_CORRELATION = "geometry_correlation"
    USER_DEFINED_K = "user_defined_k"
    EQUIVALENT_LENGTH = "equivalent_length"


def _validate_angle(angle_deg: float, label: str) -> None:
    if not math.isfinite(angle_deg) or not (0.0 < angle_deg <= 180.0):
        raise ValueError(f"{label}.angle_deg must be within (0, 180].")


def _validate_method_inputs(
    *,
    label: str,
    method: DirectionChangeMethod,
    loss_coefficient: float | None,
    equivalent_length_ratio: float | None,
) -> None:
    if method == DirectionChangeMethod.USER_DEFINED_K:
        if loss_coefficient is None:
            raise ValueError(f"{label}: method=USER_DEFINED_K requires loss_coefficient.")
        if not math.isfinite(loss_coefficient) or loss_coefficient < 0.0:
            raise ValueError(f"{label}.loss_coefficient must be non-negative and finite.")
    elif method == DirectionChangeMethod.EQUIVALENT_LENGTH:
        if equivalent_length_ratio is None:
            raise ValueError(f"{label}: method=EQUIVALENT_LENGTH requires equivalent_length_ratio.")
        if not math.isfinite(equivalent_length_ratio) or equivalent_length_ratio <= 0.0:
            raise ValueError(f"{label}.equivalent_length_ratio must be positive and finite.")


@dataclass(frozen=True)
class CircularElbowGeometry(PressureDropStageGeometry):
    """Circular-cross-section elbow/bend (45/90/180-degree, U-bend)."""

    diameter: float
    angle_deg: float
    construction: ElbowConstruction

    centerline_radius: float | None = None
    segment_count: int | None = None

    roughness: float = 0.0

    method: DirectionChangeMethod = DirectionChangeMethod.GEOMETRY_CORRELATION

    loss_coefficient: float | None = None
    equivalent_length_ratio: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.diameter) or self.diameter <= 0.0:
            raise ValueError("CircularElbowGeometry.diameter must be positive and finite.")
        _validate_angle(self.angle_deg, "CircularElbowGeometry")
        if not math.isfinite(self.roughness) or self.roughness < 0.0:
            raise ValueError("CircularElbowGeometry.roughness must be non-negative and finite.")

        if self.construction == ElbowConstruction.SMOOTH_RADIUS:
            if self.centerline_radius is None or self.centerline_radius <= 0.0:
                raise ValueError(
                    "circular_smooth_radius_elbow_requires_centerline_radius: "
                    "CircularElbowGeometry with construction=SMOOTH_RADIUS "
                    "requires a positive centerline_radius."
                )
            if self.segment_count is not None:
                raise ValueError(
                    "CircularElbowGeometry: segment_count is not applicable to "
                    "construction=SMOOTH_RADIUS."
                )
        elif self.construction == ElbowConstruction.SEGMENTED:
            if self.segment_count is None or self.segment_count < 1:
                raise ValueError(
                    "circular_segmented_elbow_requires_segment_count: "
                    "CircularElbowGeometry with construction=SEGMENTED requires "
                    "segment_count >= 1."
                )

        if self.centerline_radius is not None and (
            not math.isfinite(self.centerline_radius) or self.centerline_radius <= 0.0
        ):
            raise ValueError("CircularElbowGeometry.centerline_radius must be positive and finite.")

        _validate_method_inputs(
            label="CircularElbowGeometry",
            method=self.method,
            loss_coefficient=self.loss_coefficient,
            equivalent_length_ratio=self.equivalent_length_ratio,
        )

    @property
    def flow_area(self) -> float:
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def hydraulic_diameter(self) -> float:
        return self.diameter

    @property
    def radius_ratio(self) -> float | None:
        """``centerline_radius / diameter`` (``R/D``), or ``None`` if unset."""
        if self.centerline_radius is None:
            return None
        return self.centerline_radius / self.diameter


@dataclass(frozen=True)
class RectangularElbowGeometry(PressureDropStageGeometry):
    """Rectangular-cross-section elbow/bend (duct turn), with an explicit
    turn plane (``WIDTH`` or ``HEIGHT`` is the dimension that turns)."""

    width: float
    height: float

    turn_plane: RectangularTurnPlane
    angle_deg: float
    construction: ElbowConstruction

    centerline_radius: float | None = None
    inner_radius: float | None = None
    segment_count: int | None = None

    turning_vane_count: int = 0
    roughness: float = 0.0

    method: DirectionChangeMethod = DirectionChangeMethod.GEOMETRY_CORRELATION

    loss_coefficient: float | None = None
    equivalent_length_ratio: float | None = None

    def __post_init__(self) -> None:
        for name, value in (("width", self.width), ("height", self.height)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"RectangularElbowGeometry.{name} must be positive and finite.")
        _validate_angle(self.angle_deg, "RectangularElbowGeometry")
        if not math.isfinite(self.roughness) or self.roughness < 0.0:
            raise ValueError("RectangularElbowGeometry.roughness must be non-negative and finite.")
        if self.turning_vane_count < 0:
            raise ValueError("RectangularElbowGeometry.turning_vane_count must be non-negative.")

        if self.construction == ElbowConstruction.SMOOTH_RADIUS:
            if self.effective_centerline_radius is None or self.effective_centerline_radius <= 0.0:
                raise ValueError(
                    "rectangular_smooth_radius_elbow_requires_radius: "
                    "RectangularElbowGeometry with construction=SMOOTH_RADIUS "
                    "requires a positive centerline_radius, or an inner_radius "
                    "from which to derive it (inner_radius + turning_dimension/2)."
                )
        elif self.construction == ElbowConstruction.SEGMENTED:
            if self.segment_count is None or self.segment_count < 1:
                raise ValueError(
                    "rectangular_segmented_elbow_requires_segment_count: "
                    "RectangularElbowGeometry with construction=SEGMENTED "
                    "requires segment_count >= 1."
                )

        _validate_method_inputs(
            label="RectangularElbowGeometry",
            method=self.method,
            loss_coefficient=self.loss_coefficient,
            equivalent_length_ratio=self.equivalent_length_ratio,
        )

    @property
    def flow_area(self) -> float:
        return self.width * self.height

    @property
    def hydraulic_diameter(self) -> float:
        return 2.0 * self.width * self.height / (self.width + self.height)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def turning_dimension(self) -> float:
        """The in-plane dimension that turns: ``width`` or ``height``
        depending on ``turn_plane``."""
        return self.width if self.turn_plane == RectangularTurnPlane.WIDTH else self.height

    @property
    def effective_centerline_radius(self) -> float | None:
        """``centerline_radius`` if supplied directly; otherwise derived from
        ``inner_radius + turning_dimension / 2``; ``None`` if neither is
        available."""
        if self.centerline_radius is not None:
            return self.centerline_radius
        if self.inner_radius is not None:
            return self.inner_radius + self.turning_dimension / 2.0
        return None

    @property
    def radius_ratio(self) -> float | None:
        """``effective_centerline_radius / turning_dimension``, or ``None``
        if no radius is available."""
        radius = self.effective_centerline_radius
        if radius is None:
            return None
        return radius / self.turning_dimension


def _not_implemented_correlation_warning(stage_id: str, family_label: str):
    return make_warning(
        code="direction_change_geometry_correlation_not_implemented",
        message=(
            f"direction_change: stage '{stage_id}' requests method="
            f"GEOMETRY_CORRELATION, but no automatic geometry correlation is "
            f"implemented for {family_label} in this commit; supply "
            "method=USER_DEFINED_K or method=EQUIVALENT_LENGTH."
        ),
        source="direction_change_pressure_drop",
        severity="info",
    )


def _calculate_elbow_pressure_drop(
    *,
    geometry,
    state: "PressureDropFlowState",
    stage_id: str,
    stage_type: str,
    family_label: str,
) -> "PressureDropStageResult":
    from core.pressure_drop.flow_path import (
        PressureDropStageResult,
        PressureDropStageStatus,
        evaluate_section_flow,
    )
    from core.pressure_drop.straight_sections import (
        darcy_friction_factor,
        darcy_friction_factor_method,
    )

    flow = evaluate_section_flow(state, geometry)
    relative_roughness = geometry.roughness / geometry.hydraulic_diameter

    warnings = []
    f = None
    f_method = None

    if geometry.method == DirectionChangeMethod.USER_DEFINED_K:
        K = geometry.loss_coefficient
        method = "user_defined_loss_coefficient"
        status = PressureDropStageStatus.USER_DEFINED
    elif geometry.method == DirectionChangeMethod.EQUIVALENT_LENGTH:
        f = darcy_friction_factor(flow.reynolds, relative_roughness)
        f_method = darcy_friction_factor_method(flow.reynolds, relative_roughness)
        K = f * geometry.equivalent_length_ratio
        method = "equivalent_length"
        status = PressureDropStageStatus.CALCULATED
    else:
        K = None
        method = None
        status = PressureDropStageStatus.NOT_IMPLEMENTED
        warnings.append(_not_implemented_correlation_warning(stage_id, family_label))

    dp_irreversible = 0.0 if K is None else K * flow.dynamic_pressure

    return PressureDropStageResult(
        stage_id=stage_id,
        stage_type=stage_type,
        status=status,
        dp_irreversible=dp_irreversible,
        delta_dynamic_pressure=0.0,
        method=method,
        warnings=tuple(warnings),
        loss_coefficient=K,
        reference_area=geometry.flow_area,
        reference_velocity=flow.velocity,
        reference_dynamic_pressure=flow.dynamic_pressure,
        upstream_area=geometry.flow_area,
        downstream_area=geometry.flow_area,
        upstream_velocity=flow.velocity,
        downstream_velocity=flow.velocity,
        reynolds=flow.reynolds,
        friction_factor=f,
        friction_factor_method=f_method,
        relative_roughness=relative_roughness,
    )


def calculate_circular_elbow_pressure_drop(
    *,
    geometry: CircularElbowGeometry,
    state: "PressureDropFlowState",
    stage_id: str,
) -> "PressureDropStageResult":
    """Circular elbow pressure drop (any angle, smooth-radius or segmented).

    ``dp_irreversible = K * rho * V**2 / 2`` at the elbow's own circular
    velocity; equal-area fitting, so ``delta_dynamic_pressure = 0``. See the
    module docstring for the supported ``DirectionChangeMethod`` values.
    """
    return _calculate_elbow_pressure_drop(
        geometry=geometry,
        state=state,
        stage_id=stage_id,
        stage_type="circular_elbow",
        family_label=f"circular {geometry.construction.value} elbows",
    )


def calculate_rectangular_elbow_pressure_drop(
    *,
    geometry: RectangularElbowGeometry,
    state: "PressureDropFlowState",
    stage_id: str,
) -> "PressureDropStageResult":
    """Rectangular elbow pressure drop (any angle, smooth-radius or
    segmented, with or without turning vanes).

    ``dp_irreversible = K * rho * V**2 / 2`` at the elbow's own rectangular
    velocity; equal-area fitting, so ``delta_dynamic_pressure = 0``. Vaned
    and unvaned geometry are never treated as equivalent: ``turning_vane_count``
    is part of the geometry identity even though this commit's
    ``EQUIVALENT_LENGTH``/``USER_DEFINED_K`` methods do not vary the
    resulting K by it automatically (the caller supplies K/Le-D for the
    actual vaned or unvaned construction).
    """
    return _calculate_elbow_pressure_drop(
        geometry=geometry,
        state=state,
        stage_id=stage_id,
        stage_type="rectangular_elbow",
        family_label=f"rectangular {geometry.construction.value} elbows",
    )
