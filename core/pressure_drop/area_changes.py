# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

# NOTE ON UNITS
# -------------
# All calculations use SI units: rho [kg/m^3], V [m/s], dp [Pa], angles in
# degrees for included_angle_deg (converted internally where needed).

"""
Area-change (expansion/contraction) local pressure-drop calculations
(v0.5.6 local pressure-drop paths).

Implements the Gibson/Crane gradual and sudden expansion/contraction
correlations for ``core.geometry.pressure_drop_stages.AreaChangeGeometry``.
This module is local-pressure-drop-path-only: it is never invoked by the
standard exchanger solver (``BareTubeHeatExchanger.solve``/``simulate``/
``rate``), only by explicitly invoked local-loss calculations (see
``core.pressure_drop.flow_path.calculate_pressure_drop_assembly`` /
``calculate_tube_side_pressure_drop_path`` /
``calculate_outside_pressure_drop_path``).

Sign convention
----------------
``dp_irreversible >= 0`` always. ``delta_dynamic_pressure = q_downstream -
q_upstream``: positive for a contraction (velocity increases), negative for
an expansion (velocity decreases). The signed static-pressure difference is
``dp_static = dp_irreversible + delta_dynamic_pressure``. A diffuser may
therefore have ``dp_static < 0`` (static-pressure recovery) while its
irreversible loss remains positive.

Literature references
----------------------
- Crane Co., "Flow of Fluids Through Valves, Fittings, and Pipe" (TP-410).
- Gibson, A. H. -- gradual enlargement/contraction loss coefficients.
- Borda-Carnot sudden expansion (the limiting case of the Gibson expansion
  form at ``theta=180`` degrees, referenced to the upstream/small velocity).
- Idelchik, I. E., "Handbook of Hydraulic Resistance".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.common.warnings import ModelWarning, make_warning
from core.geometry.pressure_drop_stages import (
    AreaChangeGeometry,
    AreaChangeType,
    CircularFlowSection,
)

if TYPE_CHECKING:
    from core.pressure_drop.flow_path import PressureDropFlowState, PressureDropStageResult


def _equivalent_included_angle_deg(
    geometry: AreaChangeGeometry,
) -> tuple[float, list[ModelWarning]]:
    """The included angle used by the Gibson/Crane correlation below.

    ``SUDDEN`` always uses 180 degrees. ``GRADUAL`` uses
    ``included_angle_deg`` directly when supplied; otherwise it is derived
    from ``length`` and the area-equivalent circular diameters of both
    sections (an approximation for a non-axisymmetric transition, flagged
    with an informational warning).
    """
    if geometry.change_type == AreaChangeType.SUDDEN:
        return 180.0, []
    if geometry.included_angle_deg is not None:
        return geometry.included_angle_deg, []
    if geometry.length is None:
        raise ValueError(
            "area_change_gradual_requires_angle_or_length: a GRADUAL "
            "AreaChangeGeometry requires either included_angle_deg or a "
            "positive length to derive an equivalent included angle."
        )

    d_up = geometry.upstream_section.equivalent_circular_diameter
    d_down = geometry.downstream_section.equivalent_circular_diameter
    theta_rad = 2.0 * math.atan(abs(d_down - d_up) / (2.0 * geometry.length))
    theta_deg = math.degrees(theta_rad)

    warnings: list[ModelWarning] = []
    if not (
        isinstance(geometry.upstream_section, CircularFlowSection)
        and isinstance(geometry.downstream_section, CircularFlowSection)
    ):
        warnings.append(
            make_warning(
                code="area_change_equivalent_angle_approximation",
                message=(
                    "area_change: The transition angle is based on "
                    "area-equivalent circular diameters. This is an "
                    "approximation for a non-axisymmetric transition."
                ),
                source="area_change_pressure_drop",
                severity="info",
            )
        )
    return theta_deg, warnings


def calculate_area_change_pressure_drop(
    *,
    geometry: AreaChangeGeometry,
    state: "PressureDropFlowState",
    stage_id: str,
) -> "PressureDropStageResult":
    """
    Gibson/Crane gradual or sudden expansion/contraction pressure change.

    Expansion (``downstream_section.flow_area > upstream_section.flow_area``),
    referenced to the upstream (small-section) velocity::

        theta <= 45 deg: K = 2.6 * sin(theta/2) * (1 - R)**2
        theta >  45 deg: K = (1 - R)**2

    Contraction, referenced to the downstream (small-section) velocity::

        theta <= 45 deg: K = 0.8 * sin(theta/2) * (1 - R)
        theta >  45 deg: K = 0.5 * sqrt(sin(theta/2)) * (1 - R)

    with ``R = A_small / A_large`` (area ratio, not diameter ratio).
    """
    from core.pressure_drop.flow_path import (
        PressureDropStageResult,
        PressureDropStageStatus,
        evaluate_section_flow,
    )

    upstream_flow = evaluate_section_flow(state, geometry.upstream_section)
    downstream_flow = evaluate_section_flow(state, geometry.downstream_section)

    A_up = geometry.upstream_section.flow_area
    A_down = geometry.downstream_section.flow_area
    is_expansion = geometry.is_expansion

    A_small = min(A_up, A_down)
    A_large = max(A_up, A_down)
    R = A_small / A_large

    theta_deg, warnings = _equivalent_included_angle_deg(geometry)
    theta_rad = math.radians(theta_deg)

    if is_expansion:
        if theta_deg <= 45.0:
            K = 2.6 * math.sin(theta_rad / 2.0) * (1.0 - R) ** 2
        else:
            K = (1.0 - R) ** 2
        reference_flow = upstream_flow
        reference_area = A_up
        method = (
            "gibson_crane_sudden_expansion"
            if geometry.change_type == AreaChangeType.SUDDEN
            else "gibson_crane_gradual_expansion"
        )
    else:
        if theta_deg <= 45.0:
            K = 0.8 * math.sin(theta_rad / 2.0) * (1.0 - R)
        else:
            K = 0.5 * math.sqrt(math.sin(theta_rad / 2.0)) * (1.0 - R)
        reference_flow = downstream_flow
        reference_area = A_down
        method = (
            "gibson_crane_sudden_contraction"
            if geometry.change_type == AreaChangeType.SUDDEN
            else "gibson_crane_gradual_contraction"
        )

    dp_irreversible = K * reference_flow.dynamic_pressure
    delta_dynamic_pressure = (
        downstream_flow.dynamic_pressure - upstream_flow.dynamic_pressure
    )

    return PressureDropStageResult(
        stage_id=stage_id,
        stage_type=f"{geometry.change_type.value}_area_change",
        status=PressureDropStageStatus.CALCULATED,
        dp_irreversible=dp_irreversible,
        delta_dynamic_pressure=delta_dynamic_pressure,
        method=method,
        warnings=tuple(warnings),
        loss_coefficient=K,
        reference_area=reference_area,
        reference_velocity=reference_flow.velocity,
        reference_dynamic_pressure=reference_flow.dynamic_pressure,
        upstream_area=A_up,
        downstream_area=A_down,
        upstream_velocity=upstream_flow.velocity,
        downstream_velocity=downstream_flow.velocity,
    )
