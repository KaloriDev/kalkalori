# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Focused tests for the v0.5.6 explicit local pressure-drop path models.

Covers: PressureDropFlowState/section-flow helpers, cross-section geometry,
straight sections, gradual/sudden expansion and contraction, mixed-shape
transitions, circular/rectangular smooth-radius and segmented elbows (45,
90, 180 degrees), elbow calculation methods, flat obstructions (blockage
model), user-defined losses, a mixed assembly, the explicit tube-side/
outside-side path calculators, and standard-solver separation (the
standard solver never accepts or evaluates local-loss geometry).
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.geometry.pressure_drop_stages import (
    AreaChangeGeometry,
    AreaChangeType,
    CircularFlowSection,
    CustomFlowSection,
    DirectionChangeGeometry,
    DirectionChangeType,
    PressureDropAssemblyGeometry,
    RectangularFlowSection,
    ScreenGeometry,
    ScreenType,
    StraightSectionGeometry,
    UserDefinedPressureDropGeometry,
)
from core.geometry.outside_pressure_drop_path import SpecifiedOutsidePressureDropPath
from core.geometry.tube_side_pressure_drop_path import SpecifiedTubeSidePressureDropPath
from core.heat_transfer.outside_flow import calculate_outside_tube_bank_hydraulics
from core.heat_transfer.streams import SensibleHeatStream
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.pressure_drop.area_changes import calculate_area_change_pressure_drop
from core.pressure_drop.direction_changes import (
    CircularElbowGeometry,
    DirectionChangeMethod,
    ElbowConstruction,
    RectangularElbowGeometry,
    RectangularTurnPlane,
    calculate_circular_elbow_pressure_drop,
    calculate_rectangular_elbow_pressure_drop,
)
from core.pressure_drop.flow_path import (
    PressureDropFlowState,
    PressureDropStageResult,
    PressureDropStageStatus,
    calculate_outside_pressure_drop_path,
    calculate_pressure_drop_assembly,
    calculate_tube_side_pressure_drop_path,
    evaluate_section_flow,
)
from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
from core.pressure_drop.screens import (
    FlatObstructionGeometry,
    FlatObstructionType,
    calculate_flat_obstruction_pressure_drop,
    calculate_screen_pressure_drop,
    calculate_user_defined_pressure_drop,
)
from core.pressure_drop.straight_sections import (
    calculate_straight_section_pressure_drop,
    darcy_friction_factor,
)
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


def _props(rho: float = 1.2, mu: float = 1.8e-5, k: float = 0.026, cp: float = 1006.0) -> FluidTransportProperties:
    return FluidTransportProperties(rho=rho, mu=mu, k=k, cp=cp)


def _state(mass_flow: float = 1.0, temperature: float = 300.0, pressure: float = 101325.0, **props_kwargs) -> PressureDropFlowState:
    return PressureDropFlowState(
        mass_flow=mass_flow, temperature=temperature, pressure=pressure, props=_props(**props_kwargs),
    )


def _stage_result(
    *,
    dp_irreversible: float = 10.0,
    delta_dynamic_pressure: float = 0.0,
) -> PressureDropStageResult:
    return PressureDropStageResult(
        stage_id="stage",
        stage_type="test_stage",
        status=PressureDropStageStatus.CALCULATED,
        dp_irreversible=dp_irreversible,
        delta_dynamic_pressure=delta_dynamic_pressure,
        method="test",
        warnings=(),
    )


def _bundle(n_passes_tube: int = 1) -> TubeBundle:
    tube = BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0)
    return TubeBundle(
        tube=tube, n_rows=4, n_tubes_per_row=6,
        pitch_transverse=0.04, pitch_longitudinal=0.04,
        layout="staggered", n_passes_tube=n_passes_tube, flow_arrangement="crossflow",
    )


def _tube_bundle_result(bundle: TubeBundle):
    return calculate_tube_bundle_hydraulics(
        m_dot=1.0,
        flow_area_per_pass=bundle.internal_flow_area_per_pass,
        hydraulic_diameter=bundle.internal_hydraulic_diameter,
        hydraulic_length_total=bundle.internal_length_total,
        n_tube_passes=bundle.n_passes_tube,
        inlet_props=_props(rho=1000.0, mu=1.0e-3, k=0.6, cp=4180.0),
    )


def _tube_bank_result(bundle: TubeBundle):
    return calculate_outside_tube_bank_hydraulics(
        m_dot=1.0,
        face_area=bundle.frontal_flow_area,
        tube_outer_diameter=bundle.tube.D_o,
        tube_pitch_transverse=bundle.pitch_transverse,
        tube_pitch_longitudinal=bundle.pitch_longitudinal,
        layout=bundle.layout,
        n_rows=bundle.n_rows,
        n_tubes_per_row=bundle.n_tubes_per_row,
        inlet_props=_props(),
    )


# ---------------------------------------------------------------------------
# 16.1 Flow state
# ---------------------------------------------------------------------------

def test_flow_state_validation_rejects_non_positive_values() -> None:
    props = _props()
    with pytest.raises(ValueError):
        PressureDropFlowState(mass_flow=0.0, temperature=300.0, pressure=101325.0, props=props)
    with pytest.raises(ValueError):
        PressureDropFlowState(mass_flow=1.0, temperature=-1.0, pressure=101325.0, props=props)
    with pytest.raises(ValueError):
        PressureDropFlowState(mass_flow=1.0, temperature=300.0, pressure=0.0, props=props)
    with pytest.raises(ValueError):
        PressureDropFlowState(mass_flow=float("nan"), temperature=300.0, pressure=101325.0, props=props)


def test_evaluate_section_flow_mass_flux_velocity_dynamic_pressure_reynolds() -> None:
    section = CircularFlowSection(diameter=0.2)
    state = _state(mass_flow=2.0, rho=1.2, mu=1.8e-5)
    flow = evaluate_section_flow(state, section)

    A = math.pi * 0.2 ** 2 / 4.0
    G = 2.0 / A
    V = G / 1.2
    q = 1.2 * V ** 2 / 2.0
    Re = 1.2 * V * 0.2 / 1.8e-5

    assert math.isclose(flow.mass_flux, G)
    assert math.isclose(flow.velocity, V)
    assert math.isclose(flow.dynamic_pressure, q)
    assert math.isclose(flow.reynolds, Re, rel_tol=1e-9)


@pytest.mark.parametrize("invalid", [-1.0, math.nan, math.inf, -math.inf])
def test_stage_result_rejects_invalid_irreversible_loss(invalid: float) -> None:
    with pytest.raises(ValueError):
        _stage_result(dp_irreversible=invalid)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_stage_result_rejects_non_finite_dynamic_pressure_change(invalid: float) -> None:
    with pytest.raises(ValueError):
        _stage_result(delta_dynamic_pressure=invalid)


def test_stage_result_rejects_non_finite_derived_static_pressure_difference() -> None:
    with pytest.raises(ValueError):
        _stage_result(dp_irreversible=1.0e308, delta_dynamic_pressure=1.0e308)


def test_stage_result_allows_static_pressure_recovery() -> None:
    result = _stage_result(dp_irreversible=10.0, delta_dynamic_pressure=-25.0)

    assert result.dp_irreversible == 10.0
    assert result.delta_dynamic_pressure == -25.0
    assert result.dp_static == -15.0


# ---------------------------------------------------------------------------
# 16.2 Flow sections
# ---------------------------------------------------------------------------

def test_circular_flow_section() -> None:
    s = CircularFlowSection(diameter=0.5)
    assert math.isclose(s.flow_area, math.pi * 0.5 ** 2 / 4.0)
    assert s.hydraulic_diameter == 0.5
    assert s.equivalent_circular_diameter == 0.5


def test_rectangular_flow_section() -> None:
    s = RectangularFlowSection(width=0.4, height=0.2)
    assert math.isclose(s.flow_area, 0.08)
    assert math.isclose(s.hydraulic_diameter, 2.0 * 0.4 * 0.2 / (0.4 + 0.2))
    assert math.isclose(s.equivalent_circular_diameter, math.sqrt(4.0 * 0.08 / math.pi))


def test_custom_flow_section() -> None:
    s = CustomFlowSection(area=0.05, hydraulic_diameter=0.22)
    assert s.flow_area == 0.05
    assert s.hydraulic_diameter == 0.22
    assert math.isclose(s.equivalent_circular_diameter, math.sqrt(4.0 * 0.05 / math.pi))


def test_flow_section_validation_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        CircularFlowSection(diameter=0.0)
    with pytest.raises(ValueError):
        RectangularFlowSection(width=-1.0, height=0.2)
    with pytest.raises(ValueError):
        CustomFlowSection(area=0.05, hydraulic_diameter=0.0)


# ---------------------------------------------------------------------------
# 16.3 Straight sections
# ---------------------------------------------------------------------------

def test_straight_section_smooth_circular_reconstruction() -> None:
    geometry = StraightSectionGeometry(flow_area=0.05, hydraulic_diameter=0.25, length=10.0)
    state = _state(mass_flow=3.0)
    result = calculate_straight_section_pressure_drop(geometry=geometry, state=state, stage_id="s")

    flow = evaluate_section_flow(state, geometry)
    f = darcy_friction_factor(flow.reynolds, 0.0)
    expected = f * geometry.length / geometry.hydraulic_diameter * flow.dynamic_pressure

    assert math.isclose(result.dp_irreversible, expected)
    assert result.dp_irreversible > 0.0
    assert result.delta_dynamic_pressure == 0.0
    assert result.dp_static == result.dp_irreversible
    assert result.status == PressureDropStageStatus.CALCULATED


def test_straight_section_rough_circular_reconstruction() -> None:
    geometry = StraightSectionGeometry(flow_area=0.05, hydraulic_diameter=0.25, length=10.0, roughness=0.5e-3)
    state = _state(mass_flow=3.0)
    result = calculate_straight_section_pressure_drop(geometry=geometry, state=state, stage_id="s")

    flow = evaluate_section_flow(state, geometry)
    relative_roughness = 0.5e-3 / 0.25
    f = darcy_friction_factor(flow.reynolds, relative_roughness)
    expected = f * geometry.length / geometry.hydraulic_diameter * flow.dynamic_pressure

    assert math.isclose(result.dp_irreversible, expected)
    assert result.relative_roughness == relative_roughness
    assert result.friction_factor == f


def test_straight_section_smooth_rectangular_reconstruction() -> None:
    section = RectangularFlowSection(width=0.4, height=0.2)
    geometry = StraightSectionGeometry(flow_area=section.flow_area, hydraulic_diameter=section.hydraulic_diameter, length=5.0)
    state = _state(mass_flow=1.0)
    result = calculate_straight_section_pressure_drop(geometry=geometry, state=state, stage_id="s")

    flow = evaluate_section_flow(state, geometry)
    f = darcy_friction_factor(flow.reynolds, 0.0)
    expected = f * geometry.length / geometry.hydraulic_diameter * flow.dynamic_pressure
    assert math.isclose(result.dp_irreversible, expected)


def test_straight_section_rough_rectangular_reconstruction() -> None:
    section = RectangularFlowSection(width=0.4, height=0.2)
    geometry = StraightSectionGeometry(
        flow_area=section.flow_area, hydraulic_diameter=section.hydraulic_diameter,
        length=5.0, roughness=0.2e-3,
    )
    state = _state(mass_flow=1.0)
    result = calculate_straight_section_pressure_drop(geometry=geometry, state=state, stage_id="s")

    flow = evaluate_section_flow(state, geometry)
    relative_roughness = 0.2e-3 / geometry.hydraulic_diameter
    f = darcy_friction_factor(flow.reynolds, relative_roughness)
    expected = f * geometry.length / geometry.hydraulic_diameter * flow.dynamic_pressure
    assert math.isclose(result.dp_irreversible, expected)


# ---------------------------------------------------------------------------
# 16.4 Sudden expansion
# ---------------------------------------------------------------------------

def test_sudden_expansion_matches_borda_carnot_form() -> None:
    upstream = CircularFlowSection(diameter=0.2)
    downstream = CircularFlowSection(diameter=0.3)
    geometry = AreaChangeGeometry(upstream_section=upstream, downstream_section=downstream, change_type=AreaChangeType.SUDDEN)
    state = _state(mass_flow=2.0)
    result = calculate_area_change_pressure_drop(geometry=geometry, state=state, stage_id="ac")

    R = upstream.flow_area / downstream.flow_area
    K_expected = (1.0 - R) ** 2
    up_flow = evaluate_section_flow(state, upstream)
    down_flow = evaluate_section_flow(state, downstream)
    dp_irr_expected = K_expected * up_flow.dynamic_pressure
    delta_dynamic_expected = down_flow.dynamic_pressure - up_flow.dynamic_pressure
    dp_static_expected = dp_irr_expected + delta_dynamic_expected

    assert math.isclose(result.loss_coefficient, K_expected)
    assert math.isclose(result.dp_irreversible, dp_irr_expected)
    assert math.isclose(result.delta_dynamic_pressure, delta_dynamic_expected)
    assert math.isclose(result.dp_static, dp_static_expected)
    assert result.dp_irreversible > 0.0
    assert result.delta_dynamic_pressure < 0.0
    assert result.dp_static < 0.0


def test_area_change_rejects_equal_areas() -> None:
    with pytest.raises(ValueError):
        AreaChangeGeometry(
            upstream_section=CircularFlowSection(diameter=0.2),
            downstream_section=CircularFlowSection(diameter=0.2),
            change_type=AreaChangeType.SUDDEN,
        )


# ---------------------------------------------------------------------------
# 16.5 Gradual expansion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle_deg", [20.0, 45.0, 90.0])
def test_gradual_expansion_branches(angle_deg: float) -> None:
    upstream = CircularFlowSection(diameter=0.2)
    downstream = CircularFlowSection(diameter=0.3)
    geometry = AreaChangeGeometry(
        upstream_section=upstream, downstream_section=downstream,
        change_type=AreaChangeType.GRADUAL, included_angle_deg=angle_deg,
    )
    state = _state(mass_flow=2.0)
    result = calculate_area_change_pressure_drop(geometry=geometry, state=state, stage_id="ac")

    R = upstream.flow_area / downstream.flow_area
    theta_rad = math.radians(angle_deg)
    if angle_deg <= 45.0:
        K_expected = 2.6 * math.sin(theta_rad / 2.0) * (1.0 - R) ** 2
    else:
        K_expected = (1.0 - R) ** 2

    assert math.isclose(result.loss_coefficient, K_expected)
    up_flow = evaluate_section_flow(state, upstream)
    assert math.isclose(result.reference_velocity, up_flow.velocity)


def test_area_change_gradual_without_angle_or_length_raises_at_calculation() -> None:
    geometry = AreaChangeGeometry(
        upstream_section=CircularFlowSection(diameter=0.2),
        downstream_section=CircularFlowSection(diameter=0.3),
        change_type=AreaChangeType.GRADUAL,
    )
    state = _state(mass_flow=1.0)
    with pytest.raises(ValueError):
        calculate_area_change_pressure_drop(geometry=geometry, state=state, stage_id="ac")


# ---------------------------------------------------------------------------
# 16.6 Sudden contraction
# ---------------------------------------------------------------------------

def test_sudden_contraction_matches_crane_form() -> None:
    upstream = CircularFlowSection(diameter=0.3)
    downstream = CircularFlowSection(diameter=0.2)
    geometry = AreaChangeGeometry(upstream_section=upstream, downstream_section=downstream, change_type=AreaChangeType.SUDDEN)
    state = _state(mass_flow=2.0)
    result = calculate_area_change_pressure_drop(geometry=geometry, state=state, stage_id="ac")

    R = downstream.flow_area / upstream.flow_area
    K_expected = 0.5 * (1.0 - R)
    upstream_flow = evaluate_section_flow(state, upstream)
    downstream_flow = evaluate_section_flow(state, downstream)
    dp_irreversible_expected = K_expected * downstream_flow.dynamic_pressure
    delta_dynamic_expected = downstream_flow.dynamic_pressure - upstream_flow.dynamic_pressure

    assert math.isclose(result.loss_coefficient, K_expected)
    assert math.isclose(result.dp_irreversible, dp_irreversible_expected)
    assert math.isclose(result.delta_dynamic_pressure, delta_dynamic_expected)
    assert result.dp_irreversible > 0.0
    assert result.delta_dynamic_pressure > 0.0
    assert result.dp_static == pytest.approx(
        result.dp_irreversible + result.delta_dynamic_pressure
    )
    assert result.dp_static > result.dp_irreversible


# ---------------------------------------------------------------------------
# 16.7 Gradual contraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle_deg", [30.0, 90.0])
def test_gradual_contraction_branches(angle_deg: float) -> None:
    upstream = CircularFlowSection(diameter=0.3)
    downstream = CircularFlowSection(diameter=0.2)
    geometry = AreaChangeGeometry(
        upstream_section=upstream, downstream_section=downstream,
        change_type=AreaChangeType.GRADUAL, included_angle_deg=angle_deg,
    )
    state = _state(mass_flow=2.0)
    result = calculate_area_change_pressure_drop(geometry=geometry, state=state, stage_id="ac")

    R = downstream.flow_area / upstream.flow_area
    theta_rad = math.radians(angle_deg)
    if angle_deg <= 45.0:
        K_expected = 0.8 * math.sin(theta_rad / 2.0) * (1.0 - R)
    else:
        K_expected = 0.5 * math.sqrt(math.sin(theta_rad / 2.0)) * (1.0 - R)

    assert math.isclose(result.loss_coefficient, K_expected)
    down_flow = evaluate_section_flow(state, downstream)
    assert math.isclose(result.reference_velocity, down_flow.velocity)


# ---------------------------------------------------------------------------
# 16.8 Mixed-shape transition
# ---------------------------------------------------------------------------

def test_mixed_shape_transition_equivalent_angle_and_warning() -> None:
    upstream = CircularFlowSection(diameter=0.9)
    downstream = RectangularFlowSection(width=2.0, height=3.0)
    geometry = AreaChangeGeometry(
        upstream_section=upstream, downstream_section=downstream,
        change_type=AreaChangeType.GRADUAL, length=1.5,
    )
    state = _state(mass_flow=8.0)
    result = calculate_area_change_pressure_drop(geometry=geometry, state=state, stage_id="ac")

    assert math.isfinite(result.dp_static)
    assert any(w.code == "area_change_equivalent_angle_approximation" for w in result.warnings)
    up_flow = evaluate_section_flow(state, upstream)
    down_flow = evaluate_section_flow(state, downstream)
    assert math.isclose(result.upstream_velocity, up_flow.velocity)
    assert math.isclose(result.downstream_velocity, down_flow.velocity)


# ---------------------------------------------------------------------------
# 16.9 Circular smooth-radius elbows (45/90/180 degrees, two R/D values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle_deg", [45.0, 90.0, 180.0])
@pytest.mark.parametrize("radius_ratio", [1.0, 1.5])
def test_circular_smooth_radius_elbow_user_defined_k(angle_deg: float, radius_ratio: float) -> None:
    diameter = 0.3
    geometry = CircularElbowGeometry(
        diameter=diameter, angle_deg=angle_deg, construction=ElbowConstruction.SMOOTH_RADIUS,
        centerline_radius=radius_ratio * diameter,
        method=DirectionChangeMethod.USER_DEFINED_K, loss_coefficient=0.3,
    )
    state = _state(mass_flow=1.0)
    result = calculate_circular_elbow_pressure_drop(geometry=geometry, state=state, stage_id="e")

    flow = evaluate_section_flow(state, geometry)
    assert math.isclose(result.dp_irreversible, 0.3 * flow.dynamic_pressure)
    assert result.dp_irreversible > 0.0
    assert result.delta_dynamic_pressure == 0.0
    assert result.dp_static == result.dp_irreversible
    assert result.status == PressureDropStageStatus.USER_DEFINED
    assert math.isclose(geometry.radius_ratio, radius_ratio)


def test_circular_smooth_radius_requires_centerline_radius() -> None:
    with pytest.raises(ValueError):
        CircularElbowGeometry(diameter=0.2, angle_deg=90.0, construction=ElbowConstruction.SMOOTH_RADIUS)


# ---------------------------------------------------------------------------
# 16.10 Circular segmented elbows (45/90/180 degrees, multiple segment counts)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle_deg", [45.0, 90.0, 180.0])
@pytest.mark.parametrize("segment_count", [2, 4])
def test_circular_segmented_elbow_equivalent_length(angle_deg: float, segment_count: int) -> None:
    geometry = CircularElbowGeometry(
        diameter=0.25, angle_deg=angle_deg, construction=ElbowConstruction.SEGMENTED,
        segment_count=segment_count,
        method=DirectionChangeMethod.EQUIVALENT_LENGTH, equivalent_length_ratio=20.0,
    )
    state = _state(mass_flow=1.0)
    result = calculate_circular_elbow_pressure_drop(geometry=geometry, state=state, stage_id="e")

    flow = evaluate_section_flow(state, geometry)
    f = darcy_friction_factor(flow.reynolds, 0.0)
    K_expected = f * 20.0
    assert math.isclose(result.loss_coefficient, K_expected)
    assert result.status == PressureDropStageStatus.CALCULATED
    assert geometry.segment_count == segment_count


def test_circular_segmented_requires_segment_count() -> None:
    with pytest.raises(ValueError):
        CircularElbowGeometry(diameter=0.2, angle_deg=90.0, construction=ElbowConstruction.SEGMENTED)


# ---------------------------------------------------------------------------
# 16.11 Rectangular smooth-radius elbows (45/90/180, width/height planes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle_deg", [45.0, 90.0, 180.0])
@pytest.mark.parametrize(
    "turn_plane,width,height",
    [(RectangularTurnPlane.WIDTH, 0.6, 0.3), (RectangularTurnPlane.HEIGHT, 0.3, 0.6)],
)
def test_rectangular_smooth_radius_elbow(angle_deg: float, turn_plane: RectangularTurnPlane, width: float, height: float) -> None:
    geometry = RectangularElbowGeometry(
        width=width, height=height, turn_plane=turn_plane, angle_deg=angle_deg,
        construction=ElbowConstruction.SMOOTH_RADIUS, centerline_radius=0.5,
        method=DirectionChangeMethod.USER_DEFINED_K, loss_coefficient=0.25,
    )
    state = _state(mass_flow=1.0)
    result = calculate_rectangular_elbow_pressure_drop(geometry=geometry, state=state, stage_id="e")

    flow = evaluate_section_flow(state, geometry)
    assert math.isclose(result.dp_irreversible, 0.25 * flow.dynamic_pressure)
    assert result.delta_dynamic_pressure == 0.0
    assert result.dp_static == result.dp_irreversible
    assert math.isclose(geometry.aspect_ratio, width / height)
    assert geometry.turning_dimension == (width if turn_plane == RectangularTurnPlane.WIDTH else height)


def test_rectangular_smooth_radius_requires_radius() -> None:
    with pytest.raises(ValueError):
        RectangularElbowGeometry(
            width=0.4, height=0.3, turn_plane=RectangularTurnPlane.WIDTH, angle_deg=90.0,
            construction=ElbowConstruction.SMOOTH_RADIUS,
        )


def test_rectangular_smooth_radius_derives_from_inner_radius() -> None:
    geometry = RectangularElbowGeometry(
        width=0.4, height=0.3, turn_plane=RectangularTurnPlane.WIDTH, angle_deg=90.0,
        construction=ElbowConstruction.SMOOTH_RADIUS, inner_radius=0.1,
        method=DirectionChangeMethod.USER_DEFINED_K, loss_coefficient=0.2,
    )
    assert math.isclose(geometry.effective_centerline_radius, 0.1 + 0.4 / 2.0)


# ---------------------------------------------------------------------------
# 16.12 Rectangular segmented elbows, with/without turning vanes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle_deg", [45.0, 90.0, 180.0])
@pytest.mark.parametrize("turning_vane_count", [0, 3])
def test_rectangular_segmented_elbow_vaned_vs_unvaned(angle_deg: float, turning_vane_count: int) -> None:
    geometry = RectangularElbowGeometry(
        width=0.5, height=0.4, turn_plane=RectangularTurnPlane.WIDTH, angle_deg=angle_deg,
        construction=ElbowConstruction.SEGMENTED, segment_count=3,
        turning_vane_count=turning_vane_count,
        method=DirectionChangeMethod.EQUIVALENT_LENGTH, equivalent_length_ratio=15.0,
    )
    state = _state(mass_flow=1.0)
    result = calculate_rectangular_elbow_pressure_drop(geometry=geometry, state=state, stage_id="e")

    assert geometry.turning_vane_count == turning_vane_count
    assert result.status == PressureDropStageStatus.CALCULATED


# ---------------------------------------------------------------------------
# 16.13 Elbow calculation methods
# ---------------------------------------------------------------------------

def test_circular_elbow_geometry_correlation_not_implemented_with_warning() -> None:
    geometry = CircularElbowGeometry(
        diameter=0.2, angle_deg=90.0, construction=ElbowConstruction.SMOOTH_RADIUS, centerline_radius=0.3,
    )  # default method=GEOMETRY_CORRELATION
    state = _state(mass_flow=1.0)
    result = calculate_circular_elbow_pressure_drop(geometry=geometry, state=state, stage_id="e")

    assert result.status == PressureDropStageStatus.NOT_IMPLEMENTED
    assert result.dp_irreversible == 0.0
    assert result.delta_dynamic_pressure == 0.0
    assert result.dp_static == 0.0
    assert any(w.code == "direction_change_geometry_correlation_not_implemented" for w in result.warnings)


def test_elbow_user_defined_k_requires_loss_coefficient() -> None:
    with pytest.raises(ValueError):
        CircularElbowGeometry(
            diameter=0.2, angle_deg=90.0, construction=ElbowConstruction.SMOOTH_RADIUS,
            centerline_radius=0.3, method=DirectionChangeMethod.USER_DEFINED_K,
        )


def test_elbow_equivalent_length_requires_ratio() -> None:
    with pytest.raises(ValueError):
        RectangularElbowGeometry(
            width=0.4, height=0.3, turn_plane=RectangularTurnPlane.WIDTH, angle_deg=90.0,
            construction=ElbowConstruction.SEGMENTED, segment_count=2,
            method=DirectionChangeMethod.EQUIVALENT_LENGTH,
        )


# ---------------------------------------------------------------------------
# 16.14 Flat obstruction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blockage_percent", [0.0, 10.0, 25.0, 50.0, 75.0])
def test_flat_obstruction_blockage_formula(blockage_percent: float) -> None:
    B = blockage_percent / 100.0
    face = RectangularFlowSection(width=2.0, height=1.0)
    geometry = FlatObstructionGeometry(face_section=face, blockage_ratio=B)
    state = _state(mass_flow=5.0)
    result = calculate_flat_obstruction_pressure_drop(geometry=geometry, state=state, stage_id="fo")

    beta = 1.0 - B
    K_expected = B + (B / beta) ** 2
    assert math.isclose(result.loss_coefficient, K_expected)
    assert result.delta_dynamic_pressure == 0.0
    assert result.dp_static == result.dp_irreversible
    if B == 0.0:
        assert result.loss_coefficient == 0.0
        assert result.dp_irreversible == 0.0
        assert result.warnings == ()
    else:
        assert result.dp_irreversible > 0.0
        assert any(w.code == "flat_obstruction_blockage_only_model_limitation" for w in result.warnings)


def test_flat_obstruction_pressure_drop_monotonic_in_blockage() -> None:
    face = RectangularFlowSection(width=2.0, height=1.0)
    state = _state(mass_flow=5.0)
    dps = []
    for pct in (0.0, 10.0, 25.0, 40.0, 50.0, 60.0, 75.0):
        geometry = FlatObstructionGeometry(face_section=face, blockage_ratio=pct / 100.0)
        result = calculate_flat_obstruction_pressure_drop(geometry=geometry, state=state, stage_id="fo")
        dps.append(result.dp_irreversible)
    assert all(dps[i] < dps[i + 1] for i in range(len(dps) - 1))


def test_flat_obstruction_rejects_invalid_blockage() -> None:
    face = RectangularFlowSection(width=2.0, height=1.0)
    with pytest.raises(ValueError):
        FlatObstructionGeometry(face_section=face, blockage_ratio=-0.1)
    with pytest.raises(ValueError):
        FlatObstructionGeometry(face_section=face, blockage_ratio=1.0)


def test_flat_obstruction_type_is_descriptive_only() -> None:
    face = RectangularFlowSection(width=2.0, height=1.0)
    state = _state(mass_flow=5.0)
    values = []
    for obstruction_type in FlatObstructionType:
        geometry = FlatObstructionGeometry(face_section=face, blockage_ratio=0.3, obstruction_type=obstruction_type)
        result = calculate_flat_obstruction_pressure_drop(geometry=geometry, state=state, stage_id="fo")
        values.append(result.dp_irreversible)
    assert all(math.isclose(v, values[0]) for v in values)


# ---------------------------------------------------------------------------
# General screen with explicit K
# ---------------------------------------------------------------------------

def test_screen_with_explicit_k() -> None:
    geometry = ScreenGeometry(screen_type=ScreenType.SCREEN, upstream_area=0.03, open_flow_area=0.02, loss_coefficient=0.8)
    state = _state(mass_flow=1.5)
    result = calculate_screen_pressure_drop(geometry=geometry, state=state, stage_id="sc")

    rho = state.props.rho
    V_ref = state.mass_flow / (rho * 0.02)
    q_ref = rho * V_ref ** 2 / 2.0
    assert math.isclose(result.dp_irreversible, 0.8 * q_ref)
    assert result.delta_dynamic_pressure == 0.0
    assert result.dp_static == result.dp_irreversible
    assert result.reference_area == geometry.open_flow_area
    assert result.upstream_area == geometry.upstream_area
    assert result.downstream_area == geometry.upstream_area
    assert result.status == PressureDropStageStatus.USER_DEFINED


def test_screen_without_k_raises_from_calculator() -> None:
    geometry = ScreenGeometry(screen_type=ScreenType.SCREEN, upstream_area=0.03, open_flow_area=0.02)
    state = _state(mass_flow=1.5)
    with pytest.raises(ValueError):
        calculate_screen_pressure_drop(geometry=geometry, state=state, stage_id="sc")


def test_screen_without_k_is_not_implemented_via_assembly_dispatch() -> None:
    geometry = ScreenGeometry(screen_type=ScreenType.SCREEN, upstream_area=0.03, open_flow_area=0.02)
    assembly = PressureDropAssemblyGeometry(stages=(geometry,))
    state = _state(mass_flow=1.5)
    group = calculate_pressure_drop_assembly(group_id="inlet", geometry=assembly, state=state)
    assert group.stages[0].status == PressureDropStageStatus.NOT_IMPLEMENTED


# ---------------------------------------------------------------------------
# 16.15 User-defined loss
# ---------------------------------------------------------------------------

def test_user_defined_fixed_dp() -> None:
    geometry = UserDefinedPressureDropGeometry(pressure_drop=120.0)
    state = _state(mass_flow=1.0)
    result = calculate_user_defined_pressure_drop(geometry=geometry, state=state, stage_id="ud")
    assert result.dp_irreversible == 120.0
    assert result.delta_dynamic_pressure == 0.0
    assert result.dp_static == 120.0
    assert result.status == PressureDropStageStatus.USER_DEFINED


def test_user_defined_k_plus_reference_area() -> None:
    geometry = UserDefinedPressureDropGeometry(loss_coefficient=1.2, reference_area=0.1)
    state = _state(mass_flow=2.0)
    result = calculate_user_defined_pressure_drop(geometry=geometry, state=state, stage_id="ud")

    rho = state.props.rho
    V_ref = state.mass_flow / (rho * 0.1)
    q_ref = rho * V_ref ** 2 / 2.0
    assert math.isclose(result.dp_irreversible, 1.2 * q_ref)
    assert result.delta_dynamic_pressure == 0.0
    assert result.dp_static == result.dp_irreversible


def test_user_defined_both_supplied_raises() -> None:
    state = _state(mass_flow=1.0)
    geometry = UserDefinedPressureDropGeometry(pressure_drop=10.0, loss_coefficient=0.5, reference_area=0.1)
    with pytest.raises(ValueError):
        calculate_user_defined_pressure_drop(geometry=geometry, state=state, stage_id="ud")


def test_user_defined_k_missing_reference_area_raises() -> None:
    state = _state(mass_flow=1.0)
    geometry = UserDefinedPressureDropGeometry(loss_coefficient=0.5)
    with pytest.raises(ValueError):
        calculate_user_defined_pressure_drop(geometry=geometry, state=state, stage_id="ud")


def test_user_defined_rejects_negative_values() -> None:
    state = _state(mass_flow=1.0)
    with pytest.raises(ValueError):
        calculate_user_defined_pressure_drop(
            geometry=UserDefinedPressureDropGeometry(pressure_drop=-5.0), state=state, stage_id="ud",
        )
    with pytest.raises(ValueError):
        calculate_user_defined_pressure_drop(
            geometry=UserDefinedPressureDropGeometry(loss_coefficient=-1.0, reference_area=0.1),
            state=state, stage_id="ud",
        )


def test_user_defined_neither_supplied_is_not_implemented() -> None:
    geometry = UserDefinedPressureDropGeometry(description="future")
    state = _state(mass_flow=1.0)
    result = calculate_user_defined_pressure_drop(geometry=geometry, state=state, stage_id="ud")
    assert result.status == PressureDropStageStatus.NOT_IMPLEMENTED
    assert result.dp_irreversible == 0.0
    assert result.delta_dynamic_pressure == 0.0
    assert result.dp_static == 0.0


# ---------------------------------------------------------------------------
# 16.16 Mixed assembly
# ---------------------------------------------------------------------------

def test_mixed_assembly_dispatch_order_and_totals() -> None:
    straight = StraightSectionGeometry(flow_area=0.05, hydraulic_diameter=0.25, length=3.0)
    area_change = AreaChangeGeometry(
        upstream_section=CircularFlowSection(diameter=0.25),
        downstream_section=CircularFlowSection(diameter=0.3),
        change_type=AreaChangeType.SUDDEN,
    )
    elbow = CircularElbowGeometry(
        diameter=0.3, angle_deg=90.0, construction=ElbowConstruction.SMOOTH_RADIUS,
        centerline_radius=0.45, method=DirectionChangeMethod.USER_DEFINED_K, loss_coefficient=0.3,
    )
    obstruction = FlatObstructionGeometry(face_section=CircularFlowSection(diameter=0.3), blockage_ratio=0.1)
    user_defined = UserDefinedPressureDropGeometry(pressure_drop=50.0)
    not_implemented_stage = DirectionChangeGeometry(change_type=DirectionChangeType.RETURN_CHAMBER, angle_deg=180.0)

    assembly = PressureDropAssemblyGeometry(
        stages=(straight, area_change, elbow, obstruction, user_defined, not_implemented_stage),
    )
    state = _state(mass_flow=2.0)
    group = calculate_pressure_drop_assembly(group_id="inlet", geometry=assembly, state=state)

    assert len(group.stages) == 6
    assert [s.status for s in group.stages] == [
        PressureDropStageStatus.CALCULATED,
        PressureDropStageStatus.CALCULATED,
        PressureDropStageStatus.USER_DEFINED,
        PressureDropStageStatus.CALCULATED,
        PressureDropStageStatus.USER_DEFINED,
        PressureDropStageStatus.NOT_IMPLEMENTED,
    ]
    expected_irreversible = sum(s.dp_irreversible for s in group.stages)
    expected_delta_dynamic = sum(s.delta_dynamic_pressure for s in group.stages)
    assert group.dp_irreversible == pytest.approx(expected_irreversible)
    assert group.delta_dynamic_pressure == pytest.approx(expected_delta_dynamic)
    assert group.dp_static == pytest.approx(
        group.dp_irreversible + group.delta_dynamic_pressure
    )
    assert group.dp_irreversible > 0.0
    assert group.dp_static < 0.0
    assert group.stages[-1].dp_irreversible == 0.0
    assert group.stages[-1].delta_dynamic_pressure == 0.0
    assert group.stages[-1].dp_static == 0.0
    assert len(group.stages[-1].warnings) == 1


# ---------------------------------------------------------------------------
# 16.17 / 16.18 Explicit side-specific path calculation
# ---------------------------------------------------------------------------

def test_explicit_outside_pressure_drop_path_totals() -> None:
    bundle = _bundle(n_passes_tube=1)
    tube_bank = _tube_bank_result(bundle)
    small = CircularFlowSection(diameter=0.2)
    medium = CircularFlowSection(diameter=0.3)
    large = CircularFlowSection(diameter=0.4)
    inlet_assembly = PressureDropAssemblyGeometry(stages=(
        AreaChangeGeometry(
            upstream_section=small,
            downstream_section=large,
            change_type=AreaChangeType.SUDDEN,
        ),
    ))
    outlet_assembly = PressureDropAssemblyGeometry(stages=(
        AreaChangeGeometry(
            upstream_section=large,
            downstream_section=medium,
            change_type=AreaChangeType.SUDDEN,
        ),
    ))
    path = SpecifiedOutsidePressureDropPath(inlet=inlet_assembly, outlet=outlet_assembly)
    inlet_state = _state(mass_flow=2.0)
    outlet_state = _state(mass_flow=2.0)

    result = calculate_outside_pressure_drop_path(
        tube_bank=tube_bank, path=path, inlet_state=inlet_state, outlet_state=outlet_state,
    )
    inlet_group = result.group("inlet")
    outlet_group = result.group("outlet")
    assert inlet_group is not None
    assert outlet_group is not None

    expected_local_irreversible = inlet_group.dp_irreversible + outlet_group.dp_irreversible
    expected_local_delta_dynamic = (
        inlet_group.delta_dynamic_pressure + outlet_group.delta_dynamic_pressure
    )
    old_incorrect_static_aggregation = inlet_group.dp_static + outlet_group.dp_static

    assert result.dp_core == pytest.approx(tube_bank.dp_drag)
    assert result.delta_dynamic_pressure_core == pytest.approx(tube_bank.dp_acceleration)
    assert result.dp_static_core == pytest.approx(tube_bank.dp_total)
    assert result.dp_local == pytest.approx(expected_local_irreversible)
    assert result.delta_dynamic_pressure_local == pytest.approx(expected_local_delta_dynamic)
    assert result.dp_static_local == pytest.approx(
        result.dp_local + result.delta_dynamic_pressure_local
    )
    assert result.dp_total == pytest.approx(result.dp_core + result.dp_local)
    assert result.delta_dynamic_pressure_total == pytest.approx(
        result.delta_dynamic_pressure_core + result.delta_dynamic_pressure_local
    )
    assert result.dp_static_total == pytest.approx(
        result.dp_total + result.delta_dynamic_pressure_total
    )
    assert result.dp_local > 0.0
    assert result.dp_static_local < 0.0
    assert result.dp_local != pytest.approx(old_incorrect_static_aggregation)
    assert [g.group_id for g in result.groups] == ["inlet", "tube_bank", "outlet"]


def test_explicit_tube_side_pressure_drop_path_totals() -> None:
    bundle = _bundle(n_passes_tube=3)
    tube_bundle = _tube_bundle_result(bundle)
    inlet_assembly = PressureDropAssemblyGeometry(stages=(UserDefinedPressureDropGeometry(pressure_drop=10.0),))
    return_assembly = PressureDropAssemblyGeometry(stages=(UserDefinedPressureDropGeometry(pressure_drop=5.0),))
    outlet_assembly = PressureDropAssemblyGeometry(stages=(UserDefinedPressureDropGeometry(pressure_drop=8.0),))
    path = SpecifiedTubeSidePressureDropPath(
        inlet=inlet_assembly, returns=(return_assembly, return_assembly), outlet=outlet_assembly,
    )
    state = _state(mass_flow=1.0)
    result = calculate_tube_side_pressure_drop_path(
        tube_bundle=tube_bundle, n_tube_passes=3, path=path,
        inlet_state=state, return_states=(state, state), outlet_state=state,
    )
    expected_local = 10.0 + 5.0 + 5.0 + 8.0
    expected_core = (
        tube_bundle.dp_straight_tube_friction
        + tube_bundle.dp_tube_entrances
        + tube_bundle.dp_tube_exits
    )
    assert result.dp_core == pytest.approx(expected_core)
    assert result.delta_dynamic_pressure_core == pytest.approx(
        tube_bundle.dp_straight_tube_acceleration
    )
    assert result.dp_static_core == pytest.approx(tube_bundle.dp_tube_bundle)
    assert result.dp_local == pytest.approx(expected_local)
    assert result.delta_dynamic_pressure_local == 0.0
    assert result.dp_static_local == pytest.approx(expected_local)
    assert result.dp_total == pytest.approx(expected_core + expected_local)
    assert result.dp_static_total == pytest.approx(
        tube_bundle.dp_tube_bundle + expected_local
    )
    assert [g.group_id for g in result.groups] == ["inlet", "tube_bundle", "return_1", "return_2", "outlet"]


def test_explicit_tube_side_path_rejects_return_state_count_mismatch() -> None:
    bundle = _bundle(n_passes_tube=3)
    tube_bundle = _tube_bundle_result(bundle)
    empty = PressureDropAssemblyGeometry(stages=())
    path = SpecifiedTubeSidePressureDropPath(inlet=empty, returns=(empty, empty), outlet=empty)
    state = _state(mass_flow=1.0)
    with pytest.raises(ValueError):
        calculate_tube_side_pressure_drop_path(
            tube_bundle=tube_bundle, n_tube_passes=3, path=path,
            inlet_state=state, return_states=(state,), outlet_state=state,
        )


# ---------------------------------------------------------------------------
# 16.19 Standard solver separation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name", ["solve", "simulate", "rate"])
def test_standard_solver_does_not_accept_local_path_arguments(method_name: str) -> None:
    import inspect

    sig = inspect.signature(getattr(BareTubeHeatExchanger, method_name))
    assert "tube_side_pressure_drop_path" not in sig.parameters
    assert "outside_side_pressure_drop_path" not in sig.parameters


def _assert_standard_pressure_drop_boundary(result) -> None:
    tube_path = result.tube_side_pressure_drop
    outside_path = result.outside_side_pressure_drop

    assert tube_path.dp_local == 0.0
    assert tube_path.delta_dynamic_pressure_local == 0.0
    assert tube_path.dp_static_local == 0.0
    assert tube_path.dp_total == tube_path.dp_core
    assert tube_path.dp_static_core == pytest.approx(result.inside_dp_total)
    assert tube_path.dp_static_total == pytest.approx(result.inside_dp_total)

    assert outside_path.dp_local == 0.0
    assert outside_path.delta_dynamic_pressure_local == 0.0
    assert outside_path.dp_static_local == 0.0
    assert outside_path.dp_total == outside_path.dp_core
    assert outside_path.dp_static_core == pytest.approx(result.outside_dp_total)
    assert outside_path.dp_static_total == pytest.approx(result.outside_dp_total)

    # Deterministic legacy core-only regression values for this constant-
    # property one-pass fixture. The local-path semantic correction must not
    # alter the standard solver's established hydraulic calculations.
    assert result.inside_dp_total == pytest.approx(54.979666056152126)
    assert result.outside_dp_total == pytest.approx(61.089932891358)


def test_standard_solve_simulate_and_rate_remain_core_only() -> None:
    bundle = _bundle(n_passes_tube=1)
    hx = BareTubeHeatExchanger(bundle)
    solved = hx.solve(
        hot_stream=SensibleHeatStream(C=1000.0, T_in=350.0),
        cold_stream=SensibleHeatStream(C=500.0, T_in=300.0),
        m_dot_tube_side=1.0,
        tube_side_props=_props(rho=1000.0, mu=1.0e-3, k=0.6, cp=4180.0),
        tube_side_temperature_in=350.0, tube_side_temperature_out=340.0, tube_side_pressure=101325.0,
        m_dot_outside=2.0,
        outside_props=_props(),
        outside_temperature_in=300.0, outside_temperature_out=310.0, outside_pressure=101325.0,
    )

    inside_provider = ConstantPropertyProvider(
        _props(rho=1000.0, mu=1.0e-3, k=0.6, cp=4180.0)
    )
    outside_provider = ConstantPropertyProvider(_props())
    inside = HXSideInput(
        provider=inside_provider, m_dot=1.0, T_in=350.0, p=101325.0
    )
    outside = HXSideInput(
        provider=outside_provider, m_dot=2.0, T_in=300.0, p=101325.0
    )
    simulated = hx.simulate(inside, outside, max_iter=5)
    rated = hx.rate(
        BalanceSideSpec(
            provider=inside_provider,
            m_dot=1.0,
            p=101325.0,
            T_in=350.0,
            T_out=340.0,
        ),
        BalanceSideSpec(
            provider=outside_provider,
            m_dot=2.0,
            p=101325.0,
            T_in=300.0,
            T_out=310.0,
        ),
        Q=1000.0,
    )

    for result in (solved, simulated, rated):
        _assert_standard_pressure_drop_boundary(result)
