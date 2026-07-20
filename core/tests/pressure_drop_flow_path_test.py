# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Focused tests for the v0.5.6 pressure-drop flow-path architecture.

Covers: import compatibility (canonical vs. legacy paths), numerical
regression against the v0.5.5 straight tube-bundle / bare tube-bank scope,
stage-group ordering, placeholder (not_implemented) stage status,
return-count validation, missing-path compatibility, and the future
suggested-geometry request types being explicitly rejected rather than
silently accepted.
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.geometry.pressure_drop_stages import (
    AreaChangeGeometry,
    AreaChangeType,
    ChamberGeometry,
    CircularFlowSection,
    DirectionChangeGeometry,
    DirectionChangeType,
    PressureDropAssemblyGeometry,
    ScreenGeometry,
    ScreenType,
    StraightSectionGeometry,
    UserDefinedPressureDropGeometry,
)
from core.geometry.outside_pressure_drop_path import (
    OutsidePressureDropDesignRequest,
    SpecifiedOutsidePressureDropPath,
)
from core.geometry.tube_side_pressure_drop_path import (
    SpecifiedTubeSidePressureDropPath,
    TubeSidePressureDropDesignRequest,
    validate_specified_tube_side_path,
)
from core.heat_transfer.outside_flow import calculate_outside_tube_bank_hydraulics
from core.heat_transfer.streams import SensibleHeatStream
from core.models.bare_tube import BareTubeHeatExchanger
from core.pressure_drop.flow_path import (
    PressureDropStageStatus,
    build_outside_pressure_drop_result,
    build_tube_side_pressure_drop_result,
)
from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
from core.pressure_drop.direction_changes import (
    CircularElbowGeometry,
    ElbowConstruction,
    RectangularElbowGeometry,
    RectangularTurnPlane,
)
from core.pressure_drop.screens import FlatObstructionGeometry
from core.properties.common import FluidTransportProperties


def _props(rho: float, mu: float = 1.0e-3, k: float = 0.6, cp: float = 4180.0) -> FluidTransportProperties:
    return FluidTransportProperties(rho=rho, mu=mu, k=k, cp=cp)


def _bundle(n_passes_tube: int = 1) -> TubeBundle:
    tube = BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0)
    return TubeBundle(
        # 4 * 6 = 24 tubes total, divisible by 1, 2, and 3 passes.
        tube=tube, n_rows=4, n_tubes_per_row=6,
        pitch_transverse=0.04, pitch_longitudinal=0.04,
        layout="staggered", n_passes_tube=n_passes_tube, flow_arrangement="crossflow",
    )


def _tube_bundle_result(bundle: TubeBundle, *, variable_density: bool = False):
    point_props = (
        {
            "inlet_props": _props(1000.0),
            "midpoint_props": _props(900.0),
            "outlet_props": _props(800.0),
        }
        if variable_density
        else {"inlet_props": _props(1000.0)}
    )
    return calculate_tube_bundle_hydraulics(
        m_dot=1.0,
        flow_area_per_pass=bundle.internal_flow_area_per_pass,
        hydraulic_diameter=bundle.internal_hydraulic_diameter,
        hydraulic_length_total=bundle.internal_length_total,
        n_tube_passes=bundle.n_passes_tube,
        **point_props,
    )


def _tube_bank_result(bundle: TubeBundle, *, variable_density: bool = False):
    point_props = (
        {
            "inlet_props": _props(1.2, mu=1.8e-5, k=0.026, cp=1006.0),
            "midpoint_props": _props(1.1, mu=1.8e-5, k=0.026, cp=1006.0),
            "outlet_props": _props(1.0, mu=1.8e-5, k=0.026, cp=1006.0),
        }
        if variable_density
        else {"inlet_props": _props(1.2, mu=1.8e-5, k=0.026, cp=1006.0)}
    )
    return calculate_outside_tube_bank_hydraulics(
        m_dot=1.0,
        face_area=bundle.frontal_flow_area,
        tube_outer_diameter=bundle.tube.D_o,
        tube_pitch_transverse=bundle.pitch_transverse,
        tube_pitch_longitudinal=bundle.pitch_longitudinal,
        layout=bundle.layout,
        n_rows=bundle.n_rows,
        n_tubes_per_row=bundle.n_tubes_per_row,
        **point_props,
    )


# ---------------------------------------------------------------------------
# 18.1 Import compatibility
# ---------------------------------------------------------------------------

def test_canonical_import_from_core_pressure_drop() -> None:
    from core.pressure_drop import (
        calculate_tube_bundle_hydraulics as canonical_fn,
        EulerRequest,
        EulerResult,
        evaluate_euler,
        ExternalCliEulerProvider,
        build_tube_side_pressure_drop_result as canonical_builder,
    )
    assert canonical_fn is calculate_tube_bundle_hydraulics
    assert canonical_builder is build_tube_side_pressure_drop_result
    assert EulerRequest is not None and EulerResult is not None and evaluate_euler is not None
    assert ExternalCliEulerProvider is not None


def test_legacy_imports_resolve_to_same_implementation_objects() -> None:
    from core.heat_transfer.internal_pressure_drop import (
        calculate_tube_bundle_hydraulics as legacy_internal_fn,
        TubeBundleHydraulicResult as LegacyTubeBundleHydraulicResult,
    )
    from core.pressure_drop.internal_pressure_drop import (
        calculate_tube_bundle_hydraulics as canonical_internal_fn,
        TubeBundleHydraulicResult as CanonicalTubeBundleHydraulicResult,
    )
    assert legacy_internal_fn is canonical_internal_fn
    assert LegacyTubeBundleHydraulicResult is CanonicalTubeBundleHydraulicResult

    from core.heat_transfer.outside_pressure_drop import (
        EulerRequest as LegacyEulerRequest,
        evaluate_euler as legacy_evaluate_euler,
    )
    from core.pressure_drop.outside_pressure_drop import (
        EulerRequest as CanonicalEulerRequest,
        evaluate_euler as canonical_evaluate_euler,
    )
    assert LegacyEulerRequest is CanonicalEulerRequest
    assert legacy_evaluate_euler is canonical_evaluate_euler

    from core.heat_transfer.outside_pressure_drop_external import (
        ExternalCliEulerProvider as LegacyExternalProvider,
    )
    from core.pressure_drop.outside_pressure_drop_external import (
        ExternalCliEulerProvider as CanonicalExternalProvider,
    )
    assert LegacyExternalProvider is CanonicalExternalProvider

    # core.heat_transfer's own top-level re-exports must also resolve to the
    # same canonical objects (not a second, divergent implementation).
    import core.heat_transfer as heat_transfer_pkg
    import core.pressure_drop as pressure_drop_pkg
    assert heat_transfer_pkg.calculate_tube_bundle_hydraulics is pressure_drop_pkg.calculate_tube_bundle_hydraulics
    assert heat_transfer_pkg.EulerRequest is pressure_drop_pkg.EulerRequest
    assert heat_transfer_pkg.ExternalCliEulerProvider is pressure_drop_pkg.ExternalCliEulerProvider


# ---------------------------------------------------------------------------
# 18.2 Numerical regression and core semantic mapping
# ---------------------------------------------------------------------------

def test_tube_side_numerical_regression_against_v055_scope() -> None:
    bundle = _bundle(n_passes_tube=1)
    hx = BareTubeHeatExchanger(bundle)
    result = hx.solve(
        hot_stream=SensibleHeatStream(C=1000.0, T_in=350.0),
        cold_stream=SensibleHeatStream(C=500.0, T_in=300.0),
        m_dot_tube_side=1.0,
        tube_side_props=_props(1000.0),
        tube_side_temperature_in=350.0, tube_side_temperature_out=340.0, tube_side_pressure=101325.0,
        m_dot_outside=2.0,
        outside_props=_props(1.2, mu=1.8e-5, k=0.026, cp=1006.0),
        outside_temperature_in=300.0, outside_temperature_out=310.0, outside_pressure=101325.0,
    )
    dp = result.tube_side_pressure_drop
    tube_bundle = result.tube_bundle_hydraulic
    expected_irreversible = (
        tube_bundle.dp_straight_tube_friction
        + tube_bundle.dp_tube_entrances
        + tube_bundle.dp_tube_exits
    )
    assert dp.dp_core == pytest.approx(expected_irreversible)
    assert dp.dp_local == 0.0
    assert dp.delta_dynamic_pressure_core == pytest.approx(
        tube_bundle.dp_straight_tube_acceleration
    )
    assert dp.delta_dynamic_pressure_local == 0.0
    assert dp.dp_static_core == pytest.approx(result.inside_dp_tube_bundle)
    assert dp.dp_static_local == 0.0
    assert dp.dp_static_total == pytest.approx(result.inside_dp_total)
    assert dp.dp_total == dp.dp_core + dp.dp_local
    # Legacy standard-solver result remains the signed core pressure
    # difference and is unchanged by the explicit-path semantic correction.
    assert result.inside_dp_total == result.inside_dp_tube_bundle


def test_outside_numerical_regression_against_v055_scope() -> None:
    bundle = _bundle(n_passes_tube=1)
    hx = BareTubeHeatExchanger(bundle)
    result = hx.solve(
        hot_stream=SensibleHeatStream(C=1000.0, T_in=350.0),
        cold_stream=SensibleHeatStream(C=500.0, T_in=300.0),
        m_dot_tube_side=1.0,
        tube_side_props=_props(1000.0),
        tube_side_temperature_in=350.0, tube_side_temperature_out=340.0, tube_side_pressure=101325.0,
        m_dot_outside=2.0,
        outside_props=_props(1.2, mu=1.8e-5, k=0.026, cp=1006.0),
        outside_temperature_in=300.0, outside_temperature_out=310.0, outside_pressure=101325.0,
    )
    dp = result.outside_side_pressure_drop
    tube_bank = result.outside_tube_bank_hydraulic
    assert tube_bank is not None
    assert dp.dp_core == pytest.approx(tube_bank.dp_drag)
    assert dp.dp_local == 0.0
    assert dp.delta_dynamic_pressure_core == pytest.approx(tube_bank.dp_acceleration)
    assert dp.delta_dynamic_pressure_local == 0.0
    assert dp.dp_static_core == pytest.approx(tube_bank.dp_total)
    assert dp.dp_static_local == 0.0
    assert dp.dp_static_total == pytest.approx(result.outside_dp_total)
    assert dp.dp_total == dp.dp_core + dp.dp_local
    assert result.outside_dp_total == tube_bank.dp_total


def test_tube_side_core_mapping_preserves_nonzero_acceleration_separately() -> None:
    bundle = _bundle(n_passes_tube=1)
    tube_bundle = _tube_bundle_result(bundle, variable_density=True)
    result = build_tube_side_pressure_drop_result(
        tube_bundle, n_tube_passes=bundle.n_passes_tube,
    )
    expected_irreversible = (
        tube_bundle.dp_straight_tube_friction
        + tube_bundle.dp_tube_entrances
        + tube_bundle.dp_tube_exits
    )

    assert tube_bundle.dp_straight_tube_acceleration != 0.0
    assert result.dp_core == pytest.approx(expected_irreversible)
    assert result.delta_dynamic_pressure_core == pytest.approx(
        tube_bundle.dp_straight_tube_acceleration
    )
    assert result.dp_static_core == pytest.approx(tube_bundle.dp_tube_bundle)
    assert result.dp_total == pytest.approx(expected_irreversible)
    assert result.dp_static_total == pytest.approx(tube_bundle.dp_tube_bundle)
    assert result.dp_core != pytest.approx(result.dp_static_core)


def test_outside_core_mapping_preserves_nonzero_acceleration_separately() -> None:
    bundle = _bundle(n_passes_tube=1)
    tube_bank = _tube_bank_result(bundle, variable_density=True)
    result = build_outside_pressure_drop_result(tube_bank)

    assert tube_bank.dp_acceleration != 0.0
    assert result.dp_core == pytest.approx(tube_bank.dp_drag)
    assert result.delta_dynamic_pressure_core == pytest.approx(tube_bank.dp_acceleration)
    assert result.dp_static_core == pytest.approx(tube_bank.dp_total)
    assert result.dp_total == pytest.approx(tube_bank.dp_drag)
    assert result.dp_static_total == pytest.approx(tube_bank.dp_total)
    assert result.dp_core != pytest.approx(result.dp_static_core)


# ---------------------------------------------------------------------------
# 18.3 Stage ordering
# ---------------------------------------------------------------------------

def test_tube_side_group_order_without_path() -> None:
    bundle = _bundle(n_passes_tube=1)
    result = build_tube_side_pressure_drop_result(
        _tube_bundle_result(bundle), n_tube_passes=bundle.n_passes_tube,
    )
    assert [group.group_id for group in result.groups] == ["inlet", "tube_bundle", "returns", "outlet"]


def test_tube_side_group_order_with_multi_pass_path() -> None:
    bundle = _bundle(n_passes_tube=3)
    empty_assembly = PressureDropAssemblyGeometry(stages=())
    path = SpecifiedTubeSidePressureDropPath(
        inlet=empty_assembly,
        returns=(empty_assembly, empty_assembly),
        outlet=empty_assembly,
    )
    result = build_tube_side_pressure_drop_result(
        _tube_bundle_result(bundle), n_tube_passes=bundle.n_passes_tube, path=path,
    )
    assert [group.group_id for group in result.groups] == [
        "inlet", "tube_bundle", "return_1", "return_2", "outlet",
    ]


def test_outside_side_group_order() -> None:
    bundle = _bundle(n_passes_tube=1)
    result = build_outside_pressure_drop_result(_tube_bank_result(bundle))
    assert [group.group_id for group in result.groups] == ["inlet", "tube_bank", "outlet"]


# ---------------------------------------------------------------------------
# 18.4 Placeholder status
# ---------------------------------------------------------------------------

def test_representative_local_stages_are_not_implemented_and_zero() -> None:
    bundle = _bundle(n_passes_tube=1)
    inlet = PressureDropAssemblyGeometry(stages=(
        StraightSectionGeometry(flow_area=0.01, hydraulic_diameter=0.1, length=0.5),
        AreaChangeGeometry(
            change_type=AreaChangeType.SUDDEN,
            upstream_section=CircularFlowSection(diameter=0.16),
            downstream_section=CircularFlowSection(diameter=0.113),
        ),
        ScreenGeometry(screen_type=ScreenType.SCREEN, upstream_area=0.02, open_flow_area=0.015),
        DirectionChangeGeometry(change_type=DirectionChangeType.ELBOW, angle_deg=90.0),
        ChamberGeometry(flow_area=0.03),
        CircularElbowGeometry(
            diameter=0.1,
            angle_deg=90.0,
            construction=ElbowConstruction.SMOOTH_RADIUS,
            centerline_radius=0.15,
        ),
        RectangularElbowGeometry(
            width=0.2,
            height=0.1,
            turn_plane=RectangularTurnPlane.WIDTH,
            angle_deg=90.0,
            construction=ElbowConstruction.SMOOTH_RADIUS,
            centerline_radius=0.3,
        ),
        FlatObstructionGeometry(
            face_section=CircularFlowSection(diameter=0.16),
            blockage_ratio=0.2,
        ),
    ))
    outlet = PressureDropAssemblyGeometry(stages=())
    path = SpecifiedTubeSidePressureDropPath(inlet=inlet, returns=(), outlet=outlet)

    result = build_tube_side_pressure_drop_result(
        _tube_bundle_result(bundle), n_tube_passes=bundle.n_passes_tube, path=path,
    )
    inlet_group = result.group("inlet")
    assert inlet_group is not None
    assert len(inlet_group.stages) == 8
    for stage in inlet_group.stages:
        assert stage.status == PressureDropStageStatus.NOT_IMPLEMENTED
        assert stage.dp_irreversible == 0.0
        assert stage.delta_dynamic_pressure == 0.0
        assert stage.dp_static == 0.0
        assert stage.method is None
        assert len(stage.warnings) == 1
    # Placeholder stages must not leak non-zero pressure drop into dp_local.
    assert result.dp_local == 0.0
    assert result.dp_total == result.dp_core


def test_user_defined_fixed_dp_is_calculated_not_placeholder() -> None:
    bundle = _bundle(n_passes_tube=1)
    inlet = PressureDropAssemblyGeometry(stages=(
        UserDefinedPressureDropGeometry(pressure_drop=250.0, description="measured inlet loss"),
    ))
    outlet = PressureDropAssemblyGeometry(stages=(
        UserDefinedPressureDropGeometry(loss_coefficient=0.7, description="K only, not convertible yet"),
    ))
    path = SpecifiedTubeSidePressureDropPath(inlet=inlet, returns=(), outlet=outlet)

    result = build_tube_side_pressure_drop_result(
        _tube_bundle_result(bundle), n_tube_passes=bundle.n_passes_tube, path=path,
    )
    inlet_stage = result.group("inlet").stages[0]
    assert inlet_stage.status == PressureDropStageStatus.USER_DEFINED
    assert inlet_stage.dp_irreversible == 250.0
    assert inlet_stage.delta_dynamic_pressure == 0.0
    assert inlet_stage.dp_static == 250.0

    outlet_stage = result.group("outlet").stages[0]
    assert outlet_stage.status == PressureDropStageStatus.NOT_IMPLEMENTED
    assert outlet_stage.dp_irreversible == 0.0
    assert outlet_stage.delta_dynamic_pressure == 0.0
    assert outlet_stage.dp_static == 0.0

    assert result.dp_local == 250.0
    assert result.delta_dynamic_pressure_local == 0.0
    assert result.dp_static_local == 250.0
    assert result.dp_total == result.dp_core + 250.0


def test_geometry_only_user_defined_stage_rejects_ambiguous_modes() -> None:
    bundle = _bundle(n_passes_tube=1)
    ambiguous = PressureDropAssemblyGeometry(stages=(
        UserDefinedPressureDropGeometry(
            pressure_drop=25.0,
            loss_coefficient=0.5,
            reference_area=0.02,
        ),
    ))
    path = SpecifiedTubeSidePressureDropPath(
        inlet=ambiguous,
        returns=(),
        outlet=PressureDropAssemblyGeometry(stages=()),
    )

    with pytest.raises(ValueError, match="user_defined_pressure_drop_ambiguous"):
        build_tube_side_pressure_drop_result(
            _tube_bundle_result(bundle),
            n_tube_passes=bundle.n_passes_tube,
            path=path,
        )


# ---------------------------------------------------------------------------
# 18.5 Return-count validation
# ---------------------------------------------------------------------------

def test_return_count_matches_n_passes_minus_one() -> None:
    empty_assembly = PressureDropAssemblyGeometry(stages=())
    path = SpecifiedTubeSidePressureDropPath(
        inlet=empty_assembly, returns=(empty_assembly, empty_assembly), outlet=empty_assembly,
    )
    validate_specified_tube_side_path(path, n_tube_passes=3)  # 2 returns == 3 - 1; no raise


def test_return_count_mismatch_raises() -> None:
    empty_assembly = PressureDropAssemblyGeometry(stages=())
    path = SpecifiedTubeSidePressureDropPath(
        inlet=empty_assembly, returns=(empty_assembly,), outlet=empty_assembly,
    )
    with pytest.raises(ValueError):
        validate_specified_tube_side_path(path, n_tube_passes=3)

    bundle = _bundle(n_passes_tube=3)
    with pytest.raises(ValueError):
        build_tube_side_pressure_drop_result(
            _tube_bundle_result(bundle), n_tube_passes=bundle.n_passes_tube, path=path,
        )


# ---------------------------------------------------------------------------
# 18.6 Missing-path compatibility
# ---------------------------------------------------------------------------

def test_missing_tube_side_path_still_succeeds_with_v055_values() -> None:
    bundle = _bundle(n_passes_tube=2)
    tube_bundle_result = _tube_bundle_result(bundle)
    result = build_tube_side_pressure_drop_result(
        tube_bundle_result, n_tube_passes=bundle.n_passes_tube,
    )
    expected_irreversible = (
        tube_bundle_result.dp_straight_tube_friction
        + tube_bundle_result.dp_tube_entrances
        + tube_bundle_result.dp_tube_exits
    )
    assert result.dp_core == pytest.approx(expected_irreversible)
    assert result.dp_local == 0.0
    assert result.delta_dynamic_pressure_core == pytest.approx(
        tube_bundle_result.dp_straight_tube_acceleration
    )
    assert result.delta_dynamic_pressure_local == 0.0
    assert result.dp_total == pytest.approx(expected_irreversible)
    assert result.dp_static_total == pytest.approx(tube_bundle_result.dp_tube_bundle)


def test_missing_outside_path_still_succeeds_with_v055_values() -> None:
    bundle = _bundle(n_passes_tube=1)
    tube_bank_result = _tube_bank_result(bundle)
    result = build_outside_pressure_drop_result(tube_bank_result)
    assert result.dp_core == pytest.approx(tube_bank_result.dp_drag)
    assert result.dp_local == 0.0
    assert result.delta_dynamic_pressure_core == pytest.approx(
        tube_bank_result.dp_acceleration
    )
    assert result.delta_dynamic_pressure_local == 0.0
    assert result.dp_total == pytest.approx(tube_bank_result.dp_drag)
    assert result.dp_static_total == pytest.approx(tube_bank_result.dp_total)


def test_outside_not_specified_produces_nan_core_not_a_crash() -> None:
    result = build_outside_pressure_drop_result(None)
    assert result.groups == ()
    assert math.isnan(result.dp_core)
    assert result.dp_local == 0.0
    assert math.isnan(result.delta_dynamic_pressure_core)
    assert result.delta_dynamic_pressure_local == 0.0
    assert math.isnan(result.dp_total)
    assert math.isnan(result.delta_dynamic_pressure_total)
    assert math.isnan(result.dp_static_core)
    assert result.dp_static_local == 0.0
    assert math.isnan(result.dp_static_total)


# ---------------------------------------------------------------------------
# 18.7 Suggested-geometry mode
# ---------------------------------------------------------------------------

def test_tube_side_design_request_raises_not_implemented() -> None:
    bundle = _bundle(n_passes_tube=1)
    with pytest.raises(NotImplementedError):
        build_tube_side_pressure_drop_result(
            _tube_bundle_result(bundle),
            n_tube_passes=bundle.n_passes_tube,
            path=TubeSidePressureDropDesignRequest(maximum_pressure_drop=500.0),
        )


def test_outside_design_request_raises_not_implemented() -> None:
    bundle = _bundle(n_passes_tube=1)
    with pytest.raises(NotImplementedError):
        build_outside_pressure_drop_result(
            _tube_bank_result(bundle),
            path=OutsidePressureDropDesignRequest(minimum_velocity=1.0),
        )


def test_wrong_path_type_raises_type_error() -> None:
    bundle = _bundle(n_passes_tube=1)
    with pytest.raises(TypeError):
        build_tube_side_pressure_drop_result(
            _tube_bundle_result(bundle),
            n_tube_passes=bundle.n_passes_tube,
            path=SpecifiedOutsidePressureDropPath(
                inlet=PressureDropAssemblyGeometry(stages=()),
                outlet=PressureDropAssemblyGeometry(stages=()),
            ),
        )
