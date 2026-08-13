# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for periodic full circular-finned-tube geometry."""

from __future__ import annotations

import inspect
import math

import pytest

from core.geometry import (
    BareTube,
    CircularFinnedTube,
    TubeBundle,
    TubeOrientation,
    TubeSurfaceType,
)


def _core(**changes) -> BareTube:
    values = dict(
        D_i=0.020,
        D_o=0.025,
        length_total=2.0,
        length_effective=1.8,
        wall_k=45.0,
        roughness_inner=2.0e-5,
        roughness_outer=3.0e-5,
        tube_orientation=TubeOrientation.HORIZONTAL,
    )
    values.update(changes)
    return BareTube(**values)


def _welded(**changes) -> CircularFinnedTube:
    values = dict(
        core_tube=_core(),
        fin_k=200.0,
        D_fin=0.050,
        D_root=0.025,
        fin_thickness_root=0.001,
        fin_thickness_tip=0.001,
        fin_pitch=0.004,
        fin_contact_resistance=0.0,
    )
    values.update(changes)
    return CircularFinnedTube(**values)


def _extruded(**changes) -> CircularFinnedTube:
    values = dict(
        core_tube=_core(),
        fin_k=180.0,
        D_fin=0.052,
        D_root=0.029,
        fin_thickness_root=0.0012,
        fin_thickness_tip=0.00055,
        fin_pitch=0.0045,
        fin_contact_resistance=None,
    )
    values.update(changes)
    return CircularFinnedTube(**values)


def _bundle(
    tube, *, layout="staggered", n_rows=4, S_T=0.080, S_L=0.035
) -> TubeBundle:
    return TubeBundle(
        tube=tube,
        n_rows=n_rows,
        n_tubes_per_row=6,
        pitch_transverse=S_T,
        pitch_longitudinal=S_L,
        layout=layout,
        n_passes_tube=2,
        flow_arrangement="crossflow",
    )


def test_bare_tube_public_constructor_and_surface_type_are_backward_compatible() -> None:
    signature = inspect.signature(BareTube)
    assert list(signature.parameters) == [
        "D_i",
        "D_o",
        "length_total",
        "length_effective",
        "wall_k",
        "roughness_inner",
        "roughness_outer",
        "tube_orientation",
    ]
    tube = BareTube(0.020, 0.025, 2.0, 1.8, 45.0)
    assert tube.surface_type is TubeSurfaceType.PLAIN
    assert tube.area_outer == math.pi * 0.025 * 1.8


@pytest.mark.parametrize("name", ["D_i", "D_o", "length_total", "length_effective", "wall_k"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_bare_tube_shared_fields_reject_nonfinite(name: str, bad: float) -> None:
    values = dict(D_i=0.020, D_o=0.025, length_total=2.0, length_effective=1.8, wall_k=45.0)
    values[name] = bad
    with pytest.raises(ValueError, match="finite"):
        BareTube(**values)


def test_composition_delegates_complete_core_contract() -> None:
    core = _core()
    tube = _welded(core_tube=core)

    assert tube.surface_type is TubeSurfaceType.CIRCULAR_FINNED
    assert tube.core_tube is core
    for name in (
        "D_i",
        "D_o",
        "length_total",
        "length_effective",
        "wall_k",
        "roughness_inner",
        "roughness_outer",
        "tube_orientation",
        "relative_roughness_inner",
        "relative_roughness_outer",
        "flow_area",
        "hydraulic_diameter",
        "area_inner",
    ):
        assert getattr(tube, name) == getattr(core, name)


def test_welded_constant_thickness_periodic_areas_and_volume() -> None:
    tube = _welded()
    ri = tube.D_root / 2.0
    ro = tube.D_fin / 2.0
    t = tube.fin_thickness_root
    pitch = tube.fin_pitch

    primary_per_length = math.pi * tube.D_root * (pitch - t) / pitch
    both_sides_per_fin = 2.0 * math.pi * (ro * ro - ri * ri)
    tip_per_fin = math.pi * tube.D_fin * t
    fin_per_length = (both_sides_per_fin + tip_per_fin) / pitch
    volume_per_length = math.pi * (ro * ro - ri * ri) * t / pitch

    assert tube.fin_height == pytest.approx(0.0125)
    assert tube.root_radial_thickness == 0.0
    assert tube.fin_density == 250.0
    assert tube.effective_fin_count == 450.0
    assert tube.clear_spacing_root == pytest.approx(0.003)
    assert tube.primary_outside_area_per_length == pytest.approx(primary_per_length)
    assert tube.fin_both_sides_area_per_fin == pytest.approx(both_sides_per_fin)
    assert tube.fin_tip_area_per_fin == pytest.approx(tip_per_fin)
    assert tube.fin_area_geometric_per_length == pytest.approx(fin_per_length)
    assert tube.outside_area_geometric_per_length == pytest.approx(
        primary_per_length + fin_per_length
    )
    assert tube.fin_volume_per_length == pytest.approx(volume_per_length)
    assert tube.area_outer == pytest.approx(
        tube.outside_area_geometric_per_length * tube.length_effective
    )
    assert tube.outside_area_ratio_to_bare == pytest.approx(
        tube.area_outer / (math.pi * tube.D_o * tube.length_effective)
    )


def test_fractional_fin_count_and_periodic_area_are_continuous() -> None:
    tube = _welded(
        core_tube=_core(length_total=1.237, length_effective=1.237),
        fin_pitch=0.004,
    )
    assert tube.effective_fin_count == pytest.approx(309.25)
    assert tube.effective_fin_count != round(tube.effective_fin_count)
    assert tube.area_outer == pytest.approx(
        tube.outside_area_used_per_length * tube.length_effective
    )


def test_extruded_taper_uses_sloping_faces_tip_thickness_and_exact_volume() -> None:
    tube = _extruded()
    ri = tube.D_root / 2.0
    ro = tube.D_fin / 2.0
    height = ro - ri
    t_root = tube.fin_thickness_root
    t_tip = tube.fin_thickness_tip_effective
    slope = math.sqrt(1.0 + ((t_tip - t_root) / (2.0 * height)) ** 2)
    both_sides = 2.0 * math.pi * (ro * ro - ri * ri) * slope
    tip = 2.0 * math.pi * ro * t_tip
    exact_volume = 2.0 * math.pi * (
        ri * height * (t_root + t_tip) / 2.0
        + height * height * (t_root + 2.0 * t_tip) / 6.0
    )

    assert tube.root_radial_thickness == pytest.approx(0.002)
    assert tube.fin_side_slope_factor == pytest.approx(slope)
    assert tube.fin_both_sides_area_per_fin == pytest.approx(both_sides)
    assert tube.fin_tip_area_per_fin == pytest.approx(tip)
    assert tube.fin_volume_per_fin == pytest.approx(exact_volume)
    assert tube.fin_thickness_tip == t_tip


def test_none_tip_thickness_means_constant_thickness() -> None:
    explicit = _welded(fin_thickness_tip=0.001)
    implicit = _welded(fin_thickness_tip=None)
    assert implicit.fin_thickness_tip_effective == implicit.fin_thickness_root
    assert implicit.fin_area_per_fin == explicit.fin_area_per_fin
    assert implicit.fin_volume_per_fin == explicit.fin_volume_per_fin


def test_external_area_override_is_authoritative_but_does_not_change_flow_geometry() -> None:
    geometric = _welded()
    override_value = 1.12 * geometric.outside_area_geometric_per_length
    overridden = _welded(external_area_per_length=override_value)

    assert overridden.area_outer == pytest.approx(
        override_value * overridden.length_effective
    )
    assert overridden.area_outside_geometric == geometric.area_outside_geometric
    assert overridden.projected_blocking_area_per_length == geometric.projected_blocking_area_per_length
    assert overridden.external_area_relative_difference == pytest.approx(0.12)
    assert [warning.code for warning in overridden.geometry_warnings] == [
        "circular_finned_tube_external_area_override_difference"
    ]

    within_threshold = _welded(
        external_area_per_length=1.05 * geometric.outside_area_geometric_per_length
    )
    assert not any(
        warning.code == "circular_finned_tube_external_area_override_difference"
        for warning in within_threshold.geometry_warnings
    )


def test_unspecified_contact_warns_and_uses_documented_ideal_contact_default() -> None:
    tube = _extruded()
    assert tube.contact_resistance_used == 0.0
    assert tube.contact_thermal_resistance_per_tube == 0.0
    assert [warning.code for warning in tube.geometry_warnings] == [
        "circular_finned_tube_contact_resistance_unspecified"
    ]


def test_root_and_contact_resistance_helpers_scale_for_parallel_tubes() -> None:
    resistance_area = 2.0e-4
    tube = _extruded(fin_contact_resistance=resistance_area)
    bundle = _bundle(tube)

    expected_root_one = math.log(tube.D_root / tube.D_o) / (
        2.0 * math.pi * tube.fin_k * tube.length_effective
    )
    expected_contact_one = resistance_area / (
        math.pi * tube.D_o * tube.length_effective
    )
    assert tube.root_conduction_resistance_per_tube == pytest.approx(expected_root_one)
    assert tube.contact_thermal_resistance_per_tube == pytest.approx(expected_contact_one)
    assert bundle.root_conduction_resistance == pytest.approx(
        expected_root_one / bundle.n_tubes_total
    )
    assert bundle.fin_contact_thermal_resistance == pytest.approx(
        expected_contact_one / bundle.n_tubes_total
    )

    welded = _welded(fin_contact_resistance=resistance_area)
    welded_contact_area = (
        math.pi
        * welded.D_root
        * welded.fin_thickness_root
        * welded.effective_fin_count
    )
    assert welded.contact_area_basis == "periodic_fin_root_footprints"
    assert welded.contact_topology == "fin_branch_only"
    assert welded.contact_area_per_tube == pytest.approx(welded_contact_area)
    assert welded.contact_thermal_resistance_per_tube == pytest.approx(
        resistance_area / welded_contact_area
    )
    assert tube.contact_area_basis == "complete_core_root_layer_interface"
    assert tube.contact_topology == (
        "series_before_primary_and_fin_parallel_branches"
    )


def test_periodic_blockage_and_staggered_minimum_free_flow_area() -> None:
    tube = _welded()
    bundle = _bundle(tube, layout="staggered", S_T=0.080, S_L=0.035)
    expected_blockage = tube.D_root + (
        (tube.D_fin - tube.D_root)
        * (tube.fin_thickness_root + tube.fin_thickness_tip_effective)
        / (2.0 * tube.fin_pitch)
    )
    diagonal_pitch = math.sqrt(0.035**2 + (0.080 / 2.0) ** 2)
    transverse_gap = 0.080 - expected_blockage
    diagonal_gap = 2.0 * (diagonal_pitch - expected_blockage)
    expected_gap = min(transverse_gap, diagonal_gap)
    expected_area = 6 * tube.length_effective * expected_gap

    assert tube.projected_blocking_area_per_length == pytest.approx(expected_blockage)
    assert bundle.diagonal_pitch == pytest.approx(diagonal_pitch)
    assert bundle.minimum_free_flow_gap == pytest.approx(expected_gap)
    assert bundle.minimum_free_flow_area == pytest.approx(expected_area)
    assert bundle.minimum_free_flow_area < bundle.frontal_flow_area

    rho = 1.2
    m_dot = 2.5
    assert bundle.face_velocity(m_dot, rho) == pytest.approx(
        m_dot / (rho * bundle.frontal_flow_area)
    )
    assert bundle.reference_velocity(m_dot, rho) == pytest.approx(
        m_dot / (rho * bundle.minimum_free_flow_area)
    )
    assert bundle.reference_velocity(m_dot, rho) > bundle.face_velocity(m_dot, rho)


def test_inline_minimum_free_flow_area_uses_transverse_gap() -> None:
    tube = _welded()
    bundle = _bundle(tube, layout="inline", S_T=0.080, S_L=0.070)
    expected = (
        bundle.n_tubes_per_row
        * tube.length_effective
        * (bundle.pitch_transverse - tube.projected_blocking_area_per_length)
    )
    assert bundle.minimum_free_flow_area == pytest.approx(expected)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"D_root": 0.024}, "D_root"),
        ({"D_root": 0.050}, "D_fin"),
        ({"D_fin": math.nan}, "D_fin"),
        ({"fin_k": math.inf}, "fin_k"),
        ({"fin_thickness_root": 0.0}, "fin_thickness_root"),
        ({"fin_pitch": 0.001}, "fin_pitch"),
        ({"fin_thickness_tip": 0.0}, "fin_thickness_tip"),
        ({"fin_thickness_tip": 0.0011}, "fin_thickness_tip"),
        ({"fin_contact_resistance": -1.0e-5}, "fin_contact_resistance"),
        ({"fin_contact_resistance": math.nan}, "fin_contact_resistance"),
        ({"external_area_per_length": 0.001}, "external_area_per_length"),
    ],
)
def test_invalid_fin_geometry_is_rejected(changes, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _welded(**changes)


def test_non_bare_core_is_rejected() -> None:
    with pytest.raises(TypeError, match="core_tube"):
        _welded(core_tube=object())


def test_bundle_rejects_neighboring_fin_overlap() -> None:
    tube = _welded()
    with pytest.raises(ValueError, match="pitch_transverse"):
        _bundle(tube, S_T=tube.D_fin, S_L=0.060)
    with pytest.raises(ValueError, match="pitch_longitudinal"):
        _bundle(tube, layout="inline", S_T=0.060, S_L=tube.D_fin)
    with pytest.raises(ValueError, match="Diagonal"):
        _bundle(tube, layout="staggered", S_T=0.060, S_L=0.030)
    # Rows one and three are aligned. Their separation is 2*S_L, which can
    # control even when the nearest diagonal centre distance clears D_fin.
    with pytest.raises(ValueError, match="Twice pitch_longitudinal"):
        _bundle(tube, layout="staggered", S_T=0.100, S_L=0.020)
    # A two-row bank has no aligned second-neighbour row.
    two_rows = _bundle(
        tube, layout="staggered", n_rows=2, S_T=0.100, S_L=0.020
    )
    assert two_rows.n_rows == 2
