# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Focused tests for v0.7.3 transverse tube-pass topology."""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube


def _tube() -> BareTube:
    return BareTube(
        D_i=0.020, D_o=0.024, length_total=4.0, length_effective=4.0, wall_k=16.0,
    )


def _motivating_bundle(**overrides) -> TubeBundle:
    tube = overrides.pop("tube", _tube())
    defaults = dict(
        tube=tube,
        n_rows=15,
        n_tubes_per_row=55,
        pitch_transverse=0.05,
        pitch_longitudinal=0.05,
        layout="staggered",
        n_passes_tube=18,
        n_passes_transverse=6,
        flow_arrangement="counterflow",
    )
    defaults.update(overrides)
    return TubeBundle(**defaults)


# ----------------------------------------------------------------------
# Motivating case (section 17)
# ----------------------------------------------------------------------


def test_motivating_case_topology() -> None:
    bundle = _motivating_bundle()
    assert bundle.n_tubes_total == 825
    assert bundle.n_passes_transverse_resolved == 6
    assert bundle.n_sections_longitudinal == 3
    assert bundle.n_rows_per_section == 5
    assert bundle.n_tubes_per_pass_effective == pytest.approx(825 / 18)
    assert bundle.n_tubes_per_pass == pytest.approx(825 / 18)
    assert bundle.pass_partition_is_exact is False


def test_motivating_case_flow_area_and_length() -> None:
    tube = _tube()
    bundle = _motivating_bundle(tube=tube)
    assert bundle.internal_flow_area_per_pass == pytest.approx(
        (825 / 18) * tube.flow_area
    )
    assert bundle.internal_length_total == pytest.approx(18 * tube.length_total)


# ----------------------------------------------------------------------
# Legacy backward-compatibility (section 18)
# ----------------------------------------------------------------------


def test_legacy_exact_partition_defaults_to_none() -> None:
    tube = _tube()
    bundle = TubeBundle(
        tube=tube, n_rows=6, n_tubes_per_row=8,
        pitch_transverse=0.04, pitch_longitudinal=0.04,
        layout="inline", n_passes_tube=2, flow_arrangement="counterflow",
    )
    assert bundle.n_passes_transverse is None
    assert bundle.n_tubes_total == 48
    assert bundle.n_tubes_per_pass_effective == pytest.approx(24.0)
    assert bundle.n_tubes_per_pass == pytest.approx(24.0)
    assert bundle.pass_partition_is_exact is True
    assert bundle.n_passes_transverse_resolved == 2
    assert bundle.n_sections_longitudinal == 1
    assert bundle.n_rows_per_section == 6


def test_none_and_explicit_equal_to_n_passes_tube_are_equivalent() -> None:
    tube = _tube()
    kwargs = dict(
        tube=tube, n_rows=6, n_tubes_per_row=8,
        pitch_transverse=0.04, pitch_longitudinal=0.04,
        layout="inline", n_passes_tube=2, flow_arrangement="counterflow",
    )
    implicit = TubeBundle(**kwargs)
    explicit = TubeBundle(**kwargs, n_passes_transverse=2)
    assert implicit.n_passes_transverse_resolved == explicit.n_passes_transverse_resolved
    assert implicit.n_sections_longitudinal == explicit.n_sections_longitudinal
    assert implicit.n_rows_per_section == explicit.n_rows_per_section
    assert implicit.n_tubes_per_pass_effective == explicit.n_tubes_per_pass_effective
    assert implicit.internal_flow_area_per_pass == explicit.internal_flow_area_per_pass
    assert implicit.internal_length_total == explicit.internal_length_total


# ----------------------------------------------------------------------
# Validation (section 19)
# ----------------------------------------------------------------------


def _bundle(**overrides) -> TubeBundle:
    tube = overrides.pop("tube", _tube())
    defaults = dict(
        tube=tube,
        n_rows=15,
        n_tubes_per_row=55,
        pitch_transverse=0.05,
        pitch_longitudinal=0.05,
        layout="staggered",
        n_passes_tube=18,
        flow_arrangement="counterflow",
    )
    defaults.update(overrides)
    return TubeBundle(**defaults)


def test_t1_zero_transverse_passes_rejected() -> None:
    with pytest.raises(ValueError):
        _bundle(n_passes_transverse=0)


def test_t2_negative_transverse_passes_rejected() -> None:
    with pytest.raises(ValueError):
        _bundle(n_passes_transverse=-1)


def test_t3_transverse_passes_exceeding_total_rejected() -> None:
    with pytest.raises(ValueError):
        _bundle(n_passes_transverse=19)


def test_t4_non_divisible_transverse_pass_count_rejected() -> None:
    with pytest.raises(ValueError):
        _bundle(n_passes_transverse=5)


def test_t5_rows_not_divisible_by_sections_rejected() -> None:
    with pytest.raises(ValueError):
        _bundle(n_rows=14, n_passes_transverse=6)


def test_t6_non_divisible_tube_count_accepted() -> None:
    bundle = _motivating_bundle()
    assert bundle.n_tubes_total == 825
    assert bundle.n_passes_tube == 18


def test_t7_exact_tube_count_accepted_and_flagged() -> None:
    bundle = _bundle(n_rows=6, n_tubes_per_row=8, n_passes_tube=2, n_passes_transverse=None)
    assert bundle.n_tubes_total == 48
    assert bundle.pass_partition_is_exact is True


def test_t8_no_rounding_of_effective_tubes_per_pass() -> None:
    bundle = _motivating_bundle()
    exact = 825 / 18
    assert bundle.n_tubes_per_pass_effective == exact
    assert bundle.n_tubes_per_pass_effective != 45
    assert bundle.n_tubes_per_pass_effective != 46
    assert not math.isclose(bundle.n_tubes_per_pass_effective, 45.0)
    assert not math.isclose(bundle.n_tubes_per_pass_effective, 46.0)


def test_t9_outside_geometry_unaffected_by_transverse_passes() -> None:
    tube = _tube()
    with_sections = _motivating_bundle(tube=tube)
    without_sections = _bundle(tube=tube, n_passes_transverse=None)
    assert with_sections.total_outer_area == without_sections.total_outer_area
    assert with_sections.frontal_flow_area == without_sections.frontal_flow_area
    assert with_sections.minimum_free_flow_area == without_sections.minimum_free_flow_area
    assert with_sections.n_rows == without_sections.n_rows == 15


def test_t10_existing_divisible_case_unchanged() -> None:
    tube = _tube()
    bundle = TubeBundle(
        tube=tube, n_rows=4, n_tubes_per_row=8,
        pitch_transverse=0.03, pitch_longitudinal=0.03,
        layout="inline", n_passes_tube=2, flow_arrangement="counterflow",
    )
    assert bundle.internal_flow_area_per_pass == pytest.approx(16 * tube.flow_area)
    assert bundle.internal_length_total == pytest.approx(2 * tube.length_total)


def test_t11_velocity_independently_verified() -> None:
    bundle = _motivating_bundle()
    m_dot = 12.5
    rho = 950.0
    n_tubes_per_pass_effective = 825 / 18
    tube_flow_area = math.pi * (0.020**2) / 4.0
    expected_velocity = m_dot / (rho * n_tubes_per_pass_effective * tube_flow_area)
    actual_velocity = m_dot / (rho * bundle.internal_flow_area_per_pass)
    assert actual_velocity == pytest.approx(expected_velocity)


def test_t12_internal_length_uses_total_pass_count() -> None:
    tube = _tube()
    bundle = _motivating_bundle(tube=tube)
    assert bundle.internal_length_total == pytest.approx(18 * tube.length_total)
    assert bundle.internal_length_total != pytest.approx(6 * tube.length_total)
    assert bundle.internal_length_total != pytest.approx(3 * tube.length_total)


def main() -> None:
    test_motivating_case_topology()
    test_motivating_case_flow_area_and_length()
    test_legacy_exact_partition_defaults_to_none()
    test_none_and_explicit_equal_to_n_passes_tube_are_equivalent()
    test_t1_zero_transverse_passes_rejected()
    test_t2_negative_transverse_passes_rejected()
    test_t3_transverse_passes_exceeding_total_rejected()
    test_t4_non_divisible_transverse_pass_count_rejected()
    test_t5_rows_not_divisible_by_sections_rejected()
    test_t6_non_divisible_tube_count_accepted()
    test_t7_exact_tube_count_accepted_and_flagged()
    test_t8_no_rounding_of_effective_tubes_per_pass()
    test_t9_outside_geometry_unaffected_by_transverse_passes()
    test_t10_existing_divisible_case_unchanged()
    test_t11_velocity_independently_verified()
    test_t12_internal_length_uses_total_pass_count()
    print("ALL TRANSVERSE-PASS TESTS PASSED")


if __name__ == "__main__":
    main()
