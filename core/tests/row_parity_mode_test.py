"""Physical row numbering and its downstream area/flow consequences."""
from dataclasses import replace

import pytest

from core.geometry import BareTube, TubeBundle, RowParityMode


def bundle(**changes):
    inputs = dict(tube=BareTube(D_i=.02, D_o=.024, length_total=3.,
                               length_effective=2.5, wall_k=20.),
                  n_rows=6, n_tubes_per_row=9.5, pitch_transverse=.05,
                  pitch_longitudinal=.04, layout="staggered",
                  n_passes_tube=4, n_passes_transverse=2)
    inputs.update(changes)
    return TubeBundle(**inputs)


def test_reset_default_and_continuous_downstream_counts():
    reset = bundle()
    continuous = replace(reset, row_parity_mode="continuous")
    assert reset.row_parity_mode is RowParityMode.RESET
    assert continuous.row_parity_mode is RowParityMode.CONTINUOUS
    assert reset.n_tubes_total == 58
    assert continuous.n_tubes_total == 57
    for b, count in ((reset, 58), (continuous, 57)):
        assert b.total_inner_area == pytest.approx(count*b.tube.area_inner)
        assert b.total_outer_area == pytest.approx(count*b.tube.area_outer)
        assert b.internal_flow_area_per_pass == pytest.approx(count/4*b.tube.flow_area)
        assert b.n_tubes_per_pass_effective == count/4
    assert reset.frontal_flow_area == continuous.frontal_flow_area
    assert reset.minimum_free_flow_area == continuous.minimum_free_flow_area


@pytest.mark.parametrize("changes", [dict(n_tubes_per_row=10),
    dict(n_passes_transverse=4), dict(n_rows=8), dict(n_rows=7)])
def test_modes_agree_when_parity_cannot_change_counts(changes):
    reset = bundle(**changes)
    assert reset.n_tubes_total == replace(reset, row_parity_mode="continuous").n_tubes_total


def test_continuous_counts_do_not_depend_on_section_boundaries():
    continuous = bundle(row_parity_mode="continuous")
    for transverse in (1, 2, 4):
        assert replace(continuous, n_passes_transverse=transverse).n_tubes_total == 57


@pytest.mark.parametrize("value", ["bad", None, 1])
def test_invalid_mode_is_rejected(value):
    with pytest.raises(ValueError, match="row_parity_mode"):
        bundle(row_parity_mode=value)
