# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Free-flow-area / V_max geometry tests for circular finned-tube banks (v0.7.x)."""

from __future__ import annotations

import pytest

from core.geometry.finned_flow_geometry import (
    FinnedTubeGeometryOverlapError,
    finned_blocked_equivalent_diameter,
    finned_vmax_ratio_min_freeflow,
)
from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.tube import BareTube
from core.heat_transfer.outside_flow import vmax_ratio_min_freeflow


def _tube(**overrides) -> CircularFinnedTube:
    core = BareTube(D_i=0.020, D_o=0.025, length_total=3.0, length_effective=3.0, wall_k=45.0)
    kwargs = dict(
        core_tube=core,
        fin_k=200.0,
        D_fin=0.057,
        D_root=0.025,
        fin_thickness_root=0.0004,
        fin_pitch=0.0023,
    )
    kwargs.update(overrides)
    return CircularFinnedTube(**kwargs)


def test_blocked_diameter_exceeds_root_diameter():
    tube = _tube()
    D_blocked = finned_blocked_equivalent_diameter(tube)
    assert D_blocked > tube.D_root


def test_blocked_diameter_zero_fin_height_limit_matches_root():
    # A vanishingly small fin height should make the blockage contribution negligible.
    tube = _tube(D_fin=0.0251, fin_thickness_root=0.00005, fin_pitch=0.0023)
    D_blocked = finned_blocked_equivalent_diameter(tube)
    assert D_blocked == pytest.approx(tube.D_root, rel=1e-2)


def test_finned_vmax_ratio_exceeds_bare_tube_ratio_for_same_root_diameter():
    tube = _tube()
    S_T, S_L = 0.065, 0.055
    finned_ratio = finned_vmax_ratio_min_freeflow(tube, S_T, S_L, "staggered")
    bare_ratio = vmax_ratio_min_freeflow(tube.D_root, S_T, S_L, "staggered")
    # Fin blockage must make V_max larger relative to face velocity than
    # the bare-root-diameter-only calculation would suggest.
    assert finned_ratio > bare_ratio


def test_finned_vmax_ratio_inline_layout_supported_as_geometry():
    tube = _tube()
    ratio = finned_vmax_ratio_min_freeflow(tube, 0.065, 0.055, "inline")
    assert ratio > 1.0


def test_overlapping_fins_in_same_row_rejected():
    tube = _tube()
    with pytest.raises(FinnedTubeGeometryOverlapError):
        finned_vmax_ratio_min_freeflow(tube, tube.D_fin * 0.9, 0.055, "staggered")


def test_taller_fin_increases_vmax_ratio():
    short_fin = _tube(D_fin=0.040)
    tall_fin = _tube(D_fin=0.057)
    S_T, S_L = 0.070, 0.060
    ratio_short = finned_vmax_ratio_min_freeflow(short_fin, S_T, S_L, "staggered")
    ratio_tall = finned_vmax_ratio_min_freeflow(tall_fin, S_T, S_L, "staggered")
    assert ratio_tall > ratio_short


def test_rejects_nonpositive_pitches():
    tube = _tube()
    with pytest.raises(ValueError):
        finned_vmax_ratio_min_freeflow(tube, 0.0, 0.055, "staggered")
