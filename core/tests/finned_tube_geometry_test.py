# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Geometry tests for CircularFinnedTube (v0.7.x)."""

from __future__ import annotations

import math

import pytest

from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.tube import BareTube, TubeSurfaceType


def _core_tube(D_i: float = 0.020, D_o: float = 0.025, length_effective: float = 3.0) -> BareTube:
    return BareTube(D_i=D_i, D_o=D_o, length_total=length_effective, length_effective=length_effective, wall_k=45.0)


def _welded_tube(**overrides) -> CircularFinnedTube:
    kwargs = dict(
        core_tube=_core_tube(),
        fin_k=200.0,
        D_fin=0.057,
        D_root=0.025,
        fin_thickness_root=0.0004,
        fin_pitch=0.0023,
    )
    kwargs.update(overrides)
    return CircularFinnedTube(**kwargs)


def _extruded_tube(**overrides) -> CircularFinnedTube:
    kwargs = dict(
        core_tube=_core_tube(),
        fin_k=200.0,
        D_fin=0.057,
        D_root=0.0265,
        fin_thickness_root=0.0006,
        fin_thickness_tip=0.0002,
        fin_pitch=0.0023,
    )
    kwargs.update(overrides)
    return CircularFinnedTube(**kwargs)


# -------------------------------------------------------------------
# Identity
# -------------------------------------------------------------------

def test_bare_tube_identifies_as_plain():
    assert _core_tube().surface_type == TubeSurfaceType.PLAIN


def test_circular_finned_tube_identifies_as_circular_finned():
    assert _welded_tube().surface_type == TubeSurfaceType.CIRCULAR_FINNED


# -------------------------------------------------------------------
# Welded (constant thickness, D_root == D_o) case
# -------------------------------------------------------------------

def test_welded_constant_thickness_fin_root_matches_core_tube_od():
    tube = _welded_tube()
    assert tube.D_root == tube.D_o
    assert tube.root_radial_thickness == 0.0
    assert tube.fin_thickness_tip_used == tube.fin_thickness_root


def test_welded_tube_delegates_core_tube_fields():
    core = _core_tube()
    tube = _welded_tube(core_tube=core)
    assert tube.D_i == core.D_i
    assert tube.D_o == core.D_o
    assert tube.length_total == core.length_total
    assert tube.length_effective == core.length_effective
    assert tube.wall_k == core.wall_k
    assert tube.area_inner == core.area_inner
    assert tube.flow_area == core.flow_area
    assert tube.hydraulic_diameter == core.hydraulic_diameter


# -------------------------------------------------------------------
# Extruded (tapered, D_root > D_o) case
# -------------------------------------------------------------------

def test_extruded_tapered_fin_has_positive_root_layer_and_taper():
    tube = _extruded_tube()
    assert tube.D_root > tube.D_o
    assert tube.root_radial_thickness == pytest.approx((tube.D_root - tube.D_o) / 2.0)
    assert tube.fin_thickness_tip_used < tube.fin_thickness_root


def test_constant_thickness_fin_side_area_matches_classical_flat_annulus():
    tube = _welded_tube()
    r_root = tube.D_root / 2.0
    r_tip = tube.D_fin / 2.0
    classical = 2.0 * math.pi * (r_tip * r_tip - r_root * r_root)
    assert tube.A_fin_side_per_fin == pytest.approx(classical, rel=1e-12)


def test_tapered_fin_slant_length_exceeds_radial_height():
    tube = _extruded_tube()
    assert tube.fin_slant_length > tube.fin_height


# -------------------------------------------------------------------
# Areas per unit length / totals
# -------------------------------------------------------------------

def test_effective_fin_count_is_not_rounded():
    tube = _welded_tube()
    # length_effective / fin_pitch is very unlikely to be an integer
    assert tube.effective_fin_count != round(tube.effective_fin_count)
    assert tube.nominal_fin_count_diagnostic == round(tube.effective_fin_count)


def test_effective_fin_count_varies_smoothly_with_length_no_discontinuity():
    base = _welded_tube()
    lengths = [3.0, 3.001, 3.01, 3.1]
    areas = []
    for L in lengths:
        core = _core_tube(length_effective=L)
        tube = _welded_tube(core_tube=core)
        areas.append(tube.A_outside_geometric)
    # small length changes must produce small, monotonically increasing area changes
    for a, b in zip(areas, areas[1:]):
        assert b > a
    diffs = [b - a for a, b in zip(areas, areas[1:])]
    # ratio of consecutive derivative-like differences stays bounded (no jump)
    per_length = [d / (l2 - l1) for d, l1, l2 in zip(diffs, lengths, lengths[1:])]
    assert max(per_length) / min(per_length) < 1.01


def test_primary_area_uses_clear_spacing_not_full_pitch():
    tube = _welded_tube()
    per_length = tube.A_primary / tube.length_effective
    expected = math.pi * tube.D_root * (tube.clear_spacing_root / tube.fin_pitch)
    assert per_length == pytest.approx(expected, rel=1e-12)


def test_area_outer_equals_gross_area_by_default():
    tube = _welded_tube()
    assert tube.area_outer == pytest.approx(tube.A_outside_gross)
    assert tube.A_outside_gross == pytest.approx(tube.A_outside_geometric)


def test_total_outside_area_exceeds_bare_tube_area_by_expected_ratio():
    tube = _welded_tube()
    bare_area = math.pi * tube.D_o * tube.length_effective
    assert tube.area_outer_to_bare_ratio == pytest.approx(tube.A_outside_gross / bare_area)
    assert tube.area_outer_to_bare_ratio > 5.0  # sanity: finned area is much larger than bare


def test_fin_volume_matches_numerical_quadrature_for_tapered_profile():
    tube = _extruded_tube()
    r_root = tube.D_root / 2.0
    r_tip = tube.D_fin / 2.0
    t_root = tube.fin_thickness_root
    t_tip = tube.fin_thickness_tip_used

    n = 200000
    dr = (r_tip - r_root) / n
    total = 0.0
    for i in range(n):
        r = r_root + (i + 0.5) * dr
        frac = (r - r_root) / (r_tip - r_root)
        t = t_root + (t_tip - t_root) * frac
        total += 2.0 * math.pi * r * t * dr
    assert tube.fin_volume_per_fin == pytest.approx(total, rel=1e-4)


def test_fin_volume_constant_thickness_matches_closed_form_cylinder_shell():
    tube = _welded_tube()
    r_root = tube.D_root / 2.0
    r_tip = tube.D_fin / 2.0
    expected = math.pi * tube.fin_thickness_root * (r_tip * r_tip - r_root * r_root)
    assert tube.fin_volume_per_fin == pytest.approx(expected, rel=1e-12)


# -------------------------------------------------------------------
# external_area_per_length override
# -------------------------------------------------------------------

def test_external_area_override_is_used_and_geometric_still_available():
    tube = _welded_tube()
    per_length_override = tube.A_outside_geometric / tube.length_effective * 1.05
    overridden = _welded_tube(external_area_per_length=per_length_override)
    assert overridden.A_outside_gross == pytest.approx(per_length_override * tube.length_effective)
    assert overridden.A_outside_geometric == pytest.approx(tube.A_outside_geometric)
    assert overridden.external_area_relative_difference == pytest.approx(0.05, rel=1e-9)
    assert overridden.external_area_override_exceeds_threshold is False


def test_external_area_override_large_difference_flagged():
    tube = _welded_tube()
    per_length_override = tube.A_outside_geometric / tube.length_effective * 1.5
    overridden = _welded_tube(external_area_per_length=per_length_override)
    assert overridden.external_area_override_exceeds_threshold is True


def test_external_area_override_below_primary_area_rejected():
    tube = _welded_tube()
    tiny = tube._A_primary_per_length() * 0.5
    with pytest.raises(ValueError):
        _welded_tube(external_area_per_length=tiny)


def test_external_area_override_must_be_positive():
    with pytest.raises(ValueError):
        _welded_tube(external_area_per_length=-1.0)


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

def test_rejects_core_tube_not_a_bare_tube():
    with pytest.raises(TypeError):
        CircularFinnedTube(
            core_tube=object(),
            fin_k=200.0,
            D_fin=0.057,
            D_root=0.025,
            fin_thickness_root=0.0004,
            fin_pitch=0.0023,
        )


def test_rejects_d_root_below_core_od():
    with pytest.raises(ValueError):
        _welded_tube(D_root=0.020)


def test_rejects_d_root_equal_to_d_fin():
    with pytest.raises(ValueError):
        _welded_tube(D_root=0.057, D_fin=0.057)


def test_rejects_d_root_zero():
    with pytest.raises(ValueError):
        _welded_tube(D_root=0.0)


def test_rejects_fin_pitch_not_greater_than_thickness():
    with pytest.raises(ValueError):
        _welded_tube(fin_pitch=0.0004, fin_thickness_root=0.0004)


def test_rejects_negative_or_zero_fin_thickness_root():
    with pytest.raises(ValueError):
        _welded_tube(fin_thickness_root=0.0)


def test_rejects_fin_thickness_tip_exceeding_root():
    with pytest.raises(ValueError):
        _welded_tube(fin_thickness_root=0.0004, fin_thickness_tip=0.0008)


def test_rejects_fin_thickness_tip_zero_or_negative():
    with pytest.raises(ValueError):
        _welded_tube(fin_thickness_tip=0.0)


def test_rejects_nonpositive_fin_k():
    with pytest.raises(ValueError):
        _welded_tube(fin_k=0.0)


def test_rejects_negative_contact_resistance():
    with pytest.raises(ValueError):
        _welded_tube(fin_contact_resistance=-1.0)


def test_accepts_explicit_zero_contact_resistance_as_ideal():
    tube = _welded_tube(fin_contact_resistance=0.0)
    assert tube.fin_contact_resistance == 0.0


def test_contact_resistance_none_is_distinct_from_zero():
    tube = _welded_tube()
    assert tube.fin_contact_resistance is None
