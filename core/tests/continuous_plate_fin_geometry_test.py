# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace
import math
import pytest

from core.geometry import BareTube, TubeBundle, ContinuousPlateFinGeometry, TubeBankLattice


def bank(rows=3, count=2.5, **fin_changes):
    fin = ContinuousPlateFinGeometry(10, .0001, .002, .0001,
        .075, rows*.02, -.0125, -.01, .0102, .0018, 205.)
    return TubeBundle(BareTube(.008, .01, .02, .02, 45.), rows, count,
        .025, .02, "staggered", 1, continuous_fin_geometry=replace(fin, **fin_changes))


def test_surface_ledger_counts_each_sheet_and_every_aperture_once():
    b = bank()
    s, f = b.continuous_fin_surface, b.continuous_fin_geometry
    assert b.n_tubes_total == 8
    assert s.aperture_count == 80
    assert s.fin_face_area == pytest.approx(20*(.075*.06-8*math.pi*.0102**2/4))
    assert s.fin_edge_area == pytest.approx(20*.0001*(.075+.06))
    assert s.exposed_tube_area == pytest.approx(8*math.pi*.01*(.02-10*.0018))
    assert s.exposed_collar_area == pytest.approx(8*math.pi*.0102*10*.0017)
    assert s.collar_lip_area == pytest.approx(8*20*math.pi*(.0102**2-.01**2)/4)
    assert s.contact_area == pytest.approx(80*math.pi*.01*.0018)
    assert b.total_outer_area == pytest.approx(b.total_primary_outside_area+b.total_fin_area)
    assert b.total_outer_geometric_area == b.total_outer_area
    assert f.sheet_positions == pytest.approx(tuple(.0001+i*.002 for i in range(10)))
    assert f.pack_extent == pytest.approx((.0001, .0199))


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_manufactured_count_is_integer(value):
    with pytest.raises(ValueError, match="integer"):
        bank(sheet_count=value)


@pytest.mark.parametrize("changes", [dict(thickness=.002), dict(first_sheet_position=.004),
    dict(height=.04), dict(collar_outer_diameter=.03), dict(collar_height=.003)])
def test_invalid_pack_and_apertures(changes):
    with pytest.raises(ValueError):
        bank(**changes)


def test_finite_single_row_flow_area_independent_hand_calculation():
    b = bank(rows=1, count=3., first_sheet_position=0., collar_height=.002)
    s = b.continuous_fin_surface
    # Sealed touching collars; nine interior axial gaps plus final clear gap.
    assert s.frontal_area == pytest.approx(.075*.02)
    assert s.minimum_free_flow_area == pytest.approx((.02-10*.0001)*(.075-3*.0102))
    assert s.free_flow_ratio == pytest.approx(s.minimum_free_flow_area/s.frontal_area)
    assert s.compact_hydraulic_diameter == pytest.approx(4*s.minimum_free_flow_area*.02/s.source_face_cylinder_area)
    assert s.collar_lip_area == 0.


def test_diagonal_passage_and_actual_edge_margins():
    from core.geometry.continuous_plate_fin import _finite_throat_width
    # Two finite side passages plus two diagonal gaps, no periodic wall wrap.
    centers = ((0., 0.), (.01, .015), (0., .03))
    expected = .01+.01+2*(math.hypot(.01, .015)-.012)
    assert _finite_throat_width(centers, -.016, .046, .012) == pytest.approx(expected)
    assert _finite_throat_width(centers, -.026, .056, .012) == pytest.approx(expected+.02)


def test_layout_identity_and_legacy_gap_are_not_substituted():
    b = bank()
    assert len(b.tube_center_coordinates[1]) == 2
    inline = replace(b, lattice=TubeBankLattice("inline"), n_tubes_per_row=3.)
    assert inline.continuous_fin_surface.aperture_count == 90
    with pytest.raises(ValueError, match="periodic"):
        _ = b.minimum_free_flow_gap
    assert not hasattr(b.continuous_fin_geometry, "D_fin")


def test_plain_bundle_remains_independent():
    b = bank()
    plain = replace(b, continuous_fin_geometry=None)
    assert plain.total_outer_area == plain.n_tubes_total*plain.tube.area_outer
    assert plain.frontal_flow_area != b.frontal_flow_area or plain.total_outer_area != b.total_outer_area
