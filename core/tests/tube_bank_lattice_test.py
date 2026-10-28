# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace

import pytest

from core.geometry import LayoutClass, TubeBankLattice, TubeGeometryApplicabilityError
from core.heat_transfer.finned_ellipse_airside import (
    Zhao2017AirsideModel, ZHAO2017_IMPLEMENTED_LATTICE,
)
from core.pressure_drop.finned_ellipse import calculate_finned_ellipse_hydraulics
from core.tests.zhao2017_fixtures import AIR, source_bundle


def test_legacy_coordinates_and_deterministic_parity():
    b = replace(source_bundle(), n_rows=3, n_tubes_per_row=2.5)
    assert b.tube_bank_lattice.layout_class == LayoutClass.CONVENTIONAL_STAGGERED
    assert b.tube_center_coordinates == (
        ((0., 0.), (0., .031), (0., .062)),
        ((.060, .0155), (.060, .0465)),
        ((.120, 0.), (.120, .031), (.120, .062)),
    )
    inline = replace(b, layout='inline', n_tubes_per_row=3)
    assert all(row[0][1] == 0 for row in inline.tube_center_coordinates)
    assert len({inline.tube_bank_lattice, b.tube_bank_lattice, TubeBankLattice('shifted')}) == 3


def test_whole_rows_have_one_shared_offset():
    b = source_bundle()
    for j, row in enumerate(b.tube_center_coordinates):
        assert [y-i*b.pitch_transverse for i, (_, y) in enumerate(row)] == pytest.approx([(j % 2)*b.pitch_transverse/2]*len(row))


def test_section_reset_matches_legacy_counts_and_coordinates():
    b = replace(source_bundle(), n_rows=6, n_tubes_per_row=2.5, n_passes_tube=2, n_passes_transverse=1)
    coords = b.tube_center_coordinates
    assert sum(map(len, coords)) == b.n_tubes_total == 16
    assert coords[3][0][1] == 0
    continuous = replace(b, row_parity_mode='continuous')
    assert continuous.tube_center_coordinates[3][0][1] == .0155
    assert sum(map(len, continuous.tube_center_coordinates)) == continuous.n_tubes_total == 15


def test_per_site_shift_is_explicit_and_not_a_whole_row_translation():
    lattice = TubeBankLattice('shifted', pattern='SYNTHETIC_PER_SITE', phase='j0_i0_unshifted',
        nominal_offset=.0155, site_offsets=(((0., 0.), (.0155, 0.)), ((0., 0.), (.0155, 0.))))
    b = replace(source_bundle(), n_rows=2, n_tubes_per_row=2, lattice=lattice)
    assert b.layout == 'shifted'  # stale legacy 'staggered' cannot override physical identity
    assert b.tube_center_coordinates[0] == ((0., 0.), (.0155, .031))
    with pytest.raises(TubeGeometryApplicabilityError, match='SHIFTED'):
        Zhao2017AirsideModel().evaluate(b, 1.7, AIR)


def test_unresolved_shift_never_invents_coordinates_or_physics():
    b = replace(source_bundle(), layout='shifted')
    assert b.tube_bank_lattice.pattern == 'UNRESOLVED'
    with pytest.raises(ValueError, match='UNRESOLVED'):
        b.tube_center_coordinates
    for strict in (False, True):
        with pytest.raises(TubeGeometryApplicabilityError, match='SHIFTED'):
            Zhao2017AirsideModel(strict_source_domain=strict).evaluate(b, 1.7, AIR)
    with pytest.raises(TubeGeometryApplicabilityError, match='SHIFTED'):
        calculate_finned_ellipse_hydraulics(bundle=b, m_dot=1.7, inlet_props=AIR)


def test_zhao_signature_axis_mapping_and_unchanged_regression():
    b = source_bundle()
    r = Zhao2017AirsideModel(strict_source_domain=True).evaluate(b, 1.7, AIR)
    assert r.implemented_lattice == ZHAO2017_IMPLEMENTED_LATTICE
    assert r.source_lattice_grade == 'SOURCE_EXACT:ZHAO2017_FIG2;Pt=TRANSVERSE;Pl=LONGITUDINAL'
    assert r.flow.Xt == pytest.approx((b.pitch_transverse-.014)/.014)
    assert r.flow.Xl == pytest.approx((b.pitch_longitudinal-.036)/.036)
    assert r.flow.Xp == pytest.approx(.004/r.flow.reference_diameter)
    assert r.flow.Xf == pytest.approx(.0003/r.flow.reference_diameter)
    assert r.h_film == pytest.approx(86.47833865264215, rel=1e-12)
    assert r.pressure_drop == pytest.approx(372.055553752331, rel=1e-12)
    with pytest.raises(TubeGeometryApplicabilityError, match='INLINE'):
        Zhao2017AirsideModel().evaluate(replace(b, layout='inline'), 1.7, AIR)


@pytest.mark.parametrize('kwargs', [dict(pattern='checkerboard'), dict(phase='unknown'),
    dict(pattern='explicit', phase='p', site_offsets=(((0., float('nan')),),))])
def test_invalid_or_unrecorded_shift_patterns_reject(kwargs):
    with pytest.raises(ValueError):
        TubeBankLattice('shifted', **kwargs)


def test_shifted_round_geometry_has_no_conventional_velocity_fallback():
    from core.geometry import BareTube, TubeBundle
    b = TubeBundle(BareTube(.016, .02, 1., 1., 45.), 2, 3, .05, .06, 'shifted', 1)
    with pytest.raises(ValueError, match='SHIFTED'):
        b.reference_velocity(1., 1.2)
    assert b.tube_bank_lattice.layout_class == LayoutClass.SHIFTED


def test_source_defined_column_coordinates_do_not_enable_zhao2017():
    lattice = TubeBankLattice.zhao2016_staggered_column(
        pitch_transverse=.05334, row_counts=(2, 2, 2))
    b = replace(source_bundle(), n_rows=3, n_tubes_per_row=2,
        pitch_transverse=.05334, pitch_longitudinal=.030, lattice=lattice)
    assert b.layout == 'shifted'
    assert lattice.pattern == 'ZHAO2016_STAGGERED_COLUMN_FIG1B'
    assert lattice.nominal_offset == .02667
    assert b.tube_center_coordinates == (
        ((0., 0.), (0., .05334)),
        ((.030, .02667), (.030, .05334+.02667)),
        ((.060, 0.), (.060, .05334)),
    )
    for strict in (False, True):
        with pytest.raises(TubeGeometryApplicabilityError, match='SHIFTED'):
            Zhao2017AirsideModel(strict_source_domain=strict).evaluate(b, 1.7, AIR)
    with pytest.raises(TubeGeometryApplicabilityError, match='SHIFTED'):
        calculate_finned_ellipse_hydraulics(bundle=b, m_dot=1.7, inlet_props=AIR)


@pytest.mark.parametrize('pitch,counts', [(0., (2,)), (.05, (0,)), (.05, ())])
def test_source_defined_column_validates_finite_bank_dimensions(pitch, counts):
    with pytest.raises(ValueError):
        TubeBankLattice.zhao2016_staggered_column(pitch_transverse=pitch, row_counts=counts)
