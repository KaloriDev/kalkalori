# SPDX-License-Identifier: GPL-3.0-only
"""Analytical geometry checks, independent of the numerical fin model."""
from dataclasses import asdict, replace
from copy import deepcopy
import math

import pytest

from core.geometry import (EllipticalTube, RectangularIndividualFin,
                           RectangularFinnedEllipticalTube, TubeBundle)


def tube():
    return RectangularFinnedEllipticalTube(
        EllipticalTube(.012, .006, .010, .004, 2., 1.8, 40.),
        RectangularIndividualFin(.030, .040, .0004, .003, 200.))


@pytest.mark.parametrize('name', ['outer_height', 'outer_width', 'thickness',
                                  'axial_pitch', 'thermal_conductivity'])
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_fin_inputs(name, value):
    with pytest.raises(ValueError):
        replace(tube().fin, **{name: value})


def test_enclosure_and_spacing():
    t = tube()
    for fin in (replace(t.fin, outer_height=.024), replace(t.fin, outer_width=.010)):
        with pytest.raises(ValueError, match='enclose'):
            replace(t, fin=fin)
    for pitch in (.0004, .0003):
        with pytest.raises(ValueError, match='exceed'):
            replace(t.fin, axial_pitch=pitch)


def test_exact_surface_decomposition():
    t = tube()
    n = 1.8 / .003
    hole = math.pi * .012 * .006
    one_face = .040 * .030 - hole
    # Perimeter comes from the existing exact elliptic-integral helper.
    perimeter = t.core_tube.outside_perimeter
    exposed = perimeter * n * (.003 - .0004)
    faces = 2 * one_face * n
    edge = 2 * (.040 + .030) * .0004 * n
    assert t.aperture_area_per_fin == hole
    assert t.fin_face_area_one_side == one_face
    assert t.area_primary_outside == pytest.approx(exposed)
    assert t.area_fin_faces == pytest.approx(faces)
    assert t.area_fin_outer_edge == pytest.approx(edge)
    assert t.area_contact == pytest.approx(perimeter * n * .0004)
    assert t.area_contact + exposed == pytest.approx(t.core_tube.area_outer)
    assert t.A_physical == pytest.approx(exposed + faces + edge)
    assert t.effective_outside_area(1.) == pytest.approx(exposed + faces)
    assert t.effective_outside_area(.5) == pytest.approx(exposed + .5 * faces)
    assert t.effective_outside_area(0.) == exposed
    assert t.A_physical > t.effective_outside_area(1.)


def test_length_count_and_pass_scaling_and_copy():
    t = tube()
    longer = replace(t, core_tube=replace(t.core_tube, length_effective=.9))
    for name in ('area_outer', 'area_contact', 'area_fin_faces', 'area_primary_outside'):
        assert getattr(longer, name) == pytest.approx(.5 * getattr(t, name))
    b = TubeBundle(t, 4, 6, .060, .060, 'inline', 1)
    assert b.total_outer_area == 24 * t.area_outer
    assert b.total_outer_geometric_area == b.total_outer_area
    assert b.total_fin_area == 24 * t.area_fin
    assert replace(b, n_tubes_per_row=12).total_outer_area == 2 * b.total_outer_area
    assert replace(b, n_passes_tube=2).total_outer_area == b.total_outer_area
    assert deepcopy(t) == t
    data = asdict(t)
    reconstructed = RectangularFinnedEllipticalTube(
        EllipticalTube(**data['core_tube']), RectangularIndividualFin(**data['fin']))
    assert reconstructed == t


def test_pitch_and_conductivity_are_independent_of_tube_scale():
    t = tube()
    softer = replace(t, fin=replace(t.fin, thermal_conductivity=10.))
    assert softer.A_physical == t.A_physical
    sparse = replace(t, fin=replace(t.fin, axial_pitch=.006))
    assert sparse.area_fin_faces == .5 * t.area_fin_faces
    assert sparse.area_fin_outer_edge == .5 * t.area_fin_outer_edge
    assert sparse.area_contact == .5 * t.area_contact
    assert sparse.area_primary_outside > t.area_primary_outside


def test_projection_rotates_with_the_ellipse():
    t = tube()
    for angle in (0., .4, math.pi / 2):
        rotated = replace(t, core_tube=replace(t.core_tube, orientation_angle=angle))
        f = t.fin.thickness / t.fin.axial_pitch
        expected = ((1-f) * rotated.core_tube.projected_blocking_area_per_length
                    + f * (.040*abs(math.sin(angle)) + .030*abs(math.cos(angle))))
        assert rotated.projected_blocking_area_per_length == pytest.approx(expected)
        assert rotated.A_physical == t.A_physical
