# SPDX-License-Identifier: GPL-3.0-only
"""Independent geometry invariants for the Phase A engineering idealization."""
from dataclasses import asdict, replace
import math

import pytest

from core.geometry import (
    BareTube, CircularFinnedTube, SerratedCircularFinnedTube, TubeBundle,
    TubeSurfaceType,
)


def serrated(**changes):
    values = dict(
        core_tube=BareTube(.020, .0254, 1., 1., 45.), fin_k=45.,
        D_root=.0254, D_fin=.0534, fin_thickness_root=.0006,
        fin_pitch=.0032, fin_density_material=7850., segments_per_turn=40,
        serration_depth=.007, kerf_width_at_tip=.0003,
    )
    values.update(changes)
    return SerratedCircularFinnedTube(**values)


def solid(tube):
    return CircularFinnedTube(
        core_tube=tube.core_tube, fin_k=tube.fin_k, D_root=tube.D_root,
        D_fin=tube.D_fin, fin_thickness_root=tube.fin_thickness_root,
        fin_pitch=tube.fin_pitch,
    )


def test_audit_reference_geometry():
    t = serrated()
    s = solid(t)
    assert s.fin_volume_per_length == pytest.approx(3.24919e-4, rel=2e-6)
    assert 7850 * s.fin_volume_per_length == pytest.approx(2.5506, rel=2e-5)
    assert s.outside_area_geometric_per_length == pytest.approx(1.17935, rel=5e-6)
    assert t.fin_volume_per_length == pytest.approx(3.11234e-4, rel=2e-6)
    assert t.fin_mass_per_length == pytest.approx(2.4432, rel=1e-5)
    assert t.external_area_per_length == pytest.approx(1.23815, rel=5e-6)
    assert t.fin_area_per_length == pytest.approx(1.17331139860061)
    assert t.primary_area_per_length == pytest.approx(.06483461838845936)
    assert 100 * (t.fin_volume_per_length / s.fin_volume_per_length - 1) == pytest.approx(-4.21, abs=.01)
    assert 100 * (t.external_area_per_length / s.outside_area_geometric_per_length - 1) == pytest.approx(4.99, abs=.01)


@pytest.mark.parametrize("kerf", [0., .0003, .002])
def test_zero_depth_has_no_serration_effect(kerf):
    t = serrated(serration_depth=0., kerf_width_at_tip=kerf)
    s = solid(t)
    assert t.fin_volume_removed_per_turn == 0.
    assert t.fin_volume_per_length == s.fin_volume_per_length
    assert t.fin_area_per_length == s.fin_area_per_length
    assert t.external_area_per_length == s.outside_area_geometric_per_length
    assert t.fin_area_deltas_per_turn == (0., 0., 0., 0.)


def test_zero_kerf_does_not_remove_mass_but_has_two_slit_sides():
    t = serrated(kerf_width_at_tip=0.)
    s = solid(t)
    added_sides = 2 * 40 * .007 * .0006
    assert t.fin_volume_removed_per_turn == 0.
    assert t.fin_volume_per_length == s.fin_volume_per_length
    assert t.fin_mass_per_length == 7850 * s.fin_volume_per_length
    assert t.fin_area_deltas_per_turn == (0., 0., added_sides, 0.)
    assert t.fin_area_per_fin == pytest.approx(s.fin_area_per_fin + added_sides)


def test_independent_cut_surface_components_and_removed_volume():
    t = serrated()
    # Independent integration of n angular sectors between radii rs and r1.
    r0, r1, rs = .0127, .0267, .0197
    beta, n, thickness = .0003 / r1, 40, .0006
    annulus = math.pi * (r1**2 - r0**2)
    removed_plan_area = n * beta / 2 * (r1**2 - rs**2)
    face_delta, tip_delta, sides, bottom = t.fin_area_deltas_per_turn
    assert face_delta == pytest.approx(-2 * removed_plan_area)
    assert tip_delta == pytest.approx(-n * beta * r1 * thickness)
    assert sides == pytest.approx(2 * n * (r1 - rs) * thickness)
    assert bottom == pytest.approx(n * beta * rs * thickness)
    expected_volume = (annulus - removed_plan_area) * thickness
    expected_area = 2 * (annulus - removed_plan_area) + (2*math.pi - n*beta)*r1*thickness + sides + bottom
    assert t.fin_volume_per_fin == pytest.approx(expected_volume)
    assert t.fin_area_per_fin == pytest.approx(expected_area)


def test_area_can_increase_or_decrease_while_mass_decreases():
    t = serrated()
    s = solid(t)
    assert t.fin_volume_per_length < s.fin_volume_per_length
    assert t.external_area_per_length > s.outside_area_geometric_per_length
    wide = serrated(kerf_width_at_tip=.002)
    assert wide.fin_volume_per_length < t.fin_volume_per_length
    assert wide.external_area_per_length < s.outside_area_geometric_per_length


def test_volume_monotone_in_kerf_and_depth_including_endpoints():
    volumes = [serrated(kerf_width_at_tip=k).fin_volume_per_length
               for k in [0., .00001, .0003, .002, .004]]
    assert all(a > b for a, b in zip(volumes, volumes[1:]))
    height = serrated().fin_height
    volumes = [serrated(serration_depth=d).fin_volume_per_length
               for d in [0., .001, .007, height]]
    assert all(a > b > 0 for a, b in zip(volumes, volumes[1:]))


def test_small_serration_approaches_solid_geometry():
    t = serrated(serration_depth=1e-12)
    s = solid(t)
    assert t.fin_volume_per_length == pytest.approx(s.fin_volume_per_length, rel=1e-9)
    assert t.external_area_per_length == pytest.approx(s.outside_area_geometric_per_length, rel=1e-9)


@pytest.mark.parametrize("kerf", [math.pi*.0534/40, .005])
def test_overlapping_kerfs_rejected(kerf):
    with pytest.raises(ValueError, match="Overlapping kerfs"):
        serrated(kerf_width_at_tip=kerf)


@pytest.mark.parametrize("name", [
    "D_root", "D_fin", "fin_pitch", "fin_thickness_root", "fin_k",
    "serration_depth", "kerf_width_at_tip", "fin_density_material", "segment_width_at_tip",
])
@pytest.mark.parametrize("bad", [-.001, math.nan, math.inf, -math.inf])
def test_invalid_dimensions_and_properties_rejected(name, bad):
    with pytest.raises(ValueError):
        serrated(**{name: bad})


@pytest.mark.parametrize("n", [0, -1, 2.5, True, math.nan, math.inf])
def test_segment_count_is_explicit_positive_integer(n):
    with pytest.raises(ValueError, match="segments_per_turn"):
        serrated(segments_per_turn=n)


@pytest.mark.parametrize("changes", [
    {"D_root": .024}, {"D_fin": .0254}, {"fin_pitch": .0006},
    {"serration_depth": .015}, {"fin_density_material": 0.},
    {"fin_thickness_root": 0.}, {"fin_pitch": 0.},
])
def test_geometry_guards(changes):
    with pytest.raises(ValueError):
        serrated(**changes)


@pytest.mark.parametrize("changes", [
    {"fin_thickness_tip": .0005}, {"base_style": "u_foot"},
    {"base_style": "l_foot"}, {"segment_tip_twist_angle": .1},
])
def test_out_of_scope_geometry_rejected(changes):
    with pytest.raises(NotImplementedError):
        serrated(**changes)


def test_tip_arc_width_closure_and_no_inferred_count():
    t = serrated()
    width = math.pi * .0534 / 40 - .0003
    assert t.segment_width_at_tip == width
    assert t.slits_per_turn == t.segments_per_turn == 40
    assert t.angular_kerf_width == .0003/.0267
    assert serrated(segment_width_at_tip=width) == t
    with pytest.raises(ValueError, match="Tip arc widths"):
        serrated(segment_width_at_tip=width * 1.01)
    values = asdict(t)
    values.pop("_uncut")
    values.pop("segments_per_turn")
    with pytest.raises(TypeError, match="segments_per_turn"):
        SerratedCircularFinnedTube(**values)


def test_si_round_trip_and_dimensional_scaling():
    # Explicit external mm -> SI -> mm; core performs no unit conversions.
    t = serrated(D_root=25.4e-3, D_fin=53.4e-3, fin_pitch=3.2e-3,
                 fin_thickness_root=.6e-3, serration_depth=7e-3, kerf_width_at_tip=.3e-3)
    assert 1000*t.D_root == pytest.approx(25.4)
    assert 1000*t.kerf_width_at_tip == pytest.approx(.3)
    inputs = asdict(t)
    inputs.pop("_uncut")
    inputs["core_tube"] = BareTube(**inputs["core_tube"])
    assert SerratedCircularFinnedTube(**inputs) == t
    scale = 10
    larger = serrated(
        core_tube=BareTube(.2, .254, 10., 10., 45.),
        D_root=t.D_root*scale, D_fin=t.D_fin*scale, fin_pitch=t.fin_pitch*scale,
        fin_thickness_root=t.fin_thickness*scale, serration_depth=t.serration_depth*scale,
        kerf_width_at_tip=t.kerf_width_at_tip*scale)
    assert larger.fin_volume_per_length == pytest.approx(t.fin_volume_per_length*scale**2)
    assert larger.fin_mass_per_length == pytest.approx(t.fin_mass_per_length*scale**2)
    assert larger.external_area_per_length == pytest.approx(t.external_area_per_length*scale)


def test_composed_core_and_shared_contact_contract():
    t = serrated(fin_contact_resistance=.0001, fin_contact_efficiency=.95)
    for name in ("D_i", "D_o", "length_total", "length_effective", "wall_k",
                 "roughness_inner", "roughness_outer", "tube_orientation",
                 "relative_roughness_inner", "relative_roughness_outer",
                 "flow_area", "hydraulic_diameter", "area_inner", "inside_wetted_perimeter"):
        assert getattr(t, name) == getattr(t.core_tube, name)
    assert t.contact_input_mode == "explicit_resistance"
    assert t.contact_resistance_used == .0001
    assert t.contact_area_per_tube == pytest.approx(math.pi*.0254*.0006/.0032)
    b = TubeBundle(t, 4, 3, .08, .08, "staggered", 1)
    assert b.fin_contact_thermal_resistance == pytest.approx(.0001/t.contact_area_per_tube/b.n_tubes_total)
    layered = replace(t, D_root=.028)
    bank = replace(b, tube=layered)
    assert bank.root_conduction_resistance == layered.root_conduction_resistance_per_tube/bank.n_tubes_total


def test_geometric_area_source_provenance_and_bundle_areas():
    t = serrated()
    assert not isinstance(t, CircularFinnedTube)
    assert t.surface_type is TubeSurfaceType.SERRATED_HELICAL_FIN
    assert t.geometry_model == "IDEALIZED_RADIAL_SECTOR_CUT"
    assert t.geometry_provenance == "ENGINEERING_DERIVATION"
    assert t.physical_product_exactness == "NOT_CLAIMED"
    assert t.validated_against_manufacturer_CAD is False
    assert "SERRATED_IDEALIZED_RADIAL_SECTOR_CUT" in {w.code for w in t.geometry_warnings}
    with pytest.raises(TypeError, match="external_area_per_length"):
        serrated(external_area_per_length=2.)
    b = TubeBundle(t, 4, 3, .08, .08, "staggered", 1)
    assert b.total_outer_area == pytest.approx(b.total_primary_outside_area+b.total_fin_area)
    assert b.total_outer_area == b.total_outer_geometric_area
    assert b.total_fin_area == b.total_fin_geometric_area
    assert b.total_outer_area == b.n_tubes_total*t.length_effective*t.external_area_per_length


@pytest.mark.parametrize("layout,changes", [
    ("inline", {"pitch_transverse": .05}), ("inline", {"pitch_longitudinal": .05}),
    ("staggered", {"pitch_longitudinal": .02}),
])
def test_bundle_retains_physical_fin_envelope_clearance(layout, changes):
    values = dict(tube=serrated(), n_rows=4, n_tubes_per_row=3,
                  pitch_transverse=.08, pitch_longitudinal=.08, layout=layout, n_passes_tube=1)
    values.update(changes)
    with pytest.raises(ValueError, match="D_fin"):
        TubeBundle(**values)
