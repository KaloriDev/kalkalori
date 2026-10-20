# SPDX-License-Identifier: GPL-3.0-only
"""Physical film coefficient -> fin conduction -> area -> normal network."""
from dataclasses import replace
import inspect

import pytest

from core.geometry import (EllipticalTube, RectangularIndividualFin,
                           RectangularFinnedEllipticalTube, TubeBundle,
                           TubeGeometryApplicabilityError, TubeSurfaceType)
from core.heat_transfer.outside_dispatch import (
    calculate_resistance_network, evaluate_outside_thermal,
    evaluate_outside_hydraulics, tube_surface_type)
from core.heat_transfer.thin_plate_fin import ThinPlateFinEfficiency2D
from core.heat_transfer.thermal_iteration import fin_surface_temperatures
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.rating import run_rating
from core.models.simulation import run_simulation


def bundle():
    tube = RectangularFinnedEllipticalTube(
        EllipticalTube(.012, .006, .010, .004, 2., 1.8, 40.),
        RectangularIndividualFin(.040, .030, .0004, .003, 200.))
    return TubeBundle(tube, 4, 6, .060, .060, 'inline', 2)


def network(b=None, h=80., **kwargs):
    return calculate_resistance_network(bundle=b or bundle(), alpha_inside=1200.,
                                        outside_alpha_physical=h,
                                        resistance_core_wall=.001, **kwargs)


def test_film_coefficient_is_retained_and_efficiency_applied_once():
    b = bundle()
    n = network(b)
    t = b.tube
    eta = ThinPlateFinEfficiency2D().evaluate(t, 80.).eta_fin
    area = b.n_tubes_total * (t.area_primary_outside + eta*t.area_fin_faces)
    assert tube_surface_type(b) is TubeSurfaceType.RECTANGULAR_INDIVIDUAL_FINNED
    assert n.outside_alpha_physical == 80.
    assert n.fin_efficiency == eta
    assert n.A_physical == b.total_outer_area
    assert n.A_effective == pytest.approx(area)
    assert n.A_physical > n.A_effective
    assert n.conductance_primary_outside == 80*b.total_primary_outside_area
    assert n.conductance_fin_outside == pytest.approx(80*eta*t.area_fin_faces*b.n_tubes_total)
    assert n.resistance_outside == pytest.approx(1/(80*area))
    assert n.outside_alpha_effective_gross*n.area_outside_gross == pytest.approx(80*area)
    assert 1/n.UA == pytest.approx(1/(1200*b.total_inner_area) + .001 + 1/(80*area))
    assert n.resistance_root == n.resistance_contact == 0
    assert n.contact_area == t.area_contact*b.n_tubes_total
    # Existing reporting reconstructs the isothermal base and coolest edge.
    primary, base, edge, low, high = fin_surface_temperatures(
        network=n, outside_wall_temperature=350., outside_bulk_temperature=300.,
        heat_rate=(350.-300.)/n.resistance_outside)
    assert primary == base == high == 350.
    assert edge == low == pytest.approx(300 + 50*n.fin_efficiency_result.tip_temperature_ratio)


def test_unit_efficiency_limit_and_explicit_geometry_area_semantics():
    b = bundle()
    t = b.tube
    ideal = replace(t, fin=replace(t.fin, thermal_conductivity=1.e14))
    n = network(replace(b, tube=ideal))
    assert n.fin_efficiency == pytest.approx(1., abs=1.e-10)
    assert n.A_effective == pytest.approx(b.total_primary_outside_area + b.n_tubes_total*t.area_fin_faces)
    assert n.A_physical-n.A_effective == pytest.approx(b.n_tubes_total*t.area_fin_outer_edge)


def test_conductivity_changes_only_conduction_not_geometry_or_h():
    b = bundle()
    poorer = replace(b, tube=replace(b.tube, fin=replace(b.tube.fin, thermal_conductivity=10.)))
    n, p = network(b), network(poorer)
    assert p.A_physical == n.A_physical
    assert p.outside_alpha_physical == n.outside_alpha_physical
    assert p.fin_efficiency < n.fin_efficiency
    assert p.A_effective < n.A_effective
    assert p.conductance_primary_outside == n.conductance_primary_outside
    assert p.conductance_fin_outside < n.conductance_fin_outside


def test_pitch_scales_faces_and_contact_and_does_not_change_eta():
    b = bundle()
    sparse = replace(b, tube=replace(b.tube, fin=replace(b.tube.fin, axial_pitch=.006)))
    n, s = network(b), network(sparse)
    assert s.fin_efficiency_result is n.fin_efficiency_result
    assert s.contact_area == .5*n.contact_area
    assert s.conductance_fin_outside == .5*n.conductance_fin_outside
    assert s.conductance_primary_outside > n.conductance_primary_outside
    assert s.A_effective == pytest.approx(sparse.total_primary_outside_area
        + s.fin_efficiency*sparse.n_tubes_total*sparse.tube.area_fin_faces)


def test_h_dependency_pass_count_and_fouling_remain_explicit():
    b = bundle()
    n = network(b)
    high = network(b, h=160.)
    assert high.fin_efficiency < n.fin_efficiency
    assert high.A_effective < n.A_effective
    assert network(replace(b, n_passes_tube=1)) == n
    dirty = network(b, fouling_resistance_inside=.002, fouling_resistance_outside=.003)
    assert dirty.fin_efficiency == n.fin_efficiency
    assert dirty.A_effective == n.A_effective
    assert 1/dirty.UA == pytest.approx(1/n.UA + .002/b.total_inner_area + .003/b.total_outer_area)


def invoke(fn, **overrides):
    values = {name: None for name, p in inspect.signature(fn).parameters.items()
              if p.default is inspect.Parameter.empty
              and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    values.update(overrides)
    return fn(**values)


@pytest.mark.parametrize('fn', [evaluate_outside_thermal, evaluate_outside_hydraulics])
def test_unsupported_orientation_refuses_before_property_or_provider_access(fn):
    b=bundle()
    bad=replace(b,tube=replace(b.tube,core_tube=replace(b.tube.core_tube,orientation_angle=.2)))
    with pytest.raises(TubeGeometryApplicabilityError, match='parallel to flow'):
        invoke(fn, bundle=bad)


def test_generic_circular_throat_is_still_unavailable_for_rectangular_fins():
    hx = BareTubeHeatExchanger(bundle())
    for name in ('minimum_free_flow_gap', 'minimum_free_flow_area'):
        # Do not create a surrogate A_min/diameter definition for this surface.
        with pytest.raises(TubeGeometryApplicabilityError):
            getattr(hx.bundle, name)
