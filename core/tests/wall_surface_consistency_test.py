# SPDX-License-Identifier: GPL-3.0-only
"""v0.9.7 physical invariants, using independently constructed public cases."""
import math

import pytest

from core.geometry._thermal_topology import CounterCrossflowTopology
from core.heat_transfer.topology_reporting import representative_temperature_pairs
from core.tests.wall_temperature_envelope_test import build_hx, envelope, RecordingProvider


@pytest.mark.parametrize("topology,expected", [
    ("counterflow", ((650., 390.), (520., 300.))),
    ("cocurrentflow", ((650., 300.), (520., 390.))),
])
def test_terminal_pairs_do_not_mix_incompatible_planes(topology, expected):
    hx = build_hx()
    e = envelope(hx, RecordingProvider(1.1, []), RecordingProvider(.9, []),
                 flow_arrangement=topology)
    assert tuple((p.inside_bulk_temperature, p.outside_bulk_temperature)
                 for p in e.probes[:-1]) == expected
    assert e.inside_mean == e.probes[-1].inside_wall_temperature
    assert e.outside_mean == e.probes[-1].outside_wall_temperature
    assert e.applicability == "equivalent_flow_representative_states"


@pytest.mark.parametrize("topology", ["crossflow", CounterCrossflowTopology(3)])
def test_unresolved_crossflow_is_a_bound_not_a_local_profile(topology):
    pairs, assumption = representative_temperature_pairs(topology, 650., 520., 300., 390.)
    assert len(pairs) == 4
    assert assumption == "unresolved_crossflow_bulk_rectangle_bounds"


@pytest.mark.parametrize("topology", ["counterflow", "cocurrentflow", "crossflow"])
@pytest.mark.parametrize("temperatures", [
    (300., 300., 300., 300.),  # zero driving force
    (300.0001, 300.00009, 300., 300.00001),
    (650., 520., 300., 390.),
    (300., 390., 650., 520.),  # reverse heat direction
    (650., 649., 300., 301.),
    (650., 420., 300., 410.),
])
def test_finite_ordered_envelope_across_thermal_limits(topology, temperatures):
    ti, ui, to, uo = temperatures
    e = envelope(build_hx(), RecordingProvider(1.1, []), RecordingProvider(.9, []),
                 flow_arrangement=topology, inside_inlet_temperature=ti,
                 inside_outlet_temperature=ui, outside_inlet_temperature=to,
                 outside_outlet_temperature=uo)
    assert all(p.converged for p in e.probes)
    assert e.inside_min <= e.inside_mean <= e.inside_max
    assert e.outside_min <= e.outside_mean <= e.outside_max
    assert e.outside_skin_min <= e.outside_skin_mean <= e.outside_skin_max
    for p in e.probes:
        assert min(p.inside_bulk_temperature, p.outside_bulk_temperature) <= p.outside_skin_temperature_min
        assert p.outside_skin_temperature_max <= max(p.inside_bulk_temperature, p.outside_bulk_temperature)


def test_fouling_separates_exposed_skin_from_metal_with_signed_heat():
    from core.models.bare_tube import BareTubeHeatExchanger
    hx = build_hx()
    hx = BareTubeHeatExchanger(hx.bundle, fouling_resistance_outside=.003)
    for temps in ((650., 520., 300., 390.), (300., 390., 650., 520.)):
        ti, ui, to, uo = temps
        e = envelope(hx, RecordingProvider(1.1, []), RecordingProvider(.9, []),
                     inside_inlet_temperature=ti, inside_outlet_temperature=ui,
                     outside_inlet_temperature=to, outside_outlet_temperature=uo)
        for p in e.probes:
            expected = p.outside_wall_temperature - p.heat_rate_probe * hx.resistance_fouling_outside
            assert p.outside_skin_temperature_min == pytest.approx(expected)
            assert p.outside_surface_temperature_mean == pytest.approx(expected)
            assert p.outside_wall_temperature != pytest.approx(expected)


def test_continuous_plate_mean_exists_without_fabricated_tip_or_base():
    from core.tests.continuous_fin_integration_test import hx, WATER, AIR_PROVIDER
    from core.models.simulation import HXSideInput
    h = hx()
    r = h.simulate(HXSideInput(WATER, .1, 360., 101325.),
                   HXSideInput(AIR_PROVIDER, .15, 290., 101325.))
    e = r.wall_temperature_envelope
    assert math.isfinite(e.outside_skin_mean)
    assert math.isnan(e.outside_skin_min) and math.isnan(e.outside_skin_max)
    assert math.isnan(e.fin_base_mean) and math.isnan(e.fin_tip_mean)
    assert e.probes[-1].outside_bulk_temperature < e.outside_skin_mean < e.primary_surface_mean


def test_flow_topology_changes_extrema_without_changing_prescribed_stream_states():
    args = (build_hx(), RecordingProvider(1.1, []), RecordingProvider(.9, []))
    counter = envelope(*args, flow_arrangement="counterflow")
    parallel = envelope(*args, flow_arrangement="cocurrentflow")
    assert (counter.outside_min, counter.outside_max) != pytest.approx(
        (parallel.outside_min, parallel.outside_max))
    assert counter.outside_mean == pytest.approx(parallel.outside_mean)
