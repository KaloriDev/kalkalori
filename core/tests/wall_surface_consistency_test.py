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


@pytest.mark.parametrize("side", ["inside", "outside"])
@pytest.mark.parametrize("film,downstream", [(1e-6, .1), (.1, 1e-6), (.002, .004)])
@pytest.mark.parametrize("latent", [0., 1000.])
def test_interface_states_close_each_resistance_balance(side, film, downstream, latent):
    from core.phase_change.condensation_solver_helpers import representative_interface_envelope
    temperatures = ((400., 350., 280., 310.) if side == "inside"
                    else (280., 310., 400., 350.))
    ti, ui, to, uo = temperatures
    lo, mean, hi = representative_interface_envelope(
        topology="counterflow", side=side, T_in_inside=ti, T_out_inside=ui,
        T_in_outside=to, T_out_outside=uo, R_wet_film=film,
        R_downstream=downstream, Q_latent=latent)
    pairs = ((ti, uo), (ui, to))
    independent = []
    for i, o in pairs:
        gas, coolant = (i, o) if side == "inside" else (o, i)
        # Solve the two conductances directly (different algebraic form).
        ts = (gas / film + coolant / downstream + latent) / (1 / film + 1 / downstream)
        assert (gas - ts) / film + latent == pytest.approx((ts - coolant) / downstream, rel=1e-8)
        independent.append(ts)
    assert (lo, hi) == pytest.approx((min(independent), max(independent)))
    assert mean == pytest.approx(sum(independent) / 2)


def test_both_stream_temperature_changes_affect_wet_interface_span():
    from core.phase_change.condensation_solver_helpers import representative_interface_envelope
    kw = dict(side="outside", T_in_inside=280., T_out_inside=310.,
              T_in_outside=400., T_out_outside=350., R_wet_film=.004,
              R_downstream=.002, Q_latent=1000.)
    c = representative_interface_envelope(topology="counterflow", **kw)
    p = representative_interface_envelope(topology="cocurrentflow", **kw)
    assert c[1] == pytest.approx(p[1])
    # Countercurrent terminal surface temperatures span the two stream ends;
    # parallel temperature changes can compensate in the resistance balance.
    assert c[2] - c[0] > p[2] - p[0]


def test_unresolved_crossflow_bounds_are_not_converted_to_a_wet_area_distribution():
    from core.phase_change.condensation_solver_helpers import representative_wet_bulk_pairs
    assert representative_wet_bulk_pairs("crossflow", "outside", 280., 310., 400., 350.) == (
        (295., 400.), (295., 350.))
    assert representative_wet_bulk_pairs("crossflow", "inside", 400., 350., 280., 310.) == (
        (400., 295.), (350., 295.))


@pytest.mark.parametrize("fin_k", [1., 200., 1e8])
@pytest.mark.parametrize("contact,root", [(0., .025), (.001, .025), (.001, .028)])
def test_fin_surface_mean_uses_efficiency_and_area_not_base_tip_midpoint(fin_k, contact, root):
    from dataclasses import replace
    from core.tests.fin_surface_temperature_diagnostics_test import _finned_tube, _hx, _sides
    tube = replace(_finned_tube(contact=contact, D_root=root), fin_k=fin_k)
    hx = _hx(tube)
    a, b = _sides()
    r = hx.simulate(a, b)
    e = r.wall_temperature_envelope
    assert e.outside_skin_min <= e.outside_skin_mean <= e.outside_skin_max
    for probe in e.probes:
        from core.heat_transfer.fin_efficiency import annular_fin_efficiency
        fin = annular_fin_efficiency(tube, probe.outside_alpha_physical)
        tf = fin.mean_fin_temperature(probe.fin_base_temperature, probe.outside_bulk_temperature)
        area_mean = (fin.primary_outside_area * probe.outside_primary_surface_temperature
                     + fin.fin_area * tf) / fin.outside_area_gross
        assert probe.outside_surface_temperature_mean == pytest.approx(area_mean)
        assert probe.fin_surface_temperature_min <= tf <= probe.fin_surface_temperature_max
    if fin_k == 1e8:
        assert abs(e.fin_base_mean - e.fin_tip_mean) < .001
