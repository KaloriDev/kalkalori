# SPDX-License-Identifier: GPL-3.0-only
"""Inside interface and metal separation on an independently closed network."""
from types import SimpleNamespace

import pytest

from core.models.bare_tube import BareTubeHeatExchanger
from core.phase_change.bulk_mean_surface_reporting import inside_surface_envelope
from core.tests.wall_temperature_envelope_test import build_hx
from core.tests.inside_water_condensation_integration_test import partial_result


def test_inside_interface_and_metal_with_both_fouling_layers():
    h = BareTubeHeatExchanger(build_hx().bundle,
        fouling_resistance_inside=.003, fouling_resistance_outside=.002)
    ai, ao = h.bundle.total_inner_area, h.bundle.total_outer_area
    # Known full resistance network: inside film, inner fouling, wall,
    # outer fouling and outside film. No source or interpolated temperature.
    hi, ho = 100., 40.
    ri, ro = 1/(hi*ai), 1/(ho*ao)+h.resistance_fouling_outside
    total = ri+h.resistance_fouling_inside+h.tube_wall_resistance()+ro
    q = (400.-300.)/total
    interface = 400.-q*ri
    solution = SimpleNamespace(alfa_i_effective=hi, alfa_o=1/(ro*ao),
        T_wall_inside=interface, Q_total=q, converged=True, iterations=2,
        surface_temperature_samples=())
    e = inside_surface_envelope(h, solution, inside_mean=400., outside_mean=300.)
    assert e.inside_surface_mean == pytest.approx(interface)
    assert e.inside_mean == pytest.approx(400.-q*(ri+h.resistance_fouling_inside))
    assert e.outside_mean == pytest.approx(300.+q*ro)
    assert e.outside_skin_mean == pytest.approx(300.+q/(ho*ao))
    assert e.inside_mean-e.outside_mean == pytest.approx(q*h.tube_wall_resistance())
    assert e.inside_surface_mean > e.inside_mean > e.outside_mean > e.outside_skin_mean


def test_actual_inside_result_uses_its_native_wet_states(partial_result):
    result = partial_result
    e = result.wall_temperature_envelope
    assert e.method == "inside_bulk_mean_native_interface_envelope"
    assert e.inside_surface_mean == result.inside_phase_change.wall_temperature_mean
    assert e.inside_mean == result.thermal_state.inside_wall_temperature
    assert e.outside_mean == result.thermal_state.outside_wall_temperature
    assert e.inside_surface_min <= e.inside_surface_mean <= e.inside_surface_max


@pytest.mark.parametrize("flow", ["counterflow", "cocurrentflow", "crossflow"])
def test_inside_physics_receives_the_selected_flow_override(monkeypatch, flow):
    from core.tests.inside_water_condensation_integration_test import _hx, _wet_spec, _dry_spec
    from core.properties.gas_mixture import GasMixturePropertyProvider
    from core.models.simulation import HXSideInput
    from core.phase_change import integration
    captured = []
    def selected(hx, **options):
        captured.append(options["flow_arrangement"])
        raise RuntimeError("selected topology reached inside wet solver")
    monkeypatch.setattr(integration, "solve_inside_condensation", selected)
    a = HXSideInput(GasMixturePropertyProvider(_wet_spec()), 1., 360., 101325.)
    b = HXSideInput(GasMixturePropertyProvider(_dry_spec()), 5., 290., 101325.)
    with pytest.raises(RuntimeError, match="selected topology"):
        _hx().simulate(a, b, flow_arrangement=flow)
    assert captured == [flow]
