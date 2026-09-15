# KalKalori - GNU GPL v3 only
"""Synthetic exchanger tests exercise real enhanced thermal/hydraulic paths."""
from dataclasses import replace
import math

import pytest

from core.models.bare_tube import BareTubeHeatExchanger
from core.geometry.tube import BareTube
from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.bundle import TubeBundle
from core.models.simulation import HXSideInput
from core.models.heat_balance import BalanceSideSpec
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.water import IAPWS97WaterSteamProvider
from core.enhancements import (
    TubeSideEnhancement, TwistedTapeGeometry, Yang2020TwistedTapeProvider,
    EnhancementUnsupportedError,
)
from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
from core.enhancements.integration import hydraulic_evaluator
from core.tests.enhancement_provider_contract_test import FakeExternalProvider, fake_configuration


def bundle(finned=False, passes=1):
    tube = BareTube(D_i=.012, D_o=.014, length_total=.3, length_effective=.3, wall_k=45)
    if finned:
        tube = CircularFinnedTube(core_tube=tube, D_fin=.028, D_root=.014,
                                  fin_k=200, fin_thickness_root=.0004, fin_pitch=.002)
    return TubeBundle(tube=tube, n_rows=4, n_tubes_per_row=8,
                      pitch_transverse=.04, pitch_longitudinal=.04*math.sqrt(3)/2,
                      layout="staggered", n_passes_tube=passes, flow_arrangement="counterflow")


def configuration():
    return TubeSideEnhancement(Yang2020TwistedTapeProvider(),
                               TwistedTapeGeometry(.036, .012, .001), "liquid")


def inputs(b, re=500, provider=None):
    provider = provider or ConstantPropertyProvider(FluidTransportProperties(1000, .001, .5, 4000))
    flow = re*.001*math.pi*.012/4*b.n_tubes_per_pass_effective
    inside = HXSideInput(provider=provider, m_dot=flow, T_in=300, p=2e5)
    outside = HXSideInput(provider=ConstantPropertyProvider(
        FluidTransportProperties(1.1, 2e-5, .03, 1000)), m_dot=.2, T_in=340, p=1e5)
    return inside, outside


def test_unused_roughness_provider_warning_reaches_real_solvers():
    b = bundle()
    rough = replace(b, tube=replace(b.tube, roughness_inner=2e-5))
    inside, outside = inputs(b)
    clean_result = BareTubeHeatExchanger(b).simulate(inside, outside, tube_side_enhancement=configuration())
    hx = BareTubeHeatExchanger(rough)
    simulation = hx.simulate(inside, outside, tube_side_enhancement=configuration())
    assert simulation.inside_dp_friction == clean_result.inside_dp_friction
    rating = hx.rate(
        BalanceSideSpec(provider=inside.provider,p=inside.p,m_dot=inside.m_dot,T_in=300,T_out=simulation.T_out_inside),
        BalanceSideSpec(provider=outside.provider,p=outside.p,m_dot=outside.m_dot,T_in=340),
        include_simulation=True,tube_side_enhancement=configuration())
    for result in (simulation, rating, rating.simulation):
        assert 'yang2020_roughness_ignored' in {w.code for w in result.warnings}


@pytest.mark.parametrize("finned", [False, True])
@pytest.mark.parametrize("iterate", [False, True])
def test_real_simulation_routes_and_warning_propagation(finned, iterate):
    b = bundle(finned)
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=configuration())
    inside, outside = inputs(b)
    result = hx.simulate(inside, outside, iterate=iterate)
    assert result.converged
    enhanced = result.tube_side_enhancement
    assert enhanced.provider_id == configuration().provider.provider_id
    assert result.inside_alfa_mean == enhanced.alpha_inside
    assert result.q > 0
    assert result.T_out_inside > inside.T_in
    points = (result.inside_properties_inlet, result.inside_properties_midpoint,
              result.inside_properties_outlet)
    for point in points:
        assert point.enhancement.provider_id == enhanced.provider_id
        assert point.enhancement.reference.reynolds == pytest.approx(500)
        assert point.friction_factor == point.enhancement.f_darcy
    assert result.inside_dp_friction == pytest.approx(.3*.701682172082434659*1000/(576*.024))
    assert result.inside_dp_acceleration == 0
    codes = {w.code for w in result.warnings}
    assert {"enhancement_yang2020_scope", "enhancement_0d_reference_evaluation"} <= codes
    assert "tube_bundle_hydraulics_reynolds_outside_range" not in codes
    assert "tube_ht_laminar_regime" not in codes
    assert "tube_ht_transition_regime" not in codes
    assert b.internal_flow_area_per_pass == b.n_tubes_per_pass_effective*math.pi*.012**2/4
    if iterate:
        assert result.thermal_state.diagnostics.inside_length_correction == 1


def test_rating_simulation_bridge_and_surface_margin():
    b = bundle()
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=configuration())
    inside, outside = inputs(b)
    simulation = hx.simulate(inside, outside, surface_margin=.25)
    rating = hx.rate(
        BalanceSideSpec(provider=inside.provider, p=inside.p, m_dot=inside.m_dot,
                        T_in=inside.T_in, T_out=simulation.T_out_inside),
        BalanceSideSpec(provider=outside.provider, p=outside.p, m_dot=outside.m_dot,
                        T_in=outside.T_in),
        include_simulation=True,
    )
    assert rating.alfa_i == rating.tube_side_enhancement.alpha_inside
    assert rating.tube_side_enhancement.provider_id == simulation.tube_side_enhancement.provider_id
    assert rating.UA_actual == pytest.approx(simulation.UA_actual, rel=1e-5)
    assert simulation.overdesign_factor == pytest.approx(.25)
    assert rating.overdesign_factor == pytest.approx(.25, rel=1e-4)
    assert rating.inside_dp_friction == simulation.inside_dp_friction
    assert "enhancement_yang2020_scope" in {w.code for w in rating.warnings}


class VariableLiquid:
    def at(self, T, p):
        return FluidTransportProperties(1000-.3*(T-300), .001*math.exp(-.01*(T-300)), .5, 5000)


@pytest.mark.parametrize("iterate", [False, True])
def test_independent_hydraulic_film_reference_bootstraps_wall_state(iterate):
    class HydraulicFilmProvider(FakeExternalProvider):
        hydraulic_property_reference = "film"

        def __init__(self):
            self.states = []

        def evaluate(self, geometry, state):
            assert state.hydraulic is not None
            self.states.append((state.position, state.bulk.temperature,
                                state.hydraulic.temperature))
            return super().evaluate(geometry, state)

    b = bundle()
    inside, outside = inputs(b, provider=VariableLiquid())
    provider = HydraulicFilmProvider()
    enhancement = TubeSideEnhancement(provider, "test_geometry", "liquid")
    result = BareTubeHeatExchanger(
        b, tube_side_enhancement=enhancement,
    ).simulate(inside, outside, iterate=iterate)

    assert result.converged
    assert provider.states
    assert {position for position, _, _ in provider.states} >= {
        "thermal", "inlet", "midpoint", "outlet",
    }
    assert all(reference_temperature != bulk_temperature
               for _, bulk_temperature, reference_temperature in provider.states)


def test_wall_iteration_uses_authoritative_viscosity_once():
    b = bundle()
    inside, outside = inputs(b, provider=VariableLiquid())
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=configuration())
    result = hx.simulate(inside, outside)
    state = result.thermal_state
    e = result.tube_side_enhancement
    assert state.converged
    ratio = state.inside_bulk_props.mu/state.inside_wall_props.mu
    assert e.wall_correction == pytest.approx(ratio**.14)
    assert e.wall_correction != 1
    assert state.diagnostics.inside_combined_correction == e.wall_correction
    points = (result.inside_properties_inlet, result.inside_properties_midpoint,
              result.inside_properties_outlet)
    gradients = [p.enhancement.pressure_gradient(p.rho) for p in points]
    assert result.inside_dp_friction == pytest.approx(.3*(gradients[0]+4*gradients[1]+gradients[2])/6)
    assert result.inside_dp_acceleration > 0


def test_external_reference_diameter_and_velocity_do_not_change_local_losses():
    b = bundle(passes=2)
    inside, _ = inputs(b)
    kwargs = dict(m_dot=inside.m_dot, flow_area_per_pass=b.internal_flow_area_per_pass,
                  hydraulic_diameter=b.internal_hydraulic_diameter,
                  hydraulic_length_total=b.internal_length_total, n_tube_passes=b.n_passes_tube,
                  tube_path_type=b.tube_path_type, provider=VariableLiquid(),
                  temperature_in=300, temperature_out=310, pressure=2e5)
    smooth = calculate_tube_bundle_hydraulics(**kwargs)
    enhanced = calculate_tube_bundle_hydraulics(**kwargs,
        enhancement_evaluator=hydraulic_evaluator(fake_configuration(), b, kwargs["provider"]))
    assert enhanced.dp_straight_tube_friction != smooth.dp_straight_tube_friction
    assert enhanced.dp_straight_tube_acceleration == smooth.dp_straight_tube_acceleration
    assert enhanced.entrance_results == smooth.entrance_results
    assert enhanced.exit_results == smooth.exit_results
    assert enhanced.inlet.velocity == smooth.inlet.velocity
    assert enhanced.inlet.enhancement.reference.velocity == pytest.approx(2*smooth.inlet.velocity)
    assert enhanced.inlet.enhancement.reference.friction_diameter == .006


def test_none_is_exact_smooth_default():
    b = bundle()
    a = BareTubeHeatExchanger(b).simulate(*inputs(b))
    b_result = BareTubeHeatExchanger(b, tube_side_enhancement=None).simulate(*inputs(b))
    for name in ("q", "UA", "inside_alfa_mean", "inside_dp_friction", "inside_dp_acceleration",
                 "T_out_inside", "T_out_outside", "overdesign_factor"):
        assert getattr(a, name) == getattr(b_result, name)
    assert a.warnings == b_result.warnings
    assert a.tube_side_enhancement is None
    assert a.inside_properties_midpoint.enhancement is None


@pytest.mark.parametrize("re", [50, 2000, 3000, 20000])
def test_out_of_range_never_falls_back(re):
    b = bundle()
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=configuration())
    with pytest.raises(EnhancementUnsupportedError, match="reynolds"):
        hx.simulate(*inputs(b, re))


def test_steam_and_two_phase_are_rejected_before_special_solver():
    b = bundle()
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=configuration())
    _, outside = inputs(b)
    steam = HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=.1, p=2e5, quality_in=.5)
    with pytest.raises(EnhancementUnsupportedError, match="phase"):
        hx.simulate(steam, outside)
    with pytest.raises(EnhancementUnsupportedError, match="phase"):
        hx.rate(BalanceSideSpec(provider=steam.provider, p=2e5, m_dot=.1,
                                quality_in=1, quality_out=0),
                BalanceSideSpec(provider=outside.provider, p=1e5, m_dot=.2, T_in=300))


@pytest.mark.parametrize("iterate", [False, True])
def test_cooling_rejected_even_without_wall_iteration(iterate):
    b = bundle()
    inside, outside = inputs(b)
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=configuration())
    with pytest.raises(EnhancementUnsupportedError, match="cooling"):
        hx.simulate(replace(inside, T_in=340), replace(outside, T_in=300), iterate=iterate)


@pytest.mark.parametrize("re", [100, 1100])
def test_supported_re_endpoints_through_simulation(re):
    b = bundle()
    result = BareTubeHeatExchanger(b, tube_side_enhancement=configuration()).simulate(*inputs(b, re))
    assert result.tube_side_enhancement.reference.reynolds == pytest.approx(re)
