# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace
import math
import pytest

from core.geometry import BareTube, TubeBundle, ContinuousPlateFinGeometry, TubeOrientation, TubeSurfaceType
from core.heat_transfer.continuous_airside import ExperimentalWang2000Model
from core.heat_transfer.continuous_fin_source import ContinuousFinSourceError
from core.heat_transfer.airside_correction import MultiplicativeAirsideCorrection
from core.heat_transfer.outside_dispatch import (
    evaluate_outside_thermal, evaluate_outside_hydraulics, calculate_resistance_network,
    tube_surface_type, IncompatibleOutsideCorrelationProviderError,
)
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.water import IAPWS97WaterSteamProvider
from core.pressure_drop.flow_path import build_outside_pressure_drop_result, SourceDomainPressureEstimateResult

AIR = FluidTransportProperties(1.18, 1.85e-5, .026, 1007.)
AIR_PROVIDER = ConstantPropertyProvider(AIR)
WATER = ConstantPropertyProvider(FluidTransportProperties(990., .0007, .63, 4200.))


def geometry():
    tube = BareTube(.008, .00952, .36, .36, 45., tube_orientation=TubeOrientation.VERTICAL_DOWNWARD)
    fin = ContinuousPlateFinGeometry(180, .000115, .002, 0., .2032, .05715,
        -.0127, -.009525, .00975, .002, 205.)
    return TubeBundle(tube, 3, 7.5, .0254, .01905, "staggered", 1,
        flow_arrangement="counterflow", continuous_fin_geometry=fin)


def hx(correction=None, **kwargs):
    return BareTubeHeatExchanger(geometry(), airside_model=ExperimentalWang2000Model(True, correction), **kwargs)


def test_dispatch_and_resistance_reconstruction():
    b, model = geometry(), ExperimentalWang2000Model(True, MultiplicativeAirsideCorrection(1.2, 1.3))
    thermal = evaluate_outside_thermal(bundle=b, m_dot=.15, props=AIR, airside_model=model)
    assert thermal.surface_type is tube_surface_type(b) is TubeSurfaceType.CONTINUOUS_PLATE_FIN
    n = calculate_resistance_network(bundle=b, alpha_inside=1500., outside_alpha_physical=thermal.alpha_physical,
        resistance_core_wall=.0002, fouling_resistance_inside=.0001, fouling_resistance_outside=.0003,
        airside_model=model, outside_thermal_result=thermal)
    r = thermal.continuous_result
    assert n.resistance_contact == 0.
    assert n.contact_input_mode == "empirical_included"
    assert n.resistance_total == pytest.approx(1/(1500*b.total_inner_area)+.0001/b.total_inner_area+.0002+
        1/(r.alpha*r.efficiency.effective_area)+.0003/b.total_outer_area)
    assert n.UA == pytest.approx(1/n.resistance_total)
    assert n.UA == pytest.approx(n.U_gross_outside*b.total_outer_area)
    assert n.conductance_fin_outside == pytest.approx(r.alpha*n.fin_efficiency*b.continuous_fin_surface.fin_face_area)


@pytest.mark.parametrize("operation", ["thermal", "hydraulic", "network", "simulation", "rating"])
def test_no_auto_or_bare_fallback(operation):
    b = geometry()
    with pytest.raises(ContinuousFinSourceError, match="explicit experimental"):
        if operation == "thermal":
            evaluate_outside_thermal(bundle=b, m_dot=.15, props=AIR)
        elif operation == "hydraulic":
            evaluate_outside_hydraulics(bundle=b, m_dot=.15, inlet_props=AIR)
        elif operation == "network":
            calculate_resistance_network(bundle=b, alpha_inside=1000., outside_alpha_physical=60., resistance_core_wall=0.)
        elif operation == "simulation":
            BareTubeHeatExchanger(b).simulate(HXSideInput(WATER, .1, 360., 101325.), HXSideInput(AIR_PROVIDER, .15, 290., 101325.))
        else:
            BareTubeHeatExchanger(b).rate(BalanceSideSpec(provider=WATER, m_dot=.1, T_in=360., T_out=350., p=101325.),
                BalanceSideSpec(provider=AIR_PROVIDER, m_dot=.15, T_in=290., p=101325.))


def test_wrong_family_selector_is_rejected():
    with pytest.raises(IncompatibleOutsideCorrelationProviderError):
        evaluate_outside_thermal(bundle=geometry(), m_dot=.15, props=AIR,
            airside_model=ExperimentalWang2000Model(True), finned_heat_transfer_provider="briggs_young_1963")


def test_signed_density_term_is_not_multiplied_or_reaggregated():
    b = geometry()
    warmer = replace(AIR, rho=.9)
    args = dict(bundle=b, m_dot=.15, inlet_props=AIR, midpoint_props=AIR, outlet_props=warmer)
    a = evaluate_outside_hydraulics(**args, airside_model=ExperimentalWang2000Model(True))
    c = evaluate_outside_hydraulics(**args, airside_model=ExperimentalWang2000Model(True, MultiplicativeAirsideCorrection(1., 3.)))
    assert c.dp_drag == 3*a.dp_drag
    assert c.dp_acceleration == a.dp_acceleration
    assert c.dp_total == pytest.approx(3*a.dp_drag+a.dp_acceleration)
    assert c.reference_diameter == .00975
    assert c.reference_density == pytest.approx(2/(1/AIR.rho+1/warmer.rho))
    assert c.reference_velocity == pytest.approx(c.reference_mass_flux/c.reference_density)
    path = build_outside_pressure_drop_result(c)
    assert isinstance(path, SourceDomainPressureEstimateResult)
    assert path.dp_total == c.dp_total and not path.validated_bank_irreversible


@pytest.mark.parametrize("iterate", [False, True])
def test_rating_simulation_energy_closure_and_repeated_mass_flows(iterate):
    exchanger = hx(MultiplicativeAirsideCorrection(1.1, 1.2), fouling_resistance_inside=.0001, fouling_resistance_outside=.0002)
    inside = HXSideInput(provider=WATER, m_dot=.1, T_in=360., p=101325.)
    duties = []
    for flow in (.1, .2):
        outside = HXSideInput(provider=AIR_PROVIDER, m_dot=flow, T_in=290., p=101325.)
        sim = exchanger.simulate(inside, outside, iterate=iterate)
        assert sim.q == pytest.approx(.1*4200*(360-sim.T_out_inside), rel=1e-7)
        assert sim.q == pytest.approx(flow*1007*(sim.T_out_outside-290), rel=1e-7)
        assert math.isfinite(sim.outside_dp_total)
        assert sim.finned_tube_diagnostics.continuous_result is not None
        assert sim.finned_tube_diagnostics.pressure_drop_metadata.source_status == "SOURCE_PARTIALLY_CLOSED"
        assert not sim.final_result.outside_side_pressure_drop.flow_path.validated_bank_irreversible
        rating = exchanger.rate(
            BalanceSideSpec(provider=WATER, m_dot=.1, T_in=360., T_out=sim.T_out_inside, p=101325.),
            BalanceSideSpec(provider=AIR_PROVIDER, m_dot=flow, T_in=290., p=101325.))
        assert rating.Q_required == pytest.approx(sim.q, rel=1e-7)
        assert rating.UA_actual == pytest.approx(sim.UA, rel=2e-5)
        duties.append(sim.q)
    assert duties[1] > duties[0]


@pytest.mark.parametrize("superheated", [False, True])
def test_internal_steam_uses_same_dry_profile_and_keeps_two_phase_dp_unavailable(superheated):
    steam = IAPWS97WaterSteamProvider()
    inside = HXSideInput(provider=steam, m_dot=.01 if superheated else .002, p=1e6,
        **({"T_in":520.} if superheated else {"quality_in":1.}))
    outside = HXSideInput(provider=AIR_PROVIDER, m_dot=.15, T_in=290., p=101325.)
    result = hx().simulate(inside, outside)
    phase = result.inside_phase_change
    assert phase.active
    assert phase.Q_total == pytest.approx(inside.m_dot*(inside.h_in-phase.h_out))
    assert phase.Q_condensation > 0.
    if superheated:
        assert phase.Q_desuperheat > 0.
    assert phase.two_phase_pressure_drop_status == "not_supported"
    assert result.tube_side_hydraulic is None and math.isnan(result.inside_dp_total)
    assert result.finned_tube_diagnostics.continuous_result is not None
    assert math.isfinite(result.outside_dp_total)


def test_direct_solve_and_steam_rating_profile_propagation():
    from core.heat_transfer.streams import SensibleHeatStream
    from core.properties.adapters import to_internal_fluid_props
    exchanger = hx()
    hot, cold = SensibleHeatStream(.1*4200., 360.), SensibleHeatStream(.15*1007., 290.)
    result = exchanger.solve(hot, cold, m_dot_tube_side=.1,
        tube_side_props=to_internal_fluid_props(WATER.props), m_dot_outside=.15, outside_props=AIR)
    assert result.Q > 0. and result.finned_tube_diagnostics.continuous_result is not None
    steam = IAPWS97WaterSteamProvider()
    rating = exchanger.rate(
        BalanceSideSpec(provider=steam, m_dot=.01, p=1e6, quality_in=1., quality_out=.8),
        BalanceSideSpec(provider=AIR_PROVIDER, m_dot=.15, p=101325., T_in=290.))
    assert rating.inside_phase_change.Q_condensation > 0.
    assert rating.finned_tube_diagnostics.continuous_result is not None
    assert math.isnan(rating.inside_dp_total)


def test_three_zone_steam_energy_and_installed_area_closure():
    inside = HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=.001, p=1e6, T_in=520.)
    outside = HXSideInput(provider=AIR_PROVIDER, m_dot=.15, T_in=290., p=101325.)
    exchanger = hx()
    r = exchanger.simulate(inside, outside)
    phase = r.inside_phase_change
    assert min(phase.Q_desuperheat, phase.Q_condensation, phase.Q_subcooling) > 0.
    assert r.q == pytest.approx(phase.Q_desuperheat+phase.Q_condensation+phase.Q_subcooling)
    assert r.q == pytest.approx(.15*1007*(r.T_out_outside-290.), rel=1e-7)
    assert phase.A_total == pytest.approx(exchanger.bundle.total_outer_area, rel=2e-8)
    assert "CONTINUOUS_FIN_EDGE_CELL_APPROXIMATION" in {w.code for w in r.warnings}


def test_external_wet_provider_rejected_and_default_unchanged():
    from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
    from core.phase_change.wet_coil_provider import ElmahdyMitalasWetCoilProvider
    wet = GasMixturePropertyProvider(GasMixtureSpec(components={"N2":.75, "O2":.15, "H2O":.10}, basis="mole"))
    with pytest.raises(ContinuousFinSourceError, match="wet/condensing"):
        hx().simulate(HXSideInput(provider=WATER, m_dot=.1, T_in=280., p=101325.),
            HXSideInput(provider=wet, m_dot=.15, T_in=350., p=101325.))
    assert ElmahdyMitalasWetCoilProvider().model_id == "elmahdy_mitalas_energyplus_v25_2_adapted"


@pytest.mark.parametrize("provider_name", ["elmahdy", "legacy"])
def test_builtin_wet_provider_direct_call_cannot_treat_common_sheets_as_bare(provider_name):
    from core.phase_change.wet_coil_provider import ElmahdyMitalasWetCoilProvider, LegacyBulkMeanWetCoilProvider, WetCoilProviderUnsupportedError
    provider = ElmahdyMitalasWetCoilProvider() if provider_name == "elmahdy" else LegacyBulkMeanWetCoilProvider()
    with pytest.raises(WetCoilProviderUnsupportedError, match="continuous-sheet"):
        provider.simulate(hx(), None, None, settings=None, wet_solver_options=None, context=None)


def test_strict_source_validation_cannot_be_overridden_by_correction():
    exchanger = hx(MultiplicativeAirsideCorrection(2., 2.), strict_source_domain=True)
    with pytest.raises(ContinuousFinSourceError, match="Strict source"):
        exchanger.simulate(HXSideInput(provider=WATER, m_dot=.1, T_in=360., p=101325.),
            HXSideInput(provider=AIR_PROVIDER, m_dot=.15, T_in=290., p=101325.))
