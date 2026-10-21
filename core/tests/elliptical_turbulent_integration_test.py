# SPDX-License-Identifier: GPL-3.0-only
"""Public synthetic turbulent/nonconfocal Rating and Simulation acceptance."""
from dataclasses import replace
import math
import pytest
from core.geometry import TubeGeometryApplicabilityError
from core.heat_transfer import (EllipticalFlowAssumptions, TubeWallModel,
                                tube_wall_resistance_diagnostics)
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.models.heat_balance import BalanceSideSpec
from core.properties import ConstantPropertyProvider
from core.properties.common import FluidTransportProperties
from core.tests.zhao2017_fixtures import source_bundle, AIR


def context():
    b=source_bundle()
    # Synthetic nonconfocal 3:1 inner section; keep source outside surface.
    core=replace(b.tube.core_tube,inner_semi_major_axis=.015,
                 inner_semi_minor_axis=.005)
    b=replace(b,tube=replace(b.tube,core_tube=core))
    hx=BareTubeHeatExchanger(b)
    liquid=ConstantPropertyProvider(FluidTransportProperties(rho=1000.,mu=.0005,k=.65,cp=4200.))
    mass=24000*.0005*b.internal_flow_area_per_pass/core.hydraulic_diameter
    a=HXSideInput(provider=liquid,m_dot=mass,T_in=360.,p=500000.)
    c=HXSideInput(provider=ConstantPropertyProvider(AIR),m_dot=1.7,T_in=300.,p=101325.)
    return hx,a,c


def test_normal_rating_simulation_network_energy_and_metadata():
    hx,a,b=context()
    sim=hx.simulate(a,b)
    assert sim.converged and math.isfinite(sim.q) and sim.q>0
    assert sim.q==pytest.approx(a.m_dot*4200*(a.T_in-sim.T_out_inside),rel=1e-5)
    assert sim.q==pytest.approx(b.m_dot*AIR.cp*(sim.T_out_outside-b.T_in),rel=1e-5)
    rating=hx.rate(
        BalanceSideSpec(provider=a.provider,m_dot=a.m_dot,T_in=a.T_in,T_out=sim.T_out_inside,p=a.p),
        BalanceSideSpec(provider=b.provider,m_dot=b.m_dot,T_in=b.T_in,T_out=sim.T_out_outside,p=b.p),
        include_simulation=False)
    assert rating.UA_actual==pytest.approx(sim.UA,rel=1e-5)
    assert rating.Q_required==pytest.approx(sim.q,rel=1e-5)
    assert rating.thermal_state.diagnostics.inside_model_metadata.model_id=='elliptical_yang2018_turbulent_T'
    wall=tube_wall_resistance_diagnostics(hx.bundle.tube.core_tube,n_tubes=hx.bundle.n_tubes_total,
                                        model=TubeWallModel.ELLIPTICAL_NUMERICAL)
    network=sim.finned_tube_diagnostics
    assert network.resistance_core_wall==wall.resistance
    assert 1/sim.UA==pytest.approx(network.resistance_inside+wall.resistance+network.resistance_outside)
    hydraulic=sim.tube_side_hydraulic
    assert hydraulic.midpoint.friction_factor_method=='elliptical_jones_prandtl_turbulent_engineering'
    assert hydraulic.dp_friction>0 and hydraulic.entrance_count==hydraulic.exit_count==0
    assert any(w.code=='elliptical_turbulent_friction_engineering' for w in sim.warnings)
    assert any(w.code=='tube_end_losses_not_evaluated' for w in rating.warnings)
    for data in (sim,rating):
        assert data.finned_tube_diagnostics.airside_result.h_film>0
        assert data.finned_tube_diagnostics.fin_efficiency_result.converged


def test_confocal_guard_and_out_of_range_turbulent_flow_remain_errors():
    hx,a,b=context()
    hx.tube_wall_model=TubeWallModel.CONFOCAL_EXACT
    with pytest.raises(TubeGeometryApplicabilityError,match='not confocal'): hx.simulate(a,b)
    hx.tube_wall_model=TubeWallModel.ELLIPTICAL_NUMERICAL
    with pytest.raises(TubeGeometryApplicabilityError,match='transition'): hx.simulate(replace(a,m_dot=a.m_dot/8),b)
