# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic full-engine acceptance; no prescribed outside alpha or private data."""
from dataclasses import replace
import math
import pytest

from core.geometry import TubeGeometryApplicabilityError
from core.heat_transfer import EllipticalFlowAssumptions, TubeWallModel, ThinPlateFinEfficiency2D
from core.heat_transfer.finned_ellipse_airside import Zhao2017AirsideModel
from core.heat_transfer.airside_correction import AirsideCorrection
from core.heat_transfer.outside_dispatch import evaluate_outside_hydraulics
from core.pressure_drop.flow_path import build_outside_pressure_drop_result, _tube_bank_core_stage
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.models.heat_balance import BalanceSideSpec
from core.properties import ConstantPropertyProvider, DryAirPropertyProvider
from core.properties.common import FluidTransportProperties
from core.tests.zhao2017_fixtures import source_bundle, AIR


ASSUMPTIONS=EllipticalFlowAssumptions(regime='laminar',development='fully_developed',thermal_boundary_condition='H1')


def context(n_rows=8,model=None,outside_provider=None):
    b=source_bundle(n_rows)
    hx=BareTubeHeatExchanger(b,airside_model=model,elliptical_flow_assumptions=ASSUMPTIONS,
                             tube_wall_model=TubeWallModel.CONFOCAL_EXACT)
    # Ideal declared FD laminar H1 tube-side state; no equivalent circular law.
    liquid=ConstantPropertyProvider(FluidTransportProperties(rho=1000.,mu=.02,cp=4200.,k=.6))
    a=HXSideInput(provider=liquid,m_dot=1.,T_in=360.,p=300000.)
    c=HXSideInput(provider=outside_provider or ConstantPropertyProvider(AIR),m_dot=1.7,T_in=300.,p=101325.)
    return hx,a,c


def test_normal_simulation_and_rating_source_domain_regression():
    hx,a,b=context()
    r=hx.simulate(a,b)
    assert r.q==pytest.approx(24058.39114704213,rel=2e-9)
    assert r.UA==pytest.approx(481.30043424526707,rel=2e-9)
    assert (r.T_out_inside,r.T_out_outside)==pytest.approx((354.27181163165665,314.0675892568367),rel=2e-9)
    assert r.outside_dp_total==pytest.approx(372.055553752331,rel=2e-12)
    d=r.finned_tube_diagnostics
    assert d.airside_result is not None and not d.airside_result.extrapolated_parameters
    assert d.airside_result.h_film==d.outside_alpha_physical==pytest.approx(86.47833865264215)
    assert d.eta_fin==pytest.approx(.6539683710987041)
    assert 0<d.eta_fin<=1 and d.A_effective<d.A_outside_geometric
    assert d.conductance_primary_outside+d.conductance_fin_outside==pytest.approx(d.outside_alpha_physical*d.A_effective)
    assert d.fin_efficiency_result.converged
    assert r.tube_side_hydraulic.entrance_count==0 and r.tube_side_hydraulic.exit_count==0
    assert any(w.code=='tube_end_losses_not_evaluated' for w in r.warnings)
    inside=BalanceSideSpec(provider=a.provider,m_dot=a.m_dot,T_in=a.T_in,T_out=r.T_out_inside,p=a.p)
    outside=BalanceSideSpec(provider=b.provider,m_dot=b.m_dot,T_in=b.T_in,T_out=r.T_out_outside,p=b.p)
    rating=hx.rate(inside,outside,include_simulation=True)
    assert rating.Q_required==pytest.approx(r.q)
    assert rating.UA_actual==pytest.approx(r.UA)
    assert rating.outside_dp_total==pytest.approx(r.outside_dp_total)
    assert rating.simulation.q==pytest.approx(r.q,rel=2e-5)


@pytest.mark.parametrize('rows',[2,4])
def test_low_rows_run_and_warn_and_strict_rejects(rows):
    hx,a,b=context(rows)
    r=hx.simulate(a,b)
    assert math.isfinite(r.q) and r.outside_dp_total>0
    assert 'N' in r.finned_tube_diagnostics.airside_result.extrapolated_parameters
    assert any(w.code=='zhao2017_extrapolated_N' for w in r.warnings)
    strict,_,_=context(rows,Zhao2017AirsideModel(strict_source_domain=True))
    with pytest.raises(TubeGeometryApplicabilityError,match='N'):
        strict.simulate(a,b)


def test_pressure_estimate_is_not_an_irreversible_path_or_extra_acceleration():
    bank=evaluate_outside_hydraulics(bundle=source_bundle(),m_dot=1.7,inlet_props=AIR)
    path=build_outside_pressure_drop_result(bank)
    assert path.dp_total==bank.dp_total==bank.midpoint.airside_result.pressure_drop
    assert not path.validated_bank_irreversible and not path.components_resolved
    assert path.pressure_basis=='source_domain_inlet_to_outlet_estimate'
    assert not path.groups and path.dp_local==0
    assert path.delta_dynamic_pressure_total==0
    assert bank.dp_acceleration==0  # no separately added estimate, not a physical decomposition
    with pytest.raises(ValueError,match='irreversible'):
        _tube_bank_core_stage(bank)


def test_iteration_refreshes_h_eta_and_applies_configured_correction():
    samples=[]
    class Record:
        def correct(self,base_result,*,bundle,props):
            samples.append(base_result)
            return AirsideCorrection(provenance='synthetic_state_record',h_film_multiplier=1.1,pressure_drop_multiplier=1.2)
    hx,a,b=context(model=Zhao2017AirsideModel(correction_provider=Record()),outside_provider=DryAirPropertyProvider())
    r=hx.simulate(a,b)
    d=r.finned_tube_diagnostics
    assert len({round(s.h_film,8) for s in samples})>1
    assert d.outside_alpha_physical==pytest.approx(1.1*d.airside_result.h_film_base)
    assert d.eta_fin==ThinPlateFinEfficiency2D().evaluate(hx.bundle.tube,d.outside_alpha_physical).eta_fin
    bank=r.outside_tube_bank_hydraulic
    assert bank.dp_total==pytest.approx(1.2*bank.midpoint.airside_result.pressure_drop_base)
    assert d.hydraulic_airside_result==bank.midpoint.airside_result
    assert d.conductance_fin_outside+d.conductance_primary_outside==pytest.approx(d.outside_alpha_physical*d.A_effective)


def test_no_implicit_inside_assumptions_wall_model_or_wrong_orientation():
    hx,a,b=context()
    with pytest.raises(TubeGeometryApplicabilityError,match='EllipticalFlowAssumptions'):
        BareTubeHeatExchanger(hx.bundle).simulate(a,b)
    with pytest.raises(TubeGeometryApplicabilityError,match='CONFOCAL_EXACT'):
        BareTubeHeatExchanger(hx.bundle,elliptical_flow_assumptions=ASSUMPTIONS).simulate(a,b)
    bad=replace(hx.bundle,tube=replace(hx.bundle.tube,core_tube=replace(hx.bundle.tube.core_tube,orientation_angle=.2)))
    with pytest.raises(TubeGeometryApplicabilityError,match='parallel to flow'):
        BareTubeHeatExchanger(bad,elliptical_flow_assumptions=ASSUMPTIONS,tube_wall_model=TubeWallModel.CONFOCAL_EXACT).simulate(a,b)


def test_explicit_wet_model_is_still_rejected():
    hx,a,b=context()
    with pytest.raises(TubeGeometryApplicabilityError,match='Wet-coil'):
        hx.simulate(a,b,wet_coil_provider=object())
