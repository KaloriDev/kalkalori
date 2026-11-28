"""Preferred source-domain AUTO, explicit policies and independent channels."""
from dataclasses import replace
import math
import pytest
from core.tests.genic2006_tapered_test import bundle,GAS
from core.pressure_drop import calculate_finned_tube_pressure_drop as pressure
from core.pressure_drop import calculate_finned_tube_bank_hydraulics as bank_pressure
from core.pressure_drop import Genic2006StaggeredPressureDropProvider,RobinsonBriggs1966Provider
from core.heat_transfer import OutsideProviderPolicy
from core.heat_transfer.outside_dispatch import evaluate_outside_thermal,evaluate_outside_hydraulics
from core.properties import ConstantPropertyProvider
from core.properties.common import FluidTransportProperties
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.models.heat_balance import BalanceSideSpec

def constant():
    b=bundle();return replace(b,tube=replace(b.tube,fin_thickness_tip=b.tube.fin_thickness_root))

def test_constant_auto_uses_genic_and_records_actual_resolution():
    b=constant();a=pressure(3.7,b,GAS);g=pressure(3.7,b,GAS,provider='genic2006_staggered')
    assert a.dp_drag==g.dp_drag
    assert a.method=='genic2006_staggered'
    assert a.diagnostics['requested_provider']=='auto'
    assert a.diagnostics['resolved_provider']=='genic2006_staggered'
    assert 'Genić passes' in a.diagnostics['auto_selection_reason']
    assert not a.diagnostics['engineering_estimate']

@pytest.mark.parametrize('model',['auto','genic2006_staggered_tapered'])
def test_taper_requires_permission_and_remains_estimated(model):
    b=bundle()
    with pytest.raises(ValueError,match='allow_engineering_estimates=True'):pressure(3.7,b,GAS,provider=model)
    r=pressure(3.7,b,GAS,provider=model,allow_engineering_estimates=True)
    assert r.method=='genic2006_staggered_tapered' and r.diagnostics['engineering_estimate']
    assert r.diagnostics['prediction_status']=='UNVALIDATED_ENGINEERING_EXTENSION'
    with pytest.raises(ValueError,match='not source-validated'):
        pressure(3.7,b,GAS,provider=model,allow_engineering_estimates=True,strict_source_domain=True)

@pytest.mark.parametrize('changes',[dict(fluid_family='combustion gas'),dict(temperature_in=500.,temperature_out=350.)])
def test_auto_state_extensions_require_permission(changes):
    with pytest.raises(ValueError,match='allow_engineering_estimates=True'):pressure(3.7,constant(),GAS,**changes)
    r=pressure(3.7,constant(),GAS,allow_engineering_estimates=True,**changes)
    assert r.diagnostics['engineering_estimate']

@pytest.mark.parametrize('change',[dict(n_rows=4),dict(pitch_transverse=.12),dict(pitch_longitudinal=.12),dict(pitch_transverse=.108,pitch_longitudinal=.046)])
def test_auto_rejects_unsupported_geometry_without_reference_fallback(change):
    with pytest.raises(ValueError,match='AUTO has no applicable'):
        pressure(3.7,replace(constant(),**change),GAS,allow_engineering_estimates=True)

@pytest.mark.parametrize('mass',[.0001,10000.])
def test_auto_rejects_reynolds_extrapolation_even_if_estimates_authorized(mass):
    with pytest.raises(ValueError,match='550<Re_de<130000'):
        pressure(mass,constant(),GAS,allow_engineering_estimates=True)

@pytest.mark.parametrize('error',[RuntimeError('execution failed'),ValueError('invalid equation state')])
def test_auto_does_not_substitute_after_candidate_error(monkeypatch,error):
    def fail(self,request):raise error
    def forbidden(self,request):raise AssertionError('Reference fallback must not execute')
    monkeypatch.setattr(Genic2006StaggeredPressureDropProvider,'evaluate',fail)
    monkeypatch.setattr(RobinsonBriggs1966Provider,'evaluate',forbidden)
    with pytest.raises(type(error),match=str(error)):pressure(3.7,constant(),GAS)

@pytest.mark.parametrize('field,message', [('dp_drag','bank loss'),
    ('reynolds_number','Reynolds number'),('reference_velocity','reference velocity')])
def test_auto_rejects_invalid_candidate_output(monkeypatch,field,message):
    valid=pressure(3.7,constant(),GAS,provider='genic2006_staggered')
    monkeypatch.setattr(Genic2006StaggeredPressureDropProvider,'evaluate',lambda self,r:replace(valid,**{field:math.nan}))
    with pytest.raises(ValueError,match=message):pressure(3.7,constant(),GAS)

def test_native_bank_uses_single_loss_and_separate_acceleration():
    b=bundle();hot=replace(GAS,rho=.8);cold=replace(GAS,rho=1.3)
    a=bank_pressure(3.7,b,inlet_props=hot,midpoint_props=GAS,outlet_props=cold,allow_engineering_estimates=True)
    expected=pressure(3.7,b,GAS,provider='genic2006_staggered_tapered',allow_engineering_estimates=True)
    assert a.dp_drag==expected.dp_drag
    assert a.dp_acceleration==pytest.approx((3.7/b.frontal_flow_area)**2*(1/cold.rho-1/hot.rho))
    assert a.dp_total==a.dp_drag+a.dp_acceleration
    assert a.selection_metadata['requested_provider']=='auto'

class NamedGas(ConstantPropertyProvider):
    fluid='HEOS::CarbonDioxide'

def test_dispatch_preserves_non_air_warnings_without_numerical_changes():
    b=constant();b=replace(b,pitch_longitudinal=b.pitch_transverse*math.sqrt(3)/2)
    air=ConstantPropertyProvider(GAS);gas=NamedGas(GAS)
    h0=evaluate_outside_thermal(bundle=b,m_dot=3.7,props=GAS,property_provider=air)
    h1=evaluate_outside_thermal(bundle=b,m_dot=3.7,props=GAS,property_provider=gas)
    assert h0.alpha_physical==h1.alpha_physical
    assert any('briggs_young_1963_non_air' in w.code for w in h1.warnings)
    kwargs=dict(bundle=b,m_dot=3.7,temperature_in=450.,temperature_out=350.,pressure=101325.,finned_pressure_drop_provider='robinson_briggs_1966')
    p0=evaluate_outside_hydraulics(property_provider=air,**kwargs)
    p1=evaluate_outside_hydraulics(property_provider=gas,**kwargs)
    assert p0.dp_drag==p1.dp_drag and p0.dp_total==p1.dp_total
    assert any('non_air' in w.code for w in p1.warnings)
    assert p1.selection_metadata['engineering_estimate']

def test_rating_simulation_auto_policy_and_thermal_independence():
    b=bundle();b=replace(b,pitch_longitudinal=b.pitch_transverse*math.sqrt(3)/2)
    hx=BareTubeHeatExchanger(b,outside_provider_policy=OutsideProviderPolicy(allow_engineering_estimates=True))
    water=ConstantPropertyProvider(FluidTransportProperties(rho=980.,mu=.0006,k=.62,cp=4180.))
    sides=(HXSideInput(provider=water,m_dot=3.,T_in=310.,p=200000.),HXSideInput(provider=NamedGas(GAS),m_dot=2.,T_in=450.,p=101325.))
    reference=hx.simulate(*sides,finned_pressure_drop_provider='robinson_briggs_1966')
    auto=hx.simulate(*sides)
    assert (auto.q,auto.UA,auto.thermal_state.outside_alpha_physical)==(reference.q,reference.UA,reference.thermal_state.outside_alpha_physical)
    assert auto.outside_tube_bank_hydraulic.metadata.method=='genic2006_staggered_tapered'
    inputs=[BalanceSideSpec(provider=s.provider,m_dot=s.m_dot,T_in=s.T_in,T_out=T,p=s.p) for s,T in zip(sides,[auto.T_out_inside,auto.T_out_outside])]
    rated=hx.rate(*inputs)
    assert rated.outside_tube_bank_hydraulic.selection_metadata['requested_provider']=='auto'
    assert any('non_air' in w.code for w in rated.warnings)
