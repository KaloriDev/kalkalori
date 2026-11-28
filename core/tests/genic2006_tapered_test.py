"""Explicit tapered geometry extension; independent integrals and public paths."""
from dataclasses import replace
import math

import pytest
from scipy.integrate import quad

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.pressure_drop import (
    Genic2006InlinePressureDropProvider, Genic2006StaggeredPressureDropProvider,
    calculate_finned_tube_pressure_drop, calculate_finned_tube_bank_hydraulics,
)
from core.properties import ConstantPropertyProvider
from core.properties.common import FluidTransportProperties
from core.heat_transfer import OutsideProviderPolicy

GAS = FluidTransportProperties(rho=1.05, mu=2.05e-5, k=.029, cp=1015.)


def bundle(layout="staggered", **tube_changes):
    core = BareTube(D_i=.027, D_o=.031, length_total=1.8,
                    length_effective=1.76, wall_k=38.)
    dimensions = dict(core_tube=core, D_root=.031, D_fin=.061,
        fin_pitch=.0036, fin_thickness_root=.00075, fin_thickness_tip=.00025,
        fin_k=170.)
    dimensions.update(tube_changes)
    tube = CircularFinnedTube(**dimensions)
    return TubeBundle(tube=tube, n_rows=8, n_tubes_per_row=9,
        pitch_transverse=.074, pitch_longitudinal=.067,
        layout=layout, n_passes_tube=2, flow_arrangement="counterflow")


def independent(b):
    """Integrate local material/surface/projection without provider helpers."""
    t=b.tube; r0=t.D_root/2; r1=t.D_fin/2; pitch=t.fin_pitch
    thickness=lambda x: t.fin_thickness_root+(t.fin_thickness_tip_effective-t.fin_thickness_root)*(x-r0)/(r1-r0)
    slope=math.sqrt(1+((t.fin_thickness_tip_effective-t.fin_thickness_root)/(2*(r1-r0)))**2)
    metal=math.pi*r0*r0*pitch+quad(lambda x:2*math.pi*x*thickness(x),r0,r1,epsabs=1e-15)[0]
    surface=(math.pi*t.D_root*(pitch-t.fin_thickness_root)
        +quad(lambda x:4*math.pi*x*slope,r0,r1,epsabs=1e-15)[0]
        +math.pi*t.D_fin*t.fin_thickness_tip_effective)
    projected=t.D_root*pitch+quad(lambda x:2*thickness(x),r0,r1,epsabs=1e-15)[0]
    frontal=b.pitch_transverse*pitch-projected
    diagonal=2*(math.hypot(b.pitch_longitudinal,b.pitch_transverse/2)*pitch-projected)
    cell=b.pitch_transverse*b.pitch_longitudinal*pitch
    return dict(metal=metal, surface=surface, af=frontal, ad=diagonal,
                de=4*(cell-metal)/surface, W=surface/frontal, epsilon=1-metal/cell)


@pytest.mark.parametrize("layout", ["staggered", "inline"])
@pytest.mark.parametrize("pitch", [.002, .0036, .008])
def test_exact_taper_integrals_native_loss_and_explicit_selection(layout, pitch):
    b=bundle(layout, fin_pitch=pitch);g=independent(b);mass=3.7
    selector="genic2006_"+layout+"_tapered"
    r=calculate_finned_tube_pressure_drop(mass,b,GAS,provider=selector,allow_engineering_estimates=True)
    flux=mass/b.minimum_free_flow_area;Re=flux*g['de']/GAS.mu
    shape=(g['ad']/g['af'])**.3 if layout=='staggered' else ((b.pitch_longitudinal-b.tube.D_root)/(b.pitch_transverse-b.tube.D_root))**.7
    xi=((.43+17.7/Re**.3)*g['W']**(-.8) if layout=='staggered' else
        (.4+8.5/Re**.3)*g['W']**(-.75))*shape
    expected=b.n_rows*xi*(b.pitch_longitudinal/g['de'])*flux**2/(2*GAS.rho)
    assert r.dp_drag==pytest.approx(expected,rel=2e-14)
    assert r.diagnostics['de']==pytest.approx(g['de'],rel=2e-14)
    assert r.diagnostics['epsilon']==pytest.approx(g['epsilon'],rel=2e-14)
    assert r.diagnostics['asurf']==pytest.approx(g['surface'],rel=2e-14)
    assert r.diagnostics['metal_volume_per_period']==pytest.approx(g['metal'],rel=2e-14)
    assert r.minimum_free_flow_area==pytest.approx(b.frontal_flow_area*g['af']/(b.pitch_transverse*b.tube.fin_pitch))
    assert r.method==selector
    assert r.diagnostics['prediction_status']=='UNVALIDATED_ENGINEERING_EXTENSION'
    assert r.diagnostics['source_geometry_mapping_closed'] is False
    assert math.isnan(r.correlation_fin_thickness) and math.isnan(r.f)
    assert 'genic2006_linear_taper_geometry_extension' in {w.code for w in r.warnings}
    with pytest.raises(ValueError,match='constant-thickness'):
        calculate_finned_tube_pressure_drop(mass,b,GAS,provider='genic2006_'+layout)
    with pytest.raises(ValueError,match='not source-validated'):
        calculate_finned_tube_pressure_drop(mass,b,GAS,provider=selector,strict_source_domain=True,allow_engineering_estimates=True)


@pytest.mark.parametrize('layout',['staggered','inline'])
def test_constant_limit_is_identical_and_mean_thickness_is_not_a_taper(layout):
    b=bundle(layout);t=b.tube
    constant=replace(b,tube=replace(t,fin_thickness_root=.0005,fin_thickness_tip=.0005))
    old=calculate_finned_tube_pressure_drop(3.,constant,GAS,provider='genic2006_'+layout)
    opted=calculate_finned_tube_pressure_drop(3.,constant,GAS,provider='genic2006_'+layout+'_tapered')
    assert replace(old,diagnostics={})==replace(opted,diagnostics={})
    assert {k:v for k,v in old.diagnostics.items() if k!='requested_provider'}=={k:v for k,v in opted.diagnostics.items() if k!='requested_provider'}
    real=calculate_finned_tube_pressure_drop(3.,b,GAS,provider='genic2006_'+layout+'_tapered',allow_engineering_estimates=True)
    assert real.minimum_free_flow_area==old.minimum_free_flow_area
    assert not math.isclose(real.diagnostics['de'],old.diagnostics['de'],rel_tol=1e-3)
    assert real.diagnostics['asurf']!=old.diagnostics['asurf']


@pytest.mark.parametrize('layout',['staggered','inline'])
def test_mean_state_bank_adapter_separates_acceleration(layout):
    b=bundle(layout);selector='genic2006_'+layout+'_tapered'
    raw=calculate_finned_tube_pressure_drop(3.,b,GAS,provider=selector,allow_engineering_estimates=True)
    hot=replace(GAS,rho=.8);cold=replace(GAS,rho=1.3)
    bank=calculate_finned_tube_bank_hydraulics(3.,b,inlet_props=hot,
        midpoint_props=GAS,outlet_props=cold,temperature_in=520.,temperature_out=340.,
        pressure_drop_provider=selector,allow_engineering_estimates=True)
    assert bank.dp_drag==raw.dp_drag
    assert bank.dp_acceleration==pytest.approx((3./b.frontal_flow_area)**2*(1/cold.rho-1/hot.rho))
    assert bank.dp_total==bank.dp_drag+bank.dp_acceleration
    assert bank.native_result.method==selector
    assert math.isnan(bank.inlet.coefficient) and math.isnan(bank.outlet.coefficient)


@pytest.mark.parametrize('cls',[Genic2006InlinePressureDropProvider,Genic2006StaggeredPressureDropProvider])
def test_extension_flag_is_explicit_boolean(cls):
    with pytest.raises(TypeError,match='boolean'):cls(allow_linear_taper=1)


@pytest.mark.parametrize('layout',['inline','staggered'])
def test_rating_and_simulation_keep_default_thermal_solution(layout):
    b=bundle(layout)
    if layout=='staggered':b=replace(b,pitch_longitudinal=b.pitch_transverse*math.sqrt(3)/2)
    # No thermal provider or equation is replaced by selecting hydraulics.
    baseline=BareTubeHeatExchanger(b)
    selected=BareTubeHeatExchanger(b,outside_provider_policy=OutsideProviderPolicy(allow_engineering_estimates=True))
    selector='genic2006_'+layout+'_tapered'
    water=ConstantPropertyProvider(FluidTransportProperties(rho=980.,mu=.0006,k=.62,cp=4180.))
    gas=ConstantPropertyProvider(GAS)
    sides=[HXSideInput(provider=water,m_dot=3.,T_in=310.,p=200000.),
           HXSideInput(provider=gas,m_dot=2.,T_in=450.,p=101325.)]
    original=baseline.simulate(*sides,finned_pressure_drop_provider=('auto' if layout=='inline' else 'robinson_briggs_1966'))
    new=selected.simulate(*sides,finned_pressure_drop_provider=selector)
    assert new.q==original.q and new.UA==original.UA
    assert new.thermal_state.outside_alpha_physical==original.thermal_state.outside_alpha_physical
    rated=selected.rate(BalanceSideSpec(provider=water,m_dot=3.,T_in=310.,T_out=new.T_out_inside,p=200000.),
                        BalanceSideSpec(provider=gas,m_dot=2.,T_in=450.,T_out=new.T_out_outside,p=101325.),
                        finned_pressure_drop_provider=selector)
    assert rated.outside_tube_bank_hydraulic.native_result.method=='genic2006_'+layout+'_tapered'
    assert rated.outside_dp_drag>0.


def test_thickness_profile_change_is_visible_without_changing_minimum_area():
    b=bundle();same_mean=replace(b,tube=replace(b.tube,fin_thickness_root=.00065,fin_thickness_tip=.00035))
    a=calculate_finned_tube_pressure_drop(3.,b,GAS,provider='genic2006_staggered_tapered',allow_engineering_estimates=True)
    c=calculate_finned_tube_pressure_drop(3.,same_mean,GAS,provider='genic2006_staggered_tapered',allow_engineering_estimates=True)
    assert a.minimum_free_flow_area==c.minimum_free_flow_area
    assert a.diagnostics['fin_volume_per_period']!=c.diagnostics['fin_volume_per_period']
    assert a.dp_drag!=c.dp_drag
