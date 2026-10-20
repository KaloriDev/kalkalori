# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace
import math
import pytest
from core.geometry import EllipticalTube, TubeGeometryApplicabilityError
from core.heat_transfer import EllipticalFlowAssumptions
from core.heat_transfer.internal_flow import FluidProps
from core.heat_transfer.internal_dispatch import internal_heat_transfer_diagnostics, internal_friction
from core.heat_transfer.elliptical_turbulent import (
    elliptical_turbulent_heat_transfer, elliptical_turbulent_friction, HTC_MODEL_ID, FRICTION_MODEL_ID)

FD=EllipticalFlowAssumptions('turbulent','fully_developed','T')
PROPS=FluidProps(1000.,.001,.6,4200.)


def tube(ratio=3.):
    return EllipticalTube(.02,.018,.015,.015/ratio,2.,1.8,40.)


def htc(t=None, re=24000., props=PROPS, **kw):
    t=t or tube()
    return elliptical_turbulent_heat_transfer(t,re*props.mu*t.flow_area/t.hydraulic_diameter,
                                              props,assumptions=kw.pop('assumptions',FD),**kw)


@pytest.mark.parametrize('re,nu',[(10000.,38.6513682034),(20000.,73.6416932299),(120000.,336.116936331)])
def test_yang_eq18_unit_prandtl_reference_values(re,nu):
    # Source Table 3 does not report Pr/transport properties, so do not tune
    # properties to force reproduction of its rounded air table.
    props=FluidProps(1000.,.001,.6,600.)
    result=htc(tube(1.5),re,props)
    assert result.Nu_base==pytest.approx(nu,rel=1e-10)
    assert result.alfa_corrected==result.Nu_base*props.k/tube(1.5).hydraulic_diameter
    assert result.model_metadata.model_id==HTC_MODEL_ID
    assert result.model_metadata.approximation


@pytest.mark.parametrize('ratio,c',[(1.5,1.),(2.,1.27),(2.5,1.17),(3.,.94),(4.,.94),(5.,.94)])
def test_yang_table4_coefficients(ratio,c):
    assert htc(tube(ratio)).Nu_base/htc(tube(1.5)).Nu_base==pytest.approx(c)


@pytest.mark.parametrize('ratio',[1.,1.6,2.2,2.9,5.1])
def test_no_invented_shape_coefficient(ratio):
    with pytest.raises(TubeGeometryApplicabilityError,match='Table 4'): htc(tube(ratio))


@pytest.mark.parametrize('re',[9999.,120001.,2300.])
def test_re_bounds_no_automatic_fallback(re):
    with pytest.raises(TubeGeometryApplicabilityError,match='Eq.18'): htc(re=re)


@pytest.mark.parametrize('beta,source_fanning',[(.67,.00917),(.5,.00908)])
def test_nikitin_eq3_5_against_poiseuille_identity_and_dns(beta,source_fanning):
    result=elliptical_turbulent_friction(tube(1/beta),6000.,assumptions=FD)
    from core.heat_transfer.elliptical_internal import elliptical_laminar_friction
    laminar=elliptical_laminar_friction(tube(1/beta),6000.,
        assumptions=EllipticalFlowAssumptions('laminar','fully_developed'))
    assert result.modified_reynolds==pytest.approx(6000*64/laminar.poiseuille_number,rel=1e-14)
    # Compare with Table 3 finest-grid DNS; not an exact constitutive law.
    assert result.f_fanning==pytest.approx(source_fanning,rel=.03)
    f=result.f_darcy
    assert 1/math.sqrt(f)==pytest.approx(2*math.log10(result.modified_reynolds*math.sqrt(f))-.8,abs=1e-10)
    assert result.f_fanning==f/4
    assert result.metadata.approximation and result.metadata.source_references
    assert result.warnings[0].code=='elliptical_turbulent_friction_engineering'


def test_actual_geometry_changes_friction_without_circular_dispatch(monkeypatch):
    import core.heat_transfer.internal_dispatch as d
    monkeypatch.setattr(d,'darcy_friction_factor',lambda *a: pytest.fail('circular friction'))
    monkeypatch.setattr(d,'heat_transfer_coefficient_internal_diagnostics',lambda **a: pytest.fail('circular HTC'))
    t=tube()
    mass=24000*PROPS.mu*t.flow_area/t.hydraulic_diameter
    r=internal_heat_transfer_diagnostics(tube=t,m_dot=mass,flow_area=t.flow_area,props=PROPS,assumptions=FD)
    f,model=internal_friction(tube=t,reynolds=r.Re,assumptions=FD)
    assert r==htc(t)
    assert model==FRICTION_MODEL_ID
    assert f>elliptical_turbulent_friction(tube(1),r.Re,assumptions=FD).f_darcy
    parallel=internal_heat_transfer_diagnostics(tube=t,m_dot=mass*24,flow_area=t.flow_area*24,props=PROPS,assumptions=FD)
    assert parallel.alfa_corrected==pytest.approx(r.alfa_corrected)


@pytest.mark.parametrize('kw',[{'regime':'laminar'},{'development':'developing'},
    {'constant_properties':False},{'newtonian':False},{'no_slip':False}])
def test_turbulent_assumptions_are_not_relaxed(kw):
    a=replace(FD,**kw)
    with pytest.raises(TubeGeometryApplicabilityError): htc(assumptions=a)
    with pytest.raises(TubeGeometryApplicabilityError): elliptical_turbulent_friction(tube(),24000.,assumptions=a)


@pytest.mark.parametrize('kw',[{'thermal_boundary_condition':'H1'},{'thermal_boundary_condition':'H2'},
                             {'axial_conduction':True},{'viscous_dissipation':True}])
def test_only_constant_wall_temperature_htc(kw):
    with pytest.raises(TubeGeometryApplicabilityError,match='constant wall'): htc(assumptions=replace(FD,**kw))


def test_smooth_guard_and_short_duct_warning():
    with pytest.raises(TubeGeometryApplicabilityError,match='smooth'): htc(replace(tube(),roughness_inner=1e-6))
    with pytest.raises(TubeGeometryApplicabilityError,match='smooth'): internal_friction(tube=tube(),reynolds=24000.,relative_roughness=.001,assumptions=FD)
    assert any(w.code=='elliptical_yang2018_short_duct' for w in htc(L_heated=.01).warnings)
