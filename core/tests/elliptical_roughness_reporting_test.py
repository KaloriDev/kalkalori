# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic geometry, state-reference and independent roughness contracts."""
from dataclasses import replace
import math
import pytest
from scipy.optimize import brentq

from core.geometry import TubeGeometryApplicabilityError
from core.geometry.reporting import rectangular_fin_geometry_metadata, minimum_free_flow_area_status
from core.heat_transfer import EllipticalRoughnessPolicy
from core.heat_transfer.internal_dispatch import internal_friction, internal_heat_transfer_diagnostics
from core.heat_transfer.elliptical_turbulent import (
    elliptical_turbulent_friction, ROUGH_FRICTION_MODEL_ID, FRICTION_MODEL_ID)
from core.heat_transfer.outside_provider import (
    OutsideReferenceQuantities, OutsideThermalProviderResult, OutsideProviderResultError)
from core.heat_transfer.reference_state import result_reference_states
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.pressure_drop.outside_provider import OutsideBankPressureDropResult
from core.tests.elliptical_turbulent_test import tube, FD, PROPS, htc
from core.tests.elliptical_turbulent_integration_test import context
from core.tests.external_outside_provider_test import Film, Pressure, shifted

POLICY = EllipticalRoughnessPolicy(True, True)


def cw(re, eps):
    # Independent bracketed root of Colebrook; production uses fixed-point x.
    return brentq(lambda f: 1/math.sqrt(f)+2*math.log10(eps/3.7+2.51/(re*math.sqrt(f))),
                  .001, .2, xtol=1e-15)


@pytest.mark.parametrize('ratio',[1.,1.5,3.,5.])
@pytest.mark.parametrize('re',[4000.,24000.,120000.])
def test_rough_friction_preserves_shape_smooth_limit_and_monotonicity(ratio,re):
    t=tube(ratio)
    smooth=elliptical_turbulent_friction(t,re,assumptions=FD)
    explicit=elliptical_turbulent_friction(replace(t,roughness_inner=0.),re,assumptions=FD,roughness_policy=POLICY)
    assert explicit == smooth
    assert smooth.metadata.model_id == FRICTION_MODEL_ID
    factors=[smooth.f_darcy]
    for relative in (.0001,.001,.005,.01):
        eps=relative*t.hydraulic_diameter
        r=elliptical_turbulent_friction(replace(t,roughness_inner=eps),re,assumptions=FD,roughness_policy=POLICY)
        assert r.relative_roughness == eps/t.hydraulic_diameter
        expected=smooth.f_darcy*cw(smooth.modified_reynolds,r.relative_roughness)/cw(smooth.modified_reynolds,0.)
        assert r.f_darcy == pytest.approx(expected,rel=2e-12)
        assert r.modified_reynolds == smooth.modified_reynolds
        assert r.f_fanning*4 == r.f_darcy
        assert r.metadata.model_id == ROUGH_FRICTION_MODEL_ID and r.metadata.approximation
        assert any(w.code=='elliptical_rough_friction_engineering' for w in r.warnings)
        factors.append(r.f_darcy)
    assert all(b>a for a,b in zip(factors,factors[1:]))
    tiny=elliptical_turbulent_friction(replace(t,roughness_inner=1e-14),re,assumptions=FD,roughness_policy=POLICY)
    assert tiny.f_darcy == pytest.approx(smooth.f_darcy,rel=1e-9)


def test_thermal_and_hydraulic_permissions_are_independent():
    t=replace(tube(),roughness_inner=20e-6)
    hydraulic=EllipticalRoughnessPolicy(allow_hydraulic_roughness=True)
    thermal=EllipticalRoughnessPolicy(allow_smooth_htc_on_rough_wall=True)
    assert internal_friction(tube=t,reynolds=24000.,roughness_policy=hydraulic)[0]>0
    with pytest.raises(TubeGeometryApplicabilityError,match='HTC'): htc(t,roughness_policy=hydraulic)
    r=htc(t,roughness_policy=thermal)
    assert r.alfa_corrected == htc().alfa_corrected
    assert any(w.code=='elliptical_rough_wall_smooth_htc_approximation' for w in r.warnings)
    with pytest.raises(TubeGeometryApplicabilityError,match='hydraulic'):
        internal_friction(tube=t,reynolds=24000.,roughness_policy=thermal)


@pytest.mark.parametrize('eps',[-1.,math.nan,math.inf])
def test_invalid_absolute_roughness(eps):
    with pytest.raises(ValueError): replace(tube(),roughness_inner=eps)


@pytest.mark.parametrize('relative',[-1.,math.nan,math.inf,True])
def test_invalid_relative_roughness(relative):
    with pytest.raises(ValueError):
        internal_friction(tube=tube(),reynolds=24000.,relative_roughness=relative,roughness_policy=POLICY)


def test_roughness_domain_geometry_consistency_and_regime_guards():
    t=tube()
    with pytest.raises(ValueError,match='match actual'):
        internal_friction(tube=t,reynolds=24000.,relative_roughness=.001,roughness_policy=POLICY)
    with pytest.raises(TubeGeometryApplicabilityError,match='0.01'):
        internal_friction(tube=replace(t,roughness_inner=.011*t.hydraulic_diameter),reynolds=24000.,roughness_policy=POLICY)
    for re in (1000.,3000.):
        with pytest.raises(TubeGeometryApplicabilityError):
            internal_friction(tube=replace(t,roughness_inner=5e-6),reynolds=re,roughness_policy=POLICY)
    with pytest.raises(TypeError): EllipticalRoughnessPolicy(1,False)


def test_intrinsic_geometry_has_physical_fin_area_and_explicit_unavailable_throat(monkeypatch):
    b=shifted();t=b.tube
    import core.heat_transfer.finned_ellipse_airside as z
    monkeypatch.setattr(z,'zhao2017_flow',lambda *a: pytest.fail('SHIFTED Zhao throat'))
    g=rectangular_fin_geometry_metadata(t,n_tubes=b.n_tubes_total)
    assert g['fin_pitch_mm']==t.fin.axial_pitch*1000
    assert g['fin_density_1_m']==1/t.fin.axial_pitch
    assert g['fin_k_W_mK']==t.fin.thermal_conductivity
    assert g['fin_thickness_root_mm']==g['fin_thickness_tip_mm']==1000*t.fin.thickness
    assert g['fin_count_per_tube']==t.length_effective/t.fin.axial_pitch
    assert g['A_fin_m2']==b.total_fin_area
    assert g['A_fin_m2']==pytest.approx(g['A_fin_faces_m2']+g['A_fin_outer_edge_m2'])
    assert g['A_outside_gross_m2']==pytest.approx(g['A_primary_outside_m2']+g['A_fin_m2'])
    assert t.effective_outside_area(1.)==t.area_primary_outside+t.area_fin_faces
    assert t.effective_outside_area(1.)<t.area_outer
    unavailable=minimum_free_flow_area_status(b)
    assert unavailable.area is None and 'SHIFTED' in unavailable.reason


class ReferenceFilm(Film):
    def evaluate(self, request):
        ref=OutsideReferenceQuantities(velocity=request.face_velocity,characteristic_length=.02,
            reynolds_number=request.face_mass_flux*.02/request.mu,
            flow_area=request.frontal_area,flow_area_basis='frontal_area',mass_flux=request.face_mass_flux)
        return OutsideThermalProviderResult(self.h,self.metadata,reference=ref)


class ReferencePressure(Pressure):
    def evaluate(self, request):
        q=request.midpoint
        ref=OutsideReferenceQuantities(velocity=2*q.face_velocity,characteristic_length=.03,
            reynolds_number=2*q.face_mass_flux*.03/q.mu,
            flow_area=q.frontal_area/2,flow_area_basis='synthetic_half_face',mass_flux=2*q.face_mass_flux)
        return OutsideBankPressureDropResult(self.value,self.metadata,self.scope,'Synthetic scope',reference=ref)


def test_separate_actual_channel_states_and_geometry_independent_of_provider():
    old,a,b=context();bundle=shifted()
    h=ReferenceFilm();p=ReferencePressure()
    hx=BareTubeHeatExchanger(bundle,outside_thermal_provider=h,outside_hydraulic_provider=p)
    sim=hx.simulate(a,b)
    rating=hx.rate(BalanceSideSpec(provider=a.provider,m_dot=a.m_dot,T_in=a.T_in,T_out=sim.T_out_inside,p=a.p),
        BalanceSideSpec(provider=b.provider,m_dot=b.m_dot,T_in=b.T_in,T_out=sim.T_out_outside,p=b.p),include_simulation=False)
    g=rectangular_fin_geometry_metadata(bundle.tube,n_tubes=bundle.n_tubes_total)
    for result in (sim,rating):
        states=result_reference_states(result,bundle)
        it,ih,ot,oh=(states[k] for k in ('inside_thermal','inside_hydraulic','outside_thermal','outside_hydraulic'))
        assert it.reynolds==pytest.approx(it.mass_flux*it.diameter/result.thermal_state.inside_bulk_props.mu)
        assert ih.reynolds==result.final_result.tube_bundle_hydraulic.midpoint.reynolds
        assert ot.flow_area==bundle.frontal_flow_area and ot.area_basis=='frontal_area'
        assert oh.flow_area==bundle.frontal_flow_area/2 and oh.area_basis=='synthetic_half_face'
        assert oh.reynolds==pytest.approx(3*ot.reynolds)
        assert ot.temperature==result.thermal_state.outside_bulk_temperature
        assert oh.temperature==result.final_result.outside_tube_bank_hydraulic.midpoint.temperature
    second=BareTubeHeatExchanger(bundle,outside_thermal_provider=Film(55.),outside_hydraulic_provider=Pressure(value=42.))
    second.simulate(a,b)
    assert g==rectangular_fin_geometry_metadata(second.bundle.tube,n_tubes=bundle.n_tubes_total)


def test_roughness_reaches_all_hydraulic_points_without_changing_thermal_physics():
    old,a,b=context();smooth=old.simulate(a,b)
    tube=replace(old.bundle.tube,core_tube=replace(old.bundle.tube.core_tube,roughness_inner=20e-6))
    hx=BareTubeHeatExchanger(replace(old.bundle,tube=tube),elliptical_roughness_policy=POLICY)
    rough=hx.simulate(a,b)
    assert rough.UA==smooth.UA and rough.q==smooth.q
    sh=smooth.final_result.tube_bundle_hydraulic;rh=rough.final_result.tube_bundle_hydraulic
    assert rh.roughness_inner==tube.core_tube.roughness_inner
    assert rh.relative_roughness_inner==tube.core_tube.roughness_inner/tube.hydraulic_diameter
    assert rh.dp_friction>sh.dp_friction
    assert rh.dp_acceleration==sh.dp_acceleration
    assert rh.dp_tube_entrances==rh.dp_tube_exits==0
    for point in (rh.inlet,rh.midpoint,rh.outlet):
        assert point.friction_diagnostics.metadata.model_id==ROUGH_FRICTION_MODEL_ID
    codes={w.code for w in rough.warnings}
    assert {'elliptical_rough_friction_engineering','elliptical_rough_wall_smooth_htc_approximation','tube_end_losses_not_evaluated'}<=codes


@pytest.mark.parametrize('area,basis',[(-1.,'frontal_area'),(math.nan,'frontal_area'),(1.,''),(1.,None)])
def test_provider_reference_area_requires_a_finite_value_and_explicit_basis(area,basis):
    from core.heat_transfer.outside_provider import evaluate_external_thermal,make_outside_request
    from core.tests.zhao2017_fixtures import AIR
    film=ReferenceFilm()
    film.evaluate=lambda request: OutsideThermalProviderResult(70.,film.metadata,
        reference=OutsideReferenceQuantities(flow_area=area,flow_area_basis=basis))
    with pytest.raises(OutsideProviderResultError):
        evaluate_external_thermal(film,make_outside_request(bundle=shifted(),m_dot=1.,props=AIR))


def test_missing_provider_reference_area_is_not_inferred_from_face_geometry():
    old,a,b=context();bundle=shifted()
    hx=BareTubeHeatExchanger(bundle,outside_thermal_provider=Film(),outside_hydraulic_provider=Pressure())
    result=hx.simulate(a,b)
    states=result_reference_states(result,bundle)
    for channel in ('outside_thermal','outside_hydraulic'):
        assert states[channel].flow_area is None
        assert states[channel].reynolds is None
