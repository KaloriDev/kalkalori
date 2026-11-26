# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic contracts and full-workflow tests; no private correlations."""
from dataclasses import replace
import math
import pytest

from core.common.warnings import make_warning
from core.geometry import TubeGeometryApplicabilityError
from core.heat_transfer.outside_provider import (
    OutsideProviderMetadata, OutsideApplicability, OutsideProviderPolicy,
    OutsideReferenceQuantities, OutsideThermalProviderResult,
    OutsideProviderResultError, OutsideProviderApplicabilityError,
)
from core.pressure_drop.outside_provider import OutsideBankPressureDropResult, OutsidePressureScope
from core.heat_transfer.outside_dispatch import evaluate_outside_thermal, evaluate_outside_hydraulics
from core.pressure_drop.flow_path import build_outside_pressure_drop_result, _tube_bank_core_stage
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.heat_transfer import ThinPlateFinEfficiency2D
from core.tests.zhao2017_integration_test import context
from core.tests.zhao2017_fixtures import source_bundle, AIR


META = OutsideProviderMetadata('synthetic', 'synthetic algebra; no experimental validity', 'face velocity; no passage area')
WARNING = make_warning(code='synthetic_provider_warning', message='Synthetic test warning', source='synthetic')
POLICY = OutsideProviderPolicy(True, True)


class Film:
    def __init__(self, h=70., metadata=META):
        self.h, self.metadata, self.requests = h, metadata, []

    def evaluate(self, request):
        self.requests.append(request)
        return OutsideThermalProviderResult(self.h, self.metadata, warnings=(WARNING,))


class Pressure:
    def __init__(self, scope=OutsidePressureScope.BANK_ONLY, value=37., metadata=META, contributions=()):
        self.scope, self.value, self.metadata = scope, value, metadata
        self.contributions, self.requests = contributions, []

    def evaluate(self, request):
        self.requests.append(request)
        return OutsideBankPressureDropResult(self.value, self.metadata, self.scope,
            'Synthetic declared inlet/outlet face stations', self.contributions)


def shifted():
    return replace(source_bundle(), layout='shifted', lattice=None,
                   pitch_transverse=.062, pitch_longitudinal=.030)


def test_native_geometry_and_film_reach_existing_fin_solver_in_simulation_and_rating():
    old, a, b = context()
    bundle = shifted()
    film, pressure = Film(), Pressure()
    hx = BareTubeHeatExchanger(bundle, outside_thermal_provider=film, outside_hydraulic_provider=pressure)
    result = hx.simulate(a,b)
    d = result.finned_tube_diagnostics
    assert d.outside_alpha_physical == film.h
    assert d.eta_fin == ThinPlateFinEfficiency2D().evaluate(bundle.tube,film.h).eta_fin
    assert 0 < d.outside_alpha_effective_gross < film.h
    assert result.UA == pytest.approx(d.UA)
    assert result.outside_dp_total == pressure.value
    assert d.heat_transfer_metadata == META and d.pressure_drop_metadata == META
    assert WARNING in result.warnings
    assert all(r.bundle is bundle for r in film.requests)
    assert all(r.inlet.bundle is bundle for r in pressure.requests)
    assert bundle.layout == 'shifted' and bundle.n_rows == 8
    assert all(r.T > 0 and r.p > 0 for r in film.requests)
    inside=BalanceSideSpec(provider=a.provider,m_dot=a.m_dot,T_in=a.T_in,T_out=result.T_out_inside,p=a.p)
    outside=BalanceSideSpec(provider=b.provider,m_dot=b.m_dot,T_in=b.T_in,T_out=result.T_out_outside,p=b.p)
    rating=hx.rate(inside,outside,include_simulation=True)
    assert rating.thermal_state.UA == pytest.approx(result.UA, rel=1e-7)
    assert rating.finned_tube_diagnostics.heat_transfer_metadata == META
    assert rating.outside_dp_total == pressure.value


def test_injection_is_independent_and_default_channels_keep_their_guards():
    bundle = source_bundle()
    thermal = evaluate_outside_thermal(bundle=bundle,m_dot=1.7,props=AIR,outside_thermal_provider=Film())
    default = evaluate_outside_hydraulics(bundle=bundle,m_dot=1.7,inlet_props=AIR)
    assert thermal.alpha == 70. and default.dp_total > 0
    assert evaluate_outside_hydraulics(bundle=bundle,m_dot=1.7,inlet_props=AIR,
        outside_hydraulic_provider=Pressure()).dp_total == 37.
    bad = shifted()
    with pytest.raises(TubeGeometryApplicabilityError):
        evaluate_outside_thermal(bundle=bad,m_dot=1.7,props=AIR)
    with pytest.raises(TubeGeometryApplicabilityError):
        evaluate_outside_hydraulics(bundle=bad,m_dot=1.7,inlet_props=AIR)
    for thermal_provider, hydraulic_provider in ((Film(),None),(None,Pressure())):
        old,a,b=context()
        with pytest.raises(TubeGeometryApplicabilityError):
            BareTubeHeatExchanger(bad,outside_thermal_provider=thermal_provider,
                outside_hydraulic_provider=hydraulic_provider).simulate(a,b)


@pytest.mark.parametrize('scope,contributions,adds_acceleration', [
    (OutsidePressureScope.BANK_ONLY,(),True),
    (OutsidePressureScope.LOSS_WITH_ENTRANCE_EXIT,('entrance','exit'),True),
    (OutsidePressureScope.COMPLETE_STATIC,('entrance','exit','acceleration'),False),
    (OutsidePressureScope.UNRESOLVED_ENGINEERING,(),False),
])
def test_pressure_scope_controls_acceleration_once(scope, contributions, adds_acceleration):
    meta = replace(META, applicability=OutsideApplicability.ENGINEERING_ESTIMATE, limitations=('Synthetic unclosed scope',)) if scope is OutsidePressureScope.UNRESOLVED_ENGINEERING else META
    pressure = Pressure(scope, metadata=meta, contributions=contributions)
    bank = evaluate_outside_hydraulics(bundle=shifted(),m_dot=1.7,inlet_props=AIR,
        midpoint_props=replace(AIR,rho=1.1),outlet_props=replace(AIR,rho=1.),
        outside_hydraulic_provider=pressure,outside_provider_policy=POLICY)
    acceleration=(1.7/shifted().frontal_flow_area)**2*(1.-1./AIR.rho)
    assert bank.dp_acceleration == pytest.approx(acceleration if adds_acceleration else 0.)
    assert bank.dp_total == pytest.approx(37.+bank.dp_acceleration)
    path = build_outside_pressure_drop_result(bank)
    assert path.dp_static_total == pytest.approx(bank.dp_total)
    if scope is OutsidePressureScope.BANK_ONLY:
        assert _tube_bank_core_stage(bank).dp_irreversible == 37.
    else:
        assert not path.validated_bank_irreversible and not path.groups
        assert path.pressure_basis == scope.value
        assert path.pressure_scope is scope
        assert path.provider_result is bank.provider_result
        with pytest.raises(ValueError, match='irreversible'):
            _tube_bank_core_stage(bank)


def test_unresolved_estimate_requires_both_authorizations_and_visible_limitations():
    meta=replace(META,applicability=OutsideApplicability.ENGINEERING_ESTIMATE,limitations=('Unresolved source taps',))
    provider=Pressure(OutsidePressureScope.UNRESOLVED_ENGINEERING,metadata=meta)
    for policy in (OutsideProviderPolicy(),OutsideProviderPolicy(True,False),OutsideProviderPolicy(False,True)):
        with pytest.raises(OutsideProviderApplicabilityError):
            evaluate_outside_hydraulics(bundle=shifted(),m_dot=1.,inlet_props=AIR,
                outside_hydraulic_provider=provider,outside_provider_policy=policy)
    r=evaluate_outside_hydraulics(bundle=shifted(),m_dot=1.,inlet_props=AIR,
        outside_hydraulic_provider=provider,outside_provider_policy=POLICY)
    assert {w.code for w in r.warnings} >= {'external_outside_engineering_estimate','external_outside_pressure_scope_unresolved'}


@pytest.mark.parametrize('h',[0.,-1.,math.inf,math.nan,True,'70'])
def test_invalid_film_is_rejected(h):
    with pytest.raises(OutsideProviderResultError):
        evaluate_outside_thermal(bundle=shifted(),m_dot=1.,props=AIR,outside_thermal_provider=Film(h))


@pytest.mark.parametrize('metadata',[
    replace(META,provenance=''), replace(META,reference_basis=''),
    replace(META,applicability='INTERPOLATED'),
    replace(META,applicability=OutsideApplicability.ENGINEERING_ESTIMATE),
])
def test_invalid_metadata_is_rejected(metadata):
    with pytest.raises(OutsideProviderResultError):
        evaluate_outside_thermal(bundle=shifted(),m_dot=1.,props=AIR,
            outside_thermal_provider=Film(metadata=metadata),outside_provider_policy=POLICY)


def test_unsupported_thermal_applicability_does_not_get_replaced_by_default():
    with pytest.raises(OutsideProviderApplicabilityError):
        evaluate_outside_thermal(bundle=source_bundle(),m_dot=1.,props=AIR,
            outside_thermal_provider=Film(metadata=replace(META,applicability=OutsideApplicability.UNSUPPORTED)))


@pytest.mark.parametrize('value',[math.inf,math.nan,-1.,True])
def test_invalid_bank_loss_rejected(value):
    with pytest.raises(OutsideProviderResultError):
        evaluate_outside_hydraulics(bundle=shifted(),m_dot=1.,inlet_props=AIR,
            outside_hydraulic_provider=Pressure(value=value))


def test_static_pressure_recovery_is_signed_and_not_augmented():
    bank=evaluate_outside_hydraulics(bundle=shifted(),m_dot=1.,inlet_props=AIR,
        outside_hydraulic_provider=Pressure(OutsidePressureScope.COMPLETE_STATIC,-3.))
    assert bank.dp_total == -3. and bank.dp_acceleration == 0.


def test_contradictory_pressure_contributions_are_rejected():
    for scope, contributions in ((OutsidePressureScope.BANK_ONLY,('entrance',)),
            (OutsidePressureScope.LOSS_WITH_ENTRANCE_EXIT,()),
            (OutsidePressureScope.LOSS_WITH_ENTRANCE_EXIT,('acceleration',))):
        with pytest.raises(OutsideProviderResultError):
            evaluate_outside_hydraulics(bundle=shifted(),m_dot=1.,inlet_props=AIR,
                outside_hydraulic_provider=Pressure(scope,contributions=contributions))


def test_no_additional_paths_for_complete_or_unresolved_pressure():
    from core.geometry.outside_pressure_drop_path import SpecifiedOutsidePressureDropPath
    from core.geometry.pressure_drop_stages import PressureDropAssemblyGeometry
    path=SpecifiedOutsidePressureDropPath(PressureDropAssemblyGeometry(()),PressureDropAssemblyGeometry(()))
    for scope,contributions in ((OutsidePressureScope.COMPLETE_STATIC,()),
            (OutsidePressureScope.LOSS_WITH_ENTRANCE_EXIT,('entrance',))):
        bank=evaluate_outside_hydraulics(bundle=shifted(),m_dot=1.,inlet_props=AIR,
            outside_hydraulic_provider=Pressure(scope,contributions=contributions))
        with pytest.raises(ValueError,match='Local irreversible paths'):
            build_outside_pressure_drop_result(bank,path=path)


def test_round_geometry_and_plain_ellipse_injection_keep_provenance():
    from core.geometry import BareTube,TubeBundle
    old,a,b=context()
    for bundle in (TubeBundle(BareTube(.020,.024,1.,1.,45.),8,10,.05,.05,'inline',2),
                   replace(old.bundle,tube=old.bundle.tube.core_tube)):
        hx=BareTubeHeatExchanger(bundle,outside_thermal_provider=Film(),outside_hydraulic_provider=Pressure())
        result=hx.simulate(a,b)
        assert result.thermal_state.diagnostics.outside_provider_result.metadata == META
        assert result.final_result.outside_side_thermal.provider_result.metadata == META
        assert result.outside_dp_total == 37.


def test_independent_hydraulic_applicability_and_provider_failure():
    provider=Pressure(metadata=replace(META,applicability=OutsideApplicability.UNSUPPORTED))
    with pytest.raises(OutsideProviderApplicabilityError):
        evaluate_outside_hydraulics(bundle=source_bundle(),m_dot=1.,inlet_props=AIR,
            outside_hydraulic_provider=provider)
    class Failed:
        def evaluate(self,request): raise RuntimeError('selected provider failed')
    with pytest.raises(RuntimeError,match='selected provider failed'):
        evaluate_outside_thermal(bundle=source_bundle(),m_dot=1.,props=AIR,outside_thermal_provider=Failed())


def test_wrong_results_and_invalid_reference_are_rejected():
    class Fixed:
        def __init__(self,value): self.value=value
        def evaluate(self,request): return self.value
    for result in (object(),OutsideThermalProviderResult(70.,META,area_basis='apparent'),
            OutsideThermalProviderResult(70.,META,reference=OutsideReferenceQuantities(velocity=-1.)),
            OutsideThermalProviderResult(70.,META,warnings=('untyped warning',))):
        with pytest.raises(OutsideProviderResultError):
            evaluate_outside_thermal(bundle=shifted(),m_dot=1.,props=AIR,outside_thermal_provider=Fixed(result))
    with pytest.raises(OutsideProviderResultError):
        evaluate_outside_hydraulics(bundle=shifted(),m_dot=1.,inlet_props=AIR,
            outside_hydraulic_provider=Fixed(object()))


def test_nonphysical_inputs_and_variable_dry_mass_are_rejected():
    from types import SimpleNamespace
    invalid_rho=SimpleNamespace(rho=-1.,mu=AIR.mu,k=AIR.k,cp=AIR.cp)
    invalid_k=SimpleNamespace(rho=AIR.rho,mu=AIR.mu,k=math.nan,cp=AIR.cp)
    for mass,props in ((0.,AIR),(1.,invalid_rho),(1.,invalid_k)):
        with pytest.raises(OutsideProviderResultError):
            evaluate_outside_thermal(bundle=shifted(),m_dot=mass,props=props,outside_thermal_provider=Film())
    with pytest.raises(OutsideProviderApplicabilityError,match='constant dry mass'):
        evaluate_outside_hydraulics(bundle=shifted(),m_dot=1.,inlet_props=AIR,m_dot_outlet=.9,
            outside_hydraulic_provider=Pressure())
