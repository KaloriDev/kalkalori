# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace
import pytest

from core.heat_transfer.airside_correction import AirsideCorrection, MultiplicativeAirsideCorrection
from core.heat_transfer.finned_ellipse_airside import Zhao2017AirsideModel
from core.heat_transfer.outside_dispatch import calculate_resistance_network
from core.tests.zhao2017_fixtures import source_bundle, AIR


class Correction:
    def __init__(self,**values): self.values=values
    def correct(self,base_result,*,bundle,props):
        assert base_result.h_film==base_result.h_film_base
        return AirsideCorrection(provenance='synthetic_test',**self.values)


def evaluate(**values):
    return Zhao2017AirsideModel(correction_provider=Correction(**values)).evaluate(source_bundle(),1.7,AIR)


def test_identity_and_independent_channels():
    b=Zhao2017AirsideModel().evaluate(source_bundle(),1.7,AIR)
    identity=evaluate()
    assert identity.h_film==b.h_film and identity.pressure_drop==b.pressure_drop
    for args,expected_h,expected_dp in [
        ({'h_film_multiplier':2.},2*b.h_film,b.pressure_drop),
        ({'h_film_replacement':123.},123.,b.pressure_drop),
        ({'pressure_drop_multiplier':3.},b.h_film,3*b.pressure_drop),
        ({'pressure_drop_replacement':234.},b.h_film,234.)]:
        r=evaluate(**args)
        assert r.h_film==expected_h and r.pressure_drop==expected_dp
        assert r.h_film_base==b.h_film and r.pressure_drop_base==b.pressure_drop
        assert r.correction_provenance=='synthetic_test'


def test_correction_precedes_fin_efficiency_and_no_second_eta():
    b=source_bundle()
    a=Zhao2017AirsideModel().evaluate(b,1.7,AIR)
    c=evaluate(h_film_multiplier=2.)
    def network(r):
        return calculate_resistance_network(bundle=b,alpha_inside=500.,
            outside_alpha_physical=r.h_film,resistance_core_wall=.001)
    n,m=network(a),network(c)
    assert m.outside_alpha_physical==2*n.outside_alpha_physical
    assert m.fin_efficiency<n.fin_efficiency
    assert m.conductance_fin_outside+m.conductance_primary_outside==pytest.approx(c.h_film*m.A_effective)
    assert n.A_physical==m.A_physical
    high_k=replace(b,tube=replace(b.tube,fin=replace(b.tube.fin,thermal_conductivity=200.)))
    assert Zhao2017AirsideModel().evaluate(high_k,1.7,AIR).h_film==a.h_film


@pytest.mark.parametrize('values',[
    {'h_film_multiplier':0}, {'pressure_drop_replacement':float('nan')},
    {'h_film_multiplier':2,'h_film_replacement':2}, {'provenance':''}])
def test_invalid_correction_rejects(values):
    with pytest.raises(ValueError): AirsideCorrection(**values)


def test_provider_errors_and_wrong_return_do_not_fallback():
    class Bad:
        def correct(self,*args,**kwargs): raise RuntimeError('external failure')
    with pytest.raises(RuntimeError,match='external failure'):
        Zhao2017AirsideModel(correction_provider=Bad()).evaluate(source_bundle(),1.7,AIR)
    class Wrong:
        def correct(self,*args,**kwargs): return None
    with pytest.raises(TypeError,match='AirsideCorrection'):
        Zhao2017AirsideModel(correction_provider=Wrong()).evaluate(source_bundle(),1.7,AIR)


@pytest.mark.parametrize('ht,dp',[(1.,1.),(1.08,1.),(1.,1.20),(1.08,1.20)])
def test_generic_multiplicative_provider_independent_channels_and_eta(ht,dp):
    b=source_bundle()
    b=replace(b,tube=replace(b.tube,fin=replace(b.tube.fin,fin_turbulator_count=10)))
    base=Zhao2017AirsideModel().evaluate(b,1.7,AIR)
    provider=MultiplicativeAirsideCorrection(ht,dp,provenance='synthetic_engineering_input')
    r=Zhao2017AirsideModel(correction_provider=provider).evaluate(b,1.7,AIR)
    assert r.h_film_base==base.h_film and r.pressure_drop_base==base.pressure_drop
    assert r.h_film==ht*base.h_film and r.pressure_drop==dp*base.pressure_drop
    assert r.correction_applied==(ht!=1. or dp!=1.)
    assert r.correction_provenance=='synthetic_engineering_input'
    n=calculate_resistance_network(bundle=b,alpha_inside=500.,
        outside_alpha_physical=r.h_film,resistance_core_wall=.001)
    old=calculate_resistance_network(bundle=b,alpha_inside=500.,
        outside_alpha_physical=base.h_film,resistance_core_wall=.001)
    assert (n.fin_efficiency==old.fin_efficiency) if ht==1. else (n.fin_efficiency<old.fin_efficiency)
    assert n.outside_alpha_physical==r.h_film
    assert n.conductance_primary_outside+n.conductance_fin_outside==pytest.approx(r.h_film*n.A_effective)


@pytest.mark.parametrize('field',['h_film_multiplier','pressure_drop_multiplier'])
@pytest.mark.parametrize('value',[None,0.,-1.,True,float('nan'),float('inf'),'1.08'])
def test_generic_multiplicative_provider_rejects_invalid_factors(field,value):
    with pytest.raises((ValueError,TypeError)):
        MultiplicativeAirsideCorrection(**{field:value})
