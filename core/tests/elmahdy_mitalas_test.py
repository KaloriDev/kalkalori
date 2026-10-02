# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic source comparison with explicit, controlled property conventions."""
from dataclasses import replace
import json
from math import exp, log, pi, sqrt
from pathlib import Path

import pytest

from core.heat_transfer.elmahdy_mitalas import CoilInput, solve_source_coil, _counterflow_transfer


def _pressure(t):
    # ASHRAE Hyland-Wexler saturation pressure over liquid water, Pa.
    k = t + 273.15
    if t < .01:
        return exp(-5674.5359/k + 6.3925247 - .009677843*k + .00000062215701*k*k
                   + 2.0747825e-9*k**3 - 9.484024e-13*k**4 + 4.1635019*log(k))
    return exp(-5800.2206/k + 1.3914993 - .048640239*k + .000041764768*k*k
               - .000000014452093*k**3 + 6.5459673*log(k))


def _inverse(fn, value):
    lo, hi = -80., 90.
    assert fn(lo) <= value <= fn(hi)
    for _ in range(80):
        mid = (lo+hi)/2
        if fn(mid) < value:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def source_case(w, *, upstream_fins=False):
    """Boundary data for the native routine, not a production geometry adapter.

    Initialized air geometry coefficients and dry-fin polynomial are supplied
    directly to BOTH engines, so this verifies CalcDetailFlatFinCoolingCoil,
    not EnergyPlus InitWaterCoil or its dry-fin polynomial fitting procedure.
    Liquid rho=998.2 and cp=4180 are explicit constant property injections.
    """
    pressure=101325.
    def wsat(t):
        pv=_pressure(t)
        return max(1e-5,.62198*pv/max(pressure-pv,1000.))
    def h(t,w):
        return 1004.84*t+max(1e-5,w)*(2500940.+1858.95*t)
    def hs(t):
        return h(t,wsat(t))
    cp=1004.84+max(w,1e-5)*1858.95
    mass_flux=(1+w)*.5/.4
    reynolds=.012*mass_flux/1.846e-5
    c1=.1169302399804487 if upstream_fins else .15
    c2=-.3152533689378356 if upstream_fins else -.3
    alpha=1.23*c1*reynolds**c2*cp*mass_flux
    alphaw=alpha*(1.425-.00051*reynolds+2.63e-7*reynolds**2)
    phi=.01*sqrt(2*alpha/(200*.00015))
    coefficients = (1.00146300602055,.003803808689703925,-.6722531935051792,
                    .4001748735776073,-.07269230047516913) if upstream_fins else (.98,-.3,.02,0.,0.)
    eta_d=sum(c*phi**i for i,c in enumerate(coefficients))
    # Source's initial raised-water point is 7.3 C for this case.
    raised=7.3
    ratio=(1004.84+1858.95e-5)*(27.-raised)/(h(27.,w)-hs(raised))
    dw=min(1.,max(1e-5,abs(w-wsat(raised))))
    phi_w=min(1.,max(1e-5,.01*sqrt(2*alphaw/(200*.00015))))
    eta_w=exp(-.41718)*abs(ratio)**.09471*dw**.0108*phi_w**(-.50303)
    if eta_w>1:
        eta_w=.99
    if eta_w<0:
        eta_w=.001
    dry_r=1/(20*alpha*(1+.9*(eta_d-1)))
    wet_r=cp/(20*alphaw*(1+.9*(eta_w-1)))
    velocity=.4*4/(10*998.2*pi*.01**2)
    water_r=.01**.2/(2*1429*velocity**.8)
    metal_fouling_r=(.001/400+5e-5)/2
    return CoilInput(27.,7.,w,.5,.4*4180,cp,h(27.,w),_inverse(_pressure,pressure*w/(.62198+w)),
                     dry_r,wet_r,lambda t:water_r/(1+.0146*t)+metal_fouling_r,
                     hs,lambda enthalpy:_inverse(hs,enthalpy),
                     lambda t,enthalpy:(enthalpy-1004.84*t)/(2500940.+1858.95*t))


def test_equal_capacity_and_zero_area_limits():
    assert _counterflow_transfer(0,10,10)==0
    assert _counterflow_transfer(20,10,10)==pytest.approx(20/3)
    assert _counterflow_transfer(20,10,10+1e-9)==pytest.approx(20/3)


@pytest.mark.parametrize("humidity", [.004,.008,.0095,.01,.016])
def test_source_energy_and_mass_identity(humidity):
    x=source_case(humidity)
    r=solve_source_coil(x)
    assert r.heat==pytest.approx(x.liquid_capacity*(r.liquid_out-x.liquid_in),abs=1e-7)
    assert r.heat==pytest.approx(x.dry_mass_flow*(x.gas_h_in-r.outlet_enthalpy),abs=1e-7)
    assert r.condensate==pytest.approx(x.dry_mass_flow*(x.humidity_in-r.humidity_out),abs=1e-12)
    assert 0<=r.wet_fraction<=1
    assert x.liquid_in<r.liquid_out<x.air_in
    assert r.energy_convention=="source_no_drain_enthalpy"
    if 0<r.wet_fraction<1:
        assert r.interface_surface==pytest.approx(x.dewpoint,abs=1e-7)


def test_native_reference_cases():
    data=json.loads((Path(__file__).parent/'fixtures/elmahdy_mitalas_native.json').read_text())
    fractions=[]
    for native in data["cases"]:
        assert native["warnings"]==0
        r=solve_source_coil(source_case(native["W_in"],upstream_fins=native.get("input_family")=="native_fins"))
        assert r.heat==pytest.approx(native["Q_W"],rel=.002)
        assert r.air_out==pytest.approx(native["air_out_C"],abs=.08)
        assert r.liquid_out==pytest.approx(native["water_out_C"],abs=.08)
        assert r.humidity_out==pytest.approx(native["W_out"],abs=4e-5)
        assert r.condensate==pytest.approx(native["condensate_kg_s"],abs=2e-5)
        assert r.wet_fraction==pytest.approx(native["wet_fraction"],abs=.002)
        assert r.interface_air==pytest.approx(native["interface_air_C"],abs=.08)
        if 0<native["wet_fraction"]<1:
            assert r.interface_surface==pytest.approx(native["interface_surface_C"],abs=.08)
            assert r.interface_liquid==pytest.approx(native["interface_water_C"],abs=.08)
        if native["wet_fraction"]>0:
            assert r.cold_surface==pytest.approx(native["surface_cold_C"],abs=.08)
        if native["wet_fraction"]==1:
            assert r.hot_surface==pytest.approx(native["surface_hot_trial_C"],abs=.08)
        fractions.append(native["wet_fraction"])
    assert any(f==0 for f in fractions)
    assert any(f==1 for f in fractions)
    assert any(0<f<1 for f in fractions)


def test_invalid_properties_fail_without_fallback():
    with pytest.raises(ValueError,match="Inner resistance"):
        solve_source_coil(replace(source_case(.01),inner_resistance=lambda t:-1))


@pytest.mark.parametrize("humidity", [.00665,.0067,.00675])
@pytest.mark.parametrize("upstream_fins", [False,True])
def test_source_near_onset_humidification_is_rejected(humidity,upstream_fins):
    # The native routine also predicts Wout>Win here. No clipping, fallback
    # or claim of source-model onset acceptance is permitted.
    with pytest.raises(ValueError,match="invalid humidity"):
        solve_source_coil(source_case(humidity,upstream_fins=upstream_fins))


def test_general_solution_full_wet_bound():
    # Native routine reaches f=1 through its partial-wet iteration at .0095.
    r=solve_source_coil(source_case(.0095))
    assert r.wet_fraction==1
    assert r.heat==pytest.approx(7515.597440675073,rel=.002)
    assert r.air_out==pytest.approx(15.1425106349109,abs=.08)
    assert r.liquid_out==pytest.approx(11.4949745458583,abs=.08)
    assert r.humidity_out==pytest.approx(.008350609245642803,abs=4e-5)


def test_native_onset_limitation_is_preserved_as_evidence():
    data=json.loads((Path(__file__).parent/'fixtures/elmahdy_mitalas_native.json').read_text())
    for native in data["near_onset_limitations"]:
        assert native["W_out"]>native["W_in"]
        assert native["condensate_kg_s"]<0
        assert native["wet_fraction"]>0
        assert native["warnings"]==0


def test_counterflow_allows_outlet_temperature_cross():
    x=replace(source_case(.004),liquid_capacity=100.)
    r=solve_source_coil(x)
    assert x.liquid_in<r.air_out<r.liquid_out<x.air_in
