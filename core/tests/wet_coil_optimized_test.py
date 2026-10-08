# SPDX-License-Identifier: GPL-3.0-only
"""Production optimized/native equivalence, safeguards and structural performance."""
import gc
import weakref
import numpy as np
import pytest
from core import WetCoilSolverOptions,WetCoilTimeoutError
from core.phase_change.wet_coil_integration import forward_wet_process
from core.heat_transfer._wet_thermodynamics import _OptimizedWetThermodynamics as WetThermodynamics
from core.heat_transfer._wet_if97 import IF97WaterCache,supported
from core.heat_transfer.wet_coil_adapters import WetGasThermodynamics
from core.heat_transfer.wet_coil_solver import _WetSolveBudget
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.wet_coil_integration import _forward_wet_process_native as native
from core.properties.water import water_saturation_pressure,water_saturation_liquid_enthalpy,water_saturation_vapor_enthalpy
from core.tests.wet_coil_public_test import context


def test_shared_if97_matches_installed_values_and_derivatives_with_eviction():
    cache = IF97WaterCache({},capacity=8)
    temperatures = np.linspace(273.16,623.149,81)
    for T in temperatures:
        bundle = cache.get(float(T),values=True)
        assert bundle.pressure == water_saturation_pressure(float(T))
        assert bundle.liquid == water_saturation_liquid_enthalpy(T=float(T))
        assert bundle.vapor == water_saturation_vapor_enthalpy(T=float(T))
    assert len(cache.terms)==8
    for T in reversed(temperatures):
        bundle = cache.get(float(T),values=True)
        before = bundle.liquid,bundle.vapor
        dp,dh = cache.derivatives(bundle)
        assert (bundle.liquid,bundle.vapor)==before
    assert cache.stats['if97_term_rebuilds']>0
    assert len(cache.terms)<=8


@pytest.mark.parametrize('T',(float(np.nextafter(623.15,-np.inf)),623.15,630.,640.))
def test_region_boundary_and_region3_keep_native_value_path(T):
    _,_,outside = context()
    cap = detect_phase_change_capability(outside.provider)
    th = WetThermodynamics(outside.p,cap)
    ref = WetGasThermodynamics(outside.p,cap,_reuse_inverse_state=True)
    assert not supported(T)
    assert th.enthalpy(T,.012)==ref.enthalpy(T,.012)
    assert th.condensate_enthalpy(T)==ref.condensate_enthalpy(T)
    assert th.saturation_humidity(T)==ref.saturation_humidity(T)
    assert th.statistics['native_water_fallbacks']>0


@pytest.mark.parametrize('W,regime',[(.004,'DRY'),(.007,'PARTIALLY_WET'),(.016,'FULLY_WET'),
    (.0068219864228406-1e-6,'DRY'),(.0068219864228406+1e-6,'PARTIALLY_WET'),
    (.0077057271400947-1e-6,'PARTIALLY_WET'),(.0077057271400947+1e-6,'FULLY_WET')])
def test_strict_regimes_cold_seeded_and_final_gates(W,regime):
    hx,a,b = context(W=W)
    options = WetCoilSolverOptions(timeout_s=None)
    oracle = native(hx,a,b,wet_solver_options=options)[0]
    cold = forward_wet_process(hx,a,b,wet_solver_options=options)[0]
    seeded = forward_wet_process(hx,a,b,wet_solver_options=options,_initial_state=(cold,[0.]))[0]
    for result in (cold,seeded):
        assert result.regime==oracle.regime==regime
        assert result.heat_liquid==pytest.approx(oracle.heat_liquid,abs=.03,rel=0)
        assert result.air_out==pytest.approx(oracle.air_out,abs=1e-5,rel=0)
        assert result.liquid_out==pytest.approx(oracle.liquid_out,abs=1e-5,rel=0)
        assert result.humidity_out==pytest.approx(oracle.humidity_out,abs=3e-8,rel=0)
        assert result.wet_fraction==pytest.approx(oracle.wet_fraction,abs=2e-6,rel=0)
        d = result.diagnostics
        assert d['solver_options'] is options
        assert abs(d['energy_residual'])<=options.energy_tolerance_W
        assert abs(d['mass_residual'])<=options.mass_tolerance_kg_s
        if result.profile:
            assert d['min_wet_driving_force']>=-1e-9
            assert d['min_vapor_margin']>=-1e-9
            assert min(p.condensate_density for p in result.profile)>=-1e-11


def test_initial_guess_failure_falls_back_to_strict(monkeypatch):
    from core.heat_transfer import wet_coil_adapters as kernel
    hx,a,b = context(W=.007)
    original = kernel.solve_production_coil
    def fail_predictor(*args,**kwargs):
        if getattr(kwargs['thermodynamics'],'fidelity',None) is not None:
            raise ValueError('injected predictor-only failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(kernel,'solve_production_coil',fail_predictor)
    result = forward_wet_process(hx,a,b,wet_solver_options=WetCoilSolverOptions(timeout_s=None))[0]
    assert result.diagnostics['solver_statistics']['cold_predictor_fallbacks']==1
    oracle = native(hx,a,b,wet_solver_options=WetCoilSolverOptions(timeout_s=None))[0]
    assert result.heat_liquid==pytest.approx(oracle.heat_liquid,abs=.03,rel=0)
    assert result.regime==oracle.regime


def test_operation_caches_are_released():
    hx,a,b = context(W=.007)
    objects = forward_wet_process(hx,a,b,wet_solver_options=WetCoilSolverOptions(timeout_s=None))
    thermo = objects[1]
    assert len(thermo.water.terms)<=512
    references = [weakref.ref(thermo),*[weakref.ref(t) for t in thermo.children]]
    del thermo,objects
    gc.collect()
    assert all(ref() is None for ref in references)


def test_deadline_and_tighter_user_controls():
    hx,a,b = context(W=.007)
    expired = _WetSolveBudget(WetCoilSolverOptions(timeout_s=.001))
    expired.deadline=expired.started-1
    with pytest.raises(WetCoilTimeoutError):
        forward_wet_process(hx,a,b,wet_solver_options=expired.options,_budget=expired)
    controls = WetCoilSolverOptions(energy_tolerance_W=2e-4,mass_tolerance_kg_s=2e-10,
        outlet_temperature_tolerance_K=2e-7,timeout_s=None)
    oracle = native(hx,a,b,wet_solver_options=controls)[0]
    result = forward_wet_process(hx,a,b,wet_solver_options=controls)[0]
    assert result.diagnostics['solver_options'] is controls
    assert abs(result.diagnostics['energy_residual'])<=controls.energy_tolerance_W
    assert abs(result.diagnostics['mass_residual'])<=controls.mass_tolerance_kg_s
    assert result.heat_liquid==pytest.approx(oracle.heat_liquid,abs=1e-5,rel=0)


@pytest.mark.parametrize('T',(280.,300.,350.,450.,550.,610.))
def test_shared_derivatives_differentiate_the_native_definitions(T):
    cache = IF97WaterCache({})
    dp, dh = cache.derivatives(cache.get(T, values=True))
    def derivative(function):
        step = .01
        return (8 * (function(T + step) - function(T - step))
                - function(T + 2 * step) + function(T - 2 * step)) / (12 * step)
    assert dp == pytest.approx(derivative(water_saturation_pressure), rel=2e-7)
    assert dh == pytest.approx(derivative(lambda t: water_saturation_vapor_enthalpy(T=t)), rel=2e-7)


def test_cold_initialization_reduces_strict_work_and_seeded_skips_it():
    hx,a,b = context(W=.007)
    controls = WetCoilSolverOptions(timeout_s=None)
    result,thermo,_ = forward_wet_process(hx,a,b,wet_solver_options=controls)
    d = result.diagnostics['solver_statistics']
    assert d['cold_predictor_successes']==1
    assert d['point_bundle_reuses'] > 0
    assert d['outlet_point_evaluations_deferred'] > 0
    assert thermo.statistics['humidity_calls']==0
    assert thermo.statistics['analytic_vapor_derivatives'] > 0
    assert d.get('native_derivative_fallbacks',0)==0
    oracle=native(hx,a,b,wet_solver_options=controls)[0]
    assert d['kernel_profile_points'] < oracle.diagnostics['solver_statistics']['kernel_profile_points']
    warm,_,_ = forward_wet_process(hx,a,b,wet_solver_options=controls,_initial_state=(result,[0.]))
    assert warm.diagnostics['solver_statistics'].get('cold_predictor_attempts',0)==0
    assert warm.regime==result.regime


def test_finned_forward_keeps_native_numerics(monkeypatch):
    from core.heat_transfer import _wet_thermodynamics as props
    hx,a,b = context(finned=True,W=.004)
    def forbidden(*args,**kwargs):
        raise AssertionError('Finned must not construct optimized thermodynamics')
    monkeypatch.setattr(props,'_OptimizedWetThermodynamics',forbidden)
    result,thermo,_ = forward_wet_process(hx,a,b,wet_solver_options=WetCoilSolverOptions(timeout_s=None))
    assert type(thermo) is WetGasThermodynamics
    assert result.diagnostics['numerical_path']=='native'


def test_gauss_rules_are_exact_cached_and_read_only():
    from core.heat_transfer._wet_numerics import gauss
    from numpy.polynomial.legendre import leggauss
    nodes,weights = gauss(10)
    native_nodes,native_weights = leggauss(10)
    assert np.array_equal(nodes,native_nodes) and np.array_equal(weights,native_weights)
    assert gauss(10)[0] is nodes
    with pytest.raises(ValueError):
        nodes[0] = 0.
