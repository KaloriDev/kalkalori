# SPDX-License-Identifier: GPL-3.0-only
"""Public staged Rating keeps strict output, provider semantics and hydraulics."""
import pytest

from core import WetCoilSolverOptions, WetCoilTimeoutError
from core.heat_transfer._wet_numerics import _WetNumerics
from core.heat_transfer.wet_coil import WetCoilModelError
from core.heat_transfer import wet_coil_adapters, wet_coil_solver
from core.phase_change import wet_coil_integration as integration, _wet_rating
from core.tests.wet_coil_public_test import context, specs


@pytest.mark.parametrize('W,regime',[(.004,'DRY'),(.007,'PARTIALLY_WET'),(.016,'FULLY_WET')])
def test_public_rating_returns_only_strict_state_with_installed_hydraulics(W,regime,monkeypatch):
    hx,a,b = context(W=W)
    controls = WetCoilSolverOptions(timeout_s=None)
    target = hx.simulate(a,b,surface_margin=.1,wet_solver_options=controls)
    strict = []
    objects = []
    original = integration.forward_wet_process
    def checked(*args,**kwargs):
        assert kwargs['wet_solver_options'] is controls
        th = kwargs['_thermodynamics']
        assert th.fidelity is None and th.options==_WetNumerics()
        for child in th.children:
            assert child.fidelity is not None
            assert child.states is not th.states
            assert child._temperature_inverse_roots is not th._temperature_inverse_roots
        result = original(*args,**kwargs)
        strict.append(result[0])
        objects.append(th)
        return result
    monkeypatch.setattr(integration,'forward_wet_process',checked)
    rating = hx.rate(*specs(a,b,target.T_out_outside),wet_solver_options=controls)
    assert strict and rating.Q_required==strict[-1].heat_liquid
    assert rating.closed_balance.outside.T_out==strict[-1].air_out
    assert rating.outside_phase_change.regime==regime
    assert rating.A_required==pytest.approx(hx.bundle.total_outer_area/1.1,rel=.003)
    assert rating.final_result.A_o==hx.bundle.total_outer_area
    d = rating.wet_coil_diagnostics
    assert abs(d['independent_energy_residual'])<=controls.energy_tolerance_W
    assert abs(d['mass_residual'])<=controls.mass_tolerance_kg_s
    assert abs(d['required_area_temperature_residual'])<=controls.outlet_temperature_tolerance_K
    assert d['hydraulic_effective_length']==hx.bundle.tube.length_effective
    assert d['hydraulic_inner_flow_area']==hx.bundle.internal_flow_area_per_pass
    n=d['numerical_rating']
    assert n['final_grid']==10 and not n['approximate_inverse_caches_shared_with_final']
    assert n['statistics']['coarse_forward_solves'] > 0
    assert n['statistics']['medium_forward_solves'] > 0
    assert n['statistics']['strict_forward_solves'] > 0
    assert d['provider']['model_id']=='elmahdy_mitalas_energyplus_v25_2_adapted'


def test_joint_rating_and_included_simulation_share_operation_budget(monkeypatch):
    hx,a,b = context(W=.007)
    controls = WetCoilSolverOptions(energy_tolerance_W=2e-4,mass_tolerance_kg_s=2e-10,
                                  outlet_temperature_tolerance_K=2e-7,timeout_s=None)
    target = hx.simulate(a,b,surface_margin=.1,wet_solver_options=controls)
    budgets=[]
    original=integration.forward_wet_process
    def checked(*args,**kwargs):
        budgets.append(kwargs['_budget'])
        assert kwargs['wet_solver_options'] is controls
        return original(*args,**kwargs)
    monkeypatch.setattr(integration,'forward_wet_process',checked)
    rating=hx.rate(*specs(a,b,target.T_out_outside,target.T_out_inside,False),
                   include_simulation=True,wet_solver_options=controls)
    assert budgets and all(b is budgets[0] for b in budgets)
    assert rating.simulation.wet_coil_diagnostics['solver_options'] is controls
    assert abs(rating.wet_coil_diagnostics['required_area_temperature_residual'])<=controls.outlet_temperature_tolerance_K
    assert abs(rating.closed_balance.inside.T_out-target.T_out_inside)<=controls.outlet_temperature_tolerance_K
    assert abs(rating.wet_coil_diagnostics['independent_energy_residual'])<=controls.energy_tolerance_W


def test_finned_rating_uses_reference_search(monkeypatch):
    hx,a,b=context(finned=True,W=.004)
    def forbidden(*args,**kwargs):
        raise AssertionError('Finned must retain native Rating search')
    monkeypatch.setattr(_wet_rating,'_rate_optimized',forbidden)
    target=hx.simulate(a,b,wet_solver_options=WetCoilSolverOptions(timeout_s=None))
    rating=hx.rate(*specs(a,b,target.T_out_outside),wet_solver_options=WetCoilSolverOptions(timeout_s=None))
    assert rating.wet_coil_diagnostics['numerical_path']=='native'
    assert 'numerical_rating' not in rating.wet_coil_diagnostics


def test_rejected_initializer_fallback_retains_the_original_deadline(monkeypatch):
    hx,a,b=context(W=.007)
    now=[10.]
    monkeypatch.setattr(wet_coil_solver,'monotonic',lambda:now[0])
    controls=WetCoilSolverOptions(timeout_s=3.)
    parent=wet_coil_solver._WetSolveBudget(controls)
    original=wet_coil_adapters.solve_production_coil
    def reject(*args,**kwargs):
        child=kwargs['_budget']
        if getattr(child,'fidelity',None) is not None:
            assert child.started==parent.started and child.deadline==parent.deadline
            now[0]=parent.deadline+.1
            raise ValueError('initializer rejected after deadline')
        return original(*args,**kwargs)
    monkeypatch.setattr(wet_coil_adapters,'solve_production_coil',reject)
    with pytest.raises(WetCoilTimeoutError) as caught:
        integration.forward_wet_process(hx,a,b,wet_solver_options=controls,_budget=parent)
    assert parent.deadline==13. and caught.value.diagnostics['timeout_s']==3.
    assert caught.value.diagnostics['cold_predictor_fallbacks']==1


def test_final_physical_failure_is_not_hidden_by_native_retry(monkeypatch):
    hx,a,b=context(W=.007)
    calls=[]
    def invalid(*args,**kwargs):
        calls.append(getattr(kwargs['_budget'],'fidelity',None))
        raise WetCoilModelError('injected physical inadmissibility')
    monkeypatch.setattr(wet_coil_adapters,'_solve_profile_candidate',invalid)
    with pytest.raises(WetCoilModelError,match='physical inadmissibility'):
        integration.forward_wet_process(hx,a,b,wet_solver_options=WetCoilSolverOptions(timeout_s=None))
    assert len(calls)==2 and calls[0] is not None and calls[1] is None
