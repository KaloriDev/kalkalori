# SPDX-License-Identifier: GPL-3.0-only
"""Explicit historical provider: synthetic c80e779 cases, no project data."""
from dataclasses import replace

import pytest

from core import (LegacyBulkMeanWetCoilProvider, ElmahdyMitalasWetCoilProvider,
                  WetCoilProviderUnsupportedError, WetCoilSolverOptions, WetCoilTimeoutError)
from core.models.simulation import run_simulation
from core.phase_change.types import PhaseChangeMode
from core.phase_change.warning_codes import CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY
from core.tests.wet_coil_public_test import context, specs
from core.tests.wet_finned_simulation_test import _wet_finned_hx, _side_inputs
from core.tests.outside_water_condensation_integration_test import hx as historical_bare_hx

LEGACY = LegacyBulkMeanWetCoilProvider()


def historical_case(finned):
    inside, outside = _side_inputs()
    if finned:
        return _wet_finned_hx(), inside, outside
    return historical_bare_hx.__wrapped__(), replace(inside, T_in=290.0), replace(outside, T_in=420.0)


@pytest.fixture(scope="module", params=[False, True], ids=["bare", "finned"])
def active(request):
    hx, a, b = historical_case(request.param)
    return hx, a, b, hx.simulate(a, b, wet_coil_provider=LEGACY)


def test_contract_rating_margin_and_no_fallback(monkeypatch):
    from core.phase_change import wet_coil_integration as engine
    monkeypatch.setattr(engine, "_run_elmahdy", lambda *a, **k: pytest.fail("fallback"))
    hx, a, b = context()
    assert LEGACY.supports_simulation and not LEGACY.supports_rating
    for margin in (0.1, -0.1):
        with pytest.raises(WetCoilProviderUnsupportedError, match="surface_margin"):
            hx.simulate(a, b, wet_coil_provider=LEGACY, surface_margin=margin)
    with pytest.raises(WetCoilProviderUnsupportedError, match="rating"):
        hx.rate(*specs(a, b, 294.0), wet_coil_provider=LEGACY)
    with pytest.raises(WetCoilProviderUnsupportedError, match="not applicable"):
        hx.simulate(a, b, wet_coil_provider=LEGACY, flow_arrangement="cocurrentflow")
    with pytest.raises(ValueError, match="iterate"):
        hx.simulate(a, b, wet_coil_provider=LEGACY, iterate=False)
    r = hx.simulate(a, b, wet_coil_provider=LEGACY)
    assert r.converged and r.outside_phase_change.active


@pytest.mark.parametrize("finned", [False, True])
def test_disabled_bypasses_legacy(finned):
    hx, a, b = historical_case(finned)
    b = replace(b, phase_change_mode=PhaseChangeMode.DISABLED)
    r = hx.simulate(a, b, wet_coil_provider=LEGACY, surface_margin=0.1)
    dry = run_simulation(hx, a, b, surface_margin=0.1)
    assert (r.q, r.T_out_inside, r.T_out_outside, r.UA) == (dry.q, dry.T_out_inside, dry.T_out_outside, dry.UA)
    assert r.wet_coil_diagnostics is None


@pytest.mark.parametrize("iterate", [False, True])
def test_auto_dry_preserves_historical_baseline(iterate):
    hx, a, b = context(W=0.004)
    r = hx.simulate(a, b, wet_coil_provider=LEGACY, iterate=iterate)
    dry = run_simulation(hx, a, b, iterate=iterate)
    assert not r.outside_phase_change.active
    assert (r.q, r.T_out_inside, r.T_out_outside, r.UA) == (dry.q, dry.T_out_inside, dry.T_out_outside, dry.UA)
    assert r.wet_coil_diagnostics['surface']['wet_regime'] == 'DRY'
    assert r.wet_coil_diagnostics['surface']['surface_temperature_wet_mean'] is None


def test_active_balances_and_native_surface(active):
    hx, a, b, r = active
    pc = r.outside_phase_change
    assert r.converged and pc.active and pc.converged
    assert pc.m_dot_condensate > 0
    assert pc.m_dot_water_vapor_in == pytest.approx(pc.m_dot_water_vapor_out + pc.m_dot_condensate, abs=1e-8)
    assert r.q == pytest.approx(pc.Q_sensible + pc.Q_latent, abs=1e-7)
    assert abs(pc.mass_balance_error) < 1e-8
    assert abs(pc.energy_balance_error) < 1e-7
    s = r.wet_coil_diagnostics['surface']
    assert s['surface_temperature_min'] == pc.wall_temperature_min
    assert s['surface_temperature_max'] == pc.wall_temperature_max
    assert s['surface_temperature_wet_mean'] == pc.wall_temperature_wet_mean
    assert s['dew_point_in'] == pc.dew_point_in
    assert s['dew_point_out'] == pc.dew_point_out
    assert s['onset_margin'] == pc.onset_margin_K
    assert s['wet_surface_fraction'] == pc.wet_surface_fraction
    assert s['onset_envelope_method'] == 'two_point_0d_estimate'
    assert s['wall_envelope_method'] == (
        'two_point_0d_estimate' if r.wet_finned_surface is None else
        'bulk_mean_radial_surface_extrema_with_0d_cold_zone_offset')
    assert 'axial_wet_fraction' not in s and 'interface_temperature' not in s
    assert pc.method == LEGACY.model_id
    assert r.wet_coil_diagnostics['native_method'].startswith('outside_condensation_0d_')
    assert r.wet_coil_diagnostics is pc.wet_coil_diagnostics


def test_installed_hydraulics(active):
    from core.heat_transfer.outside_dispatch import evaluate_outside_hydraulics
    hx, a, b, r = active
    pc = r.outside_phase_change
    bank = evaluate_outside_hydraulics(
        bundle=hx.bundle, m_dot=0.5*(pc.m_dot_gas_in+pc.m_dot_gas_out),
        inlet_props=r.outside_properties_inlet, midpoint_props=r.outside_props_mean,
        outlet_props=r.outside_properties_outlet,
        temperature_in=b.T_in, temperature_out=r.T_out_outside, pressure=b.p,
        m_dot_inlet=pc.m_dot_gas_in, m_dot_midpoint=0.5*(pc.m_dot_gas_in+pc.m_dot_gas_out),
        m_dot_outlet=pc.m_dot_gas_out,
    )
    assert r.final_result.A_o == hx.bundle.total_outer_area
    assert r.outside_dp_total == pytest.approx(bank.dp_total, rel=1e-12)
    assert r.outside_tube_bank_hydraulic.midpoint.face_velocity == pytest.approx(bank.midpoint.face_velocity)
    assert r.inside_dp_total > 0


def test_historical_radial_fin_state_and_reference_dp(active):
    hx, a, b, r = active
    wet = r.wet_finned_surface
    if wet is None:
        assert 'fin_tip_temperature' not in r.wet_coil_diagnostics['surface']
        return
    assert wet is r.outside_phase_change.wet_finned_surface
    assert wet is r.thermal_state.finned_tube_diagnostics.wet_surface
    assert wet is r.final_result.wet_finned_surface
    assert wet.annular_fin.radial_cells > 1
    assert 0 < wet.fin_wet_fraction < 1
    assert wet.wet_dry_boundary_radius is not None
    s = r.wet_coil_diagnostics['surface']
    for name in ('fin_base_temperature', 'fin_tip_temperature', 'fin_wet_fraction',
                 'wet_dry_boundary_radius', 'primary_surface_temperature',
                 'core_wall_temperature', 'root_surface_temperature'):
        assert s[name] == getattr(wet, name)
    assert wet.Q_total == pytest.approx(wet.Q_primary_total + wet.Q_fin_total)
    assert wet.m_dot_condensate == pytest.approx(wet.m_dot_condensate_primary + wet.m_dot_condensate_fin)
    assert abs(wet.energy_balance_error) < 0.01
    assert r.finned_tube_diagnostics.outside_dp_reference_only
    assert not r.wet_pressure_drop_supported
    assert r.outside_dp_dry_reference == r.outside_dp_total
    assert CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY in {w.code for w in r.outside_phase_change.warnings}


@pytest.mark.parametrize('radial', [False, True])
def test_shared_deadline_interrupts_legacy_iterations(monkeypatch, radial):
    from core.heat_transfer import wet_coil_solver as controls
    from core.phase_change import outside_condensation_solver as global_solver
    from core.phase_change import wet_finned_surface as fin_solver
    now = [10.0]
    monkeypatch.setattr(controls, 'monotonic', lambda: now[0])
    module, name = (fin_solver, '_evaluate_chain') if radial else (global_solver, '_evaluate_local_wall_state')
    original = getattr(module, name)
    seen = []
    def expires(*args, **kwargs):
        budget = controls._active_budget.get()
        assert budget.started == 10 and budget.deadline == 11
        seen.append(budget)
        now[0] = 12.0
        return original(*args, **kwargs)
    monkeypatch.setattr(module, name, expires)
    hx, a, b = historical_case(radial)
    with pytest.raises(WetCoilTimeoutError):
        hx.simulate(a, b, wet_coil_provider=LEGACY, wet_solver_options=WetCoilSolverOptions(timeout_s=1))
    assert seen and all(c is seen[0] for c in seen)
    assert controls._active_budget.get() is None


def test_default_remains_elmahdy_simulation():
    hx, a, b = context(W=0.004)
    default = hx.simulate(a, b)
    explicit = hx.simulate(a, b, wet_coil_provider=ElmahdyMitalasWetCoilProvider())
    for field in ('q', 'T_out_inside', 'T_out_outside', 'UA'):
        assert getattr(default, field) == getattr(explicit, field)
    assert default.wet_coil_diagnostics['global_wet_model'] == ElmahdyMitalasWetCoilProvider().model_id


def test_endpoint_onset_uses_bounded_0d_wet_zone_when_mean_fin_is_dry() -> None:
    """Exercise the non-segmented fallback used by economizer-like pinches.

    Fixture is a synthetic finned economizer, unrelated to any specific
    project geometry, chosen empirically to trigger the same endpoint
    wet-zone-fallback code path as the case that originally motivated it.
    """

    from core.geometry.tube import BareTube
    from core.geometry.finned_tube import CircularFinnedTube
    from core.geometry.bundle import TubeBundle
    from core.models.bare_tube import BareTubeHeatExchanger
    from core.models.simulation import HXSideInput
    from core.properties.fluids import ConstantPropertyProvider
    from core.properties.common import FluidTransportProperties
    from core.properties.gas_mixture import GasMixturePropertyProvider, gas_mixture_from_dry_composition_and_water_ratio
    P = 101325.0

    core = BareTube(
        D_i=0.0189,
        D_o=0.0212,
        length_total=2.75,
        length_effective=2.72,
        wall_k=45.0,
    )
    tube = CircularFinnedTube(
        core_tube=core,
        fin_k=175.0,
        D_fin=0.0508,
        D_root=0.0224,
        fin_thickness_root=0.00035,
        fin_thickness_tip=0.00018,
        fin_pitch=0.0028,
        fin_contact_efficiency=0.92,
    )
    hx = BareTubeHeatExchanger(
        TubeBundle(
            tube=tube,
            n_rows=12,
            n_tubes_per_row=84,
            pitch_transverse=0.052,
            pitch_longitudinal=0.045,
            layout="staggered",
            n_passes_tube=16,
            n_passes_transverse=4,
            flow_arrangement="counterflow",
        )
    )
    liquid_stub = ConstantPropertyProvider(
        FluidTransportProperties(
            rho=1_010.0,
            mu=1.6e-3,
            k=0.38,
            cp=3_550.0,
        )
    )
    wet_air = GasMixturePropertyProvider(
        gas_mixture_from_dry_composition_and_water_ratio(
            dry_components={"N2": 0.77, "O2": 0.21, "CO2": 0.02},
            dry_basis="mole",
            water_ratio=0.055,
            imposed_phase="gas",
        )
    )
    result = hx.simulate(
        HXSideInput(
            provider=liquid_stub,
            m_dot=31_500.0 / 3600.0,
            T_in=293.15,
            p=250_000.0,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        HXSideInput(
            provider=wet_air,
            m_dot=142_800.0 / 3600.0,
            T_in=335.15,
            p=P,
            phase_change_mode=PhaseChangeMode.AUTO,
        ),
        wet_coil_provider=LEGACY,
    )

    phase = result.outside_phase_change
    wet = result.wet_finned_surface
    assert phase is not None and phase.active and phase.converged
    assert wet is not None and wet.m_dot_condensate > 0.0
    assert wet is phase.wet_finned_surface
    assert wet is result.final_result.wet_finned_surface
    assert wet is result.thermal_state.finned_tube_diagnostics.wet_surface
    assert wet.condensation_area_fraction < 1.0
    assert wet.condensation_temperature_offset_K < 0.0
    assert result.wet_coil_diagnostics["native_method"].endswith("with_endpoint_wet_zone_fallback")
    assert phase.residuals["outer_relaxation_factor"] == 0.25
    assert (
        "endpoint_envelope_wet_zone_0d_linear_weighting"
        in wet.assumptions
    )
    assert wet.Q_total == pytest.approx(
        wet.Q_primary_total + wet.Q_fin_total,
        abs=1.0e-7,
    )
    assert wet.m_dot_condensate == pytest.approx(
        wet.m_dot_condensate_primary + wet.m_dot_condensate_fin,
        abs=1.0e-12,
    )
    assert wet.wet_area == pytest.approx(
        wet.wet_primary_area + wet.wet_fin_area,
        abs=1.0e-10,
    )
    assert phase.Q_total == wet.Q_total
    assert phase.m_dot_condensate == wet.m_dot_condensate
    assert abs(phase.mass_balance_error) < 1.0e-6
    assert abs(phase.energy_balance_error) < 1.0e-6


def test_native_whole_stream_energy_closure(active):
    from core.phase_change.capability import detect_phase_change_capability
    from core.phase_change.wet_gas_enthalpy import WetGasEnthalpyEvaluator
    hx, a, b, r = active
    pc = r.outside_phase_change
    evaluator = WetGasEnthalpyEvaluator(b.p, detect_phase_change_capability(b.provider))
    gas_heat = pc.m_dot_dry_carrier * (
        evaluator.enthalpy(b.T_in, pc.W_in) - evaluator.enthalpy(r.T_out_outside, pc.W_out))
    assert pc.H_drain > 0
    # Native global residual convention: mean-cp sensible inside balance and
    # gas enthalpy loss minus fully drained condensate; no new tolerance mapping.
    assert gas_heat - pc.H_drain == pytest.approx(r.q, rel=1e-4)
    assert a.m_dot * r.inside_props_mean.cp * (r.T_out_inside-a.T_in) == pytest.approx(r.q, rel=1e-4)


def test_frost_remains_historical_dry_warning():
    from core.phase_change.warning_codes import FROSTING_NOT_SUPPORTED
    from core.models.simulation import HXSideInput
    from core.properties.fluids import ConstantPropertyProvider
    from core.properties.common import FluidTransportProperties
    hx, _, b = context()
    a = HXSideInput(ConstantPropertyProvider(FluidTransportProperties(
        rho=1010.0, mu=1.6e-3, k=0.38, cp=3550.0)), 0.4, 250.0, 2e5)
    r = hx.simulate(a, b, wet_coil_provider=LEGACY)
    assert not r.outside_phase_change.active
    assert FROSTING_NOT_SUPPORTED in {w.code for w in r.outside_phase_change.warnings}


def test_unsupported_outside_and_enhancement_do_not_fall_back(monkeypatch):
    from types import SimpleNamespace
    from core.phase_change import wet_coil_integration as engine
    from core.phase_change.integration import PhaseChangeSettings
    monkeypatch.setattr(engine, '_run_elmahdy', lambda *a, **k: pytest.fail('fallback'))
    hx, a, b = historical_case(False)
    with pytest.raises(WetCoilProviderUnsupportedError, match='not applicable'):
        hx.simulate(a, replace(b, provider=a.provider), wet_coil_provider=LEGACY)
    enhanced = SimpleNamespace(bundle=hx.bundle, tube_side_enhancement=object())
    assert not LEGACY.is_applicable(enhanced, a, b, settings=PhaseChangeSettings())


def test_native_iteration_settings_not_generic_residuals(monkeypatch):
    from core.phase_change import legacy_wet_coil_integration as legacy
    from core.heat_transfer import wet_coil_solver as controls
    hx, a, b = historical_case(False)
    class Captured(Exception):
        pass
    def capture(*args, **kwargs):
        assert kwargs['check'].__self__ is controls._active_budget.get()
        assert kwargs['max_iterations'] == 71
        assert kwargs['relative_Q_tolerance'] == 2e-5
        assert kwargs['temperature_tolerance_K'] == 0.03
        assert kwargs['water_ratio_tolerance'] == 1e-6
        raise Captured
    monkeypatch.setattr(legacy, 'solve_outside_condensation', capture)
    with pytest.raises(Captured):
        hx.simulate(a, b, wet_coil_provider=LEGACY,
                    phase_change_max_iterations=71,
                    phase_change_relative_Q_tolerance=2e-5,
                    phase_change_temperature_tolerance_K=0.03,
                    wet_solver_options=WetCoilSolverOptions(energy_tolerance_W=0.123,
                        mass_tolerance_kg_s=2e-10, outlet_temperature_tolerance_K=1e-7))
