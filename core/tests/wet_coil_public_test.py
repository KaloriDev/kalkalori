# SPDX-License-Identifier: GPL-3.0-only
"""Physical public roundtrips and post-solve equivalent reporting invariants."""
from dataclasses import replace

import pytest

from core import WetCoilSolverOptions
from core.tests.wet_coil_test import production_context
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput, run_simulation
from core.models.heat_balance import BalanceSideSpec
from core.phase_change.types import PhaseChangeMode
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.wet_coil_integration import (
    forward_wet_process,
)
from core.phase_change.wet_coil_reporting import equivalent_wet_process

# These regression assertions intentionally retain sub-engineering precision.
TIGHT_OPTIONS = WetCoilSolverOptions(
    energy_tolerance_W=2e-4,
    mass_tolerance_kg_s=2e-10,
    outlet_temperature_tolerance_K=2e-7,
    timeout_s=None,
)


def context(finned=False, W=0.016):
    bundle, thermo, inside = production_context(finned)
    hx = BareTubeHeatExchanger(bundle)
    a = HXSideInput(inside.provider, 0.4, 280.15, 2e5)
    b = HXSideInput(
        wet_gas_provider_at_water_ratio(thermo.capability, W),
        0.5 * (1 + W),
        300.15,
        101325.0,
    )
    return hx, a, b


def specs(a, b, target, inside_target=None, mass_known=True):
    return (
        BalanceSideSpec(
            a.provider, a.p, a.m_dot if mass_known else None, a.T_in, inside_target
        ),
        BalanceSideSpec(b.provider, b.p, b.m_dot, b.T_in, target),
    )


def check_equivalent(r):
    if hasattr(r, "UA_required"):
        assert r.EMTD == pytest.approx(r.Q_required / r.UA_required, rel=2e-13)
        assert r.U_mean == pytest.approx(r.UA_required / r.A_required, rel=2e-13)
        assert r.UA_actual == pytest.approx(r.U_mean * r.A_o, rel=2e-13)
        assert r.overdesign_factor == pytest.approx(
            r.UA_actual / r.UA_required - 1, abs=2e-13
        )
        assert r.overdesign_factor == pytest.approx(r.A_o / r.A_required - 1, abs=2e-13)
        assert r.ua_margin == r.overdesign_factor
        assert (
            r.wet_coil_diagnostics["physical_surface_overdesign"] == r.overdesign_factor
        )
    else:
        assert r.U_mean == pytest.approx(
            r.UA_process / r.wet_coil_diagnostics["thermal_outside_area"], rel=2e-13
        )
        assert r.UA_actual == pytest.approx(r.U_mean * r.final_result.A_o, rel=2e-13)
        assert r.overdesign_factor == pytest.approx(
            r.UA_actual / r.UA_process - 1, abs=2e-13
        )
        assert r.overdesign_factor == pytest.approx(r.surface_margin, abs=2e-13)
    assert r.ua_is_equivalent
    assert r.ua_reporting_basis == "wet_equivalent_secant_capacity_ntu"
    native = r.wet_coil_diagnostics
    phase = r.outside_phase_change
    assert phase.Q_total == pytest.approx(phase.Q_sensible + phase.Q_latent, abs=1e-8)
    assert native["endpoint_sensible_heat"] + native["endpoint_latent_heat"] - phase.Q_total == pytest.approx(
        native["sensible_latent_closure_residual"], abs=1e-10
    )
    assert abs(native["sensible_latent_closure_residual"]) < 0.002
    state = r.thermal_state
    ic, oc = native["inside_correlation"], native["outside_correlation"]
    assert state.diagnostics.inside_Nu_base == ic.Nu_base
    assert state.diagnostics.inside_Nu_corrected == ic.Nu_corrected
    assert state.diagnostics.inside_alfa_base == ic.alfa_base
    assert state.diagnostics.inside_alfa_corrected == ic.alfa_corrected
    assert state.diagnostics.outside_Nu_corrected == oc.nusselt_number
    assert state.residual == native["property_residual_K"]
    for probe in r.wall_temperature_envelope.probes:
        assert probe.iterations == native["property_iterations"]
        assert probe.residual == native["property_residual_K"]
    assert state.inside_wall_temperature == r.wall_temperature_envelope.inside_mean
    assert state.outside_wall_temperature == r.wall_temperature_envelope.outside_mean
    if state.finned_tube_diagnostics is not None:
        fd = state.finned_tube_diagnostics
        assert fd.thermal_reporting_basis == "native_dry_constitutive_network"
        assert fd.fin_efficiency == native["network"].fin_efficiency
        assert fd.resistance_total == native["network"].resistance_total
        assert fd.outside_alpha_effective_gross == pytest.approx(state.alfa_o, rel=1e-12)


@pytest.mark.parametrize(
    "finned,W,regime",
    [
        (False, 0.004, "DRY"),
        (False, 0.007, "PARTIALLY_WET"),
        (False, 0.016, "FULLY_WET"),
        (True, 0.004, "DRY"),
        (True, 0.01, "PARTIALLY_WET"),
        (True, 0.016, "FULLY_WET"),
    ],
)
def test_public_forward_inverse_roundtrip(finned, W, regime):
    hx, a, b = context(finned, W)
    sim = hx.simulate(a, b, surface_margin=0.1, wet_solver_options=TIGHT_OPTIONS)
    rating = hx.rate(*specs(a, b, sim.T_out_outside), wet_solver_options=TIGHT_OPTIONS)
    d = rating.wet_coil_diagnostics
    assert d["required_area_scale"] == pytest.approx(1 / 1.1, rel=2e-5)
    assert rating.A_required == pytest.approx(hx.bundle.total_outer_area / 1.1, rel=2e-5)
    assert rating.final_result.A_o == rating.A_o == hx.bundle.total_outer_area
    recovered = hx.simulate(a, b, surface_margin=rating.overdesign_factor, wet_solver_options=TIGHT_OPTIONS)
    assert recovered.q == pytest.approx(sim.q, rel=2e-6, abs=0.002)
    assert recovered.T_out_inside == pytest.approx(sim.T_out_inside, abs=2e-5)
    assert recovered.T_out_outside == pytest.approx(sim.T_out_outside, abs=2e-5)
    assert recovered.outside_water_ratio_out == pytest.approx(
        sim.outside_water_ratio_out, abs=2e-9
    )
    assert recovered.outside_condensate_mass_flow == pytest.approx(
        sim.outside_condensate_mass_flow, abs=1e-9
    )
    assert recovered.outside_phase_change.H_drain == pytest.approx(
        sim.outside_phase_change.H_drain, abs=2e-4
    )
    if regime == "DRY":
        assert not recovered.phase_change_active
        assert rating.ua_is_equivalent
        assert recovered.outside_phase_change.regime == "DRY"
        check_equivalent(sim)
        check_equivalent(rating)
    else:
        assert recovered.outside_phase_change.regime == regime
        assert rating.outside_phase_change.regime == regime
        check_equivalent(sim)
        check_equivalent(rating)
        pc = rating.outside_phase_change
        assert abs(pc.mass_balance_error) < 2e-10
        assert abs(pc.energy_balance_error) < 0.002
        assert pc.Q_total == pytest.approx(pc.Q_sensible + pc.Q_latent, abs=0.002)


@pytest.mark.parametrize("finned", [False, True])
def test_public_thermal_reserve_changes_wet_solution(finned):
    hx, a, b = context(finned)
    results = [hx.simulate(a, b, surface_margin=m) for m in (0.0, 0.05, 0.1)]
    assert results[0].q > results[1].q > results[2].q
    assert (
        results[0].T_out_outside < results[1].T_out_outside < results[2].T_out_outside
    )
    assert (
        results[0].outside_condensate_mass_flow
        > results[1].outside_condensate_mass_flow
        > results[2].outside_condensate_mass_flow
    )
    for r in results:
        check_equivalent(r)
        d = r.wet_coil_diagnostics
        assert d["hydraulic_effective_length"] == hx.bundle.tube.length_effective
        assert d["hydraulic_total_length"] == hx.bundle.tube.length_total
        assert d["hydraulic_inner_flow_area"] == hx.bundle.internal_flow_area_per_pass
        assert d["hydraulic_frontal_area"] == hx.bundle.frontal_flow_area
        assert r.Q_full == pytest.approx(results[0].q, abs=0.002)


@pytest.mark.parametrize("finned", [False, True])
def test_inside_flow_inverse_roundtrip(finned):
    hx, a, b = context(finned)
    sim = hx.simulate(a, b, surface_margin=0.1, wet_solver_options=TIGHT_OPTIONS)
    rated = hx.rate(
        *specs(a, b, sim.T_out_outside, sim.T_out_inside, False), wet_solver_options=TIGHT_OPTIONS
    )
    assert rated.closed_balance.inside.m_dot == pytest.approx(a.m_dot, rel=2e-5)
    assert rated.wet_coil_diagnostics["required_area_scale"] == pytest.approx(
        1 / 1.1, rel=2e-5
    )
    check_equivalent(rated)
    assert rated.Q_required == pytest.approx(sim.q, abs=0.002)


def test_equivalent_reduction_uses_process_duty_and_has_controlled_limits():
    from core.heat_transfer.ntu import effectiveness_ntu

    for hot_out, cold_out in ((330.0, 300.0), (320.0, 310.0), (310.0, 320.0)):
        r = equivalent_wet_process(
            heat=10000.0,
            hot_in=350.0,
            hot_out=hot_out,
            cold_in=280.0,
            cold_out=cold_out,
            flow_arrangement="counterflow",
        )
        assert effectiveness_ntu(
            r.hot_capacity, r.cold_capacity, r.ua, flow_arrangement="counterflow"
        ) == pytest.approx(r.effectiveness, rel=1e-10)
        assert r.emtd == pytest.approx(10000.0 / r.ua)
    with pytest.raises(ValueError, match="undefined"):
        equivalent_wet_process(
            heat=0.0,
            hot_in=300.0,
            hot_out=300.0,
            cold_in=280.0,
            cold_out=280.0,
            flow_arrangement="counterflow",
        )


def test_disabled_dry_numerical_path_is_unchanged():
    hx, a, b = context()
    a = HXSideInput(
        a.provider, a.m_dot, a.T_in, a.p, phase_change_mode=PhaseChangeMode.DISABLED
    )
    b = replace(b, phase_change_mode=PhaseChangeMode.DISABLED)
    actual = hx.simulate(a, b)
    reference = run_simulation(hx, a, b)
    assert (actual.q, actual.T_out_inside, actual.T_out_outside, actual.UA) == (
        reference.q,
        reference.T_out_inside,
        reference.T_out_outside,
        reference.UA,
    )
    assert not actual.ua_is_equivalent


@pytest.mark.parametrize("finned", [False, True])
@pytest.mark.parametrize("scale", [0.75, 1.0, 1.25])
def test_thermal_scale_preserves_transport_at_identical_states(finned, scale):
    from core.heat_transfer.wet_coil_adapters import (
        BareTubeAdapter,
        CircularFinnedTubeAdapter,
    )

    b, t, i = production_context(finned)
    scaled = replace(i, thermal_scale=scale)
    ri, di = i.evaluate(285.0)
    rs, ds = scaled.evaluate(285.0)
    assert rs == pytest.approx(ri / scale, rel=2e-13)
    assert scaled.bundle is b
    assert ds["film_resistance"] == pytest.approx(di["film_resistance"] / scale)
    assert ds["wall_resistance"] == pytest.approx(di["wall_resistance"] / scale)
    assert ds["inside_correlation"] == di["inside_correlation"]
    assert ds["htc"] == di["htc"]
    assert ds["reynolds"] == di["reynolds"]
    typ = CircularFinnedTubeAdapter if finned else BareTubeAdapter
    a = typ(b, t)
    c = typ(b, t, thermal_scale=scale)
    rd, d = a.evaluate(300.0, 0.016, 0.5, (ri, di))
    sd, s = c.evaluate(300.0, 0.016, 0.5, (rs, ds))
    assert sd == pytest.approx(rd / scale, rel=2e-13)
    assert d["outside_alpha_physical"] == s["outside_alpha_physical"]
    assert d["outside_reynolds"] == s["outside_reynolds"]
    assert s["dry_effective_area"] == pytest.approx(
        d["dry_effective_area"] * scale, rel=2e-13
    )
    assert s["outside_correlation"] == d["outside_correlation"]
    assert s["thermal_area"] == pytest.approx(b.total_outer_area * scale)
    if finned:
        assert s["common_root_contact_resistance"] == pytest.approx(
            d["common_root_contact_resistance"] / scale)
        full = a.response(285.0, 300.0, 0.016, d)
        reserved = c.response(285.0, 300.0, 0.016, s)
        assert reserved["heat_gas"] == pytest.approx(full["heat_gas"] * scale, rel=1e-10)
        assert reserved["wet_area"] == pytest.approx(full["wet_area"] * scale)
        assert reserved["fin_tip"] == pytest.approx(full["fin_tip"], abs=1e-7)



@pytest.mark.parametrize("finned", [False, True])
@pytest.mark.parametrize("scale", [0.75, 1.0, 1.25])
def test_rating_area_keeps_installed_hydraulics(finned, scale, monkeypatch):
    from core.phase_change import wet_coil_integration as integration

    hx, a, b = context(finned)
    # Include unheated ends so hydraulic length cannot accidentally follow area.
    tube = hx.bundle.tube
    core = tube.core_tube if finned else tube
    core = replace(core, length_total=core.length_effective + 0.2)
    hx = BareTubeHeatExchanger(replace(
        hx.bundle, tube=replace(tube, core_tube=core) if finned else core))
    bundle = hx.bundle
    target = forward_wet_process(
        hx, a, b, _area_scale=scale, wet_solver_options=TIGHT_OPTIONS)[0]
    original_forward = integration.forward_wet_process
    original_solve = BareTubeHeatExchanger.solve
    original_hydraulics = integration.evaluate_outside_hydraulics
    snapshots, banks, trials = [], [], []

    def forward(installed, *args, **kwargs):
        assert installed is hx
        trials.append(kwargs["_area_scale"])
        return original_forward(installed, *args, **kwargs)

    def solve(installed, *args, **kwargs):
        assert installed is hx
        snapshot = original_solve(installed, *args, **kwargs)
        snapshots.append(snapshot)
        return snapshot

    def hydraulics(**kwargs):
        assert kwargs["bundle"] is bundle
        bank = original_hydraulics(**kwargs)
        banks.append(bank)
        return bank

    monkeypatch.setattr(integration, "forward_wet_process", forward)
    monkeypatch.setattr(BareTubeHeatExchanger, "solve", solve)
    monkeypatch.setattr(integration, "evaluate_outside_hydraulics", hydraulics)
    rated = hx.rate(*specs(a, b, target.air_out), wet_solver_options=TIGHT_OPTIONS)
    d = rated.wet_coil_diagnostics
    if scale == 1.0:
        assert trials == [1.0]
    else:
        assert len(set(trials)) > 1
    assert hx.bundle is bundle
    assert rated.A_o == bundle.total_outer_area
    assert rated.A_required == pytest.approx(bundle.total_outer_area * scale, rel=2e-5)
    assert rated.Q_required == pytest.approx(target.heat_liquid, abs=0.002)
    assert rated.closed_balance.inside.T_out == pytest.approx(target.liquid_out, abs=2e-5)
    assert rated.closed_balance.outside.T_out == pytest.approx(target.air_out, abs=2e-7)
    assert d["thermal_outside_area"] == rated.A_required
    assert d["thermal_inside_area"] == pytest.approx(bundle.total_inner_area * scale, rel=2e-5)
    assert d["hydraulic_total_length"] == bundle.tube.length_total
    assert d["hydraulic_effective_length"] == bundle.tube.length_effective
    assert d["hydraulic_inner_flow_area"] == bundle.internal_flow_area_per_pass
    assert d["hydraulic_frontal_area"] == bundle.frontal_flow_area
    assert not ({"required_effective_length", "required_total_length",
                 "required_geometry", "physical_length_overdesign",
                 "required_length_temperature_residual"} & d.keys())
    assert "required_effective_length" not in d["solver_statistics"]
    assert len(snapshots) == len(banks) == 1
    assert rated.tube_side_hydraulic == snapshots[0].tube_side_hydraulic
    assert rated.tube_side_pressure_drop == snapshots[0].tube_side_pressure_drop
    assert rated.outside_tube_bank_hydraulic == replace(
        banks[0], midpoint_method="arithmetic_temperature_and_water_ratio")
    assert rated.outside_tube_bank_hydraulic.face_area == bundle.frontal_flow_area
    check_equivalent(rated)


def test_wet_rating_rejects_unspecified_outlet_and_incompatible_program():
    from core.phase_change.wet_coil_integration import WetRatingIncompatibilityError

    hx, a, b = context()
    with pytest.raises(WetRatingIncompatibilityError, match="outside.T_out"):
        hx.rate(*specs(a, b, None))
    sim = hx.simulate(a, b)
    with pytest.raises(WetRatingIncompatibilityError, match="incompatible"):
        hx.rate(*specs(a, b, sim.T_out_outside, sim.T_out_inside + 1.0))


def test_equivalent_reporting_cannot_feed_back_into_physical_solution(monkeypatch):
    import core.phase_change.wet_coil_integration as integration

    hx, a, b = context()
    physical = forward_wet_process(hx, a, b)
    original = integration.equivalent_wet_process

    def altered(**kw):
        r = original(**kw)
        return replace(r, ua=r.ua * 2, emtd=r.emtd / 2)

    monkeypatch.setattr(integration, "equivalent_wet_process", altered)
    result = integration._public_simulation(hx, a, b, physical)
    r = physical[0]
    assert result.q == r.heat_liquid
    assert result.T_out_inside == r.liquid_out
    assert result.T_out_outside == r.air_out
    assert result.outside_water_ratio_out == r.humidity_out
    assert result.outside_phase_change.H_drain == r.drain_enthalpy
    assert result.outside_condensate_mass_flow == r.condensate


def test_auto_dry_uses_same_physical_engine_as_wet_not_disabled_legacy():
    hx,a,b=context(False,.004)
    native=forward_wet_process(hx,a,b)[0]
    auto=hx.simulate(a,b)
    legacy=hx.simulate(a,replace(b,phase_change_mode=PhaseChangeMode.DISABLED))
    assert native.regime == auto.outside_phase_change.regime == "DRY"
    assert not auto.outside_phase_change.active
    assert auto.q == native.heat_liquid
    assert auto.residual_T_inside_K == native.diagnostics["property_change_liquid_out_K"]
    assert auto.residual_T_outside_K == native.diagnostics["property_change_air_out_K"]
    assert auto.T_out_outside == native.air_out
    assert auto.T_out_inside == native.liquid_out
    assert auto.outside_water_ratio_out == native.humidity_out
    assert auto.outside_condensate_mass_flow == 0
    assert auto.outside_phase_change.H_drain == 0
    assert abs(auto.q-legacy.q) > .01
    assert auto.ua_is_equivalent and not legacy.ua_is_equivalent


def test_public_auto_onset_is_continuous_and_exact_root_is_admissible():
    from scipy.optimize import brentq
    from core.phase_change.wet_coil_integration import _public_simulation
    cases={}
    def evaluate(w):
        if w not in cases:
            hx,a,b=context(False,w)
            cases[w]=(hx,a,b,forward_wet_process(hx,a,b))
        return cases[w]
    def criterion(w):
        return evaluate(w)[3][0].onset_margin
    onset=brentq(criterion,.0068215,.0068225,xtol=1e-14)
    states=[]
    for w in (onset-1e-9,onset,onset+1e-9,.006821986459538035):
        hx,a,b,physical=evaluate(w)
        r=physical[0]
        assert r.humidity_out <= physical[1].capability.W_in
        assert r.condensate >= 0
        assert r.drain_enthalpy >= 0
        states.append(_public_simulation(hx,a,b,physical))
    dry,at,wet,_=states
    assert dry.outside_phase_change.regime == "DRY"
    assert wet.outside_phase_change.regime == "PARTIALLY_WET"
    assert abs(wet.q-dry.q) < .002
    assert abs(wet.T_out_outside-dry.T_out_outside) < 2e-5
    assert abs(at.q-dry.q) < .002
    assert wet.outside_phase_change.W_out <= wet.outside_phase_change.W_in
    assert wet.outside_phase_change.m_dot_condensate >= 0


def test_rating_installed_simulation_preserves_explicit_inside_disabled():
    hx, inside, outside = context(W=0.004)
    # A capable inside gas is kept sensible by an explicit user setting.
    # Losing DISABLED would send the optional Simulation to another route.
    inside = HXSideInput(
        outside.provider, inside.m_dot, inside.T_in, outside.p,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )
    forward = hx.simulate(inside, outside)
    a, b = specs(inside, outside, forward.T_out_outside)
    a = replace(a, phase_change_mode=inside.phase_change_mode)
    rated = hx.rate(a, b, include_simulation=True)
    installed = rated.simulation
    assert installed.inside_phase_change.mode is PhaseChangeMode.DISABLED
    assert installed.outside_phase_change.global_wet_model == forward.outside_phase_change.global_wet_model
    assert installed.outside_phase_change.regime == "DRY"
    assert installed.q == pytest.approx(forward.q, abs=0.002)
    assert installed.T_out_outside == pytest.approx(forward.T_out_outside, abs=2e-5)
