# SPDX-License-Identifier: GPL-3.0-only
"""Exercise shared budgets, wall requirements and strict external failures."""
from dataclasses import FrozenInstanceError, replace
from importlib import import_module
from pathlib import Path
from time import monotonic, sleep
import math

import pytest

from core.enhancements import (
    EnhancementOperationContext, EnhancementProviderError,
    EnhancementTimeoutError, EnhancementUnsupportedError,
    TubeSideEnhancement, TubeSideEnhancementProvider, evaluate_enhancement,
)
from core.enhancements.integration import evaluate_for_bundle
from core.heat_transfer import thermal_iteration as thermal
from core.models import rating, simulation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.tests.enhancement_integration_test import bundle, inputs, VariableLiquid
from core.tests.enhancement_provider_contract_test import fake_configuration, sample_input


@pytest.fixture
def external(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tests" / "fixtures"))
    provider = import_module("external_enhancement_provider").ExternalProvider()
    assert not type(provider).__module__.startswith("core")
    assert isinstance(provider, TubeSideEnhancementProvider)
    return provider


def selection(provider):
    return TubeSideEnhancement(provider, provider.config, "liquid")


def solve(hx, mode, sides, **kwargs):
    inside, outside = sides
    if mode == "rating":
        return hx.rate(
            BalanceSideSpec(provider=inside.provider, p=inside.p, m_dot=inside.m_dot,
                            T_in=inside.T_in, T_out=301),
            BalanceSideSpec(provider=outside.provider, p=outside.p, m_dot=outside.m_dot,
                            T_in=outside.T_in), include_simulation=True, **kwargs,
        )
    return hx.simulate(inside, outside, iterate=mode != "snapshot", **kwargs)


def forbid_smooth_fallback(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Selected provider was replaced by smooth physics")
    monkeypatch.setattr("core.models.bare_tube.heat_transfer_coefficient_internal", forbidden)
    monkeypatch.setattr(thermal, "heat_transfer_coefficient_internal_diagnostics", forbidden)


@pytest.mark.parametrize("mode", ["rating", "simulation", "snapshot"])
def test_external_object_preserves_session_config_and_shared_context(external, monkeypatch, mode):
    now = [0.0]

    def ticking_clock():
        now[0] += .001
        return now[0]

    monkeypatch.setattr("core.enhancements.base.monotonic", ticking_clock)
    b = bundle(passes=2)
    hx = BareTubeHeatExchanger(b)
    config = selection(external)
    context = EnhancementOperationContext.from_timeout(30)
    session = external.session
    envelopes = []
    nested = []
    original_envelope = thermal.estimate_wall_temperature_envelope
    original_simulation = simulation.run_simulation

    def record_envelope(selected, *args, **kwargs):
        assert selected.tube_side_enhancement is config
        assert selected.enhancement_operation_context is context
        start = len(external.requests)
        result = original_envelope(selected, *args, **kwargs)
        envelopes.extend(external.requests[start:])
        return result

    def record_simulation(selected, *args, **kwargs):
        assert selected.tube_side_enhancement.provider is external
        assert selected.enhancement_operation_context is context
        nested.append(context.remaining_time())
        return original_simulation(selected, *args, **kwargs)

    monkeypatch.setattr(rating, "estimate_wall_temperature_envelope", record_envelope)
    monkeypatch.setattr(simulation, "estimate_wall_temperature_envelope", record_envelope)
    monkeypatch.setattr(simulation, "run_simulation", record_simulation)
    forbid_smooth_fallback(monkeypatch)
    result = solve(hx, mode, inputs(b, provider=VariableLiquid()),
                   tube_side_enhancement=config, enhancement_operation_context=context)
    assert hx.tube_side_enhancement is None
    assert hx.enhancement_operation_context is None
    assert external.session is session
    assert config.provider is external
    assert all(state.operation_context is context for state in external.requests)
    assert {state.position for state in external.requests} >= {"thermal", "inlet", "midpoint", "outlet"}
    assert len([state for state in external.requests if state.position == "thermal"]) > 4
    assert envelopes and all(state.operation_context is context for state in envelopes)
    assert all(before >= after for before, after in zip(external.remaining, external.remaining[1:]))
    assert external.remaining[0] > external.remaining[-1] > 0
    if mode == "rating":
        assert len(nested) == 1
        assert nested[0] < external.remaining[0]
        assert result.simulation.tube_side_enhancement.private_revision == external.config.revision
    enhanced = result.tube_side_enhancement
    assert enhanced.source_references == ("private:synthetic_fixture",)
    assert enhanced.source_access_basis == "private"
    assert enhanced.private_revision == external.config.revision
    assert enhanced.diagnostics[0].value == external.config.revision
    assert enhanced.nusselt is None
    assert enhanced.f_darcy == 4 * enhanced.friction_factor_native
    assert "external_fixture_notice" in {warning.code for warning in result.warnings}
    for point in (result.inside_properties_inlet, result.inside_properties_midpoint,
                  result.inside_properties_outlet):
        assert point.enhancement.private_revision == enhanced.private_revision


def test_old_provider_unlimited_input_and_solver_results_are_unchanged():
    config = fake_configuration()  # No optional capabilities or deadline logic.
    state = sample_input()
    context = EnhancementOperationContext()
    assert context.remaining_time() is None
    context.check_deadline()
    assert evaluate_enhancement(config, state) == evaluate_enhancement(
        config, replace(state, operation_context=context))
    b = bundle()
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=config)
    ordinary = hx.simulate(*inputs(b))
    unlimited = hx.simulate(*inputs(b), enhancement_operation_context=context)
    assert ordinary == unlimited
    assert hx.enhancement_operation_context is None


@pytest.mark.parametrize("field,value", [
    ("deadline", math.nan), ("deadline", math.inf), ("deadline", True),
    ("deadline", "later"), ("timeout", 0), ("timeout", -1),
    ("timeout", math.nan), ("timeout", math.inf), ("timeout", True),
])
def test_budget_validation(field, value):
    with pytest.raises(ValueError):
        if field == "deadline":
            EnhancementOperationContext(value)
        else:
            EnhancementOperationContext.from_timeout(value)


def test_context_is_immutable_and_input_is_typed():
    context = EnhancementOperationContext.from_timeout(1)
    with pytest.raises(FrozenInstanceError):
        context.deadline = 1
    with pytest.raises(TypeError, match="operation_context"):
        replace(sample_input(), operation_context=object())


@pytest.mark.parametrize("mode", ["rating", "simulation", "snapshot"])
def test_expired_budget_fails_before_any_provider_call(external, monkeypatch, mode):
    b = bundle()
    hx = BareTubeHeatExchanger(b)
    context = EnhancementOperationContext(monotonic() - 1)
    assert context.remaining_time() == 0
    forbid_smooth_fallback(monkeypatch)
    with pytest.raises(EnhancementTimeoutError, match="deadline expired"):
        solve(hx, mode, inputs(b), tube_side_enhancement=selection(external),
              enhancement_operation_context=context)
    assert external.requests == []
    assert hx.enhancement_operation_context is None


@pytest.mark.parametrize("mode", ["direct", "rating", "simulation", "snapshot"])
def test_expired_after_blocking_call_fails_even_when_provider_returns(external, monkeypatch, mode):
    # Control the clock so this is always an expired-after-call test, regardless
    # of scheduling and platform clock resolution. The real delay is only 5 ms.
    original = external.evaluate
    now = [0.0]
    monkeypatch.setattr("core.enhancements.base.monotonic", lambda: now[0])
    context = EnhancementOperationContext(.001)

    def delayed(geometry, state):
        result = original(geometry, state)
        sleep(.005)
        now[0] = .005
        return result

    external.evaluate = delayed
    with pytest.raises(EnhancementTimeoutError):
        if mode == "direct":
            evaluate_enhancement(selection(external), replace(sample_input(), operation_context=context))
        else:
            b = bundle()
            solve(BareTubeHeatExchanger(b), mode, inputs(b),
                  tube_side_enhancement=selection(external), enhancement_operation_context=context)
    assert len(external.requests) == 1


def test_cooperative_provider_can_observe_decreasing_remaining_time(external, monkeypatch):
    now = [10.0]
    monkeypatch.setattr("core.enhancements.base.monotonic", lambda: now[0])
    context = EnhancementOperationContext.from_timeout(2)
    state = replace(sample_input(), operation_context=context)
    evaluate_enhancement(selection(external), state)
    now[0] += .5
    evaluate_enhancement(selection(external), state)
    assert external.remaining == [2.0, 1.5]


@pytest.mark.parametrize("mode", ["rating", "simulation", "snapshot"])
@pytest.mark.parametrize("hydraulic_reference", ["bulk", "film"])
def test_required_wall_is_independent_of_bulk_reference(external, mode, hydraulic_reference):
    b = bundle(passes=2)
    properties = VariableLiquid()
    external.requires_wall_state = True
    external.hydraulic_property_reference = hydraulic_reference
    result = solve(BareTubeHeatExchanger(b), mode, inputs(b, provider=properties),
                   tube_side_enhancement=selection(external))
    assert external.requests
    for state in external.requests:
        assert state.wall is not None
        assert 300 <= state.wall.temperature <= 340
        assert state.wall.mu == properties.at(state.wall.temperature, state.wall.pressure).mu
        assert external.thermal_property_reference == "bulk"
        if hydraulic_reference == "film":
            assert state.hydraulic is not None and state.hydraulic is not state.wall
            assert state.hydraulic.temperature == pytest.approx(
                (state.bulk.temperature + state.wall.temperature) / 2)
            assert state.hydraulic.mu == properties.at(
                state.hydraulic.temperature, state.hydraulic.pressure).mu
        else:
            assert state.hydraulic is None
    assert {state.position for state in external.requests} >= {"thermal", "inlet", "midpoint", "outlet"}
    enhanced = result.tube_side_enhancement
    # Returned alpha already includes the wall factor; core consumes it once.
    assert enhanced.alpha_inside == 1000 * enhanced.wall_correction
    authoritative_alpha = result.alfa_i if mode == "rating" else result.inside_alfa_mean
    assert authoritative_alpha == enhanced.alpha_inside


def test_missing_required_wall_fails_without_invocation(external):
    external.requires_wall_state = True
    with pytest.raises(EnhancementUnsupportedError, match="wall_state_required"):
        evaluate_enhancement(selection(external), sample_input())
    with pytest.raises(EnhancementUnsupportedError, match="wall_state_required"):
        evaluate_enhancement(selection(external), replace(
            sample_input(), wall=replace(sample_input().bulk, temperature=None)))
    with pytest.raises(EnhancementUnsupportedError, match="wall_state_required"):
        evaluate_for_bundle(selection(external), bundle(), .1, sample_input().bulk,
                            temperature=300, pressure=2e5)
    assert external.requests == []


def test_smooth_path_does_not_consume_an_enhancement_budget():
    b = bundle()
    hx = BareTubeHeatExchanger(b)
    expected = hx.simulate(*inputs(b))
    actual = hx.simulate(*inputs(b), enhancement_operation_context=EnhancementOperationContext(-1))
    assert expected == actual
    with pytest.raises(TypeError, match="enhancement_operation_context"):
        hx.simulate(*inputs(b), enhancement_operation_context=object())


@pytest.mark.parametrize("composition", ["absolute", "correction"])
def test_composed_providers_share_context_and_wall_requirements(composition):
    from core.tests.twisted_tape_clearance_test import AbsoluteFixture, CorrectionFixture, configured

    states = []

    class RequiredAbsolute(AbsoluteFixture):
        requires_wall_state = True

        def evaluate(self, geometry, state, base_result):
            states.append(state)
            assert state.wall is not None
            return super().evaluate(geometry, state, base_result)

    class RequiredCorrection(CorrectionFixture):
        requires_wall_state = True

        def evaluate(self, geometry, state, base_result):
            states.append(state)
            assert state.wall is not None
            return super().evaluate(geometry, state, base_result)

    config = configured(RequiredAbsolute() if composition == "absolute" else RequiredCorrection())
    context = EnhancementOperationContext.from_timeout(30)
    b = bundle()
    result = BareTubeHeatExchanger(b).simulate(
        *inputs(b), tube_side_enhancement=config, enhancement_operation_context=context)
    assert all(state.operation_context is context and state.wall is not None for state in states)
    assert {state.position for state in states} >= {"thermal", "inlet", "midpoint", "outlet"}
    assert result.tube_side_enhancement.provider_id == config.clearance_provider.provider_id


def test_expired_base_call_does_not_invoke_clearance_provider(monkeypatch):
    from core.tests.twisted_tape_clearance_test import CorrectionFixture, configured

    context = EnhancementOperationContext(.001)
    now = [0.0]
    monkeypatch.setattr("core.enhancements.base.monotonic", lambda: now[0])
    config = configured(CorrectionFixture())
    original = config.provider.evaluate

    def late_base(geometry, state):
        result = original(geometry, state)
        now[0] = .005
        return result

    config.provider.evaluate = late_base
    config.clearance_provider.evaluate = lambda *args: pytest.fail("Late base invoked correction")
    with pytest.raises(EnhancementTimeoutError):
        evaluate_enhancement(config, replace(sample_input(), operation_context=context))


def test_invalid_required_wall_capability_fails_explicitly(external):
    external.requires_wall_state = "yes"
    with pytest.raises(EnhancementProviderError, match="requires_wall_state"):
        evaluate_enhancement(selection(external), sample_input())
    assert external.requests == []


@pytest.mark.parametrize("mode", ["rating", "simulation"])
@pytest.mark.parametrize("failure", ["unsupported", "backend", "timeout", "deadline", "invalid", "wrong_type"])
def test_endpoint_only_fatal_failure_propagates(external, monkeypatch, mode, failure):
    b = bundle()
    hx = BareTubeHeatExchanger(b)
    context = EnhancementOperationContext.from_timeout(30)
    original_evaluate = external.evaluate
    original_envelope = thermal.estimate_wall_temperature_envelope
    in_envelope = [False]
    entered = []

    def endpoint_failure(geometry, state):
        if in_envelope[0] and state.bulk.temperature == 300:
            entered.append(state)
            if failure == "unsupported":
                raise EnhancementUnsupportedError("endpoint unsupported")
            if failure == "backend":
                raise RuntimeError("endpoint session unavailable")
            if failure == "timeout":
                raise TimeoutError("endpoint transport timeout")
            if failure == "deadline":
                EnhancementOperationContext(monotonic() - 1).check_deadline()
            if failure == "wrong_type":
                return object()
            return replace(original_evaluate(geometry, state), alpha_inside=math.nan)
        return original_evaluate(geometry, state)

    def envelope(selected, *args, **kwargs):
        assert len(external.requests) > 3  # All primary thermal/hydraulic calls succeeded.
        in_envelope[0] = True
        return original_envelope(selected, *args, **kwargs)

    external.evaluate = endpoint_failure
    monkeypatch.setattr(rating if mode == "rating" else simulation,
                        "estimate_wall_temperature_envelope", envelope)
    forbid_smooth_fallback(monkeypatch)
    expected = (EnhancementUnsupportedError if failure == "unsupported" else
                EnhancementTimeoutError if failure in ("timeout", "deadline") else
                EnhancementProviderError)
    with pytest.raises(expected):
        solve(hx, mode, inputs(b), tube_side_enhancement=selection(external),
              enhancement_operation_context=context)
    assert entered
    assert hx.tube_side_enhancement is None
    assert hx.enhancement_operation_context is None


@pytest.mark.parametrize("mode", ["rating", "simulation"])
def test_optional_numerical_probe_failure_remains_a_warning(external, monkeypatch, mode):
    original_envelope = thermal.estimate_wall_temperature_envelope
    original_probe = thermal._solve_wall_temperature_probe

    def numerical_failure(*args, **kwargs):
        raise ArithmeticError("optional numerical envelope estimate unavailable")

    def envelope(*args, **kwargs):
        with monkeypatch.context() as local:
            local.setattr(thermal, "_solve_wall_temperature_probe", numerical_failure)
            return original_envelope(*args, **kwargs)

    monkeypatch.setattr(rating if mode == "rating" else simulation,
                        "estimate_wall_temperature_envelope", envelope)
    b = bundle()
    result = solve(BareTubeHeatExchanger(b), mode, inputs(b),
                   tube_side_enhancement=selection(external))
    assert result.tube_side_enhancement.provider_id == external.provider_id
    assert "wall_temperature_probe_not_converged" in {warning.code for warning in result.warnings}
    assert thermal._solve_wall_temperature_probe is original_probe


def test_numerical_probe_iteration_exhaustion_is_not_a_provider_failure(external):
    b = bundle()
    inside, outside = inputs(b)
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=selection(external))
    envelope = thermal.estimate_wall_temperature_envelope(
        hx, m_dot_inside=inside.m_dot, m_dot_outside=outside.m_dot,
        inside_provider=inside.provider, outside_provider=outside.provider,
        inside_inlet_temperature=300, inside_outlet_temperature=301,
        outside_inlet_temperature=340, outside_outlet_temperature=330,
        p_inside=inside.p, p_outside=outside.p, max_iterations=1,
    )
    assert external.requests  # Provider evaluations succeeded; only convergence was missing.
    assert all(not probe.converged for probe in envelope.probes)
    assert "wall_temperature_probe_not_converged" in {warning.code for warning in envelope.warnings}
