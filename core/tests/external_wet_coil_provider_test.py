# SPDX-License-Identifier: GPL-3.0-only
"""Prove a separate package needs only a provider object, no core changes."""
from pathlib import Path
from importlib import import_module

import pytest

from core import (WetCoilModelProvider, WetCoilProviderUnsupportedError,
                  WetCoilSolverOptions, WetCoilTimeoutError)
from core.heat_transfer import wet_coil_solver as controls
from core.phase_change import wet_coil_integration as engine
from core.tests.wet_coil_provider_test import dry_results
from core.tests.wet_coil_public_test import context, specs


@pytest.fixture
def external(dry_results, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tests" / "fixtures"))
    provider: WetCoilModelProvider = import_module("external_wet_provider").ExternalWetProvider(*dry_results)
    assert not type(provider).__module__.startswith("core")
    monkeypatch.setattr(engine, "_run_elmahdy", lambda *a, **k: pytest.fail("Elmahdy fallback"))
    return provider


@pytest.mark.parametrize("mode", ["simulation", "rating"])
def test_external_object_routes_with_shared_deadline_and_diagnostics(external, mode):
    hx, a, b = context()
    options = WetCoilSolverOptions(timeout_s=27)
    if mode == "simulation":
        result = hx.simulate(a, b, surface_margin=0.2,
                             wet_coil_provider=external, wet_solver_options=options)
        assert external.calls[0][3]["surface_margin"] == 0.2
    else:
        external.nested_simulation = True
        result = hx.rate(*specs(a, b, 294), wet_coil_provider=external,
                         wet_solver_options=options)
        assert [c[0] for c in external.calls] == ["rating", "simulation"]
        assert result.simulation.wet_coil_diagnostics["provider"]["model_id"] == external.model_id
    budget = external.calls[0][1]
    assert budget.deadline == budget.started + 27
    assert all(c[1] is budget and c[2] is options for c in external.calls)
    assert result.wet_coil_diagnostics["external_marker"] == mode
    assert result.wet_coil_diagnostics["provider"]["source"] == external.source
    assert result.outside_phase_change.method == external.model_id
    assert result.outside_phase_change.wet_coil_diagnostics is result.wet_coil_diagnostics
    assert controls._active_budget.get() is None


@pytest.mark.parametrize("mode", ["simulation", "rating"])
def test_external_unsupported_operation_never_falls_back(external, mode):
    hx, a, b = context()
    setattr(external, f"supports_{mode}", False)
    with pytest.raises(WetCoilProviderUnsupportedError, match=mode):
        if mode == "simulation":
            hx.simulate(a, b, wet_coil_provider=external)
        else:
            hx.rate(*specs(a, b, 294), wet_coil_provider=external)
    assert external.calls == []


def test_external_inapplicability_and_failure_never_fall_back(external):
    hx, a, b = context()
    external.applicable = False
    with pytest.raises(WetCoilProviderUnsupportedError, match="not applicable"):
        hx.simulate(a, b, wet_coil_provider=external)
    assert external.calls == []
    external.applicable = True
    external.failure = RuntimeError("external failure")
    with pytest.raises(RuntimeError, match="external failure"):
        hx.simulate(a, b, wet_coil_provider=external)
    assert controls._active_budget.get() is None


@pytest.mark.parametrize("mode", ["simulation", "rating"])
def test_external_deadline_enforced_after_provider_returns(external, mode, monkeypatch):
    hx, a, b = context()
    now = [10.0]
    monkeypatch.setattr(controls, "monotonic", lambda: now[0])
    original = external._solve

    def expire(*args, **kwargs):
        result = original(*args, **kwargs)
        now[0] = 12.0
        return result

    monkeypatch.setattr(external, "_solve", expire)
    options = WetCoilSolverOptions(timeout_s=1)
    with pytest.raises(WetCoilTimeoutError):
        if mode == "simulation":
            hx.simulate(a, b, wet_coil_provider=external, wet_solver_options=options)
        else:
            hx.rate(*specs(a, b, 294), wet_coil_provider=external, wet_solver_options=options)
    assert controls._active_budget.get() is None
