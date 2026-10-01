# SPDX-License-Identifier: GPL-3.0-only
"""Provider plumbing and exact default/explicit numerical freeze."""
from dataclasses import replace

import pytest

from core import (ElmahdyMitalasWetCoilProvider, WetCoilSolverOptions,
                  WetCoilProviderUnsupportedError, WetCoilTimeoutError)
from core.heat_transfer import wet_coil_solver as controls
from core.phase_change import wet_coil_integration as engine
from core.phase_change.types import PhaseChangeMode
from core.tests.wet_coil_public_test import context, specs


class FakeProvider:
    model_id = "test_plumbing"
    model_name = "Deterministic plumbing fixture"
    source = "Synthetic test data; not a physical model"
    applicability = "Test inputs only"
    supports_simulation = True
    supports_rating = True

    def __init__(self, simulation, rating):
        self.simulation, self.rating = simulation, rating
        self.calls = []
        self.applicable = True
        self.failure = None
        self.nest = False

    def is_applicable(self, *args, **kwargs):
        return self.applicable

    def _solve(self, mode, hx, inside, outside, *, context, wet_solver_options, **options):
        self.calls.append((mode, context, wet_solver_options, options))
        assert controls._active_budget.get() is context
        if self.failure:
            raise self.failure
        if mode == "rating" and self.nest:
            from core.models.simulation import HXSideInput
            hx.simulate(HXSideInput(inside.provider, inside.m_dot, inside.T_in, inside.p),
                        HXSideInput(outside.provider, outside.m_dot, outside.T_in, outside.p),
                        wet_solver_options=wet_solver_options, wet_coil_provider=self)
        result = self.simulation if mode == "simulation" else self.rating
        if result is None:
            return None
        return replace(result, wet_coil_diagnostics={"surface": {"wet_regime": "DRY"}})

    def simulate(self, *args, **kwargs):
        return self._solve("simulation", *args, **kwargs)

    def rate(self, *args, **kwargs):
        return self._solve("rating", *args, **kwargs)


@pytest.fixture(scope="module")
def dry_results():
    hx, a, b = context(W=0.004)
    disabled = replace(b, phase_change_mode=PhaseChangeMode.DISABLED)
    sim = hx.simulate(a, disabled)
    si, so = specs(a, disabled, sim.T_out_outside)
    rating = hx.rate(si, replace(so, phase_change_mode=PhaseChangeMode.DISABLED))
    return sim, rating


@pytest.mark.parametrize("mode", ["simulation", "rating"])
def test_dispatch_and_metadata(mode, dry_results, monkeypatch):
    hx, a, b = context()
    provider = FakeProvider(*dry_results)
    options = WetCoilSolverOptions(timeout_s=27)
    monkeypatch.setattr(engine, "_run_elmahdy", lambda *a, **k: pytest.fail("fallback"))
    if mode == "simulation":
        result = hx.simulate(a, b, surface_margin=0.2,
                             wet_coil_provider=provider, wet_solver_options=options)
        assert provider.calls[0][3]["surface_margin"] == 0.2
    else:
        provider.nest = True
        result = hx.rate(*specs(a, b, 294), wet_coil_provider=provider,
                         wet_solver_options=options)
        assert [c[0] for c in provider.calls] == ["rating", "simulation"]
    budget = provider.calls[0][1]
    assert all(c[1] is budget and c[2] is options for c in provider.calls)
    assert budget.deadline == budget.started + 27
    assert controls._active_budget.get() is None
    d = result.wet_coil_diagnostics
    assert d["provider"]["source"] == provider.source
    assert d["global_wet_model"] == provider.model_id
    assert d["surface"] == {"wet_regime": "DRY"}
    assert result.outside_phase_change.wet_coil_diagnostics is d


def test_unsupported_inapplicable_and_failure_never_fall_back(dry_results, monkeypatch):
    hx, a, b = context()
    provider = FakeProvider(*dry_results)
    monkeypatch.setattr(engine, "_run_elmahdy", lambda *a, **k: pytest.fail("fallback"))
    provider.supports_rating = False
    with pytest.raises(WetCoilProviderUnsupportedError, match="rating"):
        hx.rate(*specs(a, b, 294), wet_coil_provider=provider)
    provider.supports_simulation = False
    with pytest.raises(WetCoilProviderUnsupportedError, match="simulation"):
        hx.simulate(a, b, wet_coil_provider=provider)
    provider.supports_simulation = True
    provider.applicable = False
    with pytest.raises(WetCoilProviderUnsupportedError, match="not applicable"):
        hx.simulate(a, b, wet_coil_provider=provider)
    assert not provider.calls
    provider.applicable = True
    provider.failure = RuntimeError("provider failed")
    with pytest.raises(RuntimeError, match="provider failed"):
        hx.simulate(a, b, wet_coil_provider=provider)
    provider.failure = None
    provider.simulation = None
    with pytest.raises(TypeError, match="must return HXSimulationResult"):
        hx.simulate(a, b, wet_coil_provider=provider)


@pytest.mark.parametrize("mode", ["simulation", "rating"])
def test_disabled_bypasses_provider(mode, dry_results):
    hx, a, b = context(W=0.004)
    b = replace(b, phase_change_mode=PhaseChangeMode.DISABLED)
    provider = FakeProvider(*dry_results)
    provider.supports_simulation = provider.supports_rating = False
    if mode == "simulation":
        result = hx.simulate(a, b, wet_coil_provider=provider)
    else:
        si, so = specs(a, b, dry_results[0].T_out_outside)
        result = hx.rate(si, replace(so, phase_change_mode=PhaseChangeMode.DISABLED),
                         wet_coil_provider=provider)
    assert not provider.calls
    assert result.wet_coil_diagnostics is None


def test_shared_deadline_is_enforced(dry_results, monkeypatch):
    hx, a, b = context()
    provider = FakeProvider(*dry_results)
    now = [10.0]
    monkeypatch.setattr(controls, "monotonic", lambda: now[0])
    original = provider.simulate

    def expires(*args, **kwargs):
        result = original(*args, **kwargs)
        now[0] = 12.0
        return result

    monkeypatch.setattr(provider, "simulate", expires)
    with pytest.raises(WetCoilTimeoutError):
        hx.simulate(a, b, wet_coil_provider=provider,
                    wet_solver_options=WetCoilSolverOptions(timeout_s=1))
    assert controls._active_budget.get() is None


@pytest.mark.parametrize("finned,W", [(False, 0.004), (False, 0.016), (True, 0.016)])
def test_default_explicit_numerical_freeze(finned, W):
    hx, a, b = context(finned, W)
    default = hx.simulate(a, b, surface_margin=0.1)
    explicit = hx.simulate(a, b, surface_margin=0.1,
                           wet_coil_provider=ElmahdyMitalasWetCoilProvider())
    for name in ("q", "T_out_inside", "T_out_outside", "UA", "Q_full", "Q_derated"):
        assert getattr(default, name) == getattr(explicit, name)
    for name in ("W_out", "m_dot_condensate", "H_drain", "wet_surface_fraction"):
        assert getattr(default.outside_phase_change, name) == getattr(explicit.outside_phase_change, name)
    surface = explicit.wet_coil_diagnostics["surface"]
    assert surface == default.wet_coil_diagnostics["surface"]
    assert surface["surface_temperature_min"] == explicit.outside_phase_change.wall_temperature_min
    assert surface["onset_margin"] == explicit.outside_phase_change.onset_margin_K
    assert surface["metal_wall_temperature"] == tuple(
        p.outside_wall_temperature for p in explicit.wall_temperature_envelope.probes)
    if finned:
        assert surface["fin_tip_temperature"]
        assert all(0 <= p["fraction"] <= 1 for p in surface["radial_wet_fraction"])
    else:
        assert "fin_tip_temperature" not in surface
    if not finned:
        args = specs(a, b, default.T_out_outside)
        r0 = hx.rate(*args, include_simulation=True)
        r1 = hx.rate(*args, include_simulation=True,
                     wet_coil_provider=ElmahdyMitalasWetCoilProvider())
        for name in ("A_required", "A_o", "UA_required", "UA_actual", "Q_required", "Q_achievable"):
            assert getattr(r0, name) == getattr(r1, name)
        assert r1.simulation.q == r0.simulation.q
        assert r1.simulation.wet_coil_diagnostics["provider"] == r1.wet_coil_diagnostics["provider"]
