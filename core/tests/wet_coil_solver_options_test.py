# SPDX-License-Identifier: GPL-3.0-only
"""Engineering defaults, explicit precision, and shared operation deadlines."""
from dataclasses import FrozenInstanceError, replace

import pytest

from core import WetCoilSolverOptions, WetCoilTimeoutError
from core.heat_transfer import wet_coil_solver as controls
from core.phase_change import wet_coil_integration as integration
from core.tests.wet_coil_public_test import context, specs


def test_defaults_and_unlimited_timeout():
    options = WetCoilSolverOptions()
    assert (options.energy_tolerance_W, options.mass_tolerance_kg_s,
            options.outlet_temperature_tolerance_K, options.timeout_s) == (1.0, 5e-7, 0.01, 300.0)
    assert replace(options, timeout_s=None).timeout_s is None
    with pytest.raises(FrozenInstanceError):
        options.timeout_s = 4.0
    with pytest.raises(TypeError):
        WetCoilSolverOptions(unknown=1)
    for name in ("energy_tolerance_W", "mass_tolerance_kg_s", "outlet_temperature_tolerance_K"):
        with pytest.raises(ValueError, match=name):
            WetCoilSolverOptions(**{name: None})


@pytest.mark.parametrize("field", WetCoilSolverOptions.__dataclass_fields__)
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_controls(field, value):
    with pytest.raises(ValueError, match=field):
        WetCoilSolverOptions(**{field: value})


def test_default_forward_and_inverse_controls():
    hx, a, b = context()
    sim = hx.simulate(a, b)
    options = sim.wet_coil_diagnostics["solver_options"]
    assert options == WetCoilSolverOptions()
    assert sim.outside_water_ratio_out <= sim.outside_phase_change.W_in
    assert sim.outside_condensate_mass_flow >= 0
    assert abs(sim.wet_coil_diagnostics["energy_residual"]) <= options.energy_tolerance_W
    installed = integration.required_exchanger(hx, 2.2)
    rating = installed.rate(*specs(a, b, sim.T_out_outside), wet_solver_options=options)
    joint = installed.rate(*specs(a, b, sim.T_out_outside, sim.T_out_inside, False),
                           wet_solver_options=options)
    for result in (rating, joint):
        assert result.wet_coil_diagnostics["solver_options"] is options
        assert abs(result.closed_balance.outside.T_out - sim.T_out_outside) <= 0.01
    assert abs(joint.closed_balance.inside.T_out - sim.T_out_inside) <= 0.01
    assert joint.wet_coil_diagnostics["solver_statistics"]["joint_solver_evaluations"] > 0


def test_rating_timeout_does_not_reset_between_forward_trials(monkeypatch):
    hx, a, b = context()
    now = [0.0]
    monkeypatch.setattr(controls, "monotonic", lambda: now[0])
    options = WetCoilSolverOptions(timeout_s=3.0)
    original = integration.forward_wet_process
    budgets = []

    def forward(*args, **kwargs):
        budget = kwargs["_budget"]
        budgets.append(budget)
        assert kwargs["wet_solver_options"] is options
        result = original(*args, **kwargs)
        now[0] += 2.0
        return result

    monkeypatch.setattr(integration, "forward_wet_process", forward)
    with pytest.raises(WetCoilTimeoutError) as caught:
        hx.rate(*specs(a, b, 294.0), wet_solver_options=options)
    assert len(budgets) == 2
    assert budgets[0] is budgets[1]
    d = caught.value.diagnostics
    assert d["elapsed_s"] == 4.0
    assert d["timeout_s"] == 3.0
    assert d["forward_evaluations"] == 2
    assert d["required_effective_length"] > 0
    assert d["inside_mass_flow"] == a.m_dot
    assert d["last_regime"]
    assert controls._active_budget.get() is None


def test_timeout_reaches_profile_iteration_and_unlimited_works(monkeypatch):
    from core.heat_transfer.wet_coil import solve_wet_coil
    from core.tests.wet_coil_test import physical_case, ws

    now = [0.0]

    def clock():
        now[0] += 0.25
        return now[0]

    monkeypatch.setattr(controls, "monotonic", clock)
    kwargs = dict(liquid_enthalpy=lambda t: 4180 * t, saturation_humidity=ws)
    with pytest.raises(WetCoilTimeoutError) as caught:
        solve_wet_coil(physical_case(0.016), **kwargs,
                       wet_solver_options=WetCoilSolverOptions(timeout_s=2.0))
    assert caught.value.diagnostics["wet_fraction"] > 0
    result = solve_wet_coil(physical_case(0.016), **kwargs,
                            wet_solver_options=WetCoilSolverOptions(timeout_s=None))
    assert result.regime == "FULLY_WET"


def test_reserve_and_rating_simulation_share_options_and_deadline(monkeypatch):
    hx, a, b = context(W=0.004)
    options = WetCoilSolverOptions(timeout_s=None)
    budgets = []
    original = integration.forward_wet_process

    def forward(*args, **kwargs):
        budgets.append(kwargs["_budget"])
        assert kwargs["wet_solver_options"] is options
        return original(*args, **kwargs)

    monkeypatch.setattr(integration, "forward_wet_process", forward)
    sim = hx.simulate(a, b, surface_margin=0.1, wet_solver_options=options)
    assert len(budgets) == 2 and budgets[0] is budgets[1]
    assert sim.wet_coil_diagnostics["solver_statistics"]["forward_evaluations"] == 2
    budgets.clear()
    target = original(hx, a, b, wet_solver_options=options)[0].air_out
    rated = hx.rate(*specs(a, b, target), include_simulation=True, wet_solver_options=options)
    assert len(budgets) >= 2 and all(v is budgets[0] for v in budgets)
    assert rated.simulation.wet_coil_diagnostics["solver_options"] is options


def test_reused_saturation_brackets_preserve_property_inverse_precision():
    from core.tests.wet_coil_test import production_context

    _, thermo, _ = production_context()
    # Nonmonotone requests exercise both sides of cached brackets and their
    # coalescing endpoints without changing the configured property equation.
    for temperature in (310.0, 285.0, 320.0, 300.0, 310.0 + 1e-10, 310.0):
        enthalpy = thermo.saturation_enthalpy(temperature)
        recovered = thermo.saturation_temperature(enthalpy)
        assert recovered == pytest.approx(temperature, abs=5e-12, rel=0)
        assert thermo.saturation_enthalpy(recovered) == pytest.approx(enthalpy, abs=1e-6, rel=0)
