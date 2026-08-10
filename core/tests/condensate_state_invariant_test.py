# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Robustness tests for condensate-state and enthalpy-only invariants."""

from __future__ import annotations

import pytest

from core.phase_change import condensation_solver_helpers as helpers
from core.phase_change.condensation_solver_helpers import (
    CondensateStateInconsistentError,
    condensate_enthalpy_flow,
)
from core.phase_change.warning_codes import CONDENSATE_STATE_INCONSISTENT
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec


def test_zero_condensate_does_not_evaluate_liquid_enthalpy(monkeypatch) -> None:
    def fail_if_called(*, T=None, p=None):
        raise AssertionError("liquid enthalpy must not be evaluated")

    monkeypatch.setattr(
        helpers,
        "water_saturation_liquid_enthalpy",
        fail_if_called,
    )
    assert condensate_enthalpy_flow(
        m_dot_condensate=0.0,
        condensation_mass_tolerance=1e-8,
        wet_surface_fraction=0.0,
        wet_area=0.0,
        wall_temperature_wet_mean=None,
    ) == 0.0
    assert condensate_enthalpy_flow(
        m_dot_condensate=1e-9,
        condensation_mass_tolerance=1e-8,
        wet_surface_fraction=0.0,
        wet_area=0.0,
        wall_temperature_wet_mean=None,
    ) == 0.0


def test_active_condensate_uses_the_wet_wall_temperature(monkeypatch) -> None:
    calls: list[float] = []

    def liquid_enthalpy(*, T=None, p=None):
        calls.append(T)
        return 123_456.0

    monkeypatch.setattr(
        helpers,
        "water_saturation_liquid_enthalpy",
        liquid_enthalpy,
    )
    flow = condensate_enthalpy_flow(
        m_dot_condensate=0.02,
        condensation_mass_tolerance=1e-8,
        wet_surface_fraction=0.25,
        wet_area=1.5,
        wall_temperature_wet_mean=315.0,
    )
    assert calls == [315.0]
    assert flow == pytest.approx(0.02 * 123_456.0)


def test_positive_condensate_without_wet_state_is_controlled_error() -> None:
    with pytest.raises(CondensateStateInconsistentError) as caught:
        condensate_enthalpy_flow(
            m_dot_condensate=0.02,
            condensation_mass_tolerance=1e-8,
            wet_surface_fraction=0.0,
            wet_area=0.0,
            wall_temperature_wet_mean=None,
        )
    assert caught.value.warning_code == CONDENSATE_STATE_INCONSISTENT


def test_enthalpy_only_coolprop_call_matches_full_snapshot() -> None:
    provider = GasMixturePropertyProvider(
        GasMixtureSpec(
            components={"N2": 0.79, "O2": 0.21},
            basis="mole",
        )
    )
    full = provider.full_at(T=330.0, p=101_325.0)
    assert provider.specific_enthalpy(T=330.0, p=101_325.0) == pytest.approx(
        full.h,
        rel=1e-13,
    )
    assert provider.at(T=330.0, p=101_325.0) == full.transport
