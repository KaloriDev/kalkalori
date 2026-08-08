# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for core.phase_change.wet_gas_enthalpy (v0.6.0).

Round-trip T -> h(T,p,W) -> T, continuity in W, and the W=0 (no water)
reduction. Requires CoolProp (via GasMixturePropertyProvider) -- these are
integration-level tests against the real property backend; they are
explicitly not required to be CoolProp-free per the v0.6.0 spec's test
plan (only the pure equilibrium/algorithmic tests are).

Run:
    pytest -q core/tests/phase_change_wet_gas_enthalpy_test.py
"""

from __future__ import annotations

import pytest

from core.properties.gas_mixture import GasMixtureSpec, GasMixturePropertyProvider, check_coolprop_backend_availability
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.wet_gas_enthalpy import (
    dry_gas_enthalpy,
    h_wet_gas_dry_basis,
    temperature_from_h_wet_gas_dry_basis,
)

pytestmark = pytest.mark.skipif(
    not check_coolprop_backend_availability("HEOS").available,
    reason="CoolProp HEOS backend is not available in this environment.",
)


def _capability():
    spec = GasMixtureSpec(components={"N2": 0.75, "O2": 0.15, "CO2": 0.02, "H2O": 0.08}, basis="mole")
    return detect_phase_change_capability(GasMixturePropertyProvider(spec))


def test_round_trip_temperature_enthalpy_temperature() -> None:
    cap = _capability()
    p = 101325.0
    for T in (300.0, 350.0, 420.0):
        h = h_wet_gas_dry_basis(T, p, cap.W_in, cap)
        T_back = temperature_from_h_wet_gas_dry_basis(h, p, cap.W_in, cap)
        assert T_back == pytest.approx(T, abs=1e-3)


def test_enthalpy_continuity_for_small_change_in_W() -> None:
    cap = _capability()
    T, p = 380.0, 101325.0
    h0 = h_wet_gas_dry_basis(T, p, cap.W_in, cap)
    h1 = h_wet_gas_dry_basis(T, p, cap.W_in + 1e-6, cap)
    # A tiny change in W must produce a tiny (not discontinuous) change in h.
    assert abs(h1 - h0) < 10.0


def test_case_W_out_equals_W_in_gives_identical_enthalpy() -> None:
    cap = _capability()
    T, p = 400.0, 101325.0
    assert h_wet_gas_dry_basis(T, p, cap.W_in, cap) == h_wet_gas_dry_basis(T, p, cap.W_in, cap)


def test_W_zero_reduces_to_dry_gas_enthalpy() -> None:
    cap = _capability()
    T, p = 400.0, 101325.0
    assert h_wet_gas_dry_basis(T, p, 0.0, cap) == pytest.approx(dry_gas_enthalpy(T, p, cap), rel=1e-9)


def test_rejects_negative_W() -> None:
    cap = _capability()
    with pytest.raises(ValueError):
        h_wet_gas_dry_basis(350.0, 101325.0, -0.01, cap)


def test_inversion_raises_when_not_bracketed() -> None:
    cap = _capability()
    h_absurd = -1.0e12
    with pytest.raises(ValueError):
        temperature_from_h_wet_gas_dry_basis(h_absurd, 101325.0, cap.W_in, cap)
