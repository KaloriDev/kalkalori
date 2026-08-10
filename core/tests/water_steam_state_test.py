# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for core.properties.water.water_steam_state (v0.6.2).

Uses IAPWS-IF97 (via core.properties.water) directly -- these are
deterministic given the IAPWS-IF97 backend (the `iapws` package is a
required dependency, not optional, so no skip guard is needed here).

Run:
    pytest -q core/tests/water_steam_state_test.py
"""

from __future__ import annotations

import math

import pytest

from core.properties.water import (
    WATER_CRITICAL_PRESSURE_PA,
    WaterPhaseRegion,
    water_steam_props_iapws97,
    water_steam_state,
)

P_ATM = 101_325.0  # Pa, ~ saturation temperature 373.12 K


# ---------------------------------------------------------------------------
# Existing T+p behavior is unchanged
# ---------------------------------------------------------------------------
def test_existing_T_p_water_props_unchanged() -> None:
    # water_steam_props_iapws97 (the pre-v0.6.2 low-level adapter) must keep
    # working exactly as before; water_steam_state is an additive layer.
    props = water_steam_props_iapws97(T=300.0, p=P_ATM)
    assert props.transport.rho > 900.0
    assert props.phase in {"liquid", "compressed_liquid"} or "liquid" in props.phase


def test_T_p_superheated_matches_low_level_adapter() -> None:
    T, p = 473.15, P_ATM
    state = water_steam_state(T=T, p=p)
    props = water_steam_props_iapws97(T=T, p=p)
    assert state.h == pytest.approx(props.h, rel=1e-12)
    assert state.rho == pytest.approx(props.transport.rho, rel=1e-12)


# ---------------------------------------------------------------------------
# T+p classification
# ---------------------------------------------------------------------------
def test_T_p_superheated_vapor() -> None:
    state = water_steam_state(T=473.15, p=P_ATM)
    assert state.phase is WaterPhaseRegion.SUPERHEATED_VAPOR
    assert state.quality is None
    assert state.T_sat is not None and state.T_sat < state.T
    assert state.cp is not None and state.mu is not None and state.k is not None
    assert state.Pr is not None and state.Pr > 0.0


def test_T_p_subcooled_liquid() -> None:
    state = water_steam_state(T=280.0, p=P_ATM)
    assert state.phase is WaterPhaseRegion.SUBCOOLED_LIQUID
    assert state.quality is None
    assert state.T_sat is not None and state.T_sat > state.T
    assert state.rho is not None and state.rho > 900.0


def test_T_p_ambiguous_saturation_rejected() -> None:
    from core.properties.water import water_saturation_temperature

    T_sat = water_saturation_temperature(P_ATM)
    with pytest.raises(ValueError, match="p\\+x or p\\+h"):
        water_steam_state(T=T_sat, p=P_ATM)


def test_T_p_well_away_from_saturation_not_ambiguous() -> None:
    from core.properties.water import water_saturation_temperature

    T_sat = water_saturation_temperature(P_ATM)
    # 1 K away is well outside the ambiguity band -- must not raise.
    state_hot = water_steam_state(T=T_sat + 1.0, p=P_ATM)
    state_cold = water_steam_state(T=T_sat - 1.0, p=P_ATM)
    assert state_hot.phase is WaterPhaseRegion.SUPERHEATED_VAPOR
    assert state_cold.phase is WaterPhaseRegion.SUBCOOLED_LIQUID


# ---------------------------------------------------------------------------
# p+x
# ---------------------------------------------------------------------------
def test_p_x_saturated_vapor() -> None:
    state = water_steam_state(p=P_ATM, x=1.0)
    assert state.phase is WaterPhaseRegion.SATURATED_VAPOR
    assert state.quality == 1.0
    assert state.h == pytest.approx(state.h_g, rel=1e-9)
    # Fully defined transport properties at the saturated-vapor endpoint.
    assert state.cp is not None and state.mu is not None and state.k is not None


def test_p_x_saturated_liquid() -> None:
    state = water_steam_state(p=P_ATM, x=0.0)
    assert state.phase is WaterPhaseRegion.SATURATED_LIQUID
    assert state.quality == 0.0
    assert state.h == pytest.approx(state.h_f, rel=1e-9)
    assert state.cp is not None and state.mu is not None and state.k is not None


def test_p_x_two_phase() -> None:
    state = water_steam_state(p=P_ATM, x=0.5)
    assert state.phase is WaterPhaseRegion.TWO_PHASE
    assert state.quality == 0.5
    assert state.h == pytest.approx(state.h_f + 0.5 * state.h_fg, rel=1e-12)
    # cp/mu/k are not physically defined single-phase quantities here.
    assert state.cp is None
    assert state.mu is None
    assert state.k is None
    assert state.Pr is None
    # rho (homogeneous mixture density) remains physically defined.
    assert state.rho is not None and state.rho > 0.0
    assert any(w.code == "WATER_STEAM_TWO_PHASE_STATE" for w in state.warnings)


def test_p_x_all_states_share_same_T_sat_h_f_h_g() -> None:
    s0 = water_steam_state(p=P_ATM, x=0.0)
    s1 = water_steam_state(p=P_ATM, x=1.0)
    sm = water_steam_state(p=P_ATM, x=0.3)
    assert s0.T_sat == pytest.approx(s1.T_sat, rel=1e-12) == pytest.approx(sm.T_sat, rel=1e-12)
    assert s0.h_f == pytest.approx(sm.h_f, rel=1e-12)
    assert s1.h_g == pytest.approx(sm.h_g, rel=1e-12)


# ---------------------------------------------------------------------------
# p+h
# ---------------------------------------------------------------------------
def test_p_h_superheated() -> None:
    hg = water_steam_state(p=P_ATM, x=1.0).h_g
    state = water_steam_state(p=P_ATM, h=hg + 100_000.0)
    assert state.phase is WaterPhaseRegion.SUPERHEATED_VAPOR
    assert state.quality is None
    assert state.T > state.T_sat


def test_p_h_two_phase() -> None:
    hf = water_steam_state(p=P_ATM, x=0.0).h_f
    hg = water_steam_state(p=P_ATM, x=1.0).h_g
    h_mid = 0.5 * (hf + hg)
    state = water_steam_state(p=P_ATM, h=h_mid)
    assert state.phase is WaterPhaseRegion.TWO_PHASE
    assert state.quality == pytest.approx(0.5, rel=1e-6)
    assert state.T == pytest.approx(state.T_sat, rel=1e-12)


def test_p_h_subcooled() -> None:
    hf = water_steam_state(p=P_ATM, x=0.0).h_f
    state = water_steam_state(p=P_ATM, h=hf - 100_000.0)
    assert state.phase is WaterPhaseRegion.SUBCOOLED_LIQUID
    assert state.quality is None
    assert state.T < state.T_sat


def test_p_h_exactly_h_f() -> None:
    hf = water_steam_state(p=P_ATM, x=0.0).h_f
    state = water_steam_state(p=P_ATM, h=hf)
    assert state.phase is WaterPhaseRegion.SATURATED_LIQUID
    assert state.quality == 0.0


def test_p_h_exactly_h_g() -> None:
    hg = water_steam_state(p=P_ATM, x=1.0).h_g
    state = water_steam_state(p=P_ATM, h=hg)
    assert state.phase is WaterPhaseRegion.SATURATED_VAPOR
    assert state.quality == 1.0


def test_p_x_to_h_to_p_h_round_trip() -> None:
    for x_in in (0.0, 0.1, 0.35, 0.5, 0.8, 1.0):
        s_x = water_steam_state(p=P_ATM, x=x_in)
        s_h = water_steam_state(p=P_ATM, h=s_x.h)
        assert s_h.phase == s_x.phase
        assert s_h.quality == pytest.approx(s_x.quality, abs=1e-9)
        assert s_h.T == pytest.approx(s_x.T, rel=1e-12)


# ---------------------------------------------------------------------------
# Continuity of h at the x=0 / x=1 boundaries
# ---------------------------------------------------------------------------
def test_h_continuous_at_saturated_liquid_boundary() -> None:
    h_f = water_steam_state(p=P_ATM, x=0.0).h_f
    just_above = water_steam_state(p=P_ATM, x=1e-9)
    subcooled = water_steam_state(p=P_ATM, h=h_f - 1.0)
    assert just_above.h == pytest.approx(h_f, abs=1.0)
    assert subcooled.h < h_f


def test_h_continuous_at_saturated_vapor_boundary() -> None:
    h_g = water_steam_state(p=P_ATM, x=1.0).h_g
    just_below = water_steam_state(p=P_ATM, x=1.0 - 1e-9)
    superheated = water_steam_state(p=P_ATM, h=h_g + 1.0)
    assert just_below.h == pytest.approx(h_g, abs=1.0)
    assert superheated.h > h_g


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_invalid_x_below_zero_rejected() -> None:
    with pytest.raises(ValueError):
        water_steam_state(p=P_ATM, x=-0.1)


def test_invalid_x_above_one_rejected() -> None:
    with pytest.raises(ValueError):
        water_steam_state(p=P_ATM, x=1.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(T=300.0, p=P_ATM, x=0.5),
        dict(T=300.0, p=P_ATM, h=1.0e5),
        dict(p=P_ATM, x=0.5, h=1.0e5),
        dict(T=300.0, p=P_ATM, x=0.5, h=1.0e5),
        dict(p=P_ATM),
    ],
)
def test_conflicting_or_incomplete_state_specification_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        water_steam_state(**kwargs)


def test_missing_pressure_rejected() -> None:
    with pytest.raises(ValueError):
        water_steam_state(T=300.0)


# ---------------------------------------------------------------------------
# Supercritical
# ---------------------------------------------------------------------------
def test_supercritical_p_x_unsupported() -> None:
    with pytest.raises(ValueError):
        water_steam_state(p=WATER_CRITICAL_PRESSURE_PA + 1.0, x=0.5)


def test_supercritical_p_h_unsupported() -> None:
    with pytest.raises(ValueError):
        water_steam_state(p=WATER_CRITICAL_PRESSURE_PA + 1.0, h=2.0e6)


def test_supercritical_T_p_still_works_as_ordinary_state() -> None:
    # T+p never needed the saturation curve; this must keep working, just
    # without saturation diagnostics.
    state = water_steam_state(T=700.0, p=WATER_CRITICAL_PRESSURE_PA + 1.0e6)
    assert state.phase is WaterPhaseRegion.SUPERCRITICAL
    assert state.quality is None
    assert state.T_sat is None
    assert state.h_f is None
    assert state.h_g is None
    assert state.h_fg is None
    assert math.isfinite(state.h)


# ---------------------------------------------------------------------------
# General invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
def test_quality_never_outside_unit_interval(x: float) -> None:
    state = water_steam_state(p=P_ATM, x=x)
    assert 0.0 <= state.quality <= 1.0


def test_quality_is_none_for_superheated_and_subcooled() -> None:
    hg = water_steam_state(p=P_ATM, x=1.0).h_g
    hf = water_steam_state(p=P_ATM, x=0.0).h_f
    assert water_steam_state(p=P_ATM, h=hg + 1000.0).quality is None
    assert water_steam_state(p=P_ATM, h=hf - 1000.0).quality is None
    assert water_steam_state(T=500.0, p=P_ATM).quality is None
    assert water_steam_state(T=280.0, p=P_ATM).quality is None
