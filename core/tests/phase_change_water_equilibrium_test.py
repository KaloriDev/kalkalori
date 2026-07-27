# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for core.phase_change.water_equilibrium (v0.6.0).

Uses IAPWS-IF97 (via core.properties.water) directly -- these are the
algorithmic/equilibrium tests, deterministic given the IAPWS-IF97 backend
(the `iapws` package is a required dependency, not optional, so no skip
guard is needed here).

Run:
    pytest -q core/tests/phase_change_water_equilibrium_test.py
"""

from __future__ import annotations

import math

import pytest

from core.phase_change.water_equilibrium import (
    dry_gas_average_molar_mass,
    is_frost_regime,
    saturated_water_ratio,
    water_dew_point,
    water_mole_fraction_from_ratio,
    water_partial_pressure,
)

M_AIR = 28.9647e-3  # kg/mol, dry air


def test_water_partial_pressure_basic() -> None:
    assert water_partial_pressure(0.02, 101325.0) == pytest.approx(2026.5, rel=1e-9)


def test_water_partial_pressure_rejects_p_h2o_at_or_above_total() -> None:
    with pytest.raises(ValueError):
        water_partial_pressure(1.0, 101325.0)
    with pytest.raises(ValueError):
        water_partial_pressure(1.5, 101325.0)


def test_dew_point_round_trip_matches_saturation_pressure() -> None:
    from core.properties.water import water_saturation_pressure

    p_h2o = 2339.0  # ~ saturation pressure of water at 20 degC
    T_dew = water_dew_point(p_h2o)
    assert water_saturation_pressure(T_dew) == pytest.approx(p_h2o, rel=1e-6)
    assert T_dew == pytest.approx(293.15, abs=0.1)


def test_dew_point_below_triple_point_pressure_raises() -> None:
    with pytest.raises(ValueError):
        water_dew_point(1.0)  # far below the triple-point saturation pressure


def test_is_frost_regime() -> None:
    assert is_frost_regime(1.0) is True  # 1 Pa is far below the triple point
    assert is_frost_regime(2339.0) is False  # ~20 degC dew point


def test_saturated_water_ratio_reduces_to_classic_dry_air_formula() -> None:
    """W_sat = 0.622*p_sat/(p-p_sat) is the M_dry=M_air special case."""
    from core.properties.water import water_saturation_pressure

    T = 293.15
    p_total = 101325.0
    p_sat = water_saturation_pressure(T)
    classic = 0.621945 * p_sat / (p_total - p_sat)

    W = saturated_water_ratio(p_total=p_total, T=T, M_dry=M_AIR)
    assert W == pytest.approx(classic, rel=2e-3)


def test_saturated_water_ratio_for_non_air_dry_gas() -> None:
    """A heavier dry gas (e.g. CO2, M=44 g/mol) must give a smaller W_sat
    than dry air at the same T, p (heavier carrier -> less water per kg)."""
    T, p_total = 293.15, 101325.0
    W_air = saturated_water_ratio(p_total=p_total, T=T, M_dry=M_AIR)
    W_co2 = saturated_water_ratio(p_total=p_total, T=T, M_dry=44.0095e-3)
    assert W_co2 < W_air
    assert W_co2 == pytest.approx(W_air * (M_AIR / 44.0095e-3), rel=1e-9)


def test_saturated_water_ratio_rejects_boiling_point_and_above() -> None:
    with pytest.raises(ValueError):
        saturated_water_ratio(p_total=101325.0, T=400.0, M_dry=M_AIR)


def test_dry_gas_average_molar_mass_weighted() -> None:
    M = dry_gas_average_molar_mass({"Nitrogen": 0.8, "Oxygen": 0.2})
    expected = 0.8 * 28.0134e-3 + 0.2 * 31.9988e-3
    assert M == pytest.approx(expected, rel=1e-9)


def test_water_mole_fraction_from_ratio_round_trip() -> None:
    y = 0.05
    W = (y / (1.0 - y)) * (18.01528e-3 / M_AIR)
    y_back = water_mole_fraction_from_ratio(W, M_dry=M_AIR, M_h2o=18.01528e-3)
    assert y_back == pytest.approx(y, rel=1e-9)


def test_water_partial_pressure_rejects_nonfinite_and_negative() -> None:
    with pytest.raises(ValueError):
        water_partial_pressure(math.nan, 101325.0)
    with pytest.raises(ValueError):
        water_partial_pressure(-0.1, 101325.0)
