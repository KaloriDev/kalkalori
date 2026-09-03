# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Deterministic regressions for core.psychrometrics.

core/psychrometrics previously had no .py pytest coverage and was exercised
only manually through a notebook (since removed). This module ports the
notebook's core assertions (RH/W round trip, saturation, condensation onset,
above-boiling behavior) into real, synthetic pytest checks so the notebook
could be removed without losing coverage.
"""

from __future__ import annotations

import math

import pytest

from core.psychrometrics.condensation import check_condensation_onset
from core.psychrometrics.moist_air import (
    moist_air_state_from_t_rh,
    moist_air_state_from_t_w,
    moist_air_state_from_t_w_g_per_kg_da,
    saturated_moist_air_state,
)
from core.psychrometrics.psychrolib_adapter import (
    humidity_ratio_from_t_rh,
    max_relative_humidity_at_t_p,
    saturation_humidity_ratio,
)
from core.psychrometrics.wet_process import wet_surface_process_limit

P_ATM = 101_325.0


def test_relative_humidity_round_trip_through_humidity_ratio() -> None:
    forward = moist_air_state_from_t_rh(T=298.15, RH=0.5, p=P_ATM)
    back = moist_air_state_from_t_w(T=forward.T, W=forward.W, p=forward.p)

    assert math.isclose(back.RH, 0.5, rel_tol=1e-9)
    assert math.isclose(back.T_dew, forward.T_dew, rel_tol=1e-9)

    # Sanity range for one representative near-ambient wet-air point.
    assert 1.0 < forward.rho < 1.3
    assert forward.h > 0.0
    assert forward.T_dew < forward.T


def test_saturated_state_has_full_humidity_and_dew_point_equal_to_dry_bulb() -> None:
    sat = saturated_moist_air_state(T=283.15, p=P_ATM)

    assert math.isclose(sat.RH, 1.0, rel_tol=1e-9)
    assert math.isclose(sat.T_dew, sat.T, abs_tol=1e-6)


def test_condensation_onset_is_detected_below_dew_point() -> None:
    air = moist_air_state_from_t_rh(T=298.15, RH=0.5, p=P_ATM)

    result = check_condensation_onset(air=air, T_surface=280.0)

    assert result.will_condense is True
    assert result.dew_point_margin > 0.0
    assert result.W_surface_sat < result.W_bulk
    assert "CONDENSATION_ONSET" in [w.code for w in result.warnings]


def test_condensation_onset_is_not_detected_above_dew_point() -> None:
    air = moist_air_state_from_t_rh(T=298.15, RH=0.5, p=P_ATM)

    result = check_condensation_onset(air=air, T_surface=290.0)

    assert result.will_condense is False
    assert result.dew_point_margin < 0.0
    assert result.warnings == []


def test_surface_below_freezing_warns_regardless_of_condensation() -> None:
    air = moist_air_state_from_t_rh(T=298.15, RH=0.5, p=P_ATM)

    result = check_condensation_onset(air=air, T_surface=260.0)

    assert result.will_condense is True
    assert "SURFACE_BELOW_FREEZING" in [w.code for w in result.warnings]


def test_wet_surface_process_limit_condensing_case_removes_water_and_enthalpy() -> None:
    air = moist_air_state_from_t_rh(T=298.15, RH=0.5, p=P_ATM)

    condensing = wet_surface_process_limit(air=air, T_surface=280.0)

    assert condensing.condensable_water > 0.0
    assert condensing.enthalpy_drop > 0.0
    assert condensing.equilibrium_state.W < air.W


def test_wet_surface_process_limit_dry_case_has_no_condensate() -> None:
    air = moist_air_state_from_t_rh(T=298.15, RH=0.5, p=P_ATM)

    dry = wet_surface_process_limit(air=air, T_surface=290.0)

    assert dry.condensable_water == 0.0
    assert dry.equilibrium_state.W == pytest.approx(air.W)
    assert dry.enthalpy_drop > 0.0


def test_saturation_humidity_ratio_rejects_state_above_boiling_temperature() -> None:
    with pytest.raises(ValueError, match="boiling"):
        saturation_humidity_ratio(T=380.0, p=P_ATM)


def test_relative_humidity_input_is_rejected_when_supersaturated_above_boiling() -> None:
    rh_max = max_relative_humidity_at_t_p(T=380.0, p=P_ATM)
    assert rh_max < 1.0

    with pytest.raises(ValueError, match="not physically valid"):
        humidity_ratio_from_t_rh(T=380.0, RH=0.9, p=P_ATM)


def test_humidity_ratio_input_form_remains_valid_above_boiling_temperature() -> None:
    # Above the local boiling point, RH becomes ill-posed (see the two tests
    # above) while W stays a valid, unambiguous moisture-content input — the
    # documented reason `moist_air_state_from_t_w_g_per_kg_da` exists.
    hot = moist_air_state_from_t_w_g_per_kg_da(T=380.0, W_g_per_kg_da=80.0, p=P_ATM)

    assert hot.W == pytest.approx(0.08)
    assert 0.0 < hot.RH < 1.0
    assert hot.T_dew < hot.T
