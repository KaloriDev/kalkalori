# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for core.phase_change.inside_pure_steam_zones (v0.6.2).

Multi-zone (desuperheat / condensation / subcooling) in-tube pure water/
steam solver. Deterministic given the IAPWS-IF97 backend; no external
validation dataset is used.

Run:
    pytest -q core/tests/inside_pure_steam_zones_test.py
"""

from __future__ import annotations

import math

import pytest

from core.phase_change.inside_pure_steam_zones import (
    ReverseDirectionEvaporationNotSupportedError,
    solve_inside_steam_zones,
)
from core.properties.water import WaterPhaseRegion, water_steam_state

P = 101_325.0
D_I = 0.02
FLOW_AREA = math.pi * D_I**2 / 4.0
D_O = 0.025
L_LONG = 40.0
L_SHORT = 1.0
K_WALL = 50.0
T_SINK = 280.0
ALPHA_OUTSIDE = 1500.0
M_DOT = 0.01


def _geometry(L: float):
    A_inside = math.pi * D_I * L
    A_outside = math.pi * D_O * L
    R_wall = math.log(D_O / D_I) / (2.0 * math.pi * K_WALL * L)
    return A_inside, A_outside, R_wall


def _solve(inlet, *, L: float = L_LONG, T_sink: float = T_SINK, m_dot: float = M_DOT):
    A_inside, A_outside, R_wall = _geometry(L)
    return solve_inside_steam_zones(
        p=P,
        inlet=inlet,
        m_dot_total=m_dot,
        D_i=D_I,
        flow_area=FLOW_AREA,
        A_inside_total=A_inside,
        A_outside_total=A_outside,
        R_wall_total=R_wall,
        T_sink=T_sink,
        alpha_outside=ALPHA_OUTSIDE,
    )


def _zone_kinds(result) -> list[str]:
    return [z.kind for z in result.zones]


# ---------------------------------------------------------------------------
# Zone-combination coverage (3B/3O)
# ---------------------------------------------------------------------------
def test_superheated_to_superheated_desuperheat_only() -> None:
    inlet = water_steam_state(T=450.0, p=P)
    result = _solve(inlet, L=L_SHORT / 20.0)
    assert result.phase_out is WaterPhaseRegion.SUPERHEATED_VAPOR
    assert _zone_kinds(result) == ["desuperheat"]


def test_superheated_to_partial_condensation() -> None:
    inlet = water_steam_state(T=450.0, p=P)
    result = _solve(inlet, L=5.0)
    assert result.phase_out is WaterPhaseRegion.TWO_PHASE
    assert _zone_kinds(result) == ["desuperheat", "condensation"]
    assert 0.0 < result.quality_out < 1.0


def test_superheated_to_saturated_liquid_or_beyond() -> None:
    inlet = water_steam_state(T=450.0, p=P)
    result = _solve(inlet, L=L_LONG)
    assert result.phase_out in (WaterPhaseRegion.SATURATED_LIQUID, WaterPhaseRegion.SUBCOOLED_LIQUID)
    assert _zone_kinds(result) == ["desuperheat", "condensation", "subcooling"]


def test_superheated_to_subcooled_liquid() -> None:
    inlet = water_steam_state(T=450.0, p=P)
    result = _solve(inlet, L=L_LONG)
    assert result.phase_out is WaterPhaseRegion.SUBCOOLED_LIQUID
    assert result.T_out < result.T_sat
    assert result.quality_out is None


def test_saturated_vapor_to_partial_condensation() -> None:
    inlet = water_steam_state(p=P, x=1.0)
    result = _solve(inlet, L=1.0)
    assert result.phase_out is WaterPhaseRegion.TWO_PHASE
    assert _zone_kinds(result) == ["condensation"]
    assert 0.0 < result.quality_out < 1.0


def test_saturated_vapor_to_complete_condensation() -> None:
    inlet = water_steam_state(p=P, x=1.0)
    result = _solve(inlet, L=L_LONG)
    assert result.phase_out in (WaterPhaseRegion.SATURATED_LIQUID, WaterPhaseRegion.SUBCOOLED_LIQUID)


def test_saturated_vapor_to_subcooled_liquid() -> None:
    inlet = water_steam_state(p=P, x=1.0)
    result = _solve(inlet, L=L_LONG)
    assert result.phase_out is WaterPhaseRegion.SUBCOOLED_LIQUID
    assert _zone_kinds(result) == ["condensation", "subcooling"]


def test_wet_steam_to_wetter_steam_with_lower_quality() -> None:
    inlet = water_steam_state(p=P, x=0.8)
    result = _solve(inlet, L=1.0)
    assert result.phase_out is WaterPhaseRegion.TWO_PHASE
    assert result.quality_out < 0.8


def test_wet_steam_x_in_0p8_to_x_out_between_0_and_0p8() -> None:
    inlet = water_steam_state(p=P, x=0.8)
    result = _solve(inlet, L=1.0)
    assert 0.0 < result.quality_out < 0.8


def test_wet_steam_x_in_0p8_to_saturated_liquid() -> None:
    inlet = water_steam_state(p=P, x=0.8)
    # The per-unit-inside-area resistance model is L-invariant (see module
    # docstring), so the area needed to fully condense is portable to a
    # fresh geometry sized to exactly match it -- this lands the outlet
    # right at (or just past) the saturated-liquid boundary, with
    # negligible/no subcooling.
    probe = _solve(inlet, L=L_LONG)
    A_needed = probe.A_desuperheat + probe.A_condensation
    L_matched = A_needed / (math.pi * D_I)
    result = _solve(inlet, L=L_matched)
    assert result.phase_out in (WaterPhaseRegion.SATURATED_LIQUID, WaterPhaseRegion.SUBCOOLED_LIQUID)
    assert result.f_subcooling < 1e-3


def test_wet_steam_x_in_0p8_to_subcooled_liquid() -> None:
    inlet = water_steam_state(p=P, x=0.8)
    result = _solve(inlet, L=L_LONG)
    assert result.phase_out is WaterPhaseRegion.SUBCOOLED_LIQUID


def test_saturated_liquid_to_subcooled_liquid() -> None:
    inlet = water_steam_state(p=P, x=0.0)
    result = _solve(inlet, L=L_LONG)
    assert _zone_kinds(result) == ["subcooling"]
    assert result.phase_out is WaterPhaseRegion.SUBCOOLED_LIQUID


def test_subcooled_liquid_inlet_further_subcooled() -> None:
    inlet = water_steam_state(T=360.0, p=P)
    result = _solve(inlet, L=L_LONG, T_sink=290.0)
    assert result.phase_out is WaterPhaseRegion.SUBCOOLED_LIQUID
    assert result.T_out < 360.0


# ---------------------------------------------------------------------------
# Reverse-direction (evaporation) unsupported
# ---------------------------------------------------------------------------
def test_reverse_direction_evaporation_unsupported() -> None:
    cold_liquid = water_steam_state(T=280.0, p=P)
    with pytest.raises(ReverseDirectionEvaporationNotSupportedError):
        _solve(cold_liquid, T_sink=300.0)


def test_reverse_direction_boundary_equal_temperature_unsupported() -> None:
    inlet = water_steam_state(T=320.0, p=P)
    with pytest.raises(ReverseDirectionEvaporationNotSupportedError):
        _solve(inlet, T_sink=320.0)


# ---------------------------------------------------------------------------
# Area balance (3E/3O)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "inlet_factory,L",
    [
        (lambda: water_steam_state(T=450.0, p=P), L_LONG),
        (lambda: water_steam_state(p=P, x=1.0), 1.0),
        (lambda: water_steam_state(p=P, x=0.8), 1.0),
        (lambda: water_steam_state(p=P, x=0.0), L_LONG),
    ],
)
def test_area_fractions_sum_to_one(inlet_factory, L) -> None:
    result = _solve(inlet_factory(), L=L)
    total_fraction = result.f_desuperheat + result.f_condensation + result.f_subcooling
    assert total_fraction == pytest.approx(1.0, abs=1e-9)
    assert result.A_desuperheat + result.A_condensation + result.A_subcooling == pytest.approx(
        result.A_total, rel=1e-9
    )


def test_zero_size_zones_omitted() -> None:
    inlet = water_steam_state(p=P, x=1.0)
    result = _solve(inlet, L=1.0)  # partial condensation only
    assert result.Q_desuperheat == 0.0
    assert result.A_desuperheat == 0.0
    assert result.Q_subcooling == 0.0
    assert result.A_subcooling == 0.0
    assert "desuperheat" not in _zone_kinds(result)
    assert "subcooling" not in _zone_kinds(result)


def test_no_negative_areas_or_out_of_range_fractions() -> None:
    for inlet in (
        water_steam_state(T=450.0, p=P),
        water_steam_state(p=P, x=1.0),
        water_steam_state(p=P, x=0.5),
        water_steam_state(p=P, x=0.0),
    ):
        for L in (0.1, 1.0, L_LONG):
            result = _solve(inlet, L=L)
            assert result.A_desuperheat >= 0.0
            assert result.A_condensation >= 0.0
            assert result.A_subcooling >= 0.0
            assert -1e-9 <= result.f_desuperheat <= 1.0 + 1e-9
            assert -1e-9 <= result.f_condensation <= 1.0 + 1e-9
            assert -1e-9 <= result.f_subcooling <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Energy balance (3M/3O)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "inlet_factory,L",
    [
        (lambda: water_steam_state(T=450.0, p=P), 0.05),
        (lambda: water_steam_state(T=450.0, p=P), 5.0),
        (lambda: water_steam_state(T=450.0, p=P), L_LONG),
        (lambda: water_steam_state(p=P, x=1.0), 1.0),
        (lambda: water_steam_state(p=P, x=1.0), L_LONG),
        (lambda: water_steam_state(p=P, x=0.5), 1.0),
        (lambda: water_steam_state(p=P, x=0.0), L_LONG),
    ],
)
def test_exact_energy_balance(inlet_factory, L) -> None:
    result = _solve(inlet_factory(), L=L)
    assert result.Q_total == pytest.approx(
        result.Q_desuperheat + result.Q_condensation + result.Q_subcooling, rel=1e-10
    )
    assert result.Q_total == pytest.approx(M_DOT * (result.h_in - result.h_out), rel=1e-8)


def test_total_mass_conservation() -> None:
    # Total tube-side mass flow is echoed unchanged; condensation only
    # changes the vapor/liquid split, not the total.
    inlet = water_steam_state(p=P, x=1.0)
    result = _solve(inlet, L=1.0)
    assert result.m_dot_total == M_DOT


# ---------------------------------------------------------------------------
# Phase-boundary continuity / invariants (3N/3S)
# ---------------------------------------------------------------------------
def test_quality_never_increases_during_condensation() -> None:
    inlet = water_steam_state(p=P, x=1.0)
    result = _solve(inlet, L=1.0)
    assert result.quality_out <= result.quality_in


def test_subcooled_liquid_has_quality_none() -> None:
    inlet = water_steam_state(p=P, x=1.0)
    result = _solve(inlet, L=L_LONG)
    assert result.phase_out is WaterPhaseRegion.SUBCOOLED_LIQUID
    assert result.quality_out is None


def test_superheated_vapor_has_quality_none() -> None:
    inlet = water_steam_state(T=450.0, p=P)
    result = _solve(inlet, L=L_SHORT / 20.0)
    assert result.phase_in is WaterPhaseRegion.SUPERHEATED_VAPOR
    assert result.quality_in is None
    assert result.phase_out is WaterPhaseRegion.SUPERHEATED_VAPOR
    assert result.quality_out is None


def test_no_nan_or_inf_anywhere() -> None:
    for inlet in (
        water_steam_state(T=450.0, p=P),
        water_steam_state(p=P, x=1.0),
        water_steam_state(p=P, x=0.8),
        water_steam_state(p=P, x=0.0),
    ):
        for L in (0.05, 1.0, L_LONG):
            result = _solve(inlet, L=L)
            numeric_fields = (
                result.T_out,
                result.h_out,
                result.Q_total,
                result.A_desuperheat,
                result.A_condensation,
                result.A_subcooling,
                result.f_desuperheat,
                result.f_condensation,
                result.f_subcooling,
            )
            for value in numeric_fields:
                assert math.isfinite(value)
            for zone in result.zones:
                assert math.isfinite(zone.Q)
                assert math.isfinite(zone.A)
                assert math.isfinite(zone.alpha_inside)
                assert math.isfinite(zone.U)


# ---------------------------------------------------------------------------
# Two-phase pressure-drop diagnostic
# ---------------------------------------------------------------------------
def test_two_phase_pressure_drop_flag_false_when_condensing() -> None:
    inlet = water_steam_state(p=P, x=1.0)
    result = _solve(inlet, L=1.0)
    assert result.two_phase_pressure_drop_supported is False


def test_two_phase_pressure_drop_flag_true_when_no_condensation_zone() -> None:
    inlet = water_steam_state(T=450.0, p=P)
    result = _solve(inlet, L=L_SHORT / 20.0)
    assert "condensation" not in _zone_kinds(result)
    assert result.two_phase_pressure_drop_supported is True
