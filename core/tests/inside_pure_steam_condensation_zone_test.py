# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for core.phase_change.inside_pure_steam_condensation (v0.6.2).

Covers the pure-water/steam in-tube condensation-zone physics: exact
latent energy balance, mass conservation, HTC delegation, and the explicit
two-phase pressure-drop "not supported" diagnostic. Deterministic given
the IAPWS-IF97 backend.

Run:
    pytest -q core/tests/inside_pure_steam_condensation_zone_test.py
"""

from __future__ import annotations

import math

import pytest

from core.phase_change.inside_pure_steam_condensation import (
    solve_inside_condensation_zone,
)
from core.phase_change.warning_codes import TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED
from core.properties.water import water_steam_state

P = 101_325.0
D_I = 0.02
FLOW_AREA = math.pi * D_I**2 / 4.0
M_DOT = 0.05


def _zone(x_in: float, x_out: float):
    return solve_inside_condensation_zone(
        p=P, x_in=x_in, x_out=x_out, m_dot_total=M_DOT, D_i=D_I, flow_area=FLOW_AREA
    )


def test_saturated_vapor_inlet_partial_condensation() -> None:
    zone = _zone(1.0, 0.4)
    assert zone.x_in == 1.0
    assert zone.x_out == 0.4
    assert zone.Q_condensation > 0.0


def test_wet_inlet_produces_lower_outlet_quality() -> None:
    zone = _zone(0.8, 0.3)
    assert zone.x_out < zone.x_in


def test_exact_latent_energy_balance() -> None:
    zone = _zone(0.9, 0.2)
    expected_Q = M_DOT * zone.h_fg * (0.9 - 0.2)
    assert zone.Q_condensation == pytest.approx(expected_Q, rel=1e-10)
    assert zone.Q_condensation == pytest.approx(
        M_DOT * (zone.h_in - zone.h_out), rel=1e-12
    )


def test_total_water_mass_conserved() -> None:
    zone = _zone(0.7, 0.1)
    assert zone.m_dot_vapor_in + zone.m_dot_liquid_in == pytest.approx(M_DOT, rel=1e-12)
    assert zone.m_dot_vapor_out + zone.m_dot_liquid_out == pytest.approx(M_DOT, rel=1e-12)
    assert zone.m_dot_total == M_DOT


def test_vapor_liquid_split_matches_quality() -> None:
    zone = _zone(0.6, 0.25)
    assert zone.m_dot_vapor_in == pytest.approx(0.6 * M_DOT, rel=1e-12)
    assert zone.m_dot_liquid_in == pytest.approx(0.4 * M_DOT, rel=1e-12)
    assert zone.m_dot_vapor_out == pytest.approx(0.25 * M_DOT, rel=1e-12)
    assert zone.m_dot_liquid_out == pytest.approx(0.75 * M_DOT, rel=1e-12)


def test_x_out_cannot_exceed_x_in() -> None:
    with pytest.raises(ValueError):
        _zone(0.3, 0.7)


def test_x_out_cannot_fall_below_zero() -> None:
    with pytest.raises(ValueError):
        _zone(0.5, -0.1)


def test_x_in_cannot_exceed_one() -> None:
    with pytest.raises(ValueError):
        _zone(1.2, 0.0)


def test_zero_size_zone_rejected() -> None:
    with pytest.raises(ValueError):
        _zone(0.5, 0.5)


def test_two_phase_pressure_drop_explicitly_unsupported() -> None:
    zone = _zone(1.0, 0.0)
    assert zone.two_phase_pressure_drop_supported is False
    assert any(w.code == TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED for w in zone.warnings)


def test_complete_condensation_to_saturated_liquid() -> None:
    zone = _zone(1.0, 0.0)
    assert zone.x_out == 0.0
    assert zone.h_out == pytest.approx(zone.h_f, rel=1e-9)
    assert zone.Q_condensation == pytest.approx(M_DOT * zone.h_fg, rel=1e-10)


def test_mass_flux_matches_geometry() -> None:
    zone = _zone(0.8, 0.2)
    assert zone.G == pytest.approx(M_DOT / FLOW_AREA, rel=1e-12)


def test_saturation_properties_match_water_provider() -> None:
    zone = _zone(0.5, 0.1)
    ref = water_steam_state(p=P, x=0.0)
    assert zone.T_sat == pytest.approx(ref.T_sat, rel=1e-12)
    assert zone.h_f == pytest.approx(ref.h_f, rel=1e-12)
    assert zone.h_g == pytest.approx(ref.h_g, rel=1e-12)
    assert zone.h_fg == pytest.approx(ref.h_fg, rel=1e-12)


def test_alpha_condensation_effective_finite_and_positive() -> None:
    for x_in, x_out in [(1.0, 0.0), (1.0, 0.5), (0.9, 0.1), (0.3, 0.05)]:
        zone = _zone(x_in, x_out)
        assert math.isfinite(zone.alpha_condensation_effective)
        assert zone.alpha_condensation_effective > 0.0


def test_no_nan_or_inf_in_result() -> None:
    zone = _zone(0.95, 0.05)
    numeric_fields = (
        zone.h_in,
        zone.h_out,
        zone.T_sat,
        zone.h_f,
        zone.h_g,
        zone.h_fg,
        zone.Q_condensation,
        zone.alpha_condensation_effective,
        zone.G,
    )
    for value in numeric_fields:
        assert math.isfinite(value)
