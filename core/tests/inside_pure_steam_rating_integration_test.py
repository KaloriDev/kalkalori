# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Rating-level integration tests for pure water/steam condensation inside
tubes (v0.6.2).

Exercises the public ``BareTubeHeatExchanger.rate`` API end-to-end with
``IAPWS97WaterSteamProvider`` on the inside (a closed heat balance, both
endpoints known) and a dry-gas outside stream.

Run:
    pytest -q core/tests/inside_pure_steam_rating_integration_test.py
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import IAPWS97WaterSteamProvider, WaterPhaseRegion
from core.phase_change.integration import ReverseDirectionEvaporationNotSupportedError
from core.phase_change.types import PhaseChangeMode, WaterSteamPhaseChangeResult

P = 101_325.0


def _hx(n_rows: int = 8, n_tubes_per_row: int = 10, L: float = 10.0) -> BareTubeHeatExchanger:
    tube = BareTube(
        D_i=0.020, D_o=0.025, length_total=L, length_effective=L, wall_k=50.0,
        tube_orientation=TubeOrientation.HORIZONTAL,
    )
    bundle = TubeBundle(
        tube=tube, n_rows=n_rows, n_tubes_per_row=n_tubes_per_row,
        pitch_transverse=0.04, pitch_longitudinal=0.04,
        layout="staggered", n_passes_tube=1, flow_arrangement="crossflow",
    )
    return BareTubeHeatExchanger(bundle)


def _dry_air_spec() -> GasMixtureSpec:
    return GasMixtureSpec(components={"N2": 0.79, "O2": 0.21}, basis="mole")


def _outside(m_dot: float = 3.0, T_in: float = 280.0, T_out: float = 295.0) -> BalanceSideSpec:
    return BalanceSideSpec(provider=GasMixturePropertyProvider(_dry_air_spec()), p=P, m_dot=m_dot, T_in=T_in, T_out=T_out)


# ---------------------------------------------------------------------------
# Practical closed-balance cases (4H)
# ---------------------------------------------------------------------------
def test_rating_saturated_steam_partial_condensation() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.02, x_in=1.0, x_out=0.4),
        _outside(),
    )
    pc = result.inside_phase_change
    assert isinstance(pc, WaterSteamPhaseChangeResult)
    assert pc.active is True
    assert pc.quality_in == 1.0
    assert pc.quality_out == pytest.approx(0.4)
    assert math.isfinite(result.overdesign_factor)


def test_rating_superheated_steam() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.01, T_in=450.0, T_out=350.0),
        _outside(),
    )
    pc = result.inside_phase_change
    assert pc.phase_in is WaterPhaseRegion.SUPERHEATED_VAPOR
    assert pc.Q_desuperheat > 0.0
    assert pc.Q_condensation > 0.0


def test_rating_wet_steam_inlet() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.02, x_in=0.8, x_out=0.3),
        _outside(),
    )
    pc = result.inside_phase_change
    assert pc.quality_in == pytest.approx(0.8)
    assert pc.quality_out == pytest.approx(0.3)


def test_rating_complete_condensation() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.005, x_in=1.0, x_out=0.0),
        _outside(),
    )
    pc = result.inside_phase_change
    assert pc.quality_out == 0.0
    assert pc.phase_out is WaterPhaseRegion.SATURATED_LIQUID
    assert pc.Q_subcooling == 0.0


def test_rating_subcooling() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.005, x_in=1.0, T_out=320.0),
        _outside(),
    )
    pc = result.inside_phase_change
    assert pc.phase_out is WaterPhaseRegion.SUBCOOLED_LIQUID
    assert pc.Q_subcooling > 0.0
    assert pc.quality_out is None


# ---------------------------------------------------------------------------
# PhaseChangeMode / validation
# ---------------------------------------------------------------------------
def test_rating_disabled_with_sensible_only_span_succeeds() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(
            provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.01, T_in=450.0, T_out=400.0,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        BalanceSideSpec(provider=GasMixturePropertyProvider(_dry_air_spec()), p=P, m_dot=3.0, T_in=390.0, T_out=395.0),
    )
    assert result.inside_phase_change.active is False


def test_rating_disabled_spanning_saturation_raises() -> None:
    from core.phase_change.integration import PhaseChangeDisabledButRequiredError

    hx = _hx()
    with pytest.raises(PhaseChangeDisabledButRequiredError):
        hx.rate(
            BalanceSideSpec(
                provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.01, T_in=450.0, T_out=350.0,
                phase_change_mode=PhaseChangeMode.DISABLED,
            ),
            _outside(),
        )


def test_rating_requires_explicit_outlet_on_steam_side() -> None:
    hx = _hx()
    with pytest.raises(ValueError):
        hx.rate(
            BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.01, x_in=1.0),
            _outside(),
        )


def test_rating_reverse_direction_unsupported() -> None:
    hx = _hx()
    with pytest.raises(ReverseDirectionEvaporationNotSupportedError):
        hx.rate(
            BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.01, T_in=350.0, T_out=340.0),
            BalanceSideSpec(provider=GasMixturePropertyProvider(_dry_air_spec()), p=P, m_dot=3.0, T_in=380.0, T_out=385.0),
        )


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
def test_rating_energy_balance_and_area_invariants() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.01, T_in=450.0, T_out=350.0),
        _outside(),
    )
    pc = result.inside_phase_change
    assert pc.Q_total == pytest.approx(pc.Q_sensible + pc.Q_latent, rel=1e-8)
    assert pc.Q_total == pytest.approx(pc.Q_desuperheat + pc.Q_condensation + pc.Q_subcooling, rel=1e-8)
    assert pc.Q_total == pytest.approx(0.01 * (pc.h_in - pc.h_out), rel=1e-6)
    total_fraction = pc.f_desuperheat + pc.f_condensation + pc.f_subcooling
    assert total_fraction == pytest.approx(1.0, abs=1e-6)
    assert pc.A_desuperheat >= 0.0 and pc.A_condensation >= 0.0 and pc.A_subcooling >= 0.0
    for value in (result.overdesign_factor, result.A_required, result.UA_required, pc.Q_total):
        assert math.isfinite(value)


def test_rating_mass_conservation() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.01, x_in=1.0, x_out=0.5),
        _outside(),
    )
    assert result.inside_phase_change.m_dot_total == 0.01


# ---------------------------------------------------------------------------
# Regression: public alfa_i/UA_actual/U_mean (and the nested thermal_state)
# must reflect the real multi-zone (Shah-correlation) physics, not the
# sensible-only dry baseline's single-phase evaluation of the whole steam
# side at one bulk state. Manual validation found the reported condensation
# -side HTC orders of magnitude too small; root cause was that
# _apply_inside_pure_steam_to_rating never overrode alfa_i/UA_actual/
# U_mean/thermal_state with the corrected multi-zone values (only
# overdesign_factor/A_required/UA_required were fixed) -- see
# core.phase_change.rating_integration._apply_inside_pure_steam_to_rating.
# This test independently recomputes the expected value from the zone
# breakdown and fails against the old (unfixed) behavior.
#
# The independent recomputation combines zones through CONDUCTANCE
# (UA = sum(U_zone*A_zone), each U_zone = 1/(1/alpha_zone+R_shared_per_Ai)),
# not through a naive area-weighted arithmetic mean of alpha (a second,
# subtler bug fixed in the same v0.6.2 patch, spec section 13-14): the two
# give different answers whenever zones have different alpha, since the
# resistance mapping 1/(1/alpha+R) is nonlinear in alpha.
# ---------------------------------------------------------------------------
def test_rating_alfa_i_matches_independent_zone_conductance_resistance_identity() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.01, T_in=450.0, T_out=350.0),
        _outside(),
    )
    pc = result.inside_phase_change
    A_required_inside = pc.A_desuperheat + pc.A_condensation + pc.A_subcooling
    assert A_required_inside > 0.0

    A_inside_total = hx.bundle.total_inner_area
    A_outside_total = hx.bundle.total_outer_area
    R_wall_total = hx.tube_wall_resistance()
    alpha_outside = result.thermal_state.alfa_o
    R_shared_per_Ai = (
        R_wall_total * A_inside_total + (1.0 / alpha_outside) * (A_outside_total / A_inside_total)
    )

    zones = (
        (pc.zone_alpha_desuperheat, pc.A_desuperheat),
        (pc.zone_alpha_condensation, pc.A_condensation),
        (pc.zone_alpha_subcooling, pc.A_subcooling),
    )
    UA_required_expected = sum(
        (1.0 / (1.0 / alpha + R_shared_per_Ai)) * A
        for alpha, A in zones
        if alpha is not None and A > 0.0
    )
    U_equivalent_expected = UA_required_expected / A_required_inside
    expected_alfa_i = 1.0 / (1.0 / U_equivalent_expected - R_shared_per_Ai)

    assert result.UA_required == pytest.approx(UA_required_expected, rel=1e-9)
    assert result.alfa_i == pytest.approx(expected_alfa_i, rel=1e-9)

    # Provable bound: U_equivalent = area-weighted arithmetic mean of the
    # per-zone U_zone = 1/(1/alpha_zone+R_shared_per_Ai), so it lies within
    # [min(U_zone), max(U_zone)]; since U(alpha) is strictly increasing,
    # inverting it back to alfa_i must land within
    # [min(zone_alpha), max(zone_alpha)] too. This is a stronger, non-
    # arbitrary regression guard than comparing against a fraction of the
    # condensation zone's own alpha (which is not, in general, either
    # bound -- a poor-HTC desuperheat zone with a large area fraction can
    # legitimately pull the equivalent HTC well below the condensation
    # zone's alpha; this case's own desuperheat zone does exactly that).
    zone_alphas = [alpha for alpha, A in zones if alpha is not None and A > 0.0]
    assert min(zone_alphas) <= result.alfa_i <= max(zone_alphas)


def test_rating_thermal_state_alfa_i_consistent_with_top_level_alfa_i() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.01, T_in=450.0, T_out=350.0),
        _outside(),
    )
    assert result.thermal_state is not None
    assert result.thermal_state.alfa_i == pytest.approx(result.alfa_i, rel=1e-12)
    assert result.thermal_state.UA == pytest.approx(result.UA_actual, rel=1e-12)
    assert result.thermal_state.U == pytest.approx(result.U_mean, rel=1e-12)


def test_rating_ua_margin_uses_corrected_ua_actual() -> None:
    hx = _hx()
    result = hx.rate(
        BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P, m_dot=0.01, T_in=450.0, T_out=350.0),
        _outside(),
    )
    expected_ua_margin = result.UA_actual / result.UA_required - 1.0
    assert result.ua_margin == pytest.approx(expected_ua_margin, rel=1e-9)
