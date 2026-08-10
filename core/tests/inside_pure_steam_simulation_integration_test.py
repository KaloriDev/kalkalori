# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Simulation-level integration tests for pure water/steam condensation
inside tubes (v0.6.2).

Exercises the public ``BareTubeHeatExchanger.simulate`` API end-to-end
with ``IAPWS97WaterSteamProvider`` on the inside and a dry-gas outside
stream, covering the practical steam-heater cases from the v0.6.2 spec:
saturated/superheated/wet inlet, partial/complete condensation, and
condensate subcooling, plus ``PhaseChangeMode`` semantics and the
reverse-direction/outside-unsupported/multiple-sides guards.

Run:
    pytest -q core/tests/inside_pure_steam_simulation_integration_test.py
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import IAPWS97WaterSteamProvider, WaterPhaseRegion
from core.phase_change.integration import (
    MultiplePhaseChangeSidesError,
    PhaseChangeDisabledButRequiredError,
    PureWaterSteamCondensationNotSupportedError,
    ReverseDirectionEvaporationNotSupportedError,
    check_single_active_side_pure_steam,
)
from core.phase_change.types import PhaseChangeMode, WaterSteamPhaseChangeResult

P = 101_325.0


def _hx(n_rows: int = 4, n_tubes_per_row: int = 6, L: float = 6.0) -> BareTubeHeatExchanger:
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


def _wet_gas_spec(y_h2o: float = 0.15) -> GasMixtureSpec:
    remainder = 1.0 - y_h2o
    base = {"N2": 0.79, "O2": 0.21}
    total = sum(base.values())
    return GasMixtureSpec(
        components={k: v * remainder / total for k, v in base.items()} | {"H2O": y_h2o}, basis="mole",
    )


def _outside(m_dot: float = 2.0, T_in: float = 290.0):
    return HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=m_dot, T_in=T_in, p=P)


# ---------------------------------------------------------------------------
# Practical steam-heater cases (4G)
# ---------------------------------------------------------------------------
def test_saturated_steam_partial_condensation() -> None:
    hx = _hx()
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.02, x_in=1.0, p=P),
        _outside(),
    )
    pc = result.inside_phase_change
    assert isinstance(pc, WaterSteamPhaseChangeResult)
    assert pc.active is True
    assert pc.phase_in is WaterPhaseRegion.SATURATED_VAPOR
    assert pc.phase_out is WaterPhaseRegion.TWO_PHASE
    assert 0.0 < pc.quality_out < 1.0


def test_saturated_steam_complete_condensation() -> None:
    hx = _hx(n_rows=8, n_tubes_per_row=10, L=10.0)
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.01, x_in=1.0, p=P),
        HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=3.0, T_in=280.0, p=P),
    )
    pc = result.inside_phase_change
    assert pc.phase_out in (WaterPhaseRegion.SATURATED_LIQUID, WaterPhaseRegion.SUBCOOLED_LIQUID)
    assert pc.Q_condensation > 0.0


def test_saturated_steam_subcooled_condensate() -> None:
    hx = _hx(n_rows=10, n_tubes_per_row=10, L=12.0)
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.01, x_in=1.0, p=P),
        HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=3.0, T_in=280.0, p=P),
    )
    pc = result.inside_phase_change
    assert pc.phase_out is WaterPhaseRegion.SUBCOOLED_LIQUID
    assert pc.Q_subcooling > 0.0
    assert pc.quality_out is None


def test_superheated_steam_partial_condensation() -> None:
    hx = _hx()
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.02, T_in=450.0, p=P),
        _outside(),
    )
    pc = result.inside_phase_change
    assert pc.phase_in is WaterPhaseRegion.SUPERHEATED_VAPOR
    assert pc.Q_desuperheat > 0.0
    assert pc.Q_condensation > 0.0


def test_superheated_steam_subcooled_condensate() -> None:
    hx = _hx(n_rows=10, n_tubes_per_row=10, L=14.0)
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.01, T_in=450.0, p=P),
        HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=3.0, T_in=280.0, p=P),
    )
    pc = result.inside_phase_change
    assert pc.phase_out is WaterPhaseRegion.SUBCOOLED_LIQUID
    assert pc.Q_desuperheat > 0.0
    assert pc.Q_condensation > 0.0
    assert pc.Q_subcooling > 0.0


def test_wet_inlet_steam_further_condensation() -> None:
    hx = _hx()
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.02, x_in=0.7, p=P),
        _outside(),
    )
    pc = result.inside_phase_change
    assert pc.quality_in == pytest.approx(0.7)
    assert pc.quality_out < 0.7


# ---------------------------------------------------------------------------
# PhaseChangeMode semantics (4D)
# ---------------------------------------------------------------------------
def test_disabled_with_two_phase_inlet_requires_error() -> None:
    hx = _hx()
    with pytest.raises(PhaseChangeDisabledButRequiredError):
        hx.simulate(
            HXSideInput(
                provider=IAPWS97WaterSteamProvider(), m_dot=0.02, x_in=0.5, p=P,
                phase_change_mode=PhaseChangeMode.DISABLED,
            ),
            _outside(),
        )


def test_disabled_with_dry_baseline_crossing_saturation_requires_error() -> None:
    hx = _hx()
    with pytest.raises(PhaseChangeDisabledButRequiredError):
        hx.simulate(
            HXSideInput(
                provider=IAPWS97WaterSteamProvider(), m_dot=0.02, T_in=450.0, p=P,
                phase_change_mode=PhaseChangeMode.DISABLED,
            ),
            _outside(),
        )


def test_disabled_with_fully_sensible_case_returns_dry_result() -> None:
    hx = _hx()
    result = hx.simulate(
        HXSideInput(
            provider=IAPWS97WaterSteamProvider(), m_dot=0.02, T_in=450.0, p=P,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=20.0, T_in=440.0, p=P),
    )
    pc = result.inside_phase_change
    assert pc.active is False
    assert math.isfinite(result.T_out_inside)


# ---------------------------------------------------------------------------
# Unsupported / guarded scenarios (3B/4A/4C)
# ---------------------------------------------------------------------------
def test_reverse_direction_evaporation_unsupported() -> None:
    hx = _hx()
    with pytest.raises(ReverseDirectionEvaporationNotSupportedError):
        hx.simulate(
            HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.02, T_in=350.0, p=P),
            HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=2.0, T_in=380.0, p=P),
        )


def test_pure_steam_condensation_outside_unsupported() -> None:
    hx = _hx()
    with pytest.raises(PureWaterSteamCondensationNotSupportedError):
        hx.simulate(
            _outside(),
            HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.02, x_in=1.0, p=P),
        )


def test_simultaneous_inside_pure_steam_and_outside_wet_gas_unsupported() -> None:
    # Same unit-tested exclusivity rule used by apply_phase_change's
    # pure-steam path -- see check_single_active_side_pure_steam's
    # docstring for why a physically-engineered dual-condensing thermal
    # operating point is not needed to exercise this rule directly.
    with pytest.raises(MultiplePhaseChangeSidesError):
        check_single_active_side_pure_steam(True, True, iterate=True)


def test_pure_steam_inside_iterate_false_does_not_require_iterate() -> None:
    # Unlike the wet-gas side, inside pure-steam does not need the
    # iterative thermal state (it consumes T_mean_outside/outside_alfa_mean,
    # both populated under iterate=False too).
    check_single_active_side_pure_steam(True, False, iterate=False)


# ---------------------------------------------------------------------------
# Invariants (4S)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "side_kwargs",
    [
        dict(x_in=1.0),
        dict(x_in=0.6),
        dict(T_in=450.0),
    ],
)
def test_energy_balance_and_area_invariants(side_kwargs) -> None:
    hx = _hx(n_rows=6, n_tubes_per_row=8, L=8.0)
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.015, p=P, **side_kwargs),
        HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=3.0, T_in=285.0, p=P),
    )
    pc = result.inside_phase_change
    assert pc.Q_total == pytest.approx(pc.Q_sensible + pc.Q_latent, rel=1e-8)
    assert pc.Q_total == pytest.approx(pc.Q_desuperheat + pc.Q_condensation + pc.Q_subcooling, rel=1e-8)
    assert pc.Q_total == pytest.approx(0.015 * (pc.h_in - pc.h_out), rel=1e-6)
    total_fraction = pc.f_desuperheat + pc.f_condensation + pc.f_subcooling
    assert total_fraction == pytest.approx(1.0, abs=1e-6)
    assert pc.A_desuperheat >= 0.0
    assert pc.A_condensation >= 0.0
    assert pc.A_subcooling >= 0.0
    if pc.quality_out is not None:
        assert 0.0 <= pc.quality_out <= 1.0
    if pc.quality_in is not None:
        assert pc.quality_out is None or pc.quality_out <= pc.quality_in
    for value in (pc.Q_total, pc.T_out, pc.h_out):
        assert math.isfinite(value)


def test_subcooled_liquid_has_quality_none() -> None:
    hx = _hx(n_rows=10, n_tubes_per_row=10, L=14.0)
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.01, x_in=1.0, p=P),
        HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=3.0, T_in=280.0, p=P),
    )
    pc = result.inside_phase_change
    assert pc.phase_out is WaterPhaseRegion.SUBCOOLED_LIQUID
    assert pc.quality_out is None


def test_superheated_vapor_has_quality_none() -> None:
    hx = _hx()
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.02, T_in=450.0, p=P,
                     phase_change_mode=PhaseChangeMode.DISABLED),
        HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=20.0, T_in=440.0, p=P),
    )
    pc = result.inside_phase_change
    assert pc.quality_in is None
    assert pc.quality_out is None


def test_active_condensation_zone_flags_two_phase_dp_unsupported() -> None:
    hx = _hx()
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.02, x_in=1.0, p=P),
        _outside(),
    )
    pc = result.inside_phase_change
    assert pc.Q_condensation > 0.0
    assert pc.two_phase_pressure_drop_supported is False


def test_no_condensation_zone_reports_dp_supported() -> None:
    hx = _hx(n_rows=1, n_tubes_per_row=2, L=0.3)
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.02, T_in=450.0, p=P),
        _outside(),
    )
    pc = result.inside_phase_change
    assert pc.Q_condensation == 0.0
    assert pc.two_phase_pressure_drop_supported is True


# ---------------------------------------------------------------------------
# Regression: public inside_alfa_mean / thermal_state.alfa_i must reflect
# the real multi-zone (Shah-correlation) physics, not the sensible-only dry
# baseline's single-phase evaluation of the whole steam side at one bulk
# state. Manual validation found the reported condensation-side HTC orders
# of magnitude too small; root cause was that apply_phase_change's pure-
# steam path never overrode inside_alfa_mean/UA/thermal_state with the
# corrected multi-zone values -- see core.phase_change.integration
# _apply_inside_pure_steam_active. This test independently recomputes the
# expected value from the zone breakdown and fails against the old
# (unfixed) behavior.
#
# The independent recomputation combines zones through CONDUCTANCE
# (UA = sum(U_zone*A_zone), each U_zone = 1/(1/alpha_zone+R_shared_per_Ai)),
# not through a naive area-weighted arithmetic mean of alpha (a second,
# subtler bug fixed in the same v0.6.2 patch, spec section 13-14): the two
# give different answers whenever zones have different alpha, since the
# resistance mapping 1/(1/alpha+R) is nonlinear in alpha.
# ---------------------------------------------------------------------------
def test_inside_alfa_mean_matches_independent_zone_conductance_resistance_identity() -> None:
    hx = _hx()
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.02, T_in=450.0, p=P),
        _outside(),
    )
    pc = result.inside_phase_change
    assert pc.A_total > 0.0

    A_inside_total = hx.bundle.total_inner_area
    A_outside_total = hx.bundle.total_outer_area
    R_wall_total = hx.tube_wall_resistance()
    alpha_outside = result.outside_alfa_mean
    R_shared_per_Ai = (
        R_wall_total * A_inside_total + (1.0 / alpha_outside) * (A_outside_total / A_inside_total)
    )

    zones = (
        (pc.zone_alpha_desuperheat, pc.A_desuperheat),
        (pc.zone_alpha_condensation, pc.A_condensation),
        (pc.zone_alpha_subcooling, pc.A_subcooling),
    )
    UA_expected = sum(
        (1.0 / (1.0 / alpha + R_shared_per_Ai)) * A
        for alpha, A in zones
        if alpha is not None and A > 0.0
    )
    U_i_expected = UA_expected / pc.A_total
    expected_alfa_i = 1.0 / (1.0 / U_i_expected - R_shared_per_Ai)

    assert result.inside_alfa_mean == pytest.approx(expected_alfa_i, rel=1e-9)
    assert result.UA == pytest.approx(UA_expected, rel=1e-9)

    # Provable bound: U_equivalent = area-weighted arithmetic mean of the
    # per-zone U_zone = 1/(1/alpha_zone+R_shared_per_Ai), so it lies within
    # [min(U_zone), max(U_zone)]; since U(alpha) is strictly increasing,
    # inverting it back to alfa_i_equivalent must land within
    # [min(zone_alpha), max(zone_alpha)] too. This is a stronger, non-
    # arbitrary regression guard than comparing against a fraction of the
    # condensation zone's own alpha (which is not, in general, either
    # bound -- a poor-HTC desuperheat zone with a large area fraction can
    # legitimately pull the equivalent HTC well below the condensation
    # zone's alpha; this case's own desuperheat zone does exactly that).
    zone_alphas = [alpha for alpha, A in zones if alpha is not None and A > 0.0]
    assert min(zone_alphas) <= result.inside_alfa_mean <= max(zone_alphas)


def test_thermal_state_alfa_i_consistent_with_top_level_inside_alfa_mean() -> None:
    hx = _hx()
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.02, T_in=450.0, p=P),
        _outside(),
    )
    assert result.thermal_state is not None
    assert result.thermal_state.alfa_i == pytest.approx(result.inside_alfa_mean, rel=1e-12)
    assert result.thermal_state.UA == pytest.approx(result.UA, rel=1e-12)
    assert result.thermal_state.U == pytest.approx(result.U_mean, rel=1e-12)
