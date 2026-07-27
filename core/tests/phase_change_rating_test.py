# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Tests for BareTubeHeatExchanger.rate() with outside water condensation
(spec section 32).

Run:
    pytest -q core/tests/phase_change_rating_test.py
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.properties.gas_mixture import GasMixtureSpec, GasMixturePropertyProvider
from core.phase_change.types import PhaseChangeDirection, PhaseChangeMode


@pytest.fixture(scope="module")
def hx() -> BareTubeHeatExchanger:
    tube = BareTube(D_i=25e-3 - 2 * 1.5e-3, D_o=25e-3, length_total=2.8, length_effective=2.8, wall_k=50.0)
    bundle = TubeBundle(
        tube=tube, n_rows=20, n_tubes_per_row=30,
        pitch_transverse=35e-3, pitch_longitudinal=35e-3,
        layout="staggered", n_passes_tube=2, flow_arrangement="counterflow",
    )
    return BareTubeHeatExchanger(bundle)


def _wet_spec() -> GasMixtureSpec:
    return GasMixtureSpec(components={"N2": 0.65, "O2": 0.10, "CO2": 0.08, "H2O": 0.17}, basis="mole")


def _dry_spec() -> GasMixtureSpec:
    return GasMixtureSpec(components={"N2": 0.79, "O2": 0.21}, basis="mole")


def test_rate_with_active_outside_condensation(hx: BareTubeHeatExchanger) -> None:
    inside = BalanceSideSpec(
        provider=GasMixturePropertyProvider(_dry_spec()), p=101_325.0,
        m_dot=15.0, T_in=290.0, T_out=332.0,
    )
    outside = BalanceSideSpec(
        provider=GasMixturePropertyProvider(_wet_spec()), p=101_325.0,
        m_dot=6.0, T_in=420.0, T_out=333.0,
    )

    result = hx.rate(inside, outside)
    pc = result.outside_phase_change

    assert pc.active is True
    assert pc.direction is PhaseChangeDirection.CONDENSATION
    assert pc.m_dot_condensate > 0.0
    assert pc.W_out < pc.W_in
    assert pc.Q_latent > 0.0
    assert pc.Q_total == pytest.approx(pc.Q_sensible + pc.Q_latent, rel=1e-9)
    assert abs(pc.mass_balance_error) < 1e-6
    assert abs(pc.energy_balance_error) < 1e-6
    assert math.isfinite(result.overdesign_factor)
    assert math.isfinite(result.UA_required)


def test_rate_disabled_outside_gives_sensible_only_with_warning(hx: BareTubeHeatExchanger) -> None:
    from core.phase_change.warning_codes import PHASE_CHANGE_DISABLED_BUT_POSSIBLE

    inside = BalanceSideSpec(
        provider=GasMixturePropertyProvider(_dry_spec()), p=101_325.0,
        m_dot=15.0, T_in=290.0, T_out=332.0,
    )
    outside = BalanceSideSpec(
        provider=GasMixturePropertyProvider(_wet_spec()), p=101_325.0,
        m_dot=6.0, T_in=420.0, T_out=333.0,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )

    result = hx.rate(inside, outside)
    pc = result.outside_phase_change
    assert pc.active is False
    assert pc.m_dot_condensate == 0.0
    assert any(w.code == PHASE_CHANGE_DISABLED_BUT_POSSIBLE for w in pc.warnings)


def test_rate_not_capable_matches_dry_rating(hx: BareTubeHeatExchanger) -> None:
    from core.models.heat_balance import close_heat_balance
    from core.models.rating import run_rating

    inside = BalanceSideSpec(
        provider=GasMixturePropertyProvider(_dry_spec()), p=101_325.0,
        m_dot=15.0, T_in=290.0, T_out=320.0,
    )
    outside = BalanceSideSpec(
        provider=GasMixturePropertyProvider(_dry_spec()), p=101_325.0,
        m_dot=6.0, T_in=420.0, T_out=350.0,
    )

    expected = run_rating(hx, close_heat_balance(inside, outside))
    actual = hx.rate(inside, outside)

    assert actual.Q_required == expected.Q_required
    assert actual.UA_required == expected.UA_required
    assert actual.overdesign_factor == expected.overdesign_factor
    assert actual.outside_phase_change.capable is False


def test_rate_active_condensation_requires_explicit_m_dot(hx: BareTubeHeatExchanger) -> None:
    inside = BalanceSideSpec(
        provider=GasMixturePropertyProvider(_dry_spec()), p=101_325.0,
        m_dot=15.0, T_in=290.0, T_out=332.0,
    )
    outside_no_mdot = BalanceSideSpec(
        provider=GasMixturePropertyProvider(_wet_spec()), p=101_325.0,
        T_in=420.0, T_out=333.0,
    )
    with pytest.raises(ValueError):
        hx.rate(inside, outside_no_mdot)


def test_rate_active_condensation_requires_explicit_T_out(hx: BareTubeHeatExchanger) -> None:
    inside = BalanceSideSpec(
        provider=GasMixturePropertyProvider(_dry_spec()), p=101_325.0,
        m_dot=15.0, T_in=290.0, T_out=332.0,
    )
    outside_no_T_out = BalanceSideSpec(
        provider=GasMixturePropertyProvider(_wet_spec()), p=101_325.0,
        m_dot=6.0, T_in=420.0,
    )
    with pytest.raises(ValueError):
        hx.rate(inside, outside_no_T_out)
