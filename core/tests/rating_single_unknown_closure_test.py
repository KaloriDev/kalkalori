# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Thermal area/flow Rating closures of the shared wet forward engine.

The legacy installed-duty scalar closure, activation band and discontinuous
residual solver have been retired. AUTO dry and wet regimes share the same
production caloric/surface definitions and thermal sizing on installed geometry.
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, CircularFinnedTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.phase_change.rating_integration import (
    RatingClosureError,
)
from core.phase_change.types import PhaseChangeMode
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec

P = 101_325.0


def _exchanger() -> BareTubeHeatExchanger:
    core_tube = BareTube(D_i=0.022, D_o=0.025, length_total=2.8, length_effective=2.8, wall_k=50.0)
    tube = CircularFinnedTube(
        core_tube=core_tube, fin_k=200.0, D_fin=0.050, D_root=0.028,
        fin_thickness_root=0.0005, fin_thickness_tip=0.0003, fin_pitch=0.0024,
        fin_contact_efficiency=0.95,
    )
    pitch_transverse = 0.060
    bundle = TubeBundle(
        tube=tube, n_rows=6, n_tubes_per_row=8, pitch_transverse=pitch_transverse,
        pitch_longitudinal=math.sqrt(3.0) * pitch_transverse / 2.0,
        layout="staggered", n_passes_tube=2, flow_arrangement="counterflow",
    )
    return BareTubeHeatExchanger(bundle)


def _inside_provider() -> ConstantPropertyProvider:
    return ConstantPropertyProvider(FluidTransportProperties(rho=900.0, mu=5.0e-4, k=0.60, cp=4000.0))


def _wet_provider() -> GasMixturePropertyProvider:
    return GasMixturePropertyProvider(
        GasMixtureSpec(components={"N2": 0.65, "O2": 0.10, "CO2": 0.08, "H2O": 0.17}, basis="mole")
    )


@pytest.fixture(scope="module")
def active_unknown_m_dot_result():
    """Test 1: active condensation, inside.m_dot unknown, inside.T_out known."""
    hx = _exchanger()
    inside = BalanceSideSpec(
        provider=_inside_provider(), p=P, m_dot=None, T_in=290.0, T_out=311.0,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    outside = BalanceSideSpec(
        provider=_wet_provider(), p=P, m_dot=6.0, T_in=420.0, T_out=333.0,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    from core import WetCoilSolverOptions
    return hx.rate(inside, outside, include_simulation=False,
                   wet_solver_options=WetCoilSolverOptions(
                       outlet_temperature_tolerance_K=1e-4, timeout_s=None))


def test_active_condensation_solves_unknown_inside_mass_flow(active_unknown_m_dot_result) -> None:
    result = active_unknown_m_dot_result
    pc = result.outside_phase_change

    assert pc.active is True
    assert pc.converged is True
    solved_m_dot = result.closed_balance.inside.m_dot
    assert solved_m_dot > 0.0
    assert result.closed_balance.inside.T_out == pytest.approx(311.0)
    assert abs(pc.mass_balance_error) < 1.0e-3
    assert abs(pc.energy_balance_error) < 1.0
    assert pc.Q_total == pytest.approx(result.Q_required, rel=1.0e-6)
    assert result.wet_coil_diagnostics["required_inside_mass_flow"] == solved_m_dot
    assert result.wet_coil_diagnostics["rating_forward_evaluations"] > 0


def test_active_condensation_solves_unknown_inside_outlet_temperature(active_unknown_m_dot_result) -> None:
    """Test 2: the reverse closure (T_out unknown) must recover the same
    physically self-consistent state as Test 1's solved mass flow --
    a strong cross-check that both directions solve the same coupled
    physics rather than two independently-tuned special cases."""
    solved_m_dot = active_unknown_m_dot_result.closed_balance.inside.m_dot

    hx = _exchanger()
    inside = BalanceSideSpec(
        provider=_inside_provider(), p=P, m_dot=solved_m_dot, T_in=290.0, T_out=None,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    outside = BalanceSideSpec(
        provider=_wet_provider(), p=P, m_dot=6.0, T_in=420.0, T_out=333.0,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    result = hx.rate(
        inside, outside, include_simulation=False,
        wet_solver_options=active_unknown_m_dot_result.wet_coil_diagnostics["solver_options"],
    )
    pc = result.outside_phase_change

    assert pc.active is True
    assert pc.converged is True
    assert result.closed_balance.inside.T_out == pytest.approx(311.0, abs=1.0e-2)
    assert abs(pc.mass_balance_error) < 1.0e-3
    assert abs(pc.energy_balance_error) < 1.0
    assert result.wet_coil_diagnostics["required_area_scale"] > 0
    assert result.wet_coil_diagnostics["required_inside_mass_flow"] == solved_m_dot


def test_dry_auto_with_unknown_inside_mass_flow_still_solves() -> None:
    """AUTO-DRY solves thermal area and flow with the same production engine."""
    hx = _exchanger()
    inside = BalanceSideSpec(
        provider=_inside_provider(), p=P, m_dot=None, T_in=333.15, T_out=354.15,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    outside = BalanceSideSpec(
        provider=_wet_provider(), p=P, m_dot=6.0, T_in=420.0, T_out=380.0,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    result = hx.rate(inside, outside, include_simulation=False)
    pc = result.outside_phase_change

    assert pc.active is False
    assert pc.near_onset is False
    assert pc.converged is True
    assert result.closed_balance.inside.m_dot > 0.0
    assert pc.m_dot_condensate == 0.0
    assert pc.Q_latent == 0.0
    assert pc.Q_sensible == pytest.approx(result.Q_required)
    assert pc.Q_total == pytest.approx(result.Q_required)


def test_legacy_activation_band_does_not_change_production_dry_regime() -> None:
    """The retired activation band cannot override the physical onset."""
    hx = _exchanger()
    inside = BalanceSideSpec(
        provider=_inside_provider(), p=P, m_dot=None, T_in=333.15, T_out=354.15,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    outside = BalanceSideSpec(
        provider=_wet_provider(), p=P, m_dot=6.0, T_in=420.0, T_out=380.0,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    result = hx.rate(
        inside, outside, include_simulation=False,
        phase_change_activation_band_K=20.0,
    )
    pc = result.outside_phase_change

    assert pc.active is False
    assert pc.near_onset is False
    assert pc.possible is False
    assert pc.regime == "DRY"
    assert pc.converged is True
    assert result.closed_balance.inside.m_dot > 0.0
    assert pc.m_dot_condensate == 0.0
    assert pc.Q_latent == 0.0
    assert pc.Q_sensible == pytest.approx(result.Q_required)
    assert pc.Q_total == pytest.approx(result.Q_required)


def test_double_unknown_inside_side_remains_rejected() -> None:
    """Test 6: a genuinely underdetermined problem (both inside.m_dot and
    inside.T_out unknown) must still raise, not be silently guessed."""
    hx = _exchanger()
    inside = BalanceSideSpec(
        provider=_inside_provider(), p=P, m_dot=None, T_in=290.0, T_out=None,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    outside = BalanceSideSpec(
        provider=_wet_provider(), p=P, m_dot=6.0, T_in=420.0, T_out=333.0,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    with pytest.raises(RatingClosureError):
        hx.rate(inside, outside, include_simulation=False)


def test_no_bracket_for_unreachable_duty_raises_closure_error() -> None:
    """Test 7 (end-to-end): a dry-regime target duty that exceeds what any
    purely-sensible exchanger area could deliver, with condensation not
    activating to explain the gap, must still fail explicitly through the
    public Rating entry point -- not silently return the closest trial as
    if it were a valid answer."""
    hx = _exchanger()
    inside = BalanceSideSpec(
        provider=_inside_provider(), p=P, m_dot=8.0, T_in=419.999, T_out=None,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    outside = BalanceSideSpec(
        provider=_wet_provider(), p=P, m_dot=6.0, T_in=420.0, T_out=333.0,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    with pytest.raises(RatingClosureError):
        hx.rate(inside, outside, include_simulation=False)
