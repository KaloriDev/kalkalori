# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Regression coverage for the restored Rating single-unknown closure.

The wet-condensation Rating integration (v0.7.5) previously regressed a
pre-existing Rating capability: leaving exactly one of the non-condensing
inside side's ``m_dot``/``T_out`` unknown for ``close_heat_balance`` to
solve. Once outside H2O condensation is active, that single unknown cannot
be recovered by one more algebraic step (the wet outside stream's own duty
is not pinned down by its (T_in, T_out, m_dot) alone -- how much water
condenses is an extra degree of freedom only the mass-transfer-coefficient
-driven physics, evaluated at the real wall temperature, can resolve), so
this restores it via an outer scalar root search
(``core.phase_change.rating_integration._solve_rating_single_unknown_inside_variable``)
that reuses the existing Rating/close_heat_balance/wet-finned machinery
rather than duplicating it.

The active-condensation cases here are genuinely expensive (each trial re-
runs a full nonlinear wet-finned Rating pass), so this module keeps them to
the minimum the spec requires and reuses the smallest geometry/provider
combination already established as sufficient in
``wet_finned_rating_test.py``.
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
    _bracket_and_solve_monotonic_root,
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
    return hx.rate(inside, outside, include_simulation=False)


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
    assert "rating_closure_solved_inside_m_dot" in pc.assumptions
    assert pc.residuals["rating_closure_iterations"] >= 0.0


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
    result = hx.rate(inside, outside, include_simulation=False)
    pc = result.outside_phase_change

    assert pc.active is True
    assert pc.converged is True
    assert result.closed_balance.inside.T_out == pytest.approx(311.0, abs=1.0e-2)
    assert abs(pc.mass_balance_error) < 1.0e-3
    assert abs(pc.energy_balance_error) < 1.0
    assert "rating_closure_solved_inside_T_out" in pc.assumptions


def test_dry_auto_with_unknown_inside_mass_flow_still_solves() -> None:
    """Test 3: the same single-unknown closure must also work when AUTO
    resolves dry -- active=False is a valid converged result, not a
    failure, and the fast pre-existing close_heat_balance path (no outer
    scalar search needed) must still be reached without the old blanket
    guard rejecting it."""
    hx = _exchanger()
    inside = BalanceSideSpec(
        provider=_inside_provider(), p=P, m_dot=None, T_in=314.65, T_out=335.65,
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


def test_near_onset_auto_with_unknown_inside_mass_flow_still_solves() -> None:
    """Test 4: same as Test 3, but for the near-onset sub-regime -- this
    specifically protects the PG40-loop-discovered AUTO transition (spec
    section 5) for a Rating problem with an unknown non-condensing-side
    variable. Uses a widened activation band (a legitimate, user-facing
    ``phase_change_activation_band_K`` setting) so the same fast dry-regime
    operating point below is classified near-onset instead of plain dry,
    without needing an expensive search to land exactly inside a ~0.5 K
    default band."""
    hx = _exchanger()
    inside = BalanceSideSpec(
        provider=_inside_provider(), p=P, m_dot=None, T_in=314.65, T_out=335.65,
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
    assert pc.near_onset is True
    assert pc.possible is True
    assert pc.converged is True
    assert result.closed_balance.inside.m_dot > 0.0
    assert pc.m_dot_condensate == 0.0
    assert pc.Q_latent == 0.0
    assert pc.Q_sensible == pytest.approx(result.Q_required)
    assert pc.Q_total == pytest.approx(result.Q_required)


def test_bracket_solver_tolerates_discontinuous_residual_at_regime_kink() -> None:
    """Test 5: a root search must not fail merely because the residual has
    a kink where trial points cross a regime boundary (spec section 8) --
    exercised directly against the generic bracketed solver with a
    synthetic residual that jumps discontinuously at x=10, mirroring the
    real dry/active duty-residual discontinuity found in
    ``_rating_raw_available_duty`` (deterministic and fast; the real
    end-to-end regime-crossing case is already exercised, expensively, by
    the active/dry tests above)."""

    def kinked_residual(x: float) -> float:
        if x < 10.0:
            return 100.0 - 10.0 * x          # dry-like branch: crosses 0 at x=10
        return -50.0 - 5.0 * (x - 10.0)      # active-like branch: starts negative

    x_root, r_root, iterations = _bracket_and_solve_monotonic_root(
        kinked_residual, 5.0,
        x_min=0.0, x_max=100.0,
        x_tolerance=1.0e-6, residual_tolerance=1.0e-6,
        variable_name="synthetic_x",
    )
    assert x_root == pytest.approx(10.0, abs=1.0e-4)
    assert abs(r_root) <= 1.0e-5
    assert iterations >= 1


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
    with pytest.raises(ValueError):
        hx.rate(inside, outside, include_simulation=False)


def test_no_valid_range_for_unknown_outlet_temperature_raises_closure_error() -> None:
    """Test 7 (direct unit test): the bounds check inside the closure
    solver itself must reject an inside.T_in essentially equal to
    outside.T_in -- no positive driving force exists for any inside.T_out,
    so there is no physically valid range to even start a bracket search
    (spec section 9). Called directly (bypassing onset detection, which
    is not this check's concern) for a fast, deterministic test of the
    bounds guard alone."""
    from core.phase_change.integration import PhaseChangeSettings
    from core.phase_change.rating_integration import (
        _solve_rating_single_unknown_inside_variable,
    )

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
        _solve_rating_single_unknown_inside_variable(
            hx, inside, outside,
            cold_start_m_dot=8.0, cold_start_T_out=420.5,
            flow_arrangement=None, K_inlet=0.5, K_outlet=1.0, K_turn=1.5,
            euler_provider="zukauskas",
            finned_heat_transfer_provider=None,
            finned_pressure_drop_provider=None,
            include_simulation=False,
            over_specified_tolerance=1e-3,
            max_iterations=25,
            wall_temperature_tolerance_K=0.05,
            relative_alfa_tolerance=1e-3,
            relaxation_factor=0.5,
            settings=PhaseChangeSettings(),
        )


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
    with pytest.raises(ValueError):
        hx.rate(inside, outside, include_simulation=False)
