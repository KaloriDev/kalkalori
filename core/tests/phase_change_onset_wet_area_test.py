# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Tests for the condensation-onset (minimum wall temperature) and
partial wet-area fix (v0.6.0 patch).

Covers:
- core.phase_change.regime.evaluate_condensation_onset (pure function,
  Tmin-based, possible/active/near_onset semantics),
- core.phase_change.wet_surface_fraction.estimate_wet_surface_fraction
  (pure linear function, plus the degenerate near-zero-span fallback),
- A_wet scaling of the condensation rate (mass transfer only),
- sensible heat transfer using the full outside area regardless of
  wet_surface_fraction,
- AUTO vs DISABLED end-to-end behavior for a partially-wet case, with the
  new onset/wet-area diagnostics populated even when the wet solver did
  not run,
- a physically-sensible integration case with 0 < wet_surface_fraction < 1,
- final thermal_state consistency after an active solve.

Run:
    pytest -q core/tests/phase_change_onset_wet_area_test.py
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.properties.gas_mixture import GasMixtureSpec, GasMixturePropertyProvider
from core.phase_change.mass_heat_transfer import condensation_rate
from core.phase_change.regime import evaluate_condensation_onset
from core.phase_change.types import PhaseChangeMode
from core.phase_change.wet_surface_fraction import (
    DEGENERATE_METHOD_NAME,
    LINEAR_METHOD_NAME,
    estimate_wet_surface_fraction,
)


# ---------------------------------------------------------------------------
# Section 18: onset tests
# ---------------------------------------------------------------------------
def test_onset_case1_min_below_dew_and_mean_above_is_possible_and_active() -> None:
    """T_wall_min < T_dew < T_wall_mean -- this is the bug being fixed:
    a mean-based check would miss this, but the min-based check must not."""
    decision = evaluate_condensation_onset(
        dew_point_temperature=320.0,
        wall_temperature_min=310.0,  # clearly below dew point
        onset_tolerance_K=0.0,
        activation_band_K=0.5,
    )
    assert decision.possible is True
    assert decision.active is True  # margin=10K, far past the 0.25K half-band
    assert decision.near_onset is False
    assert decision.margin_K == pytest.approx(10.0)


def test_onset_case2_min_above_dew_is_dry() -> None:
    decision = evaluate_condensation_onset(
        dew_point_temperature=320.0,
        wall_temperature_min=325.0,
        onset_tolerance_K=0.0,
        activation_band_K=0.5,
    )
    assert decision.possible is False
    assert decision.active is False
    assert decision.near_onset is False


def test_onset_case3_near_onset_band() -> None:
    """T_wall_min slightly below T_dew, inside the activation band."""
    decision = evaluate_condensation_onset(
        dew_point_temperature=320.0,
        wall_temperature_min=319.95,  # 0.05 K margin, inside a 0.5 K band
        onset_tolerance_K=0.0,
        activation_band_K=0.5,
    )
    assert decision.possible is True
    assert decision.near_onset is True
    assert decision.active is False


def test_onset_case4_decision_depends_on_min_not_mean() -> None:
    """T_wall_mean < T_dew, but the onset function never sees a mean value
    at all -- only wall_temperature_min drives the decision."""
    # A case where the (hypothetical) mean would suggest "clearly dry" if
    # someone mistakenly compared T_dew to a warmer "mean"-like value, but
    # the actual minimum is what the function receives and decides from.
    decision = evaluate_condensation_onset(
        dew_point_temperature=320.0,
        wall_temperature_min=305.0,  # the only input; far below T_dew
        onset_tolerance_K=0.0,
        activation_band_K=0.5,
    )
    assert decision.possible is True
    assert decision.active is True
    # Confirms the function signature has no "mean" parameter at all --
    # the decision is structurally incapable of depending on it.
    import inspect

    sig = inspect.signature(evaluate_condensation_onset)
    assert "wall_temperature_mean" not in sig.parameters
    assert "wall_temperature_min" in sig.parameters


def test_onset_validates_settings() -> None:
    with pytest.raises(ValueError):
        evaluate_condensation_onset(
            dew_point_temperature=320.0, wall_temperature_min=310.0,
            onset_tolerance_K=-1.0,
        )
    with pytest.raises(ValueError):
        evaluate_condensation_onset(
            dew_point_temperature=320.0, wall_temperature_min=310.0,
            activation_band_K=0.0,
        )


# ---------------------------------------------------------------------------
# Section 19: wet_surface_fraction tests
# ---------------------------------------------------------------------------
def test_wet_fraction_dry_surface() -> None:
    result = estimate_wet_surface_fraction(
        dew_point_temperature=300.0,
        wall_temperature_min=310.0, wall_temperature_mean=320.0, wall_temperature_max=330.0,
    )
    assert result.wet_surface_fraction == 0.0
    assert result.method == LINEAR_METHOD_NAME


def test_wet_fraction_fully_wet_surface() -> None:
    result = estimate_wet_surface_fraction(
        dew_point_temperature=340.0,
        wall_temperature_min=310.0, wall_temperature_mean=320.0, wall_temperature_max=330.0,
    )
    assert result.wet_surface_fraction == 1.0


def test_wet_fraction_half_surface() -> None:
    result = estimate_wet_surface_fraction(
        dew_point_temperature=0.5 * (310.0 + 330.0),
        wall_temperature_min=310.0, wall_temperature_mean=320.0, wall_temperature_max=330.0,
    )
    assert result.wet_surface_fraction == pytest.approx(0.5)


def test_wet_fraction_one_third_surface() -> None:
    result = estimate_wet_surface_fraction(
        dew_point_temperature=323.15,
        wall_temperature_min=313.15, wall_temperature_mean=328.15, wall_temperature_max=343.15,
    )
    assert result.wet_surface_fraction == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_wet_fraction_degenerate_envelope_no_nan_or_infinity() -> None:
    result = estimate_wet_surface_fraction(
        dew_point_temperature=320.0,
        wall_temperature_min=320.0001, wall_temperature_mean=320.0001, wall_temperature_max=320.0002,
        temperature_span_tolerance_K=1e-3,
    )
    assert math.isfinite(result.wet_surface_fraction)
    assert 0.0 <= result.wet_surface_fraction <= 1.0
    assert result.method == DEGENERATE_METHOD_NAME


def test_wet_fraction_degenerate_envelope_falls_back_to_mean() -> None:
    clearly_wet = estimate_wet_surface_fraction(
        dew_point_temperature=330.0,
        wall_temperature_min=320.0, wall_temperature_mean=320.0, wall_temperature_max=320.0,
        temperature_span_tolerance_K=1e-3, activation_band_K=0.5,
    )
    assert clearly_wet.wet_surface_fraction == 1.0

    clearly_dry = estimate_wet_surface_fraction(
        dew_point_temperature=300.0,
        wall_temperature_min=320.0, wall_temperature_mean=320.0, wall_temperature_max=320.0,
        temperature_span_tolerance_K=1e-3, activation_band_K=0.5,
    )
    assert clearly_dry.wet_surface_fraction == 0.0


def test_wet_fraction_rejects_bad_tolerances() -> None:
    with pytest.raises(ValueError):
        estimate_wet_surface_fraction(
            dew_point_temperature=320.0, wall_temperature_min=310.0,
            wall_temperature_mean=320.0, wall_temperature_max=330.0,
            temperature_span_tolerance_K=0.0,
        )
    with pytest.raises(ValueError):
        estimate_wet_surface_fraction(
            dew_point_temperature=320.0, wall_temperature_min=310.0,
            wall_temperature_mean=320.0, wall_temperature_max=330.0,
            activation_band_K=-1.0,
        )


# ---------------------------------------------------------------------------
# Section 20: A_wet scaling of condensate rate
# ---------------------------------------------------------------------------
def test_condensate_scales_with_wet_area() -> None:
    common = dict(
        alfa_dry=90.0, cp_gas=1100.0, W_bulk=0.12, W_sat_surface=0.08,
        m_dot_water_vapor_available=1.0, lewis_number=1.0,
    )
    A_full = 100.0
    rate_full = condensation_rate(A_wet=A_full, **common)
    rate_half = condensation_rate(A_wet=0.5 * A_full, **common)

    assert rate_half == pytest.approx(0.5 * rate_full, rel=1e-9)


# ---------------------------------------------------------------------------
# Section 21: sensible heat transfer uses the full outside area
# ---------------------------------------------------------------------------
def test_sensible_heat_uses_full_area_regardless_of_wet_fraction() -> None:
    from core.phase_change.outside_condensation_solver import _solve_interface_state

    # Parameters chosen so the interface genuinely condenses (T_s below the
    # ~322 K dew point implied by W_bulk=0.08, M_dry=0.03 kg/mol at 1 atm):
    # a small alfa_o_dry*A_o (large outside-film resistance relative to the
    # removal network) pulls T_s well below T_bulk_outside.
    common = dict(
        alfa_o_dry=10.0, A_o=100.0, T_bulk_outside=350.0, T_bulk_inside=290.0,
        R_downstream=0.0001, cp_gas=1100.0, W_bulk=0.08, p_outside=101325.0,
        M_dry=0.0300, m_dot_water_vapor_available=0.5, lewis_number=1.0,
    )
    T_s_full, q_sens_full, q_lat_full, m_full = _solve_interface_state(A_wet=100.0, **common)
    T_s_partial, q_sens_partial, q_lat_partial, m_partial = _solve_interface_state(A_wet=30.0, **common)

    assert q_lat_full > 0.0 and q_lat_partial > 0.0  # confirm both runs actually condense

    # In both runs, q_sensible must reconstruct exactly from the FULL A_o
    # (never A_wet), at whatever T_s that run converged to.
    assert q_sens_full == pytest.approx(10.0 * 100.0 * (350.0 - T_s_full), rel=1e-6)
    assert q_sens_partial == pytest.approx(10.0 * 100.0 * (350.0 - T_s_partial), rel=1e-6)
    # A smaller wet area means less latent heat generated at any given T_s,
    # so the coupled balance settles at a *colder* interface (sensible heat
    # transfer must pick up the shortfall) -- confirms A_wet actually
    # changed the coupled solution between the two runs, in the physically
    # correct direction.
    assert T_s_partial < T_s_full
    assert q_lat_partial < q_lat_full


def test_zero_wet_area_gives_zero_condensate() -> None:
    from core.phase_change.outside_condensation_solver import _solve_interface_state

    T_s, q_sensible, q_latent, m_dot_cond = _solve_interface_state(
        alfa_o_dry=90.0, A_o=100.0, A_wet=0.0, T_bulk_outside=400.0, T_bulk_inside=310.0,
        R_downstream=0.02, cp_gas=1100.0, W_bulk=0.12, p_outside=101325.0,
        M_dry=0.0300, m_dot_water_vapor_available=0.5, lewis_number=1.0,
    )
    assert q_latent == 0.0
    assert m_dot_cond == 0.0
    assert q_sensible > 0.0


# ---------------------------------------------------------------------------
# Fixtures for end-to-end AUTO/DISABLED/integration tests
# ---------------------------------------------------------------------------
def _hx() -> BareTubeHeatExchanger:
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


# ---------------------------------------------------------------------------
# Section 22: AUTO / DISABLED for a partially-wet case
# ---------------------------------------------------------------------------
def test_auto_partial_wetting_activates_with_positive_wet_area_and_condensate() -> None:
    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_dry_spec()), m_dot=15.0, T_in=290.0, p=101325.0)
    outside = HXSideInput(provider=GasMixturePropertyProvider(_wet_spec()), m_dot=6.0, T_in=420.0, p=101325.0)

    result = hx.simulate(inside, outside)
    pc = result.outside_phase_change

    assert pc.possible is True
    assert pc.active is True
    assert pc.wet_surface_fraction is not None and pc.wet_surface_fraction > 0.0
    assert pc.wet_area is not None and pc.wet_area > 0.0
    assert pc.m_dot_condensate > 0.0


def test_disabled_reports_onset_and_wall_diagnostics_without_running_solver() -> None:
    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_dry_spec()), m_dot=15.0, T_in=290.0, p=101325.0)
    outside = HXSideInput(
        provider=GasMixturePropertyProvider(_wet_spec()), m_dot=6.0, T_in=420.0, p=101325.0,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )

    result = hx.simulate(inside, outside)
    pc = result.outside_phase_change

    assert pc.active is False
    assert pc.possible is True
    assert pc.m_dot_condensate == 0.0
    assert pc.Q_latent == 0.0
    # Onset/wall diagnostics must be populated from the dry baseline even
    # though the wet solver never ran (spec section 15).
    assert pc.onset_margin_K is not None and pc.onset_margin_K > 0.0
    assert pc.onset_wall_temperature is not None
    assert pc.onset_temperature_method is not None
    assert pc.wall_temperature_min is not None
    assert pc.wall_temperature_mean is not None
    assert pc.wall_temperature_max is not None
    assert pc.dew_point_in is not None


# ---------------------------------------------------------------------------
# Section 23: integration test with a partially-wet surface
# ---------------------------------------------------------------------------
def test_partial_wet_surface_integration() -> None:
    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_dry_spec()), m_dot=15.0, T_in=290.0, p=101325.0)
    outside = HXSideInput(provider=GasMixturePropertyProvider(_wet_spec()), m_dot=6.0, T_in=420.0, p=101325.0)

    result = hx.simulate(inside, outside)
    pc = result.outside_phase_change

    assert result.converged is True
    assert pc.converged is True
    assert 0.0 < pc.wet_surface_fraction < 1.0
    assert pc.wet_area == pytest.approx(pc.outside_total_area * pc.wet_surface_fraction, rel=1e-9)
    assert pc.m_dot_condensate > 0.0
    assert pc.W_out < pc.W_in
    assert pc.Q_latent > 0.0
    assert pc.Q_total == pytest.approx(pc.Q_sensible + pc.Q_latent, rel=1e-9)
    assert pc.wet_surface_fraction_method in (LINEAR_METHOD_NAME, DEGENERATE_METHOD_NAME)

    for value in (
        pc.wet_surface_fraction, pc.wet_area, pc.outside_total_area,
        pc.m_dot_condensate, pc.Q_sensible, pc.Q_latent, pc.Q_total,
        pc.wall_temperature_min, pc.wall_temperature_max,
    ):
        assert value is not None and math.isfinite(value)


def test_thermal_state_consistent_with_wet_solution() -> None:
    """Fix (spec section 16): thermal_state must not mix dry-baseline alfa/UA
    with a wet-solver-sourced wall_temperature_envelope/PhaseChangeResult."""
    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_dry_spec()), m_dot=15.0, T_in=290.0, p=101325.0)
    outside = HXSideInput(provider=GasMixturePropertyProvider(_wet_spec()), m_dot=6.0, T_in=420.0, p=101325.0)

    result = hx.simulate(inside, outside)
    ts = result.thermal_state
    pc = result.outside_phase_change

    assert result.inside_alfa_mean == ts.alfa_i
    assert result.outside_alfa_mean == ts.alfa_o
    assert result.U_mean == ts.U
    assert result.UA == ts.UA
    assert ts.converged == pc.converged
    assert ts.iterations == pc.iterations
    assert ts.outside_wall_temperature == pc.wall_temperature_mean
    # UA must reconstruct from alfa_i/alfa_o/wall resistance exactly like
    # the dry solver's own self-check in thermal_iteration.py.
    A_i = hx.bundle.total_inner_area
    A_o = hx.bundle.total_outer_area
    R_w = hx.tube_wall_resistance()
    UA_reconstructed = 1.0 / (1.0 / (ts.alfa_i * A_i) + R_w + 1.0 / (ts.alfa_o * A_o))
    assert UA_reconstructed == pytest.approx(ts.UA, rel=1e-9)
