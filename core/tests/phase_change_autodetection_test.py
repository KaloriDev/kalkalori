# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Autodetection behavior tests for BareTubeHeatExchanger.simulate (v0.6.0).

Covers spec section 36's scenario list: dry provider, wet gas without
onset, wet gas with active outside condensation (AUTO), DISABLED, near
onset (regime-decision level, see note below), inside condensation
(unsupported under AUTO, sensible-only under DISABLED), both sides
possible simultaneously, and the iterate=False guard.

Run:
    pytest -q core/tests/phase_change_autodetection_test.py
"""

from __future__ import annotations

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.properties.gas_mixture import GasMixtureSpec, GasMixturePropertyProvider
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.integration import (
    MultiplePhaseChangeSidesError,
    check_single_active_side,
)
from core.phase_change.regime import ThermalRegime, decide_regime
from core.phase_change.types import PhaseChangeDirection, PhaseChangeMode
from core.phase_change.wet_gas_composition import wet_gas_spec_at_water_ratio


def _hx(n_rows: int = 20, n_tubes_per_row: int = 30) -> BareTubeHeatExchanger:
    tube = BareTube(D_i=25e-3 - 2 * 1.5e-3, D_o=25e-3, length_total=2.8, length_effective=2.8, wall_k=50.0)
    bundle = TubeBundle(
        tube=tube, n_rows=n_rows, n_tubes_per_row=n_tubes_per_row,
        pitch_transverse=35e-3, pitch_longitudinal=35e-3,
        layout="staggered", n_passes_tube=2, flow_arrangement="counterflow",
    )
    return BareTubeHeatExchanger(bundle)


def _dry_air_spec() -> GasMixtureSpec:
    return GasMixtureSpec(components={"N2": 0.79, "O2": 0.21}, basis="mole")


def _wet_gas_spec(y_h2o: float) -> GasMixtureSpec:
    remainder = 1.0 - y_h2o
    base = {"N2": 0.70, "O2": 0.10, "CO2": 0.08}
    total_base = sum(base.values())
    return GasMixtureSpec(
        components={k: v * remainder / total_base for k, v in base.items()} | {"H2O": y_h2o},
        basis="mole",
    )


# ---------------------------------------------------------------------------
# Dry provider: no capability anywhere, result must equal sensible-only.
# ---------------------------------------------------------------------------
def test_dry_provider_has_no_capability_and_matches_sensible_only() -> None:
    from core.models.simulation import run_simulation

    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=15.0, T_in=290.0, p=101325.0)
    outside = HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=6.0, T_in=420.0, p=101325.0)

    expected = run_simulation(hx, inside, outside)
    result = hx.simulate(inside, outside)

    assert result.inside_phase_change.capable is False
    assert result.outside_phase_change.capable is False
    assert result.outside_phase_change.active is False
    assert result.q == expected.q
    assert result.T_out_inside == expected.T_out_inside
    assert result.T_out_outside == expected.T_out_outside


# ---------------------------------------------------------------------------
# Wet gas, capable, but the dry baseline never reaches the dew point.
# ---------------------------------------------------------------------------
def test_wet_gas_capable_but_no_condensation_matches_sensible_only() -> None:
    from core.models.simulation import run_simulation

    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=5.0, T_in=300.0, p=101325.0)
    outside = HXSideInput(provider=GasMixturePropertyProvider(_wet_gas_spec(0.08)), m_dot=8.0, T_in=450.0, p=101325.0)

    expected = run_simulation(hx, inside, outside)
    result = hx.simulate(inside, outside)

    pc = result.outside_phase_change
    assert pc.capable is True
    assert pc.active is False
    assert pc.m_dot_condensate == 0.0
    assert pc.Q_latent == 0.0
    assert result.q == expected.q
    assert result.T_out_inside == expected.T_out_inside
    assert result.T_out_outside == expected.T_out_outside


# ---------------------------------------------------------------------------
# Wet gas outside, AUTO -> active condensation.
# ---------------------------------------------------------------------------
def test_wet_gas_outside_auto_condensation_active() -> None:
    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=15.0, T_in=290.0, p=101325.0)
    outside = HXSideInput(provider=GasMixturePropertyProvider(_wet_gas_spec(0.17)), m_dot=6.0, T_in=420.0, p=101325.0)

    result = hx.simulate(inside, outside)
    pc = result.outside_phase_change

    assert pc.active is True
    assert pc.direction is PhaseChangeDirection.CONDENSATION
    assert pc.m_dot_condensate > 0.0
    assert pc.W_out < pc.W_in
    assert pc.Q_latent > 0.0
    assert pc.Q_total == pytest.approx(pc.Q_sensible + pc.Q_latent, rel=1e-9)


# ---------------------------------------------------------------------------
# Same case, DISABLED -> sensible-only, but flagged as possible.
# ---------------------------------------------------------------------------
def test_wet_gas_outside_disabled_gives_sensible_only_with_warning() -> None:
    from core.phase_change.warning_codes import PHASE_CHANGE_DISABLED_BUT_POSSIBLE

    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=15.0, T_in=290.0, p=101325.0)
    outside = HXSideInput(
        provider=GasMixturePropertyProvider(_wet_gas_spec(0.17)), m_dot=6.0, T_in=420.0, p=101325.0,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )

    result = hx.simulate(inside, outside)
    pc = result.outside_phase_change

    assert pc.active is False
    assert pc.possible is True
    assert pc.m_dot_condensate == 0.0
    assert pc.W_out == pc.W_in
    assert pc.m_dot_gas_in == outside.m_dot
    assert pc.m_dot_gas_out == outside.m_dot
    assert pc.m_dot_water_vapor_out == pc.m_dot_water_vapor_in
    assert pc.Q_latent == 0.0
    hydraulic = result.outside_tube_bank_hydraulic
    assert hydraulic.inlet.face_mass_flux == hydraulic.midpoint.face_mass_flux
    assert hydraulic.midpoint.face_mass_flux == hydraulic.outlet.face_mass_flux
    assert result.outside_properties_outlet.props == outside.provider.at(
        T=result.T_out_outside, p=outside.p
    )
    capability = detect_phase_change_capability(outside.provider)
    assert (
        wet_gas_spec_at_water_ratio(capability, pc.W_out).to_mole_fractions()
        == wet_gas_spec_at_water_ratio(capability, pc.W_in).to_mole_fractions()
    )
    assert any(w.code == PHASE_CHANGE_DISABLED_BUT_POSSIBLE for w in pc.warnings)


# ---------------------------------------------------------------------------
# Near onset: regime-decision level (deterministic; a physically-engineered
# near-onset thermal operating point is not robust to reproduce exactly).
# ---------------------------------------------------------------------------
def test_near_onset_margin_resolves_to_dry_with_warning_code() -> None:
    decision = decide_regime(
        dew_point_K=320.0,
        wall_temperature_representative_K=320.05,  # 0.05 K margin, inside a 0.5 K band
        onset_tolerance_K=0.0,
        activation_band_K=0.5,
    )
    assert decision.regime is ThermalRegime.NEAR_ONSET
    assert decision.is_near_onset is True

    clearly_dry = decide_regime(
        dew_point_K=320.0, wall_temperature_representative_K=325.0,
        onset_tolerance_K=0.0, activation_band_K=0.5,
    )
    assert clearly_dry.regime is ThermalRegime.DRY

    clearly_condensing = decide_regime(
        dew_point_K=320.0, wall_temperature_representative_K=310.0,
        onset_tolerance_K=0.0, activation_band_K=0.5,
    )
    assert clearly_condensing.regime is ThermalRegime.CONDENSING


def test_near_onset_does_not_oscillate_across_repeated_calls() -> None:
    """The same dry baseline must resolve to the same regime every time
    (no per-iteration dry/wet flip-flopping)."""
    for _ in range(5):
        decision = decide_regime(
            dew_point_K=320.0, wall_temperature_representative_K=320.05,
            onset_tolerance_K=0.0, activation_band_K=0.5,
        )
        assert decision.regime is ThermalRegime.NEAR_ONSET


# ---------------------------------------------------------------------------
# Inside wet-gas condensation possible under AUTO -> active.
# ---------------------------------------------------------------------------
def test_inside_condensation_possible_under_auto_is_active() -> None:
    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_wet_gas_spec(0.35)), m_dot=1.0, T_in=360.0, p=101325.0)
    outside = HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=25.0, T_in=290.0, p=101325.0)

    result = hx.simulate(inside, outside)
    assert result.inside_phase_change.active is True
    assert result.inside_phase_change.m_dot_condensate > 0.0
    assert result.inside_phase_change.wet_surface_fraction == pytest.approx(1.0)


def test_inside_condensation_disabled_gives_sensible_only_with_warning() -> None:
    from core.phase_change.warning_codes import PHASE_CHANGE_DISABLED_BUT_POSSIBLE

    hx = _hx()
    inside = HXSideInput(
        provider=GasMixturePropertyProvider(_wet_gas_spec(0.35)), m_dot=1.0, T_in=360.0, p=101325.0,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )
    outside = HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=25.0, T_in=290.0, p=101325.0)

    result = hx.simulate(inside, outside)
    pc = result.inside_phase_change
    assert pc.active is False
    assert pc.possible is True
    assert any(w.code == PHASE_CHANGE_DISABLED_BUT_POSSIBLE for w in pc.warnings)
    assert result.outside_phase_change.capable is False


# ---------------------------------------------------------------------------
# Both sides possible simultaneously -> MULTIPLE_PHASE_CHANGE_SIDES_NOT_SUPPORTED.
#
# Tested at the priority-check decision level (deterministic boolean inputs)
# rather than via a physically-engineered dual-condensing thermal operating
# point, which core.phase_change.integration.apply_phase_change delegates
# to exactly this function -- see its own module docstring.
# ---------------------------------------------------------------------------
def test_both_sides_possible_raises_multiple_sides_error() -> None:
    with pytest.raises(MultiplePhaseChangeSidesError):
        check_single_active_side(True, True, iterate=True)


def test_inside_only_is_supported_by_single_active_side_guard() -> None:
    check_single_active_side(True, False, iterate=True)


def test_inside_active_requires_iterative_simulation() -> None:
    with pytest.raises(ValueError):
        check_single_active_side(True, False, iterate=False)


# ---------------------------------------------------------------------------
# iterate=False guards.
# ---------------------------------------------------------------------------
def test_iterate_false_with_auto_possible_condensation_raises() -> None:
    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=15.0, T_in=290.0, p=101325.0)
    outside = HXSideInput(provider=GasMixturePropertyProvider(_wet_gas_spec(0.17)), m_dot=6.0, T_in=420.0, p=101325.0)

    with pytest.raises(ValueError):
        hx.simulate(inside, outside, iterate=False)


def test_iterate_false_with_disabled_gives_dry_result_and_warning() -> None:
    from core.phase_change.warning_codes import PHASE_CHANGE_DISABLED_BUT_POSSIBLE

    hx = _hx()
    inside = HXSideInput(provider=GasMixturePropertyProvider(_dry_air_spec()), m_dot=15.0, T_in=290.0, p=101325.0)
    outside = HXSideInput(
        provider=GasMixturePropertyProvider(_wet_gas_spec(0.17)), m_dot=6.0, T_in=420.0, p=101325.0,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )

    result = hx.simulate(inside, outside, iterate=False)
    assert result.converged is True
    pc = result.outside_phase_change
    assert pc.active is False
    assert any(w.code == PHASE_CHANGE_DISABLED_BUT_POSSIBLE for w in pc.warnings)
