# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Regression coverage for the v0.7.5 AUTO dry/near-onset/wet transition.

A real closed-loop acceptance run (KDV-26004 DK045, PG40 economizer +
air-heater loop) surfaced two related defects on a circular-finned outside
surface operating close to its condensation onset:

1. ``PhaseChangeMode.AUTO`` legitimately resolves to a DRY or NEAR_ONSET
   regime for some geometries/operating points -- ``active=False`` is a
   valid converged result, not a calculation failure. Consuming code that
   asserted ``outside_phase_change.active`` unconditionally was wrong.
2. The non-active ``PhaseChangeResult`` built for that regime reported
   ``Q_sensible == Q_latent == Q_total == 0.0`` even though the exchanger
   had a finite, real dry duty (``HXSimulationResult.q``).

This module locks in the fix for both: crossing the AUTO onset threshold
changes the reported *regime* (dry / near_onset / condensing), never turns
a physically valid exchanger result into an exception, and the non-active
``PhaseChangeResult`` always exposes the real sensible duty.

It also covers the companion hardening (spec section 8): if the dry-baseline
onset screen activates AUTO but the converged nonlinear wet-fin field itself
finds zero net condensate (a near-boundary collapse, not a solver
contradiction), the call must still return a valid, diagnostic-rich dry
result instead of raising.

Uses the same circular-finned geometry/providers as
``wet_finned_simulation_test.py`` (no CoolProp/IAPWS required); the glycol
inlet temperature sweep below reproduces the wet -> near_onset/dry -> dry
transition actually observed in the KDV-26004 acceptance geometry.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.phase_change.types import PhaseChangeMode
from core.phase_change.warning_codes import PHASE_CHANGE_WET_SOLUTION_COLLAPSED_TO_DRY
from core.phase_change.wet_finned_surface import WetFinState
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import (
    GasMixturePropertyProvider,
    gas_mixture_from_dry_composition_and_water_ratio,
)

P = 101_325.0


def _kdv_like_hx() -> BareTubeHeatExchanger:
    """Same finned economizer geometry as the KDV-26004 acceptance loop."""
    core = BareTube(
        D_i=0.016, D_o=0.018, length_total=3.33, length_effective=3.30, wall_k=15.0,
    )
    tube = CircularFinnedTube(
        core_tube=core,
        fin_k=200.0,
        D_fin=0.045,
        D_root=0.0196,
        fin_thickness_root=0.0003,
        fin_thickness_tip=0.0002,
        fin_pitch=0.0023,
        fin_contact_efficiency=0.95,
    )
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=tube,
            n_rows=15,
            n_tubes_per_row=110,
            pitch_transverse=0.047,
            pitch_longitudinal=0.0407,
            layout="staggered",
            n_passes_tube=18,
            n_passes_transverse=6,
            flow_arrangement="counterflow",
        )
    )


def _glycol_stub() -> ConstantPropertyProvider:
    return ConstantPropertyProvider(
        FluidTransportProperties(rho=1_020.0, mu=2.0e-3, k=0.42, cp=3_650.0)
    )


def _wet_air_provider() -> GasMixturePropertyProvider:
    return GasMixturePropertyProvider(
        gas_mixture_from_dry_composition_and_water_ratio(
            dry_components={"N2": 0.79, "O2": 0.21},
            dry_basis="mole",
            water_ratio=0.048,
            imposed_phase="gas",
        )
    )


def _simulate_at_glycol_inlet(T_glycol_in_K: float):
    hx = _kdv_like_hx()
    return hx.simulate(
        HXSideInput(
            provider=_glycol_stub(),
            m_dot=44_115.0 / 3600.0,
            T_in=T_glycol_in_K,
            p=300_000.0,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        HXSideInput(
            provider=_wet_air_provider(),
            m_dot=189_570.0 / 3600.0,
            T_in=340.15,
            p=P,
            phase_change_mode=PhaseChangeMode.AUTO,
        ),
    )


def _assert_valid_hx_result(result) -> None:
    """Section 16/17 closure checks, independent of the resolved regime."""
    pc = result.outside_phase_change
    assert pc is not None and pc.converged is True
    assert result.converged is True
    assert math.isfinite(result.q) and result.q > 0.0
    assert math.isfinite(result.T_out_inside)
    assert math.isfinite(result.T_out_outside)
    assert math.isfinite(pc.Q_sensible)
    assert math.isfinite(pc.Q_latent)
    assert math.isfinite(pc.Q_total)
    assert pc.Q_total == pytest.approx(pc.Q_sensible + pc.Q_latent, abs=1.0e-6)
    assert pc.m_dot_condensate >= 0.0
    assert abs(pc.mass_balance_error) < 1.0e-6
    assert abs(pc.energy_balance_error) < 1.0e-5
    # active/near_onset/possible form a single consistent regime label.
    assert pc.active + pc.near_onset <= 1  # never both True
    if pc.active:
        assert pc.possible is True
        assert pc.near_onset is False
    if pc.near_onset:
        assert pc.active is False
        assert pc.possible is True


@pytest.mark.parametrize(
    "T_glycol_in_C, expect_active, expect_near_onset, expect_wet",
    [
        # A: comfortably past onset -> active wet condensation.
        (25.0, True, False, True),
        # A': still active, but close to the activation-band boundary.
        (34.6, True, False, True),
        # B: crossed the activation band -> near-onset, held dry.
        (34.7, False, True, False),
        # C: clearly dry, well past the near-onset band.
        (40.0, False, False, False),
    ],
)
def test_auto_regime_transition_always_returns_a_valid_result(
    T_glycol_in_C, expect_active, expect_near_onset, expect_wet,
) -> None:
    result = _simulate_at_glycol_inlet(T_glycol_in_C + 273.15)
    pc = result.outside_phase_change

    _assert_valid_hx_result(result)
    assert pc.active is expect_active
    assert pc.near_onset is expect_near_onset
    assert (result.wet_finned_surface is not None) is expect_wet
    assert (pc.wet_finned_surface is not None) is expect_wet

    if expect_active:
        assert pc.m_dot_condensate > 0.0
        assert pc.Q_latent > 0.0
    else:
        # Fix (spec section 5/6): active=False is a valid dry/near-onset
        # AUTO result, not a failure, and must expose the real sensible
        # duty rather than a hardcoded zero.
        assert pc.m_dot_condensate == 0.0
        assert pc.Q_latent == 0.0
        assert pc.Q_sensible == pytest.approx(result.q)
        assert pc.Q_total == pytest.approx(result.q)


def test_crossing_onset_threshold_changes_regime_not_exception() -> None:
    """The exact transition pair: one wet, its warmer neighbor dry."""
    wet = _simulate_at_glycol_inlet(34.6 + 273.15)
    dry = _simulate_at_glycol_inlet(34.7 + 273.15)

    assert wet.outside_phase_change.active is True
    assert dry.outside_phase_change.active is False
    assert dry.outside_phase_change.near_onset is True
    # Both sides of the boundary are equally valid, finite HX solutions.
    assert math.isfinite(wet.q) and math.isfinite(dry.q)
    assert wet.q > dry.q > 0.0


def test_dry_side_of_onset_reproduces_the_legacy_disabled_dry_result() -> None:
    """Section 12: the inactive AUTO branch must not diverge from DISABLED."""
    disabled = _kdv_like_hx().simulate(
        HXSideInput(
            provider=_glycol_stub(), m_dot=44_115.0 / 3600.0,
            T_in=40.0 + 273.15, p=300_000.0,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        HXSideInput(
            provider=_wet_air_provider(), m_dot=189_570.0 / 3600.0,
            T_in=340.15, p=P, phase_change_mode=PhaseChangeMode.DISABLED,
        ),
    )
    auto = _simulate_at_glycol_inlet(40.0 + 273.15)

    assert auto.outside_phase_change.active is False
    assert auto.q == disabled.q
    assert auto.T_out_inside == disabled.T_out_inside
    assert auto.T_out_outside == disabled.T_out_outside
    assert auto.UA == disabled.UA


def test_wet_finned_solver_collapse_to_dry_returns_valid_result(monkeypatch) -> None:
    """Spec section 8: a converged-but-zero-condensate wet solve must not
    raise. It must fall back to the exact dry AUTO result with a
    diagnostic ``PHASE_CHANGE_WET_SOLUTION_COLLAPSED_TO_DRY`` warning,
    rather than forcing an internally-contradictory active state or
    failing an otherwise physically valid call."""
    import core.phase_change.outside_condensation_solver as solver_module

    real_solve = solver_module.solve_wet_finned_surface

    def collapsing_solve(*args, **kwargs):
        real_result = real_solve(*args, **kwargs)
        # Simulate the nonlinear radial field converging with no point below
        # local saturation: a self-consistent all-sensible result, not a
        # broken/unconverged one.
        return replace(
            real_result,
            fin_wet_state=WetFinState.DRY,
            fin_wet_fraction=0.0,
            wet_fin_area=0.0,
            Q_fin_latent=0.0,
            Q_fin_total=real_result.Q_fin_sensible,
            m_dot_condensate_fin=0.0,
            wet_dry_boundary_radius=None,
            Q_primary_latent=0.0,
            Q_primary_total=real_result.Q_primary_sensible,
            m_dot_condensate_primary=0.0,
            wet_primary_area=0.0,
            Q_latent=0.0,
            Q_total=real_result.Q_sensible,
            m_dot_condensate=0.0,
            condensate_enthalpy_rate=0.0,
            wet_area=0.0,
            wet_surface_fraction=0.0,
            wall_temperature_wet_mean=None,
            W_sat_wet_surface=None,
        )

    monkeypatch.setattr(solver_module, "solve_wet_finned_surface", collapsing_solve)

    # Use a comfortably-active wet operating point, so the only reason this
    # would fail to condense is the forced collapse above, not genuine onset
    # ambiguity -- isolating the collapse-handling code path.
    result = _simulate_at_glycol_inlet(25.0 + 273.15)
    pc = result.outside_phase_change

    _assert_valid_hx_result(result)
    assert pc.active is False
    assert result.wet_finned_surface is None
    assert pc.wet_finned_surface is None
    assert pc.m_dot_condensate == 0.0
    assert pc.Q_latent == 0.0
    assert pc.Q_sensible == pytest.approx(result.q)
    assert pc.Q_total == pytest.approx(result.q)
    assert PHASE_CHANGE_WET_SOLUTION_COLLAPSED_TO_DRY in {
        w.code for w in pc.warnings
    }

    # And it must reproduce the exact legacy dry (DISABLED) result -- the
    # collapse fallback must not leave a partially-wet residue behind.
    disabled = _kdv_like_hx().simulate(
        HXSideInput(
            provider=_glycol_stub(), m_dot=44_115.0 / 3600.0,
            T_in=25.0 + 273.15, p=300_000.0,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        HXSideInput(
            provider=_wet_air_provider(), m_dot=189_570.0 / 3600.0,
            T_in=340.15, p=P, phase_change_mode=PhaseChangeMode.DISABLED,
        ),
    )
    assert result.q == disabled.q
    assert result.T_out_inside == disabled.T_out_inside
    assert result.T_out_outside == disabled.T_out_outside
