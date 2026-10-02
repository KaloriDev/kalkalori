# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""AUTO uses one production caloric path through dry and wet regimes.

These existing economizer inputs exercise the physical source-profile onset;
the retired global solver's activation band and collapse-to-legacy behavior
are intentionally no longer the AUTO contract.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from core import WetCoilSolverOptions

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


def _economizer_hx() -> BareTubeHeatExchanger:
    """Synthetic finned economizer, sized only to sit near condensation onset."""
    core = BareTube(
        D_i=0.0189, D_o=0.0212, length_total=2.75, length_effective=2.72, wall_k=45.0,
    )
    tube = CircularFinnedTube(
        core_tube=core,
        fin_k=175.0,
        D_fin=0.0508,
        D_root=0.0224,
        fin_thickness_root=0.00035,
        fin_thickness_tip=0.00018,
        fin_pitch=0.0028,
        fin_contact_efficiency=0.92,
    )
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=tube,
            n_rows=12,
            n_tubes_per_row=84,
            pitch_transverse=0.052,
            pitch_longitudinal=0.045,
            layout="staggered",
            n_passes_tube=16,
            n_passes_transverse=4,
            flow_arrangement="counterflow",
        )
    )


def _liquid_stub() -> ConstantPropertyProvider:
    return ConstantPropertyProvider(
        FluidTransportProperties(rho=1_010.0, mu=1.6e-3, k=0.38, cp=3_550.0)
    )


def _wet_air_provider() -> GasMixturePropertyProvider:
    return GasMixturePropertyProvider(
        gas_mixture_from_dry_composition_and_water_ratio(
            dry_components={"N2": 0.77, "O2": 0.21, "CO2": 0.02},
            dry_basis="mole",
            water_ratio=0.055,
            imposed_phase="gas",
        )
    )


def _simulate_at_liquid_inlet(T_liquid_in_K: float):
    hx = _economizer_hx()
    # Preserve micowatt regression checks independently of engineering defaults.
    return hx.simulate(
        HXSideInput(
            provider=_liquid_stub(),
            m_dot=31_500.0 / 3600.0,
            T_in=T_liquid_in_K,
            p=250_000.0,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        HXSideInput(
            provider=_wet_air_provider(),
            m_dot=142_800.0 / 3600.0,
            T_in=335.15,
            p=P,
            phase_change_mode=PhaseChangeMode.AUTO,
        ),
        wet_solver_options=WetCoilSolverOptions(
            energy_tolerance_W=2e-4, outlet_temperature_tolerance_K=2e-7,
            timeout_s=None,
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


@pytest.fixture(scope="module")
def transition_results():
    return {c:_simulate_at_liquid_inlet(c+273.15) for c in (20.0,37.1,37.2,45.0)}


@pytest.mark.parametrize("temperature",[20.0,37.1,37.2,45.0])
def test_auto_regime_transition_always_returns_a_valid_result(transition_results,temperature):
    result=transition_results[temperature]
    pc=result.outside_phase_change
    _assert_valid_hx_result(result)
    assert pc.method == "elmahdy_mitalas_energyplus_v25_2_adapted"
    assert pc.active == (pc.regime != "DRY")
    assert pc.active == (pc.onset_margin_K > 0)
    assert not pc.near_onset  # No artificial activation band in production AUTO.
    assert result.ua_is_equivalent
    assert result.q == pytest.approx(
        31500/3600*3550*(result.T_out_inside-(temperature+273.15)),abs=1e-5)
    if pc.active:
        assert pc.m_dot_condensate > 0 and pc.Q_latent > 0
        assert pc.wet_coil_diagnostics["surface_states"]
    else:
        assert pc.m_dot_condensate == pc.Q_latent == pc.H_drain == 0
        assert pc.Q_sensible == pytest.approx(result.q)
        assert pc.Q_total == pytest.approx(result.q)


def test_crossing_physical_onset_changes_regime_not_exception(transition_results):
    ordered=list(transition_results.values())
    assert ordered[0].outside_phase_change.active
    assert ordered[-1].outside_phase_change.regime == "DRY"
    for colder,warmer in zip(ordered,ordered[1:]):
        assert colder.q > warmer.q > 0
        assert colder.outside_condensate_mass_flow >= warmer.outside_condensate_mass_flow


def test_auto_dry_and_disabled_are_explicit_different_model_paths(transition_results):
    from core.models.simulation import run_simulation
    inside=HXSideInput(_liquid_stub(),31500/3600,318.15,250000,
                      phase_change_mode=PhaseChangeMode.DISABLED)
    outside=HXSideInput(_wet_air_provider(),142800/3600,335.15,P,
                       phase_change_mode=PhaseChangeMode.DISABLED)
    hx=_economizer_hx()
    disabled=hx.simulate(inside,outside)
    legacy=run_simulation(hx,inside,outside)
    assert (disabled.q,disabled.T_out_inside,disabled.T_out_outside,disabled.UA) == (
        legacy.q,legacy.T_out_inside,legacy.T_out_outside,legacy.UA)
    auto=transition_results[45.0]
    assert not auto.outside_phase_change.active
    assert auto.ua_is_equivalent and not disabled.ua_is_equivalent
    assert auto.wet_coil_diagnostics["global_wet_model"] == "elmahdy_mitalas_energyplus_v25_2_adapted"


def test_public_wet_route_never_calls_retired_global_closure(monkeypatch):
    import core.phase_change.outside_condensation_solver as legacy
    from core.tests.wet_coil_public_test import context
    def forbidden(*args,**kwargs):
        raise AssertionError("retired outside global wet closure was invoked")
    monkeypatch.setattr(legacy,"solve_outside_condensation",forbidden)
    monkeypatch.setattr(legacy,"solve_wet_finned_surface",forbidden)
    hx,inside,outside=context(True)
    result=hx.simulate(inside,outside)
    assert result.outside_phase_change.active
    assert result.outside_phase_change.regime == "FULLY_WET"
