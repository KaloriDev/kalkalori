# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Active outside-H2O condensation on a circular-finned Simulation."""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput, run_simulation
from core.phase_change.types import PhaseChangeDirection, PhaseChangeMode
from core.phase_change.warning_codes import (
    CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY,
    PHASE_CHANGE_DISABLED_BUT_POSSIBLE,
)
from core.phase_change.wet_finned_surface import WetFinState
from core.properties.gas_mixture import (
    GasMixturePropertyProvider,
    GasMixtureSpec,
    gas_mixture_from_dry_composition_and_water_ratio,
)
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


P = 101_325.0


def _wet_finned_hx() -> BareTubeHeatExchanger:
    core = BareTube(
        D_i=0.021,
        D_o=0.025,
        length_total=2.0,
        length_effective=2.0,
        wall_k=45.0,
    )
    tube = CircularFinnedTube(
        core_tube=core,
        fin_k=200.0,
        D_fin=0.050,
        D_root=0.028,
        fin_thickness_root=0.0005,
        fin_pitch=0.0024,
    )
    bundle = TubeBundle(
        tube=tube,
        n_rows=6,
        n_tubes_per_row=8,
        pitch_transverse=0.060,
        pitch_longitudinal=0.060 * math.sqrt(3.0) / 2.0,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement="counterflow",
    )
    return BareTubeHeatExchanger(bundle)


def _side_inputs(
    *,
    mode: PhaseChangeMode = PhaseChangeMode.AUTO,
) -> tuple[HXSideInput, HXSideInput]:
    dry = GasMixturePropertyProvider(
        GasMixtureSpec(
            components={"N2": 0.79, "O2": 0.21},
            basis="mole",
        )
    )
    wet = GasMixturePropertyProvider(
        GasMixtureSpec(
            components={"N2": 0.65, "O2": 0.10, "CO2": 0.08, "H2O": 0.17},
            basis="mole",
        )
    )
    return (
        HXSideInput(
            provider=dry,
            m_dot=15.0,
            T_in=280.0,
            p=P,
            phase_change_mode=mode,
        ),
        HXSideInput(
            provider=wet,
            m_dot=6.0,
            T_in=390.0,
            p=P,
            phase_change_mode=mode,
        ),
    )


@pytest.fixture(scope="module")
def active_result():
    inside, outside = _side_inputs()
    return _wet_finned_hx().simulate(inside, outside)


def test_auto_simulation_solves_one_shared_partial_wet_finned_state(
    active_result,
) -> None:
    result = active_result
    phase = result.outside_phase_change
    wet = result.wet_finned_surface

    assert result.converged is True
    assert phase is not None and phase.converged is True
    assert phase.active is True
    assert phase.direction is PhaseChangeDirection.CONDENSATION
    assert phase.method == "outside_condensation_0d_wet_annular_fin_fvm"
    assert wet is not None
    assert wet.fin_wet_state is WetFinState.PARTIALLY_WET
    assert 0.0 < wet.fin_wet_fraction < 1.0
    assert wet.wet_dry_boundary_radius is not None

    # Every public route shares the exact converged object.  No post-hoc
    # radial diagnostic solve is permitted.
    assert phase.wet_finned_surface is wet
    assert result.final_result.wet_finned_surface is wet
    assert result.thermal_state.finned_tube_diagnostics.wet_surface is wet
    assert phase.wall_temperature_mean == pytest.approx(
        wet.outside_surface_temperature_area_mean
    )
    assert (
        phase.wall_temperature_min
        <= phase.wall_temperature_mean
        <= phase.wall_temperature_max
    )


def test_simulation_primary_fin_and_whole_side_balances_close(active_result) -> None:
    result = active_result
    phase = result.outside_phase_change
    wet = result.wet_finned_surface
    assert phase is not None and wet is not None

    assert wet.Q_primary_sensible > 0.0
    assert wet.Q_primary_latent > 0.0
    assert wet.Q_fin_sensible > 0.0
    assert wet.Q_fin_latent > 0.0
    assert wet.Q_total == pytest.approx(
        wet.Q_primary_total + wet.Q_fin_total,
        abs=1.0e-8,
    )
    assert wet.Q_total == pytest.approx(
        wet.Q_sensible + wet.Q_latent,
        abs=1.0e-8,
    )
    assert wet.m_dot_condensate == pytest.approx(
        wet.m_dot_condensate_primary + wet.m_dot_condensate_fin,
        abs=1.0e-14,
    )
    assert wet.wet_area == pytest.approx(
        wet.wet_primary_area + wet.wet_fin_area,
        abs=1.0e-12,
    )

    assert phase.Q_sensible == wet.Q_sensible
    assert phase.Q_latent == wet.Q_latent
    assert phase.Q_total == wet.Q_total
    assert phase.m_dot_condensate == wet.m_dot_condensate
    assert phase.wet_area == wet.wet_area
    assert phase.m_dot_water_vapor_in == pytest.approx(
        phase.m_dot_water_vapor_out + phase.m_dot_condensate,
        abs=1.0e-6,
    )
    assert abs(phase.mass_balance_error) < 1.0e-6
    assert abs(phase.energy_balance_error) < 1.0e-5


def test_simulation_keeps_physical_htc_and_labels_dry_dp_reference(
    active_result,
) -> None:
    result = active_result
    wet = result.wet_finned_surface
    diagnostics = result.finned_tube_diagnostics
    phase = result.outside_phase_change
    assert wet is not None and diagnostics is not None and phase is not None

    assert diagnostics.outside_alpha_physical == wet.outside_alpha_physical
    assert result.thermal_state.outside_alpha_physical == (
        wet.outside_alpha_physical
    )
    assert result.thermal_state.outside_alpha_effective_gross == (
        diagnostics.outside_alpha_effective_gross
    )
    assert (
        result.thermal_state.outside_alpha_wet_effective_gross_core_basis
        == wet.outside_alpha_wet_effective_gross_core_basis
    )
    assert result.thermal_state.outside_alpha_wet_effective_basis == (
        wet.outside_alpha_wet_effective_basis
    )

    assert math.isfinite(result.outside_dp_dry_reference)
    assert result.outside_dp_dry_reference > 0.0
    assert result.wet_pressure_drop_supported is False
    assert diagnostics.outside_dp_reference_only is True
    # Historical aliases retain the finite reference for compatibility; the
    # explicit flag and structured warning prohibit interpreting it as a wet
    # pressure-drop prediction.
    assert diagnostics.outside_dp_total == result.outside_dp_dry_reference
    assert CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY in {
        warning.code for warning in phase.warnings
    }
    assert CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY in {
        warning.code for warning in diagnostics.warnings
    }
    assert CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY in {
        warning.code for warning in result.warnings
    }


def test_disabled_mode_returns_the_exact_dry_simulation_with_sensitivity() -> None:
    hx = _wet_finned_hx()
    inside, outside = _side_inputs(mode=PhaseChangeMode.DISABLED)
    dry = run_simulation(hx, inside, outside)
    disabled = hx.simulate(inside, outside)

    assert disabled.phase_change_active is False
    assert disabled.wet_finned_surface is None
    assert disabled.q == dry.q
    assert disabled.T_out_inside == dry.T_out_inside
    assert disabled.T_out_outside == dry.T_out_outside
    assert disabled.UA == dry.UA
    assert disabled.finned_tube_diagnostics == dry.finned_tube_diagnostics
    assert PHASE_CHANGE_DISABLED_BUT_POSSIBLE in {
        warning.code for warning in disabled.outside_phase_change.warnings
    }


def test_endpoint_onset_uses_bounded_0d_wet_zone_when_mean_fin_is_dry() -> None:
    """Exercise the non-segmented fallback used by economizer-like pinches.

    Fixture is a synthetic finned economizer, unrelated to any specific
    project geometry, chosen empirically to trigger the same endpoint
    wet-zone-fallback code path as the case that originally motivated it.
    """

    core = BareTube(
        D_i=0.0189,
        D_o=0.0212,
        length_total=2.75,
        length_effective=2.72,
        wall_k=45.0,
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
    hx = BareTubeHeatExchanger(
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
    liquid_stub = ConstantPropertyProvider(
        FluidTransportProperties(
            rho=1_010.0,
            mu=1.6e-3,
            k=0.38,
            cp=3_550.0,
        )
    )
    wet_air = GasMixturePropertyProvider(
        gas_mixture_from_dry_composition_and_water_ratio(
            dry_components={"N2": 0.77, "O2": 0.21, "CO2": 0.02},
            dry_basis="mole",
            water_ratio=0.055,
            imposed_phase="gas",
        )
    )
    result = hx.simulate(
        HXSideInput(
            provider=liquid_stub,
            m_dot=31_500.0 / 3600.0,
            T_in=293.15,
            p=250_000.0,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        HXSideInput(
            provider=wet_air,
            m_dot=142_800.0 / 3600.0,
            T_in=335.15,
            p=P,
            phase_change_mode=PhaseChangeMode.AUTO,
        ),
    )

    phase = result.outside_phase_change
    wet = result.wet_finned_surface
    assert phase is not None and phase.active and phase.converged
    assert wet is not None and wet.m_dot_condensate > 0.0
    assert wet is phase.wet_finned_surface
    assert wet is result.final_result.wet_finned_surface
    assert wet is result.thermal_state.finned_tube_diagnostics.wet_surface
    assert wet.condensation_area_fraction < 1.0
    assert wet.condensation_temperature_offset_K < 0.0
    assert phase.method.endswith("with_endpoint_wet_zone_fallback")
    assert phase.residuals["outer_relaxation_factor"] == 0.25
    assert (
        "endpoint_envelope_wet_zone_0d_linear_weighting"
        in wet.assumptions
    )
    assert wet.Q_total == pytest.approx(
        wet.Q_primary_total + wet.Q_fin_total,
        abs=1.0e-7,
    )
    assert wet.m_dot_condensate == pytest.approx(
        wet.m_dot_condensate_primary + wet.m_dot_condensate_fin,
        abs=1.0e-12,
    )
    assert wet.wet_area == pytest.approx(
        wet.wet_primary_area + wet.wet_fin_area,
        abs=1.0e-10,
    )
    assert phase.Q_total == wet.Q_total
    assert phase.m_dot_condensate == wet.m_dot_condensate
    assert abs(phase.mass_balance_error) < 1.0e-6
    assert abs(phase.energy_balance_error) < 1.0e-6
