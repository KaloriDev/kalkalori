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
    return _wet_finned_hx().simulate(inside, outside, surface_margin=0.10)


def test_active_wet_finned_simulation_reports_shared_surface_margin(
    active_result,
) -> None:
    result = active_result
    assert result.outside_phase_change.active is True
    assert result.surface_margin == 0.10
    assert result.UA_actual == result.UA
    assert result.UA_process == pytest.approx(abs(result.q) / result.EMTD)
    assert result.overdesign_factor == pytest.approx(0.10)
    assert result.overdesign_factor == pytest.approx(
        result.UA_actual / result.UA_process - 1.0
    )


def test_auto_simulation_exposes_native_axial_and_radial_states(active_result) -> None:
    result=active_result
    phase=result.outside_phase_change
    native=result.wet_coil_diagnostics
    assert result.converged and phase.active and phase.converged
    assert phase.direction is PhaseChangeDirection.CONDENSATION
    assert phase.method == "elmahdy_mitalas_energyplus_v25_2_adapted"
    assert native is phase.wet_coil_diagnostics
    assert native["surface_states"]
    assert 0 < phase.wet_fraction <= 1
    assert 0 < phase.wet_surface_fraction <= 1
    assert phase.wall_temperature_min <= phase.wall_temperature_mean <= phase.wall_temperature_max
    for point in native["surface_states"]:
        assert point["inside_wall_temperature"] <= point["core_wall_temperature"]
        assert point["surface_base_temperature"] <= point["fin_tip_temperature"]


def test_simulation_profile_mass_drain_and_whole_side_balances_close(active_result) -> None:
    from numpy.polynomial.legendre import leggauss
    result=active_result
    phase=result.outside_phase_change
    profile=result.wet_coil_diagnostics["process_profile"]
    weights=leggauss(len(profile)-1)[1]*phase.wet_fraction/2
    assert phase.Q_total == pytest.approx(phase.Q_sensible+phase.Q_latent,abs=1e-8)
    assert phase.H_drain == pytest.approx(
        sum(w*p.drain_density for w,p in zip(weights,profile[1:])),abs=1e-8)
    assert phase.m_dot_water_vapor_in == pytest.approx(
        phase.m_dot_water_vapor_out+phase.m_dot_condensate,abs=1e-6)
    assert abs(phase.mass_balance_error) < 1e-6
    assert abs(phase.energy_balance_error) < 1e-5
    radial=result.wet_coil_diagnostics["surface_states"]
    assert phase.wet_area == pytest.approx(
        sum(w*p["radial_wet_area"] for w,p in zip(weights,radial)),abs=1e-12)


def test_simulation_keeps_physical_htc_and_labels_dry_dp_reference(active_result) -> None:
    result=active_result
    diagnostics=result.finned_tube_diagnostics
    native=result.wet_coil_diagnostics
    assert diagnostics.outside_alpha_physical == pytest.approx(native["outside_alpha_physical"])
    assert result.thermal_state.outside_alpha_physical == native["outside_alpha_physical"]
    assert result.thermal_state.outside_alpha_effective_gross == pytest.approx(
        diagnostics.outside_alpha_effective_gross)
    assert result.ua_is_equivalent
    assert math.isfinite(result.outside_dp_dry_reference) and result.outside_dp_dry_reference > 0
    assert result.wet_pressure_drop_supported is False
    assert diagnostics.outside_dp_reference_only is True
    assert diagnostics.outside_dp_total == result.outside_dp_dry_reference
    for warnings in (result.outside_phase_change.warnings,diagnostics.warnings,result.warnings):
        assert CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY in {w.code for w in warnings}


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


def test_endpoint_onset_uses_source_profile_with_native_radial_states() -> None:
    """Economizer-like endpoint pinch uses the native axial/radial profile."""

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

    phase=result.outside_phase_change
    assert phase is not None and phase.active and phase.converged
    assert phase.m_dot_condensate > 0
    assert phase.method == "elmahdy_mitalas_energyplus_v25_2_adapted"
    assert 0 < phase.wet_fraction <= 1
    assert phase.Q_total == pytest.approx(phase.Q_sensible+phase.Q_latent,abs=1e-7)
    assert phase.wet_coil_diagnostics["surface_states"]
    for p in phase.wet_coil_diagnostics["process_profile"]:
        assert p.surface_temperature >= p.liquid_temperature
    assert abs(phase.energy_balance_error) < 1e-5
