# KalKalori â€” Heat Exchanger Open Engine
# GNU GPL v3 only
"""Integration coverage for v0.6.1 wet-gas H2O condensation inside tubes."""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.integration import PureWaterSteamCondensationNotSupportedError
from core.phase_change.types import PhaseChangeDirection, PhaseChangeMode
from core.phase_change.warning_codes import (
    CONDENSATE_FILM_HYDRAULICS_NOT_MODELLED,
    PHASE_CHANGE_DISABLED_BUT_POSSIBLE,
)
from core.phase_change.wet_gas_composition import (
    wet_gas_provider_at_water_ratio,
    wet_gas_spec_at_water_ratio,
)
from core.phase_change.wet_gas_enthalpy import h_wet_gas_dry_basis
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import IAPWS97WaterSteamProvider, water_saturation_liquid_enthalpy


def _hx() -> BareTubeHeatExchanger:
    tube = BareTube(
        D_i=0.022,
        D_o=0.025,
        length_total=2.8,
        length_effective=2.8,
        wall_k=50.0,
    )
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=tube,
            n_rows=5,
            n_tubes_per_row=10,
            pitch_transverse=0.035,
            pitch_longitudinal=0.035,
            layout="staggered",
            n_passes_tube=2,
            flow_arrangement="counterflow",
        )
    )


def _dry_spec() -> GasMixtureSpec:
    return GasMixtureSpec(components={"N2": 0.79, "O2": 0.21}, basis="mole")


def _wet_spec(y_h2o: float = 0.08) -> GasMixtureSpec:
    remainder = 1.0 - y_h2o
    dry = {"N2": 0.70, "O2": 0.10, "CO2": 0.08}
    total = sum(dry.values())
    return GasMixtureSpec(
        components={
            name: value * remainder / total for name, value in dry.items()
        }
        | {"H2O": y_h2o},
        basis="mole",
    )


@pytest.fixture(scope="module")
def partial_result():
    return _hx().simulate(
        HXSideInput(
            provider=GasMixturePropertyProvider(_wet_spec()),
            m_dot=1.0,
            T_in=360.0,
            p=101_325.0,
        ),
        HXSideInput(
            provider=GasMixturePropertyProvider(_dry_spec()),
            m_dot=5.0,
            T_in=290.0,
            p=101_325.0,
        ),
    )


def test_no_h2o_inside_is_not_capable() -> None:
    capability = detect_phase_change_capability(
        GasMixturePropertyProvider(_dry_spec())
    )
    assert capability.capable is False


def test_wet_gas_with_dry_inside_wall_is_not_active() -> None:
    hx = _hx()
    result = hx.simulate(
        HXSideInput(
            provider=GasMixturePropertyProvider(_wet_spec()),
            m_dot=1.0,
            T_in=360.0,
            p=101_325.0,
        ),
        HXSideInput(
            provider=GasMixturePropertyProvider(_dry_spec()),
            m_dot=5.0,
            T_in=350.0,
            p=101_325.0,
        ),
    )
    pc = result.inside_phase_change
    assert pc.capable is True
    assert pc.possible is False
    assert pc.active is False
    assert pc.m_dot_condensate == 0.0


def test_partial_inside_condensation_and_key_local_onset(partial_result) -> None:
    pc = partial_result.inside_phase_change
    assert pc.direction is PhaseChangeDirection.CONDENSATION
    assert pc.possible is True and pc.active is True and pc.converged is True
    assert 0.0 < pc.wet_surface_fraction < 1.0
    assert pc.wall_temperature_min < pc.dew_point_in < pc.wall_temperature_mean
    assert pc.wall_temperature_wet_mean < pc.dew_point_in
    assert pc.W_sat_wet_surface < pc.W_mid
    assert pc.wet_area == pytest.approx(
        pc.inside_total_area * pc.wet_surface_fraction,
        rel=1e-12,
    )
    assert pc.m_dot_condensate > 0.0
    assert pc.W_out < pc.W_in
    assert pc.m_dot_gas_out < pc.m_dot_gas_in
    assert pc.Q_sensible > 0.0
    assert pc.Q_latent > 0.0
    assert pc.Q_total == pytest.approx(pc.Q_sensible + pc.Q_latent, rel=1e-12)
    assert pc.alfa_effective > pc.alfa_dry


def test_inside_water_and_full_enthalpy_balances(partial_result) -> None:
    pc = partial_result.inside_phase_change
    assert pc.m_dot_water_vapor_in == pytest.approx(
        pc.m_dot_water_vapor_out + pc.m_dot_condensate,
        abs=1e-8,
    )
    assert pc.m_dot_gas_out == pytest.approx(
        pc.m_dot_dry_carrier + pc.m_dot_water_vapor_out,
        rel=1e-12,
    )
    capability = detect_phase_change_capability(
        GasMixturePropertyProvider(_wet_spec())
    )
    h_in = h_wet_gas_dry_basis(360.0, 101_325.0, pc.W_in, capability)
    h_out = h_wet_gas_dry_basis(
        partial_result.T_out_inside,
        101_325.0,
        pc.W_out,
        capability,
    )
    h_condensate = water_saturation_liquid_enthalpy(
        T=pc.wall_temperature_wet_mean
    )
    Q_enthalpy = pc.m_dot_dry_carrier * (
        h_in - h_out - (pc.W_in - pc.W_out) * h_condensate
    )
    assert Q_enthalpy == pytest.approx(pc.Q_total, rel=5e-4)
    assert pc.energy_balance_error == pytest.approx(pc.Q_total - Q_enthalpy)


def test_inside_endpoint_properties_and_composition(partial_result) -> None:
    pc = partial_result.inside_phase_change
    capability = detect_phase_change_capability(
        GasMixturePropertyProvider(_wet_spec())
    )
    states = (
        partial_result.inside_properties_inlet,
        partial_result.inside_properties_midpoint,
        partial_result.inside_properties_outlet,
    )
    assert tuple(state.T for state in states) == pytest.approx(
        (360.0, 0.5 * (360.0 + partial_result.T_out_inside), partial_result.T_out_inside)
    )
    for index, (state, W) in enumerate(zip(states, (pc.W_in, pc.W_mid, pc.W_out))):
        expected = wet_gas_provider_at_water_ratio(capability, W).at(
            T=state.T,
            p=state.p,
        )
        tolerance = 1e-5 if index == 1 else 1e-10
        for name in ("rho", "cp", "mu", "k"):
            assert getattr(state, name) == pytest.approx(
                getattr(expected, name),
                rel=tolerance,
            )
        assert state.Pr == pytest.approx(state.mu * state.cp / state.k)

    compositions = tuple(
        wet_gas_spec_at_water_ratio(capability, W).to_mole_fractions()
        for W in (pc.W_in, pc.W_mid, pc.W_out)
    )
    water = lambda composition: composition["Water"]
    assert water(compositions[2]) < water(compositions[1]) < water(compositions[0])
    dry_ratios = tuple(
        composition["Nitrogen"] / composition["Oxygen"]
        for composition in compositions
    )
    assert dry_ratios[0] == pytest.approx(dry_ratios[1])
    assert dry_ratios[1] == pytest.approx(dry_ratios[2])
    assert pc.dew_point_out < pc.dew_point_in


def test_inside_hydraulics_use_only_remaining_gas(partial_result) -> None:
    pc = partial_result.inside_phase_change
    hydraulic = partial_result.tube_side_hydraulic.tube_bundle
    states = (hydraulic.inlet, hydraulic.midpoint, hydraulic.outlet)
    flows = tuple(
        state.mass_flux * hydraulic.flow_area_per_pass for state in states
    )
    assert flows == pytest.approx(
        (pc.m_dot_gas_in, pc.m_dot_dry_carrier * (1.0 + pc.W_mid), pc.m_dot_gas_out),
        rel=1e-12,
    )
    assert flows[2] == pytest.approx(flows[0] - pc.m_dot_condensate, rel=1e-7)
    assert states[2].mass_flux < states[1].mass_flux < states[0].mass_flux
    for state in states:
        assert state.reynolds == pytest.approx(
            state.mass_flux * hydraulic.hydraulic_diameter / state.mu
        )
    expected_acceleration = (
        states[2].mass_flux**2 / states[2].rho
        - states[0].mass_flux**2 / states[0].rho
    )
    assert hydraulic.dp_acceleration == pytest.approx(expected_acceleration)
    assert math.isfinite(hydraulic.dp_friction) and hydraulic.dp_friction > 0.0
    boundary_flows = tuple(
        state.mass_flux * state.flow_area_per_pass
        for state in hydraulic.pass_boundary_states
    )
    assert boundary_flows == pytest.approx(flows, rel=1e-12)
    assert hydraulic.dp_tube_entrances > 0.0
    assert hydraulic.dp_tube_exits > 0.0
    assert hydraulic.entrance_results[0].reference_dynamic_pressure == pytest.approx(
        hydraulic.pass_boundary_states[0].dynamic_pressure
    )
    assert hydraulic.exit_results[-1].reference_dynamic_pressure == pytest.approx(
        hydraulic.pass_boundary_states[-1].dynamic_pressure
    )
    assert any(
        warning.code == CONDENSATE_FILM_HYDRAULICS_NOT_MODELLED
        for warning in pc.warnings
    )


def test_inside_disabled_returns_sensible_only_with_warning() -> None:
    result = _hx().simulate(
        HXSideInput(
            provider=GasMixturePropertyProvider(_wet_spec()),
            m_dot=1.0,
            T_in=360.0,
            p=101_325.0,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        HXSideInput(
            provider=GasMixturePropertyProvider(_dry_spec()),
            m_dot=5.0,
            T_in=290.0,
            p=101_325.0,
        ),
    )
    pc = result.inside_phase_change
    assert pc.possible is True and pc.active is False
    assert pc.W_out == pc.W_in
    assert pc.m_dot_condensate == 0.0
    assert pc.Q_latent == 0.0
    assert any(w.code == PHASE_CHANGE_DISABLED_BUT_POSSIBLE for w in pc.warnings)


def test_pure_steam_inside_is_controlled_v062_unsupported() -> None:
    with pytest.raises(PureWaterSteamCondensationNotSupportedError, match="v0.6.2"):
        _hx().simulate(
            HXSideInput(
                provider=IAPWS97WaterSteamProvider(),
                m_dot=1.0,
                T_in=400.0,
                p=101_325.0,
            ),
            HXSideInput(
                provider=GasMixturePropertyProvider(_dry_spec()),
                m_dot=5.0,
                T_in=290.0,
                p=101_325.0,
            ),
        )
