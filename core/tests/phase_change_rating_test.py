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
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.integration import dew_point_at_ratio
from core.phase_change.types import PhaseChangeDirection, PhaseChangeMode
from core.phase_change.wet_gas_composition import (
    wet_gas_provider_at_water_ratio,
    wet_gas_spec_at_water_ratio,
)
from core.phase_change.wet_gas_enthalpy import h_wet_gas_dry_basis
from core.phase_change.water_equilibrium import saturated_water_ratio
from core.properties.water import (
    water_latent_heat_of_vaporization,
    water_saturation_liquid_enthalpy,
)


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
    assert pc.converged is True
    assert 2 <= pc.iterations <= 12
    assert pc.method == "outside_condensation_rating_wet_wall_fixed_point"
    assert math.isfinite(result.overdesign_factor)
    assert math.isfinite(result.UA_required)

    envelope = result.wall_temperature_envelope
    assert all(probe.converged for probe in envelope.probes)
    assert envelope.outside_min == min(
        probe.outside_wall_temperature for probe in envelope.probes
    )
    assert envelope.outside_max == max(
        probe.outside_wall_temperature for probe in envelope.probes
    )

    hydraulic = result.outside_tube_bank_hydraulic
    states = (
        result.outside_properties_inlet,
        result.outside_properties_midpoint,
        result.outside_properties_outlet,
    )
    assert states == (hydraulic.inlet, hydraulic.midpoint, hydraulic.outlet)
    assert hydraulic.midpoint_method == "arithmetic_temperature_and_water_ratio"
    assert states[0].T == outside.T_in
    assert states[1].T == pytest.approx(0.5 * (outside.T_in + outside.T_out))
    assert states[2].T == outside.T_out

    capability = detect_phase_change_capability(outside.provider)
    W_mid = 0.5 * (pc.W_in + pc.W_out)
    dew_point_mid = dew_point_at_ratio(capability, W_mid, p=outside.p)
    assert dew_point_mid is not None
    assert pc.wall_temperature_wet_mean is not None
    assert pc.W_sat_wet_surface is not None
    assert 0.0 < pc.wet_surface_fraction <= 1.0
    assert pc.wet_area == pytest.approx(
        pc.outside_total_area * pc.wet_surface_fraction, rel=1e-12
    )
    assert pc.wall_temperature_min <= pc.wall_temperature_mean <= pc.wall_temperature_max
    assert pc.wall_temperature_mean == pytest.approx(
        0.5 * (pc.wall_temperature_min + pc.wall_temperature_max),
        abs=1e-12,
    )
    assert pc.wall_temperature_min <= pc.wall_temperature_wet_mean
    assert pc.wall_temperature_wet_mean <= min(
        dew_point_mid, pc.wall_temperature_max
    )
    expected_wet_wall_temperature = 0.5 * (
        pc.wall_temperature_min + min(dew_point_mid, pc.wall_temperature_max)
    )
    assert pc.wall_temperature_wet_mean == pytest.approx(
        expected_wet_wall_temperature, abs=1e-8
    )
    assert pc.wall_temperature_wet_mean != pytest.approx(outside.T_out)

    expected_W_sat = saturated_water_ratio(
        p_total=outside.p,
        T=pc.wall_temperature_wet_mean,
        M_dry=capability.M_dry,
        M_h2o=capability.M_condensable,
    )
    assert pc.W_sat_wet_surface == pytest.approx(expected_W_sat, rel=1e-12)
    assert pc.W_sat_wet_surface < W_mid
    assert pc.Q_latent == pytest.approx(
        pc.m_dot_condensate
        * water_latent_heat_of_vaporization(T=pc.wall_temperature_wet_mean),
        rel=1e-12,
    )

    h_in = h_wet_gas_dry_basis(
        outside.T_in, outside.p, pc.W_in, capability
    )
    h_out = h_wet_gas_dry_basis(
        outside.T_out, outside.p, pc.W_out, capability
    )
    h_drained = (
        (pc.W_in - pc.W_out)
        * water_saturation_liquid_enthalpy(T=pc.wall_temperature_wet_mean)
    )
    Q_from_outside_enthalpy = pc.m_dot_dry_carrier * (
        h_in - h_out - h_drained
    )
    assert Q_from_outside_enthalpy == pytest.approx(pc.Q_total, abs=1e-5)
    assert pc.residuals["W_out"] < 1e-6
    assert pc.residuals["T_wall_wet_mean_K"] < 0.05
    assert pc.residuals["outside_enthalpy_balance_W"] < 1e-5

    for state, W in zip(states, (pc.W_in, W_mid, pc.W_out)):
        expected = wet_gas_provider_at_water_ratio(capability, W).at(T=state.T, p=state.p)
        for name in ("rho", "cp", "mu", "k"):
            assert getattr(state, name) == pytest.approx(getattr(expected, name), rel=1e-10)
        assert state.Pr == pytest.approx(state.mu * state.cp / state.k, rel=1e-12)

    compositions = tuple(
        wet_gas_spec_at_water_ratio(capability, W).to_mole_fractions()
        for W in (pc.W_in, W_mid, pc.W_out)
    )
    water_fraction = lambda composition: next(
        value for name, value in composition.items() if name.lower() in {"h2o", "water"}
    )
    assert (
        water_fraction(compositions[2])
        < water_fraction(compositions[1])
        < water_fraction(compositions[0])
    )
    assert pc.dew_point_out < pc.dew_point_in
    point_mass_flows = tuple(state.face_mass_flux * hydraulic.face_area for state in states)
    assert point_mass_flows[0] == pytest.approx(pc.m_dot_gas_in, rel=1e-12)
    assert point_mass_flows[1] == pytest.approx(
        pc.m_dot_dry_carrier * (1.0 + W_mid), rel=1e-12
    )
    assert point_mass_flows[2] == pytest.approx(pc.m_dot_gas_out, rel=1e-12)


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
    assert pc.W_out == pc.W_in
    assert pc.m_dot_gas_in == outside.m_dot
    assert pc.m_dot_gas_out == outside.m_dot
    assert pc.m_dot_water_vapor_out == pc.m_dot_water_vapor_in
    hydraulic = result.outside_tube_bank_hydraulic
    assert hydraulic.inlet.face_mass_flux == hydraulic.outlet.face_mass_flux
    expected_outlet = outside.provider.at(T=outside.T_out, p=outside.p)
    assert result.outside_properties_outlet.props == expected_outlet
    capability = detect_phase_change_capability(outside.provider)
    assert (
        wet_gas_spec_at_water_ratio(capability, pc.W_out).to_mole_fractions()
        == wet_gas_spec_at_water_ratio(capability, pc.W_in).to_mole_fractions()
    )
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
