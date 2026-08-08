# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Synthetic, physically-sensible outside-water-condensation integration
test (spec section 37).

outside: wet process gas (N2, O2, CO2, H2O), high inlet temperature,
    crossflow over a bare tube bank, partial H2O condensation.
inside: dry air, inside the tubes, heated, no phase change.

Checks balances, monotonicity and physical bounds -- not an arbitrary
expected Q (there is no independent reference value to check against).

Run:
    pytest -q core/tests/outside_water_condensation_integration_test.py
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.properties.gas_mixture import GasMixtureSpec, GasMixturePropertyProvider
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.types import PhaseChangeDirection
from core.phase_change.water_equilibrium import (
    water_dew_point,
    water_mole_fraction_from_ratio,
    water_partial_pressure,
)
from core.phase_change.wet_gas_composition import (
    wet_gas_provider_at_water_ratio,
    wet_gas_spec_at_water_ratio,
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


@pytest.fixture(scope="module")
def result(hx: BareTubeHeatExchanger):
    wet_process_gas = GasMixtureSpec(
        components={"N2": 0.65, "O2": 0.10, "CO2": 0.08, "H2O": 0.17}, basis="mole"
    )
    dry_air = GasMixtureSpec(components={"N2": 0.79, "O2": 0.21}, basis="mole")

    outside = HXSideInput(
        provider=GasMixturePropertyProvider(wet_process_gas), m_dot=6.0, T_in=420.0, p=101_325.0,
    )
    inside = HXSideInput(
        provider=GasMixturePropertyProvider(dry_air), m_dot=15.0, T_in=290.0, p=101_325.0,
    )
    return hx.simulate(inside, outside)


def test_solver_converged(result) -> None:
    assert result.converged is True
    assert result.outside_phase_change.converged is True
    envelope = result.wall_temperature_envelope
    valid_probes = tuple(probe for probe in envelope.probes if probe.converged)
    assert envelope.outside_min == min(
        probe.outside_wall_temperature for probe in valid_probes
    )
    assert envelope.outside_max == max(
        probe.outside_wall_temperature for probe in valid_probes
    )


def test_duty_positive_and_temperatures_move_the_right_way(result) -> None:
    assert result.q > 0.0
    assert result.T_out_outside < 420.0  # outside (process gas) is cooled
    assert result.T_out_inside > 290.0   # inside (dry air) is heated


def test_condensation_is_active_and_partial(result) -> None:
    pc = result.outside_phase_change
    assert pc.active is True
    assert pc.direction is PhaseChangeDirection.CONDENSATION
    assert pc.W_out < pc.W_in
    assert pc.W_out > 0.0  # partial, not full condensation
    assert pc.m_dot_condensate > 0.0


def test_gas_phase_mass_flow_decreases_dry_carrier_conserved(result) -> None:
    pc = result.outside_phase_change
    assert pc.m_dot_gas_out < pc.m_dot_gas_in
    assert pc.m_dot_dry_carrier == pytest.approx(pc.m_dot_gas_in / (1.0 + pc.W_in), rel=1e-9)
    # m_dot_gas_out is dry carrier + remaining water vapor only (condensate
    # is never added back to the gas-phase stream).
    assert pc.m_dot_gas_out == pytest.approx(
        pc.m_dot_dry_carrier + pc.m_dot_dry_carrier * pc.W_out, rel=1e-9
    )
    # Cross-check against independently-converged solver state (W_out,
    # m_dot_condensate each converged to their own phase_change_* tolerance,
    # so this reproduces to solver-convergence precision, not machine eps).
    assert pc.m_dot_gas_out == pytest.approx(pc.m_dot_gas_in - pc.m_dot_condensate, rel=1e-6)


def test_public_fluid_property_points_use_final_wet_states(result) -> None:
    pc = result.outside_phase_change
    hydraulic = result.outside_tube_bank_hydraulic
    capability = detect_phase_change_capability(
        GasMixturePropertyProvider(
            GasMixtureSpec(
                components={"N2": 0.65, "O2": 0.10, "CO2": 0.08, "H2O": 0.17},
                basis="mole",
            )
        )
    )
    W_mid = 0.5 * (pc.W_in + pc.W_out)
    expected_providers = (
        wet_gas_provider_at_water_ratio(capability, pc.W_in),
        wet_gas_provider_at_water_ratio(capability, W_mid),
        wet_gas_provider_at_water_ratio(capability, pc.W_out),
    )

    states = (
        result.outside_properties_inlet,
        result.outside_properties_midpoint,
        result.outside_properties_outlet,
    )
    assert states == (hydraulic.inlet, hydraulic.midpoint, hydraulic.outlet)
    assert hydraulic.midpoint_method == "arithmetic_temperature_and_water_ratio"
    assert states[0].T == 420.0
    assert states[1].T == pytest.approx(0.5 * (420.0 + result.T_out_outside))
    assert states[2].T == result.T_out_outside
    assert states[1].props == result.outside_props_mean

    for index, (state, provider) in enumerate(zip(states, expected_providers)):
        expected = provider.at(T=state.T, p=state.p)
        for name in ("rho", "cp", "mu", "k"):
            # The midpoint deliberately preserves the wet solver's last
            # actually-used bulk props; its converged relaxed endpoint can
            # differ infinitesimally from a fresh post-solve evaluation.
            tolerance = 1e-6 if index == 1 else 1e-10
            assert getattr(state, name) == pytest.approx(
                getattr(expected, name), rel=tolerance
            )
        assert state.Pr == pytest.approx(state.mu * state.cp / state.k, rel=1e-12)

    inlet_composition = wet_gas_spec_at_water_ratio(capability, pc.W_in).to_mole_fractions()
    midpoint_composition = wet_gas_spec_at_water_ratio(capability, W_mid).to_mole_fractions()
    outlet_composition = wet_gas_spec_at_water_ratio(capability, pc.W_out).to_mole_fractions()
    water_fraction = lambda composition: next(
        value for name, value in composition.items() if name.lower() in {"h2o", "water"}
    )
    assert (
        water_fraction(outlet_composition)
        < water_fraction(midpoint_composition)
        < water_fraction(inlet_composition)
    )
    assert pc.dew_point_out < pc.dew_point_in

    point_mass_flows = tuple(state.face_mass_flux * hydraulic.face_area for state in states)
    assert point_mass_flows[0] == pytest.approx(pc.m_dot_gas_in, rel=1e-12)
    assert point_mass_flows[1] == pytest.approx(
        pc.m_dot_dry_carrier * (1.0 + W_mid), rel=1e-12
    )
    assert point_mass_flows[2] == pytest.approx(pc.m_dot_gas_out, rel=1e-12)

    inside_states = (
        result.inside_properties_inlet,
        result.inside_properties_midpoint,
        result.inside_properties_outlet,
    )
    assert inside_states[0].T == 290.0
    assert inside_states[2].T == result.T_out_inside
    for state in inside_states:
        assert all(
            math.isfinite(value)
            for value in (state.T, state.p, state.rho, state.cp, state.mu, state.k, state.Pr)
        )


def test_h2o_mass_balance_closed(result) -> None:
    pc = result.outside_phase_change
    assert pc.m_dot_water_vapor_in == pytest.approx(
        pc.m_dot_water_vapor_out + pc.m_dot_condensate, rel=1e-6
    )
    assert abs(pc.mass_balance_error) < 1e-6


def test_energy_balance_closed_and_split_positive(result) -> None:
    pc = result.outside_phase_change
    assert abs(pc.energy_balance_error) < 1e-6
    assert pc.Q_latent > 0.0
    assert pc.Q_sensible > 0.0
    assert pc.Q_total == pytest.approx(pc.Q_sensible + pc.Q_latent, rel=1e-9)


def test_surface_below_dew_point_for_part_of_the_envelope(result) -> None:
    pc = result.outside_phase_change
    assert pc.wall_temperature_min < pc.dew_point_in
    assert pc.wall_temperature_wet_mean is not None
    assert pc.wall_temperature_min <= pc.wall_temperature_wet_mean
    assert pc.wall_temperature_wet_mean <= pc.wall_temperature_max
    assert pc.wall_temperature_min <= pc.wall_temperature_mean <= pc.wall_temperature_max
    assert pc.wall_temperature_mean == pytest.approx(
        0.5 * (pc.wall_temperature_min + pc.wall_temperature_max),
        abs=0.05,
    )


def test_wet_zone_saturation_drives_positive_condensation(result) -> None:
    pc = result.outside_phase_change
    W_bulk = 0.5 * (pc.W_in + pc.W_out)
    capability = detect_phase_change_capability(
        GasMixturePropertyProvider(
            GasMixtureSpec(
                components={"N2": 0.65, "O2": 0.10, "CO2": 0.08, "H2O": 0.17},
                basis="mole",
            )
        )
    )
    y_h2o_bulk = water_mole_fraction_from_ratio(
        W_bulk,
        M_dry=capability.M_dry,
        M_h2o=capability.M_condensable,
    )
    dew_point_bulk = water_dew_point(
        water_partial_pressure(y_h2o_bulk, 101_325.0)
    )

    assert pc.wall_temperature_wet_mean is not None
    assert pc.W_sat_wet_surface is not None
    assert (
        pc.wall_temperature_min
        < dew_point_bulk
        < pc.wall_temperature_mean
        < pc.wall_temperature_max
    )
    assert pc.wall_temperature_wet_mean < dew_point_bulk
    assert pc.W_sat_wet_surface < W_bulk
    assert pc.m_dot_condensate > 0.0
    assert pc.Q_latent > 0.0
    assert pc.wet_area == pytest.approx(
        pc.outside_total_area * pc.wet_surface_fraction,
        rel=1e-9,
    )
    assert pc.alfa_effective > pc.alfa_dry


def test_wet_surface_fraction_bounded(result) -> None:
    pc = result.outside_phase_change
    assert 0.0 < pc.wet_surface_fraction <= 1.0


def test_no_nan_or_infinity_anywhere_in_the_phase_change_result(result) -> None:
    pc = result.outside_phase_change
    numeric_fields = (
        pc.W_in, pc.W_out, pc.m_dot_dry_carrier, pc.m_dot_water_vapor_in,
        pc.m_dot_water_vapor_out, pc.m_dot_condensate, pc.m_dot_gas_in,
        pc.m_dot_gas_out, pc.dew_point_in, pc.wall_temperature_mean,
        pc.wall_temperature_wet_mean, pc.W_sat_wet_surface,
        pc.wet_surface_fraction, pc.alfa_dry, pc.alfa_effective,
        pc.Q_sensible, pc.Q_latent, pc.Q_total,
        pc.mass_balance_error, pc.energy_balance_error,
    )
    for value in numeric_fields:
        assert value is not None
        assert math.isfinite(value)


def test_outside_gas_phase_hydraulics_are_finite_and_positive(result) -> None:
    hyd = result.outside_tube_bank_hydraulic
    assert math.isfinite(hyd.dp_drag) and hyd.dp_drag >= 0.0
    assert math.isfinite(hyd.dp_acceleration)
    assert math.isfinite(hyd.inlet.reynolds) and hyd.inlet.reynolds > 0.0
    assert math.isfinite(hyd.outlet.reynolds) and hyd.outlet.reynolds > 0.0
    # Gas-phase mass flow decreases from inlet to outlet (condensate removed,
    # never added back to the gas-phase velocity/flux basis).
    assert hyd.outlet.face_mass_flux < hyd.inlet.face_mass_flux
