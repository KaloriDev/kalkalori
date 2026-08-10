# KalKalori â€” Heat Exchanger Open Engine
# GNU GPL v3 only
"""Rating integration for v0.6.1 inside wet-gas condensation."""

from __future__ import annotations

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.types import PhaseChangeDirection
from core.phase_change.wet_gas_enthalpy import h_wet_gas_dry_basis
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import water_saturation_liquid_enthalpy


def _wet_spec() -> GasMixtureSpec:
    y_h2o = 0.08
    remainder = 1.0 - y_h2o
    dry = {"N2": 0.70, "O2": 0.10, "CO2": 0.08}
    total = sum(dry.values())
    return GasMixtureSpec(
        components={name: value * remainder / total for name, value in dry.items()}
        | {"H2O": y_h2o},
        basis="mole",
    )


def _dry_spec() -> GasMixtureSpec:
    return GasMixtureSpec(components={"N2": 0.79, "O2": 0.21}, basis="mole")


@pytest.fixture(scope="module")
def rating_result():
    tube = BareTube(
        D_i=0.022,
        D_o=0.025,
        length_total=2.8,
        length_effective=2.8,
        wall_k=50.0,
    )
    hx = BareTubeHeatExchanger(
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
    return hx.rate(
        BalanceSideSpec(
            provider=GasMixturePropertyProvider(_wet_spec()),
            p=101_325.0,
            m_dot=1.0,
            T_in=360.0,
            T_out=320.34499765485987,
        ),
        BalanceSideSpec(
            provider=GasMixturePropertyProvider(_dry_spec()),
            p=101_325.0,
            m_dot=5.0,
            T_in=290.0,
            T_out=298.6985136488704,
        ),
    )


def test_rating_partial_inside_condensation(rating_result) -> None:
    pc = rating_result.inside_phase_change
    assert pc.direction is PhaseChangeDirection.CONDENSATION
    assert pc.active is True and pc.converged is True
    assert 0.0 < pc.wet_surface_fraction < 1.0
    assert pc.wall_temperature_min < pc.dew_point_in < pc.wall_temperature_mean
    assert pc.wall_temperature_wet_mean < pc.dew_point_in
    assert pc.W_sat_wet_surface < pc.W_mid
    assert pc.W_out < pc.W_in
    assert pc.m_dot_condensate > 0.0
    assert pc.Q_latent > 0.0
    assert pc.Q_total == pytest.approx(pc.Q_sensible + pc.Q_latent)
    assert pc.alfa_effective > pc.alfa_dry
    assert rating_result.thermal_state.alfa_i == pc.alfa_effective
    assert rating_result.UA_actual == rating_result.thermal_state.UA


def test_rating_inside_full_mass_and_enthalpy_balance(rating_result) -> None:
    pc = rating_result.inside_phase_change
    assert pc.m_dot_water_vapor_in == pytest.approx(
        pc.m_dot_water_vapor_out + pc.m_dot_condensate,
        abs=1e-12,
    )
    capability = detect_phase_change_capability(
        GasMixturePropertyProvider(_wet_spec())
    )
    h_in = h_wet_gas_dry_basis(360.0, 101_325.0, pc.W_in, capability)
    h_out = h_wet_gas_dry_basis(
        320.34499765485987,
        101_325.0,
        pc.W_out,
        capability,
    )
    h_liquid = water_saturation_liquid_enthalpy(
        T=pc.wall_temperature_wet_mean
    )
    Q_enthalpy = pc.m_dot_dry_carrier * (
        h_in - h_out - (pc.W_in - pc.W_out) * h_liquid
    )
    assert Q_enthalpy == pytest.approx(pc.Q_total, abs=1e-5)
    assert abs(pc.energy_balance_error) < 1e-5


def test_rating_inside_gas_phase_hydraulics(rating_result) -> None:
    pc = rating_result.inside_phase_change
    hydraulic = rating_result.tube_side_hydraulic.tube_bundle
    states = (hydraulic.inlet, hydraulic.midpoint, hydraulic.outlet)
    flows = tuple(
        state.mass_flux * hydraulic.flow_area_per_pass for state in states
    )
    assert flows == pytest.approx(
        (pc.m_dot_gas_in, pc.m_dot_dry_carrier * (1.0 + pc.W_mid), pc.m_dot_gas_out),
        rel=1e-12,
    )
    assert flows[2] < flows[1] < flows[0]
    assert hydraulic.dp_acceleration == pytest.approx(
        states[2].mass_flux**2 / states[2].rho
        - states[0].mass_flux**2 / states[0].rho
    )
