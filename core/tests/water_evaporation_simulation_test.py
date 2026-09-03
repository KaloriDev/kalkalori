import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput, run_simulation
from core.phase_change import warning_codes as WC
from core.phase_change.capability import PureWaterPhaseChangeProviderNotSupportedError
from core.phase_change.integration import MultiplePhaseChangeSidesError
from core.phase_change.steam_integration import (
    PhaseChangeDisabledButRequiredError,
    PureSteamOutsideNotSupportedError,
)
from core.phase_change.types import (
    PhaseChangeDirection,
    PhaseChangeMode,
    WaterSteamPhaseChangeResult,
)
from core.properties.common import FluidTransportProperties
from core.properties.coolprop_backend import CoolPropFluidProvider
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import IAPWS97WaterSteamProvider, WaterSteamPhase


P = 1.0e6
HOT = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1005.0)
)


def _hx(rows=10, columns=10, *, orientation=TubeOrientation.VERTICAL_UPWARD):
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.020, D_o=0.024, length_total=4.0,
                length_effective=4.0, wall_k=16.0,
                tube_orientation=orientation,
            ),
            n_rows=rows, n_tubes_per_row=columns,
            pitch_transverse=0.04, pitch_longitudinal=0.04,
            layout="inline", n_passes_tube=1, flow_arrangement="crossflow",
        )
    )


def _inside(*, m_dot=1.0, mode=PhaseChangeMode.AUTO, **state):
    return HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=m_dot, p=P,
        phase_change_mode=mode, **state,
    )


def _outside(*, provider=HOT, m_dot=30.0, T_in=700.0, p=101325.0):
    return HXSideInput(provider=provider, m_dot=m_dot, T_in=T_in, p=p)


def test_preheat_only_preserves_the_existing_sensible_result_and_hydraulics(monkeypatch):
    hx = _hx(rows=10, columns=10, orientation=None)
    inside = _inside(T_in=300.0, m_dot=5.0)
    outside = _outside(T_in=500.0, m_dot=20.0)
    expected = run_simulation(hx, inside, outside)

    def boiling_must_not_run(**kwargs):
        raise AssertionError("boiling transport was called for preheat-only Simulation")

    monkeypatch.setattr(
        "core.phase_change.water_evaporation.shah1982_boiling_alpha_local",
        boiling_must_not_run,
    )
    actual = hx.simulate(inside, outside)
    assert actual.q == expected.q
    assert actual.UA == expected.UA
    assert actual.T_out_inside == expected.T_out_inside
    assert actual.inside_dp_total == expected.inside_dp_total
    assert actual.tube_side_hydraulic == expected.tube_side_hydraulic
    assert actual.inside_phase_change.capable is True
    assert actual.inside_phase_change.possible is False
    assert actual.inside_phase_change.active is False


@pytest.mark.parametrize(
    ("inside", "rows", "expected_phase"),
    [
        (_inside(T_in=350.0), 10, WaterSteamPhase.TWO_PHASE),
        (_inside(quality_in=0.0), 5, WaterSteamPhase.TWO_PHASE),
        (_inside(quality_in=0.2), 5, WaterSteamPhase.TWO_PHASE),
        (_inside(quality_in=0.9), 2, WaterSteamPhase.SUPERHEATED_VAPOR),
        (_inside(h_in=_inside(quality_in=0.2).h_in), 5, WaterSteamPhase.TWO_PHASE),
    ],
)
def test_simulation_routes_supported_state_inputs_and_transitions(
    inside, rows, expected_phase
):
    result = _hx(rows=rows).simulate(inside, _outside())
    water = result.inside_phase_change
    assert isinstance(water, WaterSteamPhaseChangeResult)
    assert water.direction is PhaseChangeDirection.EVAPORATION
    assert water.capable is True and water.possible is True and water.active is True
    assert water.phase_out is expected_phase
    assert water.h_out == pytest.approx(water.h_in + result.q / inside.m_dot)
    assert result.T_out_inside == water.state_out.T
    assert result.inside_properties_inlet is water.state_in
    assert result.inside_properties_midpoint is water.state_midpoint
    assert result.inside_properties_outlet is water.state_out
    assert water.Q_total == pytest.approx(
        water.Q_preheat + water.Q_evaporation + water.Q_superheat
    )
    assert water.A_total == pytest.approx(_hx(rows=rows).bundle.total_outer_area, rel=2e-8)
    assert water.UA_total == pytest.approx(
        water.zone_UA_preheat
        + water.zone_UA_evaporation
        + water.zone_UA_superheat
    )


def test_complete_evaporation_continues_into_superheat_without_losing_duty():
    result = _hx(rows=25).simulate(_inside(T_in=350.0), _outside())
    water = result.inside_phase_change
    assert water.phase_out is WaterSteamPhase.SUPERHEATED_VAPOR
    assert water.Q_preheat > 0.0
    assert water.Q_evaporation == pytest.approx(
        water.mass_flow_total
        * (water_steam_saturated_vapor_h() - water_steam_saturated_liquid_h())
    )
    assert water.Q_superheat > 0.0
    assert water.A_preheat > 0.0
    assert water.A_evaporation > 0.0
    assert water.A_superheat > 0.0
    assert water.m_dot_evaporated == pytest.approx(water.mass_flow_total)


def water_steam_saturated_liquid_h():
    return IAPWS97WaterSteamProvider().state(x=0.0, p=P).h


def water_steam_saturated_vapor_h():
    return IAPWS97WaterSteamProvider().state(x=1.0, p=P).h


def test_surface_margin_changes_duty_quality_and_zone_allocation():
    hx = _hx(rows=10)
    baseline = hx.simulate(_inside(T_in=350.0), _outside())
    derated = hx.simulate(_inside(T_in=350.0), _outside(), surface_margin=0.10)
    base_water = baseline.inside_phase_change
    water = derated.inside_phase_change
    assert derated.q < baseline.q
    assert derated.Q_full == pytest.approx(baseline.q)
    assert derated.Q_derated == derated.q
    assert water.quality_out < base_water.quality_out
    assert water.A_total == pytest.approx(hx.bundle.total_outer_area / 1.10, rel=2e-8)
    assert water.A_evaporation < base_water.A_evaporation
    assert derated.UA_actual == derated.UA
    assert derated.UA_process == pytest.approx(abs(derated.q) / derated.EMTD)
    assert derated.overdesign_factor == pytest.approx(0.10)
    assert derated.overdesign_factor == pytest.approx(
        derated.UA_actual / derated.UA_process - 1.0
    )


def test_capable_possible_active_are_distinct_after_strong_surface_derating():
    result = _hx(rows=10).simulate(
        _inside(T_in=350.0), _outside(), surface_margin=10.0
    )
    water = result.inside_phase_change
    assert isinstance(water, WaterSteamPhaseChangeResult)
    assert water.capable is True
    assert water.possible is True
    assert water.active is False
    assert water.direction is PhaseChangeDirection.NONE
    assert water.phase_out is WaterSteamPhase.SUBCOOLED_LIQUID
    assert water.Q_evaporation == 0.0
    assert math.isfinite(result.inside_dp_total)


def test_disabled_mode_rejects_required_boiling_but_allows_derated_preheat():
    hx = _hx(rows=10)
    with pytest.raises(PhaseChangeDisabledButRequiredError) as caught:
        hx.simulate(_inside(T_in=350.0, mode=PhaseChangeMode.DISABLED), _outside())
    assert caught.value.warning_code == WC.PHASE_CHANGE_DISABLED_BUT_REQUIRED

    result = hx.simulate(
        _inside(T_in=350.0, mode=PhaseChangeMode.DISABLED),
        _outside(),
        surface_margin=10.0,
    )
    water = result.inside_phase_change
    assert water.active is False and water.possible is True
    assert WC.PHASE_CHANGE_DISABLED_BUT_POSSIBLE in {
        warning.code for warning in water.warnings
    }


def test_active_evaporation_invalidates_complete_tube_side_pressure_drop():
    result = _hx(rows=10).simulate(_inside(T_in=350.0), _outside())
    water = result.inside_phase_change
    assert result.tube_side_hydraulic is None
    assert result.tube_side_pressure_drop is None
    assert math.isnan(result.inside_dp_total)
    assert math.isnan(result.inside_dp_friction)
    assert water.two_phase_pressure_drop_supported is False
    assert water.two_phase_pressure_drop_status == "not_supported"
    assert WC.WATER_EVAPORATION_TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED in {
        warning.code for warning in water.warnings
    }


def test_public_equivalent_alpha_zone_balances_and_wall_diagnostics_are_consistent():
    hx = _hx(rows=25)
    result = hx.simulate(_inside(T_in=350.0), _outside())
    water = result.inside_phase_change
    tube = hx.bundle.tube
    reconstructed_u = 1.0 / (
        tube.D_o / (tube.D_i * water.inside_alpha_equivalent)
        + tube.D_o * math.log(tube.D_o / tube.D_i) / (2.0 * tube.wall_k)
        + 1.0 / result.outside_alfa_mean
    )
    assert reconstructed_u == pytest.approx(result.U_mean, rel=2e-12)
    assert result.UA == pytest.approx(water.UA_total)
    assert water.UA_total == pytest.approx(water.A_total * result.U_mean)
    assert result.inside_alfa_mean == water.inside_alpha_equivalent
    assert result.thermal_state.alfa_i == water.inside_alpha_equivalent
    assert result.final_result.tube_side_thermal.alfa == water.inside_alpha_equivalent
    assert result.final_result.T_hot_out == result.T_out_outside
    assert result.final_result.T_cold_out == result.T_out_inside
    assert result.inside_wall_temperature_mean > result.T_mean_inside
    assert result.outside_wall_temperature_mean < result.T_mean_outside
    assert not hasattr(water, "solution")
    assert not hasattr(water, "runtime_s")


class _FastWetGasProvider(GasMixturePropertyProvider):
    def at(self, T, p):
        return FluidTransportProperties(
            rho=12.0 * 500.0 / T,
            mu=2.0e-5,
            k=0.04,
            cp=1800.0,
        )


def test_two_active_phase_changing_sides_are_rejected():
    wet = _FastWetGasProvider(
        GasMixtureSpec(
            components={"N2": 0.49, "O2": 0.21, "H2O": 0.30}, basis="mole"
        )
    )
    outside = _outside(provider=wet, m_dot=10.0, T_in=500.0, p=5.0e6)
    with pytest.raises(MultiplePhaseChangeSidesError):
        _hx(rows=20).simulate(_inside(quality_in=0.0), outside)


def test_outside_pure_water_evaporation_is_controlled_unsupported():
    outside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0, T_in=350.0, p=P
    )
    hot_inside = HXSideInput(provider=HOT, m_dot=30.0, T_in=700.0, p=101325.0)
    with pytest.raises(PureSteamOutsideNotSupportedError) as caught:
        _hx(rows=10).simulate(hot_inside, outside)
    assert caught.value.warning_code == WC.PURE_STEAM_OUTSIDE_NOT_SUPPORTED
    assert "outside tubes is not supported" in str(caught.value)


def test_outside_subcooled_water_remaining_single_phase_is_unchanged():
    outside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=20.0, T_in=350.0, p=P
    )
    warm_inside = HXSideInput(provider=HOT, m_dot=1.0, T_in=400.0, p=101325.0)
    hx = _hx(rows=2, columns=2)
    expected = run_simulation(hx, warm_inside, outside)
    actual = hx.simulate(warm_inside, outside)
    assert actual.q == expected.q
    assert actual.T_out_outside == expected.T_out_outside
    assert actual.outside_dp_total == expected.outside_dp_total


def test_coolprop_water_crossing_is_not_silently_replaced_by_iapws():
    inside = HXSideInput(
        provider=CoolPropFluidProvider("Water"), m_dot=1.0, T_in=350.0, p=P
    )
    with pytest.raises(PureWaterPhaseChangeProviderNotSupportedError) as caught:
        _hx(rows=20).simulate(inside, _outside())
    assert caught.value.warning_code == WC.PURE_WATER_PHASE_CHANGE_PROVIDER_NOT_SUPPORTED
    assert "not silently replaced" in str(caught.value)


class _TemperatureDependentHotProvider:
    def at(self, T, p):
        return FluidTransportProperties(
            rho=1.2 * 500.0 / T,
            mu=1.4e-5 + 1.0e-8 * T,
            k=0.018 + 2.0e-5 * T,
            cp=900.0 + 0.25 * T,
        )


def test_public_result_uses_outside_properties_from_final_trial_state():
    provider = _TemperatureDependentHotProvider()
    outside = _outside(provider=provider)
    result = _hx(rows=10).simulate(_inside(T_in=350.0), outside)
    T_mean = 0.5 * (outside.T_in + result.T_out_outside)
    assert result.outside_props_mean == provider.at(T_mean, outside.p)
    assert result.outside_alfa_mean == pytest.approx(
        result.final_result.outside_side_thermal.alfa
    )
