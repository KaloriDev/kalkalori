import ast
import inspect
import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput, run_simulation
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change import warning_codes as WC
from core.phase_change.integration import MultiplePhaseChangeSidesError
from core.phase_change.steam_integration import (
    PhaseChangeDisabledButRequiredError,
    PureSteamOutsideNotSupportedError,
)
import core.phase_change.steam_integration as steam_integration
from core.phase_change.types import PhaseChangeMode, WaterSteamPhaseChangeResult
from core.phase_change.steam_heater import SteamEvaporationNotSupportedError
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import IAPWS97WaterSteamProvider, WaterSteamPhase


P = 1.0e6
ORIENTATION = TubeOrientation.VERTICAL_DOWNWARD
OUTSIDE_PROVIDER = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1005.0)
)


class TemperatureDependentOutsideProvider:
    def at(self, T, p):
        return FluidTransportProperties(
            rho=1.2 * 300.0 / T,
            mu=1.8e-5 * (T / 300.0) ** 0.7,
            k=0.026 * (T / 300.0) ** 0.8,
            cp=1005.0 + 0.25 * (T - 300.0),
        )


def _hx(n_rows=10, n_tubes_per_row=10, *, orientation=ORIENTATION):
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.020, D_o=0.024, length_total=4.0,
                length_effective=4.0, wall_k=16.0,
                tube_orientation=orientation,
            ),
            n_rows=n_rows, n_tubes_per_row=n_tubes_per_row,
            pitch_transverse=0.04, pitch_longitudinal=0.04,
            layout="inline", n_passes_tube=1, flow_arrangement="crossflow",
        )
    )


def _outside_sim(m_dot=30.0):
    return HXSideInput(
        provider=OUTSIDE_PROVIDER, m_dot=m_dot,
        T_in=300.0, p=101325.0,
    )


def _inside_sim(*, m_dot=1.0, **state):
    return HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=m_dot, p=P,
        **state,
    )


@pytest.mark.parametrize(
    ("inside", "expected_phase"),
    [
        (_inside_sim(quality_in=1.0), WaterSteamPhase.TWO_PHASE),
        (_inside_sim(quality_in=0.8), WaterSteamPhase.TWO_PHASE),
        (_inside_sim(T_in=520.0), WaterSteamPhase.TWO_PHASE),
        (_inside_sim(quality_in=1.0, m_dot=0.3), WaterSteamPhase.SUBCOOLED_LIQUID),
        (_inside_sim(T_in=520.0, m_dot=0.25), WaterSteamPhase.SUBCOOLED_LIQUID),
    ],
)
def test_simulation_public_transitions_use_typed_steam_result(inside, expected_phase):
    result = _hx().simulate(inside, _outside_sim())
    steam = result.inside_phase_change
    assert isinstance(steam, WaterSteamPhaseChangeResult)
    assert steam.phase_out is expected_phase
    assert steam.h_out == pytest.approx(inside.h_in - result.q / inside.m_dot)
    assert steam.Q_total == pytest.approx(
        steam.Q_desuperheat + steam.Q_condensation + steam.Q_subcooling
    )
    assert steam.A_total == pytest.approx(_hx().bundle.total_outer_area, rel=2e-8)
    assert steam.UA_total == pytest.approx(
        steam.zone_UA_desuperheat + steam.zone_UA_condensation + steam.zone_UA_subcooling
    )
    assert result.UA == pytest.approx(steam.UA_total)
    assert result.inside_properties_inlet is steam.state_in
    assert result.inside_properties_outlet is steam.state_out
    assert result.inside_properties_outlet.quality == steam.quality_out
    assert steam.p == P
    assert result.tube_side_hydraulic is None
    assert result.tube_side_pressure_drop is None
    assert math.isnan(result.inside_dp_total)
    assert result.outside_tube_bank_hydraulic is not None
    assert math.isfinite(result.outside_dp_total)
    assert steam.two_phase_pressure_drop_status == "not_supported"
    assert WC.STEAM_TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED in {
        warning.code for warning in steam.warnings
    }


def test_superheated_steam_remaining_superheated_is_detected_without_condensation():
    result = _hx(n_rows=2, n_tubes_per_row=2, orientation=None).simulate(
        _inside_sim(T_in=600.0, m_dot=10.0), _outside_sim()
    )
    steam = result.inside_phase_change
    assert steam.phase_out is WaterSteamPhase.SUPERHEATED_VAPOR
    assert steam.active is False
    assert steam.Q_condensation == 0.0


def test_superheated_to_superheated_rating_does_not_require_orientation():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        T_in=600.0, T_out=550.0,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=30.0, T_in=300.0,
    )
    result = _hx(orientation=None).rate(inside, outside)
    assert result.inside_phase_change.active is False
    assert result.inside_phase_change.Q_condensation == 0.0


def test_disabled_mode_allows_a_same_phase_superheated_result():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=10.0, p=P,
        T_in=600.0,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )
    result = _hx(n_rows=2, n_tubes_per_row=2, orientation=None).simulate(
        inside, _outside_sim()
    )
    assert result.inside_phase_change.phase_out is WaterSteamPhase.SUPERHEATED_VAPOR
    assert result.inside_phase_change.active is False


def test_surface_margin_derates_public_steam_simulation():
    hx = _hx()
    baseline = hx.simulate(_inside_sim(quality_in=1.0), _outside_sim())
    derated = hx.simulate(
        _inside_sim(quality_in=1.0), _outside_sim(), surface_margin=0.25
    )
    assert derated.surface_margin == 0.25
    assert derated.q < baseline.q
    assert derated.Q_full == pytest.approx(baseline.q)
    assert derated.Q_derated == derated.q
    assert derated.inside_phase_change.A_total == pytest.approx(
        hx.bundle.total_outer_area / 1.25, rel=2e-8
    )


def test_outside_properties_and_alpha_follow_final_trial_temperature():
    provider = TemperatureDependentOutsideProvider()
    cold = HXSideInput(provider=provider, m_dot=30.0, T_in=280.0, p=101325.0)
    warm = HXSideInput(provider=provider, m_dot=30.0, T_in=340.0, p=101325.0)
    cold_result = _hx().simulate(_inside_sim(quality_in=1.0), cold)
    warm_result = _hx().simulate(_inside_sim(quality_in=1.0), warm)
    for result, outside in ((cold_result, cold), (warm_result, warm)):
        T_mean = 0.5 * (outside.T_in + result.T_out_outside)
        assert result.outside_props_mean == provider.at(T_mean, outside.p)
        assert result.outside_alfa_mean == pytest.approx(
            result.final_result.outside_side_thermal.alfa
        )
    assert cold_result.outside_alfa_mean != pytest.approx(warm_result.outside_alfa_mean)


def test_steam_adapter_has_no_fake_provider_or_private_wet_gas_imports():
    source = inspect.getsource(steam_integration)
    assert "ConstantPropertyProvider" not in source
    tree = ast.parse(source)
    private_imports = [
        name.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {
            "core.phase_change.integration",
            "core.phase_change.rating_integration",
        }
        for name in node.names
        if name.name.startswith("_")
    ]
    assert private_imports == []


def test_saturated_liquid_subcooling_does_not_require_orientation():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0, p=P,
        quality_in=0.0,
    )
    result = _hx(orientation=None).simulate(inside, _outside_sim())
    assert result.inside_phase_change.phase_in is WaterSteamPhase.SATURATED_LIQUID
    assert result.inside_phase_change.phase_out is WaterSteamPhase.SUBCOOLED_LIQUID
    assert result.inside_phase_change.Q_condensation == 0.0


@pytest.mark.parametrize(
    ("inlet_kwargs", "outlet_kwargs", "expected_phase"),
    [
        ({"quality_in": 1.0}, {"quality_out": 0.4}, WaterSteamPhase.TWO_PHASE),
        ({"quality_in": 1.0}, {"quality_out": 0.0}, WaterSteamPhase.SATURATED_LIQUID),
        ({"quality_in": 1.0}, {"T_out": 350.0}, WaterSteamPhase.SUBCOOLED_LIQUID),
        ({"quality_in": 0.8}, {"quality_out": 0.2}, WaterSteamPhase.TWO_PHASE),
        ({"T_in": 520.0}, {"quality_out": 0.5}, WaterSteamPhase.TWO_PHASE),
        ({"T_in": 520.0}, {"T_out": 350.0}, WaterSteamPhase.SUBCOOLED_LIQUID),
    ],
)
def test_rating_reuses_same_zone_physics(inlet_kwargs, outlet_kwargs, expected_phase):
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        **inlet_kwargs, **outlet_kwargs,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=30.0, T_in=300.0,
    )
    result = _hx().rate(inside, outside)
    steam = result.inside_phase_change
    assert isinstance(steam, WaterSteamPhaseChangeResult)
    assert steam.phase_out is expected_phase
    assert result.UA_required == pytest.approx(steam.UA_total)
    assert result.UA_actual == pytest.approx(
        result.UA_required * result.A_o / result.A_required
    )
    assert result.inside_properties_outlet.phase is expected_phase
    assert math.isnan(result.inside_dp_total)


def test_steam_rating_rejects_effectiveness_only_target():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        quality_in=1.0,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=30.0, T_in=300.0,
    )
    with pytest.raises(ValueError, match="effectiveness-only"):
        _hx().rate(inside, outside, effectiveness=0.5)


def test_steam_rating_rejects_inconsistent_over_specified_duties():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        quality_in=1.0, quality_out=0.5,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=30.0, T_in=300.0,
    )
    with pytest.raises(ValueError, match="Over-specified"):
        _hx().rate(inside, outside, Q=1.0)


def test_zone_condensation_alpha_is_physical_and_top_level_alpha_is_reporting_only():
    result = _hx().simulate(_inside_sim(T_in=520.0), _outside_sim())
    steam = result.inside_phase_change
    assert steam.zone_alpha_condensation > 0.0
    assert result.inside_alfa_mean > 0.0
    assert result.UA == pytest.approx(steam.UA_total)
    assert steam.zone_alpha_condensation != pytest.approx(result.inside_alfa_mean)
    assert not hasattr(steam, "solution")
    assert not hasattr(steam, "runtime_s")


def test_disabled_mode_rejects_required_saturation_crossing():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0, p=P,
        quality_in=1.0,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )
    with pytest.raises(PhaseChangeDisabledButRequiredError) as caught:
        _hx().simulate(inside, _outside_sim())
    assert caught.value.warning_code == WC.PHASE_CHANGE_DISABLED_BUT_REQUIRED


def test_missing_geometry_orientation_is_rejected_only_when_condensation_is_active():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0, p=P,
        quality_in=1.0,
    )
    with pytest.raises(ValueError, match="tube_orientation on BareTube"):
        _hx(orientation=None).simulate(inside, _outside_sim())


def test_pure_steam_outside_is_controlled_unsupported_scope():
    outside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0, p=P,
        quality_in=1.0,
    )
    inside = HXSideInput(
        provider=OUTSIDE_PROVIDER, m_dot=5.0, T_in=300.0, p=101325.0,
    )
    with pytest.raises(PureSteamOutsideNotSupportedError) as caught:
        _hx().simulate(inside, outside)
    assert caught.value.warning_code == WC.PURE_STEAM_OUTSIDE_NOT_SUPPORTED


def test_rating_rejects_second_auto_phase_changing_wet_gas_side():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        quality_in=1.0, quality_out=0.5,
    )
    wet = GasMixturePropertyProvider(
        GasMixtureSpec(
            components={"N2": 0.75, "O2": 0.15, "CO2": 0.02, "H2O": 0.08},
            basis="mole",
        )
    )
    outside = BalanceSideSpec(
        provider=wet, p=101325.0, m_dot=10.0, T_in=300.0,
    )
    with pytest.raises(MultiplePhaseChangeSidesError):
        _hx().rate(inside, outside)


def test_iapws_provider_is_reported_as_pure_steam_capable():
    capability = detect_phase_change_capability(IAPWS97WaterSteamProvider())
    assert capability.capable
    assert capability.provider_kind == "pure_water_steam"
    assert capability.W_in is None


def test_subcooled_sensible_water_keeps_existing_simulation_path():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=5.0,
        T_in=300.0, p=P,
    )
    outside = HXSideInput(
        provider=OUTSIDE_PROVIDER, m_dot=20.0,
        T_in=500.0, p=101325.0,
    )
    expected = run_simulation(_hx(), inside, outside)
    actual = _hx().simulate(inside, outside)
    assert actual.q == expected.q
    assert actual.UA == expected.UA
    assert actual.inside_dp_total == expected.inside_dp_total


def test_public_reverse_boiling_is_controlled_unsupported():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0,
        T_in=300.0, p=P,
    )
    outside = HXSideInput(
        provider=OUTSIDE_PROVIDER, m_dot=30.0,
        T_in=700.0, p=101325.0,
    )
    with pytest.raises(SteamEvaporationNotSupportedError):
        _hx(n_rows=20, n_tubes_per_row=20).simulate(inside, outside)


def test_public_low_g_case_keeps_finite_condensation_zone_htc():
    hx = _hx(n_rows=20, n_tubes_per_row=20)
    result = hx.simulate(
        _inside_sim(quality_in=1.0, m_dot=0.5),
        _outside_sim(m_dot=20.0),
    )
    steam = result.inside_phase_change
    assert steam.mass_flux < 5.0
    assert steam.zone_alpha_condensation > 1000.0
    assert all(math.isfinite(value) for value in (result.q, result.UA, steam.A_total))
