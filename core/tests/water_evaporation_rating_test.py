import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.phase_change import warning_codes as WC
from core.phase_change.capability import PureWaterPhaseChangeProviderNotSupportedError
from core.phase_change.integration import MultiplePhaseChangeSidesError
from core.phase_change.steam_integration import (
    PhaseChangeDisabledButRequiredError,
    PureSteamOutsideNotSupportedError,
)
from core.phase_change.types import PhaseChangeDirection, PhaseChangeMode
from core.properties.common import FluidTransportProperties
from core.properties.coolprop_backend import CoolPropFluidProvider
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import (
    IAPWS97WaterSteamProvider,
    WaterSteamPhase,
    water_steam_props_iapws97,
)


P = 1.0e6
HOT = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1005.0)
)


def _hx(*, orientation=TubeOrientation.VERTICAL_UPWARD):
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.020, D_o=0.024, length_total=4.0,
                length_effective=4.0, wall_k=16.0,
                tube_orientation=orientation,
            ),
            n_rows=10, n_tubes_per_row=10,
            pitch_transverse=0.04, pitch_longitudinal=0.04,
            layout="inline", n_passes_tube=1, flow_arrangement="crossflow",
        )
    )


def _inside(*, m_dot=1.0, mode=PhaseChangeMode.AUTO, **states):
    return BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=m_dot,
        phase_change_mode=mode, **states,
    )


def _outside(*, provider=HOT, m_dot=30.0, T_in=700.0, T_out=None, p=101325.0,
             mode=PhaseChangeMode.AUTO):
    return BalanceSideSpec(
        provider=provider, p=p, m_dot=m_dot, T_in=T_in, T_out=T_out,
        phase_change_mode=mode,
    )


@pytest.mark.parametrize(
    ("inside", "expected_phase", "expected_quality", "zone_flags"),
    [
        (
            _inside(quality_in=0.0, quality_out=0.5),
            WaterSteamPhase.TWO_PHASE, 0.5, (False, True, False),
        ),
        (
            _inside(T_in=350.0, quality_out=1.0),
            WaterSteamPhase.SATURATED_VAPOR, 1.0, (True, True, False),
        ),
        (
            _inside(T_in=350.0, T_out=520.0),
            WaterSteamPhase.SUPERHEATED_VAPOR, None, (True, True, True),
        ),
        (
            _inside(
                T_in=350.0,
                h_out=water_steam_props_iapws97(p=P, x=0.4).h,
            ),
            WaterSteamPhase.TWO_PHASE, 0.4, (True, True, False),
        ),
        (
            _inside(
                h_in=water_steam_props_iapws97(p=P, x=0.2).h,
                quality_out=0.8,
            ),
            WaterSteamPhase.TWO_PHASE, 0.8, (False, True, False),
        ),
    ],
)
def test_rating_accepts_quality_h_and_unambiguous_temperature_targets(
    inside, expected_phase, expected_quality, zone_flags
):
    result = _hx().rate(inside, _outside())
    water = result.inside_phase_change
    assert water.direction is PhaseChangeDirection.EVAPORATION
    assert water.phase_out is expected_phase
    if expected_quality is None:
        assert water.quality_out is None
    else:
        assert water.quality_out == pytest.approx(expected_quality)
    assert tuple(value > 0.0 for value in (
        water.Q_preheat, water.Q_evaporation, water.Q_superheat
    )) == zone_flags
    assert result.Q_required == pytest.approx(
        inside.m_dot * (water.h_out - water.h_in)
    )


def test_rating_explicit_q_resolves_ph_outlet_and_does_not_clamp_at_x_one():
    inside = _inside(T_in=350.0)
    saturated_vapor = water_steam_props_iapws97(p=P, x=1.0)
    Q_to_vapor = inside.m_dot * (saturated_vapor.h - inside.h_in)
    result = _hx().rate(inside, _outside(), Q=Q_to_vapor + 100_000.0)
    water = result.inside_phase_change
    assert water.phase_out is WaterSteamPhase.SUPERHEATED_VAPOR
    assert water.Q_superheat == pytest.approx(100_000.0)
    assert water.h_out == pytest.approx(inside.h_in + result.Q_required / inside.m_dot)
    assert water.Q_total == pytest.approx(
        water.Q_preheat + water.Q_evaporation + water.Q_superheat
    )


def test_rating_duty_can_be_closed_by_the_opposing_temperature_program():
    outside = _outside(T_out=660.0)
    result = _hx().rate(_inside(T_in=350.0), outside)
    expected_Q = outside.m_dot * HOT.props.cp * (outside.T_in - outside.T_out)
    water = result.inside_phase_change
    assert result.Q_required == pytest.approx(expected_Q)
    assert water.h_out == pytest.approx(water.h_in + expected_Q / water.mass_flow_total)
    assert result.closed_balance.hot_is_inside is False
    assert result.closed_balance.outside.T_out == pytest.approx(outside.T_out)


def test_rating_area_ua_overdesign_and_equivalent_alpha_are_consistent():
    hx = _hx()
    result = hx.rate(_inside(T_in=350.0, quality_out=1.0), _outside())
    water = result.inside_phase_change
    assert result.A_required == pytest.approx(
        water.A_preheat + water.A_evaporation + water.A_superheat
    )
    assert result.UA_required == pytest.approx(
        water.zone_UA_preheat
        + water.zone_UA_evaporation
        + water.zone_UA_superheat
    )
    assert result.UA_required == pytest.approx(water.UA_total)
    assert result.UA_actual == pytest.approx(
        result.UA_required * result.A_o / result.A_required
    )
    assert result.overdesign_factor == pytest.approx(
        result.A_o / result.A_required - 1.0
    )
    assert result.ua_margin == pytest.approx(
        result.UA_actual / result.UA_required - 1.0
    )
    assert result.U_mean == pytest.approx(water.UA_total / water.A_total)
    tube = hx.bundle.tube
    reconstructed_u = 1.0 / (
        tube.D_o / (tube.D_i * water.inside_alpha_equivalent)
        + tube.D_o * math.log(tube.D_o / tube.D_i) / (2.0 * tube.wall_k)
        + 1.0 / result.alfa_o
    )
    assert reconstructed_u == pytest.approx(result.U_mean, rel=2e-12)


def test_rating_and_simulation_share_identical_zone_physics():
    hx = _hx()
    simulation = hx.simulate(
        HXSideInput(
            provider=IAPWS97WaterSteamProvider(), m_dot=1.0, T_in=350.0, p=P
        ),
        HXSideInput(
            provider=HOT, m_dot=30.0, T_in=700.0, p=101325.0
        ),
    )
    rating = hx.rate(
        _inside(T_in=350.0, h_out=simulation.inside_phase_change.h_out),
        _outside(),
    )
    sim = simulation.inside_phase_change
    rated = rating.inside_phase_change
    assert rating.A_required == pytest.approx(hx.bundle.total_outer_area, rel=2e-8)
    for name in (
        "Q_preheat", "Q_evaporation", "Q_superheat",
        "A_preheat", "A_evaporation", "A_superheat",
        "zone_alpha_preheat", "zone_alpha_evaporation", "zone_alpha_superheat",
        "zone_U_preheat", "zone_U_evaporation", "zone_U_superheat",
        "zone_UA_preheat", "zone_UA_evaporation", "zone_UA_superheat",
        "inside_alpha_equivalent",
    ):
        sim_value = getattr(sim, name)
        if sim_value is None:
            assert getattr(rated, name) is None
        else:
            assert getattr(rated, name) == pytest.approx(sim_value, rel=2e-8)


def test_include_simulation_reports_achievable_duty_and_keeps_public_results_only():
    result = _hx().rate(
        _inside(T_in=350.0, quality_out=0.5),
        _outside(),
        include_simulation=True,
    )
    assert result.simulation is not None
    assert result.Q_achievable == pytest.approx(result.simulation.q)
    assert result.Q_achievable != pytest.approx(result.Q_required)
    assert not hasattr(result.inside_phase_change, "solution")
    assert not hasattr(result.inside_phase_change, "runtime_s")


def test_active_rating_marks_tube_side_pressure_drop_unsupported():
    result = _hx().rate(_inside(quality_in=0.2, quality_out=0.8), _outside())
    water = result.inside_phase_change
    assert result.tube_side_hydraulic is None
    assert result.tube_side_pressure_drop is None
    assert math.isnan(result.inside_dp_total)
    assert water.two_phase_pressure_drop_supported is False
    assert WC.WATER_EVAPORATION_TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED in {
        warning.code for warning in water.warnings
    }


def test_effectiveness_only_rating_is_rejected():
    with pytest.raises(ValueError, match="effectiveness-only"):
        _hx().rate(_inside(quality_in=0.0), _outside(), effectiveness=0.5)


def test_inconsistent_over_specified_q_and_outlet_are_rejected():
    inside = _inside(T_in=350.0, quality_out=0.5)
    with pytest.raises(ValueError, match="Over-specified"):
        _hx().rate(inside, _outside(), Q=1.0)


def test_reverse_enthalpy_target_is_not_accepted_as_evaporation():
    inside = _inside(T_in=400.0, T_out=350.0)
    with pytest.raises(ValueError, match="increase tube-side water enthalpy"):
        _hx().rate(inside, _outside())


def test_disabled_mode_rejects_a_required_boiling_rating():
    with pytest.raises(PhaseChangeDisabledButRequiredError):
        _hx().rate(
            _inside(
                T_in=350.0, quality_out=0.5,
                mode=PhaseChangeMode.DISABLED,
            ),
            _outside(),
        )


@pytest.mark.parametrize("orientation", [None, TubeOrientation.VERTICAL_DOWNWARD])
def test_rating_requires_supported_orientation_only_for_active_boiling(orientation):
    with pytest.raises(ValueError, match="orientation"):
        _hx(orientation=orientation).rate(
            _inside(quality_in=0.0, quality_out=0.5), _outside()
        )


def test_rating_rejects_an_impossible_thermal_pinch():
    with pytest.raises(ValueError, match="temperature difference"):
        _hx().rate(
            _inside(T_in=350.0, T_out=520.0),
            _outside(T_in=460.0),
        )


def test_outside_pure_water_evaporation_rating_is_rejected():
    inside = BalanceSideSpec(provider=HOT, p=101325.0, m_dot=30.0, T_in=700.0)
    outside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        T_in=350.0, quality_out=0.5,
    )
    with pytest.raises(PureSteamOutsideNotSupportedError):
        _hx().rate(inside, outside)


def test_coolprop_water_rating_crossing_is_not_silently_replaced():
    inside = BalanceSideSpec(
        provider=CoolPropFluidProvider("Water"), p=P, m_dot=1.0,
        T_in=350.0, T_out=500.0,
    )
    with pytest.raises(PureWaterPhaseChangeProviderNotSupportedError):
        _hx().rate(inside, _outside())


class _FastWetGasProvider(GasMixturePropertyProvider):
    def at(self, T, p):
        return FluidTransportProperties(
            rho=12.0 * 500.0 / T, mu=2.0e-5, k=0.04, cp=1800.0
        )


def test_rating_rejects_second_auto_phase_changing_wet_gas_side():
    wet = _FastWetGasProvider(
        GasMixtureSpec(
            components={"N2": 0.49, "O2": 0.21, "H2O": 0.30}, basis="mole"
        )
    )
    with pytest.raises(MultiplePhaseChangeSidesError):
        _hx().rate(
            _inside(quality_in=0.0, quality_out=0.5),
            _outside(provider=wet, m_dot=10.0, T_in=500.0, p=5.0e6),
        )
