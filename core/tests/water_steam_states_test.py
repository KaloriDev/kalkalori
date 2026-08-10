import pytest

from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.properties.water import (
    IAPWS97WaterSteamProvider,
    WATER_CRITICAL_PRESSURE_PA,
    WaterSteamPhase,
    water_saturation_snapshot,
    water_steam_props_iapws97,
)


P = 1.0e6


def test_existing_temperature_pressure_provider_path_is_unchanged():
    provider = IAPWS97WaterSteamProvider()
    state = provider.full_at(T=300.0, p=P)
    assert provider.at(T=300.0, p=P) == state.transport
    assert state.phase is WaterSteamPhase.SUBCOOLED_LIQUID


@pytest.mark.parametrize(
    ("temperature", "phase"),
    [(500.0, WaterSteamPhase.SUPERHEATED_VAPOR), (300.0, WaterSteamPhase.SUBCOOLED_LIQUID)],
)
def test_temperature_pressure_classification(temperature, phase):
    state = water_steam_props_iapws97(T=temperature, p=P)
    assert state.phase is phase
    assert state.quality is None
    assert state.transport is not None


def test_temperature_pressure_saturation_ambiguity_is_rejected():
    saturation = water_saturation_snapshot(P)
    with pytest.raises(ValueError, match="saturation line"):
        water_steam_props_iapws97(T=saturation.Tsat, p=P)


@pytest.mark.parametrize(
    ("quality", "phase"),
    [
        (0.0, WaterSteamPhase.SATURATED_LIQUID),
        (0.5, WaterSteamPhase.TWO_PHASE),
        (1.0, WaterSteamPhase.SATURATED_VAPOR),
    ],
)
def test_pressure_quality_states(quality, phase):
    saturation = water_saturation_snapshot(P)
    state = water_steam_props_iapws97(p=P, x=quality)
    assert state.phase is phase
    assert state.quality == quality
    assert state.T == pytest.approx(saturation.Tsat)
    assert state.h == pytest.approx(saturation.hf + quality * saturation.hfg)
    assert (state.transport is None) is (0.0 < quality < 1.0)


@pytest.mark.parametrize(
    ("enthalpy_offset", "phase"),
    [
        (100_000.0, WaterSteamPhase.SUPERHEATED_VAPOR),
        (-100_000.0, WaterSteamPhase.SUBCOOLED_LIQUID),
    ],
)
def test_pressure_enthalpy_single_phase_states(enthalpy_offset, phase):
    saturation = water_saturation_snapshot(P)
    boundary = saturation.hg if enthalpy_offset > 0.0 else saturation.hf
    state = water_steam_props_iapws97(p=P, h=boundary + enthalpy_offset)
    assert state.phase is phase
    assert state.quality is None
    assert state.transport is not None


def test_pressure_enthalpy_two_phase_state():
    saturation = water_saturation_snapshot(P)
    state = water_steam_props_iapws97(p=P, h=saturation.hf + 0.25 * saturation.hfg)
    assert state.phase is WaterSteamPhase.TWO_PHASE
    assert state.quality == pytest.approx(0.25)
    assert state.transport is None
    assert state.cp is None
    assert state.Pr is None


@pytest.mark.parametrize(
    ("boundary_name", "phase", "quality"),
    [
        ("hf", WaterSteamPhase.SATURATED_LIQUID, 0.0),
        ("hg", WaterSteamPhase.SATURATED_VAPOR, 1.0),
    ],
)
def test_exact_saturation_enthalpy_boundaries(boundary_name, phase, quality):
    saturation = water_saturation_snapshot(P)
    state = water_steam_props_iapws97(p=P, h=getattr(saturation, boundary_name))
    assert state.phase is phase
    assert state.quality == quality


def test_pressure_quality_enthalpy_round_trip():
    first = water_steam_props_iapws97(p=P, x=0.374)
    second = water_steam_props_iapws97(p=P, h=first.h)
    assert second.T == pytest.approx(first.T)
    assert second.h == pytest.approx(first.h)
    assert second.quality == pytest.approx(first.quality)


@pytest.mark.parametrize("quality", [-0.01, 1.01])
def test_invalid_quality_is_rejected(quality):
    with pytest.raises(ValueError, match="0.0 ... 1.0"):
        water_steam_props_iapws97(p=P, x=quality)


def test_conflicting_state_inputs_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        water_steam_props_iapws97(T=500.0, p=P, x=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        HXSideInput(
            provider=IAPWS97WaterSteamProvider(), m_dot=1.0, p=P,
            T_in=500.0, quality_in=1.0,
        )


def test_supercritical_phase_interpretation_is_unsupported():
    with pytest.raises(ValueError, match="critical"):
        water_steam_props_iapws97(T=700.0, p=WATER_CRITICAL_PRESSURE_PA)
    with pytest.raises(ValueError, match="critical"):
        water_steam_props_iapws97(p=WATER_CRITICAL_PRESSURE_PA + 1.0, h=3.0e6)


def test_simulation_and_rating_inlets_resolve_pressure_quality_and_enthalpy():
    provider = IAPWS97WaterSteamProvider()
    saturation = water_saturation_snapshot(P)
    simulation_side = HXSideInput(provider=provider, m_dot=2.0, p=P, quality_in=0.5)
    rating_side = BalanceSideSpec(provider=provider, m_dot=2.0, p=P, h_in=saturation.hg)

    assert simulation_side.state_specification == "p+x"
    assert simulation_side.T_in == pytest.approx(saturation.Tsat)
    assert simulation_side.h_in == pytest.approx(saturation.hf + 0.5 * saturation.hfg)
    assert simulation_side.water_steam_state.phase is WaterSteamPhase.TWO_PHASE
    assert rating_side.state_specification == "p+h"
    assert rating_side.quality_in == 1.0
    assert rating_side.water_steam_state.phase is WaterSteamPhase.SATURATED_VAPOR
