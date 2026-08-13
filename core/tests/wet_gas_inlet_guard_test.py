# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Preflight coverage for unsupported liquid water carried by a gas."""

from __future__ import annotations

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.phase_change import warning_codes as WC
from core.phase_change.capability import (
    LiquidWaterInGasInletNotSupportedError,
    WET_GAS_INLET_SATURATION_ABSOLUTE_TOLERANCE,
    WET_GAS_INLET_SATURATION_RELATIVE_TOLERANCE,
    reject_liquid_water_in_gas_inlet,
)
from core.phase_change.types import PhaseChangeMode
from core.phase_change.water_equilibrium import (
    dry_gas_average_molar_mass,
    saturated_water_ratio,
)
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import (
    GasMixturePropertyProvider,
    GasMixtureSpec,
    gas_mixture_from_dry_composition_and_water_ratio,
)
from core.properties.water import (
    IAPWS97WaterSteamProvider,
    WATER_TRIPLE_POINT_TEMPERATURE_K,
)


P = 101_325.0
DRY_COMPONENTS = {"N2": 0.79, "O2": 0.21}
M_DRY = dry_gas_average_molar_mass(DRY_COMPONENTS)
CONSTANT = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1005.0)
)


def _provider(*, T: float, saturation_factor: float) -> GasMixturePropertyProvider:
    W_sat = saturated_water_ratio(p_total=P, T=T, M_dry=M_DRY)
    return GasMixturePropertyProvider(
        gas_mixture_from_dry_composition_and_water_ratio(
            DRY_COMPONENTS,
            water_ratio=W_sat * saturation_factor,
        )
    )


def _provider_at_ratio(W: float) -> GasMixturePropertyProvider:
    return GasMixturePropertyProvider(
        gas_mixture_from_dry_composition_and_water_ratio(
            DRY_COMPONENTS,
            water_ratio=W,
        )
    )


def _hx() -> BareTubeHeatExchanger:
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.020, D_o=0.024, length_total=2.0,
                length_effective=2.0, wall_k=16.0,
            ),
            n_rows=2, n_tubes_per_row=4,
            pitch_transverse=0.04, pitch_longitudinal=0.04,
            layout="inline", n_passes_tube=1,
            flow_arrangement="counterflow",
        )
    )


@pytest.mark.parametrize("side", ["inside", "outside"])
def test_undersaturated_wet_gas_inlet_is_accepted(side: str) -> None:
    reject_liquid_water_in_gas_inlet(
        _provider(T=330.0, saturation_factor=0.8),
        T_in=330.0,
        p=P,
        side=side,
    )


def test_exactly_saturated_inlet_is_accepted() -> None:
    reject_liquid_water_in_gas_inlet(
        _provider(T=330.0, saturation_factor=1.0),
        T_in=330.0,
        p=P,
        side="inside",
    )


def test_near_saturation_within_combined_tolerance_is_accepted() -> None:
    W_sat = saturated_water_ratio(p_total=P, T=330.0, M_dry=M_DRY)
    tolerance = (
        WET_GAS_INLET_SATURATION_ABSOLUTE_TOLERANCE
        + WET_GAS_INLET_SATURATION_RELATIVE_TOLERANCE * W_sat
    )
    reject_liquid_water_in_gas_inlet(
        _provider_at_ratio(W_sat + 0.5 * tolerance),
        T_in=330.0,
        p=P,
        side="inside",
    )


@pytest.mark.parametrize("side", ["inside", "outside"])
def test_clearly_supersaturated_inlet_is_rejected(side: str) -> None:
    with pytest.raises(LiquidWaterInGasInletNotSupportedError) as caught:
        reject_liquid_water_in_gas_inlet(
            _provider(T=330.0, saturation_factor=1.05),
            T_in=330.0,
            p=P,
            side=side,
        )
    assert caught.value.warning_code == WC.LIQUID_WATER_IN_GAS_INLET_NOT_SUPPORTED
    message = str(caught.value)
    assert "gas-liquid water mixture" in message
    assert "GasMixturePropertyProvider" in message
    assert "evaporation of liquid water carried by a gas is not modelled" in message
    assert "No ordinary sensible or condensing result was returned" in message


@pytest.mark.parametrize("mode", [PhaseChangeMode.AUTO, PhaseChangeMode.DISABLED])
def test_simulation_rejects_before_dry_solve(monkeypatch, mode: PhaseChangeMode) -> None:
    import core.models.simulation as simulation

    def invalid_dry_solve(*args, **kwargs):
        raise AssertionError("dry solve must not run for a supersaturated inlet")

    monkeypatch.setattr(simulation, "run_simulation", invalid_dry_solve)
    inside = HXSideInput(
        provider=_provider(T=330.0, saturation_factor=1.05),
        m_dot=1.0, T_in=330.0, p=P, phase_change_mode=mode,
    )
    outside = HXSideInput(provider=CONSTANT, m_dot=2.0, T_in=300.0, p=P)
    with pytest.raises(LiquidWaterInGasInletNotSupportedError):
        _hx().simulate(inside, outside)


def test_rating_rejects_before_heat_balance_closure(monkeypatch) -> None:
    import core.phase_change.rating_integration as rating_integration

    def invalid_rating(*args, **kwargs):
        raise AssertionError("rating closure must not run for a supersaturated inlet")

    monkeypatch.setattr(rating_integration, "apply_phase_change_to_rating", invalid_rating)
    inside = BalanceSideSpec(
        provider=CONSTANT, p=P, m_dot=2.0, T_in=300.0, T_out=310.0,
    )
    outside = BalanceSideSpec(
        provider=_provider(T=330.0, saturation_factor=1.05),
        p=P, m_dot=1.0, T_in=330.0,
    )
    with pytest.raises(LiquidWaterInGasInletNotSupportedError):
        _hx().rate(inside, outside)


def test_dry_non_water_and_pure_water_providers_are_unchanged() -> None:
    dry = GasMixturePropertyProvider(
        GasMixtureSpec(components=DRY_COMPONENTS, basis="mole")
    )
    for provider in (dry, CONSTANT, IAPWS97WaterSteamProvider()):
        reject_liquid_water_in_gas_inlet(
            provider, T_in=330.0, p=P, side="inside"
        )


def test_above_boiling_range_does_not_call_saturation_ratio_out_of_domain() -> None:
    reject_liquid_water_in_gas_inlet(
        _provider_at_ratio(10.0), T_in=400.0, p=P, side="outside"
    )


def test_sub_triple_point_inlet_is_left_to_existing_frost_path() -> None:
    reject_liquid_water_in_gas_inlet(
        _provider_at_ratio(1.0),
        T_in=WATER_TRIPLE_POINT_TEMPERATURE_K - 1.0,
        p=P,
        side="inside",
    )
