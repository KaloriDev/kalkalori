"""Provider-boundary policy for pure-water phase change."""

import pytest

from core.geometry import BareTube, TubeBundle, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.phase_change import warning_codes as WC
from core.phase_change.capability import (
    PureWaterPhaseChangeProviderNotSupportedError,
    detect_phase_change_capability,
)
from core.properties.common import FluidTransportProperties
from core.properties.coolprop_backend import CoolPropFluidProvider
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import IAPWS97WaterSteamProvider


P = 1.0e6
OUTSIDE = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1005.0)
)


def _hx(rows=2, columns=2):
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.020, D_o=0.024, length_total=4.0,
                length_effective=4.0, wall_k=16.0,
                tube_orientation=TubeOrientation.VERTICAL_DOWNWARD,
            ),
            n_rows=rows, n_tubes_per_row=columns,
            pitch_transverse=0.04, pitch_longitudinal=0.04,
            layout="inline", n_passes_tube=1, flow_arrangement="crossflow",
        )
    )


def _outside(T=290.0):
    return HXSideInput(OUTSIDE, 30.0, T, 101325.0)


def _pure_gas_mixture():
    return GasMixturePropertyProvider(
        GasMixtureSpec(components={"H2O": 1.0}, basis="mole")
    )


def test_only_iapws_is_officially_pure_steam_phase_change_capable():
    iapws = detect_phase_change_capability(IAPWS97WaterSteamProvider())
    coolprop = detect_phase_change_capability(CoolPropFluidProvider("Water"))
    gas_mixture = detect_phase_change_capability(_pure_gas_mixture())
    assert (iapws.capable, iapws.provider_kind) == (True, "pure_water_steam")
    assert (coolprop.capable, coolprop.provider_kind) == (
        False, "pure_water_coolprop_unsupported"
    )
    assert (gas_mixture.capable, gas_mixture.provider_kind) == (
        False, "pure_water_gas_mixture_unsupported"
    )


def test_coolprop_water_single_phase_simulation_keeps_coolprop_provider():
    provider = CoolPropFluidProvider("Water")
    inside = HXSideInput(provider, 5.0, 300.0, P)
    result = _hx().simulate(inside, _outside())
    assert result.T_out_inside < inside.T_in
    assert result.inside_phase_change.capable is False
    assert result.inside_phase_change.possible is False
    assert result.inside_phase_change.active is False
    expected = provider.at(T=result.T_mean_inside, p=P)
    assert result.inside_properties_midpoint.props == expected


def test_coolprop_water_single_phase_rating_keeps_public_provider_identity():
    provider = CoolPropFluidProvider("Water")
    inside = BalanceSideSpec(provider, P, 1.0, 600.0, 550.0)
    outside = BalanceSideSpec(OUTSIDE, 101325.0, 30.0, 500.0)
    result = _hx().rate(inside, outside)
    assert result.closed_balance.inside.provider is provider
    assert result.inside_phase_change.active is False


@pytest.mark.parametrize("provider", [CoolPropFluidProvider("Water"), _pure_gas_mixture()])
def test_non_iapws_quality_input_is_controlled_unsupported(provider):
    with pytest.raises(PureWaterPhaseChangeProviderNotSupportedError) as caught:
        HXSideInput(provider=provider, m_dot=1.0, p=P, quality_in=0.5)
    assert caught.value.warning_code == WC.PURE_WATER_PHASE_CHANGE_PROVIDER_NOT_SUPPORTED


@pytest.mark.parametrize("provider", [CoolPropFluidProvider("Water"), _pure_gas_mixture()])
def test_non_iapws_condensation_crossing_is_controlled_unsupported(provider):
    inside = HXSideInput(provider, 1.0, 500.0, P)
    with pytest.raises(PureWaterPhaseChangeProviderNotSupportedError) as caught:
        _hx(rows=10, columns=10).simulate(inside, _outside(T=300.0))
    assert caught.value.warning_code == WC.PURE_WATER_PHASE_CHANGE_PROVIDER_NOT_SUPPORTED
    assert "not silently replaced" in str(caught.value)


def test_pure_h2o_gas_mixture_stays_single_phase_gas_when_no_crossing_occurs():
    provider = _pure_gas_mixture()
    result = _hx(rows=1, columns=1).simulate(
        HXSideInput(provider, 10.0, 600.0, P),
        _outside(T=500.0),
    )
    assert result.T_out_inside > 550.0
    assert result.inside_phase_change.capable is False
    assert result.inside_phase_change.active is False
