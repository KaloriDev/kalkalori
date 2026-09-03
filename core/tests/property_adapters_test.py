# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Deterministic regressions for core.properties.adapters.

core/properties/adapters.py previously had no .py pytest coverage and was
exercised only manually through a notebook (since removed). This module
covers the field-passthrough contract synthetically (no backend required)
and ports the notebook's CoolProp-vs-IAPWS cross-check and CoolProp Air
smoke ranges as real, skip-guarded assertions.
"""

from __future__ import annotations

import math

import pytest

from core.properties.adapters import (
    from_internal_fluid_props,
    from_internal_pressure_drop_fluid_props,
    from_outside_fluid_props,
    to_internal_fluid_props,
    to_internal_pressure_drop_fluid_props,
    to_outside_fluid_props,
)
from core.properties.common import FluidTransportProperties
from core.properties.coolprop_backend import (
    CoolPropFluidProvider,
    check_coolprop_backend_availability,
    coolprop_props,
    coolprop_transport_properties,
)
from core.properties.water import water_steam_props_iapws97
from core.heat_transfer.internal_flow import FluidProps as InternalFluidProps
from core.heat_transfer.outside_flow import FluidProps as OutsideFluidProps
from core.pressure_drop.internal_pressure_drop import (
    FluidProps as InternalPressureDropFluidProps,
)

_PROPS = FluidTransportProperties(rho=997.0, mu=8.9e-4, k=0.6, cp=4180.0)


def test_to_internal_fluid_props_round_trips_all_fields() -> None:
    converted = to_internal_fluid_props(_PROPS)

    assert isinstance(converted, InternalFluidProps)
    assert converted.rho == _PROPS.rho
    assert converted.mu == _PROPS.mu
    assert converted.k == _PROPS.k
    assert converted.cp == _PROPS.cp

    back = from_internal_fluid_props(converted)
    assert back == _PROPS


def test_to_outside_fluid_props_round_trips_all_fields() -> None:
    converted = to_outside_fluid_props(_PROPS)

    assert isinstance(converted, OutsideFluidProps)
    assert converted.rho == _PROPS.rho
    assert converted.mu == _PROPS.mu
    assert converted.k == _PROPS.k
    assert converted.cp == _PROPS.cp

    back = from_outside_fluid_props(converted)
    assert back == _PROPS


def test_pressure_drop_adapter_keeps_only_rho_and_mu() -> None:
    converted = to_internal_pressure_drop_fluid_props(_PROPS)

    assert isinstance(converted, InternalPressureDropFluidProps)
    assert converted.rho == _PROPS.rho
    assert converted.mu == _PROPS.mu
    assert not hasattr(converted, "k")
    assert not hasattr(converted, "cp")


def test_pressure_drop_adapter_reverse_conversion_currently_raises() -> None:
    # Pre-existing defect, found while adding this coverage (not introduced
    # by it, and left unfixed here -- out of scope for a test-suite
    # cleanup): from_internal_pressure_drop_fluid_props()'s docstring
    # promises k/cp backfilled as NaN, but FluidTransportProperties rejects
    # any non-finite field in __post_init__, so every call currently raises
    # instead of returning the documented result. This test pins today's
    # actual behavior so a future fix (either relaxing the dataclass or
    # changing the adapter contract) is a visible, deliberate change here.
    narrow = to_internal_pressure_drop_fluid_props(_PROPS)
    with pytest.raises(ValueError, match="must be finite"):
        from_internal_pressure_drop_fluid_props(narrow)


@pytest.mark.skipif(
    not check_coolprop_backend_availability("HEOS").available,
    reason="CoolProp HEOS backend is not available in this environment.",
)
def test_coolprop_and_iapws_liquid_water_density_agree_after_adapting() -> None:
    T, p = 298.15, 101_325.0

    iapws_transport = water_steam_props_iapws97(T=T, p=p).transport
    assert iapws_transport is not None

    coolprop_transport = coolprop_transport_properties(T=T, p=p, fluid="Water")
    adapted = to_outside_fluid_props(coolprop_transport)

    # Adapting must not perturb the value used for the cross-backend check.
    assert adapted.rho == coolprop_transport.rho
    assert math.isclose(adapted.rho, iapws_transport.rho, rel_tol=0.01)


@pytest.mark.skipif(
    not check_coolprop_backend_availability("HEOS").available,
    reason="CoolProp HEOS backend is not available in this environment.",
)
def test_coolprop_air_properties_are_within_expected_range_at_ambient_state() -> None:
    air = coolprop_props(T=293.15, p=101_325.0, fluid="Air")

    assert 1.15 < air.transport.rho < 1.25
    assert 1.7e-5 < air.transport.mu < 1.9e-5
    assert 0.024 < air.transport.k < 0.027
    assert 990.0 < air.transport.cp < 1020.0

    provider = CoolPropFluidProvider("Air")
    assert provider.at(T=293.15, p=101_325.0).rho == air.transport.rho
