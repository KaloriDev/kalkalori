# SPDX-License-Identifier: GPL-3.0-only
"""Selectively recovered IF97 parity/validation tests; no historical wet workflow."""
from __future__ import annotations
import math
from types import SimpleNamespace
from iapws import IAPWS97
import pytest
from core.phase_change.water_equilibrium import saturated_water_ratio
from core.properties.water import (
    water_saturation_pressure,
    water_saturation_liquid_enthalpy,
    water_saturation_vapor_enthalpy,
    water_latent_heat_of_vaporization,
)


@pytest.mark.parametrize(
    "temperature",
    [
        273.16,
        280.013,
        295.123,
        308.15,
        312.123,
        323.15,
        330.271,
        333.15,
        335.15,
        350.125,
        373.15,
        393.15,
        500.123,
        math.nextafter(623.15, -math.inf),
        623.15,
        math.nextafter(623.15, math.inf),
        640.123,
        647.095,
    ],
)
def test_exact_query_evaluator_matches_full_state_adapter(temperature):
    # Independent full state construction. Keep the public compatibility
    # adapter's T -> pressure -> quality convention and SI roundtrip.
    tx_liquid = IAPWS97(T=temperature, x=0.0)
    tx_vapor = IAPWS97(T=temperature, x=1.0)
    pressure = float(tx_liquid.P) * 1e6
    hf = float(IAPWS97(P=pressure / 1e6, x=0.0).h) * 1000.0
    hg = float(IAPWS97(P=(float(tx_vapor.P) * 1e6) / 1e6, x=1.0).h) * 1000.0
    expected = [pressure, hf, hg, hg - hf]
    for _ in range(2):  # First evaluation and exact-key cache hit.
        actual = [
            water_saturation_pressure(temperature),
            water_saturation_liquid_enthalpy(T=temperature),
            water_saturation_vapor_enthalpy(T=temperature),
            water_latent_heat_of_vaporization(T=temperature),
        ]
        for index, (value, reference) in enumerate(zip(actual, expected)):
            assert value == pytest.approx(
                reference, rel=2e-12, abs=1e-8 if index == 0 else 1e-7
            )
    # Keep noncondensable headroom without relying on a saturation clip.
    total_pressure = 2.0 * pressure + 101325.0
    expected_ratio = (0.01801528 / 0.02897) * pressure / (total_pressure - pressure)
    assert saturated_water_ratio(
        p_total=total_pressure, T=temperature, M_dry=0.02897
    ) == pytest.approx(expected_ratio, rel=2e-12, abs=1e-14)


@pytest.mark.parametrize("layout", ["current", "legacy"])
def test_enthalpy_only_equations_support_backend_constant_layouts(monkeypatch, layout):
    from core.properties import water

    backend = water._if97
    coefficients = backend.Const
    if hasattr(coefficients, "Region2_n"):
        nj = coefficients.Region2_n * coefficients.Region2_Lj
        i, j = coefficients.Region2_Li, coefficients.Region2_Lj
        n = coefficients.Region2_n
    else:
        nj = coefficients.Region2_nr_Jr_product
        i, j, n = (
            coefficients.Region2_Ir,
            coefficients.Region2_Jr,
            coefficients.Region2_nr,
        )
    fields = {
        name: getattr(coefficients, name)
        for name in (
            "Region1_n",
            "Region1_Lj",
            "Region1_Li",
            "Region1_Lj_less_1",
            "Region2_cp0_no",
            "Region2_cp0_Jo",
        )
    }
    if layout == "current":
        fields.update(Region2_n=n, Region2_Lj=j, Region2_Li=i, Region2_Lj_less_1=j - 1)
    else:
        fields.update(Region2_nr_Jr_product=nj, Region2_Ir=i, Region2_Jr_less_1=j - 1)
    # Compute the reference through the installed backend before replacing
    # its constant namespace, so the oracle is independent of either adapter.
    states = [
        (temperature, backend._PSat_T(temperature))
        for temperature in (308.15, 323.15, 333.15, 335.15, 373.15, 393.15)
    ]
    expected = [
        (
            float(backend._Region1(t, p)["h"]) * 1000,
            float(backend._Region2(t, p)["h"]) * 1000,
        )
        for t, p in states
    ]
    monkeypatch.setattr(backend, "Const", SimpleNamespace(**fields))
    for (t, p), pair in zip(states, expected):
        assert water._region12_saturation_enthalpies(t, p) == pair


@pytest.mark.parametrize(
    "temperature", [273.15, 647.096, 650.0, math.nan, math.inf, -math.inf]
)
def test_saturation_validation_is_not_bypassed_by_cache(temperature):
    for function in [
        water_saturation_liquid_enthalpy,
        water_saturation_vapor_enthalpy,
        water_latent_heat_of_vaporization,
    ]:
        with pytest.raises(ValueError):
            function(T=temperature)
    with pytest.raises(ValueError):
        water_saturation_pressure(temperature)


def test_pressure_enthalpy_mode_and_exclusive_input_contract_are_preserved():
    pressure = 8e5
    hf = float(IAPWS97(P=pressure / 1e6, x=0.0).h) * 1000.0
    hg = float(IAPWS97(P=pressure / 1e6, x=1.0).h) * 1000.0
    assert water_saturation_liquid_enthalpy(p=pressure) == pytest.approx(
        hf, rel=2e-12, abs=1e-7
    )
    assert water_latent_heat_of_vaporization(p=pressure) == pytest.approx(
        hg - hf, rel=2e-12, abs=1e-7
    )
    for options in [{}, {"T": 312.123, "p": pressure}]:
        with pytest.raises(ValueError):
            water_saturation_liquid_enthalpy(**options)


@pytest.mark.parametrize(
    "quality,function",
    [
        (0.0, water_saturation_liquid_enthalpy),
        (1.0, water_saturation_vapor_enthalpy),
    ],
)
def test_high_temperature_enthalpy_does_not_require_unrequested_phase(
    monkeypatch, quality, function
):
    import core.properties.water as water

    calls = []

    def requested_phase_only(*, T, x):
        calls.append((T, x))
        if x != quality:
            raise ValueError("Unrequested phase is unavailable.")
        return SimpleNamespace(h=123456.0)

    water._saturation_enthalpy_at_temperature.cache_clear()
    monkeypatch.setattr(water, "water_steam_props_iapws97", requested_phase_only)
    try:
        assert function(T=640.321) == 123456.0
        assert function(T=640.321) == 123456.0
        assert calls == [(640.321, quality)]
    finally:
        water._saturation_enthalpy_at_temperature.cache_clear()
