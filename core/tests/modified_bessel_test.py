# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Deterministic validation of the dependency-free modified Bessel helper."""

from __future__ import annotations

import math

import pytest

from core.heat_transfer.modified_bessel import (
    bessel_i0,
    bessel_i1,
    bessel_k0,
    bessel_k1,
    scaled_bessel_i0,
    scaled_bessel_i1,
    scaled_bessel_k0,
    scaled_bessel_k1,
)


@pytest.mark.parametrize(
    "x, expected_i0, expected_i1, expected_k0, expected_k1",
    [
        (
            0.1,
            1.0025015629340956,
            0.050062526047092694,
            2.4270690247020164,
            9.853844780870606,
        ),
        (
            1.0,
            1.2660658777520084,
            0.5651591039924851,
            0.42102443824070834,
            0.6019072301972346,
        ),
        (
            2.0,
            2.2795853023360673,
            1.590636854637329,
            0.11389387274953344,
            0.13986588181652243,
        ),
        (
            10.0,
            2815.716628466254,
            2670.9883037012546,
            1.778006231616765e-5,
            1.8648773453825585e-5,
        ),
    ],
)
def test_modified_bessel_functions_match_frozen_reference_values(
    x: float,
    expected_i0: float,
    expected_i1: float,
    expected_k0: float,
    expected_k1: float,
) -> None:
    assert bessel_i0(x) == pytest.approx(expected_i0, rel=2.0e-6)
    assert bessel_i1(x) == pytest.approx(expected_i1, rel=2.0e-6)
    assert bessel_k0(x) == pytest.approx(expected_k0, rel=2.0e-6)
    assert bessel_k1(x) == pytest.approx(expected_k1, rel=2.0e-6)


@pytest.mark.parametrize(
    "x",
    [0.05, 0.5, 1.0, 2.0, 3.75, 10.0, 30.0, 300.0, 1000.0],
)
def test_scaled_functions_satisfy_modified_bessel_wronskian(x: float) -> None:
    # Scaling cancels between I and K, so the exact identity remains
    # I0e*K1e + I1e*K0e = 1/x without overflow at large x.
    lhs = (
        scaled_bessel_i0(x) * scaled_bessel_k1(x)
        + scaled_bessel_i1(x) * scaled_bessel_k0(x)
    )
    assert lhs == pytest.approx(1.0 / x, rel=1.0e-6)


@pytest.mark.parametrize("x", [0.1, 1.0, 2.0, 3.75, 10.0, 30.0])
def test_scaled_and_unscaled_functions_are_consistent(x: float) -> None:
    assert scaled_bessel_i0(x) == pytest.approx(
        math.exp(-x) * bessel_i0(x), rel=2.0e-15
    )
    assert scaled_bessel_i1(x) == pytest.approx(
        math.exp(-x) * bessel_i1(x), rel=2.0e-15
    )
    assert scaled_bessel_k0(x) == pytest.approx(
        math.exp(x) * bessel_k0(x), rel=2.0e-15
    )
    assert scaled_bessel_k1(x) == pytest.approx(
        math.exp(x) * bessel_k1(x), rel=2.0e-15
    )


@pytest.mark.parametrize(
    "function, switch",
    [
        (bessel_i0, 3.75),
        (bessel_i1, 3.75),
        (bessel_k0, 2.0),
        (bessel_k1, 2.0),
    ],
)
def test_piecewise_approximations_are_continuous_at_switches(
    function,
    switch: float,
) -> None:
    below = function(math.nextafter(switch, 0.0))
    above = function(math.nextafter(switch, math.inf))
    assert below == pytest.approx(above, rel=2.0e-6)


def test_first_kind_parity_and_zero_values() -> None:
    assert bessel_i0(0.0) == 1.0
    assert bessel_i1(0.0) == 0.0
    assert bessel_i0(-1.25) == bessel_i0(1.25)
    assert bessel_i1(-1.25) == -bessel_i1(1.25)


@pytest.mark.parametrize(
    "function",
    [bessel_i0, bessel_i1, scaled_bessel_i0, scaled_bessel_i1],
)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_first_kind_functions_reject_nonfinite_arguments(function, bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        function(bad)


@pytest.mark.parametrize(
    "function",
    [bessel_k0, bessel_k1, scaled_bessel_k0, scaled_bessel_k1],
)
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_second_kind_functions_reject_invalid_domain(function, bad: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        function(bad)


def test_scaled_functions_remain_finite_beyond_unscaled_exponential_range() -> None:
    x = 1000.0
    values = (
        scaled_bessel_i0(x),
        scaled_bessel_i1(x),
        scaled_bessel_k0(x),
        scaled_bessel_k1(x),
    )
    assert all(math.isfinite(value) and value > 0.0 for value in values)
