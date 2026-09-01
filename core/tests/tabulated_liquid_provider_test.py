"""Deterministic tests for core.properties.tabulated_liquid (v0.7.7).

No CoolProp dependency: TabulatedLiquidProvider is a manual-entry provider,
so these tests exercise it directly plus one small synthetic exchanger
integration test through the existing public Simulation path.
"""

from __future__ import annotations

import math

import pytest

from core.properties import (
    ConstantPropertyProvider,
    FluidTransportProperties,
    LiquidPropertyPoint,
    TabulatedLiquidProvider,
)
from core.geometry.tube import BareTube
from core.geometry.bundle import TubeBundle
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput


# ---------------------------------------------------------------------------
# LiquidPropertyPoint validation
# ---------------------------------------------------------------------------

_VALID_KWARGS = dict(T=300.0, rho=800.0, cp=2000.0, mu=0.01, k=0.12)


@pytest.mark.parametrize("field", ["T", "rho", "cp", "mu", "k"])
@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_liquid_property_point_rejects_invalid_field(field: str, bad_value: float) -> None:
    kwargs = dict(_VALID_KWARGS)
    kwargs[field] = bad_value
    with pytest.raises(ValueError):
        LiquidPropertyPoint(**kwargs)


def test_liquid_property_point_accepts_valid_values() -> None:
    point = LiquidPropertyPoint(**_VALID_KWARGS)
    assert point.T == _VALID_KWARGS["T"]
    assert point.rho == _VALID_KWARGS["rho"]
    assert point.cp == _VALID_KWARGS["cp"]
    assert point.mu == _VALID_KWARGS["mu"]
    assert point.k == _VALID_KWARGS["k"]


# ---------------------------------------------------------------------------
# One-point provider
# ---------------------------------------------------------------------------

_OIL_POINT = LiquidPropertyPoint(T=293.15, rho=870.0, cp=1900.0, mu=0.120, k=0.130)


def _oil_provider() -> TabulatedLiquidProvider:
    return TabulatedLiquidProvider(name="Turbine oil ISO VG46", points=[_OIL_POINT])


def test_single_point_properties_are_constant_across_temperature() -> None:
    oil = _oil_provider()
    for T in (250.0, 293.15, 320.0, 450.0, 900.0):
        props = oil.at(T, 101_325.0)
        assert props.rho == _OIL_POINT.rho
        assert props.cp == _OIL_POINT.cp
        assert props.mu == _OIL_POINT.mu
        assert props.k == _OIL_POINT.k


def test_single_point_properties_independent_of_pressure() -> None:
    oil = _oil_provider()
    p1 = oil.at(320.0, 101_325.0)
    p2 = oil.at(320.0, 5.0e6)
    assert p1 == p2


def test_single_point_enthalpy_reference_is_zero_at_reference_temperature() -> None:
    oil = _oil_provider()
    full = oil.full_at(_OIL_POINT.T, 101_325.0)
    assert full.h == 0.0


def test_single_point_enthalpy_matches_cp_times_delta_t() -> None:
    oil = _oil_provider()
    T1, T2 = 300.0, 340.0
    h1 = oil.full_at(T1, 101_325.0).h
    h2 = oil.full_at(T2, 101_325.0).h
    assert math.isclose(h2 - h1, _OIL_POINT.cp * (T2 - T1), rel_tol=1e-12)


def test_single_point_phase_and_fluid_name() -> None:
    oil = _oil_provider()
    full = oil.full_at(320.0, 101_325.0)
    assert full.phase == "liquid"
    assert full.fluid == "Turbine oil ISO VG46"
    assert full.warnings == []


def test_single_point_rejects_non_positive_temperature_or_pressure() -> None:
    oil = _oil_provider()
    with pytest.raises(ValueError):
        oil.at(0.0, 101_325.0)
    with pytest.raises(ValueError):
        oil.at(-10.0, 101_325.0)
    with pytest.raises(ValueError):
        oil.at(300.0, 0.0)
    with pytest.raises(ValueError):
        oil.at(300.0, -1.0)
    with pytest.raises(ValueError):
        oil.at(float("nan"), 101_325.0)


# ---------------------------------------------------------------------------
# Multi-point provider
# ---------------------------------------------------------------------------

_MULTI_POINTS_SORTED = [
    LiquidPropertyPoint(T=300.0, rho=850.0, cp=1900.0, mu=0.100, k=0.130),
    LiquidPropertyPoint(T=350.0, rho=800.0, cp=2000.0, mu=0.010, k=0.120),
    LiquidPropertyPoint(T=400.0, rho=750.0, cp=2100.0, mu=0.005, k=0.110),
]


def _multi_provider() -> TabulatedLiquidProvider:
    # Deliberately unsorted / arbitrary input order.
    unsorted = [_MULTI_POINTS_SORTED[1], _MULTI_POINTS_SORTED[2], _MULTI_POINTS_SORTED[0]]
    return TabulatedLiquidProvider(name="synthetic", points=unsorted)


def test_multi_point_accepts_unsorted_input_and_sorts_by_temperature() -> None:
    provider = _multi_provider()
    assert [pt.T for pt in provider.points] == [300.0, 350.0, 400.0]


def test_multi_point_rejects_duplicate_temperatures() -> None:
    with pytest.raises(ValueError):
        TabulatedLiquidProvider(
            name="dup",
            points=[
                LiquidPropertyPoint(T=300.0, rho=850.0, cp=1900.0, mu=0.10, k=0.13),
                LiquidPropertyPoint(T=300.0, rho=800.0, cp=2000.0, mu=0.05, k=0.12),
            ],
        )


def test_multi_point_reproduces_exact_knot_values() -> None:
    provider = _multi_provider()
    for point in _MULTI_POINTS_SORTED:
        props = provider.at(point.T, 101_325.0)
        assert props.rho == point.rho
        assert props.cp == point.cp
        assert props.mu == point.mu
        assert props.k == point.k


def test_multi_point_linear_interpolation_of_rho_cp_k() -> None:
    provider = _multi_provider()
    T_lo, T_hi = _MULTI_POINTS_SORTED[0].T, _MULTI_POINTS_SORTED[1].T
    T_mid = 0.5 * (T_lo + T_hi)
    props = provider.at(T_mid, 101_325.0)

    frac = 0.5
    lo, hi = _MULTI_POINTS_SORTED[0], _MULTI_POINTS_SORTED[1]
    assert math.isclose(props.rho, lo.rho + (hi.rho - lo.rho) * frac, rel_tol=1e-12)
    assert math.isclose(props.cp, lo.cp + (hi.cp - lo.cp) * frac, rel_tol=1e-12)
    assert math.isclose(props.k, lo.k + (hi.k - lo.k) * frac, rel_tol=1e-12)


def test_multi_point_log_linear_interpolation_of_mu() -> None:
    provider = _multi_provider()
    lo, hi = _MULTI_POINTS_SORTED[0], _MULTI_POINTS_SORTED[1]
    T_quarter = lo.T + 0.25 * (hi.T - lo.T)
    props = provider.at(T_quarter, 101_325.0)

    expected_ln_mu = math.log(lo.mu) + (math.log(hi.mu) - math.log(lo.mu)) * 0.25
    assert math.isclose(props.mu, math.exp(expected_ln_mu), rel_tol=1e-12)

    # Sanity: log-linear differs from plain linear for this mu ratio.
    plain_linear_mu = lo.mu + (hi.mu - lo.mu) * 0.25
    assert not math.isclose(props.mu, plain_linear_mu, rel_tol=1e-6)


def test_multi_point_outside_range_raises_value_error() -> None:
    provider = _multi_provider()
    with pytest.raises(ValueError):
        provider.at(299.999, 101_325.0)
    with pytest.raises(ValueError):
        provider.at(400.001, 101_325.0)
    with pytest.raises(ValueError):
        provider.full_at(200.0, 101_325.0)


def test_multi_point_enthalpy_is_continuous_at_knots() -> None:
    provider = _multi_provider()
    eps = 1.0e-6
    for point in _MULTI_POINTS_SORTED[1:-1]:
        h_at = provider.full_at(point.T, 101_325.0).h
        h_below = provider.full_at(point.T - eps, 101_325.0).h
        h_above = provider.full_at(point.T + eps, 101_325.0).h
        assert math.isclose(h_at, h_below, abs_tol=1e-2)
        assert math.isclose(h_at, h_above, abs_tol=1e-2)


def test_multi_point_enthalpy_reference_is_zero_at_lowest_temperature() -> None:
    provider = _multi_provider()
    assert provider.full_at(_MULTI_POINTS_SORTED[0].T, 101_325.0).h == 0.0


def test_multi_point_enthalpy_matches_integral_of_interpolated_cp() -> None:
    provider = _multi_provider()
    T0 = _MULTI_POINTS_SORTED[0].T

    def cp_interp(T: float) -> float:
        return provider.at(T, 101_325.0).cp

    def numerical_integral(T_target: float, n: int = 20_000) -> float:
        # Composite trapezoidal rule over many steps; cp is piecewise linear,
        # so this converges to the exact analytic integral.
        step = (T_target - T0) / n
        total = 0.0
        T_prev = T0
        cp_prev = cp_interp(T_prev)
        for i in range(1, n + 1):
            T_next = T0 + step * i
            cp_next = cp_interp(T_next)
            total += 0.5 * (cp_prev + cp_next) * step
            T_prev, cp_prev = T_next, cp_next
        return total

    for T_target in (320.0, 350.0, 375.0, 400.0):
        expected = numerical_integral(T_target)
        actual = provider.full_at(T_target, 101_325.0).h
        assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-3)


def test_multi_point_phase_and_fluid_name() -> None:
    provider = _multi_provider()
    full = provider.full_at(350.0, 101_325.0)
    assert full.phase == "liquid"
    assert full.fluid == "synthetic"
    assert full.warnings == []


def test_multi_point_requires_at_least_one_point() -> None:
    with pytest.raises(ValueError):
        TabulatedLiquidProvider(name="empty", points=[])


# ---------------------------------------------------------------------------
# Small synthetic exchanger integration test (public Simulation path)
# ---------------------------------------------------------------------------


def _build_bundle() -> TubeBundle:
    tube = BareTube(
        D_i=25e-3 - 2 * 1.5e-3,
        D_o=25e-3,
        length_total=2.8,
        length_effective=2.8,
        wall_k=50.0,
    )
    return TubeBundle(
        tube=tube,
        n_rows=6,
        n_tubes_per_row=8,
        pitch_transverse=35e-3,
        pitch_longitudinal=35e-3,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement="counterflow",
    )


def test_tabulated_liquid_provider_works_through_public_simulation_path() -> None:
    oil_provider = TabulatedLiquidProvider(
        name="synthetic oil",
        points=[
            LiquidPropertyPoint(T=300.0, rho=850.0, cp=1900.0, mu=0.050, k=0.130),
            LiquidPropertyPoint(T=380.0, rho=800.0, cp=2050.0, mu=0.008, k=0.120),
        ],
    )
    air_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=1.10, mu=2.0e-5, k=0.028, cp=1007.0)
    )

    hx = BareTubeHeatExchanger(_build_bundle())
    sim = hx.simulate(
        HXSideInput(provider=oil_provider, m_dot=1.5, T_in=360.0, p=3.0e5),
        HXSideInput(provider=air_provider, m_dot=4.0, T_in=300.0, p=101_325.0),
    )

    assert sim.converged
    assert sim.q > 0.0
    assert math.isfinite(sim.T_out_inside)
    assert math.isfinite(sim.T_out_outside)
