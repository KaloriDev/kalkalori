# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Focused tests for v0.5.6 tube surface roughness and rough-pipe friction.

Covers ``BareTube.roughness_inner``/``roughness_outer`` geometry, the
Colebrook-White rough-tube Darcy friction factor in
``core.pressure_drop.straight_sections``, its wiring into
``calculate_tube_bundle_hydraulics``, and confirmation that roughness
affects only distributed straight-tube friction (not acceleration,
tube-sheet entrance/exit, thermal, or outside-side results).
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.heat_transfer.streams import SensibleHeatStream
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
from core.pressure_drop.straight_sections import (
    COLEBROOK_MAX_ITERATIONS,
    COLEBROOK_TOLERANCE,
    darcy_friction_factor,
    darcy_friction_factor_method,
    friction_factor_smooth,
)
from core.properties.common import FluidTransportProperties
from core.properties.dry_air import DryAirPropertyProvider
from core.properties.fluids import ConstantPropertyProvider
from core.properties.moist_air_transport import MoistAirTransportProvider
from core.properties.water import IAPWS97WaterSteamProvider


def _props(rho: float, mu: float = 1.0e-3, k: float = 0.6, cp: float = 4180.0) -> FluidTransportProperties:
    return FluidTransportProperties(rho=rho, mu=mu, k=k, cp=cp)


def _hydraulic(**kwargs):
    defaults = dict(
        m_dot=1.0,
        flow_area_per_pass=0.01,
        hydraulic_diameter=0.1,
        hydraulic_length_total=2.0,
        n_tube_passes=1,
    )
    defaults.update(kwargs)
    return calculate_tube_bundle_hydraulics(**defaults)


def _bundle(roughness_inner=None, roughness_outer=None, n_passes_tube: int = 1) -> TubeBundle:
    tube = BareTube(
        D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0,
        roughness_inner=roughness_inner, roughness_outer=roughness_outer,
    )
    return TubeBundle(
        tube=tube, n_rows=4, n_tubes_per_row=6, pitch_transverse=0.04, pitch_longitudinal=0.04,
        layout="staggered", n_passes_tube=n_passes_tube, flow_arrangement="crossflow",
    )


def _solve(bundle: TubeBundle, tube_side_mu: float = 1.0e-3):
    # tube_side_mu is kept low enough (default) to land near the laminar/
    # turbulent transition for some tests, and can be lowered further by
    # callers that need a solidly turbulent Re (>> 2300) so that the
    # Colebrook-White/Petukhov smooth-limit numerical discrepancy near the
    # transition does not obscure the roughness effect being tested.
    hx = BareTubeHeatExchanger(bundle)
    return hx, hx.solve(
        hot_stream=SensibleHeatStream(C=1000.0, T_in=350.0),
        cold_stream=SensibleHeatStream(C=500.0, T_in=300.0),
        m_dot_tube_side=1.0, tube_side_props=_props(1000.0, mu=tube_side_mu),
        tube_side_temperature_in=350.0, tube_side_temperature_out=340.0, tube_side_pressure=101325.0,
        m_dot_outside=2.0, outside_props=_props(1.2, mu=1.8e-5, k=0.026, cp=1006.0),
        outside_temperature_in=300.0, outside_temperature_out=310.0, outside_pressure=101325.0,
    )


# ---------------------------------------------------------------------------
# 9.1 BareTube backward compatibility
# ---------------------------------------------------------------------------

def test_bare_tube_construction_without_roughness_still_works() -> None:
    tube = BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0)
    assert tube.roughness_inner is None
    assert tube.roughness_outer is None
    assert tube.relative_roughness_inner is None
    assert tube.relative_roughness_outer is None


# ---------------------------------------------------------------------------
# 9.2 Geometry validation
# ---------------------------------------------------------------------------

def test_roughness_validation_rejects_invalid_values() -> None:
    for bad in (-0.001, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0, roughness_inner=bad)
        with pytest.raises(ValueError):
            BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0, roughness_outer=bad)


def test_roughness_validation_accepts_valid_values() -> None:
    for good in (None, 0.0, 1.0e-6, 2.0e-4):
        tube = BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0, roughness_inner=good, roughness_outer=good)
        assert tube.roughness_inner == good
        assert tube.roughness_outer == good


# ---------------------------------------------------------------------------
# 9.3 Relative roughness
# ---------------------------------------------------------------------------

def test_relative_roughness_properties() -> None:
    tube = BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0, roughness_inner=2.0e-4, roughness_outer=1.0e-4)
    assert math.isclose(tube.relative_roughness_inner, 2.0e-4 / 0.02, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(tube.relative_roughness_outer, 1.0e-4 / 0.024, rel_tol=0.0, abs_tol=1.0e-15)


# ---------------------------------------------------------------------------
# 9.4 Smooth backward compatibility
# ---------------------------------------------------------------------------

def test_omitted_roughness_matches_pre_change_smooth_result() -> None:
    result = _hydraulic(provider=ConstantPropertyProvider(_props(950.0)), temperature_in=300.0, temperature_out=400.0, pressure=101325.0)
    # Reconstruct the pre-roughness v0.5.6 smooth calculation exactly.
    G = result.mass_flux
    for point in (result.inlet, result.midpoint, result.outlet):
        expected_f = friction_factor_smooth(point.reynolds)
        assert point.friction_factor == expected_f
        assert point.friction_factor_method == "petukhov_smooth"
    assert result.roughness_inner is None
    assert result.relative_roughness_inner is None


def test_zero_roughness_matches_none() -> None:
    common = dict(provider=ConstantPropertyProvider(_props(950.0)), temperature_in=300.0, temperature_out=400.0, pressure=101325.0)
    result_none = _hydraulic(**common)
    result_zero = _hydraulic(roughness_inner=0.0, **common)

    assert result_zero.friction_factor_in == result_none.friction_factor_in
    assert result_zero.friction_factor_mid == result_none.friction_factor_mid
    assert result_zero.friction_factor_out == result_none.friction_factor_out
    assert result_zero.mean_f_over_rho == result_none.mean_f_over_rho
    assert result_zero.dp_straight_tube_friction == result_none.dp_straight_tube_friction
    assert result_zero.dp_tube_bundle == result_none.dp_tube_bundle
    assert result_zero.relative_roughness_inner == 0.0
    assert result_none.relative_roughness_inner is None


# ---------------------------------------------------------------------------
# 9.5 Laminar flow
# ---------------------------------------------------------------------------

def test_laminar_friction_factor_independent_of_roughness() -> None:
    # High viscosity keeps Re well below 2300 at this mass flux.
    for roughness in (None, 0.0, 5.0e-4):
        result = _hydraulic(roughness_inner=roughness, inlet_props=_props(1000.0, mu=1.0))
        assert result.inlet.reynolds < 2300.0
        assert result.friction_factor_in == 64.0 / result.inlet.reynolds
        assert result.friction_factor_method_in == "laminar_64_over_re"


# ---------------------------------------------------------------------------
# 9.6 Colebrook reconstruction
# ---------------------------------------------------------------------------

def test_colebrook_white_residual_is_near_zero() -> None:
    cases = [
        (5000.0, 1.0e-4), (1.0e4, 5.0e-4), (1.0e5, 1.0e-3),
        (1.0e6, 2.0e-2), (5.0e4, 1.0e-5), (2.0e5, 1.0e-2), (3.0e3, 1.0e-3),
    ]
    for Re, relative_roughness in cases:
        f = darcy_friction_factor(Re, relative_roughness)
        assert darcy_friction_factor_method(Re, relative_roughness) == "colebrook_white"
        residual = 1.0 / math.sqrt(f) + 2.0 * math.log10(relative_roughness / 3.7 + 2.51 / (Re * math.sqrt(f)))
        assert abs(residual) < 1.0e-9, (Re, relative_roughness, f, residual)


def test_colebrook_documented_convergence_settings() -> None:
    assert COLEBROOK_TOLERANCE > 0.0
    assert COLEBROOK_MAX_ITERATIONS > 0


# ---------------------------------------------------------------------------
# 9.7 Monotonic effect
# ---------------------------------------------------------------------------

def test_rough_friction_factor_exceeds_smooth_at_constant_reynolds() -> None:
    Re = 1.0e5
    f_smooth = darcy_friction_factor(Re, None)
    f_rough = darcy_friction_factor(Re, 1.0e-3)
    assert f_rough > f_smooth


def test_increasing_roughness_increases_pressure_drop() -> None:
    bundle_smooth = _bundle(roughness_inner=None)
    bundle_light = _bundle(roughness_inner=5.0e-5)
    bundle_heavy = _bundle(roughness_inner=2.0e-4)

    # A solidly turbulent Re (well above the 2300 transition) is used here:
    # Colebrook-White and the Petukhov smooth-tube approximation are two
    # independent correlations that only agree closely away from the
    # transition region, so comparing them near Re~2300 is not a reliable
    # test of the roughness effect itself.
    tube_side_mu = 2.0e-4
    _, result_smooth = _solve(bundle_smooth, tube_side_mu=tube_side_mu)
    _, result_light = _solve(bundle_light, tube_side_mu=tube_side_mu)
    _, result_heavy = _solve(bundle_heavy, tube_side_mu=tube_side_mu)
    assert result_smooth.tube_side_thermal.Re > 1.0e4

    assert result_light.inside_dp_straight_tube_friction > result_smooth.inside_dp_straight_tube_friction
    assert result_heavy.inside_dp_straight_tube_friction > result_light.inside_dp_straight_tube_friction

    assert result_light.inside_dp_tube_bundle > result_smooth.inside_dp_tube_bundle
    assert result_heavy.inside_dp_tube_bundle > result_light.inside_dp_tube_bundle

    assert result_light.inside_dp_total > result_smooth.inside_dp_total
    assert result_heavy.inside_dp_total > result_light.inside_dp_total


# ---------------------------------------------------------------------------
# 9.8 Only friction changes
# ---------------------------------------------------------------------------

def test_only_friction_derived_quantities_change_with_roughness() -> None:
    _, result_smooth = _solve(_bundle(roughness_inner=None))
    _, result_rough = _solve(_bundle(roughness_inner=2.0e-4))

    tb_smooth = result_smooth.tube_side_hydraulic.tube_bundle
    tb_rough = result_rough.tube_side_hydraulic.tube_bundle

    for label, point_smooth, point_rough in (
        ("inlet", tb_smooth.inlet, tb_rough.inlet),
        ("midpoint", tb_smooth.midpoint, tb_rough.midpoint),
        ("outlet", tb_smooth.outlet, tb_rough.outlet),
    ):
        assert point_smooth.velocity == point_rough.velocity, label
        assert point_smooth.reynolds == point_rough.reynolds, label
        assert point_smooth.dynamic_pressure == point_rough.dynamic_pressure, label

    assert tb_smooth.dp_straight_tube_acceleration == tb_rough.dp_straight_tube_acceleration
    assert tb_smooth.dp_tube_entrances == tb_rough.dp_tube_entrances
    assert tb_smooth.dp_tube_exits == tb_rough.dp_tube_exits
    assert [r.pressure_drop for r in tb_smooth.entrance_results] == [r.pressure_drop for r in tb_rough.entrance_results]
    assert [r.pressure_drop for r in tb_smooth.exit_results] == [r.pressure_drop for r in tb_rough.exit_results]

    # Friction-derived quantities must differ.
    assert tb_smooth.friction_factor_in != tb_rough.friction_factor_in
    assert tb_smooth.dp_straight_tube_friction != tb_rough.dp_straight_tube_friction
    assert tb_smooth.dp_tube_bundle != tb_rough.dp_tube_bundle


# ---------------------------------------------------------------------------
# 9.9 Outer roughness has no current effect
# ---------------------------------------------------------------------------

def test_outer_roughness_has_no_numerical_effect() -> None:
    _, result_without = _solve(_bundle(roughness_outer=None))
    _, result_with = _solve(_bundle(roughness_outer=3.0e-4))

    assert result_without.Q == result_with.Q
    assert result_without.UA == result_with.UA
    assert result_without.tube_side_thermal.alfa == result_with.tube_side_thermal.alfa
    assert result_without.outside_side_thermal.alfa == result_with.outside_side_thermal.alfa
    assert result_without.inside_dp_total == result_with.inside_dp_total
    assert result_without.outside_dp_total == result_with.outside_dp_total


# ---------------------------------------------------------------------------
# 9.10 Variable-property integration
# ---------------------------------------------------------------------------

def test_variable_property_roughness_reconstruction() -> None:
    # A real temperature-varying provider (not ConstantPropertyProvider) is
    # required so that inlet/midpoint/outlet each have distinct local
    # Reynolds numbers, per the "own local Re" requirement below.
    result = _hydraulic(
        roughness_inner=1.5e-4,
        provider=DryAirPropertyProvider(prefer_coolprop=False),
        temperature_in=300.0, temperature_out=500.0, pressure=101325.0,
    )
    relative_roughness = 1.5e-4 / result.hydraulic_diameter
    assert math.isclose(result.relative_roughness_inner, relative_roughness, rel_tol=0.0, abs_tol=1.0e-15)

    expected_f = {}
    for label, point in (("in", result.inlet), ("mid", result.midpoint), ("out", result.outlet)):
        expected = darcy_friction_factor(point.reynolds, relative_roughness)
        expected_f[label] = expected
        assert math.isclose(point.friction_factor, expected, rel_tol=0.0, abs_tol=1.0e-12)

    expected_mean_f_over_rho = (
        result.inlet.friction_factor / result.inlet.props.rho
        + 4.0 * result.midpoint.friction_factor / result.midpoint.props.rho
        + result.outlet.friction_factor / result.outlet.props.rho
    ) / 6.0
    assert math.isclose(result.mean_f_over_rho, expected_mean_f_over_rho, rel_tol=0.0, abs_tol=1.0e-15)

    expected_dp_friction = (
        (result.hydraulic_length_total / result.hydraulic_diameter)
        * (result.mass_flux**2 / 2.0) * expected_mean_f_over_rho
    )
    assert math.isclose(result.dp_straight_tube_friction, expected_dp_friction, rel_tol=0.0, abs_tol=1.0e-9)

    # Each point retains its own local Re, distinct from the others.
    assert result.inlet.reynolds != result.midpoint.reynolds != result.outlet.reynolds


# ---------------------------------------------------------------------------
# 9.11 Standard solver integration
# ---------------------------------------------------------------------------

def test_simulate_and_rate_use_bare_tube_roughness_inner() -> None:
    bundle = _bundle(roughness_inner=2.0e-4)
    hx = BareTubeHeatExchanger(bundle)
    provider = ConstantPropertyProvider(_props(1000.0))
    outside_provider = ConstantPropertyProvider(_props(1.2, mu=1.8e-5, k=0.026, cp=1006.0))

    inside = HXSideInput(provider=provider, m_dot=1.0, T_in=350.0, p=101325.0)
    outside = HXSideInput(provider=outside_provider, m_dot=2.0, T_in=300.0, p=101325.0)
    simulated = hx.simulate(inside, outside, max_iter=5)

    tube_bundle = simulated.final_result.tube_side_hydraulic.tube_bundle
    assert tube_bundle.roughness_inner == 2.0e-4
    assert tube_bundle.friction_factor_method_in == "colebrook_white"
    assert simulated.inside_dp_tube_bundle == simulated.inside_dp_straight_tubes + simulated.inside_dp_tube_entrances + simulated.inside_dp_tube_exits

    rating = hx.rate(
        BalanceSideSpec(provider=provider, m_dot=1.0, p=101325.0, T_in=350.0, T_out=340.0),
        BalanceSideSpec(provider=outside_provider, m_dot=2.0, p=101325.0, T_in=300.0, T_out=310.0),
        Q=1000.0,
    )
    rating_tube_bundle = rating.final_result.tube_side_hydraulic.tube_bundle
    assert rating_tube_bundle.roughness_inner == 2.0e-4
    assert rating_tube_bundle.friction_factor_method_in == "colebrook_white"


# ---------------------------------------------------------------------------
# 9.12 Thermal regression
# ---------------------------------------------------------------------------

def test_thermal_results_identical_for_smooth_and_rough_tube() -> None:
    hx_smooth, result_smooth = _solve(_bundle(roughness_inner=None))
    hx_rough, result_rough = _solve(_bundle(roughness_inner=2.0e-4))

    assert result_smooth.Q == result_rough.Q
    assert result_smooth.T_hot_out == result_rough.T_hot_out
    assert result_smooth.T_cold_out == result_rough.T_cold_out
    assert result_smooth.tube_side_thermal.alfa == result_rough.tube_side_thermal.alfa
    assert result_smooth.outside_side_thermal.alfa == result_rough.outside_side_thermal.alfa
    assert result_smooth.UA == result_rough.UA

    provider = ConstantPropertyProvider(_props(1000.0))
    outside_provider = ConstantPropertyProvider(_props(1.2, mu=1.8e-5, k=0.026, cp=1006.0))
    inside = HXSideInput(provider=provider, m_dot=1.0, T_in=350.0, p=101325.0)
    outside = HXSideInput(provider=outside_provider, m_dot=2.0, T_in=300.0, p=101325.0)
    sim_smooth = hx_smooth.simulate(inside, outside, max_iter=10)
    sim_rough = hx_rough.simulate(inside, outside, max_iter=10)

    assert math.isclose(sim_smooth.q, sim_rough.q, rel_tol=1.0e-9)
    assert math.isclose(sim_smooth.T_out_inside, sim_rough.T_out_inside, rel_tol=0.0, abs_tol=1.0e-9)
    assert math.isclose(sim_smooth.T_out_outside, sim_rough.T_out_outside, rel_tol=0.0, abs_tol=1.0e-9)
    assert math.isclose(sim_smooth.inside_alfa_mean, sim_rough.inside_alfa_mean, rel_tol=1.0e-9)
    assert math.isclose(sim_smooth.outside_alfa_mean, sim_rough.outside_alfa_mean, rel_tol=1.0e-9)
    assert math.isclose(sim_smooth.U_mean, sim_rough.U_mean, rel_tol=1.0e-9)
    assert math.isclose(sim_smooth.UA, sim_rough.UA, rel_tol=1.0e-9)
    if sim_smooth.thermal_state is not None and sim_rough.thermal_state is not None:
        assert math.isclose(
            sim_smooth.thermal_state.inside_wall_temperature,
            sim_rough.thermal_state.inside_wall_temperature,
            rel_tol=1.0e-9,
        )
        assert math.isclose(
            sim_smooth.thermal_state.outside_wall_temperature,
            sim_rough.thermal_state.outside_wall_temperature,
            rel_tol=1.0e-9,
        )


# ---------------------------------------------------------------------------
# 9.13 Outside regression
# ---------------------------------------------------------------------------

def test_outside_tube_bank_result_unchanged_by_inner_roughness() -> None:
    _, result_smooth = _solve(_bundle(roughness_inner=None))
    _, result_rough = _solve(_bundle(roughness_inner=2.0e-4))

    tank_smooth = result_smooth.outside_side_hydraulic.tube_bank
    tank_rough = result_rough.outside_side_hydraulic.tube_bank
    assert tank_smooth.dp_drag == tank_rough.dp_drag
    assert tank_smooth.dp_acceleration == tank_rough.dp_acceleration
    assert tank_smooth.dp_total == tank_rough.dp_total
    assert tank_smooth.midpoint.reynolds == tank_rough.midpoint.reynolds
    assert result_smooth.outside_dp_total == result_rough.outside_dp_total


# ---------------------------------------------------------------------------
# 9.8b Universal fluids exercise the same roughness path (sanity, mirrors
# the entrance/exit universal-fluid coverage from the previous commit)
# ---------------------------------------------------------------------------

def test_universal_fluids_use_the_same_roughness_path() -> None:
    cases = (
        (DryAirPropertyProvider(prefer_coolprop=False), 300.0, 400.0),
        (MoistAirTransportProvider.from_t_rh(T=300.0, RH=0.4, p=101325.0), 300.0, 320.0),
        (IAPWS97WaterSteamProvider(), 300.0, 320.0),
        (IAPWS97WaterSteamProvider(), 320.0, 300.0),
    )
    for provider, T_in, T_out in cases:
        result = _hydraulic(roughness_inner=2.0e-4, provider=provider, temperature_in=T_in, temperature_out=T_out, pressure=101325.0)
        relative_roughness = 2.0e-4 / result.hydraulic_diameter
        for point in (result.inlet, result.midpoint, result.outlet):
            expected = darcy_friction_factor(point.reynolds, relative_roughness)
            assert math.isclose(point.friction_factor, expected, rel_tol=0.0, abs_tol=1.0e-12)
