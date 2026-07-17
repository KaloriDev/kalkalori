# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Focused tests for v0.5.5 universal tube-bundle hydraulics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from unittest import mock

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.pressure_drop.internal_pressure_drop import (
    calculate_tube_bundle_hydraulics,
    friction_factor_smooth,
)
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.dry_air import DryAirPropertyProvider
from core.properties.moist_air_transport import MoistAirTransportProvider
from core.properties.water import IAPWS97WaterSteamProvider


def _props(rho: float, mu: float = 1.0e-3) -> FluidTransportProperties:
    return FluidTransportProperties(rho=rho, mu=mu, k=0.6, cp=4180.0)


@dataclass(frozen=True)
class LinearProvider:
    rho_slope: float
    include_enthalpy: bool = False
    inverse: bool = False

    def at(self, T: float, p: float) -> FluidTransportProperties:
        return _props(1000.0 + self.rho_slope * (T - 300.0), 1.0e-3 * (1.0 + 0.001 * (T - 300.0)))

    def full_at(self, T: float, p: float):
        props = self.at(T, p)
        if not self.include_enthalpy:
            return props
        return _FullState(transport=props, h=4180.0 * T)

    def temperature_from_h_p(self, h: float, p: float) -> float:
        if not self.inverse:
            raise AttributeError("inversion intentionally unavailable")
        return h / 4180.0


@dataclass(frozen=True)
class _FullState:
    transport: FluidTransportProperties
    h: float


def _hydraulic(**kwargs):
    defaults = dict(
        m_dot=1.0,
        flow_area_per_pass=0.01,
        hydraulic_diameter=0.1,
        hydraulic_length_total=2.0,
    )
    defaults.update(kwargs)
    return calculate_tube_bundle_hydraulics(**defaults)


def test_constant_property_regression() -> None:
    result = _hydraulic(inlet_props=_props(1000.0))
    G = result.mass_flux
    f = friction_factor_smooth(G * result.hydraulic_diameter / 1.0e-3)
    expected = f * (result.hydraulic_length_total / result.hydraulic_diameter) * G**2 / (2.0 * 1000.0)
    assert result.dp_acceleration == 0.0
    assert math.isclose(result.dp_friction, expected, rel_tol=0.0, abs_tol=1.0e-12)
    assert result.dp_tube_bundle == result.dp_friction


def test_signed_acceleration_for_both_density_directions() -> None:
    decreasing = _hydraulic(provider=LinearProvider(rho_slope=-1.0), temperature_in=300.0, temperature_out=400.0, pressure=101325.0)
    increasing = _hydraulic(provider=LinearProvider(rho_slope=1.0), temperature_in=300.0, temperature_out=400.0, pressure=101325.0)
    assert decreasing.outlet.props.rho < decreasing.inlet.props.rho
    assert decreasing.dp_acceleration > 0.0
    assert decreasing.dp_tube_bundle > decreasing.dp_friction
    assert increasing.outlet.props.rho > increasing.inlet.props.rho
    assert increasing.dp_acceleration < 0.0
    assert increasing.dp_tube_bundle < increasing.dp_friction

    recovery = _hydraulic(
        provider=LinearProvider(rho_slope=10.0),
        temperature_in=300.0, temperature_out=400.0,
        hydraulic_length_total=1.0e-3,
        pressure=101325.0,
    )
    assert recovery.dp_tube_bundle < 0.0
    assert any(
        warning.code == "tube_bundle_hydraulics_negative_total_pressure_change"
        for warning in recovery.warnings
    )


def test_simpson_and_reconstruction_in_stored_states() -> None:
    result = _hydraulic(provider=LinearProvider(rho_slope=-0.5), temperature_in=300.0, temperature_out=500.0, pressure=101325.0)
    expected_mean = (
        result.inlet.friction_factor / result.inlet.props.rho
        + 4.0 * result.midpoint.friction_factor / result.midpoint.props.rho
        + result.outlet.friction_factor / result.outlet.props.rho
    ) / 6.0
    expected_dp = (result.hydraulic_length_total / result.hydraulic_diameter) * (result.mass_flux**2 / 2.0) * expected_mean
    assert math.isclose(result.mean_f_over_rho, expected_mean, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(result.dp_friction, expected_dp, rel_tol=0.0, abs_tol=1.0e-12)
    for point in (result.inlet, result.midpoint, result.outlet):
        assert math.isclose(point.props.rho * point.velocity, point.mass_flux, rel_tol=1.0e-12)
        assert math.isclose(point.reynolds, point.mass_flux * result.hydraulic_diameter / point.props.mu, rel_tol=1.0e-12)


def test_midpoint_enthalpy_and_arithmetic_fallback() -> None:
    enthalpy_result = _hydraulic(
        provider=LinearProvider(rho_slope=-0.5, include_enthalpy=True, inverse=True),
        temperature_in=300.0, temperature_out=500.0, pressure=101325.0,
    )
    fallback_result = _hydraulic(
        provider=LinearProvider(rho_slope=-0.5, include_enthalpy=False),
        temperature_in=300.0, temperature_out=500.0, pressure=101325.0,
    )
    assert enthalpy_result.midpoint_method == "enthalpy"
    assert enthalpy_result.midpoint.temperature == 400.0
    assert fallback_result.midpoint_method == "arithmetic_temperature"
    assert any(w.code == "tube_bundle_hydraulics_midpoint_temperature_fallback" for w in fallback_result.warnings)


def test_available_property_providers_use_the_same_hydraulic_structure() -> None:
    providers = (
        (DryAirPropertyProvider(prefer_coolprop=False), 300.0, 400.0),
        (MoistAirTransportProvider.from_t_rh(T=300.0, RH=0.4, p=101325.0), 300.0, 320.0),
        (IAPWS97WaterSteamProvider(), 300.0, 320.0),
    )
    for provider, T_in, T_out in providers:
        result = _hydraulic(
            provider=provider,
            temperature_in=T_in,
            temperature_out=T_out,
            pressure=101325.0,
        )
        assert result.inlet.position == "inlet"
        assert result.midpoint.position == "midpoint"
        assert result.outlet.position == "outlet"
        assert result.dp_tube_bundle == result.dp_friction + result.dp_acceleration
        assert all(point.props.rho > 0.0 and point.props.mu > 0.0 for point in (result.inlet, result.midpoint, result.outlet))


def test_multi_pass_geometry_and_no_local_losses() -> None:
    tube = BareTube(D_i=0.02, D_o=0.024, length_total=3.0, length_effective=2.5, wall_k=20.0)
    bundle_one = TubeBundle(tube=tube, n_rows=4, n_tubes_per_row=8, pitch_transverse=0.03, pitch_longitudinal=0.03, layout="inline", n_passes_tube=1, flow_arrangement="counterflow")
    bundle_two = TubeBundle(tube=tube, n_rows=4, n_tubes_per_row=8, pitch_transverse=0.03, pitch_longitudinal=0.03, layout="inline", n_passes_tube=2, flow_arrangement="counterflow")
    props = _props(1000.0)
    one = calculate_tube_bundle_hydraulics(m_dot=1.0, flow_area_per_pass=bundle_one.internal_flow_area_per_pass, hydraulic_diameter=bundle_one.internal_hydraulic_diameter, hydraulic_length_total=bundle_one.internal_length_total, inlet_props=props)
    two = calculate_tube_bundle_hydraulics(m_dot=1.0, flow_area_per_pass=bundle_two.internal_flow_area_per_pass, hydraulic_diameter=bundle_two.internal_hydraulic_diameter, hydraulic_length_total=bundle_two.internal_length_total, inlet_props=props)
    assert one.flow_area_per_pass == 32 * tube.flow_area
    assert two.flow_area_per_pass == 16 * tube.flow_area
    assert two.hydraulic_length_total == 2.0 * one.hydraulic_length_total
    assert one.mass_flux == 1.0 / one.flow_area_per_pass
    assert two.mass_flux == 1.0 / two.flow_area_per_pass

    hx = BareTubeHeatExchanger(bundle_two)
    result = hx.solve(
        hot_stream=_stream(500.0, 1000.0), cold_stream=_stream(300.0, 1000.0),
        m_dot_tube_side=1.0,
        tube_side_props=type("P", (), {"rho": 1000.0, "mu": 1.0e-3, "k": 0.6, "cp": 4180.0})(),
        m_dot_outside=1.0,
        outside_props=type("P", (), {"rho": 1.0, "mu": 1.0e-5, "k": 0.03, "cp": 1000.0})(),
        K_inlet=999.0, K_outlet=999.0, K_turn=999.0, alfa_o=100.0,
    )
    assert result.tube_side_hydraulic.dp_inlet == 0.0
    assert result.tube_side_hydraulic.dp_outlet == 0.0
    assert result.tube_side_hydraulic.dp_turns == 0.0


def _stream(T: float, C: float):
    from core.heat_transfer.streams import SensibleHeatStream
    return SensibleHeatStream(C=C, T_in=T)


def test_simulate_and_rate_expose_same_nested_result_shape() -> None:
    tube = BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0)
    bundle = TubeBundle(tube=tube, n_rows=4, n_tubes_per_row=8, pitch_transverse=0.03, pitch_longitudinal=0.03, layout="inline", n_passes_tube=1, flow_arrangement="counterflow")
    provider = ConstantPropertyProvider(_props(1000.0))
    hx = BareTubeHeatExchanger(bundle)
    inside = HXSideInput(provider=provider, m_dot=0.2, T_in=300.0, p=101325.0)
    outside = HXSideInput(provider=ConstantPropertyProvider(_props(1.0, 1.0e-5)), m_dot=0.2, T_in=500.0, p=101325.0)
    with mock.patch("core.models.bare_tube.calculate_tube_bundle_hydraulics", wraps=calculate_tube_bundle_hydraulics) as simulate_spy:
        simulated = hx.simulate(inside, outside, max_iter=3)
    assert simulate_spy.called
    assert simulated.final_result.inside_dp_tube_bundle == simulated.final_result.tube_side_hydraulic.tube_bundle.dp_tube_bundle

    rating = hx.rate(
        BalanceSideSpec(provider=provider, m_dot=0.2, p=101325.0, T_in=300.0, T_out=350.0),
        BalanceSideSpec(provider=outside.provider, m_dot=0.2, p=101325.0, T_in=500.0, T_out=450.0),
        Q=1000.0,
    )
    assert rating.final_result.inside_dp_friction == rating.tube_side_hydraulic.dp_friction
    assert rating.final_result.inside_dp_acceleration == rating.tube_side_hydraulic.dp_acceleration


def main() -> None:
    test_constant_property_regression()
    test_signed_acceleration_for_both_density_directions()
    test_simpson_and_reconstruction_in_stored_states()
    test_midpoint_enthalpy_and_arithmetic_fallback()
    test_multi_pass_geometry_and_no_local_losses()
    test_simulate_and_rate_expose_same_nested_result_shape()
    print("ALL TUBE-BUNDLE HYDRAULICS TESTS PASSED")


if __name__ == "__main__":
    main()
