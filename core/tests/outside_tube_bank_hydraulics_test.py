# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Focused tests for v0.5.5 outside tube-bank hydraulics."""

from __future__ import annotations

from dataclasses import dataclass
import math

from core.heat_transfer.outside_flow import (
    FluidProps,
    calculate_outside_tube_bank_hydraulics,
    outside_flow_from_mass_flow,
    vmax_ratio_min_freeflow,
)
from core.heat_transfer.outside_pressure_drop import EulerResult
from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
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
class FixedEulerProvider:
    eu: float
    basis: str = "complete_bank"
    rows: float | None = None

    def evaluate(self, request):
        return EulerResult(
            Eu=self.eu,
            source="test",
            model="fixed",
            euler_basis=self.basis,
            n_rows_effective=self.rows,
        )


@dataclass(frozen=True)
class LinearOutsideProvider:
    rho_slope: float
    include_enthalpy: bool = True

    def at(self, T: float, p: float) -> FluidTransportProperties:
        return _props(
            1000.0 + self.rho_slope * (T - 300.0),
            1.0e-3 * (1.0 + 0.001 * (T - 300.0)),
        )

    def full_at(self, T: float, p: float):
        props = self.at(T, p)
        return _FullState(transport=props, h=4180.0 * T) if self.include_enthalpy else props

    def temperature_from_h_p(self, h: float, p: float) -> float:
        return h / 4180.0


@dataclass(frozen=True)
class _FullState:
    transport: FluidTransportProperties
    h: float


def _hydraulic(**kwargs):
    defaults = dict(
        m_dot=1.0,
        face_area=1.0,
        tube_outer_diameter=0.02,
        tube_pitch_transverse=0.04,
        tube_pitch_longitudinal=0.04,
        layout="inline",
        n_rows=5,
        n_tubes_per_row=10,
        inlet_props=_props(1000.0),
        temperature_in=300.0,
        temperature_out=300.0,
        pressure=101325.0,
        euler_provider=FixedEulerProvider(2.0),
    )
    defaults.update(kwargs)
    return calculate_outside_tube_bank_hydraulics(**defaults)


def test_existing_providers_are_explicit_complete_bank_contracts() -> None:
    props = FluidProps(rho=2.0, mu=1.0e-3, k=0.6, cp=4180.0)
    for provider, n_rows in (("zukauskas", 5), ("gaddis_gnielinski", 6)):
        old = outside_flow_from_mass_flow(
            m_dot=1.0,
            frontal_area=1.0,
            tube_outer_diameter=0.02,
            tube_pitch_transverse=0.04,
            tube_pitch_longitudinal=0.04,
            layout="inline",
            n_rows=n_rows,
            n_tubes_per_row=10,
            props=props,
            euler_provider=provider,
        )
        assert old[-1].euler_basis == "complete_bank"
        assert old[-1].n_rows_effective == float(n_rows)

        result = _hydraulic(
            inlet_props=FluidTransportProperties(2.0, 1.0e-3, 0.6, 4180.0),
            n_rows=n_rows,
            euler_provider=provider,
        )
        assert result.euler_basis == "complete_bank"
        assert result.n_rows_effective == float(n_rows)
        # The old complete-bank result and the new constant-property Simpson
        # reconstruction must agree; this catches an accidental second row factor.
        assert math.isclose(result.dp_total, old[4], rel_tol=0.0, abs_tol=1.0e-12)


def test_row_count_is_applied_exactly_once_for_per_row_and_complete_bank() -> None:
    props = FluidTransportProperties(2.0, 1.0e-3, 0.6, 4180.0)
    per_row = _hydraulic(inlet_props=props, euler_provider=FixedEulerProvider(2.0, "per_row"))
    complete = _hydraulic(inlet_props=props, euler_provider=FixedEulerProvider(2.0, "complete_bank"))
    # ratio=2, G_ref=2 kg/(m2 s), rho=2 kg/m3, so q_ref=1 Pa.
    assert math.isclose(per_row.dp_drag, 5.0 * 2.0 * 1.0, abs_tol=1.0e-12)
    assert math.isclose(complete.dp_drag, 2.0 * 1.0, abs_tol=1.0e-12)
    assert math.isclose(per_row.dp_drag, 5.0 * complete.dp_drag, abs_tol=1.0e-12)


def test_simpson_reconstruction_and_reference_velocity_consistency() -> None:
    result = _hydraulic(
        provider=LinearOutsideProvider(rho_slope=-0.5),
        inlet_props=None,
        temperature_in=300.0,
        temperature_out=500.0,
    )
    expected_mean = (
        result.inlet.euler_number / result.inlet.props.rho
        + 4.0 * result.midpoint.euler_number / result.midpoint.props.rho
        + result.outlet.euler_number / result.outlet.props.rho
    ) / 6.0
    expected_drag = (
        result.reference_mass_flux**2
        / 2.0
        * expected_mean
    )
    assert math.isclose(result.mean_euler_over_rho, expected_mean, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(result.dp_drag, expected_drag, rel_tol=0.0, abs_tol=1.0e-12)
    assert result.midpoint_method == "enthalpy"
    assert result.midpoint.temperature == 400.0

    ratio = vmax_ratio_min_freeflow(0.02, 0.04, 0.04, "inline")
    assert result.reference_area == result.face_area / ratio
    assert result.reference_mass_flux == result.face_mass_flux * ratio
    for point in (result.inlet, result.midpoint, result.outlet):
        assert math.isclose(
            point.reference_velocity,
            point.reference_mass_flux / point.props.rho,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        assert math.isclose(
            point.reynolds,
            point.reference_mass_flux * 0.02 / point.props.mu,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )


def test_signed_acceleration_uses_face_mass_flux() -> None:
    decreasing = _hydraulic(
        provider=LinearOutsideProvider(rho_slope=-1.0),
        inlet_props=None,
        temperature_in=300.0,
        temperature_out=400.0,
    )
    increasing = _hydraulic(
        provider=LinearOutsideProvider(rho_slope=1.0),
        inlet_props=None,
        temperature_in=300.0,
        temperature_out=400.0,
    )
    expected_decreasing = decreasing.face_mass_flux**2 * (
        1.0 / decreasing.rho_out - 1.0 / decreasing.rho_in
    )
    assert math.isclose(decreasing.dp_acceleration, expected_decreasing, abs_tol=1.0e-12)
    assert decreasing.dp_acceleration > 0.0
    assert increasing.dp_acceleration < 0.0
    assert math.isclose(
        increasing.dp_acceleration,
        increasing.face_mass_flux**2 * (1.0 / increasing.rho_out - 1.0 / increasing.rho_in),
        abs_tol=1.0e-12,
    )


def test_available_outside_providers_use_the_same_universal_path() -> None:
    providers = (
        (DryAirPropertyProvider(prefer_coolprop=False), 300.0, 360.0),
        (MoistAirTransportProvider.from_t_rh(T=300.0, RH=0.4, p=101325.0), 300.0, 320.0),
        (IAPWS97WaterSteamProvider(), 300.0, 320.0),
    )
    for provider, temperature_in, temperature_out in providers:
        result = _hydraulic(
            provider=provider,
            inlet_props=None,
            temperature_in=temperature_in,
            temperature_out=temperature_out,
        )
        assert result.inlet.position == "inlet"
        assert result.midpoint.position == "midpoint"
        assert result.outlet.position == "outlet"
        assert result.dp_total == result.dp_drag + result.dp_acceleration
        assert all(
            point.props.rho > 0.0 and point.props.mu > 0.0
            for point in (result.inlet, result.midpoint, result.outlet)
        )


def test_simulate_and_rate_expose_the_same_outside_nested_result() -> None:
    tube = BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0)
    bundle = TubeBundle(
        tube=tube,
        n_rows=5,
        n_tubes_per_row=8,
        pitch_transverse=0.04,
        pitch_longitudinal=0.04,
        layout="inline",
        n_passes_tube=1,
        flow_arrangement="counterflow",
    )
    hx = BareTubeHeatExchanger(bundle)
    inside_provider = ConstantPropertyProvider(_props(1000.0))
    outside_provider = ConstantPropertyProvider(_props(2.0))
    inside = HXSideInput(provider=inside_provider, m_dot=0.2, T_in=400.0, p=101325.0)
    outside = HXSideInput(provider=outside_provider, m_dot=0.2, T_in=300.0, p=101325.0)

    simulation = hx.simulate(inside, outside, iterate=False)
    nested_simulation = simulation.outside_tube_bank_hydraulic
    assert nested_simulation is simulation.final_result.outside_side_hydraulic.tube_bank
    assert simulation.outside_dp_drag == nested_simulation.dp_drag
    assert simulation.outside_dp_acceleration == nested_simulation.dp_acceleration
    assert simulation.outside_dp_total == nested_simulation.dp_total

    rating = hx.rate(
        BalanceSideSpec(provider=inside_provider, m_dot=0.2, p=101325.0, T_in=400.0, T_out=350.0),
        BalanceSideSpec(provider=outside_provider, m_dot=0.2, p=101325.0, T_in=300.0, T_out=350.0),
        Q=1000.0,
    )
    nested_rating = rating.outside_tube_bank_hydraulic
    assert nested_rating is rating.final_result.outside_side_hydraulic.tube_bank
    assert rating.outside_dp_drag == nested_rating.dp_drag
    assert rating.outside_dp_acceleration == nested_rating.dp_acceleration
    assert rating.outside_dp_total == nested_rating.dp_total
def main() -> None:
    test_existing_providers_are_explicit_complete_bank_contracts()
    test_row_count_is_applied_exactly_once_for_per_row_and_complete_bank()
    test_simpson_reconstruction_and_reference_velocity_consistency()
    test_signed_acceleration_uses_face_mass_flux()
    test_available_outside_providers_use_the_same_universal_path()
    test_simulate_and_rate_expose_the_same_outside_nested_result()
    print("ALL OUTSIDE TUBE-BANK HYDRAULICS TESTS PASSED")


if __name__ == "__main__":
    main()
