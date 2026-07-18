# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Focused tests for v0.5.6 tube-bundle entrance and exit pressure losses.

Covers straight-tube and U-tube entrance/exit application rules,
pass-boundary hydraulic states, constant- and variable-property
reconstruction, independence from parallel-tube count, universal-fluid
coverage, result aggregation, and standard-solver (simulate/rate)
integration. See ``core.pressure_drop.screens`` for the entrance/exit
correlations themselves and ``core.pressure_drop.internal_pressure_drop``
for the tube-bundle assembly.
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle, TubePathType
from core.geometry.tube import BareTube
from core.heat_transfer.streams import SensibleHeatStream
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
from core.pressure_drop.screens import (
    TubeSheetEntranceType,
    TubeSheetExitType,
    tube_sheet_entrance_loss_coefficient,
    tube_sheet_exit_loss_coefficient,
)
from core.properties.common import FluidTransportProperties
from core.properties.dry_air import DryAirPropertyProvider
from core.properties.fluids import ConstantPropertyProvider
from core.properties.moist_air_transport import MoistAirTransportProvider
from core.properties.water import IAPWS97WaterSteamProvider

K_ENTRANCE = tube_sheet_entrance_loss_coefficient(TubeSheetEntranceType.SHARP_EDGED)
K_EXIT = tube_sheet_exit_loss_coefficient(TubeSheetExitType.NORMAL)


def _props(rho: float, mu: float = 1.0e-3, k: float = 0.6, cp: float = 4180.0) -> FluidTransportProperties:
    return FluidTransportProperties(rho=rho, mu=mu, k=k, cp=cp)


def _hydraulic(**kwargs):
    defaults = dict(
        m_dot=1.0,
        flow_area_per_pass=0.01,
        hydraulic_diameter=0.1,
        hydraulic_length_total=2.0,
    )
    defaults.update(kwargs)
    return calculate_tube_bundle_hydraulics(**defaults)


def _dynamic_pressure(mass_flux: float, rho: float) -> float:
    return mass_flux**2 / (2.0 * rho)


# ---------------------------------------------------------------------------
# 16.1 Straight tube, one pass
# ---------------------------------------------------------------------------

def test_straight_one_pass_entrance_and_exit() -> None:
    result = _hydraulic(n_tube_passes=1, inlet_props=_props(1000.0))
    assert result.tube_path_type == TubePathType.STRAIGHT
    assert result.entrance_count == 1
    assert result.exit_count == 1
    assert len(result.pass_boundary_states) == 2

    q_in = result.pass_boundary_states[0].dynamic_pressure
    q_out = result.pass_boundary_states[1].dynamic_pressure
    assert math.isclose(result.dp_tube_entrances, K_ENTRANCE * q_in, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(result.dp_tube_exits, K_EXIT * q_out, rel_tol=0.0, abs_tol=1.0e-12)


# ---------------------------------------------------------------------------
# 16.2 Straight tube, two passes
# ---------------------------------------------------------------------------

def test_straight_two_passes_boundary_reuse() -> None:
    result = _hydraulic(
        n_tube_passes=2,
        provider=ConstantPropertyProvider(_props(950.0)),
        temperature_in=300.0, temperature_out=400.0, pressure=101325.0,
    )
    assert result.entrance_count == 2
    assert result.exit_count == 2
    assert len(result.pass_boundary_states) == 3

    entrance_1, entrance_2 = result.entrance_results
    exit_1, exit_2 = result.exit_results

    assert entrance_1.pass_index == 1 and entrance_1.boundary_index == 0
    assert exit_1.pass_index == 1 and exit_1.boundary_index == 1
    assert entrance_2.pass_index == 2 and entrance_2.boundary_index == 1
    assert exit_2.pass_index == 2 and exit_2.boundary_index == 2

    # Boundary 1 contributes twice: once as pass 1's exit, once as pass 2's
    # entrance -- two distinct sequential losses at the same physical state.
    assert exit_1.boundary_index == entrance_2.boundary_index == 1
    assert exit_1.component_id != entrance_2.component_id
    assert math.isclose(exit_1.pressure_drop, K_EXIT * result.pass_boundary_states[1].dynamic_pressure, abs_tol=1.0e-12)
    assert math.isclose(entrance_2.pressure_drop, K_ENTRANCE * result.pass_boundary_states[1].dynamic_pressure, abs_tol=1.0e-12)


# ---------------------------------------------------------------------------
# 16.3 Straight tube, four passes
# ---------------------------------------------------------------------------

def test_straight_four_passes_use_correct_local_boundary_state() -> None:
    result = _hydraulic(
        n_tube_passes=4,
        provider=ConstantPropertyProvider(_props(900.0)),
        temperature_in=300.0, temperature_out=500.0, pressure=101325.0,
    )
    assert result.entrance_count == 4
    assert result.exit_count == 4
    assert len(result.pass_boundary_states) == 5

    for pass_index in range(1, 5):
        entrance = result.entrance_results[pass_index - 1]
        exit_ = result.exit_results[pass_index - 1]
        assert entrance.pass_index == pass_index
        assert exit_.pass_index == pass_index
        assert entrance.boundary_index == pass_index - 1
        assert exit_.boundary_index == pass_index

        entrance_state = result.pass_boundary_states[pass_index - 1]
        exit_state = result.pass_boundary_states[pass_index]
        assert math.isclose(entrance.pressure_drop, K_ENTRANCE * entrance_state.dynamic_pressure, abs_tol=1.0e-12)
        assert math.isclose(exit_.pressure_drop, K_EXIT * exit_state.dynamic_pressure, abs_tol=1.0e-12)
        assert entrance.reference_velocity == entrance_state.velocity
        assert exit_.reference_velocity == exit_state.velocity


# ---------------------------------------------------------------------------
# 16.4 U-tube
# ---------------------------------------------------------------------------

def test_u_tube_two_passes_single_entrance_and_exit() -> None:
    result = _hydraulic(
        n_tube_passes=2, tube_path_type=TubePathType.U_TUBE,
        provider=ConstantPropertyProvider(_props(950.0)),
        temperature_in=300.0, temperature_out=400.0, pressure=101325.0,
    )
    assert result.entrance_count == 1
    assert result.exit_count == 1
    assert len(result.pass_boundary_states) == 3  # boundaries still built for future U-bend use

    (entrance,) = result.entrance_results
    (exit_,) = result.exit_results
    assert entrance.boundary_index == 0
    assert exit_.boundary_index == len(result.pass_boundary_states) - 1
    assert entrance.pass_index is None
    assert exit_.pass_index is None
    assert entrance.component_id == "tube_bundle_entrance"
    assert exit_.component_id == "tube_bundle_exit"

    # No entrance/exit loss at the internal (intermediate) boundary, and no
    # U-bend/direction-change pressure drop is calculated: dp_tube_bundle is
    # exactly straight-tube + the single entrance + the single exit.
    assert result.dp_tube_bundle == result.dp_straight_tubes + result.dp_tube_entrances + result.dp_tube_exits
    used_boundary_indices = {entrance.boundary_index, exit_.boundary_index}
    assert 1 not in used_boundary_indices


def test_u_tube_requires_at_least_two_passes() -> None:
    with pytest.raises(ValueError):
        _hydraulic(n_tube_passes=1, tube_path_type=TubePathType.U_TUBE, inlet_props=_props(1000.0))
    with pytest.raises(ValueError):
        TubeBundle(
            tube=BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0),
            n_rows=4, n_tubes_per_row=6, pitch_transverse=0.04, pitch_longitudinal=0.04,
            layout="staggered", n_passes_tube=1, flow_arrangement="crossflow",
            tube_path_type=TubePathType.U_TUBE,
        )


# ---------------------------------------------------------------------------
# 16.5 Constant-property reconstruction
# ---------------------------------------------------------------------------

def test_constant_property_straight_entrance_exit_scale_with_pass_count() -> None:
    for n_passes in (1, 2, 3, 4):
        result = _hydraulic(n_tube_passes=n_passes, inlet_props=_props(1000.0))
        mass_flux = result.mass_flux
        q = _dynamic_pressure(mass_flux, 1000.0)
        assert math.isclose(result.dp_tube_entrances, n_passes * K_ENTRANCE * q, rel_tol=0.0, abs_tol=1.0e-9)
        assert math.isclose(result.dp_tube_exits, n_passes * K_EXIT * q, rel_tol=0.0, abs_tol=1.0e-9)


def test_constant_property_u_tube_entrance_exit_independent_of_pass_count() -> None:
    for n_passes in (2, 3, 4):
        result = _hydraulic(n_tube_passes=n_passes, tube_path_type=TubePathType.U_TUBE, inlet_props=_props(1000.0))
        mass_flux = result.mass_flux
        q = _dynamic_pressure(mass_flux, 1000.0)
        assert math.isclose(result.dp_tube_entrances, K_ENTRANCE * q, rel_tol=0.0, abs_tol=1.0e-9)
        assert math.isclose(result.dp_tube_exits, K_EXIT * q, rel_tol=0.0, abs_tol=1.0e-9)


# ---------------------------------------------------------------------------
# 16.6 Variable-property reconstruction
# ---------------------------------------------------------------------------

def test_variable_property_reconstruction_from_stored_boundary_states() -> None:
    result = _hydraulic(
        n_tube_passes=4,
        provider=ConstantPropertyProvider(_props(900.0)),
        temperature_in=300.0, temperature_out=500.0, pressure=101325.0,
    )
    # Independently reconstruct each entrance/exit from the stored
    # pass-boundary states (not from the aggregated totals).
    for pass_index, entrance in zip(range(1, 5), result.entrance_results):
        state = result.pass_boundary_states[pass_index - 1]
        expected = K_ENTRANCE * state.dynamic_pressure
        assert math.isclose(entrance.pressure_drop, expected, rel_tol=0.0, abs_tol=1.0e-12)
    for pass_index, exit_ in zip(range(1, 5), result.exit_results):
        state = result.pass_boundary_states[pass_index]
        expected = K_EXIT * state.dynamic_pressure
        assert math.isclose(exit_.pressure_drop, expected, rel_tol=0.0, abs_tol=1.0e-12)

    reconstructed_entrances = sum(r.pressure_drop for r in result.entrance_results)
    reconstructed_exits = sum(r.pressure_drop for r in result.exit_results)
    assert math.isclose(reconstructed_entrances, result.dp_tube_entrances, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(reconstructed_exits, result.dp_tube_exits, rel_tol=0.0, abs_tol=1.0e-12)


# ---------------------------------------------------------------------------
# 16.7 No multiplication by parallel tube count
# ---------------------------------------------------------------------------

def test_entrance_exit_not_multiplied_by_parallel_tube_count() -> None:
    rho = 1000.0
    m_dot = 1.0
    small_area = _hydraulic(n_tube_passes=1, flow_area_per_pass=0.01, inlet_props=_props(rho), m_dot=m_dot)
    large_area = _hydraulic(n_tube_passes=1, flow_area_per_pass=0.02, inlet_props=_props(rho), m_dot=m_dot)

    assert large_area.mass_flux == pytest.approx(small_area.mass_flux / 2.0)
    assert large_area.pass_boundary_states[0].velocity == pytest.approx(small_area.pass_boundary_states[0].velocity / 2.0)

    # dp = K * mass_flux^2 / (2*rho): doubling flow_area_per_pass (e.g. more
    # parallel tubes at the same total m_dot) halves mass_flux, so entrance/
    # exit dp scales as 1/4 -- not multiplied again by a tube count.
    assert large_area.dp_tube_entrances == pytest.approx(small_area.dp_tube_entrances / 4.0)
    assert large_area.dp_tube_exits == pytest.approx(small_area.dp_tube_exits / 4.0)

    expected_small = K_ENTRANCE * _dynamic_pressure(small_area.mass_flux, rho)
    expected_large = K_ENTRANCE * _dynamic_pressure(large_area.mass_flux, rho)
    assert math.isclose(small_area.dp_tube_entrances, expected_small, rel_tol=0.0, abs_tol=1.0e-9)
    assert math.isclose(large_area.dp_tube_entrances, expected_large, rel_tol=0.0, abs_tol=1.0e-9)


# ---------------------------------------------------------------------------
# 16.8 Universal fluids
# ---------------------------------------------------------------------------

def test_universal_fluids_use_the_same_entrance_exit_path() -> None:
    cases = (
        (DryAirPropertyProvider(prefer_coolprop=False), 300.0, 400.0),   # heated gas
        (DryAirPropertyProvider(prefer_coolprop=False), 400.0, 300.0),   # cooled gas
        (MoistAirTransportProvider.from_t_rh(T=300.0, RH=0.4, p=101325.0), 300.0, 320.0),
        (IAPWS97WaterSteamProvider(), 300.0, 320.0),   # heated liquid
        (IAPWS97WaterSteamProvider(), 320.0, 300.0),   # cooled liquid
    )
    for provider, T_in, T_out in cases:
        result = _hydraulic(
            n_tube_passes=3,
            provider=provider, temperature_in=T_in, temperature_out=T_out, pressure=101325.0,
        )
        assert result.entrance_count == 3
        assert result.exit_count == 3
        for pass_index, entrance in zip(range(1, 4), result.entrance_results):
            state = result.pass_boundary_states[pass_index - 1]
            assert math.isclose(entrance.pressure_drop, K_ENTRANCE * state.dynamic_pressure, rel_tol=0.0, abs_tol=1.0e-9)
        for pass_index, exit_ in zip(range(1, 4), result.exit_results):
            state = result.pass_boundary_states[pass_index]
            assert math.isclose(exit_.pressure_drop, K_EXIT * state.dynamic_pressure, rel_tol=0.0, abs_tol=1.0e-9)
        assert result.dp_tube_bundle == result.dp_straight_tubes + result.dp_tube_entrances + result.dp_tube_exits


# ---------------------------------------------------------------------------
# 16.9 Result aggregation
# ---------------------------------------------------------------------------

def test_result_aggregation_dp_core_local_total() -> None:
    tube = BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0)
    bundle = TubeBundle(
        tube=tube, n_rows=4, n_tubes_per_row=6, pitch_transverse=0.04, pitch_longitudinal=0.04,
        layout="staggered", n_passes_tube=2, flow_arrangement="crossflow",
    )
    hx = BareTubeHeatExchanger(bundle)
    result = hx.solve(
        hot_stream=SensibleHeatStream(C=1000.0, T_in=350.0),
        cold_stream=SensibleHeatStream(C=500.0, T_in=300.0),
        m_dot_tube_side=1.0, tube_side_props=_props(1000.0),
        tube_side_temperature_in=350.0, tube_side_temperature_out=340.0, tube_side_pressure=101325.0,
        m_dot_outside=2.0, outside_props=_props(1.2, mu=1.8e-5, k=0.026, cp=1006.0),
        outside_temperature_in=300.0, outside_temperature_out=310.0, outside_pressure=101325.0,
    )
    tube_bundle = result.tube_side_hydraulic.tube_bundle

    assert tube_bundle.dp_straight_tubes == tube_bundle.dp_straight_tube_friction + tube_bundle.dp_straight_tube_acceleration
    assert tube_bundle.dp_tube_bundle == tube_bundle.dp_straight_tubes + tube_bundle.dp_tube_entrances + tube_bundle.dp_tube_exits

    dp = result.tube_side_pressure_drop
    assert dp.dp_core == result.inside_dp_tube_bundle
    assert dp.dp_local == 0.0
    assert dp.dp_total == tube_bundle.dp_tube_bundle
    assert result.inside_dp_total == result.inside_dp_tube_bundle


# ---------------------------------------------------------------------------
# 16.10 Standard solver (simulate/rate) integration
# ---------------------------------------------------------------------------

def test_simulate_and_rate_return_updated_tube_bundle_result_without_a_path() -> None:
    tube = BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0)
    bundle = TubeBundle(
        tube=tube, n_rows=4, n_tubes_per_row=6, pitch_transverse=0.04, pitch_longitudinal=0.04,
        layout="staggered", n_passes_tube=2, flow_arrangement="crossflow",
    )
    provider = ConstantPropertyProvider(_props(1000.0))
    outside_provider = ConstantPropertyProvider(_props(1.2, mu=1.8e-5, k=0.026, cp=1006.0))
    hx = BareTubeHeatExchanger(bundle)

    inside = HXSideInput(provider=provider, m_dot=1.0, T_in=350.0, p=101325.0)
    outside = HXSideInput(provider=outside_provider, m_dot=2.0, T_in=300.0, p=101325.0)
    simulated = hx.simulate(inside, outside, max_iter=5)

    assert simulated.entrance_count == 2
    assert simulated.exit_count == 2
    assert simulated.inside_dp_tube_bundle == simulated.inside_dp_straight_tubes + simulated.inside_dp_tube_entrances + simulated.inside_dp_tube_exits
    assert simulated.inside_dp_total == simulated.inside_dp_tube_bundle

    rating = hx.rate(
        BalanceSideSpec(provider=provider, m_dot=1.0, p=101325.0, T_in=350.0, T_out=340.0),
        BalanceSideSpec(provider=outside_provider, m_dot=2.0, p=101325.0, T_in=300.0, T_out=310.0),
        Q=1000.0,
    )
    assert rating.entrance_count == 2
    assert rating.exit_count == 2
    assert rating.inside_dp_tube_bundle == rating.inside_dp_straight_tubes + rating.inside_dp_tube_entrances + rating.inside_dp_tube_exits
    assert rating.inside_dp_total == rating.inside_dp_tube_bundle


# ---------------------------------------------------------------------------
# 16.11 Outside and thermal regression
# ---------------------------------------------------------------------------

def test_outside_and_thermal_results_independent_of_tube_path_type() -> None:
    tube = BareTube(D_i=0.02, D_o=0.024, length_total=2.0, length_effective=2.0, wall_k=20.0)

    def _make_hx(tube_path_type: TubePathType) -> BareTubeHeatExchanger:
        bundle = TubeBundle(
            tube=tube, n_rows=4, n_tubes_per_row=6, pitch_transverse=0.04, pitch_longitudinal=0.04,
            layout="staggered", n_passes_tube=2, flow_arrangement="crossflow",
            tube_path_type=tube_path_type,
        )
        return BareTubeHeatExchanger(bundle)

    common_kwargs = dict(
        hot_stream=SensibleHeatStream(C=1000.0, T_in=350.0),
        cold_stream=SensibleHeatStream(C=500.0, T_in=300.0),
        m_dot_tube_side=1.0, tube_side_props=_props(1000.0),
        tube_side_temperature_in=350.0, tube_side_temperature_out=340.0, tube_side_pressure=101325.0,
        m_dot_outside=2.0, outside_props=_props(1.2, mu=1.8e-5, k=0.026, cp=1006.0),
        outside_temperature_in=300.0, outside_temperature_out=310.0, outside_pressure=101325.0,
    )

    straight_result = _make_hx(TubePathType.STRAIGHT).solve(**common_kwargs)
    u_tube_result = _make_hx(TubePathType.U_TUBE).solve(**common_kwargs)

    # Thermal results and outside pressure drop do not depend on tube-side
    # entrance/exit pressure-drop bookkeeping.
    assert straight_result.UA == u_tube_result.UA
    assert straight_result.Q == u_tube_result.Q
    assert straight_result.T_hot_out == u_tube_result.T_hot_out
    assert straight_result.T_cold_out == u_tube_result.T_cold_out
    assert straight_result.tube_side_thermal.alfa == u_tube_result.tube_side_thermal.alfa
    assert straight_result.outside_side_thermal.alfa == u_tube_result.outside_side_thermal.alfa
    assert straight_result.outside_dp_total == u_tube_result.outside_dp_total
    assert straight_result.outside_dp_drag == u_tube_result.outside_dp_drag
    assert straight_result.outside_dp_acceleration == u_tube_result.outside_dp_acceleration

    # But the tube-side entrance/exit counts (and hence dp_tube_bundle) do.
    assert straight_result.entrance_count == 2
    assert u_tube_result.entrance_count == 1
    assert straight_result.inside_dp_tube_bundle != u_tube_result.inside_dp_tube_bundle
    # Straight-tube-only friction/acceleration remain identical between the
    # two path types; only entrance/exit differ.
    assert straight_result.inside_dp_straight_tubes == u_tube_result.inside_dp_straight_tubes
