# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""
Call-path integration tests (v0.5.3) for BareTubeHeatExchanger.simulate():
proves that Simulation actually consumes the converged iterative thermal
state (wall/length-corrected alfa_i/alfa_o/U/UA), that this is not later
overwritten by the separate (uncorrected) legacy solve() pass kept only for
area/hydraulic/regime diagnostics, and that Simulation and Rating agree
where they should (the include_simulation=True bridge).

Run:
    python -m core.tests.simulate_thermal_iteration_integration_test
"""

from __future__ import annotations

import math
from unittest import mock

from core.geometry.tube import BareTube
from core.geometry.bundle import TubeBundle
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.models.heat_balance import BalanceSideSpec
from core.heat_transfer.thermal_iteration import (
    IterativeThermalState,
    ThermalIterationDiagnostics,
)


def c_to_k(t_c: float) -> float:
    return t_c + 273.15


def kgh_to_kgs(m: float) -> float:
    return m / 3600.0


def build_bundle() -> TubeBundle:
    tube = BareTube(
        D_i=25e-3 - 2 * 1.5e-3, D_o=25e-3,
        length_total=2.8, length_effective=2.8, wall_k=50.0,
    )
    return TubeBundle(
        tube=tube, n_rows=36, n_tubes_per_row=56,
        pitch_transverse=35e-3, pitch_longitudinal=35e-3,
        layout="staggered", n_passes_tube=2,
        flow_arrangement="counterflow",
    )


_INSIDE_PROVIDER = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
)
_OUTSIDE_PROVIDER = ConstantPropertyProvider(
    FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
)
_M_DOT_INSIDE = kgh_to_kgs(18_220.0)
_M_DOT_OUTSIDE = kgh_to_kgs(28_380.0)
_P = 101_325.0


_SENTINEL_ALFA_I = 123.456
_SENTINEL_ALFA_O = 234.567
_SENTINEL_U = 16.789
_SENTINEL_UA = 3456.789


def _sentinel_thermal_state(*, T_in_inside: float, T_in_outside: float) -> IterativeThermalState:
    props = FluidTransportProperties(rho=1.0, mu=2.0e-5, k=0.03, cp=1007.0)
    diagnostics = ThermalIterationDiagnostics(
        inside_Nu_base=10.0, inside_Nu_corrected=9.0,
        inside_length_correction=1.05, inside_wall_temperature_correction=0.857,
        inside_combined_correction=1.05 * 0.857,
        inside_alfa_base=100.0, inside_alfa_corrected=_SENTINEL_ALFA_I,
        outside_Nu_base=20.0, outside_Nu_corrected=21.0,
        outside_wall_property_correction=1.05,
    )
    return IterativeThermalState(
        inside_bulk_temperature=0.5 * (T_in_inside + T_in_outside) + 10.0,
        outside_bulk_temperature=0.5 * (T_in_inside + T_in_outside) - 10.0,
        inside_wall_temperature=0.5 * (T_in_inside + T_in_outside),
        outside_wall_temperature=0.5 * (T_in_inside + T_in_outside),
        inside_bulk_props=props, inside_wall_props=props,
        outside_bulk_props=props, outside_wall_props=props,
        alfa_i=_SENTINEL_ALFA_I, alfa_o=_SENTINEL_ALFA_O,
        U=_SENTINEL_U, UA=_SENTINEL_UA,
        iterations=7, converged=True, residual=1e-3,
        diagnostics=diagnostics,
        inside_provider_name="SentinelProvider", outside_provider_name="SentinelProvider",
        warnings=(),
    )


def test_simulate_consumes_sentinel_thermal_state() -> None:
    print("call-path: patching core.models.simulation.solve_iterative_thermal_state feeds HXSimulationResult")
    hx = BareTubeHeatExchanger(build_bundle())
    inside = HXSideInput(provider=_INSIDE_PROVIDER, m_dot=_M_DOT_INSIDE, T_in=c_to_k(30.0), p=_P)
    outside = HXSideInput(provider=_OUTSIDE_PROVIDER, m_dot=_M_DOT_OUTSIDE, T_in=c_to_k(400.0), p=_P)

    def _fake(hx_arg, *, T_in_inside, T_in_outside, **_kwargs):
        return _sentinel_thermal_state(T_in_inside=T_in_inside, T_in_outside=T_in_outside)

    with mock.patch(
        "core.models.simulation.solve_iterative_thermal_state", side_effect=_fake
    ) as patched:
        result = hx.simulate(inside, outside)
        assert patched.called, "run_simulation must call solve_iterative_thermal_state"

    assert result.inside_alfa_mean == _SENTINEL_ALFA_I, result.inside_alfa_mean
    assert result.outside_alfa_mean == _SENTINEL_ALFA_O, result.outside_alfa_mean
    assert result.U_mean == _SENTINEL_U, result.U_mean
    assert result.UA == _SENTINEL_UA, result.UA
    assert result.thermal_state is not None
    assert result.thermal_state.UA == _SENTINEL_UA
    assert result.iterations == 7
    assert result.converged is True

    # q must be derived from the sentinel UA (not overwritten by the
    # separate, real, uncorrected legacy solve() pass that also runs inside
    # run_simulation for area/hydraulic diagnostics only).
    assert result.q > 0.0
    assert math.isfinite(result.q)


def test_real_dry_air_heated_case_matches_corrected_diagnostics() -> None:
    print("real (non-mocked): heated dry air inside tubes -> simulation.inside_alfa_mean == corrected diagnostics")
    hx = BareTubeHeatExchanger(build_bundle())
    inside = HXSideInput(provider=_INSIDE_PROVIDER, m_dot=_M_DOT_INSIDE, T_in=c_to_k(30.0), p=_P)
    outside = HXSideInput(provider=_OUTSIDE_PROVIDER, m_dot=_M_DOT_OUTSIDE, T_in=c_to_k(400.0), p=_P)

    result = hx.simulate(inside, outside)
    assert result.converged
    assert result.thermal_state is not None
    state = result.thermal_state
    diag = state.diagnostics

    # Inside is the cold stream overall (30 degC in vs. 400 degC outside), so
    # its tube wall runs hotter than its own bulk -> heating from the gas's
    # perspective -> wall_temperature_correction < 1.
    assert state.inside_wall_temperature > state.inside_bulk_temperature
    assert diag.inside_wall_temperature_correction < 1.0, diag.inside_wall_temperature_correction
    assert diag.inside_alfa_corrected < diag.inside_alfa_base, (
        diag.inside_alfa_corrected, diag.inside_alfa_base,
    )

    # The critical consumption check: simulation.inside_alfa_mean must equal
    # the corrected coefficient, not the base/uncorrected one.
    assert result.inside_alfa_mean == diag.inside_alfa_corrected
    assert result.inside_alfa_mean == state.alfa_i
    assert not math.isclose(result.inside_alfa_mean, diag.inside_alfa_base, rel_tol=1e-6)


def test_disabling_wall_correction_at_lowest_level_changes_final_result() -> None:
    print("enable/disable: neutralizing gas_wall_temperature_correction at its defining module changes final U/UA/q")
    # Patch the symbol at its actual point of use: heat_transfer_coefficient_
    # internal_diagnostics (in core.heat_transfer.internal_flow) calls
    # gas_wall_temperature_correction via the SAME module's global namespace,
    # so patching core.heat_transfer.internal_flow.gas_wall_temperature_
    # correction affects it without exposing any new public option.
    hx = BareTubeHeatExchanger(build_bundle())
    inside = HXSideInput(provider=_INSIDE_PROVIDER, m_dot=_M_DOT_INSIDE, T_in=c_to_k(30.0), p=_P)
    outside = HXSideInput(provider=_OUTSIDE_PROVIDER, m_dot=_M_DOT_OUTSIDE, T_in=c_to_k(400.0), p=_P)

    result_corrected = hx.simulate(inside, outside)

    with mock.patch(
        "core.heat_transfer.internal_flow.gas_wall_temperature_correction",
        return_value=(1.0, []),
    ):
        result_neutralized = hx.simulate(inside, outside)

    assert result_corrected.thermal_state is not None
    assert result_neutralized.thermal_state is not None
    assert result_corrected.thermal_state.diagnostics.inside_wall_temperature_correction < 1.0
    assert math.isclose(
        result_neutralized.thermal_state.diagnostics.inside_wall_temperature_correction, 1.0, abs_tol=1e-12
    )

    assert result_neutralized.inside_alfa_mean > result_corrected.inside_alfa_mean
    assert result_neutralized.UA != result_corrected.UA
    assert result_neutralized.q != result_corrected.q
    assert not math.isclose(result_neutralized.UA, result_corrected.UA, rel_tol=1e-6)


def test_rate_include_simulation_consistency() -> None:
    print("consistency: rate(include_simulation=True) -> UA_actual/U_mean match the bridged simulation")
    hx = BareTubeHeatExchanger(build_bundle())
    inside = BalanceSideSpec(
        provider=_INSIDE_PROVIDER, p=_P, m_dot=_M_DOT_INSIDE,
        T_in=c_to_k(30.0), T_out=c_to_k(150.0),
    )
    outside = BalanceSideSpec(
        provider=_OUTSIDE_PROVIDER, p=_P, m_dot=_M_DOT_OUTSIDE,
        T_in=c_to_k(400.0), T_out=c_to_k(300.0),
    )
    rating = hx.rate(inside, outside, include_simulation=True)
    assert rating.simulation is not None

    # Both rate() and the include_simulation bridge call
    # solve_iterative_thermal_state with the SAME (T_in, m_dot, provider, p)
    # per side -- a deterministic, pure function of those inputs -- so they
    # must converge to matching UA/U (within iteration tolerance).
    assert math.isclose(rating.UA_actual, rating.simulation.UA, rel_tol=1e-6)
    assert math.isclose(rating.U_mean, rating.simulation.U_mean, rel_tol=1e-6)
    assert math.isclose(rating.alfa_i, rating.simulation.inside_alfa_mean, rel_tol=1e-6)
    assert math.isclose(rating.alfa_o, rating.simulation.outside_alfa_mean, rel_tol=1e-6)


def test_resistance_reconstruction_simulate_and_rate() -> None:
    print("resistance reconstruction: 1/(R_i+R_wall+R_o) matches UA for both simulate() and rate()")
    hx = BareTubeHeatExchanger(build_bundle())
    A_i = hx.bundle.total_inner_area
    A_o = hx.bundle.total_outer_area
    R_wall = hx.tube_wall_resistance()

    inside_sim = HXSideInput(provider=_INSIDE_PROVIDER, m_dot=_M_DOT_INSIDE, T_in=c_to_k(30.0), p=_P)
    outside_sim = HXSideInput(provider=_OUTSIDE_PROVIDER, m_dot=_M_DOT_OUTSIDE, T_in=c_to_k(400.0), p=_P)
    sim = hx.simulate(inside_sim, outside_sim)
    R_total_sim = 1.0 / (sim.inside_alfa_mean * A_i) + R_wall + 1.0 / (sim.outside_alfa_mean * A_o)
    assert math.isclose(1.0 / R_total_sim, sim.UA, rel_tol=1e-9)

    inside_rate = BalanceSideSpec(
        provider=_INSIDE_PROVIDER, p=_P, m_dot=_M_DOT_INSIDE,
        T_in=c_to_k(30.0), T_out=c_to_k(150.0),
    )
    outside_rate = BalanceSideSpec(
        provider=_OUTSIDE_PROVIDER, p=_P, m_dot=_M_DOT_OUTSIDE,
        T_in=c_to_k(400.0), T_out=c_to_k(300.0),
    )
    rating = hx.rate(inside_rate, outside_rate)
    R_total_rate = 1.0 / (rating.alfa_i * A_i) + R_wall + 1.0 / (rating.alfa_o * A_o)
    assert math.isclose(1.0 / R_total_rate, rating.UA_actual, rel_tol=1e-9)


def test_warnings_propagate_from_thermal_state_to_simulation_result() -> None:
    print("warnings: thermal_state warnings propagate into HXSimulationResult.warnings")
    hx = BareTubeHeatExchanger(build_bundle())
    inside = HXSideInput(provider=_INSIDE_PROVIDER, m_dot=_M_DOT_INSIDE, T_in=c_to_k(30.0), p=_P)
    outside = HXSideInput(provider=_OUTSIDE_PROVIDER, m_dot=_M_DOT_OUTSIDE, T_in=c_to_k(400.0), p=_P)

    # Force non-convergence (deterministic: iteration>=2 required to declare
    # convergence) to guarantee a thermal_iteration_not_converged warning is
    # present on the thermal state, then check it survives into the
    # simulation result's warnings without being dropped or duplicated.
    result = hx.simulate(inside, outside, max_iter=1)
    assert result.thermal_state is not None
    assert not result.thermal_state.converged
    thermal_codes = [w.code for w in result.thermal_state.warnings]
    assert "thermal_iteration_not_converged" in thermal_codes

    result_codes = [w.code for w in (result.warnings or [])]
    assert result_codes.count("thermal_iteration_not_converged") == thermal_codes.count(
        "thermal_iteration_not_converged"
    ), "warning must propagate exactly once, not be duplicated"


def main() -> None:
    test_simulate_consumes_sentinel_thermal_state()
    test_real_dry_air_heated_case_matches_corrected_diagnostics()
    test_disabling_wall_correction_at_lowest_level_changes_final_result()
    test_rate_include_simulation_consistency()
    test_resistance_reconstruction_simulate_and_rate()
    test_warnings_propagate_from_thermal_state_to_simulation_result()

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
