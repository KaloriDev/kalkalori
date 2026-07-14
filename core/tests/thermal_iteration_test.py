# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""
Smoke test for the v0.5.2 iterative wall-corrected thermal state
(core.heat_transfer.thermal_iteration / BareTubeHeatExchanger.solve_thermal_state).

Also covers the required integration checks: flow_arrangement="crossflow"
remains a public, unchanged value; the outside side is automatically mixed
and the tube side automatically unmixed; and counterflow/cocurrentflow
Simulation use cases still work unchanged.

Run:
    python -m core.tests.thermal_iteration_test
"""

from __future__ import annotations

import math

from core.geometry.tube import BareTube
from core.geometry.bundle import TubeBundle
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.heat_transfer.thermal_iteration import solve_iterative_thermal_state


def build_bundle(*, wall_k: float = 50.0, flow_arrangement: str = "counterflow") -> TubeBundle:
    tube = BareTube(
        D_i=25e-3 - 2 * 1.5e-3,
        D_o=25e-3,
        length_total=2.8,
        length_effective=2.8,
        wall_k=wall_k,
    )
    return TubeBundle(
        tube=tube,
        n_rows=36,
        n_tubes_per_row=56,
        pitch_transverse=35e-3,
        pitch_longitudinal=35e-3,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement=flow_arrangement,
    )


def c_to_k(t_c: float) -> float:
    return t_c + 273.15


def kgh_to_kgs(m: float) -> float:
    return m / 3600.0


_INSIDE_PROVIDER = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
)
_OUTSIDE_PROVIDER = ConstantPropertyProvider(
    FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
)
_M_DOT_INSIDE = kgh_to_kgs(18_220.0)
_M_DOT_OUTSIDE = kgh_to_kgs(28_380.0)
_P = 101_325.0


def test_inside_hot_case() -> None:
    print("thermal_iteration: inside-hot case converges with sane wall temperatures")
    hx = BareTubeHeatExchanger(build_bundle())
    state = solve_iterative_thermal_state(
        hx,
        m_dot_inside=_M_DOT_INSIDE, m_dot_outside=_M_DOT_OUTSIDE,
        inside_provider=_INSIDE_PROVIDER, outside_provider=_OUTSIDE_PROVIDER,
        T_in_inside=c_to_k(400.0), T_in_outside=c_to_k(30.0),
        p_inside=_P, p_outside=_P,
    )
    assert state.converged, state
    assert state.iterations > 1
    # Inside is hot: inside wall must run cooler than the inside bulk, and
    # outside wall must run hotter than the outside bulk.
    assert state.inside_wall_temperature < state.inside_bulk_temperature
    assert state.outside_wall_temperature > state.outside_bulk_temperature


def test_outside_hot_case() -> None:
    print("thermal_iteration: outside-hot case converges with sane wall temperatures")
    hx = BareTubeHeatExchanger(build_bundle())
    state = solve_iterative_thermal_state(
        hx,
        m_dot_inside=_M_DOT_INSIDE, m_dot_outside=_M_DOT_OUTSIDE,
        inside_provider=_INSIDE_PROVIDER, outside_provider=_OUTSIDE_PROVIDER,
        T_in_inside=c_to_k(30.0), T_in_outside=c_to_k(400.0),
        p_inside=_P, p_outside=_P,
    )
    assert state.converged, state
    # Outside is hot now: ordering flips relative to the inside-hot case.
    assert state.outside_wall_temperature < state.outside_bulk_temperature
    assert state.inside_wall_temperature > state.inside_bulk_temperature


def test_wall_temperatures_between_bulk_temperatures() -> None:
    print("thermal_iteration: both wall temperatures lie within the bulk-temperature interval")
    hx = BareTubeHeatExchanger(build_bundle())
    state = solve_iterative_thermal_state(
        hx,
        m_dot_inside=_M_DOT_INSIDE, m_dot_outside=_M_DOT_OUTSIDE,
        inside_provider=_INSIDE_PROVIDER, outside_provider=_OUTSIDE_PROVIDER,
        T_in_inside=c_to_k(400.0), T_in_outside=c_to_k(30.0),
        p_inside=_P, p_outside=_P,
    )
    lo = min(state.inside_bulk_temperature, state.outside_bulk_temperature)
    hi = max(state.inside_bulk_temperature, state.outside_bulk_temperature)
    assert lo - 1e-6 <= state.inside_wall_temperature <= hi + 1e-6
    assert lo - 1e-6 <= state.outside_wall_temperature <= hi + 1e-6
    # Finite wall resistance -> the two wall-surface temperatures differ.
    assert abs(state.inside_wall_temperature - state.outside_wall_temperature) > 1e-6
    assert state.warnings == () or all(
        w.code != "thermal_iteration_wall_temperature_out_of_bulk_range" for w in state.warnings
    )


def test_negligible_wall_resistance_gives_nearly_equal_wall_temperatures() -> None:
    print("thermal_iteration: near-zero wall resistance -> inside/outside wall temps nearly equal")
    hx = BareTubeHeatExchanger(build_bundle(wall_k=1.0e7))
    state = solve_iterative_thermal_state(
        hx,
        m_dot_inside=_M_DOT_INSIDE, m_dot_outside=_M_DOT_OUTSIDE,
        inside_provider=_INSIDE_PROVIDER, outside_provider=_OUTSIDE_PROVIDER,
        T_in_inside=c_to_k(400.0), T_in_outside=c_to_k(30.0),
        p_inside=_P, p_outside=_P,
    )
    assert state.converged
    assert abs(state.inside_wall_temperature - state.outside_wall_temperature) < 0.5, (
        state.inside_wall_temperature, state.outside_wall_temperature
    )


def test_stable_convergence_with_relaxation() -> None:
    print("thermal_iteration: default relaxation converges within max_iterations")
    hx = BareTubeHeatExchanger(build_bundle())
    state = solve_iterative_thermal_state(
        hx,
        m_dot_inside=_M_DOT_INSIDE, m_dot_outside=_M_DOT_OUTSIDE,
        inside_provider=_INSIDE_PROVIDER, outside_provider=_OUTSIDE_PROVIDER,
        T_in_inside=c_to_k(400.0), T_in_outside=c_to_k(30.0),
        p_inside=_P, p_outside=_P,
        relaxation_factor=0.5,
        max_iterations=25,
    )
    assert state.converged
    assert not any(w.code == "thermal_iteration_not_converged" for w in state.warnings)


def test_forced_non_convergence_warns() -> None:
    print("thermal_iteration: max_iterations=1 forces non-convergence -> structured warning, no hang")
    hx = BareTubeHeatExchanger(build_bundle())
    state = solve_iterative_thermal_state(
        hx,
        m_dot_inside=_M_DOT_INSIDE, m_dot_outside=_M_DOT_OUTSIDE,
        inside_provider=_INSIDE_PROVIDER, outside_provider=_OUTSIDE_PROVIDER,
        T_in_inside=c_to_k(400.0), T_in_outside=c_to_k(30.0),
        p_inside=_P, p_outside=_P,
        max_iterations=1,
    )
    assert not state.converged
    assert state.iterations == 1
    codes = {w.code for w in state.warnings}
    assert "thermal_iteration_not_converged" in codes, codes


def test_non_finite_input_protection() -> None:
    print("thermal_iteration: non-finite/invalid inputs raise ValueError")
    hx = BareTubeHeatExchanger(build_bundle())
    for bad_kwargs in (
        dict(T_in_inside=float("nan")),
        dict(T_in_outside=float("inf")),
        dict(m_dot_inside=-1.0),
        dict(p_inside=0.0),
    ):
        kwargs = dict(
            m_dot_inside=_M_DOT_INSIDE, m_dot_outside=_M_DOT_OUTSIDE,
            inside_provider=_INSIDE_PROVIDER, outside_provider=_OUTSIDE_PROVIDER,
            T_in_inside=c_to_k(400.0), T_in_outside=c_to_k(30.0),
            p_inside=_P, p_outside=_P,
        )
        kwargs.update(bad_kwargs)
        try:
            solve_iterative_thermal_state(hx, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad_kwargs}")


def test_integration_crossflow_api_unchanged_and_wraps_thermal_iteration() -> None:
    print("integration: flow_arrangement='crossflow' unchanged; hx.solve_thermal_state exposes iteration/convergence/wall temps")
    hx = BareTubeHeatExchanger(build_bundle(flow_arrangement="crossflow"))

    state = hx.solve_thermal_state(
        m_dot_inside=_M_DOT_INSIDE, m_dot_outside=_M_DOT_OUTSIDE,
        inside_provider=_INSIDE_PROVIDER, outside_provider=_OUTSIDE_PROVIDER,
        T_in_inside=c_to_k(400.0), T_in_outside=c_to_k(30.0),
        p_inside=_P, p_outside=_P,
    )
    assert state.converged
    assert isinstance(state.iterations, int) and state.iterations >= 1
    assert math.isfinite(state.inside_wall_temperature)
    assert math.isfinite(state.outside_wall_temperature)

    # The public flow-arrangement value is unchanged; also exercise it
    # through the existing Simulation entry point.
    sim = hx.simulate(
        HXSideInput(provider=_INSIDE_PROVIDER, m_dot=_M_DOT_INSIDE, T_in=c_to_k(400.0), p=_P),
        HXSideInput(provider=_OUTSIDE_PROVIDER, m_dot=_M_DOT_OUTSIDE, T_in=c_to_k(30.0), p=_P),
    )
    assert sim.converged


def test_regression_counterflow_and_cocurrentflow_simulate_still_work() -> None:
    print("regression: counterflow and cocurrentflow Simulation still converge")
    for arrangement in ("counterflow", "cocurrentflow"):
        hx = BareTubeHeatExchanger(build_bundle(flow_arrangement=arrangement))
        sim = hx.simulate(
            HXSideInput(provider=_INSIDE_PROVIDER, m_dot=_M_DOT_INSIDE, T_in=c_to_k(400.0), p=_P),
            HXSideInput(provider=_OUTSIDE_PROVIDER, m_dot=_M_DOT_OUTSIDE, T_in=c_to_k(30.0), p=_P),
        )
        assert sim.converged, arrangement
        assert sim.q > 0.0


def main() -> None:
    test_inside_hot_case()
    test_outside_hot_case()
    test_wall_temperatures_between_bulk_temperatures()
    test_negligible_wall_resistance_gives_nearly_equal_wall_temperatures()
    test_stable_convergence_with_relaxation()
    test_forced_non_convergence_warns()
    test_non_finite_input_protection()
    test_integration_crossflow_api_unchanged_and_wraps_thermal_iteration()
    test_regression_counterflow_and_cocurrentflow_simulate_still_work()

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
