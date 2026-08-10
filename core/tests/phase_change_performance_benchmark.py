"""Manual performance probe for the three phase-change execution paths.

This module is intentionally not named ``*_test.py`` and contains no timing
assertions.  Run it from the repository root, for example::

    py -3.11 -m core.tests.phase_change_performance_benchmark --case all
    py -3.11 -m core.tests.phase_change_performance_benchmark --case inside --profile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from collections.abc import Callable
from typing import Any

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec


def _small_hx() -> BareTubeHeatExchanger:
    tube = BareTube(
        D_i=0.022,
        D_o=0.025,
        length_total=2.8,
        length_effective=2.8,
        wall_k=50.0,
    )
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=tube,
            n_rows=5,
            n_tubes_per_row=10,
            pitch_transverse=0.035,
            pitch_longitudinal=0.035,
            layout="staggered",
            n_passes_tube=2,
            flow_arrangement="counterflow",
        )
    )


def _large_hx() -> BareTubeHeatExchanger:
    tube = BareTube(
        D_i=0.022,
        D_o=0.025,
        length_total=2.8,
        length_effective=2.8,
        wall_k=50.0,
    )
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=tube,
            n_rows=20,
            n_tubes_per_row=30,
            pitch_transverse=0.035,
            pitch_longitudinal=0.035,
            layout="staggered",
            n_passes_tube=2,
            flow_arrangement="counterflow",
        )
    )


def _dry_spec() -> GasMixtureSpec:
    return GasMixtureSpec(components={"N2": 0.79, "O2": 0.21}, basis="mole")


def _inside_wet_spec() -> GasMixtureSpec:
    y_h2o = 0.08
    remainder = 1.0 - y_h2o
    dry = {"N2": 0.70, "O2": 0.10, "CO2": 0.08}
    total = sum(dry.values())
    return GasMixtureSpec(
        components={
            name: value * remainder / total for name, value in dry.items()
        }
        | {"H2O": y_h2o},
        basis="mole",
    )


def _run_sensible():
    hx = BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.022,
                D_o=0.025,
                length_total=2.8,
                length_effective=2.8,
                wall_k=50.0,
            ),
            n_rows=36,
            n_tubes_per_row=56,
            pitch_transverse=0.035,
            pitch_longitudinal=0.035,
            layout="staggered",
            n_passes_tube=2,
            flow_arrangement="counterflow",
        )
    )
    inside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
    )
    outside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
    )
    return hx.simulate(
        HXSideInput(
            provider=inside_provider,
            m_dot=5.06,
            T_in=303.15,
            p=101_325.0,
        ),
        HXSideInput(
            provider=outside_provider,
            m_dot=7.88,
            T_in=673.15,
            p=101_325.0,
        ),
    )


def _run_outside():
    wet_process_gas = GasMixtureSpec(
        components={"N2": 0.65, "O2": 0.10, "CO2": 0.08, "H2O": 0.17},
        basis="mole",
    )
    return _large_hx().simulate(
        HXSideInput(
            provider=GasMixturePropertyProvider(_dry_spec()),
            m_dot=15.0,
            T_in=290.0,
            p=101_325.0,
        ),
        HXSideInput(
            provider=GasMixturePropertyProvider(wet_process_gas),
            m_dot=6.0,
            T_in=420.0,
            p=101_325.0,
        ),
    )


def _run_inside():
    return _small_hx().simulate(
        HXSideInput(
            provider=GasMixturePropertyProvider(_inside_wet_spec()),
            m_dot=1.0,
            T_in=360.0,
            p=101_325.0,
        ),
        HXSideInput(
            provider=GasMixturePropertyProvider(_dry_spec()),
            m_dot=5.0,
            T_in=290.0,
            p=101_325.0,
        ),
    )


RUNNERS: dict[str, Callable[[], Any]] = {
    "sensible": _run_sensible,
    "outside": _run_outside,
    "inside": _run_inside,
}


def _result_metrics(
    name: str,
    result: Any,
    runtime_s: float,
    dry_thermal_iterations: int,
) -> dict[str, Any]:
    pc = result.inside_phase_change if name == "inside" else result.outside_phase_change
    metrics: dict[str, Any] = {
        "case": name,
        "runtime_s": runtime_s,
        "dry_thermal_iterations": dry_thermal_iterations,
        "thermal_iterations": result.thermal_state.iterations,
        "phase_iterations": pc.iterations,
        "Q_total_W": result.q,
        "T_out_inside_K": result.T_out_inside,
        "T_out_outside_K": result.T_out_outside,
        "inside_dp_friction_Pa": result.inside_dp_straight_tube_friction,
        "inside_dp_acceleration_Pa": result.inside_dp_acceleration,
        "outside_dp_drag_Pa": result.outside_dp_drag,
        "outside_dp_acceleration_Pa": result.outside_dp_acceleration,
    }
    if pc.active:
        metrics.update(
            {
                "Q_sensible_W": pc.Q_sensible,
                "Q_latent_W": pc.Q_latent,
                "W_out": pc.W_out,
                "m_dot_condensate_kg_s": pc.m_dot_condensate,
                "wet_surface_fraction": pc.wet_surface_fraction,
                "T_wall_mean_K": pc.wall_temperature_mean,
                "T_wall_wet_mean_K": pc.wall_temperature_wet_mean,
                "alfa_dry_W_m2K": pc.alfa_dry,
                "alfa_effective_W_m2K": pc.alfa_effective,
                "mass_balance_error_kg_s": pc.mass_balance_error,
                "energy_balance_error_W": pc.energy_balance_error,
            }
        )
    return metrics


def run_case(name: str) -> tuple[Any, dict[str, Any]]:
    from core.models import simulation

    original_run_simulation = simulation.run_simulation
    dry_iterations: list[int] = []

    def capture_dry_iterations(*args, **kwargs):
        dry_result = original_run_simulation(*args, **kwargs)
        dry_iterations.append(dry_result.iterations)
        return dry_result

    simulation.run_simulation = capture_dry_iterations
    try:
        start = time.perf_counter()
        result = RUNNERS[name]()
        elapsed = time.perf_counter() - start
    finally:
        simulation.run_simulation = original_run_simulation
    if len(dry_iterations) != 1:
        raise RuntimeError(
            f"Expected one dry Simulation baseline, got {len(dry_iterations)}."
        )
    return result, _result_metrics(
        name,
        result,
        elapsed,
        dry_iterations[0],
    )


def print_profile(name: str) -> None:
    profile = cProfile.Profile()
    profile.enable()
    result, metrics = run_case(name)
    profile.disable()
    print(json.dumps(metrics, sort_keys=True))

    for sort_key, label in (
        (pstats.SortKey.CUMULATIVE, "TOP_CUMULATIVE"),
        (pstats.SortKey.CALLS, "TOP_CALLS"),
    ):
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(sort_key).print_stats(25)
        print(label)
        print(stream.getvalue())

    property_rows = []
    for (filename, line, function), values in pstats.Stats(profile).stats.items():
        primitive_calls, total_calls, own_time, cumulative_time, _callers = values
        key = f"{filename}:{line}({function})"
        key_lower = key.lower()
        if any(
            token in key_lower
            for token in (
                "coolprop",
                "gas_mixture",
                "wet_gas",
                "water.py",
                "water_equilibrium",
                "iapws",
            )
        ):
            property_rows.append(
                (cumulative_time, total_calls, primitive_calls, own_time, key)
            )
    print("PROPERTY_HOTSPOTS")
    for cumulative_time, total_calls, primitive_calls, own_time, key in sorted(
        property_rows, reverse=True
    )[:30]:
        print(
            f"{cumulative_time:.6f}s cumulative; {own_time:.6f}s own; "
            f"{total_calls}/{primitive_calls} calls; {key}"
        )
    print("SOLVER_CALL_COUNTS")
    solver_tokens = (
        "condensation_solver_helpers.py",
        "wet_gas_composition.py",
        "inside_condensation_solver.py",
        "internal_pressure_drop.py",
    )
    for (filename, line, function), values in sorted(
        pstats.Stats(profile).stats.items()
    ):
        if any(token in filename for token in solver_tokens):
            primitive_calls, total_calls, own_time, cumulative_time, _callers = values
            print(
                f"{total_calls}/{primitive_calls} calls; "
                f"{cumulative_time:.6f}s cumulative; "
                f"{filename}:{line}({function})"
            )
    del result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("all", *RUNNERS),
        default="all",
    )
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.profile:
        if args.case == "all":
            parser.error("--profile requires one concrete --case")
        print_profile(args.case)
        return
    names = tuple(RUNNERS) if args.case == "all" else (args.case,)
    for name in names:
        _result, metrics = run_case(name)
        print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
