# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Regression tests for the independent four-point 0D endpoint envelope.

Run with: ``python -m core.tests.wall_temperature_envelope_test``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import patch

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.heat_transfer import thermal_iteration as ti
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.properties.common import FluidTransportProperties


@dataclass
class RecordingProvider:
    base_rho: float
    calls: list[float]

    def at(self, T: float, p: float) -> FluidTransportProperties:
        self.calls.append(T)
        scale = T / 300.0
        return FluidTransportProperties(
            rho=max(self.base_rho * 300.0 / T, 1e-3),
            mu=1.8e-5 * scale ** 0.7,
            k=0.027 * scale ** 0.8,
            cp=1000.0 * (1.0 + 2e-4 * (T - 300.0)),
        )


def build_hx() -> BareTubeHeatExchanger:
    tube = BareTube(
        D_i=0.022, D_o=0.025, length_total=2.8,
        length_effective=2.8, wall_k=50.0,
    )
    bundle = TubeBundle(
        tube=tube, n_rows=20, n_tubes_per_row=30,
        pitch_transverse=0.035, pitch_longitudinal=0.035,
        layout="staggered", n_passes_tube=2, flow_arrangement="crossflow",
    )
    return BareTubeHeatExchanger(bundle)


def envelope(hx, inside, outside, **overrides):
    kwargs = dict(
        m_dot_inside=5.0, m_dot_outside=7.0,
        inside_provider=inside, outside_provider=outside,
        inside_inlet_temperature=650.0, inside_outlet_temperature=520.0,
        outside_inlet_temperature=300.0, outside_outlet_temperature=390.0,
        p_inside=101325.0, p_outside=101325.0,
    )
    kwargs.update(overrides)
    return ti.estimate_wall_temperature_envelope(hx, **kwargs)


def test_four_pairs_properties_coefficients_and_extrema() -> None:
    hx = build_hx()
    inside = RecordingProvider(1.1, [])
    outside = RecordingProvider(0.9, [])
    result = envelope(hx, inside, outside)
    assert result.method == "0d_endpoint_envelope"
    assert len(result.probes) == 4 and all(p.converged for p in result.probes)
    assert {(p.inside_bulk_temperature, p.outside_bulk_temperature) for p in result.probes} == {
        (650.0, 300.0), (650.0, 390.0), (520.0, 300.0), (520.0, 390.0),
    }
    # Stored properties prove that every probe evaluated its own bulk state;
    # recorded non-endpoint calls prove that wall properties were evaluated too.
    for probe in result.probes:
        assert math.isclose(probe.inside_bulk_props.rho, 1.1 * 300.0 / probe.inside_bulk_temperature)
        assert math.isclose(probe.outside_bulk_props.rho, 0.9 * 300.0 / probe.outside_bulk_temperature)
        assert probe.inside_wall_props is not None and probe.outside_wall_props is not None
    assert any(T not in {520.0, 650.0} for T in inside.calls)
    assert any(T not in {300.0, 390.0} for T in outside.calls)
    # Temperature-dependent properties mean a single mean alfa was not reused.
    assert len({round(p.alfa_i, 8) for p in result.probes}) > 1
    assert len({round(p.alfa_o, 8) for p in result.probes}) > 1
    assert result.inside_min == min(p.inside_wall_temperature for p in result.probes)
    assert result.inside_max == max(p.inside_wall_temperature for p in result.probes)
    assert result.outside_min == min(p.outside_wall_temperature for p in result.probes)
    assert result.outside_max == max(p.outside_wall_temperature for p in result.probes)


def test_signed_heat_flow_and_wall_ordering_both_directions() -> None:
    hx = build_hx()
    pi, po = RecordingProvider(1.1, []), RecordingProvider(0.9, [])
    inside_hot = envelope(hx, pi, po).probes[0]
    assert inside_hot.heat_rate_probe > 0.0
    assert (inside_hot.inside_bulk_temperature >= inside_hot.inside_wall_temperature
            >= inside_hot.outside_wall_temperature >= inside_hot.outside_bulk_temperature)

    outside_hot = envelope(
        hx, pi, po,
        inside_inlet_temperature=300.0, inside_outlet_temperature=390.0,
        outside_inlet_temperature=650.0, outside_outlet_temperature=520.0,
    ).probes[0]
    assert outside_hot.heat_rate_probe < 0.0
    assert (outside_hot.outside_bulk_temperature >= outside_hot.outside_wall_temperature
            >= outside_hot.inside_wall_temperature >= outside_hot.inside_bulk_temperature)


def test_one_failed_probe_marks_incomplete_envelope() -> None:
    hx = build_hx()
    pi, po = RecordingProvider(1.1, []), RecordingProvider(0.9, [])
    original = ti._solve_wall_temperature_probe
    calls = 0

    def one_failure(*args, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["max_iterations"] = 1 if calls == 1 else 25
        return original(*args, **kwargs)

    with patch.object(ti, "_solve_wall_temperature_probe", side_effect=one_failure):
        result = envelope(hx, pi, po)
    codes = {warning.code for warning in result.warnings}
    assert sum(probe.converged for probe in result.probes) == 3
    assert "wall_temperature_probe_not_converged" in codes
    assert "wall_temperature_envelope_incomplete" in codes
    assert "wall_temperature_envelope_0d_estimate" in codes


def test_simulation_and_rating_result_api() -> None:
    hx = build_hx()
    pi, po = RecordingProvider(1.1, []), RecordingProvider(0.9, [])
    simulation = hx.simulate(
        HXSideInput(pi, 5.0, 650.0, 101325.0),
        HXSideInput(po, 7.0, 300.0, 101325.0),
    )
    assert simulation.inside_wall_temperature_mean == simulation.thermal_state.inside_wall_temperature
    assert simulation.outside_wall_temperature_mean == simulation.thermal_state.outside_wall_temperature
    assert simulation.inside_wall_temperature_min_estimate == simulation.wall_temperature_envelope.inside_min
    assert simulation.outside_wall_temperature_max_estimate == simulation.wall_temperature_envelope.outside_max

    rating = hx.rate(
        BalanceSideSpec(pi, 101325.0, m_dot=5.0, T_in=650.0, T_out=520.0),
        BalanceSideSpec(po, 101325.0, m_dot=7.0, T_in=300.0, T_out=390.0),
    )
    assert rating.inside_wall_temperature_mean == rating.thermal_state.inside_wall_temperature
    assert rating.outside_wall_temperature_mean == rating.thermal_state.outside_wall_temperature
    assert rating.inside_wall_temperature_max_estimate == rating.wall_temperature_envelope.inside_max
    assert rating.outside_wall_temperature_min_estimate == rating.wall_temperature_envelope.outside_min
    assert rating.wall_temperature_envelope.method == simulation.wall_temperature_envelope.method


def main() -> None:
    test_four_pairs_properties_coefficients_and_extrema()
    test_signed_heat_flow_and_wall_ordering_both_directions()
    test_one_failed_probe_marks_incomplete_envelope()
    test_simulation_and_rating_result_api()
    print("ALL WALL-TEMPERATURE ENVELOPE CHECKS PASSED")


if __name__ == "__main__":
    main()
