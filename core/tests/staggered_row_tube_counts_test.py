# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Focused tests for v0.7.10 alternating (staggered) tube-row counts.

``TubeBundle.n_tubes_per_row`` remains the sole tube-row-count input, but is
now normalized on construction:

- ``layout="inline"``: normalized to the nearest integer.
- ``layout="staggered"``: normalized to the nearest half-integer, describing
  an alternating odd/even row pattern (``n_tubes_per_row_odd``/``_even``).

See ``core/geometry/bundle.py`` for the full contract.
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.heat_transfer.outside_dispatch import (
    evaluate_outside_hydraulics,
    evaluate_outside_thermal,
)
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


def _tube() -> BareTube:
    return BareTube(
        D_i=0.020, D_o=0.024, length_total=3.0, length_effective=2.5, wall_k=20.0
    )


def _bundle(*, n_rows: int, n_tubes_per_row: float, layout: str, **overrides) -> TubeBundle:
    defaults = dict(
        tube=_tube(),
        n_rows=n_rows,
        n_tubes_per_row=n_tubes_per_row,
        pitch_transverse=0.05,
        pitch_longitudinal=0.04,
        layout=layout,
        n_passes_tube=1,
    )
    defaults.update(overrides)
    return TubeBundle(**defaults)


# ---------------------------------------------------------------------------
# Normalization contract
# ---------------------------------------------------------------------------


def test_inline_integer_input_is_unchanged() -> None:
    bundle = _bundle(n_rows=4, n_tubes_per_row=8, layout="inline")
    assert bundle.n_tubes_per_row == 8.0
    assert bundle.n_tubes_per_row_odd == 8
    assert bundle.n_tubes_per_row_even == 8
    assert bundle.tube_count_normalization_warnings == ()


def test_inline_rounds_to_nearest_integer_with_warning() -> None:
    bundle = _bundle(n_rows=4, n_tubes_per_row=6.7, layout="inline")
    assert bundle.n_tubes_per_row == 7.0
    assert bundle.n_tubes_per_row_odd == 7
    assert bundle.n_tubes_per_row_even == 7
    codes = [w.code for w in bundle.tube_count_normalization_warnings]
    assert codes == ["TUBES_PER_ROW_NORMALIZED"]
    assert "6.7" in bundle.tube_count_normalization_warnings[0].message
    assert "7.0" in bundle.tube_count_normalization_warnings[0].message
    assert "inline" in bundle.tube_count_normalization_warnings[0].message


def test_staggered_integer_input_is_unchanged() -> None:
    bundle = _bundle(n_rows=4, n_tubes_per_row=8, layout="staggered")
    assert bundle.n_tubes_per_row == 8.0
    assert bundle.n_tubes_per_row_odd == 8
    assert bundle.n_tubes_per_row_even == 8
    assert bundle.tube_count_normalization_warnings == ()


def test_staggered_half_step_input_is_exact_no_warning() -> None:
    bundle = _bundle(n_rows=4, n_tubes_per_row=6.5, layout="staggered")
    assert bundle.n_tubes_per_row == 6.5
    assert bundle.n_tubes_per_row_odd == 7
    assert bundle.n_tubes_per_row_even == 6
    assert bundle.tube_count_normalization_warnings == ()


def test_staggered_arbitrary_input_rounds_to_nearest_half_with_warning() -> None:
    bundle = _bundle(n_rows=4, n_tubes_per_row=6.7, layout="staggered")
    assert bundle.n_tubes_per_row == 6.5
    assert bundle.n_tubes_per_row_odd == 7
    assert bundle.n_tubes_per_row_even == 6
    codes = [w.code for w in bundle.tube_count_normalization_warnings]
    assert codes == ["TUBES_PER_ROW_NORMALIZED"]


def test_staggered_near_integer_rounds_up_with_warning() -> None:
    bundle = _bundle(n_rows=4, n_tubes_per_row=7.9, layout="staggered")
    assert bundle.n_tubes_per_row == 8.0
    assert bundle.n_tubes_per_row_odd == 8
    assert bundle.n_tubes_per_row_even == 8
    codes = [w.code for w in bundle.tube_count_normalization_warnings]
    assert codes == ["TUBES_PER_ROW_NORMALIZED"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(6.1, 6.0), (6.2, 6.0), (6.3, 6.5), (6.7, 6.5), (6.8, 7.0)],
)
def test_staggered_deterministic_nearest_half_rounding(raw: float, expected: float) -> None:
    bundle = _bundle(n_rows=4, n_tubes_per_row=raw, layout="staggered")
    assert bundle.n_tubes_per_row == expected


@pytest.mark.parametrize(
    ("raw", "expected"), [(6.1, 6.0), (6.3, 6.0), (6.7, 7.0), (7.9, 8.0)]
)
def test_inline_deterministic_nearest_integer_rounding(raw: float, expected: float) -> None:
    bundle = _bundle(n_rows=4, n_tubes_per_row=raw, layout="inline")
    assert bundle.n_tubes_per_row == expected


def test_non_finite_and_non_positive_inputs_are_rejected() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _bundle(n_rows=4, n_tubes_per_row=bad, layout="staggered")


# ---------------------------------------------------------------------------
# Exact total tube count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_rows", "n_tubes_per_row", "expected_total"),
    [
        (4, 6.5, 26),
        (5, 6.5, 33),
        (6, 6.5, 39),
        (5, 8.0, 40),
    ],
)
def test_exact_total_tube_count_for_staggered_alternating_rows(
    n_rows: int, n_tubes_per_row: float, expected_total: int
) -> None:
    bundle = _bundle(n_rows=n_rows, n_tubes_per_row=n_tubes_per_row, layout="staggered")
    assert bundle.n_tubes_total == expected_total
    assert isinstance(bundle.n_tubes_total, int)


def test_exact_total_is_not_rows_times_periodic_average() -> None:
    bundle = _bundle(n_rows=5, n_tubes_per_row=6.5, layout="staggered")
    naive_total = bundle.n_rows * bundle.n_tubes_per_row
    assert naive_total == pytest.approx(32.5)
    assert bundle.n_tubes_total == 33
    assert bundle.n_tubes_total != naive_total


# ---------------------------------------------------------------------------
# Surface area uses the exact total
# ---------------------------------------------------------------------------


def test_total_areas_use_exact_tube_total_for_odd_row_staggered_bank() -> None:
    tube = _tube()
    bundle = _bundle(n_rows=5, n_tubes_per_row=6.5, layout="staggered", tube=tube)
    assert bundle.n_tubes_total == 33
    assert bundle.total_inner_area == pytest.approx(33 * tube.area_inner)
    assert bundle.total_outer_area == pytest.approx(33 * tube.area_outer)
    # Guard against a future regression back to n_rows * n_tubes_per_row.
    assert bundle.total_outer_area != pytest.approx(
        bundle.n_rows * bundle.n_tubes_per_row * tube.area_outer
    )


# ---------------------------------------------------------------------------
# Frontal geometry contract: uses the normalized periodic average
# ---------------------------------------------------------------------------


def test_frontal_flow_area_uses_normalized_periodic_average() -> None:
    tube = _tube()
    bundle = _bundle(n_rows=5, n_tubes_per_row=6.5, layout="staggered", tube=tube)
    assert bundle.n_tubes_per_row_odd == 7
    assert bundle.n_tubes_per_row_even == 6
    expected = 6.5 * bundle.pitch_transverse * tube.length_effective
    assert bundle.frontal_flow_area == pytest.approx(expected)
    # Not the max row count...
    assert bundle.frontal_flow_area != pytest.approx(
        bundle.n_tubes_per_row_odd * bundle.pitch_transverse * tube.length_effective
    )
    # ...and not n_tubes_total / n_rows either.
    naive_average = bundle.n_tubes_total / bundle.n_rows
    assert naive_average == pytest.approx(6.6)
    assert bundle.frontal_flow_area != pytest.approx(
        naive_average * bundle.pitch_transverse * tube.length_effective
    )


# ---------------------------------------------------------------------------
# Tube-side hydraulics use the exact total
# ---------------------------------------------------------------------------


def test_tube_side_effective_pass_count_uses_exact_total_for_odd_rows() -> None:
    bundle = _bundle(
        n_rows=5, n_tubes_per_row=6.5, layout="staggered", n_passes_tube=2
    )
    assert bundle.n_tubes_total == 33
    assert bundle.n_tubes_per_pass_effective == pytest.approx(33 / 2)
    assert bundle.internal_flow_area_per_pass == pytest.approx(
        (33 / 2) * bundle.tube.flow_area
    )


# ---------------------------------------------------------------------------
# Downstream float propagation: outside heat transfer and pressure drop
# ---------------------------------------------------------------------------


def _outside_props() -> ConstantPropertyProvider:
    return ConstantPropertyProvider(
        FluidTransportProperties(rho=1.1, mu=1.8e-5, k=0.026, cp=1007.0)
    )


def test_half_row_count_propagates_through_outside_heat_transfer() -> None:
    bundle = _bundle(n_rows=5, n_tubes_per_row=6.5, layout="staggered")
    provider = _outside_props()
    result = evaluate_outside_thermal(
        bundle=bundle, m_dot=0.5, props=provider.at(300.0, 101325.0)
    )
    assert math.isfinite(result.alpha_physical)
    assert result.alpha_physical > 0.0


def test_half_row_count_propagates_through_outside_pressure_drop() -> None:
    bundle = _bundle(n_rows=5, n_tubes_per_row=6.5, layout="staggered")
    provider = _outside_props()
    result = evaluate_outside_hydraulics(
        bundle=bundle,
        m_dot=0.5,
        property_provider=provider,
        temperature_in=300.0,
        temperature_out=305.0,
        pressure=101325.0,
    )
    assert math.isfinite(result.dp_total)
    assert result.dp_total >= 0.0


# ---------------------------------------------------------------------------
# Warning contract: topology + normalization warnings both aggregate
# ---------------------------------------------------------------------------


def test_normalization_and_topology_warnings_both_aggregate_in_bundle_warnings() -> None:
    bundle = _bundle(
        n_rows=6,
        n_tubes_per_row=6.7,
        layout="staggered",
        n_passes_tube=6,
        n_passes_transverse=2,
        flow_arrangement="auto",
    )
    codes = {w.code for w in bundle.warnings}
    assert "TUBES_PER_ROW_NORMALIZED" in codes
    assert "FLOW_ARRANGEMENT_AUTO_MULTIPASS_APPROXIMATION" in codes
    assert bundle.warnings == bundle.geometry_warnings
    assert set(bundle.geometry_warnings) == set(
        bundle.topology_warnings + bundle.tube_count_normalization_warnings
    )


def test_no_normalization_warning_for_exact_legacy_geometry() -> None:
    bundle = _bundle(n_rows=4, n_tubes_per_row=8, layout="inline")
    assert bundle.warnings == ()


# ---------------------------------------------------------------------------
# Rating / Simulation end-to-end regression
# ---------------------------------------------------------------------------


def _sensible_props(rho: float) -> ConstantPropertyProvider:
    return ConstantPropertyProvider(
        FluidTransportProperties(rho=rho, mu=1.0e-3, k=0.6, cp=4180.0)
    )


def test_legacy_integer_geometry_simulation_regression() -> None:
    bundle = _bundle(n_rows=4, n_tubes_per_row=8, layout="inline", n_passes_tube=1)
    hx = BareTubeHeatExchanger(bundle)
    inside = HXSideInput(provider=_sensible_props(1000.0), m_dot=0.5, T_in=350.0, p=101325.0)
    outside = HXSideInput(provider=_outside_props(), m_dot=0.5, T_in=300.0, p=101325.0)
    result = hx.simulate(inside, outside, iterate=False)
    assert math.isfinite(result.final_result.Q)
    assert result.final_result.Q > 0.0


def test_staggered_half_row_geometry_simulation_runs_end_to_end() -> None:
    bundle = _bundle(n_rows=5, n_tubes_per_row=6.5, layout="staggered", n_passes_tube=1)
    hx = BareTubeHeatExchanger(bundle)
    inside = HXSideInput(provider=_sensible_props(1000.0), m_dot=0.5, T_in=350.0, p=101325.0)
    outside = HXSideInput(provider=_outside_props(), m_dot=0.5, T_in=300.0, p=101325.0)
    result = hx.simulate(inside, outside, iterate=False)
    assert math.isfinite(result.final_result.Q)
    assert result.final_result.Q > 0.0
    assert bundle.n_tubes_total == 33
