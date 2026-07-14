# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""
Smoke test for the v0.5.3 finite heated-length (entrance-region) correction
in core.heat_transfer.internal_flow.internal_length_correction, and its
wiring through heat_transfer_coefficient_internal_diagnostics /
core.heat_transfer.thermal_iteration.

Run:
    python -m core.tests.internal_length_correction_test
"""

from __future__ import annotations

import math

from core.geometry.tube import BareTube
from core.geometry.bundle import TubeBundle
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.models.bare_tube import BareTubeHeatExchanger
from core.heat_transfer.thermal_iteration import solve_iterative_thermal_state
from core.heat_transfer.internal_flow import (
    FluidProps,
    internal_length_correction,
    heat_transfer_coefficient_internal_diagnostics,
    nusselt_gnielinski,
    reynolds_number,
)


_D = 0.02
_PROPS = FluidProps(rho=1.0, mu=2.0e-5, k=0.03, cp=1007.0)
_FLOW_AREA = math.pi * _D * _D / 4.0


def _turbulent_m_dot() -> float:
    m_dot = 0.15
    v = m_dot / (_PROPS.rho * _FLOW_AREA)
    Re = reynolds_number(_PROPS.rho, v, _D, _PROPS.mu)
    assert Re > 4000.0, Re
    return m_dot


def test_known_finite_ratio() -> None:
    print("length correction: known D_i/L_heated gives 1 + (D/L)^(2/3) exactly")
    D_i = 0.02
    L_heated = 2.0
    factor, warnings = internal_length_correction(D_i, L_heated)
    expected = 1.0 + (D_i / L_heated) ** (2.0 / 3.0)
    assert math.isclose(factor, expected, rel_tol=1e-12)
    assert factor > 1.0
    assert warnings == []


def test_large_length_tends_to_one() -> None:
    print("length correction: L_heated/D_i -> large gives factor -> 1")
    D_i = 0.02
    factor, warnings = internal_length_correction(D_i, 1.0e9)
    assert math.isclose(factor, 1.0, abs_tol=1e-5)
    assert warnings == []


def test_missing_length_skips_with_info_warning() -> None:
    print("length correction: L_heated=None skips correction (factor=1) with info warning")
    factor, warnings = internal_length_correction(0.02, None)
    assert factor == 1.0
    codes = {w.code for w in warnings}
    assert "tube_ht_length_correction_unavailable" in codes, codes


def test_invalid_length_warns_and_skips() -> None:
    print("length correction: non-finite/non-positive L_heated warns and skips (factor=1)")
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        factor, warnings = internal_length_correction(0.02, bad)
        assert factor == 1.0, (bad, factor)
        codes = {w.code for w in warnings}
        assert "tube_ht_length_correction_invalid" in codes, (bad, codes)


def test_wired_into_diagnostics_exactly_once() -> None:
    print("length correction: applied exactly once inside heat_transfer_coefficient_internal_diagnostics")
    m_dot = _turbulent_m_dot()
    D_i = _D
    L_heated = 1.5

    diag_no_length = heat_transfer_coefficient_internal_diagnostics(
        m_dot=m_dot, tube_inner_diameter=D_i, flow_area=_FLOW_AREA, props=_PROPS,
    )
    diag_with_length = heat_transfer_coefficient_internal_diagnostics(
        m_dot=m_dot, tube_inner_diameter=D_i, flow_area=_FLOW_AREA, props=_PROPS,
        L_heated=L_heated,
    )

    expected_factor = 1.0 + (D_i / L_heated) ** (2.0 / 3.0)
    assert math.isclose(diag_with_length.length_correction, expected_factor, rel_tol=1e-12)
    assert math.isclose(diag_no_length.length_correction, 1.0, abs_tol=1e-12)

    # Applied exactly once: Nu_corrected == Nu_base * length_correction here
    # (no wall-temperature correction requested in this case).
    assert math.isclose(
        diag_with_length.Nu_corrected,
        diag_with_length.Nu_base * diag_with_length.length_correction,
        rel_tol=1e-9,
    )
    assert diag_with_length.Nu_corrected > diag_no_length.Nu_base


def test_multi_pass_uses_per_pass_length_not_total() -> None:
    print("length correction: L_heated is tube.length_effective, independent of n_passes_tube")

    def build_bundle(n_passes: int) -> TubeBundle:
        tube = BareTube(
            D_i=25e-3 - 2 * 1.5e-3, D_o=25e-3,
            length_total=2.8, length_effective=2.8, wall_k=50.0,
        )
        return TubeBundle(
            tube=tube, n_rows=36, n_tubes_per_row=56,
            pitch_transverse=35e-3, pitch_longitudinal=35e-3,
            layout="staggered", n_passes_tube=n_passes,
            flow_arrangement="counterflow",
        )

    inside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
    )
    outside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
    )
    p = 101_325.0

    states = []
    for n_passes in (1, 2, 4):
        hx = BareTubeHeatExchanger(build_bundle(n_passes))
        # internal_flow_area_per_pass changes with n_passes (fewer tubes per
        # pass -> higher velocity/Re), so alfa_i is not expected to be
        # identical across n_passes -- what must be identical is the
        # heated-length value feeding the length correction, which the
        # module resolves from tube.length_effective directly, never from
        # bundle.internal_length_total (which scales with n_passes_tube).
        assert hx.bundle.tube.length_effective == 2.8
        assert hx.bundle.internal_length_total == n_passes * hx.bundle.tube.length_total
        states.append(n_passes)

    # Directly confirm the value used by thermal_iteration for L_heated.
    hx1 = BareTubeHeatExchanger(build_bundle(1))
    hx4 = BareTubeHeatExchanger(build_bundle(4))
    assert hx1.bundle.tube.length_effective == hx4.bundle.tube.length_effective == 2.8


def test_no_double_application_via_thermal_iteration() -> None:
    print("length correction: thermal_iteration applies the length correction exactly once")

    tube = BareTube(
        D_i=25e-3 - 2 * 1.5e-3, D_o=25e-3,
        length_total=2.8, length_effective=2.8, wall_k=50.0,
    )
    bundle = TubeBundle(
        tube=tube, n_rows=36, n_tubes_per_row=56,
        pitch_transverse=35e-3, pitch_longitudinal=35e-3,
        layout="staggered", n_passes_tube=2,
        flow_arrangement="counterflow",
    )
    hx = BareTubeHeatExchanger(bundle)

    inside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
    )
    outside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
    )
    p = 101_325.0

    state = solve_iterative_thermal_state(
        hx,
        m_dot_inside=18_220.0 / 3600.0, m_dot_outside=28_380.0 / 3600.0,
        inside_provider=inside_provider, outside_provider=outside_provider,
        T_in_inside=400.0 + 273.15, T_in_outside=30.0 + 273.15,
        p_inside=p, p_outside=p,
    )
    diag = state.diagnostics

    D_i = tube.D_i
    L_heated = tube.length_effective
    expected_length_factor = 1.0 + (D_i / L_heated) ** (2.0 / 3.0)
    assert math.isclose(diag.inside_length_correction, expected_length_factor, rel_tol=1e-6)

    # Combined correction must equal length * wall-temperature exactly once
    # each -- not length applied twice, nor folded again into alfa_corrected.
    assert math.isclose(
        diag.inside_combined_correction,
        diag.inside_length_correction * diag.inside_wall_temperature_correction,
        rel_tol=1e-9,
    )
    assert math.isclose(
        diag.inside_Nu_corrected,
        diag.inside_Nu_base * diag.inside_combined_correction,
        rel_tol=1e-9,
    )
    assert math.isclose(
        diag.inside_alfa_corrected,
        diag.inside_Nu_corrected * state.inside_bulk_props.k / D_i,
        rel_tol=1e-6,
    )


def main() -> None:
    test_known_finite_ratio()
    test_large_length_tends_to_one()
    test_missing_length_skips_with_info_warning()
    test_invalid_length_warns_and_skips()
    test_wired_into_diagnostics_exactly_once()
    test_multi_pass_uses_per_pass_length_not_total()
    test_no_double_application_via_thermal_iteration()

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
