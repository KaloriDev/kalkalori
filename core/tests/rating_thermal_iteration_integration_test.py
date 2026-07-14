# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""
Call-path integration test (v0.5.3): proves that Rating actually consumes
the converged iterative thermal state, and that its alfa_i/alfa_o/U/UA are
not later overwritten by the separate (uncorrected) legacy
BareTubeHeatExchanger.solve() pass.

Prior to v0.5.3, core.models.rating.run_rating computed alfa_i/alfa_o/UA
from a single uncorrected solve() pass and never called
solve_iterative_thermal_state at all, so the wall-temperature/finite-length
corrections were entirely invisible to Rating.

This test patches the symbol at its actual point of use --
``core.models.rating.solve_iterative_thermal_state`` -- with a sentinel
IterativeThermalState carrying distinctive alfa_i/alfa_o/U/UA values, and
verifies HXRatingResult reports exactly those values. Patching only
``core.heat_transfer.thermal_iteration.solve_iterative_thermal_state`` would
NOT be sufficient, because ``core.models.rating`` imports the function by
name (``from core.heat_transfer.thermal_iteration import
solve_iterative_thermal_state``), binding its own module-level reference.

Run:
    python -m core.tests.rating_thermal_iteration_integration_test
"""

from __future__ import annotations

import math
from unittest import mock

from core.geometry.tube import BareTube
from core.geometry.bundle import TubeBundle
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.heat_transfer.thermal_iteration import (
    IterativeThermalState,
    ThermalIterationDiagnostics,
)


def c_to_k(t_c: float) -> float:
    return t_c + 273.15


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


_SENTINEL_ALFA_I = 123.456
_SENTINEL_ALFA_O = 234.567
_SENTINEL_U = 16.789
_SENTINEL_UA = 3456.789


def _sentinel_thermal_state(*, T_in_inside: float, T_in_outside: float) -> IterativeThermalState:
    """A self-consistent-looking but clearly-fake IterativeThermalState."""
    props = FluidTransportProperties(rho=1.0, mu=2.0e-5, k=0.03, cp=1007.0)
    diagnostics = ThermalIterationDiagnostics(
        inside_Nu_base=10.0,
        inside_Nu_corrected=9.0,
        inside_length_correction=1.05,
        inside_wall_temperature_correction=0.857,
        inside_combined_correction=1.05 * 0.857,
        inside_alfa_base=100.0,
        inside_alfa_corrected=_SENTINEL_ALFA_I,
        outside_Nu_base=20.0,
        outside_Nu_corrected=21.0,
        outside_wall_property_correction=1.05,
    )
    return IterativeThermalState(
        inside_bulk_temperature=0.5 * (T_in_inside + T_in_outside) + 10.0,
        outside_bulk_temperature=0.5 * (T_in_inside + T_in_outside) - 10.0,
        inside_wall_temperature=0.5 * (T_in_inside + T_in_outside),
        outside_wall_temperature=0.5 * (T_in_inside + T_in_outside),
        inside_bulk_props=props,
        inside_wall_props=props,
        outside_bulk_props=props,
        outside_wall_props=props,
        alfa_i=_SENTINEL_ALFA_I,
        alfa_o=_SENTINEL_ALFA_O,
        U=_SENTINEL_U,
        UA=_SENTINEL_UA,
        iterations=7,
        converged=True,
        residual=1e-3,
        diagnostics=diagnostics,
        inside_provider_name="SentinelProvider",
        outside_provider_name="SentinelProvider",
        warnings=(),
    )


def test_rating_consumes_sentinel_thermal_state() -> None:
    print("call-path: patching core.models.rating.solve_iterative_thermal_state feeds HXRatingResult")

    hx = BareTubeHeatExchanger(build_bundle())

    inside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
    )
    outside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
    )
    p = 101_325.0
    inside = BalanceSideSpec(
        provider=inside_provider, p=p, m_dot=18_220.0 / 3600.0,
        T_in=c_to_k(30.0), T_out=c_to_k(150.0),
    )
    outside = BalanceSideSpec(
        provider=outside_provider, p=p, m_dot=28_380.0 / 3600.0,
        T_in=c_to_k(400.0), T_out=c_to_k(300.0),
    )

    def _fake_solve_iterative_thermal_state(hx_arg, *, T_in_inside, T_in_outside, **_kwargs):
        return _sentinel_thermal_state(T_in_inside=T_in_inside, T_in_outside=T_in_outside)

    with mock.patch(
        "core.models.rating.solve_iterative_thermal_state",
        side_effect=_fake_solve_iterative_thermal_state,
    ) as patched:
        result = hx.rate(inside, outside)
        assert patched.called, "run_rating must call solve_iterative_thermal_state"

    # The sentinel values must reach HXRatingResult verbatim -- proving the
    # iterative result is consumed and not overwritten by the separate
    # (real, uncorrected) BareTubeHeatExchanger.solve() pass that also runs
    # inside run_rating for area/regime/hydraulic diagnostics.
    assert result.alfa_i == _SENTINEL_ALFA_I, result.alfa_i
    assert result.alfa_o == _SENTINEL_ALFA_O, result.alfa_o
    assert result.U_mean == _SENTINEL_U, result.U_mean
    assert result.UA_actual == _SENTINEL_UA, result.UA_actual
    assert result.thermal_state.UA == _SENTINEL_UA
    assert result.thermal_state.alfa_i == _SENTINEL_ALFA_I

    # ua_margin/overdesign_factor must be derived FROM the sentinel UA (not
    # some other, real, value) -- U_mean is used as the working-condition U
    # for A_required, so overdesign_factor is a function of _SENTINEL_U/_UA.
    expected_A_required = result.UA_required / _SENTINEL_U
    assert math.isclose(result.A_required, expected_A_required, rel_tol=1e-9)
    expected_overdesign = result.A_o / expected_A_required - 1.0
    assert math.isclose(result.overdesign_factor, expected_overdesign, rel_tol=1e-9)


def test_rating_calls_solve_iterative_thermal_state_with_closed_balance_state() -> None:
    print("call-path: solve_iterative_thermal_state is called with the closed balance's inlet state")

    hx = BareTubeHeatExchanger(build_bundle())
    inside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
    )
    outside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
    )
    p = 101_325.0
    inside = BalanceSideSpec(
        provider=inside_provider, p=p, m_dot=18_220.0 / 3600.0,
        T_in=c_to_k(30.0), T_out=c_to_k(150.0),
    )
    outside = BalanceSideSpec(
        provider=outside_provider, p=p, m_dot=28_380.0 / 3600.0,
        T_in=c_to_k(400.0), T_out=c_to_k(300.0),
    )

    def _fake(hx_arg, *, T_in_inside, T_in_outside, m_dot_inside, m_dot_outside, **_kwargs):
        assert T_in_inside == c_to_k(30.0)
        assert T_in_outside == c_to_k(400.0)
        assert math.isclose(m_dot_inside, 18_220.0 / 3600.0)
        assert math.isclose(m_dot_outside, 28_380.0 / 3600.0)
        return _sentinel_thermal_state(T_in_inside=T_in_inside, T_in_outside=T_in_outside)

    with mock.patch(
        "core.models.rating.solve_iterative_thermal_state", side_effect=_fake
    ) as patched:
        hx.rate(inside, outside)
        assert patched.call_count == 1


def main() -> None:
    test_rating_consumes_sentinel_thermal_state()
    test_rating_calls_solve_iterative_thermal_state_with_closed_balance_state()

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
