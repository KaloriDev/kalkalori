# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""
Smoke test for heat-balance closure, ntu_from_effectiveness, and Rating
(overdesign / surface margin) -- v0.5.1.

Runs WITHOUT CoolProp / IAPWS / PsychroLib: uses only ConstantPropertyProvider,
so every closure/rating relation can be checked with tight (near-exact)
tolerances.

Run:
    python -m core.tests.heat_balance_rating_smoke
"""

from __future__ import annotations

import math

from core.geometry.tube import BareTube
from core.geometry.bundle import TubeBundle
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.models.heat_balance import BalanceSideSpec, close_heat_balance
from core.heat_transfer.ntu import effectiveness_ntu, ntu_from_effectiveness


def c_to_k(t_c: float) -> float:
    return t_c + 273.15


def cp_only_provider(cp: float) -> ConstantPropertyProvider:
    """Provider for closure-only tests: only cp matters (solve() isn't called)."""
    return ConstantPropertyProvider(FluidTransportProperties(rho=1.0, mu=1e-5, k=0.03, cp=cp))


def implied_duty(side) -> float:
    return side.C * abs(side.T_out - side.T_in)


def build_bundle() -> TubeBundle:
    """Same geometry as core/tests/simulation_smoke.py."""
    tube = BareTube(
        D_i=25e-3 - 2 * 1.5e-3,
        D_o=25e-3,
        length_total=2.8,
        length_effective=2.8,
        wall_k=50.0,
    )
    return TubeBundle(
        tube=tube,
        n_rows=36,
        n_tubes_per_row=56,
        pitch_transverse=35e-3,
        pitch_longitudinal=35e-3,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement="counterflow",
    )


# ---------------------------------------------------------------------------
# Part A - heat-balance closure
# ---------------------------------------------------------------------------
def test_closure_popular_variant() -> None:
    print("closure: popular variant (outside complete -> Q; inside T_in+T_out -> m_dot)")
    inside = BalanceSideSpec(
        provider=cp_only_provider(1007.0), p=101_325.0,
        T_in=c_to_k(30.0), T_out=c_to_k(200.0),
    )
    outside = BalanceSideSpec(
        provider=cp_only_provider(1180.0), p=101_325.0,
        m_dot=8.0, T_in=c_to_k(400.0), T_out=c_to_k(300.0),
    )
    cb = close_heat_balance(inside, outside)

    Q_expected = 8.0 * 1180.0 * 100.0
    m_dot_expected = Q_expected / (1007.0 * (200.0 - 30.0))

    assert math.isclose(cb.Q, Q_expected, rel_tol=1e-9), (cb.Q, Q_expected)
    assert math.isclose(cb.inside.m_dot, m_dot_expected, rel_tol=1e-9)
    assert math.isclose(implied_duty(cb.inside), cb.Q, rel_tol=1e-9)
    assert math.isclose(implied_duty(cb.outside), cb.Q, rel_tol=1e-9)
    assert cb.warnings is None


def test_closure_solve_T_out() -> None:
    print("closure: 'solve T_out' variant (Q given, both sides m_dot+T_in only)")
    inside = BalanceSideSpec(provider=cp_only_provider(1007.0), p=101_325.0, m_dot=2.0, T_in=c_to_k(30.0))
    outside = BalanceSideSpec(provider=cp_only_provider(1180.0), p=101_325.0, m_dot=3.0, T_in=c_to_k(400.0))
    Q = 500_000.0
    cb = close_heat_balance(inside, outside, Q=Q)

    C_hot = 3.0 * 1180.0
    C_cold = 2.0 * 1007.0
    T_hot_out_expected = c_to_k(400.0) - Q / C_hot
    T_cold_out_expected = c_to_k(30.0) + Q / C_cold

    assert math.isclose(cb.outside.T_out, T_hot_out_expected, rel_tol=1e-9)
    assert math.isclose(cb.inside.T_out, T_cold_out_expected, rel_tol=1e-9)
    assert math.isclose(cb.Q, Q, rel_tol=1e-12)
    assert math.isclose(implied_duty(cb.inside), Q, rel_tol=1e-9)
    assert math.isclose(implied_duty(cb.outside), Q, rel_tol=1e-9)


def test_closure_solve_m() -> None:
    print("closure: 'solve m' variant (Q given, both sides T_in+T_out only)")
    inside = BalanceSideSpec(provider=cp_only_provider(1007.0), p=101_325.0, T_in=c_to_k(30.0), T_out=c_to_k(100.0))
    outside = BalanceSideSpec(provider=cp_only_provider(1180.0), p=101_325.0, T_in=c_to_k(400.0), T_out=c_to_k(300.0))
    Q = 400_000.0
    cb = close_heat_balance(inside, outside, Q=Q)

    m_dot_inside_expected = Q / (1007.0 * 70.0)
    m_dot_outside_expected = Q / (1180.0 * 100.0)

    assert math.isclose(cb.inside.m_dot, m_dot_inside_expected, rel_tol=1e-9)
    assert math.isclose(cb.outside.m_dot, m_dot_outside_expected, rel_tol=1e-9)
    assert math.isclose(implied_duty(cb.inside), Q, rel_tol=1e-9)
    assert math.isclose(implied_duty(cb.outside), Q, rel_tol=1e-9)


def test_closure_effectiveness() -> None:
    print("closure: effectiveness variant (both m_dot+T_in known, T_out solved both sides)")
    inside = BalanceSideSpec(provider=cp_only_provider(1007.0), p=101_325.0, m_dot=2.0, T_in=c_to_k(30.0))
    outside = BalanceSideSpec(provider=cp_only_provider(1180.0), p=101_325.0, m_dot=3.0, T_in=c_to_k(400.0))
    eff = 0.5
    cb = close_heat_balance(inside, outside, effectiveness=eff)

    C_hot = 3.0 * 1180.0
    C_cold = 2.0 * 1007.0
    C_min = min(C_hot, C_cold)
    Q_expected = eff * C_min * (c_to_k(400.0) - c_to_k(30.0))

    assert math.isclose(cb.Q, Q_expected, rel_tol=1e-9)
    assert math.isclose(cb.effectiveness, eff, rel_tol=1e-9)
    assert math.isclose(implied_duty(cb.inside), cb.Q, rel_tol=1e-6)
    assert math.isclose(implied_duty(cb.outside), cb.Q, rel_tol=1e-6)


def test_closure_under_specified() -> None:
    print("closure: under-specified -> ValueError")
    inside = BalanceSideSpec(provider=cp_only_provider(1007.0), p=101_325.0, T_in=c_to_k(30.0))
    outside = BalanceSideSpec(provider=cp_only_provider(1180.0), p=101_325.0, T_in=c_to_k(400.0))
    try:
        close_heat_balance(inside, outside)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for under-specified balance")


def test_closure_over_specified() -> None:
    print("closure: over-specified -> ModelWarning (heat_balance_over_specified)")
    inside = BalanceSideSpec(
        provider=cp_only_provider(1007.0), p=101_325.0,
        T_in=c_to_k(30.0), T_out=c_to_k(200.0),
    )
    outside = BalanceSideSpec(
        provider=cp_only_provider(1180.0), p=101_325.0,
        m_dot=8.0, T_in=c_to_k(400.0), T_out=c_to_k(300.0),  # implies Q ~= 944 kW
    )
    cb = close_heat_balance(inside, outside, Q=800_000.0)  # explicit Q mismatches outside
    assert cb.warnings is not None
    codes = {w.code for w in cb.warnings}
    assert "heat_balance_over_specified" in codes, codes


# ---------------------------------------------------------------------------
# Part B - ntu_from_effectiveness round-trip + guard
# ---------------------------------------------------------------------------
def test_ntu_from_effectiveness_round_trip() -> None:
    print("ntu_from_effectiveness: round-trip for counterflow (Cr!=1, Cr==1), cocurrentflow, crossflow")

    def round_trip(
        C_hot: float, C_cold: float, UA: float, flow_arrangement: str,
        *, C_inside: float | None = None, C_outside: float | None = None,
    ) -> None:
        C_min = min(C_hot, C_cold)
        NTU_expected = UA / C_min
        eps = effectiveness_ntu(
            C_hot=C_hot, C_cold=C_cold, UA=UA, flow_arrangement=flow_arrangement,
            C_inside=C_inside, C_outside=C_outside,
        )
        NTU_rt = ntu_from_effectiveness(
            eps, C_hot, C_cold, flow_arrangement=flow_arrangement,
            C_inside=C_inside, C_outside=C_outside,
        )
        assert math.isclose(NTU_rt, NTU_expected, rel_tol=1e-6), (flow_arrangement, NTU_rt, NTU_expected)

    round_trip(C_hot=3540.0, C_cold=2014.0, UA=5000.0, flow_arrangement="counterflow")   # Cr != 1
    round_trip(C_hot=2000.0, C_cold=2000.0, UA=3000.0, flow_arrangement="counterflow")   # Cr == 1
    round_trip(C_hot=3540.0, C_cold=2014.0, UA=4000.0, flow_arrangement="cocurrentflow")
    # Crossflow (v0.5.2): outside mixed / inside unmixed requires C_inside/C_outside
    # to resolve which physical stream is C_min; see core.tests.ntu_crossflow_test
    # for the dedicated crossflow coverage (both branches, Cr limits, etc.).
    round_trip(
        C_hot=3540.0, C_cold=2014.0, UA=4000.0, flow_arrangement="crossflow",
        C_inside=3540.0, C_outside=2014.0,
    )


def test_ntu_from_effectiveness_guard() -> None:
    print("ntu_from_effectiveness: guard raises for unreachable effectiveness (cocurrentflow)")
    C_hot, C_cold = 3540.0, 2014.0
    C_min, C_max = min(C_hot, C_cold), max(C_hot, C_cold)
    C_r = C_min / C_max
    eps_max = 1.0 / (1.0 + C_r)
    try:
        ntu_from_effectiveness(eps_max * 1.05, C_hot, C_cold, flow_arrangement="cocurrentflow")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for effectiveness above eps_max")


# ---------------------------------------------------------------------------
# Part C - Rating (overdesign)
# ---------------------------------------------------------------------------
def test_rating() -> tuple:
    print("rating: geometry sized exactly for Q (from simulate on-zero) -> overdesign ~= 0")

    hx = BareTubeHeatExchanger(build_bundle())

    inside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
    )
    outside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
    )
    inside_m_dot = 18_220.0 / 3600.0
    outside_m_dot = 28_380.0 / 3600.0
    inside_T_in = c_to_k(30.0)
    outside_T_in = c_to_k(400.0)
    p = 101_325.0

    sim = hx.simulate(
        HXSideInput(provider=inside_provider, m_dot=inside_m_dot, T_in=inside_T_in, p=p),
        HXSideInput(provider=outside_provider, m_dot=outside_m_dot, T_in=outside_T_in, p=p),
    )

    # Baseline: feed simulate()'s own outlet temperatures back into rate().
    inside_bal0 = BalanceSideSpec(provider=inside_provider, p=p, m_dot=inside_m_dot, T_in=inside_T_in, T_out=sim.T_out_inside)
    outside_bal0 = BalanceSideSpec(provider=outside_provider, p=p, m_dot=outside_m_dot, T_in=outside_T_in, T_out=sim.T_out_outside)

    res0 = hx.rate(inside_bal0, outside_bal0)
    print(f"  overdesign_factor (baseline)         : {res0.overdesign_factor:.3e}")
    print(f"  A_required / A_o                     : {res0.A_required:.4f} / {res0.A_o:.4f}")
    assert res0.closed_balance.warnings is None, res0.closed_balance.warnings
    assert abs(res0.overdesign_factor) < 1e-6, res0.overdesign_factor
    assert abs(res0.A_required - res0.A_o) / res0.A_o < 1e-6

    # include_simulation bridge: Q_achievable should match sim.q closely.
    res0_sim = hx.rate(inside_bal0, outside_bal0, include_simulation=True)
    assert res0_sim.simulation is not None
    assert abs(res0_sim.Q_achievable - sim.q) / sim.q < 1e-6

    # Lower/higher demanded effectiveness at fixed m_dot/T_in -> overdesign
    # strictly decreasing as demanded effectiveness increases.
    C_hot = outside_m_dot * 1180.0
    C_cold = inside_m_dot * 1007.0
    C_min = min(C_hot, C_cold)
    Q_max = C_min * (outside_T_in - inside_T_in)
    eps_actual = sim.q / Q_max

    inside_bal_partial = BalanceSideSpec(provider=inside_provider, p=p, m_dot=inside_m_dot, T_in=inside_T_in)
    outside_bal_partial = BalanceSideSpec(provider=outside_provider, p=p, m_dot=outside_m_dot, T_in=outside_T_in)

    eps_lo = eps_actual * 0.7
    eps_hi = eps_actual + (1.0 - eps_actual) * 0.5

    res_lo = hx.rate(inside_bal_partial, outside_bal_partial, effectiveness=eps_lo)
    res_hi = hx.rate(inside_bal_partial, outside_bal_partial, effectiveness=eps_hi)

    print(f"  overdesign_factor lo/base/hi         : {res_lo.overdesign_factor:.4f} / {res0.overdesign_factor:.2e} / {res_hi.overdesign_factor:.4f}")

    assert res_lo.overdesign_factor > 0.0, res_lo.overdesign_factor
    assert res_hi.overdesign_factor < 0.0, res_hi.overdesign_factor
    assert res_hi.overdesign_factor < res0.overdesign_factor < res_lo.overdesign_factor

    return res0, res_lo, res_hi


def main() -> None:
    test_closure_popular_variant()
    test_closure_solve_T_out()
    test_closure_solve_m()
    test_closure_effectiveness()
    test_closure_under_specified()
    test_closure_over_specified()

    test_ntu_from_effectiveness_round_trip()
    test_ntu_from_effectiveness_guard()

    test_rating()

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
