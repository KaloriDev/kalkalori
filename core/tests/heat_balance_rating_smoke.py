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


def build_auto_topology_bundle(*, n_passes_transverse: int) -> TubeBundle:
    """Fixed 6-pass geometry whose AUTO topology is varied by one field."""
    tube = BareTube(
        D_i=25e-3 - 2 * 1.5e-3,
        D_o=25e-3,
        length_total=2.8,
        length_effective=2.8,
        wall_k=50.0,
    )
    return TubeBundle(
        tube=tube,
        n_rows=6,
        n_tubes_per_row=56,
        pitch_transverse=35e-3,
        pitch_longitudinal=35e-3,
        layout="staggered",
        n_passes_tube=6,
        n_passes_transverse=n_passes_transverse,
        flow_arrangement="auto",
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
    print("rating: geometry sized exactly for its own achievable duty -> overdesign ~= 0")

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

    # Since v0.5.3, rate() sources alfa_i/alfa_o/U/UA from the converged,
    # wall/length-corrected iterative thermal state (solve_thermal_state),
    # not from the same uncorrected single pass simulate() uses internally.
    # So simulate()'s own outlet temperatures are no longer the self-
    # consistent "zero overdesign" baseline for rate() -- that baseline must
    # instead come from solve_thermal_state()'s own achievable operating
    # point. Tight tolerances are used so the two solve_thermal_state calls
    # (this one, and the one inside rate() below) converge to matching
    # precision.
    tight_iter_kwargs = dict(
        max_iterations=200,
        wall_temperature_tolerance_K=1e-6,
        relative_alfa_tolerance=1e-9,
    )
    thermal_state = hx.solve_thermal_state(
        m_dot_inside=inside_m_dot, m_dot_outside=outside_m_dot,
        inside_provider=inside_provider, outside_provider=outside_provider,
        T_in_inside=inside_T_in, T_in_outside=outside_T_in,
        p_inside=p, p_outside=p,
        **tight_iter_kwargs,
    )
    assert thermal_state.converged, thermal_state
    T_out_inside0 = 2.0 * thermal_state.inside_bulk_temperature - inside_T_in
    T_out_outside0 = 2.0 * thermal_state.outside_bulk_temperature - outside_T_in

    # Baseline: feed solve_thermal_state()'s own achievable outlet
    # temperatures back into rate().
    inside_bal0 = BalanceSideSpec(provider=inside_provider, p=p, m_dot=inside_m_dot, T_in=inside_T_in, T_out=T_out_inside0)
    outside_bal0 = BalanceSideSpec(provider=outside_provider, p=p, m_dot=outside_m_dot, T_in=outside_T_in, T_out=T_out_outside0)

    res0 = hx.rate(inside_bal0, outside_bal0, **tight_iter_kwargs)
    print(f"  overdesign_factor (baseline)         : {res0.overdesign_factor:.3e}")
    print(f"  A_required / A_o                     : {res0.A_required:.4f} / {res0.A_o:.4f}")
    assert res0.closed_balance.warnings is None, res0.closed_balance.warnings
    assert abs(res0.overdesign_factor) < 1e-4, res0.overdesign_factor
    assert abs(res0.A_required - res0.A_o) / res0.A_o < 1e-4
    assert math.isclose(res0.UA_actual, thermal_state.UA, rel_tol=1e-6)
    assert math.isclose(res0.alfa_i, thermal_state.alfa_i, rel_tol=1e-6)
    assert math.isclose(res0.alfa_o, thermal_state.alfa_o, rel_tol=1e-6)

    # include_simulation bridge: Q_achievable should match sim.q closely.
    # (sim.q is simulate()'s own -- uncorrected -- achievable duty; it is
    # unaffected by which T_out values were used to build inside_bal0/
    # outside_bal0, since to_hx_side_input() only carries T_in/m_dot/p.)
    res0_sim = hx.rate(inside_bal0, outside_bal0, include_simulation=True)
    assert res0_sim.simulation is not None
    assert abs(res0_sim.Q_achievable - sim.q) / sim.q < 1e-6

    # Lower/higher demanded effectiveness at fixed m_dot/T_in -> overdesign
    # strictly decreasing as demanded effectiveness increases.
    eps_actual = res0.closed_balance.effectiveness

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

    # Positive margin and shortfall share one canonical UA-based result.
    # The historical area relation remains a physical invariant only.
    for result in (res_lo, res_hi):
        assert result.UA_process == result.UA_required
        assert result.overdesign_factor == result.ua_margin
        assert math.isclose(
            result.overdesign_factor,
            result.UA_actual / result.UA_process - 1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        assert math.isclose(
            result.overdesign_factor,
            result.A_o / result.A_required - 1.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )

    return res0, res_lo, res_hi


def test_rating_uses_auto_arrangement_resolved_from_tube_circuit() -> None:
    """AUTO 6/6 and 6/1 circuiting must reach Rating's epsilon-NTU inverse."""
    crossflow_bundle = build_auto_topology_bundle(n_passes_transverse=6)
    counterflow_bundle = build_auto_topology_bundle(n_passes_transverse=1)
    multipass_bundle = build_auto_topology_bundle(n_passes_transverse=2)
    assert crossflow_bundle.flow_arrangement_resolved == "crossflow"
    assert counterflow_bundle.flow_arrangement_resolved == "counterflow"
    assert multipass_bundle.flow_arrangement_resolved == "crossflow"

    inside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
    )
    outside_provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
    )
    inside = BalanceSideSpec(
        provider=inside_provider,
        p=101_325.0,
        m_dot=3.0,
        T_in=c_to_k(400.0),
    )
    outside = BalanceSideSpec(
        provider=outside_provider,
        p=101_325.0,
        m_dot=5.0,
        T_in=c_to_k(30.0),
    )

    # The same effectiveness target closes the same heat balance for both
    # geometrically identical bundles.  Constant properties keep their
    # working-condition U identical, isolating the arrangement passed to the
    # inverse epsilon-NTU relation used by Rating.
    effectiveness = 0.60
    crossflow = BareTubeHeatExchanger(crossflow_bundle).rate(
        inside, outside, effectiveness=effectiveness
    )
    counterflow = BareTubeHeatExchanger(counterflow_bundle).rate(
        inside, outside, effectiveness=effectiveness
    )
    multipass_auto = BareTubeHeatExchanger(multipass_bundle).rate(
        inside, outside, effectiveness=effectiveness
    )
    multipass_counterflow_override = BareTubeHeatExchanger(multipass_bundle).rate(
        inside,
        outside,
        effectiveness=effectiveness,
        flow_arrangement="counterflow",
    )

    assert math.isclose(
        crossflow.closed_balance.Q,
        counterflow.closed_balance.Q,
        rel_tol=1e-12,
    )
    assert math.isclose(crossflow.A_o, counterflow.A_o, rel_tol=1e-12)
    assert math.isclose(crossflow.U_mean, counterflow.U_mean, rel_tol=1e-10)
    assert math.isclose(crossflow.UA_actual, counterflow.UA_actual, rel_tol=1e-10)

    assert counterflow.A_required < crossflow.A_required
    assert crossflow.A_required / counterflow.A_required > 1.05
    assert counterflow.overdesign_factor > crossflow.overdesign_factor
    assert counterflow.overdesign_factor - crossflow.overdesign_factor > 1e-2

    auto_warning_code = "FLOW_ARRANGEMENT_AUTO_MULTIPASS_APPROXIMATION"
    assert auto_warning_code in {warning.code for warning in multipass_auto.warnings or []}
    assert auto_warning_code not in {
        warning.code for warning in multipass_counterflow_override.warnings or []
    }
    assert math.isclose(
        multipass_counterflow_override.A_required,
        counterflow.A_required,
        rel_tol=1e-10,
    )
    assert math.isclose(
        multipass_counterflow_override.overdesign_factor,
        counterflow.overdesign_factor,
        rel_tol=1e-10,
    )


def test_simulation_uses_auto_arrangement_resolved_from_tube_circuit() -> None:
    """AUTO 6/6 and 6/1 circuiting must reach Simulation's epsilon-NTU path."""
    crossflow_hx = BareTubeHeatExchanger(
        build_auto_topology_bundle(n_passes_transverse=6)
    )
    counterflow_hx = BareTubeHeatExchanger(
        build_auto_topology_bundle(n_passes_transverse=1)
    )
    assert crossflow_hx.bundle.flow_arrangement_resolved == "crossflow"
    assert counterflow_hx.bundle.flow_arrangement_resolved == "counterflow"
    inside = HXSideInput(
        provider=ConstantPropertyProvider(
            FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
        ),
        p=101_325.0,
        m_dot=3.0,
        T_in=c_to_k(400.0),
    )
    outside = HXSideInput(
        provider=ConstantPropertyProvider(
            FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
        ),
        p=101_325.0,
        m_dot=5.0,
        T_in=c_to_k(30.0),
    )

    crossflow = crossflow_hx.simulate(inside, outside, iterate=False)
    counterflow = counterflow_hx.simulate(inside, outside, iterate=False)

    C_min = min(inside.m_dot * 1007.0, outside.m_dot * 1180.0)
    Q_max = C_min * (inside.T_in - outside.T_in)
    crossflow_effectiveness = crossflow.q / Q_max
    counterflow_effectiveness = counterflow.q / Q_max

    assert counterflow.q - crossflow.q > 1e-3 * Q_max
    assert counterflow_effectiveness > crossflow_effectiveness
    assert counterflow_effectiveness - crossflow_effectiveness > 1e-3


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
    test_rating_uses_auto_arrangement_resolved_from_tube_circuit()
    test_simulation_uses_auto_arrangement_resolved_from_tube_circuit()

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
