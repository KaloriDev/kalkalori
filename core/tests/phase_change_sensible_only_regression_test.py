# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Sensible-only regression test for the v0.6.0 phase-change integration.

BareTubeHeatExchanger.simulate()/.rate() now call
core.phase_change.integration.apply_phase_change /
core.phase_change.rating_integration.apply_phase_change_to_rating after the
existing sensible-only driver (core.models.simulation.run_simulation /
core.models.rating.run_rating). This test proves that wrapping does not
perturb the sensible-only numbers at all: for a case with no phase-change
capability, .simulate()/.rate() must reproduce run_simulation()/run_rating()
bit-for-bit on every field named in spec section 38 (Q, T_out_inside,
T_out_outside, alfa_i, alfa_outside_dry, U, UA, straight-tube friction,
tube entrance/exit dp, outside dp_drag, outside dp_acceleration).

Uses the same toy geometry/providers as core/tests/simulation_smoke.py and
core/tests/heat_balance_rating_smoke.py (no CoolProp/IAPWS/PsychroLib
required), run WITHOUT going through the CoolProp-based gas-mixture
capability path at all (a ConstantPropertyProvider is never phase-change
capable), so this test is independent of the phase_change unit tests above.

Run:
    pytest -q core/tests/phase_change_sensible_only_regression_test.py
"""

from __future__ import annotations

import math

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.rating import run_rating
from core.models.heat_balance import close_heat_balance
from core.models.simulation import HXSideInput, run_simulation
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


def _bundle() -> TubeBundle:
    tube = BareTube(D_i=25e-3 - 2 * 1.5e-3, D_o=25e-3, length_total=2.8, length_effective=2.8, wall_k=50.0)
    return TubeBundle(
        tube=tube, n_rows=36, n_tubes_per_row=56,
        pitch_transverse=35e-3, pitch_longitudinal=35e-3,
        layout="staggered", n_passes_tube=2, flow_arrangement="counterflow",
    )


def _inside_provider() -> ConstantPropertyProvider:
    return ConstantPropertyProvider(FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0))


def _outside_provider() -> ConstantPropertyProvider:
    return ConstantPropertyProvider(FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0))


def _assert_fields_match(a, b) -> None:
    assert a.q == b.q
    assert a.T_out_inside == b.T_out_inside
    assert a.T_out_outside == b.T_out_outside
    assert a.inside_alfa_mean == b.inside_alfa_mean
    assert a.outside_alfa_mean == b.outside_alfa_mean
    assert a.U_mean == b.U_mean
    assert a.UA == b.UA
    assert a.inside_dp_straight_tube_friction == b.inside_dp_straight_tube_friction
    assert a.inside_dp_tube_entrances == b.inside_dp_tube_entrances
    assert a.inside_dp_tube_exits == b.inside_dp_tube_exits
    assert a.outside_dp_drag == b.outside_dp_drag
    assert a.outside_dp_acceleration == b.outside_dp_acceleration


def _assert_finite_property_state(state) -> None:
    assert all(
        math.isfinite(value)
        for value in (state.T, state.p, state.rho, state.cp, state.mu, state.k, state.Pr)
    )


def test_simulate_matches_run_simulation_bit_for_bit_when_not_capable() -> None:
    hx = BareTubeHeatExchanger(_bundle())
    inside = HXSideInput(provider=_inside_provider(), m_dot=5.06, T_in=303.15, p=101_325.0)
    outside = HXSideInput(provider=_outside_provider(), m_dot=7.88, T_in=673.15, p=101_325.0)

    expected = run_simulation(hx, inside, outside)
    actual = hx.simulate(inside, outside)

    _assert_fields_match(actual, expected)
    assert actual.outside_phase_change.capable is False
    assert actual.inside_phase_change.capable is False
    assert actual.outside_phase_change.Q_sensible == actual.q
    assert actual.outside_phase_change.Q_total == actual.q
    assert actual.inside_phase_change.Q_sensible == actual.q
    assert actual.inside_phase_change.Q_total == actual.q

    # Public endpoint accessors are direct views of the hydraulic source of
    # truth; no presentation-layer provider evaluation is involved.
    assert actual.inside_properties_inlet is actual.final_result.tube_bundle_hydraulic.inlet
    assert actual.inside_properties_midpoint is actual.final_result.tube_bundle_hydraulic.midpoint
    assert actual.inside_properties_outlet is actual.final_result.tube_bundle_hydraulic.outlet
    assert actual.inside_properties_inlet.T == inside.T_in
    assert actual.inside_properties_outlet.T == actual.T_out_inside
    assert actual.inside_properties_midpoint.T == 0.5 * (
        actual.inside_properties_inlet.T + actual.inside_properties_outlet.T
    )
    for state in (
        actual.inside_properties_inlet,
        actual.inside_properties_midpoint,
        actual.inside_properties_outlet,
    ):
        _assert_finite_property_state(state)
        assert state.props == inside.provider.at(T=state.T, p=state.p)

    outside_hydraulic = actual.outside_tube_bank_hydraulic
    assert actual.outside_properties_inlet is outside_hydraulic.inlet
    assert actual.outside_properties_midpoint is outside_hydraulic.midpoint
    assert actual.outside_properties_outlet is outside_hydraulic.outlet
    assert actual.outside_properties_inlet.T == outside.T_in
    assert actual.outside_properties_outlet.T == actual.T_out_outside
    for state in (
        actual.outside_properties_inlet,
        actual.outside_properties_midpoint,
        actual.outside_properties_outlet,
    ):
        _assert_finite_property_state(state)
        assert state.props == outside.provider.at(T=state.T, p=state.p)
        assert math.isclose(
            state.face_mass_flux * outside_hydraulic.face_area,
            outside.m_dot,
            rel_tol=1e-12,
        )


def test_simulate_iterate_false_matches_run_simulation_bit_for_bit() -> None:
    hx = BareTubeHeatExchanger(_bundle())
    inside = HXSideInput(provider=_inside_provider(), m_dot=5.06, T_in=303.15, p=101_325.0)
    outside = HXSideInput(provider=_outside_provider(), m_dot=7.88, T_in=673.15, p=101_325.0)

    expected = run_simulation(hx, inside, outside, iterate=False)
    actual = hx.simulate(inside, outside, iterate=False)

    _assert_fields_match(actual, expected)


def test_rate_matches_run_rating_bit_for_bit_when_not_capable() -> None:
    hx = BareTubeHeatExchanger(_bundle())
    inside = BalanceSideSpec(provider=_inside_provider(), p=101_325.0, m_dot=5.06, T_in=303.15, T_out=333.15)
    outside = BalanceSideSpec(provider=_outside_provider(), p=101_325.0, m_dot=7.88, T_in=673.15)

    closed_balance = close_heat_balance(inside, outside)
    expected = run_rating(hx, closed_balance)
    actual = hx.rate(inside, outside)

    assert actual.Q_required == expected.Q_required
    assert actual.UA_required == expected.UA_required
    assert actual.UA_actual == expected.UA_actual
    assert actual.U_mean == expected.U_mean
    assert actual.alfa_i == expected.alfa_i
    assert actual.alfa_o == expected.alfa_o
    assert actual.overdesign_factor == expected.overdesign_factor
    assert actual.inside_dp_straight_tube_friction == expected.inside_dp_straight_tube_friction
    assert actual.inside_dp_tube_entrances == expected.inside_dp_tube_entrances
    assert actual.inside_dp_tube_exits == expected.inside_dp_tube_exits
    assert actual.outside_dp_drag == expected.outside_dp_drag
    assert actual.outside_dp_acceleration == expected.outside_dp_acceleration
    assert actual.outside_phase_change.capable is False
    assert actual.inside_phase_change.capable is False
    assert actual.outside_phase_change.Q_sensible == actual.Q_required
    assert actual.outside_phase_change.Q_total == actual.Q_required
    assert actual.inside_phase_change.Q_sensible == actual.Q_required
    assert actual.inside_phase_change.Q_total == actual.Q_required
    assert actual.inside_properties_inlet is actual.final_result.tube_bundle_hydraulic.inlet
    assert actual.inside_properties_outlet.T == inside.T_out
    assert actual.outside_properties_inlet is actual.final_result.outside_tube_bank_hydraulic.inlet
    assert actual.outside_properties_outlet.T == actual.closed_balance.outside.T_out
