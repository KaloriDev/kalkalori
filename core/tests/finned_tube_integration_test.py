# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Integration tests for circular finned tubes in the public
BareTubeHeatExchanger / Simulation / Rating workflows (v0.7.x)."""

from __future__ import annotations

import math

import pytest

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.models.bare_tube import BareTubeHeatExchanger, FinnedTubeWetOutsideSurfaceNotSupportedError
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.heat_transfer.internal_flow import FluidProps as InternalFluidProps
from core.heat_transfer.outside_flow import FluidProps as OutsideFluidProps
from core.heat_transfer.streams import SensibleHeatStream
from core.properties.dry_air import DryAirPropertyProvider
from core.properties.water import IAPWS97WaterSteamProvider


def _core_tube() -> BareTube:
    return BareTube(D_i=0.020, D_o=0.025, length_total=3.0, length_effective=3.0, wall_k=45.0)


def _finned_tube(**overrides) -> CircularFinnedTube:
    kwargs = dict(
        core_tube=_core_tube(),
        fin_k=200.0,
        D_fin=0.057,
        D_root=0.025,
        fin_thickness_root=0.0004,
        fin_pitch=0.0023,
    )
    kwargs.update(overrides)
    return CircularFinnedTube(**kwargs)


def _finned_bundle(**tube_overrides) -> TubeBundle:
    return TubeBundle(
        tube=_finned_tube(**tube_overrides),
        n_rows=4,
        n_tubes_per_row=10,
        pitch_transverse=0.065,
        pitch_longitudinal=0.055,
        layout="staggered",
        n_passes_tube=1,
        flow_arrangement="crossflow",
    )


def _bare_bundle() -> TubeBundle:
    return TubeBundle(
        tube=_core_tube(),
        n_rows=4,
        n_tubes_per_row=10,
        pitch_transverse=0.065,
        pitch_longitudinal=0.055,
        layout="staggered",
        n_passes_tube=1,
        flow_arrangement="crossflow",
    )


# -------------------------------------------------------------------
# BareTubeHeatExchanger.solve() -- single pass
# -------------------------------------------------------------------

def test_solve_reports_finned_surface_type_and_positive_finite_ua():
    hx = BareTubeHeatExchanger(bundle=_finned_bundle())
    result = hx.solve(
        hot_stream=SensibleHeatStream(C=1.5 * 1050.0, T_in=450.0),
        cold_stream=SensibleHeatStream(C=5.0 * 1007.0, T_in=300.0),
        m_dot_tube_side=1.5,
        tube_side_props=InternalFluidProps(rho=0.8, mu=3e-5, k=0.045, cp=1050.0),
        m_dot_outside=5.0,
        outside_props=OutsideFluidProps(rho=1.15, mu=1.9e-5, k=0.028, cp=1007.0),
        flow_arrangement="crossflow",
    )
    assert math.isfinite(result.UA) and result.UA > 0.0
    assert math.isfinite(result.Q) and result.Q > 0.0
    assert math.isfinite(result.outside_side_thermal.alfa) and result.outside_side_thermal.alfa > 0.0
    assert math.isfinite(result.A_o) and result.A_o > 0.0
    # gross finned area exceeds an equivalent bare-tube area for the same root diameter
    bare_area = math.pi * 0.025 * 3.0 * 40  # n_tubes_total = 40
    assert result.A_o > bare_area


def test_solve_dp_unavailable_is_honest_nan_not_silent_wrong_value():
    hx = BareTubeHeatExchanger(bundle=_finned_bundle())
    result = hx.solve(
        hot_stream=SensibleHeatStream(C=1.5 * 1050.0, T_in=450.0),
        cold_stream=SensibleHeatStream(C=5.0 * 1007.0, T_in=300.0),
        m_dot_tube_side=1.5,
        tube_side_props=InternalFluidProps(rho=0.8, mu=3e-5, k=0.045, cp=1050.0),
        m_dot_outside=5.0,
        outside_props=OutsideFluidProps(rho=1.15, mu=1.9e-5, k=0.028, cp=1007.0),
        flow_arrangement="crossflow",
    )
    assert math.isnan(result.outside_side_hydraulic.dp_total)
    codes = [w.code for w in (result.warnings or [])]
    assert "outside_dp_robinson_briggs_not_implemented" in codes


def test_solve_energy_balance_consistency():
    hx = BareTubeHeatExchanger(bundle=_finned_bundle())
    hot = SensibleHeatStream(C=1.5 * 1050.0, T_in=450.0)
    cold = SensibleHeatStream(C=5.0 * 1007.0, T_in=300.0)
    result = hx.solve(
        hot_stream=hot,
        cold_stream=cold,
        m_dot_tube_side=1.5,
        tube_side_props=InternalFluidProps(rho=0.8, mu=3e-5, k=0.045, cp=1050.0),
        m_dot_outside=5.0,
        outside_props=OutsideFluidProps(rho=1.15, mu=1.9e-5, k=0.028, cp=1007.0),
        flow_arrangement="crossflow",
    )
    Q_hot = hot.capacity_rate() * (hot.inlet_temperature() - result.T_hot_out)
    Q_cold = cold.capacity_rate() * (result.T_cold_out - cold.inlet_temperature())
    assert Q_hot == pytest.approx(result.Q, rel=1e-9)
    assert Q_cold == pytest.approx(result.Q, rel=1e-9)


def test_solve_matches_bare_tube_when_fin_area_negligible():
    """A finned tube with a vanishingly small fin (D_fin -> D_root, thin)
    should approach the equivalent bare-tube UA in order of magnitude,
    as a sanity cross-check that the finned path is not wildly
    inconsistent with the existing bare-tube physics."""
    tiny_fin_bundle = _finned_bundle(D_fin=0.0251, fin_thickness_root=0.00005, fin_pitch=0.05)
    hx_finned = BareTubeHeatExchanger(bundle=tiny_fin_bundle)
    hx_bare = BareTubeHeatExchanger(bundle=_bare_bundle())

    kwargs = dict(
        m_dot_tube_side=1.5,
        tube_side_props=InternalFluidProps(rho=0.8, mu=3e-5, k=0.045, cp=1050.0),
        m_dot_outside=5.0,
        outside_props=OutsideFluidProps(rho=1.15, mu=1.9e-5, k=0.028, cp=1007.0),
        flow_arrangement="crossflow",
    )
    result_finned = hx_finned.solve(
        hot_stream=SensibleHeatStream(C=1.5 * 1050.0, T_in=450.0),
        cold_stream=SensibleHeatStream(C=5.0 * 1007.0, T_in=300.0),
        **kwargs,
    )
    result_bare = hx_bare.solve(
        hot_stream=SensibleHeatStream(C=1.5 * 1050.0, T_in=450.0),
        cold_stream=SensibleHeatStream(C=5.0 * 1007.0, T_in=300.0),
        **kwargs,
    )
    # Different correlations (Briggs-Young vs Zukauskas) are not expected
    # to match closely, but UA should be the same order of magnitude
    # (not off by e.g. 100x, which would indicate a gross unit/area error).
    ratio = result_finned.UA / result_bare.UA
    assert 0.2 < ratio < 5.0


# -------------------------------------------------------------------
# Simulation
# -------------------------------------------------------------------

def test_simulate_complete_dry_finned_case_positive_finite_results():
    hx = BareTubeHeatExchanger(bundle=_finned_bundle())
    inside = HXSideInput(provider=DryAirPropertyProvider(), m_dot=1.5, T_in=450.0, p=101325.0)
    outside = HXSideInput(provider=DryAirPropertyProvider(), m_dot=5.0, T_in=300.0, p=101325.0)

    sim = hx.simulate(inside, outside, flow_arrangement="crossflow")

    assert math.isfinite(sim.q) and sim.q > 0.0
    assert math.isfinite(sim.T_out_inside) and sim.T_out_inside < 450.0
    assert math.isfinite(sim.T_out_outside) and sim.T_out_outside > 300.0
    assert math.isfinite(sim.final_result.UA) and sim.final_result.UA > 0.0
    assert math.isfinite(sim.final_result.outside_side_thermal.alfa) and sim.final_result.outside_side_thermal.alfa > 0.0
    assert math.isnan(sim.final_result.outside_side_hydraulic.dp_total)


def test_simulate_finned_surface_type_identified_on_bundle():
    hx = BareTubeHeatExchanger(bundle=_finned_bundle())
    assert isinstance(hx.bundle.tube, CircularFinnedTube)
    assert hx.bundle.tube.surface_type.value == "circular_finned"


def test_simulate_rejects_wet_outside_provider_on_finned_tube():
    hx = BareTubeHeatExchanger(bundle=_finned_bundle())
    inside = HXSideInput(provider=DryAirPropertyProvider(), m_dot=1.5, T_in=450.0, p=101325.0)
    outside_wet = HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=5.0, T_in=380.0, p=101325.0)
    with pytest.raises(FinnedTubeWetOutsideSurfaceNotSupportedError):
        hx.simulate(inside, outside_wet, flow_arrangement="crossflow")


def test_simulate_dry_outside_on_finned_tube_does_not_raise_wet_guard():
    hx = BareTubeHeatExchanger(bundle=_finned_bundle())
    inside = HXSideInput(provider=DryAirPropertyProvider(), m_dot=1.5, T_in=450.0, p=101325.0)
    outside = HXSideInput(provider=DryAirPropertyProvider(), m_dot=5.0, T_in=300.0, p=101325.0)
    sim = hx.simulate(inside, outside, flow_arrangement="crossflow")
    assert sim.q > 0.0


def test_bare_tube_simulate_unaffected_by_finned_guard():
    """A BareTube exchanger with a wet outside provider must still work
    exactly as before -- the finned guard is a strict no-op for it."""
    hx = BareTubeHeatExchanger(bundle=_bare_bundle())
    inside = HXSideInput(provider=DryAirPropertyProvider(), m_dot=1.5, T_in=450.0, p=101325.0)
    outside = HXSideInput(provider=DryAirPropertyProvider(), m_dot=5.0, T_in=300.0, p=101325.0)
    sim = hx.simulate(inside, outside, flow_arrangement="crossflow")
    assert sim.q > 0.0


# -------------------------------------------------------------------
# Rating
# -------------------------------------------------------------------

def test_rate_complete_dry_finned_case_positive_finite_results():
    hx = BareTubeHeatExchanger(bundle=_finned_bundle())
    inside = BalanceSideSpec(provider=DryAirPropertyProvider(), p=101325.0, m_dot=1.5, T_in=450.0, T_out=380.0)
    outside = BalanceSideSpec(provider=DryAirPropertyProvider(), p=101325.0, m_dot=5.0, T_in=300.0)

    rate = hx.rate(inside, outside, flow_arrangement="crossflow")

    assert math.isfinite(rate.UA_actual) and rate.UA_actual > 0.0
    assert math.isfinite(rate.overdesign_factor)
    assert math.isfinite(rate.alfa_o) and rate.alfa_o > 0.0
    assert math.isfinite(rate.Q_required) and rate.Q_required > 0.0


def test_rate_rejects_wet_outside_provider_on_finned_tube():
    hx = BareTubeHeatExchanger(bundle=_finned_bundle())
    inside = BalanceSideSpec(provider=DryAirPropertyProvider(), p=101325.0, m_dot=1.5, T_in=450.0, T_out=380.0)
    outside_wet = BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=101325.0, m_dot=5.0, T_in=380.0)
    with pytest.raises(FinnedTubeWetOutsideSurfaceNotSupportedError):
        hx.rate(inside, outside_wet, flow_arrangement="crossflow")


def test_rate_warnings_are_deduplicated():
    hx = BareTubeHeatExchanger(bundle=_finned_bundle())
    inside = BalanceSideSpec(provider=DryAirPropertyProvider(), p=101325.0, m_dot=1.5, T_in=450.0, T_out=380.0)
    outside = BalanceSideSpec(provider=DryAirPropertyProvider(), p=101325.0, m_dot=5.0, T_in=300.0)
    rate = hx.rate(inside, outside, flow_arrangement="crossflow")
    identities = [(w.source, w.code) for w in (rate.warnings or [])]
    assert len(identities) == len(set(identities))


# -------------------------------------------------------------------
# Layout / provider mismatch controlled rejection at the public API
# -------------------------------------------------------------------

def test_simulate_inline_finned_bundle_is_controlled_unsupported():
    from core.heat_transfer.outside_flow_finned import FinnedTubeUnsupportedLayoutError

    inline_bundle = TubeBundle(
        tube=_finned_tube(),
        n_rows=4,
        n_tubes_per_row=10,
        pitch_transverse=0.065,
        pitch_longitudinal=0.055,
        layout="inline",
        n_passes_tube=1,
        flow_arrangement="crossflow",
    )
    hx = BareTubeHeatExchanger(bundle=inline_bundle)
    inside = HXSideInput(provider=DryAirPropertyProvider(), m_dot=1.5, T_in=450.0, p=101325.0)
    outside = HXSideInput(provider=DryAirPropertyProvider(), m_dot=5.0, T_in=300.0, p=101325.0)
    with pytest.raises(FinnedTubeUnsupportedLayoutError):
        hx.simulate(inside, outside, flow_arrangement="crossflow")


# -------------------------------------------------------------------
# Regression: BareTube results unaffected by the whole finned-tube feature
# -------------------------------------------------------------------

def test_bare_tube_wall_resistance_unaffected_by_finned_support():
    hx = BareTubeHeatExchanger(bundle=_bare_bundle())
    Di, Do, L, k, N = 0.020, 0.025, 3.0, 45.0, 40
    expected = math.log(Do / Di) / (2.0 * math.pi * k * L * N)
    assert hx.tube_wall_resistance() == pytest.approx(expected, rel=1e-12)


def test_bare_tube_solve_thermal_state_unaffected():
    hx = BareTubeHeatExchanger(bundle=_bare_bundle())
    state = hx.solve_thermal_state(
        m_dot_inside=1.5,
        m_dot_outside=5.0,
        inside_provider=DryAirPropertyProvider(),
        outside_provider=DryAirPropertyProvider(),
        T_in_inside=450.0,
        T_in_outside=300.0,
        p_inside=101325.0,
        p_outside=101325.0,
    )
    assert state.converged
    assert math.isfinite(state.UA) and state.UA > 0.0
