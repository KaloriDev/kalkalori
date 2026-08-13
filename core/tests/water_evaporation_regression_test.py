"""Final v0.6.3 public physics and performance regression matrix."""

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.phase_change.water_evaporator import rate_water_evaporator
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.water import IAPWS97WaterSteamProvider, water_steam_props_iapws97


P = 1.0e6
HOT = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1005.0)
)


def _hx(rows=10, columns=10):
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.020, D_o=0.024, length_total=4.0,
                length_effective=4.0, wall_k=16.0,
                tube_orientation=TubeOrientation.VERTICAL_UPWARD,
            ),
            n_rows=rows, n_tubes_per_row=columns,
            pitch_transverse=0.04, pitch_longitudinal=0.04,
            layout="inline", n_passes_tube=1, flow_arrangement="crossflow",
        )
    )


def _public_rating(inlet, outlet, *, hx=None, m_dot=1.0):
    provider = IAPWS97WaterSteamProvider()
    inside = BalanceSideSpec(
        provider=provider, p=P, m_dot=m_dot, h_in=inlet.h, h_out=outlet.h
    )
    outside = BalanceSideSpec(
        provider=HOT, p=101325.0, m_dot=30.0, T_in=700.0
    )
    return (hx or _hx()).rate(inside, outside)


@pytest.mark.parametrize(
    ("inlet", "outlet"),
    [
        (water_steam_props_iapws97(T=350.0, p=P), water_steam_props_iapws97(p=P, x=0.5)),
        (water_steam_props_iapws97(p=P, x=0.0), water_steam_props_iapws97(p=P, x=0.5)),
        (water_steam_props_iapws97(p=P, x=0.2), water_steam_props_iapws97(p=P, x=0.8)),
        (water_steam_props_iapws97(p=P, x=0.2), water_steam_props_iapws97(p=P, x=1.0)),
        (water_steam_props_iapws97(T=350.0, p=P), water_steam_props_iapws97(p=P, x=1.0)),
        (water_steam_props_iapws97(T=350.0, p=P), water_steam_props_iapws97(T=520.0, p=P)),
        (water_steam_props_iapws97(p=P, x=1.0e-8), water_steam_props_iapws97(p=P, x=0.1)),
        (water_steam_props_iapws97(p=P, x=0.9), water_steam_props_iapws97(p=P, x=1.0 - 1.0e-8)),
    ],
)
def test_public_evaporation_physics_invariants(inlet, outlet):
    result = _public_rating(inlet, outlet)
    water = result.inside_phase_change
    assert water.mass_flow_total == 1.0
    assert water.m_dot_condensate == 0.0
    assert water.h_out >= water.h_in
    assert water.h_out == pytest.approx(outlet.h, rel=2e-12)
    for quality in (water.quality_in, water.quality_out):
        assert quality is None or 0.0 <= quality <= 1.0
    assert all(value >= 0.0 for value in (
        water.Q_preheat, water.Q_evaporation, water.Q_superheat,
        water.A_preheat, water.A_evaporation, water.A_superheat,
        water.zone_UA_preheat, water.zone_UA_evaporation,
        water.zone_UA_superheat, water.m_dot_evaporated,
    ))
    assert water.Q_total == pytest.approx(
        water.mass_flow_total * (water.h_out - water.h_in)
    )
    assert water.Q_total == pytest.approx(
        water.Q_preheat + water.Q_evaporation + water.Q_superheat
    )
    assert water.A_total == pytest.approx(
        water.A_preheat + water.A_evaporation + water.A_superheat
    )
    assert water.UA_total == pytest.approx(
        water.zone_UA_preheat
        + water.zone_UA_evaporation
        + water.zone_UA_superheat
    )
    assert result.U_mean == pytest.approx(water.UA_total / water.A_total)
    effective_x_in = 0.0 if inlet.quality is None else inlet.quality
    effective_x_out = 1.0 if outlet.quality is None else outlet.quality
    assert water.m_dot_evaporated == pytest.approx(
        water.mass_flow_total * (effective_x_out - effective_x_in)
    )
    numeric_diagnostics = (
        water.Q_total, water.A_total, water.UA_total,
        water.inside_alpha_equivalent, water.inside_alpha_area_weighted,
        water.heat_flux_residual, result.A_required, result.UA_required,
        result.UA_actual, result.overdesign_factor, result.ua_margin,
    )
    assert all(math.isfinite(value) for value in numeric_diagnostics)
    assert math.isnan(result.inside_dp_total)
    assert not hasattr(water, "runtime_s")
    assert not hasattr(water, "solution")


@pytest.mark.parametrize(
    ("rows", "columns", "m_dot", "G_range"),
    [
        (20, 20, 0.05, (0.1, 1.0)),
        (2, 5, 1.0, (300.0, 330.0)),
        (1, 1, 1.0, (3000.0, 3300.0)),
    ],
)
def test_low_medium_high_mass_flux_remain_finite(rows, columns, m_dot, G_range):
    result = _public_rating(
        water_steam_props_iapws97(p=P, x=0.2),
        water_steam_props_iapws97(p=P, x=0.8),
        hx=_hx(rows, columns),
        m_dot=m_dot,
    )
    water = result.inside_phase_change
    assert G_range[0] <= water.mass_flux <= G_range[1]
    assert all(math.isfinite(value) and value > 0.0 for value in (
        water.zone_alpha_evaporation,
        water.zone_U_evaporation,
        water.A_evaporation,
        water.zone_UA_evaporation,
    ))


def test_self_consistent_low_medium_high_heat_flux_programs_are_ordered():
    hx = _hx()
    inlet = water_steam_props_iapws97(p=P, x=0.2)
    outlet = water_steam_props_iapws97(p=P, x=0.8)
    solutions = [
        rate_water_evaporator(
            hx,
            inlet_state=inlet,
            outlet_state=outlet,
            mass_flow_water=0.1,
            outside_provider=HOT,
            mass_flow_outside=30.0,
            T_in_outside=T_hot,
            p_outside=101325.0,
            orientation=TubeOrientation.VERTICAL_UPWARD,
        )
        for T_hot in (480.0, 600.0, 800.0)
    ]
    heat_fluxes = [solution.heat_flux_inner_evaporation for solution in solutions]
    assert heat_fluxes == sorted(heat_fluxes)
    assert heat_fluxes[0] < 10_000.0 < heat_fluxes[1] < heat_fluxes[2]
    for solution in solutions:
        assert solution.heat_flux_converged
        assert solution.heat_flux_iterations > 0
        assert solution.heat_flux_residual <= 1.0e-8
        assert solution.property_evaluations < 20
        assert solution.cache_hits > 0
        assert math.isfinite(solution.runtime_s) and solution.runtime_s >= 0.0
