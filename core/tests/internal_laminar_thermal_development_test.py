"""Source-neutral thermal-entry arithmetic and production solver regressions."""
import math
import pytest

from core.heat_transfer.internal_flow import (
    FluidProps, heat_transfer_coefficient_internal_diagnostics,
    nusselt_laminar_thermal_entry,
)
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.tests.enhancement_integration_test import bundle, inputs


def test_independent_graetz_anchor():
    # Gz=1000, hence Gz^(2/3)=100; Nu=3.66+66.8/5=17.02.
    assert nusselt_laminar_thermal_entry(500, 200, .02, 2) == pytest.approx(17.02)


def test_fully_developed_limit_and_high_pr():
    assert nusselt_laminar_thermal_entry(500, 200, .02, 1e12) == pytest.approx(3.66)
    assert nusselt_laminar_thermal_entry(500, 470, .02, 5) > nusselt_laminar_thermal_entry(500, 170, .02, 5) > 3.66


@pytest.mark.parametrize('value', [0, -1, math.inf, math.nan])
def test_invalid_thermal_length(value):
    with pytest.raises(ValueError):
        nusselt_laminar_thermal_entry(500, 200, .02, value)


def test_legacy_no_length_and_exactly_once_correction():
    args = dict(m_dot=math.pi*.02*.01*500/4, tube_inner_diameter=.02,
                flow_area=math.pi*.02**2/4, props=FluidProps(850, .01, .1, 2000))
    legacy = heat_transfer_coefficient_internal_diagnostics(**args)
    entry = heat_transfer_coefficient_internal_diagnostics(**args, L_heated=2)
    assert legacy.Nu_corrected == legacy.Nu_base == 3.66
    assert entry.Nu_corrected == pytest.approx(17.02)
    assert entry.Nu_base*entry.combined_correction == entry.Nu_corrected
    assert entry.wall_temperature_correction == 1


@pytest.mark.parametrize('iterate', [False, True])
def test_rating_and_simulation_use_heated_length(iterate):
    b = bundle(passes=2)
    hx = BareTubeHeatExchanger(b)
    inside, outside = inputs(b)
    simulation = hx.simulate(inside, outside, iterate=iterate, tube_side_enhancement=None)
    expected = nusselt_laminar_thermal_entry(500, 8, .012, .3)
    assert simulation.inside_alfa_mean == pytest.approx(expected*.5/.012)
    rating = hx.rate(
        BalanceSideSpec(provider=inside.provider, p=inside.p, m_dot=inside.m_dot,
                        T_in=300, T_out=simulation.T_out_inside),
        BalanceSideSpec(provider=outside.provider, p=outside.p, m_dot=outside.m_dot, T_in=340),
        include_simulation=True, tube_side_enhancement=None)
    assert rating.alfa_i == pytest.approx(expected*.5/.012)
    assert rating.simulation.inside_alfa_mean == pytest.approx(rating.alfa_i)
    assert simulation.inside_properties_midpoint.friction_factor == pytest.approx(64/500)
    assert hx.simulate(inside, outside, iterate=iterate).q == simulation.q
