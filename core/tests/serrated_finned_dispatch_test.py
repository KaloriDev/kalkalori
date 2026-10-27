# SPDX-License-Identifier: GPL-3.0-only
"""No explicit or automatic route may substitute solid-fin physics."""
from dataclasses import replace

import pytest

from core.geometry import (
    BareTube, SerratedCircularFinnedTube, SerratedFinPhysicsNotSupportedError,
    SerratedFlowAreaBasis, TubeBundle, TubeSurfaceType,
)
from core.heat_transfer.fin_efficiency import annular_fin_efficiency, overall_surface_efficiency
from core.heat_transfer.finned_tube_outside import calculate_finned_tube_outside_heat_transfer
from core.heat_transfer.outside_dispatch import (
    tube_surface_type, evaluate_outside_thermal, evaluate_outside_hydraulics,
    calculate_resistance_network,
)
from core.models import BalanceSideSpec
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.rating import run_rating
from core.models.simulation import HXSideInput, run_simulation
from core.phase_change.types import PhaseChangeMode
from core.pressure_drop.finned_tube_pressure_drop import calculate_finned_tube_pressure_drop, calculate_finned_tube_bank_hydraulics
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


PROPS = FluidTransportProperties(rho=1., mu=2e-5, k=.03, cp=1050.)


def bank(layout="staggered", depth=.007):
    t = SerratedCircularFinnedTube(
        core_tube=BareTube(.020, .0254, 1., 1., 45.), fin_k=45.,
        D_root=.0254, D_fin=.0534, fin_thickness_root=.0006,
        fin_pitch=.0032, fin_density_material=7850., segments_per_turn=40,
        serration_depth=depth, kerf_width_at_tip=.0003)
    return TubeBundle(t, 4, 3, .08, .08, layout, 1)


def test_distinct_surface_and_future_flow_area_bases():
    assert tube_surface_type(bank()) is TubeSurfaceType.SERRATED_HELICAL_FIN
    assert len(set(SerratedFlowAreaBasis)) == 3
    assert SerratedFlowAreaBasis.ORTHOGRAPHIC_GEOMETRIC_BLOCKAGE.value == "orthographic_geometric_blockage"
    assert SerratedFlowAreaBasis.SOURCE_EQUIVALENT_FLOW_AREA.value == "source_equivalent_flow_area"
    assert SerratedFlowAreaBasis.HYDRAULIC_PROVIDER_SPECIFIC_A_MIN.value == "hydraulic_provider_specific_A_min"


@pytest.mark.parametrize("layout", ["inline", "staggered"])
@pytest.mark.parametrize("depth", [0., .007])
@pytest.mark.parametrize("selector", ["auto", "briggs_young_1963",
    "frass_escoa_inline_solid_fin", "frass_escoa_staggered_solid_fin"])
def test_no_solid_htc_fallback(layout, depth, selector):
    b = bank(layout, depth)
    for calculate in (
        lambda: calculate_finned_tube_outside_heat_transfer(1., b, PROPS, provider=selector),
        lambda: evaluate_outside_thermal(bundle=b, m_dot=1., props=PROPS, finned_heat_transfer_provider=selector),
    ):
        with pytest.raises(SerratedFinPhysicsNotSupportedError, match="HTC.*pressure-drop.*no solid-fin fallback"):
            calculate()


@pytest.mark.parametrize("layout", ["inline", "staggered"])
@pytest.mark.parametrize("selector", ["auto", "robinson_briggs_1966", "frass_inline_solid_fin_191",
                                     "genic2006_staggered", "genic2006_inline"])
def test_no_solid_hydraulic_fallback(layout, selector):
    b = bank(layout)
    for calculate in (
        lambda: calculate_finned_tube_pressure_drop(1., b, PROPS, provider=selector),
        lambda: calculate_finned_tube_bank_hydraulics(1., b, inlet_props=PROPS, midpoint_props=PROPS,
            outlet_props=PROPS, pressure_drop_provider=selector),
        lambda: evaluate_outside_hydraulics(bundle=b, m_dot=1., inlet_props=PROPS,
            midpoint_props=PROPS, outlet_props=PROPS, finned_pressure_drop_provider=selector),
    ):
        with pytest.raises(SerratedFinPhysicsNotSupportedError, match="A_min.*fin efficiency.*no solid-fin fallback"):
            calculate()


@pytest.mark.parametrize("attribute", ["projected_blocking_area_per_length", "transverse_free_flow_gap",
    "diagonal_free_flow_gap", "minimum_free_flow_gap", "minimum_free_flow_area", "reference_flow_area",
    "maximum_to_face_velocity_ratio"])
def test_blockage_and_minimum_flow_area_are_unavailable(attribute):
    with pytest.raises(SerratedFinPhysicsNotSupportedError, match="A_min"):
        getattr(bank(), attribute)


def test_fin_efficiency_and_ua_rejected_even_with_supplied_film_coefficient():
    b = bank()
    for calculate in (
        lambda: annular_fin_efficiency(b.tube, 80.),
        lambda: overall_surface_efficiency(b.tube, .9),
        lambda: calculate_resistance_network(bundle=b, alpha_inside=1200.,
            outside_alpha_physical=80., resistance_core_wall=.001),
    ):
        with pytest.raises(SerratedFinPhysicsNotSupportedError, match="fin efficiency"):
            calculate()


@pytest.mark.parametrize("layout", ["inline", "staggered"])
@pytest.mark.parametrize("api", ["solve", "thermal_state", "rating", "simulation"])
@pytest.mark.parametrize("mode", [PhaseChangeMode.DISABLED, PhaseChangeMode.AUTO])
def test_public_workflows_cannot_bypass_unavailable_serrated_physics(layout, api, mode):
    hx = BareTubeHeatExchanger(bank(layout))
    outside = ConstantPropertyProvider(PROPS)
    inside = ConstantPropertyProvider(FluidTransportProperties(rho=900., mu=.0005, k=.6, cp=4000.))
    with pytest.raises(SerratedFinPhysicsNotSupportedError, match="geometry only"):
        if api == "solve":
            hx.solve(None, None, m_dot_tube_side=1.5, tube_side_props=inside.at(400., 101325.),
                m_dot_outside=1., outside_props=PROPS, alfa_o=80.)
        elif api == "thermal_state":
            hx.solve_thermal_state(m_dot_inside=1.5, m_dot_outside=1., inside_provider=inside,
                outside_provider=outside, T_in_inside=400., T_in_outside=600.,
                p_inside=101325., p_outside=101325.)
        else:
            cls = BalanceSideSpec if api == "rating" else HXSideInput
            i = cls(provider=inside, m_dot=1.5, T_in=400., p=101325., phase_change_mode=mode,
                    **({"T_out": 410.} if api == "rating" else {}))
            o = cls(provider=outside, m_dot=1., T_in=600., p=101325., phase_change_mode=mode)
            getattr(hx, "rate" if api == "rating" else "simulate")(i, o)


def test_direct_rating_simulation_entry_points_reject_before_execution():
    hx = BareTubeHeatExchanger(bank())
    with pytest.raises(SerratedFinPhysicsNotSupportedError, match="geometry only"):
        run_rating(hx, None)
    # Use real side contracts for Simulation; no phase detection may run.
    provider = ConstantPropertyProvider(PROPS)
    i = HXSideInput(provider=provider, m_dot=1., T_in=400., p=101325.)
    o = HXSideInput(provider=provider, m_dot=1., T_in=300., p=101325.)
    with pytest.raises(SerratedFinPhysicsNotSupportedError, match="geometry only"):
        run_simulation(hx, i, o)


def test_shifted_layout_is_not_introduced():
    with pytest.raises(ValueError, match="layout"):
        replace(bank(), layout="shifted")
