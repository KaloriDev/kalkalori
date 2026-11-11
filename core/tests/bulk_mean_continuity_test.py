"""Dry-limit equivalence uses the same caloric states, area and correlations."""
from dataclasses import replace

import pytest

from core import BulkMeanAnalogyWetCoilProvider
from core.models.bare_tube import BareTubeHeatExchanger
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.outside_condensation_solver import solve_outside_condensation
from core.phase_change.types import PhaseChangeMode
from core.tests.wet_coil_public_test import context


@pytest.mark.parametrize("flow",["counterflow","cocurrentflow","crossflow"])
@pytest.mark.parametrize("finned",[False,True])
@pytest.mark.parametrize("margin,fouling",[(0.,0.),(.1,.002)])
def test_zero_condensate_active_equations_recover_selected_provider_dry_limit(flow,finned,margin,fouling):
    hx,a,b = context(finned=finned,W=.001)
    hx = BareTubeHeatExchanger(replace(hx.bundle,flow_arrangement=flow),
        fouling_resistance_inside=fouling, fouling_resistance_outside=fouling)
    a = replace(a,phase_change_mode=PhaseChangeMode.DISABLED,h_in=None,quality_in=None)
    b = replace(b,phase_change_mode=PhaseChangeMode.DISABLED)
    dry = hx.simulate(a,b,wet_coil_provider=BulkMeanAnalogyWetCoilProvider(),
        surface_margin=margin,
        max_iter=100,temperature_tolerance_K=1e-7,relative_alfa_tolerance=1e-8)
    cap = detect_phase_change_capability(b.provider)
    wet = solve_outside_condensation(hx,inside_provider=a.provider,m_dot_inside=a.m_dot,
        T_in_inside=a.T_in,p_inside=a.p,outside_capability=cap,
        m_dot_dry_carrier=b.m_dot/(1+cap.W_in),T_in_outside=b.T_in,p_outside=b.p,
        surface_margin=margin,
        T_out_inside_init=dry.T_out_inside,T_out_outside_init=dry.T_out_outside,
        max_iterations=100,temperature_tolerance_K=1e-7,relative_Q_tolerance=1e-8,
        wall_temperature_tolerance_K=1e-7,water_ratio_tolerance=1e-10,
        condensate_tolerance_kg_s=1e-12,wet_fraction_tolerance=1e-7)
    assert dry.converged and wet.converged
    assert wet.m_dot_condensate == wet.Q_latent == wet.wet_surface_fraction == 0
    # The fin radial FVM and dry analytical fin response have independently
    # established discretization errors; the bare network has no radial mesh.
    assert wet.Q_total == pytest.approx(dry.q,rel=5e-5 if finned else 3e-7,abs=1e-3)
    assert wet.T_out_outside == pytest.approx(dry.T_out_outside,abs=.001 if finned else 2e-6)
