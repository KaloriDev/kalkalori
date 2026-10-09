# SPDX-License-Identifier: GPL-3.0-only
"""Area-basis, physical-solve, zero-freeze and hydraulic fouling regressions."""
from dataclasses import replace

import pytest

from core import LegacyBulkMeanWetCoilProvider, WetCoilSolverOptions
from core.heat_transfer.outside_dispatch import calculate_resistance_network
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.phase_change.types import PhaseChangeMode
from core.phase_change.wet_coil_integration import forward_wet_process, _forward_wet_process_native
from core.tests.thermal_iteration_test import (
    build_bundle, _INSIDE_PROVIDER, _OUTSIDE_PROVIDER,
    _M_DOT_INSIDE, _M_DOT_OUTSIDE,
)
from core.tests.wet_coil_public_test import context, specs
from core.tests.legacy_wet_coil_provider_test import historical_case
from core.tests.steam_public_integration_test import _hx, _inside_sim, _outside_sim
from core.tests.water_evaporation_simulation_test import (
    _hx as evaporation_hx, _inside, _outside,
)

OPTIONS = WetCoilSolverOptions(timeout_s=None)


def dry_context(**fouling):
    return (
        BareTubeHeatExchanger(build_bundle(), **fouling),
        HXSideInput(_INSIDE_PROVIDER, _M_DOT_INSIDE, 673.15, 101325.),
        HXSideInput(_OUTSIDE_PROVIDER, _M_DOT_OUTSIDE, 303.15, 101325.),
    )


def fouled(hx, inside=0.002, outside=0.003):
    return BareTubeHeatExchanger(hx.bundle,
        fouling_resistance_inside=inside, fouling_resistance_outside=outside)


@pytest.mark.parametrize('side', ['inside', 'outside'])
@pytest.mark.parametrize('value', [None, 0.0])
def test_defaults_equal_omitted(side, value):
    hx, a, b = dry_context(**{f'fouling_resistance_{side}': value})
    clean, _, _ = dry_context()
    actual, expected = hx.simulate(a, b), clean.simulate(a, b)
    assert getattr(hx, f'fouling_resistance_{side}') == 0.0
    assert (actual.q, actual.UA, actual.T_out_inside, actual.T_out_outside) == (
        expected.q, expected.UA, expected.T_out_inside, expected.T_out_outside)


@pytest.mark.parametrize('side', ['inside', 'outside'])
@pytest.mark.parametrize('value', [-1e-5, float('nan'), float('inf'), -float('inf')])
def test_invalid_fouling_rejected(side, value):
    with pytest.raises(ValueError, match=f'fouling_resistance_{side}'):
        dry_context(**{f'fouling_resistance_{side}': value})


@pytest.mark.parametrize('finned,override_gross', [(False, False), (True, False), (True, True)])
def test_absolute_area_basis_and_decomposition(finned, override_gross):
    hx, _, _ = context(finned=finned)
    if override_gross:
        tube = replace(hx.bundle.tube,
            external_area_per_length=hx.bundle.tube.outside_area_used_per_length * 1.2)
        hx = BareTubeHeatExchanger(replace(hx.bundle, tube=tube))
    kwargs = dict(bundle=hx.bundle, alpha_inside=1200., outside_alpha_physical=80.,
                  resistance_core_wall=hx.tube_wall_resistance())
    clean = calculate_resistance_network(**kwargs)
    dirty = calculate_resistance_network(**kwargs,
        fouling_resistance_inside=.002, fouling_resistance_outside=.003)
    assert clean.area_inside != clean.area_outside_gross
    ri, ro = .002 / hx.bundle.total_inner_area, .003 / hx.bundle.total_outer_area
    assert dirty.resistance_fouling_inside == ri
    assert dirty.resistance_fouling_outside == ro
    assert 1 / dirty.UA == pytest.approx(1 / clean.UA + ri + ro, rel=1e-14)
    assert dirty.resistance_inside == pytest.approx(clean.resistance_inside + ri)
    assert dirty.resistance_outside == pytest.approx(clean.resistance_outside + ro)
    assert dirty.resistance_core_wall == clean.resistance_core_wall
    assert dirty.area_outside_effective == clean.area_outside_effective
    assert dirty.fin_efficiency == clean.fin_efficiency
    assert dirty.resistance_contact == clean.resistance_contact
    if finned:
        assert dirty.area_outside_gross != hx.bundle.tube.bare_outside_area * hx.bundle.n_tubes_total


@pytest.mark.parametrize('inside,outside', [(.002, 0.), (0., .003), (.002, .003)])
def test_dry_simulation_rating_and_fixed_property_hydraulics(inside, outside):
    hx, a, b = dry_context()
    dirty = fouled(hx, inside, outside)
    clean, result = hx.simulate(a, b), dirty.simulate(a, b)
    assert result.converged and result.q < clean.q
    assert result.UA < clean.UA
    # Captured at merged v0.8.6 HEAD 795df3e before implementation.
    assert (clean.q, clean.UA, clean.T_out_inside, clean.T_out_outside) == pytest.approx(
        (1588271.373269159, 13841.892763549058, 361.5127481521243, 473.8890303439093), rel=1e-10)
    thermal_r = (1 / (result.inside_alfa_mean * hx.bundle.total_inner_area)
                 + dirty.resistance_fouling_inside + hx.tube_wall_resistance()
                 + 1 / (result.outside_alfa_mean * hx.bundle.total_outer_area))
    assert result.UA == pytest.approx(1 / thermal_r)
    assert result.final_result.fouling_resistance_inside == inside
    assert result.final_result.fouling_resistance_outside == outside
    for name in ('flow_area_per_pass', 'hydraulic_diameter', 'hydraulic_length_total',
                 'entrance_count', 'exit_count', 'dp_tube_bundle'):
        assert getattr(result.final_result.tube_bundle_hydraulic, name) == getattr(clean.final_result.tube_bundle_hydraulic, name)
    for point in ('inlet', 'midpoint', 'outlet'):
        for name in ('velocity', 'reynolds', 'friction_factor'):
            assert getattr(getattr(result.final_result.tube_bundle_hydraulic, point), name) == getattr(getattr(clean.final_result.tube_bundle_hydraulic, point), name)
    assert result.final_result.outside_tube_bank_hydraulic.dp_total == clean.final_result.outside_tube_bank_hydraulic.dp_total
    assert dirty.bundle is hx.bundle
    target = (
        BalanceSideSpec(a.provider, a.p, a.m_dot, a.T_in, clean.T_out_inside),
        BalanceSideSpec(b.provider, b.p, b.m_dot, b.T_in, clean.T_out_outside),
    )
    rating, rating_dirty = hx.rate(*target), dirty.rate(*target)
    assert rating_dirty.A_required > rating.A_required


@pytest.mark.parametrize('inside,outside', [(.0003, 0.), (0., .0004), (.0003, .0004)])
def test_elmahdy_changes_solved_wet_state_with_strict_closure(inside, outside):
    hx, a, b = context(W=.0075)
    clean, _, _ = forward_wet_process(hx, a, b, wet_solver_options=OPTIONS)
    dirty_hx = fouled(hx, inside, outside)
    dirty, _, _ = forward_wet_process(dirty_hx, a, b, wet_solver_options=OPTIONS)
    oracle, _, _ = _forward_wet_process_native(dirty_hx, a, b, wet_solver_options=OPTIONS)
    assert 0 < clean.wet_fraction < 1
    assert dirty.heat_liquid < clean.heat_liquid
    assert dirty.humidity_out > clean.humidity_out
    assert dirty.condensate != pytest.approx(clean.condensate, abs=1e-9)
    assert dirty.wet_fraction != pytest.approx(clean.wet_fraction, abs=1e-6)
    assert dirty.cold_surface != pytest.approx(clean.cold_surface, abs=1e-6)
    assert dirty.heat_liquid == pytest.approx(oracle.heat_liquid, abs=.03, rel=0)
    assert dirty.wet_fraction == pytest.approx(oracle.wet_fraction, abs=2e-6, rel=0)
    d = dirty.diagnostics
    assert abs(d['energy_residual']) <= OPTIONS.energy_tolerance_W
    assert abs(d['mass_residual']) <= OPTIONS.mass_tolerance_kg_s
    assert d['resistance_fouling_inside'] == dirty_hx.resistance_fouling_inside
    assert d['resistance_fouling_outside'] == dirty_hx.resistance_fouling_outside
    assert d['numerical_path'].startswith('optimized')


def test_wet_rating_fouling_increases_required_area_and_scales_resistances():
    hx, a, b = context(W=.0075)
    target = specs(a, b, 295.5)
    clean = hx.rate(*target, wet_solver_options=OPTIONS)
    dirty_hx = fouled(hx, .0003, .0004)
    dirty = dirty_hx.rate(*target, wet_solver_options=OPTIONS)
    assert dirty.A_required > clean.A_required
    scale = dirty.A_required / hx.bundle.total_outer_area
    d = dirty.wet_coil_diagnostics
    assert d['resistance_fouling_inside'] == pytest.approx(dirty_hx.resistance_fouling_inside / scale)
    assert d['resistance_fouling_outside'] == pytest.approx(dirty_hx.resistance_fouling_outside / scale)
    assert abs(d['energy_residual']) <= OPTIONS.energy_tolerance_W
    assert abs(d['mass_residual']) <= OPTIONS.mass_tolerance_kg_s


def test_wet_defaults_and_v086_numerical_freeze():
    hx, a, b = context(W=.007)
    expected = (2289.1136368353614, 281.5128964000954, 295.63766462612205,
                .006994650942929997, .23222349131850753)
    for configured in (hx, fouled(hx, None, None), fouled(hx, 0., 0.)):
        r = forward_wet_process(configured, a, b, wet_solver_options=OPTIONS)[0]
        assert (r.heat_liquid, r.liquid_out, r.air_out, r.humidity_out,
                r.wet_fraction) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize('finned', [False, True])
def test_legacy_zero_freeze_and_positive_fouling(finned):
    hx, a, b = historical_case(finned)
    opts = dict(wet_coil_provider=LegacyBulkMeanWetCoilProvider())
    clean = hx.simulate(a, b, **opts)
    zero = fouled(hx, None, None).simulate(a, b, **opts)
    expected = (452361.78009796015, 309.77365212847946, 336.40912837496386,
                .04443248677919755, .442604902596629, 6621.439654955228) if finned else (
                702644.68440337, 336.21383345649315, 336.5149577934638,
                .06982196451578865, .32591996755281527, 10784.936645928527)
    for r in (clean, zero):
        assert (r.q, r.T_out_inside, r.T_out_outside, r.outside_phase_change.m_dot_condensate,
                r.outside_phase_change.wet_surface_fraction, r.UA) == pytest.approx(expected)
    dirty = fouled(hx, .0003, .0004).simulate(a, b, **opts)
    assert dirty.converged and dirty.outside_phase_change.converged
    assert dirty.q < clean.q
    assert dirty.outside_phase_change.W_out != clean.outside_phase_change.W_out
    assert dirty.final_result.A_i == clean.final_result.A_i
    assert dirty.final_result.A_o == clean.final_result.A_o
    assert dirty.wet_coil_diagnostics['fouling_resistance_inside'] == .0003
    assert dirty.wet_coil_diagnostics['fouling_resistance_outside'] == .0004
    for name in ('flow_area_per_pass', 'hydraulic_diameter', 'hydraulic_length_total',
                 'entrance_count', 'exit_count'):
        assert getattr(dirty.final_result.tube_bundle_hydraulic, name) == getattr(clean.final_result.tube_bundle_hydraulic, name)


def test_finned_dry_path_keeps_geometry_and_fin_efficiency():
    hx, a, b = context(finned=True, W=.004)
    b = replace(b, phase_change_mode=PhaseChangeMode.DISABLED)
    clean = hx.simulate(a, b)
    dirty_hx = fouled(hx)
    dirty = dirty_hx.simulate(a, b)
    assert dirty.converged and dirty.q < clean.q
    d = dirty.finned_tube_diagnostics
    assert d.fouling_resistance_outside == .003
    assert d.resistance_fouling_outside == .003 / hx.bundle.total_outer_area
    assert dirty.final_result.A_o == clean.final_result.A_o
    assert dirty_hx.bundle is hx.bundle


def test_finned_elmahdy_wet_surface_and_energy_mass_closure():
    hx, a, b = context(finned=True)
    clean = hx.simulate(a, b, wet_solver_options=OPTIONS)
    dirty = fouled(hx, .0003, .0004).simulate(a, b, wet_solver_options=OPTIONS)
    assert dirty.converged and dirty.q < clean.q
    assert dirty.outside_phase_change.m_dot_condensate != clean.outside_phase_change.m_dot_condensate
    assert dirty.wall_temperature_envelope.outside_mean != clean.wall_temperature_envelope.outside_mean
    d = dirty.wet_coil_diagnostics
    assert abs(d['energy_residual']) <= OPTIONS.energy_tolerance_W
    assert abs(d['mass_residual']) <= OPTIONS.mass_tolerance_kg_s
    assert d['network'].resistance_fouling_outside == .0004 / hx.bundle.total_outer_area


@pytest.mark.parametrize('evaporation', [False, True])
@pytest.mark.parametrize('inside,outside', [(.002, 0.), (0., .003), (.002, .003)])
def test_inside_phase_change_zero_freeze_and_inside_fouling(evaporation, inside, outside):
    if evaporation:
        hx, a, b = evaporation_hx(), _inside(quality_in=0.), _outside()
        expected = (1597947.0520325066, 7248.095077760986)
    else:
        hx, a, b = _hx(), _inside_sim(quality_in=1.), _outside_sim()
        expected = (1008060.4071717563, 7432.31610691575)
    clean = hx.simulate(a, b)
    zero = fouled(hx, None, 0.).simulate(a, b)
    assert (clean.q, clean.UA) == pytest.approx(expected, rel=1e-10)
    assert (zero.q, zero.UA) == (clean.q, clean.UA)
    dirty_hx = fouled(hx, inside, outside)
    dirty = dirty_hx.simulate(a, b)
    assert dirty.converged and dirty.q < clean.q
    assert dirty.UA < clean.UA
    assert dirty.inside_phase_change.active
    assert dirty.final_result.fouling_resistance_inside == inside
    assert dirty.final_result.fouling_resistance_outside == outside
    resistance = (1 / (dirty.inside_alfa_mean * hx.bundle.total_inner_area)
                  + dirty_hx.resistance_fouling_inside + hx.tube_wall_resistance()
                  + 1 / (dirty.outside_alfa_mean * hx.bundle.total_outer_area))
    # Whole-exchanger equivalent HTC and final outside-property iteration
    # reconstruct within the existing phase-solver numerical convergence.
    assert dirty.UA == pytest.approx(1 / resistance, rel=1e-8)


def test_finned_tube_inside_steam_condensation_uses_fouled_common_stack():
    from core.geometry import TubeOrientation
    from core.tests.finned_tube_inside_phase_change_test import _hx as finned_hx
    hx = finned_hx(orientation=TubeOrientation.VERTICAL_DOWNWARD)
    a, b = _inside_sim(quality_in=1.), _outside_sim()
    clean = hx.simulate(a, b)
    dirty = fouled(hx, .002, .003).simulate(a, b)
    assert dirty.converged and dirty.q < clean.q
    assert dirty.finned_tube_diagnostics.resistance_fouling_inside == .002 / hx.bundle.total_inner_area
    assert dirty.finned_tube_diagnostics.resistance_fouling_outside == .003 / hx.bundle.total_outer_area
