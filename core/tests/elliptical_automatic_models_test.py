# SPDX-License-Identifier: GPL-3.0-only
"""Engineering contour, physical dispatch, and source-policy regressions."""
from dataclasses import replace
import math
import pytest

from core.geometry import EllipticalTube, TubeGeometryApplicabilityError
from core.heat_transfer import TubeWallModel, tube_wall_resistance_diagnostics
from core.heat_transfer.elliptical_turbulent import elliptical_turbulent_heat_transfer
from core.heat_transfer.internal_dispatch import internal_heat_transfer_diagnostics, internal_friction
from core.heat_transfer.internal_flow import FluidProps
from core.heat_transfer.finned_ellipse_airside import zhao2017_flow
from core.tests.elliptical_turbulent_integration_test import context
from core.tests.zhao2017_fixtures import source_bundle, AIR
from core.models.heat_balance import BalanceSideSpec

PROPS = FluidProps(1000., .001, .6, 4200.)


def nominal():
    return EllipticalTube.from_nominal_wall(outer_semi_major_axis=.018,
        outer_semi_minor_axis=.007, nominal_wall_thickness=.0015,
        length_total=.419, length_effective=.405, wall_k=45.)


def evaluate(re, tube=None, **options):
    t = tube or nominal()
    return internal_heat_transfer_diagnostics(tube=t,
        m_dot=re*PROPS.mu*t.flow_area/t.hydraulic_diameter,
        flow_area=t.flow_area, props=PROPS, **options)


def test_nominal_36_14_1p5_has_actual_33_11_inner_and_automatic_numerical_wall():
    t = nominal()
    assert (2*t.inner_semi_major_axis, 2*t.inner_semi_minor_axis)==pytest.approx((.033,.011))
    auto = tube_wall_resistance_diagnostics(t)
    forced = tube_wall_resistance_diagnostics(t, model=TubeWallModel.ELLIPTICAL_NUMERICAL)
    assert auto==forced and auto.resistance>0 and auto.convergence.converged
    with pytest.raises(TubeGeometryApplicabilityError, match='not confocal'):
        tube_wall_resistance_diagnostics(t, model=TubeWallModel.CONFOCAL_EXACT)


@pytest.mark.parametrize('thickness', [0., -.001, math.nan, math.inf, True, .007])
def test_invalid_nominal_wall_is_rejected(thickness):
    with pytest.raises(ValueError):
        EllipticalTube.from_nominal_wall(outer_semi_major_axis=.018,
            outer_semi_minor_axis=.007, nominal_wall_thickness=thickness,
            length_total=1., length_effective=1., wall_k=45.)


@pytest.mark.parametrize('re', [800., 24000., 121900.])
def test_internal_results_ignore_outer_contours_including_confocality(re):
    t = nominal()
    confocal = replace(t, outer_semi_major_axis=math.sqrt(
        t.inner_semi_major_axis**2-t.inner_semi_minor_axis**2+.007**2))
    assert evaluate(re,t)==evaluate(re,confocal)
    assert internal_friction(tube=t,reynolds=re)==internal_friction(tube=confocal,reynolds=re)


@pytest.mark.parametrize('re', [10000., 24000., 120000.])
def test_source_range_has_no_re_extrapolation_warning(re):
    assert not any(w.code=='yang2018_re_extrapolation' for w in evaluate(re).warnings)


@pytest.mark.parametrize('re,direction', [(5000.,'below'), (121900.,'above'), (500000.,'above')])
def test_turbulent_source_extrapolation_uses_actual_re_and_unchanged_equation(re,direction):
    r = evaluate(re)
    assert r.Re==pytest.approx(re)
    assert r.Nu_base==pytest.approx(.94*(.02979*re**.8-8.5626)*r.Pr**.4)
    assert math.isfinite(r.alfa_corrected) and r.alfa_corrected>0
    w = next(w for w in r.warnings if w.code=='yang2018_re_extrapolation')
    assert w.reynolds==pytest.approx(re) and w.source_lower==10000 and w.source_upper==120000
    assert w.direction==direction
    assert direction in w.message and '10000' in w.message and '120000' in w.message
    assert f'{re:g}' in w.message
    with pytest.raises(TubeGeometryApplicabilityError, match='strict source domain'):
        evaluate(re, strict_source_domain=True)


@pytest.mark.parametrize('re', [100., 2299.])
def test_automatic_laminar_selection_and_turbulent_provider_guard(re):
    assert evaluate(re).model_metadata.thermal_boundary_condition=='H1'
    t = nominal()
    with pytest.raises(TubeGeometryApplicabilityError, match='turbulent'):
        elliptical_turbulent_heat_transfer(t,re*PROPS.mu*t.flow_area/t.hydraulic_diameter,PROPS)


@pytest.mark.parametrize('re', [2300., 3000., 3999.])
def test_transition_has_no_invented_provider(re):
    with pytest.raises(TubeGeometryApplicabilityError, match='transition'):
        evaluate(re)
    with pytest.raises(TubeGeometryApplicabilityError, match='transition'):
        internal_friction(tube=nominal(), reynolds=re)


def test_fin_nomenclature_keeps_source_reconstruction_mesh_and_projection_orientation():
    b = replace(source_bundle(n_rows=2),pitch_transverse=.02667)
    assert b.tube.fin.outer_width==.026 and b.tube.fin.outer_height==.055
    assert b.pitch_transverse-b.tube.fin.outer_width==pytest.approx(.00067)
    assert b.pitch_longitudinal-b.tube.fin.outer_height==pytest.approx(.005)
    f = zhao2017_flow(b,1.7,AIR)
    fp,ft = b.tube.fin.axial_pitch,b.tube.fin.thickness
    expected=1-((fp-ft)*.014+ft*.026)/(.02667*fp)
    assert f.sigma_throat==pytest.approx(expected)
    assert not f.geometry_extrapolation


def test_high_re_rating_simulation_propagate_warning_and_strict_option():
    hx,a,b = context()
    a = replace(a,m_dot=a.m_dot*121900/24000)
    sim = hx.simulate(a,b)
    rating = hx.rate(
        BalanceSideSpec(provider=a.provider,m_dot=a.m_dot,T_in=a.T_in,T_out=sim.T_out_inside,p=a.p),
        BalanceSideSpec(provider=b.provider,m_dot=b.m_dot,T_in=b.T_in,T_out=sim.T_out_outside,p=b.p),
        include_simulation=False)
    for result in (sim,rating):
        assert any(w.code=='yang2018_re_extrapolation' for w in result.warnings)
        meta = result.thermal_state.diagnostics.inside_model_metadata
        assert meta.development=='fully_developed' and meta.thermal_boundary_condition=='T'
    hx.strict_source_domain = True
    with pytest.raises(TubeGeometryApplicabilityError,match='strict source domain'):
        hx.simulate(a,b)
