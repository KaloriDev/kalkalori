# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace
import math
import pytest
from core.geometry import EllipticalTube, BareTube, TubeGeometryApplicabilityError
from core.heat_transfer import (TubeWallModel, tube_wall_resistance_diagnostics,
    tube_wall_resistance, elliptical_wall_solution, WallConductionConvergenceError)


def result(tube,n=1):
    return tube_wall_resistance_diagnostics(tube,n_tubes=n,model=TubeWallModel.ELLIPTICAL_NUMERICAL)


def ellipse():
    return EllipticalTube(.021,.009,.018,.006,2.,1.8,40.)


def test_numerical_circle_limit_against_exact_cylinder():
    t=EllipticalTube(.012,.012,.01,.01,2.,1.8,40.)
    expected=math.log(1.2)/(2*math.pi*40*1.8)
    r=result(t)
    assert r.resistance==pytest.approx(expected,rel=1e-4)
    assert r.convergence.converged and r.approximation
    assert r.convergence.relative_heat_balance_error<1e-9
    assert r.convergence.relative_residual<1e-9


@pytest.mark.parametrize('mu',[.5,1.,2.])
def test_numerical_confocal_against_exact_coordinates(mu):
    c=.01
    t=EllipticalTube(c*math.cosh(mu+.2),c*math.sinh(mu+.2),
                     c*math.cosh(mu),c*math.sinh(mu),2.,1.8,40.)
    exact=tube_wall_resistance(t,model=TubeWallModel.CONFOCAL_EXACT)
    assert result(t).resistance==pytest.approx(exact,rel=1e-4)


def test_nonconfocal_positive_scaling_and_cache_immutability():
    t=ellipse(); r=result(t)
    with pytest.raises(TubeGeometryApplicabilityError,match='not confocal'):
        tube_wall_resistance(t,model=TubeWallModel.CONFOCAL_EXACT)
    assert r.resistance>0 and r.model_id=='elliptical_wall_laplace_P1'
    assert result(t,24).resistance==pytest.approx(r.resistance/24)
    assert result(replace(t,wall_k=80.)).resistance==r.resistance/2
    assert result(replace(t,length_effective=.9)).resistance==2*r.resistance
    assert result(replace(t,orientation_angle=.6)).resistance==r.resistance
    scaled=replace(t,**{n:getattr(t,n)*1000 for n in (
        'outer_semi_major_axis','outer_semi_minor_axis','inner_semi_major_axis','inner_semi_minor_axis')})
    assert result(scaled).resistance==pytest.approx(r.resistance,rel=1e-12)
    assert result(t).convergence is r.convergence


def test_independent_finer_mesh_agrees_and_not_converged_raises():
    t=ellipse(); normal=elliptical_wall_solution(t)
    fine=elliptical_wall_solution(t,tolerance=1e-5)
    assert fine.radial_cells>normal.radial_cells
    assert fine.unit_conductance==pytest.approx(normal.unit_conductance,rel=1e-4)
    with pytest.raises(WallConductionConvergenceError,match='did not converge'):
        elliptical_wall_solution(t,tolerance=1e-12,max_refinements=1)


@pytest.mark.parametrize('kw',[{'radial_cells':0},{'angular_cells':True},{'max_refinements':0},
                              {'tolerance':0.},{'tolerance':math.nan}])
def test_bad_solver_controls(kw):
    with pytest.raises(ValueError): elliptical_wall_solution(ellipse(),**kw)


def test_no_implicit_numerical_wall_or_circular_rerouting():
    with pytest.raises(TubeGeometryApplicabilityError,match='explicit'): tube_wall_resistance(ellipse())
    with pytest.raises(TubeGeometryApplicabilityError,match='cylindrical'):
        result(BareTube(.02,.024,2.,1.8,40.))
