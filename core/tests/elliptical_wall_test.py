# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace
import math
import pytest
from core.geometry import BareTube, EllipticalTube, TubeBundle, TubeGeometryApplicabilityError
from core.heat_transfer.wall_conduction import TubeWallModel, tube_wall_resistance, tube_wall_resistance_diagnostics
from core.models.bare_tube import BareTubeHeatExchanger


def confocal():
    # Focal c=.01; inner/outer elliptic-coordinate levels mu=1 and mu=1.2.
    c=.01
    return EllipticalTube(c*math.cosh(1.2),c*math.sinh(1.2),
                          c*math.cosh(1),c*math.sinh(1),2.,1.8,20.)


def wall(tube,n=1):
    return tube_wall_resistance(tube,n_tubes=n,model=TubeWallModel.CONFOCAL_EXACT)


def test_confocal_independent_coordinate_solution_and_scaling():
    tube=confocal()
    assert wall(tube)==pytest.approx(.2/(2*math.pi*20*1.8),rel=1e-14)
    assert wall(replace(tube,wall_k=40.))==wall(tube)/2
    assert wall(replace(tube,length_effective=.9))==wall(tube)*2
    assert wall(tube,24)==pytest.approx(wall(tube)/24,rel=1e-14)
    hx=BareTubeHeatExchanger(TubeBundle(tube,4,6,.05,.05,'inline',2))
    assert hx.tube_wall_resistance(model=TubeWallModel.CONFOCAL_EXACT)==wall(tube,24)
    result=tube_wall_resistance_diagnostics(tube,n_tubes=24,model=TubeWallModel.CONFOCAL_EXACT)
    assert result.resistance==wall(tube,24)
    assert result.model_id=='elliptical_confocal_wall_exact' and not result.approximation
    assert result.source_references


def test_circle_limit_preserves_original_circular_formula():
    ellipse=EllipticalTube(.012,.012,.01,.01,2.,1.8,20.)
    circle=BareTube(.02,.024,2.,1.8,20.)
    expected=math.log(.024/.02)/(2*math.pi*20*1.8*24)
    assert tube_wall_resistance(circle,n_tubes=24)==expected
    assert wall(ellipse,24)==pytest.approx(expected,rel=1e-14)


def test_nonconfocal_forced_exact_rejected_and_automatic_models_succeed():
    tube=replace(confocal(),inner_semi_minor_axis=confocal().inner_semi_minor_axis*.9)
    with pytest.raises(TubeGeometryApplicabilityError,match='confocal'): wall(tube)
    assert tube_wall_resistance_diagnostics(tube).model_id=='elliptical_wall_laplace_P1'
    assert tube_wall_resistance(confocal())==wall(confocal())
    for model in (TubeWallModel.CYLINDRICAL,'thin_wall'):
        with pytest.raises(TubeGeometryApplicabilityError):
            tube_wall_resistance(confocal(),model=model)


@pytest.mark.parametrize('n',[0,-1,1.5,True])
def test_invalid_tube_multiplicity(n):
    with pytest.raises(ValueError): wall(confocal(),n)


def test_confocal_tolerance_is_scale_independent_and_not_manufacturing_clearance():
    original=confocal()
    for scale in (.001,1000.):
        tube=replace(original,**{f:getattr(original,f)*scale for f in (
            'outer_semi_major_axis','outer_semi_minor_axis','inner_semi_major_axis','inner_semi_minor_axis')})
        assert wall(tube)==pytest.approx(wall(original),rel=1e-14)
        with pytest.raises(TubeGeometryApplicabilityError):
            wall(replace(tube,inner_semi_minor_axis=tube.inner_semi_minor_axis*(1+1e-8)))
