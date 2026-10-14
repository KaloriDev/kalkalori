# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import FrozenInstanceError, replace
import math
import pytest
from scipy.integrate import quad
from core.geometry import BareTube, EllipticalTube, TubeBundle, TubeGeometryApplicabilityError


def ellipse(**kwargs):
    values = dict(outer_semi_major_axis=.025, outer_semi_minor_axis=.009,
                  inner_semi_major_axis=.023, inner_semi_minor_axis=.007,
                  length_total=2., length_effective=1.8, wall_k=20.)
    return EllipticalTube(**(values | kwargs))


@pytest.mark.parametrize('beta', [1., .5, .36, .25, .01])
def test_exact_area_perimeter_against_independent_arclength_quadrature(beta):
    tube=ellipse(inner_semi_major_axis=.006, inner_semi_minor_axis=.006*beta)
    a,b=tube.inner_semi_major_axis,tube.inner_semi_minor_axis
    p=4*quad(lambda t: math.hypot(a*math.sin(t),b*math.cos(t)),0,math.pi/2,epsabs=1e-13)[0]
    assert tube.flow_area == math.pi*a*b
    assert tube.inside_wetted_perimeter == pytest.approx(p,rel=2e-12)
    assert tube.hydraulic_diameter == pytest.approx(4*math.pi*a*b/p,rel=2e-12)
    assert tube.area_inner == tube.inside_wetted_perimeter*tube.length_effective
    assert tube.area_outer == tube.outside_perimeter*tube.length_effective


@pytest.mark.parametrize('angle', [0.,math.pi/4,math.pi/2,1.137])
def test_projection_support_and_streamwise_extent(angle):
    tube=ellipse(orientation_angle=angle)
    ao,bo=tube.outer_semi_major_axis,tube.outer_semi_minor_axis
    assert tube.projected_blocking_area_per_length == pytest.approx(2*math.sqrt(ao**2*math.sin(angle)**2+bo**2*math.cos(angle)**2))
    assert tube.streamwise_extent == pytest.approx(2*math.sqrt(ao**2*math.cos(angle)**2+bo**2*math.sin(angle)**2))
    assert replace(tube,orientation_angle=angle+math.pi).streamwise_extent == pytest.approx(tube.streamwise_extent)


@pytest.mark.parametrize('angle', [0.,math.pi/4,math.pi/2])
def test_circle_limit_and_frozen_sibling_type(angle):
    tube=ellipse(outer_semi_major_axis=.012,outer_semi_minor_axis=.012,
                  inner_semi_major_axis=.01,inner_semi_minor_axis=.01,orientation_angle=angle)
    circle=BareTube(.02,.024,2.,1.8,20.)
    assert not isinstance(tube,BareTube)
    assert not hasattr(tube,'D_i') and not hasattr(tube,'D_o')
    for field in ('flow_area','hydraulic_diameter','area_inner','area_outer','inside_wetted_perimeter',
                  'outside_perimeter','streamwise_extent','projected_blocking_area_per_length'):
        assert getattr(tube,field)==pytest.approx(getattr(circle,field),rel=1e-14)
    with pytest.raises(FrozenInstanceError): tube.wall_k=30.


@pytest.mark.parametrize('field', ['outer_semi_major_axis','outer_semi_minor_axis','inner_semi_major_axis',
                                   'inner_semi_minor_axis','length_total','length_effective','wall_k'])
@pytest.mark.parametrize('value', [0.,-1.,math.nan,math.inf,-math.inf])
def test_invalid_positive_inputs(field,value):
    with pytest.raises(ValueError): ellipse(**{field:value})


@pytest.mark.parametrize('kwargs', [dict(inner_semi_minor_axis=.024),dict(outer_semi_minor_axis=.026),
    dict(inner_semi_major_axis=.025),dict(inner_semi_minor_axis=.009),dict(length_effective=2.1),
    dict(orientation_angle=math.nan),dict(roughness_inner=-1.),dict(tube_orientation='horizontal')])
def test_invalid_order_nesting_lengths_and_metadata(kwargs):
    with pytest.raises(ValueError): ellipse(**kwargs)


def test_bundle_aggregates_physical_quantities_but_rejects_bank_throat():
    tube=ellipse()
    bank=TubeBundle(tube,4,6,.06,.05,'staggered',2)
    assert bank.total_inner_area==24*tube.area_inner
    assert bank.total_outer_area==24*tube.area_outer
    assert bank.internal_flow_area_per_pass==12*tube.flow_area
    assert bank.internal_length_total==4.
    assert bank.frontal_flow_area==6*.06*1.8
    for field in ('minimum_free_flow_gap','minimum_free_flow_area','maximum_to_face_velocity_ratio'):
        with pytest.raises(TubeGeometryApplicabilityError): getattr(bank,field)
