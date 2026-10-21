# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace
import math
import numpy as np
import pytest
from core.geometry import EllipticalTube, BareTube, TubeGeometryApplicabilityError
from core.heat_transfer.internal_flow import FluidProps, heat_transfer_coefficient_internal_diagnostics
from core.heat_transfer.internal_dispatch import internal_heat_transfer_diagnostics, internal_friction, straight_friction_evaluator
from core.heat_transfer.internal_model_contract import EllipticalFlowAssumptions
from core.heat_transfer.elliptical_internal import (
    elliptical_laminar_friction, elliptical_h1_heat_transfer, elliptical_distributed_pressure_drop,
)

PROPS=FluidProps(1000.,.001,.6,4200.)
FD=EllipticalFlowAssumptions('laminar','fully_developed','H1')


def ellipse(beta=.5):
    return EllipticalTube(.012,.012,.01,.01*beta,2.,1.8,20.)


@pytest.mark.parametrize('beta,expected',[(1.,64.),(.5,67.293),(.36,70.078),(.25,72.960)])
def test_friction_source_values_and_darcy_fanning(beta,expected):
    result=elliptical_laminar_friction(ellipse(beta),800.,assumptions=FD)
    assert result.f_darcy*result.reynolds==pytest.approx(expected,abs=.0006)
    assert result.f_fanning==result.f_darcy/4
    assert result.metadata.model_id=='elliptical_fd_laminar_darcy_exact'
    assert result.metadata.provenance_id and result.metadata.source_references
    assert not result.metadata.approximation


@pytest.mark.parametrize('beta',[1.,.5,.36,.25])
def test_pressure_gradient_from_independent_momentum_solution_and_scaling(beta):
    tube=ellipse(beta)
    result=elliptical_distributed_pressure_drop(tube,.005,PROPS,assumptions=FD)
    a,b=tube.inner_semi_major_axis,tube.inner_semi_minor_axis
    expected=4*PROPS.mu*result.velocity*(a*a+b*b)/(a*a*b*b)
    assert result.pressure_gradient==pytest.approx(expected,rel=1e-14)
    assert result.pressure_drop==pytest.approx(expected*tube.length_total,rel=1e-14)
    assert not result.local_losses_included
    assert result.calculation_scope=='distributed_fully_developed_friction_only'
    assert elliptical_distributed_pressure_drop(tube,.01,PROPS,assumptions=FD).pressure_drop==pytest.approx(2*result.pressure_drop)
    assert elliptical_distributed_pressure_drop(tube,.005,PROPS,assumptions=FD,length=4.).pressure_drop==pytest.approx(2*result.pressure_drop)


@pytest.mark.parametrize('beta,expected',[(1.,48/11),(.5,4.557855),(.25,4.880272)])
def test_H1_source_values_and_alpha_reference(beta,expected):
    tube=ellipse(beta)
    result=elliptical_h1_heat_transfer(tube,.005,PROPS,assumptions=FD)
    assert result.Nu_corrected==pytest.approx(expected,abs=5e-7)
    assert result.alfa_corrected==result.Nu_corrected*PROPS.k/tube.hydraulic_diameter
    assert result.model_metadata.thermal_boundary_condition=='H1'
    assert result.model_metadata.reference_dimension=='inside_hydraulic_diameter'
    assert result.length_correction==1. and result.wall_temperature_correction==1.
    assert internal_heat_transfer_diagnostics(tube=tube,m_dot=.005,flow_area=tube.flow_area,props=PROPS,assumptions=FD)==result


@pytest.mark.parametrize('beta',[1.,.5,.36,.25])
def test_H1_independent_polynomial_energy_solution_and_bulk_quadrature(beta):
    # Solve coefficient matching of Laplacian(Tw-T)=-(1-X²-Y²), rather
    # than reproducing the closed Nu ratio. Integrate the velocity-weighted
    # bulk temperature and total heat input with independent quadrature.
    tube=ellipse(beta)
    a,b=tube.inner_semi_major_axis,tube.inner_semi_minor_axis
    A,B=1/a**2,1/b**2
    matrix=np.array([[-2*(A+B),2*A,2*B],[0,-12*A-2*B,-2*B],[0,-2*A,-2*A-12*B]])
    C0,Cx,Cy=np.linalg.solve(matrix,np.array([-1.,1.,1.]))
    r,w=np.polynomial.legendre.leggauss(12)
    r,w=(r+1)/2,w/2
    theta=(np.arange(64)+.5)*2*math.pi/64
    X,Y=r[:,None]*np.cos(theta),r[:,None]*np.sin(theta)
    dT=(1-X**2-Y**2)*(C0+Cx*X**2+Cy*Y**2)
    weight=w[:,None]*r[:,None]*(1-r[:,None]**2)*np.ones((1,64))
    bulk=np.sum(dT*weight)/np.sum(weight)
    q_per_length=tube.flow_area/2  # integral of 1-r², conductivity=1
    h=q_per_length/(tube.inside_wetted_perimeter*bulk)
    expected=h*tube.hydraulic_diameter
    actual=elliptical_h1_heat_transfer(tube,.005,PROPS,assumptions=FD)
    assert actual.Nu_corrected==pytest.approx(expected,rel=3e-14)


@pytest.mark.parametrize('change',[dict(regime='turbulent'),dict(regime='transitional'),
    dict(development='developing'),dict(constant_properties=False),dict(no_slip=False),dict(newtonian=False)])
def test_unsupported_friction_and_HTC_assumptions(change):
    assumptions=replace(FD,**change)
    for call in (lambda: elliptical_laminar_friction(ellipse(),800.,assumptions=assumptions),
                 lambda: elliptical_h1_heat_transfer(ellipse(),.005,PROPS,assumptions=assumptions)):
        with pytest.raises(TubeGeometryApplicabilityError): call()


@pytest.mark.parametrize('change',[dict(thermal_boundary_condition=bc) for bc in (None,'T','H2','constant_heat_flux','unknown')]+
    [dict(axial_conduction=True),dict(viscous_dissipation=True)])
def test_H1_identity_is_explicit(change):
    with pytest.raises(TubeGeometryApplicabilityError,match='H1'):
        elliptical_h1_heat_transfer(ellipse(),.005,PROPS,assumptions=replace(FD,**change))


def test_positive_roughness_rejected_and_automatic_regime_never_uses_circular_fallback(monkeypatch):
    import core.heat_transfer.internal_dispatch as dispatch
    monkeypatch.setattr(dispatch,'heat_transfer_coefficient_internal_diagnostics',lambda **kw: pytest.fail('Circular HTC called'))
    monkeypatch.setattr(dispatch,'darcy_friction_factor',lambda *a: pytest.fail('Circular friction called'))
    for tube,assumptions in ((replace(ellipse(),roughness_inner=1e-5),FD),):
        with pytest.raises(TubeGeometryApplicabilityError):
            internal_heat_transfer_diagnostics(tube=tube,m_dot=.005,flow_area=tube.flow_area,props=PROPS,assumptions=assumptions)
        with pytest.raises(TubeGeometryApplicabilityError):
            internal_friction(tube=tube,reynolds=800.,assumptions=assumptions)
    assert internal_heat_transfer_diagnostics(tube=ellipse(),m_dot=.005,
        flow_area=ellipse().flow_area,props=PROPS).model_metadata.regime=='laminar'
    assert straight_friction_evaluator(ellipse())(800.,None)[0]>0


def test_H1_circle_limit_does_not_change_existing_circular_T_default():
    tube=ellipse(1.)
    circle=BareTube(.02,.024,2.,1.8,20.)
    h1=elliptical_h1_heat_transfer(tube,.005,PROPS,assumptions=FD)
    old=heat_transfer_coefficient_internal_diagnostics(.005,circle.D_i,circle.flow_area,PROPS)
    assert h1.Nu_base==pytest.approx(48/11,rel=1e-14)
    assert old.Nu_base==3.66


@pytest.mark.parametrize('reynolds',[0.,-1.,math.nan,math.inf])
def test_invalid_reynolds(reynolds):
    with pytest.raises(ValueError): elliptical_laminar_friction(ellipse(),reynolds,assumptions=FD)
