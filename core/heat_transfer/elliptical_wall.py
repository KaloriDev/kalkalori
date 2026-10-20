# SPDX-License-Identifier: GPL-3.0-only
"""P1 Laplace conduction between nested concentric coaxial ellipse contours.

Independent weak-form implementation: integral(grad(T).grad(v))=0.
T_inner=1, T_outer=0; quarter-section axes have zero normal flux.
No equivalent thickness, confocal contour substitution or circular model.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from scipy.sparse.linalg import spsolve

from core.geometry.applicability import TubeGeometryApplicabilityError
from core.geometry.elliptical_tube import EllipticalTube
from core.heat_transfer.thin_plate_fin import (
    _assemble_mesh, _split_quadrilaterals, FinEfficiencyConvergenceError)


class WallConductionConvergenceError(RuntimeError):
    """No accepted numerical wall resistance is available."""


@dataclass(frozen=True)
class EllipticalWallConvergence:
    unit_conductance: float  # full section Q/(k*L*dT), dimensionless
    radial_cells: int
    angular_cells: int  # quarter section
    refinements: int
    relative_resistance_change: float
    relative_heat_balance_error: float
    relative_residual: float
    relative_area_error: float
    converged: bool = True


def _solve_level(bo, ai, bi, nr, na):
    angles=np.linspace(0,math.pi/2,na+1)
    inner=np.column_stack((ai*np.cos(angles),bi*np.sin(angles)))
    outer=np.column_stack((np.cos(angles),bo*np.sin(angles)))
    s=np.linspace(0,1,nr+1)[:,None,None]
    points=(inner[None,:,:]*(1-s)+outer[None,:,:]*s).reshape(-1,2)
    stride=na+1
    lo=(np.arange(nr)[:,None]*stride+np.arange(na)).ravel()
    quads=np.column_stack((lo,lo+1,lo+1+stride,lo+stride))
    points,triangles=_split_quadrilaterals(points,quads)
    try:
        matrix,weights=_assemble_mesh(points,triangles)
    except FinEfficiencyConvergenceError as exc:
        raise WallConductionConvergenceError('Degenerate elliptical wall mesh; check dimensions/resolution.') from exc
    inside=np.arange(stride)
    outside=nr*stride+inside
    fixed=np.concatenate((inside,outside))
    free=np.setdiff1d(np.arange(len(points)),fixed)
    t=np.zeros(len(points)); t[inside]=1.
    rhs=-np.asarray(matrix[free][:,fixed]@t[fixed]).ravel()
    t[free]=spsolve(matrix[free][:,free],rhs,permc_spec='MMD_AT_PLUS_A')
    reaction=matrix@t
    qin=float(reaction[inside].sum())
    qout=-float(reaction[outside].sum())
    balance=abs(qin-qout)/qin
    residual=float(np.linalg.norm(reaction[free])/np.linalg.norm(rhs))
    exact_area=math.pi*(bo-ai*bi)/4
    area_error=abs(float(weights.sum())/exact_area-1)
    if not (np.all(np.isfinite(t)) and math.isfinite(qin) and qin>0 and
            math.isfinite(balance) and balance<1e-8 and residual<1e-8):
        raise WallConductionConvergenceError('Elliptical wall solve failed positivity, heat-balance or residual checks.')
    return 4*qin,balance,residual,area_error


@lru_cache(maxsize=32)
def _converged_solution(bo, ai, bi, radial_cells, angular_cells, tolerance, max_refinements):
    previous=None
    for level in range(max_refinements+1):
        nr,na=radial_cells*2**level,angular_cells*2**level
        if 4*nr*na>1_000_000:
            raise WallConductionConvergenceError('Elliptical wall mesh budget exceeded before convergence.')
        conductance,balance,residual,area_error=_solve_level(bo,ai,bi,nr,na)
        resistance=1/conductance
        change=math.inf if previous is None else abs(resistance-previous)/resistance
        if change<=tolerance and area_error<=tolerance:
            return EllipticalWallConvergence(conductance,nr,na,level,change,balance,residual,area_error)
        previous=resistance
    raise WallConductionConvergenceError(
        f'Elliptical wall resistance did not converge: relative change={change:.6g}, area error={area_error:.6g}, tolerance={tolerance:.6g}.')


def elliptical_wall_solution(tube, *, radial_cells=8, angular_cells=32,
                             tolerance=1e-4, max_refinements=4):
    """Converged unit conductance; geometrically similar sections share cache.

    Both mesh directions double together. The measured resistance change is
    an a posteriori convergence indicator, not a certified continuum bound.
    Constant k, isothermal contours and negligible axial/end conduction apply.
    """
    if not isinstance(tube,EllipticalTube):
        raise TubeGeometryApplicabilityError('Numerical elliptical wall requires EllipticalTube geometry.')
    for name,value in (('radial_cells',radial_cells),('angular_cells',angular_cells),('max_refinements',max_refinements)):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'{name} must be a positive integer.')
    if not math.isfinite(tolerance) or not 0<tolerance<1:
        raise ValueError('tolerance must be finite and between 0 and 1.')
    ao=tube.outer_semi_major_axis
    return _converged_solution(tube.outer_semi_minor_axis/ao,
        tube.inner_semi_major_axis/ao,tube.inner_semi_minor_axis/ao,
        radial_cells,angular_cells,tolerance,max_refinements)
