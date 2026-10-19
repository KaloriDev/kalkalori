# SPDX-License-Identifier: GPL-3.0-only
"""Independent thin-plate conduction implementation; no air-side correlation.

Physics: Taler & Marcinkowski (2020), CET 81, 1333-1338,
https://www.aidic.it/cet/20/81/223.pdf, equations (1)-(2), and
Marcinkowski & Taler (2021), Energies 14, 577, sections 2-3,
https://doi.org/10.3390/en14030577 (CC BY 4.0).

Solve -laplacian(theta) + 2*h/(k*t)*theta = 0, theta=1 on the
elliptical contact, natural zero flux on the rectangle and symmetry axes.
P1 triangles with lumped reaction mass; only existing NumPy/SciPy are used.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from core.geometry.rectangular_finned_tube import RectangularFinnedEllipticalTube


class FinEfficiencyConvergenceError(ValueError):
    """The requested mesh accuracy or physical temperature bounds were not met."""


@dataclass(frozen=True)
class ThinPlateFinEfficiencyResult:
    """One-fin diagnostics; temperatures/coordinates describe the first quadrant.

    eta_fin is the area average of theta on the polygonal numerical domain.
    Heat rates [W/K of base excess temperature] integrate that same domain,
    mirrored four times. Geometry/network areas remain analytical. The mesh
    area error separately measures the polygonal ellipse approximation.
    convergence_estimate is an absolute successive-mesh eta difference, not
    a rigorous error bound. A zero-refinement solve is explicitly unverified.
    """

    eta_fin: float
    radial_cells: int
    angular_cells_per_sector: int
    nodes: int
    elements: int
    convergence_estimate: float | None
    converged: bool
    relative_residual: float
    energy_balance_relative_error: float
    mesh_area_relative_error: float
    base_heat_rate_per_temperature: float
    convective_heat_rate_per_temperature: float
    quarter_coordinates: tuple[tuple[float, float], ...]
    quarter_theta: tuple[float, ...]
    boundary_condition: str = "isothermal_contact_adiabatic_outer_edge"

    @property
    def fin_efficiency(self):
        return self.eta_fin


@dataclass(frozen=True)
class ThinPlateFinEfficiency2D:
    """Deterministic rectangle/ellipse solver with measured mesh refinement.

    Each angular sector spans an axis to the rectangle corner ray. Uniform
    radial rings connect exact ellipse nodes to the exact rectangle. Ellipse
    chords converge quadratically; the rectangle and symmetry axes are exact.
    No CAD mesh, contact resistance, edge convection or nonuniform h is used.
    Meshes and solutions have bounded caches keyed by physical inputs; tube
    length, count, orientation and bank rows never enter the fin PDE cache.
    """

    radial_cells: int = 16
    angular_cells_per_sector: int = 16
    tolerance: float = 5.e-4
    max_refinements: int = 3

    def __post_init__(self):
        for name in ('radial_cells', 'angular_cells_per_sector'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2.")
        if (isinstance(self.max_refinements, bool) or not isinstance(self.max_refinements, int)
                or not 0 <= self.max_refinements <= 5):
            raise ValueError("max_refinements must be an integer in [0, 5].")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be positive and finite.")

    def evaluate(self, tube: RectangularFinnedEllipticalTube, h: float) -> ThinPlateFinEfficiencyResult:
        """Evaluate using physical film h [W/m2/K], before any area efficiency."""
        if not isinstance(tube, RectangularFinnedEllipticalTube):
            raise TypeError("ThinPlateFinEfficiency2D requires RectangularFinnedEllipticalTube.")
        if not math.isfinite(h) or h < 0:
            raise ValueError("h must be non-negative and finite.")
        core, fin = tube.core_tube, tube.fin
        return _evaluate_cached(core.outer_semi_major_axis, core.outer_semi_minor_axis,
                                fin.outer_width, fin.outer_height, fin.thickness,
                                fin.thermal_conductivity, h, self)


def _assemble_mesh(points, triangles):
    """P1 weak diffusion operator and lumped planar integration weights.

    Ke_ij = area * grad(Ni).grad(Nj); each vertex receives area/3.
    This small kernel also permits an independent straight-fin benchmark.
    """
    vertices = points[triangles]
    x, y = vertices[:, :, 0], vertices[:, :, 1]
    twice_area = ((x[:, 1]-x[:, 0])*(y[:, 2]-y[:, 0])
                  - (x[:, 2]-x[:, 0])*(y[:, 1]-y[:, 0]))
    area = np.abs(twice_area) / 2
    if np.any(area <= 0) or not np.all(np.isfinite(area)):
        raise FinEfficiencyConvergenceError("Degenerate fin mesh; check dimensions/resolution.")
    gx = np.stack((y[:, 1]-y[:, 2], y[:, 2]-y[:, 0], y[:, 0]-y[:, 1]), axis=1) / twice_area[:, None]
    gy = np.stack((x[:, 2]-x[:, 1], x[:, 0]-x[:, 2], x[:, 1]-x[:, 0]), axis=1) / twice_area[:, None]
    local = area[:, None, None] * (gx[:, :, None]*gx[:, None, :] + gy[:, :, None]*gy[:, None, :])
    rows = np.repeat(triangles, 3, axis=1).ravel()
    cols = np.tile(triangles, (1, 3)).ravel()
    stiffness = coo_matrix((local.ravel(), (rows, cols)), shape=(len(points), len(points))).tocsr()
    # Enforce the constant nullspace against roundoff in element summation.
    stiffness.setdiag(0.)
    stiffness.setdiag(-np.asarray(stiffness.sum(axis=1)).ravel())
    weights = np.bincount(triangles.ravel(), weights=np.repeat(area/3, 3), minlength=len(points))
    return stiffness, weights


def _split_quadrilaterals(points, quads):
    """Four triangles about each cell center preserve reflection symmetry."""
    centers = points[quads].mean(axis=1)
    center_ids = np.arange(len(points), len(points)+len(quads))
    triangles = np.concatenate([
        np.column_stack((quads[:, i], quads[:, (i+1) % 4], center_ids))
        for i in range(4)])
    return np.concatenate((points, centers)), triangles


@lru_cache(maxsize=32)
def _ellipse_rectangle_mesh(a, b, width, height, nr, na):
    corner = math.atan2(height*a, width*b)
    angles = np.concatenate((np.linspace(0, corner, na+1),
                             np.linspace(corner, math.pi/2, na+1)[1:]))
    inner = np.column_stack((a*np.cos(angles), b*np.sin(angles)))
    inner[0, 1], inner[-1, 0] = 0., 0.
    outer = inner / np.maximum(inner[:, 0]/(width/2), inner[:, 1]/(height/2))[:, None]
    fraction = np.linspace(0, 1, nr+1)[:, None, None]
    points = (inner[None, :, :] + fraction*(outer-inner)[None, :, :]).reshape(-1, 2)
    stride = len(angles)
    index = (np.arange(nr)[:, None]*stride + np.arange(stride-1)[None, :]).ravel()
    quads = np.column_stack((index, index+stride, index+stride+1, index+1))
    points, triangles = _split_quadrilaterals(points, quads)
    base = np.arange(stride)
    stiffness, weights = _assemble_mesh(points, triangles)
    return points, triangles, base, stiffness, weights


def _solve_mesh(stiffness, weights, base, h, k, thickness):
    """Solve temperature deficit for stability as h -> 0; return diagnostics."""
    m_squared = (2 * (h / k)) / thickness
    if not math.isfinite(m_squared):
        raise ValueError("Fin m_squared is not numerically representable.")
    matrix = stiffness + diags(m_squared * weights)
    free = np.ones(len(weights), dtype=bool)
    free[base] = False
    deficit = np.zeros(len(weights))
    rhs = m_squared * weights[free]
    if h > 0:
        deficit[free] = spsolve(matrix[free][:, free], rhs, permc_spec='MMD_AT_PLUS_A')
    theta = 1 - deficit
    if (not np.all(np.isfinite(theta)) or np.min(theta) < -1.e-9
            or np.max(theta) > 1 + 1.e-9):
        raise FinEfficiencyConvergenceError("Fin temperature bounds violated; refine the mesh.")
    residual = np.asarray(matrix @ deficit)[free] - rhs
    relative_residual = float(np.max(np.abs(residual)) / max(np.max(np.abs(rhs)), np.finfo(float).tiny))
    eta = float(np.dot(weights, theta) / weights.sum())
    # Algebraic base reaction includes the lumped boundary-node convection.
    base_heat = (-float(np.sum((stiffness @ deficit)[base])) * k * thickness
                 + 2*h*float(weights[base].sum()))
    convective_heat = 2*h*float(np.dot(weights, theta))
    balance = abs(base_heat-convective_heat) / max(abs(convective_heat), np.finfo(float).tiny)
    return min(1., max(0., eta)), theta, relative_residual, balance, base_heat, convective_heat


@lru_cache(maxsize=128)
def _evaluate_cached(a, b, width, height, thickness, k, h, model):
    previous, delta = None, None
    nr, na = model.radial_cells, model.angular_cells_per_sector
    exact_quarter_area = (width*height - math.pi*a*b)/4
    for level in range(model.max_refinements+1):
        points, triangles, base, stiffness, weights = _ellipse_rectangle_mesh(a, b, width, height, nr, na)
        eta, theta, residual, balance, base_heat, convective_heat = _solve_mesh(stiffness, weights, base, h, k, thickness)
        area_error = abs(float(weights.sum())/exact_quarter_area - 1)
        if previous is not None:
            delta = abs(eta-previous)
        converged = delta is not None and delta <= model.tolerance and area_error <= model.tolerance
        if converged or level == model.max_refinements:
            break
        previous = eta
        nr, na = nr*2, na*2
    if model.max_refinements and not converged:
        raise FinEfficiencyConvergenceError(
            f"Fin mesh did not converge: eta difference={delta}, area error={area_error}; "
            "increase resolution/max_refinements or relax tolerance explicitly.")
    return ThinPlateFinEfficiencyResult(
        eta, nr, na, len(points), len(triangles), delta, converged,
        residual, balance, area_error, 4*base_heat, 4*convective_heat,
        tuple(map(tuple, points.tolist())), tuple(theta.tolist()))
