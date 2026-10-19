# SPDX-License-Identifier: GPL-3.0-only
"""Independent analytical benchmark and rectangle/ellipse convergence checks."""
from dataclasses import replace
import math

import numpy as np
import pytest

from core.geometry import EllipticalTube, RectangularIndividualFin, RectangularFinnedEllipticalTube
from core.heat_transfer.thin_plate_fin import (
    ThinPlateFinEfficiency2D, FinEfficiencyConvergenceError,
    _assemble_mesh, _solve_mesh, _ellipse_rectangle_mesh, _evaluate_cached,
    _split_quadrilaterals)


def geometry():
    return RectangularFinnedEllipticalTube(
        EllipticalTube(.012, .006, .010, .004, 2., 1.8, 40.),
        RectangularIndividualFin(.040, .030, .0004, .003, 200.))


def straight_mesh(nx):
    # Rectangular strip with an isothermal left edge and three insulated edges.
    ny, length, width = 4, .020, .008
    points = np.array([(x, y) for x in np.linspace(0, length, nx+1)
                       for y in np.linspace(0, width, ny+1)])
    quads = []
    for i in range(nx):
        for j in range(ny):
            p = i*(ny+1)+j
            quads.append((p, p+ny+1, p+ny+2, p+1))
    points, triangles = _split_quadrilaterals(points, np.array(quads))
    stiffness, weights = _assemble_mesh(points, triangles)
    return points, stiffness, weights, np.arange(ny+1)


def test_straight_insulated_tip_analytical_benchmark():
    # Solve the ODE directly: theta(x)=cosh(m*(L-x))/cosh(m*L),
    # integral(theta)/L = tanh(m*L)/(m*L). No production reference evaluator.
    h, k, t, length = 100., 200., .0005, .020
    m = math.sqrt(2*h/(k*t))
    exact_eta = math.tanh(m*length)/(m*length)
    errors = []
    for nx in (16, 32, 64):
        points, stiffness, weights, base = straight_mesh(nx)
        eta, theta, residual, balance, q_base, q_air = _solve_mesh(stiffness, weights, base, h, k, t)
        exact_theta = np.cosh(m*(length-points[:, 0]))/math.cosh(m*length)
        errors.append(abs(eta-exact_eta))
        assert np.max(np.abs(theta-exact_theta)) < 3.e-4
        assert residual < 1.e-9
        assert balance < 1.e-9
        assert q_base == pytest.approx(q_air, rel=1.e-9)
    assert errors[1] < errors[0]/3.8
    assert errors[2] < errors[1]/3.8
    assert errors[-1] < 1.e-5


@pytest.mark.parametrize('h', [0., 1.e-10, 10., 80., 300., 10000.])
def test_physical_bounds_and_diagnostics(h):
    r = ThinPlateFinEfficiency2D().evaluate(geometry(), h)
    assert 0 <= r.eta_fin <= 1
    assert min(r.quarter_theta) >= -1.e-9
    assert max(r.quarter_theta) <= 1 + 1.e-9
    assert r.converged
    assert r.convergence_estimate <= 5.e-4
    assert r.mesh_area_relative_error <= 5.e-4
    assert r.relative_residual < 1.e-8
    assert r.energy_balance_relative_error < 1.e-8
    if h <= 1.e-10:
        assert r.eta_fin == pytest.approx(1., abs=1.e-10)


def test_monotonicity_and_conductivity_thickness_limits():
    t = geometry()
    model = ThinPlateFinEfficiency2D()
    by_h = [model.evaluate(t, h).eta_fin for h in (1., 40., 100., 300.)]
    assert by_h == sorted(by_h, reverse=True)
    for name, values in [('thermal_conductivity', (10., 100., 200., 1.e12)),
                          ('thickness', (.0001, .0004, .001, 1.e8))]:
        efficiencies = []
        for value in values:
            # Changing pitch only keeps the formal t->infinity limit valid
            # as geometry; pitch does not enter the conduction problem.
            f = replace(t.fin, **{name: value}, axial_pitch=max(.003, 2*value) if name == 'thickness' else .003)
            efficiencies.append(model.evaluate(replace(t, fin=f), 80.).eta_fin)
        assert efficiencies == sorted(efficiencies)
        assert efficiencies[-1] == pytest.approx(1., abs=1.e-8)


def test_mesh_boundary_and_refinement():
    t = geometry()
    results = [ThinPlateFinEfficiency2D(n, n, max_refinements=0).evaluate(t, 80.)
               for n in (8, 16, 32, 64)]
    differences = [abs(a.eta_fin-b.eta_fin) for a, b in zip(results, results[1:])]
    assert differences[1] < differences[0]/3
    assert differences[2] < differences[1]/3
    assert differences[-1] < 5.e-5
    for r in results:
        assert not r.converged and r.convergence_estimate is None
        points = np.array(r.quarter_coordinates)
        stride = 2*r.angular_cells_per_sector + 1
        contact = points[:stride]
        assert np.allclose((contact[:, 0]/.012)**2 + (contact[:, 1]/.006)**2, 1.)
        assert np.allclose(r.quarter_theta[:stride], 1.)
        outside = points[r.radial_cells*stride:(r.radial_cells+1)*stride]
        assert np.all(np.isclose(outside[:, 0], .020) | np.isclose(outside[:, 1], .015))
    assert results[-1].mesh_area_relative_error < results[0].mesh_area_relative_error/60


def test_symmetry_and_no_spurious_transverse_straight_fin_gradient():
    # Natural zero flux on the side boundaries preserves a y-independent
    # analytical field on the same 2D operator, without imposing that field.
    _, stiffness, weights, base = straight_mesh(64)
    _, theta, *_ = _solve_mesh(stiffness, weights, base, 80., 200., .0004)
    assert np.max(np.ptp(theta[:65*5].reshape(-1, 5), axis=1)) < 1.e-12
    # A circle in a square additionally has x/y exchange symmetry. Compare
    # independently assembled angular/radial nodes reflected in x=y.
    t = geometry()
    t = replace(t, core_tube=replace(t.core_tube, outer_semi_minor_axis=.012),
                fin=replace(t.fin, outer_height=.040))
    r = ThinPlateFinEfficiency2D().evaluate(t, 80.)
    count = (r.radial_cells+1)*(2*r.angular_cells_per_sector+1)
    theta = np.array(r.quarter_theta[:count]).reshape(r.radial_cells+1, -1)
    assert np.max(np.abs(theta-theta[:, ::-1])) < 1.e-5


def test_cache_ignores_length_count_pitch_and_orientation():
    t = geometry()
    model = ThinPlateFinEfficiency2D()
    r = model.evaluate(t, 80.)
    altered = replace(t, core_tube=replace(t.core_tube, length_effective=.9, orientation_angle=.4),
                      fin=replace(t.fin, axial_pitch=.006))
    assert model.evaluate(altered, 80.) is r
    assert _evaluate_cached.cache_info().hits > 0
    assert _ellipse_rectangle_mesh.cache_info().hits > 0


def test_invalid_inputs_and_unmet_convergence_are_explicit():
    for kwargs in ({'radial_cells': True}, {'angular_cells_per_sector': 1},
                   {'tolerance': 0.}, {'max_refinements': -1}):
        with pytest.raises(ValueError):
            ThinPlateFinEfficiency2D(**kwargs)
    for h in (-1., math.inf, math.nan):
        with pytest.raises(ValueError):
            ThinPlateFinEfficiency2D().evaluate(geometry(), h)
    with pytest.raises(TypeError):
        ThinPlateFinEfficiency2D().evaluate(geometry().core_tube, 80.)
    with pytest.raises(FinEfficiencyConvergenceError, match='did not converge'):
        ThinPlateFinEfficiency2D(2, 2, tolerance=1.e-12, max_refinements=1).evaluate(geometry(), 80.)
