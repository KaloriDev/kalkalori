# SPDX-License-Identifier: GPL-3.0-only
"""Private numerical policies; approximate states initialize strict solves only."""
from dataclasses import dataclass
from functools import lru_cache
from numpy.polynomial.legendre import leggauss
from core.heat_transfer.wet_coil_solver import WetCoilSolverOptions, _WetSolveBudget

@lru_cache(maxsize=32)
def gauss(order):
    """Exact invariant nodes; read-only so callers cannot corrupt the cache."""
    nodes, weights = leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights

@dataclass(frozen=True)
class Fidelity:
    name: str
    energy_W: float
    mass_kg_s: float
    outlet_K: float
    region_temperature_K: float
    fraction_tolerance: float
    inverse_temperature_K: float
    quadrature: int

    def budget(self, parent):
        options = WetCoilSolverOptions(self.energy_W, self.mass_kg_s, self.outlet_K,
                                       parent.options.timeout_s)
        child = _WetSolveBudget(options)
        child.started, child.deadline = parent.started, parent.deadline
        child.diagnostics = parent.diagnostics
        child.fidelity = self
        return child


@dataclass(frozen=True)
class _WetNumerics:
    """Internal safeguards for the optimized path, never a public selector."""
    analytic_derivatives: bool = True
    region_continuation: bool = True
    front_continuation: bool = True
    region_acceleration: bool = True
    cold_start: bool = True
    shared_water: bool = True
    point_reuse: bool = True
    defer_outlet: bool = True


COARSE = Fidelity("coarse", 75.0, 1e-5, 0.25, 2e-6, 2e-8, 5e-9, 6)
MEDIUM = Fidelity("medium", 8.0, 1e-6, 0.04, 2e-7, 2e-9, 1e-10, 8)
INITIAL = Fidelity("initial", 75.0, 1e-5, 0.25, 2e-6, 2e-8, 5e-9, 3)
