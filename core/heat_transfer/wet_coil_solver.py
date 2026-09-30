# SPDX-License-Identifier: GPL-3.0-only
"""Numerical accuracy and one-operation wall-time budget for wet coils."""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from math import isfinite
from numbers import Real
from time import monotonic


@dataclass(frozen=True)
class WetCoilSolverOptions:
    """Convergence controls, never physical admissibility allowances.

    ``timeout_s=None`` intentionally permits unlimited validation runs.
    Rating and Simulation share the same controls throughout nested solves.
    """

    energy_tolerance_W: float = 1.0
    mass_tolerance_kg_s: float = 5e-7
    outlet_temperature_tolerance_K: float = 0.01
    timeout_s: float | None = 300.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "timeout_s" and value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite"
                                 + (" or None" if name == "timeout_s" else ""))


class WetCoilConvergenceError(RuntimeError):
    """Controlled numerical failure, distinct from an inadmissible trial."""


class WetCoilTimeoutError(WetCoilConvergenceError):
    def __init__(self, **diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"Wet-coil operation timed out: {diagnostics}")


class _WetSolveBudget:
    def __init__(self, options):
        if not isinstance(options, WetCoilSolverOptions):
            raise TypeError("wet_solver_options must be WetCoilSolverOptions")
        self.used = False
        self.options = options
        self.started = monotonic()
        self.deadline = (None if options.timeout_s is None
                         else self.started + options.timeout_s)
        self.diagnostics = dict(forward_evaluations=0, property_iterations=0,
                                joint_solver_evaluations=0)

    def check(self, **state):
        self.diagnostics.update(state)
        if self.deadline is not None:
            now = monotonic()
            if now >= self.deadline:
                raise WetCoilTimeoutError(elapsed_s=now - self.started,
                                          timeout_s=self.options.timeout_s,
                                          **self.diagnostics)

    def count(self, name):
        self.diagnostics[name] += 1
        self.check()


_active_budget = ContextVar("wet_coil_operation_budget", default=None)


def _solve_budget(options=None, budget=None):
    budget = budget or _active_budget.get()
    if budget is not None:
        if options is not None and options != budget.options:
            raise ValueError("Nested wet solve must use the operation's options")
        return budget
    return _WetSolveBudget(options if options is not None else WetCoilSolverOptions())


def _wet_operation(function):
    """Start before public preflight; nested public calls retain the budget."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        budget = _solve_budget(kwargs.get("wet_solver_options"))
        kwargs["wet_solver_options"] = budget.options
        token = _active_budget.set(budget)
        try:
            result = function(*args, **kwargs)
            if budget.used:
                budget.check()
                diagnostics = getattr(result, "wet_coil_diagnostics", None)
                if diagnostics is not None:
                    diagnostics["solver_statistics"] = dict(
                        budget.diagnostics, elapsed_s=monotonic() - budget.started)
            return result
        finally:
            _active_budget.reset(token)
    return wrapped
