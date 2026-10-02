# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic separate-package provider; imports only KalKalori's public API."""
from dataclasses import replace

from core import WetCoilOperationContext, WetCoilSolverOptions
from core.models.simulation import HXSideInput


class ExternalWetProvider:
    model_id = "synthetic_external_wet"
    model_name = "External contract fixture"
    source = "Synthetic results; no physical model"
    applicability = "Contract test inputs"
    supports_simulation = True
    supports_rating = True

    def __init__(self, simulation, rating):
        self.simulation_result = simulation
        self.rating_result = rating
        self.calls = []
        self.applicable = True
        self.failure = None
        self.nested_simulation = False

    def is_applicable(self, hx, inside, outside, *, settings, **options):
        return self.applicable

    def _solve(self, mode, *, context: WetCoilOperationContext,
               wet_solver_options: WetCoilSolverOptions, settings, **options):
        context.check(operation=mode)
        self.calls.append((mode, context, wet_solver_options, options))
        if self.failure is not None:
            raise self.failure
        result = self.simulation_result if mode == "simulation" else self.rating_result
        return replace(result, wet_coil_diagnostics={"external_marker": mode})

    def simulate(self, hx, inside, outside, **kwargs):
        return self._solve("simulation", **kwargs)

    def rate(self, hx, inside, outside, **kwargs):
        result = self._solve("rating", **kwargs)
        if self.nested_simulation:
            simulation = hx.simulate(
                HXSideInput(inside.provider, inside.m_dot, inside.T_in, inside.p),
                HXSideInput(outside.provider, outside.m_dot, outside.T_in, outside.p),
                wet_coil_provider=self,
                wet_solver_options=kwargs["wet_solver_options"],
            )
            result = replace(result, simulation=simulation)
        return result
