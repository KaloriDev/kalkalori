"""Explicit Robinson–Briggs fixture for historical thermal/regression tests.

These tests retain their original hydraulic reference and numerical assertions.
Preferred AUTO and its authorization/applicability rules have separate coverage
in extended_surface_auto_test.py. Explicit per-call overrides still take effect.
"""
from core.geometry import CircularFinnedTube,TubeSurfaceType
from core.models.bare_tube import BareTubeHeatExchanger


class RobinsonBriggsReferenceExchanger(BareTubeHeatExchanger):
    def _reference_options(self,options):
        if (isinstance(self.bundle.tube,CircularFinnedTube)
                and self.bundle.tube.surface_type is TubeSurfaceType.CIRCULAR_FINNED
                and self.bundle.layout=='staggered'):
            options.setdefault('finned_pressure_drop_provider','robinson_briggs_1966')
        return options

    def solve(self,*args,**kwargs):
        return super().solve(*args,**self._reference_options(kwargs))

    def rate(self,*args,**kwargs):
        return super().rate(*args,**self._reference_options(kwargs))

    def simulate(self,*args,**kwargs):
        return super().simulate(*args,**self._reference_options(kwargs))
