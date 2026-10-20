# SPDX-License-Identifier: GPL-3.0-only
"""Public synthetic geometry and algebra fixtures, never private notebooks."""
import math
from core.geometry import EllipticalTube, RectangularIndividualFin, RectangularFinnedEllipticalTube, TubeBundle
from core.properties.common import FluidTransportProperties


def source_bundle(n_rows=8, n_passes=2):
    # Explicit confocal inner contour, not an inferred nominal wall thickness.
    ai = .0175
    bi = math.sqrt(ai*ai-(.018**2-.007**2))
    core = EllipticalTube(.018,.007,ai,bi,1.,1.,45.,roughness_inner=0.)
    tube = RectangularFinnedEllipticalTube(core,RectangularIndividualFin(.055,.026,.0003,.004,45.))
    return TubeBundle(tube,n_rows,6,.031,.060,'staggered',n_passes,flow_arrangement='counterflow')


AIR = FluidTransportProperties(rho=1.2,mu=1.85e-5,k=.026,cp=1006.)
