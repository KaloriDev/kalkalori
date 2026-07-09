# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Physical heat exchanger models.

Models orchestrate geometry and heat-transfer correlations into
usable engineering objects.
"""

from .bare_tube import (
    BareTubeHeatExchanger,
    HXResult,
    HXOutSideThermalResults,
    HXTubeSideHydraulicResults,
    HXOutSideHydraulicResults,
)
from .mean_property_rating import (
    RatingSideInput,
    HXRatingResult,
    run_rating,
)

__all__ = [
    "BareTubeHeatExchanger",
    "HXResult",
    "HXOutSideThermalResults",
    "HXTubeSideHydraulicResults",
    "HXOutSideHydraulicResults",
    "RatingSideInput",
    "HXRatingResult",
    "run_rating",
]
