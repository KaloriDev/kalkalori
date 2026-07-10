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
from .simulation import (
    HXSideInput,
    HXSimulationResult,
    run_simulation,
)
from .heat_balance import (
    BalanceSideSpec,
    ClosedBalanceSide,
    ClosedBalance,
    close_heat_balance,
)
from .rating import (
    HXRatingResult,
    run_rating,
)

__all__ = [
    "BareTubeHeatExchanger",
    "HXResult",
    "HXOutSideThermalResults",
    "HXTubeSideHydraulicResults",
    "HXOutSideHydraulicResults",
    "HXSideInput",
    "HXSimulationResult",
    "run_simulation",
    "BalanceSideSpec",
    "ClosedBalanceSide",
    "ClosedBalance",
    "close_heat_balance",
    "HXRatingResult",
    "run_rating",
]
