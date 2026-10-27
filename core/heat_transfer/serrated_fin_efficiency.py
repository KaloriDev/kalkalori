# SPDX-License-Identifier: GPL-3.0-only
"""Future serrated conduction boundary; no efficiency model in Phase A."""
from typing import Protocol

from core.geometry.serrated_finned_tube import SerratedCircularFinnedTube


class SerratedFinEfficiencyProvider(Protocol):
    """Future fin-area-based efficiency [0..1], with explicit provenance.

    ``outside_htc`` is physical dry film HTC [W/(m2 K)]. Contact resistance
    stays in the shared network, outside this conduction calculation. This
    protocol is not yet accepted by physical Rating/Simulation execution.
    """

    provider_name: str
    source_reference: str

    def evaluate(self, tube: SerratedCircularFinnedTube, outside_htc: float) -> float: ...
