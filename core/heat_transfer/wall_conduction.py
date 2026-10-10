# SPDX-License-Identifier: GPL-3.0-only
"""Physical wall resistance [K/W], selected independently of surface fins."""
import math
from enum import Enum
from core.geometry.applicability import require_circular_tube, TubeGeometryApplicabilityError


class TubeWallModel(str, Enum):
    CYLINDRICAL = "cylindrical"
    CONFOCAL_EXACT = "confocal_exact"


def tube_wall_resistance(tube, *, n_tubes=1, model=None):
    require_circular_tube(tube, "Cylindrical wall conduction")
    if model not in (None, TubeWallModel.CYLINDRICAL):
        raise TubeGeometryApplicabilityError("Select cylindrical wall conduction for circular tubes.")
    if isinstance(n_tubes, bool) or not isinstance(n_tubes, int) or n_tubes <= 0:
        raise ValueError("n_tubes must be a positive integer.")
    k = getattr(tube, "wall_k", None)
    if k is None:
        return 0.0
    if k <= 0.0:
        raise ValueError("wall_k must be positive.")
    Di = float(getattr(tube, "D_i"))
    Do = float(getattr(tube, "D_o"))
    L_eff = float(getattr(tube, "length_effective"))
    if Do <= Di:
        raise ValueError("Tube outer diameter must exceed inner diameter.")
    # Preserve the historical circular expression AND evaluation order.
    # Cylindrical steady Laplace solution; no equivalent diameter substitution.
    return math.log(Do / Di) / (2.0 * math.pi * k * L_eff * n_tubes)
