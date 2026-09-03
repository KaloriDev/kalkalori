# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only

"""Shared physical surface-margin definition for result-level APIs."""

from __future__ import annotations

import math


def calculate_surface_margin_factor(
    *,
    UA_actual: float,
    UA_process: float,
) -> float:
    """Return the relative excess of actual UA over process UA.

    Both values represent physical conductance [W/K] and therefore must be
    finite and strictly positive. Positive output is spare capacity, zero is
    an exact match, and negative output is a shortfall.
    """
    if not math.isfinite(UA_actual) or UA_actual <= 0.0:
        raise ValueError("UA_actual must be a positive finite value [W/K].")
    if not math.isfinite(UA_process) or UA_process <= 0.0:
        raise ValueError("UA_process must be a positive finite value [W/K].")

    margin = UA_actual / UA_process - 1.0
    if not math.isfinite(margin):
        raise ValueError("UA_actual / UA_process must produce a finite margin.")
    return margin
