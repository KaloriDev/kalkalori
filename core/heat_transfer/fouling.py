# SPDX-License-Identifier: GPL-3.0-only
"""Validation of exchanger-side area-specific thermal fouling [m² K/W]."""
from math import isfinite


def normalize_fouling_resistance(value: float | None, name: str) -> float:
    """Omitted/None means clean; installed hydraulic geometry is unaffected."""
    value = 0.0 if value is None else float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative [m² K/W].")
    return value
