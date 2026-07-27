# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Stable dry/condensing regime decision from a dry sensible-only baseline.

This module implements the "possible" leg of the capable/possible/active
distinction (see ``core.phase_change.types``): given a dry baseline's wall
temperature (or wall-temperature envelope, when no converged
``IterativeThermalState`` is available -- the ``iterate=False`` case) and a
capability's dew point, decide whether the exchanger is clearly dry,
clearly condensing, or in an uncertain band close to onset.

Anti-oscillation (v0.6.0 spec, section 18)
-------------------------------------------
The regime is decided *once*, from the dry baseline, before any wet
iteration starts (``core.phase_change.integration``), and is then held
fixed for the remainder of that solver call -- there is no per-iteration
``if condenses: wet else: dry`` branch anywhere in this package. A result
landing inside the activation band is deliberately resolved to the DRY
regime (not condensing), so a borderline case never flips into an
expensive/uncertain wet solve; ``PHASE_CHANGE_NEAR_ONSET`` documents that
choice on the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ThermalRegime(str, Enum):
    DRY = "dry"
    CONDENSING = "condensing"
    NEAR_ONSET = "near_onset"


@dataclass(frozen=True)
class RegimeDecision:
    """Outcome of comparing one side's representative wall temperature
    against its dew point, once, from the dry baseline."""

    regime: ThermalRegime
    dew_point_K: float
    wall_temperature_representative_K: float
    margin_K: float  # dew_point - wall_temperature; positive => below dew point

    @property
    def is_condensing(self) -> bool:
        return self.regime is ThermalRegime.CONDENSING

    @property
    def is_near_onset(self) -> bool:
        return self.regime is ThermalRegime.NEAR_ONSET


def validate_onset_settings(onset_tolerance_K: float, activation_band_K: float) -> None:
    if not math.isfinite(onset_tolerance_K) or onset_tolerance_K < 0.0:
        raise ValueError("phase_change_onset_tolerance_K must be >= 0.")
    if not math.isfinite(activation_band_K) or activation_band_K <= 0.0:
        raise ValueError("phase_change_activation_band_K must be > 0.")


def decide_regime(
    *,
    dew_point_K: float,
    wall_temperature_representative_K: float,
    onset_tolerance_K: float = 0.0,
    activation_band_K: float = 0.5,
) -> RegimeDecision:
    """Decide DRY / CONDENSING / NEAR_ONSET from one comparison.

    ``margin_K = dew_point_K - wall_temperature_representative_K``:
    positive means the wall is below the dew point (condensation tendency).

    A stable band of half-width ``activation_band_K/2`` is centered at
    ``onset_tolerance_K`` (>= 0, an extra safety margin required before
    committing to the condensing regime): margins clearly above the band
    are CONDENSING, clearly below are DRY, and margins inside the band are
    NEAR_ONSET (resolved to DRY by the caller, see module docstring).
    """
    validate_onset_settings(onset_tolerance_K, activation_band_K)

    margin = dew_point_K - wall_temperature_representative_K
    half_band = activation_band_K / 2.0

    if margin > onset_tolerance_K + half_band:
        regime = ThermalRegime.CONDENSING
    elif margin < onset_tolerance_K - half_band:
        regime = ThermalRegime.DRY
    else:
        regime = ThermalRegime.NEAR_ONSET

    return RegimeDecision(
        regime=regime,
        dew_point_K=dew_point_K,
        wall_temperature_representative_K=wall_temperature_representative_K,
        margin_K=margin,
    )


def representative_wall_temperature(*, side: str, thermal_state, wall_envelope) -> float:
    """Return one representative wall temperature [K] for regime deciding.

    Prefers the converged mean-bulk ``IterativeThermalState`` wall
    temperature (``iterate=True``, the normal case); falls back to the
    midpoint of the 0D endpoint envelope when no converged thermal state is
    available (``iterate=False``, see ``core.phase_change.integration``).
    """
    if side not in ("inside", "outside"):
        raise ValueError(f"side must be 'inside' or 'outside', got {side!r}.")

    if thermal_state is not None:
        return (
            thermal_state.outside_wall_temperature
            if side == "outside"
            else thermal_state.inside_wall_temperature
        )

    if side == "outside":
        return 0.5 * (wall_envelope.outside_min + wall_envelope.outside_max)
    return 0.5 * (wall_envelope.inside_min + wall_envelope.inside_max)
