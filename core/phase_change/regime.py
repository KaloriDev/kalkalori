# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Stable dry/condensing regime decision from a dry sensible-only baseline.

This module implements the "possible" leg of the capable/possible/active
distinction (see ``core.phase_change.types``): given a dry baseline's
*minimum* estimated wall temperature (see ``evaluate_condensation_onset``
docstring for why the minimum, not the mean, is used) and a capability's
dew point, decide whether the exchanger is clearly dry, clearly condensing,
or in an uncertain band close to onset.

Anti-oscillation (v0.6.0 spec, section 18)
-------------------------------------------
The regime is decided *once*, from the dry baseline, before any wet
iteration starts (``core.phase_change.integration``), and is then held
fixed for the remainder of that solver call -- there is no per-iteration
``if condenses: wet else: dry`` branch anywhere in this package. A result
landing inside the activation band is deliberately resolved to *not
active* (``near_onset=True``, ``active=False``), so a borderline case
never flips into an expensive/uncertain wet solve; ``PHASE_CHANGE_NEAR_
ONSET`` documents that choice on the result.
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
class OnsetDecision:
    """Pure, numeric-only outcome of comparing a minimum wall temperature
    against a dew point (fix, v0.6.0 patch).

    ``possible`` and ``active`` are deliberately distinct (spec section 7):
    ``possible`` is true for the whole band at/above ``onset_tolerance_K -
    activation_band_K/2`` (near-onset *and* clearly-condensing margins);
    ``active`` is true only once the margin clears the *whole* activation
    band, so the solver only actually runs the wet case when it is
    confidently past onset. ``possible=True, near_onset=True, active=False``
    is an expected, valid combination -- callers must not collapse
    ``possible`` and ``active`` into the same flag.
    """

    margin_K: float  # dew_point - wall_temperature_min; positive => below dew point
    possible: bool
    active: bool
    near_onset: bool


def validate_onset_settings(onset_tolerance_K: float, activation_band_K: float) -> None:
    if not math.isfinite(onset_tolerance_K) or onset_tolerance_K < 0.0:
        raise ValueError("phase_change_onset_tolerance_K must be >= 0.")
    if not math.isfinite(activation_band_K) or activation_band_K <= 0.0:
        raise ValueError("phase_change_activation_band_K must be > 0.")


def evaluate_condensation_onset(
    *,
    dew_point_temperature: float,
    wall_temperature_min: float,
    onset_tolerance_K: float = 0.0,
    activation_band_K: float = 0.5,
) -> OnsetDecision:
    """Decide condensation onset from the *minimum* estimated wall temperature.

    Fix (v0.6.0 patch): onset must be evaluated against the coldest
    estimated point of the surface, not a mean or averaged wall
    temperature -- a mean-based check can miss condensation that has
    already started on the locally coldest part of the surface while the
    bulk-averaged wall is still nominally above the dew point. This
    function takes plain numeric values and has no dependency on
    ``MoistAirState``, PsychroLib, a gas-mixture provider, exchanger
    geometry, or which side (inside/outside) is being evaluated -- callers
    supply ``wall_temperature_min`` for whichever side/point they care
    about (``core.phase_change.integration`` passes
    ``wall_temperature_envelope.outside_min``/``.inside_min``).

    ``margin_K = dew_point_temperature - wall_temperature_min``: positive
    means the coldest point of the wall is below the dew point.

    A stable band of half-width ``activation_band_K/2`` is centered at
    ``onset_tolerance_K`` (>= 0, an extra safety margin required before
    committing to the *active* condensing regime):

    - ``margin_K <= onset_tolerance_K - activation_band_K/2``: clearly dry
      (``possible=False``, ``active=False``, ``near_onset=False``).
    - ``onset_tolerance_K - activation_band_K/2 < margin_K <=
      onset_tolerance_K + activation_band_K/2``: near onset
      (``possible=True``, ``active=False``, ``near_onset=True``).
    - ``margin_K > onset_tolerance_K + activation_band_K/2``: clearly
      condensing (``possible=True``, ``active=True``, ``near_onset=False``).
    """
    validate_onset_settings(onset_tolerance_K, activation_band_K)

    margin = dew_point_temperature - wall_temperature_min
    half_band = activation_band_K / 2.0

    active = margin > onset_tolerance_K + half_band
    possible = margin > onset_tolerance_K - half_band
    near_onset = possible and not active

    return OnsetDecision(margin_K=margin, possible=possible, active=active, near_onset=near_onset)


@dataclass(frozen=True)
class RegimeDecision:
    """Backward-compatible wrapper around ``OnsetDecision`` exposing the
    ``ThermalRegime`` enum shape used elsewhere in this package.

    ``DRY`` <-> not possible; ``NEAR_ONSET`` <-> possible and not active;
    ``CONDENSING`` <-> active. ``onset`` carries the underlying
    ``OnsetDecision`` for callers that want the raw possible/active/
    near_onset booleans directly (preferred for new code; see
    ``evaluate_condensation_onset``).
    """

    regime: ThermalRegime
    dew_point_K: float
    wall_temperature_representative_K: float
    margin_K: float
    onset: OnsetDecision

    @property
    def possible(self) -> bool:
        return self.onset.possible

    @property
    def active(self) -> bool:
        return self.onset.active

    @property
    def is_condensing(self) -> bool:
        return self.onset.active

    @property
    def is_near_onset(self) -> bool:
        return self.onset.near_onset


def decide_regime(
    *,
    dew_point_K: float,
    wall_temperature_representative_K: float,
    onset_tolerance_K: float = 0.0,
    activation_band_K: float = 0.5,
) -> RegimeDecision:
    """Decide DRY / CONDENSING / NEAR_ONSET from one wall-temperature value.

    Thin wrapper over ``evaluate_condensation_onset``. Callers deciding
    condensation *onset* should prefer passing the minimum estimated wall
    temperature here (see ``core.phase_change.integration``, which now
    passes ``wall_temperature_envelope.<side>_min``, not a mean/
    representative value -- this is the section-6 fix). This wrapper
    remains available for existing call sites and tests that use the
    ``ThermalRegime`` enum shape.
    """
    onset = evaluate_condensation_onset(
        dew_point_temperature=dew_point_K,
        wall_temperature_min=wall_temperature_representative_K,
        onset_tolerance_K=onset_tolerance_K,
        activation_band_K=activation_band_K,
    )
    if onset.active:
        regime = ThermalRegime.CONDENSING
    elif onset.near_onset:
        regime = ThermalRegime.NEAR_ONSET
    else:
        regime = ThermalRegime.DRY

    return RegimeDecision(
        regime=regime,
        dew_point_K=dew_point_K,
        wall_temperature_representative_K=wall_temperature_representative_K,
        margin_K=onset.margin_K,
        onset=onset,
    )


def representative_wall_temperature(*, side: str, thermal_state, wall_envelope) -> float:
    """Return one *mean* representative wall temperature [K].

    Used only for reporting/fallback purposes (condensate temperature,
    liquid enthalpy, mean properties -- spec section 6.3), never for the
    condensation-onset decision (section 6.1 -- use
    ``wall_envelope.<side>_min`` / ``evaluate_condensation_onset`` for
    that). Prefers the converged mean-bulk ``IterativeThermalState`` wall
    temperature (``iterate=True``, the normal case); falls back to the
    midpoint of the 0D endpoint envelope when no converged thermal state is
    available (``iterate=False``).
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
