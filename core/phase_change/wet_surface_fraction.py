# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""0D linear wet-surface-fraction estimate (spec sections 8-9, v0.6.0 fix).

This is a diagnostic *and* modelling estimate: it also scales the mass-
transfer area used inside ``core.phase_change.outside_condensation_solver``
(``A_wet = A_outside * wet_surface_fraction``), so unlike a purely
cosmetic diagnostic it must be cheap enough to evaluate every wet-solver
iteration -- see that module for how ``wall_temperature_min``/``_max`` are
obtained inexpensively per iteration (a closed-form two-point estimate,
not the full four-probe ``WallTemperatureEnvelope``).

Model: linear interpolation of the local dew point between the estimated
minimum and maximum wall temperature (spec section 8):

    wet_surface_fraction = (T_dew - T_wall_min) / (T_wall_max - T_wall_min)

clamped to ``[0, 1]``, with ``T_dew <= T_wall_min -> 0`` and
``T_dew >= T_wall_max -> 1``. This assumes an approximately linear surface-
temperature distribution between the two estimated extremes -- a
deliberate 0D simplification (see ``docs/property_models.md``), not a
spatially resolved (1D/segmented) result.

No 1D/segmented model, row-by-row interpolation, endpoint-area weighting,
or statistical temperature-distribution model is added here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


LINEAR_METHOD_NAME = "linear_wall_temperature_envelope"
DEGENERATE_METHOD_NAME = "uniform_wall_temperature_fallback"


@dataclass(frozen=True)
class WetSurfaceFractionEstimate:
    wet_surface_fraction: float
    method: str


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def estimate_wet_surface_fraction(
    *,
    dew_point_temperature: float,
    wall_temperature_min: float,
    wall_temperature_mean: float,
    wall_temperature_max: float,
    temperature_span_tolerance_K: float = 1e-3,
    activation_band_K: float = 0.5,
) -> WetSurfaceFractionEstimate:
    """Return the linear 0D wet-surface-fraction estimate.

    Args:
        dew_point_temperature: Local dew point [K] for the current wet-gas
            composition (spec section 10: use the dew point matching the
            *current* state, not a fixed inlet value, inside the wet
            solver's iteration).
        wall_temperature_min, wall_temperature_mean, wall_temperature_max:
            Estimated wall-temperature extrema and mean [K].
            ``wall_temperature_max`` must be >= ``wall_temperature_min``.
        temperature_span_tolerance_K: Guard against dividing by a
            near-zero ``(wall_temperature_max - wall_temperature_min)``
            span (spec section 9). Must be > 0.
        activation_band_K: Only used in the degenerate (near-zero span)
            fallback, to decide "clearly wet"/"clearly dry"/"near onset"
            relative to ``wall_temperature_mean``; same meaning as
            ``core.phase_change.regime``'s activation band. Must be > 0.

    Returns:
        WetSurfaceFractionEstimate with ``wet_surface_fraction`` always a
        finite value in ``[0, 1]`` (never NaN/infinity) and ``method``
        naming which branch was used
        (``"linear_wall_temperature_envelope"`` or
        ``"uniform_wall_temperature_fallback"``).
    """
    if not math.isfinite(temperature_span_tolerance_K) or temperature_span_tolerance_K <= 0.0:
        raise ValueError("temperature_span_tolerance_K must be a positive finite value.")
    if not math.isfinite(activation_band_K) or activation_band_K <= 0.0:
        raise ValueError("activation_band_K must be a positive finite value.")
    for name, value in (
        ("dew_point_temperature", dew_point_temperature),
        ("wall_temperature_min", wall_temperature_min),
        ("wall_temperature_mean", wall_temperature_mean),
        ("wall_temperature_max", wall_temperature_max),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite value, got {value!r}.")

    span = wall_temperature_max - wall_temperature_min

    if abs(span) <= temperature_span_tolerance_K:
        # Degenerate envelope (near-uniform wall temperature): fall back to
        # a mean-based ramp across the activation band -- never divide by
        # the near-zero span (spec section 9).
        margin = dew_point_temperature - wall_temperature_mean
        half_band = activation_band_K / 2.0
        if margin >= half_band:
            fraction = 1.0
        elif margin <= -half_band:
            fraction = 0.0
        else:
            fraction = _clamp01(0.5 + margin / activation_band_K)
        return WetSurfaceFractionEstimate(wet_surface_fraction=fraction, method=DEGENERATE_METHOD_NAME)

    if dew_point_temperature <= wall_temperature_min:
        fraction = 0.0
    elif dew_point_temperature >= wall_temperature_max:
        fraction = 1.0
    else:
        fraction = (dew_point_temperature - wall_temperature_min) / span

    # Numerical safety clamp only (span > tolerance > 0 already guarantees
    # fraction in [0, 1] analytically); never used to mask a sign error.
    fraction = _clamp01(fraction)

    return WetSurfaceFractionEstimate(wet_surface_fraction=fraction, method=LINEAR_METHOD_NAME)
