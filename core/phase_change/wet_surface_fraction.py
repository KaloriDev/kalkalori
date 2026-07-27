# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""0D wet-surface-fraction diagnostic (spec section 24).

This is a diagnostic estimate only -- a 0D model has no spatial resolution
to report an exact wetted fraction of the outside tube-bank surface.
Method: reuse the existing four-endpoint 0D wall-temperature envelope
(``core.heat_transfer.thermal_iteration.estimate_wall_temperature_envelope``,
the same tool the dry solver already uses for
``wall_temperature_envelope``), compare each probe's outside wall
temperature against the dew point appropriate to its position (inlet-
paired probes against the inlet dew point, outlet-paired probes against the
outlet dew point -- water content, and therefore dew point, differs between
the two ends once condensation has removed some water), ramp each
comparison to a bounded [0, 1] indicator across the same activation band
used for regime deciding, and average the available indicators.

No 1D/segmented model is implied or added; see
``docs/property_models.md`` for the stated limitations of this estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.common.warnings import ModelWarning, make_warning
from core.heat_transfer.thermal_iteration import estimate_wall_temperature_envelope

from core.phase_change import warning_codes as WC
from core.phase_change.types import PhaseChangeCapability
from core.phase_change.water_equilibrium import (
    is_frost_regime,
    water_dew_point,
    water_mole_fraction_from_ratio,
    water_partial_pressure,
)
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio

METHOD_NAME = "0d_endpoint_wet_fraction_estimate"

SOURCE = "wet_surface_fraction"


@dataclass(frozen=True)
class WetSurfaceFractionEstimate:
    wet_surface_fraction: float
    method: str
    wall_temperature_min: float
    wall_temperature_max: float
    warnings: tuple[ModelWarning, ...] = ()


def _dew_point_or_none(capability: PhaseChangeCapability, W: float, *, p: float) -> float | None:
    if W <= 0.0:
        return None
    y = water_mole_fraction_from_ratio(W, M_dry=capability.M_dry, M_h2o=capability.M_condensable)
    p_h2o = water_partial_pressure(y, p)
    if is_frost_regime(p_h2o):
        return None
    return water_dew_point(p_h2o)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def estimate_wet_surface_fraction(
    hx,
    *,
    inside_provider,
    outside_capability: PhaseChangeCapability,
    m_dot_inside: float,
    m_dot_dry_carrier: float,
    T_in_inside: float,
    T_out_inside: float,
    T_in_outside: float,
    T_out_outside: float,
    W_in: float,
    W_out: float,
    p_inside: float,
    p_outside: float,
    euler_provider: str = "zukauskas",
    activation_band_K: float = 0.5,
) -> WetSurfaceFractionEstimate:
    W_mean = 0.5 * (W_in + W_out)
    outside_provider = wet_gas_provider_at_water_ratio(outside_capability, W_mean)
    m_dot_gas_mean = m_dot_dry_carrier * (1.0 + W_mean)

    envelope = estimate_wall_temperature_envelope(
        hx,
        m_dot_inside=m_dot_inside,
        m_dot_outside=m_dot_gas_mean,
        inside_provider=inside_provider,
        outside_provider=outside_provider,
        inside_inlet_temperature=T_in_inside,
        inside_outlet_temperature=T_out_inside,
        outside_inlet_temperature=T_in_outside,
        outside_outlet_temperature=T_out_outside,
        p_inside=p_inside,
        p_outside=p_outside,
        euler_provider=euler_provider,
    )

    dew_point_in = _dew_point_or_none(outside_capability, W_in, p=p_outside)
    dew_point_out = _dew_point_or_none(outside_capability, W_out, p=p_outside)

    warnings: list[ModelWarning] = [
        make_warning(
            code=WC.WET_SURFACE_FRACTION_0D_ESTIMATE,
            message=(
                "outside: wet_surface_fraction is a 0D endpoint estimate "
                f"({METHOD_NAME}), not a spatially resolved (1D/segmented) "
                "wetted-area fraction."
            ),
            source=SOURCE,
            severity="info",
        )
    ]

    fractions: list[float] = []
    for probe in envelope.probes:
        if not probe.converged or not math.isfinite(probe.outside_wall_temperature):
            continue
        is_inlet_probe = math.isclose(
            probe.outside_bulk_temperature, T_in_outside, rel_tol=0.0, abs_tol=1e-6
        )
        dew_local = dew_point_in if is_inlet_probe else dew_point_out
        if dew_local is None:
            continue
        margin = dew_local - probe.outside_wall_temperature
        fractions.append(_clamp01(0.5 + margin / activation_band_K))

    if fractions:
        wet_fraction = sum(fractions) / len(fractions)
    else:
        # No usable probe: fall back to a coarse mean/min/max-based estimate
        # using only the min/max envelope extrema (spec section 24: "if the
        # current model does not have enough points, apply a minimal
        # estimate based on the available mean/min/max temperatures").
        warnings.append(
            make_warning(
                code="wet_surface_fraction_fallback_estimate",
                message=(
                    "outside: no wall-temperature endpoint probe could be "
                    "compared against a dew point; wet_surface_fraction "
                    "falls back to a coarse min/max-based estimate."
                ),
                source=SOURCE,
                severity="warning",
            )
        )
        dew_ref = dew_point_out or dew_point_in
        if dew_ref is None or not math.isfinite(envelope.outside_min) or not math.isfinite(envelope.outside_max):
            wet_fraction = 1.0
        else:
            wall_mean = 0.5 * (envelope.outside_min + envelope.outside_max)
            wet_fraction = _clamp01(0.5 + (dew_ref - wall_mean) / activation_band_K)

    return WetSurfaceFractionEstimate(
        wet_surface_fraction=wet_fraction,
        method=METHOD_NAME,
        wall_temperature_min=envelope.outside_min,
        wall_temperature_max=envelope.outside_max,
        warnings=tuple(warnings),
    )
