"""
Condensation onset helpers for moist-air applications.

This module only checks whether condensation is thermodynamically possible
based on the bulk moist-air dew point and a given surface temperature.

It does not calculate condensation rate, latent heat transfer, film resistance,
or wet economizer performance.

Core unit convention:
- temperature: K
- pressure: Pa
- humidity ratio: kg_water / kg_dry_air

Ref: ASHRAE Fundamentals / PsychroLib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from core.common.warnings import ModelWarning
from core.psychrometrics.moist_air import MoistAirState
from core.psychrometrics.psychrolib_adapter import saturation_humidity_ratio

SOURCE = "psychrometrics.condensation"

@dataclass(frozen=True)
class CondensationOnsetResult:
    """Result of a simple condensation onset check.

    Attributes:
        will_condense:
            True if surface temperature is below moist-air dew point.
        T_surface:
            Surface temperature [K].
        T_dew:
            Bulk moist-air dew-point temperature [K].
        dew_point_margin:
            Positive value means surface is below dew point [K].
            Defined as T_dew - T_surface.
        W_bulk:
            Bulk humidity ratio [kg_water/kg_dry_air].
        W_surface_sat:
            Saturation humidity ratio at surface temperature
            and bulk pressure [kg_water/kg_dry_air].
    """

    will_condense: bool
    T_surface: float
    T_dew: float
    dew_point_margin: float
    W_bulk: float
    W_surface_sat: float
    warnings: list[ModelWarning] = field(default_factory=list)

def _warning(code: str, message: str, severity: str = "warning") -> ModelWarning:
    return ModelWarning(
        code=code,
        message=message,
        severity=severity,
        source=SOURCE,
    )

def check_condensation_onset(
    air: MoistAirState,
    T_surface: float,
    ) -> CondensationOnsetResult:
    """Check if condensation is thermodynamically possible.

    Condensation onset criterion:

        T_surface < T_dew

    Args:
        air:
            Bulk moist-air state.
        T_surface:
            Surface temperature [K].

    Returns:
        CondensationOnsetResult.

    Notes:
        This is only an onset check. It does not calculate condensate mass flow,
        wet surface fraction, latent heat duty, or wall-temperature iteration.

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    if not math.isfinite(T_surface):
        raise ValueError("Surface temperature must be finite [K].")
    if T_surface <= 0.0:
        raise ValueError("Surface temperature must be above absolute zero [K].")

    warnings: list[ModelWarning] = []

    dew_point_margin = air.T_dew - T_surface
    will_condense = dew_point_margin > 0.0
    W_surface_sat = saturation_humidity_ratio(T_surface, air.p)

    if will_condense:
        warnings.append(
            _warning(
                code="CONDENSATION_ONSET",
                message=(
                    "Surface temperature is below moist-air dew point; "
                    "condensation is thermodynamically possible."
                ),
                severity="info",
            )
        )

    if T_surface < 273.15:
        warnings.append(
            _warning(
                code="SURFACE_BELOW_FREEZING",
                message=(
                    "Surface temperature is below 273.15 K. Ice or frost formation "
                    "is outside the current condensation-onset MVP assumptions."
                ),
                severity="warning",
            )
        )

    if will_condense and W_surface_sat >= air.W:
        warnings.append(
            _warning(
                code="CONDENSATION_NUMERICAL_INCONSISTENCY",
                message=(
                    "Condensation was detected from dew-point criterion, but "
                    "surface saturation humidity ratio is not below bulk humidity ratio."
                ),
                severity="warning",
            )
        )

    return CondensationOnsetResult(
        will_condense=will_condense,
        T_surface=T_surface,
        T_dew=air.T_dew,
        dew_point_margin=dew_point_margin,
        W_bulk=air.W,
        W_surface_sat=W_surface_sat,
        warnings=warnings,
    )
