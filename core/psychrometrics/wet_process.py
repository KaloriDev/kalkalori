"""
Psychrometric wet-process helpers.

This module provides small helper functions needed before implementing a
wet economizer solver. It works only for moist air represented by
`MoistAirState`.

It does not calculate:
- heat-exchanger UA,
- NTU/effectiveness,
- wall-temperature iteration,
- condensate mass flow in kg/s,
- gas mixtures other than moist air.

For general gas-mixture properties, a separate CoolProp-based backend should
be introduced before the wet economizer solver stage.

Core SI units:
- temperature: K
- pressure: Pa
- humidity ratio: kg_water / kg_dry_air
- moist-air enthalpy: J / kg_dry_air
- condensable water: kg_water / kg_dry_air

Refs:
- ASHRAE Fundamentals, moist air and psychrometrics.
- PsychroLib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from core.common.warnings import ModelWarning
from core.psychrometrics.condensation import (
    CondensationOnsetResult,
    check_condensation_onset,
)
from core.psychrometrics.moist_air import (
    MoistAirState,
    moist_air_state_from_t_w,
    saturated_moist_air_state,
)


SOURCE = "psychrometrics.wet_process"


@dataclass(frozen=True)
class WetSurfaceProcessResult:
    """Limiting psychrometric process result at a surface temperature.

    Attributes:
        inlet:
            Bulk moist-air state.
        T_surface:
            Surface temperature [K].
        condensation:
            Condensation onset result based on dew point.
        surface_saturated_state:
            Saturated moist-air state at surface temperature and bulk pressure.
        equilibrium_state:
            Limiting state at surface temperature:
            - saturated state if condensation is possible,
            - constant-W state if condensation is not possible.
        condensable_water:
            Maximum water that can condense per kg dry air
            [kg_water/kg_dry_air].
        enthalpy_drop:
            Enthalpy drop from inlet state to equilibrium state
            [J/kg_dry_air].
        warnings:
            Model warnings from condensation and wet-process checks.
    """

    inlet: MoistAirState
    T_surface: float
    condensation: CondensationOnsetResult
    surface_saturated_state: MoistAirState
    equilibrium_state: MoistAirState
    condensable_water: float
    enthalpy_drop: float
    warnings: list[ModelWarning] = field(default_factory=list)


def saturated_state_at_surface(
    air: MoistAirState,
    T_surface: float,
) -> MoistAirState:
    """Return saturated moist-air state at surface temperature.

    Args:
        air:
            Bulk moist-air state. Its pressure is used for the surface state.
        T_surface:
            Surface temperature [K].

    Returns:
        Saturated moist-air state at `T_surface` and `air.p`.
    """
    _validate_surface_temperature(T_surface)
    return saturated_moist_air_state(T=T_surface, p=air.p)


def humidity_ratio_drop_to_saturation(
    air: MoistAirState,
    T_surface: float,
) -> float:
    """Return humidity-ratio drop to surface saturation.

    Args:
        air:
            Bulk moist-air state.
        T_surface:
            Surface temperature [K].

    Returns:
        `max(air.W - W_sat_surface, 0.0)` [kg_water/kg_dry_air].

    Notes:
        This is the maximum water content reduction possible if air reaches
        saturation at the surface temperature.
    """
    surface_sat = saturated_state_at_surface(air=air, T_surface=T_surface)
    return max(air.W - surface_sat.W, 0.0)


def condensable_water_per_kg_dry_air(
    air: MoistAirState,
    T_surface: float,
) -> float:
    """Return maximum condensable water per kg dry air.

    This is an alias with clearer engineering meaning for
    `humidity_ratio_drop_to_saturation`.
    """
    return humidity_ratio_drop_to_saturation(air=air, T_surface=T_surface)


def enthalpy_drop_to_surface_saturation(
    air: MoistAirState,
    T_surface: float,
) -> float:
    """Return enthalpy drop to saturated state at surface temperature.

    Args:
        air:
            Bulk moist-air state.
        T_surface:
            Surface temperature [K].

    Returns:
        Enthalpy drop [J/kg_dry_air].

    Notes:
        If the surface is not below dew point, this function returns 0.0.
        The saturated state at a warmer-than-dew-point surface is not the
        correct dry cooling limit for the bulk air.
    """
    result = wet_surface_process_limit(air=air, T_surface=T_surface)
    return result.enthalpy_drop


def wet_surface_process_limit(
    air: MoistAirState,
    T_surface: float,
) -> WetSurfaceProcessResult:
    """Return limiting moist-air process quantities at a surface temperature.

    Args:
        air:
            Bulk moist-air state.
        T_surface:
            Surface temperature [K].

    Returns:
        WetSurfaceProcessResult.

    Notes:
        This function defines only a local psychrometric limit. It does not
        decide whether a heat exchanger reaches this state.
    """
    _validate_moist_air_state(air)
    _validate_surface_temperature(T_surface)

    warnings: list[ModelWarning] = []

    condensation = check_condensation_onset(
        air=air,
        T_surface=T_surface,
    )
    warnings.extend(condensation.warnings)

    surface_sat = saturated_state_at_surface(
        air=air,
        T_surface=T_surface,
    )

    if condensation.will_condense:
        equilibrium_state = surface_sat
        condensable_water = max(air.W - surface_sat.W, 0.0)
        enthalpy_drop = max(air.h - surface_sat.h, 0.0)

        if condensable_water <= 0.0:
            warnings.append(
                _warning(
                    code="WET_PROCESS_NO_W_DROP",
                    message=(
                        "Condensation was detected by dew-point criterion, but "
                        "calculated humidity-ratio drop is zero or negative."
                    ),
                )
            )

        if enthalpy_drop <= 0.0:
            warnings.append(
                _warning(
                    code="WET_PROCESS_NO_ENTHALPY_DROP",
                    message=(
                        "Condensation was detected by dew-point criterion, but "
                        "calculated enthalpy drop is zero or negative."
                    ),
                )
            )
    else:
        equilibrium_state = moist_air_state_from_t_w(
            T=T_surface,
            W=air.W,
            p=air.p,
        )
        condensable_water = 0.0
        enthalpy_drop = max(air.h - equilibrium_state.h, 0.0)

    if T_surface > air.T:
        warnings.append(
            _warning(
                code="SURFACE_ABOVE_BULK_TEMPERATURE",
                message=(
                    "Surface temperature is above bulk dry-bulb temperature. "
                    "This is outside the usual cooling / condensation use case."
                ),
                severity="info",
            )
        )

    return WetSurfaceProcessResult(
        inlet=air,
        T_surface=T_surface,
        condensation=condensation,
        surface_saturated_state=surface_sat,
        equilibrium_state=equilibrium_state,
        condensable_water=condensable_water,
        enthalpy_drop=enthalpy_drop,
        warnings=warnings,
    )


def _validate_moist_air_state(air: MoistAirState) -> None:
    _validate_temperature(air.T, "air.T")
    _validate_pressure(air.p)
    _validate_humidity_ratio(air.W)

    if not math.isfinite(air.RH):
        raise ValueError("Relative humidity must be finite [-].")
    if air.RH < 0.0:
        raise ValueError("Relative humidity must be non-negative [-].")

    if not math.isfinite(air.h):
        raise ValueError("Moist-air enthalpy must be finite [J/kg_dry_air].")

    if not math.isfinite(air.rho):
        raise ValueError("Moist-air density must be finite [kg/m3].")
    if air.rho <= 0.0:
        raise ValueError("Moist-air density must be positive [kg/m3].")

    _validate_temperature(air.T_dew, "air.T_dew")


def _validate_surface_temperature(T_surface: float) -> None:
    _validate_temperature(T_surface, "T_surface")


def _validate_temperature(T: float, name: str) -> None:
    if not math.isfinite(T):
        raise ValueError(f"{name} must be finite [K].")
    if T <= 0.0:
        raise ValueError(f"{name} must be above absolute zero [K].")


def _validate_pressure(p: float) -> None:
    if not math.isfinite(p):
        raise ValueError("Pressure must be finite [Pa].")
    if p <= 0.0:
        raise ValueError("Pressure must be positive [Pa].")


def _validate_humidity_ratio(W: float) -> None:
    if not math.isfinite(W):
        raise ValueError("Humidity ratio must be finite [kg_water/kg_dry_air].")
    if W < 0.0:
        raise ValueError("Humidity ratio must be non-negative [kg_water/kg_dry_air].")


def _warning(code: str, message: str, severity: str = "warning") -> ModelWarning:
    return ModelWarning(
        code=code,
        message=message,
        severity=severity,
        source=SOURCE,
    )