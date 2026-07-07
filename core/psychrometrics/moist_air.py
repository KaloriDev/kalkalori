"""
Moist-air state helpers for KalKalori.

KalKalori core uses SI units:
- temperature: K
- pressure: Pa
- relative humidity: fraction, 0.0 ... 1.0
- humidity ratio: kg_water / kg_dry_air
- moist-air enthalpy: J / kg_dry_air
- moist-air density: kg_moist_air / m3

Ref: ASHRAE Fundamentals / PsychroLib.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.psychrometrics.psychrolib_adapter import (
    dew_point_from_t_rh,
    dew_point_from_t_w,
    humidity_ratio_from_t_rh,
    moist_air_density_from_t_w,
    moist_air_enthalpy_from_t_w,
    relative_humidity_from_t_w,
    saturation_humidity_ratio,
)


@dataclass(frozen=True)
class MoistAirState:
    """Complete point state of moist air.

    Attributes:
        T: Dry-bulb temperature [K].
        p: Total pressure [Pa].
        W: Humidity ratio [kg_water/kg_dry_air].
        RH: Relative humidity [-].
        h: Moist-air enthalpy [J/kg_dry_air].
        rho: Moist-air density [kg_moist_air/m3].
        T_dew: Dew-point temperature [K].
    """

    T: float
    p: float
    W: float
    RH: float
    h: float
    rho: float
    T_dew: float


def moist_air_state_from_t_rh(T: float, RH: float, p: float) -> MoistAirState:
    """Build moist-air state from dry-bulb temperature and relative humidity.

    Args:
        T: Dry-bulb temperature [K].
        RH: Relative humidity [-], 0.0 ... 1.0.
        p: Total pressure [Pa].

    Returns:
        MoistAirState.

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    W = humidity_ratio_from_t_rh(T, RH, p)
    h = moist_air_enthalpy_from_t_w(T, W)
    rho = moist_air_density_from_t_w(T, W, p)
    T_dew = dew_point_from_t_rh(T, RH)

    return MoistAirState(
        T=T,
        p=p,
        W=W,
        RH=RH,
        h=h,
        rho=rho,
        T_dew=T_dew,
    )


def moist_air_state_from_t_w(T: float, W: float, p: float) -> MoistAirState:
    """Build moist-air state from dry-bulb temperature and humidity ratio.

    Args:
        T: Dry-bulb temperature [K].
        W: Humidity ratio [kg_water/kg_dry_air].
        p: Total pressure [Pa].

    Returns:
        MoistAirState.

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    RH = relative_humidity_from_t_w(T, W, p)
    h = moist_air_enthalpy_from_t_w(T, W)
    rho = moist_air_density_from_t_w(T, W, p)
    T_dew = dew_point_from_t_w(T, W, p)

    return MoistAirState(
        T=T,
        p=p,
        W=W,
        RH=RH,
        h=h,
        rho=rho,
        T_dew=T_dew,
    )


def saturated_moist_air_state(T: float, p: float) -> MoistAirState:
    """Build saturated moist-air state at given dry-bulb temperature.

    Args:
        T: Dry-bulb temperature [K].
        p: Total pressure [Pa].

    Returns:
        MoistAirState at RH = 1.0.

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    W_sat = saturation_humidity_ratio(T, p)

    return moist_air_state_from_t_w(
        T=T,
        W=W_sat,
        p=p,
    )