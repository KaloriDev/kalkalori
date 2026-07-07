"""
Psychrometric property adapter based on PsychroLib.

KalKalori core uses SI units:
- temperature: K
- pressure: Pa
- relative humidity: fraction, 0.0 ... 1.0
- humidity ratio: kg_water / kg_dry_air
- moist air enthalpy: J / kg_dry_air
- moist air density: kg_moist_air / m3

Ref: ASHRAE Fundamentals / PsychroLib.
"""

from __future__ import annotations

import math

import psychrolib

psychrolib.SetUnitSystem(psychrolib.SI)


T_0_C_IN_K = 273.15


def _k_to_c(T: float) -> float:
    """Convert temperature from K to degC."""
    return T - T_0_C_IN_K


def _c_to_k(T_C: float) -> float:
    """Convert temperature from degC to K."""
    return T_C + T_0_C_IN_K


def _validate_temperature_k(T: float) -> None:
    if not math.isfinite(T):
        raise ValueError("Temperature must be finite [K].")
    if T <= 0.0:
        raise ValueError("Temperature must be above absolute zero [K].")


def _validate_pressure_pa(p: float) -> None:
    if not math.isfinite(p):
        raise ValueError("Pressure must be finite [Pa].")
    if p <= 0.0:
        raise ValueError("Pressure must be positive [Pa].")


def _validate_relative_humidity(RH: float) -> None:
    if not math.isfinite(RH):
        raise ValueError("Relative humidity must be finite [-].")
    if RH < 0.0 or RH > 1.0:
        raise ValueError("Relative humidity RH must be in range 0.0 ... 1.0 [-].")


def _validate_humidity_ratio(W: float) -> None:
    if not math.isfinite(W):
        raise ValueError("Humidity ratio must be finite [kg_water/kg_dry_air].")
    if W < 0.0:
        raise ValueError("Humidity ratio W must be non-negative [kg_water/kg_dry_air].")


def humidity_ratio_from_t_rh(T: float, RH: float, p: float) -> float:
    """Return humidity ratio from dry-bulb temperature and relative humidity.

    Args:
        T: Dry-bulb temperature [K].
        RH: Relative humidity [-], 0.0 ... 1.0.
        p: Total pressure [Pa].

    Returns:
        Humidity ratio W [kg_water/kg_dry_air].

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    _validate_temperature_k(T)
    _validate_relative_humidity(RH)
    _validate_pressure_pa(p)

    return psychrolib.GetHumRatioFromRelHum(_k_to_c(T), RH, p)


def relative_humidity_from_t_w(T: float, W: float, p: float) -> float:
    """Return relative humidity from dry-bulb temperature and humidity ratio.

    Args:
        T: Dry-bulb temperature [K].
        W: Humidity ratio [kg_water/kg_dry_air].
        p: Total pressure [Pa].

    Returns:
        Relative humidity RH [-].

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    _validate_temperature_k(T)
    _validate_humidity_ratio(W)
    _validate_pressure_pa(p)

    return psychrolib.GetRelHumFromHumRatio(_k_to_c(T), W, p)


def dew_point_from_t_rh(T: float, RH: float) -> float:
    """Return dew-point temperature from dry-bulb temperature and relative humidity.

    Args:
        T: Dry-bulb temperature [K].
        RH: Relative humidity [-], 0.0 ... 1.0.

    Returns:
        Dew-point temperature [K].

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    _validate_temperature_k(T)
    _validate_relative_humidity(RH)

    T_dew_C = psychrolib.GetTDewPointFromRelHum(_k_to_c(T), RH)
    return _c_to_k(T_dew_C)


def dew_point_from_t_w(T: float, W: float, p: float) -> float:
    """Return dew-point temperature from dry-bulb temperature and humidity ratio.

    Args:
        T: Dry-bulb temperature [K].
        W: Humidity ratio [kg_water/kg_dry_air].
        p: Total pressure [Pa].

    Returns:
        Dew-point temperature [K].

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    _validate_temperature_k(T)
    _validate_humidity_ratio(W)
    _validate_pressure_pa(p)

    T_dew_C = psychrolib.GetTDewPointFromHumRatio(_k_to_c(T), W, p)
    return _c_to_k(T_dew_C)


def moist_air_enthalpy_from_t_w(T: float, W: float) -> float:
    """Return moist-air enthalpy from dry-bulb temperature and humidity ratio.

    Args:
        T: Dry-bulb temperature [K].
        W: Humidity ratio [kg_water/kg_dry_air].

    Returns:
        Moist-air enthalpy [J/kg_dry_air].

    Notes:
        PsychroLib in SI returns moist-air enthalpy in J/kg dry air.
        Do not multiply the result by 1000.

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    _validate_temperature_k(T)
    _validate_humidity_ratio(W)

    return psychrolib.GetMoistAirEnthalpy(_k_to_c(T), W)


def moist_air_enthalpy_from_t_rh(T: float, RH: float, p: float) -> float:
    """Return moist-air enthalpy from dry-bulb temperature and relative humidity.

    Args:
        T: Dry-bulb temperature [K].
        RH: Relative humidity [-], 0.0 ... 1.0.
        p: Total pressure [Pa].

    Returns:
        Moist-air enthalpy [J/kg_dry_air].

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    W = humidity_ratio_from_t_rh(T, RH, p)
    return moist_air_enthalpy_from_t_w(T, W)


def moist_air_density_from_t_w(T: float, W: float, p: float) -> float:
    """Return moist-air density.

    Args:
        T: Dry-bulb temperature [K].
        W: Humidity ratio [kg_water/kg_dry_air].
        p: Total pressure [Pa].

    Returns:
        Moist-air density [kg_moist_air/m3].

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    _validate_temperature_k(T)
    _validate_humidity_ratio(W)
    _validate_pressure_pa(p)

    return psychrolib.GetMoistAirDensity(_k_to_c(T), W, p)


def moist_air_density_from_t_rh(T: float, RH: float, p: float) -> float:
    """Return moist-air density from dry-bulb temperature and relative humidity.

    Args:
        T: Dry-bulb temperature [K].
        RH: Relative humidity [-], 0.0 ... 1.0.
        p: Total pressure [Pa].

    Returns:
        Moist-air density [kg_moist_air/m3].

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    W = humidity_ratio_from_t_rh(T, RH, p)
    return moist_air_density_from_t_w(T, W, p)


def saturation_humidity_ratio(T: float, p: float) -> float:
    """Return saturation humidity ratio at dry-bulb temperature and pressure.

    Args:
        T: Dry-bulb temperature [K].
        p: Total pressure [Pa].

    Returns:
        Saturation humidity ratio [kg_water/kg_dry_air].

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    _validate_temperature_k(T)
    _validate_pressure_pa(p)

    return psychrolib.GetSatHumRatio(_k_to_c(T), p)


def moist_air_enthalpy(T: float, RH: float, p: float) -> float:
    """Backward-compatible wrapper.

    Return moist-air enthalpy from dry-bulb temperature, relative humidity
    and pressure.

    Args:
        T: Dry-bulb temperature [K].
        RH: Relative humidity [-], 0.0 ... 1.0.
        p: Total pressure [Pa].

    Returns:
        Moist-air enthalpy [J/kg_dry_air].

    Notes:
        Older versions of this adapter treated the PsychroLib result as kJ/kg
        and multiplied it by 1000. That was incorrect for PsychroLib SI mode.
    """
    return moist_air_enthalpy_from_t_rh(T, RH, p)


def dew_point_temperature(T: float, RH: float) -> float:
    """Backward-compatible wrapper for dew-point temperature.

    Args:
        T: Dry-bulb temperature [K].
        RH: Relative humidity [-], 0.0 ... 1.0.

    Returns:
        Dew-point temperature [K].
    """
    return dew_point_from_t_rh(T, RH)