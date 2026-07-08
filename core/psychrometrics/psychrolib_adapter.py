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


def saturation_vapor_pressure(T: float) -> float:
    """Return saturation vapor pressure of water.

    Args:
        T: Temperature [K].

    Returns:
        Saturation vapor pressure [Pa].

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    _validate_temperature_k(T)
    return psychrolib.GetSatVapPres(_k_to_c(T))


def saturation_humidity_ratio_is_defined(T: float, p: float) -> bool:
    """Return True if saturated moist air is physically defined at T and p.

    Saturated moist air at dry-bulb temperature T requires water saturation
    vapor pressure to be below the total pressure.

    If p_ws >= p, a saturated moist-air state at this T and p is not physically
    meaningful. This commonly happens above the water boiling temperature at
    the given pressure.
    """
    _validate_temperature_k(T)
    _validate_pressure_pa(p)

    return saturation_vapor_pressure(T) < p


def max_relative_humidity_at_t_p(T: float, p: float) -> float:
    """Return maximum RH before water-vapor partial pressure reaches total pressure.

    Args:
        T: Dry-bulb temperature [K].
        p: Total pressure [Pa].

    Returns:
        Maximum RH [-].

    Notes:
        For ordinary sub-boiling psychrometric states this returns 1.0.
        Above the boiling temperature at a given pressure, this returns a value
        below 1.0. In that range RH is usually not the most convenient input
        variable; W or vapor mole fraction is clearer.
    """
    _validate_temperature_k(T)
    _validate_pressure_pa(p)

    p_ws = saturation_vapor_pressure(T)
    return min(1.0, p / p_ws)


def humidity_ratio_from_g_per_kg_dry_air(W_g_per_kg_da: float) -> float:
    """Convert humidity ratio from g_water/kg_dry_air to kg_water/kg_dry_air."""
    if not math.isfinite(W_g_per_kg_da):
        raise ValueError("Humidity ratio must be finite [g_water/kg_dry_air].")
    if W_g_per_kg_da < 0.0:
        raise ValueError("Humidity ratio must be non-negative [g_water/kg_dry_air].")

    return W_g_per_kg_da / 1000.0


def humidity_ratio_to_g_per_kg_dry_air(W: float) -> float:
    """Convert humidity ratio from kg_water/kg_dry_air to g_water/kg_dry_air."""
    _validate_humidity_ratio(W)
    return W * 1000.0


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

    p_ws = saturation_vapor_pressure(T)
    p_v = RH * p_ws

    if p_v >= p:
        rh_max = max_relative_humidity_at_t_p(T, p)
        raise ValueError(
            "The given T/RH/p state is not physically valid because "
            "RH * saturation_vapor_pressure(T) is greater than or equal to "
            "the total pressure. "
            f"T={T:.6g} K, RH={RH:.6g}, p={p:.6g} Pa, "
            f"p_ws={p_ws:.6g} Pa, p_v={p_v:.6g} Pa, "
            f"maximum RH at this T and p is about {rh_max:.6g}. "
            "For high-temperature gas mixtures, define moisture content using "
            "humidity ratio W [kg/kg dry air] or W_g_per_kg_da [g/kg dry air] "
            "instead of RH."
        )

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

    Raises:
        ValueError:
            If saturated moist air is not physically defined at T and p,
            i.e. if water saturation vapor pressure is greater than or equal
            to the total pressure.

    Ref: ASHRAE Fundamentals / PsychroLib.
    """
    _validate_temperature_k(T)
    _validate_pressure_pa(p)

    p_ws = saturation_vapor_pressure(T)

    if p_ws >= p:
        raise ValueError(
            "Saturation humidity ratio is not defined for this T and p because "
            "water saturation vapor pressure is greater than or equal to total "
            "pressure. "
            f"T={T:.6g} K, p={p:.6g} Pa, p_ws={p_ws:.6g} Pa. "
            "This usually means the dry-bulb temperature is above the water "
            "boiling temperature at the given pressure. For such gas states, "
            "define moisture content by W [kg/kg dry air], "
            "W_g_per_kg_da [g/kg dry air], or vapor mole fraction instead."
        )

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