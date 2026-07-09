"""
Dry-air property provider.

This module represents dry air only.

Use this module for:
- dry air as a standard gas medium,
- tube-side or outside-side dry air,
- sensible gas-gas heat exchangers without humidity,
- transport properties required by heat-transfer and pressure-drop solvers.

Do not use this module for:
- relative humidity,
- humidity ratio,
- dew point,
- saturation,
- condensation,
- wet-surface heat transfer.

For humid air / psychrometric calculations, use the moist-air / PsychroLib
property path. For hot gases with water vapor as a gas-phase component, use
GasMixtureSpec explicitly.

KalKalori core units:
- temperature: K
- pressure: Pa
- density: kg/m3
- dynamic viscosity: Pa*s
- thermal conductivity: W/(m*K)
- specific heat: J/(kg*K)

The preferred backend is CoolProp "Air" when CoolProp is installed.
If CoolProp is unavailable or fails, a simple engineering fallback is used
unless disabled.

The fallback is intended for robust tests and preliminary engineering checks,
not for high-accuracy property work.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.properties.common import FluidTransportProperties
from core.properties.coolprop_backend import CoolPropFluidProvider


# Dry-air constants.
DRY_AIR_MOLAR_MASS = 28.96546e-3  # kg/mol
DRY_AIR_R = 287.05287  # J/(kg*K)

# Sutherland dynamic-viscosity constants for dry air.
SUTHERLAND_T0 = 273.15  # K
SUTHERLAND_MU0 = 1.716e-5  # Pa*s
SUTHERLAND_S = 111.0  # K

# Simple fallback Prandtl number for engineering-level dry-air transport.
# Used only to derive k from mu and cp in fallback mode.
DRY_AIR_FALLBACK_PR = 0.71

# Approximate cp(T) table for dry air at low pressure [K, J/(kg*K)].
# Values are intentionally smooth engineering values for fallback mode.
DRY_AIR_CP_TABLE: tuple[tuple[float, float], ...] = (
    (200.0, 1003.0),
    (250.0, 1005.0),
    (300.0, 1007.0),
    (400.0, 1014.0),
    (500.0, 1030.0),
    (600.0, 1051.0),
    (700.0, 1075.0),
    (800.0, 1099.0),
    (900.0, 1121.0),
    (1000.0, 1142.0),
    (1100.0, 1161.0),
    (1200.0, 1179.0),
)


@dataclass(frozen=True)
class DryAirPropertyProvider:
    """Transport-property provider for dry air.

    Attributes:
        prefer_coolprop:
            If True, try CoolProp "Air" first.
        allow_fallback:
            If True, use the internal ideal-gas / Sutherland fallback when
            CoolProp is not installed or fails.
        imposed_phase:
            Optional CoolProp phase hint. The default "gas" is appropriate for
            dry-air gas-side engineering calculations.

    Notes:
        This provider is dry-air only. It does not accept RH, W, dew point, or
        saturation inputs.
    """

    prefer_coolprop: bool = True
    allow_fallback: bool = True
    imposed_phase: str | None = "gas"

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return dry-air transport properties at T [K] and p [Pa]."""
        return dry_air_props(
            T=T,
            p=p,
            prefer_coolprop=self.prefer_coolprop,
            allow_fallback=self.allow_fallback,
            imposed_phase=self.imposed_phase,
        )


def dry_air_props(
    T: float,
    p: float,
    *,
    prefer_coolprop: bool = True,
    allow_fallback: bool = True,
    imposed_phase: str | None = "gas",
) -> FluidTransportProperties:
    """Return dry-air transport properties.

    Args:
        T:
            Temperature [K].
        p:
            Absolute pressure [Pa].
        prefer_coolprop:
            If True, try CoolProp "Air" first.
        allow_fallback:
            If True, use the internal engineering fallback when CoolProp is
            unavailable or fails.
        imposed_phase:
            Optional CoolProp phase hint. The default is "gas".

    Returns:
        FluidTransportProperties in KalKalori SI units.

    Raises:
        ValueError:
            If input values are invalid.
        RuntimeError:
            If CoolProp is requested, fails, and fallback is disabled.
    """
    _validate_temperature(T)
    _validate_pressure(p)

    if prefer_coolprop:
        try:
            return _dry_air_props_coolprop(
                T=T,
                p=p,
                imposed_phase=imposed_phase,
            )
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "CoolProp dry-air property evaluation failed and fallback "
                    "is disabled. Install/configure CoolProp correctly or call "
                    "dry_air_props(..., allow_fallback=True). "
                    f"Original error: {exc}"
                ) from exc

    if allow_fallback:
        return dry_air_fallback_props(T=T, p=p)

    raise RuntimeError(
        "Dry-air property evaluation failed: prefer_coolprop=False and "
        "allow_fallback=False leaves no available dry-air backend."
    )


def dry_air_fallback_props(T: float, p: float) -> FluidTransportProperties:
    """Return fallback dry-air properties.

    Fallback model:
    - density: ideal gas, rho = p / (R_air * T)
    - viscosity: Sutherland relation
    - cp: table interpolation
    - k: derived from mu * cp / Pr

    This is intended for tests and preliminary calculations only.
    """
    _validate_temperature(T)
    _validate_pressure(p)

    rho = p / (DRY_AIR_R * T)
    mu = dry_air_sutherland_mu(T)
    cp = dry_air_cp_fallback(T)
    k = mu * cp / DRY_AIR_FALLBACK_PR

    return FluidTransportProperties(
        rho=rho,
        mu=mu,
        k=k,
        cp=cp,
    )


def dry_air_sutherland_mu(T: float) -> float:
    """Return dry-air dynamic viscosity [Pa*s] using Sutherland's relation."""
    _validate_temperature(T)

    return (
        SUTHERLAND_MU0
        * (T / SUTHERLAND_T0) ** 1.5
        * (SUTHERLAND_T0 + SUTHERLAND_S)
        / (T + SUTHERLAND_S)
    )


def dry_air_cp_fallback(T: float) -> float:
    """Return approximate dry-air cp [J/(kg*K)] by linear interpolation."""
    _validate_temperature(T)

    table = DRY_AIR_CP_TABLE

    if T <= table[0][0]:
        return table[0][1]

    if T >= table[-1][0]:
        return table[-1][1]

    for (T_low, cp_low), (T_high, cp_high) in zip(table[:-1], table[1:]):
        if T_low <= T <= T_high:
            fraction = (T - T_low) / (T_high - T_low)
            return cp_low + fraction * (cp_high - cp_low)

    # Defensive fallback; loop above should always return.
    return table[-1][1]


def _dry_air_props_coolprop(
    T: float,
    p: float,
    imposed_phase: str | None = "gas",
) -> FluidTransportProperties:
    """Return dry-air properties from CoolProp pseudo-pure fluid 'Air'."""
    try:
        provider = CoolPropFluidProvider("Air", imposed_phase=imposed_phase)
    except TypeError:
        # Compatibility with older CoolPropFluidProvider versions without
        # imposed_phase support.
        provider = CoolPropFluidProvider("Air")

    return provider.at(T=T, p=p)


def _validate_temperature(T: float) -> None:
    if not isinstance(T, (int, float)):
        raise TypeError(f"Temperature must be numeric, got {type(T).__name__}.")
    if T <= 0.0:
        raise ValueError(f"Temperature must be above 0 K, got {T!r}.")


def _validate_pressure(p: float) -> None:
    if not isinstance(p, (int, float)):
        raise TypeError(f"Pressure must be numeric, got {type(p).__name__}.")
    if p <= 0.0:
        raise ValueError(f"Pressure must be positive, got {p!r}.")
