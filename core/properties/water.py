"""
Water and steam property provider based on IAPWS-IF97.

KalKalori core uses SI floats:
- temperature: K
- pressure: Pa
- density: kg/m3
- dynamic viscosity: Pa*s
- thermal conductivity: W/(m*K)
- specific heat: J/(kg*K)
- specific enthalpy: J/kg

The external `iapws` package uses:
- temperature: K
- pressure: MPa
- enthalpy: kJ/kg
- cp: kJ/(kg*K)

Ref: IAPWS-IF97, Industrial Formulation 1997 for the Thermodynamic
Properties of Water and Steam.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from iapws import IAPWS97

from core.common.warnings import ModelWarning
from core.properties.common import FluidTransportProperties


SOURCE = "properties.water"


def _warning(code: str, message: str, severity: str = "warning") -> ModelWarning:
    return ModelWarning(
        code=code,
        message=message,
        severity=severity,
        source=SOURCE,
    )


@dataclass(frozen=True)
class WaterSteamProperties:
    """Water/steam properties at one thermodynamic state."""

    transport: FluidTransportProperties
    h: float
    phase: str
    warnings: list[ModelWarning]


def water_steam_props_iapws97(
    T: float,
    p: float,
) -> WaterSteamProperties:
    """Return water/steam properties from IAPWS-IF97.

    Args:
        T: Temperature [K].
        p: Pressure [Pa].

    Returns:
        WaterSteamProperties.

    Notes:
        This function is a thin SI-unit adapter around `iapws.IAPWS97`.

        The `iapws` package expects pressure in MPa and returns enthalpy
        and cp in kJ/kg-based units. This adapter converts them to KalKalori
        core SI units.

    Ref: IAPWS-IF97.
    """
    if not math.isfinite(T):
        raise ValueError("Temperature must be finite [K].")
    if T <= 0.0:
        raise ValueError("Temperature must be above absolute zero [K].")
    if not math.isfinite(p):
        raise ValueError("Pressure must be finite [Pa].")
    if p <= 0.0:
        raise ValueError("Pressure must be positive [Pa].")

    warnings: list[ModelWarning] = []

    p_MPa = p / 1.0e6

    try:
        state = IAPWS97(T=T, P=p_MPa)
    except Exception as exc:
        raise ValueError(
            f"IAPWS-IF97 failed for T={T} K, p={p} Pa."
        ) from exc

    # iapws units:
    # rho: kg/m3
    # mu: Pa*s
    # k: W/(m*K)
    # cp: kJ/(kg*K)
    # h: kJ/kg
    transport = FluidTransportProperties(
        rho=state.rho,
        mu=state.mu,
        k=state.k,
        cp=state.cp * 1000.0,
    )

    h = state.h * 1000.0

    phase_raw = state.phase
    phase = str(phase_raw).lower().replace(" ", "_")

    if phase in {"two_phase", "two-phase"}:
        warnings.append(
            _warning(
                code="WATER_STEAM_TWO_PHASE_STATE",
                message=(
                    "IAPWS-IF97 returned a two-phase water/steam state. "
                    "Transport properties in two-phase conditions require careful interpretation."
                ),
                severity="warning",
            )
        )

    return WaterSteamProperties(
        transport=transport,
        h=h,
        phase=phase,
        warnings=warnings,
    )


@dataclass(frozen=True)
class IAPWS97WaterSteamProvider:
    """Property provider for water/steam based on IAPWS-IF97."""

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return transport properties at T [K] and p [Pa]."""
        return water_steam_props_iapws97(T=T, p=p).transport