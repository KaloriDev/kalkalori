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
from enum import Enum
import math
from typing import Optional

from iapws import IAPWS97

from core.common.warnings import ModelWarning
from core.properties.common import FluidTransportProperties


SOURCE = "properties.water"

# IAPWS-IF97 saturation-curve (region 4) validity bounds.
# Ref: IAPWS-IF97, Region 4 (saturation line), triple point and critical
# point of water.
WATER_TRIPLE_POINT_TEMPERATURE_K = 273.16
WATER_TRIPLE_POINT_PRESSURE_PA = 611.657
WATER_CRITICAL_TEMPERATURE_K = 647.096
WATER_CRITICAL_PRESSURE_PA = 22.064e6


def _warning(code: str, message: str, severity: str = "warning") -> ModelWarning:
    return ModelWarning(
        code=code,
        message=message,
        severity=severity,
        source=SOURCE,
    )


class WaterSteamPhase(str, Enum):
    """Unambiguous pure-water phase classification, including supercritical."""

    SUBCOOLED_LIQUID = "subcooled_liquid"
    SATURATED_LIQUID = "saturated_liquid"
    TWO_PHASE = "two_phase"
    SATURATED_VAPOR = "saturated_vapor"
    SUPERHEATED_VAPOR = "superheated_vapor"
    SUPERCRITICAL_FLUID = "supercritical_fluid"


@dataclass(frozen=True)
class WaterSteamProperties:
    """Water/steam properties at one thermodynamic state.

    Attributes:
        transport:
            Single-phase transport properties in KalKalori SI units, or
            ``None`` for a two-phase equilibrium state. KalKalori does not
            invent mixture ``cp``/``mu``/``k`` values in the saturation dome.
        h:
            Specific enthalpy [J/kg].
        phase:
            Normalized phase description returned by IAPWS.
        warnings:
            Applicability / interpretation warnings.
    """

    T: float
    p: float
    transport: FluidTransportProperties | None
    h: float
    phase: WaterSteamPhase
    quality: float | None
    warnings: list[ModelWarning]

    @property
    def x(self) -> float | None:
        """Compatibility/engineering alias for vapor quality."""
        return self.quality

    @property
    def rho(self) -> float | None:
        return None if self.transport is None else self.transport.rho

    @property
    def mu(self) -> float | None:
        return None if self.transport is None else self.transport.mu

    @property
    def k(self) -> float | None:
        return None if self.transport is None else self.transport.k

    @property
    def cp(self) -> float | None:
        return None if self.transport is None else self.transport.cp

    @property
    def Pr(self) -> float | None:
        if self.transport is None:
            return None
        return self.transport.cp * self.transport.mu / self.transport.k


@dataclass(frozen=True)
class WaterSteamSaturationProperties:
    """Immutable saturation snapshot at one pressure.

    The snapshot deliberately carries both endpoint states so a zone solver
    can evaluate ``Tsat``, ``hf``, ``hg`` and transport properties once per
    nominal pressure rather than repeating identical IAPWS calls in residuals.
    """

    p: float
    Tsat: float
    hf: float
    hg: float
    hfg: float
    saturated_liquid: WaterSteamProperties
    saturated_vapor: WaterSteamProperties


def water_steam_props_iapws97(
    T: Optional[float] = None,
    p: Optional[float] = None,
    x: Optional[float] = None,
    h: Optional[float] = None,
) -> WaterSteamProperties:
    """Return water/steam properties from IAPWS-IF97.

    Supported input modes:
        T + p:
            Single-phase compressed-liquid, superheated, or supercritical state.
            T [K], p [Pa].

        p + x:
            Saturated state at pressure.
            p [Pa], x [-].

        p + h:
            State classified from pressure and specific enthalpy.
            p [Pa], h [J/kg].

        T + x:
            Saturated state at temperature. Retained as a lower-level
            compatibility mode; public exchanger inlets use p+x.

    Args:
        T:
            Temperature [K], optional.
        p:
            Pressure [Pa], optional.
        x:
            Vapor quality [-], optional.
            Use x = 0.0 for saturated liquid.
            Use x = 1.0 for saturated steam.

    Returns:
        WaterSteamProperties.

    Notes:
        This function is a thin SI-unit adapter around `iapws.IAPWS97`.

        Exactly one supported pair must be provided. Public inlet state
        specifications are T+p, p+x and p+h.

        The `iapws` package expects pressure in MPa and returns enthalpy
        and cp in kJ/kg-based units. This adapter converts them to KalKalori
        core SI units.

        For x values between 0 and 1, IAPWS-IF97 describes a two-phase state.
        Transport properties in two-phase conditions require careful
        interpretation and should not be treated as ordinary single-phase
        fluid properties without additional modelling assumptions.

    Ref: IAPWS-IF97.
    """
    has_T = T is not None
    has_p = p is not None
    has_x = x is not None
    has_h = h is not None

    mode = (has_T, has_p, has_x, has_h)
    supported = {
        (True, True, False, False),   # T+p
        (False, True, True, False),   # p+x
        (False, True, False, True),   # p+h
        (True, False, True, False),   # T+x compatibility
    }
    if mode not in supported:
        raise ValueError(
            "Provide exactly one water/steam state specification: T+p, "
            "p+x, or p+h (T+x is retained as a lower-level compatibility mode)."
        )

    if has_T:
        _validate_temperature(T)
    if has_p:
        _validate_pressure(p)
    if has_x:
        _validate_vapor_quality(x)
    if has_h:
        _validate_enthalpy(h)

    if has_p and has_x and p >= WATER_CRITICAL_PRESSURE_PA:
        raise ValueError(
            "Vapor quality is undefined at or above the water critical pressure."
        )

    try:
        if has_T and has_p:
            return _state_from_temperature_pressure(T=T, p=p)

        elif has_p and has_x:
            return _state_from_pressure_quality(p=p, x=x)

        elif has_p and has_h:
            return _state_from_pressure_enthalpy(p=p, h=h)

        elif has_T and has_x:
            state = IAPWS97(T=T, x=x)
            return _state_from_pressure_quality(p=float(state.P) * 1.0e6, x=x)

        else:
            raise ValueError(
                "Unsupported IAPWS-IF97 input mode. "
                "Supported modes are T+p, p+x, p+h, and compatibility T+x."
            )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"IAPWS-IF97 failed for T={T}, p={p}, x={x}, h={h}."
        ) from exc


@dataclass(frozen=True)
class IAPWS97WaterSteamProvider:
    """Property provider for water/steam based on IAPWS-IF97.

    This provider follows the generic `PropertyProvider.at(T, p)` shape and
    therefore supports only the ordinary T+p input mode.

    Saturated states using p+x or T+x should be obtained directly through
    `water_steam_props_iapws97(...)`.
    """

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return transport properties at T [K] and p [Pa]."""
        state = water_steam_props_iapws97(T=T, p=p)
        if state.transport is None:
            raise ValueError("Two-phase water/steam has no single-phase transport state.")
        return state.transport

    def full_at(self, T: float, p: float) -> WaterSteamProperties:
        """Return transport and specific enthalpy at T [K] and p [Pa]."""
        return water_steam_props_iapws97(T=T, p=p)

    def temperature_from_h_p(self, h: float, p: float) -> float:
        """Invert specific enthalpy and pressure to temperature [K]."""
        return water_steam_props_iapws97(p=p, h=h).T

    def state(
        self,
        *,
        T: float | None = None,
        p: float,
        h: float | None = None,
        x: float | None = None,
    ) -> WaterSteamProperties:
        """Resolve an unambiguous T+p, p+h or p+x state."""
        return water_steam_props_iapws97(T=T, p=p, h=h, x=x)


def _props_from_iapws_state(
    state: IAPWS97,
    *,
    phase: WaterSteamPhase,
    quality: float | None,
) -> WaterSteamProperties:
    """Convert an IAPWS97 state object to KalKalori SI property container."""
    warnings: list[ModelWarning] = []

    if phase is WaterSteamPhase.TWO_PHASE:
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

    transport: FluidTransportProperties | None = None
    if phase is not WaterSteamPhase.TWO_PHASE:
        try:
            transport = FluidTransportProperties(
                rho=float(state.rho),
                mu=float(state.mu),
                k=float(state.k),
                cp=float(state.cp) * 1000.0,
            )
        except Exception as exc:
            raise ValueError(
                "Failed to extract single-phase transport properties from IAPWS-IF97 state."
            ) from exc

    h = state.h * 1000.0

    return WaterSteamProperties(
        T=float(state.T),
        p=float(state.P) * 1.0e6,
        transport=transport,
        h=float(h),
        phase=phase,
        quality=quality,
        warnings=warnings,
    )


def water_saturation_snapshot(p: float) -> WaterSteamSaturationProperties:
    """Return one cached-ready IAPWS saturation snapshot at pressure p [Pa]."""
    _validate_saturation_pressure(p)
    if p >= WATER_CRITICAL_PRESSURE_PA:
        raise ValueError("Saturated liquid/vapor endpoints are undefined at the critical point.")
    try:
        liquid_raw = IAPWS97(P=_pa_to_mpa(p), x=0.0)
        vapor_raw = IAPWS97(P=_pa_to_mpa(p), x=1.0)
    except Exception as exc:
        raise ValueError(f"IAPWS-IF97 saturation snapshot failed for p={p} Pa.") from exc
    liquid = _props_from_iapws_state(
        liquid_raw, phase=WaterSteamPhase.SATURATED_LIQUID, quality=0.0
    )
    vapor = _props_from_iapws_state(
        vapor_raw, phase=WaterSteamPhase.SATURATED_VAPOR, quality=1.0
    )
    return WaterSteamSaturationProperties(
        p=float(p),
        Tsat=liquid.T,
        hf=liquid.h,
        hg=vapor.h,
        hfg=vapor.h - liquid.h,
        saturated_liquid=liquid,
        saturated_vapor=vapor,
    )


def _state_from_temperature_pressure(*, T: float, p: float) -> WaterSteamProperties:
    saturation: WaterSteamSaturationProperties | None = None
    if WATER_TRIPLE_POINT_PRESSURE_PA <= p < WATER_CRITICAL_PRESSURE_PA:
        saturation = water_saturation_snapshot(p)
        if math.isclose(T, saturation.Tsat, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(
                "T+p lies on the water saturation line and does not identify "
                "saturated liquid, saturated vapor, or mixture quality; use p+x or p+h."
            )
    raw = IAPWS97(T=T, P=_pa_to_mpa(p))
    if p >= WATER_CRITICAL_PRESSURE_PA and T >= WATER_CRITICAL_TEMPERATURE_K:
        phase = WaterSteamPhase.SUPERCRITICAL_FLUID
    elif saturation is not None:
        phase = (
            WaterSteamPhase.SUPERHEATED_VAPOR
            if T > saturation.Tsat
            else WaterSteamPhase.SUBCOOLED_LIQUID
        )
    else:
        raw_phase = _normalize_phase(raw.phase)
        phase = (
            WaterSteamPhase.SUPERHEATED_VAPOR
            if "vapour" in raw_phase or "vapor" in raw_phase
            else WaterSteamPhase.SUBCOOLED_LIQUID
        )
    return _props_from_iapws_state(raw, phase=phase, quality=None)


def _state_from_pressure_quality(*, p: float, x: float) -> WaterSteamProperties:
    saturation = water_saturation_snapshot(p)
    if x == 0.0:
        return saturation.saturated_liquid
    if x == 1.0:
        return saturation.saturated_vapor
    h = saturation.hf + x * saturation.hfg
    raw = IAPWS97(P=_pa_to_mpa(p), h=h / 1000.0)
    return _props_from_iapws_state(
        raw, phase=WaterSteamPhase.TWO_PHASE, quality=float(x)
    )


def _state_from_pressure_enthalpy(*, p: float, h: float) -> WaterSteamProperties:
    if p >= WATER_CRITICAL_PRESSURE_PA:
        raw = IAPWS97(P=_pa_to_mpa(p), h=h / 1000.0)
        phase = (
            WaterSteamPhase.SUPERCRITICAL_FLUID
            if float(raw.T) >= WATER_CRITICAL_TEMPERATURE_K
            else WaterSteamPhase.SUBCOOLED_LIQUID
        )
        return _props_from_iapws_state(raw, phase=phase, quality=None)
    saturation = water_saturation_snapshot(p)
    tolerance = max(1.0e-3, 1.0e-9 * max(abs(saturation.hf), abs(saturation.hg)))
    if abs(h - saturation.hf) <= tolerance:
        return saturation.saturated_liquid
    if abs(h - saturation.hg) <= tolerance:
        return saturation.saturated_vapor
    if saturation.hf < h < saturation.hg:
        quality = (h - saturation.hf) / saturation.hfg
        raw = IAPWS97(P=_pa_to_mpa(p), h=h / 1000.0)
        return _props_from_iapws_state(
            raw, phase=WaterSteamPhase.TWO_PHASE, quality=quality
        )
    raw = IAPWS97(P=_pa_to_mpa(p), h=h / 1000.0)
    phase = (
        WaterSteamPhase.SUPERHEATED_VAPOR
        if h > saturation.hg
        else WaterSteamPhase.SUBCOOLED_LIQUID
    )
    return _props_from_iapws_state(raw, phase=phase, quality=None)


def _normalize_phase(phase: object) -> str:
    """Normalize phase description returned by the `iapws` package."""
    return str(phase).strip().lower().replace(" ", "_")


def _pa_to_mpa(p: float) -> float:
    """Convert pressure from Pa to MPa."""
    return p / 1.0e6


def _validate_temperature(T: Optional[float]) -> None:
    if T is None:
        raise ValueError("Temperature must be provided [K].")
    if not math.isfinite(T):
        raise ValueError("Temperature must be finite [K].")
    if T <= 0.0:
        raise ValueError("Temperature must be above absolute zero [K].")


def _validate_pressure(p: Optional[float]) -> None:
    if p is None:
        raise ValueError("Pressure must be provided [Pa].")
    if not math.isfinite(p):
        raise ValueError("Pressure must be finite [Pa].")
    if p <= 0.0:
        raise ValueError("Pressure must be positive [Pa].")


def _validate_vapor_quality(x: Optional[float]) -> None:
    if x is None:
        raise ValueError("Vapor quality must be provided [-].")
    if not math.isfinite(x):
        raise ValueError("Vapor quality must be finite [-].")
    if x < 0.0 or x > 1.0:
        raise ValueError("Vapor quality x must be in range 0.0 ... 1.0 [-].")


def _validate_enthalpy(h: Optional[float]) -> None:
    if h is None or not math.isfinite(h):
        raise ValueError("Specific enthalpy must be finite [J/kg].")


# ---------------------------------------------------------------------------
# Saturation-curve helpers (v0.6.0)
#
# These are thin, single-purpose wrappers around IAPWS-IF97 region 4
# (saturation line), added so callers needing dew-point / condensation
# equilibrium (``core.phase_change``) do not each re-implement calls into
# the ``iapws`` package. All saturation-curve access for water anywhere in
# KalKalori core should go through these functions (or
# ``water_steam_props_iapws97`` above), not a second IAPWS wrapper.
#
# Validity: the saturation line (region 4) is only defined between the
# triple point and the critical point. Below the triple point, liquid-vapor
# equilibrium does not exist (that is the solid/vapor sublimation curve,
# i.e. frost/ice -- out of scope for v0.6.0, see
# ``core.phase_change.water_equilibrium.is_frost_regime``).
# ---------------------------------------------------------------------------
def water_saturation_pressure(T: float) -> float:
    """Return water saturation pressure at temperature T [K] -> Pa.

    Valid below the critical point:
    ``WATER_TRIPLE_POINT_TEMPERATURE_K <= T < WATER_CRITICAL_TEMPERATURE_K``.

    Ref: IAPWS-IF97, Region 4 (saturation line).
    """
    _validate_saturation_temperature(T)
    try:
        state = IAPWS97(T=T, x=0.0)
    except Exception as exc:
        raise ValueError(f"IAPWS-IF97 saturation pressure failed for T={T} K.") from exc
    return float(state.P) * 1.0e6


def water_saturation_temperature(p: float) -> float:
    """Return water saturation temperature at pressure p [Pa] -> K.

    Valid below the critical point:
    ``WATER_TRIPLE_POINT_PRESSURE_PA <= p < WATER_CRITICAL_PRESSURE_PA``.

    Ref: IAPWS-IF97, Region 4 (saturation line).
    """
    _validate_saturation_pressure(p)
    if p < WATER_CRITICAL_PRESSURE_PA:
        return water_saturation_snapshot(p).Tsat
    try:
        state = IAPWS97(P=_pa_to_mpa(p), x=0.0)
        return float(state.T)
    except Exception as exc:
        raise ValueError(f"IAPWS-IF97 saturation temperature failed for p={p} Pa.") from exc


def water_saturation_liquid_enthalpy(
    *, T: Optional[float] = None, p: Optional[float] = None
) -> float:
    """Return saturated-liquid specific enthalpy h_f [J/kg].

    Exactly one of ``T`` [K] or ``p`` [Pa] must be given.

    Ref: IAPWS-IF97, Region 4 (saturation line).
    """
    return _saturation_enthalpy(T=T, p=p, x=0.0)


def water_saturation_vapor_enthalpy(
    *, T: Optional[float] = None, p: Optional[float] = None
) -> float:
    """Return saturated-vapor specific enthalpy h_g [J/kg].

    Exactly one of ``T`` [K] or ``p`` [Pa] must be given.

    Used as the engineering approximation for the specific enthalpy of
    water vapor at low partial pressure within a gas mixture (the vapor's
    enthalpy is, to an excellent approximation for an ideal-gas component,
    a function of temperature alone -- the same convention used by
    psychrometric moist-air-enthalpy formulas, e.g. ASHRAE).

    Ref: IAPWS-IF97, Region 4 (saturation line).
    """
    return _saturation_enthalpy(T=T, p=p, x=1.0)


def water_latent_heat_of_vaporization(
    *, T: Optional[float] = None, p: Optional[float] = None
) -> float:
    """Return latent heat of vaporization h_fg = h_g - h_f [J/kg].

    Exactly one of ``T`` [K] or ``p`` [Pa] must be given.

    Ref: IAPWS-IF97, Region 4 (saturation line).
    """
    h_f = water_saturation_liquid_enthalpy(T=T, p=p)
    h_g = water_saturation_vapor_enthalpy(T=T, p=p)
    return h_g - h_f


def _saturation_enthalpy(
    *, T: Optional[float], p: Optional[float], x: float
) -> float:
    if (T is None) == (p is None):
        raise ValueError("Exactly one of T [K] or p [Pa] must be provided.")
    if T is not None:
        _validate_saturation_temperature(T)
        return water_steam_props_iapws97(T=T, x=x).h
    _validate_saturation_pressure(p)
    if p < WATER_CRITICAL_PRESSURE_PA:
        snapshot = water_saturation_snapshot(p)
        return snapshot.hf if x == 0.0 else snapshot.hg
    return water_steam_props_iapws97(p=p, x=x).h


def _validate_saturation_temperature(T: Optional[float]) -> None:
    _validate_temperature(T)
    if T < WATER_TRIPLE_POINT_TEMPERATURE_K or T >= WATER_CRITICAL_TEMPERATURE_K:
        raise ValueError(
            f"Saturation temperature T={T} K is outside the IAPWS-IF97 "
            f"region-4 validity range "
            f"[{WATER_TRIPLE_POINT_TEMPERATURE_K}, {WATER_CRITICAL_TEMPERATURE_K}] K."
        )


def _validate_saturation_pressure(p: Optional[float]) -> None:
    _validate_pressure(p)
    if p < WATER_TRIPLE_POINT_PRESSURE_PA or p >= WATER_CRITICAL_PRESSURE_PA:
        raise ValueError(
            f"Saturation pressure p={p} Pa is outside the IAPWS-IF97 "
            f"region-4 validity range "
            f"[{WATER_TRIPLE_POINT_PRESSURE_PA}, {WATER_CRITICAL_PRESSURE_PA}] Pa."
        )
