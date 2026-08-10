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


@dataclass(frozen=True)
class WaterSteamProperties:
    """Water/steam properties at one thermodynamic state.

    Attributes:
        transport:
            Transport properties in KalKalori SI units.
        h:
            Specific enthalpy [J/kg].
        phase:
            Normalized phase description returned by IAPWS.
        warnings:
            Applicability / interpretation warnings.
    """

    transport: FluidTransportProperties
    h: float
    phase: str
    warnings: list[ModelWarning]


def water_steam_props_iapws97(
    T: Optional[float] = None,
    p: Optional[float] = None,
    x: Optional[float] = None,
) -> WaterSteamProperties:
    """Return water/steam properties from IAPWS-IF97.

    Supported input modes:
        T + p:
            Single-phase, compressed-liquid, or superheated state.
            T [K], p [Pa].

        p + x:
            Saturated state at pressure.
            p [Pa], x [-].

        T + x:
            Saturated state at temperature.
            T [K], x [-].

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

        Exactly two of `T`, `p`, and `x` must be provided.

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

    if sum((has_T, has_p, has_x)) != 2:
        raise ValueError(
            "Exactly two of T, p and x must be provided. "
            "Supported modes are T+p, p+x, and T+x."
        )

    if has_T:
        _validate_temperature(T)
    if has_p:
        _validate_pressure(p)
    if has_x:
        _validate_vapor_quality(x)

    try:
        if has_T and has_p:
            state = IAPWS97(
                T=T,
                P=_pa_to_mpa(p),
            )

        elif has_p and has_x:
            state = IAPWS97(
                P=_pa_to_mpa(p),
                x=x,
            )

        elif has_T and has_x:
            state = IAPWS97(
                T=T,
                x=x,
            )

        else:
            raise ValueError(
                "Unsupported IAPWS-IF97 input mode. "
                "Supported modes are T+p, p+x, and T+x."
            )

    except Exception as exc:
        raise ValueError(
            f"IAPWS-IF97 failed for T={T}, p={p}, x={x}."
        ) from exc

    return _props_from_iapws_state(state)


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
        return water_steam_props_iapws97(T=T, p=p).transport

    def full_at(self, T: float, p: float) -> WaterSteamProperties:
        """Return transport and specific enthalpy at T [K] and p [Pa]."""
        return water_steam_props_iapws97(T=T, p=p)

    def temperature_from_h_p(self, h: float, p: float) -> float:
        """Invert specific enthalpy and pressure to temperature [K]."""
        if not math.isfinite(h):
            raise ValueError("Specific enthalpy must be finite [J/kg].")
        _validate_pressure(p)
        try:
            state = IAPWS97(P=_pa_to_mpa(p), h=h / 1000.0)
        except Exception as exc:
            raise ValueError(
                f"IAPWS-IF97 failed for h={h} J/kg, p={p} Pa."
            ) from exc
        return float(state.T)


def _props_from_iapws_state(state: IAPWS97) -> WaterSteamProperties:
    """Convert an IAPWS97 state object to KalKalori SI property container."""
    warnings: list[ModelWarning] = []

    phase = _normalize_phase(state.phase)

    if phase in {"two_phase", "two-phase", "two_phases"}:
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

    try:
        transport = FluidTransportProperties(
            rho=state.rho,
            mu=state.mu,
            k=state.k,
            cp=state.cp * 1000.0,
        )
    except Exception as exc:
        raise ValueError(
            "Failed to extract complete transport properties from IAPWS-IF97 state. "
            "This may happen for states where transport properties are not available "
            "or not meaningful."
        ) from exc

    h = state.h * 1000.0

    return WaterSteamProperties(
        transport=transport,
        h=h,
        phase=phase,
        warnings=warnings,
    )


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

    Valid for ``WATER_TRIPLE_POINT_TEMPERATURE_K <= T <= WATER_CRITICAL_TEMPERATURE_K``.

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

    Valid for ``WATER_TRIPLE_POINT_PRESSURE_PA <= p <= WATER_CRITICAL_PRESSURE_PA``.

    Ref: IAPWS-IF97, Region 4 (saturation line).
    """
    _validate_saturation_pressure(p)
    try:
        state = IAPWS97(P=_pa_to_mpa(p), x=0.0)
    except Exception as exc:
        raise ValueError(f"IAPWS-IF97 saturation temperature failed for p={p} Pa.") from exc
    return float(state.T)


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
    return water_steam_props_iapws97(p=p, x=x).h


def _validate_saturation_temperature(T: Optional[float]) -> None:
    _validate_temperature(T)
    if T < WATER_TRIPLE_POINT_TEMPERATURE_K or T > WATER_CRITICAL_TEMPERATURE_K:
        raise ValueError(
            f"Saturation temperature T={T} K is outside the IAPWS-IF97 "
            f"region-4 validity range "
            f"[{WATER_TRIPLE_POINT_TEMPERATURE_K}, {WATER_CRITICAL_TEMPERATURE_K}] K."
        )


def _validate_saturation_pressure(p: Optional[float]) -> None:
    _validate_pressure(p)
    if p < WATER_TRIPLE_POINT_PRESSURE_PA or p > WATER_CRITICAL_PRESSURE_PA:
        raise ValueError(
            f"Saturation pressure p={p} Pa is outside the IAPWS-IF97 "
            f"region-4 validity range "
            f"[{WATER_TRIPLE_POINT_PRESSURE_PA}, {WATER_CRITICAL_PRESSURE_PA}] Pa."
        )


# ---------------------------------------------------------------------------
# Pure water/steam thermodynamic states (v0.6.2)
#
# Unlike ``water_steam_props_iapws97``/``WaterSteamProperties`` above (a
# thin T+p / p+x / T+x property lookup, unchanged since v0.6.0), the
# ``WaterSteamState`` model below additionally classifies which side of the
# saturation dome a state sits on -- superheated vapor, saturated vapor,
# two-phase mixture, saturated liquid, or subcooled liquid -- and reports
# vapor quality plus saturation diagnostics (T_sat, h_f, h_g, h_fg) at the
# state's own pressure. This is the state representation the v0.6.2 pure
# water/steam condensation model (``core.phase_change``) is built on; it
# does not use the wet-gas humidity-ratio (W) basis at all.
# ---------------------------------------------------------------------------

# T+p is rejected as ambiguous within this band around T_sat(p): at that
# point T+p alone cannot distinguish saturated liquid, saturated vapor, and
# a two-phase mixture (all three share the same T at a given p).
_SATURATION_AMBIGUITY_TOLERANCE_K = 1e-2

# p+h boundary snap tolerance, expressed as a fraction of h_fg(p). Within
# this band of h_f/h_g, a p+h state is reported as exactly saturated
# (x=0 or x=1) instead of an infinitesimally-two-phase state; this also
# makes a p+x -> h -> p+h round trip land back on the same classification.
_SATURATION_ENTHALPY_TOLERANCE_RELATIVE = 1e-6


class WaterPhaseRegion(str, Enum):
    """Phase region of a classified ``WaterSteamState``."""

    SUBCOOLED_LIQUID = "subcooled_liquid"
    SATURATED_LIQUID = "saturated_liquid"
    TWO_PHASE = "two_phase"
    SATURATED_VAPOR = "saturated_vapor"
    SUPERHEATED_VAPOR = "superheated_vapor"
    # T+p at or above the critical pressure: no saturation line exists, so
    # the state is neither liquid nor vapor in the usual sense.
    SUPERCRITICAL = "supercritical"


@dataclass(frozen=True)
class WaterSteamState:
    """Fully classified pure-water/steam thermodynamic state.

    Built by ``water_steam_state(...)`` from exactly one of three mutually
    exclusive input modes: T+p, p+x, or p+h.

    Attributes:
        T: Temperature [K].
        p: Pressure [Pa].
        h: Specific enthalpy [J/kg].
        phase: Phase region classification.
        quality: Vapor quality [-], ``x = m_vapor / (m_vapor + m_liquid)``.
            ``None`` outside the saturation dome (superheated vapor,
            subcooled liquid, supercritical); otherwise in ``[0, 1]``.
        rho: Density [kg/m3], when physically defined.
        cp: Specific heat capacity [J/(kg*K)], when physically defined.
            ``None`` in the two-phase region (``0 < quality < 1``), where
            an ordinary single-phase isobaric cp is not defined for a
            boiling/condensing mixture.
        mu: Dynamic viscosity [Pa*s], when physically defined. ``None`` in
            the two-phase region.
        k: Thermal conductivity [W/(m*K)], when physically defined. ``None``
            in the two-phase region.
        Pr: Prandtl number [-], ``cp * mu / k``, when ``cp``, ``mu`` and
            ``k`` are all defined; ``None`` otherwise.
        T_sat: Saturation temperature at ``p`` [K]. ``None`` at or above
            the critical pressure, where no saturation line exists.
        h_f: Saturated-liquid specific enthalpy at ``p`` [J/kg]. ``None``
            at or above the critical pressure.
        h_g: Saturated-vapor specific enthalpy at ``p`` [J/kg]. ``None``
            at or above the critical pressure.
        h_fg: Latent heat of vaporization at ``p`` [J/kg], ``h_g - h_f``.
            ``None`` at or above the critical pressure.
        warnings: Applicability / interpretation warnings.
    """

    T: float
    p: float
    h: float
    phase: WaterPhaseRegion
    quality: Optional[float]
    rho: Optional[float]
    cp: Optional[float]
    mu: Optional[float]
    k: Optional[float]
    Pr: Optional[float]
    T_sat: Optional[float]
    h_f: Optional[float]
    h_g: Optional[float]
    h_fg: Optional[float]
    warnings: list[ModelWarning]


def water_steam_state(
    *,
    T: Optional[float] = None,
    p: Optional[float] = None,
    x: Optional[float] = None,
    h: Optional[float] = None,
) -> WaterSteamState:
    """Build a fully classified pure-water/steam thermodynamic state.

    Exactly one of the following mutually exclusive input modes must be
    given:

        T + p:
            Ordinary bulk-state specification. Sufficient for superheated
            vapor and subcooled liquid. Rejected with a ``ValueError`` if
            ``T`` falls within the saturation-line ambiguity band at this
            pressure -- T+p alone cannot distinguish saturated liquid,
            saturated vapor, and a two-phase mixture there; use p+x or p+h
            instead.

        p + x:
            Saturated state at pressure ``p``, vapor quality
            ``x = m_vapor / (m_vapor + m_liquid)`` in ``[0, 1]``.

        p + h:
            State at pressure ``p``, classified by specific enthalpy ``h``
            against the saturation dome (``h_f(p)``, ``h_g(p)``).

    Args:
        T: Temperature [K].
        p: Pressure [Pa]. Required in every mode.
        x: Vapor quality [-], ``0 <= x <= 1``.
        h: Specific enthalpy [J/kg].

    Returns:
        WaterSteamState.

    Raises:
        ValueError: if the input is not exactly one of T+p, p+x, p+h; if
            ``x`` is outside ``[0, 1]``; if T+p is ambiguously on the
            saturation line; or if p+x / p+h is requested at or above the
            critical pressure, where saturated/two-phase states are not
            defined.

    Ref: IAPWS-IF97.
    """
    has_T = T is not None
    has_p = p is not None
    has_x = x is not None
    has_h = h is not None

    if not has_p:
        raise ValueError("p [Pa] must be provided in every input mode (T+p, p+x, p+h).")

    if has_T and not has_x and not has_h:
        return _water_steam_state_from_T_p(T, p)
    if has_x and not has_T and not has_h:
        return _water_steam_state_from_p_x(p, x)
    if has_h and not has_T and not has_x:
        return _water_steam_state_from_p_h(p, h)

    given = ", ".join(
        name for name, flag in (("T", has_T), ("x", has_x), ("h", has_h)) if flag
    )
    raise ValueError(
        "Water/steam state must be specified using exactly one of T+p, "
        f"p+x, or p+h. Got p plus: {given or 'nothing else'}."
    )


def _water_steam_state_from_T_p(T: float, p: float) -> WaterSteamState:
    _validate_temperature(T)
    _validate_pressure(p)

    props = water_steam_props_iapws97(T=T, p=p)
    transport = props.transport
    Pr = _prandtl_number(transport.cp, transport.mu, transport.k)

    if p >= WATER_CRITICAL_PRESSURE_PA:
        return WaterSteamState(
            T=T,
            p=p,
            h=props.h,
            phase=WaterPhaseRegion.SUPERCRITICAL,
            quality=None,
            rho=transport.rho,
            cp=transport.cp,
            mu=transport.mu,
            k=transport.k,
            Pr=Pr,
            T_sat=None,
            h_f=None,
            h_g=None,
            h_fg=None,
            warnings=list(props.warnings),
        )

    T_sat = water_saturation_temperature(p)
    if abs(T - T_sat) <= _SATURATION_AMBIGUITY_TOLERANCE_K:
        raise ValueError(
            f"T={T} K at p={p} Pa is within "
            f"{_SATURATION_AMBIGUITY_TOLERANCE_K} K of the saturation "
            f"temperature T_sat={T_sat} K. T+p cannot distinguish saturated "
            "liquid, saturated vapor and a two-phase mixture at this point "
            "-- specify the state using p+x or p+h instead."
        )

    h_f = water_saturation_liquid_enthalpy(p=p)
    h_g = water_saturation_vapor_enthalpy(p=p)
    h_fg = h_g - h_f
    phase = (
        WaterPhaseRegion.SUPERHEATED_VAPOR
        if T > T_sat
        else WaterPhaseRegion.SUBCOOLED_LIQUID
    )

    return WaterSteamState(
        T=T,
        p=p,
        h=props.h,
        phase=phase,
        quality=None,
        rho=transport.rho,
        cp=transport.cp,
        mu=transport.mu,
        k=transport.k,
        Pr=Pr,
        T_sat=T_sat,
        h_f=h_f,
        h_g=h_g,
        h_fg=h_fg,
        warnings=list(props.warnings),
    )


def _water_steam_state_from_p_x(p: float, x: float) -> WaterSteamState:
    _validate_pressure(p)
    _validate_vapor_quality(x)
    _reject_supercritical_saturation(p)
    _validate_saturation_pressure(p)

    T_sat = water_saturation_temperature(p)
    h_f = water_saturation_liquid_enthalpy(p=p)
    h_g = water_saturation_vapor_enthalpy(p=p)
    h_fg = h_g - h_f
    h = h_f + x * h_fg

    rho, cp, mu, k, warnings = _saturated_transport(p, x)
    Pr = _prandtl_number(cp, mu, k)

    return WaterSteamState(
        T=T_sat,
        p=p,
        h=h,
        phase=_phase_region_for_quality(x),
        quality=x,
        rho=rho,
        cp=cp,
        mu=mu,
        k=k,
        Pr=Pr,
        T_sat=T_sat,
        h_f=h_f,
        h_g=h_g,
        h_fg=h_fg,
        warnings=warnings,
    )


def _water_steam_state_from_p_h(p: float, h: float) -> WaterSteamState:
    _validate_pressure(p)
    if not math.isfinite(h):
        raise ValueError("Specific enthalpy must be finite [J/kg].")
    _reject_supercritical_saturation(p)
    _validate_saturation_pressure(p)

    T_sat = water_saturation_temperature(p)
    h_f = water_saturation_liquid_enthalpy(p=p)
    h_g = water_saturation_vapor_enthalpy(p=p)
    h_fg = h_g - h_f
    tol = _SATURATION_ENTHALPY_TOLERANCE_RELATIVE * h_fg

    if h > h_g + tol:
        return _single_phase_state_from_p_h(
            p, h, WaterPhaseRegion.SUPERHEATED_VAPOR, T_sat, h_f, h_g, h_fg
        )
    if h < h_f - tol:
        return _single_phase_state_from_p_h(
            p, h, WaterPhaseRegion.SUBCOOLED_LIQUID, T_sat, h_f, h_g, h_fg
        )

    if h >= h_g - tol:
        x = 1.0
    elif h <= h_f + tol:
        x = 0.0
    else:
        x = (h - h_f) / h_fg

    rho, cp, mu, k, warnings = _saturated_transport(p, x)
    Pr = _prandtl_number(cp, mu, k)

    return WaterSteamState(
        T=T_sat,
        p=p,
        h=h,
        phase=_phase_region_for_quality(x),
        quality=x,
        rho=rho,
        cp=cp,
        mu=mu,
        k=k,
        Pr=Pr,
        T_sat=T_sat,
        h_f=h_f,
        h_g=h_g,
        h_fg=h_fg,
        warnings=warnings,
    )


def _single_phase_state_from_p_h(
    p: float,
    h: float,
    phase: WaterPhaseRegion,
    T_sat: float,
    h_f: float,
    h_g: float,
    h_fg: float,
) -> WaterSteamState:
    try:
        state = IAPWS97(P=_pa_to_mpa(p), h=h / 1000.0)
    except Exception as exc:
        raise ValueError(f"IAPWS-IF97 failed for p={p} Pa, h={h} J/kg.") from exc

    rho, cp, mu, k, warnings = _transport_from_iapws_state(state, two_phase=False)
    Pr = _prandtl_number(cp, mu, k)

    return WaterSteamState(
        T=float(state.T),
        p=p,
        h=h,
        phase=phase,
        quality=None,
        rho=rho,
        cp=cp,
        mu=mu,
        k=k,
        Pr=Pr,
        T_sat=T_sat,
        h_f=h_f,
        h_g=h_g,
        h_fg=h_fg,
        warnings=warnings,
    )


def _saturated_transport(
    p: float, x: float
) -> tuple[float, Optional[float], Optional[float], Optional[float], list[ModelWarning]]:
    """Return (rho, cp, mu, k, warnings) for a saturated state p+x."""
    try:
        state = IAPWS97(P=_pa_to_mpa(p), x=x)
    except Exception as exc:
        raise ValueError(f"IAPWS-IF97 saturated state failed for p={p} Pa, x={x}.") from exc
    return _transport_from_iapws_state(state, two_phase=(0.0 < x < 1.0))


def _transport_from_iapws_state(
    state: IAPWS97, *, two_phase: bool
) -> tuple[float, Optional[float], Optional[float], Optional[float], list[ModelWarning]]:
    rho = float(state.rho)
    if two_phase:
        warnings = [
            _warning(
                code="WATER_STEAM_TWO_PHASE_STATE",
                message=(
                    "Two-phase water/steam state: cp, mu and k are not "
                    "ordinary single-phase transport properties in this "
                    "region and are reported as None."
                ),
                severity="warning",
            )
        ]
        return rho, None, None, None, warnings
    cp = float(state.cp) * 1000.0
    mu = float(state.mu)
    k = float(state.k)
    return rho, cp, mu, k, []


def _prandtl_number(
    cp: Optional[float], mu: Optional[float], k: Optional[float]
) -> Optional[float]:
    if cp is None or mu is None or k is None:
        return None
    return cp * mu / k


def _phase_region_for_quality(x: float) -> WaterPhaseRegion:
    if x <= 0.0:
        return WaterPhaseRegion.SATURATED_LIQUID
    if x >= 1.0:
        return WaterPhaseRegion.SATURATED_VAPOR
    return WaterPhaseRegion.TWO_PHASE


def _reject_supercritical_saturation(p: float) -> None:
    if p >= WATER_CRITICAL_PRESSURE_PA:
        raise ValueError(
            f"p={p} Pa is at or above the critical pressure "
            f"({WATER_CRITICAL_PRESSURE_PA} Pa). Saturated/two-phase water "
            "states (p+x, or p+h phase classification against the "
            "saturation dome) are not defined there; pure-water phase "
            "change above the critical pressure is out of scope."
        )
