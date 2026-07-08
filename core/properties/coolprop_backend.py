"""
Optional CoolProp property backend.

This module provides a thin SI-unit adapter around CoolProp for pure fluids,
pseudo-pure fluids, and CoolProp fluid-string based mixtures.

KalKalori core uses SI floats:
- temperature: K
- pressure: Pa
- density: kg/m3
- dynamic viscosity: Pa*s
- thermal conductivity: W/(m*K)
- specific heat: J/(kg*K)
- specific enthalpy: J/kg

CoolProp is an optional dependency. Importing this module does not require
CoolProp to be installed, but calling its property functions does.

Refs:
- CoolProp high-level API, PropsSI / PhaseSI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

from core.common.warnings import ModelWarning
from core.properties.common import FluidTransportProperties


SOURCE = "properties.coolprop_backend"


def _warning(code: str, message: str, severity: str = "warning") -> ModelWarning:
    return ModelWarning(
        code=code,
        message=message,
        severity=severity,
        source=SOURCE,
    )


@dataclass(frozen=True)
class CoolPropProperties:
    """Fluid properties returned by the optional CoolProp backend.

    Attributes:
        transport:
            Transport properties in KalKalori SI units.
        h:
            Specific enthalpy [J/kg].
        phase:
            Phase string returned by CoolProp, normalized lightly.
        fluid:
            CoolProp fluid string used for the calculation.
        warnings:
            Applicability / backend warnings.
    """

    transport: FluidTransportProperties
    h: float
    phase: str
    fluid: str
    warnings: list[ModelWarning] = field(default_factory=list)


@dataclass(frozen=True)
class CoolPropFluidProvider:
    """Property provider for a CoolProp fluid string.

    Examples:
        CoolPropFluidProvider("Air")
        CoolPropFluidProvider("CO2")
        CoolPropFluidProvider("Water")
        CoolPropFluidProvider("HEOS::Nitrogen[0.78]&Oxygen[0.21]&CarbonDioxide[0.01]")
    """

    fluid: str

    def __post_init__(self) -> None:
        _validate_fluid_string(self.fluid)

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return transport properties at T [K] and p [Pa]."""
        return coolprop_props(T=T, p=p, fluid=self.fluid).transport

    def full_at(self, T: float, p: float) -> CoolPropProperties:
        """Return transport and thermodynamic properties at T [K] and p [Pa]."""
        return coolprop_props(T=T, p=p, fluid=self.fluid)


@dataclass(frozen=True)
class CoolPropGasMixtureProvider:
    """CoolProp provider for gas mixtures defined by mole fractions.

    Components must use CoolProp component names, for example:
        {"Nitrogen": 0.74, "Oxygen": 0.04, "CarbonDioxide": 0.12, "Water": 0.10}

    The provider builds a CoolProp mixture string such as:
        HEOS::Nitrogen[0.74]&Oxygen[0.04]&CarbonDioxide[0.12]&Water[0.10]

    Notes:
        CoolProp mixture support depends on the selected components, backend,
        and state point. Some mixture transport properties may be unavailable.
        In such cases this backend raises a ValueError with the CoolProp error
        attached as context.
    """

    components: Mapping[str, float]
    backend: str = "HEOS"
    normalize: bool = True

    def __post_init__(self) -> None:
        _validate_backend(self.backend)
        normalized = normalize_mole_fractions(self.components)
        if not normalized:
            raise ValueError("Gas mixture must contain at least one component.")

    @property
    def fluid(self) -> str:
        """Return CoolProp mixture fluid string."""
        return build_coolprop_mixture_string(
            self.components,
            backend=self.backend,
            normalize=self.normalize,
        )

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return transport properties at T [K] and p [Pa]."""
        return coolprop_props(T=T, p=p, fluid=self.fluid).transport

    def full_at(self, T: float, p: float) -> CoolPropProperties:
        """Return transport and thermodynamic properties at T [K] and p [Pa]."""
        return coolprop_props(T=T, p=p, fluid=self.fluid)


def coolprop_props(
    T: float,
    p: float,
    fluid: str,
) -> CoolPropProperties:
    """Return CoolProp properties in KalKalori SI units.

    Args:
        T: Temperature [K].
        p: Pressure [Pa].
        fluid: CoolProp fluid string.

    Returns:
        CoolPropProperties.

    Notes:
        CoolProp `PropsSI` already uses SI units for the requested properties:
        density, viscosity, thermal conductivity, cp, and enthalpy.
    """
    _validate_temperature(T)
    _validate_pressure(p)
    _validate_fluid_string(fluid)

    CP = _coolprop_module()
    warnings: list[ModelWarning] = []

    try:
        rho = CP.PropsSI("Dmass", "T", T, "P", p, fluid)
        mu = CP.PropsSI("V", "T", T, "P", p, fluid)
        k = CP.PropsSI("L", "T", T, "P", p, fluid)
        cp = CP.PropsSI("Cpmass", "T", T, "P", p, fluid)
        h = CP.PropsSI("Hmass", "T", T, "P", p, fluid)
    except Exception as exc:
        raise ValueError(
            f"CoolProp failed for fluid={fluid!r}, T={T} K, p={p} Pa."
        ) from exc

    try:
        phase = _normalize_phase(CP.PhaseSI("T", T, "P", p, fluid))
    except Exception:
        phase = "unknown"
        warnings.append(
            _warning(
                code="COOLPROP_PHASE_UNAVAILABLE",
                message=(
                    "CoolProp properties were calculated, but phase information "
                    "could not be retrieved for this state."
                ),
                severity="info",
            )
        )

    transport = FluidTransportProperties(
        rho=rho,
        mu=mu,
        k=k,
        cp=cp,
    )

    return CoolPropProperties(
        transport=transport,
        h=h,
        phase=phase,
        fluid=fluid,
        warnings=warnings,
    )


def build_coolprop_mixture_string(
    components: Mapping[str, float],
    *,
    backend: str = "HEOS",
    normalize: bool = True,
) -> str:
    """Build a CoolProp mixture string from mole fractions.

    Args:
        components: Mapping of CoolProp component name to mole fraction [-].
        backend: CoolProp backend prefix, usually "HEOS".
        normalize: If True, mole fractions are normalized to sum to 1.

    Returns:
        CoolProp mixture string, e.g.:
        "HEOS::Nitrogen[0.78]&Oxygen[0.21]&CarbonDioxide[0.01]".
    """
    _validate_backend(backend)

    if normalize:
        fractions = normalize_mole_fractions(components)
    else:
        fractions = dict(components)
        _validate_mole_fractions(fractions, require_sum_one=True)

    parts = [f"{component}[{fraction:.12g}]" for component, fraction in fractions.items()]
    return f"{backend}::" + "&".join(parts)


def normalize_mole_fractions(components: Mapping[str, float]) -> dict[str, float]:
    """Return normalized mole fractions from a component mapping."""
    fractions = dict(components)
    _validate_mole_fractions(fractions, require_sum_one=False)

    total = sum(fractions.values())
    if total <= 0.0:
        raise ValueError("Sum of mole fractions must be positive.")

    return {component: value / total for component, value in fractions.items()}


def _coolprop_module():
    try:
        import CoolProp.CoolProp as CP  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "CoolProp backend requires optional dependency 'CoolProp'. "
            "Install it with: pip install CoolProp"
        ) from exc

    return CP


def _validate_fluid_string(fluid: str) -> None:
    if not isinstance(fluid, str):
        raise ValueError("CoolProp fluid must be a string.")
    if not fluid.strip():
        raise ValueError("CoolProp fluid string must not be empty.")


def _validate_backend(backend: str) -> None:
    if not isinstance(backend, str):
        raise ValueError("CoolProp backend must be a string.")
    if not backend.strip():
        raise ValueError("CoolProp backend must not be empty.")


def _validate_mole_fractions(
    components: Mapping[str, float],
    *,
    require_sum_one: bool,
) -> None:
    if not components:
        raise ValueError("At least one component is required.")

    for component, fraction in components.items():
        if not isinstance(component, str) or not component.strip():
            raise ValueError("Component names must be non-empty strings.")
        if not math.isfinite(fraction):
            raise ValueError(f"Mole fraction for {component!r} must be finite.")
        if fraction < 0.0:
            raise ValueError(f"Mole fraction for {component!r} must be non-negative.")

    total = sum(components.values())

    if total <= 0.0:
        raise ValueError("Sum of mole fractions must be positive.")

    if require_sum_one and abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Mole fractions must sum to 1.0 when normalize=False; got {total:.12g}."
        )


def _validate_temperature(T: float) -> None:
    if not math.isfinite(T):
        raise ValueError("Temperature must be finite [K].")
    if T <= 0.0:
        raise ValueError("Temperature must be above absolute zero [K].")


def _validate_pressure(p: float) -> None:
    if not math.isfinite(p):
        raise ValueError("Pressure must be finite [Pa].")
    if p <= 0.0:
        raise ValueError("Pressure must be positive [Pa].")


def _normalize_phase(phase: object) -> str:
    return str(phase).strip().lower().replace(" ", "_")
