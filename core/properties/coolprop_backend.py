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

REFPROP support is intentionally optional and user-provided. KalKalori does not
distribute REFPROP binaries, fluid files, mixture files, or license files.
If selected as a CoolProp backend, REFPROP must be installed, licensed, and
configured locally by the user.

Refs:
- CoolProp high-level API, PropsSI / PhaseSI.
- CoolProp REFPROP backend documentation.
- NIST REFPROP documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Mapping

from core.common.warnings import ModelWarning
from core.properties.common import FluidTransportProperties


SOURCE = "properties.coolprop_backend"

SUPPORTED_IMPOSED_PHASES = {
    "liquid",
    "gas",
    "twophase",
    "supercritical_liquid",
    "supercritical_gas",
    "supercritical",
    "not_imposed",
}


def _warning(code: str, message: str, severity: str = "warning") -> ModelWarning:
    return ModelWarning(
        code=code,
        message=message,
        severity=severity,
        source=SOURCE,
    )


@dataclass(frozen=True)
class PropertyBackendAvailability:
    """Availability status for an optional property backend.

    Attributes:
        backend:
            CoolProp backend name, for example "HEOS" or "REFPROP".
        available:
            True if the backend passed a minimal runtime availability check.
        message:
            Human-readable status or recovery instruction.
    """

    backend: str
    available: bool
    message: str


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
        CoolPropFluidProvider("REFPROP::Water")

    Notes:
        `imposed_phase` is disabled by default for pure/pseudo-pure fluids.
        Use it only when the phase region is known.
    """

    fluid: str
    imposed_phase: str | None = None

    def __post_init__(self) -> None:
        _validate_fluid_string(self.fluid)
        _validate_imposed_phase(self.imposed_phase)

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return transport properties at T [K] and p [Pa]."""
        return coolprop_transport_properties(
            T=T,
            p=p,
            fluid=self.fluid,
            imposed_phase=self.imposed_phase,
        )

    def full_at(self, T: float, p: float) -> CoolPropProperties:
        """Return transport and thermodynamic properties at T [K] and p [Pa]."""
        return coolprop_props(
            T=T,
            p=p,
            fluid=self.fluid,
            imposed_phase=self.imposed_phase,
        )

    def specific_enthalpy(self, T: float, p: float) -> float:
        """Return only specific enthalpy [J/kg] at T [K], p [Pa]."""
        return coolprop_specific_enthalpy(
            T=T,
            p=p,
            fluid=self.fluid,
            imposed_phase=self.imposed_phase,
        )

    def temperature_from_h_p(self, h: float, p: float) -> float:
        """Invert specific enthalpy and pressure to temperature [K]."""
        return coolprop_temperature_from_h_p(
            h=h, p=p, fluid=self.fluid, imposed_phase=self.imposed_phase
        )

    def saturation_temperature(self, p: float) -> float | None:
        """Return backend-native saturation temperature, or ``None`` above critical."""
        return coolprop_saturation_temperature(p=p, fluid=self.fluid)


@dataclass(frozen=True)
class CoolPropGasMixtureProvider:
    """CoolProp provider for gas mixtures defined by mole fractions.

    Components must use CoolProp component names, for example:
        {"Nitrogen": 0.74, "Oxygen": 0.04, "CarbonDioxide": 0.12, "Water": 0.10}

    The provider builds a CoolProp mixture string such as:
        HEOS::Nitrogen[0.74]&Oxygen[0.04]&CarbonDioxide[0.12]&Water[0.10]

    Backend examples:
        backend="HEOS"
        backend="REFPROP"

    Notes:
        CoolProp mixture support depends on the selected components, backend,
        and state point. Some mixture transport properties may be unavailable.

        For gas mixtures, `imposed_phase="gas"` is used by default. This avoids
        some CoolProp mixture PT-flash failures when the gas phase is known from
        the engineering context. Do not use this provider for condensing or
        liquid-containing mixture states without revisiting the imposed phase.

        REFPROP is optional and must be installed/licensed/configured locally
        by the user if selected.
    """

    components: Mapping[str, float]
    backend: str = "HEOS"
    normalize: bool = True
    imposed_phase: str | None = "gas"

    def __post_init__(self) -> None:
        _validate_backend(self.backend)
        _validate_imposed_phase(self.imposed_phase)
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

    @property
    def backend_availability(self) -> PropertyBackendAvailability:
        """Return runtime availability status of the selected backend."""
        return check_coolprop_backend_availability(self.backend)

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return transport properties at T [K] and p [Pa]."""
        require_coolprop_backend(self.backend)
        return coolprop_transport_properties(
            T=T,
            p=p,
            fluid=self.fluid,
            imposed_phase=self.imposed_phase,
        )

    def full_at(self, T: float, p: float) -> CoolPropProperties:
        """Return transport and thermodynamic properties at T [K] and p [Pa]."""
        require_coolprop_backend(self.backend)
        return coolprop_props(
            T=T,
            p=p,
            fluid=self.fluid,
            imposed_phase=self.imposed_phase,
        )

    def specific_enthalpy(self, T: float, p: float) -> float:
        """Return only specific enthalpy [J/kg] at T [K], p [Pa]."""
        return coolprop_specific_enthalpy(
            T=T,
            p=p,
            fluid=self.fluid,
            imposed_phase=self.imposed_phase,
        )

    def temperature_from_h_p(self, h: float, p: float) -> float:
        """Invert specific enthalpy and pressure to temperature [K]."""
        return coolprop_temperature_from_h_p(
            h=h, p=p, fluid=self.fluid, imposed_phase=self.imposed_phase
        )


def coolprop_props(
    T: float,
    p: float,
    fluid: str,
    *,
    imposed_phase: str | None = None,
) -> CoolPropProperties:
    """Return CoolProp properties in KalKalori SI units.

    Args:
        T: Temperature [K].
        p: Pressure [Pa].
        fluid: CoolProp fluid string.
        imposed_phase:
            Optional CoolProp phase hint, for example "gas".
            If provided, properties are evaluated using the CoolProp syntax
            "T|phase", e.g. "T|gas".

    Returns:
        CoolPropProperties.

    Notes:
        CoolProp `PropsSI` already uses SI units for the requested properties:
        density, viscosity, thermal conductivity, cp, and enthalpy.

        Phase imposition should only be used when the phase region is known.
        It is useful for high-temperature gas mixtures where automatic PT flash
        can fail even though the engineering state is clearly gas-phase.
    """
    _validate_temperature(T)
    _validate_pressure(p)
    _validate_fluid_string(fluid)
    _validate_imposed_phase(imposed_phase)

    backend = backend_from_fluid_string(fluid)
    if backend is not None:
        require_coolprop_backend(backend)

    CP = _coolprop_module()
    warnings: list[ModelWarning] = []
    T_key = _input_key_with_imposed_phase("T", imposed_phase)

    try:
        rho = CP.PropsSI("Dmass", T_key, T, "P", p, fluid)
        mu = CP.PropsSI("V", T_key, T, "P", p, fluid)
        k = CP.PropsSI("L", T_key, T, "P", p, fluid)
        cp = CP.PropsSI("Cpmass", T_key, T, "P", p, fluid)
        h = CP.PropsSI("Hmass", T_key, T, "P", p, fluid)
    except Exception as exc:
        phase_note = (
            f", imposed_phase={imposed_phase!r}"
            if imposed_phase is not None
            else ""
        )
        raise ValueError(
            f"CoolProp failed for fluid={fluid!r}, T={T} K, p={p} Pa"
            f"{phase_note}. CoolProp error: {exc}"
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


def coolprop_transport_properties(
    T: float,
    p: float,
    fluid: str,
    *,
    imposed_phase: str | None = None,
) -> FluidTransportProperties:
    """Return only transport properties for one CoolProp T,p state.

    Unlike :func:`coolprop_props`, this path deliberately does not query
    specific enthalpy or phase because callers of ``PropertyProvider.at``
    cannot observe either value.
    """
    _validate_temperature(T)
    _validate_pressure(p)
    _validate_fluid_string(fluid)
    _validate_imposed_phase(imposed_phase)
    backend = backend_from_fluid_string(fluid)
    if backend is not None:
        require_coolprop_backend(backend)
    CP = _coolprop_module()
    T_key = _input_key_with_imposed_phase("T", imposed_phase)
    try:
        return FluidTransportProperties(
            rho=CP.PropsSI("Dmass", T_key, T, "P", p, fluid),
            mu=CP.PropsSI("V", T_key, T, "P", p, fluid),
            k=CP.PropsSI("L", T_key, T, "P", p, fluid),
            cp=CP.PropsSI("Cpmass", T_key, T, "P", p, fluid),
        )
    except Exception as exc:
        phase_note = (
            f", imposed_phase={imposed_phase!r}"
            if imposed_phase is not None
            else ""
        )
        raise ValueError(
            f"CoolProp transport properties failed for fluid={fluid!r}, "
            f"T={T} K, p={p} Pa{phase_note}. CoolProp error: {exc}"
        ) from exc


def coolprop_specific_enthalpy(
    T: float,
    p: float,
    fluid: str,
    *,
    imposed_phase: str | None = None,
) -> float:
    """Return only CoolProp ``Hmass`` [J/kg] for one T,p state.

    This is the enthalpy-only counterpart of :func:`coolprop_props`.  It is
    used by iterative energy-balance roots that do not need density,
    viscosity, conductivity, heat capacity, or a phase query at every trial.
    """
    _validate_temperature(T)
    _validate_pressure(p)
    _validate_fluid_string(fluid)
    _validate_imposed_phase(imposed_phase)
    backend = backend_from_fluid_string(fluid)
    if backend is not None:
        require_coolprop_backend(backend)
    CP = _coolprop_module()
    T_key = _input_key_with_imposed_phase("T", imposed_phase)
    try:
        return float(CP.PropsSI("Hmass", T_key, T, "P", p, fluid))
    except Exception as exc:
        phase_note = (
            f", imposed_phase={imposed_phase!r}"
            if imposed_phase is not None
            else ""
        )
        raise ValueError(
            f"CoolProp enthalpy failed for fluid={fluid!r}, T={T} K, "
            f"p={p} Pa{phase_note}. CoolProp error: {exc}"
        ) from exc


def coolprop_temperature_from_h_p(
    h: float,
    p: float,
    fluid: str,
    *,
    imposed_phase: str | None = None,
) -> float:
    """Return temperature [K] from CoolProp ``h,p`` inputs."""
    if not math.isfinite(h):
        raise ValueError("Specific enthalpy must be finite [J/kg].")
    _validate_pressure(p)
    _validate_fluid_string(fluid)
    _validate_imposed_phase(imposed_phase)
    backend = backend_from_fluid_string(fluid)
    if backend is not None:
        require_coolprop_backend(backend)
    CP = _coolprop_module()
    H_key = _input_key_with_imposed_phase("Hmass", imposed_phase)
    try:
        return float(CP.PropsSI("T", H_key, h, "P", p, fluid))
    except Exception as exc:
        phase_note = f", imposed_phase={imposed_phase!r}" if imposed_phase is not None else ""
        raise ValueError(
            f"CoolProp h,p inversion failed for fluid={fluid!r}, h={h} J/kg, p={p} Pa"
            f"{phase_note}. CoolProp error: {exc}"
        ) from exc


def coolprop_saturation_temperature(*, p: float, fluid: str) -> float | None:
    """Return pure-fluid saturation temperature [K] at ``p`` using CoolProp.

    ``None`` is returned at or above the backend's critical pressure. This
    helper is used only to guard unsupported phase crossings while retaining
    the caller-selected property backend; it does not substitute IAPWS.
    """
    _validate_pressure(p)
    _validate_fluid_string(fluid)
    backend = backend_from_fluid_string(fluid)
    if backend is not None:
        require_coolprop_backend(backend)
    CP = _coolprop_module()
    try:
        p_critical = float(CP.PropsSI("pcrit", fluid))
        if p >= p_critical:
            return None
        return float(CP.PropsSI("T", "P", p, "Q", 0.0, fluid))
    except Exception as exc:
        raise ValueError(
            f"CoolProp saturation temperature failed for fluid={fluid!r}, p={p} Pa. "
            f"CoolProp error: {exc}"
        ) from exc


def build_coolprop_mixture_string(
    components: Mapping[str, float],
    *,
    backend: str = "HEOS",
    normalize: bool = True,
) -> str:
    """Build a CoolProp mixture string from mole fractions.

    Args:
        components: Mapping of CoolProp component name to mole fraction [-].
        backend: CoolProp backend prefix, usually "HEOS" or "REFPROP".
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
    return f"{backend.strip()}::" + "&".join(parts)


def normalize_mole_fractions(components: Mapping[str, float]) -> dict[str, float]:
    """Return normalized mole fractions from a component mapping."""
    fractions = dict(components)
    _validate_mole_fractions(fractions, require_sum_one=False)

    total = sum(fractions.values())
    if total <= 0.0:
        raise ValueError("Sum of mole fractions must be positive.")

    return {component: value / total for component, value in fractions.items()}


def backend_from_fluid_string(fluid: str) -> str | None:
    """Return backend prefix from a CoolProp fluid string, if present.

    Examples:
        "HEOS::Water" -> "HEOS"
        "REFPROP::Water" -> "REFPROP"
        "Water" -> None
    """
    _validate_fluid_string(fluid)

    if "::" not in fluid:
        return None

    backend = fluid.split("::", 1)[0].strip()
    if not backend:
        return None

    return backend


@lru_cache(maxsize=None)
def check_coolprop_backend_availability(
    backend: str,
    *,
    test_fluid: str | None = None,
) -> PropertyBackendAvailability:
    """Check runtime availability of a CoolProp backend.

    Args:
        backend:
            CoolProp backend prefix, for example "HEOS" or "REFPROP".
        test_fluid:
            Optional fluid string used for the sanity check. If omitted, a
            backend-appropriate simple pure-fluid state is used.

    Returns:
        PropertyBackendAvailability.

    Notes:
        REFPROP availability depends on local installation, licensing, and
        CoolProp configuration. KalKalori does not distribute REFPROP.
    """
    _validate_backend(backend)
    backend_clean = backend.strip()
    backend_upper = backend_clean.upper()

    try:
        CP = _coolprop_module()
    except ImportError:
        return PropertyBackendAvailability(
            backend=backend_clean,
            available=False,
            message=(
                f"CoolProp backend {backend_clean!r} is not available because "
                "optional dependency 'CoolProp' is not installed. "
                "Install it with: pip install CoolProp"
            ),
        )

    if test_fluid is None:
        if backend_upper == "HEOS":
            test_fluid = "HEOS::Water"
        elif backend_upper == "REFPROP":
            test_fluid = "REFPROP::Water"
        else:
            test_fluid = f"{backend_clean}::Water"

    try:
        CP.PropsSI("Dmass", "T", 300.0, "P", 101325.0, test_fluid)
    except Exception as exc:
        if backend_upper == "REFPROP":
            message = (
                "REFPROP backend is not available through CoolProp. "
                "Install and license REFPROP locally, configure the CoolProp "
                "REFPROP path, or use backend='HEOS'. KalKalori does not "
                "distribute REFPROP binaries, fluid files, mixture files, or "
                f"license files. CoolProp error: {exc}"
            )
        else:
            message = (
                f"CoolProp backend {backend_clean!r} is not available or failed "
                f"the runtime sanity check with fluid {test_fluid!r}. "
                f"CoolProp error: {exc}"
            )

        return PropertyBackendAvailability(
            backend=backend_clean,
            available=False,
            message=message,
        )

    return PropertyBackendAvailability(
        backend=backend_clean,
        available=True,
        message=f"CoolProp backend {backend_clean!r} is available.",
    )


def require_coolprop_backend(backend: str) -> None:
    """Raise a clear error if a CoolProp backend is not available."""
    availability = check_coolprop_backend_availability(backend)

    if not availability.available:
        raise RuntimeError(availability.message)


def _coolprop_module():
    try:
        import CoolProp.CoolProp as CP  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "CoolProp backend requires optional dependency 'CoolProp'. "
            "Install it with: pip install CoolProp"
        ) from exc

    return CP


def _input_key_with_imposed_phase(key: str, imposed_phase: str | None) -> str:
    """Return CoolProp input key with optional phase imposition."""
    _validate_imposed_phase(imposed_phase)

    if imposed_phase is None:
        return key

    phase = imposed_phase.strip().lower()
    if phase == "not_imposed":
        return key

    return f"{key}|{phase}"


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


def _validate_imposed_phase(imposed_phase: str | None) -> None:
    if imposed_phase is None:
        return

    if not isinstance(imposed_phase, str):
        raise ValueError("CoolProp imposed phase must be a string or None.")

    phase = imposed_phase.strip().lower()
    if not phase:
        raise ValueError("CoolProp imposed phase must not be empty.")

    if phase not in SUPPORTED_IMPOSED_PHASES:
        raise ValueError(
            f"Unsupported CoolProp imposed phase {imposed_phase!r}. "
            f"Supported phases are: {sorted(SUPPORTED_IMPOSED_PHASES)}."
        )


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
