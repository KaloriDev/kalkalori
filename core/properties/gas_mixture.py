"""
Gas-mixture property helpers based on the optional CoolProp backend.

This module provides a user-facing gas-mixture specification layer above
`core.properties.coolprop_backend`.

Supported composition bases:
- mole:   mole fractions [-]
- volume: volume fractions [-], treated as mole fractions for gas mixtures
- mass:   mass fractions [-], converted to mole fractions using molar masses

Supported backend selection:
- HEOS:    default open CoolProp backend
- REFPROP: optional user-provided backend through CoolProp

KalKalori core uses SI floats:
- temperature: K
- pressure: Pa
- density: kg/m3
- dynamic viscosity: Pa*s
- thermal conductivity: W/(m*K)
- specific heat: J/(kg*K)

Notes:
    This module does not calculate flue-gas composition from fuel combustion.
    The user must provide the gas composition explicitly.

    Water vapor may be included as an explicit gas-phase component, for
    example with gas_mixture_from_dry_composition_and_water_ratio(...).
    This still creates a normal GasMixtureSpec and does not imply
    condensation, wet-surface heat transfer, or composition changes.

    For now, gas-mixture properties are delegated to the optional CoolProp
    backend. CoolProp availability and mixture support depend on installed
    CoolProp version, selected components, backend, and state point.

    Gas-mixture calculations use `imposed_phase="gas"` by default. This avoids
    some CoolProp mixture PT-flash failures for known gas-phase engineering
    states. Do not use this default for condensing or liquid-containing states
    without revisiting the phase assumption.

    REFPROP is not distributed with KalKalori. If backend="REFPROP" is selected,
    REFPROP must be installed, licensed, and configured locally by the user.

Refs:
- CoolProp documentation, high-level API and mixture fluid strings.
- CoolProp REFPROP backend documentation.
- NIST REFPROP documentation.
- Incropera et al., Fundamentals of Heat and Mass Transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from core.properties.common import FluidTransportProperties
from core.properties.coolprop_backend import (
    CoolPropGasMixtureProvider,
    CoolPropProperties,
    PropertyBackendAvailability,
    build_coolprop_mixture_string,
    check_coolprop_backend_availability,
    normalize_mole_fractions,
    require_coolprop_backend,
)


SUPPORTED_BASES = {"mole", "volume", "mass"}


# Molar masses [kg/mol].
# Keys are intended to match common CoolProp component names.
# Aliases are included for practical engineering input convenience.
COMMON_MOLAR_MASSES: dict[str, float] = {
    "Nitrogen": 28.0134e-3,
    "N2": 28.0134e-3,
    "Oxygen": 31.9988e-3,
    "O2": 31.9988e-3,
    "CarbonDioxide": 44.0095e-3,
    "CO2": 44.0095e-3,
    "Water": 18.01528e-3,
    "H2O": 18.01528e-3,
    "Argon": 39.948e-3,
    "Ar": 39.948e-3,
    "CarbonMonoxide": 28.0101e-3,
    "CO": 28.0101e-3,
    "SulfurDioxide": 64.066e-3,
    "SO2": 64.066e-3,
    "Methane": 16.0425e-3,
    "CH4": 16.0425e-3,
    "Hydrogen": 2.01588e-3,
    "H2": 2.01588e-3,
    "Helium": 4.002602e-3,
    "He": 4.002602e-3,
}


# Component aliases to preferred CoolProp names.
COMPONENT_ALIASES: dict[str, str] = {
    "N2": "Nitrogen",
    "O2": "Oxygen",
    "CO2": "CarbonDioxide",
    "H2O": "Water",
    "Ar": "Argon",
    "CO": "CarbonMonoxide",
    "SO2": "SulfurDioxide",
    "CH4": "Methane",
    "H2": "Hydrogen",
    "He": "Helium",
}


@dataclass(frozen=True)
class GasMixtureSpec:
    """Gas-mixture composition specification.

    Attributes:
        components:
            Mapping of component name to fraction [-].
            Component names should be CoolProp-compatible names, for example:
            "Nitrogen", "Oxygen", "CarbonDioxide", "Water".

            Common aliases such as "N2", "O2", "CO2", "H2O" are accepted and
            converted to preferred CoolProp names.

        basis:
            Composition basis:
            - "mole": mole fractions [-]
            - "volume": volume fractions [-], treated as mole fractions
            - "mass": mass fractions [-], converted to mole fractions

        backend:
            CoolProp backend prefix:
            - "HEOS": default open CoolProp backend
            - "REFPROP": optional user-provided backend through CoolProp

        normalize:
            If True, fractions are normalized to sum to 1.0.

        molar_masses:
            Optional molar-mass override mapping [kg/mol].
            Useful for custom components or if a component alias is not covered
            by `COMMON_MOLAR_MASSES`.

        imposed_phase:
            Optional CoolProp phase hint. The default is "gas" for gas-mixture
            engineering calculations. Set to None only if automatic phase
            detection is desired.
    """

    components: Mapping[str, float]
    basis: str = "mole"
    backend: str = "HEOS"
    normalize: bool = True
    molar_masses: Mapping[str, float] | None = None
    imposed_phase: str | None = "gas"

    def __post_init__(self) -> None:
        _validate_basis(self.basis)
        _validate_backend(self.backend)
        _validate_imposed_phase_name(self.imposed_phase)
        _validate_fraction_mapping(self.components)

        if self.molar_masses is not None:
            _validate_molar_masses(self.molar_masses)

        # Force early validation of mass-basis molar masses.
        # This must not require CoolProp or REFPROP availability.
        self.to_mole_fractions()

    def normalized_components(self) -> dict[str, float]:
        """Return normalized composition in the original basis.

        Component names are normalized to preferred CoolProp names.
        """
        components = canonicalize_component_names(self.components)

        if self.normalize:
            return _normalize_fractions(components)

        _validate_fraction_mapping(components)
        total = sum(components.values())

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Gas-mixture fractions must sum to 1.0 when normalize=False; "
                f"got {total:.12g}."
            )

        return dict(components)

    def to_mole_fractions(self) -> dict[str, float]:
        """Return normalized mole fractions for CoolProp mixture usage."""
        basis = self.basis.lower().strip()
        components = self.normalized_components()

        if basis == "mole":
            return normalize_mole_fractions(components)

        if basis == "volume":
            # For gas mixtures, volume fractions are equivalent to mole
            # fractions under the usual ideal-gas interpretation.
            return normalize_mole_fractions(components)

        if basis == "mass":
            return mass_fractions_to_mole_fractions(
                components,
                molar_masses=self.molar_masses,
            )

        raise ValueError(f"Unsupported gas-mixture basis: {self.basis!r}.")

    def backend_availability(self) -> PropertyBackendAvailability:
        """Return runtime availability status of the selected CoolProp backend."""
        return check_coolprop_backend_availability(self.backend)


@dataclass(frozen=True)
class GasMixturePropertyProvider:
    """Property provider for a gas mixture.

    The provider converts the supplied `GasMixtureSpec` to mole fractions and
    delegates property evaluation to `CoolPropGasMixtureProvider`.

    The same provider can be used to supply gas or gas-mixture properties for:
    - internal tube-side heat transfer,
    - internal tube-side pressure drop,
    - outside crossflow heat transfer / pressure drop.
    """

    spec: GasMixtureSpec

    @property
    def mole_fractions(self) -> dict[str, float]:
        """Return normalized mole fractions used by CoolProp."""
        return self.spec.to_mole_fractions()

    @property
    def fluid(self) -> str:
        """Return CoolProp mixture fluid string."""
        return build_coolprop_mixture_string(
            self.mole_fractions,
            backend=self.spec.backend,
            normalize=True,
        )

    @property
    def backend_availability(self) -> PropertyBackendAvailability:
        """Return runtime availability status of the selected CoolProp backend."""
        return check_coolprop_backend_availability(self.spec.backend)

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return transport properties at T [K] and p [Pa]."""
        require_coolprop_backend(self.spec.backend)
        return self._coolprop_provider().at(T=T, p=p)

    def full_at(self, T: float, p: float) -> CoolPropProperties:
        """Return transport and thermodynamic properties at T [K] and p [Pa]."""
        require_coolprop_backend(self.spec.backend)
        return self._coolprop_provider().full_at(T=T, p=p)

    def _coolprop_provider(self) -> CoolPropGasMixtureProvider:
        return CoolPropGasMixtureProvider(
            components=self.mole_fractions,
            backend=self.spec.backend,
            normalize=True,
            imposed_phase=self.spec.imposed_phase,
        )


def gas_mixture_props(
    T: float,
    p: float,
    spec: GasMixtureSpec,
) -> FluidTransportProperties:
    """Return gas-mixture transport properties.

    Args:
        T: Temperature [K].
        p: Pressure [Pa].
        spec: Gas-mixture composition specification.

    Returns:
        FluidTransportProperties.
    """
    return GasMixturePropertyProvider(spec).at(T=T, p=p)


def gas_mixture_full_props(
    T: float,
    p: float,
    spec: GasMixtureSpec,
) -> CoolPropProperties:
    """Return gas-mixture transport and thermodynamic properties.

    Args:
        T: Temperature [K].
        p: Pressure [Pa].
        spec: Gas-mixture composition specification.

    Returns:
        CoolPropProperties.
    """
    return GasMixturePropertyProvider(spec).full_at(T=T, p=p)


def canonicalize_component_names(
    components: Mapping[str, float],
) -> dict[str, float]:
    """Return component mapping with common aliases converted to CoolProp names.

    If aliases and preferred names are mixed, their fractions are summed.
    Example:
        {"CO2": 0.1, "CarbonDioxide": 0.2} -> {"CarbonDioxide": 0.3}
    """
    _validate_fraction_mapping(components)

    result: dict[str, float] = {}

    for component, fraction in components.items():
        canonical_name = COMPONENT_ALIASES.get(component, component)
        result[canonical_name] = result.get(canonical_name, 0.0) + fraction

    return result


def mass_fractions_to_mole_fractions(
    mass_fractions: Mapping[str, float],
    *,
    molar_masses: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Convert mass fractions to normalized mole fractions.

    Args:
        mass_fractions:
            Mapping of component name to mass fraction [-].
        molar_masses:
            Optional molar-mass overrides [kg/mol].

    Returns:
        Normalized mole fractions [-].

    Notes:
        The returned component names are preferred CoolProp names.
    """
    components = canonicalize_component_names(mass_fractions)

    if not components:
        raise ValueError("At least one gas-mixture component is required.")

    normalized_mass = _normalize_fractions(components)

    mole_amounts: dict[str, float] = {}

    for component, mass_fraction in normalized_mass.items():
        molar_mass = component_molar_mass(
            component,
            molar_masses=molar_masses,
        )
        mole_amounts[component] = mass_fraction / molar_mass

    return _normalize_fractions(mole_amounts)



def mole_fractions_to_mass_fractions(
    mole_fractions: Mapping[str, float],
    *,
    molar_masses: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Convert mole fractions to normalized mass fractions.

    Args:
        mole_fractions:
            Mapping of component name to mole fraction [-].
        molar_masses:
            Optional molar-mass overrides [kg/mol].

    Returns:
        Normalized mass fractions [-].

    Notes:
        Component names are returned as preferred CoolProp names.
    """
    components = canonicalize_component_names(mole_fractions)
    normalized_mole = normalize_mole_fractions(components)

    mass_amounts: dict[str, float] = {}

    for component, mole_fraction in normalized_mole.items():
        molar_mass = component_molar_mass(
            component,
            molar_masses=molar_masses,
        )
        mass_amounts[component] = mole_fraction * molar_mass

    return _normalize_fractions(mass_amounts)


def gas_mixture_mass_fractions_from_dry_composition_and_water_ratio(
    dry_components: Mapping[str, float],
    *,
    water_ratio: float,
    dry_basis: str = "mole",
    water_component: str = "H2O",
    normalize: bool = True,
    molar_masses: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return wet-gas mass fractions from dry gas and water ratio.

    This helper is intended for gas-phase property calculations where water
    vapor is included explicitly as a component of a gas mixture.

    Args:
        dry_components:
            Dry-gas composition excluding water.
        water_ratio:
            Water mass per dry-gas mass [kg_water / kg_dry_gas].
            Example: 120 g/kg dry gas -> 0.120.
        dry_basis:
            Basis of `dry_components`:
            - "mole": dry-gas mole fractions
            - "volume": dry-gas volume fractions, treated as mole fractions
            - "mass": dry-gas mass fractions
        water_component:
            Water component name or alias. Defaults to "H2O".
        normalize:
            If True, normalize dry-gas fractions before conversion.
        molar_masses:
            Optional molar-mass override mapping [kg/mol].

    Returns:
        Wet-gas component mass fractions [-] including water.

    Notes:
        The returned composition is on a mass basis. Use it with:
            GasMixtureSpec(..., basis="mass")

        This helper does not model condensation. It only builds a gas-phase
        mixture composition with water vapor as an explicit component.
    """
    _validate_water_ratio(water_ratio)
    _validate_basis(dry_basis)
    _validate_fraction_mapping(dry_components)

    dry_components_canonical = canonicalize_component_names(dry_components)

    if water_component in dry_components_canonical:
        raise ValueError(
            f"Dry-gas composition already contains water component "
            f"{water_component!r}. Provide dry gas without water."
        )

    water_canonical = COMPONENT_ALIASES.get(water_component, water_component)
    if water_canonical in dry_components_canonical:
        raise ValueError(
            f"Dry-gas composition already contains water component "
            f"{water_canonical!r}. Provide dry gas without water."
        )

    if normalize:
        dry_components_canonical = _normalize_fractions(dry_components_canonical)
    else:
        total = sum(dry_components_canonical.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Dry-gas fractions must sum to 1.0 when normalize=False; "
                f"got {total:.12g}."
            )

    dry_basis_normalized = dry_basis.lower().strip()

    if dry_basis_normalized in {"mole", "volume"}:
        dry_mass_fractions = mole_fractions_to_mass_fractions(
            dry_components_canonical,
            molar_masses=molar_masses,
        )
    elif dry_basis_normalized == "mass":
        dry_mass_fractions = _normalize_fractions(dry_components_canonical)
    else:
        raise ValueError(f"Unsupported dry-gas basis: {dry_basis!r}.")

    # Basis: 1 kg dry gas + water_ratio kg water vapor.
    total_wet_mass = 1.0 + water_ratio

    wet_mass_fractions = {
        component: dry_mass_fraction / total_wet_mass
        for component, dry_mass_fraction in dry_mass_fractions.items()
    }
    wet_mass_fractions[water_canonical] = water_ratio / total_wet_mass

    return _normalize_fractions(wet_mass_fractions)


def gas_mixture_from_dry_composition_and_water_ratio(
    dry_components: Mapping[str, float],
    *,
    water_ratio: float,
    dry_basis: str = "mole",
    backend: str = "HEOS",
    normalize: bool = True,
    molar_masses: Mapping[str, float] | None = None,
    imposed_phase: str | None = "gas",
    water_component: str = "H2O",
) -> GasMixtureSpec:
    """Create a gas-phase GasMixtureSpec from dry gas and water ratio.

    Args:
        dry_components:
            Dry-gas composition excluding water.
        water_ratio:
            Water mass per dry-gas mass [kg_water / kg_dry_gas].
            Example: 120 g/kg dry gas -> 0.120.
        dry_basis:
            Basis of `dry_components`: "mole", "volume", or "mass".
        backend:
            CoolProp backend, for example "HEOS" or "REFPROP".
        normalize:
            If True, normalize dry-gas fractions before conversion.
        molar_masses:
            Optional molar-mass override mapping [kg/mol].
        imposed_phase:
            Optional CoolProp phase hint. Defaults to "gas".
        water_component:
            Water component name or alias. Defaults to "H2O".

    Returns:
        GasMixtureSpec with `basis="mass"`.

    Important:
        This helper does not create a humid-air or wet-process model. It returns
        a standard GasMixtureSpec with water vapor as a gas-phase component.
        Condensation, latent heat, wet-surface heat transfer, acid dew point, and
        composition changes due to water removal are not modelled.
    """
    wet_mass_fractions = gas_mixture_mass_fractions_from_dry_composition_and_water_ratio(
        dry_components,
        water_ratio=water_ratio,
        dry_basis=dry_basis,
        water_component=water_component,
        normalize=normalize,
        molar_masses=molar_masses,
    )

    return GasMixtureSpec(
        components=wet_mass_fractions,
        basis="mass",
        backend=backend,
        normalize=True,
        molar_masses=molar_masses,
        imposed_phase=imposed_phase,
    )


def component_molar_mass(
    component: str,
    *,
    molar_masses: Mapping[str, float] | None = None,
) -> float:
    """Return component molar mass [kg/mol].

    Args:
        component:
            Component name or supported alias.
        molar_masses:
            Optional override mapping [kg/mol].

    Returns:
        Molar mass [kg/mol].
    """
    if not isinstance(component, str) or not component.strip():
        raise ValueError("Component name must be a non-empty string.")

    canonical_name = COMPONENT_ALIASES.get(component, component)

    if molar_masses is not None:
        if component in molar_masses:
            return _validated_molar_mass(component, molar_masses[component])
        if canonical_name in molar_masses:
            return _validated_molar_mass(canonical_name, molar_masses[canonical_name])

    if canonical_name in COMMON_MOLAR_MASSES:
        return COMMON_MOLAR_MASSES[canonical_name]

    raise ValueError(
        f"No molar mass is available for component {component!r}. "
        "Provide it through GasMixtureSpec(..., molar_masses={...}) "
        "when using mass-basis composition."
    )


def _normalize_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    """Normalize a non-empty fraction mapping to sum to 1.0."""
    _validate_fraction_mapping(fractions)

    total = sum(fractions.values())

    if total <= 0.0:
        raise ValueError("Sum of fractions must be positive.")

    return {component: value / total for component, value in fractions.items()}



def _validate_water_ratio(water_ratio: float) -> None:
    if not isinstance(water_ratio, (int, float)):
        raise TypeError(
            f"Water ratio must be numeric [kg_water/kg_dry_gas], "
            f"got {type(water_ratio).__name__}."
        )
    if not math.isfinite(water_ratio):
        raise ValueError(f"Water ratio must be finite, got {water_ratio!r}.")
    if water_ratio < 0.0:
        raise ValueError(f"Water ratio must be non-negative, got {water_ratio!r}.")


def _validate_fraction_mapping(fractions: Mapping[str, float]) -> None:
    if not fractions:
        raise ValueError("At least one gas-mixture component is required.")

    for component, value in fractions.items():
        if not isinstance(component, str) or not component.strip():
            raise ValueError("Component names must be non-empty strings.")
        if not math.isfinite(value):
            raise ValueError(f"Fraction for {component!r} must be finite.")
        if value < 0.0:
            raise ValueError(f"Fraction for {component!r} must be non-negative.")

    if sum(fractions.values()) <= 0.0:
        raise ValueError("Sum of fractions must be positive.")


def _validate_molar_masses(molar_masses: Mapping[str, float]) -> None:
    if not molar_masses:
        raise ValueError("Molar-mass override mapping must not be empty.")

    for component, molar_mass in molar_masses.items():
        _validated_molar_mass(component, molar_mass)


def _validated_molar_mass(component: str, molar_mass: float) -> float:
    if not isinstance(component, str) or not component.strip():
        raise ValueError("Molar-mass component names must be non-empty strings.")
    if not math.isfinite(molar_mass):
        raise ValueError(f"Molar mass for {component!r} must be finite [kg/mol].")
    if molar_mass <= 0.0:
        raise ValueError(f"Molar mass for {component!r} must be positive [kg/mol].")

    return molar_mass


def _validate_basis(basis: str) -> None:
    if not isinstance(basis, str):
        raise ValueError("Gas-mixture basis must be a string.")

    if basis.lower().strip() not in SUPPORTED_BASES:
        raise ValueError(
            f"Unsupported gas-mixture basis {basis!r}. "
            f"Supported bases are: {sorted(SUPPORTED_BASES)}."
        )


def _validate_backend(backend: str) -> None:
    if not isinstance(backend, str):
        raise ValueError("CoolProp backend must be a string.")
    if not backend.strip():
        raise ValueError("CoolProp backend must not be empty.")


def _validate_imposed_phase_name(imposed_phase: str | None) -> None:
    if imposed_phase is None:
        return

    if not isinstance(imposed_phase, str):
        raise ValueError("CoolProp imposed phase must be a string or None.")

    if not imposed_phase.strip():
        raise ValueError("CoolProp imposed phase must not be empty.")
