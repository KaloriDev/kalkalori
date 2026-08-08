# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Rebuild a wet-gas ``GasMixtureSpec`` at a different H2O content.

Given a fixed dry-gas composition (unchanged across condensation, per
``core.phase_change`` scope) and a target water ratio W [kg H2O/kg dry
gas], build a fresh ``GasMixtureSpec``/``GasMixturePropertyProvider`` for
property evaluation at that state. Reuses the existing
``gas_mixture_from_dry_composition_and_water_ratio`` helper (v0.5.x) so the
mass-fraction bookkeeping is not duplicated here.
"""

from __future__ import annotations

from core.properties.gas_mixture import (
    GasMixturePropertyProvider,
    GasMixtureSpec,
    gas_mixture_from_dry_composition_and_water_ratio,
)
from core.phase_change.types import PhaseChangeCapability


def wet_gas_spec_at_water_ratio(
    capability: PhaseChangeCapability,
    W: float,
    *,
    imposed_phase: str | None = "gas",
) -> GasMixtureSpec:
    """Return a ``GasMixtureSpec`` for the capability's dry composition at
    water ratio ``W`` [kg H2O / kg dry gas].

    The dry-composition mole fractions from ``capability`` are held fixed
    (mole-basis input to
    ``gas_mixture_from_dry_composition_and_water_ratio``); only the water
    content changes.
    """
    if not capability.capable or capability.dry_mole_fractions is None:
        raise ValueError("wet_gas_spec_at_water_ratio requires a capable PhaseChangeCapability.")

    return gas_mixture_from_dry_composition_and_water_ratio(
        dict(capability.dry_mole_fractions),
        water_ratio=W,
        dry_basis="mole",
        backend=capability.backend,
        molar_masses=capability.molar_masses,
        imposed_phase=imposed_phase,
    )


def wet_gas_provider_at_water_ratio(
    capability: PhaseChangeCapability,
    W: float,
    *,
    imposed_phase: str | None = "gas",
) -> GasMixturePropertyProvider:
    """Return a ``GasMixturePropertyProvider`` at water ratio ``W``.

    Convenience wrapper around ``wet_gas_spec_at_water_ratio`` for the
    common case of needing a full provider (for ``.at``/``.full_at``/
    ``.temperature_from_h_p``) rather than just the spec.
    """
    return GasMixturePropertyProvider(
        wet_gas_spec_at_water_ratio(capability, W, imposed_phase=imposed_phase)
    )


def dry_gas_spec(
    capability: PhaseChangeCapability,
    *,
    imposed_phase: str | None = "gas",
) -> GasMixtureSpec:
    """Return the non-condensable ("dry") part of the mixture as its own
    ``GasMixtureSpec`` (no water component at all).

    Used by ``core.phase_change.wet_gas_enthalpy`` to evaluate the dry-gas
    enthalpy contribution on an ideal-gas (pressure-independent) basis,
    without needing a full wet-mixture CoolProp flash for the enthalpy
    balance (see that module's docstring for why).
    """
    if not capability.capable or capability.dry_mole_fractions is None:
        raise ValueError("dry_gas_spec requires a capable PhaseChangeCapability.")

    return GasMixtureSpec(
        components=dict(capability.dry_mole_fractions),
        basis="mole",
        backend=capability.backend,
        normalize=True,
        molar_masses=capability.molar_masses,
        imposed_phase=imposed_phase,
    )
