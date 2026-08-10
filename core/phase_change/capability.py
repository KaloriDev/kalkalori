# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Central phase-change capability adapter.

This is the *only* place in KalKalori that should ever do
``isinstance(provider, GasMixturePropertyProvider)`` (or similar) to decide
whether a medium can undergo phase change. Solvers (``core.phase_change.
integration``, ``core.models.bare_tube``) call
``detect_phase_change_capability`` and work only with the resulting
``PhaseChangeCapability`` -- they never re-derive composition facts
themselves. This keeps "is this medium phase-change capable" centralized
instead of scattered ``if outside_is_wet: ...`` checks across solvers.

The wet-gas capability path is a
``core.properties.gas_mixture.GasMixturePropertyProvider`` whose spec
contains a positive mole fraction of water ("H2O"/"Water", any of mole,
volume, or mass composition basis -- ``GasMixtureSpec.to_mole_fractions()``
already normalizes all three bases uniformly). There is no separate
"humid-air provider" class in the codebase to special-case: a humid-air
side built via
``core.properties.gas_mixture.gas_mixture_from_dry_composition_and_water_ratio``
*is* a ``GasMixturePropertyProvider``, so it is already covered by this one
path. v0.6.2 additionally recognizes the dedicated IAPWS provider as the
distinct pure-water/steam capability path. Other providers (including
generic pure-fluid CoolProp, constant-property providers and dry gas-mixture
specs with no water) are reported as not capable.

A capable medium is not necessarily *active*: capability is a property of
the medium/spec alone, independent of the current thermal operating point.
See ``core.phase_change.types`` for the capable/possible/active distinction
and ``core.phase_change.regime`` for how "possible" is decided.
"""

from __future__ import annotations

from core.properties.gas_mixture import GasMixturePropertyProvider, component_molar_mass
from core.phase_change.types import PhaseChangeCapability

# Canonical CoolProp component name for the only condensable species
# recognized in v0.6.1 (see core.properties.gas_mixture.COMPONENT_ALIASES).
CONDENSABLE_COMPONENT_CANONICAL = "Water"
CONDENSABLE_COMPONENT_LABEL = "H2O"


def detect_phase_change_capability(provider: object) -> PhaseChangeCapability:
    """Return the phase-change capability of a property provider.

    Args:
        provider: Any object exposing the provider protocol used elsewhere
            in KalKalori (``.at(T, p)`` / ``.full_at(T, p)``). Only
            ``GasMixturePropertyProvider`` instances can currently be
            capable; every other provider type returns
            ``PhaseChangeCapability(capable=False)``.

    Returns:
        PhaseChangeCapability describing whether/how this provider can
            undergo wet-gas phase change in the v0.6.1 model.
    """
    from core.properties.water import IAPWS97WaterSteamProvider

    if isinstance(provider, IAPWS97WaterSteamProvider):
        return PhaseChangeCapability(
            capable=True,
            component=CONDENSABLE_COMPONENT_LABEL,
            provider_kind="pure_water_steam",
        )
    if isinstance(provider, GasMixturePropertyProvider):
        return _detect_gas_mixture_capability(provider)
    return PhaseChangeCapability(capable=False)


def is_pure_water_provider(provider: object) -> bool:
    """Return whether ``provider`` unambiguously represents pure H2O.

    Pure water/steam deliberately does not become a wet-gas capability: the
    dry-carrier ``W`` basis is undefined and its condensation model belongs
    to v0.6.2.
    """
    from core.properties.coolprop_backend import CoolPropFluidProvider
    from core.properties.water import IAPWS97WaterSteamProvider

    if isinstance(provider, IAPWS97WaterSteamProvider):
        return True
    if isinstance(provider, CoolPropFluidProvider):
        fluid = provider.fluid.split("::")[-1].strip().lower()
        return fluid in {"water", "h2o"}
    if isinstance(provider, GasMixturePropertyProvider):
        fractions = provider.spec.to_mole_fractions()
        return fractions.get(CONDENSABLE_COMPONENT_CANONICAL, 0.0) >= 1.0 - 1e-12
    return False


def _detect_gas_mixture_capability(
    provider: GasMixturePropertyProvider,
) -> PhaseChangeCapability:
    spec = provider.spec
    mole_fractions = spec.to_mole_fractions()

    y_h2o = mole_fractions.get(CONDENSABLE_COMPONENT_CANONICAL, 0.0)
    if y_h2o <= 0.0:
        return PhaseChangeCapability(capable=False)

    dry_raw = {
        name: fraction
        for name, fraction in mole_fractions.items()
        if name != CONDENSABLE_COMPONENT_CANONICAL
    }
    total_dry = sum(dry_raw.values())
    if total_dry <= 0.0:
        # A pure water-vapor stream has no non-condensable carrier gas; the
        # W = kg vapor / kg dry carrier basis used throughout this package
        # is undefined; the dedicated v0.6.2 water/steam adapter handles it.
        return PhaseChangeCapability(capable=False)

    dry_mole_fractions = {name: fraction / total_dry for name, fraction in dry_raw.items()}
    M_dry = sum(
        fraction * component_molar_mass(name, molar_masses=spec.molar_masses)
        for name, fraction in dry_mole_fractions.items()
    )
    M_h2o = component_molar_mass(
        CONDENSABLE_COMPONENT_CANONICAL, molar_masses=spec.molar_masses
    )

    # Ideal-gas mixing: W = kg vapor / kg dry gas = (y/(1-y)) * (M_h2o/M_dry).
    W_in = (y_h2o / (1.0 - y_h2o)) * (M_h2o / M_dry)

    return PhaseChangeCapability(
        capable=True,
        component=CONDENSABLE_COMPONENT_LABEL,
        provider_kind="gas_mixture",
        dry_mole_fractions=dry_mole_fractions,
        M_dry=M_dry,
        M_condensable=M_h2o,
        W_in=W_in,
        backend=spec.backend,
        molar_masses=spec.molar_masses,
    )
