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
distinct supported pure-water/steam capability path. CoolProp Water and
pure-H2O ``GasMixturePropertyProvider`` remain authoritative for supported
single-phase calculations, but are explicitly marked phase-change-
unsupported; constant-property providers and dry gas-mixture specs with no
water are reported as not capable.

A capable medium is not necessarily *active*: capability is a property of
the medium/spec alone, independent of the current thermal operating point.
See ``core.phase_change.types`` for the capable/possible/active distinction
and ``core.phase_change.regime`` for how "possible" is decided.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from core.properties.gas_mixture import GasMixturePropertyProvider, component_molar_mass
from core.phase_change import warning_codes as WC
from core.phase_change.types import PhaseChangeCapability

# Canonical CoolProp component name for the only condensable species
# recognized in v0.6.1 (see core.properties.gas_mixture.COMPONENT_ALIASES).
CONDENSABLE_COMPONENT_CANONICAL = "Water"
CONDENSABLE_COMPONENT_LABEL = "H2O"

# W is expressed as kg H2O / kg dry carrier gas.  The absolute term covers
# values close to a dry limit; the relative term covers round-off in
# composition-basis conversion at ordinary humidity ratios.
WET_GAS_INLET_SATURATION_ABSOLUTE_TOLERANCE = 1.0e-10
WET_GAS_INLET_SATURATION_RELATIVE_TOLERANCE = 1.0e-8


class PureWaterPhaseChangeProviderNotSupportedError(RuntimeError):
    """Raised when pure-H2O phase change needs an unsupported provider path."""

    warning_code = WC.PURE_WATER_PHASE_CHANGE_PROVIDER_NOT_SUPPORTED


class LiquidWaterInGasInletNotSupportedError(RuntimeError):
    """Raised when a gas composition implies pre-existing liquid water."""

    warning_code = WC.LIQUID_WATER_IN_GAS_INLET_NOT_SUPPORTED


@dataclass(frozen=True)
class _PureWaterSinglePhaseGuardProvider:
    """Delegate properties while preventing an unsupported phase crossing."""

    provider: object
    saturation_temperature: float
    inlet_is_vapor: bool

    def at(self, T: float, p: float):
        tolerance = 1.0e-6
        crossed = (
            T <= self.saturation_temperature + tolerance
            if self.inlet_is_vapor
            else T >= self.saturation_temperature - tolerance
        )
        if crossed:
            _raise_unsupported_provider_crossing(self.provider)
        return self.provider.at(T=T, p=p)


def detect_phase_change_capability(provider: object) -> PhaseChangeCapability:
    """Return the phase-change capability of a property provider.

    Args:
        provider: Any object exposing the provider protocol used elsewhere
            in KalKalori (``.at(T, p)`` / ``.full_at(T, p)``). Supported wet
            gas mixtures and ``IAPWS97WaterSteamProvider`` are capable;
            unsupported provider types return
            ``PhaseChangeCapability(capable=False)``.

    Returns:
        PhaseChangeCapability describing whether/how this provider can
            undergo wet-gas phase change in the v0.6.1 model.
    """
    from core.properties.coolprop_backend import CoolPropFluidProvider
    from core.properties.water import IAPWS97WaterSteamProvider

    if isinstance(provider, IAPWS97WaterSteamProvider):
        return PhaseChangeCapability(
            capable=True,
            component=CONDENSABLE_COMPONENT_LABEL,
            provider_kind="pure_water_steam",
        )
    if isinstance(provider, CoolPropFluidProvider) and is_pure_water_provider(provider):
        return PhaseChangeCapability(
            capable=False,
            component=CONDENSABLE_COMPONENT_LABEL,
            provider_kind="pure_water_coolprop_unsupported",
        )
    if isinstance(provider, GasMixturePropertyProvider):
        return _detect_gas_mixture_capability(provider)
    return PhaseChangeCapability(capable=False)


def reject_liquid_water_in_gas_inlet(
    provider: object,
    *,
    T_in: float | None,
    p: float,
    side: str,
) -> None:
    """Reject an equilibrium gas/liquid inlet before gas-only properties.

    ``GasMixturePropertyProvider`` describes the gas phase only.  For its
    wet-gas capability, an inlet is therefore valid only when its specified
    water ratio does not exceed the equilibrium vapor capacity at ``T_in``
    and ``p``.  Equality (including numerical noise within the documented
    absolute/relative tolerance) is accepted because saturation alone does
    not establish a liquid inventory.

    Temperatures below the water triple point are deliberately left to the
    existing controlled frost/ice path.  At temperatures where saturation
    pressure is at least the total pressure, a liquid phase cannot coexist
    with a carrier-gas mixture in equilibrium, so no liquid-water alarm is
    emitted and ``saturated_water_ratio`` is not called outside its domain.
    """
    capability = detect_phase_change_capability(provider)
    if capability.provider_kind != "gas_mixture" or not capability.capable:
        return
    if T_in is None:
        return

    from core.phase_change.water_equilibrium import saturated_water_ratio
    from core.properties.water import (
        WATER_CRITICAL_TEMPERATURE_K,
        WATER_TRIPLE_POINT_TEMPERATURE_K,
        water_saturation_pressure,
    )

    if T_in < WATER_TRIPLE_POINT_TEMPERATURE_K:
        return
    if T_in >= WATER_CRITICAL_TEMPERATURE_K:
        return
    p_sat = water_saturation_pressure(T_in)
    if p_sat >= p:
        return

    W_in = capability.W_in
    W_sat = saturated_water_ratio(
        p_total=p,
        T=T_in,
        M_dry=capability.M_dry,
        M_h2o=capability.M_condensable,
    )
    tolerance = (
        WET_GAS_INLET_SATURATION_ABSOLUTE_TOLERANCE
        + WET_GAS_INLET_SATURATION_RELATIVE_TOLERANCE
        * max(abs(W_in), abs(W_sat))
    )
    if W_in <= W_sat + tolerance:
        return

    raise LiquidWaterInGasInletNotSupportedError(
        f"The {side} inlet composition implies a gas-liquid water mixture "
        f"(W_in={W_in:.12g} kg/kg dry gas exceeds "
        f"W_sat={W_sat:.12g} kg/kg by more than the numerical tolerance "
        f"{tolerance:.3g}). The current GasMixturePropertyProvider "
        "represents the gas phase only; evaporation of liquid water carried "
        "by a gas is not modelled. No ordinary sensible or condensing "
        "result was returned."
    )


def is_pure_water_provider(provider: object) -> bool:
    """Return whether ``provider`` unambiguously represents pure H2O.

    Pure water/steam deliberately does not become a wet-gas capability: the
    dry-carrier ``W`` basis is undefined and its condensation model belongs
    to v0.6.2.
    """
    return pure_water_provider_kind(provider) is not None


def pure_water_provider_kind(provider: object) -> str | None:
    """Classify an unambiguous pure-H2O provider without changing backend."""
    from core.properties.coolprop_backend import CoolPropFluidProvider
    from core.properties.water import IAPWS97WaterSteamProvider

    if isinstance(provider, IAPWS97WaterSteamProvider):
        return "iapws97"
    if isinstance(provider, CoolPropFluidProvider):
        fluid = provider.fluid.split("::")[-1].strip().lower()
        if fluid in {"water", "h2o"}:
            return "coolprop_water"
    if isinstance(provider, GasMixturePropertyProvider):
        fractions = provider.spec.to_mole_fractions()
        if fractions.get(CONDENSABLE_COMPONENT_CANONICAL, 0.0) >= 1.0 - 1e-12:
            return "gas_mixture_pure_water"
    return None


def pure_water_saturation_temperature(provider: object, *, p: float) -> float | None:
    """Return saturation temperature through the provider's own backend.

    This helper is for phase-boundary guards. It never replaces a CoolProp
    provider with IAPWS and returns ``None`` above the selected backend's
    critical pressure.
    """
    kind = pure_water_provider_kind(provider)
    if kind == "iapws97":
        from core.properties.water import (
            WATER_CRITICAL_PRESSURE_PA,
            water_saturation_temperature,
        )

        if p >= WATER_CRITICAL_PRESSURE_PA:
            return None
        return water_saturation_temperature(p)
    if kind == "coolprop_water":
        return provider.saturation_temperature(p)
    if kind == "gas_mixture_pure_water":
        from core.properties.coolprop_backend import coolprop_saturation_temperature

        return coolprop_saturation_temperature(
            p=p,
            fluid=f"{provider.spec.backend}::Water",
        )
    return None


def guard_pure_water_single_phase_provider(
    provider: object,
    *,
    T_in: float,
    p: float,
) -> object:
    """Wrap unsupported pure-water providers with a phase-boundary guard."""
    kind = pure_water_provider_kind(provider)
    if kind in {None, "iapws97"}:
        return provider
    saturation_temperature = pure_water_saturation_temperature(provider, p=p)
    if saturation_temperature is None:
        return provider
    if math.isclose(T_in, saturation_temperature, rel_tol=0.0, abs_tol=1.0e-6):
        _raise_unsupported_provider_crossing(provider)
    return _PureWaterSinglePhaseGuardProvider(
        provider=provider,
        saturation_temperature=saturation_temperature,
        inlet_is_vapor=T_in > saturation_temperature,
    )


def reject_unsupported_pure_water_phase_crossing(
    provider: object,
    *,
    T_in: float,
    T_out: float,
    p: float,
) -> None:
    """Reject an accepted program that needs unsupported pure-water phase change."""
    kind = pure_water_provider_kind(provider)
    if kind in {None, "iapws97"}:
        return
    saturation_temperature = pure_water_saturation_temperature(provider, p=p)
    if saturation_temperature is None:
        return
    tolerance = 1.0e-6
    inlet_is_vapor = T_in > saturation_temperature + tolerance
    inlet_is_liquid = T_in < saturation_temperature - tolerance
    crosses_bulk = (
        inlet_is_vapor and T_out <= saturation_temperature + tolerance
    ) or (
        inlet_is_liquid and T_out >= saturation_temperature - tolerance
    )
    if not inlet_is_vapor and not inlet_is_liquid:
        crosses_bulk = True
    if crosses_bulk:
        _raise_unsupported_provider_crossing(provider)


def _raise_unsupported_provider_crossing(provider: object) -> None:
    kind = pure_water_provider_kind(provider) or type(provider).__name__
    raise PureWaterPhaseChangeProviderNotSupportedError(
        "Pure-water phase change is supported only by "
        "IAPWS97WaterSteamProvider. The selected provider "
        f"({kind}) remains authoritative for single-phase properties and "
        "was not silently replaced."
    )


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
        return PhaseChangeCapability(
            capable=False,
            component=CONDENSABLE_COMPONENT_LABEL,
            provider_kind="pure_water_gas_mixture_unsupported",
        )

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
