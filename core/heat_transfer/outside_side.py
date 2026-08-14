# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Reusable outside-side evaluation without constructing a full exchanger solve."""

from __future__ import annotations

from dataclasses import dataclass

from core.common.warnings import ModelWarning
from core.heat_transfer.outside_dispatch import (
    DEFAULT_FINNED_DP_PROVIDER,
    DEFAULT_FINNED_HT_PROVIDER,
    OutsideBankHydraulicResult,
    OutsideThermalDispatchResult,
    calculate_resistance_network,
    evaluate_outside_hydraulics,
    evaluate_outside_thermal,
)
from core.pressure_drop.flow_path import (
    PressureDropPathResult,
    build_outside_pressure_drop_result,
)
from core.properties.common import FluidTransportProperties


@dataclass(frozen=True)
class OutsideSideEvaluation:
    """Thermal and hydraulic diagnostics for one dry outside program.

    ``alpha_physical`` is the direct correlation film coefficient.  The
    generic ``alpha_corrected``/``alpha_effective_gross`` value is the
    topology-aware coefficient on the bundle's gross outside area.  They are
    identical for a plain tube and deliberately distinct for a finned tube.
    """

    T_in: float
    T_out: float
    T_mean: float
    properties_mean: FluidTransportProperties
    velocity: float
    reynolds: float
    prandtl: float
    nusselt_base: float
    nusselt_corrected: float
    alpha_base: float
    alpha_corrected: float
    alpha_physical: float
    alpha_effective_gross: float
    wall_property_correction: float
    correction_status: str
    outside_dispatch: OutsideThermalDispatchResult
    hydraulics: OutsideBankHydraulicResult
    pressure_drop: PressureDropPathResult
    warnings: tuple[ModelWarning, ...]


def evaluate_outside_side(
    hx,
    *,
    provider,
    mass_flow: float,
    T_in: float,
    T_out: float,
    p: float,
    euler_provider: str = "zukauskas",
    properties_mean: FluidTransportProperties | None = None,
    finned_heat_transfer_provider: object = DEFAULT_FINNED_HT_PROVIDER,
    finned_pressure_drop_provider: object = DEFAULT_FINNED_DP_PROVIDER,
) -> OutsideSideEvaluation:
    """Evaluate dry outside properties, HTC and three-state hydraulics.

    No tube-side stream, fake property provider, NTU solve or full exchanger
    result is created. ``properties_mean`` may be supplied by a caller-owned
    cache; identical trial states therefore do not trigger a second property
    evaluation. Surface-specific correlation and hydraulic families are
    selected only by the central dispatchers.  The reference inside
    coefficient below is used solely to materialize the outside-only
    effective coefficient; each phase zone builds its own full resistance
    network with its actual inside coefficient.
    """
    T_mean = 0.5 * (T_in + T_out)
    props = properties_mean or _transport(provider.at(T=T_mean, p=p))
    thermal = evaluate_outside_thermal(
        bundle=hx.bundle,
        m_dot=mass_flow,
        props=_outside_fluid_props(props),
        euler_provider=euler_provider,
        finned_heat_transfer_provider=finned_heat_transfer_provider,
    )
    hydraulics = evaluate_outside_hydraulics(
        bundle=hx.bundle,
        m_dot=mass_flow,
        property_provider=provider,
        temperature_in=T_in,
        temperature_out=T_out,
        pressure=p,
        euler_provider=euler_provider,
        finned_pressure_drop_provider=finned_pressure_drop_provider,
    )
    # The effective outside coefficient depends only on the physical film
    # coefficient and the declared fin topology.  The reference inside film
    # and zero core-wall resistance do not influence this outside-only term.
    reference_network = calculate_resistance_network(
        bundle=hx.bundle,
        alpha_inside=1.0,
        outside_alpha_physical=thermal.alpha_physical,
        resistance_core_wall=0.0,
    )
    warnings = _deduplicate((*thermal.warnings, *hydraulics.warnings))
    return OutsideSideEvaluation(
        T_in=T_in,
        T_out=T_out,
        T_mean=T_mean,
        properties_mean=props,
        velocity=thermal.face_velocity,
        reynolds=thermal.reynolds_number,
        prandtl=thermal.prandtl_number,
        nusselt_base=thermal.nusselt_number_base,
        nusselt_corrected=thermal.nusselt_number,
        alpha_base=thermal.alpha_physical,
        alpha_corrected=reference_network.outside_alpha_effective_gross,
        alpha_physical=thermal.alpha_physical,
        alpha_effective_gross=reference_network.outside_alpha_effective_gross,
        wall_property_correction=thermal.wall_property_correction,
        correction_status="surface_dispatch_dry_film",
        outside_dispatch=thermal,
        hydraulics=hydraulics,
        pressure_drop=build_outside_pressure_drop_result(hydraulics),
        warnings=warnings,
    )


def _transport(value) -> FluidTransportProperties:
    transport = getattr(value, "transport", value)
    if isinstance(transport, FluidTransportProperties):
        return transport
    return FluidTransportProperties(
        rho=float(transport.rho),
        mu=float(transport.mu),
        k=float(transport.k),
        cp=float(transport.cp),
    )


def _outside_fluid_props(props: FluidTransportProperties):
    # Local import avoids making this neutral helper depend on model result types.
    from core.properties.adapters import to_outside_fluid_props

    return to_outside_fluid_props(props)


def _deduplicate(warnings) -> tuple[ModelWarning, ...]:
    result = []
    seen = set()
    for warning in warnings:
        key = (warning.code, warning.message)
        if key not in seen:
            seen.add(key)
            result.append(warning)
    return tuple(result)
