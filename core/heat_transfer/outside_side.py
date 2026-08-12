# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Reusable outside-side evaluation without constructing a full exchanger solve."""

from __future__ import annotations

from dataclasses import dataclass

from core.common.warnings import ModelWarning
from core.heat_transfer.outside_flow import (
    OutsideTubeBankHydraulicResult,
    calculate_outside_tube_bank_hydraulics,
    outside_flow_from_mass_flow,
)
from core.pressure_drop.flow_path import (
    PressureDropPathResult,
    build_outside_pressure_drop_result,
)
from core.properties.common import FluidTransportProperties


@dataclass(frozen=True)
class OutsideSideEvaluation:
    """Thermal and hydraulic diagnostics for one outside temperature program."""

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
    wall_property_correction: float
    correction_status: str
    hydraulics: OutsideTubeBankHydraulicResult
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
) -> OutsideSideEvaluation:
    """Evaluate outside properties, HTC and three-state bare-bank hydraulics.

    No tube-side stream, fake property provider, NTU solve or full exchanger
    result is created. ``properties_mean`` may be supplied by a caller-owned
    cache; identical trial states therefore do not trigger a second property
    evaluation. The current steam model applies no outside wall-property
    multiplier, so corrected and base HTC diagnostics are intentionally equal.
    """
    T_mean = 0.5 * (T_in + T_out)
    props = properties_mean or _transport(provider.at(T=T_mean, p=p))
    tube = hx.bundle.tube
    velocity, reynolds, prandtl, alpha, _legacy_dp, ht_warnings, _ = (
        outside_flow_from_mass_flow(
            m_dot=mass_flow,
            frontal_area=hx.bundle.frontal_flow_area,
            tube_outer_diameter=float(tube.D_o),
            tube_pitch_transverse=hx.bundle.pitch_transverse,
            tube_pitch_longitudinal=hx.bundle.pitch_longitudinal,
            layout=hx.bundle.layout,
            n_rows=hx.bundle.n_rows,
            n_tubes_per_row=hx.bundle.n_tubes_per_row,
            props=_outside_fluid_props(props),
            euler_provider=euler_provider,
            calculate_pressure_drop=False,
        )
    )
    hydraulics = calculate_outside_tube_bank_hydraulics(
        m_dot=mass_flow,
        face_area=hx.bundle.frontal_flow_area,
        tube_outer_diameter=float(tube.D_o),
        tube_pitch_transverse=hx.bundle.pitch_transverse,
        tube_pitch_longitudinal=hx.bundle.pitch_longitudinal,
        layout=hx.bundle.layout,
        n_rows=hx.bundle.n_rows,
        n_tubes_per_row=hx.bundle.n_tubes_per_row,
        provider=provider,
        temperature_in=T_in,
        temperature_out=T_out,
        pressure=p,
        euler_provider=euler_provider,
    )
    nusselt = alpha * float(tube.D_o) / props.k
    warnings = _deduplicate((*ht_warnings, *hydraulics.warnings))
    return OutsideSideEvaluation(
        T_in=T_in,
        T_out=T_out,
        T_mean=T_mean,
        properties_mean=props,
        velocity=velocity,
        reynolds=reynolds,
        prandtl=prandtl,
        nusselt_base=nusselt,
        nusselt_corrected=nusselt,
        alpha_base=alpha,
        alpha_corrected=alpha,
        wall_property_correction=1.0,
        correction_status="not_applied_by_steam_zone_model",
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
