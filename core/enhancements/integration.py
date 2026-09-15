# KalKalori - GNU GPL v3 only
"""Adapters between enhancement contracts and existing 0D solver states."""
from dataclasses import replace

from core.common.warnings import make_warning
from .base import (
    EnhancementInput, EnhancementState, EnhancementUnsupportedError,
    evaluate_enhancement,
)


def thermal_property_reference(configuration):
    """Optional source-neutral capability; legacy providers use bulk."""
    provider = configuration.provider if configuration else None
    if configuration and configuration.clearance_provider is not None:
        from .clearance import ClearanceModelMode
        if configuration.clearance_provider.mode is ClearanceModelMode.ABSOLUTE:
            provider = configuration.clearance_provider
    reference = getattr(provider, "thermal_property_reference", "bulk")
    if reference not in ("bulk", "wall", "film"):
        raise ValueError("Unknown enhancement thermal property reference.")
    return reference


def hydraulic_property_reference(configuration):
    """Optional source-neutral hydraulic property reference; default bulk."""
    provider = configuration.provider if configuration else None
    if configuration and configuration.clearance_provider is not None:
        from .clearance import ClearanceModelMode
        if configuration.clearance_provider.mode is ClearanceModelMode.ABSOLUTE:
            provider = configuration.clearance_provider
    reference = getattr(provider, "hydraulic_property_reference", "bulk")
    if reference not in ("bulk", "wall", "film"):
        raise ValueError("Unknown enhancement hydraulic property reference.")
    return reference


def guard_phase(configuration, provider, temperature, pressure):
    """Use authoritative phase data where available; never infer from rho/cp.

    Transport-only providers rely on the explicit configuration declaration.
    Wet-gas phase-change models are excluded even if their inlet is dry.
    """
    if configuration is None or provider is None:
        return
    from core.phase_change.capability import (
        _PureWaterSinglePhaseGuardProvider, detect_phase_change_capability,
    )
    from core.properties.water import IAPWS97WaterSteamProvider
    from core.properties.coolprop_backend import CoolPropFluidProvider, CoolPropGasMixtureProvider
    from core.properties.dry_air import DryAirPropertyProvider
    from core.properties.gas_mixture import GasMixturePropertyProvider
    from core.properties.tabulated_liquid import TabulatedLiquidProvider

    if isinstance(provider, _PureWaterSinglePhaseGuardProvider):
        provider = provider.provider
    phase = None
    if isinstance(provider, (IAPWS97WaterSteamProvider, CoolPropFluidProvider, CoolPropGasMixtureProvider)):
        if temperature is None or pressure is None:
            raise EnhancementUnsupportedError("enhancement_phase_state_required")
        if isinstance(provider, IAPWS97WaterSteamProvider):
            phase = provider.state(T=temperature, p=pressure).phase.value
        else:
            phase = provider.full_at(T=temperature, p=pressure).phase
        phase = {"subcooled_liquid": "liquid", "superheated_vapor": "gas"}.get(phase, phase)
    elif isinstance(provider, TabulatedLiquidProvider):
        phase = "liquid"
    elif isinstance(provider, (DryAirPropertyProvider, GasMixturePropertyProvider)):
        if detect_phase_change_capability(provider).capable:
            raise EnhancementUnsupportedError("enhancement_wet_gas_unsupported")
        phase = "gas"
    if phase is not None and phase != configuration.fluid_phase:
        raise EnhancementUnsupportedError(f"enhancement_phase_unsupported: authoritative phase {phase!r}.")


def guard_side(configuration, side):
    if configuration is None:
        return
    # Quality endpoints are saturation states, not ordinary single-phase input.
    for name in ("quality_in", "quality_out"):
        if getattr(side, name, None) is not None:
            raise EnhancementUnsupportedError("enhancement_phase_change_unsupported: quality specified.")
    for name in ("T_in", "T_out"):
        temperature = getattr(side, name, None)
        if temperature is not None:
            guard_phase(configuration, side.provider, temperature, side.p)


def evaluate_for_bundle(configuration, bundle, mass_flow, props, *,
                        temperature=None, pressure=None, wall_props=None,
                        wall_temperature=None, property_provider=None, position="thermal",
                        heat_flow_direction="unknown"):
    if configuration is None:
        return None
    guard_phase(configuration, property_provider, temperature, pressure)
    if wall_temperature is not None:
        guard_phase(configuration, property_provider, wall_temperature, pressure)
    thermal = None
    reference = thermal_property_reference(configuration)
    if position == "thermal" and reference != "bulk":
        if wall_temperature is None or temperature is None or property_provider is None or pressure is None:
            raise EnhancementUnsupportedError("enhancement_thermal_reference_state_required: wall/bulk temperatures and property backend required.")
        thermal_temperature = (temperature + wall_temperature)/2 if reference == "film" else wall_temperature
        guard_phase(configuration, property_provider, thermal_temperature, pressure)
        thermal = EnhancementState.from_properties(
            property_provider.at(T=thermal_temperature, p=pressure), thermal_temperature, pressure)
    hydraulic = None
    hydraulic_reference = hydraulic_property_reference(configuration)
    if hydraulic_reference != "bulk":
        if wall_temperature is None or temperature is None or property_provider is None or pressure is None:
            raise EnhancementUnsupportedError("enhancement_hydraulic_reference_state_required: wall/bulk temperatures and property backend required.")
        hydraulic_temperature = ((temperature + wall_temperature)/2
                                 if hydraulic_reference == "film" else wall_temperature)
        guard_phase(configuration, property_provider, hydraulic_temperature, pressure)
        if (thermal is not None and thermal.temperature == hydraulic_temperature):
            hydraulic = thermal
        else:
            hydraulic = EnhancementState.from_properties(
                property_provider.at(T=hydraulic_temperature, p=pressure),
                hydraulic_temperature, pressure)
    state = EnhancementInput(
        mass_flow_per_tube=mass_flow/bundle.n_tubes_per_pass_effective,
        tube_inner_diameter=bundle.internal_hydraulic_diameter,
        heated_length=bundle.tube.length_effective,
        tube_length=bundle.tube.length_total,
        bulk=EnhancementState.from_properties(props, temperature, pressure),
        wall=None if wall_props is None else EnhancementState.from_properties(
            wall_props, wall_temperature, pressure),
        fluid_phase=configuration.fluid_phase, position=position,
        heat_flow_direction=heat_flow_direction,
        roughness_inner=getattr(bundle.tube, "roughness_inner", None) or 0.0,
        base_flow_area_per_tube=(bundle.internal_flow_area_per_pass
                                / bundle.n_tubes_per_pass_effective),
        hydraulic_length_total=bundle.internal_length_total,
        thermal=thermal,
        hydraulic=hydraulic,
    )
    result = evaluate_enhancement(configuration, state)
    return replace(result, warnings=result.warnings + (make_warning(
        code="enhancement_0d_reference_evaluation",
        message=("Enhancement evaluated at a 0D bulk/wall or hydraulic quadrature state; "
                 "this is not a resolved axial model. Transport-only fluid providers rely "
                 "on the declared single phase. Existing local losses and acceleration "
                 "retain their separate tube references."),
        source="tube_side_enhancement", severity="info",
    ),))


def internal_diagnostics(result, bulk_k):
    from core.heat_transfer.internal_flow import InternalHeatTransferDiagnostics
    r = result.reference
    nu = result.nusselt
    if nu is None:
        nu = result.alpha_inside*r.nusselt_length/bulk_k
    correction = result.wall_correction
    return InternalHeatTransferDiagnostics(
        v=r.velocity, Re=r.reynolds, Pr=r.prandtl,
        Nu_base=nu/correction, Nu_corrected=nu,
        length_correction=1.0, wall_temperature_correction=correction,
        combined_correction=correction, alfa_base=result.alpha_inside/correction,
        alfa_corrected=result.alpha_inside, warnings=list(result.warnings), enhancement=result,
    )


def hydraulic_evaluator(configuration, bundle, property_provider, *, wall_temperature=None,
                        heat_flow_direction="unknown"):
    """Build the same dispatcher for snapshots and refreshed hydraulic paths."""
    if configuration is None:
        return None

    def evaluate(point):
        return evaluate_for_bundle(
            configuration, bundle, point.mass_flux*bundle.internal_flow_area_per_pass,
            point.props, temperature=point.temperature, pressure=point.pressure,
            property_provider=property_provider, position=point.position,
            wall_temperature=wall_temperature,
            heat_flow_direction=heat_flow_direction,
        )
    return evaluate


def check_model_identity(*results):
    """Reject mixed thermal/hydraulic models, including a silent smooth result."""
    if all(result is None for result in results):
        return
    if any(result is None for result in results):
        raise ValueError("Enhancement result missing from a coupled solver path.")
    identities = {(r.provider_id, r.correlation_id, r.source_references,
                   r.source_access_basis) for r in results}
    if len(identities) != 1:
        raise ValueError("Thermal and hydraulic enhancement models are inconsistent.")
