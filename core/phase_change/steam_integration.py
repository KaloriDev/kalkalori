# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Public Simulation/Rating adapters for the shared steam-heater solver."""

from __future__ import annotations

from dataclasses import replace
import math

from core.common.warnings import ModelWarning, make_warning
from core.heat_transfer.outside_side import OutsideSideEvaluation, evaluate_outside_side
from core.heat_transfer.thermal_iteration import (
    IterativeThermalState,
    ThermalIterationDiagnostics,
    WallTemperatureEnvelope,
    WallTemperatureProbe,
)
from core.models.bare_tube import (
    HXOutSideHydraulicResults,
    HXOutSideThermalResults,
    HXOutsidePressureDropResults,
    HXResult,
    HXTubeSideHydraulicResults,
    HXTubeSidePressureDropResults,
)
from core.models.heat_balance import ClosedBalance, ClosedBalanceSide
from core.models.rating import HXRatingResult
from core.models.simulation import HXSideInput, HXSimulationResult
from core.pressure_drop.flow_path import build_tube_side_pressure_drop_result
from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
from core.phase_change import warning_codes as WC
from core.phase_change.capability import (
    detect_phase_change_capability,
    is_pure_water_provider,
)
from core.phase_change.integration import (
    MultiplePhaseChangeSidesError,
    PhaseChangeSettings,
    capability_only_result,
    evaluate_side_onset,
    apply_phase_change,
)
from core.phase_change.steam_heater import (
    SteamHeaterSolution,
    SteamHeaterZoneKind,
    rate_steam_heater,
    solve_steam_heater,
)
from core.phase_change.types import (
    PhaseChangeDirection,
    PhaseChangeMode,
    WaterSteamPhaseChangeResult,
)
from core.properties.water import WaterSteamPhase, water_steam_props_iapws97


class PhaseChangeDisabledButRequiredError(RuntimeError):
    warning_code = WC.PHASE_CHANGE_DISABLED_BUT_REQUIRED


class PureSteamOutsideNotSupportedError(RuntimeError):
    warning_code = WC.PURE_STEAM_OUTSIDE_NOT_SUPPORTED


def translate_saturation_crossing_error(inside, exc: ValueError) -> None:
    """Translate the dry solver's saturation ambiguity only for boiling."""
    state = getattr(inside, "water_steam_state", None)
    if (
        is_pure_water_provider(inside.provider)
        and state is not None
        and state.phase is WaterSteamPhase.SUBCOOLED_LIQUID
        and "T+p lies on the water saturation line" in str(exc)
    ):
        from core.phase_change.steam_heater import SteamEvaporationNotSupportedError

        raise SteamEvaporationNotSupportedError(
            "The sensible-only trial reached the water saturation boundary "
            "in the heating direction; boiling/evaporation is unsupported."
        ) from exc
    raise exc


def is_inside_water_steam_case(inside) -> bool:
    state = getattr(inside, "water_steam_state", None)
    return (
        is_pure_water_provider(inside.provider)
        and state is not None
        and state.phase in {
            WaterSteamPhase.SATURATED_LIQUID,
            WaterSteamPhase.TWO_PHASE,
            WaterSteamPhase.SATURATED_VAPOR,
            WaterSteamPhase.SUPERHEATED_VAPOR,
        }
    )


def reject_unsupported_outside_pure_steam(outside) -> None:
    if not is_pure_water_provider(outside.provider):
        return
    state = getattr(outside, "water_steam_state", None)
    if state is None:
        return
    if state.phase is WaterSteamPhase.SUPERCRITICAL_FLUID:
        return
    if state.phase is not WaterSteamPhase.SUBCOOLED_LIQUID:
        raise PureSteamOutsideNotSupportedError(
            "Pure water/steam condensation or two-phase flow outside tubes is "
            "outside the planned KalKalori scope."
        )


def apply_water_steam_simulation(
    hx,
    inside,
    outside,
    *,
    surface_margin: float,
    iterate: bool,
    flow_arrangement: str | None,
    K_inlet: float,
    K_outlet: float,
    K_turn: float,
    euler_provider: str,
    max_iter: int,
    temperature_tolerance_K: float,
    relative_duty_tolerance: float,
    relaxation_factor: float,
    relative_alfa_tolerance: float,
    settings: PhaseChangeSettings,
):
    if not math.isfinite(surface_margin) or surface_margin < 0.0:
        raise ValueError("surface_margin must be a non-negative finite value.")
    if not iterate:
        raise ValueError("Pure-steam multi-zone calculation requires iterate=True.")
    solve_kwargs = dict(
        inlet_state=inside.water_steam_state,
        mass_flow_steam=inside.m_dot,
        outside_provider=outside.provider,
        mass_flow_outside=outside.m_dot,
        T_in_outside=outside.T_in,
        p_outside=outside.p,
        orientation=hx.bundle.tube.tube_orientation,
    )
    full_solution = solve_steam_heater(hx, **solve_kwargs)
    solution = full_solution
    if surface_margin > 0.0:
        solution = solve_steam_heater(
            hx,
            available_area=hx.bundle.total_outer_area / (1.0 + surface_margin),
            **solve_kwargs,
        )
    steam_result = _steam_result(solution, mode=inside.phase_change_mode)
    if inside.phase_change_mode is PhaseChangeMode.DISABLED and steam_result.active:
        raise PhaseChangeDisabledButRequiredError(
            "The sensible-only solution would cross the water saturation dome; "
            "phase_change_mode=DISABLED cannot extrapolate cp through condensation."
        )

    outside_capability = detect_phase_change_capability(outside.provider)
    outside_evaluation = evaluate_outside_side(
        hx,
        provider=outside.provider,
        mass_flow=outside.m_dot,
        T_in=outside.T_in,
        T_out=solution.T_out_outside,
        p=outside.p,
        euler_provider=euler_provider,
        properties_mean=solution.outside_props_mean,
    )
    outside_result = capability_only_result(
        "outside", outside.phase_change_mode, outside_capability
    )
    result = _simulation_from_solution(
        hx,
        inside,
        outside,
        solution,
        outside_evaluation=outside_evaluation,
        steam_result=steam_result,
        outside_result=outside_result,
        surface_margin=surface_margin,
        Q_full=full_solution.Q_total,
    )
    if outside_capability.provider_kind == "gas_mixture":
        onset, *_ = evaluate_side_onset(
            side="outside", mode=outside.phase_change_mode,
            capability=outside_capability, p=outside.p,
            thermal_state=result.thermal_state,
            envelope=result.wall_temperature_envelope,
            settings=settings,
        )
        outside_active = bool(
            onset is not None and onset.active
            and outside.phase_change_mode is PhaseChangeMode.AUTO
        )
        if outside_active and steam_result.active:
            raise MultiplePhaseChangeSidesError(
                "Inside pure-steam condensation and outside wet-gas "
                "condensation are both active; only one phase-changing side is supported."
            )
        if outside_active and not steam_result.active:
            wet_result = apply_phase_change(
                hx, inside, outside, result,
                iterate=True, euler_provider=euler_provider, settings=settings,
                skip_inside_pure_steam_guard=True,
            )
            return replace(wet_result, inside_phase_change=steam_result)
    return result


def apply_water_steam_rating(
    hx,
    inside,
    outside,
    *,
    Q: float | None,
    effectiveness: float | None,
    flow_arrangement: str | None,
    K_inlet: float,
    K_outlet: float,
    K_turn: float,
    euler_provider: str,
    include_simulation: bool,
    over_specified_tolerance: float,
    max_iterations: int,
    wall_temperature_tolerance_K: float,
    relative_alfa_tolerance: float,
    relaxation_factor: float,
    settings: PhaseChangeSettings,
):
    if not math.isfinite(over_specified_tolerance) or over_specified_tolerance < 0.0:
        raise ValueError("over_specified_tolerance must be non-negative and finite.")
    if effectiveness is not None:
        raise ValueError(
            "Steam Rating requires an explicit duty or outlet state; an "
            "effectiveness-only target is not supported for a phase-changing stream."
        )
    if inside.m_dot is None or outside.m_dot is None:
        raise ValueError("Steam Rating requires explicit mass flow on both sides.")
    Q_required = _resolve_rating_duty(
        inside, outside, Q, tolerance=over_specified_tolerance
    )
    solution = rate_steam_heater(
        hx,
        inlet_state=inside.water_steam_state,
        mass_flow_steam=inside.m_dot,
        outside_provider=outside.provider,
        mass_flow_outside=outside.m_dot,
        T_in_outside=outside.T_in,
        p_outside=outside.p,
        orientation=hx.bundle.tube.tube_orientation,
        Q_total=Q_required,
    )
    steam_result = _steam_result(solution, mode=inside.phase_change_mode)
    if inside.phase_change_mode is PhaseChangeMode.DISABLED and steam_result.active:
        raise PhaseChangeDisabledButRequiredError(
            "The specified Rating duty crosses the water saturation dome while "
            "phase_change_mode=DISABLED."
        )

    outside_capability = detect_phase_change_capability(outside.provider)
    if outside_capability.provider_kind == "gas_mixture" and steam_result.active:
        # A rating dry scaffold is built below; conservative one-active-side
        # enforcement happens before any wet-gas model could be mixed into
        # steam zone conductances.
        if outside.phase_change_mode is PhaseChangeMode.AUTO:
            raise MultiplePhaseChangeSidesError(
                "Rating cannot combine inside pure-steam condensation with an "
                "AUTO wet-gas phase-changing outside side. Disable one side explicitly."
            )

    closed_balance = _closed_balance_for_steam_rating(
        inside=inside,
        outside=outside,
        solution=solution,
    )
    outside_evaluation = evaluate_outside_side(
        hx,
        provider=outside.provider,
        mass_flow=outside.m_dot,
        T_in=outside.T_in,
        T_out=solution.T_out_outside,
        p=outside.p,
        euler_provider=euler_provider,
        properties_mean=solution.outside_props_mean,
    )
    simulation = None
    Q_achievable = None
    if include_simulation:
        simulation_inside = _simulation_side_from_rating(inside)
        simulation_outside = HXSideInput(
            provider=outside.provider, m_dot=outside.m_dot,
            T_in=outside.T_in, p=outside.p,
            phase_change_mode=outside.phase_change_mode,
        )
        simulation = hx.simulate(
            simulation_inside, simulation_outside,
            flow_arrangement=flow_arrangement,
            K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
            euler_provider=euler_provider,
        )
        Q_achievable = simulation.q

    outside_result = capability_only_result(
        "outside", outside.phase_change_mode, outside_capability
    )
    return _rating_from_solution(
        hx,
        inside,
        outside,
        solution,
        outside_evaluation=outside_evaluation,
        steam_result=steam_result,
        outside_result=outside_result,
        closed_balance=closed_balance,
        simulation=simulation,
        Q_achievable=Q_achievable,
    )


def _simulation_from_solution(
    hx,
    inside,
    outside,
    solution,
    *,
    outside_evaluation,
    steam_result,
    outside_result,
    surface_margin,
    Q_full,
):
    final_result, thermal_state, envelope, warnings = _steam_diagnostics(
        hx,
        inside,
        outside,
        solution,
        outside_evaluation=outside_evaluation,
        UA=solution.UA_total,
    )
    return HXSimulationResult(
        converged=solution.converged,
        iterations=solution.iterations,
        residual_q_rel=0.0,
        residual_T_inside_K=0.0,
        residual_T_outside_K=0.0,
        T_mean_inside=0.5 * (solution.state_in.T + solution.state_out.T),
        T_mean_outside=outside_evaluation.T_mean,
        inside_props_mean=steam_result.state_midpoint.transport,
        outside_props_mean=outside_evaluation.properties_mean,
        inside_velocity_mean=final_result.tube_side_thermal.v,
        outside_velocity_mean=outside_evaluation.velocity,
        inside_Re_mean=final_result.tube_side_thermal.Re,
        outside_Re_mean=outside_evaluation.reynolds,
        inside_Pr_mean=final_result.tube_side_thermal.Pr,
        outside_Pr_mean=outside_evaluation.prandtl,
        inside_alfa_mean=solution.inside_alpha_equivalent,
        outside_alfa_mean=solution.outside_alpha,
        U_mean=solution.U_equivalent,
        UA=solution.UA_total,
        EMTD=solution.Q_total / solution.UA_total,
        q=solution.Q_total,
        T_out_inside=solution.state_out.T,
        T_out_outside=solution.T_out_outside,
        surface_margin=surface_margin,
        overdesign_factor=0.0,
        Q_full=Q_full,
        Q_derated=solution.Q_total,
        final_result=final_result,
        thermal_state=thermal_state,
        wall_temperature_envelope=envelope,
        warnings=warnings,
        inside_phase_change=steam_result,
        outside_phase_change=outside_result,
    )


def _rating_from_solution(
    hx, inside, outside, solution, *, outside_evaluation, steam_result, outside_result,
    closed_balance, simulation, Q_achievable,
):
    A_actual = hx.bundle.total_outer_area
    A_required = solution.A_total
    scale = A_actual / A_required
    UA_actual = solution.UA_total * scale
    final_result, thermal_state, envelope, warnings = _steam_diagnostics(
        hx,
        inside,
        outside,
        solution,
        outside_evaluation=outside_evaluation,
        UA=UA_actual,
    )
    return HXRatingResult(
        overdesign_factor=A_actual / A_required - 1.0,
        ua_margin=UA_actual / solution.UA_total - 1.0,
        A_o=A_actual,
        A_required=A_required,
        UA_required=solution.UA_total,
        UA_actual=UA_actual,
        U_mean=solution.U_equivalent,
        EMTD=solution.Q_total / solution.UA_total,
        alfa_i=solution.inside_alpha_equivalent,
        alfa_o=solution.outside_alpha,
        Q_required=solution.Q_total,
        Q_achievable=Q_achievable,
        closed_balance=closed_balance,
        final_result=final_result,
        simulation=simulation,
        thermal_state=thermal_state,
        wall_temperature_envelope=envelope,
        warnings=warnings,
        inside_phase_change=steam_result,
        outside_phase_change=outside_result,
    )


def _steam_diagnostics(
    hx,
    inside,
    outside,
    solution: SteamHeaterSolution,
    *,
    outside_evaluation: OutsideSideEvaluation,
    UA: float,
):
    """Build honest result diagnostics without running a sensible HX proxy."""
    active = solution.Q_condensation > 1.0e-8
    tube_hydraulic = tube_pressure_drop = None
    inside_velocity = inside_reynolds = inside_prandtl = math.nan
    midpoint = water_steam_props_iapws97(
        p=solution.state_in.p,
        h=0.5 * (solution.state_in.h + solution.state_out.h),
    )
    if not active:
        transports = (
            solution.state_in.transport,
            midpoint.transport,
            solution.state_out.transport,
        )
        if all(value is not None for value in transports):
            tube_bundle = calculate_tube_bundle_hydraulics(
                m_dot=solution.mass_flow_steam,
                flow_area_per_pass=hx.bundle.internal_flow_area_per_pass,
                hydraulic_diameter=hx.bundle.internal_hydraulic_diameter,
                hydraulic_length_total=hx.bundle.internal_length_total,
                n_tube_passes=hx.bundle.n_passes_tube,
                tube_path_type=hx.bundle.tube_path_type,
                roughness_inner=getattr(hx.bundle.tube, "roughness_inner", None),
                provider=None,
                temperature_in=solution.state_in.T,
                temperature_out=solution.state_out.T,
                pressure=solution.state_in.p,
                inlet_props=transports[0],
                midpoint_props=transports[1],
                outlet_props=transports[2],
            )
            tube_hydraulic = HXTubeSideHydraulicResults(tube_bundle=tube_bundle)
            tube_pressure_drop = HXTubeSidePressureDropResults(
                tube_bundle=tube_bundle,
                flow_path=build_tube_side_pressure_drop_result(
                    tube_bundle, n_tube_passes=hx.bundle.n_passes_tube
                ),
            )
            inside_velocity = tube_bundle.midpoint.velocity
            inside_reynolds = tube_bundle.midpoint.reynolds
            inside_prandtl = tube_bundle.midpoint.prandtl

    outside_hydraulic = HXOutSideHydraulicResults(
        dp_total=outside_evaluation.hydraulics.dp_total,
        Re=outside_evaluation.hydraulics.midpoint.reynolds,
        v=outside_evaluation.hydraulics.midpoint.face_velocity,
        tube_bank=outside_evaluation.hydraulics,
    )
    outside_pressure_drop = HXOutsidePressureDropResults(
        tube_bank=outside_evaluation.hydraulics,
        flow_path=outside_evaluation.pressure_drop,
    )
    warnings = list(solution.warnings) + list(outside_evaluation.warnings)
    if active:
        warnings.append(_two_phase_dp_warning())

    thermal_state, envelope = _steam_wall_diagnostics(
        hx,
        inside,
        outside,
        solution,
        midpoint=midpoint,
        outside_evaluation=outside_evaluation,
        UA=UA,
    )
    warnings.extend(thermal_state.warnings)
    warnings.extend(envelope.warnings)
    warnings_result = _deduplicate_warnings(warnings)
    final_result = HXResult(
        A_i=hx.bundle.total_inner_area,
        A_o=hx.bundle.total_outer_area,
        A_frontal=hx.bundle.frontal_flow_area,
        UA=UA,
        eps=_steam_effectiveness(solution, outside_evaluation, outside.m_dot),
        Q=solution.Q_total,
        T_hot_out=solution.state_out.T,
        T_cold_out=solution.T_out_outside,
        tube_side_thermal=HXOutSideThermalResults(
            v=inside_velocity,
            Re=inside_reynolds,
            Pr=inside_prandtl,
            alfa=solution.inside_alpha_equivalent,
        ),
        tube_side_hydraulic=tube_hydraulic,
        outside_side_thermal=HXOutSideThermalResults(
            v=outside_evaluation.velocity,
            Re=outside_evaluation.reynolds,
            Pr=outside_evaluation.prandtl,
            alfa=solution.outside_alpha,
        ),
        outside_side_hydraulic=outside_hydraulic,
        tube_side_pressure_drop=tube_pressure_drop,
        outside_side_pressure_drop=outside_pressure_drop,
        warnings=warnings_result,
    )
    return final_result, thermal_state, envelope, warnings_result


def _steam_wall_diagnostics(
    hx,
    inside,
    outside,
    solution: SteamHeaterSolution,
    *,
    midpoint,
    outside_evaluation: OutsideSideEvaluation,
    UA: float,
):
    """Build a 0D endpoint wall envelope using the equivalent inside HTC.

    A reported inside Nusselt number, when representative transport exists,
    is only the dimensionless form of that equivalent HTC. It is not a local
    Shah value or the Nusselt number of any individual steam zone.
    """
    tube = hx.bundle.tube
    D_i = float(tube.D_i)
    D_o = float(tube.D_o)
    T_i_mean = 0.5 * (solution.state_in.T + solution.state_out.T)
    T_o_mean = outside_evaluation.T_mean

    def probe(T_i: float, T_o: float) -> WallTemperatureProbe:
        heat_flux = solution.U_equivalent * (T_i - T_o)
        T_wall_i = T_i - heat_flux * D_o / (
            D_i * solution.inside_alpha_equivalent
        )
        T_wall_o = T_o + heat_flux / solution.outside_alpha
        inside_nusselt = (
            solution.inside_alpha_equivalent * D_i / midpoint.transport.k
            if midpoint.transport is not None
            else None
        )
        return WallTemperatureProbe(
            inside_bulk_temperature=T_i,
            outside_bulk_temperature=T_o,
            inside_wall_temperature=T_wall_i,
            outside_wall_temperature=T_wall_o,
            alfa_i=solution.inside_alpha_equivalent,
            alfa_o=solution.outside_alpha,
            converged=True,
            iterations=0,
            inside_bulk_props=midpoint.transport,
            outside_bulk_props=outside_evaluation.properties_mean,
            inside_nusselt=inside_nusselt,
            outside_nusselt=outside_evaluation.nusselt_corrected,
            heat_rate_probe=heat_flux * hx.bundle.total_outer_area,
            residual=0.0,
        )

    probes = tuple(
        probe(T_i, T_o)
        for T_i in (solution.state_in.T, solution.state_out.T)
        for T_o in (outside_evaluation.T_in, outside_evaluation.T_out)
    )
    envelope_warning = make_warning(
        code="wall_temperature_envelope_0d_estimate",
        message=(
            "Wall-temperature minimum and maximum are estimated from a 0D "
            "inlet/outlet endpoint envelope. They are not local extrema from "
            "a spatially segmented exchanger model."
        ),
        source="thermal_iteration",
        severity="info",
    )
    envelope = WallTemperatureEnvelope(
        inside_min=min(item.inside_wall_temperature for item in probes),
        inside_max=max(item.inside_wall_temperature for item in probes),
        outside_min=min(item.outside_wall_temperature for item in probes),
        outside_max=max(item.outside_wall_temperature for item in probes),
        probes=probes,
        warnings=(envelope_warning,),
        inside_mean=probe(T_i_mean, T_o_mean).inside_wall_temperature,
        outside_mean=probe(T_i_mean, T_o_mean).outside_wall_temperature,
    )
    mean_probe = probe(T_i_mean, T_o_mean)
    inside_nusselt = mean_probe.inside_nusselt
    diagnostics = ThermalIterationDiagnostics(
        inside_Nu_base=inside_nusselt,
        inside_Nu_corrected=inside_nusselt,
        inside_length_correction=None if solution.Q_condensation > 1.0e-8 else 1.0,
        inside_wall_temperature_correction=None if solution.Q_condensation > 1.0e-8 else 1.0,
        inside_combined_correction=None if solution.Q_condensation > 1.0e-8 else 1.0,
        inside_alfa_base=solution.inside_alpha_equivalent,
        inside_alfa_corrected=solution.inside_alpha_equivalent,
        outside_Nu_base=outside_evaluation.nusselt_base,
        outside_Nu_corrected=outside_evaluation.nusselt_corrected,
        outside_wall_property_correction=outside_evaluation.wall_property_correction,
    )
    state = IterativeThermalState(
        inside_bulk_temperature=T_i_mean,
        outside_bulk_temperature=T_o_mean,
        inside_wall_temperature=mean_probe.inside_wall_temperature,
        outside_wall_temperature=mean_probe.outside_wall_temperature,
        inside_bulk_props=midpoint.transport,
        inside_wall_props=None,
        outside_bulk_props=outside_evaluation.properties_mean,
        outside_wall_props=None,
        alfa_i=solution.inside_alpha_equivalent,
        alfa_o=solution.outside_alpha,
        U=solution.U_equivalent,
        UA=UA,
        iterations=solution.iterations,
        converged=solution.converged,
        residual=0.0,
        diagnostics=diagnostics,
        inside_provider_name=type(inside.provider).__name__,
        outside_provider_name=type(outside.provider).__name__,
        warnings=(),
    )
    return state, envelope


def _steam_effectiveness(
    solution: SteamHeaterSolution,
    outside_evaluation: OutsideSideEvaluation,
    mass_flow_outside: float,
) -> float:
    delta_T_steam = solution.state_in.T - solution.state_out.T
    C_steam = (
        solution.Q_total / delta_T_steam
        if delta_T_steam > 1.0e-9
        else math.inf
    )
    C_outside = mass_flow_outside * outside_evaluation.properties_mean.cp
    Q_max = min(C_steam, C_outside) * (
        solution.state_in.T - outside_evaluation.T_in
    )
    return solution.Q_total / Q_max


def _steam_result(
    solution: SteamHeaterSolution,
    *,
    mode: PhaseChangeMode,
) -> WaterSteamPhaseChangeResult:
    midpoint = water_steam_props_iapws97(
        p=solution.state_in.p,
        h=0.5 * (solution.state_in.h + solution.state_out.h),
    )
    zones = {zone.kind: zone for zone in solution.zones}

    def zone_value(kind, name, default=None):
        zone = zones.get(kind)
        return default if zone is None else getattr(zone, name)

    active = solution.Q_condensation > 1.0e-8
    return WaterSteamPhaseChangeResult(
        side="inside",
        mode=mode,
        direction=PhaseChangeDirection.CONDENSATION if active else PhaseChangeDirection.NONE,
        component="H2O",
        capable=True,
        possible=active,
        active=active,
        converged=solution.converged,
        method="water_steam_multizone_0d",
        state_in=solution.state_in,
        state_midpoint=midpoint,
        state_out=solution.state_out,
        phase_in=solution.state_in.phase,
        phase_out=solution.state_out.phase,
        T_in=solution.state_in.T,
        T_out=solution.state_out.T,
        Tsat=solution.saturation.Tsat,
        p=solution.state_in.p,
        h_in=solution.state_in.h,
        h_out=solution.state_out.h,
        quality_in=solution.state_in.quality,
        quality_out=solution.state_out.quality,
        Q_desuperheat=solution.Q_desuperheat,
        Q_condensation=solution.Q_condensation,
        Q_subcooling=solution.Q_subcooling,
        Q_total=solution.Q_total,
        A_desuperheat=solution.A_desuperheat,
        A_condensation=solution.A_condensation,
        A_subcooling=solution.A_subcooling,
        A_total=solution.A_total,
        zone_fraction_desuperheat=solution.zone_fraction_desuperheat,
        zone_fraction_condensation=solution.zone_fraction_condensation,
        zone_fraction_subcooling=solution.zone_fraction_subcooling,
        zone_alpha_desuperheat=solution.zone_alpha_desuperheat,
        zone_alpha_condensation=solution.zone_alpha_condensation,
        zone_alpha_subcooling=solution.zone_alpha_subcooling,
        zone_U_desuperheat=zone_value(SteamHeaterZoneKind.SUPERHEAT, "U"),
        zone_U_condensation=zone_value(SteamHeaterZoneKind.CONDENSATION, "U"),
        zone_U_subcooling=zone_value(SteamHeaterZoneKind.SUBCOOLING, "U"),
        zone_UA_desuperheat=solution.UA_desuperheat,
        zone_UA_condensation=solution.UA_condensation,
        zone_UA_subcooling=solution.UA_subcooling,
        UA_total=solution.UA_total,
        mass_flow_total=solution.mass_flow_steam,
        mass_flux=solution.mass_flux,
        m_dot_condensate=solution.Q_condensation / solution.saturation.hfg,
        two_phase_pressure_drop_supported=False,
        two_phase_pressure_drop_status=(
            "not_supported" if active else "not_applicable_single_phase"
        ),
        iterations=solution.iterations,
        root_iterations=solution.root_iterations,
        property_evaluations=solution.property_evaluations,
        warnings=(
            solution.warnings + (_two_phase_dp_warning(),)
            if active
            else solution.warnings
        ),
        assumptions=solution.assumptions,
        inside_alpha_equivalent=solution.inside_alpha_equivalent,
        inside_alpha_area_weighted=solution.inside_alpha_area_weighted,
    )


def _closed_balance_for_steam_rating(*, inside, outside, solution):
    delta_T_steam = solution.state_in.T - solution.state_out.T
    C_steam = (
        solution.Q_total / delta_T_steam
        if delta_T_steam > 1.0e-9
        else None
    )
    cp_steam = None if C_steam is None else C_steam / inside.m_dot
    outside_T_out = solution.T_out_outside
    outside_transport = solution.outside_props_mean
    C_outside = outside.m_dot * outside_transport.cp
    inside_closed = ClosedBalanceSide(
        provider=inside.provider, p=inside.p, m_dot=inside.m_dot,
        T_in=inside.T_in, T_out=solution.state_out.T,
        cp_mean=cp_steam, C=C_steam,
    )
    outside_closed = ClosedBalanceSide(
        provider=outside.provider, p=outside.p, m_dot=outside.m_dot,
        T_in=outside.T_in, T_out=outside_T_out,
        cp_mean=outside_transport.cp, C=C_outside,
    )
    C_min = C_outside if C_steam is None else min(C_steam, C_outside)
    Q_max = C_min * (inside.T_in - outside.T_in)
    return ClosedBalance(
        inside=inside_closed,
        outside=outside_closed,
        hot_is_inside=True,
        Q=solution.Q_total,
        Q_max=Q_max,
        effectiveness=solution.Q_total / Q_max,
        warnings=None,
    )


def _resolve_rating_duty(inside, outside, Q, *, tolerance):
    candidates = []
    if Q is not None:
        if not math.isfinite(Q) or Q <= 0.0:
            raise ValueError("Steam Rating Q must be positive and finite.")
        candidates.append(("Q", Q))
    outlet_state = inside.water_steam_outlet_state
    if outlet_state is not None:
        candidates.append((
            "inside outlet state",
            inside.m_dot * (inside.water_steam_state.h - outlet_state.h),
        ))
    if outside.T_out is not None:
        props = outside.provider.at(
            T=0.5 * (outside.T_in + outside.T_out), p=outside.p
        )
        transport = getattr(props, "transport", props)
        candidates.append((
            "outside temperature program",
            outside.m_dot * transport.cp * (outside.T_out - outside.T_in),
        ))
    if not candidates:
        raise ValueError(
            "Steam Rating requires explicit Q, a water/steam outlet state, or a "
            "fully specified opposing-side temperature program."
        )
    for label, value in candidates:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Steam Rating duty from {label} must be positive and finite.")
    reference_label, reference = candidates[0]
    for label, value in candidates[1:]:
        scale = max(abs(reference), abs(value), 1.0)
        if abs(reference - value) > tolerance * scale:
            raise ValueError(
                "Over-specified Steam Rating duties are inconsistent: "
                f"{reference_label}={reference:.9g} W versus {label}={value:.9g} W."
            )
    return reference


def _simulation_side_from_rating(inside):
    kwargs = dict(
        provider=inside.provider,
        m_dot=inside.m_dot,
        p=inside.p,
        phase_change_mode=inside.phase_change_mode,
    )
    if inside.state_specification == "p+x":
        kwargs["quality_in"] = inside.quality_in
    elif inside.state_specification == "p+h":
        kwargs["h_in"] = inside.h_in
    else:
        kwargs["T_in"] = inside.T_in
    return HXSideInput(**kwargs)


def _two_phase_dp_warning() -> ModelWarning:
    return make_warning(
        code=WC.STEAM_TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED,
        message=(
            "Tube-side pressure drop is not reported because the steam path "
            "contains a two-phase condensation zone."
        ),
        source="steam_integration",
        severity="warning",
    )


def _deduplicate_warnings(warnings):
    result = []
    seen = set()
    for warning in warnings:
        key = (warning.code, warning.message)
        if key not in seen:
            seen.add(key)
            result.append(warning)
    return result or None
