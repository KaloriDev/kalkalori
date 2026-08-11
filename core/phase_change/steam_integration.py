# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Public Simulation/Rating adapters for the shared steam-heater solver."""

from __future__ import annotations

from dataclasses import replace
import math

from core.common.warnings import ModelWarning, make_warning
from core.models.heat_balance import ClosedBalance, ClosedBalanceSide
from core.models.rating import run_rating
from core.models.simulation import HXSideInput, run_simulation
from core.phase_change import warning_codes as WC
from core.phase_change.capability import detect_phase_change_capability, is_pure_water_provider
from core.phase_change.integration import (
    MultiplePhaseChangeSidesError,
    PhaseChangeSettings,
    _capability_only_result,
    _evaluate_side_onset,
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
from core.properties.fluids import ConstantPropertyProvider
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
        and state.phase is not WaterSteamPhase.SUBCOOLED_LIQUID
    )


def reject_unsupported_outside_pure_steam(outside) -> None:
    if not is_pure_water_provider(outside.provider):
        return
    state = getattr(outside, "water_steam_state", None)
    if state is None:
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

    proxy_inside = _proxy_simulation_side(
        inside, solution, midpoint_state=steam_result.state_midpoint
    )
    scaffold = run_simulation(
        hx, proxy_inside, outside,
        surface_margin=0.0,
        iterate=True,
        flow_arrangement=flow_arrangement,
        K_inlet=K_inlet,
        K_outlet=K_outlet,
        K_turn=K_turn,
        euler_provider=euler_provider,
        max_iter=max_iter,
        temperature_tolerance_K=temperature_tolerance_K,
        relative_duty_tolerance=relative_duty_tolerance,
        relaxation_factor=relaxation_factor,
        relative_alfa_tolerance=relative_alfa_tolerance,
    )

    outside_capability = detect_phase_change_capability(outside.provider)
    if outside_capability.provider_kind == "gas_mixture":
        onset, *_ = _evaluate_side_onset(
            side="outside", mode=outside.phase_change_mode,
            capability=outside_capability, p=outside.p,
            thermal_state=scaffold.thermal_state,
            envelope=scaffold.wall_temperature_envelope,
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
                hx, inside, outside, scaffold,
                iterate=True, euler_provider=euler_provider, settings=settings,
                skip_inside_pure_steam_guard=True,
            )
            return replace(wet_result, inside_phase_change=steam_result)

    outside_result = _capability_only_result(
        "outside", outside.phase_change_mode, outside_capability
    )
    return _simulation_from_solution(
        scaffold,
        solution,
        steam_result=steam_result,
        outside_result=outside_result,
        T_in_outside=outside.T_in,
        surface_margin=surface_margin,
        Q_full=full_solution.Q_total,
    )


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

    proxy_provider = ConstantPropertyProvider(
        _representative_transport(solution, midpoint_state=steam_result.state_midpoint)
    )
    closed_scaffold, closed_public = _closed_balances_for_steam_rating(
        inside=inside,
        outside=outside,
        solution=solution,
        proxy_provider=proxy_provider,
    )
    scaffold = run_rating(
        hx, closed_scaffold,
        flow_arrangement=flow_arrangement,
        K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
        euler_provider=euler_provider,
        include_simulation=False,
        max_iterations=max_iterations,
        wall_temperature_tolerance_K=wall_temperature_tolerance_K,
        relative_alfa_tolerance=relative_alfa_tolerance,
        relaxation_factor=relaxation_factor,
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

    outside_result = _capability_only_result(
        "outside", outside.phase_change_mode, outside_capability
    )
    return _rating_from_solution(
        scaffold,
        solution,
        steam_result=steam_result,
        outside_result=outside_result,
        closed_balance=closed_public,
        simulation=simulation,
        Q_achievable=Q_achievable,
    )


def _simulation_from_solution(
    scaffold,
    solution,
    *,
    steam_result,
    outside_result,
    T_in_outside,
    surface_margin,
    Q_full,
):
    midpoint = steam_result.state_midpoint
    has_two_phase = solution.Q_condensation > 0.0
    final_result = replace(
        scaffold.final_result,
        UA=solution.UA_total,
        Q=solution.Q_total,
        T_hot_out=solution.state_out.T,
        T_cold_out=solution.T_out_outside,
        tube_side_thermal=replace(
            scaffold.final_result.tube_side_thermal,
            v=math.nan if has_two_phase else scaffold.inside_velocity_mean,
            Re=math.nan if has_two_phase else scaffold.inside_Re_mean,
            Pr=math.nan if has_two_phase else scaffold.inside_Pr_mean,
            alfa=solution.inside_alfa_mean,
        ),
        outside_side_thermal=replace(
            scaffold.final_result.outside_side_thermal,
            alfa=solution.outside_alpha,
        ),
    )
    thermal_state = scaffold.thermal_state
    if thermal_state is not None:
        thermal_state = replace(
            thermal_state,
            alfa_i=solution.inside_alfa_mean,
            alfa_o=solution.outside_alpha,
            U=solution.U_equivalent,
            UA=solution.UA_total,
        )
    warnings = list(scaffold.warnings or []) + list(solution.warnings)
    if has_two_phase:
        warnings.append(_two_phase_dp_warning())
    return replace(
        scaffold,
        converged=solution.converged,
        iterations=solution.iterations,
        residual_q_rel=0.0,
        residual_T_inside_K=0.0,
        residual_T_outside_K=0.0,
        T_mean_inside=0.5 * (solution.state_in.T + solution.state_out.T),
        T_mean_outside=0.5 * (T_in_outside + solution.T_out_outside),
        inside_props_mean=midpoint.transport,
        inside_velocity_mean=math.nan if has_two_phase else scaffold.inside_velocity_mean,
        inside_Re_mean=math.nan if has_two_phase else scaffold.inside_Re_mean,
        inside_Pr_mean=math.nan if has_two_phase else scaffold.inside_Pr_mean,
        inside_alfa_mean=solution.inside_alfa_mean,
        outside_alfa_mean=solution.outside_alpha,
        U_mean=solution.U_equivalent,
        UA=solution.UA_total,
        EMTD=solution.Q_total / solution.UA_total,
        q=solution.Q_total,
        T_out_inside=solution.state_out.T,
        T_out_outside=solution.T_out_outside,
        surface_margin=surface_margin,
        Q_full=Q_full,
        Q_derated=solution.Q_total,
        final_result=final_result,
        thermal_state=thermal_state,
        warnings=_deduplicate_warnings(warnings),
        inside_phase_change=steam_result,
        outside_phase_change=outside_result,
    )


def _rating_from_solution(
    scaffold, solution, *, steam_result, outside_result,
    closed_balance, simulation, Q_achievable,
):
    A_actual = scaffold.A_o
    A_required = solution.A_total
    scale = A_actual / A_required
    UA_actual = sum(zone.U * zone.area * scale for zone in solution.zones)
    final_result = replace(
        scaffold.final_result,
        UA=UA_actual,
        Q=solution.Q_total,
        T_hot_out=solution.state_out.T,
        T_cold_out=solution.T_out_outside,
        tube_side_thermal=replace(
            scaffold.final_result.tube_side_thermal,
            v=math.nan if solution.Q_condensation > 0.0 else scaffold.final_result.tube_side_thermal.v,
            Re=math.nan if solution.Q_condensation > 0.0 else scaffold.final_result.tube_side_thermal.Re,
            Pr=math.nan if solution.Q_condensation > 0.0 else scaffold.final_result.tube_side_thermal.Pr,
            alfa=solution.inside_alfa_mean,
        ),
        outside_side_thermal=replace(
            scaffold.final_result.outside_side_thermal,
            alfa=solution.outside_alpha,
        ),
    )
    thermal_state = replace(
        scaffold.thermal_state,
        alfa_i=solution.inside_alfa_mean,
        alfa_o=solution.outside_alpha,
        U=solution.U_equivalent,
        UA=UA_actual,
    )
    warnings = list(scaffold.warnings or []) + list(solution.warnings)
    if solution.Q_condensation > 0.0:
        warnings.append(_two_phase_dp_warning())
    return replace(
        scaffold,
        overdesign_factor=A_actual / A_required - 1.0,
        ua_margin=UA_actual / solution.UA_total - 1.0,
        A_required=A_required,
        UA_required=solution.UA_total,
        UA_actual=UA_actual,
        U_mean=solution.U_equivalent,
        EMTD=solution.Q_total / solution.UA_total,
        alfa_i=solution.inside_alfa_mean,
        alfa_o=solution.outside_alpha,
        Q_required=solution.Q_total,
        Q_achievable=Q_achievable,
        closed_balance=closed_balance,
        final_result=final_result,
        simulation=simulation,
        thermal_state=thermal_state,
        warnings=_deduplicate_warnings(warnings),
        inside_phase_change=steam_result,
        outside_phase_change=outside_result,
    )


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
        m_dot_condensate=solution.Q_condensation / solution.saturation.hfg,
        two_phase_pressure_drop_supported=False,
        two_phase_pressure_drop_status=(
            "not_supported" if active else "not_applicable_single_phase"
        ),
        iterations=solution.iterations,
        root_iterations=solution.root_iterations,
        property_evaluations=solution.property_evaluations,
        runtime_s=solution.runtime_s,
        solution=solution,
        warnings=(
            solution.warnings + (_two_phase_dp_warning(),)
            if active
            else solution.warnings
        ),
        assumptions=solution.assumptions,
    )


def _proxy_simulation_side(inside, solution, *, midpoint_state) -> HXSideInput:
    return HXSideInput(
        provider=ConstantPropertyProvider(
            _representative_transport(solution, midpoint_state=midpoint_state)
        ),
        m_dot=inside.m_dot,
        T_in=inside.T_in,
        p=inside.p,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )


def _representative_transport(
    solution: SteamHeaterSolution,
    *,
    midpoint_state=None,
):
    midpoint = midpoint_state or water_steam_props_iapws97(
        p=solution.state_in.p,
        h=0.5 * (solution.state_in.h + solution.state_out.h),
    )
    if midpoint.transport is not None:
        return midpoint.transport
    # Transport is only a scaffold for legacy result containers; all steam
    # thermal physics and endpoint reporting comes from the zone solution.
    return solution.saturation.saturated_vapor.transport


def _closed_balances_for_steam_rating(*, inside, outside, solution, proxy_provider):
    delta_T_steam = solution.state_in.T - solution.state_out.T
    C_steam = (
        solution.Q_total / delta_T_steam
        if delta_T_steam > 1.0e-9
        else 1.0e30
    )
    cp_steam = C_steam / inside.m_dot
    outside_T_out = solution.T_out_outside
    outside_props = outside.provider.at(
        T=0.5 * (outside.T_in + outside_T_out), p=outside.p
    )
    outside_transport = getattr(outside_props, "transport", outside_props)
    C_outside = outside.m_dot * outside_transport.cp
    inside_scaffold = ClosedBalanceSide(
        provider=proxy_provider, p=inside.p, m_dot=inside.m_dot,
        T_in=inside.T_in, T_out=solution.state_out.T,
        cp_mean=cp_steam, C=C_steam,
    )
    outside_closed = ClosedBalanceSide(
        provider=outside.provider, p=outside.p, m_dot=outside.m_dot,
        T_in=outside.T_in, T_out=outside_T_out,
        cp_mean=outside_transport.cp, C=C_outside,
    )
    Q_max = min(C_steam, C_outside) * (inside.T_in - outside.T_in)
    scaffold = ClosedBalance(
        inside=inside_scaffold,
        outside=outside_closed,
        hot_is_inside=True,
        Q=solution.Q_total,
        Q_max=Q_max,
        effectiveness=solution.Q_total / Q_max,
        warnings=None,
    )
    public = replace(
        scaffold,
        inside=replace(inside_scaffold, provider=inside.provider),
    )
    return scaffold, public


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
