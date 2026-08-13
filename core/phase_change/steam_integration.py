# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Public Simulation/Rating adapters for pure water/steam zone solvers."""

from __future__ import annotations

from dataclasses import replace
import math

from core.common.warnings import ModelWarning, make_warning
from core.heat_transfer.outside_side import OutsideSideEvaluation, evaluate_outside_side
from core.heat_transfer.outside_dispatch import (
    DEFAULT_FINNED_DP_PROVIDER,
    DEFAULT_FINNED_HT_PROVIDER,
)
from core.geometry.tube import TubeSurfaceType
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
from core.phase_change.water_evaporator import (
    WaterEvaporatorSolution,
    WaterEvaporatorZoneKind,
    rate_water_evaporator,
    solve_water_evaporator,
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


def translate_saturation_crossing_error(inside, exc: ValueError, outside=None) -> None:
    """Translate dry-solver saturation ambiguity into controlled scope errors."""
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
    outside_state = getattr(outside, "water_steam_state", None)
    if (
        outside is not None
        and is_pure_water_provider(outside.provider)
        and outside_state is not None
        and outside_state.phase is WaterSteamPhase.SUBCOOLED_LIQUID
        and "T+p lies on the water saturation line" in str(exc)
    ):
        raise PureSteamOutsideNotSupportedError(
            "The outside pure-water stream reached its saturation boundary "
            "in the heating direction; pure-water evaporation outside tubes "
            "is not supported."
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


def is_inside_water_evaporation_case(inside, outside) -> bool:
    """Return whether the IAPWS tube stream is heated by the outside inlet."""
    state = getattr(inside, "water_steam_state", None)
    return (
        is_pure_water_provider(inside.provider)
        and state is not None
        and state.phase is not WaterSteamPhase.SUPERCRITICAL_FLUID
        and outside.T_in > state.T
    )


def water_evaporation_reaches_saturation(
    hx,
    inside,
    outside,
    *,
    euler_provider: str,
    finned_heat_transfer_provider: object = DEFAULT_FINNED_HT_PROVIDER,
    finned_pressure_drop_provider: object = DEFAULT_FINNED_DP_PROVIDER,
    surface_margin: float = 0.0,
    iterate: bool = True,
    flow_arrangement: str | None = None,
    max_iter: int = 30,
    temperature_tolerance_K: float = 0.05,
    relative_duty_tolerance: float = 1.0e-4,
    relaxation_factor: float = 0.5,
    relative_alfa_tolerance: float = 1.0e-3,
) -> bool:
    """Test whether full geometry can reach x=0 without invoking boiling HTC."""
    state = getattr(inside, "water_steam_state", None)
    if state is None or state.phase is not WaterSteamPhase.SUBCOOLED_LIQUID:
        return True
    saturated_liquid = water_steam_props_iapws97(p=state.p, x=0.0)
    if (
        getattr(hx.bundle.tube, "surface_type", None)
        is TubeSurfaceType.CIRCULAR_FINNED
    ):
        # The legacy water-evaporator preheat probe evaluates its outside
        # coefficient through the bare-tube helper. Do not enter that path for
        # fins. A dry surface-aware Simulation gives the full-geometry sensible
        # duty; compare the corresponding outlet enthalpy with saturated liquid.
        from core.models.simulation import run_simulation

        try:
            dry = run_simulation(
                hx,
                inside,
                outside,
                euler_provider=euler_provider,
                finned_heat_transfer_provider=finned_heat_transfer_provider,
                finned_pressure_drop_provider=finned_pressure_drop_provider,
                surface_margin=surface_margin,
                iterate=iterate,
                flow_arrangement=flow_arrangement,
                max_iter=max_iter,
                temperature_tolerance_K=temperature_tolerance_K,
                relative_duty_tolerance=relative_duty_tolerance,
                relaxation_factor=relaxation_factor,
                relative_alfa_tolerance=relative_alfa_tolerance,
            )
        except ValueError as exc:
            # The generic dry iteration can land numerically on the IAPWS
            # saturation line before it has produced a duty. For this probe
            # that is exactly the positive answer; let the public caller emit
            # the dedicated finned wet-surface error instead of leaking the
            # backend's ambiguous T+p exception.
            if "T+p lies on the water saturation line" in str(exc):
                return True
            raise
        trial_outlet_enthalpy = state.h + dry.q / inside.m_dot
        return trial_outlet_enthalpy >= saturated_liquid.h - 1.0e-3
    try:
        boundary = rate_water_evaporator(
            hx,
            inlet_state=state,
            outlet_state=saturated_liquid,
            mass_flow_water=inside.m_dot,
            outside_provider=outside.provider,
            mass_flow_outside=outside.m_dot,
            T_in_outside=outside.T_in,
            p_outside=outside.p,
            orientation=None,
            euler_provider=euler_provider,
        )
    except ValueError as exc:
        if "violates a positive zone temperature difference" in str(exc):
            return False
        raise
    return boundary.A_total < hx.bundle.total_outer_area


def is_inside_water_evaporation_rating_case(
    inside,
    outside,
    *,
    Q: float | None,
    effectiveness: float | None,
) -> bool:
    """Route a water Rating when any consistent heating target reaches x=0."""
    state = getattr(inside, "water_steam_state", None)
    if (
        not is_pure_water_provider(inside.provider)
        or state is None
        or state.phase is WaterSteamPhase.SUPERCRITICAL_FLUID
    ):
        return False

    heating_duties: list[float] = []
    outlet_state = getattr(inside, "water_steam_outlet_state", None)
    if outlet_state is not None and inside.m_dot is not None:
        outlet_duty = inside.m_dot * (outlet_state.h - state.h)
        if outlet_duty > 0.0:
            heating_duties.append(outlet_duty)
    outside_is_hot = outside.T_in is not None and outside.T_in > state.T
    if outlet_state is not None and outside_is_hot and not heating_duties:
        # Route a thermodynamically inconsistent cooling target to the water
        # driver so it is rejected as an evaporation/heating request instead
        # of being interpreted through a generic sensible cp closure.
        return True
    if Q is not None and Q > 0.0 and outside_is_hot:
        heating_duties.append(Q)
    if (
        outside_is_hot
        and outside.m_dot is not None
        and outside.T_out is not None
        and outside.T_out < outside.T_in
    ):
        outside_props = outside.provider.at(
            T=0.5 * (outside.T_in + outside.T_out), p=outside.p
        )
        transport = getattr(outside_props, "transport", outside_props)
        heating_duties.append(
            outside.m_dot * transport.cp * (outside.T_in - outside.T_out)
        )
    if effectiveness is not None and outside_is_hot:
        return True
    if not heating_duties:
        return bool(
            outside_is_hot
            and state.phase is not WaterSteamPhase.SUBCOOLED_LIQUID
        )
    if state.phase is not WaterSteamPhase.SUBCOOLED_LIQUID:
        return True
    if inside.m_dot is None:
        return True
    saturated_liquid = water_steam_props_iapws97(p=state.p, x=0.0)
    return any(
        state.h + duty / inside.m_dot >= saturated_liquid.h - 1.0e-3
        for duty in heating_duties
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
            "Pure-water evaporation/condensation or two-phase flow outside "
            "tubes is outside the supported KalKalori scope."
        )


def reject_outside_pure_water_evaporation_crossing(outside, *, T_out: float) -> None:
    """Reject an IAPWS outside-liquid program only when it crosses Tsat."""
    state = getattr(outside, "water_steam_state", None)
    if (
        not is_pure_water_provider(outside.provider)
        or state is None
        or state.phase is not WaterSteamPhase.SUBCOOLED_LIQUID
    ):
        return
    saturation = water_steam_props_iapws97(p=state.p, x=0.0)
    if T_out >= saturation.T - 1.0e-6:
        raise PureSteamOutsideNotSupportedError(
            "The outside pure-water temperature program reaches saturation; "
            "pure-water evaporation is supported only inside tubes."
        )


def reject_outside_pure_water_evaporation_rating(outside, inside, *, Q) -> None:
    """Reject Rating targets that evaporate pure water on the outside side."""
    state = getattr(outside, "water_steam_state", None)
    if state is None or state.phase is not WaterSteamPhase.SUBCOOLED_LIQUID:
        return
    target = getattr(outside, "water_steam_outlet_state", None)
    target_h = None if target is None else target.h
    if (
        target_h is None
        and Q is not None
        and outside.m_dot is not None
        and inside.T_in is not None
        and inside.T_in > outside.T_in
    ):
        target_h = state.h + Q / outside.m_dot
    if target_h is None:
        return
    saturated_liquid = water_steam_props_iapws97(p=state.p, x=0.0)
    if target_h >= saturated_liquid.h - 1.0e-3:
        raise PureSteamOutsideNotSupportedError(
            "The Rating target reaches pure-water saturation on the outside "
            "side; pure-water evaporation is supported only inside tubes."
        )


def apply_water_evaporation_simulation(
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
    """Run public Simulation for an IAPWS tube stream heated from outside."""
    if not math.isfinite(surface_margin) or surface_margin < 0.0:
        raise ValueError("surface_margin must be a non-negative finite value.")
    if not iterate:
        raise ValueError("Pure-water multi-zone calculation requires iterate=True.")
    solve_kwargs = dict(
        inlet_state=inside.water_steam_state,
        mass_flow_water=inside.m_dot,
        outside_provider=outside.provider,
        mass_flow_outside=outside.m_dot,
        T_in_outside=outside.T_in,
        p_outside=outside.p,
        orientation=hx.bundle.tube.tube_orientation,
        euler_provider=euler_provider,
    )
    full_solution = solve_water_evaporator(hx, **solve_kwargs)
    solution = full_solution
    if surface_margin > 0.0:
        solution = solve_water_evaporator(
            hx,
            available_area=hx.bundle.total_outer_area / (1.0 + surface_margin),
            **solve_kwargs,
        )
    possible = full_solution.Q_evaporation > 0.0
    water_result = _water_evaporation_result(
        solution, mode=inside.phase_change_mode, possible=possible
    )
    if inside.phase_change_mode is PhaseChangeMode.DISABLED and water_result.active:
        raise PhaseChangeDisabledButRequiredError(
            "The achievable water-heating duty crosses the saturation dome; "
            "phase_change_mode=DISABLED cannot extrapolate liquid cp through boiling."
        )

    outside_capability = detect_phase_change_capability(outside.provider)
    outside_evaluation = solution.outside_evaluation
    outside_result = capability_only_result(
        "outside", outside.phase_change_mode, outside_capability
    )
    result = _simulation_from_solution(
        hx,
        inside,
        outside,
        solution,
        outside_evaluation=outside_evaluation,
        steam_result=water_result,
        outside_result=outside_result,
        surface_margin=surface_margin,
        Q_full=full_solution.Q_total,
        active=water_result.active,
        inside_is_hot=False,
        two_phase_warning=_water_evaporation_two_phase_dp_warning(),
    )
    if outside_capability.provider_kind == "gas_mixture":
        onset, *_ = evaluate_side_onset(
            side="outside",
            mode=outside.phase_change_mode,
            capability=outside_capability,
            p=outside.p,
            thermal_state=result.thermal_state,
            envelope=result.wall_temperature_envelope,
            settings=settings,
        )
        outside_active = bool(
            onset is not None
            and onset.active
            and outside.phase_change_mode is PhaseChangeMode.AUTO
        )
        if outside_active and water_result.active:
            raise MultiplePhaseChangeSidesError(
                "Inside pure-water evaporation and outside wet-gas "
                "condensation are both active; only one phase-changing side is supported."
            )
        if outside_active and not water_result.active:
            wet_result = apply_phase_change(
                hx,
                inside,
                outside,
                result,
                iterate=True,
                euler_provider=euler_provider,
                settings=settings,
                skip_inside_pure_steam_guard=True,
            )
            return replace(wet_result, inside_phase_change=water_result)
    return result


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
        active=steam_result.active,
        inside_is_hot=True,
        two_phase_warning=_two_phase_dp_warning(),
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
        active=steam_result.active,
        inside_is_hot=True,
        two_phase_warning=_two_phase_dp_warning(),
    )


def apply_water_evaporation_rating(
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
    """Run public Rating for an IAPWS tube stream heated from outside."""
    if not math.isfinite(over_specified_tolerance) or over_specified_tolerance < 0.0:
        raise ValueError("over_specified_tolerance must be non-negative and finite.")
    if effectiveness is not None:
        raise ValueError(
            "Water-evaporation Rating requires an explicit duty, outlet state, "
            "or opposing temperature program; an effectiveness-only target "
            "is not supported for a phase-changing stream."
        )
    if inside.m_dot is None or outside.m_dot is None:
        raise ValueError("Water-evaporation Rating requires mass flow on both sides.")
    Q_required = _resolve_water_evaporation_rating_duty(
        inside, outside, Q, tolerance=over_specified_tolerance
    )
    solution = rate_water_evaporator(
        hx,
        inlet_state=inside.water_steam_state,
        mass_flow_water=inside.m_dot,
        outside_provider=outside.provider,
        mass_flow_outside=outside.m_dot,
        T_in_outside=outside.T_in,
        p_outside=outside.p,
        orientation=hx.bundle.tube.tube_orientation,
        Q_total=Q_required,
        euler_provider=euler_provider,
    )
    water_result = _water_evaporation_result(
        solution,
        mode=inside.phase_change_mode,
        possible=solution.Q_evaporation > 0.0,
    )
    if inside.phase_change_mode is PhaseChangeMode.DISABLED and water_result.active:
        raise PhaseChangeDisabledButRequiredError(
            "The specified Rating duty crosses the water saturation dome while "
            "phase_change_mode=DISABLED."
        )

    outside_capability = detect_phase_change_capability(outside.provider)
    if (
        outside_capability.provider_kind == "gas_mixture"
        and water_result.active
        and outside.phase_change_mode is PhaseChangeMode.AUTO
    ):
        raise MultiplePhaseChangeSidesError(
            "Rating cannot combine inside pure-water evaporation with an "
            "AUTO wet-gas phase-changing outside side. Disable one side explicitly."
        )

    closed_balance = _closed_balance_for_water_evaporation_rating(
        inside=inside, outside=outside, solution=solution
    )
    outside_evaluation = solution.outside_evaluation
    simulation = None
    Q_achievable = None
    if include_simulation:
        simulation_inside = _simulation_side_from_rating(inside)
        simulation_outside = HXSideInput(
            provider=outside.provider,
            m_dot=outside.m_dot,
            T_in=outside.T_in,
            p=outside.p,
            phase_change_mode=outside.phase_change_mode,
        )
        simulation = hx.simulate(
            simulation_inside,
            simulation_outside,
            flow_arrangement=flow_arrangement,
            K_inlet=K_inlet,
            K_outlet=K_outlet,
            K_turn=K_turn,
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
        steam_result=water_result,
        outside_result=outside_result,
        closed_balance=closed_balance,
        simulation=simulation,
        Q_achievable=Q_achievable,
        active=water_result.active,
        inside_is_hot=False,
        two_phase_warning=_water_evaporation_two_phase_dp_warning(),
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
    active,
    inside_is_hot,
    two_phase_warning,
):
    final_result, thermal_state, envelope, warnings = _water_steam_diagnostics(
        hx,
        inside,
        outside,
        solution,
        outside_evaluation=outside_evaluation,
        UA=solution.UA_total,
        active=active,
        inside_is_hot=inside_is_hot,
        two_phase_warning=two_phase_warning,
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
    closed_balance, simulation, Q_achievable, active, inside_is_hot,
    two_phase_warning,
):
    A_actual = hx.bundle.total_outer_area
    A_required = solution.A_total
    scale = A_actual / A_required
    UA_actual = solution.UA_total * scale
    final_result, thermal_state, envelope, warnings = _water_steam_diagnostics(
        hx,
        inside,
        outside,
        solution,
        outside_evaluation=outside_evaluation,
        UA=UA_actual,
        active=active,
        inside_is_hot=inside_is_hot,
        two_phase_warning=two_phase_warning,
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


def _water_steam_diagnostics(
    hx,
    inside,
    outside,
    solution: SteamHeaterSolution | WaterEvaporatorSolution,
    *,
    outside_evaluation: OutsideSideEvaluation,
    UA: float,
    active: bool,
    inside_is_hot: bool,
    two_phase_warning: ModelWarning,
):
    """Build honest result diagnostics without running a sensible HX proxy."""
    tube_hydraulic = tube_pressure_drop = None
    inside_velocity = inside_reynolds = inside_prandtl = math.nan
    midpoint = getattr(solution, "state_midpoint", None) or water_steam_props_iapws97(
        p=solution.state_in.p, h=0.5 * (solution.state_in.h + solution.state_out.h)
    )
    if not active:
        transports = (
            solution.state_in.transport,
            midpoint.transport,
            solution.state_out.transport,
        )
        if all(value is not None for value in transports):
            tube_bundle = calculate_tube_bundle_hydraulics(
                m_dot=getattr(solution, "mass_flow_steam", None)
                or solution.mass_flow_water,
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
        warnings.append(two_phase_warning)

    thermal_state, envelope = _water_steam_wall_diagnostics(
        hx,
        inside,
        outside,
        solution,
        midpoint=midpoint,
        outside_evaluation=outside_evaluation,
        UA=UA,
        active=active,
    )
    warnings.extend(thermal_state.warnings)
    warnings.extend(envelope.warnings)
    warnings_result = _deduplicate_warnings(warnings)
    final_result = HXResult(
        A_i=hx.bundle.total_inner_area,
        A_o=hx.bundle.total_outer_area,
        A_frontal=hx.bundle.frontal_flow_area,
        UA=UA,
        eps=_water_steam_effectiveness(solution, outside_evaluation, outside.m_dot),
        Q=solution.Q_total,
        T_hot_out=(solution.state_out.T if inside_is_hot else solution.T_out_outside),
        T_cold_out=(solution.T_out_outside if inside_is_hot else solution.state_out.T),
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


def _water_steam_wall_diagnostics(
    hx,
    inside,
    outside,
    solution: SteamHeaterSolution,
    *,
    midpoint,
    outside_evaluation: OutsideSideEvaluation,
    UA: float,
    active: bool,
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
        inside_length_correction=None if active else 1.0,
        inside_wall_temperature_correction=None if active else 1.0,
        inside_combined_correction=None if active else 1.0,
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


def _water_steam_effectiveness(
    solution: SteamHeaterSolution | WaterEvaporatorSolution,
    outside_evaluation: OutsideSideEvaluation,
    mass_flow_outside: float,
) -> float:
    delta_T_inside = abs(solution.state_in.T - solution.state_out.T)
    C_inside = (
        solution.Q_total / delta_T_inside
        if delta_T_inside > 1.0e-9
        else math.inf
    )
    C_outside = mass_flow_outside * outside_evaluation.properties_mean.cp
    Q_max = min(C_inside, C_outside) * abs(
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


def _water_evaporation_result(
    solution: WaterEvaporatorSolution,
    *,
    mode: PhaseChangeMode,
    possible: bool,
) -> WaterSteamPhaseChangeResult:
    zones = {zone.kind: zone for zone in solution.zones}

    def zone_value(kind, name, default=None):
        zone = zones.get(kind)
        return default if zone is None else getattr(zone, name)

    active = solution.Q_evaporation > 0.0
    warnings = list(solution.warnings)
    if active:
        warnings.append(_water_evaporation_two_phase_dp_warning())
    elif possible and mode is PhaseChangeMode.DISABLED:
        warnings.append(
            make_warning(
                code=WC.PHASE_CHANGE_DISABLED_BUT_POSSIBLE,
                message=(
                    "inside: full geometry could reach pure-water boiling, "
                    "but the accepted derated duty remains single-phase while "
                    "phase_change_mode=DISABLED."
                ),
                source="steam_integration",
                severity="warning",
            )
        )
    return WaterSteamPhaseChangeResult(
        side="inside",
        mode=mode,
        direction=(
            PhaseChangeDirection.EVAPORATION
            if active
            else PhaseChangeDirection.NONE
        ),
        component="H2O",
        capable=True,
        possible=possible,
        active=active,
        converged=solution.converged,
        method="water_evaporator_multizone_0d",
        state_in=solution.state_in,
        state_midpoint=solution.state_midpoint,
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
        Q_desuperheat=0.0,
        Q_condensation=0.0,
        Q_subcooling=0.0,
        Q_total=solution.Q_total,
        A_desuperheat=0.0,
        A_condensation=0.0,
        A_subcooling=0.0,
        A_total=solution.A_total,
        zone_fraction_desuperheat=0.0,
        zone_fraction_condensation=0.0,
        zone_fraction_subcooling=0.0,
        zone_alpha_desuperheat=None,
        zone_alpha_condensation=None,
        zone_alpha_subcooling=None,
        zone_U_desuperheat=None,
        zone_U_condensation=None,
        zone_U_subcooling=None,
        zone_UA_desuperheat=0.0,
        zone_UA_condensation=0.0,
        zone_UA_subcooling=0.0,
        UA_total=solution.UA_total,
        mass_flow_total=solution.mass_flow_water,
        mass_flux=solution.mass_flux,
        m_dot_condensate=0.0,
        two_phase_pressure_drop_supported=not active,
        two_phase_pressure_drop_status=(
            "not_supported" if active else "not_applicable_single_phase"
        ),
        iterations=solution.iterations,
        root_iterations=solution.root_iterations,
        property_evaluations=solution.property_evaluations,
        warnings=tuple(_deduplicate_warnings(warnings) or ()),
        assumptions=solution.assumptions,
        inside_alpha_equivalent=solution.inside_alpha_equivalent,
        inside_alpha_area_weighted=solution.inside_alpha_area_weighted,
        Q_preheat=solution.Q_preheat,
        Q_evaporation=solution.Q_evaporation,
        Q_superheat=solution.Q_superheat,
        A_preheat=solution.A_preheat,
        A_evaporation=solution.A_evaporation,
        A_superheat=solution.A_superheat,
        zone_fraction_preheat=solution.zone_fraction_preheat,
        zone_fraction_evaporation=solution.zone_fraction_evaporation,
        zone_fraction_superheat=solution.zone_fraction_superheat,
        zone_alpha_preheat=solution.zone_alpha_preheat,
        zone_alpha_evaporation=solution.zone_alpha_evaporation,
        zone_alpha_superheat=solution.zone_alpha_superheat,
        zone_U_preheat=zone_value(WaterEvaporatorZoneKind.PREHEAT, "U"),
        zone_U_evaporation=zone_value(WaterEvaporatorZoneKind.EVAPORATION, "U"),
        zone_U_superheat=zone_value(WaterEvaporatorZoneKind.SUPERHEAT, "U"),
        zone_UA_preheat=solution.UA_preheat,
        zone_UA_evaporation=solution.UA_evaporation,
        zone_UA_superheat=solution.UA_superheat,
        m_dot_evaporated=solution.m_dot_evaporated,
        heat_flux_inner_evaporation=solution.heat_flux_inner_evaporation,
        heat_flux_outer_evaporation=solution.heat_flux_outer_evaporation,
        heat_flux_converged=solution.heat_flux_converged,
        heat_flux_iterations=solution.heat_flux_iterations,
        heat_flux_residual=solution.heat_flux_residual,
        cache_hits=solution.cache_hits,
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


def _closed_balance_for_water_evaporation_rating(*, inside, outside, solution):
    delta_T_water = solution.state_out.T - solution.state_in.T
    C_water = (
        solution.Q_total / delta_T_water
        if delta_T_water > 1.0e-9
        else None
    )
    cp_water = None if C_water is None else C_water / inside.m_dot
    outside_transport = solution.outside_props_mean
    C_outside = outside.m_dot * outside_transport.cp
    inside_closed = ClosedBalanceSide(
        provider=inside.provider,
        p=inside.p,
        m_dot=inside.m_dot,
        T_in=inside.T_in,
        T_out=solution.state_out.T,
        cp_mean=cp_water,
        C=C_water,
    )
    outside_closed = ClosedBalanceSide(
        provider=outside.provider,
        p=outside.p,
        m_dot=outside.m_dot,
        T_in=outside.T_in,
        T_out=solution.T_out_outside,
        cp_mean=outside_transport.cp,
        C=C_outside,
    )
    C_min = C_outside if C_water is None else min(C_water, C_outside)
    Q_max = C_min * (outside.T_in - inside.T_in)
    return ClosedBalance(
        inside=inside_closed,
        outside=outside_closed,
        hot_is_inside=False,
        Q=solution.Q_total,
        Q_max=Q_max,
        effectiveness=solution.Q_total / Q_max,
        warnings=None,
    )


def _resolve_water_evaporation_rating_duty(inside, outside, Q, *, tolerance):
    candidates = []
    if Q is not None:
        if not math.isfinite(Q) or Q <= 0.0:
            raise ValueError("Water-evaporation Rating Q must be positive and finite.")
        candidates.append(("Q", Q))
    outlet_state = inside.water_steam_outlet_state
    if outlet_state is not None:
        candidates.append((
            "inside outlet state",
            inside.m_dot * (outlet_state.h - inside.water_steam_state.h),
        ))
    if outside.T_out is not None:
        props = outside.provider.at(
            T=0.5 * (outside.T_in + outside.T_out), p=outside.p
        )
        transport = getattr(props, "transport", props)
        candidates.append((
            "outside temperature program",
            outside.m_dot * transport.cp * (outside.T_in - outside.T_out),
        ))
    if not candidates:
        raise ValueError(
            "Water-evaporation Rating requires explicit Q, a water/steam "
            "outlet state, or a fully specified opposing-side temperature program."
        )
    for label, value in candidates:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Water-evaporation Rating duty from {label} must increase "
                "tube-side water enthalpy and be positive and finite."
            )
    reference_label, reference = candidates[0]
    for label, value in candidates[1:]:
        scale = max(abs(reference), abs(value), 1.0)
        if abs(reference - value) > tolerance * scale:
            raise ValueError(
                "Over-specified Water-evaporation Rating duties are inconsistent: "
                f"{reference_label}={reference:.9g} W versus {label}={value:.9g} W."
            )
    return reference


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


def _water_evaporation_two_phase_dp_warning() -> ModelWarning:
    return make_warning(
        code=WC.WATER_EVAPORATION_TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED,
        message=(
            "Tube-side pressure drop is not reported because the pure-water "
            "path contains a two-phase evaporation zone."
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
