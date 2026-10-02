# SPDX-License-Identifier: GPL-3.0-only
"""Outside-only Simulation orchestration recovered from c80e779 integration.py.

The historical global and radial solvers remain authoritative. No Rating or
inside-condensation routing is implemented here. The native two-point wall
and wet-area estimates must not be interpreted as an axial solution.
"""
from __future__ import annotations

import math
from dataclasses import replace

from core.common.warnings import ModelWarning, make_warning
from core.geometry.tube import TubeSurfaceType
from core.heat_transfer.thermal_iteration import IterativeThermalState, estimate_wall_temperature_envelope
from core.heat_transfer.outside_dispatch import (
    DEFAULT_FINNED_DP_PROVIDER, DEFAULT_FINNED_HT_PROVIDER, evaluate_outside_hydraulics,
)
from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props
from core.properties.averaging import mean_temperature
from core.models.surface_margin import calculate_surface_margin_factor
from core.phase_change import warning_codes as WC
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.integration import (
    PhaseChangeSettings, ONSET_TEMPERATURE_METHOD, _deduplicate_warnings, _y_h2o,
    build_capability_side_result, capability_only_result, check_single_active_side,
    dew_point_at_ratio, evaluate_side_onset, raise_if_inside_pure_steam_condensation,
)
from core.phase_change.outside_condensation_solver import FrostingNotSupportedError, solve_outside_condensation
from core.phase_change.finned_tube_guard import reject_circular_finned_tube_wet_surface
from core.phase_change.types import PhaseChangeDirection, PhaseChangeMode, PhaseChangeResult
from core.phase_change.water_equilibrium import is_frost_regime, water_partial_pressure
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.wet_surface_fraction import estimate_wet_surface_fraction


def run_legacy_simulation(hx, inside, outside, *, settings, context, **options):
    """Run the historical dry screen and outside solve on installed geometry.

    Generic wet residual tolerances are intentionally not translated: legacy
    convergence retains PhaseChangeSettings and its native residual meanings.
    """
    from core.models.simulation import run_simulation
    from core.phase_change.capability import (
        guard_pure_water_single_phase_provider, reject_unsupported_pure_water_phase_crossing,
    )
    from core.phase_change.steam_integration import (
        reject_outside_pure_water_evaporation_crossing, translate_saturation_crossing_error,
    )

    context.check()
    guarded_inside = guard_pure_water_single_phase_provider(
        inside.provider, T_in=inside.T_in, p=inside.p)
    guarded_outside = guard_pure_water_single_phase_provider(
        outside.provider, T_in=outside.T_in, p=outside.p)
    dry_inside = inside if guarded_inside is inside.provider else replace(inside, provider=guarded_inside)
    dry_outside = outside if guarded_outside is outside.provider else replace(outside, provider=guarded_outside)
    try:
        dry = run_simulation(hx, dry_inside, dry_outside, **options)
    except ValueError as exc:
        translate_saturation_crossing_error(inside, exc, outside)
    reject_outside_pure_water_evaporation_crossing(outside, T_out=dry.T_out_outside)
    for side, outlet in ((inside, dry.T_out_inside), (outside, dry.T_out_outside)):
        reject_unsupported_pure_water_phase_crossing(
            side.provider, T_in=side.T_in, T_out=outlet, p=side.p)
    context.check()
    result = _apply_legacy_outside(
        hx, inside, outside, dry, settings=settings, check=context.check,
        iterate=options.get("iterate", True),
        euler_provider=options.get("euler_provider", "zukauskas"),
        finned_heat_transfer_provider=options.get("finned_heat_transfer_provider", DEFAULT_FINNED_HT_PROVIDER),
        finned_pressure_drop_provider=options.get("finned_pressure_drop_provider", DEFAULT_FINNED_DP_PROVIDER),
    )
    context.check()
    return _attach_legacy_diagnostics(result, settings=settings)


def _attach_legacy_diagnostics(result, *, settings):
    phase = result.outside_phase_change
    surface = dict(
        surface_temperature_min=phase.wall_temperature_min,
        surface_temperature_max=phase.wall_temperature_max,
        surface_temperature_wet_mean=phase.wall_temperature_wet_mean,
        dew_point_in=phase.dew_point_in, dew_point_out=phase.dew_point_out,
        onset_margin=phase.onset_margin_K,
        wet_regime=("DRY" if not phase.active else
                    "FULLY_WET" if math.isclose(phase.wet_surface_fraction, 1.0,
                                               rel_tol=0.0, abs_tol=1e-12)
                    else "PARTIALLY_WET"),
        wet_surface_fraction=phase.wet_surface_fraction,
        wet_surface_fraction_method=phase.wet_surface_fraction_method,
        wall_temperature_mean=phase.wall_temperature_mean,
        wall_envelope_method="two_point_0d_estimate",
        onset_envelope_method="two_point_0d_estimate",
    )
    if result.thermal_state is not None:
        surface["global_core_wall_temperature"] = result.thermal_state.outside_wall_temperature
    wet = phase.wet_finned_surface
    drainage = 0.0
    if phase.active:
        from core.phase_change.condensation_solver_helpers import condensate_enthalpy_flow
        drainage = (wet.condensate_enthalpy_rate if wet is not None else
                    condensate_enthalpy_flow(
                        m_dot_condensate=phase.m_dot_condensate,
                        condensation_mass_tolerance=settings.condensate_tolerance_kg_s,
                        wet_surface_fraction=phase.wet_surface_fraction,
                        wet_area=phase.wet_area,
                        wall_temperature_wet_mean=phase.wall_temperature_wet_mean))
    if wet is not None:
        surface["wall_envelope_method"] = "bulk_mean_radial_surface_extrema_with_0d_cold_zone_offset"
        for name in (
            "fin_base_temperature", "fin_tip_temperature", "wet_dry_boundary_radius",
            "fin_wet_fraction", "primary_surface_temperature", "core_wall_temperature",
            "root_surface_temperature", "condensation_area_fraction",
            "condensation_temperature_offset_K",
        ):
            surface[name] = getattr(wet, name)
    # Do not identify this 0D area fraction as an axial fraction, or invent a
    # separate condensate-film interface temperature absent from the model.
    return replace(result, wet_coil_diagnostics=dict(
        native_method=phase.method,
        surface=surface,
        H_drain=drainage,
        wet_pressure_drop_supported=False,
        outside_dp_reference_only=bool(phase.active),
        pressure_drop_basis="installed_geometry_dry_reference_correlation",
    ))


def _finned_0d_wet_zone_fallback(
    *,
    dew_point_temperature: float | None,
    wall_temperature_min: float | None,
    wall_temperature_max: float | None,
    activation_band_K: float,
) -> tuple[float, float] | None:
    """Return the dry-envelope cold-zone area and temperature offset.

    This estimate is passed to the wet circular-fin solver but used only if
    the ordinary bulk-mean radial response is entirely dry.  It reconciles a
    cold endpoint that activated AUTO with the global 0D surface response by
    reusing the established linear wet-area convention; it is not axial,
    row, circuit or pass marching.
    """

    values = (
        dew_point_temperature,
        wall_temperature_min,
        wall_temperature_max,
    )
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    assert dew_point_temperature is not None
    assert wall_temperature_min is not None
    assert wall_temperature_max is not None
    wall_temperature_mean = 0.5 * (
        wall_temperature_min + wall_temperature_max
    )
    estimate = estimate_wet_surface_fraction(
        dew_point_temperature=dew_point_temperature,
        wall_temperature_min=wall_temperature_min,
        wall_temperature_mean=wall_temperature_mean,
        wall_temperature_max=wall_temperature_max,
        activation_band_K=activation_band_K,
    )
    if (
        estimate.wet_surface_fraction <= 0.0
        or estimate.wall_temperature_wet_mean is None
    ):
        return None
    return (
        estimate.wet_surface_fraction,
        estimate.wall_temperature_wet_mean - wall_temperature_mean,
    )


def _apply_legacy_outside(
    hx,
    inside,
    outside,
    dry_result,
    *,
    iterate: bool,
    euler_provider: str = "zukauskas",
    finned_heat_transfer_provider: object = DEFAULT_FINNED_HT_PROVIDER,
    finned_pressure_drop_provider: object = DEFAULT_FINNED_DP_PROVIDER,
    settings: PhaseChangeSettings | None = None,
    check=None,
):
    """Return a new ``HXSimulationResult`` with phase-change results applied.

    Args:
        hx: ``BareTubeHeatExchanger``.
        inside, outside: the ``HXSideInput`` passed to ``.simulate()``.
        dry_result: the already-computed sensible-only ``HXSimulationResult``
            (the dry baseline).
        iterate: the ``iterate`` flag ``.simulate()`` was called with (needed
            for the "outside condensation requires iterate=True" guard).
        settings: bundled ``phase_change_*`` settings; defaults if omitted.

    Raises:
        WetCoilProviderUnsupportedError for inside condensation,
        MultiplePhaseChangeSidesError, ValueError for active AUTO with
        iterate=False. Frost retains the historical dry warning result.
    """
    settings = settings or PhaseChangeSettings()

    inside_capability = detect_phase_change_capability(inside.provider)
    outside_capability = detect_phase_change_capability(outside.provider)

    raise_if_inside_pure_steam_condensation(inside, dry_result)

    if not inside_capability.capable and not outside_capability.capable:
        return replace(
            dry_result,
            inside_phase_change=capability_only_result(
                "inside", inside.phase_change_mode, inside_capability, dry_result.q,
            ),
            outside_phase_change=capability_only_result(
                "outside", outside.phase_change_mode, outside_capability, dry_result.q,
            ),
        )

    thermal_state = dry_result.thermal_state
    envelope = dry_result.wall_temperature_envelope

    inside_onset, inside_dew_point, inside_wall_min, inside_wall_mean, inside_wall_max = evaluate_side_onset(
        side="inside", mode=inside.phase_change_mode, capability=inside_capability, p=inside.p,
        thermal_state=thermal_state, envelope=envelope, settings=settings,
    )
    outside_onset, outside_dew_point, outside_wall_min, outside_wall_mean, outside_wall_max = evaluate_side_onset(
        side="outside", mode=outside.phase_change_mode, capability=outside_capability, p=outside.p,
        thermal_state=thermal_state, envelope=envelope, settings=settings,
    )

    inside_possible = bool(inside_onset is not None and inside_onset.possible)
    outside_possible = bool(outside_onset is not None and outside_onset.possible)
    inside_near_onset = bool(inside_onset is not None and inside_onset.near_onset)
    outside_near_onset = bool(outside_onset is not None and outside_onset.near_onset)

    # "Auto possible" here means "clearly past onset" (OnsetDecision.active),
    # not merely "possible" (which also covers the near-onset band that must
    # stay on the dry path -- see core.phase_change.regime).
    inside_auto_possible = (
        inside_onset is not None and inside_onset.active and inside.phase_change_mode is PhaseChangeMode.AUTO
    )
    outside_auto_possible = (
        outside_onset is not None and outside_onset.active and outside.phase_change_mode is PhaseChangeMode.AUTO
    )

    reject_circular_finned_tube_wet_surface(
        hx,
        inside_active=inside_auto_possible,
        outside_active=outside_auto_possible,
        context="wet-gas condensation",
        outside_capability=outside_capability,
        direction=PhaseChangeDirection.CONDENSATION,
    )

    check_single_active_side(inside_auto_possible, outside_auto_possible, iterate=iterate)

    if inside_auto_possible:
        from core.phase_change.wet_coil_provider import WetCoilProviderUnsupportedError
        raise WetCoilProviderUnsupportedError("Legacy outside provider cannot solve inside condensation")

    inside_result = build_capability_side_result(
        side="inside", mode=inside.phase_change_mode, capability=inside_capability,
        possible=inside_possible, near_onset=inside_near_onset,
        dew_point=inside_dew_point, p=inside.p,
        m_dot_gas=inside.m_dot,
        onset=inside_onset, wall_temperature_min=inside_wall_min,
        wall_temperature_mean=inside_wall_mean, wall_temperature_max=inside_wall_max,
        Q_sensible_actual=dry_result.q,
    )

    if not outside_auto_possible:
        outside_result = build_capability_side_result(
            side="outside", mode=outside.phase_change_mode, capability=outside_capability,
            possible=outside_possible, near_onset=outside_near_onset,
            dew_point=outside_dew_point, p=outside.p,
            m_dot_gas=outside.m_dot,
            onset=outside_onset, wall_temperature_min=outside_wall_min,
            wall_temperature_mean=outside_wall_mean, wall_temperature_max=outside_wall_max,
            Q_sensible_actual=dry_result.q,
        )
        return replace(dry_result, inside_phase_change=inside_result, outside_phase_change=outside_result)

    # --- Outside condensation is the active, supported path -----------------
    p_h2o_in = water_partial_pressure(
        _y_h2o(outside_capability), outside.p
    )
    if is_frost_regime(p_h2o_in):
        outside_result = replace(
            build_capability_side_result(
                side="outside", mode=outside.phase_change_mode, capability=outside_capability,
                possible=True, near_onset=False, dew_point=None, p=outside.p,
                m_dot_gas=outside.m_dot,
                onset=outside_onset, wall_temperature_min=outside_wall_min,
                wall_temperature_mean=outside_wall_mean, wall_temperature_max=outside_wall_max,
                Q_sensible_actual=dry_result.q,
            ),
            warnings=(
                make_warning(
                    code=WC.FROSTING_NOT_SUPPORTED,
                    message=(
                        "outside: the equilibrium dew point for the inlet "
                        "water content is at/below the water triple point "
                        "(frost/ice regime); liquid condensation is not "
                        "applicable and frosting is not modelled in v0.6.0. "
                        "Returning the sensible-only dry baseline."
                    ),
                    source="phase_change_integration",
                    severity="warning",
                ),
            ),
        )
        return replace(dry_result, inside_phase_change=inside_result, outside_phase_change=outside_result)

    m_dot_dry_carrier = outside.m_dot / (1.0 + outside_capability.W_in)
    finned_wet_zone_fallback = None
    if (
        getattr(hx.bundle.tube, "surface_type", None)
        is TubeSurfaceType.CIRCULAR_FINNED
    ):
        finned_wet_zone_fallback = _finned_0d_wet_zone_fallback(
            dew_point_temperature=outside_dew_point,
            wall_temperature_min=outside_wall_min,
            wall_temperature_max=outside_wall_max,
            activation_band_K=settings.activation_band_K,
        )

    try:
        solution = solve_outside_condensation(
            hx,
            check=check,
            inside_provider=inside.provider,
            m_dot_inside=inside.m_dot,
            T_in_inside=inside.T_in,
            p_inside=inside.p,
            outside_capability=outside_capability,
            m_dot_dry_carrier=m_dot_dry_carrier,
            T_in_outside=outside.T_in,
            p_outside=outside.p,
            T_out_inside_init=dry_result.T_out_inside,
            T_out_outside_init=dry_result.T_out_outside,
            euler_provider=euler_provider,
            finned_heat_transfer_provider=finned_heat_transfer_provider,
            finned_condensation_area_fraction=(
                None
                if finned_wet_zone_fallback is None
                else finned_wet_zone_fallback[0]
            ),
            finned_condensation_temperature_offset_K=(
                0.0
                if finned_wet_zone_fallback is None
                else finned_wet_zone_fallback[1]
            ),
            lewis_number=settings.lewis_number,
            activation_band_K=settings.activation_band_K,
            max_iterations=settings.max_iterations,
            temperature_tolerance_K=settings.temperature_tolerance_K,
            relative_Q_tolerance=settings.relative_Q_tolerance,
            water_ratio_tolerance=settings.water_ratio_tolerance,
            condensate_tolerance_kg_s=settings.condensate_tolerance_kg_s,
            wall_temperature_tolerance_K=settings.wall_temperature_tolerance_K,
            wet_fraction_tolerance=settings.wet_fraction_tolerance,
            relaxation_factor=settings.relaxation_factor,
        )
    except FrostingNotSupportedError as exc:
        outside_result = replace(
            build_capability_side_result(
                side="outside", mode=outside.phase_change_mode, capability=outside_capability,
                possible=True, near_onset=False, dew_point=outside_dew_point, p=outside.p,
                m_dot_gas=outside.m_dot,
                onset=outside_onset, wall_temperature_min=outside_wall_min,
                wall_temperature_mean=outside_wall_mean, wall_temperature_max=outside_wall_max,
                Q_sensible_actual=dry_result.q,
            ),
            warnings=(
                make_warning(
                    code=WC.FROSTING_NOT_SUPPORTED,
                    message=f"outside: {exc}",
                    source="phase_change_integration",
                    severity="warning",
                ),
            ),
        )
        return replace(dry_result, inside_phase_change=inside_result, outside_phase_change=outside_result)

    if solution.wet_finned_surface is not None and solution.m_dot_condensate <= 0.0:
        # The dry-baseline onset screen activated this call (spec section 8),
        # but the converged nonlinear radial wet-fin field found no point
        # below the local saturation line, even with the 0D endpoint wet-
        # zone fallback (see solve_outside_condensation). This is a
        # legitimate near-boundary collapse, not a solver contradiction --
        # report it as a valid dry AUTO result (exactly the same shape as
        # the "not outside_auto_possible" dry path above, so the v0.7.4
        # legacy dry circular-fin result is reproduced bit-for-bit) with a
        # diagnostic warning rather than forcing an active wet state or
        # failing an otherwise physically valid call.
        outside_result = replace(
            build_capability_side_result(
                side="outside", mode=outside.phase_change_mode, capability=outside_capability,
                possible=True, near_onset=False, dew_point=outside_dew_point, p=outside.p,
                m_dot_gas=outside.m_dot,
                onset=outside_onset, wall_temperature_min=outside_wall_min,
                wall_temperature_mean=outside_wall_mean, wall_temperature_max=outside_wall_max,
                Q_sensible_actual=dry_result.q,
            ),
            warnings=tuple(solution.warnings),
        )
        return replace(dry_result, inside_phase_change=inside_result, outside_phase_change=outside_result)

    m_dot_water_vapor_in = m_dot_dry_carrier * outside_capability.W_in
    m_dot_water_vapor_out = m_dot_dry_carrier * solution.W_out
    m_dot_gas_in = outside.m_dot
    m_dot_gas_out = m_dot_dry_carrier + m_dot_water_vapor_out

    mass_balance_error = m_dot_water_vapor_in - (m_dot_water_vapor_out + solution.m_dot_condensate)
    energy_balance_error = solution.Q_total - (solution.Q_sensible + solution.Q_latent)

    # The outside wall envelope, wet fraction/temperature, area and
    # wet-surface saturation come straight from the solver's final
    # per-iteration state.  These are the values that actually drove mass
    # transfer, rather than a different post-hoc envelope.
    warnings_list: list[ModelWarning] = list(solution.warnings)
    wet_dp_warning = None
    if solution.wet_finned_surface is None:
        warnings_list.append(
            make_warning(
                code=WC.WET_SURFACE_FRACTION_0D_ESTIMATE,
                message=(
                    "outside: wet_surface_fraction and "
                    "wall_temperature_wet_mean are 0D linear estimates "
                    f"({solution.wet_surface_fraction_method}) based on a cheap "
                    "two-point (inlet/outlet) wall-temperature estimate, not a "
                    "spatially resolved (1D/segmented) wetted-area result."
                ),
                source="phase_change_integration",
                severity="info",
            )
        )
    else:
        if (
            solution.wet_finned_surface.condensation_area_fraction < 1.0
            or solution.wet_finned_surface
            .condensation_temperature_offset_K < 0.0
        ):
            warnings_list.append(
                make_warning(
                    code=WC.WET_SURFACE_FRACTION_0D_ESTIMATE,
                    message=(
                        "outside: the bulk-mean radial circular-fin response "
                        "was dry after endpoint onset, so condensation uses "
                        "the dry exposed-skin envelope's linear 0D cold-zone "
                        "area and representative temperature. This is not a "
                        "longitudinal, row, circuit or pass-resolved model."
                    ),
                    source="phase_change_integration",
                    severity="info",
                )
            )
        wet_dp_warning = make_warning(
            code=WC.CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY,
            message=(
                "outside: wet circular-finned thermal performance and "
                "condensate are solved, but no wet-surface pressure-drop "
                "correction is available; the reported finned-bank pressure "
                "drop is the dry Robinson-Briggs reference only."
            ),
            source="phase_change_integration",
            severity="warning",
        )
        warnings_list.append(wet_dp_warning)
    warnings_list.append(
        make_warning(
            code=WC.OUTSIDE_CONDENSATION_DETECTED,
            message=(
                "outside: the dry sensible-only baseline showed the outside "
                "tube-wall surface running below the water dew point; "
                "partial H2O condensation was solved for this call."
            ),
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list.append(
        make_warning(
            code=WC.LEWIS_NUMBER_ASSUMED,
            message=(
                f"outside: mass transfer used the Chilton-Colburn analogy "
                f"with lewis_number={settings.lewis_number:g} (a configurable "
                "first-model assumption, not a universal constant)."
            ),
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list.append(
        make_warning(
            code=WC.FULLY_DRAINED_CONDENSATE_ASSUMED,
            message=(
                "outside: condensate is assumed to be fully drained from the "
                "gas stream, leaving as saturated liquid at the representative "
                "wet-surface temperature. Film retention/re-entrainment are "
                "not modelled (v0.6.0)."
            ),
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list.append(
        make_warning(
            code=WC.CONDENSATE_FILM_RESISTANCE_NOT_MODELLED,
            message="outside: condensate film thermal resistance is not modelled in v0.6.0.",
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list.append(
        make_warning(
            code=WC.CONDENSATE_FILM_HYDRAULICS_NOT_MODELLED,
            message="outside: condensate film hydraulics are not modelled in v0.6.0.",
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list.append(
        make_warning(
            code=WC.REENTRAINMENT_NOT_MODELLED,
            message="outside: droplet carryover / re-entrainment is not modelled in v0.6.0.",
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list = _deduplicate_warnings(warnings_list)

    outside_result = PhaseChangeResult(
        side="outside",
        mode=outside.phase_change_mode,
        direction=PhaseChangeDirection.CONDENSATION,
        component=outside_capability.component,
        capable=True,
        possible=True,
        active=True,
        near_onset=False,
        onset_margin_K=None if outside_onset is None else outside_onset.margin_K,
        onset_wall_temperature=outside_wall_min,
        onset_temperature_method=ONSET_TEMPERATURE_METHOD,
        converged=solution.converged,
        iterations=solution.iterations,
        method=(
            (
                "outside_condensation_0d_wet_annular_fin_fvm_"
                "with_endpoint_wet_zone_fallback"
                if (
                    solution.wet_finned_surface is not None
                    and (
                        solution.wet_finned_surface
                        .condensation_area_fraction < 1.0
                        or solution.wet_finned_surface
                        .condensation_temperature_offset_K < 0.0
                    )
                )
                else "outside_condensation_0d_wet_annular_fin_fvm"
            )
            if solution.wet_finned_surface is not None
            else "outside_condensation_0d_bulk_mean"
        ),
        W_in=outside_capability.W_in,
        W_mid=0.5 * (outside_capability.W_in + solution.W_out),
        W_out=solution.W_out,
        m_dot_dry_carrier=m_dot_dry_carrier,
        m_dot_water_vapor_in=m_dot_water_vapor_in,
        m_dot_water_vapor_out=m_dot_water_vapor_out,
        m_dot_condensate=solution.m_dot_condensate,
        m_dot_gas_in=m_dot_gas_in,
        m_dot_gas_out=m_dot_gas_out,
        dew_point_in=outside_dew_point,
        dew_point_out=dew_point_at_ratio(outside_capability, solution.W_out, p=outside.p),
        wall_temperature_mean=(
            solution.wet_finned_surface
            .outside_surface_temperature_area_mean
            if solution.wet_finned_surface is not None
            else solution.T_wall_outside
        ),
        wall_temperature_min=solution.wall_temperature_min,
        wall_temperature_max=solution.wall_temperature_max,
        wall_temperature_wet_mean=solution.wall_temperature_wet_mean,
        wet_surface_fraction=solution.wet_surface_fraction,
        wet_surface_fraction_method=solution.wet_surface_fraction_method,
        wet_area=solution.wet_area,
        outside_total_area=solution.outside_total_area,
        W_sat_wet_surface=solution.W_sat_wet_surface,
        alfa_dry=solution.alfa_o_dry,
        alfa_effective=solution.alfa_o_effective,
        lewis_number=settings.lewis_number,
        Q_sensible=solution.Q_sensible,
        Q_latent=solution.Q_latent,
        Q_total=solution.Q_total,
        mass_balance_error=mass_balance_error,
        energy_balance_error=energy_balance_error,
        residuals=dict(solution.residuals),
        assumptions=(
            solution.wet_finned_surface.assumptions
            if solution.wet_finned_surface is not None
            else (
                "bulk_mean_property_evaluation_0d",
                "fully_drained_liquid_condensate",
                "lewis_number_chilton_colburn_analogy",
                "dry_gas_composition_unchanged_by_condensation",
                "wet_surface_fraction_two_point_inlet_outlet_estimate",
                "wet_surface_temperature_linear_envelope_estimate",
            )
        ),
        warnings=tuple(warnings_list),
        wet_finned_surface=solution.wet_finned_surface,
    )

    T_mean_inside = mean_temperature(inside.T_in, solution.T_out_inside)
    T_mean_outside = mean_temperature(outside.T_in, solution.T_out_outside)

    W_mean = 0.5 * (outside_capability.W_in + solution.W_out)
    outside_provider_inlet = wet_gas_provider_at_water_ratio(
        outside_capability, outside_capability.W_in
    )
    outside_provider_final = wet_gas_provider_at_water_ratio(
        outside_capability, W_mean
    )
    outside_provider_outlet = wet_gas_provider_at_water_ratio(
        outside_capability, solution.W_out
    )
    outside_props_inlet = outside_provider_inlet.at(T=outside.T_in, p=outside.p)
    outside_props_outlet = outside_provider_outlet.at(
        T=solution.T_out_outside, p=outside.p
    )

    # hot_stream/cold_stream only feed this snapshot's OWN (non-authoritative,
    # sensible-only) Q/T_hot_out/T_cold_out sub-fields -- hydraulics below use
    # the explicit *_temperature_out arguments (the wet solver's converged,
    # latent-inclusive outlet temperatures) directly, matching how the
    # existing dry `run_simulation` final snapshot already treats this
    # sub-result as diagnostic-only (see HXSimulationResult docstring).
    hot_is_inside = inside.T_in >= outside.T_in
    from core.heat_transfer.streams import SensibleHeatStream

    inside_stream = SensibleHeatStream(
        C=inside.m_dot * solution.inside_bulk_props.cp, T_in=inside.T_in
    )
    outside_stream = SensibleHeatStream(
        C=m_dot_dry_carrier * (1.0 + W_mean) * solution.outside_bulk_props.cp,
        T_in=outside.T_in,
    )
    hot_stream, cold_stream = (
        (inside_stream, outside_stream) if hot_is_inside else (outside_stream, inside_stream)
    )

    final_result = hx.solve(
        hot_stream=hot_stream,
        cold_stream=cold_stream,
        m_dot_tube_side=inside.m_dot,
        tube_side_props=to_internal_fluid_props(solution.inside_bulk_props),
        tube_side_provider=inside.provider,
        tube_side_temperature_in=inside.T_in,
        tube_side_temperature_out=solution.T_out_inside,
        tube_side_pressure=inside.p,
        m_dot_outside=m_dot_dry_carrier * (1.0 + W_mean),
        outside_props=to_outside_fluid_props(solution.outside_bulk_props),
        outside_provider=outside_provider_final,
        outside_temperature_in=outside.T_in,
        outside_temperature_out=solution.T_out_outside,
        outside_pressure=outside.p,
        flow_arrangement=None,
        euler_provider=euler_provider,
        finned_heat_transfer_provider=finned_heat_transfer_provider,
        finned_pressure_drop_provider=finned_pressure_drop_provider,
    )

    # Refresh the outside hydraulic snapshot using the exact wet states
    # already established by the coupled solve: W_in at inlet, arithmetic
    # (T, W) means at midpoint, and W_out at outlet. The midpoint transport
    # properties are the wet solver's own final bulk properties rather than
    # a presentation-layer re-evaluation. Actual per-point gas-phase mass
    # flow is used for the reported face flux and signed acceleration term;
    # drag/Reynolds retain the existing bulk-mean reference-flow convention.
    from dataclasses import replace as _replace

    from core.pressure_drop.flow_path import build_outside_pressure_drop_result

    outside_bank_hydraulic = evaluate_outside_hydraulics(
        bundle=hx.bundle,
        m_dot=m_dot_dry_carrier * (1.0 + W_mean),
        inlet_props=outside_props_inlet,
        midpoint_props=solution.outside_bulk_props,
        outlet_props=outside_props_outlet,
        temperature_in=outside.T_in,
        temperature_out=solution.T_out_outside,
        pressure=outside.p,
        euler_provider=euler_provider,
        finned_pressure_drop_provider=finned_pressure_drop_provider,
        m_dot_inlet=m_dot_gas_in,
        m_dot_midpoint=m_dot_dry_carrier * (1.0 + W_mean),
        m_dot_outlet=m_dot_gas_out,
    )
    outside_bank_hydraulic = _replace(
        outside_bank_hydraulic,
        midpoint_method="arithmetic_temperature_and_water_ratio",
    )
    final_result = _replace(
        final_result,
        outside_side_hydraulic=_replace(
            final_result.outside_side_hydraulic,
            dp_total=outside_bank_hydraulic.dp_total,
            Re=outside_bank_hydraulic.midpoint.reynolds,
            v=outside_bank_hydraulic.midpoint.face_velocity,
            tube_bank=outside_bank_hydraulic,
        ),
        outside_side_pressure_drop=_replace(
            final_result.outside_side_pressure_drop,
            tube_bank=outside_bank_hydraulic,
            flow_path=build_outside_pressure_drop_result(outside_bank_hydraulic),
        ),
    )
    wet_finned_diagnostics = (
        solution.finned_tube_diagnostics
        if solution.wet_finned_surface is not None
        else final_result.finned_tube_diagnostics
    )
    if solution.wet_finned_surface is not None:
        if wet_finned_diagnostics is None:
            raise ValueError(
                "Active wet circular-fin solve lost its finned-tube diagnostics."
            )
        assert wet_dp_warning is not None
        hydraulic_midpoint = outside_bank_hydraulic.midpoint
        wet_finned_diagnostics = _replace(
            wet_finned_diagnostics,
            wet_surface=solution.wet_finned_surface,
            pressure_drop_coefficient=hydraulic_midpoint.coefficient,
            pressure_drop_coefficient_definition=(
                outside_bank_hydraulic.coefficient_definition
            ),
            outside_dp_drag=outside_bank_hydraulic.dp_drag,
            outside_dp_acceleration=outside_bank_hydraulic.dp_acceleration,
            outside_dp_total=outside_bank_hydraulic.dp_total,
            outside_dp_dry_reference=outside_bank_hydraulic.dp_total,
            wet_pressure_drop_supported=False,
            outside_dp_reference_only=True,
            pressure_drop_metadata=outside_bank_hydraulic.metadata,
            warnings=tuple(
                _deduplicate_warnings(
                    [
                        *wet_finned_diagnostics.warnings,
                        *outside_bank_hydraulic.warnings,
                        wet_dp_warning,
                    ]
                )
            ),
        )
        final_result = _replace(
            final_result,
            finned_tube_diagnostics=wet_finned_diagnostics,
            warnings=_deduplicate_warnings(
                [*(final_result.warnings or []), wet_dp_warning]
            ),
        )

    envelope_wet = estimate_wall_temperature_envelope(
        hx,
        m_dot_inside=inside.m_dot,
        m_dot_outside=m_dot_dry_carrier * (1.0 + W_mean),
        inside_provider=inside.provider,
        outside_provider=outside_provider_final,
        inside_inlet_temperature=inside.T_in,
        inside_outlet_temperature=solution.T_out_inside,
        outside_inlet_temperature=outside.T_in,
        outside_outlet_temperature=solution.T_out_outside,
        p_inside=inside.p,
        p_outside=outside.p,
        euler_provider=euler_provider,
        finned_heat_transfer_provider=finned_heat_transfer_provider,
    )
    # Keep this public endpoint envelope internally consistent with its
    # four audit probes.  The exact centred two-point envelope that drove
    # wet area and mass transfer is reported separately, without
    # post-processing, on ``outside_phase_change``.

    # Fix (v0.6.0 patch, spec section 16): build a thermal_state consistent
    # with the wet solution itself -- previously this field was left as the
    # *dry baseline's* thermal_state while wall_temperature_envelope and
    # outside_phase_change already reflected the wet solution, an internal
    # inconsistency. alfa_o here is the *effective* (latent-inclusive)
    # coefficient, not alfa_dry, so that UA/U reconstruct from alfa_i/alfa_o
    # exactly as thermal_iteration.solve_iterative_thermal_state's own
    # reconstruction self-check expects; alfa_outside_dry remains separately
    # available on outside_phase_change.alfa_dry.
    wet_alpha = (
        solution.wet_finned_surface
        .outside_alpha_wet_effective_gross_core_basis
        if solution.wet_finned_surface is not None
        else solution.alfa_o_effective
    )
    wet_alpha_basis = (
        solution.wet_finned_surface.outside_alpha_wet_effective_basis
        if solution.wet_finned_surface is not None
        else (
            "gross_outside_area_and_bulk_gas_to_core_wall_"
            "temperature_difference"
        )
    )
    dry_effective_alpha = (
        wet_finned_diagnostics.outside_alpha_effective_gross
        if solution.wet_finned_surface is not None
        else solution.alfa_o_effective
    )
    wet_thermal_state = IterativeThermalState(
        inside_bulk_temperature=T_mean_inside,
        outside_bulk_temperature=T_mean_outside,
        inside_wall_temperature=solution.T_wall_inside,
        outside_wall_temperature=solution.T_wall_outside,
        inside_bulk_props=solution.inside_bulk_props,
        inside_wall_props=solution.inside_wall_props,
        outside_bulk_props=solution.outside_bulk_props,
        outside_wall_props=solution.outside_wall_props,
        alfa_i=solution.alfa_i,
        alfa_o=solution.alfa_o_effective,
        U=solution.U_effective,
        UA=solution.UA_effective,
        iterations=solution.iterations,
        converged=solution.converged,
        residual=solution.residuals.get("T_wall_outside_K", math.inf),
        diagnostics=solution.diagnostics,
        outside_alpha_physical=solution.alfa_o_dry,
        outside_alpha_effective_gross=dry_effective_alpha,
        outside_alpha_wet_effective_gross_core_basis=wet_alpha,
        outside_alpha_wet_effective_basis=wet_alpha_basis,
        inside_provider_name=type(inside.provider).__name__,
        outside_provider_name=type(outside_provider_final).__name__,
        warnings=solution.warnings,
        finned_tube_diagnostics=wet_finned_diagnostics,
    )

    # The inactive inside side's Q_sensible/Q_total were built above from the
    # dry-baseline duty (before this solve ran); the actual whole-exchanger
    # duty now includes outside's latent contribution. Refresh them to the
    # converged total so inside_phase_change stays consistent with q below
    # without requiring callers to branch on active (spec section 6).
    inside_result = replace(
        inside_result, Q_sensible=solution.Q_total, Q_total=solution.Q_total,
    )

    # Keep the phase solver and its achieved duty unchanged. The result-level
    # process UA applies the same Simulation derating contract to the final
    # wet working-state UA, then reconstructs EMTD and the canonical margin.
    UA_actual = wet_thermal_state.UA
    UA_process = UA_actual / (1.0 + dry_result.surface_margin)
    surface_margin_factor = calculate_surface_margin_factor(
        UA_actual=UA_actual,
        UA_process=UA_process,
    )

    return replace(
        dry_result,
        converged=solution.converged,
        iterations=solution.iterations,
        T_mean_inside=T_mean_inside,
        T_mean_outside=T_mean_outside,
        inside_props_mean=solution.inside_bulk_props,
        outside_props_mean=solution.outside_bulk_props,
        inside_velocity_mean=final_result.tube_side_thermal.v,
        outside_velocity_mean=final_result.outside_side_thermal.v,
        inside_Re_mean=final_result.tube_side_thermal.Re,
        outside_Re_mean=final_result.outside_side_thermal.Re,
        inside_Pr_mean=final_result.tube_side_thermal.Pr,
        outside_Pr_mean=final_result.outside_side_thermal.Pr,
        inside_alfa_mean=wet_thermal_state.alfa_i,
        outside_alfa_mean=wet_thermal_state.alfa_o,
        U_mean=wet_thermal_state.U,
        UA=UA_actual,
        EMTD=abs(solution.Q_total) / UA_process,
        q=solution.Q_total,
        T_out_inside=solution.T_out_inside,
        T_out_outside=solution.T_out_outside,
        Q_full=solution.Q_total,
        Q_derated=solution.Q_total,
        overdesign_factor=surface_margin_factor,
        final_result=final_result,
        thermal_state=wet_thermal_state,
        wall_temperature_envelope=envelope_wet,
        inside_phase_change=inside_result,
        outside_phase_change=outside_result,
        warnings=_deduplicate_warnings(
            [*(dry_result.warnings or []), *warnings_list]
        ),
    )
