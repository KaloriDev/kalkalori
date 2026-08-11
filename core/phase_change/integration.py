# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Simulation-side orchestration: capability -> regime -> (dry | wet) result.

This is the single place that wires ``core.phase_change.capability``,
``core.phase_change.regime`` and the side-specific condensation solvers
together into the automatic behavior extended in v0.6.1
spec. ``core.models.bare_tube.BareTubeHeatExchanger.simulate`` calls
``apply_phase_change`` *after* running the ordinary sensible-only
``core.models.simulation.run_simulation`` (the dry baseline) -- so the dry
path in ``run_simulation`` itself is untouched, and every sensible-only
regression stays bit-for-bit stable (no capable side => this module hands
the dry result straight back, only stamping ``inside_phase_change``/
``outside_phase_change`` onto it via ``dataclasses.replace``).

Stable regime, single active side (spec sections 18-19)
-------------------------------------------------------
The regime for each capable side is decided exactly once, from the dry
baseline (``core.phase_change.regime.decide_regime``), before any wet
solve starts; there is no per-iteration dry/wet branch anywhere in this
package (the actual iteration lives entirely inside
``core.phase_change.outside_condensation_solver``, which is only ever
invoked once regime selection is already final). At most one side may be
an *active* condensing side per call:

- both sides active under AUTO -> ``MultiplePhaseChangeSidesError``
  (``MULTIPLE_PHASE_CHANGE_SIDES_NOT_SUPPORTED``), checked first;
- exactly one active wet-gas side under AUTO -> its inside or outside wet
  solver runs and the regime remains locked for the solve.

``PhaseChangeMode.DISABLED`` never contributes to either of the above: a
disabled side can be capable and possible without blocking the other side,
it simply reports ``active=False`` with a
``PHASE_CHANGE_DISABLED_BUT_POSSIBLE`` warning and leaves the sensible-only
numbers untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from core.common.warnings import ModelWarning, make_warning
from core.heat_transfer.thermal_iteration import (
    IterativeThermalState,
    estimate_wall_temperature_envelope,
    solve_iterative_thermal_state,
)
from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props
from core.properties.averaging import mean_temperature
from core.heat_transfer.outside_flow import calculate_outside_tube_bank_hydraulics

from core.phase_change import warning_codes as WC
from core.phase_change.capability import (
    PureWaterPhaseChangeProviderNotSupportedError,
    detect_phase_change_capability,
    is_pure_water_provider,
    pure_water_provider_kind,
    pure_water_saturation_temperature,
)
from core.phase_change.mass_heat_transfer import mass_transfer_coefficient  # noqa: F401 (re-export for callers)
from core.phase_change.outside_condensation_solver import (
    FrostingNotSupportedError,
    solve_outside_condensation,
)
from core.phase_change.inside_condensation_solver import solve_inside_condensation
from core.phase_change.condensation_solver_helpers import condensate_enthalpy_flow
from core.phase_change.regime import (
    OnsetDecision,
    decide_regime,
    evaluate_condensation_onset,
    representative_wall_temperature,
    validate_onset_settings,
)
from core.phase_change.types import PhaseChangeCapability, PhaseChangeDirection, PhaseChangeMode, PhaseChangeResult
from core.phase_change.water_equilibrium import is_frost_regime, water_dew_point, water_partial_pressure
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.wet_gas_enthalpy import WetGasEnthalpyEvaluator

DEFAULT_ONSET_TOLERANCE_K = 0.0
DEFAULT_ACTIVATION_BAND_K = 0.5

ONSET_TEMPERATURE_METHOD = "wall_temperature_min_envelope_estimate"


class InsideCondensationNotSupportedError(RuntimeError):
    """Deprecated v0.6.0 compatibility exception; no longer raised in v0.6.1."""

    warning_code = WC.INSIDE_CONDENSATION_NOT_SUPPORTED


class MultiplePhaseChangeSidesError(RuntimeError):
    """Raised when both sides show an actionable (AUTO + possible) phase
    change simultaneously. At most one active phase-changing side is
    supported per call (see spec section 19)."""

    warning_code = WC.MULTIPLE_PHASE_CHANGE_SIDES_NOT_SUPPORTED


class PureWaterSteamCondensationNotSupportedError(RuntimeError):
    """Legacy direct-adapter guard; public v0.6.2 calls bypass this path."""

    warning_code = WC.PURE_WATER_STEAM_CONDENSATION_NOT_SUPPORTED


@dataclass(frozen=True)
class PhaseChangeSettings:
    """Bundled ``phase_change_*`` iteration/onset settings.

    Individual keyword parameters are threaded through
    ``BareTubeHeatExchanger.simulate``/``.rate`` (matching the existing
    thermal-iteration idiom, see ``core.heat_transfer.thermal_iteration``);
    this dataclass is only an internal convenience for passing the bundle
    around inside ``core.phase_change``.
    """

    onset_tolerance_K: float = DEFAULT_ONSET_TOLERANCE_K
    activation_band_K: float = DEFAULT_ACTIVATION_BAND_K
    lewis_number: float = 1.0
    max_iterations: int = 50
    temperature_tolerance_K: float = 0.05
    relative_Q_tolerance: float = 1e-4
    water_ratio_tolerance: float = 1e-6
    condensate_tolerance_kg_s: float = 1e-8
    wall_temperature_tolerance_K: float = 0.05
    wet_fraction_tolerance: float = 1e-3
    relaxation_factor: float = 0.5

    def __post_init__(self) -> None:
        validate_onset_settings(self.onset_tolerance_K, self.activation_band_K)
        if not math.isfinite(self.lewis_number) or self.lewis_number <= 0.0:
            raise ValueError("lewis_number must be a positive finite value.")
        if self.max_iterations <= 0:
            raise ValueError("phase_change_max_iterations must be > 0.")
        if not math.isfinite(self.temperature_tolerance_K) or self.temperature_tolerance_K <= 0.0:
            raise ValueError("phase_change_temperature_tolerance_K must be > 0.")
        if not math.isfinite(self.relative_Q_tolerance) or self.relative_Q_tolerance <= 0.0:
            raise ValueError("phase_change_relative_Q_tolerance must be > 0.")
        if not math.isfinite(self.water_ratio_tolerance) or self.water_ratio_tolerance <= 0.0:
            raise ValueError("phase_change_water_ratio_tolerance must be > 0.")
        if not math.isfinite(self.condensate_tolerance_kg_s) or self.condensate_tolerance_kg_s < 0.0:
            raise ValueError("phase_change_condensate_tolerance_kg_s must be >= 0.")
        if not math.isfinite(self.wall_temperature_tolerance_K) or self.wall_temperature_tolerance_K <= 0.0:
            raise ValueError("phase_change_wall_temperature_tolerance_K must be > 0.")
        if not math.isfinite(self.wet_fraction_tolerance) or self.wet_fraction_tolerance <= 0.0:
            raise ValueError("phase_change_wet_fraction_tolerance must be > 0.")
        if not (0.0 < self.relaxation_factor <= 1.0):
            raise ValueError("phase_change_relaxation_factor must be in (0, 1].")


def check_single_active_side(
    inside_auto_possible: bool,
    outside_auto_possible: bool,
    *,
    iterate: bool,
) -> None:
    """Enforce "at most one active phase-changing side" (spec section 19)
    and the wet-condensation ``iterate=True`` guard.

    Pure decision function (booleans in, exception or nothing out) so the
    priority ordering -- both-sides-possible takes precedence over
    inside-only-possible, which takes precedence over the iterate=False
    guard on outside -- can be unit-tested directly without needing a
    physically-engineered thermal scenario for every combination.
    """
    if inside_auto_possible and outside_auto_possible:
        raise MultiplePhaseChangeSidesError(
            "Both inside and outside show an active condensation tendency "
            "under PhaseChangeMode.AUTO in the dry baseline; at most one "
            "active phase-changing side is supported in v0.6.1. Set "
            "phase_change_mode=DISABLED on one side, or investigate the "
            "operating point."
        )
    if (inside_auto_possible or outside_auto_possible) and not iterate:
        raise ValueError(
            "Wet-gas water condensation was detected as possible under "
            "PhaseChangeMode.AUTO, but iterate=False was requested. A "
            "single-pass sensible-only approximation cannot represent "
            "condensation; call .simulate(..., iterate=True), or set "
            "outside.phase_change_mode=DISABLED to accept a sensible-only "
            "result."
        )


def _capability_only_result(side: str, mode: PhaseChangeMode, capability: PhaseChangeCapability) -> PhaseChangeResult:
    return PhaseChangeResult(
        side=side,
        mode=mode,
        direction=PhaseChangeDirection.NONE,
        component=capability.component,
        capable=capability.capable,
        possible=False,
        active=False,
        W_in=capability.W_in,
        W_out=capability.W_in,
        assumptions=("sensible_only_no_phase_change",) if capability.capable else (),
    )


def _evaluate_side_onset(
    *,
    side: str,
    mode: PhaseChangeMode,
    capability: PhaseChangeCapability,
    p: float,
    thermal_state,
    envelope,
    settings: "PhaseChangeSettings",
):
    """Return ``(onset, dew_point, wall_min, wall_mean, wall_max)`` for one
    side, or all-``None`` if the side is not capable or has no valid dew
    point.

    Fix (v0.6.0 patch, spec section 6.1): onset is decided from
    ``wall_envelope.<side>_min`` -- the dry baseline's *minimum* estimated
    wall temperature -- never from the mean/representative wall
    temperature. ``wall_mean`` is still computed (via
    ``representative_wall_temperature``) as the whole-surface mean for the
    sensible resistance network and general reporting, not for the onset
    decision or wet-zone condensate properties.
    """
    if not capability.capable:
        return None, None, None, None, None
    dew_point = _dew_point_for(capability, p=p)
    if dew_point is None:
        return None, None, None, None, None

    wall_min = envelope.outside_min if side == "outside" else envelope.inside_min
    wall_max = envelope.outside_max if side == "outside" else envelope.inside_max
    wall_mean = representative_wall_temperature(side=side, thermal_state=thermal_state, wall_envelope=envelope)

    onset = evaluate_condensation_onset(
        dew_point_temperature=dew_point,
        wall_temperature_min=wall_min,
        onset_tolerance_K=settings.onset_tolerance_K,
        activation_band_K=settings.activation_band_K,
    )
    return onset, dew_point, wall_min, wall_mean, wall_max


def _dew_point_for(capability: PhaseChangeCapability, *, p: float) -> float | None:
    if not capability.capable or capability.W_in is None:
        return None
    from core.phase_change.water_equilibrium import water_mole_fraction_from_ratio

    y_h2o = water_mole_fraction_from_ratio(capability.W_in, M_dry=capability.M_dry, M_h2o=capability.M_condensable)
    p_h2o = water_partial_pressure(y_h2o, p)
    if is_frost_regime(p_h2o):
        return None
    return water_dew_point(p_h2o)


def apply_phase_change(
    hx,
    inside,
    outside,
    dry_result,
    *,
    iterate: bool,
    euler_provider: str = "zukauskas",
    settings: PhaseChangeSettings | None = None,
    skip_inside_pure_steam_guard: bool = False,
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
        InsideCondensationNotSupportedError, MultiplePhaseChangeSidesError,
        ValueError (AUTO + possible + iterate=False), FrostingNotSupportedError.
    """
    settings = settings or PhaseChangeSettings()

    inside_capability = detect_phase_change_capability(inside.provider)
    outside_capability = detect_phase_change_capability(outside.provider)

    if not skip_inside_pure_steam_guard:
        _raise_if_inside_pure_steam_condensation(inside, dry_result)

    if not inside_capability.capable and not outside_capability.capable:
        return replace(
            dry_result,
            inside_phase_change=_capability_only_result("inside", inside.phase_change_mode, inside_capability),
            outside_phase_change=_capability_only_result("outside", outside.phase_change_mode, outside_capability),
        )

    thermal_state = dry_result.thermal_state
    envelope = dry_result.wall_temperature_envelope

    inside_onset, inside_dew_point, inside_wall_min, inside_wall_mean, inside_wall_max = _evaluate_side_onset(
        side="inside", mode=inside.phase_change_mode, capability=inside_capability, p=inside.p,
        thermal_state=thermal_state, envelope=envelope, settings=settings,
    )
    outside_onset, outside_dew_point, outside_wall_min, outside_wall_mean, outside_wall_max = _evaluate_side_onset(
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

    check_single_active_side(inside_auto_possible, outside_auto_possible, iterate=iterate)

    if inside_auto_possible:
        return _apply_inside_condensation(
            hx,
            inside=inside,
            outside=outside,
            dry_result=dry_result,
            inside_capability=inside_capability,
            outside_capability=outside_capability,
            inside_onset=inside_onset,
            inside_dew_point=inside_dew_point,
            inside_wall_min=inside_wall_min,
            inside_wall_mean=inside_wall_mean,
            inside_wall_max=inside_wall_max,
            outside_possible=outside_possible,
            outside_near_onset=outside_near_onset,
            outside_onset=outside_onset,
            outside_dew_point=outside_dew_point,
            outside_wall_min=outside_wall_min,
            outside_wall_mean=outside_wall_mean,
            outside_wall_max=outside_wall_max,
            euler_provider=euler_provider,
            settings=settings,
        )

    inside_result = _build_capability_side_result(
        side="inside", mode=inside.phase_change_mode, capability=inside_capability,
        possible=inside_possible, near_onset=inside_near_onset,
        dew_point=inside_dew_point, p=inside.p,
        m_dot_gas=inside.m_dot,
        onset=inside_onset, wall_temperature_min=inside_wall_min,
        wall_temperature_mean=inside_wall_mean, wall_temperature_max=inside_wall_max,
    )

    if not outside_auto_possible:
        outside_result = _build_capability_side_result(
            side="outside", mode=outside.phase_change_mode, capability=outside_capability,
            possible=outside_possible, near_onset=outside_near_onset,
            dew_point=outside_dew_point, p=outside.p,
            m_dot_gas=outside.m_dot,
            onset=outside_onset, wall_temperature_min=outside_wall_min,
            wall_temperature_mean=outside_wall_mean, wall_temperature_max=outside_wall_max,
        )
        return replace(dry_result, inside_phase_change=inside_result, outside_phase_change=outside_result)

    # --- Outside condensation is the active, supported path -----------------
    p_h2o_in = water_partial_pressure(
        _y_h2o(outside_capability), outside.p
    )
    if is_frost_regime(p_h2o_in):
        outside_result = replace(
            _build_capability_side_result(
                side="outside", mode=outside.phase_change_mode, capability=outside_capability,
                possible=True, near_onset=False, dew_point=None, p=outside.p,
                m_dot_gas=outside.m_dot,
                onset=outside_onset, wall_temperature_min=outside_wall_min,
                wall_temperature_mean=outside_wall_mean, wall_temperature_max=outside_wall_max,
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

    try:
        solution = solve_outside_condensation(
            hx,
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
            _build_capability_side_result(
                side="outside", mode=outside.phase_change_mode, capability=outside_capability,
                possible=True, near_onset=False, dew_point=outside_dew_point, p=outside.p,
                m_dot_gas=outside.m_dot,
                onset=outside_onset, wall_temperature_min=outside_wall_min,
                wall_temperature_mean=outside_wall_mean, wall_temperature_max=outside_wall_max,
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
        method="outside_condensation_0d_bulk_mean",
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
        dew_point_out=_dew_point_at_ratio(outside_capability, solution.W_out, p=outside.p),
        wall_temperature_mean=solution.T_wall_outside,
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
            "bulk_mean_property_evaluation_0d",
            "fully_drained_liquid_condensate",
            "lewis_number_chilton_colburn_analogy",
            "dry_gas_composition_unchanged_by_condensation",
            "wet_surface_fraction_two_point_inlet_outlet_estimate",
            "wet_surface_temperature_linear_envelope_estimate",
        ),
        warnings=tuple(warnings_list),
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

    outside_bank_hydraulic = calculate_outside_tube_bank_hydraulics(
        m_dot=m_dot_dry_carrier * (1.0 + W_mean),
        face_area=hx.bundle.frontal_flow_area,
        tube_outer_diameter=float(getattr(hx.bundle.tube, "D_o")),
        tube_pitch_transverse=hx.bundle.pitch_transverse,
        tube_pitch_longitudinal=hx.bundle.pitch_longitudinal,
        layout=hx.bundle.layout,
        n_rows=hx.bundle.n_rows,
        n_tubes_per_row=hx.bundle.n_tubes_per_row,
        inlet_props=outside_props_inlet,
        midpoint_props=solution.outside_bulk_props,
        outlet_props=outside_props_outlet,
        temperature_in=outside.T_in,
        temperature_out=solution.T_out_outside,
        pressure=outside.p,
        euler_provider=euler_provider,
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
        inside_provider_name=type(inside.provider).__name__,
        outside_provider_name=type(outside_provider_final).__name__,
        warnings=solution.warnings,
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
        UA=wet_thermal_state.UA,
        q=solution.Q_total,
        T_out_inside=solution.T_out_inside,
        T_out_outside=solution.T_out_outside,
        Q_full=solution.Q_total,
        Q_derated=solution.Q_total,
        final_result=final_result,
        thermal_state=wet_thermal_state,
        wall_temperature_envelope=envelope_wet,
        inside_phase_change=inside_result,
        outside_phase_change=outside_result,
    )


def _apply_inside_condensation(
    hx,
    *,
    inside,
    outside,
    dry_result,
    inside_capability: PhaseChangeCapability,
    outside_capability: PhaseChangeCapability,
    inside_onset: OnsetDecision,
    inside_dew_point: float,
    inside_wall_min: float,
    inside_wall_mean: float,
    inside_wall_max: float,
    outside_possible: bool,
    outside_near_onset: bool,
    outside_onset: OnsetDecision | None,
    outside_dew_point: float | None,
    outside_wall_min: float | None,
    outside_wall_mean: float | None,
    outside_wall_max: float | None,
    euler_provider: str,
    settings: PhaseChangeSettings,
):
    """Apply the active inside wet-gas solution to a dry Simulation result."""
    outside_result = _build_capability_side_result(
        side="outside",
        mode=outside.phase_change_mode,
        capability=outside_capability,
        possible=outside_possible,
        near_onset=outside_near_onset,
        dew_point=outside_dew_point,
        p=outside.p,
        m_dot_gas=outside.m_dot,
        onset=outside_onset,
        wall_temperature_min=outside_wall_min,
        wall_temperature_mean=outside_wall_mean,
        wall_temperature_max=outside_wall_max,
    )

    p_h2o_in = water_partial_pressure(_y_h2o(inside_capability), inside.p)
    if is_frost_regime(p_h2o_in):
        inside_result = replace(
            _build_capability_side_result(
                side="inside",
                mode=inside.phase_change_mode,
                capability=inside_capability,
                possible=True,
                near_onset=False,
                dew_point=None,
                p=inside.p,
                m_dot_gas=inside.m_dot,
                onset=inside_onset,
                wall_temperature_min=inside_wall_min,
                wall_temperature_mean=inside_wall_mean,
                wall_temperature_max=inside_wall_max,
            ),
            warnings=(
                make_warning(
                    code=WC.FROSTING_NOT_SUPPORTED,
                    message=(
                        "inside: inlet water content is in the frost/ice "
                        "equilibrium regime; returning the sensible-only baseline."
                    ),
                    source="phase_change_integration",
                    severity="warning",
                ),
            ),
        )
        return replace(
            dry_result,
            inside_phase_change=inside_result,
            outside_phase_change=outside_result,
        )

    m_dot_dry_carrier = inside.m_dot / (1.0 + inside_capability.W_in)
    try:
        solution = solve_inside_condensation(
            hx,
            inside_capability=inside_capability,
            m_dot_dry_carrier=m_dot_dry_carrier,
            T_in_inside=inside.T_in,
            p_inside=inside.p,
            outside_provider=outside.provider,
            m_dot_outside=outside.m_dot,
            T_in_outside=outside.T_in,
            p_outside=outside.p,
            T_out_inside_init=dry_result.T_out_inside,
            T_out_outside_init=dry_result.T_out_outside,
            euler_provider=euler_provider,
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
        inside_result = replace(
            _build_capability_side_result(
                side="inside",
                mode=inside.phase_change_mode,
                capability=inside_capability,
                possible=True,
                near_onset=False,
                dew_point=inside_dew_point,
                p=inside.p,
                m_dot_gas=inside.m_dot,
                onset=inside_onset,
                wall_temperature_min=inside_wall_min,
                wall_temperature_mean=inside_wall_mean,
                wall_temperature_max=inside_wall_max,
            ),
            warnings=(
                make_warning(
                    code=WC.FROSTING_NOT_SUPPORTED,
                    message=f"inside: {exc}",
                    source="phase_change_integration",
                    severity="warning",
                ),
            ),
        )
        return replace(
            dry_result,
            inside_phase_change=inside_result,
            outside_phase_change=outside_result,
        )

    W_in = inside_capability.W_in
    W_mid = 0.5 * (W_in + solution.W_out)
    m_dot_water_vapor_in = m_dot_dry_carrier * W_in
    m_dot_water_vapor_out = m_dot_dry_carrier * solution.W_out
    m_dot_gas_out = m_dot_dry_carrier + m_dot_water_vapor_out
    mass_balance_error = m_dot_water_vapor_in - (
        m_dot_water_vapor_out + solution.m_dot_condensate
    )

    enthalpy_evaluator = WetGasEnthalpyEvaluator(inside.p, inside_capability)
    h_in = enthalpy_evaluator.enthalpy(inside.T_in, W_in)
    h_out = enthalpy_evaluator.enthalpy(
        solution.T_out_inside,
        solution.W_out,
    )
    condensate_enthalpy_rate = condensate_enthalpy_flow(
        m_dot_condensate=solution.m_dot_condensate,
        condensation_mass_tolerance=settings.condensate_tolerance_kg_s,
        wet_surface_fraction=solution.wet_surface_fraction,
        wet_area=solution.wet_area,
        wall_temperature_wet_mean=solution.wall_temperature_wet_mean,
    )
    Q_enthalpy = (
        m_dot_dry_carrier * (h_in - h_out) - condensate_enthalpy_rate
    )
    energy_balance_error = solution.Q_total - Q_enthalpy

    warnings_list: list[ModelWarning] = list(solution.warnings)
    for code, message in (
        (
            WC.INSIDE_CONDENSATION_DETECTED,
            "inside: the minimum inside wall temperature is below the H2O dew point; partial condensation was solved.",
        ),
        (
            WC.WET_SURFACE_FRACTION_0D_ESTIMATE,
            "inside: wet fraction and wet-wall temperature are 0D linear wall-envelope estimates.",
        ),
        (
            WC.LEWIS_NUMBER_ASSUMED,
            f"inside: Chilton-Colburn mass transfer uses lewis_number={settings.lewis_number:g}.",
        ),
        (
            WC.FULLY_DRAINED_CONDENSATE_ASSUMED,
            "inside: condensate is removed from the gas at the wet-wall temperature without re-entrainment.",
        ),
        (
            WC.CONDENSATE_FILM_RESISTANCE_NOT_MODELLED,
            "inside: condensate-film thermal resistance is not modelled in v0.6.1.",
        ),
        (
            WC.CONDENSATE_FILM_HYDRAULICS_NOT_MODELLED,
            "inside: tube-side pressure drop is a gas-phase approximation; condensate-film hydraulics are not modelled.",
        ),
        (
            WC.REENTRAINMENT_NOT_MODELLED,
            "inside: liquid holdup, carryover and re-entrainment are not modelled.",
        ),
    ):
        warnings_list.append(
            make_warning(
                code=code,
                message=message,
                source="phase_change_integration",
                severity="info",
            )
        )
    warnings_list = _deduplicate_warnings(warnings_list)

    inside_result = PhaseChangeResult(
        side="inside",
        mode=inside.phase_change_mode,
        direction=PhaseChangeDirection.CONDENSATION,
        component=inside_capability.component,
        capable=True,
        possible=True,
        active=True,
        onset_margin_K=inside_onset.margin_K,
        onset_wall_temperature=inside_wall_min,
        onset_temperature_method=ONSET_TEMPERATURE_METHOD,
        converged=solution.converged,
        iterations=solution.iterations,
        method="inside_condensation_0d_bulk_mean",
        W_in=W_in,
        W_mid=W_mid,
        W_out=solution.W_out,
        m_dot_dry_carrier=m_dot_dry_carrier,
        m_dot_water_vapor_in=m_dot_water_vapor_in,
        m_dot_water_vapor_out=m_dot_water_vapor_out,
        m_dot_condensate=solution.m_dot_condensate,
        m_dot_gas_in=inside.m_dot,
        m_dot_gas_out=m_dot_gas_out,
        dew_point_in=inside_dew_point,
        dew_point_out=_dew_point_at_ratio(
            inside_capability,
            solution.W_out,
            p=inside.p,
        ),
        wall_temperature_mean=solution.T_wall_inside,
        wall_temperature_min=solution.wall_temperature_min,
        wall_temperature_max=solution.wall_temperature_max,
        wall_temperature_wet_mean=solution.wall_temperature_wet_mean,
        wet_surface_fraction=solution.wet_surface_fraction,
        wet_surface_fraction_method=solution.wet_surface_fraction_method,
        wet_area=solution.wet_area,
        inside_total_area=solution.inside_total_area,
        W_sat_wet_surface=solution.W_sat_wet_surface,
        alfa_dry=solution.alfa_i_dry,
        alfa_effective=solution.alfa_i_effective,
        lewis_number=settings.lewis_number,
        Q_sensible=solution.Q_sensible,
        Q_latent=solution.Q_latent,
        Q_total=solution.Q_total,
        mass_balance_error=mass_balance_error,
        energy_balance_error=energy_balance_error,
        residuals=dict(solution.residuals),
        assumptions=(
            "bulk_mean_property_evaluation_0d",
            "fully_drained_liquid_condensate",
            "lewis_number_chilton_colburn_analogy",
            "dry_gas_composition_unchanged_by_condensation",
            "wet_surface_fraction_two_point_inlet_outlet_estimate",
            "gas_phase_only_tube_side_hydraulics",
        ),
        warnings=tuple(warnings_list),
    )

    inside_provider_inlet = wet_gas_provider_at_water_ratio(inside_capability, W_in)
    inside_provider_mid = wet_gas_provider_at_water_ratio(inside_capability, W_mid)
    inside_provider_outlet = wet_gas_provider_at_water_ratio(
        inside_capability,
        solution.W_out,
    )
    inside_props_inlet = inside_provider_inlet.at(T=inside.T_in, p=inside.p)
    inside_props_outlet = inside_provider_outlet.at(
        T=solution.T_out_inside,
        p=inside.p,
    )
    T_mean_inside = mean_temperature(inside.T_in, solution.T_out_inside)
    T_mean_outside = mean_temperature(outside.T_in, solution.T_out_outside)
    m_dot_gas_mid = m_dot_dry_carrier * (1.0 + W_mid)

    from core.heat_transfer.streams import SensibleHeatStream

    inside_stream = SensibleHeatStream(
        C=m_dot_gas_mid * solution.inside_bulk_props.cp,
        T_in=inside.T_in,
    )
    outside_stream = SensibleHeatStream(
        C=outside.m_dot * solution.outside_bulk_props.cp,
        T_in=outside.T_in,
    )
    hot_is_inside = inside.T_in >= outside.T_in
    hot_stream, cold_stream = (
        (inside_stream, outside_stream)
        if hot_is_inside
        else (outside_stream, inside_stream)
    )
    final_result = hx.solve(
        hot_stream=hot_stream,
        cold_stream=cold_stream,
        m_dot_tube_side=m_dot_gas_mid,
        tube_side_props=to_internal_fluid_props(solution.inside_bulk_props),
        tube_side_provider=inside_provider_mid,
        tube_side_temperature_in=inside.T_in,
        tube_side_temperature_out=solution.T_out_inside,
        tube_side_pressure=inside.p,
        m_dot_outside=outside.m_dot,
        outside_props=to_outside_fluid_props(solution.outside_bulk_props),
        outside_provider=outside.provider,
        outside_temperature_in=outside.T_in,
        outside_temperature_out=solution.T_out_outside,
        outside_pressure=outside.p,
        flow_arrangement=None,
        euler_provider=euler_provider,
    )

    from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
    from core.pressure_drop.flow_path import build_tube_side_pressure_drop_result

    tube_bundle_hydraulic = calculate_tube_bundle_hydraulics(
        m_dot=m_dot_gas_mid,
        flow_area_per_pass=hx.bundle.internal_flow_area_per_pass,
        hydraulic_diameter=hx.bundle.internal_hydraulic_diameter,
        hydraulic_length_total=hx.bundle.internal_length_total,
        n_tube_passes=hx.bundle.n_passes_tube,
        tube_path_type=hx.bundle.tube_path_type,
        roughness_inner=getattr(hx.bundle.tube, "roughness_inner", None),
        inlet_props=inside_props_inlet,
        midpoint_props=solution.inside_bulk_props,
        outlet_props=inside_props_outlet,
        temperature_in=inside.T_in,
        temperature_out=solution.T_out_inside,
        pressure=inside.p,
        m_dot_inlet=inside.m_dot,
        m_dot_midpoint=m_dot_gas_mid,
        m_dot_outlet=m_dot_gas_out,
    )
    tube_bundle_hydraulic = replace(
        tube_bundle_hydraulic,
        midpoint_method="arithmetic_temperature_and_water_ratio",
    )
    final_result = replace(
        final_result,
        tube_side_hydraulic=replace(
            final_result.tube_side_hydraulic,
            tube_bundle=tube_bundle_hydraulic,
        ),
        tube_side_pressure_drop=replace(
            final_result.tube_side_pressure_drop,
            tube_bundle=tube_bundle_hydraulic,
            flow_path=build_tube_side_pressure_drop_result(
                tube_bundle_hydraulic,
                n_tube_passes=hx.bundle.n_passes_tube,
            ),
        ),
    )

    envelope_wet = estimate_wall_temperature_envelope(
        hx,
        m_dot_inside=m_dot_gas_mid,
        m_dot_outside=outside.m_dot,
        inside_provider=inside_provider_mid,
        outside_provider=outside.provider,
        inside_inlet_temperature=inside.T_in,
        inside_outlet_temperature=solution.T_out_inside,
        outside_inlet_temperature=outside.T_in,
        outside_outlet_temperature=solution.T_out_outside,
        p_inside=inside.p,
        p_outside=outside.p,
        euler_provider=euler_provider,
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
        alfa_i=solution.alfa_i_effective,
        alfa_o=solution.alfa_o,
        U=solution.U_effective,
        UA=solution.UA_effective,
        iterations=solution.iterations,
        converged=solution.converged,
        residual=solution.residuals.get("T_wall_inside_K", math.inf),
        diagnostics=solution.diagnostics,
        inside_provider_name=type(inside_provider_mid).__name__,
        outside_provider_name=type(outside.provider).__name__,
        warnings=solution.warnings,
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
        UA=wet_thermal_state.UA,
        q=solution.Q_total,
        T_out_inside=solution.T_out_inside,
        T_out_outside=solution.T_out_outside,
        Q_full=solution.Q_total,
        Q_derated=solution.Q_total,
        final_result=final_result,
        thermal_state=wet_thermal_state,
        wall_temperature_envelope=envelope_wet,
        inside_phase_change=inside_result,
        outside_phase_change=outside_result,
    )


def _raise_if_inside_pure_steam_condensation(inside, dry_result) -> None:
    """Guard pure-water phase crossings that reached the wet-gas adapter."""
    if not is_pure_water_provider(inside.provider):
        return
    from core.properties.water import WaterSteamPhase

    try:
        T_sat = pure_water_saturation_temperature(inside.provider, p=inside.p)
    except (TypeError, ValueError):
        return
    if T_sat is None:
        return
    state = getattr(inside, "water_steam_state", None)
    provider_kind = pure_water_provider_kind(inside.provider)
    if provider_kind != "iapws97":
        T_out_inside = getattr(dry_result, "T_out_inside", None)
        if T_out_inside is None:
            T_out_inside = dry_result.closed_balance.inside.T_out
        if (
            inside.T_in > T_sat + 1.0e-6
            and T_out_inside <= T_sat + 1.0e-6
        ) or (
            inside.T_in < T_sat - 1.0e-6
            and T_out_inside >= T_sat - 1.0e-6
        ):
            raise PureWaterPhaseChangeProviderNotSupportedError(
                "Pure-water phase change is supported only by "
                "IAPWS97WaterSteamProvider; the selected provider was not replaced."
            )
        return
    if (
        state is not None
        and state.phase is WaterSteamPhase.SUBCOOLED_LIQUID
        and dry_result.T_out_inside >= T_sat - 1.0e-6
    ):
        from core.phase_change.steam_heater import SteamEvaporationNotSupportedError

        raise SteamEvaporationNotSupportedError(
            "The sensible-only result crosses the water saturation boundary "
            "in the heating direction; boiling/evaporation is unsupported."
        )
    if inside.phase_change_mode is PhaseChangeMode.DISABLED:
        return
    wall_min = dry_result.wall_temperature_envelope.inside_min
    if inside.T_in >= T_sat - 1e-6 and wall_min < T_sat:
        raise PureWaterSteamCondensationNotSupportedError(
            "Pure-water/steam condensation must be dispatched through the "
            "v0.6.2 steam adapter, not the wet-gas integration function."
        )


def _y_h2o(capability: PhaseChangeCapability) -> float:
    from core.phase_change.water_equilibrium import water_mole_fraction_from_ratio

    return water_mole_fraction_from_ratio(capability.W_in, M_dry=capability.M_dry, M_h2o=capability.M_condensable)


def _dew_point_at_ratio(capability: PhaseChangeCapability, W: float, *, p: float) -> float | None:
    from core.phase_change.water_equilibrium import water_mole_fraction_from_ratio

    if W <= 0.0:
        return None
    y = water_mole_fraction_from_ratio(W, M_dry=capability.M_dry, M_h2o=capability.M_condensable)
    p_h2o = water_partial_pressure(y, p)
    if is_frost_regime(p_h2o):
        return None
    return water_dew_point(p_h2o)


def _build_capability_side_result(
    *,
    side: str,
    mode: PhaseChangeMode,
    capability: PhaseChangeCapability,
    possible: bool,
    near_onset: bool,
    dew_point: float | None,
    p: float,
    m_dot_gas: float | None = None,
    onset: OnsetDecision | None = None,
    wall_temperature_min: float | None = None,
    wall_temperature_mean: float | None = None,
    wall_temperature_max: float | None = None,
) -> PhaseChangeResult:
    """Build a non-active (capable-but-dry / near-onset / disabled)
    ``PhaseChangeResult``.

    ``onset``/``wall_temperature_*`` are optional so existing call sites
    (and the ``not capability.capable`` case, which never has a dew point)
    keep working, but every capable-with-a-dew-point caller should supply
    them (fix, v0.6.0 patch, spec section 15: "do not leave onset/wall-
    temperature diagnostics as None just because the wet solver was not
    run, if the data was available from the dry baseline").
    """
    warnings: list[ModelWarning] = []
    if near_onset:
        warnings.append(
            make_warning(
                code=WC.PHASE_CHANGE_NEAR_ONSET,
                message=(
                    f"{side}: the dry-baseline minimum wall temperature is "
                    "within the phase-change activation band of the dew "
                    "point; the regime is held at sensible-only (dry) to "
                    "avoid oscillating between dry and wet solves near "
                    "onset."
                ),
                source="phase_change_integration",
                severity="warning",
            )
        )
    if possible and mode is PhaseChangeMode.DISABLED:
        warnings.append(
            make_warning(
                code=WC.PHASE_CHANGE_DISABLED_BUT_POSSIBLE,
                message=(
                    f"{side}: phase_change_mode=DISABLED forced a sensible-"
                    "only result, but the dry baseline's minimum estimated "
                    "wall temperature runs below the dew point -- "
                    "condensation would be thermodynamically possible here. "
                    "This result is a forced single-phase approximation: it "
                    "does not remove condensate from the stream, and the "
                    "true heat duty and outlet temperature may differ."
                ),
                source="phase_change_integration",
                severity="warning",
            )
        )
    m_dot_dry_carrier = None
    m_dot_water_vapor = None
    if (
        m_dot_gas is not None
        and capability.W_in is not None
        and math.isfinite(m_dot_gas)
        and m_dot_gas > 0.0
    ):
        m_dot_dry_carrier = m_dot_gas / (1.0 + capability.W_in)
        m_dot_water_vapor = m_dot_dry_carrier * capability.W_in

    return PhaseChangeResult(
        side=side,
        mode=mode,
        direction=PhaseChangeDirection.NONE,
        component=capability.component,
        capable=capability.capable,
        possible=possible,
        active=False,
        near_onset=near_onset,
        onset_margin_K=None if onset is None else onset.margin_K,
        onset_wall_temperature=wall_temperature_min,
        onset_temperature_method=ONSET_TEMPERATURE_METHOD if onset is not None else None,
        W_in=capability.W_in,
        W_mid=capability.W_in,
        W_out=capability.W_in,
        m_dot_dry_carrier=m_dot_dry_carrier,
        m_dot_water_vapor_in=m_dot_water_vapor,
        m_dot_water_vapor_out=m_dot_water_vapor,
        m_dot_condensate=0.0,
        m_dot_gas_in=m_dot_gas,
        m_dot_gas_out=m_dot_gas,
        dew_point_in=dew_point,
        dew_point_out=dew_point,
        wall_temperature_min=wall_temperature_min,
        wall_temperature_mean=wall_temperature_mean,
        wall_temperature_max=wall_temperature_max,
        wet_surface_fraction=0.0 if capability.capable else None,
        wet_area=0.0 if capability.capable else None,
        Q_sensible=0.0,
        Q_latent=0.0,
        Q_total=0.0,
        assumptions=("sensible_only_no_phase_change",) if capability.capable else (),
        warnings=tuple(warnings),
    )


def _deduplicate_warnings(warnings: list[ModelWarning]) -> list[ModelWarning]:
    unique: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        identity = (warning.source, warning.code)
        if identity not in seen:
            seen.add(identity)
            unique.append(warning)
    return unique
