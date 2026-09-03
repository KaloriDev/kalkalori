# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Rating-side orchestration for outside water condensation (spec section 32).

Mirrors ``core.phase_change.integration`` but for
``core.models.bare_tube.BareTubeHeatExchanger.rate``. Rating is a *closed*
heat balance (both temperature programs already known), which makes the
outside condensation case algebraically simpler than Simulation: for a
trial condensate temperature, ``W_out`` is the unique root of the outside
enthalpy balance. ``Q_required`` comes from an explicit ``Q`` or from the
fully specified, non-condensing inside side -- never from the wet outside
side's sensible cp, which would silently ignore latent duty.

Active Rating wraps that algebraic root in a small, bounded fixed-point
loop. The root determines the mean wet-gas composition used by
``run_rating``.  The span of its final wall-temperature envelope is centred
on the global outside-wall mean for the active wet-surface diagnostics,
then the shared wet-surface helper determines the representative wet-wall
temperature. That temperature is fed back to the drained-liquid enthalpy
in the next root. The public four-probe envelope itself remains unchanged,
so its extrema continue to agree with its audit probes. The regime remains
locked after dry-baseline activation, and Rating deliberately does not add
Simulation's transport-limited condensation check.

The global wall mean still belongs to the whole outside surface and the
sensible resistance network. Only the representative wet-wall temperature
is used for condensate properties, the latent split, and wet-surface
equilibrium diagnostics.

Once ``W_out`` is known, a ``ClosedBalanceSide`` for the outside stream is
built with an *effective* capacity rate ``C_effective = Q_required /
|T_in-T_out|`` (spec section 25/26 "if you need an effective capacity rate
to interoperate with the existing epsilon-NTU model, it may be computed
after closing the enthalpy balance") so the existing, unmodified
``core.models.rating.run_rating`` can be reused verbatim for
``UA_required``/``A_required``/``overdesign_factor`` -- no epsilon-NTU or
overdesign logic is duplicated here.
"""

from __future__ import annotations

import math
from dataclasses import replace

from core.common.warnings import ModelWarning, make_warning
from core.geometry.tube import TubeSurfaceType
from core.heat_transfer.outside_flow import calculate_outside_tube_bank_hydraulics
from core.heat_transfer.outside_dispatch import (
    DEFAULT_FINNED_DP_PROVIDER,
    DEFAULT_FINNED_HT_PROVIDER,
    calculate_resistance_network,
    evaluate_outside_hydraulics,
)
from core.models.heat_balance import BalanceSideSpec, ClosedBalance, ClosedBalanceSide, close_heat_balance
from core.models.rating import run_rating
from core.models.surface_margin import calculate_surface_margin_factor
from core.pressure_drop.flow_path import build_outside_pressure_drop_result
from core.properties.averaging import mean_temperature

from core.phase_change import warning_codes as WC
from core.phase_change.capability import (
    detect_phase_change_capability,
    guard_pure_water_single_phase_provider,
    reject_unsupported_pure_water_phase_crossing,
)
from core.phase_change.integration import (
    ONSET_TEMPERATURE_METHOD,
    PhaseChangeSettings,
    _finned_0d_wet_zone_fallback,
    build_capability_side_result,
    dew_point_at_ratio,
    evaluate_side_onset,
    raise_if_inside_pure_steam_condensation,
    check_single_active_side,
)
from core.phase_change.types import (
    PhaseChangeCapability,
    PhaseChangeDirection,
    PhaseChangeMode,
    PhaseChangeResult,
)
from core.phase_change.finned_tube_guard import (
    reject_circular_finned_tube_wet_surface,
)
from core.properties.water import water_latent_heat_of_vaporization, water_saturation_liquid_enthalpy
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.wet_gas_enthalpy import WetGasEnthalpyEvaluator
from core.phase_change.wet_finned_surface import (
    WetFinnedSurfaceResult,
    solve_wet_finned_surface,
)
from core.phase_change.wet_surface_fraction import estimate_wet_surface_fraction
from core.phase_change.water_equilibrium import saturated_water_ratio

SOURCE = "phase_change_rating_integration"
_RATING_OUTER_MAX_ITERATIONS = 12
_RATING_ENTHALPY_SYNCHRONIZATION_TOLERANCE_W = 1e-5

_RATING_CLOSURE_MAX_BRACKET_EXPANSIONS = 40
_RATING_CLOSURE_MAX_ROOT_ITERATIONS = 60
_RATING_CLOSURE_MASS_FLOW_FLOOR_KG_S = 1e-9
_RATING_CLOSURE_TEMPERATURE_MARGIN_K = 1e-3


class RatingClosureError(ValueError):
    """Raised when a single-unknown Rating closure cannot be solved.

    Distinct from the plain ``ValueError`` used for a genuinely
    under-specified Rating problem (spec section 19): this one is raised
    only after a bracketed scalar search over the one supported unknown
    (inside mass flow or inside outlet temperature) has been attempted and
    failed -- either no sign change could be bracketed (no physically
    achievable solution for any positive/valid value) or the bracketed
    root search itself did not converge.
    """

    warning_code = "RATING_CLOSURE_NOT_SOLVED"


def _bracket_and_solve_monotonic_root(
    residual,
    x0: float,
    *,
    x_min: float,
    x_max: float,
    x_tolerance: float,
    residual_tolerance: float,
    variable_name: str,
    initial_step: float | None = None,
    max_bracket_expansions: int = _RATING_CLOSURE_MAX_BRACKET_EXPANSIONS,
    max_iterations: int = _RATING_CLOSURE_MAX_ROOT_ITERATIONS,
) -> tuple[float, float, int]:
    """Bracket and solve ``residual(x) = 0`` for ``x`` in ``(x_min, x_max)``.

    A deterministic bounded bracketed solver (geometric bracket expansion
    around ``x0``, then an Illinois/regula-falsi search with anti-stagnation
    damping and a bisection fallback) -- no SciPy dependency, matching the
    project's existing bracketed-root idiom (see
    ``core.phase_change.outside_condensation_solver._solve_interface_state``
    and the KDV acceptance notebooks' closed-loop solver). ``residual`` may
    raise on a physically/numerically invalid trial (e.g. a property outside
    its validity range); such points are treated as unusable, not as a
    valid zero-crossing, and are never returned as a bracket endpoint.

    ``initial_step`` sets the first bracket-expansion step explicitly
    (defaulting to half of ``abs(x0)`` when omitted); pass it for a variable
    whose natural scale is unrelated to its own magnitude (e.g. a Kelvin
    temperature, where ``abs(x0) * 0.5`` is a wildly oversized first jump).

    Returns ``(x_root, residual_at_root, iterations)``.

    Raises:
        RatingClosureError: no sign change could be bracketed within
            ``[x_min, x_max]``, or the root search did not converge within
            ``max_iterations``.
    """

    def safe_residual(x: float) -> float:
        if not (x_min < x < x_max):
            return math.nan
        try:
            value = residual(x)
        except Exception:
            return math.nan
        return value if math.isfinite(value) else math.nan

    r0 = safe_residual(x0)
    if math.isnan(r0):
        raise RatingClosureError(
            f"Rating closure: the initial estimate {variable_name}={x0:.6g} "
            "did not produce a valid trial state (a property or physical "
            "bound was violated); no bracket search could start."
        )
    if abs(r0) <= residual_tolerance:
        return x0, r0, 0

    lo, hi = x0, x0
    r_lo, r_hi = r0, r0
    step = (
        max(abs(x0) * 0.5, x_tolerance * 10.0)
        if initial_step is None
        else max(abs(initial_step), x_tolerance * 10.0)
    )
    expansions = 0
    while r_lo * r_hi > 0.0 and expansions < max_bracket_expansions:
        expansions += 1
        moved = False
        new_hi = min(hi + step, x_max)
        if new_hi > hi:
            r_new_hi = safe_residual(new_hi)
            if not math.isnan(r_new_hi):
                hi, r_hi = new_hi, r_new_hi
                moved = True
                if r_lo * r_hi <= 0.0:
                    break
        new_lo = max(lo - step, x_min)
        if new_lo < lo:
            r_new_lo = safe_residual(new_lo)
            if not math.isnan(r_new_lo):
                lo, r_lo = new_lo, r_new_lo
                moved = True
                if r_lo * r_hi <= 0.0:
                    break
        step *= 2.0
        if not moved and new_hi >= x_max and new_lo <= x_min:
            break

    if r_lo * r_hi > 0.0:
        raise RatingClosureError(
            f"Rating closure: could not bracket a solution for "
            f"{variable_name} -- no sign change found for "
            f"{variable_name} in [{lo:.6g}, {hi:.6g}] (residuals "
            f"{r_lo:.6g} W, {r_hi:.6g} W) after {expansions} geometric "
            f"bracket expansions from the initial estimate {x0:.6g}. The "
            "specified Rating temperature program does not appear to be "
            f"physically achievable for any valid {variable_name}."
        )

    retained_side: str | None = None
    x_mid = lo
    r_mid = r_lo
    for iteration in range(1, max_iterations + 1):
        denom = r_hi - r_lo
        if denom == 0.0:
            x_mid = 0.5 * (lo + hi)
        else:
            x_mid = hi - r_hi * (hi - lo) / denom
            if not (lo < x_mid < hi):
                x_mid = 0.5 * (lo + hi)
        r_mid = safe_residual(x_mid)
        if math.isnan(r_mid):
            # The interior point itself is invalid (e.g. a property bound):
            # fall back to a safe bisection step instead of the (possibly
            # overshooting) secant/Illinois step.
            x_mid = 0.5 * (lo + hi)
            r_mid = safe_residual(x_mid)
            if math.isnan(r_mid):
                raise RatingClosureError(
                    f"Rating closure: the interior point {variable_name}="
                    f"{x_mid:.6g} did not produce a valid trial state while "
                    "bisecting a valid bracket; the physically valid region "
                    "for this variable appears to be narrower than the "
                    "bracket found."
                )
        if abs(r_mid) <= residual_tolerance or (hi - lo) < x_tolerance:
            return x_mid, r_mid, iteration
        if r_lo * r_mid <= 0.0:
            hi, r_hi = x_mid, r_mid
            if retained_side == "lo":
                r_lo *= 0.5
            retained_side = "lo"
        else:
            lo, r_lo = x_mid, r_mid
            if retained_side == "hi":
                r_hi *= 0.5
            retained_side = "hi"

    raise RatingClosureError(
        f"Rating closure: the bracketed search for {variable_name} did not "
        f"converge within {max_iterations} iterations (last residual "
        f"{r_mid:.6g} W, bracket [{lo:.6g}, {hi:.6g}])."
    )


def _solve_rating_water_ratio_for_side(
    *,
    wet_side: BalanceSideSpec,
    capability: PhaseChangeCapability,
    Q_required: float,
    m_dot_dry_carrier: float,
    condensate_temperature: float,
    enthalpy_evaluator: WetGasEnthalpyEvaluator | None = None,
) -> tuple[float, float]:
    """Close Rating's wet-side enthalpy balance at one condensate temperature.

    Returns ``(W_out, residual_W)``. The drained saturated-liquid enthalpy
    is evaluated at ``condensate_temperature``; no mass-transfer-capacity
    constraint is imposed by this Rating-only algebraic root.
    """
    W_in = capability.W_in
    if W_in is None:
        raise ValueError("Active outside condensation requires a finite inlet water ratio.")
    if wet_side.T_out is None:
        raise ValueError("Active wet-gas condensation requires an explicit wet-side T_out.")

    evaluator = enthalpy_evaluator or WetGasEnthalpyEvaluator(
        wet_side.p,
        capability,
    )
    h_in = evaluator.enthalpy(wet_side.T_in, W_in)
    h_liquid = water_saturation_liquid_enthalpy(T=condensate_temperature)

    def residual(W_out: float) -> float:
        h_out = evaluator.enthalpy(wet_side.T_out, W_out)
        h_drained = (W_in - W_out) * h_liquid
        Q_implied = m_dot_dry_carrier * (h_in - h_out - h_drained)
        return Q_implied - Q_required

    W_lo, W_hi = 0.0, W_in
    r_lo, r_hi = residual(W_lo), residual(W_hi)
    if not (min(r_lo, r_hi) <= 0.0 <= max(r_lo, r_hi)):
        raise ValueError(
            "Rating with active outside condensation: cannot close the "
            f"outside water/enthalpy balance for Q_required={Q_required:.6g} W "
            f"within 0 <= W_out <= W_in={W_in:.6g} kg/kg at condensate "
            f"temperature {condensate_temperature:.6g} K (residuals at "
            f"bracket ends: {r_lo:.6g} W, {r_hi:.6g} W). The specified "
            "temperature program is not thermodynamically consistent with "
            "partial H2O condensation."
        )

    if r_lo == 0.0:
        return W_lo, r_lo
    if r_hi == 0.0:
        return W_hi, r_hi

    W_mid = 0.5 * (W_lo + W_hi)
    r_mid = residual(W_mid)
    for _ in range(200):
        W_mid = 0.5 * (W_lo + W_hi)
        r_mid = residual(W_mid)
        if abs(r_mid) <= 1e-7 or (W_hi - W_lo) < 1e-12:
            break
        if (r_mid > 0.0) == (r_lo > 0.0):
            W_lo, r_lo = W_mid, r_mid
        else:
            W_hi, r_hi = W_mid, r_mid

    if not (0.0 <= W_mid <= W_in):
        raise ValueError(
            f"Rating with active outside condensation: solved W_out={W_mid:.6g} "
            f"kg/kg is outside the valid range [0, {W_in:.6g}]."
        )
    return W_mid, r_mid


def _rating_enthalpy_residual_at_state(
    *,
    wet_side: BalanceSideSpec,
    capability: PhaseChangeCapability,
    Q_required: float,
    m_dot_dry_carrier: float,
    W_out: float,
    condensate_temperature: float,
    enthalpy_evaluator: WetGasEnthalpyEvaluator | None = None,
) -> float:
    """Return the outside enthalpy residual for one fully specified state."""
    W_in = capability.W_in
    if W_in is None or wet_side.T_out is None:
        raise ValueError(
            "Active outside condensation requires W_in and outside.T_out."
        )
    evaluator = enthalpy_evaluator or WetGasEnthalpyEvaluator(
        wet_side.p,
        capability,
    )
    h_in = evaluator.enthalpy(wet_side.T_in, W_in)
    h_out = evaluator.enthalpy(wet_side.T_out, W_out)
    h_liquid = water_saturation_liquid_enthalpy(
        T=condensate_temperature
    )
    return (
        m_dot_dry_carrier
        * (h_in - h_out - (W_in - W_out) * h_liquid)
        - Q_required
    )


def _centered_wall_bounds(rating_result, *, side: str) -> tuple[float, float, float]:
    """Return wet-model bounds centred on Rating's global side-wall mean.

    Only the numerical bounds used by the active wet-surface model are
    shifted.  The public four-probe envelope is deliberately not mutated.
    """
    envelope = rating_result.wall_temperature_envelope
    if side not in {"inside", "outside"}:
        raise ValueError("side must be 'inside' or 'outside'.")
    wall_mean = (
        rating_result.thermal_state.inside_wall_temperature
        if side == "inside"
        else rating_result.thermal_state.outside_wall_temperature
    )
    wall_min = envelope.inside_min if side == "inside" else envelope.outside_min
    wall_max = envelope.inside_max if side == "inside" else envelope.outside_max
    if not all(
        math.isfinite(value)
        for value in (wall_min, wall_mean, wall_max)
    ):
        raise ValueError(
            "Rating with active outside condensation requires a finite "
            "outside wall-temperature envelope."
        )
    if wall_max < wall_min:
        raise ValueError(
            "Rating outside wall-temperature envelope has max below min."
        )
    half_span = 0.5 * (wall_max - wall_min)
    return wall_mean - half_span, wall_mean, wall_mean + half_span


def _normalize_rating_wet_finned_surface(
    surface: WetFinnedSurfaceResult,
    *,
    Q_sensible: float,
    Q_latent: float,
    Q_total: float,
    m_dot_condensate: float,
    condensate_temperature: float,
) -> WetFinnedSurfaceResult:
    """Normalize only Rating's primary/fin rate split to its closed balance.

    Rating's specified temperature program fixes the authoritative whole-side
    enthalpy duty and condensate rate.  The nonlinear surface solve remains
    authoritative for temperatures, wet areas/state and the relative
    primary/fin distribution.  Scaling that distribution keeps one typed
    result internally closed without pretending that Rating is a separate
    transport-limited whole-HX solve.  Raw mismatches and scale factors remain
    explicit in ``residuals``.
    """

    def scaled_split(
        primary_raw: float,
        fin_raw: float,
        target: float,
        *,
        name: str,
    ) -> tuple[float, float, float]:
        raw_total = primary_raw + fin_raw
        scale_floor = max(abs(target), abs(raw_total), 1.0) * 1.0e-15
        if abs(raw_total) <= scale_floor:
            if abs(target) <= scale_floor:
                return 0.0, 0.0, 1.0
            raise ValueError(
                "Active wet-finned Rating cannot normalize a non-zero "
                f"{name} total from a zero raw primary/fin response."
            )
        scale = target / raw_total
        return primary_raw * scale, fin_raw * scale, scale

    Q_primary_sensible, Q_fin_sensible, sensible_scale = scaled_split(
        surface.Q_primary_sensible,
        surface.Q_fin_sensible,
        Q_sensible,
        name="sensible-duty",
    )
    Q_primary_latent, Q_fin_latent, latent_scale = scaled_split(
        surface.Q_primary_latent,
        surface.Q_fin_latent,
        Q_latent,
        name="latent-duty",
    )
    m_primary, m_fin, condensate_scale = scaled_split(
        surface.m_dot_condensate_primary,
        surface.m_dot_condensate_fin,
        m_dot_condensate,
        name="condensate",
    )

    Q_primary_total = Q_primary_sensible + Q_primary_latent
    Q_fin_total = Q_fin_sensible + Q_fin_latent
    condensate_enthalpy_rate = (
        m_dot_condensate
        * water_saturation_liquid_enthalpy(T=condensate_temperature)
    )
    raw_Q_total = surface.Q_total
    raw_condensate = surface.m_dot_condensate
    raw_Q_gap = abs(raw_Q_total - Q_total)
    raw_condensate_gap = abs(raw_condensate - m_dot_condensate)
    normalization_needed = (
        raw_Q_gap > max(abs(Q_total), 1.0) * 1.0e-12
        or raw_condensate_gap
        > max(abs(m_dot_condensate), 1.0e-12) * 1.0e-9
    )
    residuals = dict(surface.residuals)
    residuals.update(
        {
            "rating_raw_surface_Q_total_W": raw_Q_total,
            "rating_surface_Q_gap_W": raw_Q_gap,
            "rating_raw_surface_condensate_kg_s": raw_condensate,
            "rating_surface_condensate_gap_kg_s": raw_condensate_gap,
            "rating_sensible_distribution_scale": abs(sensible_scale),
            "rating_latent_distribution_scale": abs(latent_scale),
            "rating_condensate_distribution_scale": abs(condensate_scale),
        }
    )
    assumptions = surface.assumptions
    if normalization_needed:
        assumptions = (
            *assumptions,
            "rating_closed_balance_normalized_primary_fin_distribution",
            "rating_wet_effective_alpha_uses_raw_radial_transport_duty",
        )

    annular_fin = surface.annular_fin
    if annular_fin is not None and surface.equivalent_fin_count > 0.0:
        fin_count = surface.equivalent_fin_count
        fin_condensate_enthalpy = (
            condensate_enthalpy_rate
            * (m_fin / m_dot_condensate)
            if m_dot_condensate > 0.0
            else 0.0
        )
        annular_residuals = dict(annular_fin.residuals)
        annular_residuals.update(
            {
                "rating_sensible_distribution_scale": abs(sensible_scale),
                "rating_latent_distribution_scale": abs(latent_scale),
                "rating_condensate_distribution_scale": abs(condensate_scale),
            }
        )
        annular_fin = replace(
            annular_fin,
            Q_fin_sensible=Q_fin_sensible / fin_count,
            Q_fin_latent=Q_fin_latent / fin_count,
            Q_fin_total=Q_fin_total / fin_count,
            m_dot_condensate_fin=m_fin / fin_count,
            condensate_enthalpy_rate_fin=(
                fin_condensate_enthalpy / fin_count
            ),
            residuals=annular_residuals,
        )

    split_mass_error = m_dot_condensate - (m_primary + m_fin)
    split_energy_error = max(
        abs(Q_total - (Q_sensible + Q_latent)),
        abs(Q_total - (Q_primary_total + Q_fin_total)),
    )
    return replace(
        surface,
        Q_primary_sensible=Q_primary_sensible,
        Q_primary_latent=Q_primary_latent,
        Q_primary_total=Q_primary_total,
        m_dot_condensate_primary=m_primary,
        Q_fin_sensible=Q_fin_sensible,
        Q_fin_latent=Q_fin_latent,
        Q_fin_total=Q_fin_total,
        m_dot_condensate_fin=m_fin,
        Q_sensible=Q_sensible,
        Q_latent=Q_latent,
        Q_total=Q_total,
        m_dot_condensate=m_dot_condensate,
        condensate_enthalpy_rate=condensate_enthalpy_rate,
        mass_balance_error=split_mass_error,
        energy_balance_error=split_energy_error,
        residuals=residuals,
        assumptions=assumptions,
        outside_alpha_wet_effective_basis=(
            surface.outside_alpha_wet_effective_basis
            + "_using_raw_radial_transport_duty_before_rating_"
            "distribution_normalization"
            if normalization_needed
            else surface.outside_alpha_wet_effective_basis
        ),
        annular_fin=annular_fin,
    )


def _rating_raw_available_duty(result, outside: BalanceSideSpec) -> float:
    """Return the physically-driven ("raw", pre-normalization) duty [W]
    implied by one fully-evaluated Rating trial's converged state.

    For an active wet circular-fin trial, this is the nonlinear radial
    surface's own mass-transfer-coefficient-driven duty (the same raw value
    ``_normalize_rating_wet_finned_surface`` scales to match
    ``Q_required`` -- see its residual entry
    ``rating_raw_surface_Q_total_W``), *before* that normalization. For a
    dry/near-onset trial, condensation is inactive by definition, so the
    physically-driven duty is simply the outside stream's own sensible duty
    from its (unaffected-by-the-trial) closed temperature program -- exactly
    what a dry ``close_heat_balance`` would derive from that side alone.
    """
    pc = result.outside_phase_change
    if pc is not None and pc.active:
        wet = result.wet_finned_surface
        if wet is None:
            raise RatingClosureError(
                "Rating closure: active outside condensation on a non-"
                "finned (bare-tube) surface does not yet expose the raw "
                "mass-transfer-driven duty needed to solve an unknown "
                "non-condensing-side mass flow/outlet temperature; this "
                "closure is currently supported only for CircularFinnedTube."
            )
        raw_Q_total = wet.residuals.get("rating_raw_surface_Q_total_W")
        if raw_Q_total is None:
            raise RatingClosureError(
                "Rating closure: the active wet-finned trial did not "
                "report its raw pre-normalization duty."
            )
        return raw_Q_total
    cp_outside_mean = outside.provider.at(
        T=mean_temperature(outside.T_in, outside.T_out), p=outside.p
    ).cp
    return outside.m_dot * cp_outside_mean * abs(outside.T_out - outside.T_in)


def _solve_rating_single_unknown_inside_variable(
    hx,
    inside: BalanceSideSpec,
    outside: BalanceSideSpec,
    *,
    cold_start_m_dot: float,
    cold_start_T_out: float,
    flow_arrangement: str | None,
    K_inlet: float,
    K_outlet: float,
    K_turn: float,
    euler_provider: str,
    finned_heat_transfer_provider: object,
    finned_pressure_drop_provider: object,
    include_simulation: bool,
    over_specified_tolerance: float,
    max_iterations: int,
    wall_temperature_tolerance_K: float,
    relative_alfa_tolerance: float,
    relaxation_factor: float,
    settings: "PhaseChangeSettings",
):
    """Restore Rating's pre-v0.7.5 single-unknown non-condensing-side
    closure (spec sections 1-14) for active outside H2O condensation.

    Exactly one of ``inside.m_dot``/``inside.T_out`` is ``None`` here (the
    caller has already validated this and that ``Q``/``effectiveness`` were
    not given explicitly -- see the call site in
    ``apply_phase_change_to_rating``). The missing value cannot be recovered
    by one algebraic step the way the dry ``close_heat_balance`` path can,
    because the wet outside stream's own duty is not pinned down by its
    (T_in, T_out, m_dot) alone: how much water condenses (hence the total
    latent-inclusive duty) is a genuine extra degree of freedom that only
    the mass-transfer-coefficient-driven physics -- evaluated at the real
    wall temperature, which itself depends on the unknown inside flow via
    alfa_inside -- can resolve.

    This performs an outer scalar root search over the missing inside
    variable. Each trial fills in a concrete value, building a fully
    specified inside ``BalanceSideSpec`` and re-running
    ``apply_phase_change_to_rating`` on it (reusing every existing piece of
    machinery -- close_heat_balance, run_rating, the active-condensation
    fixed point, the wet-finned radial solve -- rather than duplicating any
    of it). Because the trial's inside side is complete,
    ``close_heat_balance`` resolves ``Q_required`` from it directly (its
    ``is_complete(inside)`` branch, checked first) -- this is exactly
    ``Q_required_by_inside_side(trial)`` from the spec, obtained "for free"
    instead of recomputed here. The residual compares that required duty
    against the raw, physically-driven duty the trial's converged HX state
    implies (``Q_available_from_converged_wet_HX(trial)``,
    ``_rating_raw_available_duty``); the two coincide only at the physically
    self-consistent inside variable.

    A trial may resolve dry, near-onset or actively condensing (spec
    section 8): all three are valid, finite HX states, and the residual
    formula spans the transition continuously (see
    ``_rating_raw_available_duty``); only an evaluation that itself raises
    (a non-finite iterate, a property outside its validity range, or a
    genuine solver failure) is treated as an unusable trial point.
    """
    solve_for_m_dot = inside.m_dot is None
    assert solve_for_m_dot != (inside.T_out is None), (
        "exactly one of inside.m_dot/inside.T_out must be unknown here"
    )
    T_in_inside = inside.T_in
    if T_in_inside is None:
        raise ValueError("Rating closure requires inside.T_in.")

    def evaluate(trial_value: float):
        inside_trial = (
            replace(inside, m_dot=trial_value)
            if solve_for_m_dot
            else replace(inside, T_out=trial_value)
        )
        return apply_phase_change_to_rating(
            hx, inside_trial, outside,
            Q=None, effectiveness=None,
            flow_arrangement=flow_arrangement,
            K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
            euler_provider=euler_provider,
            finned_heat_transfer_provider=finned_heat_transfer_provider,
            finned_pressure_drop_provider=finned_pressure_drop_provider,
            # The achievable-duty Simulation bridge is only useful on the
            # final, converged trial -- running it on every search step
            # would repeat an expensive nested Simulation for nothing.
            include_simulation=False,
            over_specified_tolerance=over_specified_tolerance,
            max_iterations=max_iterations,
            wall_temperature_tolerance_K=wall_temperature_tolerance_K,
            relative_alfa_tolerance=relative_alfa_tolerance,
            relaxation_factor=relaxation_factor,
            settings=settings,
        )

    def residual(trial_value: float) -> float:
        result = evaluate(trial_value)
        return _rating_raw_available_duty(result, outside) - result.Q_required

    if solve_for_m_dot:
        x0 = max(cold_start_m_dot, _RATING_CLOSURE_MASS_FLOW_FLOOR_KG_S * 10.0)
        x_min = _RATING_CLOSURE_MASS_FLOW_FLOOR_KG_S
        x_max = math.inf
        # A relative tolerance on the flow itself, not an absurdly tight
        # absolute one -- the underlying wet-finned solve's own tolerances
        # (settings.relative_Q_tolerance etc.) already bound how precisely
        # any single trial evaluation is meaningful.
        x_tolerance = max(x0 * 1e-6, 1e-9)
        variable_name = "inside.m_dot"
        initial_step = None  # relative-to-x0 default is appropriate here
        cp_inside_scale = inside.provider.at(
            T=mean_temperature(T_in_inside, inside.T_out), p=inside.p
        ).cp
        Q_scale = abs(x0 * cp_inside_scale * (inside.T_out - T_in_inside))
    else:
        hot_is_inside = T_in_inside >= outside.T_in
        margin = _RATING_CLOSURE_TEMPERATURE_MARGIN_K
        if hot_is_inside:
            x_min, x_max = outside.T_in + margin, T_in_inside - margin
        else:
            x_min, x_max = T_in_inside + margin, outside.T_in - margin
        if not (x_min < x_max):
            raise RatingClosureError(
                "Rating closure: the inside and outside inlet temperatures "
                "leave no physically valid range for the unknown inside "
                "outlet temperature (inside.T_in="
                f"{T_in_inside:.6g} K, outside.T_in={outside.T_in:.6g} K)."
            )
        x0 = min(max(cold_start_T_out, x_min), x_max)
        x_tolerance = max(wall_temperature_tolerance_K * 1.0e-2, 1e-4)
        variable_name = "inside.T_out"
        # A modest Kelvin step, not abs(x0)*0.5 (x0 is a ~300 K absolute
        # temperature, not a quantity whose natural scale is its own
        # magnitude) -- scaled to the initial approach so a cold-start
        # estimate close to T_in still gets a sensibly small first step.
        initial_step = max(abs(x0 - T_in_inside) * 0.25, 1.0)
        cp_inside_scale = inside.provider.at(
            T=mean_temperature(T_in_inside, x0), p=inside.p
        ).cp
        Q_scale = abs(inside.m_dot * cp_inside_scale * (x0 - T_in_inside))

    # A relative Watt tolerance on the residual, scaled to this problem's
    # own duty (reusing the existing settings.relative_Q_tolerance idiom
    # rather than an unscaled absolute value that would be meaninglessly
    # tight for a MW-scale duty and meaninglessly loose for a W-scale one).
    residual_tolerance = max(1.0, settings.relative_Q_tolerance * Q_scale)
    x_root, _, closure_iterations = _bracket_and_solve_monotonic_root(
        residual, x0,
        x_min=x_min, x_max=x_max,
        x_tolerance=x_tolerance,
        residual_tolerance=residual_tolerance,
        variable_name=variable_name,
        initial_step=initial_step,
    )

    inside_final = (
        replace(inside, m_dot=x_root)
        if solve_for_m_dot
        else replace(inside, T_out=x_root)
    )
    final_result = apply_phase_change_to_rating(
        hx, inside_final, outside,
        Q=None, effectiveness=None,
        flow_arrangement=flow_arrangement,
        K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
        euler_provider=euler_provider,
        finned_heat_transfer_provider=finned_heat_transfer_provider,
        finned_pressure_drop_provider=finned_pressure_drop_provider,
        include_simulation=include_simulation,
        over_specified_tolerance=over_specified_tolerance,
        max_iterations=max_iterations,
        wall_temperature_tolerance_K=wall_temperature_tolerance_K,
        relative_alfa_tolerance=relative_alfa_tolerance,
        relaxation_factor=relaxation_factor,
        settings=settings,
    )
    final_residual = (
        _rating_raw_available_duty(final_result, outside)
        - final_result.Q_required
    )
    closure_tag = (
        "rating_closure_solved_inside_m_dot"
        if solve_for_m_dot
        else "rating_closure_solved_inside_T_out"
    )
    closure_diagnostics = {
        "rating_closure_iterations": float(closure_iterations),
        "rating_closure_residual_W": final_residual,
        "rating_closure_bracket_low": x_min,
        "rating_closure_bracket_high": (
            x_max if math.isfinite(x_max) else float("nan")
        ),
    }
    outside_phase_change = replace(
        final_result.outside_phase_change,
        residuals={
            **final_result.outside_phase_change.residuals,
            **closure_diagnostics,
        },
        assumptions=(
            *final_result.outside_phase_change.assumptions,
            closure_tag,
        ),
    )
    return replace(final_result, outside_phase_change=outside_phase_change)


def apply_phase_change_to_rating(
    hx,
    inside: BalanceSideSpec,
    outside: BalanceSideSpec,
    *,
    Q: float | None = None,
    effectiveness: float | None = None,
    flow_arrangement: str | None = None,
    K_inlet: float = 0.5,
    K_outlet: float = 1.0,
    K_turn: float = 1.5,
    euler_provider: str = "zukauskas",
    finned_heat_transfer_provider: object = DEFAULT_FINNED_HT_PROVIDER,
    finned_pressure_drop_provider: object = DEFAULT_FINNED_DP_PROVIDER,
    include_simulation: bool = False,
    over_specified_tolerance: float = 1e-3,
    max_iterations: int = 25,
    wall_temperature_tolerance_K: float = 0.05,
    relative_alfa_tolerance: float = 1e-3,
    relaxation_factor: float = 0.5,
    settings: PhaseChangeSettings | None = None,
):
    """Return an ``HXRatingResult`` with phase-change results applied.

    Backing implementation of ``BareTubeHeatExchanger.rate``. See the
    module docstring for the algorithm and its scope limits (raises
    ``ValueError`` for under-specified/inconsistent condensing cases,
    rather than guessing).
    """
    settings = settings or PhaseChangeSettings()

    guarded_inside_provider = (
        inside.provider
        if inside.T_in is None
        else guard_pure_water_single_phase_provider(
            inside.provider, T_in=inside.T_in, p=inside.p
        )
    )
    guarded_outside_provider = (
        outside.provider
        if outside.T_in is None
        else guard_pure_water_single_phase_provider(
            outside.provider, T_in=outside.T_in, p=outside.p
        )
    )
    guarded_inside = (
        inside
        if guarded_inside_provider is inside.provider
        else replace(inside, provider=guarded_inside_provider)
    )
    guarded_outside = (
        outside
        if guarded_outside_provider is outside.provider
        else replace(outside, provider=guarded_outside_provider)
    )
    closed_balance = close_heat_balance(
        guarded_inside, guarded_outside, Q=Q, effectiveness=effectiveness,
        over_specified_tolerance=over_specified_tolerance,
    )
    dry_simulation_balance = closed_balance
    dry_result = run_rating(
        hx, closed_balance,
        flow_arrangement=flow_arrangement, K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
        # Defer the optional bridge until onset has selected the dry or active
        # route. An active outside wet Rating gets one phase-aware Simulation
        # after convergence, never a discarded dry snapshot before the solve.
        euler_provider=euler_provider, include_simulation=False,
        finned_heat_transfer_provider=finned_heat_transfer_provider,
        finned_pressure_drop_provider=finned_pressure_drop_provider,
        max_iterations=max_iterations, wall_temperature_tolerance_K=wall_temperature_tolerance_K,
        relative_alfa_tolerance=relative_alfa_tolerance, relaxation_factor=relaxation_factor,
        # A closed balance built from a trial value of the single-unknown
        # Rating closure (spec sections 1-14) can imply a duty beyond
        # sensible-only capacity even though the *actual* (condensation-
        # aware) problem is perfectly feasible -- this dry-baseline pass
        # exists only to estimate wall temperatures for the onset decision
        # below, not to report a final authoritative dry answer, so it must
        # not raise on that alone. A trial that turns out genuinely dry AND
        # infeasible is still rejected below, once the "not active" branch
        # is confirmed as the final route.
        allow_infeasible_sensible_effectiveness=True,
    )

    def attach_requested_dry_simulation(result):
        """Restore the legacy dry Rating bridge only on a dry return path."""
        if not include_simulation or result.simulation is not None:
            return result
        from core.models.simulation import run_simulation

        simulation = run_simulation(
            hx,
            dry_simulation_balance.inside.to_hx_side_input(),
            dry_simulation_balance.outside.to_hx_side_input(),
            flow_arrangement=flow_arrangement,
            K_inlet=K_inlet,
            K_outlet=K_outlet,
            K_turn=K_turn,
            euler_provider=euler_provider,
            finned_heat_transfer_provider=finned_heat_transfer_provider,
            finned_pressure_drop_provider=finned_pressure_drop_provider,
        )
        return replace(
            result,
            simulation=simulation,
            Q_achievable=simulation.q,
        )

    reject_unsupported_pure_water_phase_crossing(
        inside.provider,
        T_in=inside.T_in,
        T_out=closed_balance.inside.T_out,
        p=inside.p,
    )
    reject_unsupported_pure_water_phase_crossing(
        outside.provider,
        T_in=outside.T_in,
        T_out=closed_balance.outside.T_out,
        p=outside.p,
    )
    if guarded_inside is not inside or guarded_outside is not outside:
        closed_balance = replace(
            closed_balance,
            inside=replace(closed_balance.inside, provider=inside.provider),
            outside=replace(closed_balance.outside, provider=outside.provider),
        )
        dry_result = replace(dry_result, closed_balance=closed_balance)

    inside_capability = detect_phase_change_capability(inside.provider)
    outside_capability = detect_phase_change_capability(outside.provider)

    raise_if_inside_pure_steam_condensation(inside, dry_result)

    if not inside_capability.capable and not outside_capability.capable:
        dry_result = attach_requested_dry_simulation(dry_result)
        return replace(
            dry_result,
            inside_phase_change=build_capability_side_result(
                side="inside", mode=inside.phase_change_mode, capability=inside_capability,
                possible=False, near_onset=False, dew_point=None, p=inside.p,
                m_dot_gas=inside.m_dot,
                Q_sensible_actual=dry_result.Q_required,
            ),
            outside_phase_change=build_capability_side_result(
                side="outside", mode=outside.phase_change_mode, capability=outside_capability,
                possible=False, near_onset=False, dew_point=None, p=outside.p,
                m_dot_gas=outside.m_dot,
                Q_sensible_actual=dry_result.Q_required,
            ),
        )

    thermal_state = dry_result.thermal_state
    envelope = dry_result.wall_temperature_envelope

    # Fix (v0.6.0 patch, spec section 6.1): onset uses wall_envelope.<side>_min
    # (the coldest estimated point), not a mean/representative wall
    # temperature -- see core.phase_change.integration.evaluate_side_onset.
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
        context="wet-gas condensation rating",
        outside_capability=outside_capability,
        direction=PhaseChangeDirection.CONDENSATION,
    )

    # Rating has no iterate=False escape hatch, so the guard in
    # check_single_active_side never fires here (iterate=True always).
    check_single_active_side(inside_auto_possible, outside_auto_possible, iterate=True)

    if inside_auto_possible:
        return _apply_inside_condensation_to_rating(
            hx,
            inside=inside,
            outside=outside,
            closed_balance=closed_balance,
            dry_result=dry_result,
            Q=Q,
            flow_arrangement=flow_arrangement,
            K_inlet=K_inlet,
            K_outlet=K_outlet,
            K_turn=K_turn,
            euler_provider=euler_provider,
            finned_heat_transfer_provider=finned_heat_transfer_provider,
            finned_pressure_drop_provider=finned_pressure_drop_provider,
            include_simulation=include_simulation,
            max_iterations=max_iterations,
            wall_temperature_tolerance_K=wall_temperature_tolerance_K,
            relative_alfa_tolerance=relative_alfa_tolerance,
            relaxation_factor=relaxation_factor,
            settings=settings,
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
        )

    inside_result = build_capability_side_result(
        side="inside", mode=inside.phase_change_mode, capability=inside_capability,
        possible=inside_possible, near_onset=inside_near_onset,
        dew_point=inside_dew_point, p=inside.p,
        m_dot_gas=inside.m_dot,
        onset=inside_onset, wall_temperature_min=inside_wall_min,
        wall_temperature_mean=inside_wall_mean, wall_temperature_max=inside_wall_max,
        Q_sensible_actual=dry_result.Q_required,
    )

    if not outside_auto_possible:
        if not math.isfinite(dry_result.UA_required):
            # The dry-baseline pass above tolerated a sensible-only
            # effectiveness outside [0, 1) so it could still estimate wall
            # temperatures for the onset decision (see its
            # allow_infeasible_sensible_effectiveness=True call), but the
            # regime has now been confirmed dry/near-onset -- condensation
            # is not activating to explain the gap. This specified duty is
            # genuinely unreachable by any dry exchanger area.
            raise ValueError(
                "Rating: the specified/implied duty exceeds what any "
                "purely-sensible (dry) exchanger area could deliver "
                "(sensible-only effectiveness outside [0, 1)), and outside "
                "H2O condensation is not active to account for the "
                "difference. The specified Rating temperature program is "
                "not physically achievable."
            )
        outside_result = build_capability_side_result(
            side="outside", mode=outside.phase_change_mode, capability=outside_capability,
            possible=outside_possible, near_onset=outside_near_onset,
            dew_point=outside_dew_point, p=outside.p,
            m_dot_gas=outside.m_dot,
            onset=outside_onset, wall_temperature_min=outside_wall_min,
            wall_temperature_mean=outside_wall_mean, wall_temperature_max=outside_wall_max,
            Q_sensible_actual=dry_result.Q_required,
        )
        dry_result = attach_requested_dry_simulation(dry_result)
        return replace(dry_result, inside_phase_change=inside_result, outside_phase_change=outside_result)

    # --- Active outside-condensation rating ---------------------------------
    if outside.m_dot is None:
        raise ValueError(
            "Rating with active outside condensation requires outside.m_dot "
            "(the total wet-gas mass flow) to be specified explicitly; it "
            "cannot be solved for once condensation is active."
        )
    if outside.T_out is None:
        raise ValueError(
            "Rating with active outside condensation requires outside.T_out "
            "(the outlet temperature) to be specified explicitly."
        )
    inside_has_m_dot = inside.m_dot is not None
    inside_has_T_out = inside.T_out is not None
    if Q is None and effectiveness is None and inside_has_m_dot != inside_has_T_out:
        # Exactly one of the non-condensing inside side's m_dot/T_out is
        # unknown (spec sections 1-14): restore the pre-v0.7.5 Rating
        # closure capability via an outer scalar solve, rather than
        # rejecting a previously valid Rating problem outright. The wet
        # outside side's own duty is not pinned down by its (T_in, T_out,
        # m_dot) alone once condensation is active (how much water
        # condenses is a genuine extra degree of freedom), so this single
        # remaining unknown cannot be recovered by one more algebraic
        # close_heat_balance step the way the dry case can.
        return _solve_rating_single_unknown_inside_variable(
            hx, inside, outside,
            cold_start_m_dot=closed_balance.inside.m_dot,
            cold_start_T_out=closed_balance.inside.T_out,
            flow_arrangement=flow_arrangement,
            K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
            euler_provider=euler_provider,
            finned_heat_transfer_provider=finned_heat_transfer_provider,
            finned_pressure_drop_provider=finned_pressure_drop_provider,
            include_simulation=include_simulation,
            over_specified_tolerance=over_specified_tolerance,
            max_iterations=max_iterations,
            wall_temperature_tolerance_K=wall_temperature_tolerance_K,
            relative_alfa_tolerance=relative_alfa_tolerance,
            relaxation_factor=relaxation_factor,
            settings=settings,
        )
    if Q is None and not (inside_has_m_dot and inside_has_T_out):
        raise ValueError(
            "Rating with active outside condensation is underdetermined: "
            "the duty must come from an explicit Q, an explicit "
            "effectiveness, a fully specified (m_dot + T_in + T_out) "
            "inside side, or exactly one of inside.m_dot/inside.T_out left "
            "unknown for the solver to close -- it cannot be inferred from "
            "the outside (condensing) side's sensible heat capacity alone, "
            "since that would silently ignore latent duty."
        )

    Q_required = closed_balance.Q
    W_in = outside_capability.W_in
    if W_in is None:
        raise ValueError("Active outside condensation requires an inlet water ratio.")
    enthalpy_evaluator = WetGasEnthalpyEvaluator(outside.p, outside_capability)
    m_dot_dry_carrier = outside.m_dot / (1.0 + W_in)
    is_wet_finned = (
        getattr(hx.bundle.tube, "surface_type", None)
        is TubeSurfaceType.CIRCULAR_FINNED
    )
    M_dry = outside_capability.M_dry
    M_h2o = outside_capability.M_condensable
    if is_wet_finned and (M_dry is None or M_h2o is None):
        raise ValueError(
            "Active wet-finned Rating requires dry-carrier and water molar "
            "masses for the radial surface solve."
        )
    finned_wet_zone_fallback = (
        _finned_0d_wet_zone_fallback(
            dew_point_temperature=outside_dew_point,
            wall_temperature_min=outside_wall_min,
            wall_temperature_max=outside_wall_max,
            activation_band_K=settings.activation_band_K,
        )
        if is_wet_finned
        else None
    )

    dT_outside = outside.T_in - outside.T_out
    if abs(dT_outside) < 1e-9:
        raise ValueError("Rating with active outside condensation requires T_in != T_out on the outside side.")
    C_effective_outside = Q_required / abs(dT_outside)

    C_min = min(closed_balance.inside.C, C_effective_outside)
    if closed_balance.hot_is_inside:
        Th_in, Tc_in = closed_balance.inside.T_in, outside.T_in
    else:
        Th_in, Tc_in = outside.T_in, closed_balance.inside.T_in
    Q_max = C_min * (Th_in - Tc_in)
    eff = Q_required / Q_max if Q_max > 0.0 else math.nan

    balance_warnings: list[ModelWarning] = [
        make_warning(
            code=WC.EFFECTIVE_CAPACITY_RATE_0D_APPROXIMATION,
            message=(
                "outside: the outside stream's capacity rate used for the "
                "epsilon-NTU/overdesign relations is an effective value "
                f"(C_effective={C_effective_outside:.6g} W/K = Q_required/"
                "|T_in-T_out|) derived AFTER closing the mass/enthalpy "
                "balance, not a constant-cp sensible capacity rate -- a 0D "
                "approximation for interoperating with the existing "
                "epsilon-NTU overdesign model."
            ),
            source=SOURCE,
            severity="info",
        )
    ]

    def run_wet_rating(W_out_for_rating: float):
        W_mean_for_rating = 0.5 * (W_in + W_out_for_rating)
        provider_mean = wet_gas_provider_at_water_ratio(
            outside_capability, W_mean_for_rating
        )
        m_dot_mean_for_rating = m_dot_dry_carrier * (
            1.0 + W_mean_for_rating
        )
        outside_closed_side = ClosedBalanceSide(
            provider=provider_mean,
            p=outside.p,
            m_dot=m_dot_mean_for_rating,
            T_in=outside.T_in,
            T_out=outside.T_out,
            cp_mean=C_effective_outside / m_dot_mean_for_rating,
            C=C_effective_outside,
        )
        wet_balance = ClosedBalance(
            inside=closed_balance.inside,
            outside=outside_closed_side,
            hot_is_inside=closed_balance.hot_is_inside,
            Q=Q_required,
            Q_max=Q_max,
            effectiveness=eff,
            warnings=balance_warnings,
        )
        result = run_rating(
            hx,
            wet_balance,
            flow_arrangement=flow_arrangement,
            K_inlet=K_inlet,
            K_outlet=K_outlet,
            K_turn=K_turn,
            euler_provider=euler_provider,
            finned_heat_transfer_provider=finned_heat_transfer_provider,
            finned_pressure_drop_provider=finned_pressure_drop_provider,
            # A phase-aware achievable Simulation is run once after this wet
            # Rating fixed point converges. Calling the dry driver here would
            # repeat it on every outer iteration and expose a misleading
            # sensible-only nested result.
            include_simulation=False,
            max_iterations=max_iterations,
            wall_temperature_tolerance_K=wall_temperature_tolerance_K,
            relative_alfa_tolerance=relative_alfa_tolerance,
            relaxation_factor=relaxation_factor,
        )
        return result

    # Regime is locked active from the dry-baseline onset decision above.
    # Only the condensate temperature / enthalpy root / wet Rating state are
    # coupled here; Rating still does not impose a transport-limited design
    # feasibility check.
    condensate_temperature = outside.T_out
    previous_W_out: float | None = None
    wet_rating_result = None
    wet_surface = None
    wet_finned_surface = None
    dew_point_mean = None
    rating_wall_min = None
    rating_wall_max = None
    T_wall_outside_repr = None
    wet_wall_temperature = None
    W_out_iter = W_in
    root_enthalpy_residual = math.inf
    enthalpy_balance_residual = math.inf
    outer_residuals: dict[str, float] = {}
    outer_converged = False
    finned_wet_zone_fallback_locked = False
    outer_limit = min(settings.max_iterations, _RATING_OUTER_MAX_ITERATIONS)

    for outer_iteration in range(1, outer_limit + 1):
        W_out_iter, root_enthalpy_residual = _solve_rating_water_ratio_for_side(
            wet_side=outside,
            capability=outside_capability,
            Q_required=Q_required,
            m_dot_dry_carrier=m_dot_dry_carrier,
            condensate_temperature=condensate_temperature,
            enthalpy_evaluator=enthalpy_evaluator,
        )
        wet_rating_result = run_wet_rating(W_out_iter)

        W_mean_iter = 0.5 * (W_in + W_out_iter)
        dew_point_mean = dew_point_at_ratio(
            outside_capability, W_mean_iter, p=outside.p
        )
        if dew_point_mean is None:
            from core.properties.water import WATER_TRIPLE_POINT_TEMPERATURE_K

            dew_point_mean = WATER_TRIPLE_POINT_TEMPERATURE_K

        if is_wet_finned:
            thermal_state_iter = wet_rating_result.thermal_state
            dry_contact_network = calculate_resistance_network(
                bundle=hx.bundle,
                alpha_inside=thermal_state_iter.alfa_i,
                outside_alpha_physical=(
                    thermal_state_iter.outside_alpha_physical
                ),
                resistance_core_wall=hx.tube_wall_resistance(),
            )
            assert M_dry is not None and M_h2o is not None
            def solve_rating_finned_surface(
                *,
                condensation_area_fraction: float = 1.0,
                condensation_temperature_offset_K: float = 0.0,
            ) -> WetFinnedSurfaceResult:
                return solve_wet_finned_surface(
                    hx.bundle,
                    dry_contact_network,
                    gas_bulk_temperature=(
                        thermal_state_iter.outside_bulk_temperature
                    ),
                    inside_bulk_temperature=(
                        thermal_state_iter.inside_bulk_temperature
                    ),
                    cp_gas=thermal_state_iter.outside_bulk_props.cp,
                    W_bulk=W_mean_iter,
                    p_total=outside.p,
                    M_dry=M_dry,
                    M_h2o=M_h2o,
                    m_dot_water_vapor_available=(
                        m_dot_dry_carrier * W_mean_iter
                    ),
                    lewis_number=settings.lewis_number,
                    condensation_area_fraction=condensation_area_fraction,
                    condensation_temperature_offset_K=(
                        condensation_temperature_offset_K
                    ),
                    max_iterations=settings.max_iterations,
                    temperature_tolerance_K=min(
                        settings.temperature_tolerance_K,
                        settings.wall_temperature_tolerance_K,
                        1.0e-6,
                    ),
                    relative_heat_tolerance=min(
                        settings.relative_Q_tolerance,
                        1.0e-8,
                    ),
                    condensate_tolerance_kg_s=(
                        min(settings.condensate_tolerance_kg_s, 1.0e-10)
                    ),
                    # The radial solve is an internally line-searched Newton
                    # problem. Rating's outer fixed-point relaxation must not
                    # add a second damping layer to that local solve.
                    relaxation_factor=1.0,
                )

            if finned_wet_zone_fallback_locked:
                assert finned_wet_zone_fallback is not None
                wet_finned_surface = solve_rating_finned_surface(
                    condensation_area_fraction=(
                        finned_wet_zone_fallback[0]
                    ),
                    condensation_temperature_offset_K=(
                        finned_wet_zone_fallback[1]
                    ),
                )
            else:
                wet_finned_surface = solve_rating_finned_surface()
            if (
                not finned_wet_zone_fallback_locked
                and wet_finned_surface.m_dot_condensate == 0.0
                and finned_wet_zone_fallback is not None
            ):
                wet_finned_surface = solve_rating_finned_surface(
                    condensation_area_fraction=(
                        finned_wet_zone_fallback[0]
                    ),
                    condensation_temperature_offset_K=(
                        finned_wet_zone_fallback[1]
                    ),
                )
                finned_wet_zone_fallback_locked = True
            T_wall_outside_repr = wet_finned_surface.core_wall_temperature
            exposed_temperatures = (
                wet_finned_surface.primary_surface_temperature,
                wet_finned_surface.fin_base_temperature,
                wet_finned_surface.fin_tip_temperature,
            )
            rating_wall_min = min(exposed_temperatures) + (
                wet_finned_surface.condensation_temperature_offset_K
            )
            rating_wall_max = max(exposed_temperatures)
            wet_wall_temperature = (
                wet_finned_surface.wall_temperature_wet_mean
            )
            wet_fraction_iter = wet_finned_surface.wet_surface_fraction
        else:
            (
                rating_wall_min,
                T_wall_outside_repr,
                rating_wall_max,
            ) = _centered_wall_bounds(wet_rating_result, side="outside")
            wet_surface = estimate_wet_surface_fraction(
                dew_point_temperature=dew_point_mean,
                wall_temperature_min=rating_wall_min,
                wall_temperature_mean=T_wall_outside_repr,
                wall_temperature_max=rating_wall_max,
                activation_band_K=settings.activation_band_K,
            )
            wet_wall_temperature = wet_surface.wall_temperature_wet_mean
            wet_fraction_iter = wet_surface.wet_surface_fraction
        if wet_fraction_iter <= 0.0 or wet_wall_temperature is None:
            raise ValueError(
                "Rating's active outside-condensation regime produced no "
                "wet wall area after the wet-state update. The active regime "
                "is locked and is not silently replaced by a dry Rating."
            )
        if not all(
            math.isfinite(value)
            for value in (
                W_out_iter,
                wet_fraction_iter,
                wet_wall_temperature,
                rating_wall_min,
                T_wall_outside_repr,
                rating_wall_max,
                dew_point_mean,
                root_enthalpy_residual,
            )
        ):
            raise ValueError(
                "Rating with active outside condensation produced a "
                f"non-finite outer iterate at iteration {outer_iteration}."
            )

        W_out_residual = (
            abs(W_out_iter - previous_W_out)
            if previous_W_out is not None
            else math.inf
        )
        W_out_at_wet_wall, _ = _solve_rating_water_ratio_for_side(
            wet_side=outside,
            capability=outside_capability,
            Q_required=Q_required,
            m_dot_dry_carrier=m_dot_dry_carrier,
            condensate_temperature=wet_wall_temperature,
            enthalpy_evaluator=enthalpy_evaluator,
        )
        W_out_fixed_point_residual = abs(
            W_out_at_wet_wall - W_out_iter
        )
        wet_wall_residual = abs(wet_wall_temperature - condensate_temperature)
        failed_envelope_probes = sum(
            not probe.converged
            for probe in wet_rating_result.wall_temperature_envelope.probes
        )
        enthalpy_balance_residual = _rating_enthalpy_residual_at_state(
            wet_side=outside,
            capability=outside_capability,
            Q_required=Q_required,
            m_dot_dry_carrier=m_dot_dry_carrier,
            W_out=W_out_iter,
            condensate_temperature=wet_wall_temperature,
            enthalpy_evaluator=enthalpy_evaluator,
        )
        outer_residuals = {
            "W_out": max(W_out_residual, W_out_fixed_point_residual),
            "W_out_successive": W_out_residual,
            "W_out_fixed_point": W_out_fixed_point_residual,
            "T_wall_wet_mean_K": wet_wall_residual,
            "wall_envelope_failed_probes": float(failed_envelope_probes),
            "outside_enthalpy_balance_W": abs(enthalpy_balance_residual),
            "outside_enthalpy_root_W": abs(root_enthalpy_residual),
        }
        if wet_finned_surface is not None:
            outer_residuals.update(
                {
                    f"wet_finned_{name}": value
                    for name, value in wet_finned_surface.residuals.items()
                }
            )

        if (
            outer_iteration >= 2
            and W_out_residual < settings.water_ratio_tolerance
            and W_out_fixed_point_residual < settings.water_ratio_tolerance
            and wet_wall_residual < settings.wall_temperature_tolerance_K
            and abs(enthalpy_balance_residual)
            < _RATING_ENTHALPY_SYNCHRONIZATION_TOLERANCE_W
            and wet_rating_result.thermal_state.converged
            and failed_envelope_probes == 0
        ):
            outer_converged = True
            break

        previous_W_out = W_out_iter
        # This small algebraic fixed point is weakly coupled through liquid
        # enthalpy and W_mean. A direct bounded update reaches a genuinely
        # synchronized Rating state without a second post-processing loop
        # that could change W_out after the last run_rating call.
        condensate_temperature = wet_wall_temperature

    if (
        wet_rating_result is None
        or (is_wet_finned and wet_finned_surface is None)
        or (not is_wet_finned and wet_surface is None)
        or dew_point_mean is None
        or rating_wall_min is None
        or rating_wall_max is None
        or T_wall_outside_repr is None
        or wet_wall_temperature is None
    ):
        raise ValueError(
            "Rating with active outside condensation failed to produce a "
            "complete finite outer iterate."
        )

    if not outer_converged:
        balance_warnings.append(
            make_warning(
                code=WC.OUTSIDE_CONDENSATION_NOT_CONVERGED,
                message=(
                    "phase_change_rating_integration: wet-wall/enthalpy "
                    f"fixed point did not converge within {outer_limit} "
                    f"iterations. Last residuals: {outer_residuals}. "
                    "Returning the last complete, finite and internally "
                    "same-state Rating iterate."
                ),
                source=SOURCE,
                severity="warning",
            )
        )
    rating_coupled_converged = (
        outer_converged
        and wet_rating_result.thermal_state.converged
        and all(
            probe.converged
            for probe in wet_rating_result.wall_temperature_envelope.probes
        )
    )

    # The loop's last Rating run used this exact W_out. Its active surface
    # response produced this exact wet fraction and wet-wall temperature, and
    # the enthalpy residual above was evaluated at that reported state.
    W_out = W_out_iter
    W_mean = 0.5 * (W_in + W_out)
    outside_provider_mean = wet_gas_provider_at_water_ratio(
        outside_capability, W_mean
    )

    m_dot_condensate = m_dot_dry_carrier * (W_in - W_out)
    m_dot_water_vapor_in = m_dot_dry_carrier * W_in
    m_dot_water_vapor_out = m_dot_dry_carrier * W_out

    # ``run_rating`` intentionally uses the representative W_mean provider
    # for its 0D thermal calculation. Refresh only its existing outside
    # hydraulic snapshot so the public point diagnostics retain the real wet
    # endpoints: W_in/inlet gas flow, W_mean/mean gas flow, and W_out/outlet
    # gas flow. This mirrors Simulation and does not introduce a second state
    # model or change the condensation/Rating equations.
    m_dot_gas_mean = m_dot_dry_carrier * (1.0 + W_mean)
    m_dot_gas_out = m_dot_dry_carrier + m_dot_water_vapor_out
    outside_provider_inlet = wet_gas_provider_at_water_ratio(outside_capability, W_in)
    outside_provider_outlet = wet_gas_provider_at_water_ratio(outside_capability, W_out)
    T_mean_outside = 0.5 * (outside.T_in + outside.T_out)
    outside_props_inlet = outside_provider_inlet.at(
        T=outside.T_in, p=outside.p
    )
    outside_props_midpoint = outside_provider_mean.at(
        T=T_mean_outside, p=outside.p
    )
    outside_props_outlet = outside_provider_outlet.at(
        T=outside.T_out, p=outside.p
    )
    if is_wet_finned:
        outside_bank_hydraulic = evaluate_outside_hydraulics(
            bundle=hx.bundle,
            m_dot=m_dot_gas_mean,
            inlet_props=outside_props_inlet,
            midpoint_props=outside_props_midpoint,
            outlet_props=outside_props_outlet,
            temperature_in=outside.T_in,
            temperature_out=outside.T_out,
            pressure=outside.p,
            euler_provider=euler_provider,
            finned_pressure_drop_provider=finned_pressure_drop_provider,
            m_dot_inlet=outside.m_dot,
            m_dot_midpoint=m_dot_gas_mean,
            m_dot_outlet=m_dot_gas_out,
        )
    else:
        # Preserve the historical bare-tube Rating calculation exactly.
        outside_bank_hydraulic = calculate_outside_tube_bank_hydraulics(
            m_dot=m_dot_gas_mean,
            face_area=hx.bundle.frontal_flow_area,
            tube_outer_diameter=float(getattr(hx.bundle.tube, "D_o")),
            tube_pitch_transverse=hx.bundle.pitch_transverse,
            tube_pitch_longitudinal=hx.bundle.pitch_longitudinal,
            layout=hx.bundle.layout,
            n_rows=hx.bundle.n_rows,
            n_tubes_per_row=hx.bundle.n_tubes_per_row,
            inlet_props=outside_props_inlet,
            midpoint_props=outside_props_midpoint,
            outlet_props=outside_props_outlet,
            temperature_in=outside.T_in,
            temperature_out=outside.T_out,
            pressure=outside.p,
            euler_provider=euler_provider,
            m_dot_inlet=outside.m_dot,
            m_dot_midpoint=m_dot_gas_mean,
            m_dot_outlet=m_dot_gas_out,
        )
    outside_bank_hydraulic = replace(
        outside_bank_hydraulic,
        midpoint_method="arithmetic_temperature_and_water_ratio",
    )
    wet_final_result = replace(
        wet_rating_result.final_result,
        outside_side_hydraulic=replace(
            wet_rating_result.final_result.outside_side_hydraulic,
            dp_total=outside_bank_hydraulic.dp_total,
            Re=outside_bank_hydraulic.midpoint.reynolds,
            v=outside_bank_hydraulic.midpoint.face_velocity,
            tube_bank=outside_bank_hydraulic,
        ),
        outside_side_pressure_drop=replace(
            wet_rating_result.final_result.outside_side_pressure_drop,
            tube_bank=outside_bank_hydraulic,
            flow_path=build_outside_pressure_drop_result(outside_bank_hydraulic),
        ),
    )
    wet_rating_result = replace(wet_rating_result, final_result=wet_final_result)

    h_fg_representative = water_latent_heat_of_vaporization(
        T=wet_wall_temperature
    )
    Q_latent = m_dot_condensate * h_fg_representative
    Q_sensible = Q_required - Q_latent
    if Q_sensible < 0.0:
        balance_warnings.append(
            make_warning(
                code="phase_change_negative_sensible_duty",
                message=(
                    f"outside: Q_sensible={Q_sensible:.6g} W came out negative "
                    "from the closed enthalpy balance; verify the specified "
                    "temperature program and water content."
                ),
                source=SOURCE,
                severity="warning",
            )
        )

    if is_wet_finned:
        assert wet_finned_surface is not None
        raw_wet_finned_surface = wet_finned_surface
        alfa_o_dry = raw_wet_finned_surface.outside_alpha_physical
        T_wall_outside_repr = raw_wet_finned_surface.core_wall_temperature
        # This named wet coefficient comes from the same raw radial response
        # that set the converged wall temperatures and wet areas.  Its type
        # states the gross-area / bulk-gas-to-core-wall basis explicitly;
        # latent duty is never relabelled as a physical Briggs-Young HTC.
        alfa_o_effective = (
            raw_wet_finned_surface.outside_alpha_wet_effective_gross_core_basis
        )
        if not math.isfinite(alfa_o_effective) or alfa_o_effective <= 0.0:
            raise ValueError(
                "Wet-finned Rating produced a non-positive effective "
                "gross-area outside conductance."
            )

        wet_area = raw_wet_finned_surface.wet_area
        wet_surface_fraction = (
            raw_wet_finned_surface.wet_surface_fraction
        )
        wet_surface_fraction_method = (
            "nonlinear_radial_annular_fin_fvm_plus_primary_surface"
            + (
                "_with_0d_endpoint_wet_zone_fallback"
                if (
                    raw_wet_finned_surface.condensation_area_fraction < 1.0
                    or raw_wet_finned_surface
                    .condensation_temperature_offset_K < 0.0
                )
                else ""
            )
        )
        W_sat_wet_surface = raw_wet_finned_surface.W_sat_wet_surface
        wet_finned_surface = _normalize_rating_wet_finned_surface(
            raw_wet_finned_surface,
            Q_sensible=Q_sensible,
            Q_latent=Q_latent,
            Q_total=Q_required,
            m_dot_condensate=m_dot_condensate,
            condensate_temperature=wet_wall_temperature,
        )
        outer_residuals.update(
            {
                f"wet_finned_{name}": value
                for name, value in wet_finned_surface.residuals.items()
            }
        )
    else:
        assert wet_surface is not None
        alfa_o_dry = wet_rating_result.alfa_o
        T_wall_outside_repr = (
            wet_rating_result.thermal_state.outside_wall_temperature
        )
        delta_T_film = T_mean_outside - T_wall_outside_repr
        if abs(delta_T_film) < 1e-3:
            alfa_o_effective = alfa_o_dry
            balance_warnings.append(
                make_warning(
                    code=WC.EFFECTIVE_OUTSIDE_ALPHA_LIMITED,
                    message=(
                        "outside: bulk-to-surface temperature difference is "
                        "too small to reconstruct alfa_outside_effective "
                        "from Q_total; falling back to alfa_outside_dry."
                    ),
                    source=SOURCE,
                    severity="info",
                )
            )
        else:
            alfa_o_effective = Q_required / (
                wet_rating_result.A_o * delta_T_film
            )

        # Preserve the legacy bare Rating wet-envelope calculation exactly.
        wet_area = (
            wet_rating_result.A_o * wet_surface.wet_surface_fraction
        )
        wet_surface_fraction = wet_surface.wet_surface_fraction
        wet_surface_fraction_method = wet_surface.method
        if M_dry is None or M_h2o is None:
            raise ValueError(
                "Active outside condensation requires dry-carrier and water "
                "molar masses for wet-surface saturation diagnostics."
            )
        W_sat_wet_surface = saturated_water_ratio(
            p_total=outside.p,
            T=wet_wall_temperature,
            M_dry=M_dry,
            M_h2o=M_h2o,
        )

    mass_balance_error = m_dot_water_vapor_in - (m_dot_water_vapor_out + m_dot_condensate)
    energy_balance_error = Q_required - (Q_sensible + Q_latent)

    if is_wet_finned:
        assert wet_finned_surface is not None
        if (
            wet_finned_surface.condensation_area_fraction < 1.0
            or wet_finned_surface.condensation_temperature_offset_K < 0.0
        ):
            balance_warnings.append(
                make_warning(
                    code=WC.WET_SURFACE_FRACTION_0D_ESTIMATE,
                    message=(
                        "outside: the bulk-mean radial circular-fin Rating "
                        "response was dry after endpoint onset, so "
                        "condensation uses the dry exposed-skin envelope's "
                        "linear 0D cold-zone area and representative "
                        "temperature. This is not a longitudinal, row, "
                        "circuit or pass-resolved model."
                    ),
                    source=SOURCE,
                    severity="info",
                )
            )
        wet_dp_warning = make_warning(
            code=WC.CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY,
            message=(
                "outside: wet circular-finned thermal performance and "
                "condensate are solved, but no wet-surface pressure-drop "
                "correction is available; the reported finned-bank pressure "
                "drop is the selected dry-correlation reference only."
            ),
            source=SOURCE,
            severity="warning",
        )
        balance_warnings.append(wet_dp_warning)
        balance_warnings = _deduplicate_rating_warnings(balance_warnings)

        A_o = wet_rating_result.A_o
        R_outside_effective = 1.0 / (alfa_o_effective * A_o)
        R_inside = 1.0 / (
            wet_rating_result.thermal_state.alfa_i
            * wet_rating_result.final_result.A_i
        )
        R_total_effective = (
            R_inside
            + hx.tube_wall_resistance()
            + R_outside_effective
        )
        UA_effective = 1.0 / R_total_effective
        U_effective = UA_effective / A_o
        A_required_effective = wet_rating_result.UA_required / U_effective
        surface_margin_factor = calculate_surface_margin_factor(
            UA_actual=UA_effective,
            UA_process=wet_rating_result.UA_process,
        )

        thermal_finned = (
            wet_rating_result.thermal_state.finned_tube_diagnostics
        )
        final_finned = wet_rating_result.final_result.finned_tube_diagnostics
        if thermal_finned is None or final_finned is None:
            raise ValueError(
                "Active wet circular-fin Rating lost its finned diagnostics."
            )

        def attach_wet_surface(diagnostics):
            return replace(
                diagnostics,
                # Preserve all established dry-network thermal fields.  The
                # wet effective coefficient and its basis live only on the
                # explicitly named nested wet result.
                wet_surface=wet_finned_surface,
                pressure_drop_coefficient=(
                    outside_bank_hydraulic.midpoint.coefficient
                ),
                pressure_drop_coefficient_definition=(
                    outside_bank_hydraulic.coefficient_definition
                ),
                outside_dp_drag=outside_bank_hydraulic.dp_drag,
                outside_dp_acceleration=outside_bank_hydraulic.dp_acceleration,
                outside_dp_total=outside_bank_hydraulic.dp_total,
                pressure_drop_metadata=outside_bank_hydraulic.metadata,
                outside_dp_dry_reference=outside_bank_hydraulic.dp_total,
                wet_pressure_drop_supported=False,
                outside_dp_reference_only=True,
                warnings=tuple(
                    _deduplicate_rating_warnings(
                        [
                            *diagnostics.warnings,
                            *outside_bank_hydraulic.warnings,
                            wet_dp_warning,
                        ]
                    )
                ),
            )

        thermal_finned = attach_wet_surface(thermal_finned)
        final_finned = attach_wet_surface(final_finned)
        wet_thermal_state = replace(
            wet_rating_result.thermal_state,
            inside_wall_temperature=(
                wet_finned_surface.inside_wall_temperature
            ),
            outside_wall_temperature=(
                wet_finned_surface.core_wall_temperature
            ),
            alfa_o=alfa_o_effective,
            U=U_effective,
            UA=UA_effective,
            iterations=outer_iteration,
            converged=rating_coupled_converged,
            residual=outer_residuals["T_wall_wet_mean_K"],
            outside_alpha_physical=alfa_o_dry,
            outside_alpha_effective_gross=(
                thermal_finned.outside_alpha_effective_gross
            ),
            outside_alpha_wet_effective_gross_core_basis=(
                wet_finned_surface.outside_alpha_wet_effective_gross_core_basis
            ),
            outside_alpha_wet_effective_basis=(
                wet_finned_surface.outside_alpha_wet_effective_basis
            ),
            warnings=tuple(
                _deduplicate_rating_warnings(
                    [
                        *wet_rating_result.thermal_state.warnings,
                        wet_dp_warning,
                    ]
                )
            ),
            finned_tube_diagnostics=thermal_finned,
        )
        wet_final_result = replace(
            wet_rating_result.final_result,
            finned_tube_diagnostics=final_finned,
            warnings=_deduplicate_rating_warnings(
                [
                    *(wet_rating_result.final_result.warnings or []),
                    wet_dp_warning,
                ]
            ),
        )
        wet_rating_result = replace(
            wet_rating_result,
            overdesign_factor=surface_margin_factor,
            ua_margin=surface_margin_factor,
            A_required=A_required_effective,
            UA_actual=UA_effective,
            U_mean=U_effective,
            alfa_o=alfa_o_effective,
            final_result=wet_final_result,
            thermal_state=wet_thermal_state,
            warnings=_deduplicate_rating_warnings(
                [*(wet_rating_result.warnings or []), *balance_warnings]
            ),
        )

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
        converged=rating_coupled_converged,
        iterations=outer_iteration,
        method=(
            (
                "outside_condensation_rating_wet_annular_fin_fvm_"
                "with_endpoint_wet_zone_fallback"
                if (
                    wet_finned_surface is not None
                    and (
                        wet_finned_surface.condensation_area_fraction < 1.0
                        or wet_finned_surface
                        .condensation_temperature_offset_K < 0.0
                    )
                )
                else "outside_condensation_rating_wet_annular_fin_fvm"
            )
            if is_wet_finned
            else "outside_condensation_rating_wet_wall_fixed_point"
        ),
        W_in=W_in,
        W_mid=W_mean,
        W_out=W_out,
        m_dot_dry_carrier=m_dot_dry_carrier,
        m_dot_water_vapor_in=m_dot_water_vapor_in,
        m_dot_water_vapor_out=m_dot_water_vapor_out,
        m_dot_condensate=m_dot_condensate,
        m_dot_gas_in=outside.m_dot,
        m_dot_gas_out=m_dot_gas_out,
        dew_point_in=outside_dew_point,
        dew_point_out=dew_point_at_ratio(outside_capability, W_out, p=outside.p),
        wall_temperature_mean=(
            wet_finned_surface.outside_surface_temperature_area_mean
            if wet_finned_surface is not None
            else T_wall_outside_repr
        ),
        wall_temperature_min=rating_wall_min,
        wall_temperature_max=rating_wall_max,
        wall_temperature_wet_mean=wet_wall_temperature,
        wet_surface_fraction=wet_surface_fraction,
        wet_surface_fraction_method=wet_surface_fraction_method,
        wet_area=wet_area,
        outside_total_area=wet_rating_result.A_o,
        W_sat_wet_surface=W_sat_wet_surface,
        wet_finned_surface=wet_finned_surface,
        alfa_dry=alfa_o_dry,
        alfa_effective=alfa_o_effective,
        lewis_number=settings.lewis_number,
        Q_sensible=Q_sensible,
        Q_latent=Q_latent,
        Q_total=Q_required,
        mass_balance_error=mass_balance_error,
        energy_balance_error=energy_balance_error,
        residuals=outer_residuals,
        assumptions=tuple(
            dict.fromkeys(
                (
                    "rating_enthalpy_root_not_transport_limited",
                    "condensate_temperature_equals_representative_wet_wall_temperature",
                    *(
                        wet_finned_surface.assumptions
                        if wet_finned_surface is not None
                        else (
                            "wet_surface_bounds_centered_on_global_wall_mean",
                            "fully_drained_saturated_liquid_condensate_at_wet_wall_temperature",
                            "dry_gas_composition_unchanged_by_condensation",
                        )
                    ),
                )
            )
        ),
        warnings=tuple(balance_warnings),
    )

    phase_aware_simulation = None
    Q_achievable = None
    if include_simulation:
        from core.models.simulation import HXSideInput

        phase_aware_simulation = hx.simulate(
            HXSideInput(
                provider=inside.provider,
                m_dot=closed_balance.inside.m_dot,
                T_in=closed_balance.inside.T_in,
                p=inside.p,
                phase_change_mode=inside.phase_change_mode,
            ),
            HXSideInput(
                provider=outside.provider,
                m_dot=closed_balance.outside.m_dot,
                T_in=closed_balance.outside.T_in,
                p=outside.p,
                phase_change_mode=outside.phase_change_mode,
            ),
            flow_arrangement=flow_arrangement,
            K_inlet=K_inlet,
            K_outlet=K_outlet,
            K_turn=K_turn,
            euler_provider=euler_provider,
            finned_heat_transfer_provider=finned_heat_transfer_provider,
            finned_pressure_drop_provider=finned_pressure_drop_provider,
            max_iter=max_iterations,
            temperature_tolerance_K=wall_temperature_tolerance_K,
            relative_alfa_tolerance=relative_alfa_tolerance,
            relaxation_factor=relaxation_factor,
            phase_change_onset_tolerance_K=settings.onset_tolerance_K,
            phase_change_activation_band_K=settings.activation_band_K,
            lewis_number=settings.lewis_number,
            phase_change_max_iterations=settings.max_iterations,
            phase_change_temperature_tolerance_K=(
                settings.temperature_tolerance_K
            ),
            phase_change_relative_Q_tolerance=(
                settings.relative_Q_tolerance
            ),
            phase_change_water_ratio_tolerance=(
                settings.water_ratio_tolerance
            ),
            phase_change_condensate_tolerance_kg_s=(
                settings.condensate_tolerance_kg_s
            ),
            phase_change_wall_temperature_tolerance_K=(
                settings.wall_temperature_tolerance_K
            ),
            phase_change_wet_fraction_tolerance=(
                settings.wet_fraction_tolerance
            ),
            phase_change_relaxation_factor=settings.relaxation_factor,
        )
        Q_achievable = phase_aware_simulation.q

    return replace(
        wet_rating_result,
        inside_phase_change=inside_result,
        outside_phase_change=outside_result,
        simulation=phase_aware_simulation,
        Q_achievable=Q_achievable,
    )


def _apply_inside_condensation_to_rating(
    hx,
    *,
    inside: BalanceSideSpec,
    outside: BalanceSideSpec,
    closed_balance: ClosedBalance,
    dry_result,
    Q: float | None,
    flow_arrangement: str | None,
    K_inlet: float,
    K_outlet: float,
    K_turn: float,
    euler_provider: str,
    finned_heat_transfer_provider: object,
    finned_pressure_drop_provider: object,
    include_simulation: bool,
    max_iterations: int,
    wall_temperature_tolerance_K: float,
    relative_alfa_tolerance: float,
    relaxation_factor: float,
    settings: PhaseChangeSettings,
    inside_capability: PhaseChangeCapability,
    outside_capability: PhaseChangeCapability,
    inside_onset,
    inside_dew_point: float,
    inside_wall_min: float,
    inside_wall_mean: float,
    inside_wall_max: float,
    outside_possible: bool,
    outside_near_onset: bool,
    outside_onset,
    outside_dew_point: float | None,
    outside_wall_min: float | None,
    outside_wall_mean: float | None,
    outside_wall_max: float | None,
):
    """Rating integration for active wet-gas condensation inside tubes."""
    if inside.m_dot is None or inside.T_out is None:
        raise ValueError(
            "Rating with active inside condensation requires explicit "
            "inside.m_dot and inside.T_out."
        )
    if Q is None and (outside.m_dot is None or outside.T_out is None):
        raise ValueError(
            "Rating with active inside condensation requires duty from an "
            "explicit Q or a fully specified non-condensing outside side."
        )

    if Q is not None:
        Q_required = Q
    else:
        T_mean_outside = 0.5 * (outside.T_in + outside.T_out)
        cp_outside = outside.provider.at(T=T_mean_outside, p=outside.p).cp
        Q_required = outside.m_dot * cp_outside * abs(outside.T_out - outside.T_in)

    W_in = inside_capability.W_in
    if W_in is None:
        raise ValueError("Active inside condensation requires an inlet water ratio.")
    enthalpy_evaluator = WetGasEnthalpyEvaluator(inside.p, inside_capability)
    m_dot_dry_carrier = inside.m_dot / (1.0 + W_in)
    dT_inside = inside.T_in - inside.T_out
    if dT_inside <= 0.0:
        raise ValueError("Active inside condensation requires inside T_out < T_in.")
    C_effective_inside = Q_required / dT_inside
    C_min = min(C_effective_inside, closed_balance.outside.C)
    Q_max = C_min * abs(inside.T_in - outside.T_in)
    effectiveness = Q_required / Q_max if Q_max > 0.0 else math.nan

    warnings: list[ModelWarning] = [
        make_warning(
            code=WC.EFFECTIVE_CAPACITY_RATE_0D_APPROXIMATION,
            message=(
                "inside: epsilon-NTU uses an effective wet-gas capacity rate "
                "derived after closing the wet enthalpy balance."
            ),
            source=SOURCE,
            severity="info",
        )
    ]

    def run_wet_rating(W_out_for_rating: float):
        W_mean_for_rating = 0.5 * (W_in + W_out_for_rating)
        provider_mean = wet_gas_provider_at_water_ratio(
            inside_capability,
            W_mean_for_rating,
        )
        m_dot_mean = m_dot_dry_carrier * (1.0 + W_mean_for_rating)
        inside_closed = ClosedBalanceSide(
            provider=provider_mean,
            p=inside.p,
            m_dot=m_dot_mean,
            T_in=inside.T_in,
            T_out=inside.T_out,
            cp_mean=C_effective_inside / m_dot_mean,
            C=C_effective_inside,
        )
        wet_balance = ClosedBalance(
            inside=inside_closed,
            outside=closed_balance.outside,
            hot_is_inside=closed_balance.hot_is_inside,
            Q=Q_required,
            Q_max=Q_max,
            effectiveness=effectiveness,
            warnings=warnings,
        )
        return run_rating(
            hx,
            wet_balance,
            flow_arrangement=flow_arrangement,
            K_inlet=K_inlet,
            K_outlet=K_outlet,
            K_turn=K_turn,
            euler_provider=euler_provider,
            finned_heat_transfer_provider=finned_heat_transfer_provider,
            finned_pressure_drop_provider=finned_pressure_drop_provider,
            include_simulation=include_simulation,
            max_iterations=max_iterations,
            wall_temperature_tolerance_K=wall_temperature_tolerance_K,
            relative_alfa_tolerance=relative_alfa_tolerance,
            relaxation_factor=relaxation_factor,
        )

    condensate_temperature = inside.T_out
    previous_W_out: float | None = None
    wet_rating_result = None
    wet_surface = None
    wall_min = wall_mean = wall_max = wet_wall_temperature = None
    dew_point_mean = None
    W_out = W_in
    root_residual = math.inf
    enthalpy_residual = math.inf
    residuals: dict[str, float] = {}
    outer_converged = False
    outer_limit = min(settings.max_iterations, _RATING_OUTER_MAX_ITERATIONS)

    for outer_iteration in range(1, outer_limit + 1):
        W_out, root_residual = _solve_rating_water_ratio_for_side(
            wet_side=inside,
            capability=inside_capability,
            Q_required=Q_required,
            m_dot_dry_carrier=m_dot_dry_carrier,
            condensate_temperature=condensate_temperature,
            enthalpy_evaluator=enthalpy_evaluator,
        )
        wet_rating_result = run_wet_rating(W_out)
        W_mean = 0.5 * (W_in + W_out)
        dew_point_mean = dew_point_at_ratio(
            inside_capability,
            W_mean,
            p=inside.p,
        )
        if dew_point_mean is None:
            from core.properties.water import WATER_TRIPLE_POINT_TEMPERATURE_K

            dew_point_mean = WATER_TRIPLE_POINT_TEMPERATURE_K
        wall_min, wall_mean, wall_max = _centered_wall_bounds(
            wet_rating_result,
            side="inside",
        )
        wet_surface = estimate_wet_surface_fraction(
            dew_point_temperature=dew_point_mean,
            wall_temperature_min=wall_min,
            wall_temperature_mean=wall_mean,
            wall_temperature_max=wall_max,
            activation_band_K=settings.activation_band_K,
        )
        wet_wall_temperature = wet_surface.wall_temperature_wet_mean
        if wet_surface.wet_surface_fraction <= 0.0 or wet_wall_temperature is None:
            raise ValueError(
                "Rating's locked active inside-condensation regime produced no wet area."
            )

        W_at_wet_wall, _ = _solve_rating_water_ratio_for_side(
            wet_side=inside,
            capability=inside_capability,
            Q_required=Q_required,
            m_dot_dry_carrier=m_dot_dry_carrier,
            condensate_temperature=wet_wall_temperature,
            enthalpy_evaluator=enthalpy_evaluator,
        )
        successive = (
            abs(W_out - previous_W_out)
            if previous_W_out is not None
            else math.inf
        )
        fixed_point = abs(W_at_wet_wall - W_out)
        wet_wall_residual = abs(wet_wall_temperature - condensate_temperature)
        enthalpy_residual = _rating_enthalpy_residual_at_state(
            wet_side=inside,
            capability=inside_capability,
            Q_required=Q_required,
            m_dot_dry_carrier=m_dot_dry_carrier,
            W_out=W_out,
            condensate_temperature=wet_wall_temperature,
            enthalpy_evaluator=enthalpy_evaluator,
        )
        failed_probes = sum(
            not probe.converged
            for probe in wet_rating_result.wall_temperature_envelope.probes
        )
        residuals = {
            "W_out": max(successive, fixed_point),
            "W_out_successive": successive,
            "W_out_fixed_point": fixed_point,
            "T_wall_wet_mean_K": wet_wall_residual,
            "wall_envelope_failed_probes": float(failed_probes),
            "inside_enthalpy_balance_W": abs(enthalpy_residual),
            "inside_enthalpy_root_W": abs(root_residual),
        }
        if (
            outer_iteration >= 2
            and successive < settings.water_ratio_tolerance
            and fixed_point < settings.water_ratio_tolerance
            and wet_wall_residual < settings.wall_temperature_tolerance_K
            and abs(enthalpy_residual) < _RATING_ENTHALPY_SYNCHRONIZATION_TOLERANCE_W
            and wet_rating_result.thermal_state.converged
            and failed_probes == 0
        ):
            outer_converged = True
            break
        previous_W_out = W_out
        condensate_temperature = wet_wall_temperature

    if any(
        value is None
        for value in (
            wet_rating_result,
            wet_surface,
            wall_min,
            wall_mean,
            wall_max,
            wet_wall_temperature,
            dew_point_mean,
        )
    ):
        raise ValueError("Inside-condensation Rating failed to produce a complete state.")
    if not outer_converged:
        warnings.append(
            make_warning(
                code=WC.INSIDE_CONDENSATION_NOT_CONVERGED,
                message=(
                    "inside Rating wet-wall/enthalpy fixed point did not "
                    f"converge within {outer_limit} iterations."
                ),
                source=SOURCE,
                severity="warning",
            )
        )

    W_mid = 0.5 * (W_in + W_out)
    m_dot_condensate = m_dot_dry_carrier * (W_in - W_out)
    m_dot_water_vapor_in = m_dot_dry_carrier * W_in
    m_dot_water_vapor_out = m_dot_dry_carrier * W_out
    m_dot_gas_mid = m_dot_dry_carrier * (1.0 + W_mid)
    m_dot_gas_out = m_dot_dry_carrier + m_dot_water_vapor_out
    provider_in = wet_gas_provider_at_water_ratio(inside_capability, W_in)
    provider_mid = wet_gas_provider_at_water_ratio(inside_capability, W_mid)
    provider_out = wet_gas_provider_at_water_ratio(inside_capability, W_out)
    T_mean_inside = 0.5 * (inside.T_in + inside.T_out)

    from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
    from core.pressure_drop.flow_path import build_tube_side_pressure_drop_result

    tube_hydraulic = calculate_tube_bundle_hydraulics(
        m_dot=m_dot_gas_mid,
        flow_area_per_pass=hx.bundle.internal_flow_area_per_pass,
        hydraulic_diameter=hx.bundle.internal_hydraulic_diameter,
        hydraulic_length_total=hx.bundle.internal_length_total,
        n_tube_passes=hx.bundle.n_passes_tube,
        tube_path_type=hx.bundle.tube_path_type,
        roughness_inner=getattr(hx.bundle.tube, "roughness_inner", None),
        inlet_props=provider_in.at(T=inside.T_in, p=inside.p),
        midpoint_props=provider_mid.at(T=T_mean_inside, p=inside.p),
        outlet_props=provider_out.at(T=inside.T_out, p=inside.p),
        temperature_in=inside.T_in,
        temperature_out=inside.T_out,
        pressure=inside.p,
        m_dot_inlet=inside.m_dot,
        m_dot_midpoint=m_dot_gas_mid,
        m_dot_outlet=m_dot_gas_out,
    )
    tube_hydraulic = replace(
        tube_hydraulic,
        midpoint_method="arithmetic_temperature_and_water_ratio",
    )
    wet_final_result = replace(
        wet_rating_result.final_result,
        tube_side_hydraulic=replace(
            wet_rating_result.final_result.tube_side_hydraulic,
            tube_bundle=tube_hydraulic,
        ),
        tube_side_pressure_drop=replace(
            wet_rating_result.final_result.tube_side_pressure_drop,
            tube_bundle=tube_hydraulic,
            flow_path=build_tube_side_pressure_drop_result(
                tube_hydraulic,
                n_tube_passes=hx.bundle.n_passes_tube,
            ),
        ),
    )

    Q_latent = m_dot_condensate * water_latent_heat_of_vaporization(
        T=wet_wall_temperature
    )
    Q_sensible = Q_required - Q_latent
    alfa_i_dry = wet_rating_result.alfa_i
    delta_T_film = T_mean_inside - wall_mean
    if abs(delta_T_film) < 1e-3:
        alfa_i_effective = alfa_i_dry
        warnings.append(
            make_warning(
                code=WC.EFFECTIVE_INSIDE_ALPHA_LIMITED,
                message="inside Rating used alfa_inside_dry because the bulk-wall delta T is too small.",
                source=SOURCE,
                severity="info",
            )
        )
    else:
        alfa_i_effective = Q_required / (wet_rating_result.final_result.A_i * delta_T_film)
    R_i = 1.0 / (alfa_i_effective * wet_rating_result.final_result.A_i)
    R_o = 1.0 / (wet_rating_result.alfa_o * wet_rating_result.A_o)
    UA_effective = 1.0 / (R_i + hx.tube_wall_resistance() + R_o)
    U_effective = UA_effective / wet_rating_result.A_o
    wet_thermal_state = replace(
        wet_rating_result.thermal_state,
        alfa_i=alfa_i_effective,
        U=U_effective,
        UA=UA_effective,
    )
    A_required = wet_rating_result.UA_required / U_effective
    surface_margin_factor = calculate_surface_margin_factor(
        UA_actual=UA_effective,
        UA_process=wet_rating_result.UA_process,
    )

    wet_area = wet_rating_result.final_result.A_i * wet_surface.wet_surface_fraction
    W_sat_surface = saturated_water_ratio(
        p_total=inside.p,
        T=wet_wall_temperature,
        M_dry=inside_capability.M_dry,
        M_h2o=inside_capability.M_condensable,
    )
    mass_balance_error = m_dot_water_vapor_in - (
        m_dot_water_vapor_out + m_dot_condensate
    )
    energy_balance_error = enthalpy_residual
    for code, message in (
        (WC.INSIDE_CONDENSATION_DETECTED, "inside: Rating solved partial H2O condensation."),
        (WC.WET_SURFACE_FRACTION_0D_ESTIMATE, "inside: Rating wet area is a 0D wall-envelope estimate."),
        (WC.CONDENSATE_FILM_HYDRAULICS_NOT_MODELLED, "inside: Rating tube pressure drop is gas-phase only; condensate-film hydraulics are not modelled."),
        (WC.FULLY_DRAINED_CONDENSATE_ASSUMED, "inside: Rating assumes fully drained condensate without re-entrainment."),
    ):
        warnings.append(
            make_warning(code=code, message=message, source=SOURCE, severity="info")
        )

    outside_result = build_capability_side_result(
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
        Q_sensible_actual=Q_required,
    )
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
        converged=(
            outer_converged
            and wet_rating_result.thermal_state.converged
            and all(p.converged for p in wet_rating_result.wall_temperature_envelope.probes)
        ),
        iterations=outer_iteration,
        method="inside_condensation_rating_wet_wall_fixed_point",
        W_in=W_in,
        W_mid=W_mid,
        W_out=W_out,
        m_dot_dry_carrier=m_dot_dry_carrier,
        m_dot_water_vapor_in=m_dot_water_vapor_in,
        m_dot_water_vapor_out=m_dot_water_vapor_out,
        m_dot_condensate=m_dot_condensate,
        m_dot_gas_in=inside.m_dot,
        m_dot_gas_out=m_dot_gas_out,
        dew_point_in=inside_dew_point,
        dew_point_out=dew_point_at_ratio(inside_capability, W_out, p=inside.p),
        wall_temperature_mean=wall_mean,
        wall_temperature_min=wall_min,
        wall_temperature_max=wall_max,
        wall_temperature_wet_mean=wet_wall_temperature,
        wet_surface_fraction=wet_surface.wet_surface_fraction,
        wet_surface_fraction_method=wet_surface.method,
        wet_area=wet_area,
        inside_total_area=wet_rating_result.final_result.A_i,
        W_sat_wet_surface=W_sat_surface,
        alfa_dry=alfa_i_dry,
        alfa_effective=alfa_i_effective,
        lewis_number=settings.lewis_number,
        Q_sensible=Q_sensible,
        Q_latent=Q_latent,
        Q_total=Q_required,
        mass_balance_error=mass_balance_error,
        energy_balance_error=energy_balance_error,
        residuals=residuals,
        assumptions=(
            "rating_enthalpy_root_not_transport_limited",
            "wet_surface_bounds_centered_on_global_wall_mean",
            "fully_drained_saturated_liquid_condensate_at_wet_wall_temperature",
            "gas_phase_only_tube_side_hydraulics",
        ),
        warnings=tuple(_deduplicate_rating_warnings(warnings)),
    )
    return replace(
        wet_rating_result,
        overdesign_factor=surface_margin_factor,
        ua_margin=surface_margin_factor,
        A_required=A_required,
        UA_actual=UA_effective,
        U_mean=U_effective,
        alfa_i=alfa_i_effective,
        Q_required=Q_required,
        final_result=wet_final_result,
        thermal_state=wet_thermal_state,
        inside_phase_change=inside_result,
        outside_phase_change=outside_result,
    )


def _deduplicate_rating_warnings(warnings: list[ModelWarning]) -> list[ModelWarning]:
    unique: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        key = (warning.source, warning.code)
        if key not in seen:
            seen.add(key)
            unique.append(warning)
    return unique
