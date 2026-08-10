# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Coupled 0D outside partial-H2O-condensation solver.

Condensation onset is decided before this module is called, from the dry
baseline's minimum outside wall temperature.  Once active, the regime stays
locked while this module iterates outlet temperatures, water content and
the wall state.

Three wall-temperature meanings remain separate:

* ``T_wall_min`` is the cold-envelope value used by onset and wet-area
  geometry.
* ``T_wall_mean`` is the global whole-surface temperature solved by the
  sensible resistance balance.  Sensible duty uses the full ``A_o``.
* ``T_wall_wet_mean`` represents only the part with ``T_wall < T_dew``.
  Saturation, mass-transfer driving force, latent heat and drained-liquid
  enthalpy use this temperature and ``A_wet = A_o * wet_surface_fraction``.

Every outer iteration builds a cheap two-point sensible-only envelope.  Its
span is retained but both extrema are shifted together so the midpoint
tracks the current global wall mean.  The neutral linear helper in
``core.phase_change.wet_surface_fraction`` then returns the wet fraction and
representative wet-zone temperature from that same envelope.  This remains
a 0D estimate, not a spatial/row-by-row wall solution.

The global wall mean is found by bisection, not Newton iteration, from

    q_sensible(T_wall_mean)
    + q_latent(T_wall_wet_mean)
    - q_removed(T_wall_mean) = 0

The outer solve also closes water and wet-gas enthalpy balances, relaxes the
coupled iterates, checks finiteness, and requires both global and wet wall
temperatures to converge.  Frost/ice, film resistance/hydraulics and
re-entrainment remain outside the v0.6.0 scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.properties.common import FluidTransportProperties
from core.properties.fluids import PropertyProvider
from core.properties.averaging import mean_temperature
from core.properties.water import (
    WATER_TRIPLE_POINT_TEMPERATURE_K,
    water_latent_heat_of_vaporization,
    water_saturation_liquid_enthalpy,
)
from core.heat_transfer.thermal_iteration import ThermalIterationDiagnostics, _evaluate_local_wall_state
from core.common.warnings import ModelWarning, make_warning

from core.phase_change.types import PhaseChangeCapability
from core.phase_change.water_equilibrium import (
    is_frost_regime,
    saturated_water_ratio,
    water_dew_point,
    water_mole_fraction_from_ratio,
    water_partial_pressure,
)
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.wet_gas_enthalpy import (
    WetGasEnthalpyEvaluator,
    h_wet_gas_dry_basis,
    temperature_from_h_wet_gas_dry_basis,
)
from core.phase_change.mass_heat_transfer import condensation_rate
from core.phase_change.wet_surface_fraction import estimate_wet_surface_fraction
from core.phase_change import warning_codes as WC
from core.phase_change.condensation_solver_helpers import (
    condensate_enthalpy_flow,
    FrostingNotSupportedError,
    invert_wet_gas_enthalpy,
    local_dew_point_or_triple_point,
    sensible_only_wall_temperature,
    solve_condensing_interface_state,
)

if TYPE_CHECKING:
    from core.models.bare_tube import BareTubeHeatExchanger

SOURCE = "outside_condensation_solver"

# Fixed internal guard against dividing by a near-zero wall-temperature
# span in the per-iteration wet-surface-fraction estimate (spec section 9).
# Not user-exposed: it is a numerical-safety constant, not a physical
# tuning knob (unlike phase_change_wet_fraction_tolerance, the convergence
# tolerance, which is user-configurable).
WET_FRACTION_SPAN_TOLERANCE_K = 1e-3


@dataclass(frozen=True)
class OutsideCondensationSolution:
    """Converged (or last-iterate) coupled sensible+latent outside state."""

    converged: bool
    iterations: int
    residuals: dict[str, float]

    T_out_inside: float
    T_out_outside: float
    T_wall_inside: float
    T_wall_outside: float

    W_out: float
    m_dot_condensate: float

    Q_sensible: float
    Q_latent: float
    Q_total: float

    alfa_i: float
    alfa_o_dry: float
    alfa_o_effective: float
    U_effective: float
    UA_effective: float

    # Partial wet-area diagnostics (fix, v0.6.0 patch).
    wall_temperature_min: float
    wall_temperature_max: float
    wet_surface_fraction: float
    wet_surface_fraction_method: str
    wet_area: float
    outside_total_area: float

    outside_bulk_props: FluidTransportProperties
    inside_bulk_props: FluidTransportProperties
    inside_wall_props: FluidTransportProperties | None
    outside_wall_props: FluidTransportProperties | None
    diagnostics: ThermalIterationDiagnostics

    warnings: tuple[ModelWarning, ...] = ()
    # Appended diagnostics preserve the positional order of the original
    # solution fields.
    wall_temperature_wet_mean: float | None = None
    W_sat_wet_surface: float | None = None


def solve_outside_condensation(
    hx: "BareTubeHeatExchanger",
    *,
    inside_provider: PropertyProvider,
    m_dot_inside: float,
    T_in_inside: float,
    p_inside: float,
    outside_capability: PhaseChangeCapability,
    m_dot_dry_carrier: float,
    T_in_outside: float,
    p_outside: float,
    T_out_inside_init: float,
    T_out_outside_init: float,
    euler_provider: str = "zukauskas",
    lewis_number: float = 1.0,
    activation_band_K: float = 0.5,
    max_iterations: int = 50,
    temperature_tolerance_K: float = 0.05,
    relative_Q_tolerance: float = 1e-4,
    water_ratio_tolerance: float = 1e-6,
    condensate_tolerance_kg_s: float = 1e-8,
    wall_temperature_tolerance_K: float = 0.05,
    wet_fraction_tolerance: float = 1e-3,
    relaxation_factor: float = 0.5,
) -> OutsideCondensationSolution:
    """Iteratively solve the coupled outside condensation state.

    Initialized from the dry baseline (``T_out_inside_init``,
    ``T_out_outside_init``, ``W_out`` starting at ``W_in``,
    The first wet fraction comes from the dry sensible envelope rather than
    an arbitrary zero/full-area guess; subsequent updates retain the
    configured relaxation.
    """
    _validate_positive(lewis_number, "lewis_number")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0.")
    for name, value in (
        ("temperature_tolerance_K", temperature_tolerance_K),
        ("relative_Q_tolerance", relative_Q_tolerance),
        ("water_ratio_tolerance", water_ratio_tolerance),
        ("wall_temperature_tolerance_K", wall_temperature_tolerance_K),
        ("wet_fraction_tolerance", wet_fraction_tolerance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")
    if not math.isfinite(condensate_tolerance_kg_s) or condensate_tolerance_kg_s < 0.0:
        raise ValueError("condensate_tolerance_kg_s must be a non-negative finite value.")
    if not (0.0 < relaxation_factor <= 1.0):
        raise ValueError("relaxation_factor must be in (0, 1].")
    if not outside_capability.capable:
        raise ValueError("solve_outside_condensation requires a capable outside_capability.")

    bundle = hx.bundle
    A_i = bundle.total_inner_area
    A_o = bundle.total_outer_area
    R_w = hx.tube_wall_resistance()

    W_in = outside_capability.W_in
    assert W_in is not None

    T_out_inside = T_out_inside_init
    T_out_outside = T_out_outside_init
    W_out = W_in
    wet_surface_fraction = 1.0

    T_wall_inside_prev: float | None = None
    T_wall_outside_prev: float | None = None
    T_wall_wet_mean_prev: float | None = None

    Q_total_prev: float | None = None
    m_dot_condensate_prev: float | None = None

    all_warnings: list[ModelWarning] = []
    converged = False
    residuals: dict[str, float] = {}
    solution_state: dict[str, object] = {}
    enthalpy_evaluator = WetGasEnthalpyEvaluator(
        p_outside,
        outside_capability,
    )
    h_in_outside = enthalpy_evaluator.enthalpy(T_in_outside, W_in)

    for iteration in range(1, max_iterations + 1):
        T_mean_inside = mean_temperature(T_in_inside, T_out_inside)
        T_mean_outside = mean_temperature(T_in_outside, T_out_outside)
        W_mean = 0.5 * (W_in + W_out)

        outside_provider = wet_gas_provider_at_water_ratio(outside_capability, W_mean)
        m_dot_gas_mean = m_dot_dry_carrier * (1.0 + W_mean)

        local = _evaluate_local_wall_state(
            hx,
            m_dot_inside=m_dot_inside,
            m_dot_outside=m_dot_gas_mean,
            inside_provider=inside_provider,
            outside_provider=outside_provider,
            inside_bulk_temperature=T_mean_inside,
            outside_bulk_temperature=T_mean_outside,
            p_inside=p_inside,
            p_outside=p_outside,
            inside_wall_temperature=T_wall_inside_prev,
            outside_wall_temperature=T_wall_outside_prev,
            euler_provider=euler_provider,
        )
        all_warnings.extend(local.warnings)

        alfa_i = local.alfa_i
        alfa_o_dry = local.alfa_o
        cp_gas_bulk = local.outside_bulk_props.cp
        R_i = 1.0 / (alfa_i * A_i)
        R_downstream = R_i + R_w
        R_o_film = 1.0 / (alfa_o_dry * A_o)

        # --- Partial wet-area estimate for this iteration (fix) ---------
        T_wall_inlet_est = sensible_only_wall_temperature(
            T_bulk_wet_gas=T_in_outside,
            T_bulk_other=T_mean_inside,
            R_wet_film=R_o_film,
            R_downstream=R_downstream,
        )
        T_wall_outlet_est = sensible_only_wall_temperature(
            T_bulk_wet_gas=T_out_outside,
            T_bulk_other=T_mean_inside,
            R_wet_film=R_o_film,
            R_downstream=R_downstream,
        )
        T_wall_min_raw = min(T_wall_inlet_est, T_wall_outlet_est)
        T_wall_max_raw = max(T_wall_inlet_est, T_wall_outlet_est)
        T_shape_mean = 0.5 * (T_wall_min_raw + T_wall_max_raw)

        # Preserve the inexpensive two-point envelope's span, but centre it
        # on the current global wall state.  The first iteration has no
        # coupled wall iterate yet and therefore retains the sensible-only
        # midpoint; later iterations use the previous relaxed global wall
        # temperature.  This keeps Tmin/Tmean/Tmax on one consistent state
        # without introducing a spatial wall model.
        T_wall_mean_reference = (
            T_wall_outside_prev
            if T_wall_outside_prev is not None
            else T_shape_mean
        )
        wall_envelope_shift = T_wall_mean_reference - T_shape_mean
        T_wall_min_iter = T_wall_min_raw + wall_envelope_shift
        T_wall_max_iter = T_wall_max_raw + wall_envelope_shift

        dew_point_local = local_dew_point_or_triple_point(
            outside_capability, W_mean, p_wet_gas=p_outside,
        )

        wet_fraction_estimate = estimate_wet_surface_fraction(
            dew_point_temperature=dew_point_local,
            wall_temperature_min=T_wall_min_iter,
            wall_temperature_mean=T_wall_mean_reference,
            wall_temperature_max=T_wall_max_iter,
            temperature_span_tolerance_K=WET_FRACTION_SPAN_TOLERANCE_K,
            activation_band_K=activation_band_K,
        )
        wet_surface_fraction_calculated = wet_fraction_estimate.wet_surface_fraction
        if iteration == 1:
            wet_surface_fraction_new = wet_surface_fraction_calculated
        else:
            wet_surface_fraction_new = wet_surface_fraction + relaxation_factor * (
                wet_surface_fraction_calculated - wet_surface_fraction
            )
        # Numerical safety clamp only -- estimate_wet_surface_fraction
        # already returns a value in [0, 1]; relaxing between two such
        # values cannot leave the interval, this only guards float noise.
        wet_surface_fraction_new = max(0.0, min(1.0, wet_surface_fraction_new))

        T_wall_wet_mean_calculated = (
            wet_fraction_estimate.wall_temperature_wet_mean
        )
        # Use the helper's temperature directly.  It already changes
        # smoothly with the shifted linear envelope, while additionally
        # relaxing it would add a second lag to the tightly bounded outer
        # solve.  ``None`` remains a state marker and is never relaxed.
        T_wall_wet_mean_new = T_wall_wet_mean_calculated
        if T_wall_wet_mean_new is None:
            # A dry target has no temperature at which wet equilibrium can
            # be evaluated.  Keep area and temperature as one consistent
            # discrete state instead of retaining a relaxed positive area
            # paired with ``None``.
            wet_surface_fraction_new = 0.0

        A_wet = A_o * wet_surface_fraction_new
        if not (0.0 <= A_wet <= A_o):
            raise ValueError(
                f"outside_condensation_solver: A_wet={A_wet:.6g} m2 outside "
                f"the valid range [0, {A_o:.6g}] m2."
            )

        if (
            A_wet > 0.0
            and T_wall_wet_mean_new is not None
            and T_wall_wet_mean_new <= WATER_TRIPLE_POINT_TEMPERATURE_K
        ):
            raise FrostingNotSupportedError(
                "outside_condensation_solver: representative wet-surface "
                f"temperature {T_wall_wet_mean_new:.3f} K is at/below the "
                f"water triple point ({WATER_TRIPLE_POINT_TEMPERATURE_K} K); "
                "frost/ice condensate is not supported in v0.6.0."
            )

        (
            T_wall_mean,
            Q_sensible,
            Q_latent,
            m_dot_condensate,
            W_sat_wet_surface,
        ) = solve_condensing_interface_state(
            alfa_dry=alfa_o_dry,
            A_total=A_o,
            A_wet=A_wet,
            T_wall_wet_mean=T_wall_wet_mean_new,
            T_bulk_wet_gas=T_mean_outside,
            T_bulk_other=T_mean_inside,
            R_downstream=R_downstream,
            cp_gas=cp_gas_bulk,
            W_bulk=W_mean,
            p_wet_gas=p_outside,
            M_dry=outside_capability.M_dry,
            m_dot_water_vapor_available=m_dot_dry_carrier * W_mean,
            lewis_number=lewis_number,
        )

        if T_wall_mean <= WATER_TRIPLE_POINT_TEMPERATURE_K:
            raise FrostingNotSupportedError(
                f"outside_condensation_solver: global wall temperature "
                f"{T_wall_mean:.3f} K is at/below the water triple point "
                f"({WATER_TRIPLE_POINT_TEMPERATURE_K} K); frost/ice "
                "condensate is not supported in v0.6.0."
            )

        Q_total = Q_sensible + Q_latent
        W_out_new = max(0.0, W_in - m_dot_condensate / m_dot_dry_carrier)

        condensate_enthalpy_rate = condensate_enthalpy_flow(
            m_dot_condensate=m_dot_condensate,
            condensation_mass_tolerance=condensate_tolerance_kg_s,
            wet_surface_fraction=wet_surface_fraction_new,
            wet_area=A_wet,
            wall_temperature_wet_mean=T_wall_wet_mean_new,
        )
        h_drained_per_kg_dry = condensate_enthalpy_rate / m_dot_dry_carrier
        h_out_target = h_in_outside - Q_total / m_dot_dry_carrier - h_drained_per_kg_dry

        T_out_outside_new = invert_wet_gas_enthalpy(
            h_target=h_out_target,
            p_wet_gas=p_outside,
            W=W_out_new,
            capability=outside_capability,
            T_wall_mean=T_wall_mean,
            T_in_wet_gas=T_in_outside,
            evaluator=enthalpy_evaluator,
        )

        cp_inside_mean = inside_provider.at(T=T_mean_inside, p=p_inside).cp
        T_out_inside_new = T_in_inside + Q_total / (m_dot_inside * cp_inside_mean)

        T_out_inside_relaxed = T_out_inside + relaxation_factor * (T_out_inside_new - T_out_inside)
        T_out_outside_relaxed = T_out_outside + relaxation_factor * (T_out_outside_new - T_out_outside)
        W_out_relaxed = W_out + relaxation_factor * (W_out_new - W_out)
        T_wall_inside_new = T_mean_inside + Q_total * R_i

        finite_values = [
            Q_sensible,
            Q_latent,
            Q_total,
            T_out_inside_relaxed,
            T_out_outside_relaxed,
            W_out_relaxed,
            m_dot_condensate,
            T_wall_mean,
            T_wall_inside_new,
            T_wall_min_iter,
            T_wall_max_iter,
            wet_surface_fraction_new,
            A_wet,
        ]
        if T_wall_wet_mean_new is not None:
            finite_values.append(T_wall_wet_mean_new)
        if W_sat_wet_surface is not None:
            finite_values.append(W_sat_wet_surface)
        finite_ok = all(math.isfinite(v) for v in finite_values)
        if not finite_ok:
            raise ValueError(
                "outside_condensation_solver: non-finite iterate at "
                f"iteration {iteration}."
            )

        if T_wall_wet_mean_new is None and T_wall_wet_mean_prev is None:
            wet_wall_temperature_residual = 0.0
        elif (
            T_wall_wet_mean_new is None
            or T_wall_wet_mean_prev is None
        ):
            wet_wall_temperature_residual = math.inf
        else:
            wet_wall_temperature_residual = abs(
                T_wall_wet_mean_new - T_wall_wet_mean_prev
            )

        wall_envelope_center_residual = abs(
            T_wall_mean - 0.5 * (T_wall_min_iter + T_wall_max_iter)
        )
        wall_envelope_contains_mean = (
            T_wall_min_iter <= T_wall_mean <= T_wall_max_iter
        )
        mass_balance_residual = abs(
            m_dot_dry_carrier * (W_in - W_out_relaxed)
            - m_dot_condensate
        )
        residuals = {
            "Q_rel": (
                abs(Q_total - Q_total_prev) / max(abs(Q_total_prev), 1e-9)
                if Q_total_prev is not None else math.inf
            ),
            "T_out_inside_K": abs(T_out_inside_relaxed - T_out_inside),
            "T_out_outside_K": abs(T_out_outside_relaxed - T_out_outside),
            "W_out": abs(W_out_relaxed - W_out),
            "mass_balance_kg_s": mass_balance_residual,
            "m_dot_condensate_kg_s": (
                abs(m_dot_condensate - m_dot_condensate_prev)
                if m_dot_condensate_prev is not None else math.inf
            ),
            "T_wall_outside_K": (
                abs(T_wall_mean - T_wall_outside_prev)
                if T_wall_outside_prev is not None else math.inf
            ),
            "T_wall_wet_mean_K": wet_wall_temperature_residual,
            "wall_envelope_center_K": wall_envelope_center_residual,
            # Raw fixed-point error against the iterate before relaxation,
            # plus the actual successive step.  The raw form remains
            # meaningful when relaxation_factor == 1.
            "wet_surface_fraction": abs(
                wet_surface_fraction_calculated - wet_surface_fraction
            ),
            "wet_surface_fraction_step": abs(
                wet_surface_fraction_new - wet_surface_fraction
            ),
        }

        solution_state = dict(
            T_out_inside=T_out_inside_relaxed,
            T_out_outside=T_out_outside_relaxed,
            T_wall_inside=T_wall_inside_new,
            T_wall_outside=T_wall_mean,
            W_out=W_out_relaxed,
            m_dot_condensate=m_dot_condensate,
            Q_sensible=Q_sensible,
            Q_latent=Q_latent,
            Q_total=Q_total,
            alfa_i=alfa_i,
            alfa_o_dry=alfa_o_dry,
            wall_temperature_min=T_wall_min_iter,
            wall_temperature_max=T_wall_max_iter,
            wall_temperature_wet_mean=T_wall_wet_mean_new,
            wet_surface_fraction=wet_surface_fraction_new,
            wet_surface_fraction_method=wet_fraction_estimate.method,
            wet_area=A_wet,
            W_sat_wet_surface=W_sat_wet_surface,
            outside_bulk_props=local.outside_bulk_props,
            inside_bulk_props=local.inside_bulk_props,
            inside_wall_props=local.inside_wall_props,
            outside_wall_props=local.outside_wall_props,
            internal_diagnostics=local.internal_diagnostics,
            outside_nusselt_base=local.outside_nusselt_base,
            outside_nusselt=local.outside_nusselt,
            outside_wall_property_correction=local.outside_wall_property_correction,
        )

        Q_total_prev = Q_total
        m_dot_condensate_prev = m_dot_condensate
        T_wall_outside_prev = T_wall_mean
        T_wall_wet_mean_prev = T_wall_wet_mean_new
        T_wall_inside_prev = T_wall_inside_new
        T_out_inside, T_out_outside, W_out = (
            T_out_inside_relaxed, T_out_outside_relaxed, W_out_relaxed,
        )
        wet_surface_fraction = wet_surface_fraction_new

        if (
            iteration >= 2
            and residuals["Q_rel"] < relative_Q_tolerance
            and residuals["T_out_inside_K"] < temperature_tolerance_K
            and residuals["T_out_outside_K"] < temperature_tolerance_K
            and residuals["W_out"] < water_ratio_tolerance
            and residuals["mass_balance_kg_s"] <= condensate_tolerance_kg_s
            and residuals["m_dot_condensate_kg_s"] < condensate_tolerance_kg_s
            and residuals["T_wall_outside_K"] < wall_temperature_tolerance_K
            and residuals["T_wall_wet_mean_K"] < wall_temperature_tolerance_K
            and residuals["wall_envelope_center_K"]
            < wall_temperature_tolerance_K
            and wall_envelope_contains_mean
            and residuals["wet_surface_fraction"] < wet_fraction_tolerance
        ):
            converged = True
            break

    if not solution_state:
        raise ValueError("outside_condensation_solver: failed to produce a complete iterate.")

    if not (
        solution_state["wall_temperature_min"]
        <= solution_state["T_wall_outside"]
        <= solution_state["wall_temperature_max"]
    ):
        raise ValueError(
            "outside_condensation_solver: final two-point wall-temperature "
            "envelope does not contain the solved global wall mean; the "
            "coupled state is not safe to report."
        )

    if not converged:
        all_warnings.append(
            make_warning(
                code=WC.OUTSIDE_CONDENSATION_NOT_CONVERGED,
                message=(
                    "outside_condensation_solver: did not converge within "
                    f"max_iterations={max_iterations}. Last residuals: "
                    f"{residuals}. Returning the last complete, finite "
                    "iterate."
                ),
                source=SOURCE,
                severity="warning",
            )
        )

    A_o_local = A_o
    T_bulk_outside_final = mean_temperature(T_in_outside, solution_state["T_out_outside"])
    delta_T_film = T_bulk_outside_final - solution_state["T_wall_outside"]
    alfa_floor = solution_state["alfa_o_dry"] * 1.0e-3
    min_delta_T = 1.0e-3
    if abs(delta_T_film) < min_delta_T:
        alfa_o_effective = max(solution_state["alfa_o_dry"], alfa_floor)
        all_warnings.append(
            make_warning(
                code=WC.EFFECTIVE_OUTSIDE_ALPHA_LIMITED,
                message=(
                    "outside_condensation_solver: bulk-to-surface temperature "
                    f"difference ({delta_T_film:.3e} K) is too small to "
                    "reconstruct alfa_outside_effective from Q_total; "
                    "falling back to alfa_outside_dry."
                ),
                source=SOURCE,
                severity="info",
            )
        )
    else:
        alfa_o_effective = solution_state["Q_total"] / (A_o_local * delta_T_film)

    R_i_final = 1.0 / (solution_state["alfa_i"] * A_i)
    R_o_effective = 1.0 / (alfa_o_effective * A_o_local)
    UA_effective = 1.0 / (R_i_final + R_w + R_o_effective)
    U_effective = UA_effective / A_o_local if A_o_local > 0.0 else math.nan

    internal_diag = solution_state["internal_diagnostics"]
    diagnostics = ThermalIterationDiagnostics(
        inside_Nu_base=internal_diag.Nu_base,
        inside_Nu_corrected=internal_diag.Nu_corrected,
        inside_length_correction=internal_diag.length_correction,
        inside_wall_temperature_correction=internal_diag.wall_temperature_correction,
        inside_combined_correction=internal_diag.combined_correction,
        inside_alfa_base=internal_diag.alfa_base,
        inside_alfa_corrected=internal_diag.alfa_corrected,
        outside_Nu_base=solution_state["outside_nusselt_base"],
        outside_Nu_corrected=solution_state["outside_nusselt"],
        outside_wall_property_correction=solution_state["outside_wall_property_correction"],
    )

    return OutsideCondensationSolution(
        converged=converged,
        iterations=iteration,
        residuals=residuals,
        T_out_inside=solution_state["T_out_inside"],
        T_out_outside=solution_state["T_out_outside"],
        T_wall_inside=solution_state["T_wall_inside"],
        T_wall_outside=solution_state["T_wall_outside"],
        W_out=solution_state["W_out"],
        m_dot_condensate=solution_state["m_dot_condensate"],
        Q_sensible=solution_state["Q_sensible"],
        Q_latent=solution_state["Q_latent"],
        Q_total=solution_state["Q_total"],
        alfa_i=solution_state["alfa_i"],
        alfa_o_dry=solution_state["alfa_o_dry"],
        alfa_o_effective=alfa_o_effective,
        U_effective=U_effective,
        UA_effective=UA_effective,
        wall_temperature_min=solution_state["wall_temperature_min"],
        wall_temperature_max=solution_state["wall_temperature_max"],
        wall_temperature_wet_mean=solution_state["wall_temperature_wet_mean"],
        wet_surface_fraction=solution_state["wet_surface_fraction"],
        wet_surface_fraction_method=solution_state["wet_surface_fraction_method"],
        wet_area=solution_state["wet_area"],
        outside_total_area=A_o,
        W_sat_wet_surface=solution_state["W_sat_wet_surface"],
        outside_bulk_props=solution_state["outside_bulk_props"],
        inside_bulk_props=solution_state["inside_bulk_props"],
        inside_wall_props=solution_state["inside_wall_props"],
        outside_wall_props=solution_state["outside_wall_props"],
        diagnostics=diagnostics,
        warnings=tuple(all_warnings),
    )


def _sensible_only_wall_temperature(
    *, T_bulk_outside: float, T_bulk_inside: float, R_o_film: float, R_downstream: float,
) -> float:
    """Closed-form sensible-only (latent-free) outside wall temperature.

    Used only for the cheap per-iteration two-point wet-surface-fraction
    estimate (see module docstring) -- not for the actual coupled global
    wall mean used in the mass/energy balance, which is solved separately by
    ``_solve_interface_state`` including the latent contribution.
    """
    q = (T_bulk_outside - T_bulk_inside) / (R_o_film + R_downstream)
    return T_bulk_outside - q * R_o_film


def _local_dew_point_or_triple_point(
    capability: PhaseChangeCapability, W: float, *, p_outside: float,
) -> float:
    """Return the dew point for the current bulk water content, or the
    water triple-point temperature if the equilibrium state would be
    frost/ice (a bounded floor for this auxiliary wet-fraction estimate;
    the actual frost safety net is the interface-temperature check in
    ``solve_outside_condensation``, which raises
    ``FrostingNotSupportedError``)."""
    if W <= 0.0:
        return WATER_TRIPLE_POINT_TEMPERATURE_K
    y = water_mole_fraction_from_ratio(W, M_dry=capability.M_dry, M_h2o=capability.M_condensable)
    p_h2o = water_partial_pressure(y, p_outside)
    if is_frost_regime(p_h2o):
        return WATER_TRIPLE_POINT_TEMPERATURE_K
    return water_dew_point(p_h2o)


def _solve_interface_state(
    *,
    alfa_o_dry: float,
    A_o: float,
    A_wet: float,
    T_wall_wet_mean: float | None,
    T_bulk_outside: float,
    T_bulk_inside: float,
    R_downstream: float,
    cp_gas: float,
    W_bulk: float,
    p_outside: float,
    M_dry: float,
    m_dot_water_vapor_available: float,
    lewis_number: float,
    tolerance_K: float = 1e-7,
    max_iterations: int = 100,
) -> tuple[float, float, float, float, float | None]:
    """Bisect for the global wall mean and return its coupled heat state.

    The return value is ``(T_wall_mean, Q_sensible, Q_latent,
    m_dot_condensate, W_sat_wet_surface)``.  The global wall mean remains
    the unknown in the surface resistance balance.  Wet-surface saturation,
    mass transfer and latent properties are instead evaluated once at
    ``T_wall_wet_mean`` and over ``A_wet``.  Sensible heat transfer always
    uses the full outside area ``A_o``.

    When ``A_wet <= 0`` or ``T_wall_wet_mean is None``, the wet equilibrium
    functions are not called and the mass/latent terms are exactly zero.
    """

    W_sat_wet_surface: float | None = None
    m_dot_cond = 0.0
    q_latent = 0.0
    if A_wet > 0.0 and T_wall_wet_mean is not None:
        if not math.isfinite(T_wall_wet_mean):
            raise ValueError("T_wall_wet_mean must be finite when supplied.")
        if T_wall_wet_mean <= WATER_TRIPLE_POINT_TEMPERATURE_K:
            raise FrostingNotSupportedError(
                "outside_condensation_solver: representative wet-surface "
                f"temperature {T_wall_wet_mean:.3f} K is at/below the water "
                f"triple point ({WATER_TRIPLE_POINT_TEMPERATURE_K} K)."
            )
        W_sat_wet_surface = saturated_water_ratio(
            p_total=p_outside,
            T=T_wall_wet_mean,
            M_dry=M_dry,
        )
        m_dot_cond = condensation_rate(
            alfa_dry=alfa_o_dry,
            cp_gas=cp_gas,
            W_bulk=W_bulk,
            W_sat_surface=W_sat_wet_surface,
            A_wet=A_wet,
            m_dot_water_vapor_available=m_dot_water_vapor_available,
            lewis_number=lewis_number,
        )
        if m_dot_cond > 0.0:
            q_latent = m_dot_cond * water_latent_heat_of_vaporization(
                T=T_wall_wet_mean
            )

    def evaluate(T_wall_mean: float) -> tuple[float, float]:
        q_sensible = alfa_o_dry * A_o * (
            T_bulk_outside - T_wall_mean
        )
        q_removed = (T_wall_mean - T_bulk_inside) / R_downstream
        return q_sensible + q_latent - q_removed, q_sensible

    T_lo, T_hi = sorted((T_bulk_inside, T_bulk_outside))
    f_lo, _ = evaluate(T_lo)
    f_hi, _ = evaluate(T_hi)

    # With latent heat evaluated at the colder wet zone, the global mean
    # root can in principle lie slightly above the outside bulk mean.  Keep
    # the robust interval method, expanding only its upper bound until the
    # sign change is explicit.
    bracket_step = max(T_hi - T_lo, 1.0)
    for _ in range(60):
        if f_hi <= 0.0:
            break
        T_hi += bracket_step
        bracket_step *= 2.0
        f_hi, _ = evaluate(T_hi)

    if f_lo < 0.0 or f_hi > 0.0:
        raise ValueError(
            "outside_condensation_solver: global wall-temperature balance is "
            f"not bracketed on [{T_lo:.6g}, {T_hi:.6g}] K "
            f"(f_lo={f_lo:.6g}, f_hi={f_hi:.6g}); outside must be the "
            "hotter (cooling) side for condensation to be physically "
            "possible."
        )

    for _ in range(max_iterations):
        T_mid = 0.5 * (T_lo + T_hi)
        f_mid, _ = evaluate(T_mid)
        if f_mid > 0.0:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if (T_hi - T_lo) < tolerance_K:
            break

    T_wall_mean = 0.5 * (T_lo + T_hi)
    _, q_sensible = evaluate(T_wall_mean)
    return (
        T_wall_mean,
        q_sensible,
        q_latent,
        m_dot_cond,
        W_sat_wet_surface,
    )


def _invert_outside_enthalpy(
    *,
    h_target: float,
    p_outside: float,
    W: float,
    capability: PhaseChangeCapability,
    T_wall_mean: float,
    T_in_outside: float,
) -> float:
    T_lo = max(
        WATER_TRIPLE_POINT_TEMPERATURE_K + 0.5,
        min(T_wall_mean, T_in_outside) - 10.0,
    )
    T_hi = T_in_outside + 5.0
    try:
        return temperature_from_h_wet_gas_dry_basis(
            h_target,
            p_outside,
            W,
            capability,
            T_bracket=(T_lo, T_hi),
            tolerance_K=1e-6,
        )
    except ValueError:
        return temperature_from_h_wet_gas_dry_basis(
            h_target, p_outside, W, capability,
            T_bracket=(WATER_TRIPLE_POINT_TEMPERATURE_K + 0.5, T_in_outside + 10.0),
            tolerance_K=1e-6,
        )


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
