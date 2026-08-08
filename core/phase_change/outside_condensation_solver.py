# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Outside partial H2O condensation solver (v0.6.0; onset/wet-area fix).

Solves the coupled sensible + latent duty of a wet-gas stream flowing over
the outside of a bare-tube bank while a single-phase (dry) stream is heated
inside the tubes, given that the dry sensible-only baseline already showed
the outside tube-wall's *minimum* estimated temperature running below the
local water dew point (see ``core.phase_change.regime.
evaluate_condensation_onset`` for how that "possible"/"active" determination
is made, once, before this solver is ever invoked).

This module is intentionally the *only* place that iterates a phase-change
case -- ``core.phase_change.integration`` decides *whether* to call it;
``core.models.simulation``/``core.models.rating`` never duplicate this
loop.

Algorithm, per outer iteration (state: T_out_inside, T_out_outside, W_out,
wet_surface_fraction)
---------------------------------------------------------------------------
1. Bulk-mean state: ``T_mean_* = mean(T_in_*, T_out_*)``,
   ``W_mean = mean(W_in, W_out)`` (0D bulk-mean composition, the same
   simplification the rest of KalKalori uses for bulk temperature).
2. Rebuild the outside wet-gas provider at ``W_mean``
   (``core.phase_change.wet_gas_composition``) and evaluate
   ``alfa_i``/``alfa_o_dry`` via the *existing* resistance-network
   evaluator (``core.heat_transfer.thermal_iteration._evaluate_local_wall_
   state`` -- the exact same building block the dry solver uses; no
   arbitrary alfa multiplier is introduced here).
3. **Wet-area fix**: estimate the wall-temperature extrema and the
   wet-surface fraction *for this iteration*, and only then solve the
   interface temperature -- see "Partial wet-area handling" below.
4. Solve for the interface/outer-wall temperature ``T_s`` (at the
   bulk-mean state) by bisection on

       f(T_s) = q_sensible(T_s) + q_latent(T_s) - q_removed(T_s) = 0

   where ``q_sensible = alfa_o_dry*A_o*(T_bulk_outside - T_s)`` (full
   outside area -- sensible heat transfer is not restricted to the wet
   fraction, spec section 12), ``q_latent = m_dot_condensation(T_s) *
   h_fg(T_s)`` (Chilton-Colburn mass transfer,
   ``core.phase_change.mass_heat_transfer``, using ``A_wet`` -- *not*
   ``A_o`` -- as the mass-transfer area, zero when ``W_mean <=
   W_sat(T_s)``), and ``q_removed = (T_s - T_bulk_inside) / (R_i +
   R_wall)`` is the heat conducted away through the tube wall and inside
   film. Bisection, not an unbounded Newton step, for the same reason as
   before: a robust sign-changing bracket always exists between
   ``T_bulk_inside`` and ``T_bulk_outside`` for a genuinely condensing
   case.
5. Mass balance: ``m_dot_condensate = m_dot_condensation(T_s)``,
   ``W_out_new = W_in - m_dot_condensate/m_dot_dry_carrier``.
6. Enthalpy balance (not ``m_dot*cp_mean*dT``):
   ``Q_total = q_sensible + q_latent`` from step 4; then
   ``H_out = H_in - Q_total/m_dot_dry_carrier - (m_dot_condensate/
   m_dot_dry_carrier)*h_f(T_s)`` is inverted for ``T_out_outside_new``
   (``core.phase_change.wet_gas_enthalpy``, bisection).
7. The inside stream is sensible-only by construction (v0.6.0 scope):
   ``T_out_inside_new = T_in_inside + Q_total/(m_dot_inside*cp_inside)``.
8. Relaxed update of (T_out_inside, T_out_outside, W_out,
   wet_surface_fraction); wall temperatures are carried to the next
   iteration's step 2 evaluation as the previous iterate (same lagged
   wall-correction idiom as ``solve_iterative_thermal_state``).
9. Convergence requires Q, T_out_inside, T_out_outside, W_out,
   m_dot_condensate, T_wall_outside *and* wet_surface_fraction to all be
   within their respective tolerances simultaneously; every iterate is
   checked for finiteness.

Partial wet-area handling (fix, v0.6.0 patch)
------------------------------------------------
Prior to this patch, the mass-transfer step used the *full* outside area
``A_o`` unconditionally, even when only part of the surface was estimated
to be below the dew point. This module now estimates, every outer
iteration, a wet-surface fraction from a cheap **two-point** (inlet/
outlet) closed-form sensible-only wall-temperature estimate --
deliberately *not* the full four-probe ``core.heat_transfer.
thermal_iteration.WallTemperatureEnvelope`` (which would re-run its own
nested wall-temperature convergence four times per outer iteration; far
too expensive to call every iteration here). Both endpoint estimates hold
the *inside* bulk temperature at its current mean (``T_mean_inside``) and
vary only the *outside* bulk temperature between ``T_in_outside`` and the
current ``T_out_outside`` iterate -- a documented simplification distinct
from (and cheaper than) the full envelope reported elsewhere on the
result. The resulting ``(T_wall_min, T_wall_max)`` feed
``core.phase_change.wet_surface_fraction.estimate_wet_surface_fraction``
(the linear 0D model) to get this iteration's wet-surface fraction, which
is then relaxed exactly like every other iterate and used as
``A_wet = A_o * wet_surface_fraction`` for the mass-transfer step only --
sensible heat transfer always uses the full ``A_o`` (spec section 12).

``alfa_outside_effective`` (section 26 of the original v0.6.0 spec) is
reconstructed from the converged ``Q_total`` and ``(T_bulk_outside -
T_s)``, not treated as an independent correlation; ``EFFECTIVE_OUTSIDE_
ALPHA_LIMITED`` is attached and a finite floor value is used near the
``T_bulk_outside ~= T_s`` singularity rather than letting the effective
alfa blow up.

Frosting (section 15 of the original spec): if the converged (or any
intermediate) interface temperature would fall at/below the water triple
point, this is outside the liquid-condensate scope of v0.6.0;
``FrostingNotSupportedError`` is raised instead of silently clamping or
continuing.

Ref: see ``core.phase_change.mass_heat_transfer`` and
``core.phase_change.wet_gas_enthalpy`` module docstrings for the underlying
heat/mass-transfer analogy and enthalpy-basis references.
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
    h_wet_gas_dry_basis,
    temperature_from_h_wet_gas_dry_basis,
)
from core.phase_change.mass_heat_transfer import condensation_rate
from core.phase_change.wet_surface_fraction import estimate_wet_surface_fraction
from core.phase_change import warning_codes as WC

if TYPE_CHECKING:
    from core.models.bare_tube import BareTubeHeatExchanger

SOURCE = "outside_condensation_solver"

# Fixed internal guard against dividing by a near-zero wall-temperature
# span in the per-iteration wet-surface-fraction estimate (spec section 9).
# Not user-exposed: it is a numerical-safety constant, not a physical
# tuning knob (unlike phase_change_wet_fraction_tolerance, the convergence
# tolerance, which is user-configurable).
WET_FRACTION_SPAN_TOLERANCE_K = 1e-3


class FrostingNotSupportedError(RuntimeError):
    """Raised when the coupled solve would require sub-triple-point
    (frost/ice) surface conditions -- out of scope for v0.6.0 liquid-only
    condensate handling."""


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
    ``wet_surface_fraction`` starting at ``1.0`` -- the maximal, "aggressive"
    starting guess, converged down to the correct fraction; never an
    arbitrary zero) -- never from arbitrary zeros, per the v0.6.0 spec.
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

    Q_total_prev: float | None = None
    m_dot_condensate_prev: float | None = None

    all_warnings: list[ModelWarning] = []
    converged = False
    residuals: dict[str, float] = {}
    solution_state: dict[str, object] = {}

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
        T_wall_inlet_est = _sensible_only_wall_temperature(
            T_bulk_outside=T_in_outside, T_bulk_inside=T_mean_inside,
            R_o_film=R_o_film, R_downstream=R_downstream,
        )
        T_wall_outlet_est = _sensible_only_wall_temperature(
            T_bulk_outside=T_out_outside, T_bulk_inside=T_mean_inside,
            R_o_film=R_o_film, R_downstream=R_downstream,
        )
        T_wall_min_iter = min(T_wall_inlet_est, T_wall_outlet_est)
        T_wall_max_iter = max(T_wall_inlet_est, T_wall_outlet_est)
        T_wall_mean_iter = 0.5 * (T_wall_min_iter + T_wall_max_iter)

        dew_point_local = _local_dew_point_or_triple_point(
            outside_capability, W_mean, p_outside=p_outside,
        )

        wet_fraction_estimate = estimate_wet_surface_fraction(
            dew_point_temperature=dew_point_local,
            wall_temperature_min=T_wall_min_iter,
            wall_temperature_mean=T_wall_mean_iter,
            wall_temperature_max=T_wall_max_iter,
            temperature_span_tolerance_K=WET_FRACTION_SPAN_TOLERANCE_K,
            activation_band_K=activation_band_K,
        )
        wet_surface_fraction_calculated = wet_fraction_estimate.wet_surface_fraction
        wet_surface_fraction_new = wet_surface_fraction + relaxation_factor * (
            wet_surface_fraction_calculated - wet_surface_fraction
        )
        # Numerical safety clamp only -- estimate_wet_surface_fraction
        # already returns a value in [0, 1]; relaxing between two such
        # values cannot leave the interval, this only guards float noise.
        wet_surface_fraction_new = max(0.0, min(1.0, wet_surface_fraction_new))
        A_wet = A_o * wet_surface_fraction_new
        if not (0.0 <= A_wet <= A_o):
            raise ValueError(
                f"outside_condensation_solver: A_wet={A_wet:.6g} m2 outside "
                f"the valid range [0, {A_o:.6g}] m2."
            )

        T_s, Q_sensible, Q_latent, m_dot_condensate = _solve_interface_state(
            alfa_o_dry=alfa_o_dry,
            A_o=A_o,
            A_wet=A_wet,
            T_bulk_outside=T_mean_outside,
            T_bulk_inside=T_mean_inside,
            R_downstream=R_downstream,
            cp_gas=cp_gas_bulk,
            W_bulk=W_mean,
            p_outside=p_outside,
            M_dry=outside_capability.M_dry,
            m_dot_water_vapor_available=m_dot_dry_carrier * W_mean,
            lewis_number=lewis_number,
        )

        if T_s <= WATER_TRIPLE_POINT_TEMPERATURE_K:
            raise FrostingNotSupportedError(
                f"outside_condensation_solver: interface temperature "
                f"{T_s:.3f} K is at/below the water triple point "
                f"({WATER_TRIPLE_POINT_TEMPERATURE_K} K); frost/ice "
                "condensate is not supported in v0.6.0."
            )

        Q_total = Q_sensible + Q_latent
        W_out_new = max(0.0, W_in - m_dot_condensate / m_dot_dry_carrier)

        h_in_outside = h_wet_gas_dry_basis(T_in_outside, p_outside, W_in, outside_capability)
        h_drained_per_kg_dry = (
            (m_dot_condensate / m_dot_dry_carrier) * water_saturation_liquid_enthalpy(T=T_s)
        )
        h_out_target = h_in_outside - Q_total / m_dot_dry_carrier - h_drained_per_kg_dry

        T_out_outside_new = _invert_outside_enthalpy(
            h_target=h_out_target,
            p_outside=p_outside,
            W=W_out_new,
            capability=outside_capability,
            T_s=T_s,
            T_in_outside=T_in_outside,
        )

        cp_inside_mean = inside_provider.at(T=T_mean_inside, p=p_inside).cp
        T_out_inside_new = T_in_inside + Q_total / (m_dot_inside * cp_inside_mean)

        T_out_inside_relaxed = T_out_inside + relaxation_factor * (T_out_inside_new - T_out_inside)
        T_out_outside_relaxed = T_out_outside + relaxation_factor * (T_out_outside_new - T_out_outside)
        W_out_relaxed = W_out + relaxation_factor * (W_out_new - W_out)
        T_wall_inside_new = T_mean_inside + Q_total * R_i

        finite_ok = all(
            math.isfinite(v)
            for v in (
                Q_total, T_out_inside_relaxed, T_out_outside_relaxed,
                W_out_relaxed, m_dot_condensate, T_s, T_wall_inside_new,
                wet_surface_fraction_new,
            )
        )
        if not finite_ok:
            raise ValueError(
                "outside_condensation_solver: non-finite iterate at "
                f"iteration {iteration}."
            )

        residuals = {
            "Q_rel": (
                abs(Q_total - Q_total_prev) / max(abs(Q_total_prev), 1e-9)
                if Q_total_prev is not None else math.inf
            ),
            "T_out_inside_K": abs(T_out_inside_relaxed - T_out_inside),
            "T_out_outside_K": abs(T_out_outside_relaxed - T_out_outside),
            "W_out": abs(W_out_relaxed - W_out),
            "m_dot_condensate_kg_s": (
                abs(m_dot_condensate - m_dot_condensate_prev)
                if m_dot_condensate_prev is not None else math.inf
            ),
            "T_wall_outside_K": (
                abs(T_s - T_wall_outside_prev)
                if T_wall_outside_prev is not None else math.inf
            ),
            "wet_surface_fraction": abs(wet_surface_fraction_new - wet_surface_fraction),
        }

        solution_state = dict(
            T_out_inside=T_out_inside_relaxed,
            T_out_outside=T_out_outside_relaxed,
            T_wall_inside=T_wall_inside_new,
            T_wall_outside=T_s,
            W_out=W_out_relaxed,
            m_dot_condensate=m_dot_condensate,
            Q_sensible=Q_sensible,
            Q_latent=Q_latent,
            Q_total=Q_total,
            alfa_i=alfa_i,
            alfa_o_dry=alfa_o_dry,
            wall_temperature_min=T_wall_min_iter,
            wall_temperature_max=T_wall_max_iter,
            wet_surface_fraction=wet_surface_fraction_new,
            wet_surface_fraction_method=wet_fraction_estimate.method,
            wet_area=A_wet,
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
        T_wall_outside_prev = T_s
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
            and residuals["m_dot_condensate_kg_s"] < condensate_tolerance_kg_s
            and residuals["T_wall_outside_K"] < wall_temperature_tolerance_K
            and residuals["wet_surface_fraction"] < wet_fraction_tolerance
        ):
            converged = True
            break

    if not solution_state:
        raise ValueError("outside_condensation_solver: failed to produce a complete iterate.")

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
        wet_surface_fraction=solution_state["wet_surface_fraction"],
        wet_surface_fraction_method=solution_state["wet_surface_fraction_method"],
        wet_area=solution_state["wet_area"],
        outside_total_area=A_o,
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
    estimate (see module docstring) -- not for the actual coupled T_s used
    in the mass/energy balance, which is solved separately by
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
    T_bulk_outside: float,
    T_bulk_inside: float,
    R_downstream: float,
    cp_gas: float,
    W_bulk: float,
    p_outside: float,
    M_dry: float,
    m_dot_water_vapor_available: float,
    lewis_number: float,
    tolerance_K: float = 1e-5,
    max_iterations: int = 100,
) -> tuple[float, float, float, float]:
    """Bisect for the interface temperature T_s; return
    ``(T_s, Q_sensible, Q_latent, m_dot_condensate)``.

    See module docstring for the balance ``f(T_s) = q_sensible + q_latent -
    q_removed = 0`` and why bisection (not Newton) is used. Sensible heat
    transfer uses the full outside area ``A_o``; mass transfer (and hence
    latent heat) uses ``A_wet`` (spec section 12: sensible heat transfer is
    never restricted to the estimated wet fraction).
    """

    def evaluate(T_s: float) -> tuple[float, float, float, float]:
        q_sensible = alfa_o_dry * A_o * (T_bulk_outside - T_s)
        if T_s > WATER_TRIPLE_POINT_TEMPERATURE_K:
            try:
                W_sat_surface = saturated_water_ratio(p_total=p_outside, T=T_s, M_dry=M_dry)
            except ValueError:
                # T_s is at/above the boiling point of water at p_outside: no
                # saturated gas-phase state exists there, so the driving
                # force (W_bulk - W_sat_surface) is unambiguously negative
                # (no condensation at this candidate T_s).
                W_sat_surface = math.inf
        else:
            W_sat_surface = 0.0
        m_dot_cond = (
            condensation_rate(
                alfa_dry=alfa_o_dry,
                cp_gas=cp_gas,
                W_bulk=W_bulk,
                W_sat_surface=W_sat_surface,
                A_wet=A_wet,
                m_dot_water_vapor_available=m_dot_water_vapor_available,
                lewis_number=lewis_number,
            )
            if A_wet > 0.0
            else 0.0
        )
        h_fg = 0.0
        if m_dot_cond > 0.0 and T_s > WATER_TRIPLE_POINT_TEMPERATURE_K:
            from core.properties.water import water_latent_heat_of_vaporization

            h_fg = water_latent_heat_of_vaporization(T=T_s)
        q_latent = m_dot_cond * h_fg
        q_removed = (T_s - T_bulk_inside) / R_downstream
        return q_sensible + q_latent - q_removed, q_sensible, q_latent, m_dot_cond

    T_lo, T_hi = sorted((T_bulk_inside, T_bulk_outside))
    if T_hi - T_lo < 1e-9:
        # Degenerate (near-zero bulk-to-bulk difference): no meaningful
        # driving force; return the trivial no-condensation state.
        return T_bulk_outside, 0.0, 0.0, 0.0

    f_lo, *_ = evaluate(T_lo)
    f_hi, *_ = evaluate(T_hi)
    if f_lo < 0.0 or f_hi > 0.0:
        # Not bracketed by construction of the calling context (outside
        # must be the hot/cooling side for condensation to be possible at
        # all -- see core.phase_change.regime); surface any inconsistency
        # loudly rather than guessing.
        raise ValueError(
            "outside_condensation_solver: interface-temperature balance is "
            f"not bracketed on [{T_lo:.6g}, {T_hi:.6g}] K "
            f"(f_lo={f_lo:.6g}, f_hi={f_hi:.6g}); outside must be the "
            "hotter (cooling) side for condensation to be physically "
            "possible."
        )

    for _ in range(max_iterations):
        T_mid = 0.5 * (T_lo + T_hi)
        f_mid, q_sens_mid, q_lat_mid, m_cond_mid = evaluate(T_mid)
        if f_mid > 0.0:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if (T_hi - T_lo) < tolerance_K:
            break

    T_s = 0.5 * (T_lo + T_hi)
    _, q_sensible, q_latent, m_dot_cond = evaluate(T_s)
    return T_s, q_sensible, q_latent, m_dot_cond


def _invert_outside_enthalpy(
    *,
    h_target: float,
    p_outside: float,
    W: float,
    capability: PhaseChangeCapability,
    T_s: float,
    T_in_outside: float,
) -> float:
    T_lo = max(WATER_TRIPLE_POINT_TEMPERATURE_K + 0.5, min(T_s, T_in_outside) - 10.0)
    T_hi = T_in_outside + 5.0
    try:
        return temperature_from_h_wet_gas_dry_basis(
            h_target, p_outside, W, capability, T_bracket=(T_lo, T_hi)
        )
    except ValueError:
        return temperature_from_h_wet_gas_dry_basis(
            h_target, p_outside, W, capability,
            T_bracket=(WATER_TRIPLE_POINT_TEMPERATURE_K + 0.5, T_in_outside + 10.0),
        )


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
