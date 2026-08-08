# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Outside partial H2O condensation solver (v0.6.0).

Solves the coupled sensible + latent duty of a wet-gas stream flowing over
the outside of a bare-tube bank while a single-phase (dry) stream is heated
inside the tubes, given that the dry sensible-only baseline already showed
the outside tube-wall surface running below the local water dew point (see
``core.phase_change.regime`` for how that "possible" determination is
made, once, before this solver is ever invoked).

This module is intentionally the *only* place that iterates a phase-change
case -- ``core.phase_change.integration`` decides *whether* to call it;
``core.models.simulation``/``core.models.rating`` never duplicate this
loop.

Algorithm, per outer iteration (state: T_out_inside, T_out_outside, W_out)
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
3. Solve for the interface/outer-wall temperature ``T_s`` by bisection on

       f(T_s) = q_sensible(T_s) + q_latent(T_s) - q_removed(T_s) = 0

   where ``q_sensible = alfa_o_dry*A_o*(T_bulk_outside - T_s)``,
   ``q_latent = m_dot_condensation(T_s) * h_fg(T_s)`` (Chilton-Colburn mass
   transfer, ``core.phase_change.mass_heat_transfer``, zero when
   ``W_mean <= W_sat(T_s)``), and
   ``q_removed = (T_s - T_bulk_inside) / (R_i + R_wall)`` is the heat
   conducted away through the tube wall and inside film (the same
   cylindrical-wall resistance model as the dry solver,
   ``BareTubeHeatExchanger.tube_wall_resistance``). Bisection, not an
   unbounded Newton step, because both q_sensible and q_removed are affine
   in T_s and q_latent is a bounded, monotonic function of T_s over the
   bracket -- a robust sign-changing bracket always exists between
   T_bulk_inside and T_bulk_outside for a genuinely condensing case.
4. Mass balance: ``m_dot_condensate = m_dot_condensation(T_s)``,
   ``W_out_new = W_in - m_dot_condensate/m_dot_dry_carrier``.
5. Enthalpy balance (not ``m_dot*cp_mean*dT``):
   ``Q_total = q_sensible + q_latent`` from step 3; then
   ``H_out = H_in - Q_total/m_dot_dry_carrier - (m_dot_condensate/
   m_dot_dry_carrier)*h_f(T_s)`` is inverted for ``T_out_outside_new``
   (``core.phase_change.wet_gas_enthalpy``, bisection).
6. The inside stream is sensible-only by construction (v0.6.0 scope):
   ``T_out_inside_new = T_in_inside + Q_total/(m_dot_inside*cp_inside)``.
7. Relaxed update of (T_out_inside, T_out_outside, W_out); wall temperatures
   are carried to the next iteration's step 2 evaluation as the previous
   iterate (same lagged wall-correction idiom as
   ``solve_iterative_thermal_state``).
8. Convergence requires Q, T_out_inside, T_out_outside, W_out,
   m_dot_condensate and T_wall_outside to all be within their respective
   tolerances simultaneously (section 27 of the v0.6.0 task spec); every
   iterate is checked for finiteness.

``alfa_outside_effective`` (section 26) is reconstructed from the converged
``Q_total`` and ``(T_bulk_outside - T_s)``, not treated as an independent
correlation; ``EFFECTIVE_OUTSIDE_ALPHA_LIMITED`` is attached and a finite
floor value is used near the ``T_bulk_outside ~= T_s`` singularity rather
than letting the effective alfa blow up.

Frosting (section 15 of the spec): if the converged (or any intermediate)
interface temperature would fall at/below the water triple point, this is
outside the liquid-condensate scope of v0.6.0; ``FrostingNotSupportedError``
is raised instead of silently clamping or continuing.

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
from core.heat_transfer.thermal_iteration import _evaluate_local_wall_state
from core.common.warnings import ModelWarning, make_warning

from core.phase_change.types import PhaseChangeCapability
from core.phase_change.water_equilibrium import saturated_water_ratio
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.wet_gas_enthalpy import (
    h_wet_gas_dry_basis,
    temperature_from_h_wet_gas_dry_basis,
)
from core.phase_change.mass_heat_transfer import condensation_rate
from core.phase_change import warning_codes as WC

if TYPE_CHECKING:
    from core.models.bare_tube import BareTubeHeatExchanger

SOURCE = "outside_condensation_solver"


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

    outside_bulk_props: FluidTransportProperties
    inside_bulk_props: FluidTransportProperties

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
    max_iterations: int = 50,
    temperature_tolerance_K: float = 0.05,
    relative_Q_tolerance: float = 1e-4,
    water_ratio_tolerance: float = 1e-6,
    condensate_tolerance_kg_s: float = 1e-8,
    wall_temperature_tolerance_K: float = 0.05,
    relaxation_factor: float = 0.5,
) -> OutsideCondensationSolution:
    """Iteratively solve the coupled outside condensation state.

    Initialized from the dry baseline (``T_out_inside_init``,
    ``T_out_outside_init``, ``W_out`` starting at ``W_in``) -- never from
    arbitrary zeros, per the v0.6.0 spec.
    """
    _validate_positive(lewis_number, "lewis_number")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0.")
    for name, value in (
        ("temperature_tolerance_K", temperature_tolerance_K),
        ("relative_Q_tolerance", relative_Q_tolerance),
        ("water_ratio_tolerance", water_ratio_tolerance),
        ("wall_temperature_tolerance_K", wall_temperature_tolerance_K),
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

        T_s, Q_sensible, Q_latent, m_dot_condensate = _solve_interface_state(
            alfa_o_dry=alfa_o_dry,
            A_o=A_o,
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
            outside_bulk_props=local.outside_bulk_props,
            inside_bulk_props=local.inside_bulk_props,
        )

        Q_total_prev = Q_total
        m_dot_condensate_prev = m_dot_condensate
        T_wall_outside_prev = T_s
        T_wall_inside_prev = T_wall_inside_new
        T_out_inside, T_out_outside, W_out = (
            T_out_inside_relaxed, T_out_outside_relaxed, W_out_relaxed,
        )

        if (
            iteration >= 2
            and residuals["Q_rel"] < relative_Q_tolerance
            and residuals["T_out_inside_K"] < temperature_tolerance_K
            and residuals["T_out_outside_K"] < temperature_tolerance_K
            and residuals["W_out"] < water_ratio_tolerance
            and residuals["m_dot_condensate_kg_s"] < condensate_tolerance_kg_s
            and residuals["T_wall_outside_K"] < wall_temperature_tolerance_K
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
        outside_bulk_props=solution_state["outside_bulk_props"],
        inside_bulk_props=solution_state["inside_bulk_props"],
        warnings=tuple(all_warnings),
    )


def _solve_interface_state(
    *,
    alfa_o_dry: float,
    A_o: float,
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
    q_removed = 0`` and why bisection (not Newton) is used.
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
        m_dot_cond = condensation_rate(
            alfa_dry=alfa_o_dry,
            cp_gas=cp_gas,
            W_bulk=W_bulk,
            W_sat_surface=W_sat_surface,
            A_wet=A_o,
            m_dot_water_vapor_available=m_dot_water_vapor_available,
            lewis_number=lewis_number,
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
