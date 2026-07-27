# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Rating-side orchestration for outside water condensation (spec section 32).

Mirrors ``core.phase_change.integration`` but for
``core.models.bare_tube.BareTubeHeatExchanger.rate``. Rating is a *closed*
heat balance (both temperature programs already known), which makes the
outside condensation case algebraically simpler than Simulation: given
``Q_required`` (from an explicit ``Q`` or from the fully specified,
non-condensing inside side -- never inferred from the wet outside side's
sensible cp, which would silently ignore latent duty) and the outside
side's known ``T_in``/``T_out``/``m_dot``/inlet water content, the required
outlet water content ``W_out`` is the *unique* root of the enthalpy
balance -- no wall-temperature/mass-transfer iteration is needed here
(unlike ``core.phase_change.outside_condensation_solver``, used by
Simulation, where outlet conditions are unknowns to be solved for).

The representative condensate/interface temperature for the latent split
is taken as the outside outlet bulk temperature ``T_out`` (Rating does not
run the coupled wall-temperature solve Simulation does); this is a
documented 0D simplification distinct from Simulation's wall-temperature-
based interface state.

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
from core.models.heat_balance import BalanceSideSpec, ClosedBalance, ClosedBalanceSide, close_heat_balance
from core.models.rating import run_rating

from core.phase_change import warning_codes as WC
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.integration import (
    PhaseChangeSettings,
    _build_capability_side_result,
    _dew_point_at_ratio,
    _dew_point_for,
    check_single_active_side,
)
from core.phase_change.regime import decide_regime, representative_wall_temperature
from core.phase_change.types import PhaseChangeDirection, PhaseChangeMode, PhaseChangeResult
from core.properties.water import water_latent_heat_of_vaporization, water_saturation_liquid_enthalpy
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.wet_gas_enthalpy import h_wet_gas_dry_basis
from core.phase_change.wet_surface_fraction import estimate_wet_surface_fraction

SOURCE = "phase_change_rating_integration"


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

    closed_balance = close_heat_balance(
        inside, outside, Q=Q, effectiveness=effectiveness,
        over_specified_tolerance=over_specified_tolerance,
    )
    dry_result = run_rating(
        hx, closed_balance,
        flow_arrangement=flow_arrangement, K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
        euler_provider=euler_provider, include_simulation=include_simulation,
        max_iterations=max_iterations, wall_temperature_tolerance_K=wall_temperature_tolerance_K,
        relative_alfa_tolerance=relative_alfa_tolerance, relaxation_factor=relaxation_factor,
    )

    inside_capability = detect_phase_change_capability(inside.provider)
    outside_capability = detect_phase_change_capability(outside.provider)

    if not inside_capability.capable and not outside_capability.capable:
        return replace(
            dry_result,
            inside_phase_change=_build_capability_side_result(
                side="inside", mode=inside.phase_change_mode, capability=inside_capability,
                possible=False, near_onset=False, dew_point=None, p=inside.p,
            ),
            outside_phase_change=_build_capability_side_result(
                side="outside", mode=outside.phase_change_mode, capability=outside_capability,
                possible=False, near_onset=False, dew_point=None, p=outside.p,
            ),
        )

    thermal_state = dry_result.thermal_state
    envelope = dry_result.wall_temperature_envelope

    inside_regime = None
    inside_dew_point = None
    if inside_capability.capable:
        inside_dew_point = _dew_point_for(inside_capability, p=inside.p)
        if inside_dew_point is not None:
            inside_regime = decide_regime(
                dew_point_K=inside_dew_point,
                wall_temperature_representative_K=representative_wall_temperature(
                    side="inside", thermal_state=thermal_state, wall_envelope=envelope
                ),
                onset_tolerance_K=settings.onset_tolerance_K,
                activation_band_K=settings.activation_band_K,
            )

    outside_regime = None
    outside_dew_point = None
    if outside_capability.capable:
        outside_dew_point = _dew_point_for(outside_capability, p=outside.p)
        if outside_dew_point is not None:
            outside_regime = decide_regime(
                dew_point_K=outside_dew_point,
                wall_temperature_representative_K=representative_wall_temperature(
                    side="outside", thermal_state=thermal_state, wall_envelope=envelope
                ),
                onset_tolerance_K=settings.onset_tolerance_K,
                activation_band_K=settings.activation_band_K,
            )

    inside_possible = bool(inside_regime is not None and inside_regime.is_condensing)
    outside_possible = bool(outside_regime is not None and outside_regime.is_condensing)
    inside_near_onset = bool(inside_regime is not None and inside_regime.is_near_onset)
    outside_near_onset = bool(outside_regime is not None and outside_regime.is_near_onset)

    inside_auto_possible = inside_possible and inside.phase_change_mode is PhaseChangeMode.AUTO
    outside_auto_possible = outside_possible and outside.phase_change_mode is PhaseChangeMode.AUTO

    # Rating has no iterate=False escape hatch, so the guard in
    # check_single_active_side never fires here (iterate=True always).
    check_single_active_side(inside_auto_possible, outside_auto_possible, iterate=True)

    inside_result = _build_capability_side_result(
        side="inside", mode=inside.phase_change_mode, capability=inside_capability,
        possible=inside_possible, near_onset=inside_near_onset,
        dew_point=inside_dew_point, p=inside.p,
    )

    if not outside_auto_possible:
        outside_result = _build_capability_side_result(
            side="outside", mode=outside.phase_change_mode, capability=outside_capability,
            possible=outside_possible, near_onset=outside_near_onset,
            dew_point=outside_dew_point, p=outside.p,
        )
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
    if Q is None and not (inside.m_dot is not None and inside.T_out is not None):
        raise ValueError(
            "Rating with active outside condensation requires the duty to "
            "come from an explicit Q or from a fully specified "
            "(m_dot + T_in + T_out) inside side -- it cannot be inferred "
            "from the outside (condensing) side's sensible heat capacity "
            "alone, since that would silently ignore latent duty."
        )

    Q_required = closed_balance.Q
    W_in = outside_capability.W_in
    m_dot_dry_carrier = outside.m_dot / (1.0 + W_in)
    T_representative = outside.T_out

    def residual(W_out: float) -> float:
        h_in = h_wet_gas_dry_basis(outside.T_in, outside.p, W_in, outside_capability)
        h_out = h_wet_gas_dry_basis(outside.T_out, outside.p, W_out, outside_capability)
        h_drained = (W_in - W_out) * water_saturation_liquid_enthalpy(T=T_representative)
        Q_implied = m_dot_dry_carrier * (h_in - h_out - h_drained)
        return Q_implied - Q_required

    W_lo, W_hi = 0.0, W_in
    r_lo, r_hi = residual(W_lo), residual(W_hi)
    if not (min(r_lo, r_hi) <= 0.0 <= max(r_lo, r_hi)):
        raise ValueError(
            "Rating with active outside condensation: cannot close the "
            f"outside water/enthalpy balance for Q_required={Q_required:.6g} W "
            f"within 0 <= W_out <= W_in={W_in:.6g} kg/kg (residuals at "
            f"bracket ends: {r_lo:.6g} W, {r_hi:.6g} W). The specified "
            "temperature program is not thermodynamically consistent with "
            "partial H2O condensation."
        )

    for _ in range(200):
        W_mid = 0.5 * (W_lo + W_hi)
        r_mid = residual(W_mid)
        if (r_mid > 0.0) == (r_lo > 0.0):
            W_lo, r_lo = W_mid, r_mid
        else:
            W_hi, r_hi = W_mid, r_mid
        if (W_hi - W_lo) < 1e-9:
            break
    W_out = 0.5 * (W_lo + W_hi)

    if not (0.0 <= W_out <= W_in):
        raise ValueError(
            f"Rating with active outside condensation: solved W_out={W_out:.6g} "
            f"kg/kg is outside the valid range [0, {W_in:.6g}]."
        )

    m_dot_condensate = m_dot_dry_carrier * (W_in - W_out)
    m_dot_water_vapor_in = m_dot_dry_carrier * W_in
    m_dot_water_vapor_out = m_dot_dry_carrier * W_out

    dT_outside = outside.T_in - outside.T_out
    if abs(dT_outside) < 1e-9:
        raise ValueError("Rating with active outside condensation requires T_in != T_out on the outside side.")
    C_effective_outside = Q_required / abs(dT_outside)

    W_mean = 0.5 * (W_in + W_out)
    outside_provider_mean = wet_gas_provider_at_water_ratio(outside_capability, W_mean)

    outside_closed_side = ClosedBalanceSide(
        provider=outside_provider_mean, p=outside.p, m_dot=outside.m_dot,
        T_in=outside.T_in, T_out=outside.T_out,
        cp_mean=C_effective_outside / outside.m_dot, C=C_effective_outside,
    )

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

    new_closed_balance = ClosedBalance(
        inside=closed_balance.inside,
        outside=outside_closed_side,
        hot_is_inside=closed_balance.hot_is_inside,
        Q=Q_required,
        Q_max=Q_max,
        effectiveness=eff,
        warnings=balance_warnings,
    )

    wet_rating_result = run_rating(
        hx, new_closed_balance,
        flow_arrangement=flow_arrangement, K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
        euler_provider=euler_provider, include_simulation=include_simulation,
        max_iterations=max_iterations, wall_temperature_tolerance_K=wall_temperature_tolerance_K,
        relative_alfa_tolerance=relative_alfa_tolerance, relaxation_factor=relaxation_factor,
    )

    h_fg_representative = water_latent_heat_of_vaporization(T=T_representative)
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

    alfa_o_dry = wet_rating_result.alfa_o
    T_wall_outside_repr = wet_rating_result.thermal_state.outside_wall_temperature
    T_mean_outside = 0.5 * (outside.T_in + outside.T_out)
    delta_T_film = T_mean_outside - T_wall_outside_repr
    if abs(delta_T_film) < 1e-3:
        alfa_o_effective = alfa_o_dry
        balance_warnings.append(
            make_warning(
                code=WC.EFFECTIVE_OUTSIDE_ALPHA_LIMITED,
                message=(
                    "outside: bulk-to-surface temperature difference is too "
                    "small to reconstruct alfa_outside_effective from "
                    "Q_total; falling back to alfa_outside_dry."
                ),
                source=SOURCE,
                severity="info",
            )
        )
    else:
        alfa_o_effective = Q_required / (wet_rating_result.A_o * delta_T_film)

    wet_surface = estimate_wet_surface_fraction(
        hx,
        inside_provider=inside.provider,
        outside_capability=outside_capability,
        m_dot_inside=closed_balance.inside.m_dot,
        m_dot_dry_carrier=m_dot_dry_carrier,
        T_in_inside=closed_balance.inside.T_in,
        T_out_inside=closed_balance.inside.T_out,
        T_in_outside=outside.T_in,
        T_out_outside=outside.T_out,
        W_in=W_in,
        W_out=W_out,
        p_inside=inside.p,
        p_outside=outside.p,
        euler_provider=euler_provider,
        activation_band_K=settings.activation_band_K,
    )

    mass_balance_error = m_dot_water_vapor_in - (m_dot_water_vapor_out + m_dot_condensate)
    energy_balance_error = Q_required - (Q_sensible + Q_latent)

    outside_result = PhaseChangeResult(
        side="outside",
        mode=outside.phase_change_mode,
        direction=PhaseChangeDirection.CONDENSATION,
        component=outside_capability.component,
        capable=True,
        possible=True,
        active=True,
        converged=True,
        iterations=1,
        method="outside_condensation_rating_closed_form",
        W_in=W_in,
        W_out=W_out,
        m_dot_dry_carrier=m_dot_dry_carrier,
        m_dot_water_vapor_in=m_dot_water_vapor_in,
        m_dot_water_vapor_out=m_dot_water_vapor_out,
        m_dot_condensate=m_dot_condensate,
        m_dot_gas_in=outside.m_dot,
        m_dot_gas_out=m_dot_dry_carrier + m_dot_water_vapor_out,
        dew_point_in=outside_dew_point,
        dew_point_out=_dew_point_at_ratio(outside_capability, W_out, p=outside.p),
        wall_temperature_mean=T_wall_outside_repr,
        wall_temperature_min=wet_surface.wall_temperature_min,
        wall_temperature_max=wet_surface.wall_temperature_max,
        wet_surface_fraction=wet_surface.wet_surface_fraction,
        wet_surface_fraction_method=wet_surface.method,
        alfa_dry=alfa_o_dry,
        alfa_effective=alfa_o_effective,
        lewis_number=settings.lewis_number,
        Q_sensible=Q_sensible,
        Q_latent=Q_latent,
        Q_total=Q_required,
        mass_balance_error=mass_balance_error,
        energy_balance_error=energy_balance_error,
        residuals={},
        assumptions=(
            "closed_form_no_wall_iteration",
            "condensate_temperature_equals_outside_outlet_bulk_temperature",
            "fully_drained_liquid_condensate",
            "dry_gas_composition_unchanged_by_condensation",
        ),
        warnings=tuple(balance_warnings) + tuple(wet_surface.warnings),
    )

    return replace(
        wet_rating_result,
        inside_phase_change=inside_result,
        outside_phase_change=outside_result,
    )
