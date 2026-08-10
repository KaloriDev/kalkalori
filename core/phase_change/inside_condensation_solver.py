# KalKalori â€” Heat Exchanger Open Engine
# GNU GPL v3 only

"""Coupled 0D partial-H2O-condensation solver for wet gas inside tubes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.common.warnings import ModelWarning, make_warning
from core.heat_transfer.thermal_iteration import (
    ThermalIterationDiagnostics,
    _evaluate_local_wall_state,
)
from core.phase_change import warning_codes as WC
from core.phase_change.condensation_solver_helpers import (
    condensate_enthalpy_flow,
    FrostingNotSupportedError,
    invert_wet_gas_enthalpy,
    local_dew_point_or_triple_point,
    sensible_only_wall_temperature,
    solve_condensing_interface_state,
)
from core.phase_change.types import PhaseChangeCapability
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.wet_gas_enthalpy import WetGasEnthalpyEvaluator
from core.phase_change.wet_surface_fraction import estimate_wet_surface_fraction
from core.properties.averaging import mean_temperature
from core.properties.common import FluidTransportProperties
from core.properties.fluids import PropertyProvider
from core.properties.water import (
    WATER_TRIPLE_POINT_TEMPERATURE_K,
)

if TYPE_CHECKING:
    from core.models.bare_tube import BareTubeHeatExchanger


SOURCE = "inside_condensation_solver"
WET_FRACTION_SPAN_TOLERANCE_K = 1e-3


@dataclass(frozen=True)
class InsideCondensationSolution:
    """Converged (or last-iterate) coupled sensible+latent inside state."""

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

    alfa_i_dry: float
    alfa_i_effective: float
    alfa_o: float
    U_effective: float
    UA_effective: float

    wall_temperature_min: float
    wall_temperature_max: float
    wall_temperature_wet_mean: float | None
    wet_surface_fraction: float
    wet_surface_fraction_method: str
    wet_area: float
    inside_total_area: float
    W_sat_wet_surface: float | None

    inside_bulk_props: FluidTransportProperties
    outside_bulk_props: FluidTransportProperties
    inside_wall_props: FluidTransportProperties | None
    outside_wall_props: FluidTransportProperties | None
    diagnostics: ThermalIterationDiagnostics
    warnings: tuple[ModelWarning, ...] = ()


def solve_inside_condensation(
    hx: "BareTubeHeatExchanger",
    *,
    inside_capability: PhaseChangeCapability,
    m_dot_dry_carrier: float,
    T_in_inside: float,
    p_inside: float,
    outside_provider: PropertyProvider,
    m_dot_outside: float,
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
) -> InsideCondensationSolution:
    """Solve inside wet-gas cooling with a regime locked before entry."""
    if not inside_capability.capable or inside_capability.W_in is None:
        raise ValueError("solve_inside_condensation requires a capable inside stream.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0.")
    for name, value in (
        ("lewis_number", lewis_number),
        ("temperature_tolerance_K", temperature_tolerance_K),
        ("relative_Q_tolerance", relative_Q_tolerance),
        ("water_ratio_tolerance", water_ratio_tolerance),
        ("wall_temperature_tolerance_K", wall_temperature_tolerance_K),
        ("wet_fraction_tolerance", wet_fraction_tolerance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")
    if not math.isfinite(condensate_tolerance_kg_s) or condensate_tolerance_kg_s < 0.0:
        raise ValueError("condensate_tolerance_kg_s must be non-negative and finite.")
    if not 0.0 < relaxation_factor <= 1.0:
        raise ValueError("relaxation_factor must be in (0, 1].")

    bundle = hx.bundle
    A_i = bundle.total_inner_area
    A_o = bundle.total_outer_area
    R_w = hx.tube_wall_resistance()
    W_in = inside_capability.W_in

    T_out_inside = T_out_inside_init
    T_out_outside = T_out_outside_init
    W_out = W_in
    wet_surface_fraction = 1.0
    T_wall_inside_prev: float | None = None
    T_wall_outside_prev: float | None = None
    T_wall_wet_mean_prev: float | None = None
    Q_total_prev: float | None = None
    m_dot_condensate_prev: float | None = None
    warnings: list[ModelWarning] = []
    residuals: dict[str, float] = {}
    state: dict[str, object] = {}
    converged = False
    enthalpy_evaluator = WetGasEnthalpyEvaluator(p_inside, inside_capability)
    h_in = enthalpy_evaluator.enthalpy(T_in_inside, W_in)

    for iteration in range(1, max_iterations + 1):
        T_mean_inside = mean_temperature(T_in_inside, T_out_inside)
        T_mean_outside = mean_temperature(T_in_outside, T_out_outside)
        W_mean = 0.5 * (W_in + W_out)
        inside_provider = wet_gas_provider_at_water_ratio(inside_capability, W_mean)
        m_dot_gas_mean = m_dot_dry_carrier * (1.0 + W_mean)

        local = _evaluate_local_wall_state(
            hx,
            m_dot_inside=m_dot_gas_mean,
            m_dot_outside=m_dot_outside,
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
        warnings.extend(local.warnings)

        alfa_i_dry = local.alfa_i
        alfa_o = local.alfa_o
        R_i_film = 1.0 / (alfa_i_dry * A_i)
        R_o = 1.0 / (alfa_o * A_o)
        R_downstream = R_w + R_o

        T_wall_inlet_raw = sensible_only_wall_temperature(
            T_bulk_wet_gas=T_in_inside,
            T_bulk_other=T_mean_outside,
            R_wet_film=R_i_film,
            R_downstream=R_downstream,
        )
        T_wall_outlet_raw = sensible_only_wall_temperature(
            T_bulk_wet_gas=T_out_inside,
            T_bulk_other=T_mean_outside,
            R_wet_film=R_i_film,
            R_downstream=R_downstream,
        )
        wall_min_raw = min(T_wall_inlet_raw, T_wall_outlet_raw)
        wall_max_raw = max(T_wall_inlet_raw, T_wall_outlet_raw)
        shape_mean = 0.5 * (wall_min_raw + wall_max_raw)
        wall_mean_reference = (
            T_wall_inside_prev if T_wall_inside_prev is not None else shape_mean
        )
        shift = wall_mean_reference - shape_mean
        wall_min = wall_min_raw + shift
        wall_max = wall_max_raw + shift

        dew_point = local_dew_point_or_triple_point(
            inside_capability,
            W_mean,
            p_wet_gas=p_inside,
        )
        wet_estimate = estimate_wet_surface_fraction(
            dew_point_temperature=dew_point,
            wall_temperature_min=wall_min,
            wall_temperature_mean=wall_mean_reference,
            wall_temperature_max=wall_max,
            temperature_span_tolerance_K=WET_FRACTION_SPAN_TOLERANCE_K,
            activation_band_K=activation_band_K,
        )
        fraction_calculated = wet_estimate.wet_surface_fraction
        if iteration == 1:
            fraction_new = fraction_calculated
        else:
            fraction_new = wet_surface_fraction + relaxation_factor * (
                fraction_calculated - wet_surface_fraction
            )
        fraction_new = max(0.0, min(1.0, fraction_new))
        T_wall_wet_mean = wet_estimate.wall_temperature_wet_mean
        if T_wall_wet_mean is None:
            fraction_new = 0.0
        A_wet = A_i * fraction_new

        if (
            A_wet > 0.0
            and T_wall_wet_mean is not None
            and T_wall_wet_mean <= WATER_TRIPLE_POINT_TEMPERATURE_K
        ):
            raise FrostingNotSupportedError(
                "inside: representative wet-surface temperature "
                f"{T_wall_wet_mean:.3f} K is at/below the water triple point."
            )

        (
            T_wall_inside,
            Q_sensible,
            Q_latent,
            m_dot_condensate,
            W_sat_wet_surface,
        ) = solve_condensing_interface_state(
            alfa_dry=alfa_i_dry,
            A_total=A_i,
            A_wet=A_wet,
            T_wall_wet_mean=T_wall_wet_mean,
            T_bulk_wet_gas=T_mean_inside,
            T_bulk_other=T_mean_outside,
            R_downstream=R_downstream,
            cp_gas=local.inside_bulk_props.cp,
            W_bulk=W_mean,
            p_wet_gas=p_inside,
            M_dry=inside_capability.M_dry,
            m_dot_water_vapor_available=m_dot_dry_carrier * W_mean,
            lewis_number=lewis_number,
        )
        if T_wall_inside <= WATER_TRIPLE_POINT_TEMPERATURE_K:
            raise FrostingNotSupportedError(
                f"inside: global wall temperature {T_wall_inside:.3f} K "
                "is at/below the water triple point."
            )

        Q_total = Q_sensible + Q_latent
        W_out_new = max(0.0, W_in - m_dot_condensate / m_dot_dry_carrier)
        condensate_enthalpy_rate = condensate_enthalpy_flow(
            m_dot_condensate=m_dot_condensate,
            condensation_mass_tolerance=condensate_tolerance_kg_s,
            wet_surface_fraction=fraction_new,
            wet_area=A_wet,
            wall_temperature_wet_mean=T_wall_wet_mean,
        )
        h_drained = condensate_enthalpy_rate / m_dot_dry_carrier
        h_out_target = h_in - Q_total / m_dot_dry_carrier - h_drained
        T_out_inside_new = invert_wet_gas_enthalpy(
            h_target=h_out_target,
            p_wet_gas=p_inside,
            W=W_out_new,
            capability=inside_capability,
            T_wall_mean=T_wall_inside,
            T_in_wet_gas=T_in_inside,
            evaluator=enthalpy_evaluator,
        )
        cp_outside = outside_provider.at(T=T_mean_outside, p=p_outside).cp
        T_out_outside_new = T_in_outside + Q_total / (m_dot_outside * cp_outside)

        T_out_inside_relaxed = T_out_inside + relaxation_factor * (
            T_out_inside_new - T_out_inside
        )
        T_out_outside_relaxed = T_out_outside + relaxation_factor * (
            T_out_outside_new - T_out_outside
        )
        W_out_relaxed = W_out + relaxation_factor * (W_out_new - W_out)
        T_wall_outside = T_mean_outside + Q_total * R_o

        numeric = (
            T_out_inside_relaxed,
            T_out_outside_relaxed,
            W_out_relaxed,
            T_wall_inside,
            T_wall_outside,
            wall_min,
            wall_max,
            fraction_new,
            A_wet,
            Q_sensible,
            Q_latent,
            Q_total,
            m_dot_condensate,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"inside condensation produced a non-finite iteration {iteration}.")

        if T_wall_wet_mean is None and T_wall_wet_mean_prev is None:
            wet_wall_residual = 0.0
        elif T_wall_wet_mean is None or T_wall_wet_mean_prev is None:
            wet_wall_residual = math.inf
        else:
            wet_wall_residual = abs(T_wall_wet_mean - T_wall_wet_mean_prev)

        residuals = {
            "Q_rel": (
                abs(Q_total - Q_total_prev) / max(abs(Q_total_prev), 1e-9)
                if Q_total_prev is not None
                else math.inf
            ),
            "T_out_inside_K": abs(T_out_inside_relaxed - T_out_inside),
            "T_out_outside_K": abs(T_out_outside_relaxed - T_out_outside),
            "W_out": abs(W_out_relaxed - W_out),
            "mass_balance_kg_s": abs(
                m_dot_dry_carrier * (W_in - W_out_relaxed) - m_dot_condensate
            ),
            "m_dot_condensate_kg_s": (
                abs(m_dot_condensate - m_dot_condensate_prev)
                if m_dot_condensate_prev is not None
                else math.inf
            ),
            "T_wall_inside_K": (
                abs(T_wall_inside - T_wall_inside_prev)
                if T_wall_inside_prev is not None
                else math.inf
            ),
            "T_wall_wet_mean_K": wet_wall_residual,
            "wall_envelope_center_K": abs(
                T_wall_inside - 0.5 * (wall_min + wall_max)
            ),
            "wet_surface_fraction": abs(
                fraction_calculated - wet_surface_fraction
            ),
            "wet_surface_fraction_step": abs(
                fraction_new - wet_surface_fraction
            ),
        }
        contains_mean = wall_min <= T_wall_inside <= wall_max

        state = {
            "T_out_inside": T_out_inside_relaxed,
            "T_out_outside": T_out_outside_relaxed,
            "T_wall_inside": T_wall_inside,
            "T_wall_outside": T_wall_outside,
            "W_out": W_out_relaxed,
            "m_dot_condensate": m_dot_condensate,
            "Q_sensible": Q_sensible,
            "Q_latent": Q_latent,
            "Q_total": Q_total,
            "alfa_i_dry": alfa_i_dry,
            "alfa_o": alfa_o,
            "wall_temperature_min": wall_min,
            "wall_temperature_max": wall_max,
            "wall_temperature_wet_mean": T_wall_wet_mean,
            "wet_surface_fraction": fraction_new,
            "wet_surface_fraction_method": wet_estimate.method,
            "wet_area": A_wet,
            "W_sat_wet_surface": W_sat_wet_surface,
            "inside_bulk_props": local.inside_bulk_props,
            "outside_bulk_props": local.outside_bulk_props,
            "inside_wall_props": local.inside_wall_props,
            "outside_wall_props": local.outside_wall_props,
            "internal_diagnostics": local.internal_diagnostics,
            "outside_nusselt_base": local.outside_nusselt_base,
            "outside_nusselt": local.outside_nusselt,
            "outside_wall_property_correction": local.outside_wall_property_correction,
        }

        Q_total_prev = Q_total
        m_dot_condensate_prev = m_dot_condensate
        T_wall_inside_prev = T_wall_inside
        T_wall_outside_prev = T_wall_outside
        T_wall_wet_mean_prev = T_wall_wet_mean
        T_out_inside = T_out_inside_relaxed
        T_out_outside = T_out_outside_relaxed
        W_out = W_out_relaxed
        wet_surface_fraction = fraction_new

        if (
            iteration >= 2
            and residuals["Q_rel"] < relative_Q_tolerance
            and residuals["T_out_inside_K"] < temperature_tolerance_K
            and residuals["T_out_outside_K"] < temperature_tolerance_K
            and residuals["W_out"] < water_ratio_tolerance
            and residuals["mass_balance_kg_s"] <= condensate_tolerance_kg_s
            and residuals["m_dot_condensate_kg_s"] < condensate_tolerance_kg_s
            and residuals["T_wall_inside_K"] < wall_temperature_tolerance_K
            and residuals["T_wall_wet_mean_K"] < wall_temperature_tolerance_K
            and residuals["wall_envelope_center_K"] < wall_temperature_tolerance_K
            and contains_mean
            and residuals["wet_surface_fraction"] < wet_fraction_tolerance
        ):
            converged = True
            break

    if not state:
        raise ValueError("inside condensation failed to produce a complete iterate.")
    if not (
        state["wall_temperature_min"]
        <= state["T_wall_inside"]
        <= state["wall_temperature_max"]
    ):
        raise ValueError("inside wall envelope does not contain its solved mean.")
    if not converged:
        warnings.append(
            make_warning(
                code=WC.INSIDE_CONDENSATION_NOT_CONVERGED,
                message=(
                    "inside condensation did not converge within "
                    f"max_iterations={max_iterations}; returning the last finite iterate."
                ),
                source=SOURCE,
                severity="warning",
            )
        )

    T_bulk_inside_final = mean_temperature(T_in_inside, state["T_out_inside"])
    delta_T_film = T_bulk_inside_final - state["T_wall_inside"]
    if abs(delta_T_film) < 1e-3:
        alfa_i_effective = state["alfa_i_dry"]
        warnings.append(
            make_warning(
                code=WC.EFFECTIVE_INSIDE_ALPHA_LIMITED,
                message=(
                    "inside bulk-to-wall temperature difference is too small "
                    "to reconstruct alfa_inside_effective; using alfa_inside_dry."
                ),
                source=SOURCE,
                severity="info",
            )
        )
    else:
        alfa_i_effective = state["Q_total"] / (A_i * delta_T_film)
    if not math.isfinite(alfa_i_effective) or alfa_i_effective <= 0.0:
        raise ValueError("inside effective heat-transfer coefficient is not positive.")

    R_i_effective = 1.0 / (alfa_i_effective * A_i)
    R_o_final = 1.0 / (state["alfa_o"] * A_o)
    UA_effective = 1.0 / (R_i_effective + R_w + R_o_final)
    U_effective = UA_effective / A_o
    internal = state["internal_diagnostics"]
    diagnostics = ThermalIterationDiagnostics(
        inside_Nu_base=internal.Nu_base,
        inside_Nu_corrected=internal.Nu_corrected,
        inside_length_correction=internal.length_correction,
        inside_wall_temperature_correction=internal.wall_temperature_correction,
        inside_combined_correction=internal.combined_correction,
        inside_alfa_base=internal.alfa_base,
        inside_alfa_corrected=internal.alfa_corrected,
        outside_Nu_base=state["outside_nusselt_base"],
        outside_Nu_corrected=state["outside_nusselt"],
        outside_wall_property_correction=state["outside_wall_property_correction"],
    )

    return InsideCondensationSolution(
        converged=converged,
        iterations=iteration,
        residuals=residuals,
        T_out_inside=state["T_out_inside"],
        T_out_outside=state["T_out_outside"],
        T_wall_inside=state["T_wall_inside"],
        T_wall_outside=state["T_wall_outside"],
        W_out=state["W_out"],
        m_dot_condensate=state["m_dot_condensate"],
        Q_sensible=state["Q_sensible"],
        Q_latent=state["Q_latent"],
        Q_total=state["Q_total"],
        alfa_i_dry=state["alfa_i_dry"],
        alfa_i_effective=alfa_i_effective,
        alfa_o=state["alfa_o"],
        U_effective=U_effective,
        UA_effective=UA_effective,
        wall_temperature_min=state["wall_temperature_min"],
        wall_temperature_max=state["wall_temperature_max"],
        wall_temperature_wet_mean=state["wall_temperature_wet_mean"],
        wet_surface_fraction=state["wet_surface_fraction"],
        wet_surface_fraction_method=state["wet_surface_fraction_method"],
        wet_area=state["wet_area"],
        inside_total_area=A_i,
        W_sat_wet_surface=state["W_sat_wet_surface"],
        inside_bulk_props=state["inside_bulk_props"],
        outside_bulk_props=state["outside_bulk_props"],
        inside_wall_props=state["inside_wall_props"],
        outside_wall_props=state["outside_wall_props"],
        diagnostics=diagnostics,
        warnings=tuple(warnings),
    )
