# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Shared 0D p-h solver for pure water heated inside plain tubes.

At constant nominal pressure, increasing enthalpy creates the ordered zones
``PREHEAT -> EVAPORATION -> SUPERHEAT``. Each non-empty zone owns its duty,
inside/outside HTC, outer-area U, outer area, UA, and endpoint states.
Boiling heat flux is solved self-consistently on tube inside area; the
opposing hot-side properties and flow coefficient are recomputed for every
trial duty through the neutral outside-side helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from time import perf_counter

from core.common.warnings import ModelWarning, make_warning
from core.geometry.tube import TubeOrientation
from core.heat_transfer.internal_flow import heat_transfer_coefficient_internal_diagnostics
from core.heat_transfer.outside_side import OutsideSideEvaluation, evaluate_outside_side
from core.heat_transfer.tube_resistance import (
    equivalent_inside_alpha_outer_basis,
    fixed_outer_basis_resistances,
    overall_u_outer_basis,
)
from core.properties.adapters import to_internal_fluid_props
from core.properties.common import FluidTransportProperties
from core.properties.water import (
    WaterSteamPhase,
    WaterSteamProperties,
    WaterSteamSaturationProperties,
    water_saturation_snapshot,
    water_steam_props_iapws97,
)
from core.phase_change.water_evaporation import (
    WaterEvaporationZoneResult,
    solve_water_evaporation_zone,
)


class WaterEvaporatorZoneKind(str, Enum):
    PREHEAT = "preheat"
    EVAPORATION = "evaporation"
    SUPERHEAT = "superheat"


class WaterCondensationRequiredError(RuntimeError):
    """Raised when the requested enthalpy direction is cooling."""

    warning_code = "WATER_CONDENSATION_REQUIRED"


@dataclass(frozen=True)
class WaterEvaporatorZoneResult:
    kind: WaterEvaporatorZoneKind
    state_in: WaterSteamProperties
    state_out: WaterSteamProperties
    h_in: float
    h_out: float
    T_in: float
    T_out: float
    Q: float
    alpha_inside: float
    alpha_outside: float
    U: float
    area: float
    UA: float
    heat_flux_inner: float | None = None
    heat_flux_outer: float | None = None
    heat_flux_converged: bool = True
    heat_flux_iterations: int = 0
    heat_flux_residual: float = 0.0
    quality_in: float | None = None
    quality_out: float | None = None
    evaporation: WaterEvaporationZoneResult | None = None
    warnings: tuple[ModelWarning, ...] = ()


@dataclass(frozen=True)
class WaterEvaporatorSolution:
    mode: str
    state_in: WaterSteamProperties
    state_midpoint: WaterSteamProperties
    state_out: WaterSteamProperties
    saturation: WaterSteamSaturationProperties
    mass_flow_water: float
    mass_flux: float
    T_out_outside: float
    Q_preheat: float
    Q_evaporation: float
    Q_superheat: float
    Q_total: float
    A_preheat: float
    A_evaporation: float
    A_superheat: float
    A_total: float
    UA_preheat: float
    UA_evaporation: float
    UA_superheat: float
    UA_total: float
    zone_alpha_preheat: float | None
    zone_alpha_evaporation: float | None
    zone_alpha_superheat: float | None
    zone_U_preheat: float | None
    zone_U_evaporation: float | None
    zone_U_superheat: float | None
    inside_alpha_equivalent: float
    inside_alpha_area_weighted: float
    outside_alpha: float
    outside_props_mean: FluidTransportProperties
    outside_evaluation: OutsideSideEvaluation
    U_equivalent: float
    m_dot_evaporated: float
    heat_flux_inner_evaporation: float | None
    heat_flux_outer_evaporation: float | None
    heat_flux_converged: bool
    heat_flux_iterations: int
    heat_flux_residual: float
    zones: tuple[WaterEvaporatorZoneResult, ...]
    converged: bool
    iterations: int
    root_iterations: int
    property_evaluations: int
    cache_hits: int
    runtime_s: float
    two_phase_pressure_drop_supported: bool
    two_phase_pressure_drop_status: str
    warnings: tuple[ModelWarning, ...]
    assumptions: tuple[str, ...]

    @property
    def phase_in(self) -> WaterSteamPhase:
        return self.state_in.phase

    @property
    def phase_out(self) -> WaterSteamPhase:
        return self.state_out.phase

    @property
    def quality_in(self) -> float | None:
        return self.state_in.quality

    @property
    def quality_out(self) -> float | None:
        return self.state_out.quality

    @property
    def zone_fraction_preheat(self) -> float:
        return self.A_preheat / self.A_total if self.A_total > 0.0 else 0.0

    @property
    def zone_fraction_evaporation(self) -> float:
        return self.A_evaporation / self.A_total if self.A_total > 0.0 else 0.0

    @property
    def zone_fraction_superheat(self) -> float:
        return self.A_superheat / self.A_total if self.A_total > 0.0 else 0.0


@dataclass(frozen=True)
class _ZoneSpec:
    kind: WaterEvaporatorZoneKind
    h_in: float
    h_out: float


@dataclass(frozen=True)
class _TrialResult:
    h_out: float
    T_out_outside: float
    zones: tuple[WaterEvaporatorZoneResult, ...]
    required_area: float
    UA_total: float
    outside: OutsideSideEvaluation
    mass_flux: float
    warnings: tuple[ModelWarning, ...]


class _SolveCache:
    def __init__(
        self,
        p_water: float,
        outside_provider,
        p_outside: float,
        inlet_state: WaterSteamProperties,
    ):
        self.p_water = p_water
        self.outside_provider = outside_provider
        self.p_outside = p_outside
        self.water_states = {inlet_state.h: inlet_state}
        self.outside_props: dict[float, FluidTransportProperties] = {}
        self.property_evaluations = 0
        self.cache_hits = 0

    def water_state(self, h: float) -> WaterSteamProperties:
        if h in self.water_states:
            self.cache_hits += 1
            return self.water_states[h]
        state = water_steam_props_iapws97(p=self.p_water, h=h)
        self.water_states[h] = state
        self.property_evaluations += 1
        return state

    def outside_state(self, T: float) -> FluidTransportProperties:
        if T in self.outside_props:
            self.cache_hits += 1
            return self.outside_props[T]
        raw = self.outside_provider.at(T=T, p=self.p_outside)
        transport = getattr(raw, "transport", raw)
        if not isinstance(transport, FluidTransportProperties):
            transport = FluidTransportProperties(
                rho=float(transport.rho), mu=float(transport.mu),
                k=float(transport.k), cp=float(transport.cp),
            )
        self.outside_props[T] = transport
        self.property_evaluations += 1
        return transport


def solve_water_evaporator(
    hx,
    *,
    inlet_state: WaterSteamProperties,
    mass_flow_water: float,
    outside_provider,
    mass_flow_outside: float,
    T_in_outside: float,
    p_outside: float,
    orientation: TubeOrientation | None,
    available_area: float | None = None,
    euler_provider: str = "zukauskas",
    max_iterations: int = 80,
    relative_area_tolerance: float = 1.0e-8,
    heat_flux_max_iterations: int = 80,
    heat_flux_relative_tolerance: float = 1.0e-8,
) -> WaterEvaporatorSolution:
    """Simulation driver: find the increasing outlet enthalpy that fills Ao."""
    started = perf_counter()
    _validate_common_inputs(
        inlet_state=inlet_state,
        mass_flow_water=mass_flow_water,
        mass_flow_outside=mass_flow_outside,
        T_in_outside=T_in_outside,
        p_outside=p_outside,
        orientation=orientation,
    )
    if max_iterations <= 0 or relative_area_tolerance <= 0.0:
        raise ValueError("Water-evaporator area iteration controls must be positive.")
    if T_in_outside <= inlet_state.T:
        raise WaterCondensationRequiredError(
            "The opposing inlet is not hotter than the pure-water tube inlet; "
            "a positive water-heating duty is unavailable."
        )
    saturation = water_saturation_snapshot(inlet_state.p)
    cache = _SolveCache(inlet_state.p, outside_provider, p_outside, inlet_state)
    if available_area is None:
        available_area = hx.bundle.total_outer_area
    _positive_finite(available_area, "available_area")

    T_upper = T_in_outside - 1.0e-3
    if math.isclose(T_upper, saturation.Tsat, rel_tol=0.0, abs_tol=1.0e-6):
        T_upper -= 1.0e-3
    if T_upper <= 273.16:
        raise ValueError("No positive water-heating interval exists below the hot inlet.")
    upper_state = water_steam_props_iapws97(T=T_upper, p=inlet_state.p)
    if upper_state.h <= inlet_state.h:
        raise ValueError("No positive water-heating enthalpy interval exists.")
    q_water_pinch = mass_flow_water * (upper_state.h - inlet_state.h)
    # Also bound the trial by the opposing stream's positive-temperature
    # capacity.  Its 0D energy model uses cp at the current mean state, so
    # evaluating cp at the limiting mean gives the matching finite bracket
    # without ever passing a negative temperature to the property provider.
    T_hot_floor = max(inlet_state.T + 1.0e-3, 1.0e-3)
    hot_limit_props = cache.outside_state(
        0.5 * (T_in_outside + T_hot_floor)
    )
    q_hot_pinch = (
        mass_flow_outside
        * hot_limit_props.cp
        * (T_in_outside - T_hot_floor)
    )
    q_high = min(q_water_pinch, q_hot_pinch) * (1.0 - 1e-10)
    if not math.isfinite(q_high) or q_high <= 0.0:
        raise ValueError("No positive finite thermal-pinch duty is available.")
    upper_state = water_steam_props_iapws97(
        p=inlet_state.p, h=inlet_state.h + q_high / mass_flow_water
    )
    q_low = max(1e-9, q_high * 1e-12)

    def trial(q: float) -> _TrialResult:
        return _evaluate_duty(
            hx,
            inlet_state=inlet_state,
            saturation=saturation,
            mass_flow_water=mass_flow_water,
            outside_provider=outside_provider,
            mass_flow_outside=mass_flow_outside,
            T_in_outside=T_in_outside,
            p_outside=p_outside,
            orientation=orientation,
            Q_total=q,
            cache=cache,
            euler_provider=euler_provider,
            heat_flux_max_iterations=heat_flux_max_iterations,
            heat_flux_relative_tolerance=heat_flux_relative_tolerance,
        )

    # Missing orientation is irrelevant when the accepted result remains
    # liquid. Bracket at saturated liquid first and require it only if the
    # remaining geometry would enter the evaporation zone.
    if orientation is None and inlet_state.h < saturation.hf < upper_state.h:
        q_to_saturated_liquid = mass_flow_water * (saturation.hf - inlet_state.h)
        boundary = trial(q_to_saturated_liquid)
        if boundary.required_area < available_area:
            _require_boiling_orientation(orientation)
        q_high = q_to_saturated_liquid
        high = boundary
    elif saturation.hf <= inlet_state.h < saturation.hg:
        _require_boiling_orientation(orientation)
        high = trial(q_high)
    else:
        high = trial(q_high)

    if math.isfinite(high.required_area) and high.required_area < available_area:
        raise ValueError(
            "Water-evaporator area root was not bracketed before the thermal pinch."
        )

    final_trial: _TrialResult | None = None
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        q_mid = 0.5 * (q_low + q_high)
        current = trial(q_mid)
        residual = current.required_area - available_area
        final_trial = current
        if math.isfinite(residual) and abs(residual) <= relative_area_tolerance * available_area:
            converged = True
            break
        if not math.isfinite(current.required_area) or residual > 0.0:
            q_high = q_mid
        else:
            q_low = q_mid
    if final_trial is None or not converged:
        final_trial = trial(0.5 * (q_low + q_high))

    return _build_solution(
        hx=hx,
        mode="simulation",
        inlet_state=inlet_state,
        saturation=saturation,
        mass_flow_water=mass_flow_water,
        trial=final_trial,
        cache=cache,
        converged=converged,
        iterations=iterations,
        root_iterations=iterations,
        runtime_s=perf_counter() - started,
    )


def rate_water_evaporator(
    hx,
    *,
    inlet_state: WaterSteamProperties,
    mass_flow_water: float,
    outside_provider,
    mass_flow_outside: float,
    T_in_outside: float,
    p_outside: float,
    orientation: TubeOrientation | None,
    outlet_state: WaterSteamProperties | None = None,
    Q_total: float | None = None,
    euler_provider: str = "zukauskas",
    heat_flux_max_iterations: int = 80,
    heat_flux_relative_tolerance: float = 1.0e-8,
) -> WaterEvaporatorSolution:
    """Rating driver: calculate required zone areas for outlet state/duty."""
    started = perf_counter()
    _validate_common_inputs(
        inlet_state=inlet_state,
        mass_flow_water=mass_flow_water,
        mass_flow_outside=mass_flow_outside,
        T_in_outside=T_in_outside,
        p_outside=p_outside,
        orientation=orientation,
    )
    if (outlet_state is None) == (Q_total is None):
        raise ValueError("Provide exactly one of outlet_state or Q_total for water Rating.")
    if outlet_state is not None:
        if not math.isclose(outlet_state.p, inlet_state.p, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Water inlet and outlet must use the same nominal pressure.")
        Q_total = mass_flow_water * (outlet_state.h - inlet_state.h)
    if Q_total is None or not math.isfinite(Q_total) or Q_total <= 0.0:
        raise WaterCondensationRequiredError(
            "Water evaporator Rating requires increasing enthalpy and positive heat input."
        )
    saturation = water_saturation_snapshot(inlet_state.p)
    cache = _SolveCache(inlet_state.p, outside_provider, p_outside, inlet_state)
    trial = _evaluate_duty(
        hx,
        inlet_state=inlet_state,
        saturation=saturation,
        mass_flow_water=mass_flow_water,
        outside_provider=outside_provider,
        mass_flow_outside=mass_flow_outside,
        T_in_outside=T_in_outside,
        p_outside=p_outside,
        orientation=orientation,
        Q_total=Q_total,
        cache=cache,
        euler_provider=euler_provider,
        heat_flux_max_iterations=heat_flux_max_iterations,
        heat_flux_relative_tolerance=heat_flux_relative_tolerance,
    )
    if not math.isfinite(trial.required_area):
        raise ValueError("Specified water Rating violates a positive zone temperature difference.")
    return _build_solution(
        hx=hx,
        mode="rating",
        inlet_state=inlet_state,
        saturation=saturation,
        mass_flow_water=mass_flow_water,
        trial=trial,
        cache=cache,
        converged=True,
        iterations=1,
        root_iterations=0,
        runtime_s=perf_counter() - started,
    )


def _evaluate_duty(
    hx,
    *,
    inlet_state: WaterSteamProperties,
    saturation: WaterSteamSaturationProperties,
    mass_flow_water: float,
    outside_provider,
    mass_flow_outside: float,
    T_in_outside: float,
    p_outside: float,
    orientation: TubeOrientation | None,
    Q_total: float,
    cache: _SolveCache,
    euler_provider: str,
    heat_flux_max_iterations: int,
    heat_flux_relative_tolerance: float,
) -> _TrialResult:
    h_out = inlet_state.h + Q_total / mass_flow_water
    tolerance = max(1e-3, 1e-9 * max(abs(saturation.hf), abs(saturation.hg)))
    if abs(h_out - saturation.hf) <= tolerance:
        h_out = saturation.hf
    elif abs(h_out - saturation.hg) <= tolerance:
        h_out = saturation.hg
    if h_out <= inlet_state.h:
        raise WaterCondensationRequiredError("Water enthalpy must increase during heating.")

    T_out_outside = _outside_outlet_temperature(
        Q_total=Q_total,
        mass_flow_outside=mass_flow_outside,
        T_in_outside=T_in_outside,
        cache=cache,
    )
    T_mean_outside = 0.5 * (T_in_outside + T_out_outside)
    outside_props = cache.outside_state(T_mean_outside)
    outside = evaluate_outside_side(
        hx,
        provider=outside_provider,
        mass_flow=mass_flow_outside,
        T_in=T_in_outside,
        T_out=T_out_outside,
        p=p_outside,
        euler_provider=euler_provider,
        properties_mean=outside_props,
    )

    zones: list[WaterEvaporatorZoneResult] = []
    warnings: list[ModelWarning] = list(outside.warnings)
    for spec in _partition_enthalpy(inlet_state.h, h_out, saturation):
        zone = _evaluate_zone(
            hx,
            spec=spec,
            saturation=saturation,
            mass_flow_water=mass_flow_water,
            alpha_outside=outside.alpha_corrected,
            T_mean_outside=T_mean_outside,
            orientation=orientation,
            cache=cache,
            heat_flux_max_iterations=heat_flux_max_iterations,
            heat_flux_relative_tolerance=heat_flux_relative_tolerance,
        )
        zones.append(zone)
        warnings.extend(zone.warnings)
    return _TrialResult(
        h_out=h_out,
        T_out_outside=T_out_outside,
        zones=tuple(zones),
        required_area=sum(zone.area for zone in zones),
        UA_total=sum(zone.UA for zone in zones),
        outside=outside,
        mass_flux=mass_flow_water / hx.bundle.internal_flow_area_per_pass,
        warnings=tuple(_deduplicate(warnings)),
    )


def _partition_enthalpy(
    h_in: float,
    h_out: float,
    saturation: WaterSteamSaturationProperties,
) -> tuple[_ZoneSpec, ...]:
    if h_out <= h_in:
        raise WaterCondensationRequiredError("Reverse enthalpy direction requires condensation.")
    specs: list[_ZoneSpec] = []
    cursor = h_in
    if cursor < saturation.hf and h_out > cursor:
        end = min(h_out, saturation.hf)
        if end > cursor:
            specs.append(_ZoneSpec(WaterEvaporatorZoneKind.PREHEAT, cursor, end))
            cursor = end
    if cursor < saturation.hg and h_out > cursor:
        end = min(h_out, saturation.hg)
        if end > cursor:
            specs.append(_ZoneSpec(WaterEvaporatorZoneKind.EVAPORATION, cursor, end))
            cursor = end
    if h_out > cursor:
        specs.append(_ZoneSpec(WaterEvaporatorZoneKind.SUPERHEAT, cursor, h_out))
    return tuple(specs)


def _evaluate_zone(
    hx,
    *,
    spec: _ZoneSpec,
    saturation: WaterSteamSaturationProperties,
    mass_flow_water: float,
    alpha_outside: float,
    T_mean_outside: float,
    orientation: TubeOrientation | None,
    cache: _SolveCache,
    heat_flux_max_iterations: int,
    heat_flux_relative_tolerance: float,
) -> WaterEvaporatorZoneResult:
    state_in = cache.water_state(spec.h_in)
    state_out = cache.water_state(spec.h_out)
    Q = mass_flow_water * (spec.h_out - spec.h_in)
    warnings: list[ModelWarning] = []
    evaporation = None
    quality_in = quality_out = None
    q_inner = q_outer = None
    heat_flux_converged = True
    heat_flux_iterations = 0
    heat_flux_residual = 0.0
    T_in, T_out = state_in.T, state_out.T

    if spec.kind is WaterEvaporatorZoneKind.EVAPORATION:
        boiling_orientation = _require_boiling_orientation(orientation)
        quality_in = (spec.h_in - saturation.hf) / saturation.hfg
        quality_out = (spec.h_out - saturation.hf) / saturation.hfg
        delta_T = T_mean_outside - saturation.Tsat
        if delta_T <= 0.0 or not math.isfinite(delta_T):
            return _infinite_zone(
                spec=spec, state_in=state_in, state_out=state_out, Q=Q,
                alpha_outside=alpha_outside, quality_in=quality_in,
                quality_out=quality_out,
            )
        (
            evaporation,
            U,
            area,
            q_inner,
            q_outer,
            heat_flux_converged,
            heat_flux_iterations,
            heat_flux_residual,
        ) = _solve_boiling_heat_flux(
            hx,
            saturation=saturation,
            mass_flow_water=mass_flow_water,
            quality_in=quality_in,
            quality_out=quality_out,
            orientation=boiling_orientation,
            alpha_outside=alpha_outside,
            delta_T=delta_T,
            Q=Q,
            max_iterations=heat_flux_max_iterations,
            relative_tolerance=heat_flux_relative_tolerance,
        )
        alpha_inside = evaporation.zone_alpha_evaporation
        T_in = T_out = saturation.Tsat
        warnings.extend(evaporation.warnings)
        if not heat_flux_converged:
            warnings.append(
                make_warning(
                    code="WATER_EVAPORATION_HEAT_FLUX_NOT_CONVERGED",
                    message=(
                        "The self-consistent inside-area boiling heat flux did "
                        f"not converge; relative residual={heat_flux_residual:.6g}."
                    ),
                    source="water_evaporator",
                    severity="warning",
                )
            )
    else:
        midpoint = cache.water_state(0.5 * (spec.h_in + spec.h_out))
        if midpoint.transport is None:
            raise ValueError("Single-phase water zone resolved to a two-phase midpoint.")
        diagnostics = heat_transfer_coefficient_internal_diagnostics(
            m_dot=mass_flow_water,
            tube_inner_diameter=hx.bundle.internal_hydraulic_diameter,
            flow_area=hx.bundle.internal_flow_area_per_pass,
            props=to_internal_fluid_props(midpoint.transport),
            T_bulk=midpoint.T,
            L_heated=float(hx.bundle.tube.length_effective),
        )
        alpha_inside = diagnostics.alfa_corrected
        warnings.extend(diagnostics.warnings)
        U = overall_u_outer_basis(
            alpha_inside=alpha_inside,
            alpha_outside=alpha_outside,
            D_i=float(hx.bundle.tube.D_i),
            D_o=float(hx.bundle.tube.D_o),
            wall_k=float(hx.bundle.tube.wall_k),
        )
        delta_T = T_mean_outside - 0.5 * (T_in + T_out)
        if delta_T <= 0.0 or not math.isfinite(delta_T):
            area = math.inf
        else:
            area = Q / (U * delta_T)

    UA = U * area if math.isfinite(area) else math.inf
    return WaterEvaporatorZoneResult(
        kind=spec.kind,
        state_in=state_in,
        state_out=state_out,
        h_in=spec.h_in,
        h_out=spec.h_out,
        T_in=T_in,
        T_out=T_out,
        Q=Q,
        alpha_inside=alpha_inside,
        alpha_outside=alpha_outside,
        U=U,
        area=area,
        UA=UA,
        heat_flux_inner=q_inner,
        heat_flux_outer=q_outer,
        heat_flux_converged=heat_flux_converged,
        heat_flux_iterations=heat_flux_iterations,
        heat_flux_residual=heat_flux_residual,
        quality_in=quality_in,
        quality_out=quality_out,
        evaporation=evaporation,
        warnings=tuple(_deduplicate(warnings)),
    )


def _solve_boiling_heat_flux(
    hx,
    *,
    saturation: WaterSteamSaturationProperties,
    mass_flow_water: float,
    quality_in: float,
    quality_out: float,
    orientation: TubeOrientation,
    alpha_outside: float,
    delta_T: float,
    Q: float,
    max_iterations: int,
    relative_tolerance: float,
):
    if max_iterations <= 0 or relative_tolerance <= 0.0:
        raise ValueError("Heat-flux iteration controls must be positive.")
    tube = hx.bundle.tube
    D_i, D_o, wall_k = float(tube.D_i), float(tube.D_o), float(tube.wall_k)
    wall_R, outside_R = fixed_outer_basis_resistances(
        alpha_outside=alpha_outside, D_i=D_i, D_o=D_o, wall_k=wall_k
    )
    # With infinite inside HTC, wall+outside resistance gives a strict
    # physical upper bound on inner-area heat flux. This brackets the
    # fixed point without freezing an arbitrary engineering q'' value.
    q_high = delta_T * (D_o / D_i) / (wall_R + outside_R)
    q_low = max(1e-12, q_high * 1e-12)
    final = None
    converged = False
    residual = math.inf
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        q_mid = 0.5 * (q_low + q_high)
        evaporation = solve_water_evaporation_zone(
            p=saturation.p,
            mass_flow_total=mass_flow_water,
            flow_area_per_pass=hx.bundle.internal_flow_area_per_pass,
            tube_inner_diameter=hx.bundle.internal_hydraulic_diameter,
            quality_in=quality_in,
            quality_out=quality_out,
            heat_flux_inner=q_mid,
            orientation=orientation,
            saturation=saturation,
        )
        U = overall_u_outer_basis(
            alpha_inside=evaporation.zone_alpha_evaporation,
            alpha_outside=alpha_outside,
            D_i=D_i,
            D_o=D_o,
            wall_k=wall_k,
        )
        target = U * delta_T * D_o / D_i
        residual = (q_mid - target) / max(q_mid, target)
        final = (evaporation, U, q_mid, target)
        if abs(residual) <= relative_tolerance:
            converged = True
            break
        if residual > 0.0:
            q_high = q_mid
        else:
            q_low = q_mid
    if final is None:
        raise ValueError("Boiling heat-flux solve did not produce a finite iterate.")
    evaporation, U, q_inner, q_target = final
    q_outer = U * delta_T
    area = Q / q_outer
    return (
        evaporation, U, area, q_inner, q_outer, converged, iterations, abs(residual)
    )


def _infinite_zone(
    *,
    spec: _ZoneSpec,
    state_in: WaterSteamProperties,
    state_out: WaterSteamProperties,
    Q: float,
    alpha_outside: float,
    quality_in: float,
    quality_out: float,
) -> WaterEvaporatorZoneResult:
    return WaterEvaporatorZoneResult(
        kind=spec.kind,
        state_in=state_in,
        state_out=state_out,
        h_in=spec.h_in,
        h_out=spec.h_out,
        T_in=state_in.T,
        T_out=state_out.T,
        Q=Q,
        alpha_inside=math.inf,
        alpha_outside=alpha_outside,
        U=math.inf,
        area=math.inf,
        UA=math.inf,
        heat_flux_converged=False,
        heat_flux_residual=math.inf,
        quality_in=quality_in,
        quality_out=quality_out,
    )


def _outside_outlet_temperature(
    *,
    Q_total: float,
    mass_flow_outside: float,
    T_in_outside: float,
    cache: _SolveCache,
) -> float:
    T_out = T_in_outside - Q_total / (
        mass_flow_outside * cache.outside_state(T_in_outside).cp
    )
    for _ in range(12):
        T_mean = 0.5 * (T_in_outside + T_out)
        if T_mean <= 0.0:
            return -math.inf
        props = cache.outside_state(T_mean)
        updated = T_in_outside - Q_total / (mass_flow_outside * props.cp)
        if abs(updated - T_out) < 1e-9:
            return updated
        T_out = updated
    return T_out


def _build_solution(
    *,
    hx,
    mode: str,
    inlet_state: WaterSteamProperties,
    saturation: WaterSteamSaturationProperties,
    mass_flow_water: float,
    trial: _TrialResult,
    cache: _SolveCache,
    converged: bool,
    iterations: int,
    root_iterations: int,
    runtime_s: float,
) -> WaterEvaporatorSolution:
    state_out = cache.water_state(trial.h_out)
    state_midpoint = cache.water_state(0.5 * (inlet_state.h + state_out.h))
    zone_map = {zone.kind: zone for zone in trial.zones}

    def value(kind: WaterEvaporatorZoneKind, attribute: str, default=0.0):
        zone = zone_map.get(kind)
        return default if zone is None else getattr(zone, attribute)

    if not trial.zones:
        raise ValueError("Water evaporator produced no positive-duty zones.")
    A_total = sum(zone.area for zone in trial.zones)
    Q_total = sum(zone.Q for zone in trial.zones)
    UA_total = sum(zone.U * zone.area for zone in trial.zones)
    if not all(
        math.isfinite(value) and value > 0.0
        for zone in trial.zones
        for value in (zone.Q, zone.area, zone.U, zone.UA, zone.alpha_inside)
    ):
        raise ValueError("Water evaporator accepted a non-finite/non-positive zone.")
    if not math.isclose(
        Q_total,
        mass_flow_water * (state_out.h - inlet_state.h),
        rel_tol=2e-10,
        abs_tol=1e-5,
    ):
        raise ValueError("Water-evaporator zone energy balance is inconsistent.")
    if not math.isclose(UA_total, trial.UA_total, rel_tol=2e-12, abs_tol=1e-9):
        raise ValueError("Water-evaporator zone-UA aggregation is inconsistent.")

    alpha_area_weighted = sum(
        zone.alpha_inside * zone.area for zone in trial.zones
    ) / A_total
    U_equivalent = UA_total / A_total
    alpha_equivalent = equivalent_inside_alpha_outer_basis(
        U_equivalent=U_equivalent,
        alpha_outside=trial.outside.alpha_corrected,
        D_i=float(hx.bundle.tube.D_i),
        D_o=float(hx.bundle.tube.D_o),
        wall_k=float(hx.bundle.tube.wall_k),
    )
    evaporation_zone = zone_map.get(WaterEvaporatorZoneKind.EVAPORATION)
    heat_flux_converged = all(zone.heat_flux_converged for zone in trial.zones)
    overall_converged = converged and heat_flux_converged
    warnings = list(trial.warnings)
    warnings.append(
        make_warning(
            code="WATER_EVAPORATOR_ZONE_ALLOCATION_0D_ESTIMATE",
            message=(
                "Water evaporation phase-front areas are a 0D allocation "
                "using one current opposing-stream mean temperature."
            ),
            source="water_evaporator",
            severity="info",
        )
    )
    if not converged:
        warnings.append(
            make_warning(
                code="WATER_EVAPORATOR_NOT_CONVERGED",
                message="Water-evaporator area allocation did not meet its tolerance.",
                source="water_evaporator",
                severity="warning",
            )
        )
    m_dot_evaporated = (
        0.0
        if evaporation_zone is None
        else mass_flow_water
        * (evaporation_zone.quality_out - evaporation_zone.quality_in)
    )
    has_two_phase = evaporation_zone is not None
    return WaterEvaporatorSolution(
        mode=mode,
        state_in=inlet_state,
        state_midpoint=state_midpoint,
        state_out=state_out,
        saturation=saturation,
        mass_flow_water=mass_flow_water,
        mass_flux=trial.mass_flux,
        T_out_outside=trial.T_out_outside,
        Q_preheat=value(WaterEvaporatorZoneKind.PREHEAT, "Q"),
        Q_evaporation=value(WaterEvaporatorZoneKind.EVAPORATION, "Q"),
        Q_superheat=value(WaterEvaporatorZoneKind.SUPERHEAT, "Q"),
        Q_total=Q_total,
        A_preheat=value(WaterEvaporatorZoneKind.PREHEAT, "area"),
        A_evaporation=value(WaterEvaporatorZoneKind.EVAPORATION, "area"),
        A_superheat=value(WaterEvaporatorZoneKind.SUPERHEAT, "area"),
        A_total=A_total,
        UA_preheat=value(WaterEvaporatorZoneKind.PREHEAT, "UA"),
        UA_evaporation=value(WaterEvaporatorZoneKind.EVAPORATION, "UA"),
        UA_superheat=value(WaterEvaporatorZoneKind.SUPERHEAT, "UA"),
        UA_total=UA_total,
        zone_alpha_preheat=value(WaterEvaporatorZoneKind.PREHEAT, "alpha_inside", None),
        zone_alpha_evaporation=value(WaterEvaporatorZoneKind.EVAPORATION, "alpha_inside", None),
        zone_alpha_superheat=value(WaterEvaporatorZoneKind.SUPERHEAT, "alpha_inside", None),
        zone_U_preheat=value(WaterEvaporatorZoneKind.PREHEAT, "U", None),
        zone_U_evaporation=value(WaterEvaporatorZoneKind.EVAPORATION, "U", None),
        zone_U_superheat=value(WaterEvaporatorZoneKind.SUPERHEAT, "U", None),
        inside_alpha_equivalent=alpha_equivalent,
        inside_alpha_area_weighted=alpha_area_weighted,
        outside_alpha=trial.outside.alpha_corrected,
        outside_props_mean=trial.outside.properties_mean,
        outside_evaluation=trial.outside,
        U_equivalent=U_equivalent,
        m_dot_evaporated=m_dot_evaporated,
        heat_flux_inner_evaporation=(
            None if evaporation_zone is None else evaporation_zone.heat_flux_inner
        ),
        heat_flux_outer_evaporation=(
            None if evaporation_zone is None else evaporation_zone.heat_flux_outer
        ),
        heat_flux_converged=heat_flux_converged,
        heat_flux_iterations=sum(zone.heat_flux_iterations for zone in trial.zones),
        heat_flux_residual=max(zone.heat_flux_residual for zone in trial.zones),
        zones=trial.zones,
        converged=overall_converged,
        iterations=iterations,
        root_iterations=root_iterations,
        property_evaluations=cache.property_evaluations,
        cache_hits=cache.cache_hits,
        runtime_s=runtime_s,
        two_phase_pressure_drop_supported=not has_two_phase,
        two_phase_pressure_drop_status=(
            "not_supported" if has_two_phase else "not_applicable_single_phase"
        ),
        warnings=tuple(_deduplicate(warnings)),
        assumptions=(
            "constant_nominal_water_pressure",
            "shared_opposing_stream_0d_mean_temperature",
            "inside_area_self_consistent_boiling_heat_flux",
            "zone_UA_sum_is_authoritative",
        ) + (
            ("two_phase_pressure_drop_not_supported_when_evaporation_active",)
            if has_two_phase
            else ()
        ),
    )


def _require_boiling_orientation(
    orientation: TubeOrientation | None,
) -> TubeOrientation:
    if orientation is None:
        raise ValueError(
            "Active pure-water flow boiling requires explicit tube_orientation "
            "on BareTube; Shah 1982 is orientation-dependent."
        )
    if orientation not in {
        TubeOrientation.HORIZONTAL,
        TubeOrientation.VERTICAL_UPWARD,
    }:
        raise ValueError(
            "Active Shah 1982 boiling supports horizontal or vertical-upward "
            "tube_orientation only."
        )
    return orientation


def _validate_common_inputs(
    *,
    inlet_state: WaterSteamProperties,
    mass_flow_water: float,
    mass_flow_outside: float,
    T_in_outside: float,
    p_outside: float,
    orientation: TubeOrientation | None,
) -> None:
    if not isinstance(inlet_state, WaterSteamProperties):
        raise ValueError("inlet_state must be a resolved WaterSteamProperties state.")
    for name, value in (
        ("mass_flow_water", mass_flow_water),
        ("mass_flow_outside", mass_flow_outside),
        ("T_in_outside", T_in_outside),
        ("p_outside", p_outside),
    ):
        _positive_finite(value, name)
    if orientation is not None and not isinstance(orientation, TubeOrientation):
        raise ValueError("orientation must be a TubeOrientation value or None.")


def _deduplicate(warnings) -> list[ModelWarning]:
    result: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        key = (warning.code, warning.message)
        if key not in seen:
            seen.add(key)
            result.append(warning)
    return result


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
