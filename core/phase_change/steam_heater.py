# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Shared 0D multi-zone solver for a pure-water/steam tube side.

The steam coordinate is specific enthalpy at constant nominal pressure.
Crossing ``hg`` and ``hf`` creates at most three ordered zones:
desuperheating, condensation, and condensate subcooling. Each zone owns its
inside coefficient, overall U, required area, and UA. The authoritative
whole-exchanger conductance is always ``sum(U_zone * A_zone)``.

This remains a lumped 0D allocation. The opposing stream uses one overall
temperature program and a shared representative bulk temperature; it is not
artificially marched in series through phase zones for crossflow geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from time import perf_counter

from core.common.warnings import ModelWarning, make_warning
from core.heat_transfer.internal_flow import heat_transfer_coefficient_internal_diagnostics
from core.heat_transfer.outside_flow import outside_flow_from_mass_flow
from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props
from core.properties.common import FluidTransportProperties
from core.properties.water import (
    WaterSteamPhase,
    WaterSteamProperties,
    WaterSteamSaturationProperties,
    water_saturation_snapshot,
    water_steam_props_iapws97,
)
from core.phase_change.steam_condensation import (
    SteamCondensationZoneResult,
    SteamTubeOrientation,
    solve_steam_condensation_zone,
)


class SteamHeaterZoneKind(str, Enum):
    SUPERHEAT = "superheat"
    CONDENSATION = "condensation"
    SUBCOOLING = "subcooling"


class SteamEvaporationNotSupportedError(RuntimeError):
    """Raised when the requested enthalpy direction requires boiling."""

    warning_code = "STEAM_EVAPORATION_NOT_SUPPORTED"


@dataclass(frozen=True)
class SteamHeaterZoneResult:
    kind: SteamHeaterZoneKind
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
    quality_in: float | None = None
    quality_out: float | None = None
    condensation: SteamCondensationZoneResult | None = None
    warnings: tuple[ModelWarning, ...] = ()


@dataclass(frozen=True)
class SteamHeaterSolution:
    mode: str
    state_in: WaterSteamProperties
    state_out: WaterSteamProperties
    saturation: WaterSteamSaturationProperties
    mass_flow_steam: float
    mass_flux: float
    T_out_outside: float
    Q_desuperheat: float
    Q_condensation: float
    Q_subcooling: float
    Q_total: float
    A_desuperheat: float
    A_condensation: float
    A_subcooling: float
    A_total: float
    UA_desuperheat: float
    UA_condensation: float
    UA_subcooling: float
    UA_total: float
    zone_alpha_desuperheat: float | None
    zone_alpha_condensation: float | None
    zone_alpha_subcooling: float | None
    inside_alfa_mean: float
    outside_alpha: float
    U_equivalent: float
    zones: tuple[SteamHeaterZoneResult, ...]
    converged: bool
    iterations: int
    root_iterations: int
    property_evaluations: int
    runtime_s: float
    two_phase_pressure_drop_supported: bool
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
    def zone_fraction_desuperheat(self) -> float:
        return self.A_desuperheat / self.A_total if self.A_total > 0.0 else 0.0

    @property
    def zone_fraction_condensation(self) -> float:
        return self.A_condensation / self.A_total if self.A_total > 0.0 else 0.0

    @property
    def zone_fraction_subcooling(self) -> float:
        return self.A_subcooling / self.A_total if self.A_total > 0.0 else 0.0


@dataclass(frozen=True)
class _ZoneSpec:
    kind: SteamHeaterZoneKind
    h_in: float
    h_out: float


@dataclass(frozen=True)
class _TrialResult:
    h_out: float
    T_out_outside: float
    zones: tuple[SteamHeaterZoneResult, ...]
    required_area: float
    UA_total: float
    outside_alpha: float
    mass_flux: float
    warnings: tuple[ModelWarning, ...]


class _SolveCache:
    def __init__(self, p_steam: float, outside_provider, p_outside: float):
        self.p_steam = p_steam
        self.outside_provider = outside_provider
        self.p_outside = p_outside
        self.steam_states: dict[float, WaterSteamProperties] = {}
        self.outside_props: dict[float, FluidTransportProperties] = {}
        self.property_evaluations = 0

    def steam_state(self, h: float) -> WaterSteamProperties:
        if h not in self.steam_states:
            self.steam_states[h] = water_steam_props_iapws97(p=self.p_steam, h=h)
            self.property_evaluations += 1
        return self.steam_states[h]

    def outside_state(self, T: float) -> FluidTransportProperties:
        if T not in self.outside_props:
            raw = self.outside_provider.at(T=T, p=self.p_outside)
            transport = getattr(raw, "transport", raw)
            if not isinstance(transport, FluidTransportProperties):
                transport = FluidTransportProperties(
                    rho=float(transport.rho), mu=float(transport.mu),
                    k=float(transport.k), cp=float(transport.cp),
                )
            self.outside_props[T] = transport
            self.property_evaluations += 1
        return self.outside_props[T]


def solve_steam_heater(
    hx,
    *,
    inlet_state: WaterSteamProperties,
    mass_flow_steam: float,
    outside_provider,
    mass_flow_outside: float,
    T_in_outside: float,
    p_outside: float,
    orientation: SteamTubeOrientation,
    max_iterations: int = 80,
    relative_area_tolerance: float = 1.0e-8,
) -> SteamHeaterSolution:
    """Simulation: solve outlet enthalpy whose zone areas fill the geometry."""
    started = perf_counter()
    _validate_common_inputs(
        inlet_state=inlet_state, mass_flow_steam=mass_flow_steam,
        mass_flow_outside=mass_flow_outside, T_in_outside=T_in_outside,
        p_outside=p_outside, orientation=orientation,
    )
    if max_iterations <= 0 or relative_area_tolerance <= 0.0:
        raise ValueError("Steam-heater iteration controls must be positive.")
    if inlet_state.T <= T_in_outside:
        raise SteamEvaporationNotSupportedError(
            "The pure-water tube side is not hotter than the opposing inlet; "
            "the requested direction would require heating/boiling."
        )

    saturation = water_saturation_snapshot(inlet_state.p)
    cache = _SolveCache(inlet_state.p, outside_provider, p_outside)
    available_area = hx.bundle.total_outer_area

    # A lower enthalpy bound based on liquid water just above the opposing
    # inlet temperature deliberately brackets the thermal pinch. The final
    # accepted solution is always rebuilt at a finite positive driving force.
    T_lower = max(273.17, T_in_outside + 1.0e-3)
    if math.isclose(T_lower, saturation.Tsat, rel_tol=0.0, abs_tol=1.0e-6):
        T_lower += 1.0e-3
    lower_state = water_steam_props_iapws97(T=T_lower, p=inlet_state.p)
    if lower_state.h >= inlet_state.h:
        raise SteamEvaporationNotSupportedError(
            "No positive steam-side cooling range exists above the opposing inlet."
        )
    q_high = mass_flow_steam * (inlet_state.h - lower_state.h) * (1.0 - 1.0e-10)
    q_low = max(1.0e-9, q_high * 1.0e-12)

    def trial(q: float) -> _TrialResult:
        return _evaluate_duty(
            hx,
            inlet_state=inlet_state,
            saturation=saturation,
            mass_flow_steam=mass_flow_steam,
            outside_provider=outside_provider,
            mass_flow_outside=mass_flow_outside,
            T_in_outside=T_in_outside,
            p_outside=p_outside,
            orientation=orientation,
            Q_total=q,
            cache=cache,
        )

    high = trial(q_high)
    if math.isfinite(high.required_area) and high.required_area < available_area:
        raise ValueError(
            "Steam-heater area root was not bracketed before the thermal pinch."
        )

    final_trial: _TrialResult | None = None
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        q_mid = 0.5 * (q_low + q_high)
        current = trial(q_mid)
        residual = current.required_area - available_area
        if math.isfinite(residual) and abs(residual) <= relative_area_tolerance * available_area:
            final_trial = current
            converged = True
            break
        if not math.isfinite(current.required_area) or residual > 0.0:
            q_high = q_mid
        else:
            q_low = q_mid
        final_trial = current

    if final_trial is None or not converged:
        final_trial = trial(0.5 * (q_low + q_high))

    return _build_solution(
        mode="simulation",
        inlet_state=inlet_state,
        saturation=saturation,
        mass_flow_steam=mass_flow_steam,
        trial=final_trial,
        cache=cache,
        converged=converged,
        iterations=iterations,
        root_iterations=iterations,
        runtime_s=perf_counter() - started,
    )


def rate_steam_heater(
    hx,
    *,
    inlet_state: WaterSteamProperties,
    mass_flow_steam: float,
    outside_provider,
    mass_flow_outside: float,
    T_in_outside: float,
    p_outside: float,
    orientation: SteamTubeOrientation,
    outlet_state: WaterSteamProperties | None = None,
    Q_total: float | None = None,
) -> SteamHeaterSolution:
    """Rating: calculate required zone areas for a specified outlet or duty."""
    started = perf_counter()
    _validate_common_inputs(
        inlet_state=inlet_state, mass_flow_steam=mass_flow_steam,
        mass_flow_outside=mass_flow_outside, T_in_outside=T_in_outside,
        p_outside=p_outside, orientation=orientation,
    )
    if (outlet_state is None) == (Q_total is None):
        raise ValueError("Provide exactly one of outlet_state or Q_total for steam Rating.")
    if outlet_state is not None:
        if not math.isclose(outlet_state.p, inlet_state.p, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("Steam inlet and outlet must use the same nominal pressure.")
        Q_total = mass_flow_steam * (inlet_state.h - outlet_state.h)
    if Q_total is None or not math.isfinite(Q_total) or Q_total <= 0.0:
        raise SteamEvaporationNotSupportedError(
            "Steam Rating requires positive heat removal; boiling/evaporation is unsupported."
        )

    saturation = water_saturation_snapshot(inlet_state.p)
    cache = _SolveCache(inlet_state.p, outside_provider, p_outside)
    trial = _evaluate_duty(
        hx,
        inlet_state=inlet_state,
        saturation=saturation,
        mass_flow_steam=mass_flow_steam,
        outside_provider=outside_provider,
        mass_flow_outside=mass_flow_outside,
        T_in_outside=T_in_outside,
        p_outside=p_outside,
        orientation=orientation,
        Q_total=Q_total,
        cache=cache,
    )
    if not math.isfinite(trial.required_area):
        raise ValueError("Specified steam Rating violates a positive zone temperature difference.")
    return _build_solution(
        mode="rating",
        inlet_state=inlet_state,
        saturation=saturation,
        mass_flow_steam=mass_flow_steam,
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
    mass_flow_steam: float,
    outside_provider,
    mass_flow_outside: float,
    T_in_outside: float,
    p_outside: float,
    orientation: SteamTubeOrientation,
    Q_total: float,
    cache: _SolveCache,
) -> _TrialResult:
    h_out = inlet_state.h - Q_total / mass_flow_steam
    boundary_tolerance = max(
        1.0e-3, 1.0e-9 * max(abs(saturation.hf), abs(saturation.hg))
    )
    if abs(h_out - saturation.hf) <= boundary_tolerance:
        h_out = saturation.hf
    elif abs(h_out - saturation.hg) <= boundary_tolerance:
        h_out = saturation.hg
    if h_out >= inlet_state.h:
        raise SteamEvaporationNotSupportedError("Steam enthalpy must decrease during cooling/condensation.")

    T_out_outside = _outside_outlet_temperature(
        Q_total=Q_total,
        mass_flow_outside=mass_flow_outside,
        T_in_outside=T_in_outside,
        cache=cache,
    )
    T_mean_outside = 0.5 * (T_in_outside + T_out_outside)
    outside_props = cache.outside_state(T_mean_outside)
    tube = hx.bundle.tube
    _, _, _, alpha_outside, _, outside_warnings, _ = outside_flow_from_mass_flow(
        m_dot=mass_flow_outside,
        frontal_area=hx.bundle.frontal_flow_area,
        tube_outer_diameter=float(tube.D_o),
        tube_pitch_transverse=hx.bundle.pitch_transverse,
        tube_pitch_longitudinal=hx.bundle.pitch_longitudinal,
        layout=hx.bundle.layout,
        n_rows=hx.bundle.n_rows,
        n_tubes_per_row=hx.bundle.n_tubes_per_row,
        props=to_outside_fluid_props(outside_props),
        calculate_pressure_drop=False,
    )

    zones: list[SteamHeaterZoneResult] = []
    trial_warnings: list[ModelWarning] = list(outside_warnings)
    for spec in _partition_enthalpy(inlet_state.h, h_out, saturation):
        zone = _evaluate_zone(
            hx,
            spec=spec,
            saturation=saturation,
            mass_flow_steam=mass_flow_steam,
            alpha_outside=alpha_outside,
            T_mean_outside=T_mean_outside,
            orientation=orientation,
            cache=cache,
        )
        zones.append(zone)
        trial_warnings.extend(zone.warnings)

    required_area = sum(zone.area for zone in zones)
    UA_total = sum(zone.UA for zone in zones)
    return _TrialResult(
        h_out=h_out,
        T_out_outside=T_out_outside,
        zones=tuple(zones),
        required_area=required_area,
        UA_total=UA_total,
        outside_alpha=alpha_outside,
        mass_flux=mass_flow_steam / hx.bundle.internal_flow_area_per_pass,
        warnings=tuple(_deduplicate_warnings(trial_warnings)),
    )


def _partition_enthalpy(
    h_in: float,
    h_out: float,
    saturation: WaterSteamSaturationProperties,
) -> tuple[_ZoneSpec, ...]:
    if h_out >= h_in:
        raise SteamEvaporationNotSupportedError("Reverse enthalpy direction requires boiling/evaporation.")
    specs: list[_ZoneSpec] = []
    cursor = h_in
    if cursor > saturation.hg and h_out < cursor:
        end = max(h_out, saturation.hg)
        if cursor > end:
            specs.append(_ZoneSpec(SteamHeaterZoneKind.SUPERHEAT, cursor, end))
            cursor = end
    if cursor > saturation.hf and h_out < cursor:
        end = max(h_out, saturation.hf)
        if cursor > end:
            specs.append(_ZoneSpec(SteamHeaterZoneKind.CONDENSATION, cursor, end))
            cursor = end
    if h_out < cursor:
        specs.append(_ZoneSpec(SteamHeaterZoneKind.SUBCOOLING, cursor, h_out))
    return tuple(specs)


def _evaluate_zone(
    hx,
    *,
    spec: _ZoneSpec,
    saturation: WaterSteamSaturationProperties,
    mass_flow_steam: float,
    alpha_outside: float,
    T_mean_outside: float,
    orientation: SteamTubeOrientation,
    cache: _SolveCache,
) -> SteamHeaterZoneResult:
    Q = mass_flow_steam * (spec.h_in - spec.h_out)
    warnings: list[ModelWarning] = []
    quality_in = quality_out = None
    condensation = None
    if spec.kind is SteamHeaterZoneKind.CONDENSATION:
        quality_in = (spec.h_in - saturation.hf) / saturation.hfg
        quality_out = (spec.h_out - saturation.hf) / saturation.hfg
        condensation = solve_steam_condensation_zone(
            p=saturation.p,
            mass_flow_total=mass_flow_steam,
            flow_area_per_pass=hx.bundle.internal_flow_area_per_pass,
            tube_inner_diameter=hx.bundle.internal_hydraulic_diameter,
            quality_in=quality_in,
            quality_out=quality_out,
            orientation=orientation,
            saturation=saturation,
        )
        alpha_inside = condensation.zone_alpha_condensation
        T_in = T_out = saturation.Tsat
        warnings.extend(condensation.warnings)
    else:
        midpoint_state = cache.steam_state(0.5 * (spec.h_in + spec.h_out))
        if midpoint_state.transport is None:
            raise ValueError("Single-phase steam zone resolved to a two-phase midpoint.")
        diagnostics = heat_transfer_coefficient_internal_diagnostics(
            m_dot=mass_flow_steam,
            tube_inner_diameter=hx.bundle.internal_hydraulic_diameter,
            flow_area=hx.bundle.internal_flow_area_per_pass,
            props=to_internal_fluid_props(midpoint_state.transport),
            T_bulk=midpoint_state.T,
            L_heated=float(hx.bundle.tube.length_effective),
        )
        alpha_inside = diagnostics.alfa_corrected
        T_in = cache.steam_state(spec.h_in).T
        T_out = cache.steam_state(spec.h_out).T
        warnings.extend(diagnostics.warnings)

    U = _overall_u_outer_basis(
        alpha_inside=alpha_inside,
        alpha_outside=alpha_outside,
        D_i=float(hx.bundle.tube.D_i),
        D_o=float(hx.bundle.tube.D_o),
        wall_k=float(hx.bundle.tube.wall_k),
    )
    delta_T = 0.5 * (T_in + T_out) - T_mean_outside
    if delta_T <= 0.0 or not math.isfinite(delta_T):
        area = math.inf
        UA = math.inf
    else:
        area = Q / (U * delta_T)
        UA = U * area
    return SteamHeaterZoneResult(
        kind=spec.kind,
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
        quality_in=quality_in,
        quality_out=quality_out,
        condensation=condensation,
        warnings=tuple(_deduplicate_warnings(warnings)),
    )


def _outside_outlet_temperature(
    *,
    Q_total: float,
    mass_flow_outside: float,
    T_in_outside: float,
    cache: _SolveCache,
) -> float:
    T_out = T_in_outside + Q_total / (
        mass_flow_outside * cache.outside_state(T_in_outside).cp
    )
    for _ in range(8):
        props = cache.outside_state(0.5 * (T_in_outside + T_out))
        updated = T_in_outside + Q_total / (mass_flow_outside * props.cp)
        if abs(updated - T_out) < 1.0e-9:
            return updated
        T_out = updated
    return T_out


def _overall_u_outer_basis(
    *,
    alpha_inside: float,
    alpha_outside: float,
    D_i: float,
    D_o: float,
    wall_k: float,
) -> float:
    resistance_per_outer_area = (
        D_o / (D_i * alpha_inside)
        + D_o * math.log(D_o / D_i) / (2.0 * wall_k)
        + 1.0 / alpha_outside
    )
    return 1.0 / resistance_per_outer_area


def _build_solution(
    *,
    mode: str,
    inlet_state: WaterSteamProperties,
    saturation: WaterSteamSaturationProperties,
    mass_flow_steam: float,
    trial: _TrialResult,
    cache: _SolveCache,
    converged: bool,
    iterations: int,
    root_iterations: int,
    runtime_s: float,
) -> SteamHeaterSolution:
    state_out = cache.steam_state(trial.h_out)
    zone_map = {zone.kind: zone for zone in trial.zones}

    def value(kind: SteamHeaterZoneKind, attribute: str, default=0.0):
        zone = zone_map.get(kind)
        return default if zone is None else getattr(zone, attribute)

    A_total = sum(zone.area for zone in trial.zones)
    Q_total = sum(zone.Q for zone in trial.zones)
    UA_total = sum(zone.U * zone.area for zone in trial.zones)
    if not math.isclose(
        Q_total, mass_flow_steam * (inlet_state.h - state_out.h),
        rel_tol=2.0e-10, abs_tol=1.0e-5,
    ):
        raise ValueError("Steam-heater zone energy balance is internally inconsistent.")
    if not math.isclose(UA_total, trial.UA_total, rel_tol=2.0e-12, abs_tol=1.0e-9):
        raise ValueError("Steam-heater zone-UA aggregation is internally inconsistent.")

    inside_alfa_mean = sum(zone.alpha_inside * zone.area for zone in trial.zones) / A_total
    warnings = list(trial.warnings)
    warnings.append(
        make_warning(
            code="STEAM_HEATER_ZONE_ALLOCATION_0D_ESTIMATE",
            message=(
                "Steam phase-front areas are a 0D thermal allocation using a "
                "shared opposing-stream mean temperature, not axial front locations."
            ),
            source="steam_heater",
            severity="info",
        )
    )
    if not converged:
        warnings.append(
            make_warning(
                code="STEAM_HEATER_NOT_CONVERGED",
                message="Steam-heater surface allocation did not meet its area tolerance.",
                source="steam_heater",
                severity="warning",
            )
        )
    return SteamHeaterSolution(
        mode=mode,
        state_in=inlet_state,
        state_out=state_out,
        saturation=saturation,
        mass_flow_steam=mass_flow_steam,
        mass_flux=trial.mass_flux,
        T_out_outside=trial.T_out_outside,
        Q_desuperheat=value(SteamHeaterZoneKind.SUPERHEAT, "Q"),
        Q_condensation=value(SteamHeaterZoneKind.CONDENSATION, "Q"),
        Q_subcooling=value(SteamHeaterZoneKind.SUBCOOLING, "Q"),
        Q_total=Q_total,
        A_desuperheat=value(SteamHeaterZoneKind.SUPERHEAT, "area"),
        A_condensation=value(SteamHeaterZoneKind.CONDENSATION, "area"),
        A_subcooling=value(SteamHeaterZoneKind.SUBCOOLING, "area"),
        A_total=A_total,
        UA_desuperheat=value(SteamHeaterZoneKind.SUPERHEAT, "UA"),
        UA_condensation=value(SteamHeaterZoneKind.CONDENSATION, "UA"),
        UA_subcooling=value(SteamHeaterZoneKind.SUBCOOLING, "UA"),
        UA_total=UA_total,
        zone_alpha_desuperheat=value(SteamHeaterZoneKind.SUPERHEAT, "alpha_inside", None),
        zone_alpha_condensation=value(SteamHeaterZoneKind.CONDENSATION, "alpha_inside", None),
        zone_alpha_subcooling=value(SteamHeaterZoneKind.SUBCOOLING, "alpha_inside", None),
        inside_alfa_mean=inside_alfa_mean,
        outside_alpha=trial.outside_alpha,
        U_equivalent=UA_total / A_total,
        zones=trial.zones,
        converged=converged,
        iterations=iterations,
        root_iterations=root_iterations,
        property_evaluations=cache.property_evaluations,
        runtime_s=runtime_s,
        two_phase_pressure_drop_supported=False,
        warnings=tuple(_deduplicate_warnings(warnings)),
        assumptions=(
            "constant_nominal_steam_pressure",
            "shared_opposing_stream_0d_mean_temperature",
            "zone_UA_sum_is_authoritative",
            "two_phase_pressure_drop_not_supported",
        ),
    )


def _validate_common_inputs(
    *,
    inlet_state: WaterSteamProperties,
    mass_flow_steam: float,
    mass_flow_outside: float,
    T_in_outside: float,
    p_outside: float,
    orientation: SteamTubeOrientation,
) -> None:
    if not isinstance(inlet_state, WaterSteamProperties):
        raise ValueError("inlet_state must be a resolved WaterSteamProperties state.")
    for name, value in (
        ("mass_flow_steam", mass_flow_steam),
        ("mass_flow_outside", mass_flow_outside),
        ("T_in_outside", T_in_outside),
        ("p_outside", p_outside),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    if not isinstance(orientation, SteamTubeOrientation):
        raise ValueError("orientation must be an explicit SteamTubeOrientation value.")


def _deduplicate_warnings(warnings) -> list[ModelWarning]:
    result: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        key = (warning.code, warning.message)
        if key not in seen:
            seen.add(key)
            result.append(warning)
    return result
