# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Shared 0D multi-zone solver for a pure-water/steam tube side.

The steam coordinate is specific enthalpy at constant nominal pressure.
Crossing ``hg`` and ``hf`` creates at most three ordered tube-side zones:
desuperheating, condensation, and condensate subcooling. Each zone owns its
inside coefficient, overall U, required area, and UA. The authoritative
whole-exchanger conductance is always ``sum(U_zone * A_zone)``.

The tube-side thermodynamic states remain sequential. For the crossflow
steam-air-heater geometry, however, the outside stream is divided into
parallel branches in proportion to the tube-length/gross-area fraction
occupied by each zone. Every branch receives the global outside inlet state;
the branch outlets are mixed after the zone calculations. A bounded damped
fixed point makes each air-flow fraction consistent with its required-area
fraction while the whole-bank outside correlation preserves the unchanged
local face mass flux and film coefficient.

This remains a lumped multi-zone 0D allocation, not row-by-row or
longitudinal 1D resolution of a phase front.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from time import perf_counter

from core.common.warnings import ModelWarning, make_warning
from core.geometry.tube import TubeOrientation
from core.heat_transfer.internal_flow import heat_transfer_coefficient_internal_diagnostics
from core.heat_transfer.ntu import ntu_from_effectiveness
from core.heat_transfer.outside_dispatch import (
    DEFAULT_FINNED_DP_PROVIDER,
    DEFAULT_FINNED_HT_PROVIDER,
    calculate_resistance_network,
)
from core.heat_transfer.outside_side import OutsideSideEvaluation, evaluate_outside_side
from core.heat_transfer.tube_resistance import (
    equivalent_inside_alpha_outer_basis as _equivalent_inside_alpha_outer_basis,
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
from core.phase_change.steam_condensation import (
    SteamCondensationZoneResult,
    solve_steam_condensation_zone,
)


class SteamHeaterZoneKind(str, Enum):
    SUPERHEAT = "superheat"
    CONDENSATION = "condensation"
    SUBCOOLING = "subcooling"


class SteamEvaporationNotSupportedError(RuntimeError):
    """Raised when the requested enthalpy direction requires boiling."""

    warning_code = "STEAM_EVAPORATION_NOT_SUPPORTED"


class SteamZoneAllocationConvergenceError(RuntimeError):
    """Raised when the parallel air/area allocation fixed point fails."""

    def __init__(self, *, iterations: int, residual: float):
        self.iterations = iterations
        self.residual = residual
        super().__init__(
            "Steam-heater parallel zone allocation did not converge within "
            f"{iterations} iterations; max area-fraction residual={residual:.6g}."
        )


DRIVING_FORCE_ISOTHERMAL_LMTD = "isothermal_lmtd"
DRIVING_FORCE_EPSILON_NTU_COUNTERFLOW = "epsilon_ntu_counterflow"
DRIVING_FORCE_EPSILON_NTU_COCURRENTFLOW = "epsilon_ntu_cocurrentflow"
DRIVING_FORCE_EPSILON_NTU_CROSSFLOW = "epsilon_ntu_crossflow"

_EPSILON_NTU_DRIVING_FORCE_METHOD_BY_ARRANGEMENT = {
    "counterflow": DRIVING_FORCE_EPSILON_NTU_COUNTERFLOW,
    "cocurrentflow": DRIVING_FORCE_EPSILON_NTU_COCURRENTFLOW,
    "crossflow": DRIVING_FORCE_EPSILON_NTU_CROSSFLOW,
}

ZONE_ALLOCATION_METHOD_PARALLEL_BY_GEOMETRY = "parallel_by_geometry"
_ZONE_ALLOCATION_MAX_ITERATIONS = 80
_ZONE_ALLOCATION_FRACTION_TOLERANCE = 1.0e-8
_ZONE_ALLOCATION_DAMPING = 1.0
_ZONE_ALLOCATION_MIN_DAMPING = 0.125
_ZONE_ALLOCATION_STALL_RATIO = 0.95


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
    alpha_outside_physical: float
    U: float
    area: float
    UA: float
    outside_T_in: float
    outside_T_out: float
    delta_T_terminal_in: float
    delta_T_terminal_out: float
    effective_mean_temperature_difference: float
    driving_force_method: str
    area_fraction: float
    tube_length_fraction: float
    outside_mass_flow: float
    outside_mass_flow_fraction: float
    outside_frontal_area: float
    outside_face_mass_flux: float
    outside_velocity: float
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
    inside_alpha_equivalent: float
    inside_alpha_area_weighted: float
    outside_alpha: float
    outside_alpha_physical: float
    outside_props_mean: FluidTransportProperties
    U_equivalent: float
    zones: tuple[SteamHeaterZoneResult, ...]
    zone_allocation_method: str
    zone_allocation_iterations: int
    zone_allocation_converged: bool
    zone_allocation_residual: float
    sum_zone_area_fraction: float
    sum_zone_air_mass_flow: float
    mixed_outside_T_out: float
    Q_zone_sum: float
    mixed_air_energy_residual: float
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
    def inside_alfa_mean(self) -> float:
        """Historical spelling for the resistance-consistent equivalent HTC."""
        return self.inside_alpha_equivalent

    @property
    def EMTD(self) -> float:
        """Authoritative whole-exchanger driving force: ``Q_total / UA_total``.

        This is the single source of truth; it is never recomputed from
        arithmetic mean temperatures, a separate whole-exchanger LMTD, or a
        bare/finned epsilon-NTU shortcut.
        """
        return self.Q_total / self.UA_total

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
    mass_flow_outside: float
    zones: tuple[SteamHeaterZoneResult, ...]
    required_area: float
    UA_total: float
    outside_alpha: float
    outside_alpha_physical: float
    outside_props_mean: FluidTransportProperties
    outside_evaluation: OutsideSideEvaluation
    mass_flux: float
    zone_allocation_iterations: int
    zone_allocation_converged: bool
    zone_allocation_residual: float
    mixed_air_energy_residual: float
    warnings: tuple[ModelWarning, ...]


@dataclass(frozen=True)
class _ParallelZoneAllocation:
    zones: tuple[SteamHeaterZoneResult, ...]
    iterations: int
    converged: bool
    residual: float
    mixed_air_energy_residual: float


class _SolveCache:
    def __init__(self, p_steam: float, outside_provider, p_outside: float):
        self.p_steam = p_steam
        self.outside_provider = outside_provider
        self.p_outside = p_outside
        self.steam_states: dict[float, WaterSteamProperties] = {}
        self.outside_props: dict[float, FluidTransportProperties] = {}
        self.zone_fraction_history: dict[
            tuple[SteamHeaterZoneKind, ...],
            list[tuple[float, tuple[float, ...]]],
        ] = {}
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
    orientation: TubeOrientation | None,
    available_area: float | None = None,
    euler_provider: str = "zukauskas",
    finned_heat_transfer_provider: object = DEFAULT_FINNED_HT_PROVIDER,
    finned_pressure_drop_provider: object = DEFAULT_FINNED_DP_PROVIDER,
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
    if available_area is None:
        available_area = hx.bundle.total_outer_area
    if not math.isfinite(available_area) or available_area <= 0.0:
        raise ValueError("available_area must be positive and finite.")

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
            euler_provider=euler_provider,
            finned_heat_transfer_provider=finned_heat_transfer_provider,
            finned_pressure_drop_provider=finned_pressure_drop_provider,
        )

    if orientation is None and inlet_state.h > saturation.hf:
        if inlet_state.h <= saturation.hg:
            _require_condensation_orientation(orientation)
        # A superheated inlet may use all available area before reaching the
        # saturation dome. Evaluate that physical boundary without assuming
        # an orientation; require one only when the accepted solution must
        # continue into condensation.
        q_to_saturated_vapor = mass_flow_steam * (inlet_state.h - saturation.hg)
        saturated_vapor_trial = trial(q_to_saturated_vapor)
        if saturated_vapor_trial.required_area < available_area:
            _require_condensation_orientation(orientation)
        q_high = q_to_saturated_vapor
        high = saturated_vapor_trial
    else:
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
        hx=hx,
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
    orientation: TubeOrientation | None,
    outlet_state: WaterSteamProperties | None = None,
    Q_total: float | None = None,
    euler_provider: str = "zukauskas",
    finned_heat_transfer_provider: object = DEFAULT_FINNED_HT_PROVIDER,
    finned_pressure_drop_provider: object = DEFAULT_FINNED_DP_PROVIDER,
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
        euler_provider=euler_provider,
        finned_heat_transfer_provider=finned_heat_transfer_provider,
        finned_pressure_drop_provider=finned_pressure_drop_provider,
    )
    if not math.isfinite(trial.required_area):
        raise ValueError("Specified steam Rating violates a positive zone temperature difference.")
    return _build_solution(
        hx=hx,
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
    orientation: TubeOrientation | None,
    Q_total: float,
    cache: _SolveCache,
    euler_provider: str,
    finned_heat_transfer_provider: object,
    finned_pressure_drop_provider: object,
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
    outside = evaluate_outside_side(
        hx,
        provider=outside_provider,
        mass_flow=mass_flow_outside,
        T_in=T_in_outside,
        T_out=T_out_outside,
        p=p_outside,
        euler_provider=euler_provider,
        properties_mean=outside_props,
        finned_heat_transfer_provider=finned_heat_transfer_provider,
        finned_pressure_drop_provider=finned_pressure_drop_provider,
    )

    flow_arrangement = hx.bundle.flow_arrangement
    specs = _partition_enthalpy(inlet_state.h, h_out, saturation)
    allocation = _allocate_parallel_air_zones(
        hx,
        specs=specs,
        saturation=saturation,
        mass_flow_steam=mass_flow_steam,
        mass_flow_outside=mass_flow_outside,
        T_in_outside=T_in_outside,
        T_out_outside=T_out_outside,
        alpha_outside_physical=outside.alpha_physical,
        outside_velocity=outside.velocity,
        flow_arrangement=flow_arrangement,
        orientation=orientation,
        cache=cache,
    )
    zones = allocation.zones
    trial_warnings: list[ModelWarning] = list(outside.warnings)
    for zone in zones:
        trial_warnings.extend(zone.warnings)

    required_area = sum(zone.area for zone in zones)
    UA_total = sum(zone.UA for zone in zones)
    return _TrialResult(
        h_out=h_out,
        T_out_outside=T_out_outside,
        mass_flow_outside=mass_flow_outside,
        zones=tuple(zones),
        required_area=required_area,
        UA_total=UA_total,
        outside_alpha=outside.alpha_effective_gross,
        outside_alpha_physical=outside.alpha_physical,
        outside_props_mean=outside_props,
        outside_evaluation=outside,
        mass_flux=mass_flow_steam / hx.bundle.internal_flow_area_per_pass,
        zone_allocation_iterations=allocation.iterations,
        zone_allocation_converged=allocation.converged,
        zone_allocation_residual=allocation.residual,
        mixed_air_energy_residual=allocation.mixed_air_energy_residual,
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


def _allocate_parallel_air_zones(
    hx,
    *,
    specs: tuple[_ZoneSpec, ...],
    saturation: WaterSteamSaturationProperties,
    mass_flow_steam: float,
    mass_flow_outside: float,
    T_in_outside: float,
    T_out_outside: float,
    alpha_outside_physical: float,
    outside_velocity: float,
    flow_arrangement: str,
    orientation: TubeOrientation | None,
    cache: _SolveCache,
    max_iterations: int = _ZONE_ALLOCATION_MAX_ITERATIONS,
    fraction_tolerance: float = _ZONE_ALLOCATION_FRACTION_TOLERANCE,
    damping: float = _ZONE_ALLOCATION_DAMPING,
) -> _ParallelZoneAllocation:
    """Make parallel air-flow fractions consistent with zone area fractions.

    Duty fractions are used only as a neutral, energy-feasible starting
    iterate. They are never accepted as geometry: every finite multi-zone
    result must converge to ``f_air ~= A_required / sum(A_required)``.
    """
    if not specs:
        raise ValueError("Steam-heater duty did not produce an active thermal zone.")
    if max_iterations <= 0 or fraction_tolerance <= 0.0:
        raise ValueError("Parallel zone-allocation iteration controls must be positive.")
    if not math.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("Parallel zone-allocation damping must be in (0, 1].")

    duties = tuple(
        mass_flow_steam * (spec.h_in - spec.h_out) for spec in specs
    )
    zone_key = tuple(spec.kind for spec in specs)
    total_duty = math.fsum(duties)
    fractions = _parallel_fraction_starting_guess(
        cache=cache,
        zone_key=zone_key,
        total_duty=total_duty,
        duty_fractions=_normalize_positive_fractions(duties),
    )

    # The one-zone limit deliberately bypasses the iterative update. With
    # f=1 its air state, outside correlation, driving force and required area
    # are exactly the historical single-zone calculation.
    if len(specs) == 1:
        zones = _evaluate_parallel_zone_fractions(
            hx,
            specs=specs,
            fractions=(1.0,),
            saturation=saturation,
            mass_flow_steam=mass_flow_steam,
            mass_flow_outside=mass_flow_outside,
            T_in_outside=T_in_outside,
            alpha_outside_physical=alpha_outside_physical,
            outside_velocity=outside_velocity,
            flow_arrangement=flow_arrangement,
            orientation=orientation,
            cache=cache,
        )
        if not all(math.isfinite(zone.area) for zone in zones):
            return _ParallelZoneAllocation(
                zones=zones,
                iterations=1,
                converged=False,
                residual=math.inf,
                mixed_air_energy_residual=_parallel_air_energy_residual(
                    zones=zones,
                    mass_flow_outside=mass_flow_outside,
                    T_in_outside=T_in_outside,
                    T_out_outside=T_out_outside,
                    cache=cache,
                ),
            )
        zones = _set_zone_area_fractions(zones, (1.0,))
        return _ParallelZoneAllocation(
            zones=zones,
            iterations=1,
            converged=True,
            residual=0.0,
            mixed_air_energy_residual=_parallel_air_energy_residual(
                zones=zones,
                mass_flow_outside=mass_flow_outside,
                T_in_outside=T_in_outside,
                T_out_outside=T_out_outside,
                cache=cache,
            ),
        )

    last_finite_fractions: tuple[float, ...] | None = None
    last_residual = math.inf
    previous_residual = math.inf
    active_damping = damping
    tried_feasible_guess = False
    for iteration in range(1, max_iterations + 1):
        zones = _evaluate_parallel_zone_fractions(
            hx,
            specs=specs,
            fractions=fractions,
            saturation=saturation,
            mass_flow_steam=mass_flow_steam,
            mass_flow_outside=mass_flow_outside,
            T_in_outside=T_in_outside,
            alpha_outside_physical=alpha_outside_physical,
            outside_velocity=outside_velocity,
            flow_arrangement=flow_arrangement,
            orientation=orientation,
            cache=cache,
        )
        if not all(math.isfinite(zone.area) and zone.area > 0.0 for zone in zones):
            # A damped step can cross a zone pinch. Backtrack toward the last
            # finite allocation instead of accepting or clipping that state.
            if last_finite_fractions is not None:
                fractions = _normalize_positive_fractions(
                    tuple(
                        0.5 * (finite + current)
                        for finite, current in zip(last_finite_fractions, fractions)
                    )
                )
                continue
            if not tried_feasible_guess:
                tried_feasible_guess = True
                feasible_guess = _feasible_parallel_fraction_guess(
                    specs=specs,
                    duties=duties,
                    saturation=saturation,
                    mass_flow_outside=mass_flow_outside,
                    T_in_outside=T_in_outside,
                    cache=cache,
                )
                if feasible_guess is not None and feasible_guess != fractions:
                    fractions = feasible_guess
                    continue
            # This is a thermodynamic pinch/infeasible duty, not a finite
            # fixed point that failed to converge. Simulation's outer area
            # root uses the infinite required area to reduce its duty;
            # Rating converts it to its existing controlled pinch error.
            return _ParallelZoneAllocation(
                zones=zones,
                iterations=iteration,
                converged=False,
                residual=math.inf,
                mixed_air_energy_residual=_parallel_air_energy_residual(
                    zones=zones,
                    mass_flow_outside=mass_flow_outside,
                    T_in_outside=T_in_outside,
                    T_out_outside=T_out_outside,
                    cache=cache,
                ),
            )

        area_fractions = _normalize_positive_fractions(
            tuple(zone.area for zone in zones)
        )
        last_residual = max(
            abs(area_fraction - air_fraction)
            for area_fraction, air_fraction in zip(area_fractions, fractions)
        )
        if last_residual <= fraction_tolerance:
            zones = _set_zone_area_fractions(zones, area_fractions)
            _remember_parallel_fraction_solution(
                cache=cache,
                zone_key=zone_key,
                total_duty=total_duty,
                area_fractions=area_fractions,
            )
            return _ParallelZoneAllocation(
                zones=zones,
                iterations=iteration,
                converged=True,
                residual=last_residual,
                mixed_air_energy_residual=_parallel_air_energy_residual(
                    zones=zones,
                    mass_flow_outside=mass_flow_outside,
                    T_in_outside=T_in_outside,
                    T_out_outside=T_out_outside,
                    cache=cache,
                ),
            )

        if (
            math.isfinite(previous_residual)
            and last_residual >= _ZONE_ALLOCATION_STALL_RATIO * previous_residual
        ):
            active_damping = max(
                _ZONE_ALLOCATION_MIN_DAMPING,
                0.5 * active_damping,
            )
        previous_residual = last_residual
        last_finite_fractions = fractions
        fractions = _normalize_positive_fractions(
            tuple(
                (1.0 - active_damping) * air_fraction
                + active_damping * area_fraction
                for air_fraction, area_fraction in zip(fractions, area_fractions)
            )
        )

    raise SteamZoneAllocationConvergenceError(
        iterations=max_iterations,
        residual=last_residual,
    )


def _parallel_fraction_starting_guess(
    *,
    cache: _SolveCache,
    zone_key: tuple[SteamHeaterZoneKind, ...],
    total_duty: float,
    duty_fractions: tuple[float, ...],
) -> tuple[float, ...]:
    """Warm-start repeated Simulation trials without changing their physics."""
    history = cache.zone_fraction_history.get(zone_key, ())
    if not history:
        return duty_fractions
    latest_duty, latest_fractions = history[-1]
    if len(history) < 2:
        return latest_fractions
    previous_duty, previous_fractions = history[-2]
    duty_span = latest_duty - previous_duty
    if abs(duty_span) <= 1.0e-14 * max(abs(total_duty), 1.0):
        return latest_fractions
    extrapolation = (total_duty - latest_duty) / duty_span
    if not math.isfinite(extrapolation) or abs(extrapolation) > 2.0:
        return latest_fractions
    predicted = tuple(
        latest + extrapolation * (latest - previous)
        for latest, previous in zip(latest_fractions, previous_fractions)
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in predicted):
        return latest_fractions
    return _normalize_positive_fractions(predicted)


def _remember_parallel_fraction_solution(
    *,
    cache: _SolveCache,
    zone_key: tuple[SteamHeaterZoneKind, ...],
    total_duty: float,
    area_fractions: tuple[float, ...],
) -> None:
    history = cache.zone_fraction_history.setdefault(zone_key, [])
    if history and math.isclose(
        history[-1][0], total_duty, rel_tol=1.0e-14, abs_tol=1.0e-9
    ):
        history[-1] = (total_duty, area_fractions)
    else:
        history.append((total_duty, area_fractions))
        del history[:-2]


def _evaluate_parallel_zone_fractions(
    hx,
    *,
    specs: tuple[_ZoneSpec, ...],
    fractions: tuple[float, ...],
    saturation: WaterSteamSaturationProperties,
    mass_flow_steam: float,
    mass_flow_outside: float,
    T_in_outside: float,
    alpha_outside_physical: float,
    outside_velocity: float,
    flow_arrangement: str,
    orientation: TubeOrientation | None,
    cache: _SolveCache,
) -> tuple[SteamHeaterZoneResult, ...]:
    if len(specs) != len(fractions):
        raise ValueError("Steam zone specs and parallel air fractions must align.")
    frontal_area_total = hx.bundle.frontal_flow_area
    face_mass_flux = mass_flow_outside / frontal_area_total
    zones: list[SteamHeaterZoneResult] = []
    for spec, fraction in zip(specs, fractions):
        if not math.isfinite(fraction) or fraction <= 0.0:
            raise ValueError("Every active steam zone needs a positive air-flow fraction.")
        Q_zone = mass_flow_steam * (spec.h_in - spec.h_out)
        outside_mass_flow = fraction * mass_flow_outside
        outside_frontal_area = fraction * frontal_area_total
        outside_T_zone_out = _outside_outlet_temperature(
            Q_total=Q_zone,
            mass_flow_outside=outside_mass_flow,
            T_in_outside=T_in_outside,
            cache=cache,
        )
        zones.append(
            _evaluate_zone(
                hx,
                spec=spec,
                saturation=saturation,
                mass_flow_steam=mass_flow_steam,
                alpha_outside_physical=alpha_outside_physical,
                outside_T_in=T_in_outside,
                outside_T_out=outside_T_zone_out,
                outside_mass_flow=outside_mass_flow,
                outside_mass_flow_fraction=fraction,
                outside_frontal_area=outside_frontal_area,
                outside_face_mass_flux=face_mass_flux,
                outside_velocity=outside_velocity,
                flow_arrangement=flow_arrangement,
                orientation=orientation,
                cache=cache,
            )
        )
    return tuple(zones)


def _feasible_parallel_fraction_guess(
    *,
    specs: tuple[_ZoneSpec, ...],
    duties: tuple[float, ...],
    saturation: WaterSteamSaturationProperties,
    mass_flow_outside: float,
    T_in_outside: float,
    cache: _SolveCache,
) -> tuple[float, ...] | None:
    """Find a strictly positive-temperature-approach starting allocation."""
    minimum_fractions: list[float] = []
    for spec, Q_zone in zip(specs, duties):
        hot_out_temperature = (
            saturation.Tsat
            if spec.kind is SteamHeaterZoneKind.CONDENSATION
            else cache.steam_state(spec.h_out).T
        )
        approach_margin = max(1.0e-6, 1.0e-9 * abs(hot_out_temperature))
        maximum_air_outlet = hot_out_temperature - approach_margin
        if maximum_air_outlet <= T_in_outside:
            return None
        properties_at_limit = cache.outside_state(
            0.5 * (T_in_outside + maximum_air_outlet)
        )
        outside_capacity_per_fraction = (
            mass_flow_outside
            * properties_at_limit.cp
            * (maximum_air_outlet - T_in_outside)
        )
        if (
            not math.isfinite(outside_capacity_per_fraction)
            or outside_capacity_per_fraction <= 0.0
        ):
            return None
        # This is the direct inversion of the same midpoint-cp convention
        # used by _outside_outlet_temperature at the limiting outlet state.
        minimum_fraction = Q_zone / outside_capacity_per_fraction
        minimum_fraction *= 1.0 + 1.0e-10
        if not math.isfinite(minimum_fraction) or minimum_fraction >= 1.0:
            return None
        minimum_fractions.append(minimum_fraction)

    minimum_sum = math.fsum(minimum_fractions)
    if not math.isfinite(minimum_sum) or minimum_sum >= 1.0:
        return None
    duty_fractions = _normalize_positive_fractions(duties)
    remainder = 1.0 - minimum_sum
    return _normalize_positive_fractions(
        tuple(
            minimum + remainder * duty_fraction
            for minimum, duty_fraction in zip(minimum_fractions, duty_fractions)
        )
    )


def _normalize_positive_fractions(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Active steam-zone allocation weights must be positive and finite.")
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Active steam-zone allocation weights have no finite sum.")
    return tuple(value / total for value in values)


def _set_zone_area_fractions(
    zones: tuple[SteamHeaterZoneResult, ...],
    area_fractions: tuple[float, ...],
) -> tuple[SteamHeaterZoneResult, ...]:
    return tuple(
        replace(
            zone,
            area_fraction=area_fraction,
            tube_length_fraction=area_fraction,
        )
        for zone, area_fraction in zip(zones, area_fractions)
    )


def _parallel_air_energy_residual(
    *,
    zones: tuple[SteamHeaterZoneResult, ...],
    mass_flow_outside: float,
    T_in_outside: float,
    T_out_outside: float,
    cache: _SolveCache,
) -> float:
    branch_heat = math.fsum(
        _outside_sensible_heat_rate(
            mass_flow=zone.outside_mass_flow,
            T_in=zone.outside_T_in,
            T_out=zone.outside_T_out,
            cache=cache,
        )
        for zone in zones
    )
    mixed_heat = _outside_sensible_heat_rate(
        mass_flow=mass_flow_outside,
        T_in=T_in_outside,
        T_out=T_out_outside,
        cache=cache,
    )
    zone_heat = math.fsum(zone.Q for zone in zones)
    return max(
        abs(branch_heat - zone_heat),
        abs(mixed_heat - zone_heat),
        abs(branch_heat - mixed_heat),
    )


def _outside_sensible_heat_rate(
    *,
    mass_flow: float,
    T_in: float,
    T_out: float,
    cache: _SolveCache,
) -> float:
    properties_mean = cache.outside_state(0.5 * (T_in + T_out))
    return mass_flow * properties_mean.cp * (T_out - T_in)


def _log_mean_temperature_difference(delta_T_in: float, delta_T_out: float) -> float:
    """Exact terminal LMTD for an isothermal-side (condensing) zone.

    Uses the mathematically correct equal-difference limit
    (``LMTD -> delta_T_in`` as ``delta_T_out -> delta_T_in``) instead of the
    ``0/0`` closed form, so it stays finite and NaN-free at and near equal
    terminal differences.
    """
    if (
        not math.isfinite(delta_T_in)
        or not math.isfinite(delta_T_out)
        or delta_T_in <= 0.0
        or delta_T_out <= 0.0
    ):
        raise ValueError(
            "Log-mean temperature difference requires two finite, positive "
            "terminal temperature differences."
        )
    if math.isclose(delta_T_in, delta_T_out, rel_tol=1.0e-9, abs_tol=1.0e-12):
        return 0.5 * (delta_T_in + delta_T_out)
    return (delta_T_in - delta_T_out) / math.log(delta_T_in / delta_T_out)


def _evaluate_zone(
    hx,
    *,
    spec: _ZoneSpec,
    saturation: WaterSteamSaturationProperties,
    mass_flow_steam: float,
    alpha_outside_physical: float,
    outside_T_in: float,
    outside_T_out: float,
    outside_mass_flow: float,
    outside_mass_flow_fraction: float,
    outside_frontal_area: float,
    outside_face_mass_flux: float,
    outside_velocity: float,
    flow_arrangement: str,
    orientation: TubeOrientation | None,
    cache: _SolveCache,
) -> SteamHeaterZoneResult:
    Q = mass_flow_steam * (spec.h_in - spec.h_out)
    warnings: list[ModelWarning] = []
    quality_in = quality_out = None
    condensation = None
    if spec.kind is SteamHeaterZoneKind.CONDENSATION:
        condensation_orientation = _require_condensation_orientation(orientation)
        quality_in = (spec.h_in - saturation.hf) / saturation.hfg
        quality_out = (spec.h_out - saturation.hf) / saturation.hfg
        condensation = solve_steam_condensation_zone(
            p=saturation.p,
            mass_flow_total=mass_flow_steam,
            flow_area_per_pass=hx.bundle.internal_flow_area_per_pass,
            tube_inner_diameter=hx.bundle.internal_hydraulic_diameter,
            quality_in=quality_in,
            quality_out=quality_out,
            orientation=condensation_orientation,
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

    network = calculate_resistance_network(
        bundle=hx.bundle,
        alpha_inside=alpha_inside,
        outside_alpha_physical=alpha_outside_physical,
        resistance_core_wall=hx.tube_wall_resistance(),
    )
    U = network.U_gross_outside

    delta_T_terminal_in = T_in - outside_T_in
    delta_T_terminal_out = T_out - outside_T_out

    if spec.kind is SteamHeaterZoneKind.CONDENSATION:
        driving_force_method = DRIVING_FORCE_ISOTHERMAL_LMTD
        if delta_T_terminal_in <= 0.0 or delta_T_terminal_out <= 0.0:
            EMTD = math.nan
            area = math.inf
            UA = math.inf
        else:
            # Exact terminal LMTD for an isothermal condensing side; this is
            # the Cr -> 0 epsilon-NTU limit (eps = 1 - exp(-NTU)), not the
            # generic finite-Cr inversion used for the sensible zones below.
            EMTD = _log_mean_temperature_difference(
                delta_T_terminal_in, delta_T_terminal_out
            )
            UA = Q / EMTD
            area = UA / U
    else:
        driving_force_method = _EPSILON_NTU_DRIVING_FORCE_METHOD_BY_ARRANGEMENT.get(
            flow_arrangement.lower()
        )
        if driving_force_method is None:
            raise ValueError(f"Unsupported flow_arrangement: {flow_arrangement!r}")
        # Effective zone capacity rates, reconstructed from the actual zone
        # duty and terminal temperatures (steam is always the hot/inside
        # stream for this exchanger direction).
        C_inside_zone = Q / abs(T_in - T_out)
        C_outside_zone = Q / abs(outside_T_out - outside_T_in)
        C_min = min(C_inside_zone, C_outside_zone)
        Q_max_zone = C_min * delta_T_terminal_in
        EMTD = math.nan
        UA = math.inf
        area = math.inf
        if math.isfinite(Q_max_zone) and Q_max_zone > 0.0:
            eps_zone = Q / Q_max_zone
            if 0.0 <= eps_zone < 1.0:
                try:
                    NTU_zone = ntu_from_effectiveness(
                        eps_zone,
                        C_inside_zone,
                        C_outside_zone,
                        flow_arrangement=flow_arrangement,
                        C_inside=C_inside_zone,
                        C_outside=C_outside_zone,
                    )
                except ValueError:
                    NTU_zone = None
                if NTU_zone is not None and math.isfinite(NTU_zone) and NTU_zone > 0.0:
                    UA_zone = NTU_zone * C_min
                    EMTD = Q / UA_zone
                    area = UA_zone / U
                    UA = UA_zone

    return SteamHeaterZoneResult(
        kind=spec.kind,
        h_in=spec.h_in,
        h_out=spec.h_out,
        T_in=T_in,
        T_out=T_out,
        Q=Q,
        alpha_inside=alpha_inside,
        alpha_outside=network.outside_alpha_effective_gross,
        alpha_outside_physical=alpha_outside_physical,
        U=U,
        area=area,
        UA=UA,
        outside_T_in=outside_T_in,
        outside_T_out=outside_T_out,
        delta_T_terminal_in=delta_T_terminal_in,
        delta_T_terminal_out=delta_T_terminal_out,
        effective_mean_temperature_difference=EMTD,
        driving_force_method=driving_force_method,
        area_fraction=math.nan,
        tube_length_fraction=math.nan,
        outside_mass_flow=outside_mass_flow,
        outside_mass_flow_fraction=outside_mass_flow_fraction,
        outside_frontal_area=outside_frontal_area,
        outside_face_mass_flux=outside_face_mass_flux,
        outside_velocity=outside_velocity,
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


def _build_solution(
    *,
    hx,
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
    if not trial.zone_allocation_converged:
        raise ValueError(
            "Accepted steam-heater duty has no converged parallel zone allocation."
        )

    sum_zone_area_fraction = math.fsum(
        zone.area_fraction for zone in trial.zones
    )
    sum_zone_air_mass_flow = math.fsum(
        zone.outside_mass_flow for zone in trial.zones
    )
    if not math.isclose(
        sum_zone_area_fraction, 1.0, rel_tol=0.0, abs_tol=2.0e-12
    ):
        raise ValueError("Steam-heater zone area fractions do not sum to one.")
    if not math.isclose(
        sum_zone_air_mass_flow,
        trial.mass_flow_outside,
        rel_tol=2.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Steam-heater parallel zone air flows do not sum to the total.")
    if abs(trial.mixed_air_energy_residual) > max(1.0e-4, 2.0e-8 * Q_total):
        raise ValueError(
            "Steam-heater mixed outside-air energy balance is internally inconsistent."
        )

    inside_alpha_area_weighted = sum(
        zone.alpha_inside * zone.area for zone in trial.zones
    ) / A_total
    U_equivalent = UA_total / A_total
    reference_network = calculate_resistance_network(
        bundle=hx.bundle,
        alpha_inside=1.0,
        outside_alpha_physical=trial.outside_alpha_physical,
        resistance_core_wall=hx.tube_wall_resistance(),
    )
    equivalent_inside_resistance = (
        1.0 / (U_equivalent * hx.bundle.total_outer_area)
        - hx.tube_wall_resistance()
        - reference_network.resistance_outside
    )
    if (
        not math.isfinite(equivalent_inside_resistance)
        or equivalent_inside_resistance <= 0.0
    ):
        raise ValueError(
            "Equivalent steam-side HTC leaves no positive inside resistance."
        )
    inside_alpha_equivalent = 1.0 / (
        equivalent_inside_resistance * hx.bundle.total_inner_area
    )
    warnings = list(trial.warnings)
    warnings.append(
        make_warning(
            code="STEAM_HEATER_ZONE_ALLOCATION_0D_ESTIMATE",
            message=(
                "Steam phase zones remain sequential on the tube side, while "
                "crossflow air is allocated in parallel by converged geometric "
                "area fraction with a common inlet and mixed outlet. This is a "
                "multi-zone 0D allocation, not spatial phase-front resolution."
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
        inside_alpha_equivalent=inside_alpha_equivalent,
        inside_alpha_area_weighted=inside_alpha_area_weighted,
        outside_alpha=trial.outside_alpha,
        outside_alpha_physical=trial.outside_alpha_physical,
        outside_props_mean=trial.outside_props_mean,
        U_equivalent=U_equivalent,
        zones=trial.zones,
        zone_allocation_method=ZONE_ALLOCATION_METHOD_PARALLEL_BY_GEOMETRY,
        zone_allocation_iterations=trial.zone_allocation_iterations,
        zone_allocation_converged=trial.zone_allocation_converged,
        zone_allocation_residual=trial.zone_allocation_residual,
        sum_zone_area_fraction=sum_zone_area_fraction,
        sum_zone_air_mass_flow=sum_zone_air_mass_flow,
        mixed_outside_T_out=trial.T_out_outside,
        Q_zone_sum=Q_total,
        mixed_air_energy_residual=trial.mixed_air_energy_residual,
        converged=converged and trial.zone_allocation_converged,
        iterations=iterations,
        root_iterations=root_iterations,
        property_evaluations=cache.property_evaluations,
        runtime_s=runtime_s,
        two_phase_pressure_drop_supported=False,
        warnings=tuple(_deduplicate_warnings(warnings)),
        assumptions=(
            "constant_nominal_steam_pressure",
            "tube_side_zones_sequential",
            "outside_air_parallel_by_geometric_area_fraction",
            "common_outside_inlet_and_mixed_outlet",
            "zone_UA_sum_is_authoritative",
            "multi_zone_0d_not_longitudinal_1d",
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
    orientation: TubeOrientation | None,
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
    if orientation is not None and not isinstance(orientation, TubeOrientation):
        raise ValueError("orientation must be a TubeOrientation value or None.")


def _require_condensation_orientation(
    orientation: TubeOrientation | None,
) -> TubeOrientation:
    if orientation is None:
        raise ValueError(
            "Pure-steam condensation requires explicit tube_orientation on BareTube; "
            "Shah 2009 uses orientation-specific regime boundaries."
        )
    if not isinstance(orientation, TubeOrientation):
        raise ValueError("tube_orientation must be a TubeOrientation value.")
    return orientation


def _deduplicate_warnings(warnings) -> list[ModelWarning]:
    result: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        key = (warning.code, warning.message)
        if key not in seen:
            seen.add(key)
            result.append(warning)
    return result
