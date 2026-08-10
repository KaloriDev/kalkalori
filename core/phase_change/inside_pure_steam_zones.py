# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Multi-zone in-tube pure water/steam solver (v0.6.2).

Combines up to three zones for a cooling pure-water/steam tube-side
stream, in this fixed order (never reversed):

    1. DESUPERHEAT   -- superheated vapor cooling down to T_sat/h_g.
    2. CONDENSATION  -- saturated vapor / wet steam condensing from the
       zone's inlet quality down to a lower quality (see
       ``core.phase_change.inside_pure_steam_condensation``, Shah 1979).
    3. SUBCOOLING     -- saturated/subcooled liquid cooling further below
       T_sat.

Not every zone is present for every case: which zones are attempted (and
whether each is fully or only partially traversed before the available
inside heat-transfer area runs out) is determined from the inlet phase and
a single opposite-side bulk representative state.

0D scope (deliberate simplification, see ``docs/references.md`` and the
v0.6.2 roadmap entry): the opposite ("sink") side is represented by a
single bulk temperature and heat-transfer coefficient, shared by all
zones -- this model does not assume the opposite-side stream flows in
series past the desuperheat/condensation/subcooling zones (a crossflow
geometry has no such spatial arrangement); it only partitions the
steam-side thermal state and heat-transfer surface. Each zone's driving
temperature difference is therefore computed against that single bulk
sink temperature (the mathematical Cr -> 0 limit of the standard eps-NTU
relations in ``core.heat_transfer.ntu`` -- see ``_isothermal_sink_effectiveness``
below), not a true counterflow/crossflow LMTD between two spatially
varying profiles. Zone-area fractions are therefore 0D estimated
allocations, not resolved phase-front locations.

Tube-wall conduction resistance and the outside-side film resistance are
shared unchanged across all three zones (only the steam-side inside alpha
differs per zone) -- both are converted once to a per-unit-inside-area
resistance so that a physically constant per-unit-length quantity (wall
conduction resistance times inside area is invariant to how the tube's
length is partitioned into zones) does not need to be re-solved per zone.

This module does not implement evaporation/boiling: if the opposite side
is hot enough that no cooling of the inlet stream is possible at all
(evaporation would be required), it raises
``ReverseDirectionEvaporationNotSupportedError`` instead of guessing.

Ref: IAPWS-IF97 (via ``core.properties.water``); Shah, M.M. (1979) (via
``core.phase_change.inside_pure_steam_condensation``); eps-NTU method
(``core.heat_transfer.ntu``, Incropera et al.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from core.common.warnings import ModelWarning
from core.heat_transfer.internal_flow import FluidProps, heat_transfer_coefficient_internal
from core.phase_change.inside_pure_steam_condensation import (
    solve_inside_condensation_zone,
)
from core.properties.water import (
    WaterPhaseRegion,
    WaterSteamState,
    water_steam_props_iapws97,
    water_steam_state,
)

SOURCE = "phase_change.inside_pure_steam_zones"

# Below this remaining area, a zone is not worth constructing (avoids
# spurious near-zero zones from floating-point roundoff at full
# traversal).
_AREA_EPS_M2 = 1e-12

# If a partially-traversed sensible zone's solved outlet temperature lands
# within this band of the saturation boundary, snap it onto the boundary
# instead (matches the T+p ambiguity tolerance in core.properties.water --
# being this close to T_sat is not a meaningfully different outcome from
# having fully traversed the zone, and avoids reconstructing a state
# exactly on the ambiguous saturation line).
_SATURATION_SNAP_TOLERANCE_K = 1e-2


class ReverseDirectionEvaporationNotSupportedError(RuntimeError):
    """Raised when cooling the inlet stream is not possible at all.

    This means the opposite-side bulk temperature is at or above the
    inlet's own temperature (or saturation temperature, for a saturated/
    two-phase inlet): net heat flow would have to be *into* the water/
    steam side, which would require evaporation/boiling. v0.6.2 only
    implements cooling/condensation; evaporation is out of scope (see
    ``docs/roadmap.md``, v0.6.3).
    """


@dataclass(frozen=True)
class SteamZoneDiagnostics:
    """Per-zone diagnostic breakdown of one desuperheat/condensation/subcooling zone."""

    kind: str  # "desuperheat" | "condensation" | "subcooling"
    T_in: float
    T_out: float
    h_in: float
    h_out: float
    quality_in: Optional[float]
    quality_out: Optional[float]
    Q: float
    A: float
    alpha_inside: float
    U: float
    fully_traversed: bool
    warnings: list[ModelWarning] = field(default_factory=list)


@dataclass(frozen=True)
class InsideSteamZonesResult:
    """Result of the in-tube pure water/steam multi-zone solve.

    Attributes:
        phase_in, phase_out: Inlet/outlet phase region classification.
        T_in, T_out: Inlet/outlet temperature [K].
        p: Nominal tube-side pressure [Pa] (constant, see module docstring
            of ``core.properties.water`` -- two-phase pressure drop is not
            modelled in v0.6.2).
        h_in, h_out: Inlet/outlet specific enthalpy [J/kg].
        quality_in, quality_out: Inlet/outlet vapor quality [-], `None`
            outside the saturation dome.
        T_sat: Saturation temperature at `p` [K].
        Q_desuperheat, Q_condensation, Q_subcooling: Duty released by each
            zone [W] (`0.0` for a zone that was not constructed).
        Q_total: Total duty released [W],
            `Q_desuperheat + Q_condensation + Q_subcooling`, exactly equal
            to `m_dot_total * (h_in - h_out)` by construction.
        A_desuperheat, A_condensation, A_subcooling: Heat-transfer area
            allocated to each zone [m2] (`0.0` for a zone that was not
            constructed).
        A_total: Total inside heat-transfer area available, echoed input.
        f_desuperheat, f_condensation, f_subcooling: Area fractions
            [-], `A_zone / A_total`.
        zones: Per-zone diagnostic breakdown, in traversal order, only for
            zones that were actually constructed.
        m_dot_total: Total tube-side mass flow [kg/s] (constant).
        two_phase_pressure_drop_supported: `False` whenever a
            condensation zone is active in this result (see module
            docstring of ``core.phase_change.inside_pure_steam_condensation``).
        warnings: Aggregated applicability/limitation warnings from every
            constructed zone.
    """

    phase_in: WaterPhaseRegion
    phase_out: WaterPhaseRegion
    T_in: float
    T_out: float
    p: float
    h_in: float
    h_out: float
    quality_in: Optional[float]
    quality_out: Optional[float]
    T_sat: float

    Q_desuperheat: float
    Q_condensation: float
    Q_subcooling: float
    Q_total: float

    A_desuperheat: float
    A_condensation: float
    A_subcooling: float
    A_total: float

    f_desuperheat: float
    f_condensation: float
    f_subcooling: float

    zones: tuple[SteamZoneDiagnostics, ...]
    m_dot_total: float
    two_phase_pressure_drop_supported: bool
    warnings: list[ModelWarning]


def solve_inside_steam_zones(
    *,
    p: float,
    inlet: WaterSteamState,
    m_dot_total: float,
    D_i: float,
    flow_area: float,
    A_inside_total: float,
    A_outside_total: float,
    R_wall_total: float,
    T_sink: float,
    alpha_outside: float,
) -> InsideSteamZonesResult:
    """Solve the in-tube pure water/steam multi-zone cooling problem.

    Args:
        p: Nominal tube-side pressure [Pa].
        inlet: Classified inlet state (``core.properties.water.
            water_steam_state``), at this same `p`.
        m_dot_total: Total tube-side mass flow [kg/s].
        D_i: Tube inner diameter [m].
        flow_area: Total tube-side internal flow area [m2].
        A_inside_total: Total available inside heat-transfer area [m2].
        A_outside_total: Total outside heat-transfer area [m2] (used only
            to express the outside film resistance on a per-unit-inside-
            area basis).
        R_wall_total: Tube-wall conduction resistance [K/W] (e.g.
            ``BareTubeHeatExchanger.tube_wall_resistance()``).
        T_sink: Representative opposite-side bulk temperature [K].
        alpha_outside: Representative opposite-side heat transfer
            coefficient [W/(m2*K)].

    Returns:
        InsideSteamZonesResult.

    Raises:
        ReverseDirectionEvaporationNotSupportedError: if `T_sink` is at or
            above the inlet's own temperature (no cooling is possible).
        ValueError: for invalid geometry/flow inputs.
    """
    if inlet.p != p:
        raise ValueError("inlet.p must match p.")
    for name, value in (
        ("m_dot_total", m_dot_total),
        ("D_i", D_i),
        ("flow_area", flow_area),
        ("A_inside_total", A_inside_total),
        ("A_outside_total", A_outside_total),
        ("R_wall_total", R_wall_total),
        ("alpha_outside", alpha_outside),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")
    if not math.isfinite(T_sink):
        raise ValueError("T_sink must be finite.")

    if T_sink >= inlet.T:
        raise ReverseDirectionEvaporationNotSupportedError(
            f"T_sink={T_sink} K is at or above the inlet temperature "
            f"T_in={inlet.T} K; cooling this stream is not possible "
            "without evaporation/boiling, which is not implemented in "
            "v0.6.2 (see docs/roadmap.md, v0.6.3)."
        )

    # Shared resistance terms, expressed per unit INSIDE area. Both are
    # exactly invariant to how the tube length is split into zones (wall
    # resistance scales as 1/length while inside area scales as length, so
    # their product is constant; the outside/inside area ratio is a fixed
    # tube-geometry constant) -- see module docstring.
    R_wall_per_Ai = R_wall_total * A_inside_total
    R_outside_per_Ai = (1.0 / alpha_outside) * (A_outside_total / A_inside_total)
    R_shared_per_Ai = R_wall_per_Ai + R_outside_per_Ai

    state = inlet
    A_remaining = A_inside_total
    zones: list[SteamZoneDiagnostics] = []
    warnings: list[ModelWarning] = []
    Q_desuperheat = A_desuperheat = 0.0
    Q_condensation = A_condensation = 0.0
    Q_subcooling = A_subcooling = 0.0

    if A_remaining > _AREA_EPS_M2 and state.phase is WaterPhaseRegion.SUPERHEATED_VAPOR:
        zone, state, A_remaining = _solve_sensible_zone(
            kind="desuperheat",
            state=state,
            p=p,
            m_dot_total=m_dot_total,
            D_i=D_i,
            flow_area=flow_area,
            A_remaining=A_remaining,
            R_shared_per_Ai=R_shared_per_Ai,
            T_sink=T_sink,
            T_target=state.T_sat,
            snap_quality_at_target=1.0,
        )
        zones.append(zone)
        warnings.extend(zone.warnings)
        Q_desuperheat, A_desuperheat = zone.Q, zone.A

    if (
        A_remaining > _AREA_EPS_M2
        and state.phase in (WaterPhaseRegion.SATURATED_VAPOR, WaterPhaseRegion.TWO_PHASE)
        and T_sink < state.T_sat
    ):
        zone, state, A_remaining = _solve_condensation_zone(
            state=state,
            p=p,
            m_dot_total=m_dot_total,
            D_i=D_i,
            flow_area=flow_area,
            A_remaining=A_remaining,
            R_shared_per_Ai=R_shared_per_Ai,
            T_sink=T_sink,
        )
        zones.append(zone)
        warnings.extend(zone.warnings)
        Q_condensation, A_condensation = zone.Q, zone.A

    if (
        A_remaining > _AREA_EPS_M2
        and state.phase in (WaterPhaseRegion.SATURATED_LIQUID, WaterPhaseRegion.SUBCOOLED_LIQUID)
        and T_sink < state.T
    ):
        zone, state, A_remaining = _solve_sensible_zone(
            kind="subcooling",
            state=state,
            p=p,
            m_dot_total=m_dot_total,
            D_i=D_i,
            flow_area=flow_area,
            A_remaining=A_remaining,
            R_shared_per_Ai=R_shared_per_Ai,
            T_sink=T_sink,
            T_target=None,
            snap_quality_at_target=0.0,
        )
        zones.append(zone)
        warnings.extend(zone.warnings)
        Q_subcooling, A_subcooling = zone.Q, zone.A

    Q_total = Q_desuperheat + Q_condensation + Q_subcooling
    A_used = A_desuperheat + A_condensation + A_subcooling

    two_phase_dp_supported = not any(zone.kind == "condensation" for zone in zones)

    return InsideSteamZonesResult(
        phase_in=inlet.phase,
        phase_out=state.phase,
        T_in=inlet.T,
        T_out=state.T,
        p=p,
        h_in=inlet.h,
        h_out=state.h,
        quality_in=inlet.quality,
        quality_out=state.quality,
        T_sat=inlet.T_sat,
        Q_desuperheat=Q_desuperheat,
        Q_condensation=Q_condensation,
        Q_subcooling=Q_subcooling,
        Q_total=Q_total,
        A_desuperheat=A_desuperheat,
        A_condensation=A_condensation,
        A_subcooling=A_subcooling,
        A_total=A_inside_total,
        f_desuperheat=A_desuperheat / A_inside_total,
        f_condensation=A_condensation / A_inside_total,
        f_subcooling=A_subcooling / A_inside_total,
        zones=tuple(zones),
        m_dot_total=m_dot_total,
        two_phase_pressure_drop_supported=two_phase_dp_supported,
        warnings=warnings,
    )


def _isothermal_sink_effectiveness(NTU: float) -> float:
    """Effectiveness of a stream exchanging heat with an isothermal 0D bulk sink.

    This is the Cr -> 0 limit of the standard eps-NTU relations (the same
    limit ``core.heat_transfer.ntu.effectiveness_ntu`` reduces to for a
    vanishing capacity-rate ratio) -- independent of flow arrangement,
    since an isothermal sink has no spatial temperature profile for an
    arrangement to apply to.

    Ref: Incropera et al., Fundamentals of Heat and Mass Transfer (eps-NTU
    method).
    """
    return 1.0 - math.exp(-NTU)


def _solve_sensible_zone(
    *,
    kind: str,
    state: WaterSteamState,
    p: float,
    m_dot_total: float,
    D_i: float,
    flow_area: float,
    A_remaining: float,
    R_shared_per_Ai: float,
    T_sink: float,
    T_target: Optional[float],
    snap_quality_at_target: float,
) -> tuple[SteamZoneDiagnostics, WaterSteamState, float]:
    """Solve one single-phase (desuperheat or subcooling) zone.

    `T_target` is the zone's natural full-traversal boundary temperature
    (`T_sat`, for desuperheating) or `None` when there is no such natural
    boundary (subcooling: the zone always uses all of `A_remaining`).
    `snap_quality_at_target` (1.0 for desuperheat, 0.0 for subcooling) is
    which saturation-boundary state to snap onto when the solved outlet
    temperature lands within the T+p ambiguity band of `T_sat` -- which
    side of the dome a zone approaches `T_sat` from depends on the zone,
    so this must not be hardcoded to saturated vapor.
    """
    T_in_zone = state.T
    h_in_zone = state.h
    T_sat = state.T_sat

    T_rep = 0.5 * (T_in_zone + T_target) if T_target is not None else T_in_zone
    rep_props = water_steam_props_iapws97(T=T_rep, p=p)
    fluid_props = FluidProps(
        rho=rep_props.transport.rho,
        mu=rep_props.transport.mu,
        k=rep_props.transport.k,
        cp=rep_props.transport.cp,
    )
    _v, _Re, _Pr, alpha_inside, htc_warnings = heat_transfer_coefficient_internal(
        m_dot_total, D_i, flow_area, fluid_props
    )
    U = 1.0 / (1.0 / alpha_inside + R_shared_per_Ai)
    C_zone = m_dot_total * rep_props.transport.cp

    full_area: Optional[float] = None
    if T_target is not None:
        dT_in = T_in_zone - T_sink
        dT_out = T_target - T_sink
        if dT_out > 0.0:
            NTU_full = math.log(dT_in / dT_out)
            full_area = NTU_full * C_zone / U

    fully_traversed = bool(full_area is not None and full_area <= A_remaining)
    if fully_traversed:
        A_zone = full_area
        T_out_zone = T_target
    else:
        NTU = U * A_remaining / C_zone
        eps = _isothermal_sink_effectiveness(NTU)
        T_out_zone = T_in_zone - eps * (T_in_zone - T_sink)
        A_zone = A_remaining
        if abs(T_out_zone - T_sat) <= _SATURATION_SNAP_TOLERANCE_K:
            T_out_zone = T_sat
            fully_traversed = True

    if fully_traversed and abs(T_out_zone - T_sat) <= _SATURATION_SNAP_TOLERANCE_K:
        new_state = water_steam_state(p=p, x=snap_quality_at_target)
    else:
        new_state = water_steam_state(T=T_out_zone, p=p)

    Q_zone = m_dot_total * (h_in_zone - new_state.h)

    zone = SteamZoneDiagnostics(
        kind=kind,
        T_in=T_in_zone,
        T_out=new_state.T,
        h_in=h_in_zone,
        h_out=new_state.h,
        quality_in=None,
        quality_out=None,
        Q=Q_zone,
        A=A_zone,
        alpha_inside=alpha_inside,
        U=U,
        fully_traversed=fully_traversed,
        warnings=list(htc_warnings),
    )
    return zone, new_state, A_remaining - A_zone


def _solve_condensation_zone(
    *,
    state: WaterSteamState,
    p: float,
    m_dot_total: float,
    D_i: float,
    flow_area: float,
    A_remaining: float,
    R_shared_per_Ai: float,
    T_sink: float,
) -> tuple[SteamZoneDiagnostics, WaterSteamState, float]:
    x_in = state.quality
    T_sat = state.T_sat
    dT = T_sat - T_sink  # constant along the zone: both sides isothermal

    full = solve_inside_condensation_zone(
        p=p, x_in=x_in, x_out=0.0, m_dot_total=m_dot_total, D_i=D_i, flow_area=flow_area
    )
    U_full = 1.0 / (1.0 / full.alpha_condensation_effective + R_shared_per_Ai)
    A_full = full.Q_condensation / (U_full * dT)

    if A_full <= A_remaining:
        result = full
        A_zone = A_full
        U_zone = U_full
        fully_traversed = True
    else:
        x_out, result, U_zone = _solve_condensation_outlet_quality(
            x_in=x_in,
            p=p,
            m_dot_total=m_dot_total,
            D_i=D_i,
            flow_area=flow_area,
            A_target=A_remaining,
            dT=dT,
            R_shared_per_Ai=R_shared_per_Ai,
        )
        A_zone = A_remaining
        fully_traversed = False

    new_state = water_steam_state(p=p, x=result.x_out)

    zone = SteamZoneDiagnostics(
        kind="condensation",
        T_in=T_sat,
        T_out=T_sat,
        h_in=result.h_in,
        h_out=result.h_out,
        quality_in=result.x_in,
        quality_out=result.x_out,
        Q=result.Q_condensation,
        A=A_zone,
        alpha_inside=result.alpha_condensation_effective,
        U=U_zone,
        fully_traversed=fully_traversed,
        warnings=list(result.warnings),
    )
    return zone, new_state, A_remaining - A_zone


def _solve_condensation_outlet_quality(
    *,
    x_in: float,
    p: float,
    m_dot_total: float,
    D_i: float,
    flow_area: float,
    A_target: float,
    dT: float,
    R_shared_per_Ai: float,
    max_iterations: int = 60,
    quality_tolerance: float = 1e-9,
):
    """Bisect for the outlet quality that consumes exactly `A_target`.

    ``A(x_out) = Q(x_out) / (U(x_out) * dT)`` is monotonically increasing
    as `x_out` decreases from `x_in` towards 0 (more condensed => more
    duty => more area for a fixed, bounded HTC), so a single bounded
    bisection on `x_out in [0, x_in)` is well posed; this is the one
    non-closed-form solve in the multi-zone model, kept to a single
    bounded root find (not nested inside another root find).
    """

    def area_for(x_out: float) -> tuple[float, object, float]:
        result = solve_inside_condensation_zone(
            p=p, x_in=x_in, x_out=x_out, m_dot_total=m_dot_total, D_i=D_i, flow_area=flow_area
        )
        U = 1.0 / (1.0 / result.alpha_condensation_effective + R_shared_per_Ai)
        return result.Q_condensation / (U * dT), result, U

    lo, hi = 0.0, x_in
    area_lo, result_lo, U_lo = area_for(lo)
    # area(x_out) decreases monotonically to 0 as x_out -> x_in; area_lo is
    # the largest achievable (full traversal), already known > A_target by
    # construction (caller only reaches this branch when A_full > A_remaining).
    best_x, best_result, best_U = lo, result_lo, U_lo
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        area_mid, result_mid, U_mid = area_for(mid)
        best_x, best_result, best_U = mid, result_mid, U_mid
        if area_mid > A_target:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < quality_tolerance:
            break

    return best_x, best_result, best_U
