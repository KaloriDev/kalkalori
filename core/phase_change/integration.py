# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Simulation-side orchestration: capability -> regime -> (dry | wet) result.

This is the single place that wires ``core.phase_change.capability``,
``core.phase_change.regime`` and ``core.phase_change.outside_condensation_
solver`` together into the automatic behavior described in the v0.6.0 task
spec. ``core.models.bare_tube.BareTubeHeatExchanger.simulate`` calls
``apply_phase_change`` *after* running the ordinary sensible-only
``core.models.simulation.run_simulation`` (the dry baseline) -- so the dry
path in ``run_simulation`` itself is untouched, and every sensible-only
regression stays bit-for-bit stable (no capable side => this module hands
the dry result straight back, only stamping ``inside_phase_change``/
``outside_phase_change`` onto it via ``dataclasses.replace``).

Stable regime, single active side (spec sections 18-19)
-------------------------------------------------------
The regime for each capable side is decided exactly once, from the dry
baseline (``core.phase_change.regime.decide_regime``), before any wet
solve starts; there is no per-iteration dry/wet branch anywhere in this
package (the actual iteration lives entirely inside
``core.phase_change.outside_condensation_solver``, which is only ever
invoked once regime selection is already final). At most one side may be
an *active* condensing side per call:

- both sides possible under AUTO -> ``MultiplePhaseChangeSidesError``
  (``MULTIPLE_PHASE_CHANGE_SIDES_NOT_SUPPORTED``), checked first;
- only inside possible under AUTO -> ``InsideCondensationNotSupportedError``
  (``INSIDE_CONDENSATION_NOT_SUPPORTED``), condensation inside tubes is not
  solved in v0.6.0;
- only outside possible under AUTO -> the wet solver runs (the v0.6.0
  feature).

``PhaseChangeMode.DISABLED`` never contributes to either of the above: a
disabled side can be capable and possible without blocking the other side,
it simply reports ``active=False`` with a
``PHASE_CHANGE_DISABLED_BUT_POSSIBLE`` warning and leaves the sensible-only
numbers untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from core.common.warnings import ModelWarning, make_warning
from core.heat_transfer.thermal_iteration import (
    estimate_wall_temperature_envelope,
    solve_iterative_thermal_state,
)
from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props
from core.properties.averaging import mean_temperature
from core.heat_transfer.outside_flow import calculate_outside_tube_bank_hydraulics

from core.phase_change import warning_codes as WC
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.mass_heat_transfer import mass_transfer_coefficient  # noqa: F401 (re-export for callers)
from core.phase_change.outside_condensation_solver import (
    FrostingNotSupportedError,
    solve_outside_condensation,
)
from core.phase_change.regime import (
    decide_regime,
    representative_wall_temperature,
    validate_onset_settings,
)
from core.phase_change.types import PhaseChangeCapability, PhaseChangeDirection, PhaseChangeMode, PhaseChangeResult
from core.phase_change.water_equilibrium import is_frost_regime, water_dew_point, water_partial_pressure
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.wet_surface_fraction import estimate_wet_surface_fraction

DEFAULT_ONSET_TOLERANCE_K = 0.0
DEFAULT_ACTIVATION_BAND_K = 0.5


class InsideCondensationNotSupportedError(RuntimeError):
    """Raised when a dry baseline shows inside-tube condensation is possible
    under ``PhaseChangeMode.AUTO``. Inside condensation is not solved in
    v0.6.0 (see ``docs/roadmap.md``, v0.6.1)."""

    warning_code = WC.INSIDE_CONDENSATION_NOT_SUPPORTED


class MultiplePhaseChangeSidesError(RuntimeError):
    """Raised when both sides show an actionable (AUTO + possible) phase
    change simultaneously. At most one active phase-changing side is
    supported per call (see spec section 19)."""

    warning_code = WC.MULTIPLE_PHASE_CHANGE_SIDES_NOT_SUPPORTED


@dataclass(frozen=True)
class PhaseChangeSettings:
    """Bundled ``phase_change_*`` iteration/onset settings.

    Individual keyword parameters are threaded through
    ``BareTubeHeatExchanger.simulate``/``.rate`` (matching the existing
    thermal-iteration idiom, see ``core.heat_transfer.thermal_iteration``);
    this dataclass is only an internal convenience for passing the bundle
    around inside ``core.phase_change``.
    """

    onset_tolerance_K: float = DEFAULT_ONSET_TOLERANCE_K
    activation_band_K: float = DEFAULT_ACTIVATION_BAND_K
    lewis_number: float = 1.0
    max_iterations: int = 50
    temperature_tolerance_K: float = 0.05
    relative_Q_tolerance: float = 1e-4
    water_ratio_tolerance: float = 1e-6
    condensate_tolerance_kg_s: float = 1e-8
    wall_temperature_tolerance_K: float = 0.05
    relaxation_factor: float = 0.5

    def __post_init__(self) -> None:
        validate_onset_settings(self.onset_tolerance_K, self.activation_band_K)
        if not math.isfinite(self.lewis_number) or self.lewis_number <= 0.0:
            raise ValueError("lewis_number must be a positive finite value.")
        if self.max_iterations <= 0:
            raise ValueError("phase_change_max_iterations must be > 0.")
        if not math.isfinite(self.temperature_tolerance_K) or self.temperature_tolerance_K <= 0.0:
            raise ValueError("phase_change_temperature_tolerance_K must be > 0.")
        if not math.isfinite(self.relative_Q_tolerance) or self.relative_Q_tolerance <= 0.0:
            raise ValueError("phase_change_relative_Q_tolerance must be > 0.")
        if not math.isfinite(self.water_ratio_tolerance) or self.water_ratio_tolerance <= 0.0:
            raise ValueError("phase_change_water_ratio_tolerance must be > 0.")
        if not math.isfinite(self.condensate_tolerance_kg_s) or self.condensate_tolerance_kg_s < 0.0:
            raise ValueError("phase_change_condensate_tolerance_kg_s must be >= 0.")
        if not math.isfinite(self.wall_temperature_tolerance_K) or self.wall_temperature_tolerance_K <= 0.0:
            raise ValueError("phase_change_wall_temperature_tolerance_K must be > 0.")
        if not (0.0 < self.relaxation_factor <= 1.0):
            raise ValueError("phase_change_relaxation_factor must be in (0, 1].")


def check_single_active_side(
    inside_auto_possible: bool,
    outside_auto_possible: bool,
    *,
    iterate: bool,
) -> None:
    """Enforce "at most one active phase-changing side" (spec section 19)
    and the "outside condensation requires iterate=True" guard (section 21).

    Pure decision function (booleans in, exception or nothing out) so the
    priority ordering -- both-sides-possible takes precedence over
    inside-only-possible, which takes precedence over the iterate=False
    guard on outside -- can be unit-tested directly without needing a
    physically-engineered thermal scenario for every combination.
    """
    if inside_auto_possible and outside_auto_possible:
        raise MultiplePhaseChangeSidesError(
            "Both inside and outside show an active condensation tendency "
            "under PhaseChangeMode.AUTO in the dry baseline; at most one "
            "active phase-changing side is supported in v0.6.0. Set "
            "phase_change_mode=DISABLED on one side, or investigate the "
            "operating point."
        )
    if inside_auto_possible:
        raise InsideCondensationNotSupportedError(
            "The dry baseline shows the inside-tube wall running below the "
            "inside stream's water dew point under PhaseChangeMode.AUTO. "
            "Condensation inside tubes is not supported in v0.6.0 (see "
            "docs/roadmap.md, v0.6.1). Set phase_change_mode=DISABLED on "
            "the inside side to accept a sensible-only approximation."
        )
    if outside_auto_possible and not iterate:
        raise ValueError(
            "Outside water condensation was detected as possible under "
            "PhaseChangeMode.AUTO, but iterate=False was requested. A "
            "single-pass sensible-only approximation cannot represent "
            "condensation; call .simulate(..., iterate=True), or set "
            "outside.phase_change_mode=DISABLED to accept a sensible-only "
            "result."
        )


def _capability_only_result(side: str, mode: PhaseChangeMode, capability: PhaseChangeCapability) -> PhaseChangeResult:
    return PhaseChangeResult(
        side=side,
        mode=mode,
        direction=PhaseChangeDirection.NONE,
        component=capability.component,
        capable=capability.capable,
        possible=False,
        active=False,
        W_in=capability.W_in,
        W_out=capability.W_in,
        assumptions=("sensible_only_no_phase_change",) if capability.capable else (),
    )


def _dew_point_for(capability: PhaseChangeCapability, *, p: float) -> float | None:
    if not capability.capable or capability.W_in is None:
        return None
    from core.phase_change.water_equilibrium import water_mole_fraction_from_ratio

    y_h2o = water_mole_fraction_from_ratio(capability.W_in, M_dry=capability.M_dry, M_h2o=capability.M_condensable)
    p_h2o = water_partial_pressure(y_h2o, p)
    if is_frost_regime(p_h2o):
        return None
    return water_dew_point(p_h2o)


def apply_phase_change(
    hx,
    inside,
    outside,
    dry_result,
    *,
    iterate: bool,
    euler_provider: str = "zukauskas",
    settings: PhaseChangeSettings | None = None,
):
    """Return a new ``HXSimulationResult`` with phase-change results applied.

    Args:
        hx: ``BareTubeHeatExchanger``.
        inside, outside: the ``HXSideInput`` passed to ``.simulate()``.
        dry_result: the already-computed sensible-only ``HXSimulationResult``
            (the dry baseline).
        iterate: the ``iterate`` flag ``.simulate()`` was called with (needed
            for the "outside condensation requires iterate=True" guard).
        settings: bundled ``phase_change_*`` settings; defaults if omitted.

    Raises:
        InsideCondensationNotSupportedError, MultiplePhaseChangeSidesError,
        ValueError (AUTO + possible + iterate=False), FrostingNotSupportedError.
    """
    settings = settings or PhaseChangeSettings()

    inside_capability = detect_phase_change_capability(inside.provider)
    outside_capability = detect_phase_change_capability(outside.provider)

    if not inside_capability.capable and not outside_capability.capable:
        return replace(
            dry_result,
            inside_phase_change=_capability_only_result("inside", inside.phase_change_mode, inside_capability),
            outside_phase_change=_capability_only_result("outside", outside.phase_change_mode, outside_capability),
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

    check_single_active_side(inside_auto_possible, outside_auto_possible, iterate=iterate)

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

    # --- Outside condensation is the active, supported path -----------------
    p_h2o_in = water_partial_pressure(
        _y_h2o(outside_capability), outside.p
    )
    if is_frost_regime(p_h2o_in):
        outside_result = replace(
            _build_capability_side_result(
                side="outside", mode=outside.phase_change_mode, capability=outside_capability,
                possible=True, near_onset=False, dew_point=None, p=outside.p,
            ),
            warnings=(
                make_warning(
                    code=WC.FROSTING_NOT_SUPPORTED,
                    message=(
                        "outside: the equilibrium dew point for the inlet "
                        "water content is at/below the water triple point "
                        "(frost/ice regime); liquid condensation is not "
                        "applicable and frosting is not modelled in v0.6.0. "
                        "Returning the sensible-only dry baseline."
                    ),
                    source="phase_change_integration",
                    severity="warning",
                ),
            ),
        )
        return replace(dry_result, inside_phase_change=inside_result, outside_phase_change=outside_result)

    m_dot_dry_carrier = outside.m_dot / (1.0 + outside_capability.W_in)

    try:
        solution = solve_outside_condensation(
            hx,
            inside_provider=inside.provider,
            m_dot_inside=inside.m_dot,
            T_in_inside=inside.T_in,
            p_inside=inside.p,
            outside_capability=outside_capability,
            m_dot_dry_carrier=m_dot_dry_carrier,
            T_in_outside=outside.T_in,
            p_outside=outside.p,
            T_out_inside_init=dry_result.T_out_inside,
            T_out_outside_init=dry_result.T_out_outside,
            euler_provider=euler_provider,
            lewis_number=settings.lewis_number,
            max_iterations=settings.max_iterations,
            temperature_tolerance_K=settings.temperature_tolerance_K,
            relative_Q_tolerance=settings.relative_Q_tolerance,
            water_ratio_tolerance=settings.water_ratio_tolerance,
            condensate_tolerance_kg_s=settings.condensate_tolerance_kg_s,
            wall_temperature_tolerance_K=settings.wall_temperature_tolerance_K,
            relaxation_factor=settings.relaxation_factor,
        )
    except FrostingNotSupportedError as exc:
        outside_result = replace(
            _build_capability_side_result(
                side="outside", mode=outside.phase_change_mode, capability=outside_capability,
                possible=True, near_onset=False, dew_point=outside_dew_point, p=outside.p,
            ),
            warnings=(
                make_warning(
                    code=WC.FROSTING_NOT_SUPPORTED,
                    message=f"outside: {exc}",
                    source="phase_change_integration",
                    severity="warning",
                ),
            ),
        )
        return replace(dry_result, inside_phase_change=inside_result, outside_phase_change=outside_result)

    m_dot_water_vapor_in = m_dot_dry_carrier * outside_capability.W_in
    m_dot_water_vapor_out = m_dot_dry_carrier * solution.W_out
    m_dot_gas_in = outside.m_dot
    m_dot_gas_out = m_dot_dry_carrier + m_dot_water_vapor_out

    mass_balance_error = m_dot_water_vapor_in - (m_dot_water_vapor_out + solution.m_dot_condensate)
    energy_balance_error = solution.Q_total - (solution.Q_sensible + solution.Q_latent)

    wet_surface = estimate_wet_surface_fraction(
        hx,
        inside_provider=inside.provider,
        outside_capability=outside_capability,
        m_dot_inside=inside.m_dot,
        m_dot_dry_carrier=m_dot_dry_carrier,
        T_in_inside=inside.T_in,
        T_out_inside=solution.T_out_inside,
        T_in_outside=outside.T_in,
        T_out_outside=solution.T_out_outside,
        W_in=outside_capability.W_in,
        W_out=solution.W_out,
        p_inside=inside.p,
        p_outside=outside.p,
        euler_provider=euler_provider,
        activation_band_K=settings.activation_band_K,
    )

    warnings_list: list[ModelWarning] = list(solution.warnings) + list(wet_surface.warnings)
    warnings_list.append(
        make_warning(
            code=WC.OUTSIDE_CONDENSATION_DETECTED,
            message=(
                "outside: the dry sensible-only baseline showed the outside "
                "tube-wall surface running below the water dew point; "
                "partial H2O condensation was solved for this call."
            ),
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list.append(
        make_warning(
            code=WC.LEWIS_NUMBER_ASSUMED,
            message=(
                f"outside: mass transfer used the Chilton-Colburn analogy "
                f"with lewis_number={settings.lewis_number:g} (a configurable "
                "first-model assumption, not a universal constant)."
            ),
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list.append(
        make_warning(
            code=WC.FULLY_DRAINED_CONDENSATE_ASSUMED,
            message=(
                "outside: condensate is assumed to be fully drained from the "
                "gas stream, leaving as saturated liquid at the representative "
                "interface temperature. Film retention/re-entrainment are not "
                "modelled (v0.6.0)."
            ),
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list.append(
        make_warning(
            code=WC.CONDENSATE_FILM_RESISTANCE_NOT_MODELLED,
            message="outside: condensate film thermal resistance is not modelled in v0.6.0.",
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list.append(
        make_warning(
            code=WC.CONDENSATE_FILM_HYDRAULICS_NOT_MODELLED,
            message="outside: condensate film hydraulics are not modelled in v0.6.0.",
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list.append(
        make_warning(
            code=WC.REENTRAINMENT_NOT_MODELLED,
            message="outside: droplet carryover / re-entrainment is not modelled in v0.6.0.",
            source="phase_change_integration",
            severity="info",
        )
    )
    warnings_list = _deduplicate_warnings(warnings_list)

    outside_result = PhaseChangeResult(
        side="outside",
        mode=outside.phase_change_mode,
        direction=PhaseChangeDirection.CONDENSATION,
        component=outside_capability.component,
        capable=True,
        possible=True,
        active=True,
        converged=solution.converged,
        iterations=solution.iterations,
        method="outside_condensation_0d_bulk_mean",
        W_in=outside_capability.W_in,
        W_out=solution.W_out,
        m_dot_dry_carrier=m_dot_dry_carrier,
        m_dot_water_vapor_in=m_dot_water_vapor_in,
        m_dot_water_vapor_out=m_dot_water_vapor_out,
        m_dot_condensate=solution.m_dot_condensate,
        m_dot_gas_in=m_dot_gas_in,
        m_dot_gas_out=m_dot_gas_out,
        dew_point_in=outside_dew_point,
        dew_point_out=_dew_point_at_ratio(outside_capability, solution.W_out, p=outside.p),
        wall_temperature_mean=solution.T_wall_outside,
        wall_temperature_min=wet_surface.wall_temperature_min,
        wall_temperature_max=wet_surface.wall_temperature_max,
        wet_surface_fraction=wet_surface.wet_surface_fraction,
        wet_surface_fraction_method=wet_surface.method,
        alfa_dry=solution.alfa_o_dry,
        alfa_effective=solution.alfa_o_effective,
        lewis_number=settings.lewis_number,
        Q_sensible=solution.Q_sensible,
        Q_latent=solution.Q_latent,
        Q_total=solution.Q_total,
        mass_balance_error=mass_balance_error,
        energy_balance_error=energy_balance_error,
        residuals=dict(solution.residuals),
        assumptions=(
            "bulk_mean_property_evaluation_0d",
            "fully_drained_liquid_condensate",
            "lewis_number_chilton_colburn_analogy",
            "dry_gas_composition_unchanged_by_condensation",
        ),
        warnings=tuple(warnings_list),
    )

    T_mean_inside = mean_temperature(inside.T_in, solution.T_out_inside)
    T_mean_outside = mean_temperature(outside.T_in, solution.T_out_outside)

    outside_provider_final = wet_gas_provider_at_water_ratio(
        outside_capability, 0.5 * (outside_capability.W_in + solution.W_out)
    )

    # hot_stream/cold_stream only feed this snapshot's OWN (non-authoritative,
    # sensible-only) Q/T_hot_out/T_cold_out sub-fields -- hydraulics below use
    # the explicit *_temperature_out arguments (the wet solver's converged,
    # latent-inclusive outlet temperatures) directly, matching how the
    # existing dry `run_simulation` final snapshot already treats this
    # sub-result as diagnostic-only (see HXSimulationResult docstring).
    hot_is_inside = inside.T_in >= outside.T_in
    from core.heat_transfer.streams import SensibleHeatStream

    inside_stream = SensibleHeatStream(
        C=inside.m_dot * solution.inside_bulk_props.cp, T_in=inside.T_in
    )
    outside_stream = SensibleHeatStream(
        C=m_dot_dry_carrier
        * (1.0 + 0.5 * (outside_capability.W_in + solution.W_out))
        * solution.outside_bulk_props.cp,
        T_in=outside.T_in,
    )
    hot_stream, cold_stream = (
        (inside_stream, outside_stream) if hot_is_inside else (outside_stream, inside_stream)
    )

    final_result = hx.solve(
        hot_stream=hot_stream,
        cold_stream=cold_stream,
        m_dot_tube_side=inside.m_dot,
        tube_side_props=to_internal_fluid_props(solution.inside_bulk_props),
        tube_side_provider=inside.provider,
        tube_side_temperature_in=inside.T_in,
        tube_side_temperature_out=solution.T_out_inside,
        tube_side_pressure=inside.p,
        m_dot_outside=m_dot_dry_carrier * (1.0 + 0.5 * (outside_capability.W_in + solution.W_out)),
        outside_props=to_outside_fluid_props(solution.outside_bulk_props),
        outside_provider=outside_provider_final,
        outside_temperature_in=outside.T_in,
        outside_temperature_out=solution.T_out_outside,
        outside_pressure=outside.p,
        flow_arrangement=None,
        euler_provider=euler_provider,
    )

    # Refresh the outside hydraulic snapshot using the actual inlet/outlet
    # *gas-phase* mass flow (spec section 29): m_dot_gas_in > m_dot_gas_out
    # once condensate has been removed. Reynolds/Euler-provider dispatch and
    # the drag integral keep using the bulk-mean reference mass flow (same
    # convention as the dry driver); only the reported per-point face
    # velocity/flux and the signed acceleration term reflect the remaining
    # gas phase at each end -- the condensate is never added to this
    # gas-phase basis. Mirrors the same hydraulics-refresh pattern
    # ``core.models.simulation.run_simulation`` already uses after its own
    # thermal pass determines the actual outlet temperature.
    from dataclasses import replace as _replace

    from core.pressure_drop.flow_path import build_outside_pressure_drop_result

    outside_bank_hydraulic = calculate_outside_tube_bank_hydraulics(
        m_dot=m_dot_dry_carrier * (1.0 + 0.5 * (outside_capability.W_in + solution.W_out)),
        face_area=hx.bundle.frontal_flow_area,
        tube_outer_diameter=float(getattr(hx.bundle.tube, "D_o")),
        tube_pitch_transverse=hx.bundle.pitch_transverse,
        tube_pitch_longitudinal=hx.bundle.pitch_longitudinal,
        layout=hx.bundle.layout,
        n_rows=hx.bundle.n_rows,
        n_tubes_per_row=hx.bundle.n_tubes_per_row,
        provider=outside_provider_final,
        temperature_in=outside.T_in,
        temperature_out=solution.T_out_outside,
        pressure=outside.p,
        euler_provider=euler_provider,
        m_dot_inlet=m_dot_gas_in,
        m_dot_outlet=m_dot_gas_out,
    )
    final_result = _replace(
        final_result,
        outside_side_hydraulic=_replace(
            final_result.outside_side_hydraulic,
            dp_total=outside_bank_hydraulic.dp_total,
            Re=outside_bank_hydraulic.midpoint.reynolds,
            v=outside_bank_hydraulic.midpoint.face_velocity,
            tube_bank=outside_bank_hydraulic,
        ),
        outside_side_pressure_drop=_replace(
            final_result.outside_side_pressure_drop,
            tube_bank=outside_bank_hydraulic,
            flow_path=build_outside_pressure_drop_result(outside_bank_hydraulic),
        ),
    )

    envelope_wet = estimate_wall_temperature_envelope(
        hx,
        m_dot_inside=inside.m_dot,
        m_dot_outside=m_dot_dry_carrier * (1.0 + 0.5 * (outside_capability.W_in + solution.W_out)),
        inside_provider=inside.provider,
        outside_provider=outside_provider_final,
        inside_inlet_temperature=inside.T_in,
        inside_outlet_temperature=solution.T_out_inside,
        outside_inlet_temperature=outside.T_in,
        outside_outlet_temperature=solution.T_out_outside,
        p_inside=inside.p,
        p_outside=outside.p,
        euler_provider=euler_provider,
    )

    return replace(
        dry_result,
        converged=solution.converged,
        iterations=solution.iterations,
        T_mean_inside=T_mean_inside,
        T_mean_outside=T_mean_outside,
        inside_alfa_mean=solution.alfa_i,
        outside_alfa_mean=solution.alfa_o_dry,
        U_mean=solution.U_effective,
        UA=solution.UA_effective,
        q=solution.Q_total,
        T_out_inside=solution.T_out_inside,
        T_out_outside=solution.T_out_outside,
        Q_full=solution.Q_total,
        Q_derated=solution.Q_total,
        final_result=final_result,
        thermal_state=dry_result.thermal_state,
        wall_temperature_envelope=envelope_wet,
        inside_phase_change=inside_result,
        outside_phase_change=outside_result,
    )


def _y_h2o(capability: PhaseChangeCapability) -> float:
    from core.phase_change.water_equilibrium import water_mole_fraction_from_ratio

    return water_mole_fraction_from_ratio(capability.W_in, M_dry=capability.M_dry, M_h2o=capability.M_condensable)


def _dew_point_at_ratio(capability: PhaseChangeCapability, W: float, *, p: float) -> float | None:
    from core.phase_change.water_equilibrium import water_mole_fraction_from_ratio

    if W <= 0.0:
        return None
    y = water_mole_fraction_from_ratio(W, M_dry=capability.M_dry, M_h2o=capability.M_condensable)
    p_h2o = water_partial_pressure(y, p)
    if is_frost_regime(p_h2o):
        return None
    return water_dew_point(p_h2o)


def _build_capability_side_result(
    *,
    side: str,
    mode: PhaseChangeMode,
    capability: PhaseChangeCapability,
    possible: bool,
    near_onset: bool,
    dew_point: float | None,
    p: float,
) -> PhaseChangeResult:
    warnings: list[ModelWarning] = []
    if near_onset:
        warnings.append(
            make_warning(
                code=WC.PHASE_CHANGE_NEAR_ONSET,
                message=(
                    f"{side}: the dry-baseline wall temperature is within the "
                    "phase-change activation band of the dew point; the "
                    "regime is held at sensible-only (dry) to avoid "
                    "oscillating between dry and wet solves near onset."
                ),
                source="phase_change_integration",
                severity="warning",
            )
        )
    if possible and mode is PhaseChangeMode.DISABLED:
        warnings.append(
            make_warning(
                code=WC.PHASE_CHANGE_DISABLED_BUT_POSSIBLE,
                message=(
                    f"{side}: phase_change_mode=DISABLED forced a sensible-"
                    "only result, but the dry baseline shows the wall "
                    "temperature running below the dew point -- condensation "
                    "would be thermodynamically possible here. This result "
                    "is a forced single-phase approximation: it does not "
                    "remove condensate from the stream, and the true heat "
                    "duty and outlet temperature may differ."
                ),
                source="phase_change_integration",
                severity="warning",
            )
        )
    return PhaseChangeResult(
        side=side,
        mode=mode,
        direction=PhaseChangeDirection.NONE,
        component=capability.component,
        capable=capability.capable,
        possible=possible,
        active=False,
        W_in=capability.W_in,
        W_out=capability.W_in,
        m_dot_condensate=0.0,
        dew_point_in=dew_point,
        dew_point_out=dew_point,
        Q_sensible=0.0,
        Q_latent=0.0,
        Q_total=0.0,
        assumptions=("sensible_only_no_phase_change",) if capability.capable else (),
        warnings=tuple(warnings),
    )


def _deduplicate_warnings(warnings: list[ModelWarning]) -> list[ModelWarning]:
    unique: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        identity = (warning.source, warning.code)
        if identity not in seen:
            seen.add(identity)
            unique.append(warning)
    return unique


