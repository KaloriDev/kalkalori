# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Pressure-drop stage/group/path result structures and flow-path aggregation
(v0.5.6 architecture).

This module defines the generic, immutable result structures shared by both
exchanger sides (stage, stage-group, complete-path) and the functions that
aggregate the existing tube-bundle / tube-bank calculated results together
with explicitly calculated local-loss stages into a single
``PressureDropPathResult``.

Pressure-drop semantics
-----------------------
The local-path API keeps three physically distinct quantities separate:

``dp_irreversible``
    Non-negative mechanical-energy (total-pressure) loss.
``delta_dynamic_pressure``
    Signed dynamic-pressure change, ``q_downstream - q_upstream``.
``dp_static``
    Signed static-pressure difference, ``p_upstream - p_downstream``, equal
    to ``dp_irreversible + delta_dynamic_pressure``.

Only irreversible losses are aggregated into ``PressureDropPathResult``'s
``dp_core``, ``dp_local`` and ``dp_total``. Static-pressure differences are
reported separately and may be negative for pressure-recovering diffusers.

Two calculation paths (v0.5.6 local pressure-drop paths)
----------------------------------------------------------
``build_tube_side_pressure_drop_result``/``build_outside_pressure_drop_
result`` are geometry-only: they carry no flow state, so every local-loss
stage they build is a zero ``not_implemented`` placeholder (except a
``UserDefinedPressureDropGeometry`` supplying a fixed ``pressure_drop``
directly). These are the functions the standard exchanger solver
(``BareTubeHeatExchanger.solve``/``simulate``/``rate``) uses, always with
``path=None``, so the standard solver's ``dp_local`` is always exactly
``0.0`` and never evaluates any local-loss geometry or model.

``calculate_pressure_drop_assembly``/``calculate_tube_side_pressure_drop_
path``/``calculate_outside_pressure_drop_path`` are the explicitly invoked
real local-loss calculation path: given an explicit ``PressureDropFlowState``
per group, they dispatch each stage geometry to its production model (see
``core.pressure_drop.straight_sections``, ``area_changes``,
``direction_changes``, ``screens``) and are never called by the standard
solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from core.common.warnings import ModelWarning, make_warning
from core.geometry.pressure_drop_stages import (
    AreaChangeGeometry,
    ChamberGeometry,
    DirectionChangeGeometry,
    FlowSectionGeometry,
    PressureDropAssemblyGeometry,
    PressureDropStageGeometry,
    ScreenGeometry,
    StraightSectionGeometry,
    UserDefinedPressureDropGeometry,
)
from core.geometry.outside_pressure_drop_path import (
    OutsidePressureDropDesignRequest,
    SpecifiedOutsidePressureDropPath,
)
from core.geometry.tube_side_pressure_drop_path import (
    SpecifiedTubeSidePressureDropPath,
    TubeSidePressureDropDesignRequest,
    validate_specified_tube_side_path,
)
from core.pressure_drop.internal_pressure_drop import TubeBundleHydraulicResult
from core.properties.common import FluidTransportProperties

if TYPE_CHECKING:
    # Type-hint only: avoids a runtime dependency of core.pressure_drop on
    # core.heat_transfer. The tube-bank core stage is built from this
    # result's numeric fields via plain attribute access (duck-typed).
    from core.heat_transfer.outside_flow import OutsideTubeBankHydraulicResult
    from core.pressure_drop.finned_tube_pressure_drop import (
        FinnedTubeBankHydraulicResult,
    )


# ---------------------------------------------------------------------------
# Explicit pressure-drop operating state and reusable section-flow helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PressureDropFlowState:
    """A single explicit local-pressure-drop operating state.

    The front-end/application layer obtains ``props`` from an existing
    property provider and passes the result in here; local-loss calculation
    functions never perform their own property lookup (see the module docs
    for ``core.pressure_drop.area_changes``, ``direction_changes``,
    ``screens``, ``straight_sections``).

    Units: ``mass_flow`` [kg/s], ``temperature`` [K], ``pressure`` [Pa].
    """

    mass_flow: float
    temperature: float
    pressure: float
    props: FluidTransportProperties

    def __post_init__(self) -> None:
        for name, value in (
            ("mass_flow", self.mass_flow),
            ("temperature", self.temperature),
            ("pressure", self.pressure),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"PressureDropFlowState.{name} must be a positive, finite value."
                )
        # FluidTransportProperties.__post_init__ already guarantees rho, mu,
        # k, cp are finite and positive; no re-validation needed here.


@dataclass(frozen=True)
class SectionFlowResult:
    """Mass flux, velocity, dynamic pressure and Reynolds number at one
    ``FlowSectionGeometry`` (or any object exposing the same ``flow_area``/
    ``hydraulic_diameter`` attributes, e.g. ``StraightSectionGeometry``) for
    a given ``PressureDropFlowState``."""

    mass_flux: float
    velocity: float
    dynamic_pressure: float
    reynolds: float


def evaluate_section_flow(
    state: PressureDropFlowState,
    section: FlowSectionGeometry,
) -> SectionFlowResult:
    """Reusable mass-flux / velocity / dynamic-pressure / Reynolds
    calculation for a specified cross-section and flow state:

        G = mass_flow / A
        V = G / rho
        q = rho * V**2 / 2
        Re = rho * V * D_h / mu  (== G * D_h / mu)

    ``section`` may be any of ``CircularFlowSection``, ``RectangularFlow
    Section``, ``CustomFlowSection`` -- or any other object duck-typed with
    ``flow_area``/``hydraulic_diameter`` attributes, such as
    ``StraightSectionGeometry``.
    """
    area = section.flow_area
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("evaluate_section_flow: section.flow_area must be a positive, finite value.")
    hydraulic_diameter = section.hydraulic_diameter
    if not math.isfinite(hydraulic_diameter) or hydraulic_diameter <= 0.0:
        raise ValueError(
            "evaluate_section_flow: section.hydraulic_diameter must be a positive, finite value."
        )

    rho = state.props.rho
    mu = state.props.mu

    mass_flux = state.mass_flow / area
    velocity = mass_flux / rho
    dynamic_pressure = rho * velocity ** 2 / 2.0
    reynolds = mass_flux * hydraulic_diameter / mu

    return SectionFlowResult(
        mass_flux=mass_flux,
        velocity=velocity,
        dynamic_pressure=dynamic_pressure,
        reynolds=reynolds,
    )


# ---------------------------------------------------------------------------
# Stage status
# ---------------------------------------------------------------------------

class PressureDropStageStatus(str, Enum):
    """Status of a single pressure-drop stage result.

    ``CALCULATED``: a real physical calculation was performed.
    ``USER_DEFINED``: a directly user-supplied value was used verbatim.
    ``NOT_IMPLEMENTED``: the stage exists in the flow path, but no
        calculation method is available yet; reported pressure drop is
        zero, and this status makes explicit that the zero is a placeholder,
        not a completed physical result.
    ``NOT_APPLICABLE``: the stage/group does not apply to this path.
    """

    CALCULATED = "calculated"
    USER_DEFINED = "user_defined"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PressureDropStageResult:
    """Result of a single pressure-drop stage.

    ``dp_irreversible`` is the non-negative total-pressure loss and
    the quantity used for hydraulic-loss aggregation.
    ``delta_dynamic_pressure`` is the signed ``q_downstream - q_upstream``:
    normally negative in a diffuser and positive in a contraction. The
    signed static-pressure difference ``p_upstream - p_downstream`` is
    exposed by the derived ``dp_static`` property and may be negative when
    static pressure is recovered.

    When ``loss_coefficient`` is populated, ``reference_dynamic_pressure``
    is the reference ``q`` used in ``dp_irreversible = K * q``. It is a
    diagnostic, not another pressure-loss component to add. ``method`` is
    ``None`` for stages that were not calculated. A future method may return
    pressure drop directly rather than through a loss coefficient; this
    structure does not assume every stage is described only by a ``K``
    coefficient.
    """

    stage_id: str
    stage_type: str
    status: PressureDropStageStatus

    dp_irreversible: float
    delta_dynamic_pressure: float

    method: str | None
    warnings: tuple[ModelWarning, ...]

    # Extended per-stage diagnostics (v0.5.6 local pressure-drop paths).
    # Not every stage populates every field -- e.g. an area change has no
    # friction factor, a straight section has no loss_coefficient. A
    # calculated stage exposes enough of these to independently understand
    # the source of its result without recomputing it.
    loss_coefficient: float | None = None

    reference_area: float | None = None
    reference_velocity: float | None = None
    reference_dynamic_pressure: float | None = None

    upstream_area: float | None = None
    downstream_area: float | None = None

    upstream_velocity: float | None = None
    downstream_velocity: float | None = None

    reynolds: float | None = None
    friction_factor: float | None = None
    friction_factor_method: str | None = None

    relative_roughness: float | None = None

    open_area_ratio: float | None = None
    blockage_ratio: float | None = None
    open_area_velocity: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.dp_irreversible):
            raise ValueError(
                "PressureDropStageResult.dp_irreversible must be finite."
            )
        if self.dp_irreversible < 0.0:
            raise ValueError(
                "PressureDropStageResult.dp_irreversible must be non-negative."
            )
        if not math.isfinite(self.delta_dynamic_pressure):
            raise ValueError(
                "PressureDropStageResult.delta_dynamic_pressure must be finite."
            )
        if not math.isfinite(self.dp_static):
            raise ValueError("PressureDropStageResult.dp_static must be finite.")

    @property
    def dp_static(self) -> float:
        """Signed ``p_upstream - p_downstream``.

        Equal to ``dp_irreversible + delta_dynamic_pressure``; it may be
        negative when a diffuser's static-pressure recovery exceeds its
        irreversible loss.
        """
        return self.dp_irreversible + self.delta_dynamic_pressure


@dataclass(frozen=True)
class PressureDropStageGroupResult:
    """An ordered group of stage results (e.g. ``inlet``, ``return_1``).

    Irreversible loss and dynamic-pressure change are accumulated
    independently. ``dp_static`` is their signed sum and is not the group's
    hydraulic-resistance quantity.
    """

    group_id: str
    stages: tuple[PressureDropStageResult, ...]

    @property
    def dp_irreversible(self) -> float:
        return sum(stage.dp_irreversible for stage in self.stages)

    @property
    def delta_dynamic_pressure(self) -> float:
        return sum(stage.delta_dynamic_pressure for stage in self.stages)

    @property
    def dp_static(self) -> float:
        return self.dp_irreversible + self.delta_dynamic_pressure


@dataclass(frozen=True)
class PressureDropPathResult:
    """Complete-path pressure-drop result: stage groups plus aggregation.

    ``dp_core``, ``dp_local`` and ``dp_total`` are exclusively irreversible
    total-pressure losses. Signed dynamic-pressure changes and signed
    static-pressure differences are exposed separately for the core, local
    stages and complete path.
    """

    groups: tuple[PressureDropStageGroupResult, ...]

    dp_core: float
    dp_local: float

    delta_dynamic_pressure_core: float
    delta_dynamic_pressure_local: float

    warnings: tuple[ModelWarning, ...]

    @property
    def dp_total(self) -> float:
        """Total irreversible loss: ``dp_core + dp_local``."""
        return self.dp_core + self.dp_local

    @property
    def delta_dynamic_pressure_total(self) -> float:
        return (
            self.delta_dynamic_pressure_core
            + self.delta_dynamic_pressure_local
        )

    @property
    def dp_static_core(self) -> float:
        return self.dp_core + self.delta_dynamic_pressure_core

    @property
    def dp_static_local(self) -> float:
        return self.dp_local + self.delta_dynamic_pressure_local

    @property
    def dp_static_total(self) -> float:
        return self.dp_total + self.delta_dynamic_pressure_total

    def group(self, group_id: str) -> PressureDropStageGroupResult | None:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        return None


# ---------------------------------------------------------------------------
# Placeholder (not_implemented) stage construction
# ---------------------------------------------------------------------------

def _stage_type_name(geometry: PressureDropStageGeometry) -> str:
    # Local imports avoid making the common geometry layer depend on its
    # correlation-specific geometry subclasses.
    from core.pressure_drop.direction_changes import (
        CircularElbowGeometry,
        RectangularElbowGeometry,
    )
    from core.pressure_drop.screens import FlatObstructionGeometry

    if isinstance(geometry, StraightSectionGeometry):
        return "straight_section"
    if isinstance(geometry, AreaChangeGeometry):
        return f"{geometry.change_type.value}_area_change"
    if isinstance(geometry, ScreenGeometry):
        return geometry.screen_type.value
    if isinstance(geometry, DirectionChangeGeometry):
        return geometry.change_type.value
    if isinstance(geometry, ChamberGeometry):
        return "chamber"
    if isinstance(geometry, UserDefinedPressureDropGeometry):
        return "user_defined"
    if isinstance(geometry, CircularElbowGeometry):
        return "circular_elbow"
    if isinstance(geometry, RectangularElbowGeometry):
        return "rectangular_elbow"
    if isinstance(geometry, FlatObstructionGeometry):
        return f"flat_obstruction_{geometry.obstruction_type.value}"
    raise TypeError(f"Unsupported pressure-drop stage geometry: {geometry!r}")


def _build_stage_result(
    group_id: str,
    index: int,
    geometry: PressureDropStageGeometry,
) -> PressureDropStageResult:
    stage_type = _stage_type_name(geometry)
    stage_id = f"{group_id}_{index}_{stage_type}"

    if (
        isinstance(geometry, UserDefinedPressureDropGeometry)
        and geometry.pressure_drop is not None
        and geometry.loss_coefficient is not None
    ):
        raise ValueError(
            "user_defined_pressure_drop_ambiguous: supply either pressure_drop "
            "or loss_coefficient (+ reference_area), not both."
        )

    if isinstance(geometry, UserDefinedPressureDropGeometry) and geometry.pressure_drop is not None:
        dp = float(geometry.pressure_drop)
        return PressureDropStageResult(
            stage_id=stage_id,
            stage_type=stage_type,
            status=PressureDropStageStatus.USER_DEFINED,
            dp_irreversible=dp,
            delta_dynamic_pressure=0.0,
            method="user_defined_fixed_dp",
            warnings=(),
        )

    warning = make_warning(
        code="pressure_drop_stage_not_implemented",
        message=(
            f"pressure_drop_flow_path: stage '{stage_id}' ({stage_type}) has "
            "no implemented calculation method yet; reporting a zero "
            "placeholder, not a completed physical result."
        ),
        source="pressure_drop_flow_path",
        severity="info",
    )
    return PressureDropStageResult(
        stage_id=stage_id,
        stage_type=stage_type,
        status=PressureDropStageStatus.NOT_IMPLEMENTED,
        dp_irreversible=0.0,
        delta_dynamic_pressure=0.0,
        method=None,
        warnings=(warning,),
    )


def _group_from_assembly(
    group_id: str,
    assembly: PressureDropAssemblyGeometry | None,
) -> PressureDropStageGroupResult:
    if assembly is None:
        return PressureDropStageGroupResult(group_id=group_id, stages=())
    stages = tuple(
        _build_stage_result(group_id, index, geometry)
        for index, geometry in enumerate(assembly.stages)
    )
    return PressureDropStageGroupResult(group_id=group_id, stages=stages)


def _deduplicate_warnings(warnings: list[ModelWarning]) -> tuple[ModelWarning, ...]:
    unique: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        identity = (warning.source, warning.code)
        if identity not in seen:
            seen.add(identity)
            unique.append(warning)
    return tuple(unique)


def _assemble_path_result(
    groups: tuple[PressureDropStageGroupResult, ...],
    *,
    core_group_id: str,
) -> PressureDropPathResult:
    dp_core = 0.0
    dp_local = 0.0
    delta_dynamic_pressure_core = 0.0
    delta_dynamic_pressure_local = 0.0
    warnings: list[ModelWarning] = []
    for group in groups:
        for stage in group.stages:
            warnings.extend(stage.warnings)
        if group.group_id == core_group_id:
            dp_core += group.dp_irreversible
            delta_dynamic_pressure_core += group.delta_dynamic_pressure
        else:
            dp_local += group.dp_irreversible
            delta_dynamic_pressure_local += group.delta_dynamic_pressure
    return PressureDropPathResult(
        groups=groups,
        dp_core=dp_core,
        dp_local=dp_local,
        delta_dynamic_pressure_core=delta_dynamic_pressure_core,
        delta_dynamic_pressure_local=delta_dynamic_pressure_local,
        warnings=_deduplicate_warnings(warnings),
    )


# ---------------------------------------------------------------------------
# Tube-side aggregation
# ---------------------------------------------------------------------------

def _tube_bundle_core_stage(tube_bundle: TubeBundleHydraulicResult) -> PressureDropStageResult:
    # Friction and tube-sheet entrance/exit losses are all irreversible;
    # only the signed acceleration term is kept separate (v0.5.6).
    dp_irreversible = (
        tube_bundle.dp_straight_tube_friction
        + tube_bundle.dp_tube_entrances
        + tube_bundle.dp_tube_exits
    )
    return PressureDropStageResult(
        stage_id="tube_bundle",
        stage_type="straight_tube_bundle",
        status=PressureDropStageStatus.CALCULATED,
        dp_irreversible=dp_irreversible,
        delta_dynamic_pressure=tube_bundle.dp_straight_tube_acceleration,
        method="tube_bundle_hydraulics",
        warnings=tube_bundle.warnings,
    )


def build_tube_side_pressure_drop_result(
    tube_bundle: TubeBundleHydraulicResult,
    *,
    n_tube_passes: int,
    path: SpecifiedTubeSidePressureDropPath | TubeSidePressureDropDesignRequest | None = None,
) -> PressureDropPathResult:
    """Aggregate the tube-side flow path into a ``PressureDropPathResult``.

    Groups are always, in order: ``inlet``, ``tube_bundle`` (the existing
    calculated straight tube-bundle result), ``return_1..return_{n-1}`` (or
    a single empty ``returns`` group when ``path`` is ``None``), ``outlet``.

    Raises:
        NotImplementedError: if ``path`` is a
            ``TubeSidePressureDropDesignRequest`` (future suggested-geometry
            mode; not implemented).
        TypeError: if ``path`` is neither ``None``, a
            ``SpecifiedTubeSidePressureDropPath``, nor a
            ``TubeSidePressureDropDesignRequest``.
        ValueError: if ``path`` is specified and its return-assembly count
            does not equal ``n_tube_passes - 1``.
    """
    if isinstance(path, TubeSidePressureDropDesignRequest):
        raise NotImplementedError(
            "tube_side_pressure_drop_suggested_geometry_not_implemented: "
            "TubeSidePressureDropDesignRequest describes a future "
            "suggested-geometry sizing mode that is not implemented yet. "
            "Supply a SpecifiedTubeSidePressureDropPath, or omit the path "
            "to use the existing straight tube-bundle result only."
        )
    if path is not None and not isinstance(path, SpecifiedTubeSidePressureDropPath):
        raise TypeError(
            "path must be a SpecifiedTubeSidePressureDropPath, a "
            "TubeSidePressureDropDesignRequest, or None."
        )

    core_stage = _tube_bundle_core_stage(tube_bundle)
    tube_bundle_group = PressureDropStageGroupResult(group_id="tube_bundle", stages=(core_stage,))

    if path is None:
        inlet_group = PressureDropStageGroupResult(group_id="inlet", stages=())
        returns_groups: tuple[PressureDropStageGroupResult, ...] = (
            PressureDropStageGroupResult(group_id="returns", stages=()),
        )
        outlet_group = PressureDropStageGroupResult(group_id="outlet", stages=())
    else:
        validate_specified_tube_side_path(path, n_tube_passes=n_tube_passes)
        inlet_group = _group_from_assembly("inlet", path.inlet)
        returns_groups = tuple(
            _group_from_assembly(f"return_{i + 1}", assembly)
            for i, assembly in enumerate(path.returns)
        )
        outlet_group = _group_from_assembly("outlet", path.outlet)

    groups = (inlet_group, tube_bundle_group, *returns_groups, outlet_group)
    return _assemble_path_result(groups, core_group_id="tube_bundle")


# ---------------------------------------------------------------------------
# Outside-side aggregation
# ---------------------------------------------------------------------------

def _tube_bank_core_stage(
    tube_bank: "OutsideTubeBankHydraulicResult | FinnedTubeBankHydraulicResult",
) -> PressureDropStageResult:
    from core.pressure_drop.finned_tube_pressure_drop import (
        FinnedTubeBankHydraulicResult,
    )

    is_finned = isinstance(tube_bank, FinnedTubeBankHydraulicResult)
    metadata = getattr(tube_bank, "metadata", None)
    finned_method = getattr(metadata, "method", "dedicated_provider")
    return PressureDropStageResult(
        stage_id="tube_bank",
        stage_type="circular_finned_tube_bank" if is_finned else "bare_tube_bank",
        status=PressureDropStageStatus.CALCULATED,
        dp_irreversible=tube_bank.dp_drag,
        delta_dynamic_pressure=tube_bank.dp_acceleration,
        method=(
            f"finned_tube_bank_hydraulics_{finned_method}"
            if is_finned
            else "outside_tube_bank_hydraulics"
        ),
        warnings=tube_bank.warnings,
    )


def build_outside_pressure_drop_result(
    tube_bank: "OutsideTubeBankHydraulicResult | FinnedTubeBankHydraulicResult | None",
    *,
    path: SpecifiedOutsidePressureDropPath | OutsidePressureDropDesignRequest | None = None,
) -> PressureDropPathResult:
    """Aggregate the outside-side flow path into a ``PressureDropPathResult``.

    Groups are always, in order: ``inlet``, ``tube_bank`` (the calculated
    plain- or circular-finned-tube-bank result), ``outlet``.

    When ``tube_bank`` is ``None`` (the outside side was not specified for
    this calculation), a NaN-valued result with no groups is returned,
    consistent with the existing outside-hydraulic NaN convention.

    Raises:
        NotImplementedError: if ``path`` is an
            ``OutsidePressureDropDesignRequest`` (future suggested-geometry
            mode; not implemented).
        TypeError: if ``path`` is neither ``None``, a
            ``SpecifiedOutsidePressureDropPath``, nor an
            ``OutsidePressureDropDesignRequest``.
    """
    if isinstance(path, OutsidePressureDropDesignRequest):
        raise NotImplementedError(
            "outside_pressure_drop_suggested_geometry_not_implemented: "
            "OutsidePressureDropDesignRequest describes a future "
            "suggested-geometry sizing mode that is not implemented yet. "
            "Supply a SpecifiedOutsidePressureDropPath, or omit the path "
            "to use the existing calculated tube-bank result only."
        )
    if path is not None and not isinstance(path, SpecifiedOutsidePressureDropPath):
        raise TypeError(
            "path must be a SpecifiedOutsidePressureDropPath, an "
            "OutsidePressureDropDesignRequest, or None."
        )

    if tube_bank is None:
        return PressureDropPathResult(
            groups=(),
            dp_core=math.nan,
            dp_local=0.0,
            delta_dynamic_pressure_core=math.nan,
            delta_dynamic_pressure_local=0.0,
            warnings=(),
        )

    core_stage = _tube_bank_core_stage(tube_bank)
    tube_bank_group = PressureDropStageGroupResult(group_id="tube_bank", stages=(core_stage,))

    if path is None:
        inlet_group = PressureDropStageGroupResult(group_id="inlet", stages=())
        outlet_group = PressureDropStageGroupResult(group_id="outlet", stages=())
    else:
        inlet_group = _group_from_assembly("inlet", path.inlet)
        outlet_group = _group_from_assembly("outlet", path.outlet)

    groups = (inlet_group, tube_bank_group, outlet_group)
    return _assemble_path_result(groups, core_group_id="tube_bank")


# ---------------------------------------------------------------------------
# Generic local-loss assembly calculator (v0.5.6 local pressure-drop paths)
# ---------------------------------------------------------------------------

def _dispatch_calculated_stage(
    group_id: str,
    index: int,
    geometry: object,
    state: PressureDropFlowState,
) -> PressureDropStageResult:
    """Dispatch one stage geometry to its production calculation, or to the
    ``not_implemented`` placeholder builder when no calculation applies.

    Local imports avoid a circular import: ``area_changes``,
    ``direction_changes``, ``screens`` and ``straight_sections`` all need
    ``PressureDropFlowState``/``PressureDropStageResult`` from this module.
    """
    from core.pressure_drop.area_changes import calculate_area_change_pressure_drop
    from core.pressure_drop.direction_changes import (
        CircularElbowGeometry,
        RectangularElbowGeometry,
        calculate_circular_elbow_pressure_drop,
        calculate_rectangular_elbow_pressure_drop,
    )
    from core.pressure_drop.screens import (
        FlatObstructionGeometry,
        calculate_flat_obstruction_pressure_drop,
        calculate_screen_pressure_drop,
        calculate_user_defined_pressure_drop,
    )
    from core.pressure_drop.straight_sections import calculate_straight_section_pressure_drop

    prefix = f"{group_id}_{index}_"

    if isinstance(geometry, StraightSectionGeometry):
        return calculate_straight_section_pressure_drop(
            geometry=geometry, state=state, stage_id=f"{prefix}straight_section",
        )
    if isinstance(geometry, AreaChangeGeometry):
        return calculate_area_change_pressure_drop(
            geometry=geometry, state=state,
            stage_id=f"{prefix}{geometry.change_type.value}_area_change",
        )
    if isinstance(geometry, CircularElbowGeometry):
        return calculate_circular_elbow_pressure_drop(
            geometry=geometry, state=state, stage_id=f"{prefix}circular_elbow",
        )
    if isinstance(geometry, RectangularElbowGeometry):
        return calculate_rectangular_elbow_pressure_drop(
            geometry=geometry, state=state, stage_id=f"{prefix}rectangular_elbow",
        )
    if isinstance(geometry, FlatObstructionGeometry):
        return calculate_flat_obstruction_pressure_drop(
            geometry=geometry, state=state,
            stage_id=f"{prefix}flat_obstruction_{geometry.obstruction_type.value}",
        )
    if isinstance(geometry, UserDefinedPressureDropGeometry):
        return calculate_user_defined_pressure_drop(
            geometry=geometry, state=state, stage_id=f"{prefix}user_defined",
        )
    if isinstance(geometry, ScreenGeometry) and geometry.loss_coefficient is not None:
        return calculate_screen_pressure_drop(
            geometry=geometry, state=state, stage_id=f"{prefix}{geometry.screen_type.value}",
        )

    # ScreenGeometry with no supplied K, DirectionChangeGeometry (the
    # generic placeholder -- not the CircularElbowGeometry/
    # RectangularElbowGeometry taxonomy above), ChamberGeometry, or any
    # other geometry with no implemented calculation: the same
    # not_implemented placeholder used by build_tube_side_pressure_drop_
    # result / build_outside_pressure_drop_result.
    return _build_stage_result(group_id, index, geometry)


def calculate_pressure_drop_assembly(
    *,
    group_id: str,
    geometry: PressureDropAssemblyGeometry,
    state: PressureDropFlowState,
) -> PressureDropStageGroupResult:
    """
    Calculate every stage of an ordered local-loss assembly against a
    single explicit flow state, dispatching each stage geometry to its
    production model:

        StraightSectionGeometry         -> straight_sections
        AreaChangeGeometry               -> area_changes
        CircularElbowGeometry            -> direction_changes
        RectangularElbowGeometry         -> direction_changes
        FlatObstructionGeometry          -> screens
        ScreenGeometry (with loss_coefficient) -> screens
        UserDefinedPressureDropGeometry  -> screens

    Stage order is preserved exactly as supplied in ``geometry.stages``.
    The returned group's hydraulic resistance is the sum of its stages'
    non-negative ``dp_irreversible`` values. Signed dynamic-pressure changes
    and ``dp_static`` are reported separately.

    Geometries with no implemented calculation (a bare ``ScreenGeometry``
    with no supplied K, a generic ``DirectionChangeGeometry``, a standalone
    ``ChamberGeometry``, or a ``UserDefinedPressureDropGeometry`` supplying
    neither ``pressure_drop`` nor ``loss_coefficient``) fall back to the
    same zero, informational-warning ``not_implemented`` placeholder used by
    ``build_tube_side_pressure_drop_result``/``build_outside_pressure_drop_
    result``.

    Purely a function of ``geometry``/``state``: no exchanger object is
    required, no heat-transfer calculation is invoked, and neither argument
    is mutated.
    """
    stages = tuple(
        _dispatch_calculated_stage(group_id, index, stage_geometry, state)
        for index, stage_geometry in enumerate(geometry.stages)
    )
    return PressureDropStageGroupResult(group_id=group_id, stages=stages)


# ---------------------------------------------------------------------------
# Explicit side-specific local pressure-drop path calculation
# (v0.5.6 local pressure-drop paths -- never invoked by the standard solver)
# ---------------------------------------------------------------------------

def calculate_tube_side_pressure_drop_path(
    *,
    tube_bundle: TubeBundleHydraulicResult,
    n_tube_passes: int,
    path: SpecifiedTubeSidePressureDropPath,
    inlet_state: PressureDropFlowState,
    return_states: tuple[PressureDropFlowState, ...],
    outlet_state: PressureDropFlowState,
) -> PressureDropPathResult:
    """
    Explicitly invoked tube-side local-loss path calculation.

    Calculates every stage in ``path.inlet``/``path.returns``/``path.outlet``
    against the supplied per-group flow states via
    ``calculate_pressure_drop_assembly``, and aggregates them with the
    existing calculated tube-bundle core exactly like
    ``build_tube_side_pressure_drop_result``. ``dp_core`` and ``dp_local``
    contain only irreversible losses; the tube-bundle acceleration term and
    local area-change dynamic-pressure terms are reported separately.
    ``len(return_states)`` must equal ``len(path.returns)``
    (== ``n_tube_passes - 1``).

    Never invoked by the standard exchanger solver: geometry, per-group
    states, and the result are all supplied/consumed explicitly by the
    caller (front-end/application layer).
    """
    validate_specified_tube_side_path(path, n_tube_passes=n_tube_passes)
    if len(return_states) != len(path.returns):
        raise ValueError(
            "tube_side_pressure_drop_path_return_state_count_mismatch: "
            f"path.returns has {len(path.returns)} assembl"
            f"{'y' if len(path.returns) == 1 else 'ies'}, but {len(return_states)} "
            "return_states were supplied."
        )

    core_stage = _tube_bundle_core_stage(tube_bundle)
    tube_bundle_group = PressureDropStageGroupResult(group_id="tube_bundle", stages=(core_stage,))

    inlet_group = calculate_pressure_drop_assembly(group_id="inlet", geometry=path.inlet, state=inlet_state)
    returns_groups = tuple(
        calculate_pressure_drop_assembly(
            group_id=f"return_{i + 1}", geometry=assembly, state=return_states[i],
        )
        for i, assembly in enumerate(path.returns)
    )
    outlet_group = calculate_pressure_drop_assembly(group_id="outlet", geometry=path.outlet, state=outlet_state)

    groups = (inlet_group, tube_bundle_group, *returns_groups, outlet_group)
    return _assemble_path_result(groups, core_group_id="tube_bundle")


def calculate_outside_pressure_drop_path(
    *,
    tube_bank: "OutsideTubeBankHydraulicResult | FinnedTubeBankHydraulicResult",
    path: SpecifiedOutsidePressureDropPath,
    inlet_state: PressureDropFlowState,
    outlet_state: PressureDropFlowState,
) -> PressureDropPathResult:
    """
    Explicitly invoked outside-side local-loss path calculation.

    Mirrors ``calculate_tube_side_pressure_drop_path`` for the two-group
    (``inlet``, ``outlet``) outside-side flow path around the existing
    calculated plain or circular-finned tube-bank core
    (``dp_core=tube_bank.dp_drag``), with the tube-bank acceleration
    contribution reported separately.

    Never invoked by the standard exchanger solver.
    """
    core_stage = _tube_bank_core_stage(tube_bank)
    tube_bank_group = PressureDropStageGroupResult(group_id="tube_bank", stages=(core_stage,))

    inlet_group = calculate_pressure_drop_assembly(group_id="inlet", geometry=path.inlet, state=inlet_state)
    outlet_group = calculate_pressure_drop_assembly(group_id="outlet", geometry=path.outlet, state=outlet_state)

    groups = (inlet_group, tube_bank_group, outlet_group)
    return _assemble_path_result(groups, core_group_id="tube_bank")
