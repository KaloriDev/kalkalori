# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Pressure-drop stage/group/path result structures and flow-path aggregation
(v0.5.6 architecture).

This module defines the generic, immutable result structures shared by both
exchanger sides (stage, stage-group, complete-path) and the functions that
aggregate the existing tube-bundle / tube-bank calculated results together
with (currently unimplemented) local-loss stages into a single
``PressureDropPathResult``.

Scope of this commit
---------------------
No new local-loss correlations are implemented. Every stage built from a
``core.geometry.pressure_drop_stages`` geometry object is reported with
``status=PressureDropStageStatus.NOT_IMPLEMENTED`` and zero pressure drop,
except a ``UserDefinedPressureDropGeometry`` that directly supplies a fixed
``pressure_drop`` value (a trivial pass-through, not a new calculation). The
existing tube-bundle and tube-bank hydraulic results remain the only
calculated stages (``tube_bundle`` / ``tube_bank``).
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

if TYPE_CHECKING:
    # Type-hint only: avoids a runtime dependency of core.pressure_drop on
    # core.heat_transfer. The tube-bank core stage is built from this
    # result's numeric fields via plain attribute access (duck-typed).
    from core.heat_transfer.outside_flow import OutsideTubeBankHydraulicResult


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

    ``method`` is ``None`` for stages that were not calculated. A future
    method may return pressure drop directly rather than through a loss
    coefficient; this structure does not assume every stage is described
    only by a ``K`` coefficient.
    """

    stage_id: str
    stage_type: str
    status: PressureDropStageStatus

    dp_irreversible: float
    dp_acceleration: float
    dp_total: float

    method: str | None
    warnings: tuple[ModelWarning, ...]


@dataclass(frozen=True)
class PressureDropStageGroupResult:
    """An ordered group of stage results (e.g. ``inlet``, ``return_1``)."""

    group_id: str
    stages: tuple[PressureDropStageResult, ...]

    @property
    def dp_irreversible(self) -> float:
        return sum(stage.dp_irreversible for stage in self.stages)

    @property
    def dp_acceleration(self) -> float:
        return sum(stage.dp_acceleration for stage in self.stages)

    @property
    def dp_total(self) -> float:
        return sum(stage.dp_total for stage in self.stages)


@dataclass(frozen=True)
class PressureDropPathResult:
    """Complete-path pressure-drop result: stage groups plus aggregation.

    ``dp_core`` is the existing calculated tube-bundle or tube-bank result.
    ``dp_local`` is the sum of all additional inlet/return/outlet local
    stages (currently always ``0.0``: no local-loss correlations are
    implemented yet). ``dp_total = dp_core + dp_local`` is the single source
    of truth for the complete-path pressure drop; no other alias is
    authoritative.
    """

    groups: tuple[PressureDropStageGroupResult, ...]

    dp_core: float
    dp_local: float
    dp_total: float

    warnings: tuple[ModelWarning, ...]

    def group(self, group_id: str) -> PressureDropStageGroupResult | None:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        return None


# ---------------------------------------------------------------------------
# Placeholder (not_implemented) stage construction
# ---------------------------------------------------------------------------

def _stage_type_name(geometry: PressureDropStageGeometry) -> str:
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
    raise TypeError(f"Unsupported pressure-drop stage geometry: {geometry!r}")


def _build_stage_result(
    group_id: str,
    index: int,
    geometry: PressureDropStageGeometry,
) -> PressureDropStageResult:
    stage_type = _stage_type_name(geometry)
    stage_id = f"{group_id}_{index}_{stage_type}"

    if isinstance(geometry, UserDefinedPressureDropGeometry) and geometry.pressure_drop is not None:
        dp = float(geometry.pressure_drop)
        return PressureDropStageResult(
            stage_id=stage_id,
            stage_type=stage_type,
            status=PressureDropStageStatus.USER_DEFINED,
            dp_irreversible=dp,
            dp_acceleration=0.0,
            dp_total=dp,
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
        dp_acceleration=0.0,
        dp_total=0.0,
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
    warnings: list[ModelWarning] = []
    for group in groups:
        for stage in group.stages:
            warnings.extend(stage.warnings)
        if group.group_id == core_group_id:
            dp_core += group.dp_total
        else:
            dp_local += group.dp_total
    return PressureDropPathResult(
        groups=groups,
        dp_core=dp_core,
        dp_local=dp_local,
        dp_total=dp_core + dp_local,
        warnings=_deduplicate_warnings(warnings),
    )


# ---------------------------------------------------------------------------
# Tube-side aggregation
# ---------------------------------------------------------------------------

def _tube_bundle_core_stage(tube_bundle: TubeBundleHydraulicResult) -> PressureDropStageResult:
    return PressureDropStageResult(
        stage_id="tube_bundle",
        stage_type="straight_tube_bundle",
        status=PressureDropStageStatus.CALCULATED,
        dp_irreversible=tube_bundle.dp_friction,
        dp_acceleration=tube_bundle.dp_acceleration,
        dp_total=tube_bundle.dp_tube_bundle,
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

def _tube_bank_core_stage(tube_bank: "OutsideTubeBankHydraulicResult") -> PressureDropStageResult:
    return PressureDropStageResult(
        stage_id="tube_bank",
        stage_type="bare_tube_bank",
        status=PressureDropStageStatus.CALCULATED,
        dp_irreversible=tube_bank.dp_drag,
        dp_acceleration=tube_bank.dp_acceleration,
        dp_total=tube_bank.dp_total,
        method="outside_tube_bank_hydraulics",
        warnings=tube_bank.warnings,
    )


def build_outside_pressure_drop_result(
    tube_bank: "OutsideTubeBankHydraulicResult | None",
    *,
    path: SpecifiedOutsidePressureDropPath | OutsidePressureDropDesignRequest | None = None,
) -> PressureDropPathResult:
    """Aggregate the outside-side flow path into a ``PressureDropPathResult``.

    Groups are always, in order: ``inlet``, ``tube_bank`` (the existing
    calculated bare tube-bank result), ``outlet``.

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
            "to use the existing bare tube-bank result only."
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
            dp_total=math.nan,
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
