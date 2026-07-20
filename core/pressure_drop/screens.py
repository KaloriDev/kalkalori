# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

# NOTE ON UNITS
# -------------
# All calculations use SI units:
# - rho [kg/m^3], v [m/s], dp [Pa]

"""
Screen / opening-array pressure-drop calculations (v0.5.6).

A tube sheet is an array of parallel openings, i.e. a screen/opening-array
element in the common pressure-drop geometry vocabulary introduced by the
first v0.5.6 commit (``core.geometry.pressure_drop_stages.ScreenType.
TUBE_SHEET_OPENING_ARRAY``). This module implements the standard tube-sheet
entrance and exit losses that are always part of the tube-bundle core (see
``core.pressure_drop.internal_pressure_drop.calculate_tube_bundle_
hydraulics``), plus (since the local pressure-drop paths commit) three
further, explicitly-invoked-only local-loss calculations that share the
same "loss coefficient times a reference dynamic pressure" structure:

    - ``calculate_flat_obstruction_pressure_drop``: a high-Re,
      blockage-ratio-only model for flat planar obstructions (bird nets,
      wire meshes, grilles, louvers, perforated plates) -- see
      ``FlatObstructionGeometry`` below.
    - ``calculate_screen_pressure_drop``: a general ``ScreenGeometry`` with
      an explicit user-supplied ``loss_coefficient``, referenced to the
      velocity through ``open_flow_area``.
    - ``calculate_user_defined_pressure_drop``: a
      ``UserDefinedPressureDropGeometry`` supplying either a fixed pressure
      drop, or a loss coefficient plus a reference area.

None of the three are invoked by the standard exchanger solver; they are
reached only via ``core.pressure_drop.flow_path.
calculate_pressure_drop_assembly`` (or the tube-side/outside-side explicit
path calculators built on it).

Scope of this commit (tube-sheet entrance/exit)
------------------------------------------------
Only the standard models are implemented:
    - sharp-edged, flush tube entrance (``TubeSheetEntranceType.SHARP_EDGED``)
    - normal discharge into a comparatively large chamber or header
      (``TubeSheetExitType.NORMAL``)

Both assume:
    - a sharp, flush entry from (or normal discharge into) a comparatively
      large upstream/downstream chamber -- not a detailed finite-chamber
      geometry model;
    - uniform flow distribution among parallel tubes.

Future entrance/exit geometries (rounded, beveled, projecting, re-entrant
entrances; other exit geometries) are architecturally anticipated via
additional enum members and coefficient-table entries, but are not
implemented here. No empirical calibration multipliers are applied.

Literature references
----------------------
- Idelchik, I. E., "Handbook of Hydraulic Resistance"
- Crane Co., "Flow of Fluids Through Valves, Fittings, and Pipe" (TP-410)
- Rennels, D. C., Hudson, H. M., "Pipe Flow: A Practical and Comprehensive
  Guide"

Statelessness contract
------------------------
``calculate_tube_sheet_entrance_loss``/``calculate_tube_sheet_exit_loss``
operate only on a reference dynamic pressure and a selected method; they do
not know which tube pass, tube path type (straight/U-tube), or calculation
mode (simulate/rate) the reference state came from. The calling tube-bundle
function decides how many times to apply them and at which pass-boundary
state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from core.common.warnings import make_warning
from core.geometry.pressure_drop_stages import (
    FlowSectionGeometry,
    PressureDropStageGeometry,
    ScreenGeometry,
    UserDefinedPressureDropGeometry,
)

if TYPE_CHECKING:
    from core.pressure_drop.flow_path import PressureDropFlowState, PressureDropStageResult


class TubeSheetEntranceType(str, Enum):
    """Supported tube-sheet entrance geometries.

    Only ``SHARP_EDGED`` is implemented in this commit. Future additions
    (rounded, beveled, projecting, re-entrant) are anticipated but not
    implemented.
    """

    SHARP_EDGED = "sharp_edged"


class TubeSheetExitType(str, Enum):
    """Supported tube-sheet exit geometries.

    Only ``NORMAL`` (discharge into a comparatively large chamber or
    header) is implemented in this commit.
    """

    NORMAL = "normal"


# Sharp-edged, flush entrance from a comparatively large chamber.
# Ref: Idelchik; Crane TP-410; Rennels & Hudson, Pipe Flow.
_ENTRANCE_LOSS_COEFFICIENTS: dict[TubeSheetEntranceType, float] = {
    TubeSheetEntranceType.SHARP_EDGED: 0.5,
}

# Normal discharge into a comparatively large chamber or header.
# Ref: Idelchik; Crane TP-410; Rennels & Hudson, Pipe Flow.
_EXIT_LOSS_COEFFICIENTS: dict[TubeSheetExitType, float] = {
    TubeSheetExitType.NORMAL: 1.0,
}


def tube_sheet_entrance_loss_coefficient(
    entrance_type: TubeSheetEntranceType = TubeSheetEntranceType.SHARP_EDGED,
) -> float:
    """Loss coefficient K for a tube-sheet entrance."""
    try:
        return _ENTRANCE_LOSS_COEFFICIENTS[entrance_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported TubeSheetEntranceType: {entrance_type!r}") from exc


def tube_sheet_exit_loss_coefficient(
    exit_type: TubeSheetExitType = TubeSheetExitType.NORMAL,
) -> float:
    """Loss coefficient K for a tube-sheet exit."""
    try:
        return _EXIT_LOSS_COEFFICIENTS[exit_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported TubeSheetExitType: {exit_type!r}") from exc


def calculate_tube_sheet_entrance_loss(
    *,
    dynamic_pressure: float,
    entrance_type: TubeSheetEntranceType = TubeSheetEntranceType.SHARP_EDGED,
) -> tuple[float, float]:
    """Tube-sheet entrance pressure loss: ``dp = K * dynamic_pressure``.

    ``dynamic_pressure`` is the local mean tube velocity's dynamic pressure
    (``rho*v^2/2``, equivalently ``mass_flux^2/(2*rho)``) at the entrance
    reference state -- the caller supplies it (from geometry, fluid
    density, and reference velocity) rather than this function computing
    velocity itself, so a caller can equally pass a directly known
    dynamic pressure.

    Returns:
        ``(loss_coefficient, pressure_drop)`` in ``[-]``, ``[Pa]``.
    """
    if not math.isfinite(dynamic_pressure) or dynamic_pressure < 0.0:
        raise ValueError(
            "tube_sheet_entrance_loss_invalid_dynamic_pressure: "
            "dynamic_pressure must be finite and non-negative."
        )
    K = tube_sheet_entrance_loss_coefficient(entrance_type)
    return K, K * dynamic_pressure


def calculate_tube_sheet_exit_loss(
    *,
    dynamic_pressure: float,
    exit_type: TubeSheetExitType = TubeSheetExitType.NORMAL,
) -> tuple[float, float]:
    """Tube-sheet exit pressure loss: ``dp = K * dynamic_pressure``.

    See ``calculate_tube_sheet_entrance_loss`` for the statelessness
    contract and the meaning of ``dynamic_pressure``.

    Returns:
        ``(loss_coefficient, pressure_drop)`` in ``[-]``, ``[Pa]``.
    """
    if not math.isfinite(dynamic_pressure) or dynamic_pressure < 0.0:
        raise ValueError(
            "tube_sheet_exit_loss_invalid_dynamic_pressure: "
            "dynamic_pressure must be finite and non-negative."
        )
    K = tube_sheet_exit_loss_coefficient(exit_type)
    return K, K * dynamic_pressure


# ---------------------------------------------------------------------------
# Flat planar obstructions (bird nets, wire meshes, grilles, louvers, ...)
# ---------------------------------------------------------------------------

class FlatObstructionType(str, Enum):
    """Descriptive obstruction family. In the blockage-only model below this
    is purely descriptive: the calculated result depends only on
    ``blockage_ratio`` and the fluid state, never on ``obstruction_type``
    itself -- see ``calculate_flat_obstruction_pressure_drop``."""

    GENERIC = "generic"
    BIRD_NET = "bird_net"
    WIRE_MESH = "wire_mesh"
    GRILLE = "grille"
    LOUVER = "louver"


@dataclass(frozen=True)
class FlatObstructionGeometry(PressureDropStageGeometry):
    """A flat planar obstruction spanning the full gross face area of a
    duct/plenum section (bird net, wire mesh, grille, louver, perforated
    plate, ...), described only by its blockage ratio.

    ``blockage_ratio`` is the fraction of ``face_section.flow_area`` that is
    physically blocked (``0`` = fully open, approaching ``1`` = fully
    closed). ``obstruction_type`` is descriptive only in this commit's
    high-Re blockage-only model; see
    ``calculate_flat_obstruction_pressure_drop``.
    """

    face_section: FlowSectionGeometry
    blockage_ratio: float
    obstruction_type: FlatObstructionType = FlatObstructionType.GENERIC

    def __post_init__(self) -> None:
        if not math.isfinite(self.blockage_ratio) or not (0.0 <= self.blockage_ratio < 1.0):
            raise ValueError(
                "FlatObstructionGeometry.blockage_ratio must be within [0, 1)."
            )


_FLAT_OBSTRUCTION_LIMITATION_MESSAGE = (
    "This blockage-only model treats the element as a normal, "
    "high-Reynolds-number screen-equivalent obstruction. Actual pressure "
    "loss may also depend on wire or bar shape, thickness, spacing, mesh "
    "construction, Reynolds number, incidence angle, louver angle and other "
    "construction details."
)


def calculate_flat_obstruction_pressure_drop(
    *,
    geometry: FlatObstructionGeometry,
    state: "PressureDropFlowState",
    stage_id: str,
) -> "PressureDropStageResult":
    """
    High-Re, blockage-ratio-only pressure drop for a flat planar
    obstruction (Idelchik screen-equivalent form, ``K_mesh=1``, ``K_Re=1``):

        beta = 1 - blockage_ratio                    (open-area ratio)
        K = blockage_ratio + (blockage_ratio / beta)**2
        dp_irreversible = K * rho * V_face**2 / 2

    referenced to the gross face (approach) velocity, not the open-area
    velocity. ``delta_dynamic_pressure = 0``: the gross duct area is unchanged
    across an in-place obstruction. ``blockage_ratio=0`` gives ``K=0`` (and
    thus ``dp=0``) directly from the formula, with no special-cased branch.

    A limitation warning is attached whenever ``blockage_ratio > 0``: this
    is not a universally exact model for any specific bird net, wire mesh,
    grille, or louver construction -- see the module-level note above and
    ``FlatObstructionType``.
    """
    from core.pressure_drop.flow_path import (
        PressureDropStageResult,
        PressureDropStageStatus,
        evaluate_section_flow,
    )

    face_flow = evaluate_section_flow(state, geometry.face_section)
    B = geometry.blockage_ratio
    beta = 1.0 - B
    K = B + (B / beta) ** 2
    dp_irreversible = K * face_flow.dynamic_pressure

    face_area = geometry.face_section.flow_area
    open_area = face_area * beta
    open_area_velocity = state.mass_flow / (state.props.rho * open_area)

    warnings = ()
    if B > 0.0:
        warnings = (
            make_warning(
                code="flat_obstruction_blockage_only_model_limitation",
                message=(
                    f"flat_obstruction ({geometry.obstruction_type.value}): "
                    f"{_FLAT_OBSTRUCTION_LIMITATION_MESSAGE}"
                ),
                source="flat_obstruction_pressure_drop",
                severity="info",
            ),
        )

    return PressureDropStageResult(
        stage_id=stage_id,
        stage_type=f"flat_obstruction_{geometry.obstruction_type.value}",
        status=PressureDropStageStatus.CALCULATED,
        dp_irreversible=dp_irreversible,
        delta_dynamic_pressure=0.0,
        method="idelchik_high_re_blockage_only",
        warnings=warnings,
        loss_coefficient=K,
        reference_area=face_area,
        reference_velocity=face_flow.velocity,
        reference_dynamic_pressure=face_flow.dynamic_pressure,
        upstream_area=face_area,
        downstream_area=face_area,
        upstream_velocity=face_flow.velocity,
        downstream_velocity=face_flow.velocity,
        open_area_ratio=beta,
        blockage_ratio=B,
        open_area_velocity=open_area_velocity,
    )


# ---------------------------------------------------------------------------
# General screen with an explicit user-supplied loss coefficient
# ---------------------------------------------------------------------------

def calculate_screen_pressure_drop(
    *,
    geometry: ScreenGeometry,
    state: "PressureDropFlowState",
    stage_id: str,
) -> "PressureDropStageResult":
    """
    ``ScreenGeometry`` with an explicit user-supplied ``loss_coefficient``:

        dp_irreversible = loss_coefficient * rho * V_open**2 / 2

    referenced to the velocity through ``open_flow_area`` (the physically
    meaningful reference for a general screen/opening-array element, unlike
    the gross-face-referenced ``FlatObstructionGeometry`` blockage-only
    model). Raises if ``geometry.loss_coefficient`` is ``None`` -- a screen
    with no supplied K has no implemented calculation and must be reported
    as ``not_implemented`` by the caller instead of calling this function.
    """
    from core.pressure_drop.flow_path import (
        PressureDropStageResult,
        PressureDropStageStatus,
    )

    if geometry.loss_coefficient is None:
        raise ValueError(
            "calculate_screen_pressure_drop requires geometry.loss_coefficient; "
            "a screen with no supplied K is not_implemented, not an error to "
            "raise from the dispatcher."
        )
    K = geometry.loss_coefficient
    if not math.isfinite(K) or K < 0.0:
        raise ValueError("ScreenGeometry.loss_coefficient must be non-negative and finite.")

    open_flow_area = geometry.open_flow_area
    if not math.isfinite(open_flow_area) or open_flow_area <= 0.0:
        raise ValueError("ScreenGeometry.open_flow_area must be positive and finite.")

    rho = state.props.rho
    V_ref = state.mass_flow / (rho * open_flow_area)
    q_ref = rho * V_ref ** 2 / 2.0
    dp_irreversible = K * q_ref

    open_area_ratio = (
        open_flow_area / geometry.upstream_area if geometry.upstream_area > 0.0 else None
    )

    return PressureDropStageResult(
        stage_id=stage_id,
        stage_type=geometry.screen_type.value,
        status=PressureDropStageStatus.USER_DEFINED,
        dp_irreversible=dp_irreversible,
        delta_dynamic_pressure=0.0,
        method="user_defined_loss_coefficient",
        warnings=(),
        loss_coefficient=K,
        reference_area=open_flow_area,
        reference_velocity=V_ref,
        reference_dynamic_pressure=q_ref,
        upstream_area=geometry.upstream_area,
        downstream_area=geometry.upstream_area,
        open_area_ratio=open_area_ratio,
    )


# ---------------------------------------------------------------------------
# User-defined local loss (fixed dp, or K + reference area)
# ---------------------------------------------------------------------------

def calculate_user_defined_pressure_drop(
    *,
    geometry: UserDefinedPressureDropGeometry,
    state: "PressureDropFlowState",
    stage_id: str,
) -> "PressureDropStageResult":
    """
    ``UserDefinedPressureDropGeometry`` supplying exactly one of:

        - ``pressure_drop``: reported verbatim as ``dp_irreversible`` with
          ``delta_dynamic_pressure=0`` (and therefore
          ``dp_static=dp_irreversible``).
        - ``loss_coefficient`` + ``reference_area``: ``V_ref = mass_flow /
          (rho * reference_area)``, ``dp_irreversible = loss_coefficient *
          rho * V_ref**2 / 2``.

    Rejects supplying both, ``loss_coefficient`` without ``reference_area``,
    negative ``loss_coefficient``, negative ``pressure_drop``, and a
    non-positive ``reference_area``. A geometry supplying neither is
    reported as ``not_implemented`` (mirrors
    ``core.pressure_drop.flow_path._build_stage_result``'s placeholder
    convention for the same case).
    """
    from core.pressure_drop.flow_path import (
        PressureDropStageResult,
        PressureDropStageStatus,
    )

    if geometry.pressure_drop is not None and geometry.loss_coefficient is not None:
        raise ValueError(
            "user_defined_pressure_drop_ambiguous: supply either pressure_drop "
            "or loss_coefficient (+ reference_area), not both."
        )

    if geometry.pressure_drop is not None:
        dp = float(geometry.pressure_drop)
        if not math.isfinite(dp) or dp < 0.0:
            raise ValueError("UserDefinedPressureDropGeometry.pressure_drop must be non-negative and finite.")
        return PressureDropStageResult(
            stage_id=stage_id,
            stage_type="user_defined",
            status=PressureDropStageStatus.USER_DEFINED,
            dp_irreversible=dp,
            delta_dynamic_pressure=0.0,
            method="user_defined_fixed_dp",
            warnings=(),
        )

    if geometry.loss_coefficient is not None:
        K = geometry.loss_coefficient
        if not math.isfinite(K) or K < 0.0:
            raise ValueError("UserDefinedPressureDropGeometry.loss_coefficient must be non-negative and finite.")
        if geometry.reference_area is None:
            raise ValueError(
                "user_defined_pressure_drop_missing_reference_area: "
                "loss_coefficient requires a reference_area."
            )
        A_ref = geometry.reference_area
        if not math.isfinite(A_ref) or A_ref <= 0.0:
            raise ValueError("UserDefinedPressureDropGeometry.reference_area must be a positive, finite value.")

        rho = state.props.rho
        V_ref = state.mass_flow / (rho * A_ref)
        q_ref = rho * V_ref ** 2 / 2.0
        dp_irreversible = K * q_ref
        return PressureDropStageResult(
            stage_id=stage_id,
            stage_type="user_defined",
            status=PressureDropStageStatus.USER_DEFINED,
            dp_irreversible=dp_irreversible,
            delta_dynamic_pressure=0.0,
            method="user_defined_loss_coefficient",
            warnings=(),
            loss_coefficient=K,
            reference_area=A_ref,
            reference_velocity=V_ref,
            reference_dynamic_pressure=q_ref,
        )

    warning = make_warning(
        code="pressure_drop_stage_not_implemented",
        message=(
            f"pressure_drop_flow_path: stage '{stage_id}' (user_defined) has "
            "no pressure_drop or loss_coefficient supplied; reporting a zero "
            "placeholder, not a completed physical result."
        ),
        source="pressure_drop_flow_path",
        severity="info",
    )
    return PressureDropStageResult(
        stage_id=stage_id,
        stage_type="user_defined",
        status=PressureDropStageStatus.NOT_IMPLEMENTED,
        dp_irreversible=0.0,
        delta_dynamic_pressure=0.0,
        method=None,
        warnings=(warning,),
    )
