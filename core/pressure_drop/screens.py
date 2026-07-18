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
TUBE_SHEET_OPENING_ARRAY``). This module implements the first real,
family-specific pressure-drop calculation: the standard tube-sheet entrance
and exit losses that are always part of the tube-bundle core (see
``core.pressure_drop.internal_pressure_drop.calculate_tube_bundle_
hydraulics``).

Scope of this commit
---------------------
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
from enum import Enum


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
