# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only

"""Dry-only phase-change boundary for circular-finned-tube workflows."""

from __future__ import annotations

from core.geometry.tube import TubeSurfaceType
from core.phase_change import warning_codes as WC


class CircularFinnedTubeWetSurfaceNotSupportedError(NotImplementedError):
    """Raised before a wet/condensing phase-change solver can start."""

    warning_code = WC.CIRCULAR_FINNED_TUBE_WET_SURFACE_NOT_SUPPORTED


def reject_circular_finned_tube_wet_surface(
    hx,
    *,
    inside_active: bool = False,
    outside_active: bool = False,
    context: str = "wet/condensing phase change",
) -> None:
    """Reject an active wet path while leaving dry/disabled paths untouched."""

    surface_type = getattr(hx.bundle.tube, "surface_type", None)
    if (
        surface_type is TubeSurfaceType.CIRCULAR_FINNED
        and (inside_active or outside_active)
    ):
        active_sides = ", ".join(
            side
            for side, active in (
                ("inside", inside_active),
                ("outside", outside_active),
            )
            if active
        )
        raise CircularFinnedTubeWetSurfaceNotSupportedError(
            "CircularFinnedTube currently supports dry single-phase operation "
            "only. The requested active "
            f"{context} path ({active_sides}) is unsupported. A dry sensible "
            "result is available only when the requested operating point "
            "remains single phase."
        )
