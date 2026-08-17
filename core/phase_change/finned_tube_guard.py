# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only

"""Wet-outside boundary for circular-finned-tube workflows.

Dry circular fins are now supported opposite existing tube-side phase-change
models.  What remains deliberately unsupported is a wet/condensing *outside*
surface: its film, drainage and fin-contact physics require a separate model.
"""

from __future__ import annotations

from core.geometry.tube import TubeSurfaceType
from core.phase_change import warning_codes as WC


class CircularFinnedTubeWetSurfaceNotSupportedError(NotImplementedError):
    """Raised before a wet/condensing outside-surface solver can start."""

    warning_code = WC.CIRCULAR_FINNED_TUBE_WET_SURFACE_NOT_SUPPORTED


def reject_circular_finned_tube_wet_surface(
    hx,
    *,
    inside_active: bool = False,
    outside_active: bool = False,
    context: str = "wet/condensing phase change",
) -> None:
    """Reject an active wet *outside* path while allowing dry finned outside.

    ``inside_active`` is accepted for compatibility and diagnostics, but does
    not itself reject circular fins: inside evaporation/condensation is
    coupled to the dry outside dispatch and topology-aware resistance network.
    """

    surface_type = getattr(hx.bundle.tube, "surface_type", None)
    if (
        surface_type is TubeSurfaceType.CIRCULAR_FINNED
        and outside_active
    ):
        raise CircularFinnedTubeWetSurfaceNotSupportedError(
            "CircularFinnedTube supports a dry outside surface only. The "
            f"requested active outside {context} path is unsupported because "
            "wet fin/condensate-film transport is not modelled."
        )
