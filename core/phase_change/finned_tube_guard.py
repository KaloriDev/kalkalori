# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only

"""Capability boundary for wet circular-finned outside workflows.

v0.7.5 permits the explicitly supported H2O-condensation path from a wet gas
with a non-condensable carrier.  Historical calls and all other wet species,
directions, or provider families remain conservative controlled errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.geometry.tube import TubeSurfaceType
from core.phase_change import warning_codes as WC

if TYPE_CHECKING:
    from core.phase_change.types import PhaseChangeCapability, PhaseChangeDirection


class CircularFinnedTubeWetSurfaceNotSupportedError(NotImplementedError):
    """Raised before an unsupported wet circular-fin path can start."""

    warning_code = WC.CIRCULAR_FINNED_TUBE_WET_SURFACE_NOT_SUPPORTED


def reject_circular_finned_tube_wet_surface(
    hx,
    *,
    inside_active: bool = False,
    outside_active: bool = False,
    context: str = "wet/condensing phase change",
    outside_capability: "PhaseChangeCapability | None" = None,
    direction: "PhaseChangeDirection | str | None" = None,
) -> None:
    """Reject unsupported active wet outside paths on circular fins.

    ``inside_active`` is accepted for compatibility and diagnostics, but does
    not itself reject circular fins: inside evaporation/condensation is
    coupled to the dry outside dispatch and topology-aware resistance network.

    Active H2O condensation from a gas mixture is the sole supported wet
    circular-fin path.  Callers must opt into that path explicitly by passing
    both its detected ``outside_capability`` and condensation ``direction``.
    Calls using the historical signature remain conservative and continue to
    reject an active wet outside surface.
    """

    surface_type = getattr(hx.bundle.tube, "surface_type", None)
    if surface_type is not TubeSurfaceType.CIRCULAR_FINNED or not outside_active:
        return

    direction_value = getattr(direction, "value", direction)
    supported_h2o_condensation = (
        outside_capability is not None
        and getattr(outside_capability, "capable", False)
        and str(getattr(outside_capability, "component", "")).upper() == "H2O"
        and getattr(outside_capability, "provider_kind", None) == "gas_mixture"
        and direction_value == "condensation"
    )
    if not supported_h2o_condensation:
        raise CircularFinnedTubeWetSurfaceNotSupportedError(
            "CircularFinnedTube supports a dry outside surface only for this "
            "requested path. Active wet outside operation is supported only "
            "for H2O condensation from a non-condensable carrier-gas mixture; "
            f"the requested active outside {context} path is unsupported."
        )
