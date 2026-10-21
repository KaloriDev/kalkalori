# SPDX-License-Identifier: GPL-3.0-only
"""Cross-section applicability, independent of outside surface treatment."""
from core.geometry.tube import TubeCrossSectionFamily, TubeSurfaceType


class TubeGeometryApplicabilityError(ValueError):
    """The selected physics cannot describe the declared tube cross-section."""


def tube_cross_section_family(tube) -> TubeCrossSectionFamily:
    value = getattr(tube, "cross_section_family", None)
    try:
        return TubeCrossSectionFamily(value)
    except (ValueError, TypeError):
        raise TubeGeometryApplicabilityError(
            "Tube geometry must explicitly declare a supported cross_section_family."
        ) from None


def require_circular_tube(tube, model: str) -> None:
    family = tube_cross_section_family(tube)
    if family is not TubeCrossSectionFamily.CIRCULAR:
        raise TubeGeometryApplicabilityError(
            f"{model} supports circular tubes only; {family.value} geometry is unsupported."
        )


def require_automatic_outside_bank(tube, model: str) -> None:
    """Reject source gaps before evaluating any surrogate bank physics."""
    if getattr(tube, 'surface_type', None) is TubeSurfaceType.RECTANGULAR_INDIVIDUAL_FINNED:
        # Source-specific model selection/geometry validation belongs to the
        # outside dispatcher. This is not circular or smooth-ellipse physics.
        return
    require_circular_tube(tube, model)
