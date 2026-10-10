# SPDX-License-Identifier: GPL-3.0-only
"""Cross-section applicability, independent of outside surface treatment."""
from core.geometry.tube import TubeCrossSectionFamily


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
