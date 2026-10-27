# SPDX-License-Identifier: GPL-3.0-only
"""Future serrated area contracts; Phase A implements none of these models."""
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from core.geometry.bundle import TubeBundle


class SerratedFlowAreaBasis(str, Enum):
    """Distinct definitions; orthographic blockage does not establish A_min."""

    ORTHOGRAPHIC_GEOMETRIC_BLOCKAGE = "orthographic_geometric_blockage"
    SOURCE_EQUIVALENT_FLOW_AREA = "source_equivalent_flow_area"
    HYDRAULIC_PROVIDER_SPECIFIC_A_MIN = "hydraulic_provider_specific_A_min"


class SerratedFlowAreaProvider(Protocol):
    """Future area [m2] with explicit basis and provider/source provenance.

    An orthographic provider returns blocked area, the other bases return
    open flow area. Consumers must explicitly select the required basis;
    these values are not interchangeable. No provider is installed in Phase A.
    """

    basis: SerratedFlowAreaBasis
    provider_name: str
    source_reference: str

    def evaluate(self, bundle: "TubeBundle") -> float: ...
