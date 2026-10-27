# SPDX-License-Identifier: GPL-3.0-only
"""Distinct serrated flow-area contracts and explicit Holfeld source area."""
from dataclasses import dataclass
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
    """Area [m2] with explicit basis and provider/source provenance.

    An orthographic provider returns blocked area, the other bases return
    open flow area. Consumers must explicitly select the required basis;
    these values are not interchangeable. No provider is selected by AUTO.
    """

    basis: SerratedFlowAreaBasis
    provider_name: str
    source_reference: str

    def evaluate(self, bundle: "TubeBundle") -> float: ...


@dataclass(frozen=True)
class Holfeld2016SerratedFlowAreaProvider:
    """Holfeld p.89 Eq.5-9; source-equivalent, not orthographic blockage."""

    basis = SerratedFlowAreaBasis.SOURCE_EQUIVALENT_FLOW_AREA
    provider_name = "holfeld2016_serrated_flow_area"
    source_reference = "Holfeld (2016), NTNU 2016:303, p.89 Eq.5-9"

    def evaluate(self, bundle: "TubeBundle") -> float:
        from core.geometry.serrated_source import positive, require_source_tube
        t = bundle.tube
        require_source_tube(t, "FLOW_AREA")
        width = bundle.pitch_transverse - t.D_o - 2*t.fin_height*t.fin_thickness_root/t.fin_pitch
        positive(width, "Holfeld minimum free width")
        area = t.length_effective * bundle.n_tubes_per_row * width
        positive(area, "Holfeld A_min")
        if area > bundle.frontal_flow_area:
            raise ValueError("Holfeld A_min must not exceed A_face.")
        return area

    def diagnostics(self, bundle: "TubeBundle") -> dict[str, object]:
        return {
            "provider": self.provider_name, "source": self.source_reference,
            "flow_area_basis": self.basis.value, "flow_area_source": "HOLFELD_2016",
            "equation": "5-9", "equation_closure": "CLOSED",
            "access_status": "INSTITUTIONAL_FREE_ACCESS_LICENCE_UNESTABLISHED",
            "A_min_source": self.evaluate(bundle), "A_face": bundle.frontal_flow_area,
            "FLOW_AREA_DOMAIN": "PASS", "layout": bundle.layout,
            "row_basis": "transverse flow cells; independent of longitudinal rows",
            "reference_velocity": "mdot/(rho*A_min_source)",
            "reference_diameter": "bare do", "coefficient_definition": "area [m2]",
            "calibration": "source geometric equivalent; not a fitted kerf correction",
        }
