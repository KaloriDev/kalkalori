# SPDX-License-Identifier: GPL-3.0-only
"""Source geometry, separate from Phase A geometric integrals.

Original-page mapping: docs/validation/serrated_source_mapping_v095.md.
"""
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from core.common.warnings import ApplicabilityRange
from core.geometry.serrated_finned_tube import SerratedCircularFinnedTube

if TYPE_CHECKING:
    from core.geometry.bundle import TubeBundle


@dataclass(frozen=True)
class SerratedTubeBankRequest:
    """Explicit source dimensions and dry-air bulk transport, all SI.

    Separate from solid-fin requests so a solid provider cannot accept this
    geometry by structural coincidence. Properties are at the arithmetic
    inlet/outlet mean temperature; the DP adapter supplies mean endpoint rho.
    """

    bundle: "TubeBundle"
    m_dot: float
    props: object
    source_segment_width: float
    minimum_free_flow_area: float
    fluid_family: str = "air"

    def __post_init__(self):
        positive(self.m_dot, "m_dot")
        positive(self.source_segment_width, "source_segment_width [m]")
        positive(self.minimum_free_flow_area, "A_min")
        for name in ("rho", "mu", "k", "cp"):
            positive(getattr(self.props, name), name)
        if self.fluid_family != "air":
            raise SerratedSourceDomainError("FLUID", ("source calibration is dry air",))
        from core.geometry.serrated_flow_area import Holfeld2016SerratedFlowAreaProvider
        expected = Holfeld2016SerratedFlowAreaProvider().evaluate(self.bundle)
        if not math.isclose(self.minimum_free_flow_area, expected, rel_tol=1e-12, abs_tol=0.):
            raise SerratedSourceDomainError("FLOW_AREA", ("request must use Holfeld Eq.5-9 A_min",))

    @property
    def source_geometry(self):
        return HolfeldSerratedSourceGeometry(self.bundle, self.source_segment_width)

    @property
    def reynolds_number(self) -> float:
        return self.m_dot*self.bundle.tube.D_o/(self.minimum_free_flow_area*self.props.mu)


def holfeld_applicability(component: str) -> tuple[ApplicabilityRange, ...]:
    ht = component == "HT"
    return (
        ApplicabilityRange("Re", None, 50000.),  # open upper, enforced below
        ApplicabilityRange("n_rows", 4., None, "rows"),
        ApplicabilityRange("angle_deg", None, 45., "degrees"),  # open upper
        ApplicabilityRange("do", .01720 if ht else .01283, .06350, "m"),
        ApplicabilityRange("hf", .00861 if ht else .00635, .03175, "m"),
        ApplicabilityRange("Nf", 78.7, 446.5, "1/m"),
        ApplicabilityRange("hf/do", .26, .94),
        ApplicabilityRange("hs/hf", .50, .99),
        ApplicabilityRange("sf/do", .08, .33),
        ApplicabilityRange("Pt/do", 1.75, 3.50),
        ApplicabilityRange("Pt/Pl", .75, 2.),  # open upper
        ApplicabilityRange("Ar", 4.75, 18.89),
        ApplicabilityRange("Aht/A_min", 8.2, 60.6 if ht else 61.7),
        ApplicabilityRange("A_min/Af_fin", 1., 3.24),
    )


def holfeld_domain(request: SerratedTubeBankRequest, component: str) -> dict[str, float]:
    if not isinstance(request, SerratedTubeBankRequest):
        raise TypeError(f"Holfeld {component} requires SerratedTubeBankRequest, not a solid-fin request.")
    if request.bundle.layout != "staggered":
        raise SerratedSourceDomainError(component, ("STAGGERED layout required",))
    values = request.source_geometry.ratios(request.minimum_free_flow_area)
    values["Re"] = request.reynolds_number
    enforce_ranges(component, values, holfeld_applicability(component),
                   open_upper=("Re", "Pt/Pl", "angle_deg"))
    return values


def holfeld_diagnostics(request: SerratedTubeBankRequest, component: str,
                        values: dict[str, float]) -> dict[str, object]:
    from core.geometry.serrated_flow_area import Holfeld2016SerratedFlowAreaProvider
    return {
        **Holfeld2016SerratedFlowAreaProvider().diagnostics(request.bundle),
        f"{component}_DOMAIN": "PASS", "values": values,
        "property_state": "arithmetic inlet/outlet mean temperature; DP uses endpoint-mean rho",
        "source_surface_area": request.source_geometry.total_area,
        "source_surface_area_basis": "Holfeld Eqs.2-17/2-19 native I-foot strip",
        "phase_a_geometric_area": request.bundle.total_outer_geometric_area,
        "source_segment_width": request.source_segment_width,
        "segment_width_mapping": "explicit rectangular strip width, not Phase A tip arc",
        "layout": "staggered", "row_range": (4, None),
        "row_correction": "NONE",
        "unreported_bounds": ("lower Re", "upper row count", "fin thickness", "ws", "Pl/do", "Pr"),
        "open_upper_bounds": ("Re < 50000", "Pt/Pl < 2", "angle < 45 deg"),
    }


class SerratedSourceDomainError(ValueError):
    """A named physical component failed a source-domain check."""

    def __init__(self, component: str, failures: tuple[str, ...]):
        self.component = component
        self.failures = failures
        self.diagnostics = {f"{component}_DOMAIN": "FAIL", "failures": failures}
        super().__init__(f"{component}_DOMAIN_FAIL: " + "; ".join(failures))


def positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def require_source_tube(tube: SerratedCircularFinnedTube, component: str) -> None:
    if not isinstance(tube, SerratedCircularFinnedTube):
        raise TypeError(f"{component} requires SerratedCircularFinnedTube.")
    failures = []
    if not math.isclose(tube.D_root, tube.D_o, rel_tol=1e-12, abs_tol=1e-15):
        failures.append("D_root must equal bare core D_o; root-layer mapping is unsupported")
    if tube.kerf_width_at_tip != 0:
        failures.append("removed kerf material is outside the source strip geometry")
    if failures:
        raise SerratedSourceDomainError(component, tuple(failures))


def enforce_ranges(component: str, values: dict[str, float],
                   ranges: tuple[ApplicabilityRange, ...], *,
                   open_upper: tuple[str, ...] = ()) -> None:
    failures = []
    for bound in ranges:
        value = values[bound.name]
        if not math.isfinite(value):
            failures.append(f"{bound.name} must be finite")
        elif bound.lower is not None and value < bound.lower - 1e-12 * abs(bound.lower):
            failures.append(f"{bound.name}={value:g} < {bound.lower:g}")
        elif bound.upper is not None:
            if bound.name in open_upper and value >= bound.upper:
                failures.append(f"{bound.name}={value:g} >= {bound.upper:g} (open upper bound)")
            elif value > bound.upper + 1e-12 * abs(bound.upper):
                failures.append(f"{bound.name}={value:g} > {bound.upper:g}")
    if failures:
        raise SerratedSourceDomainError(component, tuple(failures))


@dataclass(frozen=True)
class HolfeldSerratedSourceGeometry:
    """Native I-foot areas; none override the geometry object's own areas.

    source_segment_width is rectangular cut-strip width [m], distinct from
    the outer-radius arc width derived by Phase A.
    """

    bundle: "TubeBundle"
    source_segment_width: float

    def __post_init__(self):
        require_source_tube(self.bundle.tube, "GEOMETRY")
        positive(self.source_segment_width, "source_segment_width [m]")

    @property
    def area_ratio(self) -> float:
        t = self.bundle.tube
        h, s, b, d = t.fin_height, t.serration_depth, t.fin_thickness_root, t.D_o
        w = self.source_segment_width
        # Holfeld p.30 Eq.2-19, expanded to remove division by (hf-hs).
        return 1 + 2 / t.fin_pitch * (
            h + s*b/w + (h-s)*(h+s+b+2*s*b/w)/d
        )

    @property
    def total_area_per_length(self) -> float:
        return self.area_ratio * math.pi * self.bundle.tube.D_o

    @property
    def primary_area_per_length(self) -> float:
        t = self.bundle.tube
        return math.pi * t.D_o * (1-t.fin_thickness_root/t.fin_pitch)

    @property
    def fin_area_per_length(self) -> float:
        return self.total_area_per_length - self.primary_area_per_length

    @property
    def total_area(self) -> float:
        return self.total_area_per_length * self.bundle.tube.length_effective * self.bundle.n_tubes_total

    def ratios(self, minimum_area: float) -> dict[str, float]:
        b, t = self.bundle, self.bundle.tube
        # Af,fin is FLOW area, never the heat-transfer surface of the fins.
        between_fins = (t.length_effective * b.n_tubes_per_row * 2*t.fin_height
                        * (1-t.fin_thickness_root/t.fin_pitch))
        positive(between_fins, "Af_fin")
        return {
            "do": t.D_o, "hf": t.fin_height, "Nf": 1/t.fin_pitch,
            "hf/do": t.fin_height/t.D_o, "hs/hf": t.serration_depth/t.fin_height,
            "sf/do": t.fin_pitch/t.D_o, "Pt/do": b.pitch_transverse/t.D_o,
            "Pl/do": b.pitch_longitudinal/t.D_o,
            "Pt/Pl": b.pitch_transverse/b.pitch_longitudinal,
            "angle_deg": math.degrees(math.atan(b.pitch_transverse/(2*b.pitch_longitudinal))),
            "Ar": self.area_ratio,
            "Aht/A_min": (self.total_area_per_length * t.length_effective
                          * b.n_tubes_per_row / minimum_area),
            "A_min/Af_fin": minimum_area/between_fins, "Af_fin": between_fins,
            "n_rows": b.n_rows,
        }
