# SPDX-License-Identifier: GPL-3.0-only
"""Hashizume segmented-annulus theory and STAGGERED experimental correction.

1997 pp.4030–4031 Eqs.1–11/16; 2002 p.2830 Eqs.3–4. See the source
mapping note for the 2000 main-equation/appendix symbol discrepancy.
No production solid annular efficiency is called here.
"""
from dataclasses import dataclass
import math
from typing import Protocol

from core.common.warnings import ApplicabilityRange
from core.geometry.serrated_finned_tube import SerratedCircularFinnedTube
from core.geometry.serrated_source import enforce_ranges, positive, require_source_tube
from core.heat_transfer.modified_bessel import (
    scaled_bessel_i0, scaled_bessel_i1, scaled_bessel_k0, scaled_bessel_k1,
)


HASHIZUME_STAGGERED_APPLICABILITY = (
    ApplicabilityRange("df/do", 1.8, 2.15),
    ApplicabilityRange("hs/hf", 0., .86),
    ApplicabilityRange("hs/ws", 0., 2.60),
    ApplicabilityRange("Re_o", 5000., 50000.),
    ApplicabilityRange("m*hf", 0., 2.5),
)


@dataclass(frozen=True)
class SerratedFinEfficiencyResult:
    fin_efficiency_theoretical: float
    fin_efficiency: float
    correction: float
    m_height: float
    theoretical_fin_area_per_turn: float
    theoretical_tip_temperature_ratio: float
    diagnostics: dict[str, object]
    # Empirical mean eta does not determine a corrected temperature field.
    tip_temperature_ratio: None = None

    @property
    def eta_fin(self) -> float:
        return self.fin_efficiency


class SerratedFinEfficiencyProvider(Protocol):
    provider_name: str
    source_reference: str

    def evaluate(self, tube: SerratedCircularFinnedTube, outside_htc: float, *,
                 reynolds_number: float, source_segment_width: float,
                 layout: str = "staggered") -> SerratedFinEfficiencyResult: ...


def hashizume_theoretical_efficiency(tube: SerratedCircularFinnedTube,
                                    outside_htc: float) -> tuple[float, float, float]:
    """Return theory eta, theoretical tip ratio and broad-face area/turn.

    The root annulus feeds independent straight segments. At their junction
    u'(rs)+m*tanh(m*hs)*u(rs)=0. Scaled I0/I1/K0/K1 implement the source's
    two-region boundary problem without overflowing Bessel products.
    """
    require_source_tube(tube, "ETA_THEORY")
    if not math.isfinite(outside_htc) or outside_htc < 0:
        raise ValueError("outside_htc must be finite and nonnegative [W/(m2 K)].")
    r0, r2 = tube.D_o/2, tube.D_fin/2
    hs = tube.serration_depth
    rs = r2-hs
    area = 2*math.pi*(rs*rs-r0*r0+2*rs*hs)
    m = math.sqrt(2*outside_htc/(tube.fin_k*tube.fin_thickness_root))
    if m*tube.fin_height < 1e-7:
        return 1., 1., area
    if rs == r0:
        mh = m*hs
        tip = 1/math.cosh(mh) if mh < 700 else 0.
        return math.tanh(mh)/mh, tip, area
    x0, xs = m*r0, m*rs
    gamma = math.tanh(m*hs)
    ci = scaled_bessel_i1(xs)+gamma*scaled_bessel_i0(xs)
    ck = scaled_bessel_k1(xs)-gamma*scaled_bessel_k0(xs)
    attenuation = math.exp(-2*(xs-x0))
    denominator = ci*scaled_bessel_k0(x0)+attenuation*ck*scaled_bessel_i0(x0)
    gradient = m*(ci*scaled_bessel_k1(x0)-attenuation*ck*scaled_bessel_i1(x0))/denominator
    heat_per_K = 2*math.pi*r0*tube.fin_k*tube.fin_thickness_root*gradient
    eta = heat_per_K/(outside_htc*area)
    junction = math.exp(-(xs-x0))*(ci*scaled_bessel_k0(xs)+ck*scaled_bessel_i0(xs))/denominator
    tip = junction/math.cosh(m*hs) if m*hs < 700 else 0.
    if not 0 < eta <= 1+1e-6:
        raise ArithmeticError("Hashizume boundary solution produced invalid theoretical eta.")
    return min(1., eta), tip, area


@dataclass(frozen=True)
class HashizumeSerratedFinEfficiencyProvider:
    provider_name = "hashizume_serrated_fin_efficiency"
    source_reference = "Hashizume 1997 Eqs.1–11/16; 2002 Eqs.3–4 (STAGGERED)"
    applicability = HASHIZUME_STAGGERED_APPLICABILITY

    def evaluate(self, tube: SerratedCircularFinnedTube, outside_htc: float, *,
                 reynolds_number: float, source_segment_width: float,
                 layout: str = "staggered") -> SerratedFinEfficiencyResult:
        require_source_tube(tube, "ETA")
        positive(source_segment_width, "source_segment_width [m]")
        positive(reynolds_number, "Re_o")
        if layout != "staggered":
            from core.geometry.serrated_source import SerratedSourceDomainError
            raise SerratedSourceDomainError("ETA", ("STAGGERED correction only",))
        if not math.isfinite(outside_htc) or outside_htc < 0:
            raise ValueError("outside_htc must be finite and nonnegative.")
        mh = math.sqrt(2*outside_htc/(tube.fin_k*tube.fin_thickness_root))*tube.fin_height
        values = {"df/do": tube.D_fin/tube.D_o, "hs/hf": tube.serration_depth/tube.fin_height,
                  "hs/ws": tube.serration_depth/source_segment_width,
                  "Re_o": reynolds_number, "m*hf": mh}
        enforce_ranges("ETA", values, self.applicability)
        theory, tip, area = hashizume_theoretical_efficiency(tube, outside_htc)
        psi_raw = 1-(.016*values["hs/ws"]+.14*values["df/do"]**2.7
                     *(1-.097*math.log(reynolds_number)))*mh
        psi = min(1., psi_raw)  # source-prescribed upper cap, not a fit
        if psi <= 0:
            raise ArithmeticError("Hashizume correction is nonpositive within declared inputs.")
        diagnostics = {
            "provider": self.provider_name, "source": self.source_reference,
            "equation_closure": "CLOSED_USING_2002_STAGGERED_RESTATEMENT",
            "access_status": "FREE_ACCESS_COPYRIGHTED", "ETA_DOMAIN": "PASS",
            "layout": layout, "row_basis": "STAGGERED correction has no row multiplier",
            "reference_diameter": "bare do", "reference_velocity": "source minimum-passage velocity",
            "property_state": "bulk mean film h; caller-specified constant fin k",
            "coefficient_definition": "eta_corrected = eta_theory * min(1,psi)",
            "calibration": "2000 staggered experiments; 2002 explicit restatement",
            "theory_area_basis": "two-region broad faces; insulated tips/sides",
            "corrected_area_basis": "physical fin area including tip/sides, source strip geometry",
            "temperature_field_status": "THEORETICAL_ONLY; corrected tip field unreported",
            "unreported_bounds": ("fin thickness", "fin pitch", "upper/lower row count"),
            "psi_raw": psi_raw, "values": values,
        }
        return SerratedFinEfficiencyResult(theory, theory*psi, psi, mh, area, tip, diagnostics)
