# SPDX-License-Identifier: GPL-3.0-only
"""Explicit source-bounded Holfeld STAGGERED serrated physical film HTC."""
from dataclasses import dataclass
import math

from core.common.warnings import make_warning
from core.geometry.serrated_source import (
    SerratedTubeBankRequest, holfeld_applicability, holfeld_domain, holfeld_diagnostics,
)
from core.heat_transfer.finned_tube_outside import (
    FinnedTubeHeatTransferMetadata, FinnedTubeHeatTransferResult,
)


HOLFELD_2016_HT_METADATA = FinnedTubeHeatTransferMetadata(
    method="holfeld2016_serrated_staggered_ht",
    source="A. Holfeld (2016), NTNU 2016:303, pp.30,54–57,89,112–115,126; hdl:11250/2432512",
    equation="Eq.6-6: Nu*Pr^(-1/3)=.184*Re^.696*Ar^(-.655)*(Pt/do)^.262*(hf/do)^.602*(sf/do)^(-.729)",
    geometry_family="HF-welded constant-thickness untwisted serrated I-foot round tube bank",
    velocity_basis="u_max=mdot/(rho*A_min_source), Eq.5-9",
    reynolds_basis="Re=mdot*do/(A_min_source*mu)",
    reference_diameter="bare tube do (D_root=core.D_o)",
    area_basis="PHYSICAL_FILM on Holfeld native I-foot surface; eta applied separately once",
    row_basis="mean bank film; Nl>=4, upper unreported; no row correction",
    source_fluid="dry air",
    supported_layouts=("staggered",), supported_fin_profiles=("serrated_constant_untwisted_welded_i_foot",),
    applicability=holfeld_applicability("HT"),
)


@dataclass(frozen=True)
class Holfeld2016SerratedStaggeredHeatTransferProvider:
    provider_name = "holfeld2016_serrated_staggered_ht"
    source_reference = HOLFELD_2016_HT_METADATA.source

    def evaluate(self, request: SerratedTubeBankRequest) -> FinnedTubeHeatTransferResult:
        v = holfeld_domain(request, "HT")
        b, t, p = request.bundle, request.bundle.tube, request.props
        Re = request.reynolds_number
        Pr = p.cp*p.mu/p.k
        reduced_nu = (.184*Re**.696*v["Ar"]**(-.655)*v["Pt/do"]**.262
                      *v["hf/do"]**.602*v["sf/do"]**(-.729))
        Nu = reduced_nu*Pr**(1/3)
        h = Nu*p.k/t.D_o
        diagnostics = {
            **holfeld_diagnostics(request, "HT", v),
            "provider": self.provider_name, "source": self.source_reference, "equation": "6-6",
            "row_basis": HOLFELD_2016_HT_METADATA.row_basis,
            "htc_semantic": "PHYSICAL_FILM", "eta_included": False,
            "coefficient_definition": "Nu=h*do/k; physical film before eta",
            "Nu_Pr_minus_one_third": reduced_nu,
            "calibration": "mixed literature + Holfeld experiments; differing eta-reduction conventions",
            "eta_reduction_caveat": "own data use Weierman, not Hashizume; no recalibration performed",
            "source_reported_regression_coverage": {"classification": "SOURCE_REPORTED_REGRESSION_COVERAGE",
                "points": 284, "fraction_within_10_percent": .81, "fraction_within_21_percent": .95, "R_squared": .958},
        }
        return FinnedTubeHeatTransferResult(
            alpha=h, nusselt_number=Nu, colburn_j_factor=Nu/(Re*Pr**(1/3)),
            stanton_number=Nu/(Re*Pr), reynolds_number=Re, prandtl_number=Pr,
            face_area=b.frontal_flow_area, minimum_free_flow_area=request.minimum_free_flow_area,
            face_mass_flux=request.m_dot/b.frontal_flow_area,
            reference_mass_flux=request.m_dot/request.minimum_free_flow_area,
            face_velocity=request.m_dot/(p.rho*b.frontal_flow_area),
            reference_velocity=request.m_dot/(p.rho*request.minimum_free_flow_area),
            reference_diameter_value=t.D_o, fin_height=t.fin_height,
            correlation_fin_thickness=t.fin_thickness_root, correlation_fin_spacing=t.fin_pitch,
            diagonal_pitch=math.hypot(b.pitch_longitudinal,b.pitch_transverse/2),
            equilateral_relative_deviation=abs(b.pitch_longitudinal/(math.sqrt(3)*b.pitch_transverse/2)-1),
            metadata=HOLFELD_2016_HT_METADATA,
            warnings=(make_warning(code="serrated_source_surface_and_eta_reduction_conventions",
                message="Explicit native strip surface differs from Phase A geometric area; Holfeld calibration uses mixed eta reduction conventions.",
                source=self.provider_name, severity="info"),), diagnostics=diagnostics,
        )
