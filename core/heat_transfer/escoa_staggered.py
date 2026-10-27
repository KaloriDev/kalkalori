# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""ESCOA STAGGERED solid-fin thermal model, explicit selection only.

Implemented from the legal secondary reproduction in Frass/Hofmann/Ponweiser
(2015), printed pp.17-18, Eqs.(60)-(62):
https://www.wseas.us/e-library/conferences/2015/Frass/Frass.pdf
The original ESCOA 1979 manual was not inspected. Neither a complete source
domain nor experimental accuracy is established by this reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

from core.heat_transfer._escoa_common import _evaluate_escoa_solid_fin
from core.heat_transfer.finned_tube_outside import (
    FinnedTubeHeatTransferMetadata, FinnedTubeHeatTransferRequest,
    FinnedTubeHeatTransferResult,
)


FRASS_ESCOA_STAGGERED_METADATA = FinnedTubeHeatTransferMetadata(
    method="frass_escoa_staggered_solid_fin",
    source=("ESCOA solid-fin STAGGERED model, implemented from the legal secondary "
            "reproduction in Frass/Hofmann/Ponweiser (2015), printed pp.17-18, "
            "Eqs.(60)-(62). Original ESCOA 1979 manual not inspected."),
    equation=("Nu=0.25*Re^0.65*Pr^(1/3)*(Tg/Tfin)^0.25*(Dfin/dA)^0.5*C3*C5; "
              "C3=0.35+0.65*exp(-0.25*h/(Fp-tf)); "
              "C5=0.7+(0.7-0.8*exp(-0.15*n_rows^2))*exp(-Pl/Pt)"),
    geometry_family="solid individual spiral/radial fins on round tubes",
    velocity_basis="u_reference=m_dot/(rho_supplied*A_min), actual resolved STAGGERED free area",
    reynolds_basis="Re=(m_dot/A_min)*d_A/mu_supplied; ENGINEERING_CONVENTION",
    reference_diameter=("source bare tube OD d_A; gas-facing D_root is the explicit "
                        "engineering equivalent for a continuous root layer"),
    area_basis="physical film HTC before fin/overall surface efficiency",
    row_basis="consecutive physical rows n_rows squared in C5; no additional row correction",
    source_fluid="gas; original fluid/accuracy envelope unresolved",
    supported_layouts=("staggered",),
    supported_fin_profiles=("solid_circular_constant_or_linear_taper_engineering_mapping",),
    applicability=(),  # Missing limits do not imply an unbounded valid domain.
)

UNRESOLVED_SOURCE_RANGES = (
    "Re", "d_A", "fin_height", "fin_thickness", "fin_pitch",
    "pitch_transverse", "pitch_longitudinal", "Pl/Pt", "accuracy_deviation",
)


@dataclass(frozen=True)
class FrassEscoaStaggeredSolidFinProvider:
    """Physical film HTC on resolved staggered round solid-fin geometry.

    Geometry, supplied-property and strict-domain policies are qualified in
    diagnostics. This provider does not impose Briggs-Young's equilateral
    source restriction, accept arbitrary shifted banks, or change AUTO.
    """

    requires_fin_temperature: ClassVar[bool] = True

    def evaluate(self, request: FinnedTubeHeatTransferRequest) -> FinnedTubeHeatTransferResult:
        if request.geometry_family != "solid_circular_finned_tube_bank":
            raise ValueError("ESCOA STAGGERED requires round tubes with solid individual radial/spiral fins; serrated, segmented, louvered, elliptical and plain geometries are unsupported.")
        if request.layout != "staggered":
            raise ValueError("FrassEscoaStaggeredSolidFinProvider supports only resolved STAGGERED layout; inline and arbitrary shifted layouts are unsupported.")
        return _evaluate_escoa_solid_fin(
            request, layout="staggered", provider_name=type(self).__name__,
            metadata=FRASS_ESCOA_STAGGERED_METADATA,
            unresolved_source_ranges=UNRESOLVED_SOURCE_RANGES,
            factors=_staggered_factors,
        )


def _staggered_factors(request: FinnedTubeHeatTransferRequest, height: float,
                       spacing: float) -> tuple[float, float]:
    # Frass (2015), Eq.(61): STAGGERED SOLID, not the serrated Eq.(63).
    C3 = 0.35 + 0.65 * math.exp(-0.25 * height / spacing)
    # Eq.(62): the source has n_R SQUARED inside this exponential.
    C5 = (0.7 + (0.7 - 0.8 * math.exp(-0.15 * request.n_rows**2))
          * math.exp(-request.pitch_longitudinal / request.pitch_transverse))
    return C3, C5
