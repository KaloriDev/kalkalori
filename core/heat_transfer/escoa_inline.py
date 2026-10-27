# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Old ESCOA INLINE solid-fin thermal model, reproduced by Frass (2015).

Implementation source: Frass/Hofmann/Ponweiser, Principles of Finned-Tube
Heat Exchanger Design, 2nd ed., p.39 Eqs.(96)-(98), p.18 Eq.(60).
https://www.wseas.us/e-library/conferences/2015/Frass/Frass.pdf
The inaccessible 1979 manual was not used directly. No TURB-X revision
coefficients, invented domain limits, or additional row multipliers are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

from core.heat_transfer.finned_tube_outside import (
    FinnedTubeHeatTransferMetadata, FinnedTubeHeatTransferRequest,
    FinnedTubeHeatTransferResult,
)


FRASS_ESCOA_INLINE_METADATA = FinnedTubeHeatTransferMetadata(
    method="frass_escoa_inline_solid_fin",
    source=("ESCOA 1979 solid-fin model, implemented from legal secondary "
            "reproduction in Frass/Hofmann/Ponweiser (2015), p.39 Eqs.(96)-(98); "
            "temperature definitions p.18 Eq.(60). Primary manual not inspected."),
    equation=("Nu=0.25*Re^0.65*Pr^(1/3)*(Tg/Tfin)^0.25*(Dfin/dA)^0.5*C3*C5; "
              "C3=0.20+0.65*exp(-0.25*h/(Fp-tf)); "
              "C5=1.1-(0.75-1.5*exp(-0.70*n_rows))*exp(-2*Pl/Pt)"),
    geometry_family="solid individual spiral/radial fins on round tubes",
    velocity_basis="u_reference=m_dot/(rho_supplied*A_min), actual INLINE free area",
    reynolds_basis="Re=(m_dot/A_min)*d_A/mu_supplied; ENGINEERING_CONVENTION",
    reference_diameter=("source bare tube OD d_A; gas-facing D_root is the explicit "
                        "engineering equivalent for a continuous root layer"),
    area_basis="physical film HTC before fin/overall surface efficiency",
    row_basis="consecutive physical rows in C5; no additional row correction",
    source_fluid="gas; original fluid/accuracy envelope unresolved",
    supported_layouts=("inline",),
    supported_fin_profiles=("solid_circular_constant_or_linear_taper_engineering_mapping",),
    applicability=(),  # Unknown source ranges are not unbounded valid ranges.
)

UNRESOLVED_SOURCE_RANGES = (
    "Re", "d_A", "fin_height", "fin_thickness", "fin_pitch",
    "pitch_transverse", "pitch_longitudinal", "Pl/Pt", "accuracy_deviation",
)


@dataclass(frozen=True)
class FrassEscoaInlineSolidFinProvider:
    """Physical film HTC; prescribed mean fin temperature or coupled adapter.

    Supplied transport properties are reused without temperature reevaluation.
    Workflows supply outside bulk properties. This engineering convention is
    explicit and is not claimed source-exact for the 1979 ESCOA model.
    """

    requires_fin_temperature: ClassVar[bool] = True

    def evaluate(self, request: FinnedTubeHeatTransferRequest) -> FinnedTubeHeatTransferResult:
        if request.geometry_family != "solid_circular_finned_tube_bank":
            raise ValueError("ESCOA INLINE requires round tubes with solid individual radial/spiral fins; serrated, segmented and elliptical geometries are unsupported.")
        if request.layout != "inline":
            raise ValueError("FrassEscoaInlineSolidFinProvider supports only INLINE layout.")
        from core.heat_transfer._escoa_common import _evaluate_escoa_solid_fin
        return _evaluate_escoa_solid_fin(
            request, layout="inline", provider_name=type(self).__name__,
            metadata=FRASS_ESCOA_INLINE_METADATA,
            unresolved_source_ranges=UNRESOLVED_SOURCE_RANGES,
            factors=_inline_factors,
        )


def _inline_factors(request: FinnedTubeHeatTransferRequest, height: float,
                    spacing: float) -> tuple[float, float]:
    # Frass (2015), Eq.97: INLINE SOLID constant 0.20.
    C3 = 0.20 + 0.65 * math.exp(-0.25 * height / spacing)
    # Eq.98 includes physical rows; no borrowed/local-row multiplier.
    pitch_ratio = request.pitch_longitudinal / request.pitch_transverse
    C5 = 1.1 - (0.75 - 1.5 * math.exp(-0.70 * request.n_rows)) * math.exp(-2.0 * pitch_ratio)
    return C3, C5
