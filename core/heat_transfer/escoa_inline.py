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

from core.common.warnings import make_warning
from core.heat_transfer.finned_tube_outside import (
    FinnedTubeHeatTransferMetadata, FinnedTubeHeatTransferRequest,
    FinnedTubeHeatTransferResult, _require_positive_finite,
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
        core_diameter = request.tube_outer_diameter or request.D_root
        if core_diameter > request.D_root:
            raise ValueError("Bare tube OD must not exceed the gas-facing fin-root diameter.")
        # Source D=d_A+2*h with actual radial fin height identifies the
        # gas-facing root cylinder. For a continuous layer over a smaller
        # metal core this is an explicitly qualified engineering mapping,
        # authorized for non-strict calculations, not experimental closure.
        d_A = request.D_root
        root_layer_mapping = core_diameter < request.D_root
        tapered_mapping = request.fin_thickness_root != request.fin_thickness_tip
        height = 0.5 * (request.D_fin - d_A)
        # Existing circular-finned correlation convention: map only the
        # empirical thickness/clear-spacing terms to the arithmetic mean.
        # Actual taper stays in geometry, projected blockage and conduction.
        thickness = 0.5 * (request.fin_thickness_root + request.fin_thickness_tip)
        spacing = request.fin_pitch - thickness
        blockage = d_A + 2.0 * height * thickness / request.fin_pitch
        if request.pitch_transverse <= blockage:
            raise ValueError("ESCOA INLINE requires pitch_transverse > physical transverse blockage.")
        if request.strict_source_domain:
            raise ValueError("SOURCE_RANGE_UNRESOLVED: ESCOA INLINE required source limits and 1979 property convention cannot be established in strict_source_domain mode.")
        if request.gas_temperature is None or request.mean_fin_temperature is None:
            raise ValueError("ESCOA INLINE requires gas_temperature and mean_fin_temperature [K]; supply a coupled fin-temperature model, not a root/bulk substitution.")

        flux = request.m_dot / request.minimum_free_flow_area
        face_flux = request.m_dot / request.face_area
        reynolds = flux * d_A / request.mu
        prandtl = request.cp * request.mu / request.k
        ratio = request.gas_temperature / request.mean_fin_temperature
        pitch_ratio = request.pitch_longitudinal / request.pitch_transverse
        # Frass (2015), Eq.97: INLINE SOLID constant 0.20.
        C3 = 0.20 + 0.65 * math.exp(-0.25 * height / spacing)
        # Eq.98 includes physical rows; no borrowed/local-row multiplier.
        C5 = 1.1 - (0.75 - 1.5 * math.exp(-0.70 * request.n_rows)) * math.exp(-2.0 * pitch_ratio)
        Nu = (0.25 * reynolds**0.65 * prandtl**(1.0/3.0)
              * ratio**0.25 * (request.D_fin/d_A)**0.5 * C3 * C5)
        alpha = Nu * request.k / d_A
        for name, value in (("Re", reynolds), ("Pr", prandtl), ("Nu", Nu), ("h_air", alpha)):
            _require_positive_finite(value, name)
        warnings = [
            make_warning(code="SOURCE_RANGE_UNRESOLVED", source="outside_ht",
                         message="ESCOA INLINE source limits unresolved: " + ", ".join(UNRESOLVED_SOURCE_RANGES) + "; unknown does not mean in-domain."),
            make_warning(code="escoa_property_convention_engineering", source="outside_ht",
                         message="Re, Pr and conductivity reuse supplied transport properties (outside bulk in workflows); exact 1979 ESCOA property convention is not source-closed."),
        ]
        if root_layer_mapping:
            warnings.append(make_warning(
                code="escoa_root_diameter_engineering_mapping", source="outside_ht",
                message=("ENGINEERING_CONVENTION: continuous root layer uses gas-facing "
                         "D_root as equivalent ESCOA d_A; metal tube OD remains separate "
                         "in the conduction/contact network. This mapping is not source-validated."),
            ))
        if tapered_mapping:
            warnings.append(make_warning(
                code="escoa_tapered_fin_engineering_mapping", source="outside_ht",
                message=("ENGINEERING_CONVENTION: empirical tf is the arithmetic mean "
                         "of root/tip thickness, following the existing circular-finned "
                         "mapping convention. Actual taper remains in area, blockage and "
                         "fin conduction. This is not source-validated tapered-fin ESCOA."),
            ))
        if request.n_rows == 1:
            warnings.append(make_warning(code="escoa_single_row_caveat", source="outside_ht",
                message="Frass notes single-row layout formulas may disagree (up to about 15% for small height/clear spacing); C5 retains Pl/Pt even though a single row has no longitudinal pitch. Computability is not single-row validation."))
        metadata = FRASS_ESCOA_INLINE_METADATA
        diagnostics = dict(
            provider=type(self).__name__, source_provenance=metadata.source,
            layout=request.layout, n_rows=request.n_rows, d_A=d_A, h=height,
            D_root=request.D_root, tube_outer_diameter=core_diameter,
            diameter_mapping=("gas_facing_root_cylinder" if root_layer_mapping else "bare_tube_OD"),
            diameter_mapping_provenance=("ENGINEERING_CONVENTION" if root_layer_mapping else "SOURCE_GEOMETRY_DEFINITION"),
            source_geometry_mapping_closed=not (root_layer_mapping or tapered_mapping),
            Fp=request.fin_pitch, tf=thickness, Pt=request.pitch_transverse,
            fin_thickness_root=request.fin_thickness_root,
            fin_thickness_tip=request.fin_thickness_tip,
            thickness_mapping=("arithmetic_mean_root_tip" if tapered_mapping else "constant_thickness"),
            thickness_mapping_provenance=("ENGINEERING_CONVENTION" if tapered_mapping else "SOURCE_GEOMETRY_DEFINITION"),
            Pl=request.pitch_longitudinal, pitch_ratio=pitch_ratio,
            u_reference=flux/request.rho, Re=reynolds, Pr=prandtl,
            T_g_mean=request.gas_temperature, T_fin_mean=request.mean_fin_temperature,
            temperature_ratio=ratio, C3=C3, C5=C5, Nu=Nu, h_air=alpha,
            property_convention="supplied_transport_properties; workflow_outside_bulk",
            property_convention_provenance="ENGINEERING_CONVENTION",
            source_property_convention_closed=False,
            source_range_status="SOURCE_RANGE_UNRESOLVED",
            unresolved_source_ranges=(UNRESOLVED_SOURCE_RANGES +
                (("continuous_root_layer_mapping",) if root_layer_mapping else ()) +
                (("linear_taper_thickness_mapping",) if tapered_mapping else ())),
            warnings=tuple(warnings),
        )
        diagnostics["Pl/Pt"] = pitch_ratio
        j = Nu / (reynolds * prandtl**(1.0/3.0))
        return FinnedTubeHeatTransferResult(
            alpha=alpha, nusselt_number=Nu, colburn_j_factor=j,
            stanton_number=Nu/(reynolds*prandtl), reynolds_number=reynolds,
            prandtl_number=prandtl, face_area=request.face_area,
            minimum_free_flow_area=request.minimum_free_flow_area,
            face_mass_flux=face_flux, reference_mass_flux=flux,
            face_velocity=face_flux/request.rho, reference_velocity=flux/request.rho,
            reference_diameter_value=d_A, fin_height=height,
            correlation_fin_thickness=thickness, correlation_fin_spacing=spacing,
            diagonal_pitch=math.nan, equilateral_relative_deviation=math.nan,
            metadata=metadata, warnings=tuple(warnings), diagnostics=diagnostics,
        )
