# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Native per-row xi and empirical INLINE bank drag from Frass Eq. (191).

Source: Frass/Hofmann/Ponweiser (2015), pp. 86/88/108/118-119,
Eqs. (139)/(144)/(190)-(191), legal publisher reproduction:
https://www.wseas.us/e-library/conferences/2015/Frass/Frass.pdf
Heated normalization and experimental loss decomposition remain partial.
The explicit v0.9.4 engineering policy permits qualified non-strict use.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

from core.common.warnings import make_warning
from core.pressure_drop.finned_tube_pressure_drop import (
    FRASS_INLINE_XI_COEFFICIENT_BASIS, FinnedTubePressureDropMetadata,
    FinnedTubePressureDropRequest, FinnedTubePressureDropResult, _require_positive_finite,
)


FRASS_INLINE_METADATA = FinnedTubePressureDropMetadata(
    method="frass_inline_solid_fin_191",
    source="Frass/Hofmann/Ponweiser (2015), pp.86/88/108/118-119, Eqs.(139)/(144)/(190)-(191); ITE INLINE spiral solid-fin measurements. Heated normalization and loss decomposition partial; qualified engineering bank-drag policy.",
    equation="xi=4.534*Re^-0.34*(Pt/dA)^-0.5*(Fp/dA)^-0.7*(h/dA)^0.5*(Pl/dA)^0.4; dp_row=xi*rho_mean*w_E^2/2; bank_delta_p=n_rows*Kz*dp_row",
    coefficient_definition=FRASS_INLINE_XI_COEFFICIENT_BASIS,
    coefficient_basis=FRASS_INLINE_XI_COEFFICIENT_BASIS,
    geometry_family="round tubes with solid individual radial/spiral fins",
    velocity_basis="w_E=m_dot/(rho_mean*A_min); actual INLINE minimum free-flow area",
    reynolds_basis="Re=w_E*d_A/nu_mean=G_min*d_A/mu_mean; supplied mean outside gas state",
    reference_diameter="bare tube OD d_A; explicit gas-facing D_root engineering mapping for root layers",
    area_basis="actual minimum projected flow area; no empirical area correction",
    row_basis="native xi per row; physical n_rows and Eq.190 Kz each applied once for 2..8; Kz=1 above 8 is ENGINEERING_ASYMPTOTIC_CONVENTION; one row rejected",
    source_fluid="gas; three INLINE geometries, approximately Re 7000..25000 and rows 2/4/6/8; incomplete experimental envelope",
    supported_layouts=("inline",),
    supported_fin_profiles=("solid_circular_constant", "linear_taper_unvalidated_extension"),
    applicability=(),
)

UNRESOLVED_GEOMETRY_RANGES = (
    "d_A", "fin_height", "fin_pitch", "fin_thickness", "Pt", "Pl", "Pl/Pt", "accuracy_deviation",
)


@dataclass(frozen=True)
class FrassInlineSolidFinPressureDropProvider:
    """Evaluate native xi and return whole-bank drag at one mean gas state.

    No Eq.143 correction, Robinson-Briggs conversion, acceleration or external
    local losses enter the provider. Their source uncertainty stays explicit.
    """

    hydraulic_state_basis: ClassVar[str] = "mean_outside_gas"

    def evaluate(self, request: FinnedTubePressureDropRequest) -> FinnedTubePressureDropResult:
        if request.geometry_family != "solid_circular_finned_tube_bank":
            raise ValueError("Frass INLINE requires round solid individual radial/spiral fins; other fin/cross-section families are unsupported.")
        if request.layout != "inline":
            raise ValueError("FrassInlineSolidFinPressureDropProvider supports only INLINE layout.")
        if request.n_rows == 1:
            raise ValueError("Frass INLINE requires at least 2 rows; one-row hydraulics are unsupported in v0.9.4.")
        core_od = request.tube_outer_diameter or request.D_root
        if core_od > request.D_root:
            raise ValueError("Metal tube OD must not exceed D_root.")
        if request.pitch_transverse <= request.D_fin or request.pitch_longitudinal <= request.D_fin:
            raise ValueError("INLINE solid fins require positive transverse and longitudinal fin-tip clearance.")
        root_mapping = core_od < request.D_root
        taper = request.fin_thickness_root != request.fin_thickness_tip
        nonisothermal = (request.temperature_in is not None and request.temperature_out is not None
                         and request.temperature_in != request.temperature_out)
        if request.strict_source_domain:
            if nonisothermal:
                raise ValueError("frass_inline_heated_normalization_unresolved: strict source-domain mode rejects nonisothermal use.")
            if request.n_rows > 8:
                raise ValueError("Strict source-domain mode rejects rows >8 and ENGINEERING_ASYMPTOTIC_CONVENTION.")
            if root_mapping or taper:
                raise ValueError("Strict source-domain mode rejects root-layer mapping and unvalidated taper geometry extension.")
            raise ValueError("SOURCE_RANGE_UNRESOLVED: required Frass INLINE geometry limits and complete loss-scope provenance are unresolved.")

        d_A = request.D_root
        height = (request.D_fin - d_A)/2.0
        flux = request.m_dot/request.minimum_free_flow_area
        face_flux = request.m_dot/request.face_area
        velocity = flux/request.rho
        Re = velocity*d_A/(request.mu/request.rho)
        xi = (4.534*Re**(-0.34)*(request.pitch_transverse/d_A)**(-0.5)
              *(request.fin_pitch/d_A)**(-0.7)*(height/d_A)**0.5
              *(request.pitch_longitudinal/d_A)**0.4)
        Kz = 1.0 if request.n_rows > 8 else 1.0-0.1*(8-request.n_rows)/6.0
        dynamic_pressure = request.rho*velocity**2/2.0
        dp_row = xi*dynamic_pressure
        bank_delta_p = request.n_rows*Kz*dp_row
        for name, value in (("Re", Re), ("xi", xi), ("Kz", Kz), ("dp_row", dp_row), ("bank_delta_p", bank_delta_p)):
            _require_positive_finite(value, name)
        warnings = [
            make_warning(code="SOURCE_RANGE_UNRESOLVED", source="outside_dp", message="Frass INLINE geometry/accuracy source ranges remain unresolved: "+", ".join(UNRESOLVED_GEOMETRY_RANGES)),
            make_warning(code="frass_inline_heated_normalization_unresolved", source="outside_dp", message="Eq.191 heated/isothermal regression normalization is not source-closed. Eq.143 is NOT_APPLIED under the explicit engineering policy; evaluate at mean outside gas state."),
            make_warning(code="frass_inline_loss_scope_partial", source="outside_dp", message="Eq.191 is treated as empirical aerodynamic bank_drag by engineering policy. Experimental tap/channel/acceleration decomposition is partial; acceleration and external losses remain separate."),
        ]
        if Re < 7000.0 or Re > 25000.0:
            warnings.append(make_warning(code="frass_inline_re_evidence_extrapolation", source="outside_dp", message="Re lies outside the approximately reported 7000..25000 experimental context; this is not a complete verified applicability domain."))
        if request.n_rows > 8:
            row_status = "ENGINEERING_ASYMPTOTIC_CONVENTION"
            warnings.append(make_warning(code="frass_inline_rows_asymptotic_convention", source="outside_dp", message="ENGINEERING_ASYMPTOTIC_CONVENTION: rows >8 use Kz=1; Eq.190 is not extrapolated above unity."))
        elif request.n_rows in (3, 5, 7):
            row_status = "CORRELATION_INTERPOLATION"
            warnings.append(make_warning(code="frass_inline_row_interpolation", source="outside_dp", message="Odd row count uses analytic Eq.190 interpolation; directly reported row anchors are 2/4/6/8."))
        else:
            row_status = "SOURCE_ROW_ANCHOR"
        if root_mapping:
            warnings.append(make_warning(code="frass_inline_root_diameter_engineering_mapping", source="outside_dp", message="ENGINEERING_CONVENTION: gas-facing D_root is equivalent d_A; metal tube OD remains separate. This root-layer mapping is not source-validated."))
        if taper:
            warnings.append(make_warning(code="frass_inline_taper_unvalidated_extension", source="outside_dp", message="Linear taper is an unvalidated geometry extension. Actual projected blockage/A_min is retained; Eq.191 has no fin-thickness term or empirical mean-thickness substitution."))
        diagnostics = dict(
            provider=type(self).__name__, source_provenance=FRASS_INLINE_METADATA.source,
            layout=request.layout, n_rows=request.n_rows, row_status=row_status,
            xi=xi, Kz=Kz, dp_row=dp_row, bank_delta_p=bank_delta_p,
            coefficient_basis=FRASS_INLINE_XI_COEFFICIENT_BASIS,
            dynamic_pressure=dynamic_pressure, w_E=velocity, Re=Re,
            rho_mean=request.rho, mu_mean=request.mu, nu_mean=request.mu/request.rho,
            hydraulic_state_basis=self.hydraulic_state_basis,
            temperature_in=request.temperature_in, temperature_out=request.temperature_out,
            nonisothermal=nonisothermal, heated_normalization="NOT_APPLIED",
            heated_normalization_source_closed=False,
            loss_scope="EMPIRICAL_AERODYNAMIC_BANK_DRAG_ENGINEERING_POLICY",
            loss_scope_source_closed=False,
            includes_acceleration=False, includes_external_losses=False,
            d_A=d_A, D_root=request.D_root, tube_outer_diameter=core_od, h=height,
            diameter_mapping_provenance="ENGINEERING_CONVENTION" if root_mapping else "SOURCE_GEOMETRY_DEFINITION",
            source_geometry_mapping_closed=not (root_mapping or taper),
            fin_thickness_root=request.fin_thickness_root, fin_thickness_tip=request.fin_thickness_tip,
            taper_status="UNVALIDATED_GEOMETRY_EXTENSION" if taper else "CONSTANT_THICKNESS",
            Fp=request.fin_pitch, Pt=request.pitch_transverse, Pl=request.pitch_longitudinal,
            minimum_free_flow_area=request.minimum_free_flow_area, face_area=request.face_area,
            u_face=face_flux/request.rho, source_range_status="SOURCE_RANGE_UNRESOLVED",
            Re_evidence_interval_approx=(7000.0, 25000.0),
            unresolved_source_ranges=UNRESOLVED_GEOMETRY_RANGES,
            warnings=tuple(warnings),
        )
        diagnostics["Pl/Pt"] = request.pitch_longitudinal/request.pitch_transverse
        return FinnedTubePressureDropResult(
            coefficient=xi, coefficient_definition=FRASS_INLINE_XI_COEFFICIENT_BASIS,
            coefficient_basis=FRASS_INLINE_XI_COEFFICIENT_BASIS, dp_drag=bank_delta_p,
            reynolds_number=Re, face_area=request.face_area,
            minimum_free_flow_area=request.minimum_free_flow_area,
            face_mass_flux=face_flux, reference_mass_flux=flux,
            face_velocity=face_flux/request.rho, reference_velocity=velocity,
            reference_diameter_value=d_A, n_rows_effective=request.n_rows,
            fin_height=height, correlation_fin_thickness=math.nan,
            correlation_fin_spacing=math.nan, diagonal_pitch=math.nan,
            equilateral_relative_deviation=math.nan, metadata=FRASS_INLINE_METADATA,
            warnings=tuple(warnings), diagnostics=diagnostics,
        )
