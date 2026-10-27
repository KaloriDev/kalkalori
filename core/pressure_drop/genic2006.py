# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Explicit Genić et al. (2006) normal-staggered/inline solid-fin models.

Independent implementation of Eqs.(10), (12), (15) and their geometry
definitions. DOI: 10.1016/j.applthermaleng.2005.07.017. Equation definitions
are closed; calibration provenance is partial and raw data were not recovered.
AUTO selection does not use these providers. Dense packing and <5 rows are
excluded because their separate equations are outside this implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

from core.common.warnings import ApplicabilityRange, make_warning
from .finned_tube_pressure_drop import (
    GENIC_2006_XI_COEFFICIENT_BASIS,
    FinnedTubePressureDropMetadata,
    FinnedTubePressureDropRequest,
    FinnedTubePressureDropResult,
    _require_positive_finite,
)


def _metadata(layout: str) -> FinnedTubePressureDropMetadata:
    inline = layout == "inline"
    equation = ("xi=(0.4+8.5/Re_de^0.3)*W^-0.75*Ri^0.7" if inline else
                "xi=(0.43+17.7/Re_de^0.3)*W^-0.8*Rd^0.3")
    return FinnedTubePressureDropMetadata(
        method=f"genic2006_{layout}",
        source="Genić, Jaćimović and Latinović (2006), Applied Thermal Engineering 26, 478-485; DOI 10.1016/j.applthermaleng.2005.07.017. Calibration provenance PARTIAL; raw calibration data NOT RECOVERED.",
        equation=equation+"; delta_p=n_rows*xi*(Pl/de)*rho*w_min^2/2 (Eq.15)",
        coefficient_definition=GENIC_2006_XI_COEFFICIENT_BASIS,
        coefficient_basis=GENIC_2006_XI_COEFFICIENT_BASIS,
        geometry_family="round tubes with solid constant-thickness individual helical/annular fins",
        velocity_basis="w_min=m_dot/(rho*A_min); source frontal passage af",
        reynolds_basis="Re_de=rho*w_min*de/mu; de=4*epsilon/Sv",
        reference_diameter="source equivalent diameter de=4*epsilon/Sv",
        area_basis="af per fin-pitch segment; A_min=A_face*af/(Pt*sf); frontal passage controls normal staggered Rd>=1",
        row_basis="physical n_rows>=5, no short-bank correction; xi multiplied once by n_rows*Pl/de",
        source_fluid="air",
        supported_layouts=(layout,),
        supported_fin_profiles=("solid_circular_constant",),
        applicability=(
            # ApplicabilityRange represents closed intervals only. The open
            # Reynolds interval is exposed separately in result diagnostics.
            (ApplicabilityRange("Ri", .5, 2.),) if inline else
            (ApplicabilityRange("Rd", 1., None),
             ApplicabilityRange("Pt/d", None, 3.5),
             ApplicabilityRange("Pl/d", None, 3.5))
        ) + (ApplicabilityRange("n_rows", 5., None, "rows"),),
    )


@dataclass(frozen=True)
class Genic2006StaggeredPressureDropProvider:
    """Normal staggered Eq.(10), native Eq.(15) bank drag; explicit only."""

    hydraulic_state_basis: ClassVar[str] = "mean_outside_gas"

    def evaluate(self, request: FinnedTubePressureDropRequest) -> FinnedTubePressureDropResult:
        return _evaluate(request, "staggered")


@dataclass(frozen=True)
class Genic2006InlinePressureDropProvider:
    """Inline Eq.(12), native Eq.(15) bank drag; explicit only."""

    hydraulic_state_basis: ClassVar[str] = "mean_outside_gas"

    def evaluate(self, request: FinnedTubePressureDropRequest) -> FinnedTubePressureDropResult:
        return _evaluate(request, "inline")


def _evaluate(r: FinnedTubePressureDropRequest, layout: str) -> FinnedTubePressureDropResult:
    if r.geometry_family != "solid_circular_finned_tube_bank":
        raise ValueError("Genić 2006 requires round solid individual fins.")
    if r.layout != layout:
        raise ValueError(f"Genić 2006 {layout} provider supports only {layout} layout.")
    if r.n_rows < 5:
        raise ValueError("Genić 2006 requires n_rows>=5; the separate C_Nr correction is not implemented.")
    if r.fin_thickness_root != r.fin_thickness_tip:
        raise ValueError("Genić 2006 requires constant-thickness fins; taper mapping is not source-closed.")
    metal_od = r.tube_outer_diameter or r.D_root
    if metal_od > r.D_root:
        raise ValueError("Metal tube OD must not exceed D_root.")
    d, D, sf, delta = r.D_root, r.D_fin, r.fin_pitch, r.fin_thickness_root
    Pt, Pl = r.pitch_transverse, r.pitch_longitudinal
    if Pt <= d or Pl <= d:
        raise ValueError("Genić 2006 requires positive root clearances Pt-d and Pl-d.")
    sd = math.hypot(Pl, Pt/2.)
    af = (Pt-d)*(sf-delta)+delta*(Pt-D)
    ad = 2.*((sd-d)*(sf-delta)+delta*(sd-D))
    asurf = math.pi*d*(sf-delta)+math.pi/2.*(D**2-d**2)+math.pi*D*delta
    epsilon = 1.-math.pi*(d**2*(sf-delta)+D**2*delta)/(4.*Pt*Pl*sf)
    Sv = asurf/(Pt*Pl*sf)
    for name, value in (("af", af), ("ad", ad), ("asurf", asurf), ("epsilon", epsilon), ("Sv", Sv)):
        _require_positive_finite(value, name)
    de = 4.*epsilon/Sv
    Rd, Ri, W = ad/af, (Pl-d)/(Pt-d), asurf/af
    if layout == "staggered" and Rd < 1.:
        raise ValueError("Genić Eq.(10) requires Rd>=1; densely-packed Eq.(11) is not implemented.")
    expected_area = r.face_area*af/(Pt*sf)
    if not math.isclose(r.minimum_free_flow_area, expected_area, rel_tol=1e-10, abs_tol=0.):
        raise ValueError("Genić source frontal af does not match supplied minimum_free_flow_area.")
    flux = r.m_dot/r.minimum_free_flow_area
    face_flux = r.m_dot/r.face_area
    velocity = flux/r.rho
    Re = flux*de/r.mu
    failures = []
    if layout == "staggered":
        if Pt/d > 3.5:
            failures.append("Pt/d<=3.5")
        if Pl/d > 3.5:
            failures.append("Pl/d<=3.5")
        if (not 550. < Re < 130000. or
                math.isclose(Re, 550., rel_tol=1e-14) or
                math.isclose(Re, 130000., rel_tol=1e-14)):
            failures.append("550<Re_de<130000 (open interval)")
        xi = (.43+17.7/Re**.3)*W**(-.8)*Rd**.3
    else:
        # Inclusive bounds tolerate roundoff in the subtracted clearances.
        if ((Ri < .5 and not math.isclose(Ri, .5, rel_tol=1e-14)) or
                (Ri > 2. and not math.isclose(Ri, 2., rel_tol=1e-14))):
            failures.append("0.5<=Ri<=2")
        if (not 6000. < Re < 200000. or
                math.isclose(Re, 6000., rel_tol=1e-14) or
                math.isclose(Re, 200000., rel_tol=1e-14)):
            failures.append("6000<Re_de<200000 (open interval)")
        xi = (.4+8.5/Re**.3)*W**(-.75)*Ri**.7
    warnings = []
    if failures:
        warnings.append(make_warning(code="genic2006_source_domain_extrapolation", source="outside_dp",
            message="SOURCE_DOMAIN_FAIL: "+", ".join(failures)))
    extensions = []
    if metal_od < d:
        extensions.append("root_diameter_engineering_mapping")
    nonisothermal = (r.temperature_in is not None and r.temperature_out is not None
                     and r.temperature_in != r.temperature_out)
    if nonisothermal:
        extensions.append("mean_state_nonisothermal_extension")
    if r.fluid_family.strip().lower() != "air":
        extensions.append("non_air_extension")
    for extension in extensions:
        warnings.append(make_warning(code="genic2006_"+extension, source="outside_dp",
            message="ENGINEERING_CONVENTION: "+extension+" is outside the closed source geometry/state. No heated-flow multiplier is applied; acceleration stays separate."))
    if r.strict_source_domain and (failures or extensions):
        raise ValueError("Genić strict source-domain rejection: "+", ".join(failures+extensions))
    diagnostics = dict(
        provider=("Genic2006InlinePressureDropProvider" if layout == "inline" else "Genic2006StaggeredPressureDropProvider"),
        layout=layout, source_equation=f"Genić 2006 Eq.({'12' if layout == 'inline' else '10'})",
        source_equation_closed=True, calibration_provenance="PARTIAL",
        raw_calibration_data_recovered=False,
        source_domain_status="SOURCE_DOMAIN_FAIL" if failures else "SOURCE_DOMAIN_PASS",
        source_domain_failures=tuple(failures), source_domain_coverage="AUDITED_EXPLICIT_LIMITS_ONLY",
        geometry_state_extensions=tuple(extensions),
        d=d, D=D, sf=sf, delta=delta, Pt=Pt, Pl=Pl, sd=sd, af=af, ad=ad,
        asurf=asurf, Rd=Rd, Ri=Ri, W=W, epsilon=epsilon, Sv=Sv, de=de,
        tube_outer_diameter=metal_od, minimum_free_flow_area=r.minimum_free_flow_area,
        source_frontal_area=expected_area, w_min=velocity, rho=r.rho, mu=r.mu, Re_de=Re,
        xi=xi, zeta_row=xi*Pl/de, row_correction=1., n_rows=r.n_rows,
        row_status="BASE_EQUATION_NR_GE_5", hydraulic_state_basis="mean_outside_gas",
        heated_correction="NOT_APPLIED", includes_acceleration=False, includes_external_losses=False,
    )
    if layout == "inline":
        calibration = dict(d=.032, D=.050, sf=.006, delta=.0013, Nr=7)
        actual = dict(d=d, D=D, sf=sf, delta=delta, Nr=r.n_rows)
        # Equality to the recovered family, not an invented tolerance band for accuracy.
        extrapolation = any(not math.isclose(actual[k], v, rel_tol=1e-12, abs_tol=0.)
                            for k, v in calibration.items())
        diagnostics.update(calibration_source="Yudin 1982 via Genić",
            reynolds_source_interval=dict(lower=6000., upper=200000.,
                lower_inclusive=False, upper_inclusive=False),
            known_calibration_geometry=calibration, calibration_bundles=6,
            calibration_regimes=91, calibration_geometry_narrow=True,
            calibration_geometry_status=("CALIBRATION_GEOMETRY_EXTRAPOLATION" if extrapolation else "KNOWN_CALIBRATION_GEOMETRY_MATCH"),
            calibration_geometry_comparison="Equality to recovered dimensions/rows; not a source-derived applicability bound",
            calibration_geometry_ratios={k: actual[k]/v for k, v in calibration.items()})
        if extrapolation:
            warnings.append(make_warning(code="genic2006_inline_calibration_geometry_extrapolation", source="outside_dp",
                message="Geometry/rows differ from the narrow recovered Yudin calibration family. This coverage warning is separate from Eq.(12) source-domain limits and is not a hard bound."))
    else:
        diagnostics["calibration_geometry_status"] = "PARTIAL_PROVENANCE"
        diagnostics["reynolds_source_interval"] = dict(
            lower=550., upper=130000., lower_inclusive=False, upper_inclusive=False)
    dynamic_pressure = r.rho*velocity**2/2.
    dp_row = xi*(Pl/de)*dynamic_pressure
    bank_dp = r.n_rows*dp_row
    for name, value in (("de", de), ("Re_de", Re), ("xi", xi), ("bank_delta_p", bank_dp)):
        _require_positive_finite(value, name)
    diagnostics.update(dp_row=dp_row, bank_delta_p=bank_dp, dynamic_pressure=dynamic_pressure,
                       coefficient_basis=GENIC_2006_XI_COEFFICIENT_BASIS)
    metadata = _metadata(layout)
    return FinnedTubePressureDropResult(
        coefficient=xi, coefficient_definition=GENIC_2006_XI_COEFFICIENT_BASIS,
        coefficient_basis=GENIC_2006_XI_COEFFICIENT_BASIS, dp_drag=bank_dp,
        reynolds_number=Re, face_area=r.face_area, minimum_free_flow_area=r.minimum_free_flow_area,
        face_mass_flux=face_flux, reference_mass_flux=flux, face_velocity=face_flux/r.rho,
        reference_velocity=velocity, reference_diameter_value=de, n_rows_effective=r.n_rows,
        fin_height=(D-d)/2., correlation_fin_thickness=delta,
        correlation_fin_spacing=sf-delta, diagonal_pitch=sd,
        equilateral_relative_deviation=abs(sd/Pt-1.), metadata=metadata,
        warnings=tuple(warnings), diagnostics=diagnostics,
    )
