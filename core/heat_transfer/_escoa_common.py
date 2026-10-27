# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Shared mechanics for the legally reproduced ESCOA solid-fin equations.

Layout-specific factors and metadata live in escoa_inline / escoa_staggered.
Fin-temperature iteration is owned by the existing provider-neutral adapter,
not by this result builder. No efficiency correction is applied to film HTC.
"""
from __future__ import annotations

import math
from typing import Callable

from core.common.warnings import make_warning
from core.heat_transfer.finned_tube_outside import (
    FinnedTubeHeatTransferMetadata, FinnedTubeHeatTransferRequest,
    FinnedTubeHeatTransferResult, _require_positive_finite,
)


def _evaluate_escoa_solid_fin(
    request: FinnedTubeHeatTransferRequest,
    *,
    layout: str,
    provider_name: str,
    metadata: FinnedTubeHeatTransferMetadata,
    unresolved_source_ranges: tuple[str, ...],
    factors: Callable[[FinnedTubeHeatTransferRequest, float, float], tuple[float, float]],
) -> FinnedTubeHeatTransferResult:
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
        raise ValueError(f"ESCOA {layout.upper()} requires pitch_transverse > physical transverse blockage.")
    if request.strict_source_domain:
        raise ValueError(f"SOURCE_RANGE_UNRESOLVED: ESCOA {layout.upper()} required source limits and 1979 property convention cannot be established in strict_source_domain mode.")
    if request.gas_temperature is None or request.mean_fin_temperature is None:
        raise ValueError(f"ESCOA {layout.upper()} requires gas_temperature and mean_fin_temperature [K]; supply a coupled fin-temperature model, not a root/bulk substitution.")

    flux = request.m_dot / request.minimum_free_flow_area
    face_flux = request.m_dot / request.face_area
    reynolds = flux * d_A / request.mu
    prandtl = request.cp * request.mu / request.k
    ratio = request.gas_temperature / request.mean_fin_temperature
    pitch_ratio = request.pitch_longitudinal / request.pitch_transverse
    C3, C5 = factors(request, height, spacing)
    Nu = (0.25 * reynolds**0.65 * prandtl**(1.0/3.0)
          * ratio**0.25 * (request.D_fin/d_A)**0.5 * C3 * C5)
    alpha = Nu * request.k / d_A
    for name, value in (("Re", reynolds), ("Pr", prandtl), ("Nu", Nu), ("h_air", alpha)):
        _require_positive_finite(value, name)
    warnings = [
        make_warning(code="SOURCE_RANGE_UNRESOLVED", source="outside_ht",
                     message=f"ESCOA {layout.upper()} source limits unresolved: " + ", ".join(unresolved_source_ranges) + "; unknown does not mean in-domain."),
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
    diagnostics = dict(
        provider=provider_name, source_provenance=metadata.source,
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
        unresolved_source_ranges=(unresolved_source_ranges +
            (("continuous_root_layer_mapping",) if root_layer_mapping else ()) +
            (("linear_taper_thickness_mapping",) if tapered_mapping else ())),
        warnings=tuple(warnings),
    )
    diagnostics.update(
        D_fin=request.D_fin, metal_tube_od=core_diameter,
        A_face=request.face_area, A_min=request.minimum_free_flow_area,
        u_face=face_flux/request.rho,
        root_mapping=root_layer_mapping, taper_mapping=tapered_mapping,
        correlation_fin_thickness=thickness,
        coupling_iterations=0, coupling_residual=0.0,
        coupling_converged=True, coupling_method="prescribed_mean_fin_temperature",
    )
    diagnostics["Tg/Tfin"] = ratio
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
        diagonal_pitch=(math.hypot(request.pitch_longitudinal, .5*request.pitch_transverse)
                        if layout == "staggered" else math.nan),
        equilateral_relative_deviation=math.nan,
        metadata=metadata, warnings=tuple(warnings), diagnostics=diagnostics,
    )
