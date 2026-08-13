# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
#
# -------------------------------------------------------------------------
# OUTSIDE FLOW — CROSSFLOW OVER CIRCULAR FINNED-TUBE BANK (v0.7.x)
# -------------------------------------------------------------------------
#
# Dry, single-phase, gas-side, full circular (annular) fins, staggered
# (triangular pitch) tube banks only. This module never modifies or
# extends core.heat_transfer.outside_flow (the bare-tube module); it is
# a self-contained sibling with its own velocity/Re/Nu definitions, so
# bare-tube results are provably unaffected.
#
# Heat transfer (Nusselt number):
#   Briggs, D.E.; Young, E.H. (1963), "Convection Heat Transfer and
#   Pressure Drop of Air Flowing across Triangular Pitch Banks of Finned
#   Tubes", Chemical Engineering Progress Symposium Series, Vol. 59,
#   No. 41, pp. 1-10.
#
#       Nu = 0.134 * Re^0.681 * Pr^(1/3) * (s/l)^0.2 * (s/b)^0.1134
#
#   with:
#     - Re = rho * V_max * D_root / mu   (V_max on the fin-blockage-aware
#       minimum free-flow area; D_root is the bare/root tube OD, i.e.
#       CircularFinnedTube.D_root)
#     - Nu = h * D_root / k              (h is the PHYSICAL, not
#       fin-efficiency-weighted, coefficient on the true finned area)
#     - s  = clear axial gap between fin roots (fin_pitch - fin_thickness_root)
#     - l  = fin_height
#     - b  = fin_thickness_root
#   Independently corroborated (see docs/finned_tube_model.md) against:
#     (1) Camaraza-Medina et al. (2018), Math. Model. Eng. Probl. 5(4), Eq. 15
#     (2) Caleb Bell, `ht` (MIT license), ht.air_cooler.h_Briggs_Young
#         docstring (arXiv-independent open-source cross-check; formula
#         only, not copied as code)
#   Reported range of applicability: 1000 < Re < 8000, 11.13 mm < D_root
#   < 40.89 mm, 1.42 mm < fin_height < 16.57 mm, 0.33 mm <
#   fin_thickness_root < 2.02 mm, 1.30 mm < fin_pitch < 4.06 mm,
#   24.49 mm < S_T < 111 mm; staggered (triangular pitch) tube banks.
#
# Pressure drop:
#   Delegated to core.pressure_drop.outside_pressure_drop's
#   RobinsonBriggsEulerProvider, which is a documented, geometry-gated
#   blocker in this pass (see that module and
#   docs/finned_tube_model.md, "Unresolved limitations"). Calling this
#   module with calculate_pressure_drop=True does not crash: the
#   NotImplementedError is caught and surfaced as dp_o=NaN plus a
#   critical ModelWarning, so HTC/UA results remain usable even though
#   Delta p is not available.
#
# Fin efficiency and area weighting are NOT performed in this module
# (see core.heat_transfer.fin_efficiency /
# core.heat_transfer.finned_tube_resistance): this module returns the
# PHYSICAL outside heat-transfer coefficient only.
# -------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

from core.common.warnings import ModelWarning, make_warning
from core.geometry.finned_flow_geometry import finned_vmax_ratio_min_freeflow
from core.geometry.finned_tube import CircularFinnedTube
from core.heat_transfer.outside_flow import FluidProps, prandtl_number, reynolds_number
from core.pressure_drop.outside_pressure_drop import (
    EulerProvider,
    EulerRequest,
    EulerResult,
    check_outside_dp_applicability,
    evaluate_euler,
    pressure_drop_from_euler,
)

METHOD_NAME = "briggs_young_1963"
SOURCE_CITATION = (
    "Briggs, D.E.; Young, E.H. (1963), \"Convection Heat Transfer and "
    "Pressure Drop of Air Flowing across Triangular Pitch Banks of "
    "Finned Tubes\", Chemical Engineering Progress Symposium Series, "
    "Vol. 59, No. 41, pp. 1-10."
)
GEOMETRY_FAMILY = "circular_finned_tube_bank"
VELOCITY_BASIS = "maximum_gap_velocity_fin_blockage_aware"
REYNOLDS_BASIS = "root_diameter_Vmax"
REFERENCE_DIAMETER = "D_root"
AREA_BASIS = "physical_finned_surface_no_fin_efficiency"
ROW_BASIS = "not_row_resolved_0D"

# Applicability envelope reported for the Briggs & Young (1963)
# correlation (see module docstring for provenance).
RE_MIN, RE_MAX = 1000.0, 8000.0
D_ROOT_MIN, D_ROOT_MAX = 0.01113, 0.04089
FIN_HEIGHT_MIN, FIN_HEIGHT_MAX = 0.00142, 0.01657
FIN_THICKNESS_MIN, FIN_THICKNESS_MAX = 0.00033, 0.00202
FIN_PITCH_MIN, FIN_PITCH_MAX = 0.00130, 0.00406
ST_MIN, ST_MAX = 0.02449, 0.111


@dataclass(frozen=True)
class FinnedCorrelationContract:
    """Explicit self-reporting contract for a finned-tube outside provider
    (see docs/finned_tube_model.md for the full table)."""

    method: str
    source: str
    geometry_family: str
    velocity_basis: str
    reynolds_basis: str
    reference_diameter: str
    area_basis: str
    row_basis: str


HTC_CONTRACT = FinnedCorrelationContract(
    method=METHOD_NAME,
    source=SOURCE_CITATION,
    geometry_family=GEOMETRY_FAMILY,
    velocity_basis=VELOCITY_BASIS,
    reynolds_basis=REYNOLDS_BASIS,
    reference_diameter=REFERENCE_DIAMETER,
    area_basis=AREA_BASIS,
    row_basis=ROW_BASIS,
)


class FinnedTubeUnsupportedLayoutError(ValueError):
    """Raised when a finned-tube correlation is asked to evaluate a
    tube-bank layout it does not cover (never silently extrapolated)."""


def nusselt_briggs_young(
    Re: float,
    Pr: float,
    *,
    clear_spacing_root: float,
    fin_height: float,
    fin_thickness_root: float,
) -> float:
    """Briggs & Young (1963) Nusselt correlation (see module docstring)."""
    if Re <= 0.0 or Pr <= 0.0:
        raise ValueError("Re and Pr must be positive.")
    if clear_spacing_root <= 0.0 or fin_height <= 0.0 or fin_thickness_root <= 0.0:
        raise ValueError("clear_spacing_root, fin_height, fin_thickness_root must be positive.")

    return (
        0.134
        * Re**0.681
        * Pr ** (1.0 / 3.0)
        * (clear_spacing_root / fin_height) ** 0.2
        * (clear_spacing_root / fin_thickness_root) ** 0.1134
    )


def check_finned_outside_ht_applicability(
    tube: CircularFinnedTube,
    Re: float,
    S_T: float,
    layout: str,
) -> list[ModelWarning]:
    warnings: list[ModelWarning] = []

    if layout != "staggered":
        # Callers must already have raised FinnedTubeUnsupportedLayoutError
        # before reaching here; this branch only guards direct callers of
        # the applicability check.
        warnings.append(
            make_warning(
                code="outside_ht_finned_layout_unsupported",
                message="outside_ht_finned: Briggs-Young (1963) covers staggered (triangular pitch) layouts only.",
                source="outside_ht_finned",
                severity="critical",
            )
        )

    if not (RE_MIN <= Re <= RE_MAX):
        warnings.append(
            make_warning(
                code="outside_ht_finned_re_out_of_range",
                message=(
                    f"outside_ht_finned: Re={Re:.4g} is outside the reported "
                    f"Briggs-Young applicability range [{RE_MIN:.0f}, {RE_MAX:.0f}]."
                ),
                source="outside_ht_finned",
                severity="warning",
            )
        )
    if not (D_ROOT_MIN <= tube.D_root <= D_ROOT_MAX):
        warnings.append(
            make_warning(
                code="outside_ht_finned_d_root_out_of_range",
                message="outside_ht_finned: D_root is outside the reported Briggs-Young geometry range.",
                source="outside_ht_finned",
                severity="warning",
            )
        )
    if not (FIN_HEIGHT_MIN <= tube.fin_height <= FIN_HEIGHT_MAX):
        warnings.append(
            make_warning(
                code="outside_ht_finned_fin_height_out_of_range",
                message="outside_ht_finned: fin_height is outside the reported Briggs-Young geometry range.",
                source="outside_ht_finned",
                severity="warning",
            )
        )
    if not (FIN_THICKNESS_MIN <= tube.fin_thickness_root <= FIN_THICKNESS_MAX):
        warnings.append(
            make_warning(
                code="outside_ht_finned_fin_thickness_out_of_range",
                message="outside_ht_finned: fin_thickness_root is outside the reported Briggs-Young geometry range.",
                source="outside_ht_finned",
                severity="warning",
            )
        )
    if not (FIN_PITCH_MIN <= tube.fin_pitch <= FIN_PITCH_MAX):
        warnings.append(
            make_warning(
                code="outside_ht_finned_fin_pitch_out_of_range",
                message="outside_ht_finned: fin_pitch is outside the reported Briggs-Young geometry range.",
                source="outside_ht_finned",
                severity="warning",
            )
        )
    if not (ST_MIN <= S_T <= ST_MAX):
        warnings.append(
            make_warning(
                code="outside_ht_finned_st_out_of_range",
                message="outside_ht_finned: S_T (transverse pitch) is outside the reported Briggs-Young geometry range.",
                source="outside_ht_finned",
                severity="warning",
            )
        )
    if tube.fin_thickness_tip_used != tube.fin_thickness_root:
        warnings.append(
            make_warning(
                code="outside_ht_finned_tapered_fin_not_in_source_dataset",
                message=(
                    "outside_ht_finned: Briggs-Young (1963) was correlated "
                    "against constant-thickness fins; this tube has a "
                    "root-to-tip taper. fin_thickness_root is used as the "
                    "representative thickness (documented approximation)."
                ),
                source="outside_ht_finned",
                severity="info",
            )
        )

    return warnings


@dataclass(frozen=True)
class FinnedOutsideFlowResult:
    v: float
    V_max: float
    Re: float
    Pr: float
    alfa_o_physical: float
    dp_o: float
    dp_available: bool
    warnings: tuple[ModelWarning, ...]
    euler_result: EulerResult | None
    contract: FinnedCorrelationContract


def finned_outside_flow_from_mass_flow(
    m_dot: float,
    frontal_area: float,
    tube: CircularFinnedTube,
    tube_pitch_transverse: float,
    tube_pitch_longitudinal: float,
    layout: str,
    n_rows: int,
    n_tubes_per_row: int,
    props: FluidProps,
    *,
    euler_provider: str | EulerProvider = "robinson_briggs",
    calculate_pressure_drop: bool = True,
) -> FinnedOutsideFlowResult:
    """Dry, single-phase outside forced convection for a circular
    finned-tube bank (0D, single-state; see module docstring).

    Returns the PHYSICAL outside heat-transfer coefficient
    (``alfa_o_physical``, referenced to the true finned surface, not yet
    fin-efficiency-weighted) -- combine with
    ``core.heat_transfer.finned_tube_resistance.build_finned_tube_resistance_network``
    to get UA.
    """
    if m_dot <= 0.0:
        raise ValueError("m_dot must be positive.")
    if frontal_area <= 0.0:
        raise ValueError("frontal_area must be positive.")
    if n_rows <= 0 or n_tubes_per_row <= 0:
        raise ValueError("n_rows and n_tubes_per_row must be positive.")
    if layout not in ("inline", "staggered"):
        raise ValueError("layout must be 'inline' or 'staggered'.")
    if layout != "staggered":
        raise FinnedTubeUnsupportedLayoutError(
            "Briggs-Young (1963)/Robinson-Briggs (1966) finned-tube "
            "correlations in this implementation cover staggered "
            "(triangular pitch) tube banks only; inline finned-tube banks "
            f"are geometrically valid but unsupported by these providers "
            f"(got layout={layout!r})."
        )

    m_dot_tube = m_dot / float(n_tubes_per_row)
    frontal_area_per_tube = frontal_area / float(n_tubes_per_row)
    v = m_dot_tube / (props.rho * frontal_area_per_tube)

    ratio = finned_vmax_ratio_min_freeflow(tube, tube_pitch_transverse, tube_pitch_longitudinal, layout)
    V_max = v * ratio

    Re = reynolds_number(props.rho, V_max, tube.D_root, props.mu)
    Pr = prandtl_number(props.cp, props.mu, props.k)

    Nu = nusselt_briggs_young(
        Re,
        Pr,
        clear_spacing_root=tube.clear_spacing_root,
        fin_height=tube.fin_height,
        fin_thickness_root=tube.fin_thickness_root,
    )
    alfa_o_physical = Nu * props.k / tube.D_root

    warnings_list = check_finned_outside_ht_applicability(tube, Re, tube_pitch_transverse, layout)

    if calculate_pressure_drop:
        ST_over_D = tube_pitch_transverse / tube.D_root
        SL_over_D = tube_pitch_longitudinal / tube.D_root
        request = EulerRequest(
            Re=Re,
            ST_over_D=ST_over_D,
            SL_over_D=SL_over_D,
            layout=layout,
            n_rows=n_rows,
            is_finned=True,
            geometry_meta={
                "m_dot_total": m_dot,
                "rho": props.rho,
                "mu": props.mu,
                "v_ref": V_max,
                "fin_height": tube.fin_height,
                "fin_thickness_root": tube.fin_thickness_root,
                "fin_pitch": tube.fin_pitch,
                "clear_spacing_root": tube.clear_spacing_root,
                "D_root": tube.D_root,
                "D_fin": tube.D_fin,
            },
        )
        warnings_list.extend(
            check_outside_dp_applicability(request, euler_provider=euler_provider, use_vmax_for_dp=True)
        )
        try:
            euler_result = evaluate_euler(request, euler_provider=euler_provider)
            dp_o = pressure_drop_from_euler(props.rho, V_max, euler_result.Eu)
            dp_available = True
        except NotImplementedError as exc:
            euler_result = None
            dp_o = float("nan")
            dp_available = False
            warnings_list.append(
                make_warning(
                    code="outside_dp_finned_unavailable",
                    message=f"outside_dp_finned: pressure drop is not available ({exc}).",
                    source="outside_dp_finned",
                    severity="critical",
                )
            )
    else:
        euler_result = None
        dp_o = float("nan")
        dp_available = False

    return FinnedOutsideFlowResult(
        v=v,
        V_max=V_max,
        Re=Re,
        Pr=Pr,
        alfa_o_physical=alfa_o_physical,
        dp_o=dp_o,
        dp_available=dp_available,
        warnings=tuple(warnings_list),
        euler_result=euler_result,
        contract=HTC_CONTRACT,
    )
