# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
#
# -------------------------------------------------------------------------
# OUTSIDE FLOW – CROSSFLOW OVER TUBE BANK (0D CORE MODEL)
# -------------------------------------------------------------------------
#
# This module implements engineering correlations for:
#   • Heat transfer coefficient: Zukauskas-type Nu(Re,Pr) with:
#       - Re-range constants (C,m)
#       - finite-row correction C2(N_L)
#       - optional (Pr/Pr_s)^0.25 correction
#   • Geometry-consistent velocity definitions:
#       V_inf (approach) and V_max based on minimum free-flow area concept
#   • Pressure-drop integration through outside_pressure_drop dispatcher:
#       selectable euler_provider = "zukauskas" | "gaddis_gnielinski" | "esdu" | custom provider
#
# Production outside hydraulics use three provider states (inlet, midpoint,
# outlet), Simpson integration of the provider-contract quantity Eu/rho, and
# a separate signed acceleration term based on face mass flux.
#
# All correlations used here are taken from OPEN LITERATURE sources:
#
# Heat transfer (tube banks in crossflow):
#   - Zukauskas, A. (1972), "Heat Transfer from Tubes in Crossflow"
#   - Incropera et al., Fundamentals of Heat and Mass Transfer
#   - VDI Heat Atlas (Tube Banks in Crossflow)
#   - Khan, W.A. (2004), PhD Thesis, Univ. Waterloo (finite row correction)
#
# Pressure drop architecture:
#   - delegated to outside_pressure_drop.py
#
# NOTE:
#   External closed-data providers may be attached only through the
#   euler_provider interface, without importing proprietary code here.
#
# -------------------------------------------------------------------------
# UNITS (SI ONLY)
# -------------------------------------------------------------------------
# m_dot [kg/s]
# frontal_area [m^2]
# D, tube pitches [m]
# rho [kg/m^3]
# mu [Pa*s]
# k [W/(m*K)]
# cp [J/(kg*K)]
# v [m/s]
# alfa [W/(m^2*K)]
# dp [Pa]
# -------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Literal

from core.common.warnings import ModelWarning, make_warning
from core.pressure_drop.outside_pressure_drop import (
    EulerProvider,
    EulerRequest,
    EulerResult,
    check_outside_dp_applicability,
    evaluate_euler,
    pressure_drop_from_euler,
)

if TYPE_CHECKING:
    from core.properties.common import FluidTransportProperties


# -------------------------------------------------------------------------
# Fluid properties container
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class FluidProps:
    rho: float  # [kg/m^3]
    mu: float   # [Pa*s]
    k: float    # [W/(m*K)]
    cp: float   # [J/(kg*K)]


OutsideHydraulicPosition = Literal["inlet", "midpoint", "outlet"]


@dataclass(frozen=True)
class OutsideHydraulicPoint:
    """Three-state outside tube-bank hydraulic point."""

    position: OutsideHydraulicPosition
    temperature: float
    pressure: float
    props: FluidTransportProperties
    enthalpy: float | None
    face_mass_flux: float
    face_velocity: float
    reference_area: float
    reference_mass_flux: float
    reference_velocity: float
    reynolds: float
    euler_number: float
    dynamic_pressure_reference: float
    prandtl: float

    @property
    def Pr(self) -> float:
        return self.prandtl

    @property
    def h(self) -> float | None:
        return self.enthalpy

    @property
    def dynamic_pressure(self) -> float:
        """Compatibility alias for reference dynamic pressure."""
        return self.dynamic_pressure_reference

    @property
    def maximum_gap_velocity(self) -> float:
        """Velocity on the minimum-free-flow reference area."""
        return self.reference_velocity

    @property
    def maximum_gap_mass_flux(self) -> float:
        """Mass flux on the minimum-free-flow reference area."""
        return self.reference_mass_flux


@dataclass(frozen=True)
class OutsideTubeBankHydraulicResult:
    """Variable-property drag and signed acceleration through a tube bank."""

    inlet: OutsideHydraulicPoint
    midpoint: OutsideHydraulicPoint
    outlet: OutsideHydraulicPoint
    midpoint_method: str
    euler_basis: str
    reference_velocity_definition: str
    face_area: float
    face_mass_flux: float
    reference_area: float
    reference_mass_flux: float
    reference_diameter: float
    n_rows_effective: float
    mean_euler_over_rho: float
    dp_drag: float
    dp_acceleration: float
    dp_total: float
    warnings: tuple[ModelWarning, ...]

    @property
    def dp_outside_drag(self) -> float:
        return self.dp_drag

    @property
    def dp_outside_acceleration(self) -> float:
        return self.dp_acceleration

    @property
    def dp_outside_total(self) -> float:
        return self.dp_total

    @property
    def mean_Eu_over_rho(self) -> float:
        return self.mean_euler_over_rho

    @property
    def rho_in(self) -> float:
        return self.inlet.props.rho

    @property
    def rho_mid(self) -> float:
        return self.midpoint.props.rho

    @property
    def rho_out(self) -> float:
        return self.outlet.props.rho

    @property
    def mu_in(self) -> float:
        return self.inlet.props.mu

    @property
    def mu_mid(self) -> float:
        return self.midpoint.props.mu

    @property
    def mu_out(self) -> float:
        return self.outlet.props.mu

    @property
    def reference_velocity_in(self) -> float:
        return self.inlet.reference_velocity

    @property
    def reference_velocity_mid(self) -> float:
        return self.midpoint.reference_velocity

    @property
    def reference_velocity_out(self) -> float:
        return self.outlet.reference_velocity

    @property
    def Re_in(self) -> float:
        return self.inlet.reynolds

    @property
    def Re_mid(self) -> float:
        return self.midpoint.reynolds

    @property
    def Re_out(self) -> float:
        return self.outlet.reynolds

    @property
    def Eu_in(self) -> float:
        return self.inlet.euler_number

    @property
    def Eu_mid(self) -> float:
        return self.midpoint.euler_number

    @property
    def Eu_out(self) -> float:
        return self.outlet.euler_number


# -------------------------------------------------------------------------
# Dimensionless numbers
# -------------------------------------------------------------------------

def reynolds_number(rho: float, v: float, D: float, mu: float) -> float:
    """
    Reynolds number definition (external flow over cylinder):

        Re_D = rho * v * D / mu

    Reference:
        Incropera et al., Fundamentals of Heat and Mass Transfer
    """
    if rho <= 0.0 or mu <= 0.0 or D <= 0.0 or v <= 0.0:
        raise ValueError("rho, mu, D, v must be positive.")
    return rho * v * D / mu


def prandtl_number(cp: float, mu: float, k: float) -> float:
    """
    Prandtl number:

        Pr = cp * mu / k

    Reference:
        Standard thermophysical definition (all heat transfer textbooks)
    """
    if cp <= 0.0 or mu <= 0.0 or k <= 0.0:
        raise ValueError("cp, mu, k must be positive.")
    return cp * mu / k


# -------------------------------------------------------------------------
# Geometry-consistent velocities (minimum free-flow area concept)
# -------------------------------------------------------------------------

def vmax_ratio_min_freeflow(
    tube_outer_diameter: float,
    S_T: float,
    S_L: float,
    layout: str,
) -> float:
    """
    Returns ratio (V_max / V_inf) based on minimum free-flow area concept.

    Inline:
        V_max / V_inf = S_T / (S_T - D)

    Staggered:
        Consider two candidate "gaps" per common tube-bank treatments:
          A_T = (S_T - D)             transverse gap
          A_D = 2*(S_D - D)           diagonal gap, where S_D = sqrt(S_L^2 + (S_T/2)^2)
        Use the minimum gap to represent minimum free-flow area (per unit depth),
        leading to:
          V_max / V_inf = S_T / min(A_T, A_D)

    NOTE:
      This is a 0D geometric velocity model. For higher-fidelity layouts,
      additional corrections may apply.
    """
    if tube_outer_diameter <= 0.0 or S_T <= 0.0 or S_L <= 0.0:
        raise ValueError("tube_outer_diameter, S_T, S_L must be positive.")
    if S_T <= tube_outer_diameter:
        raise ValueError("S_T must be > D to have a valid transverse flow gap.")

    A_T = S_T - tube_outer_diameter

    if layout == "inline":
        A_min = A_T
    elif layout == "staggered":
        S_D = math.sqrt(S_L * S_L + (S_T * 0.5) ** 2)
        if S_D <= tube_outer_diameter:
            raise ValueError("Invalid geometry: diagonal pitch S_D must be > D.")
        A_D = 2.0 * (S_D - tube_outer_diameter)
        A_min = min(A_T, A_D)
    else:
        raise ValueError("layout must be 'inline' or 'staggered'.")

    return S_T / A_min


def calculate_outside_tube_bank_hydraulics(
    *,
    m_dot: float,
    face_area: float,
    tube_outer_diameter: float,
    tube_pitch_transverse: float,
    tube_pitch_longitudinal: float,
    layout: str,
    n_rows: int,
    n_tubes_per_row: int,
    provider: Any | None = None,
    temperature_in: float | None = None,
    temperature_out: float | None = None,
    pressure: float | None = None,
    euler_provider: str | EulerProvider = "zukauskas",
    inlet_props: Any | None = None,
    midpoint_props: Any | None = None,
    outlet_props: Any | None = None,
    use_vmax_for_dp: bool = True,
    pressure_drop_geometry_meta: dict | None = None,
    m_dot_inlet: float | None = None,
    m_dot_midpoint: float | None = None,
    m_dot_outlet: float | None = None,
) -> OutsideTubeBankHydraulicResult:
    """Calculate three-state variable-property crossflow tube-bank hydraulics.

    Existing Euler providers return a complete-bank coefficient by default,
    so the row count is already represented in ``Eu`` and is not multiplied
    again.  A provider explicitly marked ``per_row`` is multiplied by its
    reported effective row count exactly once.  In both cases the provider's
    Reynolds number and pressure conversion use the same maximum-gap
    reference mass flux by default.

    The drag reference is the existing minimum-free-flow velocity basis
    (``V_max`` by default).  The acceleration term uses the face mass flux,
    because inlet and outlet are the same physical face reference sections in
    this 0D model.  Pressure is held at one representative value for all
    three property evaluations; no pressure-property iteration is performed.

    ``m_dot_inlet``/``m_dot_midpoint``/``m_dot_outlet`` (v0.6.0, all default
    ``None``): optional per-point *gas-phase* mass flow override [kg/s], for
    the case where the remaining gas-phase stream itself loses mass along
    the flow path (partial condensate removal --
    ``core.phase_change.outside_condensation_solver``); the liquid
    condensate is never added to this gas-phase velocity/Reynolds basis.
    When none are given (the default, and every pre-v0.6.0 call site), the
    single ``m_dot`` is used everywhere exactly as before -- this is a
    strictly additive extension, not a behavior change, for the
    non-condensing default path (``dp_drag``, ``dp_acceleration``, and every
    other field are bit-for-bit identical to prior versions in that case).
    """
    _validate_outside_hydraulic_inputs(
        m_dot=m_dot,
        face_area=face_area,
        tube_outer_diameter=tube_outer_diameter,
        tube_pitch_transverse=tube_pitch_transverse,
        tube_pitch_longitudinal=tube_pitch_longitudinal,
        layout=layout,
        n_rows=n_rows,
        n_tubes_per_row=n_tubes_per_row,
    )

    ratio = vmax_ratio_min_freeflow(
        tube_outer_diameter,
        tube_pitch_transverse,
        tube_pitch_longitudinal,
        layout,
    )
    reference_area = face_area / ratio if use_vmax_for_dp else face_area
    reference_velocity_definition = (
        "maximum_gap_velocity" if use_vmax_for_dp else "face_velocity"
    )
    face_mass_flux = m_dot / face_area
    reference_mass_flux = m_dot / reference_area

    # Per-point *face* mass flux for the remaining gas phase (v0.6.0).
    # Reynolds/Euler-provider dispatch and the drag integral below
    # deliberately keep using the single shared reference_mass_flux (see the
    # docstring: this stays a bulk/reference-property calculation, matching
    # the existing three-state contract); only the reported per-point face
    # velocity/flux and the signed acceleration term (below) use the
    # per-point gas-phase mass flow when it differs from point to point.
    face_mass_flux_points = {
        "inlet": m_dot_inlet / face_area if m_dot_inlet is not None else face_mass_flux,
        "midpoint": m_dot_midpoint / face_area if m_dot_midpoint is not None else face_mass_flux,
        "outlet": m_dot_outlet / face_area if m_dot_outlet is not None else face_mass_flux,
    }

    warnings: list[ModelWarning] = []
    if provider is None:
        if inlet_props is None:
            raise ValueError(
                "provider or inlet_props must be supplied for outside tube-bank hydraulics."
            )
        props_in, h_in = _outside_transport_and_enthalpy(inlet_props)
        props_mid, h_mid = _outside_transport_and_enthalpy(midpoint_props or inlet_props)
        props_out, h_out = _outside_transport_and_enthalpy(outlet_props or inlet_props)
        T_in = 300.0 if temperature_in is None else float(temperature_in)
        T_out = T_in if temperature_out is None else float(temperature_out)
        T_mid = 0.5 * (T_in + T_out)
        p = 101325.0 if pressure is None else float(pressure)
        midpoint_method = "arithmetic_temperature"
    else:
        T_in = _outside_require_temperature(temperature_in, "temperature_in")
        T_out = _outside_require_temperature(temperature_out, "temperature_out")
        p = _outside_require_pressure(pressure)
        props_in, h_in = _outside_evaluate_provider_state(provider, T_in, p)
        props_out, h_out = _outside_evaluate_provider_state(provider, T_out, p)
        T_mid = 0.5 * (T_in + T_out)
        midpoint_method = "arithmetic_temperature"

        if _outside_finite_enthalpy(h_in) and _outside_finite_enthalpy(h_out):
            h_mid_target = (float(h_in) + float(h_out)) / 2.0
            T_mid_enthalpy = _outside_try_temperature_from_enthalpy(
                provider, h=h_mid_target, p=p
            )
            if T_mid_enthalpy is not None:
                T_mid = T_mid_enthalpy
                midpoint_method = "enthalpy"
            else:
                warnings.append(_outside_midpoint_fallback_warning("inversion is unavailable or failed"))
        else:
            warnings.append(_outside_midpoint_fallback_warning("complete inlet/outlet enthalpy data are unavailable"))

        if midpoint_props is not None:
            props_mid, h_mid = _outside_transport_and_enthalpy(midpoint_props)
        else:
            props_mid, h_mid = _outside_evaluate_provider_state(provider, T_mid, p)

    points: list[OutsideHydraulicPoint] = []
    euler_basis: str | None = None
    n_rows_effective: float | None = None
    for position, temperature, props, enthalpy in (
        ("inlet", T_in, props_in, h_in),
        ("midpoint", T_mid, props_mid, h_mid),
        ("outlet", T_out, props_out, h_out),
    ):
        reynolds = reference_mass_flux * tube_outer_diameter / props.mu
        if not math.isfinite(reynolds) or reynolds <= 0.0:
            raise ValueError(
                "outside_bank_hydraulics_invalid_reynolds: Reynolds number must be finite and positive."
            )

        geometry_meta = dict(pressure_drop_geometry_meta or {})
        # Outside v0.5.5 hydraulics are bulk-property only; wall-property
        # corrections belong to the thermal path and are not injected into
        # the pressure-drop provider.
        geometry_meta.pop("mu_wall", None)
        geometry_meta.pop("mu_surface", None)
        # These reference quantities are owned by this integrator.  Do not
        # let optional provider metadata silently mix a different velocity,
        # area, density, or viscosity basis into Re or pressure conversion.
        geometry_meta.update(
            {
                "m_dot_total": m_dot,
                "rho": props.rho,
                "mu": props.mu,
                "v_ref": reference_mass_flux / props.rho,
                "face_area": face_area,
                "reference_area": reference_area,
                "face_mass_flux": face_mass_flux,
                "reference_mass_flux": reference_mass_flux,
                "tube_outer_diameter": tube_outer_diameter,
                "tube_pitch_transverse": tube_pitch_transverse,
                "tube_pitch_longitudinal": tube_pitch_longitudinal,
                "mu_bulk": props.mu,
            }
        )

        request = EulerRequest(
            Re=reynolds,
            ST_over_D=tube_pitch_transverse / tube_outer_diameter,
            SL_over_D=tube_pitch_longitudinal / tube_outer_diameter,
            layout=layout,
            n_rows=n_rows,
            geometry_meta=geometry_meta,
        )
        euler_result = evaluate_euler(request, euler_provider=euler_provider)
        basis = getattr(euler_result, "euler_basis", None)
        if basis is None:
            basis = "complete_bank"
            warnings.append(
                make_warning(
                    code="outside_bank_hydraulics_ambiguous_euler_basis",
                    message=(
                        "outside_bank_hydraulics: Euler provider did not declare "
                        "whether Eu is per-row or complete-bank; assuming complete-bank."
                    ),
                    source="outside_bank_hydraulics",
                    severity="warning",
                )
            )
        if basis not in ("per_row", "complete_bank"):
            raise ValueError(
                "outside_bank_hydraulics_ambiguous_euler_basis: unsupported Euler basis."
            )
        if euler_basis is None:
            euler_basis = basis
        elif euler_basis != basis:
            raise ValueError(
                "outside_bank_hydraulics_ambiguous_euler_basis: Euler basis changed between states."
            )

        effective_rows_raw = getattr(euler_result, "n_rows_effective", None)
        effective_rows = (
            float(n_rows)
            if effective_rows_raw is None
            else float(effective_rows_raw)
        )
        if not math.isfinite(effective_rows) or effective_rows <= 0.0:
            raise ValueError(
                "outside_bank_hydraulics_invalid_row_count: effective row count must be finite and positive."
            )
        if n_rows_effective is None:
            n_rows_effective = effective_rows
        elif not math.isclose(n_rows_effective, effective_rows, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "outside_bank_hydraulics_invalid_row_count: effective row count changed between states."
            )

        eu = float(euler_result.Eu)
        if not math.isfinite(eu) or eu < 0.0:
            raise ValueError(
                "outside_bank_hydraulics_invalid_euler: Euler number must be finite and non-negative."
            )

        point_face_mass_flux = face_mass_flux_points[position]
        reference_velocity = reference_mass_flux / props.rho
        face_velocity = point_face_mass_flux / props.rho
        dynamic_pressure_reference = reference_mass_flux**2 / (2.0 * props.rho)
        prandtl = props.mu * props.cp / props.k
        if not all(
            math.isfinite(value)
            for value in (
                temperature,
                p,
                face_velocity,
                reference_velocity,
                dynamic_pressure_reference,
                prandtl,
            )
        ):
            raise ValueError(
                "outside_bank_hydraulics_nonfinite_state: outside hydraulic point is not finite."
            )

        points.append(
            OutsideHydraulicPoint(
                position=position,
                temperature=float(temperature),
                pressure=float(p),
                props=props,
                enthalpy=enthalpy,
                face_mass_flux=point_face_mass_flux,
                face_velocity=face_velocity,
                reference_area=reference_area,
                reference_mass_flux=reference_mass_flux,
                reference_velocity=reference_velocity,
                reynolds=reynolds,
                euler_number=eu,
                dynamic_pressure_reference=dynamic_pressure_reference,
                prandtl=prandtl,
            )
        )
        warnings.extend(
            check_outside_dp_applicability(
                request,
                euler_provider=euler_provider,
                use_vmax_for_dp=use_vmax_for_dp,
            )
        )

    if euler_basis is None or n_rows_effective is None:
        raise ValueError("outside_bank_hydraulics_nonfinite_state: no hydraulic states were evaluated.")

    mean_euler_over_rho = (
        points[0].euler_number / points[0].props.rho
        + 4.0 * points[1].euler_number / points[1].props.rho
        + points[2].euler_number / points[2].props.rho
    ) / 6.0
    row_multiplier = n_rows_effective if euler_basis == "per_row" else 1.0
    # Three-state 0D+ approximation of the local tube-bank drag integral;
    # this integrates the Euler-correlation quantity, not Darcy f/rho.
    dp_drag = row_multiplier * (reference_mass_flux**2 / 2.0) * mean_euler_over_rho

    # The inlet and outlet are the same face-flow reference sections in this
    # 0D bank model, so acceleration uses face mass flux rather than V_max mass
    # flux.  Refs: White, Fluid Mechanics; Idelchik; Crane TP-410.
    #
    # v0.6.0: generalized to independent inlet/outlet face mass flux, for the
    # remaining-gas-phase momentum change when the gas stream itself loses
    # mass along the path (condensate removal -- the condensate is never
    # part of this gas-phase flux). When m_dot_inlet/m_dot_outlet are not
    # given (every pre-v0.6.0 call site), points[2]/points[0] face_mass_flux
    # are both the same shared face_mass_flux, so this reduces to exactly
    # the prior expression bit-for-bit (same single value squared, factored
    # the same way as before).
    dp_acceleration = (
        points[2].face_mass_flux**2 / points[2].props.rho
        - points[0].face_mass_flux**2 / points[0].props.rho
    ) if (m_dot_inlet is not None or m_dot_outlet is not None) else face_mass_flux**2 * (
        1.0 / points[2].props.rho - 1.0 / points[0].props.rho
    )
    dp_total = dp_drag + dp_acceleration

    if not math.isfinite(dp_drag) or dp_drag < 0.0:
        raise ValueError(
            "outside_bank_hydraulics_invalid_drag: bank drag must be finite and non-negative."
        )
    if not math.isfinite(dp_acceleration) or not math.isfinite(dp_total):
        raise ValueError(
            "outside_bank_hydraulics_nonfinite_state: outside pressure change is not finite."
        )
    if dp_total < 0.0:
        warnings.append(
            make_warning(
                code="outside_bank_hydraulics_negative_total_pressure_change",
                message=(
                    "outside_bank_hydraulics: signed outside bank pressure change "
                    f"is negative ({dp_total:.6g} Pa), representing net pressure "
                    "recovery after irreversible bank drag."
                ),
                source="outside_bank_hydraulics",
                severity="warning",
            )
        )

    return OutsideTubeBankHydraulicResult(
        inlet=points[0],
        midpoint=points[1],
        outlet=points[2],
        midpoint_method=midpoint_method,
        euler_basis=euler_basis,
        reference_velocity_definition=reference_velocity_definition,
        face_area=float(face_area),
        face_mass_flux=face_mass_flux,
        reference_area=float(reference_area),
        reference_mass_flux=reference_mass_flux,
        reference_diameter=float(tube_outer_diameter),
        n_rows_effective=n_rows_effective,
        mean_euler_over_rho=mean_euler_over_rho,
        dp_drag=dp_drag,
        dp_acceleration=dp_acceleration,
        dp_total=dp_total,
        warnings=_deduplicate_outside_hydraulic_warnings(warnings),
    )


outside_tube_bank_hydraulics = calculate_outside_tube_bank_hydraulics


def _validate_outside_hydraulic_inputs(
    *,
    m_dot: float,
    face_area: float,
    tube_outer_diameter: float,
    tube_pitch_transverse: float,
    tube_pitch_longitudinal: float,
    layout: str,
    n_rows: int,
    n_tubes_per_row: int,
) -> None:
    checks = (
        ("mass flow", m_dot, "outside_bank_hydraulics_invalid_mass_flux", True),
        ("face area", face_area, "outside_bank_hydraulics_invalid_face_area", True),
        ("reference diameter", tube_outer_diameter, "outside_bank_hydraulics_invalid_reference_diameter", True),
        ("transverse pitch", tube_pitch_transverse, "outside_bank_hydraulics_invalid_geometry", True),
        ("longitudinal pitch", tube_pitch_longitudinal, "outside_bank_hydraulics_invalid_geometry", True),
    )
    for label, value, code, strictly_positive in checks:
        valid = math.isfinite(value) and (value > 0.0 if strictly_positive else value >= 0.0)
        if not valid:
            raise ValueError(f"{code}: {label} must be finite and positive.")
    if n_rows <= 0 or n_tubes_per_row <= 0:
        raise ValueError(
            "outside_bank_hydraulics_invalid_row_count: row and tube counts must be positive."
        )
    if layout not in ("inline", "staggered"):
        raise ValueError("outside_bank_hydraulics_invalid_geometry: layout is invalid.")


def _outside_require_temperature(value: float | None, name: str) -> float:
    if value is None or not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"outside_bank_hydraulics_nonfinite_state: {name} must be finite and positive."
        )
    return float(value)


def _outside_require_pressure(value: float | None) -> float:
    if value is None or not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "outside_bank_hydraulics_nonfinite_state: pressure must be finite and positive."
        )
    return float(value)


def _outside_transport_and_enthalpy(raw: Any) -> tuple[FluidTransportProperties, float | None]:
    from core.properties.common import FluidTransportProperties

    transport = getattr(raw, "transport", raw)
    if isinstance(transport, FluidTransportProperties):
        props = transport
    else:
        props = FluidTransportProperties(
            rho=float(getattr(transport, "rho")),
            mu=float(getattr(transport, "mu")),
            k=float(getattr(transport, "k")),
            cp=float(getattr(transport, "cp")),
        )
    enthalpy = getattr(raw, "h", getattr(raw, "enthalpy", None))
    return props, _outside_finite_float_or_none(enthalpy)


def _outside_evaluate_provider_state(
    provider: Any,
    temperature: float,
    pressure: float,
) -> tuple[FluidTransportProperties, float | None]:
    full_at = getattr(provider, "full_at", None)
    raw = (
        full_at(T=temperature, p=pressure)
        if callable(full_at)
        else provider.at(T=temperature, p=pressure)
    )
    return _outside_transport_and_enthalpy(raw)


def _outside_try_temperature_from_enthalpy(
    provider: Any,
    *,
    h: float,
    p: float,
) -> float | None:
    for method_name in ("temperature_from_h_p", "T_from_h_p"):
        method = getattr(provider, method_name, None)
        if not callable(method):
            continue
        try:
            raw = method(h=h, p=p)
        except Exception:
            continue
        if isinstance(raw, (int, float)):
            value = float(raw)
        else:
            value = _outside_finite_float_or_none(getattr(raw, "T", None))
        return value if value is not None and value > 0.0 else None
    return None


def _outside_finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _outside_finite_enthalpy(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _outside_midpoint_fallback_warning(reason: str) -> ModelWarning:
    return make_warning(
        code="outside_bank_hydraulics_midpoint_temperature_fallback",
        message=(
            "outside_bank_hydraulics: " + reason + "; using arithmetic "
            "temperature midpoint."
        ),
        source="outside_bank_hydraulics",
        severity="info",
    )


def _deduplicate_outside_hydraulic_warnings(
    warnings: list[ModelWarning],
) -> tuple[ModelWarning, ...]:
    unique: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        identity = (warning.source, warning.code)
        if identity not in seen:
            seen.add(identity)
            unique.append(warning)
    return tuple(unique)


# -------------------------------------------------------------------------
# Finite row correction (Zukauskas method)
# -------------------------------------------------------------------------

def _interp_1d(x: float, xs: list[float], ys: list[float]) -> float:
    """Clamped linear interpolation."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("xs and ys must have same length >= 2.")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def finite_row_correction_c2(n_rows: int) -> float:
    """
    Finite row correction factor C2(N_L).

    Tabulated anchor values based on:
        Khan (2004), PhD Thesis, Univ. Waterloo
        Zukauskas method summaries in literature

    For N_L >= 20 → C2 ≈ 1.0
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    xs = [1, 2, 3, 4, 5, 7, 10, 13, 16, 20]
    ys = [0.64, 0.76, 0.84, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00]

    return _interp_1d(float(n_rows), [float(v) for v in xs], ys)


# -------------------------------------------------------------------------
# Zukauskas heat transfer correlation
# -------------------------------------------------------------------------

def nusselt_zukauskas(
    Re: float,
    Pr: float,
    n_rows: int,
    *,
    Pr_s: float | None = None,
    apply_finite_row_correction: bool = True,
) -> float:
    """
    Zukauskas-type correlation for tube banks in crossflow:

        Nu = C * Re^m * Pr^0.36

    with:
        - Re-range dependent C, m
        - optional finite-row correction C2(N_L)
        - optional (Pr/Pr_s)^0.25 correction

    References:
        Zukauskas (1972)
        Incropera et al.
        VDI Heat Atlas
    """
    if Re <= 0.0 or Pr <= 0.0:
        raise ValueError("Re and Pr must be positive.")
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    if Re < 1e2:
        C, m = 0.90, 0.40
    elif Re < 1e3:
        C, m = 0.52, 0.50
    elif Re < 2e5:
        C, m = 0.27, 0.63
    else:
        C, m = 0.021, 0.84

    Nu = C * (Re ** m) * (Pr ** 0.36)

    if apply_finite_row_correction:
        Nu *= finite_row_correction_c2(n_rows)

    if Pr_s is not None:
        if Pr_s <= 0.0:
            raise ValueError("Pr_s must be positive.")
        Nu *= (Pr / Pr_s) ** 0.25

    return Nu


# -------------------------------------------------------------------------
# Heat-transfer applicability
# -------------------------------------------------------------------------

def check_outside_ht_applicability(
    Re: float,
    Pr: float,
    tube_outer_diameter: float,
    tube_pitch_transverse: float,
    tube_pitch_longitudinal: float,
    layout: str,
    n_rows: int,
    *,
    use_vmax_for_ht: bool = True,
) -> list[ModelWarning]:
    """
    Applicability / diagnostic checks for the outside heat-transfer model.
    """
    warnings: list[ModelWarning] = []

    if Re <= 0.0:
        warnings.append(
            make_warning(
                code="outside_ht_re_nonpositive",
                message="outside_ht: Reynolds number must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )
    if Pr <= 0.0:
        warnings.append(
            make_warning(
                code="outside_ht_pr_nonpositive",
                message="outside_ht: Prandtl number must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )
    if tube_outer_diameter <= 0.0:
        warnings.append(
            make_warning(
                code="outside_ht_diameter_nonpositive",
                message="outside_ht: tube_outer_diameter must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )
    if tube_pitch_transverse <= 0.0:
        warnings.append(
            make_warning(
                code="outside_ht_pitch_transverse_nonpositive",
                message="outside_ht: tube_pitch_transverse must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )
    if tube_pitch_longitudinal <= 0.0:
        warnings.append(
            make_warning(
                code="outside_ht_pitch_longitudinal_nonpositive",
                message="outside_ht: tube_pitch_longitudinal must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )
    if n_rows <= 0:
        warnings.append(
            make_warning(
                code="outside_ht_n_rows_nonpositive",
                message="outside_ht: n_rows must be positive.",
                source="outside_ht",
                severity="critical",
            )
        )

    if (
        tube_outer_diameter <= 0.0
        or tube_pitch_transverse <= 0.0
        or tube_pitch_longitudinal <= 0.0
    ):
        return warnings

    ST_over_D = tube_pitch_transverse / tube_outer_diameter
    SL_over_D = tube_pitch_longitudinal / tube_outer_diameter

    if ST_over_D <= 1.0:
        warnings.append(
            make_warning(
                code="outside_ht_st_over_d_invalid",
                message="outside_ht: ST/D <= 1.0 is geometrically invalid for crossflow tube banks.",
                source="outside_ht",
                severity="critical",
            )
        )
    elif ST_over_D < 1.1:
        warnings.append(
            make_warning(
                code="outside_ht_st_over_d_near_blockage",
                message="outside_ht: ST/D is very close to blockage limit; results may be highly sensitive.",
                source="outside_ht",
                severity="warning",
            )
        )

    if SL_over_D <= 1.0:
        warnings.append(
            make_warning(
                code="outside_ht_sl_over_d_invalid",
                message="outside_ht: SL/D <= 1.0 is outside the intended geometry range.",
                source="outside_ht",
                severity="critical",
            )
        )
    elif SL_over_D < 1.1:
        warnings.append(
            make_warning(
                code="outside_ht_sl_over_d_small",
                message="outside_ht: SL/D is very small; wake interaction may be strong and correlation confidence is reduced.",
                source="outside_ht",
                severity="warning",
            )
        )

    if ST_over_D > 4.0:
        warnings.append(
            make_warning(
                code="outside_ht_st_over_d_large",
                message="outside_ht: ST/D is unusually large for the current 0D tube-bank model; verify applicability.",
                source="outside_ht",
                severity="warning",
            )
        )

    if SL_over_D > 4.0:
        warnings.append(
            make_warning(
                code="outside_ht_sl_over_d_large",
                message="outside_ht: SL/D is unusually large for the current 0D tube-bank model; verify applicability.",
                source="outside_ht",
                severity="warning",
            )
        )

    if Re > 0.0:
        if Re < 30.0:
            warnings.append(
                make_warning(
                    code="outside_ht_re_extremely_low",
                    message="outside_ht: Re is extremely low for tube-bank crossflow; current outside HT correlation is low-confidence.",
                    source="outside_ht",
                    severity="critical",
                )
            )
        elif Re < 1.0e2:
            warnings.append(
                make_warning(
                    code="outside_ht_re_low",
                    message="outside_ht: Re is in a very low range; confirm that the selected outside HT correlation is appropriate.",
                    source="outside_ht",
                    severity="warning",
                )
            )
        elif Re > 2.0e5:
            warnings.append(
                make_warning(
                    code="outside_ht_re_high",
                    message="outside_ht: Re exceeds the main mid-range branch of the current Zukauskas-style implementation; verify high-Re applicability.",
                    source="outside_ht",
                    severity="warning",
                )
            )

    if Pr > 0.0:
        if Pr < 0.6:
            warnings.append(
                make_warning(
                    code="outside_ht_pr_low",
                    message="outside_ht: very low Pr may be outside the intended use of the current gas-side correlation.",
                    source="outside_ht",
                    severity="warning",
                )
            )
        elif Pr > 500.0:
            warnings.append(
                make_warning(
                    code="outside_ht_pr_high",
                    message="outside_ht: very high Pr is outside the usual gas-side engineering range; verify applicability.",
                    source="outside_ht",
                    severity="warning",
                )
            )

    if n_rows == 1:
        warnings.append(
            make_warning(
                code="outside_ht_single_row",
                message="outside_ht: single-row bank; finite-row effects dominate and uncertainty is elevated.",
                source="outside_ht",
                severity="warning",
            )
        )
    elif n_rows < 5:
        warnings.append(
            make_warning(
                code="outside_ht_few_rows",
                message="outside_ht: very small number of rows; finite-row correction has strong influence on the result.",
                source="outside_ht",
                severity="warning",
            )
        )

    if not use_vmax_for_ht:
        warnings.append(
            make_warning(
                code="outside_ht_velocity_reference_nonstandard",
                message="outside_ht: HT is not using V_max as reference velocity; literature tube-bank correlations are commonly referenced to maximum gap velocity.",
                source="outside_ht",
                severity="info",
            )
        )

    if layout == "staggered" and SL_over_D < 1.2:
        warnings.append(
            make_warning(
                code="outside_ht_staggered_tight_sl",
                message="outside_ht: staggered layout with very tight longitudinal pitch may require more geometry-specific validation.",
                source="outside_ht",
                severity="warning",
            )
        )

    return warnings


# -------------------------------------------------------------------------
# Main 0D solver
# -------------------------------------------------------------------------

def outside_flow_from_mass_flow(
    m_dot: float,
    frontal_area: float,
    tube_outer_diameter: float,
    tube_pitch_transverse: float,
    tube_pitch_longitudinal: float,
    layout: str,
    n_rows: int,
    n_tubes_per_row: int,
    props: FluidProps,
    *,
    Pr_s: float | None = None,
    apply_finite_row_correction: bool = True,
    euler_provider: str | EulerProvider = "zukauskas",
    use_vmax_for_ht: bool = True,
    use_vmax_for_dp: bool = True,
    calculate_pressure_drop: bool = True,
    is_finned: bool = False,
    pressure_drop_geometry_meta: dict | None = None,
) -> tuple[float, float, float, float, float, list[ModelWarning], EulerResult | None]:
    """
    Outside forced convection for tube bank (0D core).

    Parameters
    ----------
    euler_provider:
        Either:
                    - built-in provider name: "zukauskas", "gaddis_gnielinski", "esdu"
          - custom provider object implementing EulerProvider

    Returns
    -------
    tuple
        (v, Re, Pr, alfa_o, dp_o, warnings_list, euler_result)
        - v: approach velocity [m/s]
        - Re: heat-transfer Reynolds number [-]
        - Pr: Prandtl number [-]
        - alfa_o: outside heat transfer coefficient [W/(m^2*K)]
        - dp_o: pressure drop [Pa]
        - warnings_list: structured applicability warnings
        - euler_result: metadata about selected pressure-drop backend, or
          ``None`` when ``calculate_pressure_drop=False``
    """

    if m_dot <= 0.0:
        raise ValueError("m_dot must be positive.")
    if frontal_area <= 0.0:
        raise ValueError("frontal_area must be positive.")
    if tube_outer_diameter <= 0.0:
        raise ValueError("tube_outer_diameter must be positive.")
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    if n_tubes_per_row <= 0:
        raise ValueError("n_tubes_per_row must be positive.")

    # Mass flow per tube in row
    m_dot_tube = m_dot / float(n_tubes_per_row)

    # Frontal area per tube in row
    frontal_area_per_tube = frontal_area / float(n_tubes_per_row)

    # Approach velocity (per tube)
    v = m_dot_tube / (props.rho * frontal_area_per_tube)

    # Geometry-consistent Vmax
    ratio = vmax_ratio_min_freeflow(
        tube_outer_diameter,
        tube_pitch_transverse,
        tube_pitch_longitudinal,
        layout,
    )
    V_max = v * ratio

    # Choose reference velocities
    V_ref_ht = V_max if use_vmax_for_ht else v
    V_ref_dp = V_max if use_vmax_for_dp else v

    # Heat transfer
    Re = reynolds_number(props.rho, V_ref_ht, tube_outer_diameter, props.mu)
    Pr = prandtl_number(props.cp, props.mu, props.k)

    Nu = nusselt_zukauskas(
        Re,
        Pr,
        n_rows,
        Pr_s=Pr_s,
        apply_finite_row_correction=apply_finite_row_correction,
    )
    alfa_o = Nu * props.k / tube_outer_diameter

    # Applicability warnings
    warnings_list = check_outside_ht_applicability(
        Re,
        Pr,
        tube_outer_diameter,
        tube_pitch_transverse,
        tube_pitch_longitudinal,
        layout,
        n_rows,
        use_vmax_for_ht=use_vmax_for_ht,
    )

    if calculate_pressure_drop:
        # This compatibility helper retains the historical one-state API.
        # Production BareTubeHeatExchanger pressure drop uses the universal
        # three-state outside-bank function instead.
        ST_over_D = tube_pitch_transverse / tube_outer_diameter
        SL_over_D = tube_pitch_longitudinal / tube_outer_diameter
        Re_dp = reynolds_number(props.rho, V_ref_dp, tube_outer_diameter, props.mu)

        pressure_drop_geometry_meta = dict(pressure_drop_geometry_meta or {})
        pressure_drop_geometry_meta.setdefault("m_dot_total", m_dot)
        pressure_drop_geometry_meta.setdefault("rho", props.rho)
        pressure_drop_geometry_meta.setdefault("mu", props.mu)
        pressure_drop_geometry_meta.setdefault("v_ref", V_ref_dp)
        pressure_drop_geometry_meta.setdefault("tube_outer_diameter", tube_outer_diameter)
        pressure_drop_geometry_meta.setdefault("tube_pitch_transverse", tube_pitch_transverse)
        pressure_drop_geometry_meta.setdefault("tube_pitch_longitudinal", tube_pitch_longitudinal)
        pressure_drop_geometry_meta.setdefault("mu_bulk", props.mu)

        request = EulerRequest(
            Re=Re_dp,
            ST_over_D=ST_over_D,
            SL_over_D=SL_over_D,
            layout=layout,
            n_rows=n_rows,
            is_finned=is_finned,
            geometry_meta=pressure_drop_geometry_meta,
        )

        euler_result = evaluate_euler(
            request,
            euler_provider=euler_provider,
        )
        dp_o = pressure_drop_from_euler(props.rho, V_ref_dp, euler_result.Eu)

        warnings_list.extend(
            check_outside_dp_applicability(
                request,
                euler_provider=euler_provider,
                use_vmax_for_dp=use_vmax_for_dp,
            )
        )
    else:
        euler_result = None
        dp_o = float("nan")

    return v, Re, Pr, alfa_o, dp_o, warnings_list, euler_result
