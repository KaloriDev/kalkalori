# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only

"""Pressure loss across staggered banks of circular-finned tubes.

This module has a dedicated contract because the Robinson-Briggs
coefficient is neither an unnamed Euler number nor a bare-tube coefficient.
Its definition is kept explicit throughout::

    f_RB = delta_p / (2 * n_rows * rho * V_max**2)

``V_max`` and the corresponding Reynolds number use the bundle's actual
minimum free-flow area, including periodic fin blockage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Protocol, runtime_checkable

from core.common.warnings import ApplicabilityRange, ModelWarning, make_warning
from core.geometry.bundle import TubeBundle
from core.geometry.finned_tube import CircularFinnedTube
from core.properties.common import FluidTransportProperties


FinnedTubeLayout = Literal["inline", "staggered"]
FinnedTubeHydraulicPosition = Literal["inlet", "midpoint", "outlet"]
FinnedTubePressureCoefficientBasis = Literal[
    "delta_p/(2*n_rows*rho*V_max^2)"
]

ROBINSON_BRIGGS_COEFFICIENT_BASIS: FinnedTubePressureCoefficientBasis = (
    "delta_p/(2*n_rows*rho*V_max^2)"
)


ROBINSON_BRIGGS_1966_APPLICABILITY: tuple[ApplicabilityRange, ...] = (
    ApplicabilityRange("Re_D_root", 2000.0, 50000.0),
    ApplicabilityRange("clear_fin_spacing/fin_height", 0.15, 0.19),
    ApplicabilityRange("clear_fin_spacing/mean_fin_thickness", 3.75, 6.03),
    ApplicabilityRange("fin_height/D_root", 0.35, 0.56),
    ApplicabilityRange("mean_fin_thickness/D_root", 0.011, 0.025),
    ApplicabilityRange("pitch_transverse/D_root", 1.86, 4.60),
    ApplicabilityRange("D_root", 0.0186, 0.0409, "m"),
    ApplicabilityRange("fin_density", 311.0, 431.0, "1/m"),
    ApplicabilityRange("source_test_rows", 6.0, 6.0, "rows"),
    ApplicabilityRange("secondary_recommended_rows", 4.0, None, "rows"),
)


@dataclass(frozen=True)
class FinnedTubePressureDropMetadata:
    """Machine-readable provenance and basis of a pressure-drop result."""

    method: str
    source: str
    equation: str
    coefficient_definition: str
    coefficient_basis: FinnedTubePressureCoefficientBasis
    geometry_family: str
    velocity_basis: str
    reynolds_basis: str
    reference_diameter: str
    area_basis: str
    row_basis: str
    source_fluid: str
    supported_layouts: tuple[str, ...]
    supported_fin_profiles: tuple[str, ...]
    applicability: tuple[ApplicabilityRange, ...]


ROBINSON_BRIGGS_1966_METADATA = FinnedTubePressureDropMetadata(
    method="robinson_briggs_1966",
    source=(
        "K. K. Robinson and D. E. Briggs, 'Pressure Drop of Air Flowing "
        "Across Triangular Pitch Banks of Finned Tubes', Chemical "
        "Engineering Progress Symposium Series, Vol. 62, No. 64, 1966, "
        "pp. 177-184."
    ),
    equation=(
        "f_RB=9.465*Re_Droot^-0.316*(P_t/D_root)^-0.927*"
        "(P_t/P_d)^0.515; delta_p=2*f_RB*n_rows*rho*V_max^2"
    ),
    coefficient_definition=ROBINSON_BRIGGS_COEFFICIENT_BASIS,
    coefficient_basis=ROBINSON_BRIGGS_COEFFICIENT_BASIS,
    geometry_family=(
        "periodic solid circular/annular fins on individual round tubes"
    ),
    velocity_basis=(
        "V_max=m_dot/(rho*A_min), using the global minimum free-flow area "
        "including root and periodic fin blockage"
    ),
    reynolds_basis="Re_Droot=(m_dot/A_min)*D_root/mu",
    reference_diameter="fin-root diameter D_root",
    area_basis=(
        "A_min is the physical minimum free-flow area; face-section mass "
        "flux is used only for the separate acceleration term"
    ),
    row_basis=(
        "f_RB is a per-row modified Fanning-type coefficient; multiply "
        "exactly once by 2*n_rows*rho*V_max^2. Source tests used six rows; "
        ">=4 rows is a later handbook recommendation"
    ),
    source_fluid="air",
    supported_layouts=("staggered_equilateral_triangular",),
    supported_fin_profiles=("solid_circular_constant_or_linear_taper",),
    applicability=ROBINSON_BRIGGS_1966_APPLICABILITY,
)


@dataclass(frozen=True)
class FinnedTubePressureDropRequest:
    """Complete bulk state and geometry supplied to a pressure provider."""

    m_dot: float
    face_area: float
    minimum_free_flow_area: float
    rho: float
    mu: float
    D_root: float
    D_fin: float
    fin_pitch: float
    fin_thickness_root: float
    fin_thickness_tip: float
    pitch_transverse: float
    pitch_longitudinal: float
    layout: FinnedTubeLayout
    n_rows: int
    fluid_family: str = "air"
    geometry_family: str = "solid_circular_finned_tube_bank"

    def __post_init__(self) -> None:
        for name, value in (
            ("m_dot", self.m_dot),
            ("face_area", self.face_area),
            ("minimum_free_flow_area", self.minimum_free_flow_area),
            ("rho", self.rho),
            ("mu", self.mu),
            ("D_root", self.D_root),
            ("D_fin", self.D_fin),
            ("fin_pitch", self.fin_pitch),
            ("fin_thickness_root", self.fin_thickness_root),
            ("fin_thickness_tip", self.fin_thickness_tip),
            ("pitch_transverse", self.pitch_transverse),
            ("pitch_longitudinal", self.pitch_longitudinal),
        ):
            _require_positive_finite(value, name)
        if self.minimum_free_flow_area > self.face_area:
            raise ValueError(
                "minimum_free_flow_area must not exceed face_area."
            )
        if self.D_fin <= self.D_root:
            raise ValueError("D_fin must be greater than D_root.")
        if self.fin_pitch <= self.fin_thickness_root:
            raise ValueError(
                "fin_pitch must be greater than fin_thickness_root."
            )
        if self.fin_thickness_tip > self.fin_thickness_root:
            raise ValueError(
                "fin_thickness_tip must not exceed fin_thickness_root."
            )
        if not isinstance(self.n_rows, int) or isinstance(self.n_rows, bool):
            raise TypeError("n_rows must be an integer.")
        if self.n_rows <= 0:
            raise ValueError("n_rows must be positive.")
        if self.layout not in ("inline", "staggered"):
            raise ValueError("layout must be 'inline' or 'staggered'.")
        if not self.fluid_family.strip():
            raise ValueError("fluid_family must be non-empty.")
        if not self.geometry_family.strip():
            raise ValueError("geometry_family must be non-empty.")


@dataclass(frozen=True)
class FinnedTubePressureDropResult:
    """Single-state bank drag and the unambiguous coefficient behind it."""

    coefficient: float
    coefficient_definition: str
    coefficient_basis: FinnedTubePressureCoefficientBasis
    dp_drag: float
    reynolds_number: float
    face_area: float
    minimum_free_flow_area: float
    face_mass_flux: float
    reference_mass_flux: float
    face_velocity: float
    reference_velocity: float
    reference_diameter_value: float
    n_rows_effective: int
    fin_height: float
    correlation_fin_thickness: float
    correlation_fin_spacing: float
    diagonal_pitch: float
    equilateral_relative_deviation: float
    metadata: FinnedTubePressureDropMetadata
    warnings: tuple[ModelWarning, ...]

    @property
    def f(self) -> float:
        return self.coefficient

    @property
    def Re(self) -> float:
        return self.reynolds_number

    @property
    def method(self) -> str:
        return self.metadata.method

    @property
    def source(self) -> str:
        return self.metadata.source

    @property
    def geometry_family(self) -> str:
        return self.metadata.geometry_family

    @property
    def velocity_basis(self) -> str:
        return self.metadata.velocity_basis

    @property
    def reynolds_basis(self) -> str:
        return self.metadata.reynolds_basis

    @property
    def reference_diameter(self) -> str:
        return self.metadata.reference_diameter

    @property
    def area_basis(self) -> str:
        return self.metadata.area_basis

    @property
    def row_basis(self) -> str:
        return self.metadata.row_basis

    @property
    def applicability(self) -> tuple[ApplicabilityRange, ...]:
        return self.metadata.applicability


@runtime_checkable
class FinnedTubePressureDropProvider(Protocol):
    """Provider protocol independent of the generic bare-bank Eu API."""

    def evaluate(
        self, request: FinnedTubePressureDropRequest
    ) -> FinnedTubePressureDropResult:
        ...


@dataclass(frozen=True)
class RobinsonBriggs1966Provider:
    """Robinson-Briggs pressure-loss correlation for triangular banks."""

    equilateral_relative_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.equilateral_relative_tolerance)
            or self.equilateral_relative_tolerance < 0.0
        ):
            raise ValueError(
                "equilateral_relative_tolerance must be finite and non-negative."
            )

    def evaluate(
        self, request: FinnedTubePressureDropRequest
    ) -> FinnedTubePressureDropResult:
        if request.geometry_family != "solid_circular_finned_tube_bank":
            raise ValueError(
                "RobinsonBriggs1966Provider supports only solid "
                "circular-finned tube banks."
            )
        if request.layout != "staggered":
            raise ValueError(
                "RobinsonBriggs1966Provider supports only staggered "
                "triangular tube banks; inline layout is unsupported."
            )
        if request.n_rows < 4:
            raise ValueError(
                "RobinsonBriggs1966Provider requires at least four tube "
                "rows; the source correlation was obtained from six-row banks."
            )

        diagonal_pitch = math.hypot(
            request.pitch_longitudinal, 0.5 * request.pitch_transverse
        )
        equilateral_deviation = abs(
            diagonal_pitch / request.pitch_transverse - 1.0
        )
        if equilateral_deviation > self.equilateral_relative_tolerance:
            raise ValueError(
                "RobinsonBriggs1966Provider requires an equilateral triangular "
                "bank (P_d approximately equal to P_t). The isosceles mapping "
                "is outside the verified v0.7.0 geometry scope."
            )
        t_mean = 0.5 * (
            request.fin_thickness_root + request.fin_thickness_tip
        )
        spacing = request.fin_pitch - t_mean
        height = 0.5 * (request.D_fin - request.D_root)
        if spacing <= 0.0:
            raise ValueError(
                "Mean-thickness correlation mapping leaves no positive fin spacing."
            )

        face_mass_flux = request.m_dot / request.face_area
        reference_mass_flux = request.m_dot / request.minimum_free_flow_area
        face_velocity = face_mass_flux / request.rho
        reference_velocity = reference_mass_flux / request.rho
        reynolds = reference_mass_flux * request.D_root / request.mu
        coefficient = (
            9.465
            * reynolds ** -0.316
            * (request.pitch_transverse / request.D_root) ** -0.927
            * (request.pitch_transverse / diagonal_pitch) ** 0.515
        )
        dp_drag = (
            2.0
            * request.n_rows
            * coefficient
            * reference_mass_flux**2
            / request.rho
        )

        for name, value in (
            ("Re", reynolds),
            ("f_RB", coefficient),
            ("dp_drag", dp_drag),
        ):
            _require_positive_finite(value, name)

        warnings = _robinson_briggs_warnings(
            request=request,
            reynolds=reynolds,
            spacing=spacing,
            t_mean=t_mean,
            height=height,
        )
        return FinnedTubePressureDropResult(
            coefficient=coefficient,
            coefficient_definition=ROBINSON_BRIGGS_COEFFICIENT_BASIS,
            coefficient_basis=ROBINSON_BRIGGS_COEFFICIENT_BASIS,
            dp_drag=dp_drag,
            reynolds_number=reynolds,
            face_area=request.face_area,
            minimum_free_flow_area=request.minimum_free_flow_area,
            face_mass_flux=face_mass_flux,
            reference_mass_flux=reference_mass_flux,
            face_velocity=face_velocity,
            reference_velocity=reference_velocity,
            reference_diameter_value=request.D_root,
            n_rows_effective=request.n_rows,
            fin_height=height,
            correlation_fin_thickness=t_mean,
            correlation_fin_spacing=spacing,
            diagonal_pitch=diagonal_pitch,
            equilateral_relative_deviation=equilateral_deviation,
            metadata=ROBINSON_BRIGGS_1966_METADATA,
            warnings=warnings,
        )


def evaluate_finned_tube_pressure_drop(
    request: FinnedTubePressureDropRequest,
    *,
    provider: str | FinnedTubePressureDropProvider = "robinson_briggs_1966",
) -> FinnedTubePressureDropResult:
    """Evaluate a request with a built-in name or custom provider."""

    return _resolve_pressure_drop_provider(provider).evaluate(request)


def calculate_finned_tube_pressure_drop(
    m_dot: float,
    bundle: TubeBundle,
    props: FluidTransportProperties,
    *,
    provider: str | FinnedTubePressureDropProvider = "robinson_briggs_1966",
    fluid_family: str = "air",
) -> FinnedTubePressureDropResult:
    """Calculate single-state drag directly from ``(m_dot, bundle, props)``."""

    tube = _require_circular_finned_bundle(bundle)
    transport, _ = _transport_and_enthalpy(props)
    request = _request_from_bundle(
        m_dot=m_dot,
        bundle=bundle,
        tube=tube,
        props=transport,
        fluid_family=fluid_family,
    )
    return evaluate_finned_tube_pressure_drop(request, provider=provider)


@dataclass(frozen=True)
class FinnedTubeHydraulicPoint:
    """One of the three bulk-property states used for bank hydraulics."""

    position: FinnedTubeHydraulicPosition
    temperature: float
    pressure: float
    props: FluidTransportProperties
    enthalpy: float | None
    face_mass_flux: float
    face_velocity: float
    reference_area: float
    reference_mass_flux: float
    reference_velocity: float
    reynolds_number: float
    coefficient: float
    coefficient_definition: str
    local_dp_drag: float
    metadata: FinnedTubePressureDropMetadata
    warnings: tuple[ModelWarning, ...]

    @property
    def T(self) -> float:
        return self.temperature

    @property
    def p(self) -> float:
        return self.pressure

    @property
    def rho(self) -> float:
        return self.props.rho

    @property
    def mu(self) -> float:
        return self.props.mu

    @property
    def cp(self) -> float:
        return self.props.cp

    @property
    def k(self) -> float:
        return self.props.k

    @property
    def h(self) -> float | None:
        return self.enthalpy

    @property
    def Re(self) -> float:
        return self.reynolds_number

    @property
    def reynolds(self) -> float:
        return self.reynolds_number

    @property
    def prandtl(self) -> float:
        return self.props.cp * self.props.mu / self.props.k

    @property
    def Pr(self) -> float:
        return self.prandtl

    @property
    def f(self) -> float:
        return self.coefficient

    @property
    def dynamic_pressure_reference(self) -> float:
        return 0.5 * self.props.rho * self.reference_velocity**2

    @property
    def maximum_gap_velocity(self) -> float:
        return self.reference_velocity

    @property
    def maximum_gap_mass_flux(self) -> float:
        return self.reference_mass_flux


@dataclass(frozen=True)
class FinnedTubeBankHydraulicResult:
    """Three-state variable-property drag plus signed acceleration."""

    inlet: FinnedTubeHydraulicPoint
    midpoint: FinnedTubeHydraulicPoint
    outlet: FinnedTubeHydraulicPoint
    midpoint_method: str
    coefficient_definition: str
    face_area: float
    face_mass_flux: float
    reference_area: float
    reference_mass_flux: float
    reference_diameter: float
    n_rows_effective: int
    mean_coefficient_over_rho: float
    dp_drag: float
    dp_acceleration: float
    dp_total: float
    metadata: FinnedTubePressureDropMetadata
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
    def mean_f_over_rho(self) -> float:
        return self.mean_coefficient_over_rho


def calculate_finned_tube_bank_hydraulics(
    m_dot: float,
    bundle: TubeBundle,
    *,
    property_provider: Any | None = None,
    temperature_in: float | None = None,
    temperature_out: float | None = None,
    pressure: float | None = None,
    inlet_props: Any | None = None,
    midpoint_props: Any | None = None,
    outlet_props: Any | None = None,
    pressure_drop_provider: (
        str | FinnedTubePressureDropProvider
    ) = "robinson_briggs_1966",
    fluid_family: str = "air",
    m_dot_inlet: float | None = None,
    m_dot_midpoint: float | None = None,
    m_dot_outlet: float | None = None,
) -> FinnedTubeBankHydraulicResult:
    """Calculate inlet/midpoint/outlet finned-bank hydraulics.

    Simpson's rule is applied to ``f_RB/rho`` for drag.  The signed
    acceleration term uses the face sections, not ``A_min``.  Either a
    thermophysical ``property_provider`` or explicit point properties must
    be supplied.  This function never constructs or dispatches an
    ``EulerRequest``.
    """

    tube = _require_circular_finned_bundle(bundle)
    _require_positive_finite(m_dot, "m_dot")
    corr_provider = _resolve_pressure_drop_provider(pressure_drop_provider)
    warnings: list[ModelWarning] = []

    if property_provider is None:
        if inlet_props is None:
            raise ValueError(
                "property_provider or inlet_props must be supplied for "
                "finned-tube bank hydraulics."
            )
        if outlet_props is not None and midpoint_props is None:
            raise ValueError(
                "midpoint_props must be supplied when outlet_props is supplied "
                "without a property_provider; three-state drag cannot infer a "
                "physical midpoint from endpoint transport properties."
            )
        props_in, h_in = _transport_and_enthalpy(inlet_props)
        props_mid, h_mid = _transport_and_enthalpy(
            inlet_props if midpoint_props is None else midpoint_props
        )
        props_out, h_out = _transport_and_enthalpy(
            inlet_props if outlet_props is None else outlet_props
        )
        T_in = 300.0 if temperature_in is None else _require_temperature(
            temperature_in, "temperature_in"
        )
        T_out = T_in if temperature_out is None else _require_temperature(
            temperature_out, "temperature_out"
        )
        T_mid = 0.5 * (T_in + T_out)
        p = 101325.0 if pressure is None else _require_pressure(pressure)
        midpoint_method = "explicit_properties"
    else:
        T_in = _require_temperature(temperature_in, "temperature_in")
        T_out = _require_temperature(temperature_out, "temperature_out")
        p = _require_pressure(pressure)
        props_in, h_in = _evaluate_property_state(property_provider, T_in, p)
        props_out, h_out = _evaluate_property_state(property_provider, T_out, p)
        T_mid = 0.5 * (T_in + T_out)
        midpoint_method = "arithmetic_temperature"
        if _finite_enthalpy(h_in) and _finite_enthalpy(h_out):
            T_from_h = _try_temperature_from_enthalpy(
                property_provider,
                h=0.5 * (float(h_in) + float(h_out)),
                p=p,
            )
            if T_from_h is None:
                warnings.append(_midpoint_fallback_warning("enthalpy inversion failed"))
            else:
                T_mid = T_from_h
                midpoint_method = "enthalpy"
        else:
            warnings.append(
                _midpoint_fallback_warning(
                    "complete inlet/outlet enthalpy data are unavailable"
                )
            )
        if midpoint_props is None:
            props_mid, h_mid = _evaluate_property_state(
                property_provider, T_mid, p
            )
        else:
            props_mid, h_mid = _transport_and_enthalpy(midpoint_props)
            midpoint_method = "explicit_properties"

    face_area = bundle.frontal_flow_area
    reference_area = bundle.minimum_free_flow_area
    face_mass_flux = m_dot / face_area
    reference_mass_flux = m_dot / reference_area
    point_mass_flows = {
        "inlet": m_dot if m_dot_inlet is None else m_dot_inlet,
        "midpoint": m_dot if m_dot_midpoint is None else m_dot_midpoint,
        "outlet": m_dot if m_dot_outlet is None else m_dot_outlet,
    }
    for name, value in point_mass_flows.items():
        _require_positive_finite(value, f"m_dot_{name}")

    points: list[FinnedTubeHydraulicPoint] = []
    metadata: FinnedTubePressureDropMetadata | None = None
    for position, temperature, props, enthalpy in (
        ("inlet", T_in, props_in, h_in),
        ("midpoint", T_mid, props_mid, h_mid),
        ("outlet", T_out, props_out, h_out),
    ):
        state = corr_provider.evaluate(
            _request_from_bundle(
                m_dot=m_dot,
                bundle=bundle,
                tube=tube,
                props=props,
                fluid_family=fluid_family,
            )
        )
        if state.coefficient_basis != ROBINSON_BRIGGS_COEFFICIENT_BASIS:
            raise ValueError(
                "Finned pressure provider must return coefficient on the "
                "delta_p/(2*n_rows*rho*V_max^2) basis."
            )
        if metadata is None:
            metadata = state.metadata
        elif state.metadata != metadata:
            raise ValueError(
                "Finned pressure-provider metadata changed between states."
            )
        point_face_mass_flux = point_mass_flows[position] / face_area
        points.append(
            FinnedTubeHydraulicPoint(
                position=position,
                temperature=float(temperature),
                pressure=p,
                props=props,
                enthalpy=enthalpy,
                face_mass_flux=point_face_mass_flux,
                face_velocity=point_face_mass_flux / props.rho,
                reference_area=reference_area,
                reference_mass_flux=reference_mass_flux,
                reference_velocity=reference_mass_flux / props.rho,
                reynolds_number=state.reynolds_number,
                coefficient=state.coefficient,
                coefficient_definition=state.coefficient_definition,
                local_dp_drag=state.dp_drag,
                metadata=state.metadata,
                warnings=state.warnings,
            )
        )
        warnings.extend(state.warnings)

    if metadata is None:
        raise ValueError("No finned-tube hydraulic states were evaluated.")
    mean_coefficient_over_rho = (
        points[0].coefficient / points[0].rho
        + 4.0 * points[1].coefficient / points[1].rho
        + points[2].coefficient / points[2].rho
    ) / 6.0
    dp_drag = (
        2.0
        * bundle.n_rows
        * reference_mass_flux**2
        * mean_coefficient_over_rho
    )
    dp_acceleration = (
        points[2].face_mass_flux**2 / points[2].rho
        - points[0].face_mass_flux**2 / points[0].rho
    )
    dp_total = dp_drag + dp_acceleration
    if not math.isfinite(dp_drag) or dp_drag < 0.0:
        raise ValueError("Finned-tube bank drag must be finite and non-negative.")
    if not math.isfinite(dp_acceleration) or not math.isfinite(dp_total):
        raise ValueError("Finned-tube bank pressure change must be finite.")
    if dp_total < 0.0:
        warnings.append(
            make_warning(
                code="finned_tube_bank_hydraulics_negative_total_pressure_change",
                message=(
                    "Signed finned-tube bank pressure change is negative, "
                    "representing net pressure recovery after bank drag."
                ),
                source="finned_tube_bank_hydraulics",
                severity="warning",
            )
        )
    return FinnedTubeBankHydraulicResult(
        inlet=points[0],
        midpoint=points[1],
        outlet=points[2],
        midpoint_method=midpoint_method,
        coefficient_definition=ROBINSON_BRIGGS_COEFFICIENT_BASIS,
        face_area=face_area,
        face_mass_flux=face_mass_flux,
        reference_area=reference_area,
        reference_mass_flux=reference_mass_flux,
        reference_diameter=tube.D_root,
        n_rows_effective=bundle.n_rows,
        mean_coefficient_over_rho=mean_coefficient_over_rho,
        dp_drag=dp_drag,
        dp_acceleration=dp_acceleration,
        dp_total=dp_total,
        metadata=metadata,
        warnings=_deduplicate_warnings(warnings),
    )


finned_tube_pressure_drop_from_mass_flow = calculate_finned_tube_pressure_drop
finned_tube_bank_hydraulics = calculate_finned_tube_bank_hydraulics


def _request_from_bundle(
    *,
    m_dot: float,
    bundle: TubeBundle,
    tube: CircularFinnedTube,
    props: FluidTransportProperties,
    fluid_family: str,
) -> FinnedTubePressureDropRequest:
    return FinnedTubePressureDropRequest(
        m_dot=m_dot,
        face_area=bundle.frontal_flow_area,
        minimum_free_flow_area=bundle.minimum_free_flow_area,
        rho=props.rho,
        mu=props.mu,
        D_root=tube.D_root,
        D_fin=tube.D_fin,
        fin_pitch=tube.fin_pitch,
        fin_thickness_root=tube.fin_thickness_root,
        fin_thickness_tip=tube.fin_thickness_tip_effective,
        pitch_transverse=bundle.pitch_transverse,
        pitch_longitudinal=bundle.pitch_longitudinal,
        layout=bundle.layout.lower(),
        n_rows=bundle.n_rows,
        fluid_family=fluid_family,
    )


def _resolve_pressure_drop_provider(
    provider: str | FinnedTubePressureDropProvider,
) -> FinnedTubePressureDropProvider:
    if isinstance(provider, str):
        name = provider.strip().lower().replace("-", "_")
        if name in (
            "robinson_briggs",
            "robinson_briggs_1966",
            "rb1966",
        ):
            return RobinsonBriggs1966Provider()
        raise ValueError(
            f"Unknown finned-tube pressure-drop provider '{provider}'. "
            "Supported built-in provider: 'robinson_briggs_1966'."
        )
    if not isinstance(provider, FinnedTubePressureDropProvider):
        raise TypeError(
            "Custom finned-tube pressure-drop provider must implement "
            "evaluate(FinnedTubePressureDropRequest)."
        )
    return provider


def _require_circular_finned_bundle(bundle: TubeBundle) -> CircularFinnedTube:
    if not isinstance(bundle, TubeBundle):
        raise TypeError("bundle must be a TubeBundle instance.")
    if not isinstance(bundle.tube, CircularFinnedTube):
        raise TypeError(
            "Dedicated finned-tube pressure drop requires bundle.tube to "
            "be CircularFinnedTube; bare tubes are unsupported."
        )
    return bundle.tube


def _robinson_briggs_warnings(
    *,
    request: FinnedTubePressureDropRequest,
    reynolds: float,
    spacing: float,
    t_mean: float,
    height: float,
) -> tuple[ModelWarning, ...]:
    warnings: list[ModelWarning] = []
    values = (
        ("reynolds", "Re_D_root", reynolds, 2000.0, 50000.0, "-"),
        ("spacing_height", "s/H", spacing / height, 0.15, 0.19, "-"),
        ("spacing_thickness", "s/t_mean", spacing / t_mean, 3.75, 6.03, "-"),
        ("height_diameter", "H/D_root", height / request.D_root, 0.35, 0.56, "-"),
        (
            "thickness_diameter",
            "t_mean/D_root",
            t_mean / request.D_root,
            0.011,
            0.025,
            "-",
        ),
        (
            "pitch_diameter",
            "P_t/D_root",
            request.pitch_transverse / request.D_root,
            1.86,
            4.60,
            "-",
        ),
        ("root_diameter", "D_root", request.D_root, 0.0186, 0.0409, "m"),
        (
            "fin_density",
            "fin_density",
            1.0 / request.fin_pitch,
            311.0,
            431.0,
            "1/m",
        ),
    )
    for key, label, value, lower, upper, units in values:
        warning = _outside_range_warning(
            method="robinson_briggs_1966",
            key=key,
            label=label,
            value=value,
            lower=lower,
            upper=upper,
            units=units,
            source="finned_tube_pressure_drop",
        )
        if warning is not None:
            warnings.append(warning)
    if request.n_rows != 6:
        warnings.append(
            make_warning(
                code="robinson_briggs_1966_row_count_secondary_extension",
                message=(
                    "Robinson-Briggs was correlated from six-row banks. The "
                    f"requested {request.n_rows} rows satisfy the later "
                    ">=4-row recommendation; f_RB is multiplied per row."
                ),
                source="finned_tube_pressure_drop",
                severity="info",
            )
        )
    if not math.isclose(
        request.fin_thickness_root,
        request.fin_thickness_tip,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        warnings.append(
            make_warning(
                code="robinson_briggs_1966_linear_taper_mean_thickness_mapping",
                message=(
                    "The real linearly tapered fin geometry is retained, "
                    "while range diagnostics use arithmetic mean thickness."
                ),
                source="finned_tube_pressure_drop",
                severity="info",
            )
        )
    if request.fluid_family.strip().lower() != "air":
        warnings.append(
            make_warning(
                code="robinson_briggs_1966_non_air_source_extrapolation",
                message=(
                    "The source experiments used air; application to another "
                    "dry single-phase gas is an unvalidated extrapolation."
                ),
                source="finned_tube_pressure_drop",
                severity="warning",
            )
        )
    return _deduplicate_warnings(warnings)


def _outside_range_warning(
    *,
    method: str,
    key: str,
    label: str,
    value: float,
    lower: float,
    upper: float,
    units: str,
    source: str,
) -> ModelWarning | None:
    if lower <= value <= upper:
        return None
    suffix = "below_range" if value < lower else "above_range"
    return make_warning(
        code=f"{method}_{key}_{suffix}",
        message=(
            f"{label}={value:.6g} {units} is outside the published "
            f"Robinson-Briggs range [{lower:.6g}, {upper:.6g}] {units}."
        ),
        source=source,
        severity="warning",
    )


def _transport_and_enthalpy(
    raw: Any,
) -> tuple[FluidTransportProperties, float | None]:
    transport = getattr(raw, "transport", raw)
    if isinstance(transport, FluidTransportProperties):
        props = transport
    else:
        try:
            props = FluidTransportProperties(
                rho=float(getattr(transport, "rho")),
                mu=float(getattr(transport, "mu")),
                k=float(getattr(transport, "k")),
                cp=float(getattr(transport, "cp")),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "Properties must expose positive rho, mu, k and cp values."
            ) from exc
    enthalpy = getattr(raw, "h", getattr(raw, "enthalpy", None))
    return props, _finite_float_or_none(enthalpy)


def _evaluate_property_state(
    provider: Any,
    temperature: float,
    pressure: float,
) -> tuple[FluidTransportProperties, float | None]:
    full_at = getattr(provider, "full_at", None)
    at = getattr(provider, "at", None)
    if callable(full_at):
        raw = full_at(T=temperature, p=pressure)
    elif callable(at):
        raw = at(T=temperature, p=pressure)
    else:
        raise TypeError("property_provider must implement at(T, p) or full_at(T, p).")
    return _transport_and_enthalpy(raw)


def _try_temperature_from_enthalpy(
    provider: Any,
    *,
    h: float,
    p: float,
) -> float | None:
    for name in ("temperature_from_h_p", "T_from_h_p"):
        method = getattr(provider, name, None)
        if not callable(method):
            continue
        try:
            raw = method(h=h, p=p)
        except Exception:
            continue
        value = raw if isinstance(raw, (int, float)) else getattr(raw, "T", None)
        parsed = _finite_float_or_none(value)
        return parsed if parsed is not None and parsed > 0.0 else None
    return None


def _midpoint_fallback_warning(reason: str) -> ModelWarning:
    return make_warning(
        code="finned_tube_bank_hydraulics_midpoint_temperature_fallback",
        message=f"{reason}; using arithmetic temperature midpoint.",
        source="finned_tube_bank_hydraulics",
        severity="info",
    )


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_enthalpy(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _require_temperature(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} must be supplied with property_provider.")
    _require_positive_finite(value, name)
    return float(value)


def _require_pressure(value: float | None) -> float:
    if value is None:
        raise ValueError("pressure must be supplied with property_provider.")
    _require_positive_finite(value, "pressure")
    return float(value)


def _deduplicate_warnings(
    warnings: list[ModelWarning],
) -> tuple[ModelWarning, ...]:
    unique: list[ModelWarning] = []
    seen: set[tuple[str, str, str, str]] = set()
    for warning in warnings:
        key = (
            warning.code,
            warning.message,
            warning.severity,
            warning.source,
        )
        if key not in seen:
            seen.add(key)
            unique.append(warning)
    return tuple(unique)


def _require_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")


__all__ = [
    "FinnedTubeLayout",
    "FinnedTubeHydraulicPosition",
    "FinnedTubePressureCoefficientBasis",
    "FinnedTubePressureDropMetadata",
    "FinnedTubePressureDropRequest",
    "FinnedTubePressureDropResult",
    "FinnedTubePressureDropProvider",
    "RobinsonBriggs1966Provider",
    "FinnedTubeHydraulicPoint",
    "FinnedTubeBankHydraulicResult",
    "ROBINSON_BRIGGS_COEFFICIENT_BASIS",
    "ROBINSON_BRIGGS_1966_APPLICABILITY",
    "ROBINSON_BRIGGS_1966_METADATA",
    "evaluate_finned_tube_pressure_drop",
    "calculate_finned_tube_pressure_drop",
    "finned_tube_pressure_drop_from_mass_flow",
    "calculate_finned_tube_bank_hydraulics",
    "finned_tube_bank_hydraulics",
]
