# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only

"""Dry outside heat transfer for banks of circular-finned tubes.

The contracts in this module are deliberately separate from the bare-tube
outside-flow API.  In particular, the reference diameter is the fin-root
diameter and the reference velocity is based on the bundle's actual minimum
free-flow area, including periodic fin blockage.

The built-in implementation is an independent implementation of the
published Briggs-Young (1963) correlation.  It returns the physical mean
outside film coefficient.  Fin efficiency and extended-surface area are not
folded into that coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Protocol, runtime_checkable

from core.common.warnings import ApplicabilityRange, ModelWarning, make_warning
from core.geometry.bundle import TubeBundle
from core.geometry.tube import CircularFinnedTube
from core.properties.common import FluidTransportProperties


FinnedTubeLayout = Literal["inline", "staggered"]


BRIGGS_YOUNG_1963_APPLICABILITY: tuple[ApplicabilityRange, ...] = (
    ApplicabilityRange("Re_D_root", 1100.0, 18000.0),
    ApplicabilityRange("clear_fin_spacing/fin_height", 0.13, 0.63),
    ApplicabilityRange("clear_fin_spacing/mean_fin_thickness", 1.01, 6.62),
    ApplicabilityRange("fin_height/D_root", 0.09, 0.69),
    ApplicabilityRange("mean_fin_thickness/D_root", 0.011, 0.15),
    ApplicabilityRange("pitch_transverse/D_root", 1.54, 8.23),
    ApplicabilityRange("D_root", 0.0111, 0.0409, "m"),
    ApplicabilityRange("fin_density", 246.0, 768.0, "1/m"),
    ApplicabilityRange("source_test_rows", 6.0, 6.0, "rows"),
    ApplicabilityRange("secondary_recommended_rows", 4.0, None, "rows"),
)


@dataclass(frozen=True)
class FinnedTubeHeatTransferMetadata:
    """Machine-readable provenance and basis of an HTC result."""

    method: str
    source: str
    equation: str
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


BRIGGS_YOUNG_1963_METADATA = FinnedTubeHeatTransferMetadata(
    method="briggs_young_1963",
    source=(
        "D. E. Briggs and E. H. Young, 'Convection Heat Transfer and "
        "Pressure Drop of Air Flowing Across Triangular Pitch Banks of "
        "Finned Tubes', Chemical Engineering Progress Symposium Series, "
        "Vol. 59, No. 41, 1963, pp. 1-10."
    ),
    equation=(
        "j=0.134*Re_Droot^-0.319*(s/H)^0.2*(s/t_mean)^0.1134; "
        "Nu_Droot=j*Re_Droot*Pr^(1/3); alpha=Nu_Droot*k/D_root"
    ),
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
        "physical mean outside film coefficient on the gross exposed "
        "outside surface; apply fin efficiency separately as "
        "UA=alpha*(A_primary+eta_fin*A_fin)"
    ),
    row_basis=(
        "mean coefficient correlated from six-row banks; no row correction "
        "is contained in the equation; use for >=4 rows is a later handbook "
        "recommendation and is reported diagnostically"
    ),
    source_fluid="air",
    supported_layouts=("staggered_equilateral_triangular",),
    supported_fin_profiles=("solid_circular_constant_or_linear_taper",),
    applicability=BRIGGS_YOUNG_1963_APPLICABILITY,
)


@dataclass(frozen=True)
class FinnedTubeHeatTransferRequest:
    """Complete state and geometry supplied to an outside-HTC provider.

    ``fin_thickness_root`` and ``fin_thickness_tip`` retain the real fin
    profile.  Briggs-Young alone maps that profile to the arithmetic mean
    thickness for its empirical ``s/t`` terms; the underlying geometry is
    never replaced by a mean-thickness fin.
    """

    m_dot: float
    face_area: float
    minimum_free_flow_area: float
    rho: float
    mu: float
    k: float
    cp: float
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
            ("k", self.k),
            ("cp", self.cp),
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
class FinnedTubeHeatTransferResult:
    """Physical outside film coefficient and its complete source context."""

    alpha: float
    nusselt_number: float
    colburn_j_factor: float
    stanton_number: float
    reynolds_number: float
    prandtl_number: float
    face_area: float
    minimum_free_flow_area: float
    face_mass_flux: float
    reference_mass_flux: float
    face_velocity: float
    reference_velocity: float
    reference_diameter_value: float
    fin_height: float
    correlation_fin_thickness: float
    correlation_fin_spacing: float
    diagonal_pitch: float
    equilateral_relative_deviation: float
    metadata: FinnedTubeHeatTransferMetadata
    warnings: tuple[ModelWarning, ...]

    @property
    def h(self) -> float:
        return self.alpha

    @property
    def alfa(self) -> float:
        return self.alpha

    @property
    def outside_htc(self) -> float:
        return self.alpha

    @property
    def Nu(self) -> float:
        return self.nusselt_number

    @property
    def j(self) -> float:
        return self.colburn_j_factor

    @property
    def Re(self) -> float:
        return self.reynolds_number

    @property
    def Pr(self) -> float:
        return self.prandtl_number

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
class FinnedTubeHeatTransferProvider(Protocol):
    """Provider protocol kept independent of bare-tube correlations."""

    def evaluate(
        self, request: FinnedTubeHeatTransferRequest
    ) -> FinnedTubeHeatTransferResult:
        ...


@dataclass(frozen=True)
class BriggsYoung1963Provider:
    """Briggs-Young correlation for equilateral staggered finned banks."""

    # A 0.1% engineering tolerance accepts ordinary rounded nominal pitches
    # without extending the model to a materially isosceles bank.
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
        self, request: FinnedTubeHeatTransferRequest
    ) -> FinnedTubeHeatTransferResult:
        if request.geometry_family != "solid_circular_finned_tube_bank":
            raise ValueError(
                "BriggsYoung1963Provider supports only solid circular-finned "
                "tube banks."
            )
        if request.layout != "staggered":
            raise ValueError(
                "BriggsYoung1963Provider supports only staggered triangular "
                "tube banks; inline layout is unsupported."
            )
        if request.n_rows < 4:
            raise ValueError(
                "BriggsYoung1963Provider requires at least four tube rows; "
                "the source correlation was obtained from six-row banks."
            )

        diagonal_pitch = math.hypot(
            request.pitch_longitudinal, 0.5 * request.pitch_transverse
        )
        equilateral_deviation = abs(
            diagonal_pitch / request.pitch_transverse - 1.0
        )
        if equilateral_deviation > self.equilateral_relative_tolerance:
            raise ValueError(
                "BriggsYoung1963Provider requires an equilateral triangular "
                "bank (P_d approximately equal to P_t); isosceles layouts are "
                "outside the implemented source scope."
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
        prandtl = request.cp * request.mu / request.k

        j_factor = (
            0.134
            * reynolds ** -0.319
            * (spacing / height) ** 0.2
            * (spacing / t_mean) ** 0.1134
        )
        nusselt = j_factor * reynolds * prandtl ** (1.0 / 3.0)
        alpha = nusselt * request.k / request.D_root
        stanton = j_factor / prandtl ** (2.0 / 3.0)

        for name, value in (
            ("Re", reynolds),
            ("Pr", prandtl),
            ("j", j_factor),
            ("Nu", nusselt),
            ("alpha", alpha),
        ):
            _require_positive_finite(value, name)

        warnings = _briggs_young_warnings(
            request=request,
            reynolds=reynolds,
            spacing=spacing,
            t_mean=t_mean,
            height=height,
        )

        return FinnedTubeHeatTransferResult(
            alpha=alpha,
            nusselt_number=nusselt,
            colburn_j_factor=j_factor,
            stanton_number=stanton,
            reynolds_number=reynolds,
            prandtl_number=prandtl,
            face_area=request.face_area,
            minimum_free_flow_area=request.minimum_free_flow_area,
            face_mass_flux=face_mass_flux,
            reference_mass_flux=reference_mass_flux,
            face_velocity=face_velocity,
            reference_velocity=reference_velocity,
            reference_diameter_value=request.D_root,
            fin_height=height,
            correlation_fin_thickness=t_mean,
            correlation_fin_spacing=spacing,
            diagonal_pitch=diagonal_pitch,
            equilateral_relative_deviation=equilateral_deviation,
            metadata=BRIGGS_YOUNG_1963_METADATA,
            warnings=warnings,
        )


def evaluate_finned_tube_heat_transfer(
    request: FinnedTubeHeatTransferRequest,
    *,
    provider: str | FinnedTubeHeatTransferProvider = "briggs_young_1963",
) -> FinnedTubeHeatTransferResult:
    """Evaluate a request with a built-in name or custom provider."""

    resolved = _resolve_heat_transfer_provider(provider)
    return resolved.evaluate(request)


def calculate_finned_tube_outside_heat_transfer(
    m_dot: float,
    bundle: TubeBundle,
    props: FluidTransportProperties,
    *,
    provider: str | FinnedTubeHeatTransferProvider = "briggs_young_1963",
    fluid_family: str = "air",
) -> FinnedTubeHeatTransferResult:
    """Calculate dry outside HTC from ``(m_dot, bundle, props)``.

    The adapter intentionally takes ``A_min`` from ``TubeBundle``.  It never
    reconstructs a smooth-tube gap or substitutes ``D_fin`` for a diameter.
    """

    tube = _require_circular_finned_bundle(bundle)
    transport = _coerce_transport_properties(props)
    request = FinnedTubeHeatTransferRequest(
        m_dot=m_dot,
        face_area=bundle.frontal_flow_area,
        minimum_free_flow_area=bundle.minimum_free_flow_area,
        rho=transport.rho,
        mu=transport.mu,
        k=transport.k,
        cp=transport.cp,
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
    return evaluate_finned_tube_heat_transfer(request, provider=provider)


# Concise public alias for later workflow integration.
finned_tube_outside_heat_transfer = calculate_finned_tube_outside_heat_transfer


def _resolve_heat_transfer_provider(
    provider: str | FinnedTubeHeatTransferProvider,
) -> FinnedTubeHeatTransferProvider:
    if isinstance(provider, str):
        name = provider.strip().lower().replace("-", "_")
        if name in ("briggs_young", "briggs_young_1963", "by1963"):
            return BriggsYoung1963Provider()
        raise ValueError(
            f"Unknown finned-tube heat-transfer provider '{provider}'. "
            "Supported built-in provider: 'briggs_young_1963'."
        )
    if not isinstance(provider, FinnedTubeHeatTransferProvider):
        raise TypeError(
            "Custom finned-tube heat-transfer provider must implement "
            "evaluate(FinnedTubeHeatTransferRequest)."
        )
    return provider


def _require_circular_finned_bundle(bundle: TubeBundle) -> CircularFinnedTube:
    if not isinstance(bundle, TubeBundle):
        raise TypeError("bundle must be a TubeBundle instance.")
    if not isinstance(bundle.tube, CircularFinnedTube):
        raise TypeError(
            "Dedicated finned-tube heat transfer requires "
            "bundle.tube to be CircularFinnedTube; bare tubes are unsupported."
        )
    return bundle.tube


def _coerce_transport_properties(raw: object) -> FluidTransportProperties:
    if isinstance(raw, FluidTransportProperties):
        return raw
    transport = getattr(raw, "transport", raw)
    try:
        return FluidTransportProperties(
            rho=float(getattr(transport, "rho")),
            mu=float(getattr(transport, "mu")),
            k=float(getattr(transport, "k")),
            cp=float(getattr(transport, "cp")),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "props must be FluidTransportProperties or expose positive "
            "rho, mu, k and cp values."
        ) from exc


def _briggs_young_warnings(
    *,
    request: FinnedTubeHeatTransferRequest,
    reynolds: float,
    spacing: float,
    t_mean: float,
    height: float,
) -> tuple[ModelWarning, ...]:
    warnings: list[ModelWarning] = []
    values = (
        ("reynolds", "Re_D_root", reynolds, 1100.0, 18000.0, "-"),
        ("spacing_height", "s/H", spacing / height, 0.13, 0.63, "-"),
        ("spacing_thickness", "s/t_mean", spacing / t_mean, 1.01, 6.62, "-"),
        ("height_diameter", "H/D_root", height / request.D_root, 0.09, 0.69, "-"),
        (
            "thickness_diameter",
            "t_mean/D_root",
            t_mean / request.D_root,
            0.011,
            0.15,
            "-",
        ),
        (
            "pitch_diameter",
            "P_t/D_root",
            request.pitch_transverse / request.D_root,
            1.54,
            8.23,
            "-",
        ),
        ("root_diameter", "D_root", request.D_root, 0.0111, 0.0409, "m"),
        (
            "fin_density",
            "fin_density",
            1.0 / request.fin_pitch,
            246.0,
            768.0,
            "1/m",
        ),
    )
    for key, label, value, lower, upper, units in values:
        warning = _outside_range_warning(
            method="briggs_young_1963",
            key=key,
            label=label,
            value=value,
            lower=lower,
            upper=upper,
            units=units,
            source="finned_tube_outside_ht",
        )
        if warning is not None:
            warnings.append(warning)

    if request.n_rows != 6:
        warnings.append(
            make_warning(
                code="briggs_young_1963_row_count_secondary_extension",
                message=(
                    "Briggs-Young was correlated from six-row banks. The "
                    f"requested {request.n_rows} rows satisfy the later >=4-row "
                    "recommendation, but no row correction is present."
                ),
                source="finned_tube_outside_ht",
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
                code="briggs_young_1963_linear_taper_mean_thickness_mapping",
                message=(
                    "The real linearly tapered fin geometry is retained, while "
                    "Briggs-Young uses the arithmetic mean of root and tip "
                    "thickness only in its empirical s/H and s/t terms."
                ),
                source="finned_tube_outside_ht",
                severity="info",
            )
        )
    if request.fluid_family.strip().lower() != "air":
        warnings.append(
            make_warning(
                code="briggs_young_1963_non_air_source_extrapolation",
                message=(
                    "The source experiments used air; application to another "
                    "dry single-phase gas is an unvalidated fluid extrapolation."
                ),
                source="finned_tube_outside_ht",
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
            f"Briggs-Young range [{lower:.6g}, {upper:.6g}] {units}."
        ),
        source=source,
        severity="warning",
    )


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
    "FinnedTubeHeatTransferMetadata",
    "FinnedTubeHeatTransferRequest",
    "FinnedTubeHeatTransferResult",
    "FinnedTubeHeatTransferProvider",
    "BriggsYoung1963Provider",
    "BRIGGS_YOUNG_1963_APPLICABILITY",
    "BRIGGS_YOUNG_1963_METADATA",
    "evaluate_finned_tube_heat_transfer",
    "calculate_finned_tube_outside_heat_transfer",
    "finned_tube_outside_heat_transfer",
]
