# SPDX-License-Identifier: GPL-3.0-only
"""Independent dry outside-film providers. All quantities use SI units.

The actual bundle is passed unchanged. Reference lengths, velocities and
applicability belong to the provider; no conventional tube-bank throat is
constructed here. The core retains fin conduction and the resistance network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
from typing import Mapping, Protocol

from core.common.warnings import ModelWarning, make_warning
from core.geometry.bundle import TubeBundle


class OutsideProviderError(ValueError):
    """Selected external outside physics could not be evaluated."""


class OutsideProviderApplicabilityError(OutsideProviderError):
    """Unsupported state or missing explicit engineering authorization."""


class OutsideProviderResultError(OutsideProviderError):
    """Invalid external result, metadata or reference quantities."""


class OutsideApplicability(str, Enum):
    SUPPORTED_EXACT = "SUPPORTED_EXACT"
    INTERPOLATED = "INTERPOLATED"
    ENGINEERING_ESTIMATE = "ENGINEERING_ESTIMATE"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class OutsideProviderPolicy:
    allow_engineering_estimates: bool = False
    allow_unresolved_pressure_scope: bool = False

    def __post_init__(self):
        if not all(isinstance(v, bool) for v in (
                self.allow_engineering_estimates, self.allow_unresolved_pressure_scope)):
            raise TypeError("Outside provider policy flags must be boolean.")


@dataclass(frozen=True)
class OutsideProviderMetadata:
    provider_name: str
    provenance: str
    reference_basis: str
    applicability: OutsideApplicability = OutsideApplicability.SUPPORTED_EXACT
    limitations: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OutsideProviderRequest:
    """Physical state and unchanged geometry; T/p may be unavailable.

    The bundle exposes dimensions, material, layout, physical rows, lattice,
    circuit and areas. A provider needing a passage area must define it itself.
    """
    bundle: TubeBundle
    T: float | None
    p: float | None
    rho: float
    mu: float
    k: float
    cp: float
    mass_flow: float
    frontal_area: float
    physical_outside_area: float
    face_velocity: float

    @property
    def Pr(self): return self.cp * self.mu / self.k

    @property
    def face_mass_flux(self): return self.mass_flow / self.frontal_area


@dataclass(frozen=True)
class OutsideReferenceQuantities:
    """Optional, provider-defined diagnostics; absent values remain unknown."""
    velocity: float | None = None
    characteristic_length: float | None = None
    reynolds_number: float | None = None
    nusselt_number: float | None = None
    coefficient: float | None = None
    coefficient_definition: str = ""
    flow_area: float | None = None
    flow_area_basis: str = ""
    mass_flux: float | None = None


@dataclass(frozen=True)
class OutsideThermalProviderResult:
    """Physical film h [W/(m² K)], before fins, contact, wall or fouling."""
    alpha_outside: float
    metadata: OutsideProviderMetadata
    area_basis: str = "physical_outside_area"
    warnings: tuple[ModelWarning, ...] = ()
    reference: OutsideReferenceQuantities = field(default_factory=OutsideReferenceQuantities)


class OutsideThermalProvider(Protocol):
    def evaluate(self, request: OutsideProviderRequest) -> OutsideThermalProviderResult: ...


def finite_number(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or (positive and value <= 0) or (nonnegative and value < 0)):
        raise OutsideProviderResultError(f"{name} must be finite and have a physical sign.")


def make_outside_request(*, bundle, m_dot, props, temperature=None, pressure=None):
    values = dict(rho=props.rho, mu=props.mu, k=props.k, cp=props.cp,
                  mass_flow=m_dot, frontal_area=bundle.frontal_flow_area,
                  physical_outside_area=bundle.total_outer_area)
    for name, value in values.items():
        finite_number(value, name, positive=True)
    for name, value in (("T", temperature), ("p", pressure)):
        if value is not None:
            finite_number(value, name, positive=True)
    return OutsideProviderRequest(bundle=bundle, T=temperature, p=pressure, **values,
        face_velocity=m_dot / (props.rho * values['frontal_area']))


def validate_metadata(metadata, warnings, reference, policy):
    if not isinstance(metadata, OutsideProviderMetadata):
        raise OutsideProviderResultError("Expected OutsideProviderMetadata.")
    if not all(isinstance(v, str) and v.strip() for v in (
            metadata.provider_name, metadata.provenance, metadata.reference_basis)):
        raise OutsideProviderResultError("Provider name, provenance and reference basis are required.")
    if not isinstance(metadata.applicability, OutsideApplicability):
        raise OutsideProviderResultError("Applicability must be OutsideApplicability.")
    if (not isinstance(metadata.limitations, tuple)
            or not all(isinstance(v, str) and v.strip() for v in metadata.limitations)
            or not isinstance(metadata.diagnostics, Mapping)):
        raise OutsideProviderResultError("Invalid limitations or diagnostics.")
    if not isinstance(warnings, tuple) or not all(isinstance(w, ModelWarning) for w in warnings):
        raise OutsideProviderResultError("Provider warnings must be a tuple of ModelWarning.")
    if not isinstance(reference, OutsideReferenceQuantities):
        raise OutsideProviderResultError("Expected OutsideReferenceQuantities.")
    for name in ('velocity', 'characteristic_length', 'reynolds_number', 'nusselt_number', 'flow_area', 'mass_flux'):
        value = getattr(reference, name)
        if value is not None:
            finite_number(value, name, positive=True)
    if not isinstance(reference.flow_area_basis, str):
        raise OutsideProviderResultError('Reference flow-area basis must be a string.')
    if reference.flow_area is not None and not reference.flow_area_basis.strip():
        raise OutsideProviderResultError('A provider reference flow area requires its actual basis.')
    if reference.coefficient is not None:
        finite_number(reference.coefficient, 'coefficient', nonnegative=True)
        if not isinstance(reference.coefficient_definition, str) or not reference.coefficient_definition.strip():
            raise OutsideProviderResultError("A pressure coefficient requires its definition.")
    if metadata.applicability is OutsideApplicability.UNSUPPORTED:
        raise OutsideProviderApplicabilityError(f"{metadata.provider_name}: unsupported operating point.")
    if metadata.applicability is OutsideApplicability.ENGINEERING_ESTIMATE:
        if not policy.allow_engineering_estimates:
            raise OutsideProviderApplicabilityError("External engineering estimate requires explicit authorization.")
        if not metadata.limitations:
            raise OutsideProviderResultError("Engineering estimates must declare limitations.")
        warnings += (make_warning(code='external_outside_engineering_estimate',
            message=f"{metadata.provider_name}: ENGINEERING_ESTIMATE; " + '; '.join(metadata.limitations),
            source='outside_provider'),)
    return warnings


def evaluate_external_thermal(provider, request, policy=OutsideProviderPolicy()):
    result = provider.evaluate(request)
    if not isinstance(result, OutsideThermalProviderResult):
        raise OutsideProviderResultError("Expected OutsideThermalProviderResult.")
    warnings = validate_metadata(result.metadata, result.warnings, result.reference, policy)
    finite_number(result.alpha_outside, 'Outside physical film HTC', positive=True)
    if result.area_basis != 'physical_outside_area':
        raise OutsideProviderResultError("Outside film HTC requires physical_outside_area basis.")
    from dataclasses import replace
    return replace(result, warnings=warnings)


def require_outside_models(hx):
    """Retain default geometry guards for every channel without injection."""
    from core.geometry.applicability import require_automatic_outside_bank
    if hx.outside_thermal_provider is None or hx.outside_hydraulic_provider is None:
        require_automatic_outside_bank(hx.bundle.tube, 'Automatic full exchanger outside-bank physics',
                                       airside_model=hx.airside_model)
