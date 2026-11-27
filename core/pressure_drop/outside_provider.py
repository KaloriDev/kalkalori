# SPDX-License-Identifier: GPL-3.0-only
"""Independent complete-bank outside pressure results with explicit scope."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Protocol

from core.common.warnings import make_warning
from core.heat_transfer.outside_provider import (
    OutsideApplicability, OutsideProviderMetadata, OutsideProviderPolicy,
    OutsideProviderRequest, OutsideProviderResultError, OutsideProviderApplicabilityError,
    OutsideReferenceQuantities, make_outside_request, validate_metadata, finite_number,
)


class OutsidePressureScope(str, Enum):
    BANK_ONLY = 'BANK_ONLY'
    LOSS_WITH_ENTRANCE_EXIT = 'LOSS_WITH_ENTRANCE_EXIT'
    COMPLETE_STATIC = 'COMPLETE_STATIC'
    UNRESOLVED_ENGINEERING = 'UNRESOLVED_ENGINEERING'


@dataclass(frozen=True)
class OutsideBankPressureDropRequest:
    inlet: OutsideProviderRequest
    midpoint: OutsideProviderRequest
    outlet: OutsideProviderRequest
    midpoint_method: str


@dataclass(frozen=True)
class OutsideBankPressureDropResult:
    """Pa over declared scope. Static differences may be signed.

    BANK_ONLY and LOSS_WITH_ENTRANCE_EXIT exclude acceleration: the core adds
    face momentum change once. COMPLETE_STATIC and UNRESOLVED_ENGINEERING
    are used as supplied, with no added acceleration or local losses.
    """
    delta_p: float
    metadata: OutsideProviderMetadata
    scope: OutsidePressureScope
    scope_description: str
    included_contributions: tuple[str, ...] = ()
    warnings: tuple = ()
    reference: OutsideReferenceQuantities = field(default_factory=OutsideReferenceQuantities)


class OutsideHydraulicProvider(Protocol):
    def evaluate(self, request: OutsideBankPressureDropRequest) -> OutsideBankPressureDropResult: ...


@dataclass(frozen=True)
class ExternalOutsideHydraulicPoint:
    position: str
    request: OutsideProviderRequest
    props: object
    enthalpy: float | None
    reference: OutsideReferenceQuantities

    @property
    def temperature(self): return self.request.T
    @property
    def pressure(self): return self.request.p
    @property
    def face_velocity(self): return self.request.face_velocity
    @property
    def reference_velocity(self): return self.reference.velocity if self.reference.velocity is not None else math.nan
    @property
    def reynolds(self): return self.reference.reynolds_number if self.reference.reynolds_number is not None else math.nan
    @property
    def reynolds_number(self): return self.reynolds
    @property
    def coefficient(self): return self.reference.coefficient if self.reference.coefficient is not None else math.nan


@dataclass(frozen=True)
class ExternalOutsideBankHydraulicResult:
    inlet: ExternalOutsideHydraulicPoint
    midpoint: ExternalOutsideHydraulicPoint
    outlet: ExternalOutsideHydraulicPoint
    midpoint_method: str
    provider_result: OutsideBankPressureDropResult
    dp_acceleration: float
    warnings: tuple

    @property
    def dp_drag(self):
        """Compatibility slot; only BANK_ONLY certifies bank irreversible loss."""
        return self.provider_result.delta_p
    @property
    def dp_total(self): return self.provider_result.delta_p + self.dp_acceleration
    @property
    def metadata(self): return self.provider_result.metadata
    @property
    def pressure_scope(self): return self.provider_result.scope
    @property
    def pressure_basis(self): return self.pressure_scope.value
    @property
    def scope_description(self): return self.provider_result.scope_description
    @property
    def validated_bank_irreversible(self): return self.pressure_scope is OutsidePressureScope.BANK_ONLY
    @property
    def coefficient_definition(self): return self.provider_result.reference.coefficient_definition


def evaluate_external_hydraulics(*, hydraulic_provider, bundle, m_dot,
        policy=OutsideProviderPolicy(), property_provider=None, temperature_in=None,
        temperature_out=None, pressure=None, inlet_props=None, midpoint_props=None,
        outlet_props=None, m_dot_inlet=None, m_dot_midpoint=None, m_dot_outlet=None):
    from core.pressure_drop.finned_tube_pressure_drop import _transport_and_enthalpy, _evaluate_property_state
    if any(v is not None and v != m_dot for v in (m_dot_inlet, m_dot_midpoint, m_dot_outlet)):
        raise OutsideProviderApplicabilityError("External outside pressure providers require constant dry mass flow.")
    Tin, Tout = temperature_in, temperature_out if temperature_out is not None else temperature_in
    Tmid = None if Tin is None or Tout is None else .5*(Tin+Tout)
    if property_provider is not None:
        if any(v is None for v in (Tin, Tout, pressure)):
            raise OutsideProviderApplicabilityError("Property evaluation requires inlet/outlet T and p.")
        for name, value in (('T_in', Tin), ('T_out', Tout), ('p', pressure)):
            finite_number(value, name, positive=True)
        states = [_evaluate_property_state(property_provider, T, pressure) for T in (Tin, Tmid, Tout)]
        if midpoint_props is not None:
            states[1] = _transport_and_enthalpy(midpoint_props)
        method = 'arithmetic_temperature'
    else:
        if inlet_props is None or (outlet_props is not None and midpoint_props is None):
            raise OutsideProviderApplicabilityError("Supply a property provider or complete explicit properties.")
        states = [_transport_and_enthalpy(v if v is not None else inlet_props)
                  for v in (inlet_props, midpoint_props, outlet_props)]
        method = 'supplied_properties'
    requests = tuple(make_outside_request(bundle=bundle, m_dot=m_dot, props=s[0], temperature=T, pressure=pressure)
                     for T,s in zip((Tin,Tmid,Tout), states))
    result = hydraulic_provider.evaluate(OutsideBankPressureDropRequest(*requests, midpoint_method=method))
    if not isinstance(result, OutsideBankPressureDropResult):
        raise OutsideProviderResultError("Expected OutsideBankPressureDropResult.")
    warnings = validate_metadata(result.metadata, result.warnings, result.reference, policy)
    if not isinstance(result.scope, OutsidePressureScope):
        raise OutsideProviderResultError("Pressure scope must be OutsidePressureScope.")
    if not isinstance(result.scope_description, str) or not result.scope_description.strip():
        raise OutsideProviderResultError("Pressure result must declare its physical scope.")
    contributions = result.included_contributions
    if (not isinstance(contributions, tuple)
            or any(not isinstance(v, str) or v not in ('entrance', 'exit', 'acceleration') for v in contributions)
            or len(set(contributions)) != len(contributions)):
        raise OutsideProviderResultError("Invalid included pressure contributions.")
    if result.scope is OutsidePressureScope.BANK_ONLY and contributions:
        raise OutsideProviderResultError("BANK_ONLY must exclude entrance, exit and acceleration.")
    if result.scope is OutsidePressureScope.LOSS_WITH_ENTRANCE_EXIT and (
            not contributions or 'acceleration' in contributions):
        raise OutsideProviderResultError("Loss scope must specify entrance/exit and exclude acceleration.")
    if result.scope is OutsidePressureScope.UNRESOLVED_ENGINEERING:
        if (result.metadata.applicability is not OutsideApplicability.ENGINEERING_ESTIMATE
                or not policy.allow_unresolved_pressure_scope):
            raise OutsideProviderApplicabilityError("Unresolved pressure scope requires explicit engineering and scope authorization.")
        warnings += (make_warning(code='external_outside_pressure_scope_unresolved',
            message='Engineering pressure estimate; measurement scope unresolved: '+result.scope_description,
            source='outside_provider'),)
    finite_number(result.delta_p, 'Outside pressure difference', nonnegative=result.scope is not OutsidePressureScope.COMPLETE_STATIC)
    acceleration = 0.
    if result.scope in (OutsidePressureScope.BANK_ONLY, OutsidePressureScope.LOSS_WITH_ENTRANCE_EXIT):
        acceleration = requests[0].face_mass_flux**2*(1/states[2][0].rho-1/states[0][0].rho)
        finite_number(acceleration, 'Face acceleration')
    points = [ExternalOutsideHydraulicPoint(pos, req, s[0], s[1], result.reference)
              for pos,req,s in zip(('inlet','midpoint','outlet'), requests, states)]
    return ExternalOutsideBankHydraulicResult(*points, method, result, acceleration, warnings)
