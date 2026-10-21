# SPDX-License-Identifier: GPL-3.0-only
"""Zhao source-domain pressure estimate at the normal 0D mean bulk state.

This result deliberately does not claim a bank-only irreversible loss or
decompose unknown inlet/exit/acceleration contributions. No extra acceleration,
row multiplier or entry/exit loss is added to the source-normalized estimate.
"""
from dataclasses import dataclass

from core.common.warnings import deduplicate_warnings
from core.heat_transfer.finned_ellipse_airside import Zhao2017AirsideModel, Zhao2017AirsideResult, validate_zhao2017_geometry
from core.pressure_drop.finned_tube_pressure_drop import _transport_and_enthalpy, _evaluate_property_state


@dataclass(frozen=True)
class Zhao2017HydraulicPoint:
    position: str
    temperature: float
    pressure: float
    props: object
    enthalpy: float | None
    m_dot: float
    airside_result: Zhao2017AirsideResult

    @property
    def reynolds_number(self): return self.airside_result.flow.reynolds_number
    @property
    def reynolds(self): return self.reynolds_number
    @property
    def face_velocity(self): return self.airside_result.flow.face_velocity
    @property
    def reference_velocity(self): return self.airside_result.flow.reference_velocity
    @property
    def coefficient(self): return self.airside_result.factors.friction_factor


@dataclass(frozen=True)
class Zhao2017BankHydraulicResult:
    inlet: Zhao2017HydraulicPoint
    midpoint: Zhao2017HydraulicPoint
    outlet: Zhao2017HydraulicPoint
    warnings: tuple
    midpoint_method: str = 'arithmetic_temperature_source_mean'
    coefficient_definition: str = 'f=2*DeltaP*Aa/(rho*um^2*A); source-domain inlet-to-outlet'
    validated_bank_irreversible: bool = False
    pressure_scope: str = 'source_domain_inlet_to_outlet_estimate'

    @property
    def dp_total(self): return self.midpoint.airside_result.pressure_drop
    @property
    def dp_estimate(self): return self.dp_total
    @property
    def dp_drag(self):
        """Legacy pressure slot: source-domain estimate, NOT certified drag."""
        return self.dp_total
    @property
    def dp_acceleration(self):
        """No separate acceleration correction is added to the estimate."""
        return 0.
    @property
    def metadata(self): return self.midpoint.airside_result


def calculate_finned_ellipse_hydraulics(*, bundle, m_dot, model=None,
        property_provider=None, temperature_in=None, temperature_out=None, pressure=None,
        inlet_props=None, midpoint_props=None, outlet_props=None,
        m_dot_inlet=None, m_dot_midpoint=None, m_dot_outlet=None):
    validate_zhao2017_geometry(bundle)
    model = model if model is not None else Zhao2017AirsideModel()
    if not isinstance(model,Zhao2017AirsideModel):
        raise TypeError('Rectangular-finned ellipse requires Zhao2017AirsideModel.')
    if any(value is not None and value != m_dot for value in (m_dot_inlet,m_dot_midpoint,m_dot_outlet)):
        raise ValueError('Zhao2017 dry pressure estimate requires constant air mass flow; wet/mass-transfer models are unsupported.')
    from core.heat_transfer.finned_ellipse_airside import _positive
    _positive(m_dot=m_dot)
    Tin = 300. if temperature_in is None else temperature_in
    Tout = Tin if temperature_out is None else temperature_out
    p = 101325. if pressure is None else pressure
    _positive(temperature_in=Tin,temperature_out=Tout,pressure=p)
    if property_provider is not None and (temperature_in is None or temperature_out is None or pressure is None):
        raise ValueError('Provider evaluation requires inlet/outlet temperatures and pressure.')
    if property_provider is None and (inlet_props is None or (outlet_props is not None and midpoint_props is None)):
        raise ValueError('Supply a provider or complete explicit point properties.')
    points=[]
    for position,T,explicit in (
        ('inlet',Tin,inlet_props),('midpoint',.5*(Tin+Tout),midpoint_props),('outlet',Tout,outlet_props)):
        if property_provider is None:
            props,h=_transport_and_enthalpy(inlet_props if explicit is None else explicit)
        else:
            props,h=_evaluate_property_state(property_provider,T,p)
        result=model.evaluate(bundle,m_dot,props)
        points.append(Zhao2017HydraulicPoint(position,T,p,props,h,m_dot,result))
    warnings=tuple(deduplicate_warnings(tuple(w for point in points for w in point.airside_result.warnings)))
    return Zhao2017BankHydraulicResult(*points,warnings)
