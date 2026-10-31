# SPDX-License-Identifier: GPL-3.0-only
"""Separate experimental compact-HX friction and pressure-estimate contract."""
from dataclasses import dataclass
from typing import Protocol

from core.heat_transfer.continuous_fin_source import (
    ContinuousFinRequest, ContinuousFinSourceCard, WANG2000_SOURCE,
    kernel_arguments, positive, source_warnings, wang2000_f,
)


@dataclass(frozen=True)
class ContinuousFinDPResult:
    coefficient: float
    core_friction_estimate: float
    metadata: ContinuousFinSourceCard = WANG2000_SOURCE
    coefficient_definition: str = "dp_core_reference=f*(Ao/Ac)*Gc^2/(2*rho); compact-HX, not Darcy"
    warnings: tuple = ()


class ContinuousFinDPProvider(Protocol):
    def evaluate(self, request: ContinuousFinRequest) -> ContinuousFinDPResult: ...


@dataclass(frozen=True)
class ExperimentalWang2000DP:
    def evaluate(self, request):
        f = positive(wang2000_f(**kernel_arguments(request)), "compact-HX f")
        flow = request.flow
        dp = f*(flow.source_face_cylinder_area/flow.minimum_free_flow_area)*request.mass_flux**2/(2*request.props.rho)
        return ContinuousFinDPResult(f, positive(dp, "core friction estimate"), warnings=source_warnings())


def wang_density_change_estimate(*, mass_flux, sigma, density_in, density_out):
    """Signed source-form density term, separate from the core friction estimate.

    Not a separately validated irreversible loss or a generic momentum term.
    Entry/exit terms are unavailable. Never apply a positive drag correction to
    this signed contribution and never append another acceleration term.
    """
    for name, value in (("mass_flux", mass_flux), ("density_in", density_in), ("density_out", density_out), ("sigma", sigma)):
        positive(value, name)
    if sigma > 1:
        raise ValueError("sigma must not exceed one.")
    return mass_flux**2/(2*density_in)*(1+sigma**2)*(density_in/density_out-1)


@dataclass(frozen=True)
class ContinuousFinHydraulicPoint:
    position: str
    temperature: float
    pressure: float
    props: object
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
    metadata: object
    warnings: tuple

    @property
    def rho(self): return self.props.rho
    @property
    def mu(self): return self.props.mu
    @property
    def k(self): return self.props.k
    @property
    def cp(self): return self.props.cp
    @property
    def h(self): return self.enthalpy
    @property
    def reynolds(self): return self.reynolds_number
    @property
    def prandtl(self): return self.cp*self.mu/self.k


@dataclass(frozen=True)
class ContinuousFinBankHydraulicResult:
    inlet: ContinuousFinHydraulicPoint
    midpoint: ContinuousFinHydraulicPoint
    outlet: ContinuousFinHydraulicPoint
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
    metadata: object
    warnings: tuple
    continuous_result: object
    validated_bank_irreversible: bool = False
    components_resolved: bool = False
    pressure_basis: str = "experimental_compact_HX_core_plus_source_density_term_entry_exit_unresolved"


def calculate_continuous_fin_hydraulics(*, bundle, m_dot, model,
        property_provider=None, temperature_in=None, temperature_out=None, pressure=None,
        inlet_props=None, midpoint_props=None, outlet_props=None,
        m_dot_inlet=None, m_dot_midpoint=None, m_dot_outlet=None, strict_source_domain=False):
    """Harmonic endpoint density for friction; separate signed source density term.

    Compatibility dp_drag/dp_acceleration names expose research estimates only.
    The unresolved flag prevents irreversible-loss aggregation. Corrections act
    on the positive core friction estimate, before adding the signed term.
    """
    from core.properties.common import FluidTransportProperties
    from core.pressure_drop.finned_tube_pressure_drop import _evaluate_property_state, _transport_and_enthalpy
    from core.heat_transfer.continuous_fin_source import ContinuousFinSourceError
    positive(m_dot, "m_dot")
    for value in (m_dot_inlet, m_dot_midpoint, m_dot_outlet):
        if value is not None and value != m_dot:
            raise ContinuousFinSourceError("Continuous dry-air hydraulics requires constant external mass flow.")
    if property_provider is not None:
        model.guard_property_provider(property_provider)
        tin, tout, p = positive(temperature_in, "temperature_in"), positive(temperature_out, "temperature_out"), positive(pressure, "pressure")
        states = tuple(_evaluate_property_state(property_provider, T, p) for T in (tin, (tin+tout)/2, tout))
    else:
        if inlet_props is None or (outlet_props is not None and midpoint_props is None):
            raise ValueError("Provide a property provider or explicit inlet/midpoint/outlet properties.")
        states = tuple(_transport_and_enthalpy(p) for p in (
            inlet_props, inlet_props if midpoint_props is None else midpoint_props,
            inlet_props if outlet_props is None else outlet_props))
        tin = 300. if temperature_in is None else positive(temperature_in, "temperature_in")
        tout = tin if temperature_out is None else positive(temperature_out, "temperature_out")
        p = 101325. if pressure is None else positive(pressure, "pressure")
    pin, pmid, pout = tuple(state[0] for state in states)
    rho = 2/(1/pin.rho+1/pout.rho)
    mean = FluidTransportProperties(rho, pmid.mu, pmid.k, pmid.cp)
    result = model.evaluate(bundle, m_dot, mean, strict_source_domain=strict_source_domain)
    native = result.baseline.pressure_drop
    area, face = bundle.minimum_free_flow_area, bundle.frontal_flow_area
    diameter = bundle.continuous_fin_geometry.collar_outer_diameter
    points = tuple(ContinuousFinHydraulicPoint(position, T, p, props, enthalpy,
        m_dot/face, m_dot/(face*props.rho), area, m_dot/area, m_dot/(area*props.rho),
        m_dot*diameter/(area*props.mu), native.coefficient, native.coefficient_definition,
        result.core_friction_estimate*rho/props.rho, result.metadata, result.warnings)
        for position, T, (props, enthalpy) in zip(("inlet", "midpoint", "outlet"), (tin, (tin+tout)/2, tout), states))
    density_term = wang_density_change_estimate(mass_flux=m_dot/area, sigma=area/face,
        density_in=pin.rho, density_out=pout.rho)
    return ContinuousFinBankHydraulicResult(*points, "mean_temperature_transport_harmonic_endpoint_density",
        native.coefficient_definition, face, m_dot/face, area, m_dot/area, diameter,
        bundle.n_rows, native.coefficient/rho, result.core_friction_estimate, density_term,
        result.core_friction_estimate+density_term, result.metadata, result.warnings, result)
