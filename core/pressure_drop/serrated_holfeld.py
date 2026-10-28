# SPDX-License-Identifier: GPL-3.0-only
"""Holfeld native Eu and separate face-area acceleration, explicit only."""
from dataclasses import dataclass
import math

from core.geometry.serrated_source import (
    SerratedTubeBankRequest, holfeld_applicability, holfeld_domain, holfeld_diagnostics, positive,
)
from core.pressure_drop.finned_tube_pressure_drop import (
    FinnedTubePressureDropMetadata, FinnedTubePressureDropResult,
    FinnedTubeHydraulicPoint, FinnedTubeBankHydraulicResult,
    HOLFELD_2016_EU_COEFFICIENT_BASIS,
    _transport_and_enthalpy, _evaluate_property_state,
)
from core.properties.common import FluidTransportProperties


HOLFELD_2016_DP_METADATA = FinnedTubePressureDropMetadata(
    method="holfeld2016_serrated_staggered_dp",
    source="A. Holfeld (2016), NTNU 2016:303, pp.56–57,89,112–115,130; hdl:11250/2432512",
    equation="Eq.6-10: Eu=9.661*Re^(-.197)*(Pt/Pl)^.931*(A_min/Af_fin)^(-.776)*(1-hs/hf)^.112; Eq.4-8: dp=Eu*Nl*rho*u_max^2/2",
    coefficient_definition=HOLFELD_2016_EU_COEFFICIENT_BASIS,
    coefficient_basis=HOLFELD_2016_EU_COEFFICIENT_BASIS,
    geometry_family="HF-welded constant-thickness untwisted serrated I-foot round tube bank",
    velocity_basis="u_max=mdot/(rho*A_min_source), Eq.5-9",
    reynolds_basis="Re=mdot*do/(A_min_source*mu)",
    reference_diameter="bare tube do (D_root=core.D_o)",
    area_basis="source-equivalent A_min Eq.5-9; acceleration uses face sections separately",
    row_basis="native Eu per longitudinal row; multiply Nl once; Nl>=4, upper unreported",
    source_fluid="dry air; adiabatic bundle measurements",
    supported_layouts=("staggered",), supported_fin_profiles=("serrated_constant_untwisted_welded_i_foot",),
    applicability=holfeld_applicability("DP"),
)


@dataclass(frozen=True)
class Holfeld2016SerratedStaggeredPressureDropProvider:
    provider_name = "holfeld2016_serrated_staggered_dp"
    source_reference = HOLFELD_2016_DP_METADATA.source
    hydraulic_state_basis = "mean_outside_gas"

    def evaluate(self, request: SerratedTubeBankRequest) -> FinnedTubePressureDropResult:
        v = holfeld_domain(request, "DP")
        b, t, p = request.bundle, request.bundle.tube, request.props
        Re = request.reynolds_number
        Eu = 9.661*Re**(-.197)*v["Pt/Pl"]**.931*v["A_min/Af_fin"]**(-.776)*(1-v["hs/hf"])**.112
        velocity = request.m_dot/(p.rho*request.minimum_free_flow_area)
        dp = Eu*b.n_rows*p.rho*velocity**2/2
        diagnostics = {
            **holfeld_diagnostics(request, "DP", v),
            "provider": self.provider_name, "source": self.source_reference, "equation": "6-10 / 4-8",
            "row_basis": HOLFELD_2016_DP_METADATA.row_basis,
            "coefficient_definition": HOLFELD_2016_EU_COEFFICIENT_BASIS,
            "dp_scope": "adiabatic measured bundle-section loss; source neglects ~1% acceleration",
            "pressure_tap_details": "test-section differential; exact distances and blank-channel subtraction unreported",
            "calibration": "mixed literature + adiabatic Holfeld bundle experiments",
            "conversion_to_RB_Darcy_Fanning_Frass_Genic": "NONE",
            "source_reported_regression_coverage": {"classification": "SOURCE_REPORTED_REGRESSION_COVERAGE",
                "points": 442, "fraction_within_10_percent": .44, "fraction_within_34_percent": .95, "R_squared": .739},
        }
        return FinnedTubePressureDropResult(
            coefficient=Eu, coefficient_definition=HOLFELD_2016_EU_COEFFICIENT_BASIS,
            coefficient_basis=HOLFELD_2016_EU_COEFFICIENT_BASIS, dp_drag=dp,
            reynolds_number=Re, face_area=b.frontal_flow_area,
            minimum_free_flow_area=request.minimum_free_flow_area,
            face_mass_flux=request.m_dot/b.frontal_flow_area,
            reference_mass_flux=request.m_dot/request.minimum_free_flow_area,
            face_velocity=request.m_dot/(p.rho*b.frontal_flow_area), reference_velocity=velocity,
            reference_diameter_value=t.D_o, n_rows_effective=b.n_rows,
            fin_height=t.fin_height, correlation_fin_thickness=t.fin_thickness_root,
            correlation_fin_spacing=t.fin_pitch,
            diagonal_pitch=math.hypot(b.pitch_longitudinal,b.pitch_transverse/2),
            equilateral_relative_deviation=abs(b.pitch_longitudinal/(math.sqrt(3)*b.pitch_transverse/2)-1),
            metadata=HOLFELD_2016_DP_METADATA, warnings=(), diagnostics=diagnostics,
        )


def calculate_holfeld_serrated_hydraulics(*, bundle, m_dot, model,
        property_provider=None, temperature_in=None, temperature_out=None, pressure=None,
        inlet_props=None, midpoint_props=None, outlet_props=None,
        m_dot_inlet=None, m_dot_midpoint=None, m_dot_outlet=None):
    """Mean-temperature Eu bank loss plus endpoint face momentum change.

    No Simpson/RB conversion. Endpoint coefficients are the one evaluated
    mean-state Eu, not three independently evaluated regression coefficients.
    """
    model.validate_bundle(bundle)
    positive(m_dot, "m_dot")
    if property_provider is not None:
        model.guard_property_provider(property_provider, "outside")
        Tin, Tout, pressure = (positive(temperature_in, "temperature_in"),
                               positive(temperature_out, "temperature_out"), positive(pressure, "pressure"))
        states = [_evaluate_property_state(property_provider, T, pressure)
                  for T in (Tin, (Tin+Tout)/2, Tout)]
        method = "arithmetic_temperature_endpoint_mean_density"
    else:
        if inlet_props is None:
            raise ValueError("property_provider or inlet_props is required.")
        if outlet_props is not None and midpoint_props is None:
            raise ValueError("midpoint_props required with explicit endpoint properties.")
        states = [_transport_and_enthalpy(p) for p in (inlet_props,
            inlet_props if midpoint_props is None else midpoint_props,
            inlet_props if outlet_props is None else outlet_props)]
        Tin = 300. if temperature_in is None else positive(temperature_in,"temperature_in")
        Tout = Tin if temperature_out is None else positive(temperature_out,"temperature_out")
        pressure = 101325. if pressure is None else positive(pressure,"pressure")
        method = "explicit_mean_temperature_properties_endpoint_mean_density"
    pin, pmid, pout = [s[0] for s in states]
    rho = (pin.rho+pout.rho)/2  # Holfeld Eq.4-8, not density at mean T
    mean_props = FluidTransportProperties(rho, pmid.mu, pmid.k, pmid.cp)
    model.evaluate_thermal(bundle, m_dot, mean_props)  # complete-domain intersection
    native = model.pressure_drop_provider.evaluate(model.request(bundle, m_dot, mean_props))
    area, face = native.minimum_free_flow_area, bundle.frontal_flow_area
    mass_flows = [m_dot if x is None else positive(x, "point mass flow")
                  for x in (m_dot_inlet,m_dot_midpoint,m_dot_outlet)]
    if any(x != m_dot for x in mass_flows):
        from core.geometry.serrated_source import SerratedSourceDomainError
        raise SerratedSourceDomainError("DP", ("dry single-phase constant mass flow required",))
    points = []
    for position, T, state in zip(("inlet","midpoint","outlet"),
                                  (Tin,(Tin+Tout)/2,Tout), states):
        p, enthalpy = state
        points.append(FinnedTubeHydraulicPoint(
            position=position, temperature=T, pressure=pressure, props=p, enthalpy=enthalpy,
            face_mass_flux=m_dot/face, face_velocity=m_dot/(face*p.rho),
            reference_area=area, reference_mass_flux=m_dot/area,
            reference_velocity=m_dot/(area*p.rho), reynolds_number=m_dot*bundle.tube.D_o/(area*p.mu),
            coefficient=native.coefficient, coefficient_definition=native.coefficient_definition,
            local_dp_drag=native.coefficient*bundle.n_rows*(m_dot/area)**2/(2*p.rho),
            metadata=native.metadata, warnings=(),
        ))
    acceleration = (m_dot/face)**2*(1/pout.rho-1/pin.rho)
    return FinnedTubeBankHydraulicResult(
        inlet=points[0], midpoint=points[1], outlet=points[2], midpoint_method=method,
        coefficient_definition=native.coefficient_definition, face_area=face,
        face_mass_flux=m_dot/face, reference_area=area, reference_mass_flux=m_dot/area,
        reference_diameter=bundle.tube.D_o, n_rows_effective=bundle.n_rows,
        mean_coefficient_over_rho=native.coefficient/rho, dp_drag=native.dp_drag,
        dp_acceleration=acceleration, dp_total=native.dp_drag+acceleration,
        metadata=native.metadata, warnings=(), native_result=native,
    )
