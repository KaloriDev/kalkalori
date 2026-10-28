# SPDX-License-Identifier: GPL-3.0-only
"""Original-source facts, independent arithmetic, conduction and routing checks.

Provenance and rounding: docs/validation/serrated_source_mapping_v095.md.
No source PDF, figure, full data table or extracted text is distributed.
"""
from dataclasses import replace
import math
import importlib

import numpy as np
import pytest
from scipy.integrate import solve_bvp, quad

from core.geometry import BareTube, SerratedCircularFinnedTube, TubeBundle, TubeBankLattice
from core.geometry.serrated_flow_area import Holfeld2016SerratedFlowAreaProvider, SerratedFlowAreaBasis
from core.geometry.serrated_source import (
    HolfeldSerratedSourceGeometry, SerratedTubeBankRequest, SerratedSourceDomainError,
    enforce_ranges, holfeld_applicability,
)
from core.heat_transfer.serrated_fin_efficiency import (
    HashizumeSerratedFinEfficiencyProvider, hashizume_theoretical_efficiency,
    HASHIZUME_STAGGERED_APPLICABILITY,
)
from core.heat_transfer.serrated_holfeld import Holfeld2016SerratedStaggeredHeatTransferProvider
from core.heat_transfer.serrated_airside import Holfeld2016SerratedStaggeredModel
from core.heat_transfer.outside_dispatch import (
    evaluate_outside_thermal, evaluate_outside_hydraulics, calculate_resistance_network,
    IncompatibleOutsideCorrelationProviderError, fin_surface_temperatures,
)
from core.pressure_drop.serrated_holfeld import Holfeld2016SerratedStaggeredPressureDropProvider
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.models.bare_tube import BareTubeHeatExchanger
from core.models import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.phase_change.types import PhaseChangeMode

PROPS = FluidTransportProperties(1.2, 2e-5, .03, 1005.)
WS = .0045
FLOW = Holfeld2016SerratedFlowAreaProvider()
HT = Holfeld2016SerratedStaggeredHeatTransferProvider()
DP = Holfeld2016SerratedStaggeredPressureDropProvider()
ETA = HashizumeSerratedFinEfficiencyProvider()


def source_bank(**changes):
    """Holfeld Geometry 1 flow cells; not an active-tube heat-duty fixture."""
    tube = SerratedCircularFinnedTube(
        core_tube=BareTube(.025, .03175, .5, .5, 45.), fin_k=45.,
        D_root=.03175, D_fin=.06775, fin_thickness_root=.001,
        fin_pitch=1/277, serration_depth=.011, kerf_width_at_tip=0.,
        segments_per_turn=32, fin_density_material=7850.)
    return replace(TubeBundle(tube,8,4.5,.06975,.06975*math.sqrt(3)/2,"staggered",1,
                             flow_arrangement="counterflow"), **changes)


def request(Re=20000., bank=None, props=PROPS):
    bank = source_bank() if bank is None else bank
    area = FLOW.evaluate(bank)
    mass = Re*area*props.mu/bank.tube.D_o
    return SerratedTubeBankRequest(bank,mass,props,WS,area)


def test_flow_source_case_and_independent_surface_definition():
    b = source_bank()
    assert FLOW.evaluate(b) == pytest.approx(.063063, abs=1e-14)
    g = HolfeldSerratedSourceGeometry(b,WS)
    assert g.area_ratio == pytest.approx(16.58761154855643)
    assert g.ratios(FLOW.evaluate(b))["Aht/A_min"] == pytest.approx(59.03170811203398)
    assert g.ratios(FLOW.evaluate(b))["A_min/Af_fin"] == pytest.approx(1.0768403258029814)
    assert FLOW.basis is SerratedFlowAreaBasis.SOURCE_EQUIVALENT_FLOW_AREA
    assert FLOW.diagnostics(b)["flow_area_source"] == "HOLFELD_2016"
    assert FLOW.evaluate(b) <= b.frontal_flow_area
    assert g.total_area != pytest.approx(b.total_outer_geometric_area)
    assert g.total_area_per_length == pytest.approx(1.6545407149640885)
    from core.geometry import SerratedFinPhysicsNotSupportedError
    # Equal numerical values in special zero-kerf limits do not establish
    # interchangeable provenance. Orthographic geometry stays unimplemented.
    with pytest.raises(SerratedFinPhysicsNotSupportedError):
        _ = b.minimum_free_flow_area


@pytest.mark.parametrize("depth,count", [(0.,32),(.009,32),(.015,40)])
def test_flow_contains_no_depth_count_mass_or_surface_correction(depth,count):
    b = source_bank()
    changed = replace(b,tube=replace(b.tube,serration_depth=depth,
                                    segments_per_turn=count,segment_width_at_tip=None))
    assert FLOW.evaluate(changed) == FLOW.evaluate(b)
    if depth != b.tube.serration_depth:
        assert changed.tube.fin_area_per_length != b.tube.fin_area_per_length


def test_flow_rejects_invalid_free_width_and_unsupported_source_mapping():
    # A minimal bundle contract isolates the area law from envelope validation.
    from types import SimpleNamespace
    b = source_bank()
    invalid = SimpleNamespace(tube=b.tube,pitch_transverse=.02,n_tubes_per_row=4.5,frontal_flow_area=.05)
    with pytest.raises(ValueError,match="minimum free width"):
        FLOW.evaluate(invalid)
    for tube in (replace(b.tube,kerf_width_at_tip=.0001,segment_width_at_tip=None),
                 replace(b.tube,D_root=.032)):
        with pytest.raises(SerratedSourceDomainError,match="kerf|D_root"):
            FLOW.evaluate(replace(b,tube=tube))


# SOURCE_EQUATION_ARITHMETIC: 45-digit Decimal evaluation of the original,
# unexpanded printed Ar equation and regressions, separate from production.
ARITHMETIC = [
    (5000.,40.91498855693277,38.65983170733805,1.7522457524754135,57.9410420972712),
    (20000.,107.37821693181694,101.45973253400026,1.3334882633507793,705.5056015413221),
    (49000.,200.3437426961911,189.30117420112545,1.117695439756834,3549.497841461428),
]


@pytest.mark.parametrize("Re,Nu,h,Eu,dp", ARITHMETIC)
def test_source_equation_arithmetic(Re,Nu,h,Eu,dp):
    r = request(Re)
    ht, drag = HT.evaluate(r), DP.evaluate(r)
    assert ht.Re == pytest.approx(Re)
    assert ht.Nu == pytest.approx(Nu,rel=2e-13)
    assert ht.alpha == pytest.approx(h,rel=2e-13)
    assert drag.eu == pytest.approx(Eu,rel=2e-13)
    assert drag.dp_drag == pytest.approx(dp,rel=2e-13)
    assert ht.reference_diameter_value == r.bundle.tube.D_o
    assert ht.reference_velocity == pytest.approx(r.m_dot/(r.props.rho*r.minimum_free_flow_area))
    assert drag.dp_drag == pytest.approx(drag.eu*r.bundle.n_rows*r.props.rho*drag.reference_velocity**2/2)
    assert math.isnan(drag.f) and math.isnan(drag.xi)
    assert drag.coefficient_basis == "2*delta_p/(n_rows*rho*u_max^2)"
    assert ht.diagnostics["htc_semantic"] == "PHYSICAL_FILM"
    assert not ht.diagnostics["eta_included"]


def test_native_row_factor_and_prandtl_power():
    r = request()
    a,b = DP.evaluate(r),DP.evaluate(replace(r,bundle=replace(r.bundle,n_rows=4)))
    assert a.eu == b.eu
    assert a.dp_drag == 2*b.dp_drag
    highcp = replace(r,props=replace(PROPS,cp=8*PROPS.cp))
    assert HT.evaluate(highcp).alpha == pytest.approx(2*HT.evaluate(r).alpha)
    assert DP.evaluate(highcp).eu == a.eu


def source_air_transport(T_C, p=101325.):
    """Test-only Holfeld p.57–58 Eq.4-11 k/mu; cp supplied, not refitted."""
    T = T_C+273.15
    k = sum(c*T**i for i,c in enumerate((-.908e-3,.112e-3,-.084333e-6,.056964e-9,-.015631e-12)))
    mu = sum(c*T**i for i,c in enumerate((-.01702e-5,.79965e-7,-.72183e-10,.04960e-12,-.01388e-15)))
    return FluidTransportProperties(p/(287*T),mu,k,1007.)


# SOURCE_MEASURED_POINT: Holfeld Appendix II pp.180–181, minimal facts.
# HTC pressure was not tabulated; comparisons of Re/reduced Nu/h do not use it.
MEASURED_HT = [(.34,117.9,32.,8312.,73.,87.),
               (.81,120.3,48.2,19196.,126.,147.),
               (1.43,123.6,63.9,33078.,177.,202.)]
MEASURED_DP = [(.33,18.6,100092.,9257.,173.,1.84),
               (.74,20.3,100077.,20380.,678.,1.47),
               (1.29,23.2,100114.,35127.,1806.,1.29)]


@pytest.mark.parametrize("mass,Tin,Tout,Re_meas,h_meas,reduced_nu_meas",MEASURED_HT)
def test_source_measured_ht_points(mass,Tin,Tout,Re_meas,h_meas,reduced_nu_meas):
    b = source_bank()
    p = source_air_transport((Tin+Tout)/2)
    r = SerratedTubeBankRequest(b,mass,p,WS,FLOW.evaluate(b))
    ht = HT.evaluate(r)
    assert ht.Re == pytest.approx(Re_meas,rel=.025)  # printed flow/geometry rounding
    # Measured residual, not an exact reproduction or uncertainty guarantee.
    assert .75 < ht.alpha/h_meas < 1.02
    assert .75 < ht.diagnostics["Nu_Pr_minus_one_third"]/reduced_nu_meas < 1.02
    assert ht.diagnostics["source_reported_regression_coverage"]["points"] == 284


@pytest.mark.parametrize("mass,T,pin,Re_meas,dp_meas,Eu_meas",MEASURED_DP)
def test_source_measured_pressure_points(mass,T,pin,Re_meas,dp_meas,Eu_meas):
    b = source_bank()
    props = source_air_transport(T,pin-dp_meas/2)  # source endpoint-mean density
    result = DP.evaluate(SerratedTubeBankRequest(b,mass,props,WS,FLOW.evaluate(b)))
    assert result.reynolds_number == pytest.approx(Re_meas,rel=.025)
    assert .80 < result.eu/Eu_meas < 1.05
    assert .80 < result.dp_drag/dp_meas < 1.05
    assert "adiabatic measured bundle-section" in result.diagnostics["dp_scope"]


@pytest.mark.parametrize("mh,estimate",[(.5,.91),(1.,.74),(1.5,.58)])
def test_digitized_hashizume_theoretical_graph(mh,estimate):
    # DIGITIZED_SOURCE_ESTIMATE: 1999 p.1501 Fig.6, theoretical graph only.
    # df/do=1.8; hs/h=2/3; Re=None; experimental correction NOT applied.
    base = source_bank().tube
    t = replace(base,D_fin=base.D_o*1.8,serration_depth=base.D_o*.4*2/3,
                segment_width_at_tip=None)
    film = (mh/t.fin_height)**2*t.fin_k*t.fin_thickness_root/2
    eta,tip,area = hashizume_theoretical_efficiency(t,film)
    assert eta == pytest.approx(estimate,abs=.02)
    assert 0 < tip < eta < 1 and area > 0


@pytest.mark.parametrize("depth_fraction",[0.,.5,.86,1.])
@pytest.mark.parametrize("mh",[.1,1.,2.5])
def test_hashizume_against_independent_boundary_value_integration(depth_fraction,mh):
    base = source_bank().tube
    t = replace(base,serration_depth=base.fin_height*depth_fraction,segment_width_at_tip=None)
    r0, r2 = t.D_o/2,t.D_fin/2
    rs = r2-t.serration_depth
    m = mh/t.fin_height
    film = m*m*t.fin_k*t.fin_thickness_root/2
    # Independent two-region ODE on radial coordinate. Circumference is
    # fixed at the junction in the straight part, continuous in the root.
    def ode(r,y):
        return np.vstack((y[1],m*m*y[0]-np.where(r<rs,1/r,0)*y[1]))
    def bc(left,right):
        return np.array([left[0]-1,right[1]])
    x = np.unique(np.r_[np.linspace(r0,r2,250),rs])
    solution = solve_bvp(ode,bc,x,np.vstack((np.ones_like(x),np.zeros_like(x))),tol=1e-7,max_nodes=10000)
    # The differential coefficient jumps at rs; the integrated state is
    # independently verified even if mesh convergence flags that one point.
    integrand = lambda r: min(r,rs)*float(solution.sol(r)[0])
    denominator = (rs*rs-r0*r0)/2+rs*(r2-rs)
    integral = quad(integrand,r0,r2,points=[rs],epsabs=1e-11)[0]/denominator
    eta,tip,_ = hashizume_theoretical_efficiency(t,film)
    assert eta == pytest.approx(integral,rel=2e-6)
    assert tip == pytest.approx(float(solution.sol(r2)[0]),rel=2e-6)


@pytest.mark.parametrize("film",[0.,1e-12,1e-8,.001,1.,50.,200.])
def test_hashizume_limits_and_no_solid_eta_call(film,monkeypatch):
    def forbidden(*a,**kw):
        raise AssertionError("solid annular eta fallback")
    monkeypatch.setattr(importlib.import_module("core.heat_transfer.fin_efficiency"),"annular_fin_efficiency",forbidden)
    t = source_bank().tube
    eta,tip,_ = hashizume_theoretical_efficiency(t,film)
    assert 0 < tip <= eta <= 1
    if film < .001:
        assert eta == pytest.approx(1.,abs=1e-6)


def test_corrected_eta_uses_hs_over_ws_natural_log_and_explicit_cap():
    t = source_bank().tube
    eta = ETA.evaluate(t,100.,reynolds_number=20000.,source_segment_width=WS)
    expected = 1-(.016*(.011/.0045)+.14*(.06775/.03175)**2.7
                  *(1-.097*math.log(20000.)))*math.sqrt(2*100/(45*.001))*.018
    assert eta.correction == pytest.approx(expected)
    assert eta.fin_efficiency == pytest.approx(eta.fin_efficiency_theoretical*expected)
    assert eta.tip_temperature_ratio is None
    assert eta.diagnostics["temperature_field_status"].startswith("THEORETICAL_ONLY")
    # Small hs/ws and upper source Re can exceed unity before the source cap.
    shallow = replace(t,serration_depth=0.,segment_width_at_tip=None)
    capped = ETA.evaluate(shallow,10.,reynolds_number=50000.,source_segment_width=WS)
    assert capped.diagnostics["psi_raw"] > 1
    assert capped.correction == 1


@pytest.mark.parametrize("component",["HT","DP"])
@pytest.mark.parametrize("Re",[4999.,5000.,49999.,50000.,50001.])
def test_individual_holfeld_vs_complete_re_intersection(component,Re):
    r = request(Re)
    provider = HT if component == "HT" else DP
    if Re >= 50000:
        with pytest.raises(SerratedSourceDomainError,match=component+"_DOMAIN_FAIL"):
            provider.evaluate(r)
    else:
        provider.evaluate(r)  # Holfeld lower Re is unreported
    model = Holfeld2016SerratedStaggeredModel(WS)
    if 5000 <= Re < 50000:
        model.evaluate_thermal(r.bundle,r.m_dot,r.props)
    else:
        with pytest.raises(SerratedSourceDomainError,match="DOMAIN_FAIL"):
            model.evaluate_thermal(r.bundle,r.m_dot,r.props)


HOLFIELD_FINITE_BOUNDS = [(c,b.name,side) for c in ("HT","DP") for b in holfeld_applicability(c)
                        for side in ("lower","upper") if getattr(b,side) is not None]


@pytest.mark.parametrize("component,name,side",HOLFIELD_FINITE_BOUNDS)
def test_each_declared_holfeld_bound_is_strictly_guarded(component,name,side):
    # Isolate every published scalar interval, including bounds that cannot
    # be varied independently in a manufactured geometry.
    ranges = holfeld_applicability(component)
    values = request().source_geometry.ratios(FLOW.evaluate(source_bank()))
    values["Re"] = 20000.
    bound = next(x for x in ranges if x.name == name)
    limit = getattr(bound,side)
    values[name] = limit - 1e-6*max(abs(limit),.01) if side == "lower" else limit + 1e-6*max(abs(limit),.01)
    with pytest.raises(SerratedSourceDomainError,match=component+"_DOMAIN_FAIL"):
        enforce_ranges(component,values,ranges,open_upper=("Re","Pt/Pl","angle_deg"))


@pytest.mark.parametrize("name",[x.name for x in HASHIZUME_STAGGERED_APPLICABILITY])
@pytest.mark.parametrize("side",["lower","upper"])
def test_hashizume_each_source_bound(name,side):
    ranges = HASHIZUME_STAGGERED_APPLICABILITY
    values = {"df/do":2.,"hs/hf":.6,"hs/ws":2.,"Re_o":20000.,"m*hf":1.}
    bound = next(x for x in ranges if x.name == name)
    values[name] = getattr(bound,side)+(-1 if side == "lower" else 1)*1e-5
    with pytest.raises(SerratedSourceDomainError,match="ETA_DOMAIN_FAIL"):
        enforce_ranges("ETA",values,ranges)


def test_component_failure_is_visible_in_complete_profile():
    b = source_bank()
    # Deep, narrow strips pass Holfeld; Hashizume hs/ws fails.
    model = Holfeld2016SerratedStaggeredModel(.0042)
    r = request()
    with pytest.raises(SerratedSourceDomainError) as e:
        model.evaluate_thermal(b,r.m_dot,PROPS)
    assert e.value.component == "ETA"
    assert e.value.diagnostics["HT_DOMAIN"] == e.value.diagnostics["DP_DOMAIN"] == "PASS"
    assert e.value.diagnostics["ETA_DOMAIN"] == "FAIL"


@pytest.mark.parametrize("change",["diameter_ratio","serration_ratio","conduction_parameter"])
def test_real_geometry_can_pass_holfeld_and_fail_hashizume(change):
    b = source_bank()
    if change == "diameter_ratio":
        t = replace(b.tube,D_fin=.0565,serration_depth=.007,segment_width_at_tip=None)
    elif change == "serration_ratio":
        t = replace(b.tube,serration_depth=.0162,segment_width_at_tip=None)
    else:
        t = replace(b.tube,fin_k=1.)
    b = replace(b,tube=t)
    r = request(bank=b)
    HT.evaluate(r)
    DP.evaluate(r)
    with pytest.raises(SerratedSourceDomainError,match="ETA_DOMAIN_FAIL"):
        Holfeld2016SerratedStaggeredModel(WS).evaluate_thermal(b,r.m_dot,r.props)


def test_source_profile_requires_explicit_width_and_distinct_request_type():
    with pytest.raises(TypeError,match="source_segment_width"):
        Holfeld2016SerratedStaggeredModel()
    for provider in (HT,DP):
        with pytest.raises(TypeError,match="SerratedTubeBankRequest"):
            provider.evaluate(object())
    with pytest.raises(SerratedSourceDomainError,match="dry air"):
        replace(request(),fluid_family="water")
    with pytest.raises(SerratedSourceDomainError,match="Eq.5-9"):
        replace(request(),minimum_free_flow_area=.05)


def test_network_uses_native_areas_and_eta_exactly_once():
    b = source_bank()
    model = Holfeld2016SerratedStaggeredModel(WS)
    r = request()
    thermal = evaluate_outside_thermal(bundle=b,m_dot=r.m_dot,props=PROPS,airside_model=model)
    network = calculate_resistance_network(bundle=b,alpha_inside=1200.,
        outside_alpha_physical=thermal.alpha_physical,resistance_core_wall=.001,
        airside_model=model,outside_thermal_result=thermal)
    g = model.source_geometry(b)
    eta = thermal.serrated_fin_efficiency_result.fin_efficiency
    primary = g.primary_area_per_length*b.tube.length_effective*b.n_tubes_total
    fin = g.fin_area_per_length*b.tube.length_effective*b.n_tubes_total
    conductance = thermal.alpha_physical*(primary+eta*fin)
    assert network.resistance_outside == pytest.approx(1/conductance)
    assert network.conductance_fin_outside == pytest.approx(thermal.alpha_physical*eta*fin)
    assert network.area_outside_gross == g.total_area
    assert network.area_outside_geometric == b.total_outer_geometric_area
    assert network.area_outside_gross != network.area_outside_geometric
    assert network.area_outside_effective == pytest.approx(primary+eta*fin)
    assert network.UA == pytest.approx(1/(1/(1200*b.total_inner_area)+.001+1/conductance))
    with pytest.raises(NotImplementedError,match="temperature field"):
        fin_surface_temperatures(network=network,outside_wall_temperature=330.,
                                 outside_bulk_temperature=400.,heat_rate=-100.)


@pytest.mark.parametrize("rho_out",[.9,1.2,1.5])
def test_native_hydraulics_mean_density_and_separate_signed_acceleration(rho_out):
    r = request()
    model = Holfeld2016SerratedStaggeredModel(WS)
    out = replace(PROPS,rho=rho_out)
    result = evaluate_outside_hydraulics(bundle=r.bundle,m_dot=r.m_dot,airside_model=model,
        inlet_props=PROPS,midpoint_props=replace(PROPS,rho=9.),outlet_props=out)
    rho_mean = (PROPS.rho+rho_out)/2
    direct = DP.evaluate(replace(r,props=replace(PROPS,rho=rho_mean)))
    acceleration = (r.m_dot/r.bundle.frontal_flow_area)**2*(1/rho_out-1/PROPS.rho)
    assert result.dp_drag == direct.dp_drag
    assert result.dp_acceleration == pytest.approx(acceleration)
    assert result.dp_total == pytest.approx(direct.dp_drag+acceleration)
    assert math.isnan(result.midpoint.f) and math.isnan(result.mean_f_over_rho)
    assert result.midpoint.eu == direct.eu


def test_temperature_dependent_hydraulic_properties_use_arithmetic_temperature():
    from core.properties.dry_air import DryAirPropertyProvider
    r = request()
    model = Holfeld2016SerratedStaggeredModel(WS)
    provider = DryAirPropertyProvider(prefer_coolprop=False)
    result = evaluate_outside_hydraulics(bundle=r.bundle,m_dot=r.m_dot,airside_model=model,
        property_provider=provider,temperature_in=450.,temperature_out=350.,pressure=101325.)
    assert result.midpoint.temperature == 400.
    assert result.midpoint.props.mu == provider.at(400.,101325.).mu
    density_mean = (provider.at(450.,101325.).rho+provider.at(350.,101325.).rho)/2
    assert result.native_result.reference_velocity == pytest.approx(r.m_dot/(density_mean*r.minimum_free_flow_area))
    assert result.dp_acceleration < 0  # cooling gas decelerates


@pytest.mark.parametrize("api",["thermal_state","rating","simulation"])
@pytest.mark.parametrize("mode",[PhaseChangeMode.DISABLED,PhaseChangeMode.AUTO])
@pytest.mark.parametrize("explicit_lattice", [False, True])
def test_explicit_complete_profile_executes_public_workflows(api,mode,explicit_lattice):
    b = source_bank()
    if explicit_lattice:
        b = replace(b, layout="inline", lattice=TubeBankLattice("conventional_staggered"))
    model = Holfeld2016SerratedStaggeredModel(WS)
    hx = BareTubeHeatExchanger(b,airside_model=model)
    inside = ConstantPropertyProvider(FluidTransportProperties(900.,.0005,.6,4000.))
    outside = ConstantPropertyProvider(PROPS)
    mass = request().m_dot
    if api == "thermal_state":
        result = hx.solve_thermal_state(m_dot_inside=1.5,m_dot_outside=mass,
            inside_provider=inside,outside_provider=outside,T_in_inside=350.,T_in_outside=450.,
            p_inside=101325.,p_outside=101325.)
        state = result
    else:
        cls = BalanceSideSpec if api == "rating" else HXSideInput
        i = cls(provider=inside,m_dot=1.5,T_in=350.,p=101325.,phase_change_mode=mode,
                **({"T_out":355.} if api == "rating" else {}))
        o = cls(provider=outside,m_dot=mass,T_in=450.,p=101325.,phase_change_mode=mode)
        result = getattr(hx,"rate" if api == "rating" else "simulate")(i,o)
        state = result.thermal_state
        assert result.outside_tube_bank_hydraulic.metadata.method == DP.provider_name
        assert math.isnan(result.finned_tube_diagnostics.f)
        assert result.finned_tube_diagnostics.eu == result.outside_tube_bank_hydraulic.native_result.eu
        assert result.final_result.A_o == pytest.approx(model.source_geometry(b).total_area)
    assert state.converged
    diagnostics = state.finned_tube_diagnostics
    assert diagnostics.heat_transfer_result.method == HT.provider_name
    assert diagnostics.heat_transfer_result.diagnostics["complete_domain_status"] == "INTERSECTION_PASS"
    assert diagnostics.fin_efficiency_result.diagnostics["ETA_DOMAIN"] == "PASS"
    assert diagnostics.fin_tip_temperature_ratio is None


@pytest.mark.parametrize("field",["flow_area_provider","fin_efficiency_provider","heat_transfer_provider","pressure_drop_provider"])
def test_incomplete_or_solid_component_sets_reject(field):
    from core.geometry import SerratedFinPhysicsNotSupportedError
    for value in (None,"auto","briggs_young_1963"):
        with pytest.raises(SerratedFinPhysicsNotSupportedError,match=field):
            Holfeld2016SerratedStaggeredModel(WS,**{field:value})


@pytest.mark.parametrize("layout,rows",[("inline",8),("staggered",1),("staggered",2),("staggered",3)])
def test_explicit_profile_rejects_layout_and_shallow_rows(layout,rows):
    pattern = "layout" if layout == "inline" else "PROFILE_DOMAIN_FAIL"
    with pytest.raises(ValueError,match=pattern):
        BareTubeHeatExchanger(source_bank(layout=layout,n_rows=rows,
            **({"lattice":TubeBankLattice("inline")} if layout == "inline" else {})),
            airside_model=Holfeld2016SerratedStaggeredModel(WS))


@pytest.mark.parametrize("selector",["briggs_young_1963","frass_escoa_staggered_solid_fin","genic2006_staggered"])
def test_solid_overrides_do_not_replace_explicit_profile(selector):
    r = request()
    with pytest.raises(IncompatibleOutsideCorrelationProviderError,match="solid/bare"):
        evaluate_outside_thermal(bundle=r.bundle,m_dot=r.m_dot,props=PROPS,
            airside_model=Holfeld2016SerratedStaggeredModel(WS),finned_heat_transfer_provider=selector)


def test_fouling_and_phase_change_provider_rejected_before_wet_execution():
    from core.properties.water import IAPWS97WaterSteamProvider
    b = source_bank()
    hx = BareTubeHeatExchanger(b,airside_model=Holfeld2016SerratedStaggeredModel(WS),
                              fouling_resistance_outside=.0001)
    outside = ConstantPropertyProvider(PROPS)
    with pytest.raises(SerratedSourceDomainError,match="fouling"):
        hx.solve_thermal_state(m_dot_inside=1.5,m_dot_outside=request().m_dot,inside_provider=outside,
            outside_provider=outside,T_in_inside=350.,T_in_outside=450.,p_inside=101325.,p_outside=101325.)
    model = Holfeld2016SerratedStaggeredModel(WS)
    with pytest.raises(SerratedSourceDomainError,match="dry single-phase"):
        model.guard_property_provider(IAPWS97WaterSteamProvider(),"outside")


@pytest.mark.parametrize("mode",[PhaseChangeMode.DISABLED,PhaseChangeMode.AUTO])
def test_wet_gas_is_rejected_even_if_condensation_is_disabled(mode):
    from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
    wet = GasMixturePropertyProvider(GasMixtureSpec({"Nitrogen":.95,"Water":.05},basis="mole"))
    inside = ConstantPropertyProvider(PROPS)
    i = HXSideInput(provider=inside,m_dot=1.,T_in=350.,p=101325.,phase_change_mode=mode)
    o = HXSideInput(provider=wet,m_dot=request().m_dot,T_in=450.,p=101325.,phase_change_mode=mode)
    hx = BareTubeHeatExchanger(source_bank(),airside_model=Holfeld2016SerratedStaggeredModel(WS))
    with pytest.raises(SerratedSourceDomainError,match="dry single-phase"):
        hx.simulate(i,o)
