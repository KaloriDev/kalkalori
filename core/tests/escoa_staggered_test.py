# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Independent equation fixtures and explicit ESCOA staggered integration.

Synthetic inputs verify mathematics and contracts, not source-domain closure
or experimental accuracy. No private ECO values are stored here.
"""
from dataclasses import replace
import math

import pytest

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.heat_transfer import (
    FrassEscoaInlineSolidFinProvider, FrassEscoaStaggeredSolidFinProvider,
    FRASS_ESCOA_STAGGERED_METADATA,
)
from core.heat_transfer.fin_efficiency import annular_fin_efficiency
from core.heat_transfer.finned_tube_outside import (
    FinnedTubeHeatTransferRequest, calculate_finned_tube_outside_heat_transfer,
    evaluate_finned_tube_heat_transfer,
)
from core.heat_transfer.outside_dispatch import (
    calculate_resistance_network, evaluate_outside_thermal,
    evaluate_outside_hydraulics, make_mean_fin_temperature_model,
)
from core.models.bare_tube import BareTubeHeatExchanger
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


SELECTOR = "frass_escoa_staggered_solid_fin"
PROPS = FluidTransportProperties(rho=1., mu=2e-5, k=.03, cp=1050.)


def request(**changes):
    values = dict(
        m_dot=20000*2e-5/.038*.5, face_area=1., minimum_free_flow_area=.5,
        rho=1., mu=2e-5, k=.03, cp=1050., D_root=.038, D_fin=.070,
        fin_pitch=1/150, fin_thickness_root=.001, fin_thickness_tip=.001,
        pitch_transverse=.08, pitch_longitudinal=.08, layout="staggered", n_rows=8,
        gas_temperature=600., mean_fin_temperature=450., tube_outer_diameter=.038,
    )
    return FinnedTubeHeatTransferRequest(**{**values, **changes})


def bundle(**changes):
    tube = CircularFinnedTube(
        core_tube=BareTube(D_i=.032, D_o=.038, length_total=1.,
                           length_effective=1., wall_k=45.),
        D_root=.038, D_fin=.070, fin_k=45., fin_thickness_root=.001,
        fin_pitch=1/150,
    )
    values = dict(tube=tube, n_rows=6, n_tubes_per_row=3, pitch_transverse=.08,
                  pitch_longitudinal=math.sqrt(3)/2*.08, layout="staggered",
                  n_passes_tube=1, flow_arrangement="counterflow")
    return TubeBundle(**{**values, **changes})


# Calculated independently using 65-digit Decimal ln/exp arithmetic. Each row:
# n_rows, Pl/Pt, H, Re, C3, C5, Nu, h. Fixed Pr=.7, Tg/Tfin=600/450,
# dA=.038, Dfin=dA+2H, Fp=1/150, tf=.001, k=.03.
FIXTURES = [
    (1,.75,.008,3000,.8067020397309471,.705400859126231,29.452607014644354,23.252058169456067),
    (1,1.,.016,10000,.6708873124529346,.704206193316767,60.89019247598042,48.07120458630033),
    (1,1.25,.024,24000,.57546136941698,.7032757866488479,102.1356320086758,80.63339369105984),
    (2,.75,.008,10000,.8067020397309471,.8232643784019971,75.17929820262373,59.35207752838715),
    (2,1.,.016,24000,.6708873124529346,.7959983944242853,121.58976118654476,95.99191672621954),
    (2,1.25,.024,3000,.57546136941698,.774763624751231,29.12140647531571,22.99058405945977),
    (4,.75,.008,24000,.8067020397309471,.9963748854250781,160.73817093897998,126.89855600445787),
    (4,1.,.016,3000,.6708873124529346,.9308169928517488,36.799390472114965,29.05215037272234),
    (4,1.25,.024,10000,.57546136941698,.8797604547791288,72.3233850607488,57.097409258485904),
    (8,.75,.008,3000,.8067020397309471,1.0306309926868882,43.03194306046025,33.972586626679146),
    (8,1.,.016,10000,.6708873124529346,.9574956760122244,82.7912287063312,65.36149634710357),
    (8,1.25,.024,24000,.57546136941698,.9005378341158211,130.78368768164117,103.25027974866408),
    (40,.75,.008,10000,.8067020397309471,1.0306565869187103,94.11805116949311,74.30372460749456),
    (40,1.,.016,24000,.6708873124529346,.9575156088200096,146.26171990336613,115.46977887107853),
    (40,1.25,.024,3000,.57546136941698,.9005533578021331,33.84952461299026,26.723308904992308),
]


@pytest.mark.parametrize("rows,pitch_ratio,H,Re,C3,C5,Nu,h", FIXTURES)
def test_independent_decimal_equation_fixtures(rows, pitch_ratio, H, Re, C3, C5, Nu, h):
    result = FrassEscoaStaggeredSolidFinProvider().evaluate(request(
        n_rows=rows, pitch_longitudinal=.08*pitch_ratio,
        D_fin=.038+2*H, m_dot=Re*2e-5/.038*.5))
    assert result.diagnostics["C3"] == pytest.approx(C3, rel=2e-14)
    assert result.diagnostics["C5"] == pytest.approx(C5, rel=2e-14)
    assert result.Nu == pytest.approx(Nu, rel=2e-14)
    assert result.h == pytest.approx(h, rel=2e-14)
    assert result.Re == pytest.approx(Re)
    assert result.Pr == pytest.approx(.7)


def test_squared_row_term_and_staggered_row_trend():
    provider = FrassEscoaStaggeredSolidFinProvider()
    r = provider.evaluate(request(n_rows=4))
    assert r.diagnostics["C5"] == pytest.approx(.9308169928517488, rel=2e-14)
    wrong_unsquared = .7+(.7-.8*math.exp(-.15*4))*math.exp(-1.)
    assert abs(r.diagnostics["C5"]-wrong_unsquared) > .1
    values = [provider.evaluate(request(n_rows=n)).h for n in (1,2,4,8,40)]
    assert values == sorted(values)


def test_metadata_uncertainty_and_single_row_are_explicit():
    r = FrassEscoaStaggeredSolidFinProvider().evaluate(request(n_rows=1))
    assert r.metadata is FRASS_ESCOA_STAGGERED_METADATA
    assert "secondary" in r.source and "not inspected" in r.source
    assert "(60)-(62)" in r.source and r.metadata.supported_layouts == ("staggered",)
    assert r.applicability == ()
    assert r.diagnostics["source_range_status"] == "SOURCE_RANGE_UNRESOLVED"
    assert r.diagnostics["property_convention_provenance"] == "ENGINEERING_CONVENTION"
    assert r.diagnostics["source_property_convention_closed"] is False
    assert "escoa_single_row_caveat" in {w.code for w in r.warnings}
    with pytest.raises(ValueError, match="SOURCE_RANGE_UNRESOLVED"):
        FrassEscoaStaggeredSolidFinProvider().evaluate(request(strict_source_domain=True))


@pytest.mark.parametrize("layout", ["inline", "shifted", "arbitrary_offset"])
def test_other_layouts_rejected(layout):
    with pytest.raises(ValueError, match="STAGGERED|layout"):
        FrassEscoaStaggeredSolidFinProvider().evaluate(request(layout=layout))


@pytest.mark.parametrize("family", ["serrated", "segmented", "louvered", "elliptical", "plain"])
def test_other_surface_families_rejected(family):
    with pytest.raises(ValueError, match="solid individual"):
        FrassEscoaStaggeredSolidFinProvider().evaluate(request(geometry_family=family))


@pytest.mark.parametrize("changes,pattern", [
    ({"tube_outer_diameter":.040}, "must not exceed"),
    ({"pitch_transverse":.04}, "blockage"),
    ({"n_rows":0}, "n_rows"),
    ({"mean_fin_temperature":None}, "mean_fin_temperature"),
    ({"gas_temperature":None}, "gas_temperature"),
    ({"mean_fin_temperature":0.}, "mean_fin_temperature"),
])
def test_invalid_requests_rejected(changes, pattern):
    with pytest.raises(ValueError, match=pattern):
        FrassEscoaStaggeredSolidFinProvider().evaluate(request(**changes))


def test_supplied_properties_and_temperature_ratio_without_efficiency():
    p = FrassEscoaStaggeredSolidFinProvider()
    r = p.evaluate(request())
    density = p.evaluate(request(rho=2.))
    assert density.Re == r.Re and density.h == r.h
    assert density.reference_velocity == r.reference_velocity/2
    assert p.evaluate(request(gas_temperature=1200.)).h/r.h == pytest.approx(2**.25)
    assert r.h == pytest.approx(r.Nu*request().k/request().D_root)
    assert r.j == pytest.approx(r.Nu/(r.Re*r.Pr**(1/3)))


def test_same_family_representative_case_has_higher_staggered_film_htc():
    s = FrassEscoaStaggeredSolidFinProvider().evaluate(request())
    i = FrassEscoaInlineSolidFinProvider().evaluate(request(layout="inline"))
    assert s.h > i.h  # This case only; no universal layout percentage imposed.


def test_root_and_taper_conventions_preserve_actual_geometry():
    p = FrassEscoaStaggeredSolidFinProvider()
    r = p.evaluate(request(tube_outer_diameter=.034, fin_thickness_tip=.0005))
    d = r.diagnostics
    assert d["metal_tube_od"] == .034 and d["D_root"] == .038 and d["D_fin"] == .070
    assert d["diameter_mapping_provenance"] == "ENGINEERING_CONVENTION"
    assert d["thickness_mapping_provenance"] == "ENGINEERING_CONVENTION"
    assert d["root_mapping"] and d["taper_mapping"]
    assert d["source_geometry_mapping_closed"] is False
    assert d["tf"] == .00075
    assert d["fin_thickness_root"] == .001 and d["fin_thickness_tip"] == .0005
    equivalent = p.evaluate(request(tube_outer_diameter=.034,
        fin_thickness_root=.00075, fin_thickness_tip=.00075))
    assert r.h == equivalent.h  # Empirical term only.
    bank = bundle()
    tube = replace(bank.tube,core_tube=replace(bank.tube.core_tube,D_o=.034),
                   fin_thickness_tip=.0005)
    actual = replace(bank,tube=tube)
    mean_profile = replace(actual,tube=replace(tube,fin_thickness_root=.00075,
                                             fin_thickness_tip=.00075))
    assert actual.tube.fin_thickness_tip_effective == .0005
    assert actual.minimum_free_flow_area == mean_profile.minimum_free_flow_area
    assert annular_fin_efficiency(tube,r.h).fin_efficiency != pytest.approx(
        annular_fin_efficiency(mean_profile.tube,r.h).fin_efficiency,rel=1e-4)
    for changes in ({"tube_outer_diameter":.034},{"fin_thickness_tip":.0005}):
        with pytest.raises(ValueError,match="SOURCE_RANGE_UNRESOLVED"):
            p.evaluate(request(strict_source_domain=True,**changes))


def test_explicit_dispatch_and_auto_keep_existing_guards():
    req = request()
    assert evaluate_finned_tube_heat_transfer(req,provider=SELECTOR).method == SELECTOR
    # Square staggered requests calculate in ESCOA, retain unresolved status;
    # their rejection by BY AUTO must remain intact.
    with pytest.raises(ValueError, match="equilateral"):
        evaluate_finned_tube_heat_transfer(req,provider="auto")
    tri = replace(req,pitch_longitudinal=math.sqrt(3)/2*req.pitch_transverse)
    assert evaluate_finned_tube_heat_transfer(tri,provider="auto").method == "briggs_young_1963"
    assert evaluate_finned_tube_heat_transfer(replace(req,layout="inline"),provider="auto").method == "frass_escoa_inline_solid_fin"
    with pytest.raises(ValueError, match="STAGGERED"):
        evaluate_finned_tube_heat_transfer(replace(req,layout="inline"),provider=SELECTOR)


def test_physical_mean_fin_temperature_coupling_and_diagnostics():
    bank = bundle()
    hx = BareTubeHeatExchanger(bank)
    model = make_mean_fin_temperature_model(bundle=bank,inside_bulk_temperature=400.,
        outside_bulk_temperature=600.,alpha_inside=900.,
        resistance_core_wall=hx.tube_wall_resistance())
    r = calculate_finned_tube_outside_heat_transfer(1.,bank,PROPS,
        provider=SELECTOR,gas_temperature=600.,fin_temperature_model=model)
    d = r.diagnostics
    assert d["T_fin_mean"] == pytest.approx(model(r.h),abs=1e-7)
    assert 400. < d["T_fin_mean"] < 600.
    assert d["coupling_converged"] and d["coupling_iterations"] > 1
    assert d["coupling_residual"] <= 1e-10
    assert d["coupling_residual"] == d["coupling_relative_residual"]
    expected = {"provider","source_provenance","layout","n_rows","d_A", "D_root",
        "metal_tube_od","D_fin","h","Fp","tf","Pt","Pl","A_face","A_min",
        "u_face","u_reference","Re","Pr","T_g_mean","T_fin_mean","Tg/Tfin",
        "C3","C5","Nu","h_air","property_convention","source_range_status",
        "root_mapping","taper_mapping","warnings","coupling_iterations","coupling_residual"}
    assert expected <= d.keys()
    n = calculate_resistance_network(bundle=bank,alpha_inside=900.,
        outside_alpha_physical=r.h,resistance_core_wall=hx.tube_wall_resistance())
    assert n.outside_alpha_physical == r.h
    assert n.outside_alpha_effective_gross < r.h
    assert n.UA == pytest.approx(1/n.resistance_total)


def test_coupling_failure_strict_and_surface_boundaries():
    bank = bundle();calls=[]
    def oscillating(h):
        calls.append(h)
        return 1. if len(calls)%2 else 1e9
    with pytest.raises(RuntimeError,match="did not converge"):
        calculate_finned_tube_outside_heat_transfer(1.,bank,PROPS,provider=SELECTOR,
            gas_temperature=600.,fin_temperature_model=oscillating)
    with pytest.raises(ValueError,match="SOURCE_RANGE_UNRESOLVED"):
        evaluate_outside_thermal(bundle=bank,m_dot=1.,props=PROPS,
            finned_heat_transfer_provider=SELECTOR,strict_source_domain=True)
    plain = replace(bank,tube=bank.tube.core_tube)
    with pytest.raises(TypeError,match="plain|CircularFinnedTube"):
        evaluate_outside_thermal(bundle=plain,m_dot=1.,props=PROPS,
            finned_heat_transfer_provider=SELECTOR)
    with pytest.raises(TypeError,match="heat-transfer|heat transfer|HTC"):
        evaluate_outside_hydraulics(bundle=bank,m_dot=1.,inlet_props=PROPS,
            finned_pressure_drop_provider=FrassEscoaStaggeredSolidFinProvider())


@pytest.mark.parametrize("workflow", ["thermal", "rating", "simulation"])
def test_explicit_provider_reaches_complete_workflows(workflow):
    from core.models import BalanceSideSpec
    from core.models.simulation import HXSideInput
    from core.phase_change.types import PhaseChangeMode
    hx = BareTubeHeatExchanger(bundle())
    inside = ConstantPropertyProvider(FluidTransportProperties(rho=900.,mu=.0005,k=.6,cp=4000.))
    outside = ConstantPropertyProvider(PROPS)
    kw = dict(finned_heat_transfer_provider=SELECTOR)
    if workflow == "thermal":
        state = hx.solve_thermal_state(m_dot_inside=1.5,m_dot_outside=1.,
            inside_provider=inside,outside_provider=outside,T_in_inside=400.,
            T_in_outside=600.,p_inside=101325.,p_outside=101325.,**kw)
    else:
        cls = BalanceSideSpec if workflow == "rating" else HXSideInput
        i = cls(provider=inside,m_dot=1.5,T_in=400.,p=101325.,
                phase_change_mode=PhaseChangeMode.DISABLED,
                **({"T_out":410.} if workflow == "rating" else {}))
        o = cls(provider=outside,m_dot=1.,T_in=600.,p=101325.,
                phase_change_mode=PhaseChangeMode.DISABLED)
        result = getattr(hx,"rate" if workflow == "rating" else "simulate")(i,o,**kw)
        state = result.thermal_state
        assert result.final_result.finned_tube_diagnostics.heat_transfer_result.method == SELECTOR
        assert result.outside_tube_bank_hydraulic.metadata.method == "robinson_briggs_1966"
    assert state.converged
    assert state.finned_tube_diagnostics.heat_transfer_result.method == SELECTOR
    assert state.finned_tube_diagnostics.heat_transfer_result.diagnostics["coupling_converged"]
