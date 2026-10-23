# KalKalori - GNU GPL v3 only
"""Public synthetic equation/convention anchors and coupled solver checks."""
from dataclasses import replace
import math

import pytest

from core.enhancements import (
    ChakrounAlFahed1995TwistedTapeProvider, EnhancementInput, EnhancementState,
    EnhancementUnsupportedError, TubeSideEnhancement, TwistedTapeGeometry,
    evaluate_enhancement,
)
from core.enhancements.integration import evaluate_for_bundle
from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.tabulated_liquid import LiquidPropertyPoint, TabulatedLiquidProvider


SOURCE = TwistedTapeGeometry(.014*3.6, .0126, .0005)
TARGET = TwistedTapeGeometry(.076, .019, .0009)


def configured(geometry=SOURCE, strict=False):
    return TubeSideEnhancement(ChakrounAlFahed1995TwistedTapeProvider(strict), geometry, "liquid")


def sample(re=1000, diameter=.014):
    bulk = EnhancementState(870, .02, .13, 2200, 310, 2e5)
    return EnhancementInput(re*bulk.mu*math.pi*diameter/4, diameter, 2, 3, bulk,
                            wall=replace(bulk, mu=.01, temperature=330),
                            fluid_phase="liquid", heat_flow_direction="heating")


def diagnostics(result):
    return {d.name: d.value for d in result.diagnostics}


def codes(result):
    return {w.code for w in result.warnings}


def test_source_geometry_and_equation_chain():
    state = sample()
    result = evaluate_enhancement(configured(strict=True), state)
    d = diagnostics(result)
    area = math.pi*.014**2/4
    ac = area - .0005*.0126
    s = math.sqrt(1 + (math.pi/7.2)**2)
    va = state.mass_flow_per_tube/(870*ac)
    re_sw = 870*va*s*.014/.02
    sw = re_sw/math.sqrt(3.6)
    nu = .318*sw**.569*(2200*.02/.13)**.3*2**.14
    f = 15.767*((math.pi+2-2*.0005/.014)/(math.pi-4*.0005/.014))**2
    f *= (1+1e-6*sw**2.55)**(1/6)/re_sw
    assert d["Y"] == pytest.approx(3.6)
    assert d["Ac"] == pytest.approx(ac, rel=1e-14)
    assert d["Va"]/d["V0"] == pytest.approx(area/ac)
    assert d["phi"] == pytest.approx(area/ac)
    assert d["helical_factor"] == pytest.approx(s)
    assert d["Vs"] == pytest.approx(va*s)
    assert d["Re_sw"] == pytest.approx(re_sw)
    assert d["Sw_actual_area"] == pytest.approx(sw)
    assert d["Nu"] == result.nusselt == pytest.approx(nu)
    assert result.alpha_inside == pytest.approx(nu*.13/.014)
    assert d["f_Fanning_swirl"] == pytest.approx(f)
    assert d["f_Darcy_swirl"] == 4*d["f_Fanning_swirl"]
    assert result.friction_factor_native == d["f_Fanning_swirl"]
    assert result.friction_basis == "fanning"
    assert result.thermal_property_reference == "bulk"
    assert result.applicability == "within_range"


def test_public_target_geometry_and_thickness_limitation():
    result = evaluate_enhancement(configured(TARGET), sample(diameter=.021))
    d = diagnostics(result)
    assert d["Y"] == pytest.approx(3.619047619)
    assert d["W_over_D"] == pytest.approx(.9047619048)
    assert d["clearance_over_D"] == pytest.approx(.04761904762)
    assert d["delta_over_D"] == pytest.approx(.04285714286)
    assert d["clearance"] == pytest.approx(.001)
    assert "chakroun1995_thickness_ratio_extrapolation" in codes(result)
    assert "chakroun1995_width_ratio_extrapolation" not in codes(result)
    assert result.applicability == "extrapolated"
    with pytest.raises(EnhancementUnsupportedError, match="thickness_ratio"):
        evaluate_enhancement(configured(TARGET, strict=True), sample(diameter=.021))


def test_equation16_audited_fanning_darcy_anchor():
    # Synthetic Sw=1000, not an experimental datum or plot digitization.
    source = evaluate_enhancement(configured(), sample())
    re = 1000*1000/diagnostics(source)["Sw_actual_area"]
    result = evaluate_enhancement(configured(), sample(re=re))
    d = diagnostics(result)
    assert d["Sw_actual_area"] == pytest.approx(1000)
    assert d["f_Fanning_swirl"]*d["Re_sw"] == pytest.approx(85.2156457, rel=1e-9)
    assert d["f_Darcy_swirl"]*d["Re_sw"] == pytest.approx(340.862583, rel=2e-9)
    assert d["f_Darcy_swirl"] == 4*d["f_Fanning_swirl"]


@pytest.mark.parametrize("geometry,diameter", [(SOURCE, .014), (TARGET, .021)])
@pytest.mark.parametrize("length", [.25, 3, 24])
def test_pressure_drop_helical_and_empty_axial_bases_match(geometry, diameter, length):
    state = sample(diameter=diameter)
    result = evaluate_enhancement(configured(geometry), state)
    d = diagnostics(result)
    helical_dp = d["f_Darcy_swirl"]*(length*d["helical_factor"]/diameter)*state.bulk.rho*d["Vs"]**2/2
    axial_dp = result.pressure_gradient(state.bulk.rho)*length
    assert axial_dp == pytest.approx(helical_dp, rel=1e-13)
    assert result.reference.flow_area == pytest.approx(math.pi*diameter**2/4)
    assert result.reference.hydraulic_diameter == result.reference.friction_diameter == diameter
    assert result.f_darcy == d["f_Darcy_axial_equivalent"]
    assert result.f_darcy == pytest.approx(4*result.friction_factor_native*result.friction_normalization)


def test_actual_width_predicts_and_nominal_eq11_is_only_diagnostic():
    narrow = evaluate_enhancement(configured(), sample())
    wide = evaluate_enhancement(configured(replace(SOURCE, tape_width=.0132)), sample())
    a, b = diagnostics(narrow), diagnostics(wide)
    assert a["Sw_printed_nominal"] == b["Sw_printed_nominal"]
    assert a["Sw_actual_area"] < b["Sw_actual_area"] < b["Sw_printed_nominal"]
    assert narrow.nusselt < wide.nusselt
    assert wide.nusselt/narrow.nusselt == pytest.approx((a["Ac"]/b["Ac"])**.569)


def test_full_width_limit_coincides_with_printed_eq11():
    result = evaluate_enhancement(configured(replace(SOURCE, tape_width=.014)), sample())
    d = diagnostics(result)
    assert d["Sw_actual_area"] == pytest.approx(d["Sw_printed_nominal"], rel=1e-14)
    assert d["clearance"] == 0
    assert "chakroun1995_width_ratio_extrapolation" in codes(result)


def test_bulk_basis_wall_viscosity_and_cooling_extrapolation():
    state = sample()
    base = evaluate_enhancement(configured(), replace(state, wall=state.bulk))
    heating = evaluate_enhancement(configured(), state)
    cooling_state = replace(state, wall=replace(state.wall, mu=.04, temperature=290),
                            heat_flow_direction="cooling")
    cooling = evaluate_enhancement(configured(), cooling_state)
    assert cooling.nusselt < base.nusselt < heating.nusselt
    assert cooling.nusselt/base.nusselt == pytest.approx(.5**.14)
    assert cooling.wall_correction == pytest.approx(.5**.14)
    assert cooling.f_darcy == base.f_darcy == heating.f_darcy
    assert "chakroun1995_cooling_extrapolation" in codes(cooling)
    assert cooling.applicability == "extrapolated"
    with pytest.raises(EnhancementUnsupportedError, match="cooling_extrapolation"):
        evaluate_enhancement(configured(strict=True), cooling_state)
    # Temperature evidence detects cooling even when direction is unspecified.
    assert "chakroun1995_cooling_extrapolation" in codes(evaluate_enhancement(
        configured(), replace(cooling_state, heat_flow_direction="unknown")))
    alien_film = EnhancementState(1, .9, 5, 20, 320, 2e5)
    assert evaluate_enhancement(configured(), replace(state, thermal=alien_film)).nusselt == heating.nusselt


@pytest.mark.parametrize("re", [1, 229, 230, 1000, 2300])
def test_re_applicability_and_strict_mode(re):
    result = evaluate_enhancement(configured(), sample(re))
    assert ("chakroun1995_re_below_source_range" in codes(result)) == (re < 230)
    if re < 230:
        with pytest.raises(EnhancementUnsupportedError, match="re_below_source_range"):
            evaluate_enhancement(configured(strict=True), sample(re))
    else:
        assert evaluate_enhancement(configured(strict=True), sample(re)).applicability == "within_range"


@pytest.mark.parametrize("re", [2300.01, 3000, 5000])
@pytest.mark.parametrize("strict", [False, True])
def test_transition_and_turbulence_always_rejected(re, strict):
    with pytest.raises(EnhancementUnsupportedError, match="regime_unsupported"):
        evaluate_enhancement(configured(strict=strict), sample(re))


@pytest.mark.parametrize("field,value,code", [
    ("half_turn_length", .014*3, "twist_ratio_extrapolation"),
    ("half_turn_length", .014*8, "twist_ratio_extrapolation"),
    ("tape_width", .010, "width_ratio_extrapolation"),
    ("tape_width", .0138, "width_ratio_extrapolation"),
    ("tape_thickness", .0006, "thickness_ratio_extrapolation"),
])
def test_geometry_extrapolations_and_strict_mode(field, value, code):
    geometry = replace(SOURCE, **{field: value})
    assert "chakroun1995_"+code in codes(evaluate_enhancement(configured(geometry), sample()))
    with pytest.raises(EnhancementUnsupportedError, match=code):
        evaluate_enhancement(configured(geometry, strict=True), sample())


@pytest.mark.parametrize("y,width", [(3.6, .0108), (7.1, .0132)])
def test_source_geometry_endpoints(y, width):
    geometry = replace(SOURCE, half_turn_length=y*.014, tape_width=width)
    assert evaluate_enhancement(configured(geometry, strict=True), sample()).applicability == "within_range"


def test_required_wall_phase_geometry_and_diagnostic_provenance():
    with pytest.raises(EnhancementUnsupportedError, match="wall_state_required"):
        evaluate_enhancement(configured(), replace(sample(), wall=None))
    with pytest.raises(EnhancementUnsupportedError, match="phase_unsupported"):
        ChakrounAlFahed1995TwistedTapeProvider().evaluate(SOURCE, replace(sample(), fluid_phase="gas"))
    with pytest.raises(EnhancementUnsupportedError, match="geometry_unsupported"):
        ChakrounAlFahed1995TwistedTapeProvider().evaluate(object(), sample())
    with pytest.raises(ValueError, match="width"):
        evaluate_enhancement(configured(replace(SOURCE, tape_width=.015)), sample())
    with pytest.raises(EnhancementUnsupportedError, match="blockage"):
        evaluate_enhancement(configured(TwistedTapeGeometry(.05, .013, .012)), sample())
    with pytest.raises(TypeError, match="bool"):
        ChakrounAlFahed1995TwistedTapeProvider("yes")
    result = evaluate_enhancement(configured(), sample())
    assert result.source_references == ("https://doi.org/10.1115/95-GT-042",)
    assert result.source_access_basis == "open"
    assert diagnostics(result)["provenance"].startswith("Chakroun & Al-Fahed 1995")
    assert diagnostics(result)["source_pr_range"] == "unknown"
    assert {"chakroun1995_source_width_interpretation", "chakroun1995_fanning_to_darcy_conversion",
            "chakroun1995_constant_wall_temperature_source"} <= codes(result)
    assert all(w.severity == "info" for w in result.warnings)


@pytest.mark.parametrize("cp", [20, 2e6])
def test_pr_limits_are_not_invented(cp):
    state = sample()
    result = evaluate_enhancement(configured(strict=True), replace(state, bulk=replace(state.bulk, cp=cp)))
    assert diagnostics(result)["Pr_bulk"] == cp*state.bulk.mu/state.bulk.k
    assert diagnostics(result)["source_pr_range"] == "unknown"


def synthetic_case(cooling=False):
    # Invented public transport states; no named oil or private property data.
    liquid = TabulatedLiquidProvider("Synthetic high-Pr liquid", (
        LiquidPropertyPoint(T=270, rho=900, mu=.04, k=.13, cp=2200),
        LiquidPropertyPoint(T=380, rho=850, mu=.004, k=.15, cp=2400),
    ))
    tube = BareTube(D_i=.021, D_o=.024, length_total=2, length_effective=2, wall_k=45)
    bundle = TubeBundle(tube=tube, n_rows=4, n_tubes_per_row=8,
                        pitch_transverse=.05, pitch_longitudinal=.045,
                        layout="staggered", n_passes_tube=1, flow_arrangement="counterflow")
    inlet = 340 if cooling else 300
    flow = 500*liquid.at(inlet, 2e5).mu*math.pi*.021/4*bundle.n_tubes_per_pass_effective
    inside = HXSideInput(provider=liquid, m_dot=flow, T_in=inlet, p=2e5)
    outside = HXSideInput(provider=ConstantPropertyProvider(
        FluidTransportProperties(1.1, 2e-5, .03, 1000)), m_dot=.8,
        T_in=300 if cooling else 340, p=1e5)
    return bundle, inside, outside


@pytest.mark.parametrize("cooling", [False, True])
@pytest.mark.parametrize("iterate", [False, True])
def test_simulation_rating_and_coupled_wall_viscosity(cooling, iterate):
    bundle, inside, outside = synthetic_case(cooling)
    hx = BareTubeHeatExchanger(bundle, tube_side_enhancement=configured(TARGET))
    simulation = hx.simulate(inside, outside, iterate=iterate)
    assert simulation.converged
    assert (simulation.T_out_inside < inside.T_in) == cooling
    rating = hx.rate(
        BalanceSideSpec(provider=inside.provider, p=inside.p, m_dot=inside.m_dot,
                        T_in=inside.T_in, T_out=simulation.T_out_inside),
        BalanceSideSpec(provider=outside.provider, p=outside.p, m_dot=outside.m_dot, T_in=outside.T_in),
        include_simulation=True)
    assert rating.simulation.converged
    for result in (simulation, rating, rating.simulation):
        e = result.tube_side_enhancement
        d = diagnostics(e)
        assert e.provider_id == configured().provider.provider_id
        assert math.isfinite(e.alpha_inside) and e.alpha_inside > 0
        assert math.isfinite(result.inside_dp_friction) and result.inside_dp_friction > 0
        assert "chakroun1995_thickness_ratio_extrapolation" in codes(result)
        assert ("chakroun1995_cooling_extrapolation" in codes(result)) == cooling
        assert d["mu_wall"] == pytest.approx(inside.provider.at(d["T_wall"], inside.p).mu)
        assert d["mu_bulk"] == pytest.approx(inside.provider.at(d["T_bulk"], inside.p).mu)
        assert (d["mu_bulk_over_wall"] < 1) == cooling
        assert e.wall_correction == pytest.approx(d["mu_bulk_over_wall"]**.14)
        assert e.wall_correction != 1
    points = (simulation.inside_properties_inlet, simulation.inside_properties_midpoint,
              simulation.inside_properties_outlet)
    for p in points:
        assert p.enhancement.alpha_inside is None and p.enhancement.nusselt is None
        assert p.enhancement.provider_id == configured().provider.provider_id
    gradients = [p.enhancement.pressure_gradient(p.rho) for p in points]
    assert simulation.inside_dp_friction == pytest.approx(2*(gradients[0]+4*gradients[1]+gradients[2])/6)


def test_adapter_requires_wall_backend_and_does_not_add_thermal_correction():
    bundle, inside, outside = synthetic_case()
    bulk = inside.provider.at(300, inside.p)
    e = evaluate_for_bundle(configured(TARGET), bundle, inside.m_dot, bulk,
                            temperature=300, pressure=inside.p, wall_temperature=320,
                            property_provider=inside.provider, heat_flow_direction="heating")
    d = diagnostics(e)
    assert e.alpha_inside == pytest.approx(.318*d["Sw_actual_area"]**.569
        *d["Pr_bulk"]**.3*(bulk.mu/inside.provider.at(320, inside.p).mu)**.14*bulk.k/.021)
    with pytest.raises(EnhancementUnsupportedError, match="wall_state_required"):
        evaluate_for_bundle(configured(TARGET), bundle, inside.m_dot, bulk,
                            temperature=300, pressure=inside.p)
