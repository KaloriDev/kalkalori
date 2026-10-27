# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Independent Genić arithmetic, applicability and native adapter checks."""
from dataclasses import replace
import math

import pytest

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.pressure_drop import (
    GENIC_2006_XI_COEFFICIENT_BASIS,
    FinnedTubePressureDropRequest,
    Genic2006InlinePressureDropProvider,
    Genic2006StaggeredPressureDropProvider,
    calculate_finned_tube_bank_hydraulics,
    calculate_finned_tube_pressure_drop,
    evaluate_finned_tube_pressure_drop,
)
from core.properties.common import FluidTransportProperties
from core.heat_transfer.outside_dispatch import evaluate_outside_thermal, evaluate_outside_hydraulics


DE = .0106745541842112201061920671642741196930755991596834968984098
PROPS = FluidTransportProperties(rho=1.2, mu=1.81e-5, k=.026, cp=1005.)


def request(layout="inline", Re=8000., **changes):
    values = dict(face_area=1., D_root=.026, D_fin=.053,
        fin_pitch=.0042, fin_thickness_root=.0008, fin_thickness_tip=.0008,
        pitch_transverse=.060, pitch_longitudinal=.054, n_rows=7,
        rho=1.2, mu=1.81e-5, layout=layout)
    values.update(changes)
    # Source geometry constructed independently from the public providers.
    d, D, s, t, Pt = [values[k] for k in (
        "D_root", "D_fin", "fin_pitch", "fin_thickness_root", "pitch_transverse")]
    area = values["face_area"]*((Pt-d)*(s-t)+t*(Pt-D))/(Pt*s)
    values.setdefault("minimum_free_flow_area", area)
    Pl=values["pitch_longitudinal"]
    # Select the requested Reynolds state after changing geometry, not the
    # original reference case's diameter (the assertions use fixed fixtures).
    surface=math.pi*(d*(s-t)+(D*D-d*d)/2+D*t)
    equivalent=4*(Pt*Pl*s-math.pi*(d*d*(s-t)+D*D*t)/4)/surface
    values.setdefault("m_dot", Re*PROPS.mu/equivalent*area)
    return FinnedTubePressureDropRequest(**values)


def bank(layout="inline", **changes):
    core = BareTube(D_i=.022, D_o=.026, length_total=1.5, length_effective=1.5, wall_k=45.)
    tube = CircularFinnedTube(core_tube=core, D_root=.026, D_fin=.053,
        fin_pitch=.0042, fin_thickness_root=.0008, fin_k=45.)
    values = dict(tube=tube, n_rows=7, n_tubes_per_row=4, pitch_transverse=.060,
        pitch_longitudinal=.054, layout=layout, n_passes_tube=1, flow_arrangement="counterflow")
    return TubeBundle(**{**values, **changes})


# 60-digit Decimal calculation independent of production functions.
@pytest.mark.parametrize("layout,xi,zeta,dp", [
    ("staggered", .1303820192300451242470, .6595712492458244287115, 353.9860490568428774559),
    ("inline", .06462480059328675611082, .3269213094818679573512, 175.4557719553426265098),
])
def test_audited_reference_geometry_re_xi_and_eq15(layout, xi, zeta, dp):
    result = evaluate_finned_tube_pressure_drop(request(layout), provider="genic2006_"+layout)
    diag = result.diagnostics
    for key, expected in dict(W=31.034891783358576274, epsilon=.737646539422439665,
        de=DE, Rd=2.122935306153010032, Ri=.823529411764705882,
        w_min=11.304141098945870954, Re_de=8000., xi=xi, zeta_row=zeta).items():
        assert diag[key] == pytest.approx(expected, rel=3e-14)
    assert result.bank_delta_p == pytest.approx(dp, rel=3e-14)
    assert result.dp_row == pytest.approx(dp/7, rel=3e-14)
    assert result.Re == pytest.approx(8000., rel=3e-14)
    assert result.reference_diameter_value == pytest.approx(DE, rel=3e-14)
    assert result.xi == pytest.approx(xi, rel=3e-14)
    assert math.isnan(result.f)
    assert result.coefficient_basis == GENIC_2006_XI_COEFFICIENT_BASIS
    assert diag["source_equation_closed"] is True
    assert diag["calibration_provenance"] == "PARTIAL"
    assert diag["raw_calibration_data_recovered"] is False
    assert diag["source_domain_status"] == "SOURCE_DOMAIN_PASS"


@pytest.mark.parametrize("layout", ["inline", "staggered"])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 7, 12])
def test_no_invented_short_row_correction(layout, rows):
    if rows < 5:
        with pytest.raises(ValueError, match="C_Nr"):
            evaluate_finned_tube_pressure_drop(request(layout, n_rows=rows), provider="genic2006_"+layout)
    else:
        r = evaluate_finned_tube_pressure_drop(request(layout, n_rows=rows), provider="genic2006_"+layout)
        assert r.diagnostics["row_correction"] == 1.
        expected = 353.9860490568428775 if layout == "staggered" else 175.4557719553426265
        assert r.dp_drag == pytest.approx(expected*rows/7, rel=3e-14)


@pytest.mark.parametrize("layout", ["inline", "staggered"])
@pytest.mark.parametrize("Pt,Pl", [(.060,.054), (.065,.070), (.080,.075)])
def test_live_minimum_area_is_exact_source_frontal_passage(layout, Pt, Pl):
    b = bank(layout, pitch_transverse=Pt, pitch_longitudinal=Pl)
    af = (Pt-.026)*(.0042-.0008)+.0008*(Pt-.053)
    expected = 4*1.5*af/.0042
    assert b.minimum_free_flow_area == pytest.approx(expected, rel=3e-14)
    r = calculate_finned_tube_pressure_drop(5., b, PROPS, provider="genic2006_"+layout)
    assert r.minimum_free_flow_area == pytest.approx(expected, rel=3e-14)
    assert r.reference_velocity == pytest.approx(5./(PROPS.rho*expected), rel=3e-14)


@pytest.mark.parametrize("strict", [False, True])
def test_dense_staggered_is_always_rejected(strict):
    with pytest.raises(ValueError, match="Rd>=1"):
        Genic2006StaggeredPressureDropProvider().evaluate(request("staggered",
            pitch_transverse=.09, pitch_longitudinal=.03, strict_source_domain=strict))


@pytest.mark.parametrize("pitch", ["pitch_transverse", "pitch_longitudinal"])
def test_staggered_pitch_bounds_warn_or_reject_in_strict(pitch):
    provider = Genic2006StaggeredPressureDropProvider()
    r = request("staggered", **{pitch:.092})
    result = provider.evaluate(r)
    assert result.diagnostics["source_domain_status"] == "SOURCE_DOMAIN_FAIL"
    assert "genic2006_source_domain_extrapolation" in {w.code for w in result.warnings}
    with pytest.raises(ValueError, match="3.5"):
        provider.evaluate(replace(r, strict_source_domain=True))
    assert provider.evaluate(request("staggered", **{pitch:3.5*.026}, strict_source_domain=True)).diagnostics["source_domain_status"] == "SOURCE_DOMAIN_PASS"


@pytest.mark.parametrize("Ri,passes", [(.49,False), (.5,True), (2.,True), (2.01,False)])
def test_inline_ri_bounds(Ri, passes):
    r = request(pitch_longitudinal=.026+Ri*(.060-.026))
    provider = Genic2006InlinePressureDropProvider()
    assert (provider.evaluate(r).diagnostics["source_domain_status"] == "SOURCE_DOMAIN_PASS") is passes
    if passes:
        provider.evaluate(replace(r, strict_source_domain=True))
    else:
        with pytest.raises(ValueError, match="Ri"):
            provider.evaluate(replace(r, strict_source_domain=True))


@pytest.mark.parametrize("Re,passes", [(5999.,False),(6000.,False),(6001.,True),
    (199999.,True),(200000.,False),(200001.,False)])
def test_inline_re_open_interval(Re, passes):
    r = request(Re=Re)
    result = Genic2006InlinePressureDropProvider().evaluate(r)
    assert result.Re == pytest.approx(Re)
    assert (result.diagnostics["source_domain_status"] == "SOURCE_DOMAIN_PASS") is passes
    if not passes:
        with pytest.raises(ValueError, match="Re_de"):
            Genic2006InlinePressureDropProvider().evaluate(replace(r, strict_source_domain=True))


@pytest.mark.parametrize("Re,passes", [(549.,False),(550.,False),(551.,True),
    (129999.,True),(130000.,False),(130001.,False)])
def test_staggered_re_open_interval(Re, passes):
    provider = Genic2006StaggeredPressureDropProvider()
    r = request("staggered", Re=Re)
    result = provider.evaluate(r)
    assert result.Re == pytest.approx(Re)
    assert result.diagnostics["reynolds_source_interval"] == dict(lower=550.,upper=130000.,
        lower_inclusive=False,upper_inclusive=False)
    assert (result.diagnostics["source_domain_status"] == "SOURCE_DOMAIN_PASS") is passes
    if passes:
        provider.evaluate(replace(r,strict_source_domain=True))
    else:
        assert "genic2006_source_domain_extrapolation" in {w.code for w in result.warnings}
        with pytest.raises(ValueError,match="Re_de"):
            provider.evaluate(replace(r,strict_source_domain=True))


def test_calibration_coverage_does_not_become_a_hard_source_bound():
    r = Genic2006InlinePressureDropProvider().evaluate(request(strict_source_domain=True))
    d = r.diagnostics
    assert d["calibration_geometry_status"] == "CALIBRATION_GEOMETRY_EXTRAPOLATION"
    assert d["calibration_source"] == "Yudin 1982 via Genić"
    assert d["reynolds_source_interval"] == dict(lower=6000.,upper=200000.,
        lower_inclusive=False,upper_inclusive=False)
    assert (d["calibration_bundles"], d["calibration_regimes"]) == (6, 91)
    assert "genic2006_inline_calibration_geometry_extrapolation" in {w.code for w in r.warnings}
    known = request(D_root=.032,D_fin=.050,fin_pitch=.006,fin_thickness_root=.0013,
        fin_thickness_tip=.0013)
    known_result = Genic2006InlinePressureDropProvider().evaluate(known)
    assert known_result.diagnostics["calibration_geometry_status"] == "KNOWN_CALIBRATION_GEOMETRY_MATCH"


@pytest.mark.parametrize("layout", ["inline", "staggered"])
def test_native_bank_adapter_has_no_extra_rows_or_rb_conversion(layout):
    b = bank(layout)
    selector = "genic2006_"+layout
    raw = calculate_finned_tube_pressure_drop(5., b, PROPS, provider=selector)
    out = calculate_finned_tube_bank_hydraulics(5., b, inlet_props=PROPS, pressure_drop_provider=selector)
    assert out.dp_drag == raw.dp_drag == out.native_result.dp_drag
    assert out.dp_total == out.dp_drag
    assert out.dp_acceleration == 0.
    assert out.midpoint.Re == raw.Re
    assert out.reference_diameter == raw.reference_diameter_value
    assert math.isnan(out.midpoint.f) and math.isnan(out.mean_f_over_rho)
    assert math.isnan(out.inlet.coefficient) and math.isnan(out.outlet.coefficient)
    hot = calculate_finned_tube_bank_hydraulics(5., b, inlet_props=replace(PROPS,rho=1.),
        midpoint_props=PROPS, outlet_props=replace(PROPS,rho=1.4), temperature_in=600.,
        temperature_out=400., pressure_drop_provider=selector)
    assert hot.dp_drag == raw.dp_drag
    assert hot.dp_acceleration == pytest.approx((5./b.frontal_flow_area)**2*(1/1.4-1.))
    assert hot.dp_total == hot.dp_drag+hot.dp_acceleration
    assert "genic2006_mean_state_nonisothermal_extension" in {w.code for w in hot.warnings}


@pytest.mark.parametrize("layout", ["inline", "staggered"])
@pytest.mark.parametrize("changes,pattern", [
    ({"minimum_free_flow_area":.4},"frontal af"),
    ({"fin_thickness_tip":.0004},"constant-thickness"),
    ({"geometry_family":"serrated"},"round solid"),
    ({"tube_outer_diameter":.03},"must not exceed"),
    ({"pitch_longitudinal":.025},"root clearances"),
])
def test_unsupported_geometry_and_area_do_not_silently_map(layout, changes, pattern):
    with pytest.raises(ValueError, match=pattern):
        evaluate_finned_tube_pressure_drop(request(layout,**changes),provider="genic2006_"+layout)


@pytest.mark.parametrize("changes,warning", [
    ({"tube_outer_diameter":.025},"root_diameter_engineering_mapping"),
    ({"temperature_in":600.,"temperature_out":400.},"mean_state_nonisothermal_extension"),
    ({"fluid_family":"water"},"non_air_extension"),
])
def test_state_extensions_are_explicit_and_rejected_in_strict(changes, warning):
    provider=Genic2006InlinePressureDropProvider()
    r=request(**changes)
    assert "genic2006_"+warning in {w.code for w in provider.evaluate(r).warnings}
    with pytest.raises(ValueError, match=warning):
        provider.evaluate(replace(r,strict_source_domain=True))


@pytest.mark.parametrize("provider", [Genic2006InlinePressureDropProvider(), Genic2006StaggeredPressureDropProvider()])
def test_wrong_layout_and_surface_dispatch_guards(provider):
    supported="inline" if isinstance(provider,Genic2006InlinePressureDropProvider) else "staggered"
    wrong="staggered" if supported=="inline" else "inline"
    with pytest.raises(ValueError,match="only"):
        provider.evaluate(request(wrong))
    with pytest.raises(TypeError,match="pressure-drop provider"):
        evaluate_outside_thermal(bundle=bank(supported),m_dot=5.,props=PROPS,finned_heat_transfer_provider=provider)
    with pytest.raises(TypeError):
        evaluate_outside_hydraulics(bundle=replace(bank(),tube=bank().tube.core_tube),
            m_dot=5.,inlet_props=PROPS,finned_pressure_drop_provider=provider)


def test_auto_and_thermal_dispatch_stay_unchanged():
    from core.heat_transfer.finned_tube_outside import calculate_finned_tube_outside_heat_transfer
    for layout, hydraulic, thermal in [
        ("inline","frass_inline_solid_fin_191","frass_escoa_inline_solid_fin"),
        ("staggered","robinson_briggs_1966","briggs_young_1963")]:
        b=bank(layout,pitch_transverse=.065,pitch_longitudinal=math.sqrt(3)*.065/2)
        auto=calculate_finned_tube_pressure_drop(5.,b,PROPS)
        explicit=calculate_finned_tube_pressure_drop(5.,b,PROPS,provider=hydraulic)
        assert auto==explicit
        ht=calculate_finned_tube_outside_heat_transfer(5.,b,PROPS,
            gas_temperature=600.,mean_fin_temperature=450.)
        assert ht.method==thermal


@pytest.mark.parametrize("layout", ["inline", "staggered"])
@pytest.mark.parametrize("workflow", ["rate", "simulate"])
def test_explicit_selectors_reach_rating_and_simulation(layout, workflow):
    from core.models import BalanceSideSpec
    from core.models.bare_tube import BareTubeHeatExchanger
    from core.models.simulation import HXSideInput
    from core.properties.fluids import ConstantPropertyProvider
    from core.phase_change.types import PhaseChangeMode
    b=bank(layout,pitch_transverse=.065,pitch_longitudinal=math.sqrt(3)*.065/2)
    hx=BareTubeHeatExchanger(b)
    cls=BalanceSideSpec if workflow=="rate" else HXSideInput
    inside=cls(provider=ConstantPropertyProvider(FluidTransportProperties(
        rho=900.,mu=.0005,k=.6,cp=4000.)),m_dot=1.5,T_in=400.,p=101325.,
        phase_change_mode=PhaseChangeMode.DISABLED,
        **({"T_out":410.} if workflow=="rate" else {}))
    outside=cls(provider=ConstantPropertyProvider(PROPS),m_dot=5.,T_in=600.,p=101325.,
        phase_change_mode=PhaseChangeMode.DISABLED)
    result=getattr(hx,workflow)(inside,outside,finned_pressure_drop_provider="genic2006_"+layout)
    hydraulic=result.outside_tube_bank_hydraulic
    assert hydraulic.metadata.method=="genic2006_"+layout
    assert hydraulic.native_result.coefficient_basis==GENIC_2006_XI_COEFFICIENT_BASIS
    assert hydraulic.dp_total==hydraulic.dp_drag+hydraulic.dp_acceleration
    diag=result.final_result.finned_tube_diagnostics
    assert math.isnan(diag.f)
