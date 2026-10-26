# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Native xi fixtures and qualified mean-state INLINE hydraulic workflows."""
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.pressure_drop import (
    FRASS_INLINE_XI_COEFFICIENT_BASIS, FrassInlineSolidFinPressureDropProvider,
    FinnedTubePressureDropRequest, RobinsonBriggs1966Provider,
    calculate_finned_tube_bank_hydraulics, calculate_finned_tube_pressure_drop,
    evaluate_finned_tube_pressure_drop,
)
from core.heat_transfer.outside_dispatch import evaluate_outside_hydraulics, evaluate_outside_thermal
from core.heat_transfer import FrassEscoaInlineSolidFinProvider
from core.pressure_drop.flow_path import build_outside_pressure_drop_result
from core.properties.common import FluidTransportProperties


PROPS = FluidTransportProperties(rho=1., mu=2e-5, k=.03, cp=1050.)
XI = .3338430036837449


def request(**changes):
    values = dict(m_dot=(20000*2e-5/.038)*.5, face_area=1.,
        minimum_free_flow_area=.5, rho=1., mu=2e-5, D_root=.038, D_fin=.070,
        fin_pitch=1/150, fin_thickness_root=.001, fin_thickness_tip=.001,
        pitch_transverse=.08, pitch_longitudinal=.09, layout="inline", n_rows=8,
        temperature_in=500., temperature_out=500.)
    return FinnedTubePressureDropRequest(**{**values, **changes})


def bundle(**changes):
    tube = CircularFinnedTube(
        core_tube=BareTube(D_i=.032,D_o=.038,length_total=1.,length_effective=1.,wall_k=45.),
        D_root=.038,D_fin=.070,fin_k=45.,fin_thickness_root=.001,fin_pitch=1/150)
    values = dict(tube=tube,n_rows=8,n_tubes_per_row=3,pitch_transverse=.08,
                  pitch_longitudinal=.09,layout="inline",n_passes_tube=1,
                  flow_arrangement="counterflow")
    return TubeBundle(**{**values, **changes})


# Independent 60-digit Decimal ln/exp/power calculations, never core-derived.
@pytest.mark.parametrize("rows,kz,dp", [
    (2,.9,33.2918230820355040),
    (4,.9333333333333333,69.0497071331106749),
    (6,.9666666666666667,107.2736521532255128),
    (8,1.,147.9636581423800177),
])
def test_independent_xi_row_and_bank_fixtures(rows,kz,dp):
    result = FrassInlineSolidFinPressureDropProvider().evaluate(request(n_rows=rows))
    assert result.xi == pytest.approx(XI,rel=2e-14)
    assert result.Kz == pytest.approx(kz,rel=2e-14)
    assert result.dp_row == pytest.approx(18.4954572677975022,rel=2e-14)
    assert result.bank_delta_p == pytest.approx(dp,rel=2e-14)
    assert result.coefficient_basis == FRASS_INLINE_XI_COEFFICIENT_BASIS
    assert result.coefficient == result.xi
    assert math.isnan(result.f)
    assert math.isnan(result.correlation_fin_thickness)
    assert math.isnan(result.correlation_fin_spacing)
    assert math.isnan(result.diagonal_pitch)
    assert result.diagnostics["row_status"] == "SOURCE_ROW_ANCHOR"
    assert result.diagnostics["dynamic_pressure"] == pytest.approx(55.4016620498614958)


@pytest.mark.parametrize("Re,xi,extrapolated", [
    (6000,.50271408403671197,True),
    (7000,.47704477573467007,False),
    (10000,.42256398326354459,False),
    (25000,.30945170135639631,False),
    (30000,.29085146196250782,True),
])
def test_independent_re_fixtures_and_approximate_evidence(Re,xi,extrapolated):
    result = FrassInlineSolidFinPressureDropProvider().evaluate(request(m_dot=Re*2e-5/.038*.5))
    assert result.Re == pytest.approx(Re)
    assert result.xi == pytest.approx(xi,rel=2e-14)
    codes = {w.code for w in result.warnings}
    assert ("frass_inline_re_evidence_extrapolation" in codes) is extrapolated
    assert "SOURCE_RANGE_UNRESOLVED" in codes
    assert result.applicability == ()  # approximate Re evidence is not a verified box


@pytest.mark.parametrize("rows,kz", [(3,.9166666666666667),(5,.95),(7,.9833333333333333)])
def test_odd_rows_are_diagnosed_interpolation(rows,kz):
    result = FrassInlineSolidFinPressureDropProvider().evaluate(request(n_rows=rows))
    assert result.Kz == pytest.approx(kz)
    assert result.diagnostics["row_status"] == "CORRELATION_INTERPOLATION"
    assert "frass_inline_row_interpolation" in {w.code for w in result.warnings}


@pytest.mark.parametrize("rows", [9,20,100])
def test_rows_above_eight_use_explicit_asymptotic_convention(rows):
    result = FrassInlineSolidFinPressureDropProvider().evaluate(request(n_rows=rows))
    assert result.Kz == 1.
    assert result.bank_delta_p == pytest.approx(rows*18.4954572677975022)
    assert result.diagnostics["row_status"] == "ENGINEERING_ASYMPTOTIC_CONVENTION"
    with pytest.raises(ValueError,match="rows >8"):
        FrassInlineSolidFinPressureDropProvider().evaluate(request(n_rows=rows,strict_source_domain=True))


@pytest.mark.parametrize("changes,pattern", [
    ({"n_rows":1},"at least 2"),
    ({"n_rows":0},"positive"),
    ({"tube_outer_diameter":.040},"must not exceed"),
    ({"pitch_transverse":.069},"fin-tip clearance"),
    ({"pitch_longitudinal":.069},"fin-tip clearance"),
    ({"layout":"staggered"},"only INLINE"),
    ({"layout":"shifted"},"layout"),
    ({"geometry_family":"serrated"},"round solid"),
    ({"geometry_family":"elliptical"},"round solid"),
])
def test_physical_and_family_guards(changes,pattern):
    with pytest.raises(ValueError,match=pattern):
        FrassInlineSolidFinPressureDropProvider().evaluate(request(**changes))


@pytest.mark.parametrize("changes,pattern", [
    ({},"SOURCE_RANGE_UNRESOLVED"),
    ({"temperature_out":600.},"heated_normalization_unresolved"),
    ({"tube_outer_diameter":.034},"root-layer mapping"),
    ({"fin_thickness_tip":.0005},"taper geometry"),
])
def test_strict_rejects_unresolved_required_conditions(changes,pattern):
    with pytest.raises(ValueError,match=pattern):
        FrassInlineSolidFinPressureDropProvider().evaluate(request(strict_source_domain=True,**changes))


def test_heated_policy_applies_no_temperature_multiplier_and_excludes_other_losses():
    provider = FrassInlineSolidFinPressureDropProvider()
    iso = provider.evaluate(request())
    hot = provider.evaluate(request(temperature_in=900.,temperature_out=400.))
    assert hot.bank_delta_p == iso.bank_delta_p
    assert hot.diagnostics["heated_normalization"] == "NOT_APPLIED"
    assert hot.diagnostics["heated_normalization_source_closed"] is False
    assert hot.diagnostics["loss_scope_source_closed"] is False
    assert hot.diagnostics["includes_acceleration"] is False
    assert hot.diagnostics["includes_external_losses"] is False
    assert "frass_inline_heated_normalization_unresolved" in {w.code for w in hot.warnings}


def test_root_mapping_and_taper_do_not_change_empirical_terms_at_prescribed_area():
    provider = FrassInlineSolidFinPressureDropProvider()
    base = provider.evaluate(request())
    root = provider.evaluate(request(tube_outer_diameter=.034))
    taper = provider.evaluate(request(fin_thickness_tip=.0005))
    assert base.xi == root.xi == taper.xi
    assert root.diagnostics["d_A"] == root.diagnostics["D_root"] == .038
    assert root.diagnostics["tube_outer_diameter"] == .034
    assert root.diagnostics["diameter_mapping_provenance"] == "ENGINEERING_CONVENTION"
    assert taper.diagnostics["taper_status"] == "UNVALIDATED_GEOMETRY_EXTENSION"
    assert "frass_inline_taper_unvalidated_extension" in {w.code for w in taper.warnings}
    assert math.isnan(taper.correlation_fin_thickness)


@pytest.mark.parametrize("pitch", [1/100,1/150,1/250])
@pytest.mark.parametrize("Pt,Pl", [(.08,.09),(.09,.08),(.08,.08)])
def test_live_inline_minimum_area_matches_frass_constant_fin_throat(pitch,Pt,Pl):
    bank=bundle(pitch_transverse=Pt,pitch_longitudinal=Pl)
    bank=replace(bank,tube=replace(bank.tube,fin_pitch=pitch))
    aperture=Pt-.038-2*.016*(.001/pitch)
    assert bank.minimum_free_flow_area == pytest.approx(3*1.*aperture,rel=2e-14)
    assert bank.minimum_free_flow_gap == pytest.approx(aperture,rel=2e-14)


def test_actual_taper_changes_area_and_re_without_empirical_thickness_term():
    bank=bundle()
    tapered=replace(bank,tube=replace(bank.tube,fin_thickness_tip=.0005))
    actual=calculate_finned_tube_pressure_drop(1.,tapered,PROPS)
    constant=calculate_finned_tube_pressure_drop(1.,bank,PROPS)
    assert tapered.minimum_free_flow_area > bank.minimum_free_flow_area
    assert actual.Re < constant.Re
    assert actual.diagnostics["minimum_free_flow_area"] == tapered.minimum_free_flow_area
    assert actual.diagnostics["fin_thickness_tip"] == .0005
    assert math.isnan(actual.correlation_fin_thickness)


def test_auto_dispatch_explicit_guards_and_rb_regression():
    auto=evaluate_finned_tube_pressure_drop(request())
    explicit=evaluate_finned_tube_pressure_drop(request(),provider="frass_inline_solid_fin_191")
    assert auto == explicit
    with pytest.raises(ValueError,match="staggered triangular"):
        evaluate_finned_tube_pressure_drop(request(),provider=RobinsonBriggs1966Provider())
    staggered=request(layout="staggered",pitch_longitudinal=math.sqrt(3)*.08/2)
    assert evaluate_finned_tube_pressure_drop(staggered) == RobinsonBriggs1966Provider().evaluate(staggered)
    assert isinstance(hash(evaluate_finned_tube_pressure_drop(staggered)),int)
    class DescriptiveLegacyProvider:
        def evaluate(self,r):
            return replace(RobinsonBriggs1966Provider().evaluate(r),
                           coefficient_definition="custom Robinson-Briggs description")
    legacy_bank=calculate_finned_tube_bank_hydraulics(1.,
        bundle(layout="staggered",pitch_longitudinal=math.sqrt(3)*.08/2),
        inlet_props=PROPS,pressure_drop_provider=DescriptiveLegacyProvider())
    assert legacy_bank.midpoint.f == legacy_bank.midpoint.coefficient
    assert replace(legacy_bank,coefficient_definition="custom description").mean_f_over_rho == legacy_bank.mean_coefficient_over_rho
    plain=replace(bundle(),tube=bundle().tube.core_tube)
    old=evaluate_outside_hydraulics(bundle=plain,m_dot=1.,inlet_props=PROPS,
                                  finned_pressure_drop_provider="robinson_briggs_1966")
    assert old == evaluate_outside_hydraulics(bundle=plain,m_dot=1.,inlet_props=PROPS)
    with pytest.raises(TypeError):
        evaluate_outside_hydraulics(bundle=plain,m_dot=1.,inlet_props=PROPS,
                                   finned_pressure_drop_provider=FrassInlineSolidFinPressureDropProvider())
    with pytest.raises(TypeError,match="pressure-drop provider"):
        evaluate_outside_thermal(bundle=bundle(),m_dot=1.,props=PROPS,
            finned_heat_transfer_provider=FrassInlineSolidFinPressureDropProvider())
    with pytest.raises(TypeError,match="heat-transfer provider"):
        evaluate_outside_hydraulics(bundle=bundle(),m_dot=1.,inlet_props=PROPS,
            finned_pressure_drop_provider=FrassEscoaInlineSolidFinProvider())


def test_factor_four_is_diagnostic_only_for_matched_bases():
    result=FrassInlineSolidFinPressureDropProvider().evaluate(request())
    assert result.Kz == 1.
    # This relation is a test diagnostic, never an integration conversion.
    equivalent_f=result.xi/4
    matched_pressure=2*8*equivalent_f*1.*(10.5263157894736842**2)
    assert matched_pressure == pytest.approx(result.bank_delta_p)
    assert math.isnan(result.f)


@pytest.mark.parametrize("Tin,Tout", [(400.,600.),(600.,400.),(500.,500.)])
def test_native_bank_uses_one_mean_state_and_separate_acceleration(Tin,Tout):
    bank=bundle()
    calls=[]
    class CountingProvider:
        hydraulic_state_basis="mean_outside_gas"
        def evaluate(self,r):
            calls.append(r)
            return replace(FrassInlineSolidFinPressureDropProvider().evaluate(r),
                           coefficient_definition="custom native xi description")
    class Gas:
        def at(self,T,p):
            return SimpleNamespace(transport=FluidTransportProperties(rho=600./T,mu=2e-5*T/500.,k=.03,cp=1000.),h=T*T)
        def temperature_from_h_p(self,h,p):
            raise AssertionError("Native mean-state policy must use arithmetic temperature, not enthalpy inversion")
    result=calculate_finned_tube_bank_hydraulics(1.,bank,property_provider=Gas(),
        temperature_in=Tin,temperature_out=Tout,pressure=101325.,pressure_drop_provider=CountingProvider())
    assert len(calls)==1
    assert calls[0].rho == 1.2
    assert calls[0].mu == 2e-5
    assert result.midpoint.temperature == 500.
    assert result.dp_drag == result.native_result.bank_delta_p
    assert math.isnan(result.inlet.coefficient) and math.isnan(result.outlet.coefficient)
    assert math.isnan(result.midpoint.f) and math.isnan(result.mean_f_over_rho)
    expected_acceleration=(1/bank.frontal_flow_area)**2*(Tout-Tin)/600.
    assert result.dp_acceleration == pytest.approx(expected_acceleration,abs=1e-12)
    assert result.dp_total == pytest.approx(result.dp_drag+expected_acceleration)
    path=build_outside_pressure_drop_result(result)
    assert path.dp_core == pytest.approx(result.dp_drag)
    assert path.delta_dynamic_pressure_core == pytest.approx(expected_acceleration)
    assert path.dp_local == 0.


def test_strict_state_reaches_hydraulic_adapter():
    with pytest.raises(ValueError,match="heated_normalization_unresolved"):
        evaluate_outside_hydraulics(bundle=bundle(),m_dot=1.,inlet_props=PROPS,
            temperature_in=400.,temperature_out=600.,strict_source_domain=True)


def test_explicit_hydraulic_layout_guards_reach_live_adapter():
    with pytest.raises(ValueError,match="RobinsonBriggs1966Provider.*staggered"):
        evaluate_outside_hydraulics(bundle=bundle(),m_dot=1.,inlet_props=PROPS,
                                   finned_pressure_drop_provider=RobinsonBriggs1966Provider())
    staggered=bundle(layout="staggered",pitch_longitudinal=math.sqrt(3)*.08/2)
    with pytest.raises(ValueError,match="only INLINE"):
        evaluate_outside_hydraulics(bundle=staggered,m_dot=1.,inlet_props=PROPS,
                                   finned_pressure_drop_provider=FrassInlineSolidFinPressureDropProvider())


def test_external_local_losses_are_aggregated_once_outside_native_provider():
    from core.geometry.outside_pressure_drop_path import SpecifiedOutsidePressureDropPath
    from core.geometry.pressure_drop_stages import PressureDropAssemblyGeometry, UserDefinedPressureDropGeometry
    from core.pressure_drop.flow_path import PressureDropFlowState, calculate_outside_pressure_drop_path
    bank=calculate_finned_tube_bank_hydraulics(1.,bundle(),inlet_props=PROPS,
        temperature_in=400.,temperature_out=600.)
    path=SpecifiedOutsidePressureDropPath(
        inlet=PressureDropAssemblyGeometry(stages=(UserDefinedPressureDropGeometry(pressure_drop=15.),)),
        outlet=PressureDropAssemblyGeometry(stages=(UserDefinedPressureDropGeometry(pressure_drop=25.),)))
    state=PressureDropFlowState(mass_flow=1.,props=PROPS,temperature=500.,pressure=101325.)
    aggregate=calculate_outside_pressure_drop_path(tube_bank=bank,path=path,
        inlet_state=state,outlet_state=state)
    assert aggregate.dp_core == bank.native_result.bank_delta_p
    assert aggregate.dp_local == 40.
    assert aggregate.dp_total == pytest.approx(bank.dp_drag+40.)
    assert aggregate.delta_dynamic_pressure_core == bank.dp_acceleration
