# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Equation verification, constitutive coupling and guarded INLINE workflows.

These synthetic fixtures verify the reproduced mathematics, not experimental
accuracy or unknown ESCOA source-domain bounds.
"""
from dataclasses import replace
import math

import pytest

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.heat_transfer import FrassEscoaInlineSolidFinProvider
from core.heat_transfer.fin_efficiency import annular_fin_efficiency
from core.heat_transfer.finned_tube_outside import (
    BriggsYoung1963Provider, FinnedTubeHeatTransferRequest,
    calculate_finned_tube_outside_heat_transfer, evaluate_finned_tube_heat_transfer,
)
from core.heat_transfer.outside_dispatch import (
    build_finned_tube_diagnostics, calculate_resistance_network,
    evaluate_outside_thermal, make_mean_fin_temperature_model,
)
from core.models.bare_tube import BareTubeHeatExchanger
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


PROPS = FluidTransportProperties(rho=1.0, mu=2e-5, k=.03, cp=1050.)


def request(**changes):
    values = dict(
        m_dot=20000*2e-5/.038*.5, face_area=1., minimum_free_flow_area=.5,
        rho=1., mu=2e-5, k=.03, cp=1050., D_root=.038, D_fin=.070,
        fin_pitch=1/150, fin_thickness_root=.001, fin_thickness_tip=.001,
        pitch_transverse=.08, pitch_longitudinal=.08, layout="inline", n_rows=8,
        gas_temperature=600., mean_fin_temperature=450.,
    )
    return FinnedTubeHeatTransferRequest(**{**values, **changes})


def bundle(**changes):
    tube = CircularFinnedTube(
        core_tube=BareTube(D_i=.032, D_o=.038, length_total=1.,
                           length_effective=1., wall_k=45.),
        D_root=.038, D_fin=.070, fin_k=45., fin_thickness_root=.001,
        fin_pitch=1/150,
    )
    values = dict(tube=tube, n_rows=8, n_tubes_per_row=3, pitch_transverse=.08,
                  pitch_longitudinal=.09, layout="inline", n_passes_tube=1,
                  flow_arrangement="counterflow")
    return TubeBundle(**{**values, **changes})


# Independently evaluated with 60-digit decimal arithmetic (ln/exp/powers).
# Each row freezes C5, Nu, physical h for Re=20000, Pr=.7, Tg/Tfin=600/450,
# dA=.038, hfin=.016, Fp=1/150, tf=.001, k=.03. C3=.5208873124529347.
FIXTURES = [
    (1,.75,1.0988571174321784,115.7583771758126,91.38819250722047),
    (1,1.,1.0993068066821652,115.8057493928061,91.42559162589956),
    (1,1.25,1.0995795569996252,115.83448209483124,91.44827533802466),
    (2,.75,1.0151872099732884,106.9442260434755,84.42965213958591),
    (2,1.,1.0485584425130297,110.45969649170283,87.20502354608118),
    (2,1.25,1.0687991182007825,112.59193710186644,88.88837139621035),
    (4,.75,.953005218406979,100.39370521680367,79.25818832905553),
    (4,1.,1.0108431581460706,106.48660477324023,84.06837218940018),
    (4,1.25,1.0459236418924411,110.18213516076851,86.98589617955409),
    (8,.75,.9338900372735764,98.38002908701243,77.66844401606245),
    (8,1.,.9992492147227015,105.26524846179036,83.10414352246607),
    (8,1.25,1.0388915597391941,109.44134511144063,86.40106193008471),
]


@pytest.mark.parametrize("rows,pitch_ratio,C5,Nu,h", FIXTURES)
def test_independent_equation_fixtures(rows, pitch_ratio, C5, Nu, h):
    result = FrassEscoaInlineSolidFinProvider().evaluate(request(
        n_rows=rows, pitch_longitudinal=.08*pitch_ratio))
    assert result.diagnostics["C3"] == pytest.approx(.5208873124529347, rel=2e-14)
    assert result.diagnostics["C5"] == pytest.approx(C5, rel=2e-14)
    assert result.Nu == pytest.approx(Nu, rel=2e-14)
    assert result.alpha == pytest.approx(h, rel=2e-14)
    assert result.Re == pytest.approx(20000)
    assert result.Pr == pytest.approx(.7)
    assert result.j == pytest.approx(result.Nu/(result.Re*result.Pr**(1/3)))
    assert math.isnan(result.diagonal_pitch)  # no triangular identity in INLINE


def test_equation_implied_trends_and_supplied_property_basis():
    provider = FrassEscoaInlineSolidFinProvider()
    by_rows = [provider.evaluate(request(n_rows=n)).alpha for n in [1,2,4,8]]
    assert by_rows == sorted(by_rows, reverse=True)
    by_pitch = [provider.evaluate(request(pitch_longitudinal=.08*r)).alpha
                for r in [.75,1.,1.25]]
    assert by_pitch == sorted(by_pitch)
    original = provider.evaluate(request())
    changed_density = provider.evaluate(request(rho=2.))
    assert changed_density.Re == original.Re
    assert changed_density.reference_velocity == original.reference_velocity/2
    assert changed_density.alpha == original.alpha
    hotter_gas = provider.evaluate(request(gas_temperature=1200.))
    assert hotter_gas.alpha/original.alpha == pytest.approx(2**.25)


def test_source_uncertainty_and_single_row_caveat_are_visible():
    result = FrassEscoaInlineSolidFinProvider().evaluate(request(n_rows=1))
    diagnostics = result.diagnostics
    assert diagnostics["source_range_status"] == "SOURCE_RANGE_UNRESOLVED"
    assert diagnostics["property_convention_provenance"] == "ENGINEERING_CONVENTION"
    assert diagnostics["source_property_convention_closed"] is False
    assert result.applicability == ()
    assert {w.code for w in result.warnings} == {
        "SOURCE_RANGE_UNRESOLVED", "escoa_property_convention_engineering",
        "escoa_single_row_caveat",
    }
    assert "secondary" in result.source
    assert "C5" in result.row_basis
    with pytest.raises(ValueError, match="SOURCE_RANGE_UNRESOLVED"):
        FrassEscoaInlineSolidFinProvider().evaluate(request(strict_source_domain=True))


@pytest.mark.parametrize("layout", ["staggered", "shifted"])
def test_escoa_rejects_other_layouts(layout):
    with pytest.raises(ValueError, match="INLINE|layout"):
        FrassEscoaInlineSolidFinProvider().evaluate(request(layout=layout))


@pytest.mark.parametrize("family", ["serrated", "segmented", "louvered", "elliptical"])
def test_escoa_rejects_other_geometry_families(family):
    with pytest.raises(ValueError, match="solid individual"):
        FrassEscoaInlineSolidFinProvider().evaluate(request(geometry_family=family))


@pytest.mark.parametrize("changes,pattern", [
    ({"fin_thickness_tip":.0015}, "fin_thickness_tip"),
    ({"tube_outer_diameter":.040}, "must not exceed"),
    ({"pitch_transverse":.04}, "blockage"),
    ({"fin_pitch":.001}, "fin_pitch"),
    ({"n_rows":0}, "n_rows"),
    ({"mean_fin_temperature":None}, "mean_fin_temperature"),
    ({"gas_temperature":0.}, "gas_temperature"),
])
def test_invalid_or_unmapped_input_fails_explicitly(changes, pattern):
    with pytest.raises(ValueError, match=pattern):
        FrassEscoaInlineSolidFinProvider().evaluate(request(**changes))


def test_auto_and_explicit_provider_selection():
    auto = evaluate_finned_tube_heat_transfer(request())
    explicit = evaluate_finned_tube_heat_transfer(request(), provider="frass_escoa_inline_solid_fin")
    assert auto == explicit
    with pytest.raises(ValueError, match="staggered triangular"):
        evaluate_finned_tube_heat_transfer(request(), provider=BriggsYoung1963Provider())
    staggered = request(layout="staggered", pitch_longitudinal=math.sqrt(3)*.08/2)
    staggered_result = evaluate_finned_tube_heat_transfer(staggered)
    assert staggered_result.method == "briggs_young_1963"
    assert isinstance(hash(staggered_result), int)  # additive diagnostics preserve the frozen result's hashing
    plain = replace(bundle(), tube=bundle().tube.core_tube)
    previous_default = evaluate_outside_thermal(
        bundle=plain, m_dot=1., props=PROPS,
        finned_heat_transfer_provider="briggs_young_1963",
    )
    assert previous_default == evaluate_outside_thermal(bundle=plain,m_dot=1.,props=PROPS)


def coupled_thermal(bank, *, T_inside=400., T_gas=600., contact_fouling=0.):
    wall_resistance = math.log(bank.tube.D_o/bank.tube.D_i) / (
        2*math.pi*bank.tube.wall_k*bank.tube.length_effective*bank.n_tubes_total)
    model = make_mean_fin_temperature_model(
        bundle=bank, inside_bulk_temperature=T_inside, outside_bulk_temperature=T_gas,
        alpha_inside=900., resistance_core_wall=wall_resistance,
        fouling_resistance_outside=contact_fouling,
    )
    mass_flow = 20000*PROPS.mu/.038*bank.minimum_free_flow_area
    result = evaluate_outside_thermal(
        bundle=bank, m_dot=mass_flow, props=PROPS, gas_temperature=T_gas,
        fin_temperature_model=model,
    )
    return result, model, wall_resistance


def test_coupled_temperature_has_independent_bessel_network_fixture():
    # Fixture computed independently with scipy iv/kv convective-tip Bessel
    # solution, explicit parallel conductances and scipy brentq, not core.
    # r0=.019,r1=.035,tf=.001,kfin=45,hi=900,Ti=400,Tg=600,
    # 8x3 one-metre tubes, d_i=.032, tube k=45, ideal contact, no fouling.
    bank = bundle()
    thermal, model, wall_resistance = coupled_thermal(bank)
    result = thermal.finned_result
    assert result.method == "frass_escoa_inline_solid_fin"
    assert result.alpha == pytest.approx(82.10440467639526, rel=3e-7)
    assert result.diagnostics["T_fin_mean"] == pytest.approx(515.8880856745463, abs=2e-5)
    assert result.diagnostics["T_fin_mean"] == pytest.approx(model(result.alpha), abs=1e-7)
    assert result.diagnostics["coupling_converged"] is True
    assert result.diagnostics["coupling_relative_residual"] <= 1e-10
    network = calculate_resistance_network(
        bundle=bank, alpha_inside=900., outside_alpha_physical=result.alpha,
        resistance_core_wall=wall_resistance,
    )
    assert network.fin_efficiency == pytest.approx(.700322935791175, rel=3e-7)
    assert network.outside_alpha_effective_gross < result.alpha
    assert network.conductance_fin_outside == pytest.approx(
        result.alpha*network.fin_efficiency*network.area_fin)
    diagnostics = build_finned_tube_diagnostics(network=network, thermal=thermal)
    assert diagnostics.heat_transfer_result is result
    assert diagnostics.outside_alpha_physical == result.alpha


@pytest.mark.parametrize("T_inside,T_gas", [(400.,600.), (600.,400.), (500.,500.)])
@pytest.mark.parametrize("contact,fouling", [(0.,0.), (.002,.001)])
def test_coupling_handles_both_heat_directions_contact_and_fouling(T_inside,T_gas,contact,fouling):
    bank = bundle()
    bank = replace(bank, tube=replace(bank.tube, fin_contact_resistance=contact))
    thermal, model, _ = coupled_thermal(bank, T_inside=T_inside, T_gas=T_gas, contact_fouling=fouling)
    result = thermal.finned_result
    Tfin = result.diagnostics["T_fin_mean"]
    assert min(T_inside,T_gas) <= Tfin <= max(T_inside,T_gas)
    assert Tfin == pytest.approx(model(result.alpha), abs=1e-7)
    if T_inside == T_gas:
        assert Tfin == T_gas
        assert result.diagnostics["temperature_ratio"] == 1.
    else:
        assert Tfin != T_gas
        assert Tfin != T_inside


def test_mean_temperature_is_fin_only_independent_of_area_override():
    tube = bundle().tube
    fin = annular_fin_efficiency(tube, 100.)
    changed = annular_fin_efficiency(replace(tube, external_area_per_length=20.),100.)
    assert fin.mean_fin_temperature(400.,600.) == changed.mean_fin_temperature(400.,600.)
    actual_mean = fin.mean_fin_temperature(400.,600.)
    q = 100*tube.fin_area_per_fin*(actual_mean-600.)
    assert q == pytest.approx(fin.fin_heat_rate_per_base_temperature*(400.-600.))
    assert actual_mean != 600.+fin.overall_surface_efficiency*(400.-600.)


def test_root_layer_uses_explicit_engineering_mapping_and_real_network():
    baseline = FrassEscoaInlineSolidFinProvider().evaluate(request())
    mapped = FrassEscoaInlineSolidFinProvider().evaluate(request(tube_outer_diameter=.034))
    assert mapped.alpha == baseline.alpha  # same prescribed mean fin temperature
    assert mapped.diagnostics["d_A"] == .038
    assert mapped.diagnostics["D_root"] == .038
    assert mapped.diagnostics["tube_outer_diameter"] == .034
    assert mapped.diagnostics["diameter_mapping_provenance"] == "ENGINEERING_CONVENTION"
    assert mapped.diagnostics["source_geometry_mapping_closed"] is False
    assert "escoa_root_diameter_engineering_mapping" in {w.code for w in mapped.warnings}
    assert "continuous_root_layer_mapping" in mapped.diagnostics["unresolved_source_ranges"]
    with pytest.raises(ValueError, match="SOURCE_RANGE_UNRESOLVED"):
        FrassEscoaInlineSolidFinProvider().evaluate(request(tube_outer_diameter=.034,strict_source_domain=True))

    bank = bundle()
    metal = replace(bank.tube.core_tube, D_o=.034)
    bank = replace(bank,tube=replace(bank.tube,core_tube=metal,
        fin_contact_resistance=.001,fin_thickness_tip=.0005))
    thermal, model, wall_resistance = coupled_thermal(bank)
    result = thermal.finned_result
    network = calculate_resistance_network(bundle=bank,alpha_inside=900.,
        outside_alpha_physical=result.alpha,resistance_core_wall=wall_resistance)
    assert network.resistance_root > 0.
    assert network.contact_topology == "series_before_primary_and_fin_parallel_branches"
    assert result.diagnostics["T_fin_mean"] == pytest.approx(model(result.alpha), abs=1e-7)
    assert result.diagnostics["coupling_converged"]


def test_tapered_fin_empirical_mean_does_not_replace_physical_fin():
    provider = FrassEscoaInlineSolidFinProvider()
    tapered = provider.evaluate(request(fin_thickness_tip=.0005))
    equivalent = provider.evaluate(request(fin_thickness_root=.00075,fin_thickness_tip=.00075))
    assert tapered.alpha == equivalent.alpha
    assert tapered.diagnostics["tf"] == .00075
    assert tapered.diagnostics["thickness_mapping_provenance"] == "ENGINEERING_CONVENTION"
    assert tapered.diagnostics["source_geometry_mapping_closed"] is False
    assert "escoa_tapered_fin_engineering_mapping" in {w.code for w in tapered.warnings}
    assert "linear_taper_thickness_mapping" in tapered.diagnostics["unresolved_source_ranges"]
    with pytest.raises(ValueError,match="SOURCE_RANGE_UNRESOLVED"):
        provider.evaluate(request(fin_thickness_tip=.0005,strict_source_domain=True))
    bank = bundle()
    tube = replace(bank.tube,fin_thickness_tip=.0005)
    real = annular_fin_efficiency(tube,tapered.alpha)
    constant = annular_fin_efficiency(replace(tube,fin_thickness_root=.00075,
        fin_thickness_tip=.00075),tapered.alpha)
    assert real.fin_efficiency != pytest.approx(constant.fin_efficiency,rel=1e-4)
    assert real.method == "conservative_finite_volume_linear_taper_annular_fin"


def test_coupling_nonconvergence_is_an_error():
    calls = []
    def alternating_temperature(h):
        calls.append(h)
        return 1. if len(calls)%2 else 1e9
    bank = bundle()
    with pytest.raises(RuntimeError, match="did not converge"):
        calculate_finned_tube_outside_heat_transfer(
            1., bank, PROPS, gas_temperature=600.,
            fin_temperature_model=alternating_temperature,
        )


def test_strict_domain_is_threaded_through_automatic_dispatch():
    with pytest.raises(ValueError, match="SOURCE_RANGE_UNRESOLVED"):
        evaluate_outside_thermal(bundle=bundle(),m_dot=1.,props=PROPS,
                                 strict_source_domain=True)


@pytest.mark.parametrize("workflow", ["thermal", "simulation", "rating"])
def test_inline_workflows_close_thermal_then_stop_at_hydraulic_guard(workflow):
    from core.models.simulation import HXSideInput
    from core.models import BalanceSideSpec
    from core.phase_change.types import PhaseChangeMode
    bank = bundle()
    hx = BareTubeHeatExchanger(bank)
    inside = ConstantPropertyProvider(FluidTransportProperties(rho=900.,mu=.0005,k=.6,cp=4000.))
    outside = ConstantPropertyProvider(PROPS)
    if workflow == "thermal":
        state = hx.solve_thermal_state(
            m_dot_inside=1.5,m_dot_outside=1.,inside_provider=inside,outside_provider=outside,
            T_in_inside=400.,T_in_outside=600.,p_inside=101325.,p_outside=101325.)
        assert state.converged
        result = state.finned_tube_diagnostics.heat_transfer_result
        assert result.method == "frass_escoa_inline_solid_fin"
        assert result.diagnostics["coupling_converged"]
    elif workflow == "simulation":
        i = HXSideInput(provider=inside,m_dot=1.5,T_in=400.,p=101325.,phase_change_mode=PhaseChangeMode.DISABLED)
        o = HXSideInput(provider=outside,m_dot=1.,T_in=600.,p=101325.,phase_change_mode=PhaseChangeMode.DISABLED)
        with pytest.raises(ValueError, match="RobinsonBriggs1966Provider.*staggered"):
            hx.simulate(i,o)
    else:
        i = BalanceSideSpec(provider=inside,m_dot=1.5,T_in=400.,T_out=410.,p=101325.)
        o = BalanceSideSpec(provider=outside,m_dot=1.,T_in=600.,p=101325.)
        with pytest.raises(ValueError, match="RobinsonBriggs1966Provider.*staggered"):
            hx.rate(i,o)
