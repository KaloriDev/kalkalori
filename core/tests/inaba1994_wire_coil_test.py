# KalKalori - GNU GPL v3 only
"""Independent Inaba (1994) source, normalization and integration regressions."""
from dataclasses import replace
import math

import pytest

from core.enhancements import (
    EnhancementInput, EnhancementState, EnhancementUnsupportedError,
    Inaba1994WireCoilProvider, TubeSideEnhancement, WireCoilGeometry,
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


def geometry(e=.002, q=20.0):
    return WireCoilGeometry(wire_diameter=e, pitch=q*e)


def state(re=1000.0, pr=6.0, coil=None, diameter=.016, position="thermal",
          heat_flow_direction="heating"):
    coil = coil or geometry()
    helix = math.sqrt(coil.pitch**2 + math.pi**2*(diameter-coil.wire_diameter)**2)/coil.pitch
    dh = diameter/(1 + coil.wire_diameter*helix/diameter)
    bulk = EnhancementState(1000.0, .001, .6, 4200.0, 300.0, 2e5)
    film = EnhancementState(995.0, .0012, .6, pr*.6/.0012, 320.0, 2e5)
    velocity = re*(film.mu/film.rho)/dh
    mass_flow = velocity*bulk.rho*math.pi*diameter**2/4
    return EnhancementInput(
        mass_flow, diameter, 2.0, 2.0, bulk,
        wall=replace(bulk, temperature=340.0), thermal=film,
        hydraulic=film, fluid_phase="liquid", position=position,
        heat_flow_direction=heat_flow_direction,
        base_flow_area_per_tube=math.pi*diameter**2/4,
        hydraulic_length_total=2.0,
    )


def configured(policy="error", coil=None):
    return TubeSideEnhancement(Inaba1994WireCoilProvider(policy), coil or geometry(), "liquid")


def diagnostics(result):
    return {item.name: item.value for item in result.diagnostics}


def source_bundle(diameter=.016):
    tube = BareTube(D_i=diameter, D_o=.020 if diameter == .016 else diameter+.004,
                    length_total=.3, length_effective=.3, wall_k=45)
    return TubeBundle(tube=tube, n_rows=4, n_tubes_per_row=8,
                      pitch_transverse=.04, pitch_longitudinal=.04*math.sqrt(3)/2,
                      layout="staggered", n_passes_tube=1,
                      flow_arrangement="counterflow")


def test_geometry_primitives_and_validation_are_source_neutral():
    coil = geometry()
    assert coil.pitch_ratio == 20.0
    for kwargs in ({"wire_diameter": 0, "pitch": .04},
                   {"wire_diameter": .002, "pitch": 0}):
        with pytest.raises(ValueError, match="positive"):
            WireCoilGeometry(**kwargs)
    with pytest.raises(ValueError, match="smaller"):
        geometry(e=.016).validate_for(.016)


def test_independent_source_geometry_and_equation_anchors():
    result = evaluate_enhancement(configured(), state())
    d = diagnostics(result)
    assert d["pitch_ratio"] == 20.0
    assert d["helix_length_ratio"] == pytest.approx(1.4862794283490055, rel=1e-14)
    assert d["source_hydraulic_diameter"] == pytest.approx(.013493172003502448, rel=1e-14)
    assert d["source_area_ratio_As_A0"] == pytest.approx(1.1857849285436257, rel=1e-14)
    assert d["source_friction_factor"] == pytest.approx(.0573854437268498, rel=1e-13)
    assert d["source_nusselt"] == pytest.approx(24.382142790394262, rel=1e-13)
    assert result.friction_basis == "fanning"
    assert result.friction_factor_native == d["source_friction_factor"]
    assert result.f_darcy == pytest.approx(.2721871771563474, rel=1e-13)
    assert result.nusselt == pytest.approx(34.28338711039142, rel=1e-13)


def test_nominal_bore_velocity_and_film_reynolds_are_independent():
    s = state()
    result = evaluate_enhancement(configured(), s)
    d = diagnostics(result)
    expected_area = math.pi*.016**2/4
    expected_velocity = s.mass_flow_per_tube/(s.bulk.rho*expected_area)
    expected_re = expected_velocity*d["source_hydraulic_diameter"]/(s.hydraulic.mu/s.hydraulic.rho)
    assert d["source_flow_area"] == expected_area
    assert d["source_volumetric_flow"] == s.mass_flow_per_tube/s.bulk.rho
    assert d["source_velocity"] == expected_velocity
    assert result.reference.velocity == expected_velocity
    assert result.reference.reynolds == pytest.approx(expected_re)
    assert expected_re == pytest.approx(1000.0)


def test_pressure_gradient_and_heat_flow_reference_transformations():
    s = state()
    result = evaluate_enhancement(configured(), s)
    d = diagnostics(result)
    rho, velocity, D, dh = s.bulk.rho, d["source_velocity"], .016, d["source_hydraulic_diameter"]
    source_gradient = 4*d["source_friction_factor"]*rho*velocity**2/(2*dh)
    engine_gradient = result.f_darcy*rho*velocity**2/(2*D)
    assert result.pressure_gradient(rho) == pytest.approx(source_gradient, rel=1e-14)
    assert engine_gradient == pytest.approx(source_gradient, rel=1e-14)
    A0 = math.pi*D*s.heated_length
    As = A0*d["source_area_ratio_As_A0"]
    assert d["source_alpha"]*As == pytest.approx(result.alpha_inside*A0, rel=1e-14)
    assert result.alpha_inside == pytest.approx(result.nusselt*s.thermal.k/D, rel=1e-14)


def test_high_re_anchor_and_exact_2000_branch_rule():
    high = evaluate_enhancement(configured(), state(4000.0))
    assert diagnostics(high)["source_nusselt"] == pytest.approx(64.4024949186253, rel=1e-13)
    assert high.regime == "eq10_high_re"
    cases = (
        (1999.999999, "eq11_low_re", 42.4517762641078),
        (2000.0, "eq11_low_re", 42.45177628108852),
        (2000.000001, "eq10_high_re", 41.615373317209894),
    )
    for re, branch, expected in cases:
        result = evaluate_enhancement(configured(), state(re))
        assert result.regime == branch
        assert diagnostics(result)["nusselt_equation_branch"] == branch
        assert diagnostics(result)["source_nusselt"] == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("re", [400.0, 6000.0])
@pytest.mark.parametrize("pr", [4.21, 8.12])
def test_source_reynolds_and_prandtl_boundaries(re, pr):
    assert evaluate_enhancement(configured(), state(re, pr)).applicability == "within_range"


@pytest.mark.parametrize("q", [math.nextafter(10.0, math.inf), 10.0000001, 50.3])
def test_supported_pitch_boundaries(q):
    coil = geometry(q=q)
    assert evaluate_enhancement(configured(coil=coil), state(coil=coil)).applicability == "within_range"


@pytest.mark.parametrize("q", [10.0, 9.999999, 2.5])
@pytest.mark.parametrize("policy", ["error", "warn"])
def test_unimplemented_low_pitch_family_is_never_extrapolated(q, policy):
    coil = geometry(q=q)
    with pytest.raises(EnhancementUnsupportedError, match="unsupported_branch"):
        evaluate_enhancement(configured(policy, coil), state(coil=coil))


@pytest.mark.parametrize("re,pr,q,code", [
    (399, 6, 20, "reynolds_source"), (6001, 6, 20, "reynolds_source"),
    (1000, 4.20, 20, "prandtl_film"), (1000, 8.13, 20, "prandtl_film"),
    (1000, 6, 50.31, "pitch_ratio"),
])
def test_numerical_extrapolation_requires_explicit_warning_mode(re, pr, q, code):
    coil = geometry(q=q)
    with pytest.raises(EnhancementUnsupportedError, match=code):
        evaluate_enhancement(configured(coil=coil), state(re, pr, coil))
    result = evaluate_enhancement(configured("warn", coil), state(re, pr, coil))
    assert result.applicability == "extrapolated"
    assert f"inaba1994_{code}_extrapolated" in {w.code for w in result.warnings}


@pytest.mark.parametrize("e", [.002, .0025, .003])
def test_exact_source_wire_ratios(e):
    coil = geometry(e=e)
    d = diagnostics(evaluate_enhancement(configured(coil=coil), state(coil=coil)))
    assert d["geometry_ratio_status"] == "tested_ratio"
    assert d["geometry_scale_status"] == "source_diameter"


def test_geometry_interpolation_scaling_and_extrapolation_are_distinct():
    interpolated = geometry(e=.00225)
    direct = evaluate_enhancement(configured(coil=interpolated), state(coil=interpolated))
    assert diagnostics(direct)["geometry_ratio_status"] == "geometry_interpolation"
    assert "inaba1994_geometry_interpolation" in {w.code for w in direct.warnings}
    scaled = geometry(e=.0015)
    with pytest.raises(EnhancementUnsupportedError, match="tube_diameter_scaling"):
        evaluate_enhancement(configured(coil=scaled), state(coil=scaled, diameter=.012))
    scaled_result = evaluate_enhancement(configured("warn", scaled), state(coil=scaled, diameter=.012))
    assert diagnostics(scaled_result)["geometry_scale_status"] == "diameter_scaling"
    extrapolated = geometry(e=.0015)
    with pytest.raises(EnhancementUnsupportedError, match="wire_diameter_ratio"):
        evaluate_enhancement(configured(coil=extrapolated), state(coil=extrapolated))


class NonlinearWaterLike:
    def __init__(self):
        self.temperatures = []

    def at(self, T, p):
        self.temperatures.append(T)
        return FluidTransportProperties(1000-.2*(T-300),
                                        .0012*math.exp(-.012*(T-320)),
                                        .55+.00002*(T-315)**2, 4000+2*(T-300))


def test_backend_evaluates_film_state_without_averaging_properties():
    backend = NonlinearWaterLike()
    b = source_bundle()
    bulk, wall = backend.at(300, 2e5), backend.at(340, 2e5)
    result = evaluate_for_bundle(configured("warn"), b, .04, bulk,
                                 temperature=300, pressure=2e5,
                                 wall_props=wall, wall_temperature=340,
                                 property_provider=backend,
                                 heat_flow_direction="heating")
    film = backend.at(320, 2e5)
    d = diagnostics(result)
    assert 320 in backend.temperatures
    assert d["thermal_reference_temperature"] == 320
    assert d["hydraulic_reference_temperature"] == 320
    assert result.reference.prandtl == pytest.approx(film.cp*film.mu/film.k)
    assert film.k != (bulk.k+wall.k)/2
    assert result.alpha_inside == pytest.approx(result.nusselt*film.k/.016)


def test_missing_or_inconsistent_hydraulic_film_is_rejected():
    s = state()
    with pytest.raises(EnhancementUnsupportedError, match="hydraulic_film_state_required"):
        evaluate_enhancement(configured(), replace(s, hydraulic=None))
    with pytest.raises(EnhancementUnsupportedError, match="hydraulic_film_temperature_inconsistent"):
        evaluate_enhancement(configured(), replace(s, hydraulic=replace(s.hydraulic, temperature=319)))


def test_rating_simulation_and_three_point_hydraulics_use_same_model():
    b = source_bundle()
    backend = NonlinearWaterLike()
    area_tube = math.pi*.016**2/4
    film = backend.at(320, 2e5)
    helix = math.sqrt(.04**2+math.pi**2*(.016-.002)**2)/.04
    dh = .016/(1+.002*helix/.016)
    target_re = 1000
    velocity = target_re*(film.mu/film.rho)/dh
    flow = velocity*backend.at(300,2e5).rho*area_tube*b.n_tubes_per_pass_effective
    inside = HXSideInput(provider=backend, m_dot=flow, T_in=300, p=2e5)
    outside = HXSideInput(provider=ConstantPropertyProvider(
        FluidTransportProperties(1.1, 2e-5, .03, 1000)), m_dot=.25, T_in=340, p=1e5)
    simulation = BareTubeHeatExchanger(b, tube_side_enhancement=configured("warn")).simulate(inside, outside)
    rating = BareTubeHeatExchanger(b).rate(
        BalanceSideSpec(provider=backend, p=2e5, m_dot=flow, T_in=300,
                        T_out=simulation.T_out_inside),
        BalanceSideSpec(provider=outside.provider, p=1e5, m_dot=.25, T_in=340),
        include_simulation=True, tube_side_enhancement=configured("warn"))
    assert simulation.converged and rating.simulation.converged
    for result in (simulation, rating, rating.simulation):
        assert result.tube_side_enhancement.provider_id == "inaba_1994_wire_coil"
        assert "inaba1994_source_context" in {w.code for w in result.warnings}
    for point in (simulation.inside_properties_inlet,
                  simulation.inside_properties_midpoint,
                  simulation.inside_properties_outlet):
        assert point.enhancement.provider_id == simulation.tube_side_enhancement.provider_id
        assert point.enhancement.alpha_inside is None
        d = diagnostics(point.enhancement)
        assert "inaba1994_heating_context_unmatched" not in {
            warning.code for warning in point.enhancement.warnings}
        assert d["hydraulic_reference_temperature"] == pytest.approx(
            (point.temperature + diagnostics(simulation.tube_side_enhancement)["thermal_reference_temperature"]*2
             - .5*(300+simulation.T_out_inside))/2, rel=2e-2)
    gradients = [p.enhancement.pressure_gradient(p.rho) for p in (
        simulation.inside_properties_inlet, simulation.inside_properties_midpoint,
        simulation.inside_properties_outlet)]
    assert simulation.inside_dp_friction == pytest.approx(.3*(gradients[0]+4*gradients[1]+gradients[2])/6)


def test_noniterative_simulation_retains_hydraulic_film_and_heating_context():
    b = source_bundle()
    backend = NonlinearWaterLike()
    inside = HXSideInput(provider=backend, m_dot=.04, T_in=300, p=2e5)
    outside = HXSideInput(provider=ConstantPropertyProvider(
        FluidTransportProperties(1.1, 2e-5, .03, 1000)), m_dot=.25, T_in=340, p=1e5)

    result = BareTubeHeatExchanger(
        b, tube_side_enhancement=configured("warn"),
    ).simulate(inside, outside, iterate=False)

    assert result.converged
    for point in (result.inside_properties_inlet,
                  result.inside_properties_midpoint,
                  result.inside_properties_outlet):
        assert point.enhancement.provider_id == "inaba_1994_wire_coil"
        assert diagnostics(point.enhancement)["hydraulic_reference_temperature"] != point.temperature
        assert "inaba1994_heating_context_unmatched" not in {
            warning.code for warning in point.enhancement.warnings}


def test_source_context_and_unmodelled_local_loss_diagnostics_are_explicit():
    result = evaluate_enhancement(configured(), state())
    codes = {w.code for w in result.warnings}
    assert {"inaba1994_source_context", "inaba1994_local_losses_outside_model"} <= codes
    assert diagnostics(result)["provenance"].startswith("Inaba, Ozaki & Kanaoka 1994")
