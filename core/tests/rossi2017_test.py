# KalKalori - GNU GPL v3 only
"""Independent Rossi Eq9/10 anchors and real solver/property-reference paths."""
from dataclasses import replace
import math

import pytest

from core.enhancements import (
    EnhancementInput, EnhancementState, EnhancementUnsupportedError,
    Rossi2017TwistedTapeProvider, TubeSideEnhancement, TwistedTapeGeometry,
    evaluate_enhancement,
)
from core.enhancements.integration import evaluate_for_bundle
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.properties.common import FluidTransportProperties
from core.tests.enhancement_integration_test import bundle, inputs


def sample(re=500, pr=45):
    bulk = EnhancementState(1000, .001, .5, 4000, 300, 2e5)
    film = EnhancementState(995, .002, .4, pr*.4/.002, 310, 2e5)
    return EnhancementInput(re*.001*math.pi*.0135/4, .0135, 3, 3, bulk,
                            wall=replace(bulk, temperature=320), thermal=film,
                            fluid_phase="liquid")


def config(policy="error", diameter=.0135, y=4.44, width=None):
    return TubeSideEnhancement(Rossi2017TwistedTapeProvider(policy),
                              TwistedTapeGeometry(y*diameter, width or diameter, .0005), "liquid")


@pytest.mark.parametrize("re,pr,friction,nu", [
    # Independently evaluated with 45-digit Decimal powers of published
    # Eq9/10 coefficients; not obtained from this provider or another backend.
    (500, 45, .1328951596404086826008619185, 27.18228906023405298299787484),
    (2100, 50, .05214503349698122327563950239, 63.61274262259606459821588264),
])
def test_independent_anchors(re, pr, friction, nu):
    result = evaluate_enhancement(config(), sample(re, pr))
    assert result.nusselt == pytest.approx(nu, rel=1e-13)
    assert result.alpha_inside == pytest.approx(nu*.4/.0135, rel=1e-13)
    assert result.f_darcy == pytest.approx(friction, rel=1e-13)
    assert result.friction_factor_native == result.f_darcy
    assert result.friction_basis == "darcy"
    assert result.f_fanning == result.f_darcy/4
    assert result.wall_correction == 1
    assert result.reference.reynolds == pytest.approx(re)
    assert result.reference.prandtl == pytest.approx(pr)
    assert result.applicability == "within_range"
    assert result.source_access_basis == "open"


@pytest.mark.parametrize("re,pr,y,code", [(209,45,4.44,"reynolds"), (3101,45,4.44,"reynolds"),
                                        (500,43,4.44,"prandtl_film"), (500,52,4.44,"prandtl_film"),
                                        (500,45,4.43,"twist_ratio"), (500,45,4.45,"twist_ratio")])
def test_extrapolation_is_explicit(re, pr, y, code):
    with pytest.raises(EnhancementUnsupportedError, match=code):
        evaluate_enhancement(config(y=y), sample(re,pr))
    result = evaluate_enhancement(config("warn", y=y), sample(re,pr))
    assert result.applicability == "extrapolated"
    assert f"rossi2017_{code}_extrapolated" in {w.code for w in result.warnings}


@pytest.mark.parametrize("re,pr", [(210,44), (3100,51)])
def test_source_boundaries(re, pr):
    assert evaluate_enhancement(config(), sample(re,pr)).applicability == "within_range"


def test_width_thickness_roughness_are_not_corrections():
    a = evaluate_enhancement(config(), sample())
    c = config(width=.012)
    c = replace(c, geometry=replace(c.geometry, tape_thickness=.001))
    b = evaluate_enhancement(c, replace(sample(), roughness_inner=.00002))
    assert (a.alpha_inside, a.f_darcy) == (b.alpha_inside, b.f_darcy)
    assert {"rossi2017_width_clearance_not_represented", "rossi2017_thickness_ignored",
            "rossi2017_roughness_ignored"} <= {w.code for w in b.warnings}
    with pytest.raises(ValueError, match="width"):
        evaluate_enhancement(config(width=.014), sample())


def test_missing_film_does_not_fall_back_but_hydraulics_do_not_need_it():
    state = replace(sample(), wall=None, thermal=None)
    with pytest.raises(EnhancementUnsupportedError, match="film_state_required"):
        evaluate_enhancement(config(), state)
    result = evaluate_enhancement(config(), replace(state, position="midpoint"))
    assert result.nusselt is None and result.alpha_inside is None
    assert result.f_darcy == pytest.approx(.13289515964040868)
    assert not any("prandtl_film" in w.code for w in result.warnings)


class NonlinearLiquid:
    def __init__(self):
        self.temperatures = []

    def at(self, T, p):
        self.temperatures.append(T)
        return FluidTransportProperties(1000-.2*(T-300), .003*math.exp(-.015*(T-300)),
                                        .3+.00001*(T-290)**2, 4500)


def test_backend_evaluates_temperature_not_averaged_properties():
    backend = NonlinearLiquid()
    bulk, wall = backend.at(300,2e5), backend.at(340,2e5)
    result = evaluate_for_bundle(config("warn", diameter=.012), bundle(), .05, bulk,
                                 temperature=300, pressure=2e5, wall_props=wall,
                                 wall_temperature=340, property_provider=backend)
    assert 320 in backend.temperatures
    film = backend.at(320,2e5)
    diag = {d.name:d.value for d in result.diagnostics}
    assert diag["T_film"] == 320
    assert diag["k_film"] == film.k
    assert diag["Pr_film"] == film.cp*film.mu/film.k
    assert film.k != (bulk.k+wall.k)/2
    assert result.alpha_inside == result.nusselt*film.k/.012


@pytest.mark.parametrize("finned", [False, True])
@pytest.mark.parametrize("iterate", [False, True])
def test_simulation_film_and_local_hydraulic_states(finned, iterate):
    b = bundle(finned)
    backend = NonlinearLiquid()
    inside, outside = inputs(b, provider=backend)
    hx = BareTubeHeatExchanger(b)
    c = config("warn", diameter=.012)
    result = hx.simulate(inside, outside, tube_side_enhancement=c, iterate=iterate)
    assert result.converged and result.q > 0
    assert hx.tube_side_enhancement is None
    thermal = result.tube_side_enhancement
    diag = {d.name:d.value for d in thermal.diagnostics}
    assert diag["T_film"] == (diag["T_bulk"]+diag["T_wall"])/2
    assert diag["T_bulk"] < diag["T_wall"] < outside.T_in
    assert diag["T_film"] in backend.temperatures
    film = backend.at(diag["T_film"], inside.p)
    assert thermal.alpha_inside == thermal.nusselt*film.k/.012
    assert thermal.wall_correction == 1
    points = [result.inside_properties_inlet, result.inside_properties_midpoint, result.inside_properties_outlet]
    gradients = []
    for point in points:
        e = point.enhancement
        assert e.provider_id == thermal.provider_id and e.correlation_id == thermal.correlation_id
        assert e.alpha_inside is None and e.nusselt is None
        assert e.f_darcy == pytest.approx(19.48*e.reference.reynolds**(-.6519)*4.44**(-.6281))
        gradients.append(e.pressure_gradient(point.rho))
    assert result.inside_dp_friction == pytest.approx(.3*(gradients[0]+4*gradients[1]+gradients[2])/6)


def test_rating_and_bridge_keep_film_reference():
    b = bundle()
    inside, outside = inputs(b, provider=NonlinearLiquid())
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=config("warn", diameter=.012))
    result = hx.rate(
        BalanceSideSpec(provider=inside.provider,p=inside.p,m_dot=inside.m_dot,T_in=300,T_out=301),
        BalanceSideSpec(provider=outside.provider,p=outside.p,m_dot=outside.m_dot,T_in=340),
        include_simulation=True)
    for e in (result.tube_side_enhancement, result.simulation.tube_side_enhancement):
        assert e.thermal_property_reference == "film"
        assert e.alpha_inside > 0
        assert "rossi2017_scope" in {w.code for w in e.warnings}


def test_contract_rejects_wrong_conductivity_or_missing_thermal_coefficient():
    class Broken(Rossi2017TwistedTapeProvider):
        def evaluate(self, geometry, state):
            r = super().evaluate(geometry, state)
            return replace(r, alpha_inside=r.nusselt*state.bulk.k/state.tube_inner_diameter)
    with pytest.raises(ValueError, match="reference bases"):
        evaluate_enhancement(replace(config(), provider=Broken()), sample())
    class Missing(Rossi2017TwistedTapeProvider):
        def evaluate(self, geometry, state):
            return replace(super().evaluate(geometry,state), alpha_inside=None, nusselt=None)
    with pytest.raises(ValueError, match="requires alpha"):
        evaluate_enhancement(replace(config(), provider=Missing()), sample())


def test_invalid_policy_film_and_absent_backend():
    with pytest.raises(ValueError, match="policy"):
        Rossi2017TwistedTapeProvider("silent")
    state = sample()
    with pytest.raises(EnhancementUnsupportedError, match="inconsistent"):
        evaluate_enhancement(config(), replace(state, thermal=replace(state.thermal, temperature=301)))
    with pytest.raises(EnhancementUnsupportedError, match="reference_state_required"):
        evaluate_for_bundle(config("warn", diameter=.012), bundle(), .05,
                            FluidTransportProperties(1000,.001,.5,4000), temperature=300, pressure=2e5)


@pytest.mark.parametrize("reference,expected_temperature", [("bulk",300), ("wall",340), ("film",320)])
def test_source_neutral_property_reference(reference, expected_temperature):
    from core.tests.enhancement_provider_contract_test import FakeExternalProvider
    class ReferenceFixture(FakeExternalProvider):
        thermal_property_reference = reference
        def evaluate(self, geometry, state):
            selected = state.bulk if reference == "bulk" else state.thermal
            assert selected.temperature == expected_temperature
            r = super().evaluate("test_geometry",state)
            return replace(r, alpha_inside=r.nusselt*selected.k/r.reference.nusselt_length,
                           thermal_property_reference=reference)
    backend = NonlinearLiquid()
    c = replace(config("warn", diameter=.012), provider=ReferenceFixture())
    result = evaluate_for_bundle(c, bundle(), .05, backend.at(300,2e5), temperature=300,
                                 pressure=2e5, wall_temperature=340,
                                 wall_props=backend.at(340,2e5), property_provider=backend)
    assert result.thermal_property_reference == reference


def test_hydraulic_only_generic_clearance_fixture_and_absolute_reference():
    from core.enhancements.integration import thermal_property_reference
    from core.tests.twisted_tape_clearance_test import configured, CorrectionFixture, AbsoluteFixture, NominalFixture
    from core.tests.enhancement_provider_contract_test import sample_input
    class HydraulicFixture(NominalFixture):
        def evaluate(self, geometry, state):
            return replace(super().evaluate(geometry,state), alpha_inside=None,nusselt=None)
    c = replace(configured(CorrectionFixture()), provider=HydraulicFixture())
    r = evaluate_enhancement(c, replace(sample_input(), position="inlet"))
    assert r.alpha_inside is None and r.nusselt is None and r.f_darcy == pytest.approx(.048)
    # An absolute model replaces the base, including its property requirement.
    c = replace(configured(AbsoluteFixture()), provider=Rossi2017TwistedTapeProvider())
    assert thermal_property_reference(c) == "bulk"


def test_declared_reference_cannot_change_in_result():
    class WrongReference(Rossi2017TwistedTapeProvider):
        def evaluate(self, geometry, state):
            return replace(super().evaluate(geometry,state), thermal_property_reference="bulk")
    with pytest.raises(ValueError, match="provider declaration"):
        evaluate_enhancement(replace(config(), provider=WrongReference()), sample())
