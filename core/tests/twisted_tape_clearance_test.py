# KalKalori - GNU GPL v3 only
"""Synthetic contract fixtures only; no literature or manufacturer equations."""
from dataclasses import replace
import math
import pytest

from core.enhancements import (
    ClearanceCorrection, ClearanceModelMode, TwistedTapeClearanceResult,
    EnhancementUnsupportedError, TubeSideEnhancement, TwistedTapeGeometry,
    evaluate_enhancement, require_nominal_twisted_tape,
)
from core.common.warnings import make_warning
from core.tests.enhancement_provider_contract_test import FakeExternalProvider, sample_input
from core.tests.enhancement_integration_test import bundle, inputs, VariableLiquid
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec


class NominalFixture(FakeExternalProvider):
    def evaluate(self, geometry, state):
        require_nominal_twisted_tape(geometry, state.tube_inner_diameter)
        result = super().evaluate("test_geometry", state)
        wall = .9 if state.wall else 1
        return replace(result, nusselt=result.nusselt*wall,
                       alpha_inside=result.alpha_inside*wall, wall_correction=wall)


class CorrectionFixture:
    provider_id = "clearance_correction_fixture"
    mode = ClearanceModelMode.CORRECTION

    def base_geometry_for(self, geometry, state):
        return replace(geometry, tape_width=state.tube_inner_diameter)

    def evaluate(self, geometry, state, base_result):
        assert base_result is not None
        return TwistedTapeClearanceResult(self.mode,
            geometry.clearance_for(state.tube_inner_diameter).radial_clearance/state.tube_inner_diameter,
            "radial_clearance / tube_inner_diameter",
            correction=ClearanceCorrection("relative_fixture", ("external:relative_fixture",),
                "external", (base_result.correlation_id,), .8, .6, base_result.regime,
                "extrapolated", (make_warning(code="clearance_fixture_warning",
                source=self.provider_id, message="Synthetic extrapolation test fixture."),)))


class AbsoluteFixture:
    provider_id = "clearance_absolute_fixture"
    mode = ClearanceModelMode.ABSOLUTE

    def evaluate(self, geometry, state, base_result):
        assert base_result is None
        result = FakeExternalProvider().evaluate("test_geometry", state)
        wall = .85 if state.wall else 1
        result = replace(result, provider_id=self.provider_id, correlation_id="absolute_fixture",
            source_references=("external:absolute_fixture",), source_access_basis="external",
            nusselt=15*wall, alpha_inside=15*wall*state.bulk.k/result.reference.nusselt_length,
            f_darcy=.03, friction_factor_native=.0075, wall_correction=wall)
        return TwistedTapeClearanceResult(self.mode,
            geometry.clearance_for(state.tube_inner_diameter).diametral_clearance/state.tube_inner_diameter,
            "diametral_clearance / tube_inner_diameter", absolute=result)


def configured(model=None):
    return TubeSideEnhancement(NominalFixture(), TwistedTapeGeometry(.036,.011,.001),
                               "liquid", clearance_provider=model)


def diagnostics(result):
    return {d.name:d.value for d in result.diagnostics}


@pytest.mark.parametrize("mode", ["correction", "absolute"])
def test_composed_model_cannot_return_a_different_thermal_reference(mode):
    # Synthetic fixtures deliberately declare film but return the legacy
    # bulk-based Nu/alpha pair. Composition must enforce the same contract
    # as direct provider dispatch, before accepting the wrong conductivity.
    class MisdeclaredBase(NominalFixture):
        thermal_property_reference = "film"

    class MisdeclaredAbsolute(AbsoluteFixture):
        thermal_property_reference = "film"

    config = (replace(configured(CorrectionFixture()), provider=MisdeclaredBase())
              if mode == "correction" else configured(MisdeclaredAbsolute()))
    state = sample_input()
    state = replace(state, thermal=replace(state.bulk, k=.3))
    with pytest.raises(ValueError, match="provider declaration"):
        evaluate_enhancement(config, state)


@pytest.mark.parametrize("mode", ["correction", "absolute"])
def test_composed_model_preserves_a_valid_film_reference(mode):
    def with_film(result, state):
        return replace(result, thermal_property_reference="film",
                       alpha_inside=result.nusselt*state.thermal.k/result.reference.nusselt_length)

    class FilmBase(NominalFixture):
        thermal_property_reference = "film"

        def evaluate(self, geometry, state):
            return with_film(super().evaluate(geometry, state), state)

    class FilmAbsolute(AbsoluteFixture):
        thermal_property_reference = "film"

        def evaluate(self, geometry, state, base_result):
            model = super().evaluate(geometry, state, base_result)
            return replace(model, absolute=with_film(model.absolute, state))

    config = (replace(configured(CorrectionFixture()), provider=FilmBase())
              if mode == "correction" else configured(FilmAbsolute()))
    state = sample_input()
    state = replace(state, thermal=replace(state.bulk, k=.3))
    result = evaluate_enhancement(config, state)
    assert result.thermal_property_reference == "film"
    assert result.alpha_inside == pytest.approx(result.nusselt*.3/result.reference.nusselt_length)


@pytest.mark.parametrize("width,diametral", [(.012,0),(.011,.001),(.010,.002)])
def test_unambiguous_centered_gap_geometry(width, diametral):
    gap=TwistedTapeGeometry(.036,width,.001).clearance_for(.012)
    assert gap.diametral_clearance == pytest.approx(diametral)
    assert gap.radial_clearance == pytest.approx(diametral/2)
    assert gap.nominal_full_width is (diametral == 0)


@pytest.mark.parametrize("width", [0,-1,math.nan,math.inf,.0120001,.013])
def test_invalid_width_rejected_before_absolute_dispatch(width):
    with pytest.raises(ValueError):
        evaluate_enhancement(replace(configured(AbsoluteFixture()),
            geometry=TwistedTapeGeometry(.036,width,.001)),sample_input())


def test_finite_gap_no_silent_nominal_fallback_and_none_unchanged():
    with pytest.raises(EnhancementUnsupportedError,match="clearance_model_required"):
        evaluate_enhancement(configured(),sample_input())
    assert evaluate_enhancement(None,sample_input()) is None
    full=replace(configured(),geometry=TwistedTapeGeometry(.036,.012,.001))
    assert evaluate_enhancement(full,sample_input()) == full.provider.evaluate(full.geometry,sample_input())


def test_relative_results_provenance_and_fanning_conversion():
    state=sample_input()
    config=configured(CorrectionFixture())
    result=evaluate_enhancement(config,state)
    d=diagnostics(result)
    assert result.nusselt == 16
    assert result.f_darcy == pytest.approx(.048)
    assert result.f_fanning == pytest.approx(.012)
    assert result.reference.flow_area == pytest.approx(math.pi*.012**2/8)
    assert d['Nu_before_clearance']==20 and d['Nu_after_clearance']==16
    assert d['heat_transfer_factor']==.8 and d['friction_factor_factor']==.6
    assert d['base_tape_width']==.012 and d['tape_width']==.011
    assert result.source_access_basis=='private'
    assert result.source_references==('private:test_dataset','external:relative_fixture')
    assert result.applicability=='extrapolated'
    assert {'test_external_notice','clearance_fixture_warning'} <= {w.code for w in result.warnings}


def test_absolute_replaces_even_an_unsupported_base_without_evaluating_it():
    class ForbiddenBase:
        provider_id='never_evaluate_me'
        def evaluate(self,*args):
            raise AssertionError('An absolute clearance model cannot depend on base evaluation.')
    config=replace(configured(AbsoluteFixture()),provider=ForbiddenBase())
    result=evaluate_enhancement(config,sample_input())
    d=diagnostics(result)
    assert result.provider_id=='clearance_absolute_fixture'
    assert result.nusselt==15 and result.f_darcy==.03
    assert result.source_references==('external:absolute_fixture',)
    assert d['replaced_base_provider']=='never_evaluate_me'
    assert 'Nu_before_clearance' not in d and 'heat_transfer_factor' not in d


@pytest.mark.parametrize("change", [dict(compatible_base_correlation_ids=('other',)),dict(regime='different')])
def test_incompatible_relative_normalization_rejected(change):
    class Bad(CorrectionFixture):
        def evaluate(self,*args):
            r=super().evaluate(*args)
            return replace(r,correction=replace(r.correction,**change))
    with pytest.raises(EnhancementUnsupportedError,match='incompatible'):
        evaluate_enhancement(configured(Bad()),sample_input())


@pytest.mark.parametrize("mode", [CorrectionFixture,AbsoluteFixture])
def test_explicit_clearance_failure_propagates_at_finite_and_zero_gap(mode):
    class Reject(mode):
        def evaluate(self,*args):
            raise EnhancementUnsupportedError('fixture_transition_clearance_unsupported')
    for width in (.011,.012):
        config=replace(configured(Reject()),geometry=TwistedTapeGeometry(.036,width,.001))
        with pytest.raises(EnhancementUnsupportedError,match='fixture_transition_clearance_unsupported'):
            evaluate_enhancement(config,sample_input())


@pytest.mark.parametrize("change", [dict(heat_transfer_factor=0),dict(friction_factor_factor=math.nan),
    dict(compatible_base_correlation_ids=('*',)),dict(source_references=()),
    dict(applicability='extrapolated',warnings=())])
def test_correction_contract_rejects_incomplete_or_invalid_result(change):
    c=CorrectionFixture().evaluate(configured().geometry,sample_input(),
        FakeExternalProvider().evaluate('test_geometry',sample_input())).correction
    with pytest.raises((ValueError,TypeError)):
        replace(c,**change)


def test_mode_payload_and_provider_mode_must_agree():
    result=AbsoluteFixture().evaluate(configured().geometry,sample_input(),None)
    with pytest.raises(TypeError):
        replace(result,mode=ClearanceModelMode.CORRECTION)
    class Bad(AbsoluteFixture):
        mode='absolute'
    with pytest.raises(TypeError):
        configured(Bad())
    class Wrong(AbsoluteFixture):
        def evaluate(self,*args):
            return replace(super().evaluate(*args),absolute=replace(result.absolute,provider_id='wrong'))
    with pytest.raises(ValueError,match='identity'):
        evaluate_enhancement(configured(Wrong()),sample_input())


@pytest.mark.parametrize("factory", [CorrectionFixture,AbsoluteFixture])
@pytest.mark.parametrize("iterate", [False,True])
def test_clearance_through_real_simulation_wall_and_distributed_hydraulics(factory,iterate):
    b=bundle(passes=2)
    hx=BareTubeHeatExchanger(b)
    sides=inputs(b,provider=VariableLiquid())
    result=hx.simulate(*sides,iterate=iterate,tube_side_enhancement=configured(factory()))
    e=result.tube_side_enhancement
    assert result.inside_alfa_mean==e.alpha_inside
    assert e.provider_id==factory.provider_id
    if iterate:
        assert result.thermal_state.diagnostics.inside_combined_correction==e.wall_correction
        assert e.wall_correction == (.9 if factory is CorrectionFixture else .85)
    else:
        assert result.thermal_state is None and e.wall_correction == 1
    points=[getattr(result,'inside_properties_'+p) for p in ('inlet','midpoint','outlet')]
    gradients=[]
    for point in points:
        velocity=sides[0].m_dot/b.n_tubes_per_pass_effective/(point.rho*math.pi*.012**2/8)
        f=.048 if factory is CorrectionFixture else .03
        gradients.append(f*point.rho*velocity**2/(2*.006))
        assert point.friction_factor==pytest.approx(f)
        assert point.enhancement.provider_id==e.provider_id
    assert result.inside_dp_friction==pytest.approx(b.internal_length_total*(gradients[0]+4*gradients[1]+gradients[2])/6)
    assert result.inside_dp_total==pytest.approx(sum(getattr(result,f) for f in (
        'inside_dp_friction','inside_dp_acceleration','inside_dp_local',
        'inside_dp_tube_entrances','inside_dp_tube_exits')))
    assert hx.tube_side_enhancement is None


@pytest.mark.parametrize("factory", [CorrectionFixture,AbsoluteFixture])
def test_rating_bridge_area_and_local_terms(factory):
    b=bundle()
    hx=BareTubeHeatExchanger(b)
    i,o=inputs(b)
    sides=(BalanceSideSpec(provider=i.provider,p=i.p,m_dot=i.m_dot,T_in=300,T_out=301),
           BalanceSideSpec(provider=o.provider,p=o.p,m_dot=o.m_dot,T_in=340))
    result=hx.rate(*sides,tube_side_enhancement=configured(factory()),include_simulation=True)
    base=hx.rate(*sides,tube_side_enhancement=replace(configured(),geometry=TwistedTapeGeometry(.036,.012,.001)))
    assert result.A_required>base.A_required
    assert result.Q_required==base.Q_required
    assert result.alfa_i<base.alfa_i
    assert result.inside_dp_friction<base.inside_dp_friction
    for field in ('inside_dp_acceleration','inside_dp_local','inside_dp_tube_entrances','inside_dp_tube_exits'):
        assert getattr(result,field)==getattr(base,field)
    assert result.simulation.tube_side_enhancement.provider_id==factory.provider_id


def test_public_base_provider_can_use_same_architecture():
    from core.enhancements import Yang2020TwistedTapeProvider
    config=replace(configured(CorrectionFixture()),provider=Yang2020TwistedTapeProvider())
    result=evaluate_enhancement(config,sample_input())
    assert diagnostics(result)['base_enhancement_provider']=='yang_2020_twisted_tape'
