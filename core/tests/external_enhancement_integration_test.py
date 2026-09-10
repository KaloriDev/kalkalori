# KalKalori - GNU GPL v3 only
"""An external package only needs the public contract, not solver changes."""
from dataclasses import dataclass, replace
import math

import pytest

from core.enhancements import (
    EnhancementResult, TubeSideEnhancement, TwistedTapeGeometry,
    EnhancementUnsupportedError,
)
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.tests.enhancement_provider_contract_test import FakeExternalProvider
from core.tests.enhancement_integration_test import bundle, inputs, VariableLiquid


@dataclass(frozen=True)
class PrivateFixtureDetails:
    revision: int
    dataset: str


@dataclass(frozen=True)
class PrivateFixtureResult(EnhancementResult):
    private_details: PrivateFixtureDetails | None = None


class ExternalTapeFixture(FakeExternalProvider):
    """No M&B/manufacturer equations: fixed synthetic fixture values only."""
    def __init__(self):
        self.requests = []

    def evaluate(self, geometry, state):
        if not isinstance(geometry, TwistedTapeGeometry):
            raise EnhancementUnsupportedError("external_fixture_wrong_geometry")
        self.requests.append(state)
        base = super().evaluate("test_geometry", state)
        # Deliberately alpha-only; external metadata stays typed and opaque.
        return PrivateFixtureResult(**{
            **vars(base), "nusselt": None, "applicability": "extrapolated",
            "private_details": PrivateFixtureDetails(7, "synthetic_private_fixture"),
        })


@pytest.mark.parametrize("iterate", [False, True])
def test_external_alpha_only_provider_through_simulation(iterate):
    b = bundle()
    provider = ExternalTapeFixture()
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=TubeSideEnhancement(
        provider, TwistedTapeGeometry(.05, .01, .0005), "liquid"))
    result = hx.simulate(*inputs(b, provider=VariableLiquid()), iterate=iterate)
    enhanced = result.tube_side_enhancement
    assert result.inside_alfa_mean == enhanced.alpha_inside
    assert enhanced.nusselt is None
    assert enhanced.applicability == "extrapolated"
    assert enhanced.source_access_basis == "private"
    assert enhanced.private_details == PrivateFixtureDetails(7, "synthetic_private_fixture")
    assert "test_external_notice" in {w.code for w in result.warnings}
    assert {"thermal", "inlet", "midpoint", "outlet"} <= {s.position for s in provider.requests}
    if iterate:
        assert any(s.wall is not None for s in provider.requests)
        assert result.thermal_state.diagnostics.inside_alfa_corrected == enhanced.alpha_inside
    # Same fixture result and its own area/D at all three hydraulic nodes.
    points = [result.inside_properties_inlet, result.inside_properties_midpoint,
              result.inside_properties_outlet]
    gradients = []
    for point in points:
        assert point.enhancement.private_details == enhanced.private_details
        assert point.enhancement.friction_basis == "fanning"
        area = math.pi*.012**2/8
        velocity = inputs(b)[0].m_dot/b.n_tubes_per_pass_effective/(point.rho*area)
        gradients.append(.08*point.rho*velocity**2/(2*.006))
    assert result.inside_dp_friction == pytest.approx(.3*(gradients[0]+4*gradients[1]+gradients[2])/6)


def test_external_provider_rating_and_simulation_bridge():
    b = bundle()
    provider = ExternalTapeFixture()
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=TubeSideEnhancement(
        provider, TwistedTapeGeometry(.05, .01, .0005), "liquid"))
    inside, outside = inputs(b)
    result = hx.rate(
        BalanceSideSpec(provider=inside.provider, p=inside.p, m_dot=inside.m_dot,
                        T_in=300, T_out=301),
        BalanceSideSpec(provider=outside.provider, p=outside.p, m_dot=outside.m_dot, T_in=340),
        include_simulation=True,
    )
    assert result.alfa_i == result.tube_side_enhancement.alpha_inside
    assert result.tube_side_enhancement.source_references == ("private:test_dataset",)
    assert result.inside_properties_outlet.enhancement.correlation_id == result.tube_side_enhancement.correlation_id
    assert "test_external_notice" in {w.code for w in result.warnings}
    assert result.overdesign_factor == pytest.approx(result.UA_actual/result.UA_process-1)


def test_external_model_mismatch_is_rejected_by_real_solver():
    class InconsistentProvider(ExternalTapeFixture):
        def evaluate(self, geometry, state):
            result = super().evaluate(geometry, state)
            if state.position != "thermal":
                return replace(result, correlation_id="different_model")
            return result
    b = bundle()
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=TubeSideEnhancement(
        InconsistentProvider(), TwistedTapeGeometry(.05, .01, .0005), "liquid"))
    with pytest.raises(ValueError, match="inconsistent"):
        hx.simulate(*inputs(b))


def test_external_unsupported_is_not_swallowed_by_rating():
    class RejectingProvider(ExternalTapeFixture):
        def evaluate(self, geometry, state):
            raise EnhancementUnsupportedError("external_fixture_out_of_range")
    b = bundle()
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=TubeSideEnhancement(
        RejectingProvider(), TwistedTapeGeometry(.05, .01, .0005), "liquid"))
    inside, outside = inputs(b)
    with pytest.raises(EnhancementUnsupportedError, match="external_fixture_out_of_range"):
        hx.rate(BalanceSideSpec(provider=inside.provider, p=inside.p, m_dot=inside.m_dot,
                               T_in=300, T_out=301),
                BalanceSideSpec(provider=outside.provider, p=outside.p, m_dot=outside.m_dot, T_in=340))


@pytest.mark.parametrize("field,value", [("correlation_id", "changed_model"),
                                        ("source_access_basis", "external")])
def test_refreshed_hydraulics_cannot_change_thermal_model(field, value):
    class ChangingProvider(ExternalTapeFixture):
        def __init__(self):
            super().__init__()
            self.hydraulic_requests = 0

        def evaluate(self, geometry, state):
            result = super().evaluate(geometry, state)
            if state.position != "thermal":
                self.hydraulic_requests += 1
                # Initial snapshot is consistent; all refreshed nodes then
                # switch together. Hydraulic-node-only checks cannot catch it.
                if self.hydraulic_requests > 3:
                    return replace(result, **{field: value})
            return result

    b = bundle()
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=TubeSideEnhancement(
        ChangingProvider(), TwistedTapeGeometry(.05, .01, .0005), "liquid"))
    with pytest.raises(ValueError, match="inconsistent"):
        hx.simulate(*inputs(b), iterate=False)


def test_hydraulic_nodes_cannot_change_source_access_basis():
    class ChangingProvenance(ExternalTapeFixture):
        def evaluate(self, geometry, state):
            result = super().evaluate(geometry, state)
            if state.position == "outlet":
                return replace(result, source_access_basis="external")
            return result

    from core.enhancements.integration import hydraulic_evaluator
    from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
    b = bundle()
    inside, _ = inputs(b)
    config = TubeSideEnhancement(ChangingProvenance(), TwistedTapeGeometry(.05, .01, .0005), "liquid")
    with pytest.raises(ValueError, match="inconsistent"):
        calculate_tube_bundle_hydraulics(
            m_dot=inside.m_dot, flow_area_per_pass=b.internal_flow_area_per_pass,
            hydraulic_diameter=b.internal_hydraulic_diameter,
            hydraulic_length_total=b.internal_length_total,
            n_tube_passes=b.n_passes_tube, tube_path_type=b.tube_path_type,
            provider=inside.provider, temperature_in=300, temperature_out=301, pressure=inside.p,
            enhancement_evaluator=hydraulic_evaluator(config, b, inside.provider))


@pytest.mark.parametrize("iterate", [False, True])
def test_external_provider_can_support_dry_gas(iterate):
    from core.properties.dry_air import DryAirPropertyProvider
    b = bundle()
    provider = ExternalTapeFixture()
    inside, outside = inputs(b)
    inside = replace(inside, provider=DryAirPropertyProvider(), m_dot=.02)
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=TubeSideEnhancement(
        provider, TwistedTapeGeometry(.05, .01, .0005), "gas"))
    result = hx.simulate(inside, outside, iterate=iterate)
    assert result.q > 0
    assert result.tube_side_enhancement.provider_id == provider.provider_id
    assert all(state.fluid_phase == "gas" for state in provider.requests)


class NotebookFixtureProvider(ExternalTapeFixture):
    """A notebook-style object using only public contract inputs/outputs.

    The temperature multiplier is synthetic test arithmetic, not an
    engineering correlation or any part of Manglik-Bergles.
    """
    def evaluate(self, geometry, state):
        result = super().evaluate(geometry, state)
        factor = 1 if state.wall is None else 1 + .001*(
            state.wall.temperature - state.bulk.temperature)
        return replace(result, alpha_inside=result.alpha_inside*factor,
                       wall_correction=factor)


def notebook_configuration(provider):
    return TubeSideEnhancement(provider, TwistedTapeGeometry(.05, .01, .0005), "liquid")


def notebook_solve(hx, mode, inside, outside, **kwargs):
    if mode == "rating":
        return hx.rate(
            BalanceSideSpec(provider=inside.provider, p=inside.p, m_dot=inside.m_dot,
                            T_in=inside.T_in, T_out=301),
            BalanceSideSpec(provider=outside.provider, p=outside.p, m_dot=outside.m_dot,
                            T_in=outside.T_in), include_simulation=True, **kwargs)
    return hx.simulate(inside, outside, iterate=mode != "snapshot", **kwargs)


@pytest.mark.parametrize("mode", ["rating", "simulation", "snapshot"])
def test_notebook_object_selected_per_call_owns_thermal_and_hydraulic_results(mode):
    b = bundle(passes=2)
    provider = NotebookFixtureProvider()
    config = notebook_configuration(provider)
    hx = BareTubeHeatExchanger(b)
    inside, outside = inputs(b, provider=VariableLiquid())
    result = notebook_solve(hx, mode, inside, outside, tube_side_enhancement=config)
    assert hx.tube_side_enhancement is None
    assert config.provider is provider
    enhanced = result.tube_side_enhancement
    assert enhanced.source_access_basis == "private"
    assert enhanced.source_references == ("private:test_dataset",)
    assert enhanced.private_details.dataset == "synthetic_private_fixture"
    assert "test_external_notice" in {w.code for w in result.warnings}

    area = math.pi*.012**2/4
    flow = inside.m_dot/b.n_tubes_per_pass_effective
    for state in provider.requests:
        assert state.mass_flow_per_tube == pytest.approx(flow)
        assert state.base_flow_area_per_tube == pytest.approx(area)
        assert state.base_mass_flux == pytest.approx(flow/area)
        assert state.heated_length == state.tube_length == .3
        assert state.hydraulic_length_total == .6
        assert state.tube_inner_diameter == .012
        assert state.bulk.temperature is not None
        assert state.bulk.pressure == inside.p

    points = (result.inside_properties_inlet, result.inside_properties_midpoint,
              result.inside_properties_outlet)
    gradients = []
    for point in points:
        assert point.enhancement.f_darcy == .08
        assert any(s.position == point.position and s.bulk.temperature == point.temperature
                   and s.bulk.rho == point.rho and s.bulk.mu == point.mu
                   for s in provider.requests)
        # Independent Darcy pressure gradient on the synthetic half-area,
        # half-diameter reference, integrated over both tube passes.
        gradients.append(.08*point.rho*(flow/(point.rho*area/2))**2/(2*.006))
    assert result.inside_dp_friction == pytest.approx(.6*(gradients[0]+4*gradients[1]+gradients[2])/6)
    if mode == "rating":
        assert result.simulation is not None
        assert result.simulation.tube_side_enhancement.provider_id == provider.provider_id
        assert result.simulation.inside_properties_outlet.enhancement.source_access_basis == "private"

    if mode != "snapshot":
        thermal = result.thermal_state
        assert thermal.converged
        matches = [s for s in provider.requests if s.wall is not None
                   and s.bulk.mu == thermal.inside_bulk_props.mu
                   and s.wall.mu == thermal.inside_wall_props.mu]
        assert matches
        state = matches[-1]
        factor = 1 + .001*(state.wall.temperature-state.bulk.temperature)
        assert factor != 1
        assert enhanced.wall_correction == pytest.approx(factor)
        assert enhanced.alpha_inside == pytest.approx(20*state.bulk.k/.012*factor)
        assert thermal.diagnostics.inside_combined_correction == pytest.approx(factor)
    else:
        assert result.thermal_state is None
    # Omission after an override still follows the original smooth default.
    assert notebook_solve(hx, mode, inside, outside).tube_side_enhancement is None


@pytest.mark.parametrize("mode", ["rating", "simulation", "snapshot"])
def test_notebook_override_inherit_and_explicit_none_are_isolated(mode):
    b = bundle()
    inside, outside = inputs(b)
    default = notebook_configuration(ExternalTapeFixture())
    hx = BareTubeHeatExchanger(b, tube_side_enhancement=default)
    inherited = notebook_solve(hx, mode, inside, outside)
    assert inherited.tube_side_enhancement is not None
    before = len(default.provider.requests)
    smooth = notebook_solve(hx, mode, inside, outside, tube_side_enhancement=None)
    legacy = notebook_solve(BareTubeHeatExchanger(b), mode, inside, outside)
    assert smooth.tube_side_enhancement is None
    assert smooth.warnings == legacy.warnings
    assert smooth.final_result.tube_side_thermal == legacy.final_result.tube_side_thermal
    for name in ("inside_dp_friction", "inside_dp_acceleration", "overdesign_factor"):
        assert getattr(smooth, name) == getattr(legacy, name)
    assert len(default.provider.requests) == before
    assert hx.tube_side_enhancement is default

    class RejectingNotebookProvider(ExternalTapeFixture):
        def evaluate(self, geometry, state):
            raise EnhancementUnsupportedError("notebook_fixture_unsupported")

    with pytest.raises(EnhancementUnsupportedError, match="notebook_fixture_unsupported"):
        notebook_solve(hx, mode, inside, outside,
                       tube_side_enhancement=notebook_configuration(RejectingNotebookProvider()))
    with pytest.raises(TypeError, match="configuration"):
        notebook_solve(hx, mode, inside, outside, tube_side_enhancement=object())
    assert hx.tube_side_enhancement is default
    assert len(default.provider.requests) == before
