# KalKalori - GNU GPL v3 only
"""Synthetic providers test a public contract, never private physics."""
from dataclasses import replace
import math

import pytest

from core.enhancements import (
    EnhancementDiagnostic, EnhancementInput, EnhancementReferenceState,
    EnhancementResult, EnhancementState, EnhancementUnsupportedError,
    TubeSideEnhancement, TwistedTapeGeometry, evaluate_enhancement,
)
from core.common.warnings import make_warning


class FakeExternalProvider:
    provider_id = "test_external"

    def evaluate(self, geometry, state):
        if geometry != "test_geometry":
            raise EnhancementUnsupportedError("incompatible_geometry")
        # Deliberately uses a different reference area AND diameter. The
        # distributed hydraulic solver must honor these together.
        area = math.pi*state.tube_inner_diameter**2/8
        velocity = state.mass_flow_per_tube/(state.bulk.rho*area)
        ref = EnhancementReferenceState(area, velocity, 1234.0, 8.0,
                                        state.tube_inner_diameter/2,
                                        state.tube_inner_diameter/3,
                                        state.tube_inner_diameter)
        return EnhancementResult(
            self.provider_id, "test_fixture_only", ("private:test_dataset",),
            "private", ref, 20*state.bulk.k/ref.nusselt_length,
            0.08, 0.02, "fanning", "test_regime", nusselt=20,
            warnings=(make_warning(code="test_external_notice", message="Fixture notice",
                                   source=self.provider_id),),
            diagnostics=(EnhancementDiagnostic("test_dataset_revision", "one"),),
        )


def sample_input():
    return EnhancementInput(0.01, 0.012, 0.3, 0.3,
                            EnhancementState(1000, 0.001, 0.5, 4000, 300, 1e5),
                            fluid_phase="liquid")


def fake_configuration():
    return TubeSideEnhancement(FakeExternalProvider(), "test_geometry", "liquid")


def test_no_enhancement_does_not_dispatch():
    assert evaluate_enhancement(None, sample_input()) is None


def test_external_contract_conventions_provenance_and_reference_geometry():
    state = sample_input()
    result = evaluate_enhancement(fake_configuration(), state)
    assert result.source_access_basis == "private"
    assert result.source_references == ("private:test_dataset",)
    assert result.warnings[0].code == "test_external_notice"
    assert result.diagnostics[0].value == "one"
    assert result.f_fanning == 0.02
    assert result.f_darcy == 0.08
    # Hand: A=pi*.012^2/8, v=.01/(1000*A), D_ref=.006.
    assert result.pressure_gradient(1000) == pytest.approx(208.4798017332055)


@pytest.mark.parametrize("field", ["half_turn_length", "tape_width", "tape_thickness"])
@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_geometry_rejects_invalid(field, value):
    values = dict(half_turn_length=.036, tape_width=.012, tape_thickness=.001)
    values[field] = value
    with pytest.raises(ValueError):
        TwistedTapeGeometry(**values)


def test_unambiguous_half_turn_and_derived_diameter_ratio():
    tape = TwistedTapeGeometry(.036, .011, .001)
    assert tape.twist_ratio_for(.012) == pytest.approx(3)
    assert tape.twist_ratio_for(.018) == 2
    with pytest.raises(ValueError):
        tape.twist_ratio_for(0)
    with pytest.raises(ValueError):
        TwistedTapeGeometry(.03, .001, .002)


def test_errors_do_not_fall_back():
    with pytest.raises(EnhancementUnsupportedError, match="incompatible_geometry"):
        evaluate_enhancement(replace(fake_configuration(), geometry="wrong"), sample_input())
    with pytest.raises(EnhancementUnsupportedError, match="phase"):
        evaluate_enhancement(fake_configuration(), replace(sample_input(), fluid_phase="gas"))


@pytest.mark.parametrize("change", [
    {"f_darcy": .02}, {"friction_basis": "unknown"},
    {"alpha_inside": math.nan}, {"f_darcy": -1},
    {"nusselt": 0}, {"wall_correction": 0}, {"source_references": ()},
    {"source_access_basis": "unverified"}, {"provider_id": ""},
    {"applicability": "unsupported"}, {"warnings": []},
    {"applicability": "extrapolated", "warnings": ()},
])
def test_invalid_result_is_rejected(change):
    result = FakeExternalProvider().evaluate("test_geometry", sample_input())
    with pytest.raises((ValueError, TypeError)):
        replace(result, **change)


@pytest.mark.parametrize("change", [
    {"provider_id": "different"}, {"alpha_inside": 1},
])
def test_dispatch_validates_coherence(change):
    class BadProvider(FakeExternalProvider):
        def evaluate(self, geometry, state):
            return replace(super().evaluate(geometry, state), **change)
    with pytest.raises(ValueError):
        evaluate_enhancement(TubeSideEnhancement(BadProvider(), "test_geometry", "liquid"),
                             sample_input())


def test_reference_velocity_must_match_provider_area():
    class BadProvider(FakeExternalProvider):
        def evaluate(self, geometry, state):
            result = super().evaluate(geometry, state)
            return replace(result, reference=replace(result.reference, velocity=1))
    with pytest.raises(ValueError, match="area"):
        evaluate_enhancement(TubeSideEnhancement(BadProvider(), "test_geometry", "liquid"),
                             sample_input())


def test_configuration_rejects_nonprovider_and_unsupported_phase():
    with pytest.raises(TypeError):
        TubeSideEnhancement(object(), "geometry", "liquid")
    with pytest.raises(EnhancementUnsupportedError):
        TubeSideEnhancement(FakeExternalProvider(), "test_geometry", "two_phase")


@pytest.mark.parametrize("field", ["base_flow_area_per_tube", "hydraulic_length_total"])
@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_solver_reference_inputs_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(sample_input(), **{field: value})


def test_standalone_reference_inputs_are_optional_and_mass_flux_is_derived():
    assert sample_input().base_mass_flux is None
    state = replace(sample_input(), base_flow_area_per_tube=.002, hydraulic_length_total=1.2)
    assert state.base_mass_flux == 5
