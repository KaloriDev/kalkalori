# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import FrozenInstanceError, replace
import pytest
from core.tests.continuous_plate_fin_geometry_test import bank
from core.heat_transfer.airside_correction import AirsideCorrection, MultiplicativeAirsideCorrection
from core.heat_transfer.continuous_airside import ExperimentalWang2000Model
from core.heat_transfer.continuous_fin_source import ContinuousFinSourceError, ContactPolicy
from core.heat_transfer.continuous_fin_efficiency import SchmidtEquivalentCell
from core.properties.common import FluidTransportProperties

AIR = FluidTransportProperties(1.18, 1.85e-5, .026, 1007.)


def test_efficiency_limits_and_both_pitches():
    b, model = bank(), SchmidtEquivalentCell()
    low, normal = model.evaluate(b, 0.), model.evaluate(b, 60.)
    assert low.fin_efficiency == 1.
    assert 0 < normal.fin_efficiency < 1.
    high_k = replace(b, continuous_fin_geometry=replace(b.continuous_fin_geometry, conductivity=1e20))
    assert model.evaluate(high_k, 60.).fin_efficiency == pytest.approx(1.)
    longer = replace(b, pitch_longitudinal=.024, continuous_fin_geometry=replace(b.continuous_fin_geometry, depth=.075))
    assert model.evaluate(longer, 60.).equivalent_radius > normal.equivalent_radius
    assert normal.tip_temperature_ratio is None
    assert normal.overall_surface_efficiency == pytest.approx(normal.effective_area/b.total_outer_area)
    assert "finite_edge_cells_unresolved" in normal.cell_basis


def test_correct_once_before_eta_and_keep_source_status():
    b = bank()
    base = ExperimentalWang2000Model(True).evaluate(b, .005, AIR)
    model = ExperimentalWang2000Model(True, MultiplicativeAirsideCorrection(2., 3., "synthetic_test"))
    r = model.evaluate(b, .005, AIR)
    again = model.evaluate(b, .005, AIR)
    assert r == again
    assert r.baseline == base.baseline
    assert r.alpha == 2*base.alpha
    assert r.core_friction_estimate == 3*base.core_friction_estimate
    assert r.efficiency.fin_efficiency < base.efficiency.fin_efficiency
    assert r.metadata.source_status == "SOURCE_PARTIALLY_CLOSED"
    assert r.warnings == base.warnings
    with pytest.raises(FrozenInstanceError):
        r.baseline.heat_transfer.alpha = 5.


@pytest.mark.parametrize("args", [{}, dict(acknowledge_source_limits=True, contact_resistance=.001),
    dict(acknowledge_source_limits=True, contact_policy=ContactPolicy.IDEAL_CONTACT)])
def test_opt_in_and_empirical_contact(args):
    with pytest.raises(ContinuousFinSourceError):
        ExperimentalWang2000Model(**args)


def test_bad_correction_does_not_fallback():
    class Wrong:
        def correct(self, *a, **kw): return None
    with pytest.raises(TypeError, match="AirsideCorrection"):
        ExperimentalWang2000Model(True, Wrong()).evaluate(bank(), .005, AIR)
    class Failing:
        def correct(self, *a, **kw): raise RuntimeError("correction failure")
    with pytest.raises(RuntimeError, match="correction failure"):
        ExperimentalWang2000Model(True, Failing()).evaluate(bank(), .005, AIR)


def test_replacements_are_independent_and_baseline_is_frozen():
    class Replace:
        def correct(self, base_result, **kwargs):
            assert base_result.heat_transfer.alpha > 0.
            return AirsideCorrection("replacement_test", h_film_replacement=50., pressure_drop_replacement=2.)
    r = ExperimentalWang2000Model(True, Replace()).evaluate(bank(), .005, AIR)
    assert r.alpha == 50. and r.core_friction_estimate == 2.
