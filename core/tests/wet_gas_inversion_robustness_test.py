"""Caloric domain, bounded roots and explicit numerical failure classification."""
import math

import pytest

from core.phase_change import wet_gas_enthalpy as caloric
from core.phase_change.condensation_solver_helpers import invert_wet_gas_enthalpy
from core.tests.phase_change_wet_gas_enthalpy_test import _capability


@pytest.mark.parametrize("T,W", [(273.160001, 0.), (280., .001),
                                (320., .04), (560., .12), (623.149, .04)])
def test_supported_domain_round_trip_including_small_approaches(T, W):
    cap = _capability()
    evaluator = caloric.WetGasEnthalpyEvaluator(101325., cap, reuse_dry_backend_state=True)
    h = evaluator.enthalpy(T, W)
    answer = caloric.temperature_from_h_wet_gas_dry_basis(h, 101325., W, cap,
        T_bracket=(273.16, 623.1499), tolerance_K=1e-7, evaluator=evaluator)
    assert answer == pytest.approx(T, abs=1e-7)


@pytest.mark.parametrize("kw", [dict(h_target=math.nan), dict(W=-.01),
    dict(T_bracket=(260., 300.)), dict(T_bracket=(300., 650.)),
    dict(T_bracket=(350., 300.)), dict(tolerance_K=0.)])
def test_invalid_domains_are_classified_before_property_evaluation(kw):
    inputs = dict(h_target=1., p=101325., W=.01, capability=_capability())
    inputs.update(kw)
    with pytest.raises(caloric.WetGasEnthalpyError) as error:
        caloric.temperature_from_h_wet_gas_dry_basis(**inputs)
    assert error.value.reason == "invalid_input"


def test_iteration_exhaustion_is_not_reported_as_a_temperature():
    cap = _capability()
    h = caloric.h_wet_gas_dry_basis(333.3333, 101325., .01, cap)
    with pytest.raises(caloric.WetGasEnthalpyError) as error:
        caloric.temperature_from_h_wet_gas_dry_basis(h, 101325., .01, cap,
            max_iterations=1, tolerance_K=1e-9)
    assert error.value.reason == "numerical_stagnation"


def test_nonmonotone_mapping_is_distinct_from_a_bad_bracket(monkeypatch):
    monkeypatch.setattr(caloric, "h_wet_gas_dry_basis", lambda T, *a: -T)
    with pytest.raises(caloric.WetGasEnthalpyError) as error:
        caloric.temperature_from_h_wet_gas_dry_basis(-310., 101325., .01,
                                                   _capability(), T_bracket=(300., 330.))
    assert error.value.reason == "non_monotonic_mapping"


def test_backend_failure_is_not_retried_in_a_wider_domain(monkeypatch):
    calls = []
    def unavailable(T, *a):
        calls.append(T)
        raise ValueError("backend failed")
    monkeypatch.setattr(caloric, "h_wet_gas_dry_basis", unavailable)
    with pytest.raises(caloric.WetGasEnthalpyError) as error:
        invert_wet_gas_enthalpy(h_target=1., p_wet_gas=101325., W=.01,
            capability=_capability(), T_wall_mean=310., T_in_wet_gas=360.)
    assert error.value.reason == "property_backend_failure"
    assert len(calls) == 1


def test_supersaturated_outlet_is_excluded_by_its_own_dewpoint():
    cap = _capability()
    h = caloric.h_wet_gas_dry_basis(280., 101325., .04, cap)
    with pytest.raises(caloric.WetGasEnthalpyError) as error:
        invert_wet_gas_enthalpy(h_target=h, p_wet_gas=101325., W=.04,
            capability=cap, T_wall_mean=310., T_in_wet_gas=360.)
    assert error.value.reason == "bad_bracket"
