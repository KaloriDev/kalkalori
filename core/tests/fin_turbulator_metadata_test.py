# SPDX-License-Identifier: GPL-3.0-only
"""Per-fin metadata never selects an implicit enhancement law."""
from copy import deepcopy
from dataclasses import asdict, replace, FrozenInstanceError
import json
import pytest

from core.geometry import RectangularIndividualFin
from core.heat_transfer import Zhao2017AirsideModel
from core.tests.zhao2017_fixtures import source_bundle, AIR


def with_count(count):
    b=source_bundle()
    return replace(b,tube=replace(b.tube,fin=replace(b.tube.fin,fin_turbulator_count=count)))


@pytest.mark.parametrize('count',[None,0,4,8,10])
def test_optional_count_is_per_fin_metadata(count):
    old=source_bundle().tube.fin  # Historical five-argument construction.
    assert old.fin_turbulator_count is None and old.effective_fin_turbulator_count==0
    b=with_count(count)
    f=b.tube.fin
    assert f.effective_fin_turbulator_count==(0 if count is None else count)
    assert b.total_outer_area==source_bundle().total_outer_area
    assert b.tube.projected_blocking_area_per_length==source_bundle().tube.projected_blocking_area_per_length


@pytest.mark.parametrize('count',[-1,True,False,4.,4.5,float('nan'),float('inf'),'4','',complex(4)])
def test_invalid_count_is_not_coerced(count):
    with pytest.raises(ValueError,match='fin_turbulator_count'):
        with_count(count)


def test_serialization_copy_hash_and_geometry_identity():
    f=with_count(4).tube.fin
    assert deepcopy(f)==f and hash(deepcopy(f))==hash(f)
    assert RectangularIndividualFin(**json.loads(json.dumps(asdict(f))))==f
    old=asdict(f);old.pop('fin_turbulator_count')
    assert RectangularIndividualFin(**old).effective_fin_turbulator_count==0
    assert len({f,replace(f,fin_turbulator_count=10)})==2
    with pytest.raises(FrozenInstanceError):f.fin_turbulator_count=10


def test_omitted_none_zero_baselines_are_identical_including_diagnostics():
    model=Zhao2017AirsideModel()
    omitted=model.evaluate(source_bundle(),1.7,AIR)
    assert model.evaluate(with_count(None),1.7,AIR)==omitted
    assert model.evaluate(with_count(0),1.7,AIR)==omitted
    assert omitted.fin_turbulator_count==0 and not omitted.correction_applied
    assert not any(w.code=='fin_turbulators_present_uncorrected' for w in omitted.warnings)


@pytest.mark.parametrize('count',[4,10])
def test_positive_count_baseline_does_not_depend_on_passes(count):
    b=with_count(count);model=Zhao2017AirsideModel()
    r=model.evaluate(b,1.7,AIR);smooth=model.evaluate(source_bundle(),1.7,AIR)
    assert r.h_film==smooth.h_film and r.pressure_drop==smooth.pressure_drop
    assert r.factors==smooth.factors and r.flow==smooth.flow
    assert r.fin_turbulator_count==count and not r.correction_applied
    assert any(w.code=='fin_turbulators_present_uncorrected' for w in r.warnings)
    assert model.evaluate(replace(b,n_passes_tube=4),1.7,AIR)==r


@pytest.mark.parametrize('count',[None,0,4,10])
def test_count_runs_normal_simulation_and_rating(count):
    from core.models.bare_tube import BareTubeHeatExchanger
    from core.models.heat_balance import BalanceSideSpec
    from core.tests.zhao2017_integration_test import context
    template,inside,outside=context()
    hx=BareTubeHeatExchanger(with_count(count),
        elliptical_flow_assumptions=template.elliptical_flow_assumptions,
        tube_wall_model=template.tube_wall_model)
    result=hx.simulate(inside,outside)
    assert result.q==pytest.approx(24058.39114704213,rel=2e-9)
    assert result.outside_dp_total==pytest.approx(372.055553752331,rel=2e-12)
    a=BalanceSideSpec(provider=inside.provider,m_dot=inside.m_dot,T_in=inside.T_in,
                      T_out=result.T_out_inside,p=inside.p)
    b=BalanceSideSpec(provider=outside.provider,m_dot=outside.m_dot,T_in=outside.T_in,
                      T_out=result.T_out_outside,p=outside.p)
    rating=hx.rate(a,b,include_simulation=False)
    assert rating.Q_required==pytest.approx(result.q)
    assert rating.outside_dp_total==result.outside_dp_total
    assert result.finned_tube_diagnostics.airside_result.fin_turbulator_count==(count or 0)
