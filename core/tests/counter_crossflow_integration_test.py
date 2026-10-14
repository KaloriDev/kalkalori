"""Dry call-path consistency and geometry/topology independence."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.geometry._thermal_topology import (
    CounterCrossflowTopology, resolve_thermal_topology, reject_phase_change_topology,
)
from core.heat_transfer.ntu import effectiveness_ntu, ntu_from_effectiveness
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


def exchanger(finned=False, **changes):
    tube = BareTube(D_i=.02, D_o=.024, length_total=2., length_effective=1.9, wall_k=20.)
    if finned:
        tube = CircularFinnedTube(core_tube=tube, D_root=.024, D_fin=.04,
            fin_thickness_root=.0004, fin_thickness_tip=.0004, fin_pitch=.003,
            fin_k=200., fin_contact_efficiency=.95)
    inputs = dict(tube=tube, n_rows=6, n_tubes_per_row=9.5,
                  pitch_transverse=.06, pitch_longitudinal=.052,
                  layout="staggered", n_passes_tube=4, n_passes_transverse=2)
    inputs.update(changes)
    return BareTubeHeatExchanger(TubeBundle(**inputs))


def sides(hot_inside=True):
    pi = ConstantPropertyProvider(FluidTransportProperties(rho=1000., mu=.001, k=.6, cp=1000.))
    po = ConstantPropertyProvider(FluidTransportProperties(rho=1.1, mu=1.8e-5, k=.026, cp=700.))
    return (HXSideInput(provider=pi, m_dot=1., T_in=350. if hot_inside else 300., p=101325.),
            HXSideInput(provider=po, m_dot=1., T_in=300. if hot_inside else 350., p=101325.))


@pytest.mark.parametrize("finned", [False, True])
@pytest.mark.parametrize("hot_inside", [False, True])
@pytest.mark.parametrize("iterate,margin", [(True,0.), (True,.3), (False,0.), (False,.3)])
def test_simulation_uses_shared_forward_in_every_dry_path(finned, hot_inside, iterate, margin):
    hx = exchanger(finned)
    inside, outside = sides(hot_inside)
    result = hx.simulate(inside, outside, iterate=iterate, surface_margin=margin)
    eps = effectiveness_ntu(1000., 700., result.UA/(1+margin),
        flow_arrangement=hx.bundle.thermal_topology_resolved, C_inside=1000., C_outside=700.)
    assert abs(result.q) == pytest.approx(eps*700.*50., rel=1e-10)
    assert result.EMTD == pytest.approx(abs(result.q)/(result.UA/(1+margin)))
    assert result.overdesign_factor == pytest.approx(margin, abs=1e-10)
    assert result.thermal_topology_model == "counter_crossflow_equal_ua"
    assert result.n_thermal_sections == 2
    assert result.inter_section_mixing == "complete"
    assert result.F_equivalent == pytest.approx(result.EMTD/result.LMTD_countercurrent)
    assert 0 < result.F_equivalent <= 1.
    assert result.final_result.n_thermal_sections == 2


@pytest.mark.parametrize("finned", [False, True])
def test_rating_inverse_and_public_simulation_bridge_agree(finned):
    hx = exchanger(finned)
    inside, outside = sides()
    sim = hx.simulate(inside, outside)
    ri = BalanceSideSpec(provider=inside.provider, m_dot=1., T_in=inside.T_in,
                        T_out=sim.T_out_inside, p=inside.p)
    ro = BalanceSideSpec(provider=outside.provider, m_dot=1., T_in=outside.T_in,
                        T_out=sim.T_out_outside, p=outside.p)
    rate = hx.rate(ri, ro, include_simulation=True)
    expected = ntu_from_effectiveness(rate.Q_required/(700.*50.),1000.,700.,
        flow_arrangement=hx.bundle.thermal_topology_resolved, C_inside=1000., C_outside=700.)*700.
    assert rate.UA_required == pytest.approx(expected, rel=1e-10)
    assert rate.UA_actual == pytest.approx(sim.UA, rel=1e-10)
    assert rate.UA_required == pytest.approx(sim.UA, rel=1e-9)
    assert rate.overdesign_factor == pytest.approx(0., abs=1e-9)
    assert rate.simulation.q == pytest.approx(sim.q, rel=1e-10)
    assert rate.thermal_topology_model == sim.thermal_topology_model
    assert rate.n_thermal_sections == sim.n_thermal_sections == 2
    assert rate.F_equivalent == pytest.approx(sim.F_equivalent, rel=1e-9)


@pytest.mark.parametrize("override", ["crossflow", "counterflow", "cocurrentflow"])
def test_explicit_method_overrides_freeze_legacy_law(override):
    hx = exchanger()
    inside, outside = sides()
    result = hx.simulate(inside, outside, flow_arrangement=override, iterate=False)
    eps = effectiveness_ntu(1000.,700.,result.UA,flow_arrangement=override,
                            C_inside=1000.,C_outside=700.)
    assert result.q == pytest.approx(eps*700.*50., rel=1e-12)
    explicit = exchanger(flow_arrangement=override).simulate(inside,outside,iterate=False)
    assert result.q == explicit.q
    assert result.n_thermal_sections == 1
    if override == "counterflow":
        assert result.F_equivalent == pytest.approx(1., abs=1e-13)


def test_auto_resolution_and_parity_interaction():
    b = exchanger().bundle
    assert b.n_tubes_total == 58
    c = replace(b,row_parity_mode="continuous")
    assert c.n_tubes_total == 57
    assert b.thermal_topology_resolved == c.thermal_topology_resolved == CounterCrossflowTopology(2)
    assert replace(b,n_passes_transverse=4).thermal_topology_resolved == "crossflow"
    assert replace(b,n_passes_transverse=1).thermal_topology_resolved == "counterflow"
    assert resolve_thermal_topology(replace(b,flow_arrangement="counterflow"),"auto") == CounterCrossflowTopology(2)
    for transverse in (1,2,4):
        assert replace(c,n_passes_transverse=transverse).n_tubes_total == 57
    assert c.total_outer_area/b.total_outer_area == pytest.approx(57/58)


def test_nonexact_rows_keep_integer_sections_without_inventing_weights():
    b = exchanger(n_rows=7).bundle
    assert b.rows_partition_is_exact is False
    assert b.thermal_topology_resolved == CounterCrossflowTopology(2)


@pytest.mark.parametrize("field", ["n_passes_tube", "n_passes_transverse"])
@pytest.mark.parametrize("value", [2.5, True])
def test_invalid_pass_grouping_is_rejected(field,value):
    with pytest.raises(ValueError,match="positive integer"):
        exchanger(**{field:value})


def test_phase_model_guard_is_explicit_and_overrides_remain_available():
    b = exchanger().bundle
    with pytest.raises(ValueError,match="dry single-phase"):
        reject_phase_change_topology(b)
    for explicit in ("crossflow","counterflow","cocurrentflow"):
        reject_phase_change_topology(b,explicit)


def test_wet_candidate_rejects_before_provider_execution():
    from core.phase_change.wet_coil_provider import dispatch_wet_coil
    from core.phase_change.integration import PhaseChangeSettings
    hx = exchanger()
    inside,outside = sides()
    with patch('core.phase_change.wet_coil_provider.ElmahdyMitalasWetCoilProvider.simulate') as invoke:
        with pytest.raises(ValueError,match="dry single-phase"):
            dispatch_wet_coil(hx,inside,outside,mode="simulation",
                              settings=PhaseChangeSettings(),force_candidate=True)
        invoke.assert_not_called()


def test_wet_equivalent_reporting_rejects_dry_descriptor():
    from core.phase_change.wet_coil_reporting import equivalent_wet_process
    with pytest.raises(ValueError,match="unsupported for wet"):
        equivalent_wet_process(heat=100.,hot_in=350.,hot_out=340.,
            cold_in=300.,cold_out=310.,flow_arrangement=CounterCrossflowTopology(2))


def test_zero_duty_diagnostics_leave_factor_undefined():
    hx = exchanger()
    inside,outside = sides()
    result = hx.simulate(replace(inside,T_in=300.),outside,iterate=False)
    assert result.q == 0.
    assert result.LMTD_countercurrent == 0.
    assert result.F_equivalent is None


def test_rating_diagnostic_counterflow_and_simple_crossflow():
    inside,outside = sides()
    ri = BalanceSideSpec(provider=inside.provider,m_dot=1.,T_in=350.,T_out=336.,p=inside.p)
    ro = BalanceSideSpec(provider=outside.provider,m_dot=1.,T_in=300.,T_out=320.,p=outside.p)
    for arrangement in ('counterflow','crossflow'):
        result = exchanger().rate(ri,ro,flow_arrangement=arrangement,include_simulation=True)
        assert result.n_thermal_sections == 1
        assert result.simulation.n_thermal_sections == 1
        assert result.F_equivalent == pytest.approx(result.EMTD/result.LMTD_countercurrent)
        if arrangement=='counterflow':
            assert result.F_equivalent == pytest.approx(1.,abs=1e-13)
