"""v0.9.9 public Rating flow closure and preserved water/steam limits."""

import math

import pytest

from core.models.simulation import HXSideInput
from core.phase_change import warning_codes as WC
from core.phase_change.integration import MultiplePhaseChangeSidesError
from core.phase_change.steam_integration import PhaseChangeDisabledButRequiredError
from core.phase_change.types import PhaseChangeMode
from core.properties.water import water_steam_props_iapws97
from core.geometry.tube import TubeOrientation
from core.tests.water_evaporation_rating_test import (
    HOT, P, _FastWetGasProvider, _hx, _inside, _outside,
)
from core.properties.gas_mixture import GasMixtureSpec


@pytest.mark.parametrize("duty_source", ["Q", "outside", "both"])
@pytest.mark.parametrize("target", [
    {"quality_out": 1.0},
    {"quality_out": 0.45},
    {"h_out": water_steam_props_iapws97(p=P, x=0.6).h},
    {"T_out": 520.0},
])
def test_unknown_water_flow_matches_known_flow_and_conserves_energy(target, duty_source):
    hx = _hx()
    inside = _inside(m_dot=None, T_in=350.0, **target)
    snapshot = vars(inside).copy()
    Q = 800_000.0
    outside = _outside(T_out=(700.0 - Q / (30.0 * HOT.props.cp))
                       if duty_source != "Q" else None)
    result = hx.rate(inside, outside, Q=Q if duty_source != "outside" else None)
    expected_flow = Q / (inside.h_out - inside.h_in)
    known = hx.rate(_inside(m_dot=expected_flow, T_in=350.0, **target), outside)
    water = result.inside_phase_change

    assert vars(inside) == snapshot
    assert inside.m_dot is None
    assert result.closed_balance.inside.m_dot == pytest.approx(expected_flow)
    assert result.closed_balance.outside.m_dot == outside.m_dot
    assert result.closed_balance.hot_is_inside is False
    assert result.Q_required == pytest.approx(Q)
    assert result.closed_balance.Q == pytest.approx(Q)
    assert water.mass_flow_total == pytest.approx(expected_flow)
    assert water.h_out == pytest.approx(inside.h_out, rel=2e-12)
    assert water.Q_total == pytest.approx(expected_flow * (water.h_out - water.h_in))
    assert water.Q_total == pytest.approx(
        outside.m_dot * result.closed_balance.outside.cp_mean
        * (outside.T_in - result.closed_balance.outside.T_out)
    )
    assert water.Q_total == pytest.approx(
        water.Q_preheat + water.Q_evaporation + water.Q_superheat
    )
    assert water.A_total == pytest.approx(sum(zone.area for zone in water.zones))
    assert water.UA_total == pytest.approx(sum(zone.UA for zone in water.zones))
    x_out = 1.0 if water.quality_out is None else water.quality_out
    assert water.m_dot_evaporated == pytest.approx(expected_flow * x_out)
    assert water.m_dot_condensate == 0.0
    assert water.mass_flux == pytest.approx(
        expected_flow / (hx.bundle.n_tubes_per_pass_effective * math.pi * hx.bundle.tube.D_i**2 / 4)
    )
    for name in ("A_required", "UA_required", "UA_actual", "U_mean", "alfa_i",
                 "alfa_o"):
        assert getattr(result, name) == pytest.approx(getattr(known, name), rel=2e-10)
    assert result.closed_balance.inside.T_out == pytest.approx(known.closed_balance.inside.T_out)
    assert result.closed_balance.outside.T_out == pytest.approx(known.closed_balance.outside.T_out)
    for zone, reference in zip(water.zones, known.inside_phase_change.zones):
        assert zone.kind == reference.kind
        for name in ("Q", "area", "UA", "alpha_inside", "U", "heat_flux_inner"):
            assert getattr(zone, name) == pytest.approx(getattr(reference, name), rel=2e-10)
    assert result.tube_side_hydraulic is None
    assert result.tube_side_pressure_drop is None
    assert math.isnan(result.inside_dp_total)
    assert not water.two_phase_pressure_drop_supported
    assert WC.WATER_EVAPORATION_TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED in {
        warning.code for warning in result.warnings
    }
    assert not hasattr(water, "solution")
    assert not hasattr(water, "runtime_s")


@pytest.mark.parametrize("inlet", [
    {"quality_in": 0.0},
    {"quality_in": 0.2},
    {"h_in": water_steam_props_iapws97(p=P, x=0.2).h},
    {"T_in": water_steam_props_iapws97(p=P, x=0.0).T - 0.01},
])
def test_unknown_flow_preserves_canonical_inlet_in_optional_simulation(inlet):
    hx = _hx()
    inside = _inside(m_dot=None, quality_out=0.8, **inlet)
    result = hx.rate(inside, _outside(), Q=600_000.0, include_simulation=True)
    expected = hx.simulate(
        HXSideInput(provider=inside.provider, p=P,
                    m_dot=result.closed_balance.inside.m_dot, **inlet),
        HXSideInput(provider=HOT, p=101325.0, m_dot=30.0, T_in=700.0),
    )
    assert result.Q_achievable == pytest.approx(expected.q, rel=2e-10)
    assert result.simulation.q == pytest.approx(expected.q, rel=2e-10)
    assert result.simulation.inside_phase_change.mass_flow_total == pytest.approx(
        result.closed_balance.inside.m_dot
    )
    assert result.simulation.inside_phase_change.h_in == pytest.approx(inside.h_in)
    assert inside.m_dot is None


def test_unknown_flow_rating_reconstructs_simulation_geometry_and_zone_physics():
    hx = _hx()
    simulation = hx.simulate(
        HXSideInput(provider=_inside(T_in=350.0).provider, p=P, m_dot=1.0, T_in=350.0),
        HXSideInput(provider=HOT, p=101325.0, m_dot=30.0, T_in=700.0),
    )
    inside = _inside(m_dot=None, T_in=350.0, h_out=simulation.inside_phase_change.h_out)
    rating = hx.rate(inside, _outside(), Q=simulation.q, include_simulation=True)
    assert rating.closed_balance.inside.m_dot == pytest.approx(1.0, rel=2e-12)
    assert rating.A_required == pytest.approx(hx.bundle.total_outer_area, rel=2e-8)
    assert rating.UA_required == pytest.approx(simulation.UA, rel=2e-8)
    assert rating.Q_achievable == pytest.approx(simulation.q, rel=2e-8)
    for name in ("Q_preheat", "Q_evaporation", "Q_superheat", "A_preheat",
                 "A_evaporation", "A_superheat", "inside_alpha_equivalent"):
        assert getattr(rating.inside_phase_change, name) == pytest.approx(
            getattr(simulation.inside_phase_change, name), rel=2e-8
        )


@pytest.mark.parametrize("Q", [None, 600_000.0])
def test_unknown_flow_requires_explicit_outlet_instead_of_defaulting_to_vapor(Q):
    with pytest.raises(ValueError, match="explicit.*outlet"):
        _hx().rate(_inside(m_dot=None, T_in=350.0), _outside(), Q=Q)


def test_unknown_flow_requires_independent_duty():
    with pytest.raises(ValueError, match="explicit Q|opposing-side temperature program"):
        _hx().rate(_inside(m_dot=None, T_in=350.0, quality_out=1.0), _outside())


@pytest.mark.parametrize("inside_flow", [None, 1.0])
def test_outside_flow_must_remain_known(inside_flow):
    with pytest.raises(ValueError, match="outside mass flow"):
        _hx().rate(_inside(m_dot=inside_flow, T_in=350.0, quality_out=1.0),
                   _outside(m_dot=None, T_out=650.0), Q=600_000.0)


@pytest.mark.parametrize("Q", [0.0, -1.0, math.nan, math.inf])
def test_unknown_flow_rejects_nonpositive_or_nonfinite_duty(Q):
    with pytest.raises(ValueError, match="positive and finite"):
        _hx().rate(_inside(m_dot=None, T_in=350.0, quality_out=1.0), _outside(), Q=Q)


@pytest.mark.parametrize("T_out", [700.0, 710.0])
def test_unknown_flow_rejects_hot_side_without_positive_duty(T_out):
    with pytest.raises(ValueError, match="positive and finite"):
        _hx().rate(_inside(m_dot=None, T_in=350.0, quality_out=1.0),
                   _outside(T_out=T_out))


def test_unknown_flow_rejects_conflicting_independent_duties():
    with pytest.raises(ValueError, match="Over-specified.*inconsistent"):
        _hx().rate(_inside(m_dot=None, T_in=350.0, quality_out=1.0),
                   _outside(T_out=650.0), Q=600_000.0)


@pytest.mark.parametrize("target", [{"T_out": 350.0}, {"T_out": 340.0}])
def test_unknown_flow_rejects_zero_or_negative_enthalpy_rise(target):
    with pytest.raises(ValueError, match="higher specific enthalpy"):
        _hx().rate(_inside(m_dot=None, T_in=350.0, **target), _outside(), Q=600_000.0)


@pytest.mark.parametrize("orientation", [None, TubeOrientation.VERTICAL_DOWNWARD])
def test_unknown_flow_keeps_boiling_orientation_guards(orientation):
    with pytest.raises(ValueError, match="orientation"):
        _hx(orientation=orientation).rate(
            _inside(m_dot=None, T_in=350.0, quality_out=1.0), _outside(), Q=600_000.0
        )


def test_unknown_flow_keeps_disabled_phase_change_guard():
    with pytest.raises(PhaseChangeDisabledButRequiredError):
        _hx().rate(_inside(m_dot=None, T_in=350.0, quality_out=1.0,
                           mode=PhaseChangeMode.DISABLED), _outside(), Q=600_000.0)


def test_unknown_flow_keeps_simultaneous_phase_change_guard():
    wet = _FastWetGasProvider(GasMixtureSpec(
        components={"N2": 0.49, "O2": 0.21, "H2O": 0.30}, basis="mole"
    ))
    with pytest.raises(MultiplePhaseChangeSidesError):
        _hx().rate(_inside(m_dot=None, quality_in=0.0, quality_out=0.5),
                   _outside(provider=wet, m_dot=10.0, T_in=500.0, p=5.0e6), Q=600_000.0)


def test_unknown_flow_keeps_thermal_pinch_guard():
    with pytest.raises(ValueError, match="temperature difference"):
        _hx().rate(_inside(m_dot=None, T_in=350.0, T_out=520.0),
                   _outside(T_in=460.0), Q=600_000.0)


@pytest.mark.parametrize("states", [
    {"T_in": water_steam_props_iapws97(p=P, x=0.0).T, "quality_out": 1.0},
    {"T_in": 350.0, "T_out": water_steam_props_iapws97(p=P, x=1.0).T},
])
def test_unknown_flow_keeps_saturation_temperature_ambiguity_guard(states):
    with pytest.raises(ValueError, match="saturation line"):
        _inside(m_dot=None, **states)


def test_unknown_flow_sensible_target_has_finite_hydraulics_without_orientation():
    result = _hx(orientation=None).rate(
        _inside(m_dot=None, T_in=350.0, T_out=400.0), _outside(), Q=100_000.0
    )
    assert not result.inside_phase_change.active
    assert result.tube_side_hydraulic is not None
    assert math.isfinite(result.inside_dp_total)
    assert result.closed_balance.inside.m_dot == pytest.approx(
        100_000.0 / (result.inside_phase_change.h_out - result.inside_phase_change.h_in)
    )


def test_unknown_condensation_flow_default_and_optional_simulation_are_preserved():
    inside = _inside(m_dot=None, quality_in=1.0)
    outside = _outside(T_in=300.0)
    Q = 600_000.0
    result = _hx(orientation=TubeOrientation.VERTICAL_DOWNWARD).rate(
        inside, outside, Q=Q, include_simulation=True
    )
    liquid = water_steam_props_iapws97(p=P, x=0.0)
    expected_flow = Q / (inside.h_in - liquid.h)
    assert result.closed_balance.inside.m_dot == pytest.approx(expected_flow)
    assert result.inside_phase_change.quality_out == 0.0
    assert result.Q_required == pytest.approx(Q)
    assert result.simulation.inside_phase_change.mass_flow_total == pytest.approx(expected_flow)
    assert inside.m_dot is None
    assert inside.water_steam_outlet_state is None
