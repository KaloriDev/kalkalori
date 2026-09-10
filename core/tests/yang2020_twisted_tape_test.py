# KalKalori - GNU GPL v3 only
"""Independent arithmetic anchors for the OPEN Yang2020 Eqs.21-22, not M&B.

Inputs use the paper's D=.012, delta=.001, L=.3; coefficients are read
directly from its PDF. Constants below were calculated separately with
45-digit decimal arithmetic from the published equations, not this provider.
Y's Table 2 is not an exact Eq.21/22 benchmark and is deliberately not used.
"""
from dataclasses import replace
import math

import pytest

from core.enhancements import (
    EnhancementInput, EnhancementState, EnhancementUnsupportedError,
    TubeSideEnhancement, TwistedTapeGeometry, Yang2020TwistedTapeProvider,
    evaluate_enhancement,
)


def state_for(re=500, pr=8, ratio=1):
    bulk = EnhancementState(1000, .001, .5, pr*500, 300, 1e5)
    wall = replace(bulk, temperature=320, mu=bulk.mu/ratio)
    return EnhancementInput(re*.001*math.pi*.012/4, .012, .3, .3,
                            bulk, wall, "liquid")


def evaluate(state=None, y=3):
    return evaluate_enhancement(TubeSideEnhancement(Yang2020TwistedTapeProvider(),
                                TwistedTapeGeometry(y*.012, .012, .001), "liquid"),
                                state or state_for())


@pytest.mark.parametrize("re,y,pr,ratio,f,nu,sw,gz", [
    (500, 3, 8, 1, .701682172082434659, 21.03189900824799046,
     364.53001013898545, 125.66370614359173),
    (100, 2, 7, 2, 3.66578632176205536, 12.14982647078097218,
     100.58484901986960, 21.991148575128553),
    (1100, 4, 900, 1, .339254689837772258, 145.369276966793112,
     661.02556018741574, 31101.767270538953),
])
def test_published_equation_hand_anchors(re, y, pr, ratio, f, nu, sw, gz):
    result = evaluate(state_for(re, pr, ratio), y)
    assert result.f_darcy == pytest.approx(f, rel=1e-12)
    assert result.nusselt == pytest.approx(nu, rel=1e-12)
    assert result.alpha_inside == pytest.approx(nu*.5/.012, rel=1e-12)
    d = {v.name: v.value for v in result.diagnostics}
    assert d["swirl_parameter"] == pytest.approx(sw, rel=1e-12)
    assert d["graetz"] == pytest.approx(gz, rel=1e-12)


def test_native_darcy_not_multiplied_by_four():
    result = evaluate()
    assert result.friction_basis == "darcy"
    assert result.friction_factor_native == pytest.approx(.701682172082434659)
    assert result.f_fanning == pytest.approx(.175420543020608665)
    # Eq.2 with u=1/24 m/s, rho=1000, D=.012: f*1000/(576*.024).
    assert result.pressure_gradient(1000) == pytest.approx(.701682172082434659*1000/(576*.024))


def test_blockage_owned_once_and_physical_wall_basis():
    result = evaluate()
    r = result.reference
    d = {v.name: v.value for v in result.diagnostics}
    assert d["open_axial_area"] == pytest.approx(.000101097335529232557)
    assert d["phi"] == pytest.approx(1.118697490266993)
    assert d["psi"] == pytest.approx(1.771533686735454)
    assert r.hydraulic_diameter == pytest.approx(.00677379159642928)
    assert r.flow_area == pytest.approx(math.pi*.012**2/4)
    assert r.velocity == pytest.approx(1/24)
    assert r.friction_diameter == r.nusselt_length == .012
    assert d["axial_velocity"] > r.velocity


@pytest.mark.parametrize("re", [99.9, 1100.1, 2300, 3000, 10000])
def test_unsupported_re_including_transition_and_turbulent(re):
    with pytest.raises(EnhancementUnsupportedError, match="reynolds"):
        evaluate(state_for(re))


@pytest.mark.parametrize("pr", [6.999, 900.001])
def test_prandtl_bounds(pr):
    with pytest.raises(EnhancementUnsupportedError, match="prandtl"):
        evaluate(state_for(pr=pr))


@pytest.mark.parametrize("y", [1.999, 4.001])
def test_twist_bounds(y):
    with pytest.raises(EnhancementUnsupportedError, match="twist_ratio"):
        evaluate(y=y)


@pytest.mark.parametrize("change", [
    {"tube_inner_diameter": .02}, {"heated_length": 1}, {"tube_length": 1},
    {"fluid_phase": "gas"}, {"fluid_phase": "two_phase"},
    {"fluid_phase": "supercritical"}, {"roughness_inner": 1e-6},
])
def test_unsupported_scope(change):
    with pytest.raises(EnhancementUnsupportedError):
        Yang2020TwistedTapeProvider().evaluate(TwistedTapeGeometry(.036, .012, .001),
                                              replace(state_for(), **change))


def test_cooling_and_incompatible_tape_rejected():
    state = state_for()
    with pytest.raises(EnhancementUnsupportedError, match="cooling"):
        evaluate(replace(state, wall=replace(state.wall, temperature=290)))
    for tape in (object(), TwistedTapeGeometry(.036, .011, .001),
                 TwistedTapeGeometry(.036, .012, .0005)):
        with pytest.raises(EnhancementUnsupportedError):
            Yang2020TwistedTapeProvider().evaluate(tape, state)


def test_provider_owns_wall_correction_and_missing_wall_is_visible():
    base = evaluate()
    corrected = evaluate(state_for(ratio=2))
    assert corrected.nusselt/base.nusselt == pytest.approx(2**.14)
    assert corrected.f_darcy == base.f_darcy
    no_wall = evaluate(replace(state_for(), wall=None))
    assert no_wall.nusselt == base.nusselt
    assert "enhancement_wall_state_unavailable" in {w.code for w in no_wall.warnings}
    assert base.source_access_basis == "open"
    assert "yang2020_eq21_eq22" in base.correlation_id
    assert base.regime == "laminar"


def test_laminar_sanity_and_continuity_inside_supported_interval():
    low, mid, high = (evaluate(state_for(re)) for re in (200, 500, 1000))
    assert low.nusselt < mid.nusselt < high.nusselt
    assert low.f_darcy > mid.f_darcy > high.f_darcy
    assert evaluate(state_for(500+1e-5)).nusselt == pytest.approx(mid.nusselt, rel=1e-7)
