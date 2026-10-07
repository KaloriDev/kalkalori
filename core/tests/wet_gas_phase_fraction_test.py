# SPDX-License-Identifier: GPL-3.0-only
"""Process phase fractions share the inlet mass basis, including drained water."""
from dataclasses import asdict, replace

import pytest

from core.phase_change.types import PhaseChangeDirection, PhaseChangeMode, PhaseChangeResult


def wet_result(**changes):
    result = PhaseChangeResult(
        side="outside", mode=PhaseChangeMode.AUTO,
        direction=PhaseChangeDirection.CONDENSATION,
        component="H2O", capable=True, possible=True, active=True,
        W_in=0.25, W_mid=0.20, W_out=0.125,
        m_dot_dry_carrier=800.0 / 3600.0,
        m_dot_gas_in=1000.0 / 3600.0,
        m_dot_gas_out=900.0 / 3600.0,
        m_dot_condensate=100.0 / 3600.0,
    )
    return replace(result, **changes)


@pytest.mark.parametrize("W", [0.0, 0.016])
def test_dry_or_noncondensing_wet_gas_is_entirely_gas(W):
    pc = wet_result(W_in=W, W_mid=W, W_out=W, active=False,
                    direction=PhaseChangeDirection.NONE, m_dot_condensate=0.0)
    for position in ("in", "mid", "out"):
        assert getattr(pc, f"gas_phase_mass_fraction_{position}") == 1.0
        assert getattr(pc, f"liquid_phase_mass_fraction_{position}") == 0.0


def test_known_condensation_uses_total_inlet_mass_and_actual_mean_humidity():
    pc = wet_result()
    before = asdict(pc)
    # W_mid is deliberately not the endpoint average: use the stored state.
    for position, gas, liquid in (("in", 1.0, 0.0), ("mid", 0.96, 0.04), ("out", 0.9, 0.1)):
        actual_gas = getattr(pc, f"gas_phase_mass_fraction_{position}")
        actual_liquid = getattr(pc, f"liquid_phase_mass_fraction_{position}")
        assert actual_gas == pytest.approx(gas)
        assert actual_liquid == pytest.approx(liquid)
        assert actual_liquid == pytest.approx(1.0 - actual_gas)
        assert actual_gas + actual_liquid == pytest.approx(1.0)
    assert pc.gas_phase_mass_fraction_out == pytest.approx(pc.m_dot_gas_out / pc.m_dot_gas_in)
    assert pc.liquid_phase_mass_fraction_out == pytest.approx(pc.m_dot_condensate / pc.m_dot_gas_in)
    assert pc.m_dot_condensate == pytest.approx(pc.m_dot_dry_carrier * (pc.W_in - pc.W_out))
    assert asdict(pc) == before


@pytest.mark.parametrize("position", ["in", "mid", "out"])
def test_missing_inlet_reference_has_no_phase_fraction(position):
    pc = wet_result(W_in=None)
    assert getattr(pc, f"gas_phase_mass_fraction_{position}") is None
    assert getattr(pc, f"liquid_phase_mass_fraction_{position}") is None


@pytest.mark.parametrize("position", ["mid", "out"])
def test_missing_state_humidity_is_not_reconstructed(position):
    pc = wet_result(**{f"W_{position}": None})
    assert getattr(pc, f"gas_phase_mass_fraction_{position}") is None
    assert getattr(pc, f"liquid_phase_mass_fraction_{position}") is None


def test_inconsistent_humidity_is_visible_without_clipping():
    # Supported solvers prohibit re-evaporation. An inconsistent result must
    # expose the negative cumulative removal rather than disguise it as dry.
    pc = wet_result(W_out=0.30)
    assert pc.gas_phase_mass_fraction_out == pytest.approx(1.04)
    assert pc.liquid_phase_mass_fraction_out == pytest.approx(-0.04)


@pytest.mark.parametrize("finned,W", [(False, 0.004), (False, 0.016), (True, 0.016)])
def test_public_solved_wet_gas_fractions_agree_with_mass_bookkeeping(finned, W):
    from core.tests.wet_coil_public_test import context, TIGHT_OPTIONS

    hx, inside, outside = context(finned, W)
    result = hx.simulate(inside, outside, wet_solver_options=TIGHT_OPTIONS)
    pc = result.outside_phase_change
    for position in ("in", "mid", "out"):
        gas = getattr(pc, f"gas_phase_mass_fraction_{position}")
        liquid = getattr(pc, f"liquid_phase_mass_fraction_{position}")
        humidity = getattr(pc, f"W_{position}")
        assert gas == pytest.approx((1.0 + humidity) / (1.0 + pc.W_in))
        assert gas + liquid == pytest.approx(1.0)
    assert pc.W_mid == pytest.approx((pc.W_in + pc.W_out) / 2.0)
    assert pc.gas_phase_mass_fraction_out == pytest.approx(pc.m_dot_gas_out / outside.m_dot)
    assert pc.liquid_phase_mass_fraction_out == pytest.approx(pc.m_dot_condensate / outside.m_dot)
    assert pc.m_dot_condensate == pytest.approx(pc.m_dot_dry_carrier * (pc.W_in - pc.W_out))
    if pc.active:
        assert pc.gas_phase_mass_fraction_out < 1.0
    else:
        assert pc.gas_phase_mass_fraction_out == 1.0


def test_legacy_wet_gas_phase_fractions_preserve_pre_patch_physics():
    from core.tests.legacy_wet_coil_provider_test import historical_case, LEGACY

    hx, inside, outside = historical_case(False)
    result = hx.simulate(inside, outside, wet_coil_provider=LEGACY)
    pc = result.outside_phase_change
    # Captured before this bookkeeping patch on the same synthetic bare-bank
    # case used by the existing Legacy numerical-freeze regression.
    assert (result.q, result.T_out_inside, result.T_out_outside,
            pc.W_out, pc.m_dot_condensate) == pytest.approx(
        (702644.68440337, 336.21383345649315, 336.5149577934638,
         0.10978447100327163, 0.06982196451578865), rel=1e-12,
    )
    for position in ("in", "mid", "out"):
        gas = getattr(pc, f"gas_phase_mass_fraction_{position}")
        liquid = getattr(pc, f"liquid_phase_mass_fraction_{position}")
        assert gas == pytest.approx((1 + getattr(pc, f"W_{position}")) / (1 + pc.W_in))
        assert liquid == pytest.approx(1 - gas)
        assert gas + liquid == pytest.approx(1.0)
    assert pc.gas_phase_mass_fraction_out == pytest.approx(pc.m_dot_gas_out / outside.m_dot)
    # Preserve the existing Legacy solver's mass-balance tolerance.
    assert pc.liquid_phase_mass_fraction_out * outside.m_dot == pytest.approx(
        pc.m_dot_condensate, abs=1e-8,
    )
