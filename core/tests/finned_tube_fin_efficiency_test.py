# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Fin-efficiency tests for circular (annular) finned tubes (v0.7.x)."""

from __future__ import annotations

import math
import random

import pytest

from core.heat_transfer.modified_bessel import bessel_i0, bessel_i1, bessel_k0, bessel_k1
from core.heat_transfer.fin_efficiency import (
    fin_efficiency_constant_thickness,
    fin_efficiency_tapered,
    overall_surface_efficiency,
)


# -------------------------------------------------------------------
# Modified Bessel function sanity (building block correctness)
# -------------------------------------------------------------------

def test_bessel_functions_match_known_reference_values():
    assert bessel_i0(1.0) == pytest.approx(1.2660658777520084, rel=1e-6)
    assert bessel_i1(1.0) == pytest.approx(0.5651591039924851, rel=1e-6)
    assert bessel_k0(1.0) == pytest.approx(0.42102443824070834, rel=1e-6)
    assert bessel_k1(1.0) == pytest.approx(0.6019072301972346, rel=1e-6)


def test_bessel_functions_satisfy_wronskian_identity():
    # I0(x)*K1(x) + I1(x)*K0(x) = 1/x, exact for the true functions;
    # independent of any transcription error in the coefficients.
    rng = random.Random(0)
    for _ in range(500):
        x = rng.uniform(0.05, 30.0)
        lhs = bessel_i0(x) * bessel_k1(x) + bessel_i1(x) * bessel_k0(x)
        assert lhs == pytest.approx(1.0 / x, rel=1e-5)


# -------------------------------------------------------------------
# Constant-thickness closed form vs. independent numerical reference
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "D_root,D_fin,t,k,h",
    [
        (0.025, 0.057, 0.0004, 200.0, 80.0),
        (0.020, 0.045, 0.0006, 150.0, 50.0),
        (0.030, 0.070, 0.0003, 80.0, 150.0),
        (0.025, 0.050, 0.0008, 400.0, 300.0),
    ],
)
def test_constant_thickness_closed_form_matches_independent_numerical_ode(D_root, D_fin, t, k, h):
    closed = fin_efficiency_constant_thickness(D_root=D_root, D_fin=D_fin, fin_thickness=t, fin_k=k, h_o=h)
    numerical = fin_efficiency_tapered(
        D_root=D_root, D_fin=D_fin, fin_thickness_root=t, fin_thickness_tip=t, fin_k=k, h_o=h, n_steps=4000
    ).efficiency
    assert closed == pytest.approx(numerical, rel=2e-4)


def test_closed_form_is_within_unit_bounds():
    eta = fin_efficiency_constant_thickness(D_root=0.025, D_fin=0.057, fin_thickness=0.0004, fin_k=200.0, h_o=80.0)
    assert 0.0 < eta <= 1.0


# -------------------------------------------------------------------
# Directional / limiting behavior
# -------------------------------------------------------------------

def test_higher_fin_conductivity_increases_efficiency():
    lo = fin_efficiency_constant_thickness(D_root=0.025, D_fin=0.057, fin_thickness=0.0004, fin_k=50.0, h_o=80.0)
    hi = fin_efficiency_constant_thickness(D_root=0.025, D_fin=0.057, fin_thickness=0.0004, fin_k=400.0, h_o=80.0)
    assert hi > lo


def test_higher_htc_decreases_efficiency():
    lo = fin_efficiency_constant_thickness(D_root=0.025, D_fin=0.057, fin_thickness=0.0004, fin_k=200.0, h_o=20.0)
    hi = fin_efficiency_constant_thickness(D_root=0.025, D_fin=0.057, fin_thickness=0.0004, fin_k=200.0, h_o=300.0)
    assert hi < lo


def test_taller_fin_decreases_efficiency():
    short = fin_efficiency_constant_thickness(D_root=0.025, D_fin=0.032, fin_thickness=0.0004, fin_k=200.0, h_o=80.0)
    tall = fin_efficiency_constant_thickness(D_root=0.025, D_fin=0.080, fin_thickness=0.0004, fin_k=200.0, h_o=80.0)
    assert tall < short


def test_small_fin_or_high_conductivity_limit_approaches_unity():
    eta = fin_efficiency_constant_thickness(D_root=0.025, D_fin=0.0255, fin_thickness=0.0004, fin_k=200.0, h_o=80.0)
    assert eta > 0.999


def test_thicker_fin_increases_efficiency():
    thin = fin_efficiency_constant_thickness(D_root=0.025, D_fin=0.057, fin_thickness=0.0002, fin_k=200.0, h_o=80.0)
    thick = fin_efficiency_constant_thickness(D_root=0.025, D_fin=0.057, fin_thickness=0.0008, fin_k=200.0, h_o=80.0)
    assert thick > thin


# -------------------------------------------------------------------
# Tapered fin stability and consistency
# -------------------------------------------------------------------

def test_tapered_solver_converges_with_step_count():
    kwargs = dict(D_root=0.025, D_fin=0.057, fin_thickness_root=0.0008, fin_thickness_tip=0.0002, fin_k=200.0, h_o=80.0)
    coarse = fin_efficiency_tapered(**kwargs, n_steps=100).efficiency
    fine = fin_efficiency_tapered(**kwargs, n_steps=4000).efficiency
    assert coarse == pytest.approx(fine, rel=1e-4)


def test_tapered_fin_differs_from_naive_mean_thickness_shortcut():
    tapered = fin_efficiency_tapered(
        D_root=0.025, D_fin=0.057, fin_thickness_root=0.0008, fin_thickness_tip=0.0002,
        fin_k=200.0, h_o=80.0, n_steps=4000,
    ).efficiency
    naive_mean = fin_efficiency_constant_thickness(
        D_root=0.025, D_fin=0.057, fin_thickness=0.0005, fin_k=200.0, h_o=80.0
    )
    # the exact tapered solution and the "just average the thickness"
    # shortcut must not silently coincide (the task explicitly forbids
    # collapsing root/tip thickness into a single mean for the geometric
    #/thermal model)
    assert abs(tapered - naive_mean) / naive_mean > 0.01


def test_tapered_fin_reduces_to_constant_thickness_case():
    tapered = fin_efficiency_tapered(
        D_root=0.025, D_fin=0.057, fin_thickness_root=0.0004, fin_thickness_tip=0.0004,
        fin_k=200.0, h_o=80.0, n_steps=4000,
    ).efficiency
    constant = fin_efficiency_constant_thickness(
        D_root=0.025, D_fin=0.057, fin_thickness=0.0004, fin_k=200.0, h_o=80.0
    )
    assert tapered == pytest.approx(constant, rel=2e-4)


def test_tapered_fin_requires_valid_inputs():
    with pytest.raises(ValueError):
        fin_efficiency_tapered(
            D_root=0.025, D_fin=0.020, fin_thickness_root=0.0004, fin_thickness_tip=0.0004,
            fin_k=200.0, h_o=80.0,
        )
    with pytest.raises(ValueError):
        fin_efficiency_tapered(
            D_root=0.025, D_fin=0.057, fin_thickness_root=0.0, fin_thickness_tip=0.0004,
            fin_k=200.0, h_o=80.0,
        )


# -------------------------------------------------------------------
# Overall surface efficiency
# -------------------------------------------------------------------

def test_overall_surface_efficiency_between_fin_efficiency_and_one():
    eta_fin = 0.8
    eta_o = overall_surface_efficiency(A_primary=0.2, A_fin=5.5, fin_efficiency=eta_fin)
    assert eta_fin < eta_o < 1.0


def test_overall_surface_efficiency_no_fin_area_equals_one():
    assert overall_surface_efficiency(A_primary=1.0, A_fin=0.0, fin_efficiency=0.5) == pytest.approx(1.0)


def test_overall_surface_efficiency_all_fin_area_equals_fin_efficiency():
    assert overall_surface_efficiency(A_primary=0.0, A_fin=1.0, fin_efficiency=0.63) == pytest.approx(0.63)


def test_overall_surface_efficiency_rejects_out_of_range_fin_efficiency():
    with pytest.raises(ValueError):
        overall_surface_efficiency(A_primary=1.0, A_fin=1.0, fin_efficiency=1.5)
