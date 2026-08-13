"""Independent equation-level tests for Shah (1982) flow boiling."""

from __future__ import annotations

import math

import pytest

from core.geometry.tube import TubeOrientation
from core.heat_transfer.evaporation_inside_shah1982 import (
    GRAVITY,
    SHAH_1982_BOILING_CORRELATION,
    shah1982_boiling_alpha_local,
)


BASE = dict(
    p=1.0e6,
    pcritical=22.064e6,
    tube_inner_diameter=0.020,
    orientation=TubeOrientation.VERTICAL_UPWARD,
    liquid_density=887.1274516747791,
    vapor_density=5.145385853182684,
    liquid_viscosity=0.00015048492650911248,
    liquid_conductivity=0.6713377268728457,
    liquid_prandtl=0.9873802822607724,
    latent_heat=2_014_436.6933492515,
)


def _reference(*, mass_flux, quality, heat_flux_inner, orientation):
    """Direct transcription of Shah (1982), equations 1--14."""
    D = BASE["tube_inner_diameter"]
    rho_l = BASE["liquid_density"]
    rho_v = BASE["vapor_density"]
    mu_l = BASE["liquid_viscosity"]
    k_l = BASE["liquid_conductivity"]
    Pr_l = BASE["liquid_prandtl"]
    h_fg = BASE["latent_heat"]
    G = mass_flux
    x = quality
    Bo = heat_flux_inner / (G * h_fg)
    Co = ((1.0 / x) - 1.0) ** 0.8 * (rho_v / rho_l) ** 0.5
    Fr_L = G**2 / (rho_l**2 * GRAVITY * D)
    Re_L = G * (1.0 - x) * D / mu_l
    h_l = 0.023 * Re_L**0.8 * Pr_l**0.4 * k_l / D
    N = (
        0.38 * Fr_L**-0.3 * Co
        if orientation is TubeOrientation.HORIZONTAL and Fr_L <= 0.04
        else Co
    )
    psi_cb = 1.8 / N**0.8
    F = 14.7 if Bo >= 11e-4 else 15.43
    psi_nb = None
    psi_bs = None
    if N > 1.0:
        psi_nb = (
            230.0 * Bo**0.5
            if Bo > 0.3e-4
            else 1.0 + 46.0 * Bo**0.5
        )
        if psi_nb >= psi_cb:
            regime, psi = "nucleate_boiling", psi_nb
        else:
            regime, psi = "convective_boiling", psi_cb
    else:
        psi_bs = F * Bo**0.5 * math.exp(
            2.74 * N**-0.1 if N > 0.1 else 2.47 * N**-0.15
        )
        if psi_bs >= psi_cb:
            regime, psi = "bubble_suppression", psi_bs
        else:
            regime, psi = "convective_boiling", psi_cb
    return dict(
        Bo=Bo, Co=Co, Fr_L=Fr_L, Re_L=Re_L, h_l=h_l, N=N, F=F,
        psi_cb=psi_cb, psi_nb=psi_nb, psi_bs=psi_bs,
        regime=regime, psi=psi, alpha=psi * h_l,
    )


@pytest.mark.parametrize("mass_flux", [20.0, 100.0, 500.0])
@pytest.mark.parametrize("heat_flux_inner", [10_000.0, 100_000.0, 500_000.0])
@pytest.mark.parametrize("quality", [0.01, 0.5, 0.9])
def test_low_medium_high_G_heat_flux_and_quality_match_independent_equations(
    mass_flux, heat_flux_inner, quality
):
    result = shah1982_boiling_alpha_local(
        mass_flux=mass_flux,
        heat_flux_inner=heat_flux_inner,
        quality=quality,
        **BASE,
    )
    expected = _reference(
        mass_flux=mass_flux,
        heat_flux_inner=heat_flux_inner,
        quality=quality,
        orientation=BASE["orientation"],
    )
    assert result.correlation == SHAH_1982_BOILING_CORRELATION
    assert result.regime == expected["regime"]
    assert result.Bo == pytest.approx(expected["Bo"], rel=2e-13)
    assert result.Co == pytest.approx(expected["Co"], rel=2e-13)
    assert result.Fr_L == pytest.approx(expected["Fr_L"], rel=2e-13)
    assert result.Re_L == pytest.approx(expected["Re_L"], rel=2e-13)
    assert result.N == pytest.approx(expected["N"], rel=2e-13)
    assert result.F == expected["F"]
    assert result.alpha_liquid == pytest.approx(expected["h_l"], rel=2e-13)
    assert result.psi_convective == pytest.approx(expected["psi_cb"], rel=2e-13)
    if expected["psi_nb"] is None:
        assert result.psi_nucleate is None
    else:
        assert result.psi_nucleate == pytest.approx(expected["psi_nb"], rel=2e-13)
    if expected["psi_bs"] is None:
        assert result.psi_bubble_suppression is None
    else:
        assert result.psi_bubble_suppression == pytest.approx(
            expected["psi_bs"], rel=2e-13
        )
    assert result.psi == pytest.approx(expected["psi"], rel=2e-13)
    assert result.alpha == pytest.approx(expected["alpha"], rel=2e-13)
    assert math.isfinite(result.alpha) and result.alpha > 0.0


def test_grid_exercises_all_documented_regime_selections():
    regimes = {
        shah1982_boiling_alpha_local(
            mass_flux=G, heat_flux_inner=q, quality=x, **BASE
        ).regime
        for G in (20.0, 100.0, 500.0)
        for q in (1_000.0, 10_000.0, 500_000.0)
        for x in (0.005, 0.1, 0.9)
    }
    assert regimes == {
        "nucleate_boiling", "bubble_suppression", "convective_boiling"
    }


@pytest.mark.parametrize("target_N", [1.0, 0.1])
def test_N_transition_points_use_published_branch_inequalities(target_N):
    density_factor = math.sqrt(BASE["vapor_density"] / BASE["liquid_density"])
    ratio = (target_N / density_factor) ** (1.0 / 0.8)
    quality = 1.0 / (1.0 + ratio)
    result = shah1982_boiling_alpha_local(
        mass_flux=100.0, heat_flux_inner=100_000.0,
        quality=quality, **BASE,
    )
    expected = _reference(
        mass_flux=100.0, heat_flux_inner=100_000.0,
        quality=quality, orientation=BASE["orientation"],
    )
    assert result.N == pytest.approx(target_N, rel=2e-13)
    assert result.psi == pytest.approx(expected["psi"], rel=2e-13)


@pytest.mark.parametrize("target_Bo", [0.3e-4, 11e-4])
def test_boiling_number_transition_points_are_deterministic(target_Bo):
    G = 100.0
    q = target_Bo * G * BASE["latent_heat"]
    result = shah1982_boiling_alpha_local(
        mass_flux=G, heat_flux_inner=q, quality=0.01, **BASE
    )
    expected = _reference(
        mass_flux=G, heat_flux_inner=q, quality=0.01,
        orientation=BASE["orientation"],
    )
    assert result.Bo == pytest.approx(target_Bo, rel=2e-13)
    assert result.F == expected["F"]
    assert result.psi == pytest.approx(expected["psi"], rel=2e-13)


def test_horizontal_froude_correction_and_transition_match_reference():
    G_transition = BASE["liquid_density"] * math.sqrt(
        0.04 * GRAVITY * BASE["tube_inner_diameter"]
    )
    for G in (0.99 * G_transition, G_transition, 1.01 * G_transition):
        inputs = dict(BASE, orientation=TubeOrientation.HORIZONTAL)
        result = shah1982_boiling_alpha_local(
            mass_flux=G, heat_flux_inner=100_000.0, quality=0.5, **inputs
        )
        expected = _reference(
            mass_flux=G, heat_flux_inner=100_000.0, quality=0.5,
            orientation=TubeOrientation.HORIZONTAL,
        )
        assert result.N == pytest.approx(expected["N"], rel=2e-13)
        assert result.alpha == pytest.approx(expected["alpha"], rel=2e-13)


@pytest.mark.parametrize("quality", [1e-9, 1.0 - 1e-9])
def test_endpoint_safe_interior_points_are_finite(quality):
    result = shah1982_boiling_alpha_local(
        mass_flux=100.0, heat_flux_inner=100_000.0,
        quality=quality, **BASE,
    )
    assert math.isfinite(result.alpha) and result.alpha > 0.0


@pytest.mark.parametrize("quality", [0.0, 1.0, -0.1, 1.1, math.nan])
def test_exact_or_invalid_quality_is_rejected(quality):
    with pytest.raises(ValueError, match="0 < x < 1"):
        shah1982_boiling_alpha_local(
            mass_flux=100.0, heat_flux_inner=100_000.0,
            quality=quality, **BASE,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("mass_flux", 0.0),
        ("heat_flux_inner", -1.0),
        ("tube_inner_diameter", math.nan),
        ("latent_heat", math.inf),
    ],
)
def test_invalid_positive_finite_inputs_are_rejected(name, value):
    kwargs = dict(
        BASE, mass_flux=100.0, heat_flux_inner=100_000.0, quality=0.5
    )
    kwargs[name] = value
    with pytest.raises(ValueError):
        shah1982_boiling_alpha_local(**kwargs)


@pytest.mark.parametrize(
    "orientation",
    [TubeOrientation.VERTICAL_DOWNWARD, TubeOrientation.DOWNWARD_INCLINED_15_PLUS],
)
def test_undocumented_orientation_is_rejected(orientation):
    with pytest.raises(ValueError, match="horizontal or vertical-upward"):
        shah1982_boiling_alpha_local(
            mass_flux=100.0, heat_flux_inner=100_000.0,
            quality=0.5, **dict(BASE, orientation=orientation),
        )


def test_low_G_and_published_range_warnings_are_explicit_and_unclipped():
    result = shah1982_boiling_alpha_local(
        p=0.9 * BASE["pcritical"],
        pcritical=BASE["pcritical"],
        tube_inner_diameter=0.05,
        mass_flux=2.0,
        quality=0.9,
        heat_flux_inner=100.0,
        orientation=TubeOrientation.HORIZONTAL,
        liquid_density=BASE["liquid_density"],
        vapor_density=BASE["vapor_density"],
        liquid_viscosity=BASE["liquid_viscosity"],
        liquid_conductivity=BASE["liquid_conductivity"],
        liquid_prandtl=BASE["liquid_prandtl"],
        latent_heat=BASE["latent_heat"],
    )
    codes = {warning.code for warning in result.warnings}
    assert "WATER_BOILING_SHAH_1982_OUTSIDE_RANGE" in codes
    assert "WATER_BOILING_SHAH_1982_LOW_LIQUID_REYNOLDS" in codes
    assert "WATER_BOILING_SHAH_1982_HORIZONTAL_LOW_FR_LOW_BO" in codes
    assert "WATER_BOILING_DRYOUT_CHF_NOT_MODELLED" in codes
    assert math.isfinite(result.alpha) and result.alpha > 0.0


def test_critical_pressure_and_density_order_are_rejected():
    with pytest.raises(ValueError, match="critical pressure"):
        shah1982_boiling_alpha_local(
            mass_flux=100.0, heat_flux_inner=100_000.0, quality=0.5,
            **dict(BASE, p=BASE["pcritical"]),
        )
    with pytest.raises(ValueError, match="greater than"):
        shah1982_boiling_alpha_local(
            mass_flux=100.0, heat_flux_inner=100_000.0, quality=0.5,
            **dict(BASE, liquid_density=BASE["vapor_density"]),
        )
