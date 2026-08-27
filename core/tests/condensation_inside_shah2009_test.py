"""Transport-only regression coverage for the Shah (2009) correlation."""

import math

import pytest

from core.geometry.tube import TubeOrientation
from core.heat_transfer.condensation_inside_shah2009 import (
    SHAH_2009_CORRELATION,
    shah2009_condensation_alpha_local,
)


# Fixed saturated-water transport snapshot at p=1 MPa. The correlation test
# intentionally has no dependency on IAPWS or any heat-exchanger object.
PLAIN_SI_INPUTS = dict(
    p=1.0e6,
    pcritical=22.064e6,
    tube_inner_diameter=0.020,
    orientation=TubeOrientation.VERTICAL_DOWNWARD,
    liquid_density=887.1274516747791,
    vapor_density=5.145385853182684,
    liquid_viscosity=0.00015048492650911248,
    vapor_viscosity=1.4981316222701132e-05,
    liquid_conductivity=0.6713377268728457,
    liquid_specific_heat=4405.112049727733,
)


def _independent_reference(*, mass_flux, quality, orientation):
    """Direct transcription of Shah (2009), equations 4--12."""
    p = PLAIN_SI_INPUTS["p"]
    pcritical = PLAIN_SI_INPUTS["pcritical"]
    D = PLAIN_SI_INPUTS["tube_inner_diameter"]
    rho_l = PLAIN_SI_INPUTS["liquid_density"]
    rho_v = PLAIN_SI_INPUTS["vapor_density"]
    mu_l = PLAIN_SI_INPUTS["liquid_viscosity"]
    mu_v = PLAIN_SI_INPUTS["vapor_viscosity"]
    k_l = PLAIN_SI_INPUTS["liquid_conductivity"]
    cp_l = PLAIN_SI_INPUTS["liquid_specific_heat"]
    x = quality
    G = mass_flux
    p_r = p / pcritical
    Re_LT = G * D / mu_l
    Re_GT = G * D / mu_v
    Re_LS = G * (1.0 - x) * D / mu_l
    Pr_l = cp_l * mu_l / k_l
    Z = ((1.0 - x) / x) ** 0.8 * p_r**0.4
    J_g = x * G / math.sqrt(9.80665 * D * rho_v * (rho_l - rho_v))
    h_LT = 0.023 * Re_LT**0.8 * Pr_l**0.4 * k_l / D
    n = 0.0058 + 0.557 * p_r
    h_I = h_LT * (mu_l / (14.0 * mu_v)) ** n * (
        (1.0 - x) ** 0.8
        + 3.8 * x**0.76 * (1.0 - x) ** 0.04 / p_r**0.38
    )
    h_Nu = 1.32 * Re_LS ** (-1.0 / 3.0) * (
        rho_l * (rho_l - rho_v) * 9.80665 * k_l**3 / mu_l**2
    ) ** (1.0 / 3.0)
    if orientation is TubeOrientation.HORIZONTAL:
        regime = "I" if J_g >= 0.98 * (Z + 0.263) ** (-0.62) else "II"
    else:
        boundary_I = 1.0 / (2.4 * Z + 0.73)
        boundary_III = 0.89 - 0.93 * math.exp(-0.087 * Z ** (-1.17))
        regime = "I" if J_g >= boundary_I else "III" if J_g <= boundary_III else "II"
    alpha = h_I if regime == "I" else h_Nu if regime == "III" else h_I + h_Nu
    return {
        "Re_LT": Re_LT,
        "Re_GT": Re_GT,
        "Re_LS": Re_LS,
        "Pr_l": Pr_l,
        "p_r": p_r,
        "Z": Z,
        "J_g": J_g,
        "h_I": h_I,
        "h_Nu": h_Nu,
        "regime": regime,
        "alpha": alpha,
    }


@pytest.mark.parametrize(
    ("mass_flux", "quality", "regime", "alpha"),
    [
        (4.0, 0.9, "III", 16430.19830525919),
        (4.0, 0.5, "III", 9608.438257386568),
        (4.0, 0.1, "II", 8250.170609835557),
        (20.0, 0.9, "III", 9608.438257386568),
        (20.0, 0.5, "II", 8811.248447595906),
        (20.0, 0.1, "II", 5892.530573144826),
        (100.0, 0.9, "I", 15921.854888307133),
        (100.0, 0.5, "I", 11568.207006548264),
        (100.0, 0.1, "I", 4614.220063163214),
    ],
)
def test_plain_si_low_medium_high_flux_quality_grid_is_stable(
    mass_flux, quality, regime, alpha
):
    result = shah2009_condensation_alpha_local(
        mass_flux=mass_flux,
        quality=quality,
        **PLAIN_SI_INPUTS,
    )
    assert result.correlation == SHAH_2009_CORRELATION
    assert result.regime == regime
    assert result.alpha == pytest.approx(alpha, rel=2.0e-12)
    assert result.h_I == result.h_forced
    assert result.h_Nu == result.h_gravity
    assert result.p_r == result.reduced_pressure
    assert result.n == result.exponent_n
    reference = _independent_reference(
        mass_flux=mass_flux,
        quality=quality,
        orientation=PLAIN_SI_INPUTS["orientation"],
    )
    assert result.Re_LT == pytest.approx(reference["Re_LT"], rel=2.0e-12)
    assert result.Re_GT == pytest.approx(reference["Re_GT"], rel=2.0e-12)
    assert result.Re_LS == pytest.approx(reference["Re_LS"], rel=2.0e-12)
    assert result.Pr_liquid == pytest.approx(reference["Pr_l"], rel=2.0e-12)
    assert result.p_r == pytest.approx(reference["p_r"], rel=2.0e-12)
    assert result.Z == pytest.approx(reference["Z"], rel=2.0e-12)
    assert result.J_g == pytest.approx(reference["J_g"], rel=2.0e-12)
    assert result.h_I == pytest.approx(reference["h_I"], rel=2.0e-12)
    assert result.h_Nu == pytest.approx(reference["h_Nu"], rel=2.0e-12)
    assert result.regime == reference["regime"]
    assert result.alpha == pytest.approx(reference["alpha"], rel=2.0e-12)


@pytest.mark.parametrize(
    ("mass_flux", "quality", "regime"),
    [(300.0, 0.7, "I"), (3.0, 0.5, "II")],
)
def test_horizontal_regime_map_matches_independent_reference(
    mass_flux, quality, regime
):
    inputs = dict(PLAIN_SI_INPUTS, orientation=TubeOrientation.HORIZONTAL)
    result = shah2009_condensation_alpha_local(
        mass_flux=mass_flux, quality=quality, **inputs
    )
    reference = _independent_reference(
        mass_flux=mass_flux,
        quality=quality,
        orientation=TubeOrientation.HORIZONTAL,
    )
    assert result.regime == reference["regime"] == regime
    assert result.alpha == pytest.approx(reference["alpha"], rel=2.0e-12)


def test_horizontal_low_re_behavior_is_explicit_and_never_selects_regime_iii():
    inputs = dict(PLAIN_SI_INPUTS, orientation=TubeOrientation.HORIZONTAL)
    for quality in (0.01, 0.1, 0.5, 0.9, 0.99):
        result = shah2009_condensation_alpha_local(
            mass_flux=3.0, quality=quality, **inputs
        )
        assert result.regime in {"I", "II"}
        assert math.isfinite(result.alpha) and result.alpha > 0.0
    midpoint = shah2009_condensation_alpha_local(
        mass_flux=3.0, quality=0.5, **inputs
    )
    assert midpoint.Re_GT <= 35_000.0
    assert "STEAM_CONDENSATION_HORIZONTAL_LOW_RE_GT_UNVERIFIED" in {
        warning.code for warning in midpoint.warnings
    }


def test_regime_transition_sweep_remains_finite_without_numerical_blowup():
    previous = None
    for index in range(1, 100):
        result = shah2009_condensation_alpha_local(
            mass_flux=20.0,
            quality=index / 100.0,
            **PLAIN_SI_INPUTS,
        )
        assert result.regime in {"I", "II", "III"}
        assert math.isfinite(result.alpha) and result.alpha > 0.0
        if previous is not None:
            assert previous / 50.0 < result.alpha < previous * 50.0


def test_vertical_upward_orientation_is_explicitly_rejected():
    inputs = dict(PLAIN_SI_INPUTS, orientation=TubeOrientation.VERTICAL_UPWARD)
    with pytest.raises(ValueError, match="upward"):
        shah2009_condensation_alpha_local(mass_flux=20.0, quality=0.5, **inputs)
        previous = result.alpha
