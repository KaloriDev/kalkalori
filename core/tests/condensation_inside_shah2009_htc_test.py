# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for core.heat_transfer.condensation_inside_shah2009 (v0.6.2 patch).

Shah (2009) in-tube condensation correlation: deterministic formula tests,
regime-selection tests (I/II/III, horizontal vs vertical/inclined),
quadrature/applicability behavior, and a direct low-mass-flux comparison
against the legacy Shah (1979) implementation that motivated this patch.

Run:
    pytest -q core/tests/condensation_inside_shah2009_htc_test.py
"""

from __future__ import annotations

import math

import pytest

from core.geometry.tube import TubeOrientation
from core.heat_transfer.condensation_inside import shah_condensation_alpha_local
from core.heat_transfer.condensation_inside_shah2009 import (
    GRAVITY_M_S2,
    SHAH_2009_HORIZONTAL_MIN_VAPOR_REYNOLDS,
    SHAH_2009_LIQUID_REYNOLDS_RANGE,
    SHAH_2009_MASS_FLUX_RANGE_KG_M2S,
    condensation_zone_alpha_effective_2009,
    shah2009_condensation_alpha_local,
)

# Representative saturated-water-like properties at ~1 bar, well inside the
# correlation's documented applicability range (Table 4).
P = 1.0e5
P_CRITICAL = 22.064e6
D_I = 0.02
MU_L = 2.82e-4
MU_G = 1.2e-5
K_L = 0.679
CP_L = 4216.0
RHO_L = 958.0
RHO_G = 0.598
G_HIGH = 300.0  # kg/(m2*s), well inside Shah 1979's own range too
G_LOW = 3.0  # kg/(m2*s), far below Shah 1979's 10.8 lower bound


def _local(x: float, *, G: float = G_HIGH, orientation: TubeOrientation = TubeOrientation.HORIZONTAL):
    return shah2009_condensation_alpha_local(
        x, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, orientation=orientation,
        mu_L=MU_L, mu_G=MU_G, k_L=K_L, cp_L=CP_L, rho_L=RHO_L, rho_G=RHO_G,
    )


# ---------------------------------------------------------------------------
# Local correlation: finiteness, positivity, determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [0.001, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.999])
@pytest.mark.parametrize("orientation", list(TubeOrientation))
@pytest.mark.parametrize("G", [G_LOW, G_HIGH])
def test_local_alpha_finite_and_positive(x: float, orientation: TubeOrientation, G: float) -> None:
    result = _local(x, G=G, orientation=orientation)
    assert math.isfinite(result.alpha)
    assert result.alpha > 0.0
    assert result.regime in ("I", "II", "III")


def test_no_nan_near_x_to_zero() -> None:
    for x in (1e-6, 1e-9, 1e-12):
        assert math.isfinite(_local(x).alpha)


def test_no_nan_near_x_to_one() -> None:
    for x in (1.0 - 1e-6, 1.0 - 1e-9, 1.0 - 1e-12):
        assert math.isfinite(_local(x).alpha)


def test_x_zero_and_x_one_rejected_on_local_correlation() -> None:
    with pytest.raises(ValueError):
        _local(0.0)
    with pytest.raises(ValueError):
        _local(1.0)


def test_horizontal_regime_iii_not_defined() -> None:
    # Shah (2009) defines no third (gravity-only) regime for horizontal
    # tubes -- module docstring, "A third regime is expected at very low
    # flow rates. Analyzable data were not available for such conditions."
    for x in (0.99, 0.9, 0.5, 0.1, 0.01):
        result = _local(x, G=0.05, orientation=TubeOrientation.HORIZONTAL)
        assert result.regime in ("I", "II")


# ---------------------------------------------------------------------------
# Deterministic formula reference (Eq. 4-12, Shah 2009)
# ---------------------------------------------------------------------------
def _reference_terms(x: float, G: float):
    Re_LT = G * D_I / MU_L
    Re_LS = G * (1.0 - x) * D_I / MU_L
    Re_GT = G * D_I / MU_G
    Pr_L = CP_L * MU_L / K_L
    p_r = P / P_CRITICAL
    Z = ((1.0 - x) / x) ** 0.8 * p_r**0.4
    Jg = (x * G) / math.sqrt(GRAVITY_M_S2 * D_I * RHO_G * (RHO_L - RHO_G))
    h_LT = 0.023 * Re_LT**0.8 * Pr_L**0.4 * (K_L / D_I)
    n = 0.0058 + 0.557 * p_r
    h_I = h_LT * (MU_L / (14.0 * MU_G)) ** n * (
        (1.0 - x) ** 0.8 + 3.8 * x**0.76 * (1.0 - x) ** 0.04 / p_r**0.38
    )
    h_Nu = 1.32 * Re_LS ** (-1.0 / 3.0) * (RHO_L * (RHO_L - RHO_G) * GRAVITY_M_S2 * K_L**3 / MU_L**2) ** (1.0 / 3.0)
    return Re_LT, Re_LS, Re_GT, Pr_L, p_r, Z, Jg, h_I, h_Nu, n


def test_deterministic_formula_reference_horizontal_regime_I() -> None:
    x = 0.7
    result = _local(x, G=G_HIGH, orientation=TubeOrientation.HORIZONTAL)
    Re_LT, Re_LS, Re_GT, Pr_L, p_r, Z, Jg, h_I, h_Nu, n = _reference_terms(x, G_HIGH)
    boundary = 0.98 * (Z + 0.263) ** (-0.62)
    assert Jg >= boundary  # sanity: this case is engineered to land in Regime I
    assert result.regime == "I"
    assert result.alpha == pytest.approx(h_I, rel=1e-10)
    assert result.h_I == pytest.approx(h_I, rel=1e-10)
    assert result.h_Nu == pytest.approx(h_Nu, rel=1e-10)
    assert result.Jg == pytest.approx(Jg, rel=1e-12)
    assert result.Z == pytest.approx(Z, rel=1e-12)
    assert result.Re_LT == pytest.approx(Re_LT, rel=1e-12)
    assert result.Re_LS == pytest.approx(Re_LS, rel=1e-12)
    assert result.Re_GT == pytest.approx(Re_GT, rel=1e-12)
    assert result.Pr_L == pytest.approx(Pr_L, rel=1e-12)
    assert result.p_r == pytest.approx(p_r, rel=1e-12)
    assert result.n == pytest.approx(n, rel=1e-12)


def test_deterministic_formula_reference_horizontal_regime_II() -> None:
    x = 0.5
    result = _local(x, G=G_LOW, orientation=TubeOrientation.HORIZONTAL)
    Re_LT, Re_LS, Re_GT, Pr_L, p_r, Z, Jg, h_I, h_Nu, n = _reference_terms(x, G_LOW)
    boundary = 0.98 * (Z + 0.263) ** (-0.62)
    assert Jg < boundary  # sanity: engineered to land in Regime II
    assert result.regime == "II"
    assert result.alpha == pytest.approx(h_I + h_Nu, rel=1e-10)


def test_deterministic_formula_reference_vertical_regime_I() -> None:
    x = 0.7
    result = _local(x, G=G_HIGH, orientation=TubeOrientation.VERTICAL_DOWNFLOW)
    Re_LT, Re_LS, Re_GT, Pr_L, p_r, Z, Jg, h_I, h_Nu, n = _reference_terms(x, G_HIGH)
    boundary_I = 1.0 / (2.4 * Z + 0.73)
    assert Jg >= boundary_I  # sanity
    assert result.regime == "I"
    assert result.alpha == pytest.approx(h_I, rel=1e-10)


def test_deterministic_formula_reference_vertical_regime_III() -> None:
    x = 0.3
    result = _local(x, G=G_LOW, orientation=TubeOrientation.VERTICAL_DOWNFLOW)
    Re_LT, Re_LS, Re_GT, Pr_L, p_r, Z, Jg, h_I, h_Nu, n = _reference_terms(x, G_LOW)
    boundary_III = 0.89 - 0.93 * math.exp(-0.087 * Z ** (-1.17))
    assert Jg <= boundary_III  # sanity: engineered to land in Regime III
    assert result.regime == "III"
    assert result.alpha == pytest.approx(h_Nu, rel=1e-10)


def test_deterministic_formula_reference_vertical_regime_II() -> None:
    x = 0.6
    G = 15.0
    result = _local(x, G=G, orientation=TubeOrientation.VERTICAL_DOWNFLOW)
    Re_LT, Re_LS, Re_GT, Pr_L, p_r, Z, Jg, h_I, h_Nu, n = _reference_terms(x, G)
    boundary_I = 1.0 / (2.4 * Z + 0.73)
    boundary_III = 0.89 - 0.93 * math.exp(-0.087 * Z ** (-1.17))
    assert boundary_III < Jg < boundary_I  # sanity: engineered to land in Regime II
    assert result.regime == "II"
    assert result.alpha == pytest.approx(h_I + h_Nu, rel=1e-10)


def test_inclined_downward_uses_same_regime_logic_as_vertical() -> None:
    # Shah (2009): tubes inclined >= 15 deg downward follow the same
    # vertical-tube regime boundaries (module docstring).
    for x in (0.7, 0.3):
        vertical = _local(x, G=G_LOW, orientation=TubeOrientation.VERTICAL_DOWNFLOW)
        inclined = _local(x, G=G_LOW, orientation=TubeOrientation.INCLINED_DOWNWARD)
        assert inclined.regime == vertical.regime
        assert inclined.alpha == pytest.approx(vertical.alpha, rel=1e-12)


# ---------------------------------------------------------------------------
# Applicability warnings
# ---------------------------------------------------------------------------
def test_out_of_range_mass_flux_warns() -> None:
    result = _local(0.5, G=G_LOW, orientation=TubeOrientation.HORIZONTAL)
    assert any(w.code == "SHAH_2009_CONDENSATION_OUT_OF_RANGE" for w in result.warnings)
    assert G_LOW < SHAH_2009_MASS_FLUX_RANGE_KG_M2S[0]


def test_in_range_case_has_no_applicability_warning() -> None:
    result = _local(0.5, G=G_HIGH, orientation=TubeOrientation.VERTICAL_DOWNFLOW)
    assert result.warnings == []


def test_horizontal_low_reGT_warns_vertical_does_not() -> None:
    horizontal = _local(0.5, G=G_LOW, orientation=TubeOrientation.HORIZONTAL)
    vertical = _local(0.5, G=G_LOW, orientation=TubeOrientation.VERTICAL_DOWNFLOW)
    assert horizontal.Re_GT < SHAH_2009_HORIZONTAL_MIN_VAPOR_REYNOLDS
    assert any(w.code == "SHAH_2009_HORIZONTAL_LOW_REGT_EXTRAPOLATION" for w in horizontal.warnings)
    assert not any(w.code == "SHAH_2009_HORIZONTAL_LOW_REGT_EXTRAPOLATION" for w in vertical.warnings)


# ---------------------------------------------------------------------------
# Zone-averaged (quadrature) HTC
# ---------------------------------------------------------------------------
def test_zone_alpha_effective_finite_full_range() -> None:
    zone = condensation_zone_alpha_effective_2009(
        x_in=1.0, x_out=0.0, p=P, p_critical=P_CRITICAL, G=G_HIGH, D_i=D_I,
        orientation=TubeOrientation.HORIZONTAL,
        mu_L=MU_L, mu_G=MU_G, k_L=K_L, cp_L=CP_L, rho_L=RHO_L, rho_G=RHO_G,
    )
    assert math.isfinite(zone.alpha_effective)
    assert zone.alpha_effective > 0.0


def test_zone_rejects_x_out_at_or_above_x_in() -> None:
    with pytest.raises(ValueError):
        condensation_zone_alpha_effective_2009(
            x_in=0.5, x_out=0.5, p=P, p_critical=P_CRITICAL, G=G_HIGH, D_i=D_I,
            orientation=TubeOrientation.HORIZONTAL,
            mu_L=MU_L, mu_G=MU_G, k_L=K_L, cp_L=CP_L, rho_L=RHO_L, rho_G=RHO_G,
        )


def test_zone_alpha_matches_manual_quadrature_reference() -> None:
    nodes = (-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664)
    weights = (0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891)
    x_in, x_out = 0.9, 0.1
    half = 0.5 * (x_in - x_out)
    mid = 0.5 * (x_in + x_out)
    total = 0.0
    for node, weight in zip(nodes, weights):
        total += weight * _local(mid + half * node, G=G_LOW, orientation=TubeOrientation.HORIZONTAL).alpha
    expected = total * half / (x_in - x_out)

    zone = condensation_zone_alpha_effective_2009(
        x_in=x_in, x_out=x_out, p=P, p_critical=P_CRITICAL, G=G_LOW, D_i=D_I,
        orientation=TubeOrientation.HORIZONTAL,
        mu_L=MU_L, mu_G=MU_G, k_L=K_L, cp_L=CP_L, rho_L=RHO_L, rho_G=RHO_G,
    )
    assert zone.alpha_effective == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Regime-transition continuity (no NaN/blow-up crossing a boundary)
# ---------------------------------------------------------------------------
def test_no_nan_or_blowup_scanning_across_regime_boundaries() -> None:
    previous_alpha = None
    for x100 in range(1, 100):
        x = x100 / 100.0
        result = _local(x, G=15.0, orientation=TubeOrientation.VERTICAL_DOWNFLOW)
        assert math.isfinite(result.alpha)
        assert result.alpha > 0.0
        if previous_alpha is not None:
            # A regime boundary can add/remove the h_Nu contribution
            # discontinuously (an inherent, published characteristic of
            # Shah 2009's regime formulation -- see module docstring), but
            # it must never look like a numerical blow-up.
            assert result.alpha < 50.0 * previous_alpha
            assert result.alpha > previous_alpha / 50.0
        previous_alpha = result.alpha


# ---------------------------------------------------------------------------
# Section 16/17: low-flow vs high-flow comparison against the legacy
# Shah (1979) implementation -- the direct evidence that the new model
# fixes the "condensation HTC ~1000x too small at low G" symptom without
# any calibration factor.
# ---------------------------------------------------------------------------
def test_low_flow_shah2009_far_exceeds_legacy_shah1979() -> None:
    x = 0.5
    shah1979 = shah_condensation_alpha_local(
        x, p=P, p_critical=P_CRITICAL, G=G_LOW, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
    )
    shah2009 = _local(x, G=G_LOW, orientation=TubeOrientation.HORIZONTAL)
    assert shah2009.regime != "I"  # gravity contribution must be active at this G
    assert shah2009.alpha > 5.0 * shah1979.alpha


def test_high_flow_shah2009_regime_I_reasonably_close_to_legacy_shah1979() -> None:
    # Inside the range Shah (1979) was itself validated for, Shah (2009)'s
    # Regime I selection should behave similarly in order of magnitude
    # (both are the same forced-convective physics; 2009 additionally
    # applies the published viscosity-ratio correction of Eq. 8a).
    x = 0.5
    shah1979 = shah_condensation_alpha_local(
        x, p=P, p_critical=P_CRITICAL, G=G_HIGH, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
    )
    shah2009 = _local(x, G=G_HIGH, orientation=TubeOrientation.HORIZONTAL)
    assert shah2009.regime == "I"
    ratio = shah2009.alpha / shah1979.alpha
    assert 0.2 < ratio < 5.0
