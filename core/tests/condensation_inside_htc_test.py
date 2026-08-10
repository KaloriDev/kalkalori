# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for core.heat_transfer.condensation_inside (v0.6.2).

Shah (1979) in-tube condensation correlation: deterministic formula tests
plus quadrature/applicability behavior. No external validation dataset is
used (see module docstring of the source module for the citation).

Run:
    pytest -q core/tests/condensation_inside_htc_test.py
"""

from __future__ import annotations

import math

import pytest

from core.heat_transfer.condensation_inside import (
    SHAH_1979_LIQUID_REYNOLDS_RANGE,
    SHAH_1979_MASS_FLUX_RANGE_KG_M2S,
    SHAH_1979_REDUCED_PRESSURE_RANGE,
    condensation_zone_alpha_effective,
    shah_condensation_alpha_local,
)

# Representative saturated-water-like liquid properties at ~1 bar, well
# inside the correlation's documented applicability range.
P = 1.0e5
P_CRITICAL = 22.064e6
D_I = 0.02
MU_L = 2.82e-4
K_L = 0.679
CP_L = 4216.0
G = 100.0


def _alpha(x: float) -> float:
    return shah_condensation_alpha_local(
        x, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
    ).alpha


# ---------------------------------------------------------------------------
# Local correlation: finiteness, positivity, determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [0.001, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.999])
def test_local_alpha_finite_and_positive(x: float) -> None:
    alpha = _alpha(x)
    assert math.isfinite(alpha)
    assert alpha > 0.0


def test_no_nan_near_x_to_zero() -> None:
    for x in (1e-6, 1e-9, 1e-12):
        assert math.isfinite(_alpha(x))


def test_no_nan_near_x_to_one() -> None:
    for x in (1.0 - 1e-6, 1.0 - 1e-9, 1.0 - 1e-12):
        assert math.isfinite(_alpha(x))


def test_x_zero_and_x_one_rejected_on_local_correlation() -> None:
    with pytest.raises(ValueError):
        shah_condensation_alpha_local(
            0.0, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
        )
    with pytest.raises(ValueError):
        shah_condensation_alpha_local(
            1.0, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
        )


def test_deterministic_formula_reference() -> None:
    # h_LO/(k_L/D) = 0.023 Re_LO^0.8 Pr_L^0.4 ; h_TP/h_LO = (1-x)^0.8 +
    # 3.8 x^0.76 (1-x)^0.04 / p_r^0.38 -- Shah (1979).
    x = 0.4
    Re_LO = G * D_I / MU_L
    Pr_L = CP_L * MU_L / K_L
    p_r = P / P_CRITICAL
    alpha_LO = 0.023 * Re_LO**0.8 * Pr_L**0.4 * (K_L / D_I)
    factor = (1.0 - x) ** 0.8 + 3.8 * x**0.76 * (1.0 - x) ** 0.04 / p_r**0.38
    expected = alpha_LO * factor

    result = shah_condensation_alpha_local(
        x, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
    )
    assert result.alpha == pytest.approx(expected, rel=1e-12)
    assert result.alpha_LO == pytest.approx(alpha_LO, rel=1e-12)
    assert result.Re_LO == pytest.approx(Re_LO, rel=1e-12)
    assert result.Pr_L == pytest.approx(Pr_L, rel=1e-12)
    assert result.p_r == pytest.approx(p_r, rel=1e-12)


# ---------------------------------------------------------------------------
# Applicability warnings
# ---------------------------------------------------------------------------
def test_out_of_range_reynolds_warns() -> None:
    # Very small G/D combination drives Re_LO below the documented range.
    result = shah_condensation_alpha_local(
        0.5, p=P, p_critical=P_CRITICAL, G=0.5, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
    )
    assert any(w.code == "SHAH_CONDENSATION_OUT_OF_RANGE" for w in result.warnings)
    assert result.Re_LO < SHAH_1979_LIQUID_REYNOLDS_RANGE[0]


def test_in_range_case_has_no_applicability_warning() -> None:
    result = shah_condensation_alpha_local(
        0.5, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
    )
    assert result.warnings == []
    assert SHAH_1979_REDUCED_PRESSURE_RANGE[0] <= result.p_r <= SHAH_1979_REDUCED_PRESSURE_RANGE[1]
    assert SHAH_1979_MASS_FLUX_RANGE_KG_M2S[0] <= G <= SHAH_1979_MASS_FLUX_RANGE_KG_M2S[1]


# ---------------------------------------------------------------------------
# Zone-averaged (quadrature) HTC
# ---------------------------------------------------------------------------
def test_zone_alpha_effective_finite_full_range() -> None:
    zone = condensation_zone_alpha_effective(
        x_in=1.0, x_out=0.0, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
    )
    assert math.isfinite(zone.alpha_effective)
    assert zone.alpha_effective > 0.0


def test_zone_alpha_effective_finite_partial_range() -> None:
    zone = condensation_zone_alpha_effective(
        x_in=0.8, x_out=0.2, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
    )
    assert math.isfinite(zone.alpha_effective)
    assert zone.alpha_effective > 0.0


def test_zone_rejects_x_out_at_or_above_x_in() -> None:
    with pytest.raises(ValueError):
        condensation_zone_alpha_effective(
            x_in=0.5, x_out=0.5, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
        )
    with pytest.raises(ValueError):
        condensation_zone_alpha_effective(
            x_in=0.3, x_out=0.6, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
        )


def test_zone_quadrature_independent_of_endpoint_singularities() -> None:
    # Spans the full [0, 1] range, including both boundary degeneracies of
    # the local correlation; must not raise and must be finite (the
    # quadrature nodes never touch x=0 or x=1 exactly).
    zone = condensation_zone_alpha_effective(
        x_in=1.0, x_out=0.0, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
    )
    assert math.isfinite(zone.alpha_effective)


def test_zone_alpha_matches_manual_quadrature_reference() -> None:
    # Cross-check against an independent hand-rolled 5-point Gauss-Legendre
    # mean, using the same nodes/weights but computed independently here.
    nodes = (-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664)
    weights = (0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891)
    x_in, x_out = 0.9, 0.1
    half = 0.5 * (x_in - x_out)
    mid = 0.5 * (x_in + x_out)
    total = 0.0
    for node, weight in zip(nodes, weights):
        total += weight * _alpha(mid + half * node)
    expected = total * half / (x_in - x_out)

    zone = condensation_zone_alpha_effective(
        x_in=x_in, x_out=x_out, p=P, p_critical=P_CRITICAL, G=G, D_i=D_I, mu_L=MU_L, k_L=K_L, cp_L=CP_L
    )
    assert zone.alpha_effective == pytest.approx(expected, rel=1e-12)
