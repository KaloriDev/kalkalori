# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""
Smoke test for the v0.5.2 crossflow ε-NTU correction (outside mixed / inside
unmixed) in core.heat_transfer.ntu.

Also covers forward/inverse consistency and confirms counterflow /
cocurrentflow are unchanged.

Run:
    python -m core.tests.ntu_crossflow_test
"""

from __future__ import annotations

import math

from core.heat_transfer.ntu import effectiveness_ntu, ntu_from_effectiveness


def _eps_case_a(NTU: float, C_r: float) -> float:
    """Independent reference: outside (mixed) stream is C_min (Incropera Table 11.4)."""
    if C_r < 1e-9:
        return 1.0 - math.exp(-NTU)
    return 1.0 - math.exp(-(1.0 - math.exp(-C_r * NTU)) / C_r)


def _eps_case_b(NTU: float, C_r: float) -> float:
    """Independent reference: inside (unmixed) stream is C_min (Incropera Table 11.4)."""
    if C_r < 1e-9:
        return 1.0 - math.exp(-NTU)
    return (1.0 / C_r) * (1.0 - math.exp(-C_r * (1.0 - math.exp(-NTU))))


def test_outside_is_cmin() -> None:
    print("crossflow: C_outside < C_inside -> outside (mixed) stream is C_min")
    C_inside, C_outside, UA = 3540.0, 2014.0, 4000.0
    C_min, C_max = min(C_inside, C_outside), max(C_inside, C_outside)
    C_r = C_min / C_max
    NTU = UA / C_min

    eps = effectiveness_ntu(
        C_hot=C_inside, C_cold=C_outside, UA=UA,
        flow_arrangement="crossflow", C_inside=C_inside, C_outside=C_outside,
    )
    expected = _eps_case_a(NTU, C_r)
    assert math.isclose(eps, expected, rel_tol=1e-9), (eps, expected)


def test_inside_is_cmin() -> None:
    print("crossflow: C_inside < C_outside -> inside (unmixed) stream is C_min")
    C_inside, C_outside, UA = 2014.0, 3540.0, 4000.0
    C_min, C_max = min(C_inside, C_outside), max(C_inside, C_outside)
    C_r = C_min / C_max
    NTU = UA / C_min

    eps = effectiveness_ntu(
        C_hot=C_outside, C_cold=C_inside, UA=UA,
        flow_arrangement="crossflow", C_inside=C_inside, C_outside=C_outside,
    )
    expected = _eps_case_b(NTU, C_r)
    assert math.isclose(eps, expected, rel_tol=1e-9), (eps, expected)


def test_branch_selection_matters() -> None:
    print("crossflow: swapping which physical side is C_min changes eps (branch actually used)")
    C_a, C_b, UA = 3540.0, 2014.0, 4000.0

    eps_outside_min = effectiveness_ntu(
        C_hot=C_a, C_cold=C_b, UA=UA,
        flow_arrangement="crossflow", C_inside=C_a, C_outside=C_b,
    )
    eps_inside_min = effectiveness_ntu(
        C_hot=C_a, C_cold=C_b, UA=UA,
        flow_arrangement="crossflow", C_inside=C_b, C_outside=C_a,
    )
    assert abs(eps_outside_min - eps_inside_min) > 1e-6, (eps_outside_min, eps_inside_min)


def test_cr_near_zero() -> None:
    print("crossflow: Cr -> 0 limit (both branches reduce to 1 - exp(-NTU))")
    NTU = 1.5
    UA = 1000.0
    C_min = UA / NTU

    # Case A: outside is the (vanishingly small) C_min.
    C_outside_tiny = C_min
    C_inside_huge = C_min * 1.0e6
    eps_a = effectiveness_ntu(
        C_hot=C_inside_huge, C_cold=C_outside_tiny, UA=UA,
        flow_arrangement="crossflow", C_inside=C_inside_huge, C_outside=C_outside_tiny,
    )
    assert math.isclose(eps_a, 1.0 - math.exp(-NTU), rel_tol=1e-4), eps_a

    # Case B: inside is the (vanishingly small) C_min.
    C_inside_tiny = C_min
    C_outside_huge = C_min * 1.0e6
    eps_b = effectiveness_ntu(
        C_hot=C_outside_huge, C_cold=C_inside_tiny, UA=UA,
        flow_arrangement="crossflow", C_inside=C_inside_tiny, C_outside=C_outside_huge,
    )
    assert math.isclose(eps_b, 1.0 - math.exp(-NTU), rel_tol=1e-4), eps_b


def test_cr_near_one() -> None:
    print("crossflow: Cr -> 1 (nearly balanced capacity rates)")
    C_inside, C_outside, UA = 2000.0, 2001.0, 3000.0
    C_min, C_max = min(C_inside, C_outside), max(C_inside, C_outside)
    C_r = C_min / C_max
    NTU = UA / C_min

    eps = effectiveness_ntu(
        C_hot=C_inside, C_cold=C_outside, UA=UA,
        flow_arrangement="crossflow", C_inside=C_inside, C_outside=C_outside,
    )
    expected = _eps_case_b(NTU, C_r)  # C_outside > C_inside -> inside is C_min
    assert math.isclose(eps, expected, rel_tol=1e-9)
    assert 0.0 < eps < 1.0


def test_low_and_moderate_ntu() -> None:
    print("crossflow: low NTU (0.1) and moderate NTU (2.0)")
    C_inside, C_outside = 2500.0, 1800.0
    C_min = min(C_inside, C_outside)

    for NTU in (0.1, 2.0):
        UA = NTU * C_min
        eps = effectiveness_ntu(
            C_hot=C_inside, C_cold=C_outside, UA=UA,
            flow_arrangement="crossflow", C_inside=C_inside, C_outside=C_outside,
        )
        assert 0.0 < eps < 1.0
        # Monotonic sanity: higher NTU must not give lower effectiveness.
    eps_low = effectiveness_ntu(
        C_hot=C_inside, C_cold=C_outside, UA=0.1 * C_min,
        flow_arrangement="crossflow", C_inside=C_inside, C_outside=C_outside,
    )
    eps_mod = effectiveness_ntu(
        C_hot=C_inside, C_cold=C_outside, UA=2.0 * C_min,
        flow_arrangement="crossflow", C_inside=C_inside, C_outside=C_outside,
    )
    assert eps_mod > eps_low


def test_forward_inverse_round_trip() -> None:
    print("crossflow: forward/inverse consistency for both branches")
    cases = [
        (3540.0, 2014.0, 4000.0),  # outside is C_min
        (2014.0, 3540.0, 4000.0),  # inside is C_min
        (2000.0, 2001.0, 3000.0),  # Cr near 1
    ]
    for C_inside, C_outside, UA in cases:
        C_min = min(C_inside, C_outside)
        NTU_expected = UA / C_min
        eps = effectiveness_ntu(
            C_hot=C_inside, C_cold=C_outside, UA=UA,
            flow_arrangement="crossflow", C_inside=C_inside, C_outside=C_outside,
        )
        NTU_rt = ntu_from_effectiveness(
            eps, C_inside, C_outside,
            flow_arrangement="crossflow", C_inside=C_inside, C_outside=C_outside,
        )
        assert math.isclose(NTU_rt, NTU_expected, rel_tol=1e-6), (NTU_rt, NTU_expected)


def test_crossflow_requires_side_identity() -> None:
    print("crossflow: missing C_inside/C_outside raises ValueError")
    try:
        effectiveness_ntu(C_hot=3000.0, C_cold=2000.0, UA=4000.0, flow_arrangement="crossflow")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when C_inside/C_outside are not supplied")


def test_inverse_eps_max_guard() -> None:
    print("crossflow: inverse raises for effectiveness above the branch's eps_max")
    C_inside, C_outside, UA = 3540.0, 2014.0, 100.0  # small UA -> small eps_max at huge NTU request
    C_min, C_max = min(C_inside, C_outside), max(C_inside, C_outside)
    C_r = C_min / C_max
    eps_max = 1.0 - math.exp(-1.0 / C_r)  # Case A (outside is C_min)
    try:
        ntu_from_effectiveness(
            eps_max * 1.05, C_inside, C_outside,
            flow_arrangement="crossflow", C_inside=C_inside, C_outside=C_outside,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for effectiveness above eps_max")


def test_counterflow_unchanged() -> None:
    print("regression: counterflow forward/inverse unchanged by crossflow fix")
    C_hot, C_cold, UA = 3540.0, 2014.0, 5000.0
    C_min, C_max = min(C_hot, C_cold), max(C_hot, C_cold)
    C_r = C_min / C_max
    NTU = UA / C_min
    expected_eps = (1.0 - math.exp(-NTU * (1.0 - C_r))) / (1.0 - C_r * math.exp(-NTU * (1.0 - C_r)))

    eps = effectiveness_ntu(C_hot=C_hot, C_cold=C_cold, UA=UA, flow_arrangement="counterflow")
    assert math.isclose(eps, expected_eps, rel_tol=1e-9)

    NTU_rt = ntu_from_effectiveness(eps, C_hot, C_cold, flow_arrangement="counterflow")
    assert math.isclose(NTU_rt, NTU, rel_tol=1e-6)


def test_cocurrentflow_unchanged() -> None:
    print("regression: cocurrentflow forward/inverse unchanged by crossflow fix")
    C_hot, C_cold, UA = 3540.0, 2014.0, 4000.0
    C_min, C_max = min(C_hot, C_cold), max(C_hot, C_cold)
    C_r = C_min / C_max
    NTU = UA / C_min
    expected_eps = (1.0 - math.exp(-NTU * (1.0 + C_r))) / (1.0 + C_r)

    eps = effectiveness_ntu(C_hot=C_hot, C_cold=C_cold, UA=UA, flow_arrangement="cocurrentflow")
    assert math.isclose(eps, expected_eps, rel_tol=1e-9)

    NTU_rt = ntu_from_effectiveness(eps, C_hot, C_cold, flow_arrangement="cocurrentflow")
    assert math.isclose(NTU_rt, NTU, rel_tol=1e-6)


def main() -> None:
    test_outside_is_cmin()
    test_inside_is_cmin()
    test_branch_selection_matters()
    test_cr_near_zero()
    test_cr_near_one()
    test_low_and_moderate_ntu()
    test_forward_inverse_round_trip()
    test_crossflow_requires_side_identity()
    test_inverse_eps_max_guard()
    test_counterflow_unchanged()
    test_cocurrentflow_unchanged()

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
