# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Regressions for core.heat_transfer.outside_flow.check_outside_ht_applicability.

This Zukauskas-style applicability/diagnostic check previously had no .py
pytest coverage and was exercised only manually, without assertions, through
a notebook (since removed). This module ports the notebook's Re/Pr/
geometry sweep into a compact, parametrized, assertion-based regression.
"""

from __future__ import annotations

import pytest

from core.heat_transfer.outside_flow import check_outside_ht_applicability

# A synthetic mid-range tube bank with no applicability concerns of its own,
# so each parametrized case can vary exactly one input at a time.
_BASELINE: dict[str, object] = dict(
    Re=1_000.0,
    Pr=0.7,
    tube_outer_diameter=0.025,
    tube_pitch_transverse=0.060,  # ST/D = 2.4
    tube_pitch_longitudinal=0.060,  # SL/D = 2.4
    layout="inline",
    n_rows=10,
    use_vmax_for_ht=True,
)


def _codes(overrides: dict[str, object]) -> list[str]:
    warnings = check_outside_ht_applicability(**{**_BASELINE, **overrides})
    return [w.code for w in warnings]


def test_nominal_case_has_no_applicability_warnings() -> None:
    assert check_outside_ht_applicability(**_BASELINE) == []


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        # Reynolds-number applicability boundary.
        ({"Re": 0.0}, "outside_ht_re_nonpositive"),
        ({"Re": 10.0}, "outside_ht_re_extremely_low"),
        ({"Re": 50.0}, "outside_ht_re_low"),
        ({"Re": 3.0e5}, "outside_ht_re_high"),
        # Prandtl-number applicability boundary.
        ({"Pr": 0.0}, "outside_ht_pr_nonpositive"),
        ({"Pr": 0.3}, "outside_ht_pr_low"),
        ({"Pr": 600.0}, "outside_ht_pr_high"),
        # Row-count applicability.
        ({"n_rows": 1}, "outside_ht_single_row"),
        ({"n_rows": 3}, "outside_ht_few_rows"),
        # Transverse / longitudinal pitch-ratio geometry.
        ({"tube_pitch_transverse": 0.020}, "outside_ht_st_over_d_invalid"),
        ({"tube_pitch_transverse": 0.026}, "outside_ht_st_over_d_near_blockage"),
        ({"tube_pitch_transverse": 0.110}, "outside_ht_st_over_d_large"),
        ({"tube_pitch_longitudinal": 0.020}, "outside_ht_sl_over_d_invalid"),
        ({"tube_pitch_longitudinal": 0.026}, "outside_ht_sl_over_d_small"),
        ({"tube_pitch_longitudinal": 0.110}, "outside_ht_sl_over_d_large"),
        # Non-standard velocity reference and tight staggered geometry.
        ({"use_vmax_for_ht": False}, "outside_ht_velocity_reference_nonstandard"),
        (
            {"layout": "staggered", "tube_pitch_longitudinal": 0.028},
            "outside_ht_staggered_tight_sl",
        ),
    ],
)
def test_applicability_boundary_raises_expected_code(
    overrides: dict[str, object], expected_code: str
) -> None:
    assert expected_code in _codes(overrides)


def test_nonpositive_geometry_short_circuits_before_ratio_checks() -> None:
    warnings = check_outside_ht_applicability(
        **{**_BASELINE, "tube_outer_diameter": 0.0}
    )
    codes = [w.code for w in warnings]

    assert "outside_ht_diameter_nonpositive" in codes
    # ST/D and SL/D are undefined once the diameter is non-positive; the
    # function must not also emit a ratio-based warning in that case.
    assert "outside_ht_st_over_d_invalid" not in codes
    assert "outside_ht_sl_over_d_invalid" not in codes
