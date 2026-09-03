# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for the shared physical surface-margin definition."""

from __future__ import annotations

import math

import pytest

from core.models.surface_margin import calculate_surface_margin_factor


@pytest.mark.parametrize(
    ("UA_actual", "UA_process", "expected"),
    [
        (110.0, 100.0, 0.10),
        (100.0, 100.0, 0.0),
        (90.0, 100.0, -0.10),
    ],
)
def test_calculate_surface_margin_factor_uses_shared_ua_definition(
    UA_actual: float,
    UA_process: float,
    expected: float,
) -> None:
    assert calculate_surface_margin_factor(
        UA_actual=UA_actual,
        UA_process=UA_process,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("UA_actual", "UA_process"),
    [
        (0.0, 100.0),
        (-0.0, 100.0),
        (-1.0, 100.0),
        (math.nan, 100.0),
        (math.inf, 100.0),
        (-math.inf, 100.0),
        (100.0, 0.0),
        (100.0, -0.0),
        (100.0, -1.0),
        (100.0, math.nan),
        (100.0, math.inf),
        (100.0, -math.inf),
    ],
)
def test_calculate_surface_margin_factor_rejects_nonphysical_ua(
    UA_actual: float,
    UA_process: float,
) -> None:
    with pytest.raises(ValueError):
        calculate_surface_margin_factor(
            UA_actual=UA_actual,
            UA_process=UA_process,
        )


def test_calculate_surface_margin_factor_rejects_nonfinite_ratio() -> None:
    with pytest.raises(ValueError):
        calculate_surface_margin_factor(
            UA_actual=1.0e308,
            UA_process=1.0e-308,
        )
