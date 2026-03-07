# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Warning and applicability structures for engineering calculations.

Purpose
-------
This module defines lightweight data structures for:
- applicability checks,
- model limitations,
- diagnostic warnings.

It is intentionally generic so it can be reused across:
- outside heat transfer,
- internal heat transfer,
- pressure drop,
- NTU calculations,
- future phase-change models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


WarningSeverity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class ModelWarning:
    """
    Single diagnostic warning or applicability notice.

    Parameters
    ----------
    code : str
        Short machine-readable identifier, e.g.:
        'outside_ht_re_out_of_range'
    message : str
        Human-readable explanation.
    severity : {"info", "warning", "critical"}
        Importance level.
    source : str
        Origin of the warning, e.g.:
        'outside_ht', 'outside_dp', 'internal_ht', 'ntu'
    """

    code: str
    message: str
    severity: WarningSeverity
    source: str


@dataclass(frozen=True)
class ApplicabilityRange:
    """
    Declared applicability interval for a scalar parameter.

    Parameters
    ----------
    name : str
        Parameter name, e.g. 'Re', 'Pr', 'S_T/D_o'
    lower : float | None
        Lower bound (inclusive), or None if not defined
    upper : float | None
        Upper bound (inclusive), or None if not defined
    units : str
        Units label for reporting; use '-' for dimensionless quantities
    """

    name: str
    lower: float | None
    upper: float | None
    units: str = "-"


def check_range(
    *,
    name: str,
    value: float,
    lower: float | None = None,
    upper: float | None = None,
    units: str = "-",
    source: str,
    code_prefix: str,
    severity: WarningSeverity = "warning",
) -> list[ModelWarning]:
    """
    Generate warnings when a value is outside a declared applicability range.
    """
    warnings: list[ModelWarning] = []

    if lower is not None and value < lower:
        warnings.append(
            ModelWarning(
                code=f"{code_prefix}_{name.lower()}_below_range",
                message=(
                    f"{name} = {value:.6g} {units} is below the recommended "
                    f"lower limit {lower:.6g} {units}."
                ),
                severity=severity,
                source=source,
            )
        )

    if upper is not None and value > upper:
        warnings.append(
            ModelWarning(
                code=f"{code_prefix}_{name.lower()}_above_range",
                message=(
                    f"{name} = {value:.6g} {units} is above the recommended "
                    f"upper limit {upper:.6g} {units}."
                ),
                severity=severity,
                source=source,
            )
        )

    return warnings


def make_warning(
    *,
    code: str,
    message: str,
    source: str,
    severity: WarningSeverity = "warning",
) -> ModelWarning:
    """
    Convenience constructor for ad hoc warnings.
    """
    return ModelWarning(
        code=code,
        message=message,
        severity=severity,
        source=source,
    )


def has_critical_warnings(warnings: list[ModelWarning] | tuple[ModelWarning, ...]) -> bool:
    """
    Return True if any warning has severity == 'critical'.
    """
    return any(w.severity == "critical" for w in warnings)