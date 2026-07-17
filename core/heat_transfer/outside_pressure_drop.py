# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
#
# -------------------------------------------------------------------------
# DEPRECATED compatibility shim (since v0.5.6)
# -------------------------------------------------------------------------
#
# Outside (crossflow tube-bank) pressure-drop code now lives in
# ``core.pressure_drop.outside_pressure_drop``. This module re-exports the
# same objects so that previous import paths keep working; there is exactly
# one implementation, in ``core.pressure_drop``. Prefer importing from
# ``core.pressure_drop`` (or ``core.pressure_drop.outside_pressure_drop``)
# directly in new code.
#
# -------------------------------------------------------------------------

from __future__ import annotations

from core.pressure_drop.outside_pressure_drop import (
    Layout,
    EulerRequest,
    EulerResult,
    EulerProvider,
    ZukauskasEulerProvider,
    EsduEulerProvider,
    GaddisGnielinskiEulerProvider,
    evaluate_euler,
    pressure_drop_from_euler,
    check_outside_dp_applicability,
)

__all__ = [
    "Layout",
    "EulerRequest",
    "EulerResult",
    "EulerProvider",
    "ZukauskasEulerProvider",
    "EsduEulerProvider",
    "GaddisGnielinskiEulerProvider",
    "evaluate_euler",
    "pressure_drop_from_euler",
    "check_outside_dp_applicability",
]
