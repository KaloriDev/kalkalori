# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
#
# -------------------------------------------------------------------------
# DEPRECATED compatibility shim (since v0.5.6)
# -------------------------------------------------------------------------
#
# The external Euler-provider adapter now lives in
# ``core.pressure_drop.outside_pressure_drop_external``. This module
# re-exports the same object so that previous import paths keep working;
# there is exactly one implementation, in ``core.pressure_drop``. Prefer
# importing from ``core.pressure_drop`` (or ``core.pressure_drop.
# outside_pressure_drop_external``) directly in new code.
#
# -------------------------------------------------------------------------

from __future__ import annotations

from core.pressure_drop.outside_pressure_drop_external import (
    ExternalCliEulerProvider,
)

__all__ = [
    "ExternalCliEulerProvider",
]
