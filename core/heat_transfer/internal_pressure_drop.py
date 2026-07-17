# KalKalori — Heat Exchanger Open Engine
# Copyright (C) 2025  KalKalori Project Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
DEPRECATED compatibility shim (since v0.5.6).

Tube-side pressure-drop code now lives in ``core.pressure_drop.
internal_pressure_drop``. This module re-exports the same objects so that
previous import paths keep working; there is exactly one implementation, in
``core.pressure_drop``. Prefer importing from ``core.pressure_drop`` (or
``core.pressure_drop.internal_pressure_drop``) directly in new code.
"""

from __future__ import annotations

from core.pressure_drop.internal_pressure_drop import (
    FluidProps,
    HydraulicPosition,
    TubeSideHydraulicPoint,
    TubeBundleHydraulicResult,
    calculate_tube_bundle_hydraulics,
    tube_bundle_hydraulics,
    mean_velocity,
    reynolds_number,
    friction_factor_smooth,
    dynamic_pressure,
    pressure_drop_tubes,
    pressure_drop_inlet,
    pressure_drop_outlet,
    pressure_drop_turns,
    pressure_drop_internal_total,
)

__all__ = [
    "FluidProps",
    "HydraulicPosition",
    "TubeSideHydraulicPoint",
    "TubeBundleHydraulicResult",
    "calculate_tube_bundle_hydraulics",
    "tube_bundle_hydraulics",
    "mean_velocity",
    "reynolds_number",
    "friction_factor_smooth",
    "dynamic_pressure",
    "pressure_drop_tubes",
    "pressure_drop_inlet",
    "pressure_drop_outlet",
    "pressure_drop_turns",
    "pressure_drop_internal_total",
]
