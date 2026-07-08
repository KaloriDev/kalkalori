# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Psychrometric helpers and adapters.

This package intentionally isolates external psychrometric libraries
(e.g. PsychroLib) from the core heat transfer logic.
"""
from core.psychrometrics.moist_air import (
    MoistAirState,
    moist_air_state_from_t_rh,
    moist_air_state_from_t_w,
    saturated_moist_air_state,
    moist_air_state_from_t_w_g_per_kg_da,
)

from core.psychrometrics.condensation import (
    CondensationOnsetResult,
    check_condensation_onset,
)

from core.psychrometrics.wet_process import (
    WetSurfaceProcessResult,
    condensable_water_per_kg_dry_air,
    enthalpy_drop_to_surface_saturation,
    humidity_ratio_drop_to_saturation,
    saturated_state_at_surface,
    wet_surface_process_limit,
)

from core.psychrometrics.psychrolib_adapter import (
    humidity_ratio_from_g_per_kg_dry_air,
    humidity_ratio_to_g_per_kg_dry_air,
    max_relative_humidity_at_t_p,
    saturation_humidity_ratio,
    saturation_humidity_ratio_is_defined,
    saturation_vapor_pressure,
)

__all__ = [
    "MoistAirState",
    "moist_air_state_from_t_rh",
    "moist_air_state_from_t_w",
    "saturated_moist_air_state",
    "CondensationOnsetResult",
    "check_condensation_onset",
    "WetSurfaceProcessResult",
    "condensable_water_per_kg_dry_air",
    "enthalpy_drop_to_surface_saturation",
    "humidity_ratio_drop_to_saturation",
    "saturated_state_at_surface",
    "wet_surface_process_limit",
    "humidity_ratio_from_g_per_kg_dry_air",
    "humidity_ratio_to_g_per_kg_dry_air",
    "max_relative_humidity_at_t_p",
    "saturation_humidity_ratio",
    "saturation_humidity_ratio_is_defined",
    "saturation_vapor_pressure",
]
