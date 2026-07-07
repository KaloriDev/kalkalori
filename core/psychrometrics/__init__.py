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
)

from core.psychrometrics.condensation import (
    CondensationOnsetResult,
    check_condensation_onset,
)

__all__ = [
    "MoistAirState",
    "moist_air_state_from_t_rh",
    "moist_air_state_from_t_w",
    "saturated_moist_air_state",
    "CondensationOnsetResult",
    "check_condensation_onset",
]
