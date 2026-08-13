# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Phase-change support for KalKalori's 0D bare-tube heat-exchanger model.

The package supports partial H2O condensation from a wet gas containing a
non-condensable carrier on either exchanger side, plus pure-water/steam
cooling, condensation, heating and evaporation inside tubes. Pure-water
phase change outside tubes remains unsupported.

Layering (see individual module docstrings for detail)
--------------------------------------------------------
1. **Capability** -- ``core.phase_change.capability``: does the medium
   contain a condensable component and expose enough structure to evaluate
   phase equilibrium?
2. **Equilibrium** -- ``core.phase_change.water_equilibrium``,
   ``core.phase_change.wet_gas_composition``: dew point, saturated water
   content, dry/wet composition bookkeeping.
3. **Mass and enthalpy balance** -- ``core.phase_change.wet_gas_enthalpy``:
   the per-kg-dry-gas enthalpy function and its bisection inverse.
4. **Heat and mass transport** -- ``core.phase_change.mass_heat_transfer``:
   the Chilton-Colburn analogy coupling condensation rate to the applicable
   existing dry side heat-transfer coefficient.
5. **Phase handling / solver** -- the inside/outside condensation solvers
   and ``core.phase_change.regime``: coupled wet solves and the stable
   dry/condensing regime decision.
6. **Result and diagnostics** -- ``core.phase_change.types``
   (``PhaseChangeResult`` etc.), ``core.phase_change.wet_surface_fraction``.

``core.phase_change.integration`` wires layers 1-6 together for
``core.models.bare_tube.BareTubeHeatExchanger.simulate``;
``core.phase_change.rating_integration`` does the equivalent for ``.rate``.

Wet-gas H2O remains limited to partial condensation, while pure water/steam
inside tubes can cool through condensation or heat through partial/complete
evaporation and optional superheating. At most one phase-changing side is
active per call. See ``docs/roadmap.md`` for carried-liquid evaporation,
film retention, freezing, multiple-species and phase-change-hydraulic work that can extend
this package without breaking its public shape
(``PhaseChangeMode``/``PhaseChangeDirection``/``PhaseChangeCapability``/
``PhaseChangeResult`` are deliberately named for the general concept, not
for "outside water condensation" specifically).
"""

from core.phase_change.types import (
    PhaseChangeCapability,
    PhaseChangeDirection,
    PhaseChangeMode,
    PhaseChangeResult,
    WaterSteamPhaseChangeResult,
)
from core.phase_change.capability import (
    PureWaterPhaseChangeProviderNotSupportedError,
)
from core.phase_change.finned_tube_guard import (
    CircularFinnedTubeWetSurfaceNotSupportedError,
    reject_circular_finned_tube_wet_surface,
)
from core.phase_change.steam_condensation import (
    SteamCondensationLocalResult,
    SteamCondensationZoneResult,
    SteamTubeOrientation,
    local_steam_condensation_alpha,
    solve_steam_condensation_zone,
    steam_mass_flux,
)
from core.phase_change.steam_heater import (
    SteamEvaporationNotSupportedError,
    SteamHeaterSolution,
    SteamHeaterZoneKind,
    SteamHeaterZoneResult,
    rate_steam_heater,
    solve_steam_heater,
)
from core.phase_change.water_evaporation import (
    WaterEvaporationZoneResult,
    local_water_evaporation_alpha,
    solve_water_evaporation_zone,
    water_mass_flux,
)
from core.phase_change.water_evaporator import (
    WaterCondensationRequiredError,
    WaterEvaporatorSolution,
    WaterEvaporatorZoneKind,
    WaterEvaporatorZoneResult,
    rate_water_evaporator,
    solve_water_evaporator,
)

__all__ = [
    "PhaseChangeCapability",
    "PhaseChangeDirection",
    "PhaseChangeMode",
    "PhaseChangeResult",
    "WaterSteamPhaseChangeResult",
    "PureWaterPhaseChangeProviderNotSupportedError",
    "CircularFinnedTubeWetSurfaceNotSupportedError",
    "reject_circular_finned_tube_wet_surface",
    "SteamCondensationLocalResult",
    "SteamCondensationZoneResult",
    "SteamTubeOrientation",
    "local_steam_condensation_alpha",
    "solve_steam_condensation_zone",
    "steam_mass_flux",
    "SteamEvaporationNotSupportedError",
    "SteamHeaterSolution",
    "SteamHeaterZoneKind",
    "SteamHeaterZoneResult",
    "rate_steam_heater",
    "solve_steam_heater",
    "WaterEvaporationZoneResult",
    "local_water_evaporation_alpha",
    "solve_water_evaporation_zone",
    "water_mass_flux",
    "WaterCondensationRequiredError",
    "WaterEvaporatorSolution",
    "WaterEvaporatorZoneKind",
    "WaterEvaporatorZoneResult",
    "rate_water_evaporator",
    "solve_water_evaporator",
]
