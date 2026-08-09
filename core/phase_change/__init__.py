# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Phase-change support for KalKalori's 0D bare-tube heat-exchanger model.

v0.6.1 supports partial H2O condensation from a wet gas containing a
non-condensable carrier on either the inside or outside surface.

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

Only H2O from wet gas, only partial condensation, at most one active
phase-changing side per call. Pure water/steam condensation inside tubes is
planned for v0.6.2; pure-steam condensation outside tubes is outside planned
scope. See ``docs/roadmap.md`` for later evaporation, film retention,
freezing, multiple-species and phase-change-hydraulic work that can extend
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
)

__all__ = [
    "PhaseChangeCapability",
    "PhaseChangeDirection",
    "PhaseChangeMode",
    "PhaseChangeResult",
]
