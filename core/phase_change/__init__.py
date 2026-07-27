# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Phase-change support for KalKalori's 0D bare-tube heat-exchanger model
(v0.6.0: partial H2O condensation from a water-containing gas outside bare
tubes).

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
   the Chilton-Colburn analogy coupling condensation rate to the existing
   dry outside heat-transfer coefficient.
5. **Phase handling / solver** -- ``core.phase_change.
   outside_condensation_solver``, ``core.phase_change.regime``: the
   iterative coupled solve and the stable dry/condensing regime decision.
6. **Result and diagnostics** -- ``core.phase_change.types``
   (``PhaseChangeResult`` etc.), ``core.phase_change.wet_surface_fraction``.

``core.phase_change.integration`` wires layers 1-6 together for
``core.models.bare_tube.BareTubeHeatExchanger.simulate``;
``core.phase_change.rating_integration`` does the equivalent for ``.rate``.

Only H2O, only outside, only partial condensation, at most one active
phase-changing side per call -- see ``docs/roadmap.md`` for what later
patches (inside condensation, full steam condensation, evaporation, film
retention, freezing, multiple species, two-phase hydraulics) are expected
to add without breaking this package's public shape
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
