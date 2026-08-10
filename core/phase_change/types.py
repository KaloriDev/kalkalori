# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Core neutral types for phase-change handling (v0.6.1).

Three distinct concepts (see module docstrings in this package for where
each is produced):

1. **capability** (`PhaseChangeCapability`): does this medium/provider even
   contain a condensable component and expose enough structure (dry
   composition, water content, molar masses) to evaluate phase equilibrium?
   Capability says nothing about the current thermal operating point.
2. **possibility** ("possible" on `PhaseChangeResult`): given the current
   dry sensible-only baseline (wall temperatures vs. dew point), *could*
   phase change occur at this operating point?
3. **activity** ("active" on `PhaseChangeResult`): did the solver actually
   solve a phase-change case for this call (as opposed to falling back to
   sensible-only, e.g. because ``PhaseChangeMode.DISABLED`` was set, or
   because the exchanger is thermodynamically in the dry regime)?

Naming is deliberately neutral (not "outside water condensation ..."): the
same types remain stable as later patches add evaporation, multiple
condensable species, etc. (see ``docs/roadmap.md``). Only wet-gas H2O
``CONDENSATION`` (inside or outside) is solved in v0.6.1; ``EVAPORATION``
exists only to keep the enum's API shape stable for later patches and must
not be produced by a v0.6.1 code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

from core.common.warnings import ModelWarning
from core.properties.water import WaterPhaseRegion


class PhaseChangeMode(str, Enum):
    """User-facing control: how should the solver treat phase change.

    Assigned per side/input (``HXSideInput.phase_change_mode`` /
    ``BalanceSideSpec.phase_change_mode``), not on the exchanger as a
    whole -- there is deliberately no ``condensing_side`` selector anywhere
    in this package; the solver determines which side (if any) is
    condensing from capability + thermal state, see
    ``core.phase_change.integration``.

    AUTO:
        Default. The solver detects phase-change capability and, if the
        current thermal operating point makes phase change possible, solves
        it. A capable medium that never reaches its onset condition still
        produces a plain sensible-only result under AUTO -- capability does
        not imply activity.
    DISABLED:
        Force a sensible-only ("dry") solve on this side, even if phase
        change would otherwise be possible. The solver still evaluates and
        reports whether phase change *would* have been possible (see
        ``PhaseChangeResult.possible``) and attaches
        ``PHASE_CHANGE_DISABLED_BUT_POSSIBLE`` when it would.
    """

    AUTO = "auto"
    DISABLED = "disabled"


class PhaseChangeDirection(str, Enum):
    """Which direction of phase change (if any) is active for a side.

    Only ``NONE`` and ``CONDENSATION`` are solved in v0.6.1.
    ``EVAPORATION`` is a reserved value for a later patch (droplet/mist
    evaporation, re-evaporation of condensate) and must never be returned
    by v0.6.1 code.
    """

    NONE = "none"
    CONDENSATION = "condensation"
    EVAPORATION = "evaporation"


@dataclass(frozen=True)
class PhaseChangeCapability:
    """Uniform capability facts about a medium/provider, independent of the
    current thermal operating point.

    Produced by ``core.phase_change.capability.detect_phase_change_capability``.
    All fields other than ``capable``/``component``/``provider_kind`` are
    ``None`` when ``capable`` is False.

    Attributes:
        capable: True if this provider/spec contains a positive amount of a
            condensable component this model understands (H2O only, v0.6.0)
            and exposes enough structure (dry composition + molar masses) to
            evaluate phase equilibrium.
        component: Name of the condensable component ("H2O") or None.
        provider_kind: Short label identifying which capability path
            matched (e.g. "gas_mixture"), for diagnostics.
        dry_mole_fractions: Mole fractions of the non-condensable ("dry")
            part of the mixture, renormalized to sum to 1.0.
        M_dry: Mean molar mass of the dry part of the mixture [kg/mol].
        M_condensable: Molar mass of the condensable component [kg/mol].
        W_in: Inlet water content, kg condensable vapor / kg dry carrier gas
            [-]. This is a capability-time snapshot of the *input*
            composition; it is not updated as condensation proceeds (see
            ``PhaseChangeResult.W_out`` for that).
        backend: CoolProp backend string carried through from the spec
            (e.g. "HEOS"), used to rebuild gas-mixture specs at a different
            water content later in the solve.
        molar_masses: Optional molar-mass override mapping carried through
            from the spec (``GasMixtureSpec.molar_masses``).
    """

    capable: bool
    component: str | None = None
    provider_kind: str | None = None
    dry_mole_fractions: Mapping[str, float] | None = None
    M_dry: float | None = None
    M_condensable: float | None = None
    W_in: float | None = None
    backend: str = "HEOS"
    molar_masses: Mapping[str, float] | None = None


@dataclass(frozen=True)
class PhaseChangeResult:
    """Neutral, side-scoped phase-change result.

    One instance describes one side (``side="inside"`` or
    ``side="outside"``) of one exchanger call. For ``direction=NONE``
    (no active phase change), most of the mass/energy-split fields are
    still populated with their sensible-only-equivalent values
    (``W_out == W_in``, ``m_dot_condensate == 0``, ``Q_latent == 0``,
    ``Q_sensible == Q_total``) rather than left as ``None``, so callers do
    not need to branch on ``active`` before reading them; the exceptions are
    per-solve diagnostics that are meaningless without an active solve
    (``iterations``, ``residuals``) which default to ``0``/an empty mapping.

    For active wet-gas condensation, ``wall_temperature_mean`` is the
    global whole-surface mean used by the sensible resistance network,
    whereas ``wall_temperature_wet_mean`` represents only the active wet
    zone. ``W_sat_wet_surface`` is saturation at that wet-zone temperature;
    latent heat and the drained saturated-liquid condensate enthalpy use the
    same temperature.
    """

    side: str  # "inside" | "outside"
    mode: PhaseChangeMode
    direction: PhaseChangeDirection
    component: str | None

    capable: bool
    possible: bool
    active: bool
    near_onset: bool = False

    converged: bool = True
    iterations: int = 0
    method: str = "sensible_only"

    # Onset diagnostics (fix, v0.6.0 patch): the coldest estimated wall
    # point and the margin used to decide possible/active/near_onset (see
    # core.phase_change.regime.evaluate_condensation_onset). Populated from
    # the dry baseline even when the wet solver was never run (capable but
    # dry, near-onset, disabled), not left as None just because active is
    # False.
    onset_margin_K: float | None = None
    onset_wall_temperature: float | None = None
    onset_temperature_method: str | None = None

    # Water/vapor mass basis: kg / kg dry carrier gas [-], and kg/s.
    W_in: float | None = None
    W_out: float | None = None
    m_dot_dry_carrier: float | None = None
    m_dot_water_vapor_in: float | None = None
    m_dot_water_vapor_out: float | None = None
    m_dot_condensate: float = 0.0
    m_dot_gas_in: float | None = None
    m_dot_gas_out: float | None = None

    dew_point_in: float | None = None
    dew_point_out: float | None = None

    # Global mean for the whole side surface.  Active outside condensation
    # additionally reports the representative temperature of only the wet
    # part below.
    wall_temperature_mean: float | None = None
    wall_temperature_min: float | None = None
    wall_temperature_max: float | None = None

    wet_surface_fraction: float | None = None
    wet_surface_fraction_method: str | None = None
    wet_area: float | None = None
    outside_total_area: float | None = None

    alfa_dry: float | None = None
    alfa_effective: float | None = None
    lewis_number: float = 1.0

    Q_sensible: float = 0.0
    Q_latent: float = 0.0
    Q_total: float = 0.0

    mass_balance_error: float = 0.0
    energy_balance_error: float = 0.0

    residuals: Mapping[str, float] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    warnings: tuple[ModelWarning, ...] = ()

    # Appended diagnostics preserve the positional order of every field
    # that existed before the wet-zone-temperature extension.
    wall_temperature_wet_mean: float | None = None
    # Saturated water ratio evaluated at wall_temperature_wet_mean.  It is
    # deliberately distinct from saturation at the global wall mean.
    W_sat_wet_surface: float | None = None
    # v0.6.1 tube-side diagnostics. ``outside_total_area`` remains for
    # backwards compatibility with the v0.6.0 public result.
    inside_total_area: float | None = None
    W_mid: float | None = None

    @property
    def total_area(self) -> float | None:
        """Total side area used by full-area sensible convection."""
        return self.inside_total_area if self.side == "inside" else self.outside_total_area

    @property
    def is_condensing(self) -> bool:
        return self.active and self.direction is PhaseChangeDirection.CONDENSATION


@dataclass(frozen=True)
class WaterSteamPhaseChangeResult:
    """Pure water/steam phase-change result for one side (v0.6.2).

    Deliberately a *separate* typed result from ``PhaseChangeResult``
    rather than an extension of it: pure H2O condensation inside tubes is
    a direct vapor-liquid phase-equilibrium problem parameterized by
    vapor quality, not the dry-carrier humidity-ratio (W) / dew-point /
    Chilton-Colburn mass-transfer problem ``PhaseChangeResult`` was
    designed around (see ``core.phase_change.inside_pure_steam_zones``).
    Fields that are semantically shared between the two result types
    (``side``, ``mode``, ``direction``, ``active``, ``converged``,
    ``Q_sensible``, ``Q_latent``, ``Q_total``, ``warnings``) use the same
    names and meaning so callers that only care about those can treat
    either result uniformly; steam-specific fields have no W-based
    equivalent and are simply absent (not padded with unused ``None``
    wet-gas fields).

    ``HXSimulationResult.inside_phase_change`` and
    ``HXRatingResult.inside_phase_change`` accept either this type or
    ``PhaseChangeResult`` -- callers should check the concrete type (or
    duck-type on the shared fields above) before reading steam-specific
    fields.

    Attributes:
        side: Always ``"inside"`` in v0.6.2 (pure-steam condensation
            outside tubes is out of scope, see ``docs/roadmap.md``).
        mode: The ``PhaseChangeMode`` this side was evaluated under.
        direction: ``CONDENSATION`` if any condensation zone was active,
            else ``NONE`` (pure sensible desuperheat/subcool only).
        capable: Whether the provider is recognized as pure water/steam
            (``core.phase_change.capability.is_pure_water_provider``).
        possible: Whether cooling this stream at all is thermodynamically
            possible given the opposite-side bulk state (`False` only for
            the reverse-direction/evaporation-required case).
        active: Whether the multi-zone solver actually ran (`False` under
            ``PhaseChangeMode.DISABLED``, or when the inlet/outlet never
            leave the single sensible-only region it started in).
        converged: Whether the (bounded) internal root-find for a
            partially-traversed condensation zone converged; `True` when
            no such solve was needed.
        phase_in, phase_out: Inlet/outlet phase region
            (``core.properties.water.WaterPhaseRegion``).
        p: Nominal tube-side pressure [Pa] (constant; two-phase pressure
            drop is not modelled, see `two_phase_pressure_drop_supported`).
        T_in, T_out: Inlet/outlet temperature [K].
        T_sat: Saturation temperature at `p` [K].
        h_in, h_out: Inlet/outlet specific enthalpy [J/kg].
        quality_in, quality_out: Inlet/outlet vapor quality [-], `None`
            outside the saturation dome.
        Q_desuperheat, Q_condensation, Q_subcooling: Duty released by each
            zone [W] (`0.0` for a zone not constructed).
        Q_sensible: `Q_desuperheat + Q_subcooling` [W].
        Q_latent: `Q_condensation` [W].
        Q_total: `Q_sensible + Q_latent` [W], exactly
            `m_dot_total * (h_in - h_out)`.
        A_desuperheat, A_condensation, A_subcooling: Heat-transfer area
            allocated to each zone [m2].
        A_total: Total inside heat-transfer area available [m2].
        f_desuperheat, f_condensation, f_subcooling: Area fractions [-].
        zone_alpha_desuperheat, zone_alpha_condensation,
            zone_alpha_subcooling: Zone-representative inside heat
            transfer coefficient [W/(m2*K)], `None` for a zone not
            constructed.
        m_dot_total: Total tube-side mass flow [kg/s] (constant).
        two_phase_pressure_drop_supported: `False` whenever a
            condensation zone is active; the tube-side pressure drop
            result does not represent a full two-phase dp in that case
            (see ``core.phase_change.inside_pure_steam_condensation``).
        assumptions: Short machine-readable labels for the 0D modelling
            assumptions in effect (e.g. the isothermal-bulk-sink zone
            allocation, see ``core.phase_change.inside_pure_steam_zones``).
        warnings: Applicability/limitation warnings.
    """

    side: str  # "inside" (only value supported in v0.6.2)
    mode: PhaseChangeMode
    direction: PhaseChangeDirection
    capable: bool
    possible: bool
    active: bool
    converged: bool = True

    phase_in: Optional[WaterPhaseRegion] = None
    phase_out: Optional[WaterPhaseRegion] = None
    p: Optional[float] = None
    T_in: Optional[float] = None
    T_out: Optional[float] = None
    T_sat: Optional[float] = None
    h_in: Optional[float] = None
    h_out: Optional[float] = None
    quality_in: Optional[float] = None
    quality_out: Optional[float] = None

    Q_desuperheat: float = 0.0
    Q_condensation: float = 0.0
    Q_subcooling: float = 0.0
    Q_sensible: float = 0.0
    Q_latent: float = 0.0
    Q_total: float = 0.0

    A_desuperheat: float = 0.0
    A_condensation: float = 0.0
    A_subcooling: float = 0.0
    A_total: Optional[float] = None

    f_desuperheat: float = 0.0
    f_condensation: float = 0.0
    f_subcooling: float = 0.0

    zone_alpha_desuperheat: Optional[float] = None
    zone_alpha_condensation: Optional[float] = None
    zone_alpha_subcooling: Optional[float] = None

    m_dot_total: Optional[float] = None
    two_phase_pressure_drop_supported: bool = True

    assumptions: tuple[str, ...] = ()
    warnings: tuple[ModelWarning, ...] = ()

    @property
    def is_condensing(self) -> bool:
        return self.active and self.direction is PhaseChangeDirection.CONDENSATION
