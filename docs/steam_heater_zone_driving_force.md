# Steam-Heater Zone Driving Force and Fin-Surface Diagnostics (v0.7.1)

This document describes two related v0.7.1 corrections to the pure-steam
tube-side model in `core.phase_change.steam_heater`, and the new fin-surface
temperature diagnostics that accompany them. Both are 0D engineering
refinements to the existing multi-zone steam-heater solver; neither changes
Briggs–Young, Robinson–Briggs, fin efficiency, contact topology, alpha
semantics, pressure drop, or the phase-change dispatch.

## 1. The corrected zone driving force

### 1.1 The previous approximation

The steam-heater solver partitions the tube-side enthalpy drop into up to
three ordered zones — `SUPERHEAT`, `CONDENSATION`, `SUBCOOLING` — each with
its own inside coefficient, `U`, and required area. Before this correction,
every zone sized against the *same* driving force:

```text
T_mean_outside = 0.5 * (T_in_outside + T_out_outside)
delta_T_zone   = 0.5 * (T_zone_in + T_zone_out) - T_mean_outside
A_zone         = Q_zone / (U_zone * delta_T_zone)
```

`T_mean_outside` is one arithmetic mean built from the *whole exchanger's*
inlet/outlet outside temperatures, shared unchanged across all three zones.
For a condensing zone at high pressure this collapses toward the difference
between the (isothermal) saturation temperature and the whole-exchanger mean
outside temperature — for a representative 17 bara steam/air case this
produces an EMTD around 71–72 K, well above what a zone-resolved
counterflow-consistent driving force gives for the same terminal states.

### 1.2 The corrected model

Each zone now sizes against its own thermodynamic driving force. An
equivalent outside-stream temperature *path* is marched through the zones,
in the same order the steam itself passes through them
(`SUPERHEAT -> CONDENSATION -> SUBCOOLING`), reusing the exact same outside
cp/property convention already used to compute the whole-exchanger outside
outlet temperature (`_outside_outlet_temperature`):

```text
T_outside_zone_0 = T_in_outside
for each active zone, in partition order:
    Q_zone           = m_dot_steam * (h_zone_in - h_zone_out)
    T_air_zone_out    = outside_temperature_after(T_air_zone_in, Q_zone, m_dot_outside)
    # this zone's driving force is resolved from (T_air_zone_in, T_air_zone_out)
    next T_air_zone_in = T_air_zone_out
```

The final zone's `T_air_zone_out` is required to close onto the
independently-computed whole-exchanger outside outlet temperature within a
tight numerical tolerance (the two are different fixed-point evaluations of
the same energy balance — one coarse pass over the whole duty, several finer
passes over each zone's sub-duty — so they need not be bit-identical for a
temperature-dependent outside cp, only mutually consistent).

This is explicitly an **equivalent 0D series allocation**, not a claim about
the physical axial position of phase fronts in a two-dimensional crossflow
bundle — see the `STEAM_HEATER_ZONE_ALLOCATION_0D_ESTIMATE` info warning
attached to every steam-heater result.

**Condensation zone.** The steam side is isothermal at `Tsat`. The zone uses
the exact terminal log-mean driving force:

```text
dT_in  = Tsat - T_air_zone_in
dT_out = Tsat - T_air_zone_out
EMTD_zone = (dT_in - dT_out) / ln(dT_in / dT_out)      # -> dT_in as dT_out -> dT_in
UA_zone   = Q_zone / EMTD_zone
```

This is the `Cr -> 0` epsilon-NTU limit for an isothermal side
(`eps = 1 - exp(-NTU)`); it is not obtained by calling the generic finite-`Cr`
inversion. Non-positive terminal differences (`dT_in <= 0` or `dT_out <= 0`)
are a genuine pinch violation for the assumed zone pairing and are never
silently averaged, clipped, or replaced by a fake finite area — the zone
reports an infeasible (non-finite) area, which the existing Rating/Simulation
entry points already turn into a `ValueError`.

**Sensible zones (`SUPERHEAT`, `SUBCOOLING`).** Both sides are sensible
within the zone. Effective zone capacity rates are reconstructed from the
zone's own duty and terminal temperatures (`C = Q_zone / |delta_T|`), and the
established `core.heat_transfer.ntu.ntu_from_effectiveness` inversion is
reused for the exchanger's declared `flow_arrangement`
(`counterflow`/`cocurrentflow`/`crossflow`) — no epsilon-NTU equation is
duplicated in `steam_heater.py`.

### 1.3 Whole-exchanger EMTD

The whole-exchanger effective mean temperature difference is a single
derived identity, never a separate calculation:

```text
UA_total  = sum(UA_zone)
EMTD_total = Q_total / UA_total     # SteamHeaterSolution.EMTD
```

Public `Simulation`/`Rating` results expose the same identity through
`result.EMTD == result.Q_required / result.UA_required` (or
`result.Q_total / result.UA_total` for a `Simulation`).

### 1.4 What did not change

Actual geometry (`A_actual`, `UA_actual` for the same geometry/properties),
Briggs–Young, Robinson–Briggs, fin efficiency, contact topology, wall
resistance, the Shah (2009) condensation correlation, the steam property
provider, and the v0.7.1 derived-steam-mass-flow Rating path
(`m_dot = Q_required / (h_in - h_out_target)`) are all untouched. Only the
phase-zone driving force / required UA / required area allocation changes,
so `A_required`, `UA_required`, `EMTD` and `overdesign` for a Rating are
expected to move.

## 2. Fin-surface temperature diagnostics

### 2.1 Existing wall-temperature semantics (unchanged)

`outside_wall_temperature_mean/min_estimate/max_estimate` (on
`WallTemperatureProbe`/`WallTemperatureEnvelope`, and the corresponding
public `Simulation`/`Rating` properties) represent the existing
outside/core-wall thermal-network node. For a `BareTube` this already *is*
the exposed outside skin. For a `CircularFinnedTube` it is **not**
necessarily the exposed primary surface, the fin base, the fin tip, or the
minimum metal skin temperature — it is the node the resistance network
resolves at the tube's core wall, upstream of the fin/root/contact topology.
These fields keep their exact historical values and meaning; nothing about
them is redefined.

### 2.2 New: primary surface, fin base, fin tip, outside skin

New diagnostics report the actually exposed extended-surface temperatures,
using the existing resistance-network topology
(`core.heat_transfer.outside_dispatch.calculate_resistance_network`) and the
existing annular-fin temperature ratio
(`FinEfficiencyResult.tip_temperature_ratio`, unchanged) — no new fin solver
and no change to fin efficiency physics.

**Welded fin, `"fin_branch_only"` topology (`D_root == D_o`).** The exposed
primary surface bypasses the fin/contact resistance entirely:

```text
T_primary_surface = T_core
q_fin      = G_fin * (T_core - T_bulk)
T_fin_base = T_core - q_fin * R_contact          # -> T_core as R_contact -> 0
```

**Continuous root layer, `"series_before_primary_and_fin_parallel_branches"`
topology (`D_root > D_o`).** The whole outside heat rate passes through the
common root/contact resistance before the primary/fin branches split, so
contact is never applied a second time to the fin branch:

```text
R_common       = R_root + R_contact
T_root_surface = T_core - q_total * R_common
T_primary_surface = T_root_surface
T_fin_base        = T_root_surface
```

**Fin tip (either topology),** from the existing annular-fin ratio:

```text
T_fin_tip = T_bulk + tip_temperature_ratio * (T_fin_base - T_bulk)
```

Both formulas are direction-agnostic: for a hot outside stream the ordering
reverses naturally (`T_bulk >= T_fin_tip >= T_fin_base >= T_core`) without
relying on `abs()`.

**Outside skin.** At each local probe, `outside_skin_temperature_min/max` is
the min/max of the physically exposed temperatures
(`T_primary_surface`, `T_fin_base`, `T_fin_tip`) for a finned tube, or simply
`T_core` for a bare tube — so `outside_skin_min/max == outside_min/max`
exactly for `BareTube`. The whole-exchanger envelope aggregates this across
the existing four-probe 0D endpoint envelope, same as
`outside_wall_temperature_min/max_estimate` — it is still an inlet/outlet
endpoint estimate, not a spatially segmented solution.

Public `Simulation`/`Rating` results add
`outside_skin_temperature_min/max_estimate` (always populated) and
`fin_base_temperature_min/max_estimate`,
`fin_tip_temperature_min/max_estimate` (`None` for a bare tube), alongside
the existing `outside_wall_temperature_*` properties.

### 2.3 Using this for engineering comparison

If comparing against skin-temperature output from an independent engineering
reference tool, compare that reference against
`outside_skin_temperature_min/max_estimate`, **not**
`outside_wall_temperature_min/max_estimate`. For outside film HTC, compare
against `outside_alpha_physical`, **not** `outside_alpha_effective_gross`.

The fin-tip temperature is strongly sensitive to the configured `fin_k` and
contact resistance, which this model never infers from a displayed tube
material — report the exact configured `fin_k`, contact resistance, and
root/tip thickness alongside any such comparison. This 0D endpoint envelope
remains an estimate, not a segmented spatial solution, so differences
against an independent reference are not automatically a defect; they may
reflect genuine modelling differences (discretization, fin geometry
assumptions, or configured material properties) rather than a bug. This
document does not claim external validation is complete.
