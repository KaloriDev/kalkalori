# Steam-Heater Zone Driving Force and Fin-Surface Diagnostics (v0.7.1/v0.7.6)

This document records the v0.7.1 zone-driving-force and fin-surface
diagnostic work, together with the v0.7.6 correction to the outside-air
topology in `core.phase_change.steam_heater`. Both are global 0D engineering
refinements to the multi-zone steam-heater solver. Neither changes
Briggs–Young, Robinson–Briggs, fin efficiency, contact topology, alpha
semantics, pressure drop, the steam property model, or phase-change dispatch.

## 1. Steam-zone driving force and outside-air topology

### 1.1 Historical approximations

The steam-heater solver partitions the tube-side enthalpy drop into up to
three ordered zones — `SUPERHEAT`, `CONDENSATION`, `SUBCOOLING` — each with
its own inside coefficient, `U`, required area, and `UA`.

Before v0.7.1, every zone was sized against the same whole-exchanger mean
outside temperature:

```text
T_mean_outside = 0.5 * (T_in_outside + T_out_outside)
delta_T_zone   = 0.5 * (T_zone_in + T_zone_out) - T_mean_outside
A_zone         = Q_zone / (U_zone * delta_T_zone)
```

For a representative 17 bara steam/air case this gave the condensation zone
an EMTD around 71–72 K. v0.7.1 corrected that common-driving-force
approximation by resolving each zone's terminals, but it marched one
equivalent outside stream through the tube-side zones in series. That
series-marched air path was a documented 0D approximation, not a physical
claim about the crossflow bundle, and is superseded by v0.7.6.

### 1.2 Current v0.7.6 topology

The tube-side thermodynamic path remains sequential and its enthalpy-derived
zone duties remain authoritative:

```text
steam inlet
    -> SUPERHEAT
    -> CONDENSATION
    -> SUBCOOLING
    -> condensate outlet
```

The crossflow air does not follow that sequence. Each occupied tube-length or
height region receives a parallel share of the global outside flow, every
branch receives the same global outside inlet state, and the branch outlets
are mixed only after all active zones have been evaluated:

```text
                       -> SUPERHEAT branch    -> T_air,out,superheat
T_air,in (global) -----+-> CONDENSATION branch -> T_air,out,condensation -> MIX
                       -> SUBCOOLING branch   -> T_air,out,subcooling
```

For geometry that is uniform along the active tube length, the converged
gross outside-area fraction is also the represented tube-length and frontal
area fraction:

```text
f_z = A_required,z / sum(A_required,z)

m_dot_air,z   = f_z * m_dot_air,total
A_frontal,z   = f_z * A_frontal,total
L_z / L_total = f_z
```

Consequently, geometrical partitioning does not reduce the local face mass
flux or velocity:

```text
m_dot_air,z / A_frontal,z
    = m_dot_air,total / A_frontal,total
```

The existing outside correlation is therefore evaluated on the complete
bank and its physical film coefficient is retained for every branch. The
partitioned branch mass flow is used with the correspondingly partitioned
frontal region for the branch energy balance and capacity rate; it is never
passed through the full frontal area as though its velocity had fallen.

The area fractions are not heat-duty fractions. They are obtained from a
deterministic bounded fixed point because the required area depends on branch
air flow, while branch air flow depends on required area:

```text
choose finite positive trial fractions
    -> set m_dot_air,z = f_z * m_dot_air,total
    -> solve every zone from the common T_air,in
    -> set f_target,z = A_required,z / sum(A_required,z)
    -> update the trial fractions and repeat to tolerance
```

The implementation uses explicit iteration and residual limits, bounded
feasibility handling near a temperature pinch, and a maximum iteration count.
It raises a controlled non-convergence error rather than accepting the last
iterate. A single active zone is the exact `f_z = 1` limiting case. Duty
fractions may provide an initial guess, but are never accepted as the final
geometric allocation unless the required-area fixed point independently
converges to the same values.

Each branch outlet is evaluated with the existing outside-property convention.
The mixed outlet is recovered from the total outside sensible-energy balance
using that same convention, and the branch sum is checked against it:

```text
sum(m_dot_air,z * delta_h_air,z)
    ~= m_dot_air,total * delta_h_air,mixed
    = Q_total
```

For a constant-`cp` outside fluid this reduces to the ordinary mass-weighted
outlet temperature. No second fluid-energy model or separate property backend
is introduced for mixing.

### 1.3 Per-zone driving forces

**Condensation zone.** The steam side is isothermal at `Tsat`. The branch
uses the exact terminal log-mean driving force:

```text
dT_in  = Tsat - T_air,in
dT_out = Tsat - T_air,out,z
EMTD_z = (dT_in - dT_out) / ln(dT_in / dT_out)
UA_z   = Q_z / EMTD_z
```

This is the `Cr -> 0` epsilon-NTU limit for an isothermal side
(`eps = 1 - exp(-NTU)`); it is not obtained by calling the generic finite-`Cr`
inversion. Non-positive terminal differences (`dT_in <= 0` or `dT_out <= 0`)
are genuine branch pinch violations and are never silently averaged, clipped,
or replaced by a fake finite area.

**Sensible zones (`SUPERHEAT`, `SUBCOOLING`).** Both sides are sensible
within a branch. Effective capacity rates are reconstructed from that zone's
duty and terminal temperatures (`C = Q_z / |delta_T_z|`), including the
branch air capacity rate. The established
`core.heat_transfer.ntu.ntu_from_effectiveness` inversion is reused for the
exchanger's declared `flow_arrangement`
(`counterflow`/`cocurrentflow`/`crossflow`); no epsilon-NTU equation is
duplicated in `steam_heater.py`.

### 1.4 Whole-exchanger EMTD

The whole-exchanger effective mean temperature difference remains a derived
identity, never a separate calculation:

```text
UA_total   = sum(UA_zone)
EMTD_total = Q_total / UA_total
```

Public `Simulation`/`Rating` results expose the same identity through
`result.EMTD == result.Q_required / result.UA_required` (or
`result.Q_total / result.UA_total` for a `Simulation`). Per-zone diagnostics
also expose the converged area/air fractions, branch outside mass flow,
frontal area, face mass flux or velocity, outside inlet/outlet temperatures,
and local heat-transfer quantities. Global diagnostics identify
`parallel_by_geometry` and report allocation convergence and mixed-air energy
closure.

### 1.5 What did not change and model limits

The actual exchanger geometry, Briggs–Young, Robinson–Briggs, fin efficiency,
contact topology, wall resistance, Shah (2009) condensation correlation,
single-phase inside correlations, steam properties, tube-side zone duties,
pressure-drop correlations, and the v0.7.1 derived-steam-mass-flow Rating path
(`m_dot = Q_required / (h_in - h_out_target)`) are unchanged. Required zone
areas, branch outlet temperatures, `UA_required`, `EMTD`, and Rating overdesign
may change for multi-zone cases because the outside coupling is intentionally
different.

This remains a **global multi-zone 0D model**. The converged area/tube-length
fractions represent the occupied portions needed by the model; they are not a
row-by-row or longitudinal temperature field and do not claim local phase-front
accuracy. No falling-film, liquid-level, drainage-hydraulic, flooded-pipe, or
general 1D model is introduced. The
`STEAM_HEATER_ZONE_ALLOCATION_0D_ESTIMATE` information warning reports this
scope on steam-heater results.

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
