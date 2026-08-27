# Circular-Finned-Tube Model (v0.7.0; applicability extended through v0.7.5)

This document describes the v0.7.0 release-candidate dry
circular-finned-tube model and its v0.7.5 wet-surface extension, pending
independent reference validation.
It is a 0D engineering model with explicitly declared geometry and
correlation boundaries. v0.7.2 adds a practical dimensionless
`fin_contact_efficiency` alternative to the explicit `fin_contact_resistance`
input; see [Fin/root contact input](#finroot-contact-input) below. v0.7.4
makes positive one- to three-row Briggs--Young requests calculable with an
explicit unvalidated-extrapolation warning. Neither change introduces a new
correlation or changes the fin-efficiency, root-conduction, Briggs--Young or
Robinson--Briggs equations described here. v0.7.5 adds nonlinear H2O
condensation from a wet gas on the exposed primary/root surface and annular
fins without changing the legacy dry route.

## Scope

The model covers round core tubes with individual, full circular fins and a
dry single-phase gas, or a gas-phase mixture with H2O as its single
condensable species, in crossflow. A continuous helical fin is represented as
an equivalent periodic train of complete annular fins. The 0D model uses
the fractional count `length_effective / fin_pitch`; it does not round the
number of fins and therefore does not introduce area jumps when tube length
changes.

The implemented empirical correlations cover staggered equilateral triangular
banks.
`TubeBundle(layout="inline")` remains valid public geometry, but the built-in
Briggs--Young and Robinson--Briggs providers reject it as unsupported.

Segmented, serrated, wavy and plate fins; elliptical or flattened tubes;
frost/ice; acid or multiple-species condensation; condensate retention,
flooding, bridging, re-evaporation, carry-over or re-entrainment; drainage
geometry; wet pressure-drop correction; radiation; and detailed fin-to-tube
joint mechanics are outside this model. No smooth-tube correlation is used as
a fallback.

## Public geometry

`CircularFinnedTube` is defined in `core.geometry.finned_tube` and remains
available through the public `core.geometry` package. It composes an existing
`BareTube` as `core_tube`. The
inside diameter, outside core diameter, total/effective lengths, wall
conductivity, roughness and orientation are delegated to that object. This
keeps one source of truth for core-tube validation and tube-side hydraulics.

The additional SI inputs are:

- `fin_k` [W/(m K)];
- `fin_contact_resistance` [m2 K/W] or `None` -- advanced, physically
  explicit areal contact resistance;
- `fin_contact_efficiency`, dimensionless, `0 < value <= 1`, or `None` --
  practical contact-conductance-degradation input added in v0.7.2; see
  [Fin/root contact input](#finroot-contact-input);
- `D_fin`, `D_root` [m];
- `fin_thickness_root`, optional `fin_thickness_tip` [m];
- `fin_pitch` [m], centre-to-centre in the axial direction;
- optional authoritative `external_area_per_length` [m2/m].

`fin_thickness_tip=None` means the root thickness is used at the tip.
The constraints are

```text
D_i < D_o <= D_root < D_fin
fin_pitch > fin_thickness_root > 0
0 < fin_thickness_tip <= fin_thickness_root
```

`D_root=D_o` represents a welded fin without an additional root sleeve;
`D_root>D_o` represents a radial root layer, for example an integral
extruded construction. No material name or manufacturing method is inferred.

## Periodic areas and volume

Let

```text
r_root = D_root / 2
r_tip  = D_fin / 2
H      = r_tip - r_root
t_r    = fin_thickness_root
t_t    = resolved fin_thickness_tip
p_f    = fin_pitch
S      = sqrt(1 + ((t_t - t_r)/(2 H))**2)
```

For one fin:

```text
A_fin_sides = 2 pi (r_tip**2 - r_root**2) S
A_fin_tip   = pi D_fin t_t
A_fin       = A_fin_sides + A_fin_tip
```

The inclined-face factor `S` retains the actual linearly tapered profile.
The root edge is not exposed. Per unit axial length:

```text
A_primary'  = pi D_root (p_f - t_r) / p_f
A_fin'      = A_fin / p_f
A_gross,geometric' = A_primary' + A_fin'
```

The exact volume of a linearly tapered annular fin is

```text
V_fin = 2 pi [r_root H (t_r + t_t)/2
              + H**2 (t_r + 2 t_t)/6]
```

If `external_area_per_length` is supplied, it is the authoritative gross
area used by the thermal solver. The geometric gross area remains available,
and a structured warning is produced when the absolute relative difference
exceeds 5%. The override must be positive and no smaller than `A_primary'`.
It never changes blockage, reference velocity, Reynolds number or correlation
applicability.

## Free-flow area, velocity and Reynolds number

The periodic projected blockage per unit tube length is

```text
D_block = D_root
          + (D_fin - D_root) (t_r + t_t)/(2 p_f)
```

Thus, for transverse pitch `P_t`, longitudinal row spacing `P_l` and
diagonal centre distance `P_d = sqrt(P_l**2 + (P_t/2)**2)`:

```text
g_t = P_t - D_block
g_d = 2 (P_d - D_block)
g_min = g_t                 # inline
g_min = min(g_t, g_d)       # staggered

A_face = n_tubes_per_row P_t length_effective
A_min  = n_tubes_per_row g_min length_effective
V_face = m_dot/(rho A_face)
V_max  = V_ref = m_dot/(rho A_min)
G_max  = m_dot/A_min
Re_Droot = G_max D_root/mu
```

Physical fin collision is checked separately with `D_fin`, rather than the
period-averaged `D_block`. This is not the smooth-tube gap formula with
`D_fin` substituted for `D_o`.

## Fin efficiency and resistance network

The linearly tapered fin is solved as one-dimensional radial conduction:

```text
d/dr [k_fin A_c(r) d(theta)/dr] - h_o P_s(r) theta = 0
A_c(r) = 2 pi r t(r)
```

The root temperature is prescribed, both sloping faces convect, and the tip
has a convective boundary condition. A deterministic conservative
finite-volume/tridiagonal solve is used without a production dependency on
NumPy or SciPy. The constant-thickness case is tested against an independent
classical modified-Bessel solution.

The physical film coefficient, fin efficiency and area remain separate:

```text
A_out,effective = A_primary + eta_fin A_fin,used
eta_overall = A_out,effective / A_out,gross
```

For all tubes in the bundle the common inside/core terms are

```text
R_i    = 1/(h_i A_i)
R_wall = ln(D_o/D_i)/(2 pi wall_k L N)
```

The outside topology then depends on whether the root layer is continuous.
For a welded fin (`D_root=D_o`) the exposed primary cylinder bypasses the
fin contact. The physical contact basis is the periodic root footprint,
`A_contact=pi D_root L N (t_root/p_f)`, and

```text
G_primary = h_o A_primary
G_fin     = 1/[R_contact''/A_contact + 1/(h_o eta_fin A_fin)]
R_outside,branches = 1/(G_primary + G_fin)
R_outside = R_outside,branches
R_total   = R_i + R_wall + R_outside
```

For an extruded geometry (`D_root>D_o`) the root layer is continuous. Its
core interface has basis `A_contact=pi D_o L N`; root conduction and contact
are therefore common series terms before the parallel outside convection:

```text
R_root    = ln(D_root/D_o)/(2 pi fin_k L N)
R_contact = R_contact''/A_contact
R_outside,branches = 1/[h_o (A_primary + eta_fin A_fin)]
R_outside = R_root + R_contact + R_outside,branches
R_total   = R_i + R_wall + R_outside
UA        = 1/R_total
U_out,gross = UA/A_out,gross
```

This is an explicit 0D equivalent network, not a detailed joint-mechanics
model. `R_contact''` above is the *resolved* areal contact resistance,
produced by the precedence rule in
[Fin/root contact input](#finroot-contact-input): an explicit
`fin_contact_resistance` (even `0.0`, meaning known ideal contact) is used
directly; otherwise a supplied `fin_contact_efficiency` is converted to an
equivalent `R_contact''` at the current operating point; otherwise (both
omitted) `R_contact''=0` with no diagnostic warning -- ideal contact is the
documented default, not an unspecified/uncertain state.

Fin efficiency and contact resistance are each applied exactly once.

## Fin/root contact input

v0.7.2 adds a practical, dimensionless alternative to the existing
physically explicit `fin_contact_resistance`. Both inputs describe the same
additional fin/root contact loss; they are mutually exclusive in physics,
but the constructor accepts both fields and resolves one authoritative
value by precedence.

### Two input styles

- **`fin_contact_resistance`** [m2 K/W] -- the advanced, physically explicit
  areal interface resistance from [Periodic areas and volume](#periodic-areas-and-volume)
  onward. It is a material/interface property supplied directly by the
  caller and is used exactly as before.
- **`fin_contact_efficiency`** -- dimensionless, `0 < value <= 1`, added in
  v0.7.2 for callers who cannot estimate an areal resistance. `1.0` is ideal
  contact; `0.95`/`0.80` mean 95%/80% contact efficiency. Percent notation
  (`95`) is rejected, not silently divided by 100. It is defined as

  ```text
  eta_contact = G_actual / G_ideal
  ```

  the conductance of the contact-affected path *with* contact loss, divided
  by the conductance of the same path with ideal contact, at the current
  operating point. `eta_contact` is **not** `eta_fin` and **not**
  `eta_overall`; it is a separate, additional thermal loss applied on top of
  fin efficiency, never multiplied into it.

### Precedence

```text
explicit fin_contact_resistance (even 0.0)
    > fin_contact_efficiency
    > ideal-contact default (R_contact'' = 0)
```

`None` on either field is an input-selection sentinel, not a declaration of
uncertainty. Leaving both `None` selects the documented ideal-contact
default without a diagnostic warning -- the pre-v0.7.2 behaviour of warning
whenever `fin_contact_resistance` was unset has been removed, because ideal
contact is now the normal, documented default rather than an unspecified
state. Supplying both fields together does not raise a warning either: the
resolution is deterministic (explicit resistance wins), and one `info`-level
diagnostic (`circular_finned_tube_contact_efficiency_ignored`) records that
`fin_contact_efficiency` was ignored.

### Converting efficiency to an equivalent resistance

Unlike `fin_contact_resistance`, `fin_contact_efficiency` is **not**
converted into one fixed `R_contact''` at geometry-construction time. The
equivalent resistance it implies depends on the ideal (contact-free)
conductance of the contact-affected path at the current operating point --
`h_o`, `eta_fin`, the relevant areas and, for a continuous root layer, root
conduction -- so it is resolved only inside the resistance-network
calculation, after those quantities are known:

- welded fin (`D_root=D_o`, only the fin branch is contact-affected):

  ```text
  G_fin,ideal      = h_o eta_fin A_fin
  R_contact,equiv  = (1/eta_contact - 1) / G_fin,ideal
  ```

- continuous root layer (`D_root>D_o`, the whole downstream path is
  contact-affected):

  ```text
  R_branches,ideal = 1/[h_o (A_primary + eta_fin A_fin)]
  R_path,ideal     = R_root + R_branches,ideal
  R_contact,equiv  = (1/eta_contact - 1) R_path,ideal
  ```

`eta_contact=1` gives `R_contact,equiv=0` exactly in both cases. The
resulting `R_contact,equiv` is then folded into `R_contact''` above exactly
like an explicit resistance would be -- welded fins penalise only the fin
branch, a continuous root layer treats it as a common series term. Because
`R_contact,equiv` depends on the operating point, its areal equivalent
(`contact_resistance_equivalent_areal` in the diagnostics below) must not be
read as a fixed material constant the way an explicit `fin_contact_resistance`
is; it is a resolved value for the current `h_o`/`eta_fin`/geometry, not an
interface property that transfers unchanged to a different operating point
or a different tube.

### Diagnostics

`ThermalResistanceNetwork` and `FinnedTubeDiagnostics` expose:

- `contact_input_mode`: `"ideal_default"`, `"explicit_resistance"` or
  `"contact_efficiency"`.
- `fin_contact_efficiency_input`: the caller-declared efficiency, or `None`.
- `fin_contact_efficiency_effective`: the actual conductance ratio implied
  by the resolved network (`1.0` for ideal contact; reproduces
  `fin_contact_efficiency_input` to numerical tolerance in efficiency mode;
  computed from the actual resistance in explicit-resistance mode).
- `contact_resistance_equivalent_areal`: the resolved areal equivalent
  [m2 K/W] -- the exact supplied value in explicit-resistance mode, the
  operating-point equivalent above in efficiency mode, `0.0` for the ideal
  default.
- `resistance_contact`: unchanged in meaning -- the actual absolute [K/W]
  contact resistance used by the network.
- `contact_resistance_used`: now reports the network's resolved areal
  equivalent (identical to `contact_resistance_equivalent_areal`) rather
  than only mirroring an unset explicit resistance, so it is never
  misleading in efficiency mode; explicit-resistance cases keep the same
  numeric value as before v0.7.2.

### Practical guidance for fin/tube contact

KalKalori does **not** choose a contact resistance or efficiency from
material name, fin type, manufacturing method or temperature; the ideal
default remains `fin_contact_efficiency=1.0`. The table below is published
engineering guidance for a caller deciding what to enter -- not a predictive
model and not applied automatically.

| Source | Reported quantity | Approx. range | Context |
|---|---|---|---|
| Caruso, Giannetti & Naviglio (2014), *Heat Transfer Engineering* 36(2), 212--221, DOI: 10.1080/01457632.2014.909224 | Thermal contact conductance `h_c`; reciprocal `R''=1/h_c` | `h_c` approx. 3500--11000 W/(m2 K), i.e. `R''` approx. 9.1e-5--2.9e-4 m2 K/W (larger `h_c` gives smaller `R''`) | Mechanically bound annular finned tubes; the same study reports contact resistance contributing roughly 30--50% of total air-side resistance *for their tubes* -- a study-specific observation, not a generic design recommendation |
| Jeong, Kim, Youn & Kim (2004), *Int. J. Heat and Fluid Flow* 25(6), 1006--1014, DOI: 10.1016/j.ijheatfluidflow.2004.03.005 | Correlation between thermal contact conductance and joint factors | Not reduced to one scalar here | Tube expansion ratio, fin type, fin spacing and fin coating are all reported to materially affect contact conductance |
| Jeong, Kim & Youn (2005), *Int. J. Heat and Mass Transfer* 49(7--8), 1547--1555, DOI: 10.1016/j.ijheatmasstransfer.2005.10.042 | Thermal contact conductance, 7 mm tube | Not reduced to one scalar here | Further evidence that fin type and tube manufacturing method shift contact conductance; supports treating contact resistance as geometry/process-dependent rather than a universal constant |
| Cheng & Madhusudana (2006), *Applied Thermal Engineering* 26(17--18), 2119--2131, DOI: 10.1016/j.applthermaleng.2006.04.016 | Effect of electroplating on fin-tube interface conductance | Qualitative only | Interface treatment/plating and interstitial medium affect contact conductance |

These are reported scalar ranges and qualitative conclusions from the cited
studies, not a universal recommended default and not proprietary/vendor
data. Use project-specific engineering judgement, or the explicit
`fin_contact_resistance` input with a project-derived value, when ideal
contact is not an adequate assumption.

## Physical film HTC vs effective gross-area HTC vs overall U

All three quantities use units of `W/(m2 K)`, but they have deliberately
different meanings and must not be compared interchangeably.

- `outside_alpha_physical` is the physical mean film HTC returned directly by
  Briggs--Young. It has no fin-efficiency, area-enhancement, contact, or
  root-conduction adjustment. Use it to validate the heat-transfer
  correlation against published literature or an independent physical
  film-HTC reference.
- `outside_alpha_effective_gross` is the equivalent outside-side coefficient
  referenced to the authoritative gross outside area `A_outside_gross`. It
  is calculated from the complete topology-aware outside path, including a
  finite fin contact and, where present, the continuous root layer:

  ```text
  R_outside = 1/(outside_alpha_effective_gross A_outside_gross)
  ```

  Use it for generic outside-resistance reporting. With ideal contact on a
  simple extended surface it reduces to

  ```text
  outside_alpha_effective_gross = outside_alpha_physical
      (A_primary + eta_fin A_fin) / A_outside_gross
  ```

  but that shortcut is not valid for finite contact resistance.
- `U_gross_outside` is the overall exchanger coefficient on the same gross
  area, including inside convection and core-wall conduction as well as the
  whole outside path. The whole-exchanger source of truth is `UA`, with
  `U_gross_outside = UA/A_outside_gross`.

During active condensation, `outside_alpha_physical` keeps exactly the same
sensible-film meaning. Latent heat is not folded into that field. The wet
surface result additionally reports
`outside_alpha_wet_effective_gross_core_basis`, defined diagnostically as

```text
outside_alpha_wet_effective_gross_core_basis
    = Q_total / [A_outside_gross (T_gas,bulk - T_core_wall)]
```

This is a named reconstruction on the authoritative gross outside area and
the bulk-gas-to-core-wall driving-force basis, not a new physical film
correlation. It is the standard `Q = h A deltaT` coefficient definition used
as a reporting identity (Incropera et al., *Fundamentals of Heat and Mass
Transfer*), not an empirical wet-surface law. Sensible, latent and total
duties remain separate. The existing dry `outside_alpha_effective_gross`
field is not redefined.

For compatibility, `OutsideThermalDispatchResult.alpha` and
`FinnedTubeDiagnostics.outside_htc` remain aliases for the **physical** film
HTC. Generic top-level `alfa_o` / `outside_alfa_mean` fields instead report
the **effective gross-area** value (the values coincide for a plain tube).

## Wet annular-fin condensation (v0.7.5)

`PhaseChangeMode.AUTO` first evaluates the dry surface-temperature envelope.
If no condensation is possible, execution stays on the unchanged v0.7.4 dry
`CircularFinnedTube` route. If the outside provider declares a gas mixture
with H2O condensation capability, the existing whole-exchanger wet-gas solver
calls a radial wet-surface component. The outer 0D iteration remains
authoritative for gas outlet temperature, outlet humidity ratio, dry-carrier
and water-vapor flows, condensate availability and the overall enthalpy
balance. `PhaseChangeMode.DISABLED` still returns the dry result and warns
when phase change would have been possible.

### Local heat and mass transfer

The wet solver retains the dry annular-fin finite-volume geometry: linear
radial conduction through a constant or linearly tapered fin, both exposed
faces, and the physical tip boundary. Each surface control volume at
temperature `T_s` uses

```text
h_m = (h_o / cp_gas) Le**(-2/3)
m_cond'' = h_m max[W_bulk - W_sat(T_s, p), 0]
q_sensible'' = h_o (T_gas,bulk - T_s)
q_latent'' = m_cond'' h_fg(T_s)
q_total'' = q_sensible'' + q_latent''
```

The `h_m` expression and dry-carrier humidity-ratio basis are the existing
Chilton--Colburn analogy documented under [Outside Water
Condensation](references.md#outside-water-condensation-v060). The coupled
wet-annular-fin source form follows Sharqawy, Moinuddin & Zubair (2012) and
Sharqawy & Zubair (2007); the partial-wet radial treatment is also consistent
with Rosario & Rahman (1999). `W_sat` and `h_fg` come from the existing IAPWS
water-property path. Full bibliographic details and DOI links are in
[Wet annular fins (v0.7.5)](references.md#wet-annular-fins-v075).

A cell for which `W_bulk <= W_sat` has exactly zero condensate and latent
source and therefore behaves as a dry-fin cell. The wet/dry boundary is never
an input. The nonlinear temperature and saturation field determines one of
the typed states `DRY`, `PARTIALLY_WET` or `FULLY_WET`.

For a partially wet fin, `wet_dry_boundary_radius` is the single radial
dew-point crossing obtained by linearly interpolating the humidity-ratio
driving force between adjacent radial temperature nodes. The wet face area is
integrated from the root radius to that crossing,

```text
A_wet,faces = 2 pi S (r_boundary**2 - r_root**2)
```

where `S` is the tapered-face slope factor defined under [Periodic areas and
volume](#periodic-areas-and-volume). The tip area is wet only when the tip
itself is below its local dew point. The radius is `None` for dry and fully
wet fins. It is also `None` in the special aggregate-partial case described
below when the representative cold-zone fin is wet to its tip but only part
of the 0D endpoint envelope is wet: no radial crossing exists to report.
This radial boundary and area convention follows the partial-wet radial-fin
formulation of Sharqawy & Zubair (2007) and Rosario & Rahman (1999).

The dry onset audit spans inlet/outlet endpoint probes, whereas the nonlinear
radial response normally uses one bulk-mean state. If that mean radial field
is completely dry after a cold endpoint has already activated `AUTO`, the
solver reuses the established linear 0D wall-envelope estimate as a bounded
cold-zone fallback. Its area fraction multiplies mass/latent sources, and its
representative wet-zone temperature is applied to saturation, latent heat and
drained-liquid enthalpy; sensible convection remains on the full area at the
solved bulk-mean metal temperature. This onset-consistency fallback is exposed
through `condensation_area_fraction`,
`condensation_temperature_offset_K`, assumptions and the wet-area warning. It
does not march temperatures axially and does not use rows, circuits or
`n_passes_transverse` as thermal segments.

### AUTO regime and non-active results

`PhaseChangeMode.AUTO` legitimately resolves to any of `DRY`, `NEAR_ONSET`,
`PARTIALLY_WET` or `FULLY_WET`; `outside_phase_change.active == False` is a
valid converged result (the dry or near-onset regime), never a calculation
failure. The near-onset activation band exists so a borderline operating
point does not oscillate between the dry and wet solve on repeated calls
close to the dew point; it is deliberately held on the dry route rather than
shrunk to force activation. A regime-independent result must not be read by
first checking `active`: whenever phase change was evaluated,
`Q_sensible`/`Q_total` on the returned `PhaseChangeResult` always equal the
real exchanger duty (`HXSimulationResult.q` / `HXRatingResult.Q_required`)
and `Q_latent`/`m_dot_condensate` are `0.0` for a non-active result, exactly
as for any other side/provider without condensation capability.

If the dry-baseline onset screen (a cheap two-point linear wall-temperature
envelope) activates `AUTO`, but the converged nonlinear radial field --
including its 0D endpoint wet-zone fallback -- finds no point below the
local saturation line, the call returns the same dry AUTO result described
above together with a `PHASE_CHANGE_WET_SOLUTION_COLLAPSED_TO_DRY` warning,
instead of raising. This is a legitimate near-boundary collapse (the coarse
onset screen is more conservative than the resolved fin-efficiency field),
not a solver contradiction: `solve_wet_finned_surface` only ever returns a
converged, internally consistent iterate (an unconverged nonlinear solve
raises `WetFinConvergenceError` instead, which is not converted to a dry
result).

### Primary surface, area and contact accounting

The exposed primary/root cylinder is a separate nonlinear surface node and
can condense independently of the fin. The authoritative area network closes
without overlap:

```text
Q_outside = Q_primary + Q_fin
m_dot_condensate = m_dot_condensate_primary + m_dot_condensate_fin
A_wet = A_wet,primary + A_wet,fin
```

Primary and fin duties each retain sensible, latent and total components.
These are conservation identities on the disjoint authoritative primary and
fin areas. If `external_area_per_length` is supplied, the physical per-fin
temperature field is retained while an equivalent fin count is derived from
the authoritative fin area; this preserves the existing override without
double-counting area.

The v0.7.2 contact precedence is unchanged:

```text
explicit fin_contact_resistance (including 0.0)
    > fin_contact_efficiency
    > ideal-contact default
```

The established dry operating-point network first resolves the equivalent
contact resistance, then the wet nonlinear chain applies that loss exactly
once. For `D_root == D_o`, the exposed primary surface bypasses the contact
affected fin branch. For `D_root > D_o`, root conduction and contact are
common series terms before the primary and fin branches. Humidity does not
alter mechanical contact quality, and no separate wet contact efficiency is
defined.

### Numerical method and diagnostics

The default wet mesh has 160 radial cells. A latent-free linear solve supplies
the dry-field initialization, followed by a damped Newton solve of the coupled
tridiagonal heat/mass residual. A residual-RMS backtracking line search keeps
the step within the bulk-temperature bounds. At the non-differentiable
dew-point switch, a frozen-latent-source Picard direction is tried if the
Newton direction cannot reduce the merit norm; the same full nonlinear
residual is still used for acceptance and final convergence. Convergence
requires scale-aware maximum heat-equation and energy residuals, a bounded
temperature step and a bounded condensate-rate step. The
default iteration limit is 80; failure raises `WetFinConvergenceError` with
the iteration count and last residuals instead of returning an unconverged
field. Production code requires neither NumPy nor SciPy.

IAPWS saturation ratio, latent heat and saturated-liquid condensate enthalpy
are evaluated from the authoritative project functions and linearly
interpolated on a deterministic 0.25 K temperature grid. The fin response is
then coupled to the existing whole-HX convergence controls and water-
availability bound.

`WetFinnedSurfaceResult`, reachable as `wet_finned_surface` from `HXResult`,
`HXSimulationResult` and `HXRatingResult` (and as `wet_surface` inside the
finned diagnostics), reports at least:

- the fin wet state, wet fraction and wet/dry boundary radius;
- wet and dry fin area, plus wet primary and whole-surface area/fraction;
- fin and primary sensible, latent and total duties and condensate rates;
- fin-base, fin-tip, primary, root, core-wall, exposed-area-mean and wet-mean
  temperatures; generic phase-change min/mean/max use exposed-surface values,
  while the core temperature remains explicitly nested for the wet-effective
  coefficient basis;
- physical and explicitly named wet-effective coefficients and their bases;
- contact topology and resolved contact-input diagnostics;
- iterations, temperature/heat/mass residuals, split mass/energy errors and
  assumptions. Applicable structured integration warnings, including the
  dry-reference-only pressure-drop warning, are propagated through the
  authoritative phase-change and finned-diagnostics results.

The generic `outside_phase_change` result remains the authoritative
whole-side condensate, sensible/latent/total duty, outlet composition and
mass/energy balance. The nested wet-finned result supplies only the
geometry-specific primary/fin decomposition of that same converged state.
For Rating, the specified temperature program and closed enthalpy balance set
the authoritative whole-side duty and condensate rate. The converged radial
response sets temperatures, wet state/area and the primary/fin distribution;
that distribution is normalized to the closed Rating totals, with its raw
transport duty/rate, gaps and scale factors retained in `residuals` and the
normalization declared in `assumptions`.
When `Rating(..., include_simulation=True)` is requested, the achievable-duty
bridge runs the public phase-aware Simulation once after Rating convergence;
it is a separate achievable operating point and is never a dry internal
Rating snapshot.

All formed condensate is removed from the modelled gas phase as fully drained
saturated liquid. No liquid inventory is retained. The existing dry
Robinson--Briggs bank pressure loss continues to be calculated, but active
wet-finned results expose it as `outside_dp_dry_reference`, set
`wet_pressure_drop_supported = False` and `outside_dp_reference_only = True`,
and emit
`circular_finned_tube_wet_pressure_drop_reference_only`. It must not be read
as the actual pressure drop of a wet bank.

## Correlations and applicability

| Model | Source | Application | Reference diameter | Reference velocity | Reynolds range | Layout | Area basis | Main limitations |
|---|---|---|---|---|---:|---|---|---|
| Annular-fin conduction | Gardner (1945); extended-surface texts | Fin and overall surface efficiency | `D_root` to `D_fin` radial domain | n/a | n/a | Individual full circular fin | Actual two faces plus tip | 1D radial conduction, uniform physical `h_o`, linear taper |
| Wet annular-fin conduction and condensation | Sharqawy, Moinuddin & Zubair (2012); Sharqawy & Zubair (2007); Rosario & Rahman (1999) | Outside H2O condensation from a non-condensable carrier gas | `D_root` to `D_fin` radial domain | Same Briggs--Young `V_max` basis | Same dry-correlation applicability | Individual full circular fin in the supported staggered bank | Authoritative disjoint primary and fin areas; both faces plus tip | Nonlinear 0D radial field; one condensable; full drainage; no wet hydraulic correction |
| Briggs--Young (1963) | *Chem. Eng. Prog. Symp. Ser.* 59(41), 1--10 | Dry outside HTC | `D_root` | `V_max` on `A_min` | 1,100--18,000 | Staggered equilateral triangular; positive row count; source banks had 6 and a later recommendation is at least 4 | Physical film coefficient on exposed gross surface; efficiency separate | Air data; 1--3 rows and other gases are unvalidated extrapolations; no inline or row correction |
| Robinson--Briggs (1966) | *Chem. Eng. Prog. Symp. Ser.* 62(64), 177--184 | Dry bank pressure loss | `D_root` | `V_max` on `A_min` | 2,000--50,000 | Staggered equilateral triangular; at least 4 rows, source banks had 6 | Dynamic pressure based on `V_max`; coefficient is per row | Isothermal air data; materially isosceles layouts are outside the verified v0.7.0 scope |

### Briggs--Young heat transfer

For the empirical correlation only, a tapered fin is mapped to
`t_mean=(t_r+t_t)/2` and `s=p_f-t_mean`. This does not replace the real
profile in area or fin-efficiency calculations. The source dataset does not
define arbitrary linearly tapered fins, so this is a controlled engineering
mapping and emits an information warning; it is not a source-validated
tapered-fin Briggs--Young correlation.

```text
j = 0.134 Re_Droot**(-0.319)
    (s/H)**0.2 (s/t_mean)**0.1134
Nu_Droot = j Re_Droot Pr**(1/3)
h_o = Nu_Droot k/D_root
```

Published geometry diagnostics include `0.13 <= s/H <= 0.63`,
`1.01 <= s/t_mean <= 6.62`, `0.09 <= H/D_root <= 0.69`,
`0.011 <= t_mean/D_root <= 0.15` and
`1.54 <= P_t/D_root <= 8.23`, with `11.1 mm <= D_root <= 40.9 mm`
and `246 <= 1/p_f <= 768 1/m`. Source banks had six rows; use down to
four rows is reported as a later, secondary recommendation. Provider metadata
therefore retains `source_test_rows` as exactly 6 rows and
`secondary_recommended_rows` as 4 rows or more. These ranges describe the
evidence and later recommendation rather than a mathematical requirement of
the equation.

Every positive row count is calculable. For one to three rows, v0.7.4 uses
the same uncorrected Briggs--Young equation shown above and returns
`briggs_young_1963_small_row_count_extrapolation` with severity `warning`.
Its source is `finned_tube_outside_ht`. The diagnostic states that the request
is an unvalidated small-row-count extrapolation and that no row correction was
applied. It replaces, rather than duplicates, the general row-count diagnostic.
Four or five rows and row counts above six retain
`briggs_young_1963_row_count_secondary_extension` with severity `info`; six
rows produce no row-count diagnostic. Other independent applicability
warnings may coexist with these diagnostics.

The equilateral check uses a 0.1% relative engineering tolerance so rounded
nominal dimensions are not rejected as materially isosceles.

### Robinson--Briggs pressure loss

```text
f_RB = 9.465 Re_Droot**(-0.316)
       (P_t/D_root)**(-0.927) (P_t/P_d)**0.515
delta_p_drag = 2 f_RB n_rows rho V_max**2
```

`f_RB` is a source-defined per-row friction coefficient. Relative to the
usual dynamic pressure `q=rho V_max**2/2`, the complete-bank coefficient is
`4 f_RB n_rows`; this distinction avoids a factor-of-two error. Variable
properties are evaluated at inlet, midpoint and outlet and integrated on
the same `G_max`, while the signed inlet-to-outlet acceleration term remains
separate.

Published geometry diagnostics include `0.15 <= s/H <= 0.19`,
`3.75 <= s/t_mean <= 6.03`, `0.35 <= H/D_root <= 0.56`,
`0.011 <= t_mean/D_root <= 0.025` and
`1.86 <= P_t/D_root <= 4.60`, with `18.6 mm <= D_root <= 40.9 mm`
and `311 <= 1/p_f <= 431 1/m`. Explicit endpoint properties require an
explicit midpoint property state for the three-state drag integral; the code
does not silently substitute the inlet state.

## Diagnostics and unsupported cases

Provider results report method/source, geometry family, velocity and
Reynolds bases, reference diameter, area and row bases, applicability ranges
and structured warnings. The built-in models reject bare tubes, inline
banks, materially isosceles banks and geometry-family mismatches rather than
falling back to a smooth-tube model. Robinson--Briggs pressure loss also
rejects banks with fewer than four rows. Briggs--Young heat transfer instead
calculates every positive row count and reports the dedicated extrapolation
warning for one to three rows. A 0.1% relative tolerance on `P_d/P_t` accepts
ordinary rounded dimensions without extending the model to arbitrary
isosceles geometry.

An active **outside** H2O-condensing surface on a `CircularFinnedTube` is
supported when the outside capability is a gas mixture with a
non-condensable carrier and H2O as the single condensable species. The public
`CircularFinnedTubeWetSurfaceNotSupportedError` remains available for
genuinely unsupported capability, species, direction or simultaneous phase-
change combinations rather than being removed. Existing tube-side
evaporation, steam-condensation and wet-gas-condensation paths remain
available opposite a dry finned outside surface and use the same topology-
aware outside resistance network. Wet-gas cases also remain available as dry
sensible calculations when phase change is explicitly disabled and the
requested state remains valid.

## Validation limits

The original Briggs--Young experiments used air, and the integrated default
labels its correlation state as air. The low-level contracts can label another
gas and emit an extrapolation warning; the wet path requires an explicit
provider capability declaring the supported gas-mixture/H2O-condensation
case rather than inferring phase behavior from arbitrary properties.
Briggs--Young was based on equilateral triangular banks. The tracked
Robinson--Briggs provenance does not establish an unambiguous production
mapping for arbitrary isosceles pitch geometry, so v0.7.0 conservatively
supports only the equilateral case. Correlation warnings are engineering
diagnostics, not permission to extrapolate blindly. Industrial/vendor data
should be used for final design validation.

The v0.7.5 wet extension remains one global 0D exchanger. It does not perform
row-by-row or longitudinal marching, circuit-by-circuit temperature mapping,
or thermal segmentation through `n_passes_transverse`. It does not cover
freezing, acid dew point, multiple condensables, flow maldistribution,
condensate-film resistance, retention, flooding, bridging, re-evaporation,
carryover/re-entrainment, explicit drainage geometry or wet pressure-drop
correction. At most one exchanger side may have active phase change.
