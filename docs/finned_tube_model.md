# Circular-Finned-Tube Model (v0.7.0; fin/root contact input extended in v0.7.2)

This document describes the v0.7.0 release-candidate dry
circular-finned-tube model pending independent reference validation.
It is a 0D engineering model with explicitly declared geometry and
correlation boundaries. v0.7.2 adds a practical dimensionless
`fin_contact_efficiency` alternative to the explicit `fin_contact_resistance`
input; see [Fin/root contact input](#finroot-contact-input) below. It is not
a new correlation and does not change the fin-efficiency, root-conduction,
Briggs--Young or Robinson--Briggs physics described here.

## Scope

The model covers round core tubes with individual, full circular fins and a
dry, single-phase gas in crossflow. A continuous helical fin is represented
as an equivalent periodic train of complete annular fins. The 0D model uses
the fractional count `length_effective / fin_pitch`; it does not round the
number of fins and therefore does not introduce area jumps when tube length
changes.

The implemented empirical correlations cover staggered equilateral triangular
banks.
`TubeBundle(layout="inline")` remains valid public geometry, but the built-in
Briggs--Young and Robinson--Briggs providers reject it as unsupported.

Segmented, serrated, wavy and plate fins; elliptical or flattened tubes; wet
surfaces; condensation; frost/ice; condensate retention or carry-over;
radiation; and detailed fin-to-tube joint mechanics are outside this model.
No smooth-tube correlation is used as a fallback.

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

For compatibility, `OutsideThermalDispatchResult.alpha` and
`FinnedTubeDiagnostics.outside_htc` remain aliases for the **physical** film
HTC. Generic top-level `alfa_o` / `outside_alfa_mean` fields instead report
the **effective gross-area** value (the values coincide for a plain tube).

## Correlations and applicability

| Model | Source | Application | Reference diameter | Reference velocity | Reynolds range | Layout | Area basis | Main limitations |
|---|---|---|---|---|---:|---|---|---|
| Annular-fin conduction | Gardner (1945); extended-surface texts | Fin and overall surface efficiency | `D_root` to `D_fin` radial domain | n/a | n/a | Individual full circular fin | Actual two faces plus tip | 1D radial conduction, uniform physical `h_o`, linear taper |
| Briggs--Young (1963) | *Chem. Eng. Prog. Symp. Ser.* 59(41), 1--10 | Dry outside HTC | `D_root` | `V_max` on `A_min` | 1,100--18,000 | Staggered equilateral triangular; at least 4 rows, source banks had 6 | Physical film coefficient on exposed gross surface; efficiency separate | Air data; no inline or row correction; other gases are an extrapolation |
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
four rows is reported as a secondary recommendation. The equilateral check
uses a 0.1% relative engineering tolerance so rounded nominal dimensions are
not rejected as materially isosceles.

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
banks, materially isosceles banks, too few rows and geometry-family mismatches
rather than falling back to a smooth-tube model. A 0.1% relative tolerance on
`P_d/P_t` accepts ordinary rounded dimensions without extending the model to
arbitrary isosceles geometry.

An active wet or condensing **outside** surface on a `CircularFinnedTube` is
conservatively rejected with a dedicated controlled unsupported error. The
existing tube-side evaporation, steam-condensation and wet-gas-condensation
paths remain available when the finned outside surface is dry; they use the
same topology-aware outside resistance network. Wet-gas cases also remain
available as dry sensible calculations when phase change is explicitly
disabled and the requested state remains valid.

## Validation limits

The original experiments used air, and the integrated default labels the
outside correlation state as air. The low-level contracts can label another
dry gas and emit an extrapolation warning; arbitrary property providers do
not expose enough phase metadata for the engine to infer gas versus liquid,
so the caller remains responsible for supplying a dry single-phase gas state.
Briggs--Young was based on equilateral triangular banks. The tracked
Robinson--Briggs provenance does not establish an unambiguous production
mapping for arbitrary isosceles pitch geometry, so v0.7.0 conservatively
supports only the equilateral case. Correlation warnings are engineering
diagnostics, not permission to extrapolate blindly. Industrial/vendor data
should be used for final design validation.
