# Experimental Circular-Finned-Tube Model

This document describes the experimental dry circular-finned-tube model on
the `experiment/v0.7.x-finned-tubes-codex` branch. It is not a released
`v0.7.0` feature and does not change the project version.

## Scope

The model covers round core tubes with individual, full circular fins and a
dry, single-phase gas in crossflow. A continuous helical fin is represented
as an equivalent periodic train of complete annular fins. The 0D model uses
the fractional count `length_effective / fin_pitch`; it does not round the
number of fins and therefore does not introduce area jumps when tube length
changes.

The implemented empirical correlations cover staggered triangular banks.
`TubeBundle(layout="inline")` remains valid public geometry, but the built-in
Briggs--Young and Robinson--Briggs providers reject it as unsupported.

Segmented, serrated, wavy and plate fins; elliptical or flattened tubes; wet
surfaces; condensation; frost/ice; condensate retention or carry-over;
radiation; and detailed fin-to-tube joint mechanics are outside this model.
No smooth-tube correlation is used as a fallback.

## Public geometry

`CircularFinnedTube` composes an existing `BareTube` as `core_tube`. The
inside diameter, outside core diameter, total/effective lengths, wall
conductivity, roughness and orientation are delegated to that object. This
keeps one source of truth for core-tube validation and tube-side hydraulics.

The additional SI inputs are:

- `fin_k` [W/(m K)];
- `fin_contact_resistance` [m2 K/W] or `None`;
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
R_total   = R_i + R_wall + 1/(G_primary + G_fin)
```

For an extruded geometry (`D_root>D_o`) the root layer is continuous. Its
core interface has basis `A_contact=pi D_o L N`; root conduction and contact
are therefore common series terms before the parallel outside convection:

```text
R_root    = ln(D_root/D_o)/(2 pi fin_k L N)
R_contact = R_contact''/A_contact
R_total   = R_i + R_wall + R_root + R_contact
            + 1/[h_o (A_primary + eta_fin A_fin)]
UA        = 1/R_total
U_out,gross = UA/A_out,gross
```

This is an explicit 0D equivalent network, not a detailed joint-mechanics
model. `fin_contact_resistance=0.0` means known ideal contact; `None` uses
zero but produces a warning that data were not supplied.

Fin efficiency and contact resistance are each applied exactly once.

## Correlations and applicability

| Model | Source | Application | Reference diameter | Reference velocity | Reynolds range | Layout | Area basis | Main limitations |
|---|---|---|---|---|---:|---|---|---|
| Annular-fin conduction | Gardner (1945); extended-surface texts | Fin and overall surface efficiency | `D_root` to `D_fin` radial domain | n/a | n/a | Individual full circular fin | Actual two faces plus tip | 1D radial conduction, uniform physical `h_o`, linear taper |
| Briggs--Young (1963) | *Chem. Eng. Prog. Symp. Ser.* 59(41), 1--10 | Dry outside HTC | `D_root` | `V_max` on `A_min` | 1,100--18,000 | Staggered equilateral triangular; at least 4 rows, source banks had 6 | Physical film coefficient on exposed gross surface; efficiency separate | Air data; no inline or row correction; other gases are an extrapolation |
| Robinson--Briggs (1966) | *Chem. Eng. Prog. Symp. Ser.* 62(64), 177--184 | Dry bank pressure loss | `D_root` | `V_max` on `A_min` | 2,000--50,000 | Staggered triangular; at least 4 rows, source banks had 6 | Dynamic pressure based on `V_max`; coefficient is per row | Isothermal air data; only two isosceles banks; high uncertainty outside original geometry |

### Briggs--Young heat transfer

For the empirical correlation only, a tapered fin is mapped to
`t_mean=(t_r+t_t)/2` and `s=p_f-t_mean`. This does not replace the real
profile in area or fin-efficiency calculations.

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
banks, too few rows and geometry-family mismatches rather than falling back
to a smooth-tube model.

Any attempt to activate wet-gas condensation or another wet/phase-change path
on a `CircularFinnedTube` is conservatively rejected with a dedicated
controlled unsupported error, including tube-side phase change in this first
experiment. Wet-gas cases remain available as dry sensible calculations when
phase change is explicitly disabled and the requested state remains valid.

## Validation limits

The original experiments used air, and the integrated default labels the
outside correlation state as air. The low-level contracts can label another
dry gas and emit an extrapolation warning; arbitrary property providers do
not expose enough phase metadata for the engine to infer gas versus liquid,
so the caller remains responsible for supplying a dry single-phase gas state.
Briggs--Young was based on equilateral
triangular banks. Robinson--Briggs included fifteen equilateral and only two
isosceles banks; later comparisons report substantial disagreement outside
that narrow data family. Correlation warnings are therefore engineering
diagnostics, not permission to extrapolate blindly. Industrial/vendor data
should be used for final design validation.
