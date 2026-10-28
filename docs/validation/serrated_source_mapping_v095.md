# v0.9.5 serrated source mapping

Original PDFs supplied in the private source pack were inspected directly,
including rendered equation pages, before implementing equations. Page numbers
below are printed pages; Holfeld PDF pages are printed page + 13. This is an
independent implementation record, not a reproduction of source tables.

## Sources and access

- A. Holfeld (2016), NTNU thesis 2016:303, *Experimental investigation of heat
  transfer and pressure drop in compact waste heat recovery units*,
  [institutional record](https://hdl.handle.net/11250/2432512).
  INSTITUTIONAL_FREE_ACCESS; no open redistribution licence established.
- Hashizume et al. (1997), pp. 4029–4035,
  [DOI 10.1299/kikaib.63.4029](https://doi.org/10.1299/kikaib.63.4029).
- Hashizume et al. (1999), pp. 1498–1501,
  [DOI 10.1299/kikaib.65.1498](https://doi.org/10.1299/kikaib.65.1498).
- Hashizume et al. (2000), pp. 2125–2132,
  [DOI 10.1299/kikaib.66.648_2125](https://doi.org/10.1299/kikaib.66.648_2125).
- Hashizume et al. (2002), pp. 2827–2833,
  [DOI 10.1299/kikaib.68.2827](https://doi.org/10.1299/kikaib.68.2827).
- Kawaguchi et al. (2003), pp. 2670–2676,
  [DOI 10.1299/kikaib.69.2670](https://doi.org/10.1299/kikaib.69.2670),
  and (2005), pp. 1854–1861,
  [DOI 10.1299/kikaib.71.1854](https://doi.org/10.1299/kikaib.71.1854).

The Japanese journal originals are FREE_ACCESS_COPYRIGHTED. No CC licence is
asserted. PDFs, extracted text and rendered pages remain private and untracked.
The two short conference papers and the 2006 INLINE paper were inspected as
supplementary material; they do not authorize a complete INLINE execution.

## Common geometry and state contract

All lengths in implementation are metres. Holfeld `do` is core bare OD,
`df = do + 2 hf`, `bf`/`tf` is constant thickness, `hs` is segmented radial
height, `ws` is the physical rectangular strip-segment width. `Nf` is turns/m,
`sf` is **fin pitch**, and `gf = sf - tf` is clear spacing (notation, pp. x–xi;
p. 30). `Pt`, `Pl` are centre pitches across/along flow; `Nl` counts rows along
flow. `lt` is effective axial tube length; `Nt` is the effective transverse
flow-cell count. No total-turn count is substituted for `Nf`.

The explicit source geometry requires `D_root = core.D_o`, constant thickness,
untwisted welded I-foot, and no removed kerf material. The required
`source_segment_width` is an independent measured strip dimension. Phase A's
derived outer-radius arc width is not substituted for it. Phase A radial-sector
volume, mass and geometric areas remain unchanged. A source-equivalent surface
area is reported separately and is used only by the explicit source network.

## Holfeld equations

| Source location | Mapping and definition | Reference contract |
| --- | --- | --- |
| p. 89 Eq. 5-9 | `A_min = lt Nt (Pt - do - 2 Nf hf tf)` [m2]; `Af_fin = lt Nt 2 hf (1-Nf tf)` [m2] (notation) | SOURCE_EQUIVALENT_FLOW_AREA; not orthographic Phase A blockage; `A_face = lt Nt Pt` |
| p. 30 Eqs. 2-17, 2-19 | `Ar = Aht/Abt`; I-foot expression evaluated algebraically as `1 + 2 Nf [hf + hs tf/ws + (hf-hs)(hf+hs+tf+2hs tf/ws)/do]` | `Abt' = pi do`; native `Aht' = Ar pi do`; native fin area = native total minus exposed primary `pi do (1-tf/sf)`; all per axial metre |
| p. 89 Eq. 5-7; notation | `Re = (mdot/A_min) do/mu`; `u_max = mdot/(rho A_min)` | bare OD; source minimum area; no hydraulic diameter |
| p. 57 Eq. 4-9 | `Tref = (Tin+Tout)/2` [K] | caller supplies bulk mean-state transport properties; not wall/film-temperature properties; dry air calibration |
| p. 126 Eq. 6-6 | `Nu Pr^(-1/3) = .184 Re^.696 Ar^(-.655) (Pt/do)^.262 (hf/do)^.602 (sf/do)^(-.729)` | `Nu = h do/k`, `Pr=cp mu/k`; **physical film h**, before eta; no row multiplier |
| pp. 54–56 Eqs. 4-4 to 4-6, sec. 4.2 | experimental outside film is recovered using `eta_f Af + At` in the resistance reduction | eta is removed from measured overall conductance; apply the selected eta once in the normal network |
| pp. 56–57 Eqs. 4-7, 4-8 | `Eu=2 dp rho / [(mdot/A_min)^2 Nl]`; invert as `dp=Eu Nl rho u_max^2/2` [Pa] | native Eu, not RB/Darcy/Fanning/Frass/Genic; source density is arithmetic inlet/outlet mean |
| p. 130 Eq. 6-10 | `Eu=9.661 Re^(-.197) (Pt/Pl)^.931 (A_min/Af_fin)^(-.776) (1-hs/hf)^.112` | adiabatic bundle pressure difference; bank-section loss, not isolated skin friction; rows counted once |

Pressure scope: sec. 4.3 explicitly identifies the measured difference as
across the bundle, separates friction and acceleration, and neglects acceleration
(about 1% of measured loss) under adiabatic conditions. Fig. 3-1 and sec. 3.3
locate the differential measurement at the test section. Exact tap distances and
a separate empty-channel subtraction are **not reported**. The implementation
retains this measured bundle-section scope, including bundle entrance/exit wake
effects; it does not claim an independently isolated wall/skin-friction loss or
add separate inlet/exit coefficients. Acceleration from changing bulk state
remains a separate face-area momentum term. This source convention is an
approximation, not an exact subtraction of the measured 1% acceleration.

Holfeld's own serrated film data use Weierman's corrected theoretical eta
(sec. 4.2), while the mixed regression database uses several reduction
conventions. Selecting Hashizume eta does not recalibrate that regression.
This calibration difference is exposed, not removed by tuning.

### Hard source bounds

Sec. 6.1 p. 112: staggered, `Re < 50000`, `Nl >= 4`, arrangement angle
`atan(Pt/(2 Pl)) < 45 degrees`, `Pt/Pl < 2` (strict upper inequalities).
Table 6-2 p. 115 gives database bounds plus the present tests in parentheses.
The implementation includes both sets: `hf/do` upper .94 and `hs/hf` lower .50.
Other finite bounds are implemented component by component, including different
HT/DP minimum diameters/heights and maximum `Aht/A_min`. No lower Re, upper row
count, thickness, `ws`, separate `Pl/do`, or property-ratio bound is reported.
Those bounds remain explicitly unreported. Data envelope membership does not
guarantee accuracy for every combination of dimensions.

Area-ratio normalization follows the notation's single-cell, unit-axial-length
areas: `Aht/A_min = Aht'/(Pt-do-2 Nf hf tf)`. There is no longitudinal row
factor in this geometric descriptor. Scaling numerator and denominator by
`lt Nt` gives the same ratio. As an independent cross-check, Geometry 1 gives
about 59, matching the present-test range ending at 59.1 in Table 6-2; using
whole-bank heat-transfer area would incorrectly give about 472.

Source-reported regression coverage (not prediction uncertainty): Eq. 6-6,
284 points, 81% within 10%, 95% within 21%, R2 .958; Eq. 6-10, 442 points,
44% within 10%, 95% within 34%, R2 .739 (Tables 6-4, 6-8).

Discrepancies retained in the audit: p. 89's prose areas .72/.63 m2 conflict by
a factor of ten with Eq. 5-9 and test dimensions (about .072/.063 m2). Use the
verified equation and measured Reynolds basis. The p. 113 prose height-ratio
upper value .91 differs from Table 6-2's .94; use the explicit table. The .99
serration limit includes L-foot substitution in the database; L-foot remains
unsupported here.

## Hashizume theory and correction

1997 pp. 4030–4031 Eqs. 1–11, 16: `r0=do/2`, `r2=df/2`, `r1=r2-hs`,
`theta=T-T_air`, `m=sqrt(2 h/(k_fin tf))` [1/m]. The continuous root annulus
satisfies `u''+u'/r-m^2 u=0`, with `I0(mr), K0(mr)`; the equivalent straight
segment satisfies `v''-m^2 v=0`, with sinh/cosh. Conditions are prescribed
base `u(r0)=theta0`, continuity of temperature and derivative at `r1`, and
adiabatic tip `v'(r2)=0`. Segment sides are insulated in this theoretical
idealization; both broad faces exchange heat with uniform film h.

The theoretical denominator in Eq. 11 is
`theta0 [(r1^2-r0^2)/2 + r1 (r2-r1)]`. Equivalently the two broad-face areas
are `A_theory = 2 pi [r1^2-r0^2+2 r1 hs]` per turn. This is not Phase A's
full annulus/slit-side/tip area. The implemented scaled-Bessel boundary solution
uses the equivalent Robin condition `u'(r1)+m tanh(m hs) u(r1)=0` and computes
eta from root heat flow divided by `h A_theory theta0`. This is an algebraic
solution of the source boundary problem, not a call to solid annular eta.
The theoretical normalized temperature field does not establish a corrected
experimental tip temperature.

1999 p. 1500 Eq. 9 interpolates straight and Schmidt annular approximations
using `cos(pi hs/(2 hf))^n`, `n=1.6-.094 (m hf)(df/do)` and
`h_equiv=hf[1+.35 ln(df/do)]`. Reported agreement with theory is within 1.5%
for `df/do<=3`, `m hf<=2` (p. 1501). This approximation is documented only;
the exact source boundary solution is used instead. Fig. 6 supplies three
coarsely digitized theoretical checks, not experimental measurements.

2000 p. 2127 Eq. 4 defines **experimental** eta using total physical fin area,
including tips/sides; Eq. 6 is the theoretical solution. The empirical correction
translates between these conventions. The 2000 main Eq. 10 (p. 2129) prints
`hs/h`, but its own appendix Eqs. 21–23 (p. 2132) uses `hs/ws`.
**Do not silently repair Eq. 10.** The implemented correction is the explicit
STAGGERED restatement in **2002 p. 2830 Eq. 4**, which prints:

`psi = 1 - [.016 hs/ws + .14 (df/do)^2.7 (1-.097 ln Re_o)] m hf`.

Apply `min(1, psi)` as directed by the source, and `eta_corrected=eta_theory psi`.
Natural logarithm; `Re_o=rho u_min do/mu`; physical bulk mean film h;
`k_fin` [W/(m K)] constant at caller's declared material state; no row correction
for STAGGERED. Source bounds from 2000 p. 2131 sec. 5: `1.80<=df/do<=2.15`,
`0<=hs/hf<=.86`, `0<=hs/ws<=2.60`, `5000<=Re_o<=50000`, `0<=m hf<=2.5`.
The narrower 2002 INLINE bounds (p. 2833) do not replace these STAGGERED bounds.
All bounds are strict guards in the explicit complete path, even if the generic
engine's optional strict flag is false. Unknown thickness/pitch/row bounds
are not invented. The complete Re intersection is therefore `[5000,50000)`.

## Independent Kawaguchi comparison contract

2003 p. 2672 defines `f=2 rho dp/(Ga^2 Nl)`, `Ga=mdot/A_min`,
`Re_h=Ga dh/mu`, `dh=4 A_flow/P_wetted`. P. 2675 Eq. 2:
`f=6.46 Re_h^(-.179) (s_f/t_f)^(-.354)` for `3000<=Re_h<=30000`,
`3.07<=s_f/t_f<=5.07`. Here `s_f` is **clear spacing**, not Holfeld pitch.
The tested transverse/longitudinal pitches and 3–6 rows are explicit. The
segment width/depth needed for a common Holfeld geometry are not both given
numerically for SR200/SR300. Their reported hydraulic diameters are source
facts, not permission to substitute `do` or an invented hydraulic diameter.

2005 p. 1857 defines `Re_v=Ga dv/mu`, `Nu=h dv/k`; `dv` is the diameter of
a cylinder with the bare-tube-plus-fin material volume, not projected blockage.
`A_min` is duct area minus first-row projection. Data reduction removes eta
iteratively (film HTC). P. 1860 Eqs. 2–3 give low/high-Re serrated regressions,
with explicit row constants for 3–6 rows, `hf/dv` .31–.45 and clear-spacing/pitch
.76–.82. Missing strip width prevents an exact common-geometry/source-surface
mapping without an additional assumption. These families remain validation-only;
no optional production selector or fabricated overlap comparison is enabled.
Comparison limitations must be reported alongside any test-only arithmetic.

For an explicitly labelled **equation comparison**, a native strip-volume
idealization can be stated independently: integrating the same annulus plus
straight segments used in the 1997 theory gives
`V_fin' = pi tf Nf [hf(do+hf)-hs^2]` and hence
`dv^2 = do^2 + 4 tf Nf [hf(do+hf)-hs^2]`. This is an engineering derivation
of the source's volume-equivalent-diameter definition, not a reported measured
diameter or Phase A's radial-sector volume. A comparison using it must retain
that qualification and unreported segment-shape bounds. It does not establish
matched-experiment agreement. The 2005 six-row scalar equations used for that
private arithmetic are `Nu=.068 Re_v^.81 Pr^(1/3) (hf/dv)^.24 (sf/Pf)^(-.07)`
below Re_v=30000, and `.041 Re_v^.88 Pr^(1/3) (hf/dv)^.44 (sf/Pf)^(-.38)`
above it. The boundary belongs to both reported intervals; compare away from
it without inventing a blending rule. Property state is the caller's bulk
mean air state, matching the reported data reduction.

A measured-case fallback is also limited: the 2005 SR210 geometry has
`hs/hf=6.3/13=.4846`, below Holfeld's .50 lower bound; the lower fins likewise
do not supply a matched complete Hashizume/Holfeld case. The 2003 SR200/SR300
strip dimensions and an unambiguous wetted-perimeter construction for a
cross-family hydraulic-diameter mapping remain unresolved. Its plotted
3.938/3.312 mm labels must not be substituted as hydraulic diameters: those
labels concern fin spacing. No valid DP-bias ratios can be claimed from that
substitution.

INLINE preparation is documentation only. The 2002 paper distinguishes its
first row from deeper rows and supplies a narrower INLINE eta domain. The
2006 paper p.559 Eq.7 prints `.44 (s/h)^.60 Re^.12` and requires at least
five rows and a separately valid staggered baseline. That baseline is not
registered as a serrated INLINE provider; no INLINE DP is supplied.

## Minimal numerical facts

Holfeld Geometry 1 (pp. 49–51): do 31.75 mm, hf 18 mm, hs 11 mm, ws 4.5 mm,
tf 1 mm, Nf 277/m, Pt 69.75 mm, Pl=sqrt(3) Pt/2, eight rows, lt .5 m,
4.5 transverse flow cells (four active tubes plus boundary half tubes).
Flow-cell count is not substituted for the measured count of active tubes
when reconstructing heat duty.

Appendix II p. 180: three measured `(mdot kg/s, Tin C, Tout C, Re, h W/m2K,
Nu Pr^(-1/3))` facts:
`(.34,117.9,32.0,8312,73,87)`, `(.81,120.3,48.2,19196,126,147)`,
`(1.43,123.6,63.9,33078,177,202)`.
P. 181 measured `(mdot,Tmean C,p_in Pa,Re,dp Pa,Eu)`:
`(.33,18.6,100092,9257,173,1.84)`, `(.74,20.3,100077,20380,678,1.47)`,
`(1.29,23.2,100114,35127,1806,1.29)`.
Printed mass flows and coefficients are rounded. Measured comparisons retain
the observed residuals; neither exact arithmetic nor source coverage is a
guarantee that all three data points lie inside a selected percentage band.

Hashizume 1999 p. 1501 Fig. 6: df/do=1.8, hs/hf=2/3, theoretical serrated
eta approximately .91, .74, .58 at m hf=.5,1,1.5 respectively (digitization
tolerance .02). Re is not applicable; experimental correction is not applied.
These are DIGITIZED_SOURCE_ESTIMATE/THEORETICAL_GRAPH, not measured eta.
