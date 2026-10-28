# Zhao lattice and YC classification audit — 2026-10-27

Branch: `fix/ellipse-yc-shifted-layout`. Base: merged default branch `main`,
`ff4f9137554257af6fd1d7ea57a35f9169c30288`, refreshed from origin before
branch creation. The unrelated serrated-fin commits are excluded. No history
was rewritten. Numerical YC inventories and comparisons remain in ignored
`.kon/_worklogs/ellipse_yc_shifted_20261027/`; no raw YC data are published here.

Closure update: the recovered Zhao 2017 Fig. 2 and Zhao & Yang 2016 Fig. 1
resolve the source lattice identities and Pt/Pl axes. The user confirms YC
SHIFTED follows the Zhao STAGGERED_COLUMN design basis. Geometry identity is
closed; a generic SHIFTED physical correlation remains outside this branch.

## Physical identities and API

`core.geometry.TubeBankLattice` is independent of tube section, fin type and
correlation. Coordinates use x along outside flow, y transverse to flow,
zero-based downstream row j and transverse site i:

| Identity | Coordinates/meaning |
|---|---|
| INLINE | x=j Pl, y=i Pt |
| CONVENTIONAL_STAGGERED (legacy `staggered`) | x=j Pl, y=i Pt+(j mod 2) Pt/2; all sites in a downstream row share the offset |
| SHIFTED | Per-site (dx,dy) offsets; a whole-row translation is not inferred from a nominal Pt/2 |

Unqualified SHIFTED still has `pattern='UNRESOLVED'` and cannot invent
coordinates. The small `TubeBankLattice.zhao2016_staggered_column` preset now
provides Fig. 1(b) offsets for specified finite row/site counts, with pattern
`ZHAO2016_STAGGERED_COLUMN_FIG1B` and a recorded canonical phase. Other resolved
SHIFTED cases require explicit SI offsets indexed [row][site], pattern and
phase. No generic checkerboard/striped pattern is inferred. Offset arrays must
match physical site counts; coincident centers are rejected. Shape-specific
collision/clearance analysis remains outside this geometry foundation. SHIFTED
currently requires integer sites per row.

`TubeBundle(..., lattice=...)` makes explicit physical identity authoritative
over a stale legacy label, canonicalizing `bundle.layout` accordingly. Legacy
inputs retain their existing count normalization, section RESET/CONTINUOUS
parity and numerical results. `tube_center_coordinates` exposes that parity.
This is a geometry foundation, not a general free-area model for arbitrary
lattices. Conventional minimum-gap/reference-velocity formulas reject SHIFTED.

Zhao thermal and pressure entry points now reject INLINE/SHIFTED before
calculation, independently of strict-domain mode. Built-in conventional
finned-bank request guards remain in force (Briggs–Young, Robinson–Briggs,
Frass/ESCOA and Genić); SHIFTED cannot select a conventional pressure AUTO
provider. Bare-bank automatic reference velocity also rejects SHIFTED.
**Generic SHIFTED HTC=NO; DP=NO; AUTO=NO.**

## YC classification facts and traced error

The user's 2026-10-27 classification supersedes historical labels:

| Family | Correct grouping | Pitch | Feature count |
|---|---|---|---|
| CO30R | Two conventional STAGGERED repeats combined; one separate SHIFTED series | 3.0 mm | 10 |
| CO35P | Separate SHIFTED material case | 3.5 mm | 4 |
| CM35P | Separate SHIFTED material case | 3.5 mm | 4 |

The user's follow-up confirms nominal Pt/Pl of 26.67/60.00 mm for YC STAGGERED
and 53.34/30.00 mm for YC SHIFTED, with 26.67 mm nominal SHIFTED displacement.
The closure instruction supersedes the earlier request to leave identity
unresolved: YC SHIFTED was intentionally built on Zhao STAGGERED_COLUMN.
Record `USER_CONFIRMED_DESIGN_BASIS_ZHAO_STAGGERED_COLUMN`, not
`SOURCE_PROVEN_FROM_YC_DRAWING`. No separate YC drawing was recovered or is
required for this design-basis identity. Pt is transverse, Pl follows flow;
the source preset supplies a canonical origin/phase. It does not reconstruct
unreported finite-edge cuts from the nominal periods or replace measured face
area. No TEMA conversion or substitution into Zhao2017 physics is made.

The locally available CO30R reference already combined only its two standard
series at source-defined nominal stations. The separate third series was
mislabelled as a wide conventional staggered case. There is **no evidence in
the traced averaging path of a three-series CO30R mean**. Relabelling changes
zero standard working points and has zero dataset-mixing effect on that path.
Individual source points are retained. Nominal-station averaging is historical
source policy; it is not asserted that all actual flows/Re match exactly.

The latest material research used an assumed conventional bank, not verified
SHIFTED geometry. A consistency gate prevented its proposed common material
curve from closing. No combined measured correction had been deployed. The
corrected workflow prohibits material pooling even if curves numerically agree.
Physical fin conductivity/thickness, area and efficiency remain separate.
Historical research results are retained as surrogate comparisons; no derived
SHIFTED multiplier is installed. Existing trial engineering multipliers are
retained with neutral `YC_CASE_CORRECTION` provenance and target identity/status.

Repository `fin_turbulator_count` means features per individual rectangular fin.
The corresponding YC location/count basis cannot be independently established
from the available drawings: `TURBULATOR_SEMANTICS=UNRESOLVED`, metadata only.
Pressure-tap/fan/system-loss scope is also unresolved; no thermal factor is
reused as a pressure factor. Proprietary source/material details, per-point
residuals and historical versus corrected tables are in the private report.

**Previous underestimation explained by layout error: UNRESOLVED.** Actual
CO30R grouping effect is 0%; a physical baseline-layout contribution cannot be
quantified without source-closed SHIFTED physics. Material, circuit, pitch and
feature effects are confounded. No numerical causal layout attribution is made.

## Recovered source evidence

Production equations rely on Zhao, Gu, Gao, Yang (2017), IJHMT 114,
1314–1330, [DOI 10.1016/j.ijheatmasstransfer.2017.06.049](https://doi.org/10.1016/j.ijheatmasstransfer.2017.06.049).
The full 17-page article is recovered from the user-supplied local archive
`kalkalori0.9.2refs.zip` (entry `372035990-GRE-Prep-1.pdf`). Nomenclature on
p.1315 defines Pt as transverse pitch and Pl as longitudinal pitch. Fig. 2
on p.1317 supplies the flow arrow and staggered bank; the text on p.1316
calls it a six-staggered-row arrangement. **ZHAO2017_SOURCE_LATTICE=RESOLVED:
CONVENTIONAL_STAGGERED; ZHAO_SOURCE_PT_PL=RESOLVED; PT_PL_SWAP_BUG=NO.**
Equations and coefficients remain unchanged. Full texts/rendered figures and
their hashes remain in the ignored private worklog, not in the public package.

Zhao and Yang (2016), *Effect of Tube Arrangement on Performance of Rectangular
Finned Elliptical Tube Bundles*, Tongji 44(2), 298–302,
[official record](https://tjxb.ijournals.cn/jtuns/article/abstract/15023),
DOI 10.11908/j.issn.0253-374x.2016.02.020, is recovered from the same archive,
entry `管束排列方式对矩形翅片椭圆管束性能的影响.pdf`. Direct inspection of Fig. 1(a,b)
on p.299, rather than its conflicting abstract analogies, resolves ROW and
COLUMN as the two coordinate constructions documented below.

Zhao, Wang, Wang, Yang (2019), *Aero-thermal behavior and performance optimization
of rectangular finned elliptical heat exchangers with different tube
arrangements*, IJHMT 133, 1196–1218,
[publisher record](https://www.sciencedirect.com/science/article/abs/pii/S0017931018339711),
DOI 10.1016/j.ijheatmasstransfer.2018.12.138, distinguishes inline, staggered row
and staggered column. It is supporting bibliographic context; no coordinates
or coefficient transfer are inferred from its preview. The resolved preset
in this branch is specifically from the recovered 2016 figure.

The different 2019 DOI 10.11949/0438-1157.20190527 in earlier repository source
notes belongs to **Tang, Du, Zeng**, not Zhao's arrangement paper. It is not a
second production Zhao model. Related Tongji-2016 pitch/performance material
is supporting context, not an independent implemented provider.

### Source-figure coordinates

Choose x along the left-to-right flow arrow and y transverse to flow. Set an
arbitrary origin at an unshifted center; i indexes transverse sites and j
indexes longitudinal stations. Half-period signs differ only by the chosen
origin/site indexing in the periodic construction.

| Source identity | Center coordinates | Evidence |
|---|---|---|
| Zhao 2016 STAGGERED_ROW | x=(j+(i mod 2)/2) Pl; y=i Pt | SOURCE_EXACT, Fig. 1(a), p.299: alternate transverse lines displaced along flow |
| Zhao 2016 STAGGERED_COLUMN | x=j Pl; y=(i+(j mod 2)/2) Pt | SOURCE_EXACT, Fig. 1(b), p.299: alternate longitudinal columns displaced transversely |
| Zhao 2017 conventional staggered source | ROW construction above | SOURCE_EXACT, Fig. 2(b), p.1317 |
| YC SHIFTED design basis | COLUMN construction above; nominal Pt/Pl=53.34/30 mm | USER_CONFIRMED_DESIGN_BASIS_ZHAO_STAGGERED_COLUMN |

**ZHAO_STAGGERED_ROW_COORDINATES=RESOLVED;
ZHAO_STAGGERED_COLUMN_COORDINATES=RESOLVED;
ZHAO_COLUMN_EQUALS_YC_SHIFTED=YES by user-confirmed design basis.**

Source arrangement names must not be substituted for API row indexing. The
legacy conventional coordinate helper uses the COLUMN-form expression in this
table, whereas the recovered 2017 drawing uses the ROW construction at its
printed pitches. The source Pt/Pl axes themselves are correctly mapped; there
is no Pt/Pl swap. This closure records that indexing/phase distinction without
rewriting legacy coordinates or changing correlation arithmetic. The new
COLUMN preset deliberately retains SHIFTED identity even where its numeric
center array equals that legacy helper: identical helper coordinates alone do
not transfer coefficients between source arrangement/pitch conventions.

ROW and COLUMN must not be collapsed at the same source Pt/Pl. Finite YC face
area, tube counts and material remain separate construction inputs. No source
full texts/images or private measurement fixtures are committed.

## Symbol audit

The provider's source family is **conventional staggered**, major axis parallel
to flow; Pt is transverse and Pl longitudinal. The coordinate-helper naming
distinction above is recorded explicitly and does not change its arithmetic.
`ZHAO2017_IMPLEMENTED_LATTICE` and `ZHAO2017_SOURCE_LATTICE_GRADE` expose both.
The previous guard was TOO_BROAD: layout mismatch merely produced an optional
geometry-extrapolation warning. Even INLINE could calculate in default mode.
The narrower guard rejects unsupported identities. `within_source_domain` is
a parameter/envelope check; neither it nor `strict_source_domain` extends
coefficients to the newly identified COLUMN/SHIFTED target. Resolved source
geometry does not upgrade engineering flow normalization to SOURCE_EXACT.

| Quantity / mapping in retained implementation | Evidence grade / finding |
|---|---|
| Source Eqs.18–20 coefficients/exponents | SOURCE_EXACT within retained authorized transcription; unchanged |
| N -> bundle.n_rows (outside-flow rows), not tube passes/count | SOURCE_EXACT source row count; API coordinate-helper distinction documented above |
| Pt -> pitch_transverse (y), Pl -> pitch_longitudinal (x) | SOURCE_EXACT, nomenclature p.1315 and Fig. 2; RESOLVED, no swap |
| Dl -> twice outer semi-major axis, aligned with x | SOURCE_EXACT, Figs. 1/2 and flow arrow |
| Dt -> twice outer semi-minor axis, aligned with y | SOURCE_EXACT, Figs. 1/2 |
| Fin streamwise extent -> outer_height=.055; transverse -> outer_width=.026 | SOURCE_EXACT dimensions/orientation, Fig. 1 p.1316; retained API height/width convention |
| Tube count -> bundle.n_tubes_total; length -> installed physical area/frontal area | ENGINEERING_CONVENTION finite-bank scaling, not an extra N law |
| Xt=(Pt-Dt)/Dt; Xl=(Pl-Dl)/Dl | SOURCE_EXACT transcription; ratio association matches implementation |
| Xp=Fp/de; Xf=Ft/de | SOURCE_EXACT transcription; Fp=axial pitch, Ft=fin thickness |
| de=4 ellipse_area / ellipse_perimeter | SOURCE_INFERRED, not a printed source-exact diameter definition |
| Sg=Pt Fp; Savg=Sg-[(Fp-Ft)Se+Ft Wx Wy]/Pl | ENGINEERING_CONVENTION periodic-volume-average reconstruction |
| Aa=A_frontal Savg/Sg; u_m=m_dot/(rho Aa) | ENGINEERING_CONVENTION reference area/velocity; not source figure recovery |
| Re=rho u_m de/mu; Pr=cp mu/k | Retained source normalization with inferred de/reconstructed u_m |
| Eq.18 j_eff | SOURCE_EXACT transcription; includes fin resistance, diagnostic only |
| Eq.20 j_prime; h=j_prime rho cp u_m / Pr^(2/3) | SOURCE_EXACT transcription/algebra; physical film h, eta applied separately |
| Nu=h de/k (if required) | Algebraic conversion; no additional Nu correlation implemented |
| Eq.19 f; dp=f rho u_m²/2 A_physical/Aa | SOURCE_EXACT transcription/algebra with reconstructed normalization |
| Density/properties at mean bulk state | ENGINEERING_CONVENTION 0D approximation to source mean flow-field properties |
| Pressure taps, bank-only irreversible decomposition | UNRESOLVED; source-domain inlet-to-outlet estimate only |
| Full rectangular fin vs holes/tabs/clipped corners | ENGINEERING_CONVENTION diagnosed simplification |

Retained intervals: N=6–10; Re=5500–23800; Xt=.93–1.50; Xl=.56–.78;
Xp=.10–.52; Xf=.01–.03. Dt=.014 and Dl=.036 imply pitch families
Pt=27.02–35.00 mm and Pl=56.16–64.08 mm under the implementation convention.
The recovered nomenclature/figure, rather than those numerical magnitudes,
prove the axes. `Pt/Pl SWAP BUG=NO`; no equation/axis/coefficient change made.
YC nominal 53.34/30.00 mm remains a separate COLUMN design-basis case.

Existing synthetic 8×6 baseline remains h=86.47833865264215 W/(m² K),
dp=372.055553752331 Pa. Equation fixture at N=8, Re=13000 retains
j_eff=.003596209097641707, j_prime=.005572441209273412,
f=.028160566614707627. The earlier approximately 735 vs 404.49 Pa source
normalization discrepancy remains unresolved and is not fitted away.

## Verification

Branch-closure verification on 2026-10-28:

- Broader affected public geometry/provider/dispatch regression: **436 passed**,
  zero failures, errors, skips or xfails (`closure-public.xml`). Includes the
  source-defined COLUMN coordinates, unchanged Zhao h/dp fixture, INLINE
  rejection, named COLUMN rejection in both domain modes, round providers and
  legacy section parity/count behavior.
- Applicable private YC suite rerun: **92 passed**, zero failures, errors,
  skips or xfails (`closure-private.xml`). Source/design-basis metadata and
  finite-column article coordinates are checked without modifying raw data.
- Six direct AUTO probes with the resolved COLUMN preset all reject
  unsupported physics (`closure-dispatch-guards.json`). No provider guard,
  equation, coefficient or deployed correction multiplier was broadened.
- The named preset is additive; existing coordinate arithmetic is unchanged.
  The affected broader suite above is the closure regression; the earlier
  expensive full-suite run was not repeated for this bounded addition.

Initial branch verification, retained as historical evidence:

- Final targeted geometry, lattice, Zhao thermal/pressure/integration and
  provider checks: **335 passed**.
- Complete public coverage: **2338 collected**, **2338 passed after two
  explicit isolated rechecks**, **0 remaining failures/errors**, **0 skipped**,
  **0 xfailed**. The four-process run passed 2336 cases and hit the unchanged
  300-second wet-rating deadline in two cases. Rechecking those two cases
  alone passed in 183.81 and 241.83 seconds. Parallel wall time was 3174.45 s;
  the separate two-case recheck took 428.60 s. This is full-suite coverage
  with documented retries, not a claim of one uninterrupted clean run.
- Applicable private classification/correction suite: **92 passed**, no
  failures, errors, skips or xfails. Six additional direct AUTO probes
  rejected SHIFTED for both channels across bare round, solid-finned round
  and rectangular-finned elliptical tubes.

An earlier incomplete serial run exposed two diagnostic-precedence failures
for geometry that was both rotated and INLINE. Orientation validation again
precedes lattice validation; the final targeted run covers that correction.
No solver deadline or physical correlation coefficient was changed. Original
parallel failures, isolated receipts and the explicitly annotated combined
verification receipt are retained in the ignored audit worklog. Raw-source
hashes and historical snapshots are retained privately. No public YC
measurement fixture was added.
