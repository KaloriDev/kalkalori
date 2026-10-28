# Zhao lattice and YC classification audit — 2026-10-27

Branch: `fix/ellipse-yc-shifted-layout`. Base: merged default branch `main`,
`ff4f9137554257af6fd1d7ea57a35f9169c30288`, refreshed from origin before
branch creation. The unrelated serrated-fin commits are excluded. No history
was rewritten. Numerical YC inventories and comparisons remain in ignored
`.kon/_worklogs/ellipse_yc_shifted_20261027/`; no raw YC data are published here.

The classification/geometry correction is implemented. Original-source lattice
verification and corrected SHIFTED prediction remain **UNRESOLVED**. This record
does not claim the source-only provider acceptance criterion is fully closed.

## Physical identities and API

`core.geometry.TubeBankLattice` is independent of tube section, fin type and
correlation. Coordinates use x along outside flow, y transverse to flow,
zero-based downstream row j and transverse site i:

| Identity | Coordinates/meaning |
|---|---|
| INLINE | x=j Pl, y=i Pt |
| CONVENTIONAL_STAGGERED (legacy `staggered`) | x=j Pl, y=i Pt+(j mod 2) Pt/2; all sites in a downstream row share the offset |
| SHIFTED | Per-site (dx,dy) offsets; a whole-row translation is not inferred from a nominal Pt/2 |

SHIFTED without a drawing has `pattern='UNRESOLVED'`; requesting coordinates
raises. Resolved SHIFTED requires explicit SI offsets indexed [row][site], a
pattern and phase. Synthetic per-site offsets in tests do not describe YC or
Zhao. There is no automatic checkerboard/striped selection or equivalent-pitch
substitution. Offset arrays must match physical site counts; coincident centers
are rejected. Shape-specific collision/clearance analysis for custom lattices
is not implemented. SHIFTED currently requires integer sites per row.

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
SHIFTED flow-axis convention and tube-center phase remain unverified. These
are recorded as nominal geometry, never converted to TEMA 60 degrees or
substituted into Zhao's conventional geometry. The user explicitly asked to
leave SHIFTED phase and Zhao source Pt/Pl axis mapping unresolved for now.

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

## Publications and recovery limits

Production equations rely on Zhao, Gu, Gao, Yang (2017), IJHMT 114,
1314–1330, [DOI 10.1016/j.ijheatmasstransfer.2017.06.049](https://doi.org/10.1016/j.ijheatmasstransfer.2017.06.049).
The retained evidence is the previously authorized equation transcription;
the complete original layout figure is not retained locally. The publisher
preview establishes the subject and separation of fin resistance, not the
missing coordinate construction. Full article access failed during this audit.

Zhao and Yang (2016), *Effect of Tube Arrangement on Performance of Rectangular
Finned Elliptical Tube Bundles*, Tongji 44(2), 298–302,
[official record](https://tjxb.ijournals.cn/jtuns/article/abstract/15023),
DOI 10.11908/j.issn.0253-374x.2016.02.020, distinguishes staggered row and
column. The accessible Chinese and English descriptions do not reliably settle
coordinate equivalence (their round-bank comparison wording differs). The PDF
endpoint did not yield a readable full paper. Neither abstract nor naming
establishes a lattice.

Zhao, Wang, Wang, Yang (2019), *Aero-thermal behavior and performance optimization
of rectangular finned elliptical heat exchangers with different tube
arrangements*, IJHMT 133, 1196–1218,
[publisher record](https://www.sciencedirect.com/science/article/abs/pii/S0017931018339711),
DOI 10.1016/j.ijheatmasstransfer.2018.12.138, distinguishes inline, staggered row
and staggered column. No coordinates or coefficient transfer are inferred from
the preview. Its original figures are needed to reconstruct the three lattices.

The different 2019 DOI 10.11949/0438-1157.20190527 in earlier repository source
notes belongs to **Tang, Du, Zeng**, not Zhao's arrangement paper. It is not a
second production Zhao model. Related Tongji-2016 pitch/performance material
is supporting context, not an independent implemented provider.

`ZHAO_ROW_COLUMN_EQUIVALENCE=UNRESOLVED`;
`ZHAO_COLUMN_EQUALS_YC_SHIFTED=UNRESOLVED`. Exact evidence needed next: original
2017 plan-view geometry with flow arrow, Pt/Pl dimensions and axis labels;
2016/2019 dimensioned arrangement figures; YC as-built center coordinates,
flow direction and feature drawing. No source full texts/images are committed.

## Symbol audit

The implementation represents **conventional whole-row half-transverse-pitch
staggering**, major axis parallel to flow. That is an explicit implementation
convention, not a newly verified identification of Zhao's original lattice.
`ZHAO2017_IMPLEMENTED_LATTICE` and `ZHAO2017_SOURCE_LATTICE_GRADE` expose both.
The previous guard was TOO_BROAD: layout mismatch merely produced an optional
geometry-extrapolation warning. Even INLINE could calculate in default mode.
The narrower guard rejects unsupported identities; original-source equivalence
remains unverified. `within_source_domain` is a legacy parameter/envelope check;
neither it nor `strict_source_domain` certifies the missing original figure.

| Quantity / mapping in retained implementation | Evidence grade / finding |
|---|---|
| Source Eqs.18–20 coefficients/exponents | SOURCE_EXACT within retained authorized transcription; unchanged |
| N -> bundle.n_rows (outside-flow rows), not tube passes/count | SOURCE_EXACT transcription; actual source row/column coordinate identification UNRESOLVED |
| Pt -> pitch_transverse (y), Pl -> pitch_longitudinal (x) | ENGINEERING_CONVENTION; consistent with transcription; original figure mapping UNRESOLVED |
| Dl -> twice outer semi-major axis, aligned with x | Transcription geometry / ENGINEERING_CONVENTION orientation; source arrow verification UNRESOLVED |
| Dt -> twice outer semi-minor axis, aligned with y | Transcription geometry / ENGINEERING_CONVENTION orientation |
| Fin streamwise extent -> outer_height=.055; transverse -> outer_width=.026 | Retained geometry convention; source dimensioned orientation UNRESOLVED |
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
This arithmetic does **not** prove the original axes and does not justify
swapping YC's 53.34/30.00 mm. `Pt/Pl SWAP BUG=UNRESOLVED`; no demonstrated
transcription bug, no equation/axis/coefficient change made.

Existing synthetic 8×6 baseline remains h=86.47833865264215 W/(m² K),
dp=372.055553752331 Pa. Equation fixture at N=8, Re=13000 retains
j_eff=.003596209097641707, j_prime=.005572441209273412,
f=.028160566614707627. The earlier approximately 735 vs 404.49 Pa source
normalization discrepancy remains unresolved and is not fitted away.

## Verification

Verification completed on 2026-10-28:

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
