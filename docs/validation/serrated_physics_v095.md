# v0.9.5 Phase B validation and limitations

Development branch: `feature/v0.9.5-serrated-fins`. Version **0.9.5**, unreleased.
Phase A commit: `1be1443f5495f635adfc2caec6507bfcddaf0bcc`; history preserved.
Source pack: all ten supplied PDFs inspected privately. Equation pages were
rendered and checked directly before implementation; transcriptions were not
accepted solely from the request or a previous audit. The
[source mapping](serrated_source_mapping_v095.md) records definitions, units,
reference areas/velocities/diameters, property/row/layout contracts and access.

## Implemented scope

Four separate components are implemented: Holfeld Eq.5-9 source-equivalent
flow area, Hashizume's two-region serrated theory with the 2002 STAGGERED
experimental correction, Holfeld Eq.6-6 physical-film HTC, and Holfeld
Eqs.6-10/4-8 native Eu bank-section loss. Complete explicit STAGGERED execution
works through `Holfeld2016SerratedStaggeredModel`; no AUTO registration occurs.
Normal Rating/Simulation and the resistance network retain those providers
and apply eta once.

Initial physical geometry is HF-welded I-foot, round tube, untwisted constant
thickness, ideal weld contact, zero removed kerf and `D_root=core.D_o`.
Physical rectangular strip width is a separate required input, not Phase A's
tip arc. Native source surface is distinct from the unchanged geometric surface.
Dry-air calibration is retained; wet/condensing/phase-capable provider routes,
fouling, INLINE complete execution, SHIFTED and solid-fin substitution reject.

The complete interval is `5000 <= Re_do < 50000`, `Nl >= 4` (no published upper
row bound), plus **all** active component geometry/area/conduction bounds.
Notable intersections include `1.80<=df/do<=2.15`, `.50<=hs/hf<=.86`,
`hs/ws<=2.60`, `m*hf<=2.5`, `Pt/Pl<2` and arrangement angle <45 degrees.
No generic row correction is added. Lower Re is unreported for the individual
Holfeld regressions but the Hashizume correction supplies the complete lower
bound. Other unreported bounds remain diagnosed; the supported envelope is
not inferred from the widest component.

Holfeld film h is before eta: Eqs.4-4 to 4-6 use eta in experimental resistance
reduction, and Eq.6-6 returns `Nu=h do/k`. The regression combines differing
literature reduction conventions; Holfeld's own serrated measurements use
Weierman, not Hashizume. That calibration difference is disclosed without
re-fitting. Hashizume mean corrected eta closes conductance but supplies no
corrected tip temperature; tip/skin extrema remain unavailable.

Pressure scope is the adiabatic measured bundle-section pressure difference
treated as friction by the source after neglecting about 1% acceleration.
Exact tap distances and an empty-channel subtraction are unreported.
Native `Eu=2 dp/(Nl rho Vmax^2)` is inverted directly, with mean-temperature
transport and endpoint-mean density. Separate signed acceleration uses face
sections. No RB/Frass/Darcy/Fanning/Genic normalization or separate bank
inlet/exit coefficient is introduced.

## Independent numerical checks

The new public module `core/tests/serrated_source_physics_test.py` contains
138 tests. It distinguishes source-measured facts, source-equation arithmetic
and digitized theoretical graph estimates. Decimal reference arithmetic uses
the original unexpanded printed Ar form. Twelve independent boundary-value
solutions integrate the two-region conduction ODE instead of the production
scaled-Bessel implementation. Three Fig.6 (1999) eta estimates use a .02
digitization tolerance and are explicitly theoretical, with no Re or empirical
correction. They are not labelled measured eta.

Three low/mid/high HTC and three DP measured facts from Holfeld Geometry 1,
Appendix II pp.180–181, are checked independently of arithmetic fixtures.
The results below include printed mass-flow/geometry rounding and use the
source k/mu polynomials; the HTC cp is a declared constant approximation.
HTC pressure is not tabulated in the source; the compared Re/reduced-Nu/h
quantities do not depend on the chosen density. No heat-duty reconstruction
confuses 4.5 flow cells with the number of active heated tubes.

| Measured source Re | Predicted/measured reduced Nu | Predicted/measured h |
| ---: | ---: | ---: |
| 8312 | .7601 | .7600 |
| 19196 | .8117 | .8115 |
| 33078 | .8654 | .8642 |

| Measured source Re | Predicted/measured Eu | Predicted/measured dp |
| ---: | ---: | ---: |
| 9257 | .8453 | .8245 |
| 20380 | .9033 | .9108 |
| 35127 | .9240 | .9470 |

The low-Re HTC residual exceeds 21%. Source-reported 95% data coverage within
21% (HTC) and 34% (DP) is **SOURCE_REPORTED_REGRESSION_COVERAGE**, not an
uncertainty guarantee or a claim that every selected point satisfies that band.
Tests retain these residuals and exact arithmetic checks separately; no
coefficient, area mapping or correction was tuned to measured values.

The audit also retains source disagreements: Holfeld's p.89 prose flow areas
have a factor-ten discrepancy with Eq.5-9/test dimensions; its prose/table
height-ratio limits differ; the original 2000 Hashizume main correction prints
a different height ratio from its appendix and the 2002 restatement. See the
mapping for the explicit decisions. None were reconciled silently.

## Kawaguchi comparison

Both supplied originals were inspected. They remain **validation-only**;
no optional production provider was registered. The private equation comparison
uses 2005 Eqs. 2/3, six-row coefficients, native volume-equivalent diameter and
the declared geometric integration in the source mapping. It is not matched
experimental validation. Geometry [mm]: do 25.3, hf 13, hs 7, tf .9, pf 4.5,
ws 4.5, Pt 58.5, Pl 51; six rows. The native strip-volume idealization gives
dv 31.6103 mm,
hf/dv .41126 and clear-spacing/pitch .80. Holfeld/Hashizume active guards and
Kawaguchi's published scalar bounds pass; Kawaguchi segment-shape bounds remain
unreported.

| Re on bare do | Kawaguchi Re on dv | Holfeld h / Kawaguchi h |
| ---: | ---: | ---: |
| 10000 | 12494.2 | .7878 |
| 20000 | 24988.4 | .7279 |
| 30000 | 37482.5 | .6148 |

These differences are retained, with no tuning. They quantify disagreement
between the equations under the stated volume idealization; they do not
establish bias against independently matched measured cases.

**DP comparison remains blocked.** The 2003 source does not close the common
serrated strip dimensions and wetted-perimeter mapping needed to translate
its hydraulic-diameter Reynolds number to Holfeld's do-based Reynolds number.
Fin-spacing labels in the plots are not hydraulic diameters. The supplied
2005 HTC measured cases likewise do not supply a strict matched fallback:
SR210's hs/hf is .4846, below Holfeld's .50 bound, and strip widths are missing.
No ratio is manufactured for those incompatible/undefined conditions. This
unfulfilled independent hydraulic validation is a release limitation.

## Representative industrial smoke case

This is execution verification, not an experiment. Geometry uses do 31.75 mm,
df 67.75 mm, hf 18 mm, hs 11 mm, tf 1 mm, 277 turns/m, ws 4.5 mm, zero kerf,
32 Phase A segments, Pt 69.75 mm, Pl 60.4053 mm, eight by eight tubes, length 2 m.
Inside is a declared single-phase constant-property liquid at 350 K, 4 kg/s;
outside is dry air at 450 K, 7.41696 kg/s, 101325 Pa. The dry-air provider uses
its existing ideal-gas/Sutherland engineering property fallback explicitly.
Rating specifies inside outlet 355 K; Simulation predicts the installed-bank
response. Every active domain label is PASS.

| Quantity | Rating | Simulation |
| --- | ---: | ---: |
| Source A_min [m2] | .448448 | .448448 |
| Face area [m2] | 1.116 | 1.116 |
| Thermal Vmax [m/s] | 19.9679 | 19.9679 |
| Thermal Re_do | 21957.6 | 21957.6 |
| Film h [W/(m2 K)] | 126.193 | 126.193 |
| Eta theoretical / corrected | .60342 / .54489 | .60342 / .54489 |
| Outside effective conductance [W/K] | 15092.5 | 15092.5 |
| UA [W/K] | 5621.23 | 5621.23 |
| DP reference Re_do | 21309.3 | 21957.9 |
| DP reference Vmax [m/s] | 20.8345 | 19.9050 |
| Native Eu | 1.31693 | 1.30918 |
| Bundle-section drag [Pa] | 1815.19 | 1723.99 |
| Signed acceleration [Pa] | -1.32173 | -5.96807 |
| Total pressure change [Pa] | 1813.86 | 1718.02 |

The existing engine distinguishes its converged representative thermal state
from the closed Rating/final Simulation hydraulic endpoint states. Consequently
those reference Re/velocities are reported separately. Source-reference gross
surface is 211.7812 m2, while Phase A geometric surface remains 241.2349 m2.
HT_DOMAIN, DP_DOMAIN, ETA_DOMAIN and FLOW_AREA_DOMAIN all pass. The Phase A
finite-kerf reference example remains unchanged at volume .0003112338269 m3/m,
mass 2.443185541 kg/m, fin area 1.173311399 m2/m, primary area .06483461839 m2/m,
total 1.238146017 m2/m; that example does not itself acquire a finite-kerf
source-physics mapping.

## Public suite and package

The complete public collection contains **2590 tests** (2452 baseline +138).
Final result on 2026-10-27: **2590 passed, 0 failed, 0 skipped, 0 xfailed,
0 xpassed**. All four regular worker processes and the final serial process
returned exit code zero.

An earlier tool-attached process ended without a final report. Only its 44
passes from fully completed modules were retained, after verifying identical
core/test source hashes and collection. The interrupted module was retried in
full. The persistent run executed the remaining 2544 regular cases and both
original default-deadline inverse-area cases, the latter serially after all
regular workers. Their original deadlines and solver tolerances were unchanged.
The final reconciliation verifies disjoint, complete assignment of all 2590
node IDs, identical collections, no missing reports, and unchanged core/test
source hashes. The two serial cases both pass (411.49 seconds combined).

Layered checks completed: the original 545-test target selection (including
131 initial Phase B tests), both ESCOA modules (83 tests), and the final
138-test Phase B module pass. The unchanged Phase A geometry/dispatch tests
(124 tests) pass. After separating the native Eu diagnostic from the generic
friction-factor alias, the affected 222-test selection also passes. The
complete result above covers their union and the rest of the public suite.

The final wheel check is **PASS**: `kalkalori-0.9.5-py3-none-any.whl`,
896919 bytes, 253 entries, SHA-256
`6ac00bdf7746fcfae72ebaddc11d98561de4442e7141c85f547c68e588713f94`.
All packaged core Python files match the working sources. The archive contains
no PDFs, ZIPs, screenshots, private research inputs or helper reports. A
wheel-first import outside the repository passes with the existing Python 3.11
dependencies, including user-site packages (`-E -P`); this is not a fresh
dependency-environment installation test. Private research PDFs, extracted
text, screenshots, reports and helper scripts remain ignored. Version stays
0.9.5; no tag, publication or push.

Readiness is **CONDITIONAL**, pending independent matched hydraulic validation.
Complete regression and package verification pass. The explicit source model is
available within its diagnosed conventions; it is not promoted to AUTO or a
fully independently validated serrated release.
