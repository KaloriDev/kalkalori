# STAGGERED round solid-fin thermal model comparison — v0.9.4

2026-10-26. Synthetic public inputs only. The private ECO matrix is stored separately under ignored `.kon`; no private input or result is reproduced here. This comparison adds no coefficients, provider tuning, hydraulic modification or AUTO change.

## Inventory and provenance

Inventory of public `core/heat_transfer` providers, the circular-finned selector and surface dispatch finds exactly two independently comparable STAGGERED dry film models: `BriggsYoung1963Provider` and `FrassEscoaStaggeredSolidFinProvider`. The generic provider/correction protocols are not additional correlations. INLINE ESCOA, Zhao rectangular-finned ellipse physics, bare-tube Zukauskas, phase-change wet operators, fin-conduction efficiency solvers, internal-flow enhancements and private calibration providers are excluded by geometry or output.

Briggs–Young is the existing 1963 triangular-bank correlation, unchanged. ESCOA staggered is implemented from the legal secondary reproduction in Frass/Hofmann/Ponweiser (2015), printed pp17–18, Eqs60–62; the primary 1979 manual was not inspected. [Publisher PDF](https://www.wseas.us/e-library/conferences/2015/Frass/Frass.pdf). The page was visually verified, including the squared row term. See the [provider guide](../staggered_round_finned_thermal.md) for equations, diagnostics and source limitations.

## A. Production-valid pathways and source distinction

Both providers are evaluated through the normal production circular-finned adapter and the identical resistance-network functions, with default guards. Briggs–Young has no source-range warnings at this synthetic geometry or any reported production sweep point. ESCOA passes production geometry guards but **its required source ranges and exact original property convention remain unresolved**; no fully experimentally verified joint ESCOA domain can be claimed. In every table its status is therefore `PRODUCTION / SOURCE-RANGE-UNRESOLVED`, never source-validated.

| Synthetic input | Value |
| --- | --- |
| metal/root diameter m | 0.025 |
| fin diameter m | 0.049 |
| fin thickness root/tip m | 0.0008 |
| fin centre pitch m | 0.0036 |
| Pt m | 0.065 |
| Pl m | 0.0562916512 |
| n_rows | 6 |
| tubes/row | 8 |
| tube length m | 1.2 |
| fin conductivity W/mK | 205 |
| core conductivity W/mK | 45 |
| fin contact efficiency | 0.95 |
| mass flow kg/s | 2.449408 |
| rho kg/m3 | 0.9 |
| mu Pa s | 2.3e-05 |
| k W/mK | 0.036 |
| cp J/kgK | 1080 |
| gas mean K | 500 |
| inside mean K (network-coupled comparison) | 370 |
| inside HTC W/m2K | 1500 |
| inside fouling m2K/W | 0.0015 |
| outside fouling m2K/W | 0.0002 |

The primary raw comparison below fixes the **same fin-base temperature at 450 K**. The existing annular-fin callback computes each model’s own physical mean fin temperature from its physical HTC. Thus it preserves the source mean-fin definition without substituting base temperature into the source ratio. BY does not use fin temperature; its physical fin mean is shown separately. The conductance/resistance columns evaluate the identical physical network at those film coefficients; they are a constitutive comparison, not a claim that both networks at inside bulk 370 K solve to base 450 K. Independently solved exchanger results follow separately.

| Quantity [SI] | Briggs–Young reference | ESCOA staggered |
| --- | --- | --- |
| provider | briggs_young_1963 | frass_escoa_staggered_solid_fin |
| applicability | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| A_min | 0.3328 | 0.3328 |
| u_reference | 8.17777778 | 8.17777778 |
| d_ref | 0.025 | 0.025 |
| Re | 8000 | 8000 |
| Pr | 0.69 | 0.69 |
| S_over_H | 0.233333333 | 0.233333333 |
| S_over_tmean | 3.5 | 3.5 |
| C3 | None | 0.572637256 |
| C5 | None | 0.992914207 |
| Tg_over_Tfin | None | 1.10222023 |
| Dfin_over_dA | 1.96 | 1.96 |
| Pl_over_Pt | 0.866025404 | 0.866025404 |
| Nu | 46.4178246 | 62.0465995 |
| j | 0.00656616461 | 0.00877697714 |
| h_air | 66.8416674 | 89.3471033 |
| T_fin_mean | None | 453.629854 |
| physical_fin_mean | 452.772959 | 453.629854 |
| physical_fin_base | 450 | 450 |
| eta_fin | 0.94454083 | 0.927402919 |
| eta_overall | 0.948433872 | 0.932498984 |
| A | 50.1247391 | 50.1247391 |
| h_eta_A | 3177.6529 | 4176.19692 |
| R_air | 0.000314697682 | 0.00023945231 |
| R_outside_total | 0.000333965368 | 0.000255050409 |
| R_total | 0.000914835944 | 0.000835920985 |
| R_air_over_R_total | 0.343993569 | 0.286453282 |
| UA | 1093.09216 | 1196.28532 |


| Relative to Briggs–Young | Value |
| --- | --- |
| h_ES_over_BY | 1.3366977 |
| Nu_ES_over_BY | 1.3366977 |
| UA_ES_over_BY | 1.09440481 |

Briggs–Young j = .134 Re^−.319 (S/H)^.2 (S/t_mean)^.1134; Nu=j Re Pr^(1/3). ESCOA C3/C5 and Nu are as reproduced in the provider guide. Re, reference area, reference diameter and transport properties are identical between models, so no compensating geometry/velocity mapping explains the model difference. `R_air=1/(h eta_o A)` isolates the ideal convective/fin-conduction resistance. `R_outside_total` also includes the declared contact/root/fouling topology; R_total includes inside and wall terms. Contact is not added twice.

The normal complete network callback instead solves each model’s own fin-base temperature at the same inside/outside bulk states, inside HTC, wall, contact and fouling inputs. Its results are:

| Quantity [SI] | Briggs–Young reference | ESCOA staggered |
| --- | --- | --- |
| provider | briggs_young_1963 | frass_escoa_staggered_solid_fin |
| applicability | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| A_min | 0.3328 | 0.3328 |
| u_reference | 8.17777778 | 8.17777778 |
| d_ref | 0.025 | 0.025 |
| Re | 8000 | 8000 |
| Pr | 0.69 | 0.69 |
| S_over_H | 0.233333333 | 0.233333333 |
| S_over_tmean | 3.5 | 3.5 |
| C3 | None | 0.572637256 |
| C5 | None | 0.992914207 |
| Tg_over_Tfin | None | 1.07425168 |
| Dfin_over_dA | 1.96 | 1.96 |
| Pl_over_Pt | 0.866025404 | 0.866025404 |
| Nu | 46.4178246 | 61.6491937 |
| j | 0.00656616461 | 0.00872076098 |
| h_air | 66.8416674 | 88.7748389 |
| T_fin_mean | None | 465.440278 |
| physical_fin_mean | 457.924824 | 465.440278 |
| physical_fin_base | 455.45436 | 462.752094 |
| eta_fin | 0.94454083 | 0.927829905 |
| eta_overall | 0.948433872 | 0.932895997 |
| A | 50.1247391 | 50.1247391 |
| h_eta_A | 3177.6529 | 4151.2152 |
| R_air | 0.000314697682 | 0.000240893317 |
| R_outside_total | 0.000333965368 | 0.000256561697 |
| R_total | 0.000914835944 | 0.000837432273 |
| R_air_over_R_total | 0.343993569 | 0.287657074 |
| UA | 1093.09216 | 1194.12642 |


| Relative to Briggs–Young | Value |
| --- | --- |
| h_ES_over_BY | 1.32813621 |
| Nu_ES_over_BY | 1.32813621 |
| UA_ES_over_BY | 1.09242977 |

At the common-base reference, the ESCOA temperature factor is 1.02463006; algebraically removing just that factor would leave h_ES/h_BY=1.30456616. Most of the offset therefore comes from the correlation family and geometry factors, with a smaller contribution from fin-temperature treatment. This is a factor decomposition, not a modified production equation. Neither model is treated as ground truth.

## Complete normal Rating and Simulation

Separate fresh normal production runs use this synthetic geometry, constant supplied properties, inside mass flow 1.8 kg/s, cp=4200 J/kgK, inlets 370/500 K and identical wall/contact/fouling inputs. Simulation has configured margin 0.1 and compares actual duty/outlets, not that invariant margin. Rating fixes inside outlet 385 K. The converged bulk/wall states respond to each model; these are distinct from the fixed-state constitutive tables. Both use unchanged Robinson–Briggs hydraulics with normal guards.

| Metric | Briggs–Young | ESCOA staggered |
| --- | --- | --- |
| UA | 937.114854 | 1007.63307 |
| Q_full | 97444.341 | 103154.024 |
| q | 90291.2857 | 95700.4407 |
| T_out_inside | 381.943292 | 382.658788 |
| T_out_outside | 465.868062 | 463.823292 |
| UA_actual | 937.114854 | 1007.63307 |
| UA_required | 1140.24515 | 1140.24515 |
| overdesign | -0.178146164 | -0.11630138 |
| hydraulic provider | robinson_briggs_1966 | robinson_briggs_1966 |


## Flow sensitivity: production paths, network-coupled mean fin temperature


| Flow/reference | Re (both) | h BY | h ES | h ES/BY | UA BY | UA ES | UA ES/BY | BY status | ES status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 4000 | 41.6913394 | 56.9476913 | 1.36593576 | 906.007012 | 1031.72542 | 1.13876097 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| 0.75 | 6000 | 54.9495317 | 73.8250234 | 1.34350596 | 1017.69107 | 1129.68209 | 1.11004422 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| 1 | 8000 | 66.8416674 | 88.7748389 | 1.32813621 | 1093.09216 | 1194.12642 | 1.09242977 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| 1.25 | 10000 | 77.8113578 | 102.441404 | 1.31653535 | 1148.53654 | 1240.77401 | 1.0803087 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| 1.5 | 12000 | 88.0978729 | 115.167272 | 1.30726507 | 1191.54661 | 1276.58215 | 1.07136568 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |

A systematic ESCOA-above-BY offset persists across this range, with a weak Re dependence and no crossing. BY Nu scales as Re^0.681 and ESCOA as Re^0.65 plus the weak coupled temperature term. Agreement is not a single-point coincidence; the tested film offset remains about 31–37%. UA differences shrink as the air-side share of resistance decreases with increasing flow.

## Small geometry sensitivity

Mass flow and supplied gas/inside bulk properties remain fixed. Fin pitch and transverse pitch vary independently in a small one-factor matrix. For transverse-pitch changes, Pl follows the equilateral relation, keeping Pl/Pt fixed, because this is the meaningful overlap with BY. These changes legitimately alter A_min/Re; within each row both models use the same recomputed geometry.

| Parameter | Value m | Re (both) | h ES/BY | Nu ES/BY | UA ES/BY | BY status | ES status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fp | 0.0032 | 8156.86275 | 1.30362476 | 1.30362476 | 1.08246203 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| Fp | 0.0036 | 8000 | 1.32813621 | 1.32813621 | 1.09242977 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| Fp | 0.004 | 7878.78788 | 1.34613217 | 1.34613217 | 1.10082829 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| Pt | 0.06 | 9348.31461 | 1.32000915 | 1.32000915 | 1.08383046 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| Pt | 0.065 | 8000 | 1.32813621 | 1.32813621 | 1.09242977 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |
| Pt | 0.07 | 6991.59664 | 1.33527679 | 1.33527679 | 1.10039629 | PRODUCTION / SOURCE-QUALIFIED | PRODUCTION / SOURCE-RANGE-UNRESOLVED |

Independent finite changes of Pl/Pt are incompatible with BY’s equilateral source restriction. They cannot be presented as a two-model production-valid sweep. The normal guard results below preserve this limitation; no relaxed tolerance is used in this production table.

| Pl/Pt | BY applicability | ESCOA applicability | ESCOA h | BY/ES ratio |
| --- | --- | --- | --- | --- |
| 0.8 | NOT APPLICABLE | PRODUCTION / SOURCE-RANGE-UNRESOLVED | 90.5385177 | NOT APPLICABLE |
| 1 | NOT APPLICABLE | PRODUCTION / SOURCE-RANGE-UNRESOLVED | 85.5341814 | NOT APPLICABLE |

The small valid matrix shows modest geometry dependence of the ratio, not evidence of a universal offset. Unknown ESCOA bounds preclude claiming a fully verified multidimensional overlap. Larger independent pitch-ratio comparisons would require a separately labelled diagnostic extrapolation or another source-validated BY geometry model.

## STAGGERED comparison summary


| Model | Re | h_air | eta_o | UA W/K | Applicability |
| --- | --- | --- | --- | --- | --- |
| briggs_young_1963 | 8000 | 66.8416674 | 0.948433872 | 1093.09216 | PRODUCTION / SOURCE-QUALIFIED |
| frass_escoa_staggered_solid_fin | 8000 | 89.3471033 | 0.932498984 | 1196.28532 | PRODUCTION / SOURCE-RANGE-UNRESOLVED |

ESCOA_STAGGERED / BRIGGS_YOUNG h = 1.3366977

ESCOA_STAGGERED / BRIGGS_YOUNG UA = 1.09440481

MODEL DIFFERENCE MAINLY CAUSED BY = correlation family, with smaller fin-temperature treatment and Re-dependence contributions; geometry/velocity bases match.

BRIGGS-YOUNG vs ESCOA STAGGERED AGREEMENT = POOR for film HTC over this tested matrix (roughly 31–37% systematic difference), despite smaller UA differences. This is a numerical comparison, not an experimental error ranking.

## AUTO decision support

Recommend **D. NO AUTO CHANGE: expose both STAGGERED providers explicitly until further validation**. Source-qualified Briggs–Young has a narrower verified geometry scope; reproduced ESCOA provides useful independent pitch and row terms plus a same-family INLINE comparison, but its complete source range/property convention/accuracy remains unresolved. Numerical consistency and expected layout direction alone do not justify replacing AUTO. This implementation leaves STAGGERED AUTO=Briggs–Young and INLINE AUTO=ESCOA. No domain-based fallback or silent relaxation is introduced.

## Validation record

359 focused public tests passed, including 40 new staggered cases, existing INLINE fixtures, fin geometry/conduction/contact, coupling, Rating, Simulation and hydraulic regressions. Fifteen new equation fixtures were independently calculated with 65-digit Decimal arithmetic. An additional 503 direct/coupled INLINE comparisons against the pre-refactor implementation were numerically identical. Briggs?Young equations, warnings and applicability, and the AUTO branch were preserved by AST comparison. Existing hydraulic source and version files remained unchanged. No commit was made.
