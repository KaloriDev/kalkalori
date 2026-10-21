# Rectangular-finned ellipse: targeted source closure for v0.9.2

Historical decision under the 2026-10-19 free-source-only requirement. The
2026-10-20 explicit authorization of supplied Zhao transcriptions and graded
engineering reconstructions supersedes the implementation stop for that
baseline; it does not change the access findings recorded below. See the
[current implementation evidence](zhao2017_v092.md).

Access checked: **2026-10-19**. Decision: **NOT READY for predictive bank
implementation**. No new empirical model or Rating/Simulation capability is
claimed. The requested final-release acceptance case remains blocked.

## Base and decision boundary

Preflight found a clean tracked tree on `private/universal-v4-finned-ellipse`,
HEAD `124ca51e61e33da3c5353efe38062cdb3f4a84f5`. Work resumed on the existing
`feature/v0.9.2-finned-elliptical-foundation` at the same HEAD; no branch was
created. The complete difference from merged v0.9.1
`36031d2de6a8f2d747c6929b4d386bf68df8cfff` was:

- `7c841e6`: rectangular-fin geometry and physical areas;
- `051cd2d`: deterministic 2D fin efficiency;
- `cafa8ab`: effective-area network and applicability guards;
- `124ca51`: public documentation, tests and version.

The abandoned calibrated-provider series is not in this ancestry. `.kon` is
outside this task and was not edited.

The supplied `Free_Source_Airside_Models_KalKalori_Audit_2026-10-19.md` was a
source map, not implementation authority. Its historical instructions to
start another branch or perform no implementation do not override this task.
Only the free primary sources below were examined for equations. No paid or
private equations, tables, software or provider data were used. Source PDFs,
screenshots and extracted article text are not included in the repository.

## Decisions

| Candidate | Free primary material verified | Thermal decision | Hydraulic decision | Row treatment |
|---|---|---|---|---|
| Tang, Du, Zeng (2019) | Full publisher PDF, including equations | NOT READY: equation recovered; geometry/flow/property definitions incomplete for bank dispatch | NOT READY: no air-side predictive pressure equation in this paper | Fixed N=2 established; no rule for other N |
| Zhao, Yang (2016) | Official abstract; full-text retrieval not verified | NOT READY | NOT READY | Not established from accessible material |
| Ribeiro Jr., Andrade (2004) | Full SciELO paper | NOT READY: equation recovered; flow construction and domain incomplete | NOT READY: no bank pressure model | Source N/domain not specified |
| Ribeiro (2001) | Official university catalogue confirms thesis; full thesis not located | NOT READY | NOT READY | Not established |
| Chiu, Lin, Jang (2007), audit F04 | Free full conference paper, checked for hydraulic closure | NOT READY for this surface | NOT READY: definition and plots, no predictive correlation | No transferable row law |

No candidate qualifies as IMPLEMENT, IMPLEMENT AS FIXED MODEL or IMPLEMENT AS
FAMILY MODEL. Knowing N=2 alone does not close the other required inputs.

## A. Tang, Du, Zeng (2019)

[Publisher record](https://hgxb.cip.com.cn/EN/10.11949/0438-1157.20190527),
[free full PDF](https://hgxb.cip.com.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=20741).
DOI: 10.11949/0438-1157.20190527. Download verified as an eight-page PDF:
SHA256 `f84f28367547b935fe6854e377bd126d6a1254ec975641ef0f9c346f7b752607`.

Printed pp.140–142, Eqs.(1), (17)–(19):

```text
Dc = 4*a*b / sqrt(2*(a*a + b*b))
u_max = u_fr / sigma
Re_Dc = rho_air*u_max*Dc/mu_air
Nu = h_o*Dc/lambda_air
Nu = 4.39 * Re_Dc**0.311 * (theta_deg/90)**0.211
4000 <= Re_Dc <= 7500; 30 <= theta_deg <= 90; N_rows = 2
```

Here a/b are semi-axes. Table 1: full axes 36/14 mm; Pt/Pl=26.7/60 mm;
fin thickness .35 mm; reported fin spacing Fs=2.5 mm. Eqs.(6), (9)–(16)
separate h_o from surface/fin efficiency: h_o is physical film h, not UA or
efficiency-weighted h. Eq.(19) has no independent Pr factor; no general-fluid
extension is justified.

Missing: source construction/value of sigma; independent rectangular fin
width/height and confirmation of the single-aperture individual-fin topology;
explicit air-property evaluation temperature. Fs is not silently identified
with our axial pitch. Figure 4 is not a dimensioned fin drawing. No air-side
pressure correlation is supplied; Eq.(8) is water-side Darcy friction.
**NOT READY**, including as a fixed N=2 bank model.

## B. Zhao, Yang (2016), tube arrangement

[Official record](https://tjxb.ijournals.cn/jtuns/article/abstract/15023),
DOI: 10.11908/j.issn.0253-374x.2016.02.020, pp.298–302.
The abstract compares 24 staggered-row/column structures and attributes a
substantial pressure loss to the inlet. It provides no complete correlation,
coefficient set, velocity/pressure normalization or row applicability.

The linked [PDF endpoint](https://tjxb.ijournals.cn/jtuns/article/pdf/15023)
returned a 58-byte HTML guard script, not a PDF.
The [HTML endpoint](https://tjxb.ijournals.cn/jtuns/article/html/15023)
redirected to the abstract. The legacy create_pdf route also returned the
guard script. These are retrieval limitations, not proof that the full paper
is paywalled or lacks a model. **NOT READY** until a legal free full text can
be checked. No equations or applicability ranges are inferred from its abstract.

## C. Ribeiro Jr., Andrade (2004)

[SciELO full text](https://www.scielo.br/j/bjce/a/bKpbLp6QDGsbmkQQxxF6tnw/?lang=en),
[free PDF](https://www.scielo.br/j/bjce/a/bKpbLp6QDGsbmkQQxxF6tnw/?format=pdf&lang=en).
DOI: 10.1590/S0104-66322004000200025.
PDF SHA256 `acc224268abfa747d9467fea415c9e88637827df3c684138d45a4875e4637e8c`.
Printed pp.348–350, 354; Eqs.(21), (22), nomenclature:

```text
Nu = 0.06 * Re**0.65 * Pr**0.36 * S1*D2_ext/(S2*D1_ext)
Deq = 4*A_min*[n_transv*B + (n_transv+1)*(S2-B)]/A_T
Re = rho*v_min*Deq/mu; Pr = cp*mu/k; Nu = h_sl*Deq/k
```

D1 is the full minor axis, D2 the full major axis; S1/S2 are transverse/
longitudinal pitches. Properties use inlet temperature; velocity uses A_min.
h_sl is physical film h, separate from fin efficiency in Eq.(2).
Demonstrated geometry: 36×14 mm ellipse, 55×26×.23 mm rectangular fins,
triangular layout. Table 1 gives four pitch/fin-density configurations.

Missing: a reproducible A_min construction, configuration n_transv/N values,
and the correlation's Re/Pr/geometric/row validity bounds. Eq.(22) itself is
available; its complete bank inputs and domain are not. Total tube counts do
not determine row arrangement. No hydraulic equation or row rule closes here.
**NOT READY** for either fixed or family predictive dispatch. No substitute
diameter, inferred N or guessed free area is permitted.

## D. Ribeiro (2001), institutional follow-up

The [official UFMG master's catalogue](https://ppgeq.deq.ufmg.br/wp-content/uploads/2023/10/Egressos-Mestrado.pdf),
PDF p.4, entry 33, confirms Cláudio Patrício Ribeiro Júnior, adviser Maria
Helena Caño de Andrade, defence 2001-02-19, thesis *Modelagem de Sistemas de
Troca Térmica a Placas e Simulação do processo Industrial de Produção de
Leite em Pó da Embaré*.

The official [repository search API](https://repositorio.ufmg.br/server/api/discover/search/objects)
was queried by the exact title fragment and author, including accented
spelling. No matching full thesis was located: the author search returned an
unrelated Kühni-column thesis. A catalogue is not a source for missing
equations. **NOT READY**; availability elsewhere is not ruled out.

## E. Chiu, Lin, Jang (2007), specific hydraulic follow-up

[Free primary conference paper](https://www.irbnet.de/daten/iconda/CIB7212.pdf),
*Numerical and Experimental Analysis of Elliptic Finned-Tube Heat Exchangers
under Misted Conditions*, Clima 2007. Audit F04 was checked specifically for
an air-side pressure model. Eq.(10) defines a pressure-derived f; Figures 6–8
show pressure/HTC results. They do not provide a closed predictive f(Re,
geometry,N) correlation with coefficients and validity bounds. Governing CFD
equations do not constitute that empirical bank model. No pressure curve was
digitized or fitted, and no coefficients were transferred across geometries.
**NOT READY** for closing the target hydraulics.

## Engine consequence and stop decision

The normal public path evaluates outside hydraulics unconditionally when
outside flow/properties are present: `BareTubeHeatExchanger.solve` calls
`evaluate_outside_hydraulics`; Rating and Simulation use that normal path.
There is no established thermal-only rectangular-finned-ellipse result contract
that would permit skipping the hydraulic model. The NaN branch for absent
outside inputs is not such an option for a normal two-stream calculation.

Therefore neither an unknown pressure coefficient nor zero/NaN pressure is
inserted to unlock a nominal acceptance case. The existing outside-bank guards
remain appropriate. Source-specific dispatch can replace them only after a
complete applicable model exists. No generic calibrated-provider architecture
or private hook is restored.

The request's stop conditions apply: implementing from the inspected sources
would require a guessed free area/sigma, unsupported geometry mapping, missing
applicability limits or an undefined hydraulic prediction. The source-defined
N=2 restriction is useful recovered evidence, not a universal row correction.
No new HTC/hydraulic/row classes or empirical tests are added for an unclosed
model. The existing public conduction/network capability remains usable.

Next evidence needed is narrow: Tang's dimensioned fin/topology and free-area
definition plus air-property convention; a verified free Tongji full text;
or Ribeiro's missing flow construction and original validity/N specification.
Any proposed hydraulics must additionally define velocity, density, pressure
type, row/bank scope, inlet/exit and acceleration treatment. No author was
contacted, and no paid/private source was consulted.

## Validation and release status

Focused verification on the unchanged public physics:

```text
py -3.11 -m pytest core/tests/rectangular_finned_geometry_test.py core/tests/thin_plate_fin_test.py core/tests/rectangular_finned_network_test.py -q
45 passed in 5.27s
```

Zero failures, skips, xfails or warnings were reported in that run. It covers
geometry, conduction, physical-h/effective-area coupling and explicit refusal
before unrelated predictive fallback. It is not validation of a new empirical
model. The full public suite was not rerun: implementation stopped at source
closure and executable code is unchanged. The earlier 1930-pass run belongs
to the [first-series validation](finned_foundation_v092.md), not this series.

**Final v0.9.2 release acceptance is not achieved.** No source-supported normal
Rating/Simulation operating point, duty, outlet temperatures, predictive h,
UA or bank pressure drop can be reported from this second series. Existing
prescribed-film examples must not be relabeled as predictive acceptance.
