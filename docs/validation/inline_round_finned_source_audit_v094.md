# v0.9.4 forensic reconstruction: INLINE solid-finned round tubes

Research date: 2026-10-24. Base: `f5a3cdf`, v0.9.3.
Branch: `feature/v0.9.4-inline-round-finned`.

**Historical source-reconstruction snapshot, completed before implementation.**
The subsequent authorized [thermal implementation](../inline_round_finned_thermal.md)
uses this report as its source basis. Runtime/validation statements in section 9
describe the reconstruction stage, not the current implementation state.
The user's revised policy accepts legally accessible secondary reproductions.
Unavailable original ESCOA/ITE reports are future verification tasks, not an
automatic veto. Unknown ranges remain unknown. This report supersedes the
earlier primary-source-only gate and its provisional Reynolds conversion.

The preferred thermal development route is **ESCOA INLINE solid-fin Eqs. (96)-(98)
as reproduced by Frass**, because row and pitch dependence are explicit.
The expression is reproducible; the original experimental envelope and exact
property convention are not fully recovered. Eq. (100) is a separate candidate
with an explicit Re interval, but its regression row basis is **UNRESOLVED**.
An ESCOA row multiplier must not be attached to Eq. (100).

## 1. Source register and access classification

Frass page references use the bottom-of-page chapter numbers, not the inconsistent
contents/figure-list pagination. Add 20 for the one-based PDF page. For Hofmann's
2009 thesis add 21. No third-party full text, figures or tables are redistributed.

| ID | Source and inspected locations | Access/evidence classification |
| --- | --- | --- |
| F15 | F. Frass, edited/translated by R. Hofmann and K. Ponweiser, *Principles of Finned-Tube Heat Exchanger Design for Enhanced Heat Transfer*, 2nd ed., WSEAS Press, 2015, ISBN 978-960-474-389-6. [Publisher full PDF](https://www.wseas.us/e-library/conferences/2015/Frass/Frass.pdf). | Full 152-page text reviewed, including rendered figures. Primary author account of ITE measurements; legal secondary reproduction of other authors' models. |
| H09 | R. Hofmann, *Experimental and Numerical Gas-Side Performance Evaluation of Finned-Tube Heat Exchangers*, TU Wien dissertation, 2009. [Institutional full PDF](https://repositum.tuwien.at/bitstream/20.500.12708/12978/2/Hofmann%20Rene%20-%202009%20-%20Experimental%20and%20numerical%20gas-side%20performance%20evaluation...pdf). Sec. 4.1.2 pp. 85-88; p. 109; Appendix C p. 153; bibliography pp. 154-159. | Full text reviewed. Primary for new experiments, secondary for earlier Frasz measurements/1992 paper. The literal ellipsis belongs to the URL. |
| H07 | Hofmann, Frasz, Ponweiser, *Heat Transfer and Pressure Drop Performance Comparison of Finned-Tube Bundles in Forced Convection*, WSEAS Transactions on Heat and Mass Transfer 2(4), 2007, 72-88. [Publisher full PDF](https://wseas.us/e-library/transactions/heat/2007/25-669N.pdf), Table 1 p. 75, pp. 76-79. | Full text reviewed. Geometry cross-confirms H09; related serrated/staggered ESCOA reproduction only partly covers the present target. |
| H08 | Hofmann, Frasz, Ponweiser, *Experimental Heat Transfer Investigation of Tube Row Effects at Air Side Heat Exchanger with Serrated Finned-Tubes*, Thermal Engineering and Environment, 2008, 193-201. [Publisher PDF](https://www.wseas.us/e-library/conferences/2008/rhodes/hte/hte29.pdf). | Legal derivative row-study evidence; staggered U-segmented fins, not INLINE solid-fin calibration. |
| W99 | M. Wetter, *Simulation Model: Finned Water-to-Air Coil without Condensation*, LBNL-42355, January 1999 (internal title dated November 1998). [Institutional full PDF](https://simulationresearch.lbl.gov/dirpubs/42355.pdf), pp. 14, 19. | Full text reviewed. Secondary Yudin/Stasiulevicius chain; different characteristic length. |
| P20 | M. Petrik, G. L. Szepesi, *Experimental and Numerical Investigation of the Air Side Heat Transfer of a Finned Tubes Heat Exchanger*, Processes 8(7), 773, 2020. [Publisher article](https://www.mdpi.com/2227-9717/8/7/773), Eqs. (15)-(20). | Open primary experiment with secondary Schmidt reproduction. Its new geometry is not Schmidt's original matrix. |
| MG00 | H. Martin, V. Gnielinski, *Calculation of heat transfer from pressure drop in tube bundles*, ETSC 2000, 1155-1160. [KIT full text](https://publikationen.bibliothek.kit.edu/1000034867/2579184), Table 1/nomenclature. | Legal primary paper reusing Scholz data; supports plain-tube classification. Not Scholz's original full text. |
| S68 | F. Scholz, *Einfluß der Rohrreihenzahl auf den Druckverlust und Wärmeübergang von Rohrbündeln bei hohen Reynolds-Zahlen*, Chemie Ingenieur Technik 40(20), 1968, 988-995. [Publisher DOI](https://doi.org/10.1002/cite.330402003). | Abstract/metadata evidence; original full text not obtained. Direct publisher fetch was blocked; indexed records and derivative bibliographies were used. F15's page 983 differs from the DOI's bibliographic pagination. |
| K96 | S. P. Kearney, A. M. Jacobi, *Local Convective Behavior and Fin Efficiency in Shallow Banks of In-Line and Staggered, Annularly Finned Tubes*, J. Heat Transfer 118(2), 1996, 317-326. [University abstract](https://experts.illinois.edu/en/publications/local-convective-behavior-and-fin-efficiency-in-shallow-banks-of-/), [DOI](https://doi.org/10.1115/1.2825847). | Primary abstract, not full geometry/data; independent evidence, not Eq. (100) provenance. |
| K95 | Kearney/Jacobi, *Local and Average Heat Transfer and Pressure Drop Characteristics of Annularly Finned Tube Heat Exchangers*, ACRC TR-69, January 1995. [Institutional record](https://www.ideals.illinois.edu/items/11008), [handle](https://hdl.handle.net/2142/10990). | Official indexed abstract/metadata located. Full PDF requests returned HTTP 403; full text not reviewed. |
| L16 | H. Lim, *An Investigation into the Level of Fidelity Required to Model Heat Exchangers of a Heat Recovery Steam Generator*, AUT master's thesis, December 2016. [Institutional full text](https://openrepository.aut.ac.nz/server/api/core/bitstreams/f2f59e6e-35fb-4800-bc87-df84a47560a3/content), pp. 27-28, Appendix A pp. 82-88. | Legal secondary reproduction of the **1993 TURB-X HF revision**, with bulk-property definitions; coefficients differ from the 1979 solid-fin target. |
| ESG6 | Yarden, Aoki, Taborek, *Isothermal Pressure Drop Tests on 2-inch Serrated and Solid Finned Tubes*, HTRI ESG-6, 1978. [Public primary abstract](https://www.htri.net/esg-6). | Abstract only; insufficient to reconstruct the full hydraulic correlation. |
| HTRI-TOC | [HTRI Design Manual contents](https://horizons.htri.net/design-manual). | ONLY CITATION/contents evidence for INLINE/ESCOA sections; no equations recovered. |

F15 full-file SHA-256:
`0be9db984191438c606d0b522ccd1fc9c4bd491bc22d04da4bede3e41f8aa072`.
The similarly named WSEAS PDF under `main/books/2015/Frass/` is a short preview.
Scribd ESCOA uploads and unofficial commercial-book scans are excluded provenance.

## 2. Citation-chain reconstruction

### [1] Schmidt 1963

Path: **Schmidt 1963 -> F15 pp. 12-13, Eq. (36), and p. 38, Eq. (93)**.
With tube-OD dimensionless groups:

```text
Nu_0 = C Re_dA^0.625 Pr^(1/3) (A_tot/A_R)^(-0.375)
C_INLINE = 0.30; C_STAGGERED = 0.45
```

Path: **Schmidt -> P20 Eqs. (15)-(18) -> algebraic cross-check**. P20 uses
`d_F=d_A*(A_tot/A_R)` and `Nu_F=C*Re_F^0.625*Pr^(1/3)`. Converting Nu and Re
back to `d_A` gives the area exponent `0.625-1=-0.375`. This checks the equation,
not the original experimental range. P20's modified fit and experiment must not
be relabeled Schmidt's domain.

F15 describes an aggregate fit mostly to annular fins and discusses a possible
increase for welded spiral fins; that is not an independently selected correction.
Original row count, Re bounds, fin dimensions/ranges and pitch envelope remain
UNKNOWN. H09 [56] confirms the first installment at 98-102; later bibliographies
also identify 370-378. F15's abbreviated page 88 is not a reliable locator.
Schmidt's 1963 film-HTC expression is distinct from Schmidt fin-efficiency methods.

### [3] Stasiulevicius and Skrinska, including predecessors

Path: **Stasiulevicius/Skrinska -> F15 Sec. 3.5.3 pp. 58-59** recovers a
fourth-row asymptotic claim for INLINE flow. The explanation starts with local
downstream-row wake behavior, then discusses bundle row-count dependence. The
original local-to-bank averaging data and a multiplier attached to Eq. (100) are
not supplied. The fifth-row and pitch-ratio observations there are attributed
to ESCOA; they are not an ITE test matrix.

The English book has 1987/1988 dates in different records; H09 [58] cites
Hemisphere 1988, edited by Zukauskas/Hewitt. Earlier Vilnius, Mintis 1974 editions
are traceable bibliographically. An [institutional Lithuanian bibliography](https://gs.elaba.lt/object/elaba%3A5656516/5656516.pdf)
provides a catalogue route, not an inspectable INLINE experimental matrix.
Predecessor search leads include Skrinska/Stasiulevicius on nonuniform fin heat
transfer (1965), Skrinska/Zukauskas/Stasiulevicius on local spiral-fin transfer
(1964), and Stasiulevicius/Skrinska/Survila/Samoska on transfer and drag (1971).
Original full texts/numerical INLINE matrices were not recovered; conflicting
transliterations and pagination are not converted into technical facts.

Path: **Yudin/Tokhtarova and Yudin et al. -> W99 p. 14, bibliography p. 19**
recovers a finned-tube expression with a 40,000 threshold, but its Reynolds
length is **fin pitch**, not tube OD. It cannot confirm Eq. (100)'s interval or
rows. W99 identifies the 1968 TsKTI vol. 82 correlation of spiral/circumferential
fin data. These are library leads, not recovered ITE provenance.

H09's introductory large Re range for Stasiulevicius concerns staggered data.
F15 Fig. 58 explicitly lacks the zero-angle INLINE point (about 35-57 degrees
shown). F15 p. 105 describes HEDH hydraulic ancestry as annularly milled fins
of trapezoidal section and cautions against equating them to welded spiral fins.
None of this closes the INLINE spiral-fin geometry envelope.

### [6] Frasz/Frass and Linzer 1992: high-priority result

The author appears as Frasz and Frass. Both spellings and German/transliterated
titles were searched. H09 [15] and H07 [27] confirm **BWK 44, issue 7/8, 333-336
(1992)**. No verified legal open full text of that paper was recovered.

Two different derivative paths were found:

1. **1992 -> F15 Sec. 3.5.1 pp. 54-55, Fig. 43/Table 2 -> H09 Sec. 4.1.2
   pp. 85-88.** F15 specifically cites Fig. 6 of [6] for measurements at
   1/2/4/6 rows relative to eight. The section/figure are **staggered**. H09
   p. 87 describes the matching Frasz-1 model as staggered, `d_A=38 mm`, with
   the same pitches as that thesis's experiment (`Pt=85`, `Pl=79 mm`). This
   closes a staggered row-study path, not the INLINE regression membership.
   H09's U-segmented fin dimensions cannot be assigned to the original tubes.
2. **1992 -> F15 p. 49, Sec. 3.3.** Frass cites [6] for measured INLINE/staggered
   transfer differences smaller than Schmidt's factor 1.5. Thus the 1992 work
   is relevant to comparing layouts. The citation does not establish that
   Eq. (100), coefficient 0.1306, geometry factors or row basis appeared there.

The first publication date of Eq. (100) remains unresolved. Its complete form is
inspectable in F15. H09 cites the 2008 first monograph edition, but a bibliography
entry cannot prove that every equation in the 2015 edition was already present.

### [12] ESCOA Solidfin HF 1979

Path: **ESCOA 1979 -> F15 p. 39 Eqs. (96)-(98), definitions p. 18** is a
**COMPLETE LEGAL SECONDARY REPRODUCTION of the algebraic INLINE solid-fin
thermal model**, including physical row number and pitch ratio. Reconstruction
of the full original method is **PARTIAL**: original numeric ranges, deviation
envelope and a model-specific property-state instruction remain unavailable.
No primary-manual access is claimed.

H07/H08: **PARTIAL REPRODUCTION** for this target, because the reproduced
models concern serrated/staggered fins. HTRI contents: **ONLY CITATION**.
L16: substantial legal reproduction with `Re=G_n*d_A/mu_bulk`, bulk `cp` and
conductivity, and film HTC before fin efficiency, but explicitly the **1993
TURB-X revision**. Its INLINE `C1`, `C3` and temperature exponent differ from
F15's old solid-fin form. This corroborates family practice, not the exact 1979
property convention. Neither its coefficients nor its quoted 1993 tube-size
range can silently complete the 1979 model.

### [16] Scholz 1968

The publisher abstract identifies one staggered/two INLINE bundles and Re about
`10^4...10^6`; it does not supply a finned-tube matrix. Path: **Scholz -> MG00
Table 1 dataset 120 -> plain-tube modeling/nomenclature** supports
**STRONG_RECONSTRUCTION: smooth/plain tube-bank evidence**. Original surface
construction was not inspected, so this is not SOURCE_EXACT. F15's citation of
a high-Re pressure-drop discontinuity does not make it a finned INLINE row
correction. Exclude it from thermal row treatment without explicit finned evidence.

### [22] Kearney and Jacobi 1996

Path: **K96 university abstract -> independent shallow-bank evidence**: INLINE
and staggered annular fins, second-row local data, Re `5000...28000` based on
**hydraulic diameter and minimum-area velocity**. K95's indexed official
abstract describes one-row ordinary/baffled and two-row INLINE/staggered
experiments over `5000...30000`. These different intervals are not Eq. (100)'s
tube-OD interval.

Path: **K96 -> F15 p. 22** adds fixed geometry/pitches and touching fin edges.
Numerical geometry was not recovered from primary full text. The K96 abstract
reports comparable INLINE/staggered average transfer/pressure behavior, but
there are no recovered numerical bank fixtures. This is independent qualitative
validation, not Eq. (100) provenance. KalKalori requires positive fin-tip gaps;
these touching-fin cases cannot be run unchanged through its geometry guard.

## 3. Eq. (100): equation, output and definitions

F15 pp. 47-48, Sec. 3.2.3; `Pt=t_q`, `Pl=t_l`, `Fp=t_R`, `tf=s_R`:

```text
Nu_0 = 0.1306 Re_dA^0.75 Pr^(1/3) (A_tot/A_R)^(-0.375)
       * ((Pt-d_A)/(2*h))^0.21 * (Pl/Pt)^0.4
10,000 < Re_dA < 40,000                 # open source interval
alpha_0 = Nu_0 * lambda_gas / d_A
```

| Quantity | Definition/location | Closure |
| --- | --- | --- |
| Construction/layout | INLINE spiral-finned round tubes, p. 47 | Exact; arbitrary taper/root layers/separate annular manufacture not experimentally equated |
| `alpha_0` | Physical convective film HTC **before** surface efficiency, pp. 10-12 Eqs. (33)-(35) | Closed |
| `d_A` | Bare tube outside diameter and Schmidt-family characteristic length, pp. 12-13 | Closed |
| `A_R` | Complete unfinned outside cylindrical area per tube length, `pi*d_A`, not exposed primary area `A_Ro` | Closed |
| `A_tot` | Actual exposed tube-plus-fin surface per tube length, p. 13 | Closed; an arbitrary area override cannot redefine it |
| `h`, `D` | Radial fin height; fin outside diameter `D=d_A+2*h` | Closed |
| `Fp`, `tf` | Fin center pitch/thickness; clear spacing `Fp-tf` | Closed |
| `Pt`, `Pl` | Independent transverse/longitudinal center pitches | Closed; no square/triangular identity |
| Velocity | Narrowest-free-cross-section velocity, p. 14 | Closed; not face velocity |
| Re | Tube OD, minimum-area velocity; temperature convention below | Partial |
| Surface efficiency | Applied after HTC: `(A_Ri*eta_fin+A_Ro)/A_tot`, Eq. (33) | Closed; do not apply twice |

### Heated-gas Reynolds conversion: internal inconsistency

F15 Sec. 3.4 pp. 52-53 specifies
`T_B=(T_wall,area_mean+T_gas,mean)/2`; the area mean includes fin and exposed-root
temperatures. It specifies viscosity at `T_B`, velocity at `T_g`, and writes
`Re=w*d/nu`. Its next instruction for mass-flux evaluation prints a multiplier
**`T_B/T_g`**, with absolute temperatures. Rendered PDF page 73 confirms that
this is not a text-extraction error.

Independently, for `G=mass_flow/A_min`, `w_g=G/rho_g`, `nu_B=mu_B/rho_B`:

```text
Re = w_g*d_A/nu_B
   = (G*d_A/mu_B)*(rho_B/rho_g)
   = (G*d_A/mu_B)*(T_g/T_B)   # ideal gas, same pressure/composition
```

The derived ratio is reciprocal to the printed instruction. At `T_g=600 K`,
`T_B=450 K`, the factors are `4/3` and `3/4`; they coincide only at equal
temperatures. This report does not silently repair the source or call either
heated-gas choice fully source-closed. A provider must declare its convention,
keep the ambiguity visible, and make fixtures consistent with that choice.
H07 pp. 76-77 separately distinguishes bulk-property Nu and boundary-layer Nu;
the existing single bulk-property request cannot be assumed to implement F15's
Schmidt-family convention.

## 4. Recoverable ITE matrix

Confidence is attached to **each numeric cell**, not to a whole pooled domain:

- **E = SOURCE_EXACT:** explicitly in inspected text/caption/legend/table.
- **C = CROSS_SOURCE_CONFIRMED:** explicitly matched at two locations; does not
  imply independent experiments.
- **R = STRONG_RECONSTRUCTION:** transparent arithmetic or explicit contextual
  matching; derivation identified.
- **W = WEAK_INFERENCE:** plausible hypothesis, not a source acceptance limit.
- **U = UNKNOWN:** missing, never filled from another experiment.

All dimensions mm. Braced lists do not assert all cross-combinations were tested.
Re columns contain reported experimental conditions where recovered, not graph
axis limits or correlation validity bounds. The layout/fin column prevents
pooling incompatible measurements.

| Source / layout / fin | Tube OD | Height | Thickness | Fin pitch | Pt | Pl | n_rows | Re_min | Re_max | Measured quantity | Confidence / membership |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F15 Fig. 39 p. 47; INLINE spiral solid A | 38 E | 16 E | 1 E | 1000/150=6.6667 R | 80 E | **80 E** | U | U | U | HTC: reduced `Ku_Da2` versus `Re_dA` | Caption/rendered legend; directly precedes Eq. (100) |
| F15 Fig. 39; INLINE spiral solid B | 38 E | 16 E | 1 E | 6.6667 R | 80 E | **90 E** | U | U | U | Same | Second exact legend pitch |
| F15 Fig. 59 pp. 64-65; zero-angle INLINE point, spiral | 31.8 E | U | U | U | U | U | U | 20000 E | 20000 E | Nu at one labeled INLINE Re condition | Re=10000 markers occur at nonzero angles only; Eq. (100) fit membership U |
| F15 Fig. 60 p. 66; zero-angle INLINE point, spiral | 38 E | U | U | U | U | U | U | 30000 E | 30000 E | Same | Re=10000 has no measured zero-angle marker; fit membership U |
| F15 pp. 118-119; INLINE hydraulic geometries 1/2/3, not individually specified | U | U | U | U | U | U | {2,4,6,8} E each | approx. 7000 E | approx. 25000 E | Per-row pressure coefficient | Exact aggregate description; Eq. (191) context |
| F15 pp. 32-35; staggered spiral aggregate | U | U | U | U | U | U | 8 E | 4000 E | 32000 E | Thermal measurements | Range for staggered Eq. (82), not Eq. (100) |
| F15 Figs. 43-48 pp. 53-58; staggered row series | U | U | U | {3.62,6.67} E rounded | U | U | {1,2,4,6,8} E | U | U | Bank HTC/reduced Nu and rows | Pitch groups identified p. 57; full cross-matrix U |
| 1992 -> H09 p. 87 Frasz-1; staggered | 38 E | U | U | U | 85 R | 79 R | {1,2,4,6} relative to 8 E via F15 | U | U | Thermal row correction | Pitches from explicit same-pitch statement, not original numeric table |
| F15 Fig. 74 p. 86; spiral, layout not explicit in caption | 38 E | 16 E | 1 E | 6.6667 R | 85 E | {70,90} E | U | U | U | Heated pressure evaluation | Do not assign this full geometry to INLINE experiments |
| F15 p. 92; staggered-section comparison | 44.5 E | 16 E | U | U | U | U | U | U | U | Pressure comparison | No Eq. (100) membership established |
| F15 p. 92; staggered-section comparison | {31.8,38} E | 15 E | U | U | U | U | U | U | U | Same | Cannot join to other figures' fin dimensions |
| F15 Fig. 90 pp. 103-104; staggered spiral | 38 E | 16 E | 1 E | {9.0909,6.6667,3.62319} R from {110,150,276}/m E | U | U | U | 8000 E | 15000 E | Pressure at two labeled Re | Density sweep, not INLINE bounds |
| F15 Fig. 91 p. 104; staggered spiral | 31.8 E | U | U | U | 75 E | {60,70,80} E | U | U | U | Pressure versus Pl | Do not attach Fig. 90 dimensions |
| F15 Fig. 92 pp. 104-105; staggered spiral | 38 E | U | U | U | 85 E | {70,80,90} E | U | U | U | Same | Joining Fig. 90 dimensions remains U |
| H07 Table 1 p. 75 -> H09 Table C.1 p. 153; staggered I-solid | 38 C | 15 C | 1 C | 3.62319 R from 276/m C | 85 C | 79 C | 8 C | U | U | Thermal/pressure | Wall 2.6 mm, active length 500 mm explicit; same earlier Frasz data, not independent replication |
| H07/H09 C.1; staggered I-segmented, excluded fin type | 38 C | 15.5 C | 1 C | 3.62319 R | 85 C | 79 C | {2,4,8} E in H09 | U | U | Thermal/pressure | Cannot extend solid-fin model from segmented data |
| H07 Table 1/H09 Table 2.3 p. 26, H08 row study; staggered U-segmented, excluded fin type | 38 C | 20 C | 0.8 C | 3.38983 R from 295/m C | 85 C | 79 C | {1,2,4,6,8} E H09 | 4500 E study aggregate | 35000 E study aggregate | Thermal/pressure/row effects | Study Re range is not a recovered interval for the I-solid sample |
| F15 appendix pp. 122-123; rig capability | U | U | U | U | variable E | variable E | usually 8; possible {6,4,2}, special 1 E | U | U | Simultaneous thermal/pressure E p. 84 | Active width 500 mm E; capacity is not fit membership |

Fig. 39 recovers **two discrete Pl values**, not a continuous tested interval
or the complete Eq. (100) matrix. Fin OD is 70 mm (R, `38+2*16`), pitch ratios
are 1 and 1.125 (R); both have positive fin-tip gaps. The ordinate is reduced
`Ku_Da2`, not raw Nu: its regression lines cannot directly become Nu fixtures.
Graph axes do not establish exact measured Re endpoints. The separate
**equation** interval is `10000 < Re < 40000` E.

In Figs. 59/60 the Re=10000 trend lines reach zero angle but their measured
markers do not. Those line intercepts are not additional INLINE test points.
The visible INLINE markers are Re=20000 and 30000 respectively; these are
discrete conditions, not recovered intervals. This distinction follows the
text on p. 64 describing one INLINE measurement for each tube diameter.

Figs. 40/41 and the other calculated parameter sweeps are correlation comparisons,
not experimental limits. H07's abstract wording about equal pitches must not
override Table 1's `Pt=85`, `Pl=79`: equal pitches across specimens do not imply
a square bank.

## 5. Eq. (100) row basis

**ROW BASIS = UNRESOLVED.** An eight-row reference bank is a **WEAK_INFERENCE**,
not an implementation parameter. No evidence-based probability can be assigned.

| Interpretation | Evidence | Missing link / decision |
| --- | --- | --- |
| Fixed eight-row bundle | Normal rig configuration; explicit eight-row staggered experiments; later solid I-fin table | No source attaches eight to Fig. 39/Eq. (100). Cannot set source-exact `n_rows=8`. |
| Developed/asymptotic bank | Sec. 3.5.3 fourth/fifth-row discussion | Cited local trends do not define the regression reference. Cannot impose `n_rows>=4` or `>=5`. |
| Average of several row counts | Three INLINE hydraulic geometries each at 2/4/6/8; p. 84 says heat and pressure measured together | Plausible thermal data exist, but regression inclusion/weighting are unknown. |
| Baseline needing correction | No row term; bank HTC may depend on rows | No attached multiplier/normalization recovered; inventing one changes the model. |

The strongest evidence quantifies one usual row count (8), three alternative
even counts (2/4/6), special one-row capability, and **three unspecified INLINE
hydraulic geometries**, each tested at four row counts. None identifies the
thermal regression membership.

F15 pp. 58-59 attributes fourth-row asymptotic behavior to [3], fifth-row to
ESCOA, weak row dependence above `Pl/Pt=1.1`, and increasing dependence below 1.
These are contextual observations, not Eq. (100) boundaries or corrections.
General minimum-row statements elsewhere span 4-20 for INLINE literature and
cannot select one threshold. Local-row asymptotes differ from bank averages.

## 6. Route B equations and route comparison

INLINE **solid** spiral fins, F15 p. 39:

```text
Nu = 0.25 Re^0.65 Pr^(1/3) (T_g_mean/T_fin_mean)^0.25
     * ((d_A+2*h)/d_A)^0.5 * C3 * C5
C3 = 0.20 + 0.65 exp(-0.25*h/(Fp-tf))
C5 = 1.1 - (0.75 - 1.5 exp(-0.70*n_rows)) * exp(-2*Pl/Pt)
```

Temperatures are absolute. `T_fin_mean` means the mean **fin** temperature
defined with Eq. (60), not root temperature or fin-plus-tube area-mean temperature.
`n_rows` counts consecutive physical rows, not the index of the current row.
Treat this as the supplied bank correlation; do not apply local wake decline
again. The solid INLINE constant is **0.20**, not staggered solid 0.35,
staggered serrated 0.55 or INLINE serrated 0.35.

F15 p. 22 says both ESCOA layout formulas can compute one row but may disagree
by up to about 15% for small height/clear-spacing ratio. Computability is not
single-row validation; the later row discussion flags nonphysical dependence
on longitudinal pitch when there is only one row. Preserve that caveat.

| Requirement | Route A: Eq. (100) | Route B: old ESCOA INLINE solid-fin |
| --- | --- | --- |
| Legal algebra | Closed in F15 | Closed in F15 Eqs. (96)-(98), secondary provenance |
| Output | Physical film HTC before fin efficiency | Same |
| Definitions | Tube OD, areas, minimum-section velocity closed | Tube/fin OD, clear spacing, rows and pitches recoverable |
| Numeric Re limits | Open 10000-40000 | Not recovered; must not borrow Route A's |
| Heated-property convention | Instruction recovered but mass-flux conversion contradicts it | Target-specific 1979 convention partial; later bulk convention corroborates family practice only |
| Row treatment | Regression basis open | Explicit `C5(n_rows,Pl/Pt)`; no extra multiplier |
| Geometry envelope | Partial Fig. 39 anchors, incomplete fit membership | Original envelope unresolved |
| Deviation envelope | Unknown | Unknown for this target form |
| Defensible use | Qualified comparison baseline, unresolved rows | Preferred conditional provider, explicit property and validation qualifications |

**Recommendation:** Route B as its own named secondary-source provider; Eq. (100)
as a separate comparison candidate. A bulk-property ESCOA implementation can be
made reproducible by explicitly declaring that engineering convention and its
evidentiary limit; it cannot claim the unavailable 1979 manual or F15 has
source-closed it. A release requiring source-exact heated-property evaluation
still needs that detail. Numerical reconstruction with prescribed properties
and temperatures is possible without calling it a validated heated-HX solver.

## 7. Minimum defensible domain and diagnostics policy

There is currently **no complete experimentally VERIFIED_DOMAIN for Eq. (100)**:
even the two Fig. 39 geometries lack an attached row basis. Store them as
`SOURCE_GEOMETRY_ANCHORS`, not a rectangular range. Do not interpolate Pl into
a validated interval, attach H09's eight rows, or pool unrelated test extrema.
For Route B, formula verification and experimental-domain verification must
also remain separate metadata.

1. Enforce physical validity and selected layout/fin construction. Geometry API
   support for taper, root layers or separate annular fins does not establish
   equivalence to the source's solid spiral fins.
2. Diagnose Eq. (100)'s **open** Re interval. ESCOA Re limits remain unresolved;
   do not inherit Frass's interval or TURB-X revision ranges.
3. Missing diameter, height, pitch, thickness, pitch-ratio and accuracy limits
   receive `SOURCE_RANGE_UNRESOLVED`, never invented min/max values. Unknown
   does not mean verified in-domain.
4. Non-strict calculation may proceed with explicit unresolved-range/property
   warnings, plus Eq. (100)'s unresolved-row warning. Strict source-domain mode
   rejects required conditions that cannot be established, distinguishing
   **unknown** from known out-of-range. Current evidence cannot support a strict
   full heated-workflow domain-success claim for either route.
5. Do not clamp Re, change layout, attach arbitrary multipliers or silently
   switch models. Automatic selection must retain model identity and diagnostics
   through Rating/Simulation.
6. Source reproducibility does not establish accuracy for arbitrary INLINE banks.
   Primary-manual absence alone need not prevent a qualified provider under the
   accepted secondary-source policy. Preserve unresolved library verification
   separately from release claims.

## 8. Hydraulic evidence and integration constraints

F15 Eq. (191), p. 119:

```text
xi = 4.534 Re^(-0.34) (Pt/d_A)^(-0.5) (Fp/d_A)^(-0.7)
     * (h/d_A)^0.5 (Pl/d_A)^0.4
Kz = 1 - 0.1*(8-n_rows)/6    # Eq. (190), proposed below eight rows
```

Three unspecified INLINE geometries, each at 2/4/6/8 rows, approximately Re
7000-25000, precede this equation. This is not a full geometry envelope or
expressly universal validity interval. The hydraulic row factor is not thermal.
Heated/isothermal normalization in Sec. 5.2 needs a separate audit before the
output can be labeled irreversible bank drag.

ESG6 confirms 2-inch tubes, solid/serrated fins, 90-degree INLINE, 4.5-inch pitch
and about 5-40 ft/s minimum-area velocity. A separate INLINE 2-inch tube/4-inch
fin-diameter study finds a strong pitch effect. Its abstract cannot reconstruct
the correlation or a complete row matrix. The HTRI contents confirm dedicated
sections B3.4.6, B3.4.10, B2.4.5 and B2.4.7, not access to their equations.

Existing architecture findings retained for later implementation:

- Circular-finned defaults are Briggs-Young thermal and Robinson-Briggs hydraulic,
  independently of layout. Both explicitly reject INLINE. This is the reported
  failure path; a notebook topology substitution is inappropriate.
- `TubeBundle` already permits independent INLINE pitches and requires circular
  fin clearance. INLINE minimum area uses transverse periodic blockage. No
  triangular/diagonal identity is needed.
- `CircularFinnedTube` supplies physical geometry/areas. Fin conduction and
  resistance-network surface efficiency occur after physical HTC.
- Thermal requests currently carry one transport state; wall iteration passes
  bulk outside properties. Film properties and mean fin temperature need an
  explicit property/iteration contract, not a silent use of existing bulk Re.
- Current three-state hydraulics use `f_RB=dp/(2*n_rows*rho*Vmax^2)`. F15
  Eq. (144) uses `dp_row=xi*rho*wE^2/2`; matching requires `f_RB=xi/4` after
  sourced row treatment, or explicit native-basis integration. Returning xi
  as f_RB would multiply drag by four.
- Flow-path aggregation separates drag, acceleration and external losses. A
  source must establish which losses it covers before integration.
- `ApplicabilityRange` currently uses inclusive bounds; Eq. (100) needs explicit
  open-bound handling. Strict-domain state must reach the adapters.
- Preserve staggered providers/guards. Elliptical INLINE and serrated fins remain
  different model families.

## 9. Validation state and remaining library tasks

Only this report and the reference-index link are changed. No runtime code,
numerical outputs, version or release changelog were changed.

Focused unchanged-baseline tests: **179 passed in 7.75 s**, Python 3.11.9:

```text
py -3.11 -m pytest -q core/tests/finned_tube_correlations_test.py core/tests/circular_finned_tube_geometry_test.py core/tests/circular_fin_efficiency_test.py core/tests/circular_finned_tube_integration_test.py core/tests/finned_tube_dry_release_regression_test.py core/tests/pressure_drop_flow_path_test.py
```

The full command `py -3.11 -m pytest -q core/tests tests` collected 2113 tests
and was **interrupted**, not completed/passed. Workspace `.venv` is Python 3.8.3,
below the project's minimum 3.10, and cannot collect these tests. No new INLINE
provider tests or empirical fixtures are claimed. The private `eco_nominal`/
`eco_max`, `ECO_LT_B1`/`ECO_LT_B2` cases were not run: public implementation is
required first. No private engineering data entered the report. Version is 0.9.3.

Remaining checks have specific objectives:

- BWK 1992 pp. 333-336: whether Eq. (100) or its actual INLINE matrix appears;
  distinguish Fig. 6's staggered row tests.
- ITE tables behind Fig. 39: rows, actual Re endpoints, regression membership,
  additional INLINE geometry and uncertainty.
- ESCOA Solidfin HF 1979/another complete legal reproduction: property state,
  fin versus surface temperature, numeric ranges and row interpretation without
  importing the 1993 coefficients.
- Schmidt/Stasiulevicius predecessors: original INLINE matrices and local-to-bank
  definitions, without transferring staggered ranges.
- Kearney/Jacobi accessible full report: geometry, numerical bank results,
  hydraulic-diameter conversion and touching-fin treatment.
- Frass erratum/author clarification for temperature conversion; Eq. (191)
  hydraulic normalization and loss scope.

## 10. Final decision

```text
FRASS EQ.100 EQUATION = CLOSED
FRASS RE DEFINITION = PARTIAL
FRASS VELOCITY BASIS = CLOSED
FRASS GEOMETRY DOMAIN = PARTIAL
FRASS ROW BASIS = OPEN
ESCOA INLINE SECONDARY RECONSTRUCTION = PARTIAL
PUBLIC IMPLEMENTATION POSSIBLE = CONDITIONAL
RECOMMENDED v0.9.4 THERMAL MODEL = ESCOA INLINE solid-fin Eqs.96-98 via Frass; separate named secondary-source provider with explicit property-convention qualification
RECOMMENDED SOURCE-DOMAIN POLICY = Stated bounds only; SOURCE_RANGE_UNRESOLVED for missing limits; geometry anchors are not VERIFIED_DOMAIN; strict mode rejects unresolved required conditions; no hybrid row correction
```
