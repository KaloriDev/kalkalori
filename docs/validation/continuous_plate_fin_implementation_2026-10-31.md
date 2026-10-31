# Continuous plate-fin implementation record — 2026-10-31

Decision: **IMPLEMENTED_BUT_SOURCE_BLOCKED**. This is a local implementation
checkpoint, not release approval. Source and physical validation gates remain
open even when software tests pass. No final PR is prepared while they are open.

## A. Repository state

Starting branch: `main`; starting HEAD:
`a2b87e1b4a83bdeed9a53e0f25bfc36a2d4ed87e`.
The connected GitHub service independently returned the same current
`main` HEAD. Local shell network access was unavailable; no fetch/rebase was
needed. Work uses `feature/v0.9.6-continuous-plate-fins` throughout.

One pre-existing untracked file, `continuous_plate_fin_architecture_api_audit_2026-10-31.md`,
was read and left unmodified/uncommitted. The complete forensic handover was
read from Downloads under the matching filename without `(1)`. Its source
status takes precedence over the earlier architecture report's conditional
Wang proposals. Neither input report was copied into this change. The user
authorized implementation supersedes the handovers' earlier read-only scopes.

Local checkpoints:

| Commit | Stage |
|---|---|
| `c0be0a6` | Finite common-sheet geometry |
| `35d6761` | Wang arithmetic kernels and source metadata |
| `cd4600e` | Schmidt/contact/correction research profile |
| `69dd8f5` | Rating/Simulation, pressure semantics and steam propagation |

The final documentation/regression commit is identified by the branch HEAD
and the delivery response, avoiding a self-referential commit hash here.
`.kon` remains ignored and untracked; no private files are staged or packaged.

## B. Implemented API and modules

| Module | Additions / changes |
|---|---|
| `core/geometry/continuous_plate_fin.py` | `ContinuousPlateFinGeometry`, `ContinuousPlateFinSurface`, `FinAttachment`; finite surface and throat construction |
| `core/geometry/bundle.py` | Optional keyword-only `continuous_fin_geometry`; immutable `continuous_fin_surface`; shared area/flow adapters |
| `core/geometry/tube.py` | Bundle dispatch identity `TubeSurfaceType.CONTINUOUS_PLATE_FIN`; core remains `BareTube` |
| `core/heat_transfer/continuous_fin_source.py` | `ContinuousFinRequest`, `ContinuousFinSourceCard`, `ContinuousFinHTCProvider`, `ExperimentalWang2000HTC`, `ContactPolicy`, `wang2000_j`, `wang2000_f` |
| `core/pressure_drop/continuous_fin.py` | `ContinuousFinDPProvider`, `ExperimentalWang2000DP`, typed point/bank/results and signed source density term |
| `core/heat_transfer/continuous_fin_efficiency.py` | `SchmidtEquivalentCell`, `ContinuousFinEfficiencyResult`; explicit finite-cell limitation |
| `core/heat_transfer/continuous_airside.py` | `ExperimentalWang2000Model`, frozen baseline/corrected snapshots, dry/source/contact guards |
| `core/heat_transfer/outside_dispatch.py` | Continuous-specific dispatch/network/diagnostics; shared execution guard |
| `core/heat_transfer/outside_side.py` | Forward profile and evaluated thermal result into helper/network paths |
| `core/models/bare_tube.py`, `core/models/rating.py`, `core/models/simulation.py`, `core/heat_transfer/thermal_iteration.py` | Profile acceptance and applicability propagation; numerical algorithms retained |
| `core/phase_change/steam_heater.py`, `steam_integration.py` | Zone/reference network profile propagation; unavailable common-fin extrema |
| `core/phase_change/wet_coil_provider.py` | Built-in wet providers reject direct common-sheet calls; default selection unchanged |
| `core/pressure_drop/flow_path.py` | Preserve explicit pressure basis on unresolved source estimates |

Geometry, heat-transfer and pressure-drop package namespaces export the new
typed additions. Existing scalar result/export columns are retained; new
common-sheet snapshots use nested diagnostics. No annular diameter/root
fields are populated for shared sheets. A future private reporter would need
family-aware extraction from these new snapshots; it was not modified.

## C. Upstream reuse

Only the small Schmidt equivalent-radius/efficiency kernel is adapted from
`PlainContinuousFin` in python-hvac `fins.py`, commit
`94247560a667905bfbf67e45a58965a58254fce8`. MIT copyright and permission are
preserved in `core/heat_transfer/python_hvac_MIT.txt`, included as package data,
and attributed in `THIRD_PARTY.md`. Changes include SI scalars, pitch-dependent
bisectors, admissibility guards, zero-h limit, adiabatic cell boundaries and
explicit finite-edge warnings. No full exchanger solver, upstream geometry,
ACHP code or python-hvac runtime dependency is included. Wang kernels and
finite geometry are independently implemented from equations/derivations.

## D. Provider/source evidence and Kim alternative

Wang: **SOURCE_PARTIALLY_CLOSED / SYNTHETIC_ARITHMETIC_ONLY**.
The [author university record](https://scholar.nycu.edu.tw/en/publications/heat-transfer-and-friction-characteristics-of-plain-fin-and-tube--2/)
confirms empirical inclusion of contact, but supplies no primary full text.
The [publisher DOI](https://doi.org/10.1016/S0017-9310(99)00333-6) remains the
required primary reference. The existing forensic handover supplies the
provisional equations and coefficient-conflict evidence. P2/P5 retain plus
terms, F3 retains its minus term and P6 uses 1.21. Unknown Re and source area
limits remain unknown; secondary software ranges are not promoted.

Kim–Youn–Webb 1999 was actively investigated as a separate paired alternative.
The [author-hosted primary paper](https://www.researchgate.net/profile/Baek-Youn/publication/245362115_Air-Side_Heat_Transfer_and_Friction_Correlations_for_Plain_Fin-and-Tube_Heat_Exchangers_With_Staggered_Tube_Arrangements/links/60e3bbe7299bf1ea9ee52c37/Air-Side-Heat-Transfer-and-Friction-Correlations-for-Plain-Fin-and-Tube-Heat-Exchangers-With-Staggered-Tube-Arrangements.pdf?origin=scientificContributions)
is accessible and confirms different HTC/friction domains and mixed original
data reduction/contact/entry-exit treatments. This preserves the forensic
finding that primary equations/tables were recovered. The handover does not
transcribe the full row multiplier or bare-bank drag equation; current text
extraction garbles their exponents. No ambiguous coefficient was implemented.
No independent experimental numerical checkpoints or finite-pack/efficiency
validation were obtained. Consequently **no complete Kim provider is promoted**.
This is a validation/transcription gate, not a claim that its source is absent,
and does not make Wang mandatory for a future production release.

Kim paired domain intersection from the handover, still requiring its own
verified implementation:

| Rows | Re_D | St/Sl | St/D | clear s/D |
|---|---|---|---|---|
| 1–2 | 591–4430 | 1.154–1.654 | 2.399–2.876 | 0.135–0.300 |
| >=3 | 505–19766 | 0.857–1.654 | 1.996–2.876 | 0.081–0.641 |

A large-diameter HTC-only sample is not paired friction evidence. Neither
provider establishes support for industrial tube OD 25–50 mm.

## E. Geometry and efficiency verification

Independent hand calculations verify integer sheets, axial placement,
aperture counts, collar contacts, exposed cylinders, finite faces/edges/lips
and the area ledger, including sealed side edges and flush axial broad faces.
Finite row and diagonal examples include actual edge
margins; exhaustive chain enumeration independently checks the gap graph.
Bare/collar axial passages are superposed independently. Sigma and the
explicit candidate `4 Ac L/Ao` identity are checked. These are mathematical
geometry tests, not experimental flow validation or manufactured-part metrology.

Schmidt tests verify h→0, k→infinity, monotonic h response, both pitches and
physical/gross area reconstruction and one-face sealed terminal sheets.
Boundary cells are explicitly unresolved.
No fictitious finite-sheet temperature field is reported. B05 is only partly
addressed; independent source-geometry and finite-sheet conduction validation
remain necessary.

The following are additional **software arithmetic checkpoints only**, not
experiments. They retain the five handover input pitches, diameters, row/Re
values and 0.36 m axial width, with 8 tubes in every physical row, H=9 Pt,
depth=(N+1) Pl, y_origin=-Pt/2, x_origin=-Pl, first sheet at zero and touching
collars hc=Fp. Actual finite margins replace the handover's provisional
boundary description. Mass flow is recomputed to retain the requested Re_Dc.
Ao below is the candidate face/cylinder source area, not physical gross area.

| Case | Ac (m2) | Ao candidate (m2) | Dh (mm) | j | h (W/m2K) | core estimate (Pa) | face eta |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1, N=1 | 0.04902030 | 4.93906835 | 1.51257144 | 0.03150650 | 60.51288 | 8.56509 | 0.85453380 |
| B2, N=2 | 0.05161815 | 3.71707774 | 3.17451233 | 0.01528886 | 44.04676 | 8.18173 | 0.88894429 |
| B3, N=4 | 0.04318158 | 5.36102809 | 2.04589867 | 0.01260058 | 81.59144 | 71.99915 | 0.88674517 |
| B4, N=6 | 0.06311572 | 12.31531260 | 3.94623405 | 0.00777627 | 49.27231 | 71.62897 | 0.79677571 |
| B5, N=3 | 0.05109858 | 5.85967936 | 2.65796919 | 0.00664283 | 79.74081 | 123.28633 | 0.81842419 |

The B3 finite separator differs from the provisional transverse-only area;
the sealed first broad face reduces Ao in all five cases. Both lattice pitches,
adiabatic cell boundaries and one-face terminal weighting change Schmidt eta.
These differences are not tuned away. Source/experimental gates remain open.

## F. Integration evidence

Tests exercise direct solve, mean-state iteration, Rating, Simulation at two
airflows, energy closure, source status retention, inside wall/fouling
resistances, UA reconstruction and exactly-once engineering corrections.
Baseline snapshots are frozen. Hydraulic correction multiplies only the
positive core estimate; the signed density contribution remains unchanged.
Unresolved pressure estimates cannot aggregate installation/local losses.

IAPWS97/Shah saturated condensation, desuperheating/condensation, three-zone
desuperheating/condensation/subcooling and steam Rating execute with the same
dry profile. Existing two-phase DP remains unavailable. Shared-sheet inter-zone
conduction is neglected and explicitly warned. With the test geometry, air
0.15 kg/s at 290 K, steam 0.001 kg/s at 1 MPa and 520 K is a successful
three-zone case. Steam 0.002 kg/s at the same conditions reaches the existing
80-iteration zone-allocation guard (final-tree residual about 3.4e-7); that guard and
tolerances are preserved. External wet providers are rejected; the default
`ElmahdyMitalasWetCoilProvider` remains unchanged.

## G. Validation execution

Python 3.11 is available and supported. The other installed interpreter is
3.8, below the declared >=3.10 minimum, and is not used. Python 3.10 is not
installed, so no separate 3.10 execution is claimed.

- New feature suite on the final source tree: **62 passed**. Final sealed-
  boundary, one-face terminal-sheet and direct-wet guards are included.
- Selected legacy regression: **299 passed**, including serrated dispatch and
  source physics, existing steam, individual fins, wet-provider injection,
  fin contact and fouling.
- Final wet-provider/integration regression: **64 passed**.
- Complete public suite: **2,662 passed, 2 failed** across all **2,664 cases
  in 118 files**, using four independent test-file workers (2,991 seconds).
  Both failures are the pre-existing individual circular-fin wet Rating
  `test_inverse_area_rating_with_default_deadline[False/True]` cases, which
  exceeded their unchanged 300-second wall-clock budget under parallel load.
  The unchanged `main` export passed the `[False]` control in 282 seconds.
  The exact two feature-branch cases then passed sequentially with the same
  300-second per-operation budget: **2 passed, 4 deselected in 418.84 seconds**.
  Thus every one of the 2,664 collected public cases has passing coverage,
  while the full parallel invocation retains its two recorded deadline failures.
  This host/load sensitivity is not hidden by relaxing controls or tolerances.
  The earlier serial whole-suite sweep was intentionally stopped to run this
  complete final-tree collection; no public test file was omitted.
- All five forensic synthetic Wang cases reproduce j/f and independently
  reconstructed h/core-pressure values with rounded-reference tolerances.
  Corrected finite geometry/Schmidt values deliberately are not forced to
  match the provisional handover's periodic geometry or regular-hexagon eta.
- Wheel built with `pip wheel --no-deps --no-build-isolation`; source archive
  built with the declared `setuptools.build_meta` backend. The `build` frontend
  is not installed (`build/` is a local namespace directory), so its module
  invocation is not claimed as successful.
- Wheel and sdist include the MIT notice and continuous geometry/provider
  modules. Both archives were inspected for private artifacts/PDFs; none found.
- Wheel installed to an isolated target directory and its **62 feature tests
  passed** from that directory, without importing the repository source tree.
  Wheel version stays 0.9.5 because release promotion is blocked. Artifacts
  remain local under ignored `dist/continuous-fin-check/`.

## H–J. Limitations, release decision and required next actions

| Gate | Current outcome | Minimum closure |
|---|---|---|
| B01 | Open | Lawfully accessible primary Wang Part II and equation-by-equation check |
| B02 | Open | Primary Reynolds envelope, collar/passage geometry and source area definitions |
| B03 | Open | Pressure reduction, reference density and entry/core/exit reconciliation |
| B04 | Open | At least three independent experimental j/f/pressure checkpoints |
| B05 | Partially implemented, open | Independently validate finite passage mapping and finite-edge Schmidt/thermal treatment |
| Kim alternative | Source recovered, implementation/validation incomplete | Verify full Eq.2/Eq.8, implement its own paired domains/pressure convention, obtain independent checkpoints |
| Industrial OD | Unsupported | Matched HTC/DP evidence for continuous fins at 25–50 mm |

Keep the dedicated branch and logical local history. Do not merge, push a final
PR, publish a package, tag a release or promote version metadata until a complete
paired provider passes the applicable source and physical validation gates.
Wet-sheet physics, common-fin inter-zone conduction, detailed field extrema,
two-phase tube-side DP and fan equipment logic are not implemented here.
