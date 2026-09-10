# Tube-side enhancement: architecture audit and source gate

Status: **CORRELATION SOURCE GAP**. This is a pre-implementation audit,
not a frozen correlation specification or an announcement of delivered
v0.8.0 functionality. Package version remains v0.7.10.

## Authoritative references

- R. M. Manglik and A. E. Bergles (1993), *Heat Transfer and Pressure Drop
  Correlations for Twisted-Tape Inserts in Isothermal Tubes: Part I —
  Laminar Flows*, Journal of Heat Transfer 115(4), 881–889.
  DOI: [10.1115/1.2911383](https://doi.org/10.1115/1.2911383).
- R. M. Manglik and A. E. Bergles (1993), *Heat Transfer and Pressure Drop
  Correlations for Twisted-Tape Inserts in Isothermal Tubes: Part II —
  Transition and Turbulent Flows*, Journal of Heat Transfer 115(4), 890–896.
  DOI: [10.1115/1.2911384](https://doi.org/10.1115/1.2911384).

Verification attempted on 2026-09-10: DOI/publisher pages could not be
retrieved. Searches did not yield readable Part I full text. An indexed
[Part II reproduction](https://www.scribd.com/document/677049528/manglik1993)
provided prose but damaged mathematical text. It is insufficient to
transcribe the complete production model. No full-text reference works
are included in this repository.

The Part II transition discussion (pp. 894–895) distinguishes continuous
asymptotic friction matching from heat-transfer interpolation between
laminar and turbulent estimates, identifying Sw = 1400 and Re = 10000
as engineering transition criteria. The full definitions, equations and
endpoint construction still need verification from readable originals.
The smooth-tube Re = 2300...4000 blend must not be substituted.

## Missing source definitions and equations

Each item below must be resolved with a page/equation/figure citation
before freezing the corresponding API or production calculation.

| Required item | Unresolved verification |
| --- | --- |
| Insert geometry | Exact definitions of internal diameter, tape thickness, width/clearance, H and y; confirmation that H is the axial 180-degree twist length and identification of the denominator in y. |
| Flow geometry | Empty-tube versus tape-blocked axial area, wetted perimeter, hydraulic diameter, and which quantities each equation uses. |
| Velocity and dimensionless groups | Empty-tube, axial and swirl/reference velocities; every Reynolds number, swirl Reynolds number, Sw and Pr definition. |
| Heat-transfer basis | Nu definition, characteristic length, mean/length dependence and physical reference area. |
| Friction basis | Primary pressure-drop definition of literature f and its associated velocity/length scales; explicit confirmation of the Fanning convention. |
| Part I physics | Complete laminar friction and mean-Nu equations (including the Part I Eq. (17) cited by Part II), coefficients, exponents, developing-flow terms and bulk/wall corrections. |
| Part II physics | Unambiguous turbulent friction Eq. (7), Nu Eqs. (8)–(9), laminar asymptote Eq. (10), continuous friction Eq. (11), and property corrections. |
| Heat-transfer transition | Exact independent variable for the recommended linear fit, endpoint evaluation, and treatment when the Sw and Re bounds overlap or change order for a geometry. |
| Applicability | Separate validated and extrapolation ranges for Re, Pr, y, thickness/diameter, clearance and length; tight/loose fit assumptions; liquid/gas and heating/cooling restrictions in each regime. |
| Independent anchors | At least one trustworthy numerical point each for Part I and Part II, with all inputs, reference conventions, provenance and justified tolerance. |

No equation or numerical anchor has been reconstructed from memory or
accepted from an unverified secondary transcription. Neither gas nor liquid
enhancement applicability is implemented at this checkpoint.

## Audit of the existing production paths

The audit baseline is main commit `42b9da4`, after the v0.7.10 merge.

- `core/geometry/bundle.py`: `internal_flow_area_per_pass` multiplies the
  effective tubes per pass by the tube's flow area;
  `internal_hydraulic_diameter` delegates to the tube. These quantities
  also serve existing hydraulic/local-loss paths. Do not globally reduce
  them for tape blockage. Keep physical inner-wall heat-transfer area.
- `core/heat_transfer/internal_flow.py`: smooth thermal evaluation owns
  laminar/Gnielinski Nu, its Re = 2300...4000 blend, optional gas wall
  correction and finite heated-length correction. An enhancement result
  must bypass those correlation-specific operations.
- `core/heat_transfer/thermal_iteration.py`: `_evaluate_local_wall_state`
  evaluates authoritative bulk and wall properties, calls the internal
  diagnostic correlation, and collects `ModelWarning` values. It is shared
  by the iterative thermal state and wall-temperature probes. Dispatch here
  must pass bulk/wall states to the selected enhancement provider once,
  with no subsequent smooth-tube wall or length multiplier.
- `core/models/bare_tube.py`: `BareTubeHeatExchanger.solve` calls internal
  heat transfer and `calculate_tube_bundle_hydraulics` separately. A shared
  enhancement configuration must govern both paths, including the
  constant-property route.
- `core/pressure_drop/internal_pressure_drop.py`: inlet, midpoint and
  outlet hydraulic points currently use Darcy friction factors. Straight
  friction uses Simpson integration of f/rho (or f*G^2/rho for differing
  point flows). Signed acceleration is a separate endpoint momentum term.
  The provider's friction must be integrated with its own verified
  reference area, velocity and diameter, without a second blockage factor.
  Preserve nozzle, chamber, tube-sheet and return-loss models separately.
- `core/models/rating.py` and `core/models/simulation.py`: authoritative
  iterative thermal states coexist with a `solve` snapshot for hydraulics.
  Both must expose the same provider identity and applicable diagnostics.
  Preserve `core/models/surface_margin.py` semantics and physical areas.
- `core/common/warnings.py` provides typed `ModelWarning` and applicability
  helpers. Provider warnings must reach standard result `warnings`.
  Existing controlled phase-boundary errors in `core/phase_change/capability.py`
  provide a pattern for explicit rejection, not silent smooth fallback.

## Implementation constraints after source verification

Use one enhancement provider returning coherent thermal/hydraulic results
and typed reference-state diagnostics. External providers may supply
performance without disclosing equations. Retain the exact existing path
when no enhancement is configured; do not independently select thermal and
friction models for an insert.

The required production friction convention is Darcy. Once the primary
Fanning definition is verified, convert explicitly with
`f_darcy = 4 * f_fanning`, exposing both and locking the conversion with an
independent regression test. This conversion does not authorize changing
the associated reference velocity or diameter.

Keep geometry fields provisional until the nomenclature gate is closed.
Prefer `half_turn_length` if H is confirmed as the 180-degree axial length;
never use ambiguous `pitch`. Derive twist ratio from the verified geometry.
Reject nonfinite or invalid inputs and report applicability violations.

Planned scope: single-phase circular internal flow, full-length continuous
tape, Rating and Simulation, thermal iteration, distributed friction and
separate acceleration; bare or externally finned compatible core tubes.
Gas support is conditional on source verification.

Excluded: internal boiling/condensation, two-phase and supercritical flow,
perforated/broken/short/alternate-axis tape, wire matrix, bonded tape fin
area, insert fouling interaction and insert-specific entrance/exit losses.
No CALGAVIN/hiTRAN correlation, reverse engineering or private dependency
belongs in the GPL implementation.

## Continuation gate

Obtain readable, legitimately accessible Part I and Part II pages, resolve
the table above and complete the frozen reference specification first.
Then follow the separate provider, standalone correlation, integration and
external-provider commits from the development brief. This audit commit
does **not** complete the originally requested Commit 1 source freeze.

Validation must reside in `core/tests`: independent literature anchors,
factor-of-four, geometry/blockage, transition continuity, applicability,
wall correction ownership, fake provider, warning propagation, unsupported
phases, Rating/Simulation, surface margin and exact smooth regressions.
No public notebooks, experiments or manual harnesses are required.

Audit validation: Python 3.11, 78 existing tests passed across
`thermal_iteration_test.py`, `internal_gas_wall_correction_test.py`,
`tube_bundle_hydraulics_test.py` and `staggered_row_tube_counts_test.py`,
collected in that order. Collecting the hydraulic module first instead
failed with an existing circular import through `core.properties`,
`core.heat_transfer` and `finned_tube_pressure_drop`. Production files were
unchanged; this import-order limitation is not repaired by the audit.
The full suite and enhancement-specific tests have not been run.

Do not release or mark v0.8.0 delivered at this checkpoint. The final
release commit remains subject to explicit later user authorization.
