# Changelog

All notable changes to KalKalori are documented in this file.

The project follows **Semantic Versioning (SemVer)**:
`MAJOR.MINOR.PATCH`.

## [Unreleased]

- Enforced provider-declared thermal references through relative and absolute
  clearance composition as well as direct dispatch. A mismatched bulk/film
  result is rejected instead of bypassing the conductivity-reference contract.
- Corrected the Yang2020 guide to describe supplied roughness and differing
  physical length as diagnostically ignored correlation inputs, consistent
  with existing behavior; the hydraulic path retains its physical length.

- Added `Rossi2017TwistedTapeProvider`: public open-secondary reconstruction
  of Rossi (2017) Eqs.9/10 for paired thermal and Darcy distributed friction,
  with explicit extrapolation and unrepresented width/clearance diagnostics.
  Historical Agarwal attribution is not primary-source verification;
  pychemqt is not a backend. See `docs/twisted_tape_rossi2017.md`.
  The private XSC workflow compares this nominal sensitivity with smooth,
  public Yang2020 and private M&B paths; real-clearance physics remains unsupported.
- Added optional provider-declared bulk/wall/film thermal property references.
  Film properties come from the existing fluid backend at the mean of bulk
  and iterated wall temperatures, never averaged transport values. Hydraulic
  nodes can omit thermal outputs while retaining the same friction model.
  Smooth and existing bulk-reference providers retain their defaults.

## [0.8.0] — Tube-side enhancements and laminar thermal development

### Added and changed

- The open twisted-tape provider ignores supplied inner roughness with an
  informational diagnostic because its equation has no roughness term;
  provider-owned distributed friction is not corrected a second time.
  Physical tube length is independent of the correlation's heated length.

- Smooth laminar internal flow now uses Hausen mean thermal development when
  a positive heated length is supplied. The constant-wall-temperature and
  bulk-property assumptions are visible in warnings. Calls without length
  retain Nu=3.66; hydraulic path lengths and enhanced-provider physics remain
  separate. See `docs/internal_laminar_thermal_development.md`.

- Added explicit radial/diametral twisted-tape clearance geometry and optional
  public clearance-provider composition. Relative models require documented
  base-geometry selection and compatible correlation IDs; absolute models
  supply their own thermal/hydraulic result without evaluating the base.
  Provenance, applicability, wall conventions and separate local losses are
  preserved. No built-in clearance correlation or AUTO fallback is enabled;
  source-definition gaps are documented in `docs/twisted_tape_clearance.md`.

- Added the standalone open-source `Yang2020TwistedTapeProvider`, using
  Yang et al. (2020) Eqs.21-22 for conventional continuous tape: laminar
  liquid heating, Re 100..1100, Pr 7..900, H/D 2..4, and the studied
  D=12 mm, thickness=1 mm, L=300 mm zero-clearance geometry. Native Darcy
  friction, reference blockage geometry and viscosity correction are
  explicit. Other regimes/geometries raise unsupported errors.

- Added the generic coherent tube-side enhancement contract, explicit
  half-turn twisted-tape geometry, provenance and provider-owned reference
  states. Private literature/manufacturer providers can implement this
  interface without disclosing equations. Existing solvers select the
  smooth-tube model when no enhancement is configured, including the
  laminar thermal-development correction described above.
- Integrated enhancement selection on `BareTubeHeatExchanger` with Rating,
  Simulation (including `iterate=False`), wall-state evaluation, hydraulic
  quadrature and surface margin. Results expose authoritative thermal
  enhancement diagnostics and separate per-point hydraulic reference states.
  Provider gradients affect distributed friction only; local losses and
  signed acceleration keep their existing definitions. Enhanced phase
  change and unsupported operating states raise controlled errors.
- Built-in correlation source policy now requires legally and freely
  accessible material; M&B Part I/II remain future private-provider sources.
- Verified external alpha-only providers through Rating and both Simulation
  modes, including native Fanning conversion, provider-owned hydraulic
  references, opaque typed metadata, provenance, applicability and warnings.
  Inconsistent thermal/hydraulic model identities and unsupported results
  fail explicitly. No private M&B or manufacturer physics is included.
- Documented the public/external provider contract and current open-model
  limits in `docs/tube_side_enhancements.md`. Preserved the positional order
  of existing `HXResult` fields when appending enhancement diagnostics.
- Added call-local `tube_side_enhancement` selection to Rating and Simulation
  for notebook-created external provider objects. Omission inherits the
  exchanger configuration; explicit `None` selects the exact smooth default.
  Selection leaves the original exchanger unchanged and governs all nested
  thermal/hydraulic evaluations, including the Rating-to-Simulation bridge.
- Exposed solver-owned per-tube base flow area, derived base mass flux and
  total hydraulic length in the generic enhancement input. Added the private
  notebook provider readiness audit and tests of authoritative bulk/wall
  inputs and provider-owned correction without a second smooth correction.

---

## [0.7.10] — Alternating tube counts in staggered rows

### Changed

- `TubeBundle.n_tubes_per_row` remains the sole tube-row-count input but now
  accepts `float` and is normalized on construction: to the nearest integer
  for `layout="inline"`, and to the nearest half-integer (`..., 6.0, 6.5,
  7.0, ...`) for `layout="staggered"`. A half-integer value such as `6.5`
  represents a real alternating `7 / 6 / 7 / 6 / ...` tube-per-row pattern
  via two new derived properties, `n_tubes_per_row_odd` and
  `n_tubes_per_row_even`; `n_tubes_per_row` itself remains their periodic
  average, `(odd + even) / 2` -- deliberately not `n_tubes_total / n_rows`.
- `n_tubes_total` is now computed exactly from the alternating odd/even row
  pattern instead of `n_rows * n_tubes_per_row`, so every total
  heat-transfer area and tube-side flow-geometry quantity
  (`n_tubes_per_pass_effective`, `internal_flow_area_per_pass`, etc.) stays
  exact for an odd `n_rows` staggered bank. `frontal_flow_area` keeps using
  the normalized periodic-average `n_tubes_per_row`, unchanged from the
  existing 0D model.
- When rows partition exactly between multiple longitudinal sections
  (`rows_partition_is_exact`, from circuit topology), odd/even row parity
  now resets to "odd" at the start of every section instead of continuing
  globally across the whole bundle -- each exact section is a repeated
  physical module. For example 3 exact sections of 5 rows each at
  `n_tubes_per_row=12.5` gives `13/12/13/12/13` per section (189 tubes
  total), not a single 15-row global sequence (188). Single-section bundles
  (the default) are unaffected. A non-exact row/section partition falls
  back to the pre-existing global-row approximation, now flagged by a new
  `ALTERNATING_ROWS_NONEXACT_SECTION_PARTITION` warning when it actually
  matters (alternating odd/even counts, non-exact partition, more than one
  section).
- A new `TUBES_PER_ROW_NORMALIZED` warning is raised when normalization
  actually changes the declared input, surfaced through the existing
  `bundle.warnings`/`geometry_warnings` contract alongside the pre-existing
  `FLOW_ARRANGEMENT_AUTO_MULTIPASS_APPROXIMATION` warning.
- Legacy integer `n_tubes_per_row` inputs normalize to themselves for both
  layouts, so every pre-v0.7.10 numeric result (areas, hydraulics, Rating,
  Simulation) is unchanged.

---

## [0.7.9] — Unified surface margin reporting

### Changed

- Rating and Simulation now share the canonical surface-margin definition
  `overdesign_factor = UA_actual / UA_process - 1` and expose the same
  result-level `UA_actual`, `UA_process`, and `overdesign_factor` contract for
  standard reporting.
- `HXSimulationResult.overdesign_factor` now reports the exchanger's physical
  UA margin for the simulated process instead of the historical constant
  `0.0`. The existing Simulation `surface_margin` input remains available and
  continues to control UA derating independently of the canonical result.

---

## [0.7.8] — Tube circuit topology resolution

### Changed

- Non-exact row partitions across longitudinal sections are now accepted and
  represented by `n_rows_per_section_effective` as an effective 0D average.
- Automatic flow-arrangement resolution now derives the global 0D
  `flow_arrangement` from the tube circuit topology.
- Explicit `crossflow`, `counterflow` and `cocurrentflow` arrangement
  overrides remain available and unchanged.
- Standard tube-circuit configuration tests were expanded to cover topology
  validation, AUTO resolution and explicit overrides.

---

## [0.7.7] — Tabulated liquid property provider

### Added

- Added `LiquidPropertyPoint` and `TabulatedLiquidProvider` to
  `core.properties` for manually supplied single-phase liquid properties,
  without requiring CoolProp. One supplied point gives constant rho/cp/mu/k
  at every positive temperature; two or more points interpolate rho/cp/k
  linearly and mu log-linearly versus T. Specific enthalpy is derived by
  exact piecewise-linear-cp integration, with `h = 0` at the reference
  temperature (the single point's T, or the lowest supplied T for 2+
  points). A multi-point query outside the supplied table raises
  `ValueError` instead of silently extrapolating.
- `TabulatedLiquidProvider.full_at()` reuses the existing `CoolPropProperties`
  container (`transport`/`h`/`phase`/`fluid`/`warnings`) as the smallest
  backward-compatible full-state type; no new generic full-state dataclass
  was introduced and the CoolProp backend itself is unchanged.

### Notes

- `ConstantPropertyProvider` is unchanged and remains available for fixed,
  temperature-independent properties. `TabulatedLiquidProvider` is simply
  the preferred new path for manually entered liquids because it also
  supports temperature dependency and enthalpy.
- Pressure is accepted by `TabulatedLiquidProvider` only for interface
  compatibility; properties are pressure-independent, matching the existing
  point-property provider contract.
- No new dependency was added.

---

## [0.7.6] — Parallel air-side coupling for steam-heater zones

### Changed

- Steam-heater desuperheating, condensation and subcooling zones now share
  the same crossflow-air inlet state and receive air flow in proportion to
  their converged geometric area fraction.
- Zone outlet air streams are mixed after the parallel zone calculations
  instead of being marched sequentially through tube-side phase zones.
- Multi-zone area allocation is iterated to consistency between zone area
  fraction and air-flow fraction while preserving the whole-bank face
  velocity and outside heat-transfer correlation.

### Notes

- Existing steam/condensate phase decomposition and local heat-transfer
  correlations are unchanged.
- The model remains global multi-zone 0D; no falling-film, drainage or
  longitudinal 1D model is introduced.

---

## [0.7.5] — Wet circular-finned outside condensation

### Added

- Added nonlinear radial wet annular-fin heat and mass transfer for outside
  H2O condensation from a gas mixture with a non-condensable carrier,
  including naturally resolved dry, partially wet and fully wet fin states.
- Added independent primary/root and fin sensible duty, latent duty, total
  duty, condensate, wet-area and temperature diagnostics, together with
  typed wet-finned convergence, residual, assumption and warning results.
- Added a separately named wet effective coefficient on the gross-outside-area
  and bulk-gas-to-core-wall driving-force basis without redefining the
  physical sensible film coefficient.
- Added explicit wet-hydraulic contract diagnostics: the calculated
  finned-bank pressure drop is a dry reference and wet pressure-drop support
  is false.

### Changed

- Integrated active wet circular-finned outside surfaces into Simulation,
  Rating and `PhaseChangeMode.AUTO`; dry onset checks continue to use the
  unchanged v0.7.4 circular-finned result route, and
  `PhaseChangeMode.DISABLED` retains its dry sensitivity and warning.
- Preserved welded-fin and continuous-root/contact topology, authoritative
  primary/fin areas, external-area overrides and v0.7.2 contact-input
  precedence while applying contact and fin effects once in the wet solve.
- Narrowed the wet-finned capability guard to permit the supported outside
  gas-mixture/H2O-condensation case while retaining controlled errors for
  unsupported species, directions and simultaneous phase-change cases.
- `PhaseChangeMode.AUTO` dry/near-onset results now expose a complete
  sensible-only `PhaseChangeResult`: `Q_sensible`/`Q_total` report the real
  exchanger duty (matching `HXSimulationResult.q` / `HXRatingResult.
  Q_required`) instead of `0.0`, so callers no longer need to branch on
  `active` before reading them. `active=False` is a valid converged dry or
  near-onset AUTO result, never a calculation failure.
- Restored Rating closure of a single unknown non-condensing-side mass flow
  or outlet temperature when outside H2O condensation is active.
- Hardened the active wet circular-fin solve against a near-boundary
  collapse: if the dry-baseline onset screen activates AUTO but the
  converged nonlinear radial field (including its 0D endpoint wet-zone
  fallback) finds zero net condensate, the call now returns the exact dry
  AUTO result with a `PHASE_CHANGE_WET_SOLUTION_COLLAPSED_TO_DRY` warning
  instead of raising.

### Notes

- The wet extension remains a global 0D model. It uses a deterministic
  160-cell radial finite-volume field and assumes all formed condensate drains
  from the modelled gas phase. When a cold endpoint activates condensation but
  the bulk-mean radial field is dry, a declared linear 0D endpoint wet-zone
  fallback reconciles onset without adding longitudinal segmentation. The new
  wet-finned whole-HX fixed point uses a bounded relaxation step near thermal
  pinches; bare condensation keeps its existing controls.
- Wet-surface pressure-drop correction is not supported. Active condensation
  emits `circular_finned_tube_wet_pressure_drop_reference_only`; the exposed
  pressure drop must be interpreted only as the dry-bank reference.
- Frost/ice, acid or multiple-condensable condensation, liquid-film inventory
  and resistance, retention, flooding, drainage geometry, carryover,
  re-entrainment, re-evaporation, flow maldistribution and thermal
  segmentation remain outside scope.

---

## [0.7.4] — Briggs–Young small-row applicability

### Changed

- `BriggsYoung1963Provider` now evaluates valid circular-finned banks with
  any positive row count instead of rejecting banks with fewer than four rows.
- One- to three-row banks return the
  `briggs_young_1963_small_row_count_extrapolation` warning because they are
  outside the later recommended range. The uncorrected Briggs–Young equation
  is still used; no row-count correction is applied.
- Existing hard validation for invalid inputs, unsupported geometries and
  unsupported layouts remains unchanged.

---

## [0.7.3] — Transverse tube-pass topology

### Added

- Added `TubeBundle.n_passes_transverse` to describe how tube-side passes are
  grouped into longitudinal sections of the outside tube bank. Its default of
  `None` preserves the interpretation and numerical results of earlier bundles.
- Added the derived topology diagnostics
  `n_passes_transverse_resolved`, `n_sections_longitudinal`,
  `n_rows_per_section`, `n_tubes_per_pass_effective` and
  `pass_partition_is_exact`.

### Changed

- Removed the requirement that the total tube count divide exactly by the
  tube-side pass count. Unequal integer pass populations are represented by
  the unrounded effective average `n_tubes_total / n_passes_tube`.
- Changed `n_tubes_per_pass` from a floored integer to the effective floating-
  point average used for tube-side flow area, velocity, Reynolds number, heat
  transfer and pressure drop. Previously exact divisions retain unchanged
  hydraulic and thermal results within floating-point tolerance.

### Notes

- Hydraulic length continues to use the total tube-side pass count, while the
  complete outside-bank geometry and heat-transfer area remain unchanged.
- The transverse-pass topology does not introduce thermal segmentation;
  `flow_arrangement` still describes one global 0D thermal element.

---

## [0.7.2] — Fin-contact efficiency input

### Added

- Added the dimensionless `CircularFinnedTube.fin_contact_efficiency` input
  (`0 < value <= 1`) as a practical alternative to the physically explicit
  `fin_contact_resistance`.
- Added `contact_input_mode`, `fin_contact_efficiency_input`,
  `fin_contact_efficiency_effective` and
  `contact_resistance_equivalent_areal` diagnostics.
- Documented the contact-input precedence and published engineering guidance
  in `docs/finned_tube_model.md`.

### Changed

- Defined contact-input precedence as explicit `fin_contact_resistance`
  (including `0.0`), then `fin_contact_efficiency`, then ideal contact.
- Resolved the equivalent areal resistance implied by
  `fin_contact_efficiency` at each operating point rather than fixing it at
  geometry construction.
- Simplified ordinary finned-tube examples and validation cases to use
  `fin_contact_efficiency = 1.0` for ideal contact while retaining explicit
  resistance inputs for deliberately non-ideal cases.
- Removed the warning previously emitted when `fin_contact_resistance` was
  omitted, because ideal contact is now the documented default. Supplying both
  contact inputs reports that efficiency is ignored.
- Changed `contact_resistance_used` to report the resolved equivalent areal
  resistance. Existing calculations with explicit `fin_contact_resistance`
  remain numerically unchanged.

---

## [0.7.1] — Steam-heater rating and surface diagnostics

### Added

- Added Rating support for deriving the required tube-side pure-steam mass
  flow from the opposing-side heat balance or an explicit duty and requested
  steam outlet state. When no outlet target is supplied, unknown steam flow
  defaults to complete condensation (`quality_out = 0.0`).
- Added fin-surface temperature diagnostics distinct from the existing
  outside/core-wall network node, including primary-surface, fin-base,
  fin-tip and outside-skin estimates in `FinnedTubeDiagnostics`,
  `WallTemperatureProbe`, `WallTemperatureEnvelope`, Simulation and Rating.
- Added public access to raw per-zone steam results and their driving-force
  diagnostics.

### Changed

- Corrected steam-heater phase-zone driving forces so desuperheating,
  condensation and subcooling are each sized against their own thermodynamic
  driving force over an equivalent 0D series-marched opposing-stream path.
- Used exact terminal LMTD for the isothermal condensation zone and the
  established ε–NTU inversion for sensible zones. Whole-exchanger EMTD remains
  the derived identity `Q_total / UA_total`.
- The corrected driving forces may change required area, UA, EMTD and
  overdesign for the same Rating geometry. Actual area and UA, outside
  correlations, fin efficiency, contact topology, pressure drop and the
  derived steam-flow path are unchanged.

### Notes

- The added fin-surface temperatures are reporting-only and do not change fin
  efficiency, contact topology or thermal results.

---

## [0.7.0] — Dry circular-finned tubes

### Added

- Added `CircularFinnedTube` geometry with constant or linearly tapered
  annular fins, explicit root geometry and fin/root contact resistance.
- Added annular-fin efficiency, topology-aware root/contact thermal resistance
  and an analytical fast path for constant-thickness fins.
- Added Briggs–Young (1963) dry outside heat transfer and the verified-scope
  Robinson–Briggs finned-bank pressure-drop model.
- Integrated dry circular-finned tubes into Simulation and Rating.
- Added deterministic regressions for dry water/air finned banks, geometry,
  efficiency, correlation applicability and supported inside phase-change
  routes.

### Changed

- Clarified the distinction between physical and effective-gross outside
  heat-transfer coefficients for extended surfaces.
- Refined circular-finned geometry boundaries and restricted the built-in
  Robinson–Briggs model to verified equilateral staggered banks.
- Enabled supported inside evaporation and condensation opposite a dry
  circular-finned outside surface.

### Notes

- Wet or condensing circular-finned outside surfaces remain controlled
  unsupported cases.

---

## [0.6.3] — In-tube water evaporation

### Added

- Added pure-water inlet/outlet evaporation states with liquid preheating,
  partial or complete in-tube evaporation, and optional vapor superheating.
- Integrated shared Shah (1982) flow-boiling zone physics into Simulation
  and Rating.
- Added a controlled guard for wet-gas inlets whose composition implies
  unsupported carried liquid water.

---

## [0.6.2] — Steam condensation

### Added

- Added water-steam inlet states based on temperature, enthalpy or vapor quality.
- Added in-tube heat transfer for saturated and wet-steam condensation.
- Added automatic desuperheating, condensation and condensate-subcooling zones.
- Integrated pure-steam condensation into simulation and rating.
- Added superheated-steam cooling and condensate subcooling.
- Added gravity-aware low-mass-flux steam condensation with Shah (2009) applicability diagnostics.

### Changed

- Corrected equivalent steam-side HTC and wall diagnostics for multi-zone results.
- Refined the in-tube steam-condensation model architecture.
- Restored water/steam property compatibility and clarified pure-water phase-change provider support.
- Simplified steam-side integration and final-state diagnostics.

---

## [0.6.1] — Inside wet-gas water condensation

### Added

- Added partial H2O condensation from wet gas inside bare tubes.
- Added inside wet-surface heat and mass transfer with gas-phase composition and flow updates.

### Changed

- Extended automatic phase-change detection to the tube side.

---

## [0.6.0] — Outside water condensation

### Added

- Added automatic partial H2O condensation for water-containing gas outside bare tubes.
- Added sensible/latent heat and condensate mass-balance results.
- Added a dry-only override with condensation warnings.

### Notes

- The model remains 0D and supports one phase-changing side.
- Inside condensation, evaporation, film effects, freezing and two-phase hydraulics are not yet included.

---
## [0.5.x] - Simulation/Rating split, pressure-drop, heat-balance and thermal-model improvements

### Added

- Simulation/Rating split with shared heat-balance closure, NTU inverse
  support, and one iterative thermal state used consistently across both
  driver layers.
- Wall-temperature-aware and length-corrected thermal diagnostics, including a
  four-point 0D wall-temperature envelope and a shared `thermal_state` as the
  authoritative source for corrected coefficients.
- Variable-property hydraulic calculations for tube-side and outside-side
  flow, based on reusable inlet/midpoint/outlet states instead of simplified
  mean-property handling.
- General pressure-drop flow-path architecture with stage-based aggregation,
  separate tube-side and outside-side paths, straight-tube and U-tube support,
  tube-sheet entrance/exit losses, and clearer hydraulic diagnostics.
- Optional tube roughness support for internal pressure drop, plus expanded
  tests, warnings, and notebook coverage for the updated workflows.
- Added explicit local pressure-drop calculations for straight ducts, area changes, elbows, planar obstructions, screens, and user-defined elements.
- Added circular and rectangular fitting geometry, including gradual and sudden transitions, smooth-radius and segmented elbows, and 45°, 90°, and 180° bends.
- Added explicit tube-side and outside-side pressure-drop paths with separate reporting of irreversible loss, dynamic-pressure change, and static-pressure difference.

### Changed

- The previous mean-property rating API was renamed to `Simulation`, while
  `Simulation` and `Rating` now both rely on the same converged iterative
  thermal backbone by default.
- Crossflow NTU handling, corrected heat-transfer coefficients, and notebook
  diagnostics were aligned so reported results come from the production thermal
  state instead of parallel recomputation.
- Tube-side and outside-side hydraulic calculations were upgraded from single
  representative states to universal variable-property methods with clearer
  result separation for straight-tube, tube-bank, local, and total pressure
  drop.
- Pressure-drop modules were moved from `core.heat_transfer` to
  `core.pressure_drop`, while earlier import paths remain available through
  compatibility re-exports.
- Internal roughness now affects distributed straight-tube pressure drop when
  specified, while the previous smooth-tube behavior is preserved when
  roughness is omitted or zero.
- Standard exchanger calculations no longer include local pressure-drop paths; these are now evaluated explicitly by the application layer, while standard bundle and tube-bank results remain unchanged.
- Pressure-drop stage results now provide detailed flow and geometry diagnostics, with separate values for irreversible loss, dynamic-pressure change, and static-pressure difference.

### Notes

- Scope remains 0D, with no condensation or latent-heat modelling.
- Local pressure-drop paths are calculated explicitly and are not included automatically in `solve()`, `simulate()`, or `rate()`.
- Elbow losses currently require a user-defined `K` or `Le/D`; return chambers, nozzles, and general external pipework remain unsupported.
- Flat-obstruction losses use a simplified high-Re blockage-based model.
- Outer tube roughness is stored in geometry but is not yet used in outside heat-transfer or pressure-drop calculations.

---
## [0.4.x] - Property Layer Foundation

### Added

- Added initial property layer for future wet economizer and gas-gas calculations.
- Added PsychroLib-based moist-air helpers.
- Added dew-point, saturation, and condensation-onset helpers for moist air.
- Added IAPWS-IF97 water/steam property provider.
- Added optional CoolProp backend for pure fluids and mixtures.
- Added CoolProp backend availability checks for `HEOS` and optional `REFPROP`.
- Added gas-mixture property provider with mole, volume, and mass composition bases.
- Added gas-phase imposition for gas-mixture calculations.
- Added dedicated dry-air property provider.
- Added common mass-flow / actual-volume-flow helpers.
- Added helper for converting dry gas composition plus water ratio to `GasMixtureSpec`.
- Added property-model selection documentation.
- Added functional test notebooks for moist air, water/steam, CoolProp, dry air, and gas mixtures.

### Changed

- Corrected moist-air enthalpy unit handling.
- Clarified SI unit conventions across the property layer.
- Clarified when to use dry air, moist air, gas mixtures, and water/steam property paths.
- Clarified third-party dependency and licensing notes.

### Notes

- `v0.4.x` focuses on the property foundation, not on full heat exchanger solver refactoring.
- High-temperature humid gas should be represented explicitly as a gas mixture with `H2O` as a gas-phase component.
- Gas-mixture calculations do not model condensation, latent heat, wet-surface heat transfer, or water removal.
- Large-temperature-change gas-gas rating requires iterative mean-property calculation and is planned for `v0.5.x`.


---
## [0.3.x] — Outside Flow Model Accuracy

### Added
- Improved outside-side forced-flow model for bare tube bundles
- Mass-flow-driven outside-side calculations
- Calculation of:
  - approach velocity `V_inf`
  - maximum tube-bank velocity `V_max`
  - Reynolds number based on `V_max`
  - Prandtl number
  - outside heat transfer coefficient `alfa_o`
  - outside pressure drop `dp_o`
- Improved outside pressure drop / Euler-based modelling approach
- Applicability checks for outside-side calculations
- Warning system for model limitations and out-of-range conditions
- Validation cases for outside-side heat transfer and pressure drop
- Support for geometry-dependent outside flow inputs:
  - transverse pitch `S_T`
  - longitudinal pitch `S_L`
  - tube outside diameter `D_o`
  - number of rows
  - inline / staggered layout

### Changed
- Outside heat transfer and outside hydraulic calculations are treated as one coupled model area.
- The original split between:
  - heat transfer accuracy,
  - hydraulic accuracy,
  was removed for the outside side because both depend on the same tube-bank flow description.
- The outside-side model now consistently uses the same geometric and velocity basis for `alfa_o` and `dp_o`.

### Notes
- This stage covers the priority outside-flow scope for the current MVP.
- Internal-side hydraulic refinement is deferred and is not part of the current priority path.
- The outside-side model remains 0D / lumped and does not include segmentation.

---

## [0.2.x] — Bare Tube Heat Exchanger MVP

### Added
- `BareTubeHeatExchanger` model
- Bare tube geometry
- Tube bundle geometry
- Tube-side multi-pass support
- Effective and total tube length distinction:
  - `length_effective`
  - `length_total`
- Flow arrangement stored as a geometry property:
  - `counterflow`
  - `cocurrentflow`
  - `crossflow`
- ε–NTU thermal solver
- Result object designed as a full calculation snapshot
- Basic test notebooks for:
  - NTU counterflow
  - bare tube heat exchanger
  - moist air psychrometric checks

### Changed
- Geometry became the source of flow topology.
- Solver no longer defines flow arrangement directly.
- Heat transfer coefficient naming moved toward `alfa` to avoid confusion with enthalpy `h`.

---

## [0.1.x] — Initial Architecture

### Added
- Initial project structure
- Core package layout:
  - `geometry`
  - `heat_transfer`
  - `models`
  - `psychrometrics`
- Energy stream abstractions
- Initial NTU method implementation
- Basic internal flow helpers
- Initial documentation:
  - `README.md`
  - `CONTRIBUTING.md`
  - `roadmap.md`
  - `THIRD_PARTY.md`
  - `POLICY_CODE_ACCEPTANCE.md`

---


## Versioning Notes

- Versions below `1.0.0` indicate an evolving API.
- `1.0.0` marks a stable, validated 0D modelling core.
- `2.0.0` introduces a new physical modelling paradigm.
- Version numbers are defined by Git tags and this changelog, not by per-file metadata.
