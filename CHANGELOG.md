# Changelog

All notable changes to KalKalori are documented in this file.

The project follows **Semantic Versioning (SemVer)**:
`MAJOR.MINOR.PATCH`.

## [0.6.2] — Pure water/steam cooling and condensation inside tubes

### Added

- Added water/steam inlet states based on temperature, enthalpy or vapor quality.

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
