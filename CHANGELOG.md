# Changelog

All notable changes to KalKalori are documented in this file.

The project follows **Semantic Versioning (SemVer)**:
`MAJOR.MINOR.PATCH`.

## [0.7.x]

### Added

- Rating can derive the required tube-side pure-steam mass flow from the
  opposing-side heat balance (or explicit duty) and a requested steam outlet
  state. Unknown steam flow defaults to complete condensation
  (``quality_out=0.0``) when no outlet target is supplied.
- Added dry circular-finned-tube geometry, annular-fin efficiency, and
  topology-aware root/contact thermal resistance.
- Added Briggs--Young dry outside heat transfer and verified-scope
  Robinson--Briggs finned-bank pressure loss to Simulation and Rating.
- Added an analytical fast path for constant-thickness circular fins.
- Added deterministic release regressions for dry water/air finned banks,
  geometry, efficiency, correlation bounds, and inside phase-change routes.
- Added fin-surface temperature diagnostics distinct from the existing
  outside/core-wall network node: ``FinnedTubeDiagnostics.fin_tip_temperature_ratio``,
  ``WallTemperatureProbe``/``WallTemperatureEnvelope`` primary-surface/fin-base/
  fin-tip/outside-skin fields, and the corresponding public
  ``outside_skin_temperature_*_estimate`` /
  ``fin_base_temperature_*_estimate`` / ``fin_tip_temperature_*_estimate``
  Simulation/Rating properties. Reporting-only; no fin efficiency, contact
  topology, or thermal-result change. See
  `docs/steam_heater_zone_driving_force.md`.
- Added a practical dimensionless ``fin_contact_efficiency`` fin/root contact
  input (``0 < value <= 1``) on ``CircularFinnedTube``, alongside the
  existing physically explicit ``fin_contact_resistance``. Precedence is
  explicit resistance (even ``0.0``) over efficiency over the ideal-contact
  default; the equivalent areal resistance implied by an efficiency is
  resolved per operating point, not fixed at geometry-construction time. See
  `docs/finned_tube_model.md`.

### Changed

- Clarified physical and effective outside heat-transfer coefficients for
  extended surfaces.
- Refined circular-finned-tube geometry boundaries and limited the built-in
  Robinson--Briggs pressure-drop model to verified equilateral staggered banks.
- Enabled supported inside evaporation and condensation paths opposite a dry
  circular-finned outside surface; wet/condensing finned outside paths remain
  controlled unsupported cases.
- Corrected steam-heater phase-zone driving forces: each desuperheat/
  condensation/subcooling zone now sizes against its own thermodynamic
  driving force (exact terminal LMTD for the isothermal condensation zone,
  the established epsilon-NTU inversion for sensible zones) over an
  equivalent 0D series-marched opposing-stream temperature path, instead of
  one shared arithmetic opposing-stream mean temperature. Whole-exchanger
  EMTD remains the single derived identity ``Q_total / UA_total``. Required
  area/UA/EMTD/overdesign for a steam Rating are expected to change for the
  same geometry; actual area/UA, Briggs--Young, Robinson--Briggs, fin
  efficiency, contact topology, alpha semantics, pressure drop, and the
  derived steam-mass-flow Rating path are unchanged. See
  `docs/steam_heater_zone_driving_force.md`.
- Simplified ordinary finned-tube example/validation configuration to set
  ``fin_contact_efficiency = 1.0`` (ideal contact) instead of an areal
  ``fin_contact_resistance``; the advanced physical input remains fully
  supported and is still used by cases with deliberate non-ideal contact.
  Removed the old diagnostic warning that fired merely because
  ``fin_contact_resistance`` was left unspecified, since ideal contact is
  now the documented default. ``FinnedTubeDiagnostics``/
  ``ThermalResistanceNetwork`` gained ``contact_input_mode``,
  ``fin_contact_efficiency_input``/``_effective`` and
  ``contact_resistance_equivalent_areal``; ``contact_resistance_used`` now
  reports the network's resolved areal equivalent instead of only mirroring
  an unset explicit resistance. Explicit ``fin_contact_resistance`` numerics
  are unchanged.

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
