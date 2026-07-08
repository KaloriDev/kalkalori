# Changelog

All notable changes to KalKalori are documented in this file.

The project follows **Semantic Versioning (SemVer)**:
`MAJOR.MINOR.PATCH`.

---

## [Unreleased]

### Planned
- Property layer required for wet economizer modelling
- Psychrometric property adapter
- Fluid property adapter for water / condensate / tube-side media
- Condensation onset checks
- Preparation for wet economizer 0D model

---
## v0.4.3 - Property-Driven Fluid Inputs
### Added
- Added adapters between common transport properties and existing heat-transfer FluidProps containers.
- Added mean-temperature property evaluation helper.
- Added moist-air transport property helper for outside-flow calculations.
- Added psychrometric wet-process helpers for surface saturation, condensable water, and enthalpy-drop limits.
- Extended IAPWS-IF97 water/steam property access to support T+p, p+x, and T+x input modes.
- Added high-temperature moist-air handling for cases where saturation state is not defined at given T and p.
- Added humidity-ratio helpers for g_water/kg_dry_air input and display.

### Notes
- No heat-transfer solver refactor added yet.
- No wet economizer solver or phase-change heat balance added yet.

---
## v0.4.2 - IAPWS Water/Steam Provider

### Added
- Added IAPWS-IF97 based water/steam property provider.
- Added `iapws` as a project dependency.

### Notes
- No wet economizer solver or phase-change heat balance added yet.

---
## [0.4.1] — Outside Flow Model Accuracy

### Status
Implemented in the current development line.

### Added
- Added `MoistAirState` and saturated moist-air state helpers.
- Added simple condensation onset check with `ModelWarning` support.
- Added minimal single-phase fluid property structures and constant property provider.

### Changed
- Corrected moist-air enthalpy units in the PsychroLib adapter.
- Added clearer psychrometric helper functions and SI unit docstrings.
- Kept backward-compatible psychrometric wrappers.

### Notes
- Starts the v0.4.x property-layer line.
- No wet economizer solver added yet.
- No wet economizer solver or phase-change heat balance added yet.

---
## [0.3.x] — Outside Flow Model Accuracy

### Status
Implemented in the current development line.

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