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

## Planned Releases

### 0.4.x — Reserved / Deferred Internal Hydraulic Refinement

This version range is currently reserved.

Possible future scope:
- refined internal-side pressure drop
- return-bend / header losses
- detailed tube-side hydraulic modelling
- internal-side applicability and warnings

This stage is intentionally deferred because the current priority is wet economizer development.

---

### 0.5.x — Property Layer for Wet Economizer

Planned scope:
- moist air property layer
- PsychroLib adapter
- water / condensate / tube-side media property adapter
- CoolProp / IAPWS integration where appropriate
- dew point and saturation checks
- condensation onset utilities

---

### 0.6.x — Phase Change / Wet Economizer 0D

Planned scope:
- moisture condensation from outside air
- sensible + latent heat balance
- condensate mass flow estimate
- wet economizer 0D model
- applicability and warnings for wet operation

---

### 0.7.x — Finned Tubes

Planned scope:
- finned tube geometry
- fin efficiency
- corrected outside heat transfer area
- outside-side heat transfer and pressure drop corrections for finned tubes

---

### 0.8.x — Non-Standard Tube Geometries

Planned scope:
- elliptical tubes
- flattened tubes
- proprietary or manufacturer-specific tube profiles
- empirical correction interface
- possible commercial extension points for licensed datasets

---

## Future Major Releases

### 1.0.0 — Production-Ready 0D Engine

Target:
- stable public core API
- validated 0D models
- documented applicability limits
- engineering-grade reliability within declared scope

---

### 2.0.0 — Segmented / Distributed Solver

Target:
- segmented 1D solver
- local property variation
- local dry / wet regions
- HTRI-like series, parallel, and longitudinal arrangements
- iterative solution architecture

---

## Versioning Notes

- Versions below `1.0.0` indicate an evolving API.
- `1.0.0` marks a stable, validated 0D modelling core.
- `2.0.0` introduces a new physical modelling paradigm.
- Version numbers are defined by Git tags and this changelog, not by per-file metadata.