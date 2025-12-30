# Changelog

All notable changes to KalKalori are documented in this file.

The project follows **Semantic Versioning (SemVer)**:
`MAJOR.MINOR.PATCH`.

---

## [Unreleased]

- Ongoing calibration of heat transfer correlations
- Documentation improvements
- Additional validation notebooks

---

## [0.3.0] — MVP_0D Stabilization

### Added
- `BareTubeHeatExchanger` model
- Explicit tube bundle geometry with:
  - number of passes,
  - flow arrangement,
  - effective vs total tube length
- ε–NTU thermal solver:
  - counterflow
  - cocurrentflow
  - crossflow (both sides mixed)
- Detailed tube-side pressure drop model:
  - friction losses
  - inlet losses
  - outlet losses
  - return (pass) losses
- Outside flow model driven by mass flow rate
- Comprehensive `HXResult` snapshot:
  - inputs
  - geometry
  - thermal results
  - hydraulic results

### Changed
- Flow arrangement defined as a geometry property
- Clear separation between geometry, correlations, and models
- Consistent naming for tube-side vs outside-side results

### Notes
- This release represents a **fully functional MVP**
- Focus is on correctness and transparency rather than feature breadth

---

## [0.2.x] — Geometry and Flow Foundations

### Added
- Tube and bundle geometry primitives
- Internal flow correlations
- Initial NTU framework
- Energy stream abstractions

---

## [0.1.x] — Initial Architecture

### Added
- Project structure
- Licensing and contribution model
- Foundational abstractions

---

## Planned Releases

### 0.4.x — Hydraulic Accuracy Improvements
- Refined pressure drop models
- Improved outside Δp correlations

### 0.5.x — Thermophysical Properties
- CoolProp / IAPWS integration

### 0.6.x — Phase Change (0D)
- Steam condensation
- Moist air condensation

### 0.7.x — Finned Tubes

### 0.8.x — Non-Standard Tube Geometries (Empirical)

### 1.0.0 — Production-Ready 0D Engine

### 2.0.0 — Segmented / Distributed Solver (1D)

---

## Versioning Notes

- Versions below `1.0.0` indicate an evolving API
- `1.0.0` marks a stable, validated 0D modelling core
- `2.0.0` introduces a new physical modelling paradigm
