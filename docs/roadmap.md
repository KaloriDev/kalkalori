# 🧭 KalKalori — Development Roadmap DRAFT
# KalKalori — Roadmap

This document describes the planned development path of the **KalKalori** (Heat Exchanger Open Engine) computational engine.

The roadmap reflects a deliberate engineering-first approach:
**numerical accuracy and trustworthiness precede model complexity**.

---

## Versioning Philosophy

KalKalori follows **Semantic Versioning (SemVer)**:


Interpretation in KalKalori:

- **PATCH** — bug fixes, refactoring, documentation, calibration
- **MINOR** — significant extensions within the same physical paradigm (0D)
- **MAJOR** — new physical paradigm (e.g. segmentation, iterative solvers)

### Meaning of major versions

- **0.x** — incubation phase  
  Models are usable, but correlations and assumptions are still being refined.
  API may evolve.

- **1.0.0** — production-ready **0D heat exchanger engine**  
  Stable API, validated results within declared applicability limits.

- **2.0.0** — segmented / distributed (1D) solver  
  Fundamental change in modelling approach.

---

## Current Status

**Current version:** `v0.7.4`
**Model level:** MVP_0D  
**Scope:** Bare tube heat exchanger, forced external flow, 0D thermal balance
and straight-tube-bundle hydraulic balance; local nozzle/chamber/tube-sheet/
return losses remain future work.

---

## Roadmap Overview

---


### Deferred and unassigned work

These topics remain on the roadmap but are not assigned to a release number
or implementation order.

#### Condensate management

- film retention and drainage
- carryover and droplet re-entrainment
- re-evaporation
- wet-surface and drainage diagnostics

These effects are strongly geometry-dependent (bundle layout, fins and drainage
path) and should be reassessed after finned-tube geometry is implemented.
The current 0D model continues to explicitly assume full condensate drainage
where that assumption is documented.

#### Freezing / frost / ice

- freezing and solid-water deposits remain deferred
- implementation can be before or after the 1.x line
- current explicit `unsupported` diagnostics remain the intended behavior

#### Acid dew point and multiple condensables

- keep acid-dew and multiple-condensables scope visible in the roadmap
- diagnostic acid-dew-point evaluation can be performed outside KalKalori
  core (for example in a separate tool or orchestrator)
- KalKalori may later accept an externally computed limiting temperature and
  compare it against wall temperature
- full acid condensation, liquid-phase chemistry and corrosion are outside
  current planned core scope

#### Phase-change hydraulics

- two-phase pressure-drop modelling and further hydraulic-accuracy upgrades
  remain future work
- no release number or order is assigned; implementation can be before or
  after `v1.0.0`
- long-term ownership of dedicated hydraulic models belongs to KalFluxi
- KalKalori should provide thermodynamic states needed by hydraulics, including
  phase, vapor quality, mass flux, pressure and enthalpy
- coupled thermal-hydraulic iteration should be managed by an orchestrator
- absent two-phase pressure-drop support remains explicitly `unsupported`, not
  replaced by an approximate single-phase result

**Out of scope for the whole v0.6.x line:** corrosion and material
selection remain outside the solver's scope.

**Outside planned scope:** pure-steam condensation outside tubes.


---

### v0.8.x — Non-Standard Tube Geometries (Empirical)

**Goal:**  
Enable modelling of geometries that cannot be described purely theoretically.

**Examples:**
- elliptical tubes
- flattened tubes
- proprietary manufacturer profiles

**Approach:**
- empirical correction factors
- tabulated or curve-fitted data
- separation between:
  - open-source core (interfaces, mechanisms)
  - optional commercial modules (licensed datasets)

**Notes:**  
This stage explicitly anticipates **commercial extensions** based on
manufacturer data and experimental correlations.

---
## v0.9.x — Supercritical CO₂

Initial support for **supercritical CO₂ (sCO₂) inside smooth circular tubes**.

The first implementation will focus on establishing the thermodynamic and model infrastructure required for future high-accuracy segmented calculations, while providing a limited 0D calculation capability for operating conditions sufficiently far from the pseudocritical region.

### Scope

* pure CO₂ on the tube side
* smooth circular tubes
* supercritical single-phase operation
* heating and cooling
* CoolProp/HEOS as the default open property backend
* generic thermodynamic property-provider interface supporting both integrated and external providers
* optional compatibility with REFPROP or other proprietary/external property providers
* thermodynamic state evaluation using both `T-p` and `p-h` inputs
* energy balance based on enthalpy rather than constant or mean heat capacity
* representative 0D state based on mean pressure and enthalpy
* conventional smooth-tube turbulent heat-transfer correlation as a baseline model
* supercritical-state detection
* detection of pseudocritical-region crossing and excessive property variation
* applicability warnings where the 0D approximation is not considered reliable

### Limitations of the 0D implementation

The 0D model will not attempt to accurately represent heat transfer where strong variations of density, heat capacity, viscosity or thermal conductivity occur across the exchanger.

Cases with significant pseudocritical effects should return an applicability warning such as:

`SCO2_SEGMENTATION_REQUIRED`

rather than extrapolating the standard smooth-tube correlation outside its reliable range.

### Deferred to segmented calculation

Advanced sCO₂ modelling will be implemented together with the segmented heat-exchanger solver, including:

* local `p-h` state tracking
* local thermophysical properties
* wall-temperature-dependent properties
* dedicated supercritical heat-transfer correlations
* Krasnoshchekov–Protopopov-type models
* Jackson-type models
* buoyancy effects
* flow-acceleration effects
* acceleration pressure drop
* adaptive segmentation through the pseudocritical region
* validation against published experimental sCO₂ heat-transfer datasets

---


## v1.0.0 — Production-Ready 0D Engine

This release marks the **maturity of the 0D modelling approach**.

**Declaration:**
> KalKalori 1.0 provides reliable, validated 0D heat exchanger calculations
> within documented applicability limits.

**Characteristics:**
- stable core API
- documented assumptions and limitations
- validated correlations
- suitable for academic, engineering, and commercial use

---

## v2.0.0 — Segmented / Distributed Model (1D)

**New physical paradigm.**

**Scope:**
- axial segmentation
- local temperature and property variation
- dry/wet coil regions
- series, parallel, and longitudinal arrangements
- iterative solvers (row-wise / segmented industrial heat-exchanger capability)

**Impact:**
- breaking API changes
- fundamentally new solver architecture

---

## Summary

KalKalori development prioritizes:

1. **Trustworthy numbers**
2. **Clear physical assumptions**
3. **Extensible architecture**
4. **Clean separation between open core and commercial extensions**

The path from `v0.x` to `v1.0.0` is about **accuracy and confidence**.  
The jump to `v2.0.0` is about **model fidelity and complexity**.
