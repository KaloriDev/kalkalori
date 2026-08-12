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

**Current version:** `v0.6.2`
**Model level:** MVP_0D  
**Scope:** Bare tube heat exchanger, forced external flow, 0D thermal balance
and straight-tube-bundle hydraulic balance; local nozzle/chamber/tube-sheet/
return losses remain future work.

---

## Roadmap Overview

---

### v0.6.x — Phase Change (0D)

**Goal:**
Support phase-change phenomena within a lumped-parameter framework.

**Notes:**
This is a major functional extension, but **not** a new modelling paradigm.
The patch list below is a plan, not a commitment: later patch numbering and
scope may be adjusted based on validation results as each step lands.

#### v0.6.0 — Outside wet-gas water condensation — IMPLEMENTED

- Partial H2O condensation from a water-containing gas outside bare tubes.
- Automatic outside condensation detection and partial wet-surface handling.

#### v0.6.1 — Inside wet-gas water condensation — IMPLEMENTED

- Partial H2O condensation from wet gas inside bare tubes.
- Automatic inside condensation detection.
- Inside wet-surface estimation.
- Wet-wall-temperature heat and mass transfer.
- Gas composition and gas-phase mass-flow update.
- Gas-phase hydraulic-state update.
- One active phase-changing side per call.

#### v0.6.2 — Pure water/steam cooling and condensation inside tubes — IMPLEMENTED / ready for validation

- Superheated-steam desuperheating.
- Saturated-steam condensation.
- Two-phase steam/condensate inlet with vapor quality `0 < x < 1`.
- Partial and complete condensation.
- Saturated-liquid state handling.
- Optional condensate subcooling.
- Automatic allocation of surface between vapor, condensation and liquid zones.
- Gravity-aware Shah (2009) condensation at low, medium and high mass flux.
- Shared Simulation/Rating zone physics with final pressure/enthalpy endpoint states.
- Pure-steam condensation outside tubes is outside planned scope.

#### v0.6.3 — Explicit-inventory evaporation — PLANNED

- Partial evaporation from an explicitly specified liquid or droplet
  inventory in a gas stream.

#### v0.6.4 — Condensate management — PLANNED

- Condensate film retention and drainage.
- Carryover, re-entrainment and re-evaporation.

#### v0.6.5 — Freezing / ice — PLANNED

- Freezing and solid water deposits.

#### v0.6.6 — Multiple condensables / acid-dew providers — PLANNED

- Multiple condensable species.
- Replaceable acid-dew-point and phase-equilibrium providers.

#### v0.6.7 — Phase-change hydraulics — PLANNED

- Condensate-film and applicable two-phase pressure-drop models.

**Out of scope for the whole v0.6.x line:** corrosion and material
selection remain outside the solver's scope.

**Outside planned scope:** pure-steam condensation outside tubes.

---

### v0.7.x — Finned Tubes

**Goal:**  
Extend geometry support to real-world air coolers.

**Focus areas:**
- finned tube geometry
- fin efficiency
- corrected outside heat transfer coefficients
- corrected outside pressure drop

**Outcome:**  
Support for common industrial air-cooled heat exchangers.

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

### v1.1.x — Tube side Hydraulic Accuracy

**Goal:**  
Provide reliable pressure drop predictions.

**Focus areas:**
- verification and refinement of:
  - tube-side friction losses
  - inlet / outlet losses
  - return (pass) losses

**Outcome:**  
Δp values suitable for real design decisions.

---

## v2.0.0 — Segmented / Distributed Model (1D)

**New physical paradigm.**

**Scope:**
- axial segmentation
- local temperature and property variation
- dry/wet coil regions
- series, parallel, and longitudinal arrangements
- iterative solvers (HTRI-like capability)

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
