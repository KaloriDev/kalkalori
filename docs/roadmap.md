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

**Current version:** `v0.3.0`  
**Model level:** MVP_0D  
**Scope:** Bare tube heat exchanger, forced external flow, complete thermal and hydraulic balance.

---

## Roadmap Overview

### v0.3.x — Heat Transfer Accuracy (CURRENT PHASE)

**Goal:**  
Improve confidence in thermal results without changing the 0D model structure.

**Focus areas:**
- review and calibration of heat transfer correlations:
  - tube-side convection
  - outside (air/gas) convection
- clear applicability ranges (Re, Pr, geometry)
- correlation selection mechanisms
- diagnostic warnings (e.g. laminar flow, extreme NTU)

**Outcome:**  
Thermal results (Q, UA, ε) that can be trusted for engineering use.

---

### v0.4.x — Hydraulic Accuracy

**Goal:**  
Provide reliable pressure drop predictions for pump and fan selection.

**Focus areas:**
- verification and refinement of:
  - tube-side friction losses
  - inlet / outlet losses
  - return (pass) losses
- improved outside pressure drop models
- distinction between:
  - preliminary engineering estimates
  - refined calculations

**Outcome:**  
Δp values suitable for real design decisions.

---

### v0.5.x — Fluid Properties (CoolProp / IAPWS)

**Goal:**  
Move beyond constant-property assumptions.

**Focus areas:**
- optional integration with:
  - CoolProp (general fluids)
  - IAPWS (water/steam)
- temperature- and pressure-dependent properties
- pointwise (lumped) property evaluation — still 0D

**Outcome:**  
Improved realism without introducing segmentation.

---

### v0.6.x — Phase Change (0D)

**Goal:**  
Support phase-change phenomena within a lumped-parameter framework.

**Focus areas:**
- condensation of steam on tube side
- moisture condensation from humid air on outside
- sensible + latent heat balance
- effective heat transfer coefficients

**Notes:**  
This is a major functional extension, but **not** a new modelling paradigm.

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
