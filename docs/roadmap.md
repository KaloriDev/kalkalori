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

**Current version:** `v0.7.5`
**Model level:** MVP_0D  
**Scope:** Bare and circular-finned tube heat exchangers, forced external flow,
0D sensible/phase-change thermal balance and tube-bank hydraulic balance;
active wet-finned pressure drop, local nozzle/chamber/tube-sheet/return losses
and distributed thermal resolution remain future work.

---

## Roadmap Overview

---

### v0.7.5 — Wet Circular-Finned Outside Condensation

**Delivered scope:**

- H2O condensation from a gas mixture with a non-condensable dry carrier on
  the exposed primary/root cylinder and circular annular fins;
- a deterministic nonlinear radial finite-volume fin field with dry,
  partially wet and fully wet states, both faces and the physical fin tip;
- independent primary/fin sensible duty, latent duty and condensate splits,
  with authoritative whole-side mass and energy closure;
- preserved welded-fin and continuous-root/contact topology, including the
  v0.7.2 contact-input precedence and unchanged legacy dry route;
- Simulation, Rating and `PhaseChangeMode.AUTO`, with the existing
  `PhaseChangeMode.DISABLED` dry-result warning behavior;
- typed wet-surface state, temperature, wet-area, iteration, residual,
  assumption and warning diagnostics;
- an explicit dry/reference-only finned-bank pressure drop during active
  condensation; wet pressure-drop support remains false;
- a regime-independent `PhaseChangeMode.AUTO` result contract: dry and
  near-onset are valid converged regimes (`active=False` is not a failure),
  the returned `Q_sensible`/`Q_total` always equal the real exchanger duty,
  and a wet-fin solve that converges to zero net condensate near onset
  returns the exact dry result with a diagnostic warning instead of raising.

This is a global 0D extension. It does not add row-wise, longitudinal or
circuit-wise thermal marching, and `n_passes_transverse` is not thermal
segmentation. A cold endpoint / dry bulk-mean mismatch uses the declared
linear 0D endpoint wet-zone fallback, not a spatial temperature map. Formed
condensate is assumed to drain completely from the
modelled gas phase. Film retention, drainage geometry, flooding/bridging,
droplet carryover or re-entrainment, re-evaporation, frost/ice, acid dew point,
multiple condensables, flow maldistribution, condensate-film resistance and
wet hydraulics remain excluded.

---


### Deferred and unassigned work

These topics remain on the roadmap but are not assigned to a release number
or implementation order.

#### Condensate management

- film retention and drainage
- carryover and droplet re-entrainment
- re-evaporation
- liquid-inventory, retention and drainage-path diagnostics

These effects are strongly geometry-dependent (bundle layout, fin joints and
drainage path) and require a dedicated liquid-inventory model beyond the
implemented wet thermal surface response.
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
- absent two-phase pressure-drop support remains explicit. Active wet-finned
  condensation may expose the existing dry-bank value only when labelled as a
  reference, never as an approximate actual wet pressure drop

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

### v0.9.x — External Tube-Performance Provider Architecture

**Goal:**
Generalize v0.8.x's empirical non-standard-geometry support into a formal
provider architecture for tube-side and/or outside performance data that
cannot be derived from KalKalori's built-in correlations.

**Intended for:**
* elliptical / flattened tubes
* proprietary tube profiles
* externally supplied empirical correlations or performance data

**Scope:**
* generic tube-performance provider interface supporting both integrated and
  externally supplied data
* support for externally computed, tabulated or curve-fitted heat-transfer
  and/or pressure-drop performance as an alternative to internal correlations
* applicability/validation boundaries for externally supplied data
* continued separation between the open-source core (interfaces, mechanisms)
  and optional commercial modules (licensed datasets)

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

### v2.x.x — Supercritical Fluids

Initial target: **pure supercritical CO₂ inside smooth circular tubes**.
Supercritical water may follow later using the same architecture.

Full sCO₂ exchanger rating is intentionally deferred until the segmented
solver is available. Thermophysical properties can vary very strongly near
the pseudocritical temperature, and an accurate property backend (e.g.
CoolProp/HEOS, with optional REFPROP) can give accurate properties at an
individual state — but that does not solve the limitation of representing
a whole exchanger with one 0D mean state. A heat exchanger may cross the
pseudocritical region while its mean bulk temperature remains far from it,
and dedicated supercritical heat-transfer correlations require meaningful
local bulk and wall states. Local `p-h-T` states, wall temperatures and
property evaluation are therefore handled segment-by-segment rather than
as a single 0D approximation.

**Expected scope:**
* local `p-h` thermodynamic state tracking
* CoolProp/HEOS property backend, with optional REFPROP
* pseudocritical-temperature detection and pseudocritical crossing
* local bulk/wall properties
* dedicated public supercritical smooth-tube correlations
* adaptive refinement/segmentation where property gradients are large
* later consideration of acceleration and buoyancy effects

---

## Summary

KalKalori development prioritizes:

1. **Trustworthy numbers**
2. **Clear physical assumptions**
3. **Extensible architecture**
4. **Clean separation between open core and commercial extensions**

The path from `v0.x` to `v1.0.0` is about **accuracy and confidence**.  
The jump to `v2.0.0` is about **model fidelity and complexity**.
