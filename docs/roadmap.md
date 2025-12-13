# 🧭 KalKalori — Development Roadmap DRAFT

This roadmap defines the **technical evolution** of the KalKalori engine.
The focus is correctness, transparency, and long-term maintainability.

---

## Phase 1 — Core Foundations

* Fluid property layer (CoolProp, IAPWS)
* Unit handling and validation
* Base data structures (geometry, flow states)
* Initial test framework (pytest)

**Outcome:** reproducible thermophysical calculations

---

## Phase 2 — Heat Transfer Models

* LMTD method with correction factors
* Effectiveness–NTU formulation
* Internal flow correlations (Dittus–Boelter, Gnielinski)
* External crossflow correlations (Zukauskas, Kern)

**Outcome:** reliable prediction of heat duty and outlet temperatures

---

## Phase 3 — Pressure Drop Models

* Tube-side pressure loss (Darcy–Weisbach)
* Local losses (inlet, outlet)
* Air-side pressure loss models for tube banks and fins

**Outcome:** coupled thermal–hydraulic performance prediction

---

## Phase 4 — Integrated Heat Exchanger Models

* Bare-tube air coolers
* Finned-tube heat exchangers
* Condensing steam inside tubes

**Outcome:** full exchanger simulation objects

---

## Phase 5 — Validation and Documentation

* Comparison with literature and handbook data
* Reference test cases
* Engineering documentation and theory notes

**Outcome:** engineering-grade credibility

---

## Phase 6 — External Interfaces (separate projects)

* REST API
* Excel add-in
* Commercial UI frontends

**Outcome:** ecosystem around a stable GPL engine
