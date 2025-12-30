# KalKalori — Heat Exchanger Open Engine

**KalKalori** is an open-source computational core for **thermal and hydraulic
design of heat exchangers**, with a strong focus on engineering correctness,
clear physical assumptions, and long-term extensibility.

The project is designed to serve:
- engineering and R&D teams,
- academic and student projects,
- as a reliable computational core for commercial tools.

KalKalori deliberately separates **physics and numerics** from **UI, APIs,
and serialization**, enabling both open collaboration and commercial adoption.

---

## Project Scope

### What KalKalori *is*

- A **numerical engine** for heat exchanger calculations
- Focused on:
  - heat transfer,
  - pressure drop,
  - geometry-driven modelling
- Built around **explicit physical assumptions**
- Intended as a reusable **core library**, not an end-user application

### What KalKalori *is not*

- Not a UI or GUI tool
- Not a complete process simulator
- Not a replacement for CFD
- Not a monolithic “all-in-one” application

---

## Current Capabilities (v0.3.x)

- Bare tube heat exchangers
- Tube-side forced convection
- Outside forced convection (mass-flow driven)
- Multi-pass tube bundles
- Thermal analysis using ε–NTU method:
  - counterflow
  - cocurrentflow
  - crossflow (both sides mixed, lumped 0D)
- Detailed tube-side pressure drop:
  - friction losses
  - inlet losses
  - outlet losses
  - return (pass) losses
- Explicit geometry modelling:
  - tube dimensions
  - effective vs total length
  - bundle layout
- Complete result snapshot (`HXResult`):
  - inputs
  - geometry
  - thermal performance
  - hydraulic performance

---

## Modelling Philosophy

KalKalori prioritizes **trustworthy results over model complexity**.

The development path follows this principle:
1. Accurate 0D calculations
2. Well-documented applicability limits
3. Incremental physical extensions
4. Segmentation and distributed models only when justified

This mirrors proven industrial practice rather than academic novelty.

---

## Architecture Overview

High-level structure:

- `geometry`  
  Defines exchanger construction (tubes, bundles, layout, flow arrangement)

- `heat_transfer`  
  Correlations for heat transfer, pressure drop, and thermophysical behaviour

- `models`  
  Orchestrates geometry and correlations into usable exchanger models

- `notebooks`  
  Reference examples and validation studies (documentation by example)

The core contains **no UI code, no JSON serialization, and no external I/O**.

---

## Licensing

KalKalori is released under the **GNU General Public License v3 (GPLv3 only)**.

This ensures:
- openness of the computational core,
- freedom for academic and collaborative development,
- clear separation from proprietary extensions.

Commercial usage is possible, including:
- internal engineering tools,
- external services,
- proprietary extensions (as separate modules).

---

## Roadmap

The development roadmap is documented in detail in:

➡️ [`roadmap.md`](roadmap.md)

In short:
- **0.x** — model calibration and accuracy improvements
- **1.0.0** — stable, production-ready 0D engine
- **2.0.0** — segmented / distributed (1D) solver

---

## Getting Started

KalKalori is intended to be used as a Python library.

Typical usage:
1. Define geometry (`BareTube`, `TubeBundle`)
2. Define energy streams
3. Solve using a heat exchanger model
4. Inspect results from `HXResult`

See the example notebooks for reference workflows.

---

## Project Status

KalKalori is **actively developed**.

The API is stabilizing but may still evolve until `v1.0.0`.
Feedback, validation studies, and contributions are welcome.

---

## Contributing

Contributions are welcome from:
- industry engineers,
- researchers,
- students.

Please see `CONTRIBUTING.md` for guidelines.

---

## Disclaimer

KalKalori is provided **without any warranty**.

Results must always be validated against:
- engineering judgment,
- applicable standards,
- experimental or vendor data where required.

The authors assume no liability for the use of results in real-world designs.
