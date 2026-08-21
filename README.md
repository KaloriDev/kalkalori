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

## Current Capabilities (v0.7.4)

- Rating can derive the required tube-side pure-steam mass flow from an
  independently known duty (explicit `Q`, or a fully specified opposing-side
  temperature program) and a requested steam outlet state, defaulting to
  complete condensation (`quality_out=0.0`) when no outlet target is given.
- Pure-water heating and evaporation inside tubes with a dry outside surface,
  including liquid
  preheating, partial or complete evaporation, and optional vapor
  superheating through Simulation and Rating. The IAPWS-backed model uses
  Shah (1982) saturated flow boiling with a self-consistent inside-area heat
  flux; see
  [`docs/property_models.md`](docs/property_models.md#17-v063--pure-water-heating-and-evaporation-inside-tubes)

- Pure water/steam cooling inside tubes with a dry outside surface, including superheated steam,
  saturated or wet-steam inlets, partial/complete condensation, and optional
  condensate subcooling through the public Simulation and Rating APIs. The
  IAPWS-backed model includes gravity-aware low-mass-flux Shah (2009)
  condensation; see
  [`docs/property_models.md`](docs/property_models.md#16-v062--pure-watersteam-cooling-inside-tubes)
- Partial H2O condensation from a water-containing gas mixture either
  inside tubes with a dry outside surface or outside a bare-tube bank, with automatic detection
  (`PhaseChangeMode.AUTO`) and a dry-only override
  (`PhaseChangeMode.DISABLED`); see
  [`docs/property_models.md`](docs/property_models.md#15-v061--wet-gas-water-condensation)
- Bare tube heat exchangers
- Dry circular-finned-tube exchangers. `CircularFinnedTube` composes a
  `BareTube` core and supports constant or linearly tapered annular fins,
  welded or continuous-root construction, ideal contact by default, and
  either a practical dimensionless `fin_contact_efficiency` or the advanced
  physical `fin_contact_resistance` for non-ideal fin/root contact,
  Briggs--Young dry outside heat transfer, verified-scope Robinson--Briggs
  pressure loss, Simulation, Rating, and the established inside-side phase
  change paths when the finned outside surface remains dry. The Briggs--Young
  heat-transfer provider accepts every positive bank row count; one- to
  three-row results carry an explicit unvalidated-extrapolation warning and
  have no row-count correction. See
  [`docs/finned_tube_model.md`](docs/finned_tube_model.md). Wet/condensing
  finned outside surfaces remain unsupported.
- Tube-side forced convection
- Outside forced convection (mass-flow driven)
- Multi-pass tube bundles
- Thermal analysis using ε–NTU method:
  - counterflow
  - cocurrentflow
  - crossflow (both sides mixed, lumped 0D)
- Three-state variable-property tube-side pressure change:
  - distributed straight-tube friction
  - signed acceleration pressure change
- Three-state variable-property outside tube-bank pressure change:
  - irreversible crossflow bank drag
  - signed inlet-to-outlet acceleration pressure change
- Outside reference-velocity, Euler-number, and inlet/midpoint/outlet
  hydraulic diagnostics
- Public inside/outside inlet, midpoint, and outlet fluid-property
  diagnostics on `HXResult`, `HXSimulationResult`, and `HXRatingResult`
- Explicit geometry modelling:
  - tube dimensions
  - effective vs total length
  - bundle layout
- Complete result snapshot (`HXResult`):
  - inputs
  - geometry
  - thermal performance
  - hydraulic performance

Outside pressure change is limited to the calculated plain or dry
circular-finned tube bank. Duct, plenum,
casing-transition, screen, louver, damper, fan, and external-piping losses are
not included.

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
1. Define geometry (`BareTube` or `CircularFinnedTube`, then
   `TubeBundle`)
2. Define energy streams
3. Solve using a heat exchanger model
4. Inspect results from `HXResult`

See the [local pressure-drop examples](tests/local_pressure_drop_examples.ipynb)
for straight sections, transitions, elbows, planar obstructions, user-defined
losses, and explicit path assembly.

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


## Property Model Selection

Before selecting a property provider, see:

[`docs/property_models.md`](docs/property_models.md)
