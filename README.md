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

## Current Capabilities (v0.9.4)

Round individual solid-finned INLINE banks now select ESCOA thermal HTC
(as reproduced by Frass) and native Frass Eq.191 hydraulics automatically.
Mean fin temperature is coupled to the physical film coefficient. Hydraulic
drag uses mean gas state with separate acceleration and external losses.
This is qualified non-strict engineering support: heated normalization is
unapplied, source ranges and loss decomposition remain partial, and strict
mode rejects unresolved required conditions. Staggered Briggs--Young and
Robinson--Briggs behavior is unchanged. See the [thermal guide](docs/inline_round_finned_thermal.md)
and [native hydraulic guide](docs/inline_round_finned_hydraulics.md).

The explicitly selected public [Chakroun–Al-Fahed 1995 twisted-tape provider](docs/twisted_tape_chakroun1995.md)
supports paired laminar liquid heat transfer and distributed friction using
actual tape-width blockage and coupled wall viscosity, with source-domain
diagnostics and explicit Fanning-to-Darcy conversion. Cooling remains an
extrapolation; arbitrary-clearance validity is not established.

- Rectangular-finned elliptical baseline: `RectangularIndividualFin` composes with
  `EllipticalTube` through `RectangularFinnedEllipticalTube`. Exact physical
  surface components and deterministic `ThinPlateFinEfficiency2D` conduction
  feed effective area into the common thermal resistance network for a
  physical film coefficient. Normal dry Rating/Simulation can select the
  Zhao-2017 engineering baseline, with explicit inferred diameter, reconstructed
  passage velocity, source-domain pressure estimate and source-geometry warnings.
  The source fit covers N=6–10; other positive row counts calculate with
  extrapolation warnings, or reject in optional strict mode. The current
  internal ellipse path automatically selects laminar H1 or turbulent Yang T
  from Re, with an explicit unsupported transition interval. General elliptical
  walls use numerical conduction automatically; confocal walls admit the exact
  solution. Yang source-Re exceedance calculates with a warning by default.
  Fin width is transverse and height streamwise (a pre-release API correction). See the
  [geometry, physics and limitations guide](docs/rectangular_finned_ellipse.md).

- Smooth mathematical `EllipticalTube` geometry with exact perimeter and
  hydraulic diameter, explicit fully developed laminar Darcy friction and H1
  internal heat transfer, turbulent Yang T, and automatic numerical/exact wall selection.
  These are standalone foundation calculations; automatic bare-ellipse-bank
  Rating/Simulation remains unsupported. See [elliptical tubes](docs/elliptical_tubes.md).

- Independent inside/outside thermal fouling resistances [m² K/W] on the
  exchanger, with omitted/`None` values resolving to zero. Rating and
  Simulation include them in the solved dry/wet thermal circuit, including
  finned and tube-side phase-change paths. Installed hydraulic geometry
  remains unchanged; see [thermal fouling](docs/thermal_fouling.md).

- Rating and Simulation expose a common surface-margin result contract:
  `UA_actual` is the full actual exchanger UA, `UA_process` is the UA required
  or consumed by the reported process state, and
  `overdesign_factor = UA_actual / UA_process - 1`. Simulation keeps
  `surface_margin` as its backward-compatible input derating.
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
  inside tubes with a dry outside surface or outside a bare or circular-finned
  tube bank, with automatic detection
  (`PhaseChangeMode.AUTO`) and a dry-only override
  (`PhaseChangeMode.DISABLED`); see
  [`docs/property_models.md`](docs/property_models.md#15-v061--wet-gas-water-condensation)
- Bare tube heat exchangers
- Circular-finned-tube exchangers. `CircularFinnedTube` composes a `BareTube`
  core and supports constant or linearly tapered annular fins,
  welded or continuous-root construction, ideal contact by default, and
  either a practical dimensionless `fin_contact_efficiency` or the advanced
  physical `fin_contact_resistance` for non-ideal fin/root contact,
  Briggs--Young dry outside heat transfer and verified-scope Robinson--Briggs
  pressure loss for staggered banks, the qualified INLINE models linked above,
  Simulation, Rating, and the established inside-side phase
  change paths when the finned outside surface remains dry. The unchanged dry
  Briggs--Young heat-transfer provider accepts every positive bank row count;
  one- to
  three-row results carry an explicit unvalidated-extrapolation warning and
  have no row-count correction. See
  [`docs/finned_tube_model.md`](docs/finned_tube_model.md). For outside H2O
  condensation from a non-condensable carrier gas, the same geometry supports
  a nonlinear radial annular-fin calculation with independently condensing
  primary and fin surfaces, typed dry/partially-wet/fully-wet diagnostics,
  separate sensible/latent duties, drained-condensate balance, Simulation,
  Rating and `PhaseChangeMode.AUTO`.
- Tube-side forced convection
- Outside forced convection (mass-flow driven)
- Multi-pass tube bundles
- Thermal analysis using ε–NTU method:
  - counterflow
  - cocurrentflow
  - crossflow (outside mixed, inside unmixed)
  - topology-aware AUTO: equal-UA countercurrent series of crossflow sections
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

Outside pressure-change support includes the calculated plain or dry
circular-finned tube bank and an explicitly reconstructed Zhao source-domain
estimate for rectangular-finned ellipses. The latter is not a validated
bank-only irreversible loss. During active wet-finned condensation the
reported finned-bank value is explicitly a dry pressure-drop reference, not an
actual wet-bank pressure drop. Duct, plenum,
casing-transition, screen, louver, damper, fan, and external-piping losses are
not included.

### Tube circuit topology

`TubeBundle.n_passes_tube` is the total number of tube-side passes, while
`n_passes_transverse` is the number of passes grouped within each longitudinal
section of the outside bank. The latter must divide the former, and
`n_sections_longitudinal` is their quotient. Rows need not divide exactly
between those sections: `n_rows_per_section_effective` reports the possibly
non-integer effective 0D average, while `rows_partition_is_exact` indicates
whether the physical partition is exact.

With the default `flow_arrangement="auto"`, a single longitudinal section
resolves to global `crossflow`, and one transverse pass per section with more
than one tube-side pass resolves to global `counterflow`. Intermediate
circuits containing both multiple transverse passes and multiple longitudinal
sections use an internal N-section counter-crossflow model: each section has
`UA_total/N`, outside mixed / inside unmixed crossflow, and complete mixing
between sections. This restricted dry model retains full serial capacity
rates and countercurrent section coupling. An explicit `crossflow`,
`counterflow` or `cocurrentflow` value, including a method-level override,
always overrides AUTO resolution. No additional public arrangement string is
needed. The compatibility `flow_arrangement_resolved` label stays `crossflow`
for this branch; dry results expose the actual `thermal_topology_model`,
`n_thermal_sections`, and `inter_section_mixing`.

`row_parity_mode="reset"` (default) preserves repeated physical row modules
at exact section boundaries and the legacy global fallback for nonexact row
partitions. `"continuous"` numbers alternating rows across the whole bank,
independently of thermal boundaries. For 6 rows in two 3-row modules with
10/9 tubes per row, reset gives 58 tubes and continuous gives 57. Areas and
tube-side flow geometry use that physical count; transverse frontal geometry
retains its periodic row average.

See [the thermal topology guide](docs/thermal_topology.md) for equations,
public provenance, reachability limits, diagnostics and dry-only scope.

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

- `core/tests`  
  Automated validation and regression tests

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

➡️ [`docs/roadmap.md`](docs/roadmap.md)

In short:
- **0.x** — model calibration and accuracy improvements
- **1.0.0** — stable, production-ready 0D engine
- **2.0.0** — segmented / distributed (1D) solver

---

## Getting Started

KalKalori is intended to be used as a Python library.

Typical usage:
1. Define geometry (`BareTube`, `CircularFinnedTube` or a supported
   `RectangularFinnedEllipticalTube` configuration, then
   `TubeBundle`)
2. Define energy streams
3. Solve using a heat exchanger model
4. Inspect results from `HXResult`

For the tube-side enhancement API, including public Yang2020 and Rossi2017
twisted-tape models, the released Inaba1994 wire-coil model, and external
providers, see
[`docs/tube_side_enhancements.md`](docs/tube_side_enhancements.md).
The current package version is v0.8.8, closing the planned 0.8.x line with
shared tube-enhancement deadlines, independent required wall states and
explicit provider failures during endpoint probes. It retains
[thermal fouling resistance support](docs/thermal_fouling.md) and the
[BareTube wet numerical performance improvements](docs/wet_numerical_performance.md).
Smooth laminar flow with a supplied
heated length now includes [thermal development](docs/internal_laminar_thermal_development.md).

For wet coils, Elmahdy-Mitalas is the default for Simulation and thermal-area
Rating. Legacy bulk-mean Simulation and external provider objects are explicit
selections. Hydraulics use installed geometry; wet-film pressure-drop
correction is not implemented. See [the wet-coil provider guide](docs/wet_coil_providers.md).

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
