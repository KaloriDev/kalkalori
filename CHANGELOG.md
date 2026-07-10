# Changelog

All notable changes to KalKalori are documented in this file.

The project follows **Semantic Versioning (SemVer)**:
`MAJOR.MINOR.PATCH`.

---
## [0.5.1] - Simulation/Rating split, heat-balance closure, surface margin

### Changed (breaking, internal API)

- Renamed the previous mean-property "rating" feature to **Simulation**,
  matching design-practice terminology (known geometry + both inlet
  temperatures + flow rates -> achievable outlet temperatures):
  - `BareTubeHeatExchanger.rate(...)` -> `BareTubeHeatExchanger.simulate(...)`
  - `run_rating(...)` -> `run_simulation(...)`
  - `HXRatingResult` -> `HXSimulationResult`
  - `RatingSideInput` -> `HXSideInput`
  - `core/models/mean_property_rating.py` -> `core/models/simulation.py`
  - `core/tests/mean_property_rating_smoke.py` -> `core/tests/simulation_smoke.py`

### Added

- Added `surface_margin: float = 0.0` to `simulate(...)` (derating input):
  `UA_eff = UA_full/(1+surface_margin)` -> `NTU_eff` -> `eps` -> `Q` -> `T_out`.
  `surface_margin=0.0` reproduces the previous behavior bit-for-bit for
  constant properties. Result gains `surface_margin`, `Q_full`, `Q_derated`.
- Added **Rating** (overdesign / surface margin as an output), reclaiming the
  `.rate(...)` / `run_rating(...)` / `HXRatingResult` names for their correct
  meaning: given geometry and a *closed* heat balance, report
  `overdesign_factor = A_o/A_required - 1` (and `ua_margin`, "for free").
  `U` is evaluated once at the closed balance's working conditions and held
  constant in the area-margin arithmetic (standard over-surface practice;
  a property-iteration refinement of `U` itself is out of scope for v0.5.1).
  New `core/models/rating.py`.
- Added heat-balance closure (`core/models/heat_balance.py`):
  `BalanceSideSpec` (partially known per-side flow/temperatures),
  `close_heat_balance(...)` resolving `Q` (explicit, from a fully specified
  side, or from `effectiveness`) and then each side's missing `m_dot`/`T_out`,
  and `ClosedBalance`/`ClosedBalanceSide` (with a bridge back to
  `HXSideInput` so a closed balance can be re-run through `simulate()` for
  comparison via `rate(..., include_simulation=True)`). Under-specified
  balances raise `ValueError`; over-specified (mismatched) balances emit a
  `ModelWarning` (`heat_balance_over_specified`) rather than failing.
- Added `ntu_from_effectiveness(...)` to `core/heat_transfer/ntu.py`, the
  inverse of `effectiveness_ntu(...)`, for counterflow, cocurrentflow and
  crossflow, with a guard raising `ValueError` when the requested
  effectiveness exceeds the arrangement's thermodynamic maximum regardless
  of area.
- Added dependency-free smoke test
  (`core/tests/heat_balance_rating_smoke.py`) covering closure variants,
  the `ntu_from_effectiveness` round-trip + guard, and Rating (overdesign
  ~= 0/greater/less, plus the `A_required == A_o` consistency check against
  a `simulate()`-derived operating point).

---
## [0.5.x] - Iterative Mean-Property HX Rating

### Added

- Added `BareTubeHeatExchanger.rate(...)` as the default rating entry point,
  running an iterative mean-property outer loop around the existing `solve()`
  kernel.
- Added `RatingSideInput` / `HXRatingResult` carrying per-side boundary
  conditions and full rating diagnostics (convergence residuals, mean bulk
  temperatures, mean transport properties, velocity, Re, Pr, alfa, U_mean,
  UA, q, pressure drop).
- Added `run_rating` driver (`core/models/mean_property_rating.py`);
  `run_rating`, `RatingSideInput`, `HXRatingResult` exported from `core.models`.
- Added single-pass shortcut: when both sides use `ConstantPropertyProvider`
  (or `iterate=False` is passed), rating collapses to one `solve()` pass —
  with constant properties UA and C do not depend on guessed outlet
  temperatures, so one pass is exact.
- Added dependency-free smoke test (`core/tests/mean_property_rating_smoke.py`).
- Added demonstration notebook (`core/tests/dry_gas_gas_mean_property_test.ipynb`)
  on real dry-air / gas-mixture providers, comparing mean-property vs
  inlet-only rating and reporting velocity and pressure drop for both sides.

### Notes

- `solve()` remains the single-pass physics kernel; `rate()` is a driver and
  does not duplicate any correlation.
- Convergence defaults: `max_iter=30`, `temperature_tolerance_K=0.05`,
  `relative_duty_tolerance=1e-4`, `relaxation_factor=0.5`.
- Non-convergence returns the last iterate with `converged=False` and a
  `ModelWarning` (`mean_property_rating_not_converged`) instead of raising
  or hanging.
- Out of scope for this stage: condensation, wet surface, latent heat, water
  removal, acid dew point, full wet economizer, row-by-row/segmented models,
  wall-temperature iteration.

---
## [0.4.x] - Property Layer Foundation

### Added

- Added initial property layer for future wet economizer and gas-gas calculations.
- Added PsychroLib-based moist-air helpers.
- Added dew-point, saturation, and condensation-onset helpers for moist air.
- Added IAPWS-IF97 water/steam property provider.
- Added optional CoolProp backend for pure fluids and mixtures.
- Added CoolProp backend availability checks for `HEOS` and optional `REFPROP`.
- Added gas-mixture property provider with mole, volume, and mass composition bases.
- Added gas-phase imposition for gas-mixture calculations.
- Added dedicated dry-air property provider.
- Added common mass-flow / actual-volume-flow helpers.
- Added helper for converting dry gas composition plus water ratio to `GasMixtureSpec`.
- Added property-model selection documentation.
- Added functional test notebooks for moist air, water/steam, CoolProp, dry air, and gas mixtures.

### Changed

- Corrected moist-air enthalpy unit handling.
- Clarified SI unit conventions across the property layer.
- Clarified when to use dry air, moist air, gas mixtures, and water/steam property paths.
- Clarified third-party dependency and licensing notes.

### Notes

- `v0.4.x` focuses on the property foundation, not on full heat exchanger solver refactoring.
- High-temperature humid gas should be represented explicitly as a gas mixture with `H2O` as a gas-phase component.
- Gas-mixture calculations do not model condensation, latent heat, wet-surface heat transfer, or water removal.
- Large-temperature-change gas-gas rating requires iterative mean-property calculation and is planned for `v0.5.x`.


---
## [0.3.x] — Outside Flow Model Accuracy

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


## Versioning Notes

- Versions below `1.0.0` indicate an evolving API.
- `1.0.0` marks a stable, validated 0D modelling core.
- `2.0.0` introduces a new physical modelling paradigm.
- Version numbers are defined by Git tags and this changelog, not by per-file metadata.