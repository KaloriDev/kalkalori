# Circular Finned Tubes (v0.7.x, experimental)

**Status: experimental, not released.** This document describes the
circular finned-tube feature developed on the experimental branch
`experiment/v0.7.x-finned-tubes-claude`, starting from commit `0475635`.
It is **not** merged into `main`, not tagged, and the package version
in `pyproject.toml` is unchanged. See `docs/roadmap.md` for the
`v0.7.x — Finned Tubes` roadmap entry this branch implements.

## 1. Scope

Implemented:
- full (non-segmented) circular round-tube geometry: root/foot collar
  layer, constant-thickness or linearly root-to-tip tapered fins,
- fin surface areas, fin metal volume, fin-blockage-aware minimum
  free-flow area / maximum velocity,
- annular-fin efficiency (closed-form Bessel solution for constant
  thickness; deterministic numerical ODE solver for tapered fins),
- an explicit resistance network (inside convection, core-tube wall
  conduction, root/foot-layer conduction, contact resistance, parallel
  primary/fin convection),
- a dry, single-phase, gas-side outside heat-transfer coefficient
  (Briggs & Young, 1963),
- integration into `BareTubeHeatExchanger.solve/simulate/rate` for
  staggered (triangular pitch) layouts,
- a controlled rejection of wet/condensing outside providers on a
  finned tube.

**Not implemented** (documented blocker, see §6): the Robinson & Briggs
(1966) outside pressure-drop correlation. The provider architecture
exists and is geometry/layout-gated, but raises `NotImplementedError`
rather than a guessed formula.

Out of scope entirely for this feature (see the task brief and
`docs/roadmap.md`): segmented/serrated fins, wavy fins, continuous
plate lamella spanning multiple tubes, elliptical/flattened tubes, wet
finned surfaces, condensation on fins, frost/ice, condensate retention
and runoff, radiation, detailed fin-root joint mechanics.

A full helical/spiral fin is modeled as an **equivalent periodic array
of full annular (ring) fins** of pitch `fin_pitch`. This is an explicit
approximation: the true helical lead angle and its (generally small,
for low lead angles typical of finned tubes) effect on local
conduction/flow is not resolved.

## 2. Geometry: composition, not duplication

`core.geometry.finned_tube.CircularFinnedTube` composes an existing
`BareTube` as `core_tube` -- the single source of truth for `D_i`,
`D_o`, `length_total`, `length_effective`, `wall_k`, roughness, and
`tube_orientation`. It never copies or re-validates those fields.

```python
from core.geometry import BareTube, CircularFinnedTube

core = BareTube(D_i=0.020, D_o=0.025, length_total=3.0,
                 length_effective=3.0, wall_k=45.0)

tube = CircularFinnedTube(
    core_tube=core,
    fin_k=200.0,
    D_fin=0.057,
    D_root=0.025,               # == core.D_o for a welded fin
    fin_thickness_root=0.0004,
    fin_pitch=0.0023,
    # fin_thickness_tip=None -> constant thickness
    # fin_contact_resistance=None -> ideal contact, warns
    # external_area_per_length=None -> geometric area used
)
```

`BareTube.surface_type == TubeSurfaceType.PLAIN`;
`CircularFinnedTube.surface_type == TubeSurfaceType.CIRCULAR_FINNED`.
`TubeBundle`/`BareTubeHeatExchanger` dispatch on
`isinstance(bundle.tube, CircularFinnedTube)`, unchanged for
`BareTube`.

## 3. Definitions used consistently throughout

| Quantity | Definition |
|---|---|
| `D_root` | Fin root/collar outer diameter. `D_o <= D_root < D_fin`. `D_root == core_tube.D_o` for a welded fin; `D_root > core_tube.D_o` for an extruded fin with a foot layer. |
| `fin_height` | `(D_fin - D_root) / 2` |
| `root_radial_thickness` | `(D_root - D_o) / 2`; the extruded foot-layer conduction path, zero for welded fins |
| `fin_pitch` | axial center-to-center repeat distance between fins |
| `clear_spacing_root` | `fin_pitch - fin_thickness_root`, the exposed bare-root axial gap |
| `effective_fin_count` | `length_effective / fin_pitch`, **not rounded** (avoids 0D thermal discontinuities for small length changes); a rounded value is available separately as `nominal_fin_count_diagnostic` and never feeds the thermal result |
| `A_primary` | exposed root/base surface between fins |
| `A_fin` | total fin area (both sloped/flat side faces + tip edge), all fins |
| `A_outside_geometric` | `A_primary + A_fin`, always computed from geometry |
| `A_outside_gross` (== `area_outer`) | the *used* total outside area: `A_outside_geometric` unless `external_area_per_length` overrides it |
| `A_outside_effective` | fin-efficiency-weighted area, `A_primary + eta_fin*A_fin_used`, computed in the resistance network, not on the tube itself (it depends on the flow condition through `eta_fin`) |
| `V_max` / Reynolds basis | fin-blockage-aware minimum-free-flow velocity (`core.geometry.finned_flow_geometry`), **not** the bare-tube `S_T/(S_T-D)` ratio with `D_fin` substituted in; Reynolds number uses `D_root`, never `D_fin` |

## 4. Fin efficiency

`core.heat_transfer.fin_efficiency` solves the standard annular-fin
conduction/convection equation
`d/dr[k_fin*A_c(r)*dtheta/dr] - h_o*P(r)*theta = 0` with a base Dirichlet
condition and a **convective tip** condition (both flat/sloped faces
convect; the tip edge convects with `fin_thickness_tip`).

- **Constant thickness**: closed-form Gardner (1945) / Kern-Kraus
  Bessel-function solution, using the standard corrected-radius
  approximation (`r_e -> r_e + t/2`) to approximate the convective tip
  within that classical (originally adiabatic-tip) closed form.
- **Tapered** (`fin_thickness_tip < fin_thickness_root`): deterministic
  numerical solution (RK4 integration + linear-superposition shooting
  -- exact for this linear BVP, no iterative root-finding, no new
  production dependency).
- **Cross-validation**: in the constant-thickness limit, the two agree
  to ~2e-5 relative for representative finned-tube geometries (see
  `core/tests/finned_tube_fin_efficiency_test.py`). Modified Bessel
  functions (`core.heat_transfer.modified_bessel`, Abramowitz & Stegun
  1964 polynomial approximations) are independently verified against
  known reference values and the exact Wronskian identity
  `I0(x)K1(x)+I1(x)K0(x) = 1/x`.
- Direction/limit checks: higher `fin_k` -> higher efficiency; higher
  `h_o` -> lower efficiency; taller fin -> lower efficiency; small-fin
  or high-conductivity limit -> efficiency -> 1.
- The tapered solver's result is **not** approximated by averaging
  `fin_thickness_root`/`fin_thickness_tip` into one constant thickness;
  a test asserts the two differ meaningfully (`finned_tube_fin_efficiency_test.py::
  test_tapered_fin_differs_from_naive_mean_thickness_shortcut`).

## 5. Resistance network

`core.heat_transfer.finned_tube_resistance.build_finned_tube_resistance_network`
explicitly separates, per bundle of `n_tubes`:

```
R_total = R_inside_convection
        + R_wall_conduction          (core tube wall, existing formula)
        + R_root_conduction          (0 when D_root == D_o)
        + R_contact                  (0 when ideal/unspecified)
        + R_outside_convection       (parallel combination below)

R_outside_convection = 1 / (1/R_primary_convection + 1/R_fin_convection)
R_primary_convection = 1 / (alfa_o_physical * A_primary)
R_fin_convection      = 1 / (alfa_o_physical * fin_efficiency * A_fin_used)
```

Fin efficiency and contact resistance are each applied exactly once
(verified by
`finned_tube_resistance_network_test.py::test_fin_efficiency_and_contact_resistance_not_double_counted`).
`fin_contact_resistance=None` assumes ideal (zero) contact **and**
raises an explicit warning (`finned_tube_contact_resistance_unknown`)
-- contact resistance is never inferred from fin manufacturing
technology (welded/extruded/wrapped).

`conduction_and_contact_resistance` (the alfa-independent conduction
path) is reused by both this network and
`BareTubeHeatExchanger.tube_wall_resistance()` (which becomes
finned-aware for a `CircularFinnedTube`), so the two never drift apart.

`alfa_o_gross_basis` is the equivalent overall-surface-efficiency-
weighted HTC on the `A_outside_gross` basis, i.e. the value threaded
into the *unmodified* `R_o = 1/(alfa_o*A_o)` formula already used
throughout `core.heat_transfer.thermal_iteration` and
`core.models.bare_tube` -- this is how a finned tube plugs into the
existing pipeline without changing that formula's code.

## 6. Correlations and their contracts

| Model | Source | Application | Reference diameter | Reference velocity | Re range | Layout | Area basis | Limitations |
|---|---|---|---|---|---|---|---|---|
| `nusselt_briggs_young` | Briggs & Young (1963), CEPS 59(41) | dry outside HTC, circular finned tubes | `D_root` | fin-blockage-aware `V_max` | 1000-8000 (reported) | staggered (triangular pitch) only | physical, on true finned surface (not yet fin-efficiency-weighted) | constant-thickness fin dataset; tapered fins flagged as outside the source dataset (info warning) |
| `RobinsonBriggsEulerProvider` | Robinson & Briggs (1966), CEPS 62(64) | dry outside Δp, circular finned tubes | `D_root` (geometry-gate only) | -- | ~2000-50000 (reported, not implemented) | staggered only (geometry gate implemented) | -- | **documented blocker**: raises `NotImplementedError`; exact closed-form equation not independently verified in this pass |
| `ZukauskasEulerProvider` / `GaddisGnielinskiEulerProvider` (existing) | -- | bare tubes only | -- | -- | -- | inline/staggered | -- | explicitly reject `is_finned=True` (unchanged) |

`FinnedCorrelationContract` (`core.heat_transfer.outside_flow_finned.HTC_CONTRACT`)
self-reports method/source/geometry family/velocity basis/Reynolds
basis/reference diameter/area basis/row basis, per the same
introspectable-contract convention as the existing `EulerRequest`/
`EulerResult` architecture (`core.pressure_drop.outside_pressure_drop`),
which was extended (not modified) with `RobinsonBriggsEulerProvider`
following the exact reserved-stub pattern already used there for
`EsduEulerProvider`.

Automatic provider selection by geometry
(`resolve_finned_euler_provider`): a caller that leaves `euler_provider`
at the shared bare-tube default (`"zukauskas"`) gets `"robinson_briggs"`
automatically for a finned tube. A caller who explicitly names a
bare-tube provider (or passes a `ZukauskasEulerProvider`/
`GaddisGnielinskiEulerProvider` instance directly) still gets that
provider's own controlled `ValueError` from its `is_finned` gate --
never a silent substitution.

## 7. Wet-surface / condensation guard

`BareTubeHeatExchanger._reject_finned_tube_wet_outside_surface` runs at
the top of `.simulate()`/`.rate()`: if `bundle.tube` is a
`CircularFinnedTube` and `detect_phase_change_capability(outside.provider)`
reports the outside provider as phase-change capable (wet gas mixture
or pure IAPWS water/steam), it raises
`FinnedTubeWetOutsideSurfaceNotSupportedError` **before** any wet-
surface solve is attempted. Inside (tube-side) phase change is
unaffected -- fins in this feature are outside-only.

## 8. Known simplifications versus the bare-tube path

- **Single-state outside evaluation.** `BareTube` outside hydraulics use
  a 3-state (inlet/midpoint/outlet, Simpson-integrated) treatment
  (`core.heat_transfer.outside_flow.calculate_outside_tube_bank_hydraulics`).
  The finned path (`finned_outside_flow_from_mass_flow`) is
  single-state (evaluated at the mean bulk state) in this pass. This
  is a real, documented scope reduction, not a hidden approximation:
  `outside_side_hydraulic.tube_bank` is `None` for a finned tube, and
  `dp_drag`/`dp_acceleration` correctly report NaN (they depend on the
  3-state result) via the existing `tube_bank is None` fallback
  already used when the outside side is unspecified.
- **No outside wall-property correction.** Briggs & Young (1963) has no
  documented `(Pr/Pr_s)^n`-style wall correction (unlike Zukauskas), so
  none is applied.
- **Tapered-fin dimensionless groups.** The Briggs & Young `(s/l)`/`(s/b)`
  groups use `fin_thickness_root` as the representative thickness for a
  tapered fin (an info-level applicability warning is raised); the fin
  *efficiency* itself does resolve the true taper (§4).
- **Fin-blockage V_max uses average thickness.** `finned_blocked_equivalent_diameter`
  uses `(fin_thickness_root + fin_thickness_tip)/2` as the representative
  axial footprint for a tapered fin's flow blockage -- a documented,
  controlled approximation, not an attempt at an exact swept-volume
  integral for this specific purpose.

## 9. Unresolved limitations

1. **Robinson & Briggs (1966) pressure drop is not implemented.** Despite
   a genuine, multi-source research effort (see `docs/references.md`),
   the exact closed-form Euler/friction-factor equation could not be
   independently confirmed to this project's standard. Even the
   rigorously-sourced open-source `ht`/`fluids` libraries (which do
   implement Briggs & Young 1963 precisely) do not implement Robinson &
   Briggs (1966) either -- corroborating that this is a genuine
   accessibility gap, not an oversight. `RobinsonBriggsEulerProvider`
   is a ready, geometry-gated stub for a future contributor with
   primary-source access to CEPS 62(64), pp. 177-184.
2. **No 3-state outside hydraulic snapshot for finned tubes** (§8).
3. **Helical fin approximated as a periodic ring array** (§1); the true
   lead-angle effect is not resolved.
4. **No row-position/row-count correction** analogous to Zukauskas'
   `finite_row_correction_c2` is applied to the finned Nusselt number;
   Briggs & Young (1963) itself does not report one distinctly from its
   correlation dataset (rows >= ~2-4 per various secondary sources).
5. **Contact resistance and detailed fin-root joint mechanics** are
   modeled only as a single area-basis resistance value, per the
   explicit task scope.

## 10. Notebook

`core/tests/finned_tube_dry_air_cooler_example.ipynb` demonstrates: a
welded constant-thickness tube and an extruded tapered tube, their
geometry outputs, fin efficiency, Briggs & Young HTC, the pressure-drop
blocker's controlled `NotImplementedError`, and a complete dry
`BareTubeHeatExchanger.simulate()` run.
