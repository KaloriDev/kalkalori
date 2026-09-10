# Tube-side enhancements (unreleased v0.8.x)

The feature branch adds a coherent provider for inside heat transfer and
distributed straight-tube friction. The package version remains v0.7.10
until an explicitly authorized release. Omitting `tube_side_enhancement`
(or setting it to `None`) retains the exact legacy smooth-tube route.

## Selecting the public model

```python
from core.enhancements import (
    TubeSideEnhancement, TwistedTapeGeometry, Yang2020TwistedTapeProvider,
)
from core.models.bare_tube import BareTubeHeatExchanger

# bundle is an existing TubeBundle with the matching dimensions below.
enhancement = TubeSideEnhancement(
    provider=Yang2020TwistedTapeProvider(),
    geometry=TwistedTapeGeometry(
        half_turn_length=0.036,  # axial length for a 180-degree turn, metres
        tape_width=0.012,
        tape_thickness=0.001,
    ),
    fluid_phase="liquid",
)
hx = BareTubeHeatExchanger(bundle, tube_side_enhancement=enhancement)
# Use the existing hx.simulate(...) or hx.rate(...) inputs and options.
```

The configuration is shared by thermal iteration, wall probes, Rating,
Simulation (also `iterate=False`), and inlet/midpoint/outlet hydraulics.
It works with the existing bare and circular-finned outside-surface routes.
Tube wall heat-transfer area and surface-margin definitions are unchanged;
the tape does not add a conductive fin area.

`result.tube_side_enhancement` exposes the authoritative thermal
`EnhancementResult`. On iterated results it comes from `thermal_state`,
rather than the preliminary hydraulic snapshot. The hydraulic points
`inside_properties_inlet`, `inside_properties_midpoint` and
`inside_properties_outlet` each carry their own `.enhancement` result.
Their legacy point velocity/Re fields retain the empty-tube basis used by
existing local losses; read `.enhancement.reference` for provider references.
All provider warnings propagate to the normal result `warnings` collection.

## Public model and limits

The implementation authority is the freely accessible publisher article
by Yang et al. (2020),
[DOI 10.3389/fenrg.2020.00178](https://doi.org/10.3389/fenrg.2020.00178).
Only the ordinary single-tape specialization of its new Eqs.21-22 is used.
It is **not** the canonical Manglik-Bergles 1993 model or a cross-tape model.
The [source audit](twisted_tape_manglik_bergles.md) records equations,
definitions, open-access verification and unresolved regimes.

| Quantity | Supported public scope |
| --- | --- |
| Fluid/heat flow | Single-phase liquid heating; incompressible forced convection |
| Empty-tube Reynolds number | 100 through 1100 |
| Bulk Prandtl number | 7 through 900 |
| Twist ratio | `half_turn_length / tube_inner_diameter`, 2 through 4 |
| Circular tube inside diameter | 0.012 m |
| Tape width/thickness | 0.012 m / 0.001 m, zero clearance |
| Physical and heated tube lengths | Both 0.300 m; full-length continuous tape |
| Inner roughness | Zero; no added roughness correction |
| Thermal assumptions | Adiabatic tape, negligible buoyancy and radiation |
| Friction convention | Native Darcy, `f_Fanning = f_Darcy / 4` |

These are deliberately the verified study dimensions, not an inferred
geometrically similar family. Every thermal and hydraulic evaluation must
remain in range. Transition, turbulence, gas, cooling, phase change,
supercritical states, loose/partial tape, other dimensions and positive
roughness are unsupported. There is no extrapolation or smooth-tube blend.
An unsupported state raises `EnhancementUnsupportedError`; the selected
enhancement never silently falls back to a smooth correlation.

Nu and friction use empty-tube velocity and inside diameter. The blockage
area, wetted perimeter, hydraulic diameter and swirl corrections are already
inside the published correlation. The solver does not multiply by another
blockage factor or substitute the smaller hydraulic diameter into Darcy's
pressure-gradient equation. Gz uses one tube's heated length; straight-tube
pressure drop integrates over the full hydraulic length through the passes.

The source is a numerical laminar fit, reporting deviations of 20% for Nu
and 12% for friction. Those figures are not an uncertainty guarantee for a
complete exchanger. The source's inlet-reference/isothermal-wall model is
applied at KalKalori's bulk/wall and hydraulic quadrature states as an
explicit 0D engineering approximation, not a resolved axial calculation.

Wall viscosity is supplied by the authoritative thermal iteration and the
provider applies `(mu_bulk / mu_wall)**0.14` once. Initialization,
`iterate=False`, and hydraulic evaluations have no resolved local wall state;
the public model uses unity correction with a diagnostic. Hydraulic friction
in this selected model does not depend on wall viscosity. Direct low-level
`solve`/provider callers must supply known heating direction or wall states;
an unknown direction without a wall is only a provisional evaluation, not
evidence that a cooling application is supported.

| Warning code | Meaning |
| --- | --- |
| `enhancement_yang2020_scope` | Numerical fit and declared physical assumptions |
| `enhancement_wall_state_unavailable` | Unity viscosity correction for this evaluation |
| `enhancement_0d_reference_evaluation` | Bulk/wall or hydraulic quadrature approximation and separate local/momentum references |

Local nozzle, chamber, return, and tube-sheet entrance/exit losses retain
their existing architecture and reference geometry. Signed acceleration
retains the empty-tube endpoint momentum term. Insert-specific local losses
and an insert-specific momentum model are not included.

## Implementing an external provider

A separate package supplies an object with `provider_id` and
`evaluate(geometry, state) -> EnhancementResult`. Select it through the same
`TubeSideEnhancement` constructor; no registry or solver modification is
needed. `geometry` may be `TwistedTapeGeometry` or a provider-owned typed
object. The external provider validates geometry compatibility and its own
physical limits. Twisted-tape geometry uses an explicit 180-degree length;
providers with a 360-degree or width-based ratio convert at their boundary.

All core contract types are exported from `core.enhancements`:

| Contract | Required interpretation |
| --- | --- |
| `EnhancementInput` | SI per-tube mass flow, inside diameter, physical/heated lengths, roughness, bulk state, optional wall state, declared phase, position and heating/cooling direction |
| `EnhancementState` | Positive finite density, viscosity, conductivity and cp; optional temperature and pressure |
| `EnhancementReferenceState` | Per-tube flow area and consistent mass-flow velocity, provider-owned Re/Pr, friction diameter, separate hydraulic diameter and Nu reference length |
| `EnhancementResult` | Positive `alpha_inside` and `f_darcy`, native friction and explicit `darcy`/`fanning` basis, reference state, regime, applicability, provenance and optional Nu |
| `EnhancementDiagnostic` | Named scalar/string diagnostic and units, opaque to the solver |

An alpha-only model may leave `nusselt=None`; core derives Nu solely for
legacy thermal diagnostic presentation. If Nu is supplied, it must satisfy
`alpha_inside = Nu * bulk.k / reference.nusselt_length`. Core validates
`velocity = mass_flow_per_tube / (rho * flow_area)` and the native-to-Darcy
conversion. It integrates each provider gradient
`f_darcy * rho * velocity**2 / (2 * friction_diameter)` with Simpson weights
at inlet/midpoint/outlet. It never substitutes generic smooth-tube Re,
friction or wall/length corrections into the enhanced result.

Provenance includes `provider_id`, `correlation_id`, nonempty
`source_references` and `source_access_basis` (`open`, `private` or `external`).
Thermal and hydraulic results must share this model identity. If one model
has several regimes, identify the common paired model in `correlation_id`
and expose the local branch in `regime`/diagnostics. Different identities or
missing enhanced results in a coupled path are rejected.

`applicability` is `within_range` or `extrapolated`; the latter requires a
warning and is a provider decision (the built-in model never extrapolates).
Return immutable tuples of `ModelWarning` and `EnhancementDiagnostic`.
For richer typed data, subclass the frozen `EnhancementResult` dataclass
with defaulted additional fields; result adaptation preserves that subtype
and data without interpreting proprietary semantics. Private reference
identifiers need not expose confidential data, but values placed in results
are visible to callers and should be chosen accordingly.

The provider must return a paired thermal/hydraulic result at each requested
state, including initialization and hydraulic nodes with `wall=None`.
There is no resolved axial wall-temperature field. A model that requires
unavailable wall information must raise `EnhancementUnsupportedError`
instead of guessing. External property tables/software own their interpolation,
extrapolation policy, unavailable-data errors and any external I/O; core adds
no dependency or proprietary equation.

Single-phase `liquid` or `gas` must be declared. Where a property backend
exposes authoritative phase data, core checks it, including wall states.
Transport-only providers rely on the declaration and cannot independently
prove phase. Quality inputs, active inside phase-change paths, and wet-gas
phase-capable inside providers are excluded. Gas support in the generic
contract does not extend the liquid-only public model.

## Source policy and private-model boundary

Built-in engineering correlations must be independently reproducible from
legally and freely accessible source material. Models requiring paywalled
publications, licensed datasets, proprietary software or manufacturer-confidential
information are integrated through external providers. This is a project
policy; see [CONTRIBUTING.md](../CONTRIBUTING.md).

Manglik-Bergles Part I/II are reserved as sources for a future private
external implementation from legitimately acquired full papers. No Part I/II
implementation is in this GPL core, and no private package is required to use
the public model. CALGAVIN/hiTRAN, supplier tables and licensed software may
use the interface, but no manufacturer physics has been implemented or
reverse engineered; sharing an interface does not imply shared physics.

Synthetic fixtures exist only in `core/tests` to verify dispatch, alpha-only
results, native Fanning factors, independent hydraulic references, typed
metadata, warnings, Rating and both Simulation modes. They contain no M&B
or manufacturer correlation. Public-model tests include independently
hand-calculated equation anchors, boundary/unsupported cases and end-to-end
laminar solves; the source's CFD comparison tables are not presented as
exact anchors for Eqs.21-22.
