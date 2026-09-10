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
It may also be supplied directly to either public solver call:

```python
simulation = hx.simulate(inside, outside, tube_side_enhancement=enhancement)
rating = hx.rate(inside_balance, outside_balance,
                 tube_side_enhancement=enhancement, include_simulation=True)
```

Omitting the keyword inherits the exchanger's constructor configuration.
Passing a configuration overrides it only for that call; passing `None`
explicitly selects the legacy smooth path for that call. Selection uses an
exchanger copy with the same geometry and exact provider instance, leaving
the original exchanger unchanged even if the solve raises. The complete
configuration is passed as one object so HTC and friction cannot be selected
independently. See the private notebook readiness audit below.

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
| `EnhancementInput` | SI per-tube mass flow, inside diameter, physical/heated lengths, base per-tube flow area, total hydraulic length, roughness, bulk state, optional wall state, declared phase, position and heating/cooling direction |
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

## PRIVATE M&B PROVIDER READINESS

**Interface readiness: yes.** A future privately supplied
`ManglikBergles1993Provider` can implement the public
`TubeSideEnhancementProvider` contract and be used by private notebooks
without registering its name, editing a dispatcher, monkey-patching, or
adding an import of private code to core. No further core change is required
to select the object and consume its coherent result within the documented
single-phase 0D state contract. This audit verifies integration readiness;
it does not verify or implement the paywalled model's physics or applicability.

The notebook imports its private class using its own Python import setup
and constructs it normally. With that `provider` object already created,
the public side of the notebook code is:

```python
from core.enhancements import TubeSideEnhancement, TwistedTapeGeometry

tape = TwistedTapeGeometry(
    half_turn_length=0.036, tape_width=0.012, tape_thickness=0.001,
)
selection = TubeSideEnhancement(provider=provider, geometry=tape, fluid_phase="liquid")
rating = hx.rate(inside_balance, outside_balance, tube_side_enhancement=selection)
simulation = hx.simulate(inside, outside, tube_side_enhancement=selection)
```

These geometry values illustrate the public API only; they do not declare
the private model's validity. A conceptual private location such as
`.kon/_shared/tube_side_enhancement/manglik_bergles_1993.py` has no special
meaning to core. No such module or notebook is created, imported, inspected,
or required by public implementation/tests. Dependency direction is solely
private notebook/provider -> public contract.

| Requirement | Public data/path available to the private provider |
| --- | --- |
| Physical geometry | `evaluate(geometry, state)` receives the public tape geometry, without Sw, friction or private model parameters |
| Local flow | `state.mass_flow_per_tube`; `base_flow_area_per_tube`; derived `base_mass_flux` in kg/(m2 s). Core already divides bundle mass flow/area by effective parallel tube count, so the provider need not access bundle internals |
| Tube and lengths | `tube_inner_diameter`, `tube_length` (one physical tube), `heated_length` (one heated tube), `hydraulic_length_total` (complete path through the passes), `roughness_inner` |
| Bulk state | `state.bulk.rho`, `.mu`, `.k`, `.cp`, `.temperature` [K], `.pressure` [Pa]; phase/direction/position separately on input |
| Wall state | `state.wall` carries the authoritative wall transport properties, temperature and pressure when thermal iteration evaluates them; it is optional during initialization and absent at hydraulic quadrature nodes |
| Coherent output | Required positive `alpha_inside`, explicit `f_darcy`, native factor/basis, `reference`, `regime`, provider/correlation/source identifiers; optional `nusselt`, declared applicability, warnings and correction diagnostic |
| Provider-specific details | Immutable scalar `EnhancementDiagnostic` tuples or defaulted typed fields on a frozen `EnhancementResult` subclass; solver does not interpret Sw, Re variants or private factors |
| Darcy hydraulic consumption | At each inlet/midpoint/outlet, the same provider supplies `f_darcy` with its own reference velocity and friction diameter; Simpson integration uses its pressure gradient over `hydraulic_length_total` |
| Thermal correction | The external provider returns its already corrected alpha. Core applies no additional smooth-tube wall or length correction |
| Selection and failures | Constructor default or direct `rate`/`simulate` configuration; no registry. Unsupported errors propagate and model identity must agree across thermal/hydraulic paths |

The two added geometry fields are optional for independently constructed
standalone `EnhancementInput` values, preserving existing calls. Exchanger
adapters always fill them. `base_mass_flux` is `None` when a standalone caller
omits the base area. Provider-owned blocked area, reference velocities,
hydraulic diameters and dimensionless groups remain derived by the provider;
the base inputs do not prescribe its correlation convention.

The 0D availability boundary is explicit: there is no axial wall-temperature
field and no invented wall state in hydraulics. The future provider must
handle provisional/missing-wall evaluations according to its verified model
or reject unsupported states. This is not a promise that an arbitrary model
requiring a spatially resolved wall solution can be evaluated by a 0D solver.
Extra private source data and additional property evaluation, if needed,
belong to the external object and must not depend on private core internals.

`core/tests/external_enhancement_integration_test.py` defines synthetic
providers entirely in tests and verifies notebook-style selection through
Rating, both Simulation modes and the Rating-to-Simulation bridge. It checks
per-tube inputs and two-pass length, authoritative bulk/wall properties,
an independently specified synthetic wall multiplier applied once, explicit
Darcy pressure gradients, provenance/typed detail/warning propagation,
unchanged defaults after overrides/errors, exact explicit-None smooth
results, and controlled unsupported failure. These fixtures contain no M&B
equations, paywalled numerical anchors or manufacturer data.

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
