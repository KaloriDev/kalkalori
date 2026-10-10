# Tube-side enhancements

The [Inaba1994 wire-coil provider](wire_coil_inaba1994.md) adds a distinct
insert family without using twisted-tape geometry or clearance semantics. Its
hydraulic and thermal film properties, source diameter/area normalization and
restricted `P/e > 10` applicability are documented on the dedicated page.

The additional [Rossi2017 open-secondary comparator](twisted_tape_rossi2017.md)
uses film thermal properties and bulk hydraulic states, with explicit
extrapolation policy. The Yang2020 example and limits below remain separate.

Optional source-neutral [clearance model composition](twisted_tape_clearance.md)
supports explicit correction and absolute modes. No built-in clearance fit or
AUTO fallback is currently enabled.

Version v0.8.0 adds a coherent provider for inside heat transfer and
distributed straight-tube friction. Omitting `tube_side_enhancement`
(or setting it to `None`) selects the smooth-tube route, including the
[laminar thermal-development correction](internal_laminar_thermal_development.md)
when a positive heated length is supplied.

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
| Heated tube length | 0.300 m; full-length continuous tape; physical length stays solver-owned |
| Inner roughness | Not an equation input; supplied roughness is ignored with a diagnostic |
| Thermal assumptions | Adiabatic tape, negligible buoyancy and radiation |
| Friction convention | Native Darcy, `f_Fanning = f_Darcy / 4` |

These are deliberately the verified study dimensions, not an inferred
geometrically similar family. Every thermal and hydraulic evaluation must
remain in range. Transition, turbulence, gas, cooling, phase change,
supercritical states, loose/partial tape and other correlation dimensions
are unsupported. A different physical tube length is reported as ignored
by the correlation; the hydraulic path still uses its actual length.
Positive roughness is also reported as ignored, not applied as a correction.
There is no extrapolation or smooth-tube blend.
An unsupported state raises `EnhancementUnsupportedError`; the selected
enhancement never silently falls back to a smooth correlation.

For liquid cooling, `Yang2020TwistedTapeProvider` raises
`EnhancementUnsupportedError` with
`yang2020_cooling_unsupported: liquid heating only.` Cooling is outside the
supported source scope, not warning-mode extrapolation: the heating correlation
is not reused for cooling. This restriction belongs to the Yang2020 provider;
it is not a general limitation of KalKalori's Rating or Simulation solvers.

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
needed. Structural compatibility requires no registration, discovery,
inheritance or import of the external provider into KalKalori.
`geometry` may be `TwistedTapeGeometry` or a provider-owned typed
object. The external provider validates geometry compatibility and its own
physical limits. Twisted-tape geometry uses an explicit 180-degree length;
providers with a 360-degree or width-based ratio convert at their boundary.

All core contract types are exported from `core.enhancements`:

| Contract | Required interpretation |
| --- | --- |
| `EnhancementInput` | SI per-tube mass flow, inside diameter, physical/heated lengths, base per-tube flow area, total hydraulic length, roughness, bulk state, optional wall and backend-evaluated thermal/hydraulic reference states, declared phase, position, heating/cooling direction and optional shared `operation_context` |
| `EnhancementOperationContext` | Immutable absolute monotonic deadline [s] or unlimited; `remaining_time()` and `check_deadline()` |
| `EnhancementState` | Positive finite density, viscosity, conductivity and cp; optional temperature and pressure |
| `EnhancementReferenceState` | Per-tube flow area and consistent mass-flow velocity, provider-owned Re/Pr, friction diameter, separate hydraulic diameter and Nu reference length |
| `EnhancementResult` | Positive thermal `alpha_inside` and canonical `f_darcy`, native friction with explicit `darcy`/`fanning` basis and reference normalization, reference state, regime, applicability, provenance, thermal property reference and optional Nu; hydraulic-only nodes may omit alpha and Nu |
| `EnhancementDiagnostic` | Named scalar/string diagnostic and units, opaque to the solver |

An alpha-only model may leave `nusselt=None`; core derives a bulk-equivalent
Nu solely for legacy thermal diagnostic presentation, not a canonical
provider Nu. If Nu is supplied, it must satisfy
`alpha_inside = Nu * k_reference / reference.nusselt_length`. The default
reference is bulk; providers may declare `thermal_property_reference` as
`"wall"` or `"film"`. Integration evaluates the backend at the requested
temperature and supplies `state.thermal`; it never averages properties.
The result must declare the same reference. Core validates
`velocity = mass_flow_per_tube / (rho * flow_area)` and the native-to-Darcy
conversion. It integrates each provider gradient
`f_darcy * rho * velocity**2 / (2 * friction_diameter)` with Simpson weights
at inlet/midpoint/outlet. It never substitutes generic smooth-tube Re,
friction or wall/length corrections into the enhanced result.

Providers may independently declare `hydraulic_property_reference` as bulk,
wall, or film. For wall/film hydraulics, integration evaluates the same fluid
backend at the required temperature for every hydraulic quadrature point and
supplies `state.hydraulic`; it never averages transport properties. A native
friction factor whose source diameter/velocity differs from the canonical
returned Darcy reference uses explicit `friction_normalization`, which is
validated together with the ordinary Fanning factor of four.

Provenance includes `provider_id`, `correlation_id`, nonempty
`source_references` and `source_access_basis` (`open`, `private` or `external`).
Thermal and hydraulic results must share this model identity. If one model
has several regimes, identify the common paired model in `correlation_id`
and expose the local branch in `regime`/diagnostics. Different identities or
missing enhanced results in a coupled path are rejected.

`applicability` is `within_range` or `extrapolated`; the latter requires a
warning and is a provider decision (Yang2020 never extrapolates; Rossi2017
supports explicitly requested warning-mode extrapolation).
Return immutable tuples of `ModelWarning` and `EnhancementDiagnostic`.
For richer typed data, subclass the frozen `EnhancementResult` dataclass
with defaulted additional fields; result adaptation preserves that subtype
and data without interpreting proprietary semantics. Private reference
identifiers need not expose confidential data, but values placed in results
are visible to callers and should be chosen accordingly.

The same provider owns thermal and hydraulic performance. Hydraulic nodes
may return friction only (`alpha_inside=None`, `nusselt=None`) when the thermal
reference is unavailable; thermal requests require alpha. Wall requirements
reuse the existing iterative wall solver, as described below; a standalone
call without required wall data fails clearly.
External property tables/software own their interpolation,
extrapolation policy, unavailable-data errors and any external I/O; core adds
no dependency or proprietary equation.

This canonical result does not require every upstream backend to natively
return `Nu` and Darcy friction. An adapter may pass through a native
`alpha_inside`, convert a documented Fanning factor to Darcy, or normalize a
distributed pressure drop to Darcy using the backend's own reference velocity,
diameter and length. It must do that conversion at the adapter boundary and
return the complete coherent `EnhancementResult`; core will not guess missing
physics or combine unrelated heat-transfer and friction models. A backend that
supplies only one side of the coupled model is therefore not by itself a usable
enhancement provider.

Optional software adapters remain ordinary explicitly selected providers.
They should import/probe their runtime locally and raise a controlled provider
error when unavailable, so importing KalKalori never depends on that software.
Installation alone must not place an external model in `AUTO`. Backend name,
revision and execution mode may be carried in opaque diagnostics or defaulted
fields on a result subtype alongside the required provider/model/source
identity; the solver needs no second provider hierarchy for this.

### Shared operation deadline

Rating and Simulation accept the optional `enhancement_operation_context`
keyword. Create the budget once, at the start of the operation:

```python
from core.enhancements import EnhancementOperationContext

context = EnhancementOperationContext.from_timeout(30.0)
result = hx.rate(
    inside_balance, outside_balance,
    tube_side_enhancement=selection,
    enhancement_operation_context=context,
    include_simulation=True,
)
```

The exact same immutable object reaches `state.operation_context` during
initial calls, thermal/wall iterations, hydraulic nodes, endpoint probes,
repeated Rating forward evaluations and optional nested Simulation. Its
deadline is never restarted per evaluation. Selection and context remain
local to this operation; the original exchanger is unchanged and the provider
is not cloned.
The exchanger is shallow-copied for call-local options while retaining the
same provider, config and private session.

Omission preserves unlimited behavior (`state.operation_context is None`).
`EnhancementOperationContext()` or `from_timeout(None)` also means unlimited.
`from_timeout(seconds)` requires a positive finite timeout. A directly
supplied `deadline` is in absolute monotonic seconds, not a wall-clock date.
`remaining_time()` returns nonnegative seconds or `None` when unlimited;
`check_deadline()` raises `EnhancementTimeoutError` when expired. Standalone
provider callers can populate `EnhancementInput.operation_context` directly.
The context applies to selected tube enhancement calls and does not replace
the separate wet-coil solver controls or impose a timeout on smooth physics.

Core checks before invocation and after the provider returns. It cannot
forcibly interrupt an arbitrary external blocking call. A private adapter
must pass `remaining_time()` into its own transport/request timeout and
check the context during cooperative work. No thread/process executor or
external runtime dependency is required by core.

### Required wall state and property references

`requires_wall_state` is an optional boolean provider attribute, default
`False`; it is not a required structural protocol member. A provider can
correctly declare:

```python
thermal_property_reference = "bulk"
hydraulic_property_reference = "bulk"
requires_wall_state = True
```

The property-reference attributes select which fluid state supplies the
primary thermal/hydraulic transport properties. The wall requirement means
that a separate wall state is also needed, for example for a wall-viscosity
correction. It never changes the meaning of bulk, wall or film reference.

For required-wall providers, initialization uses a bounded representative
wall temperature between the two bulk temperatures and evaluates properties
at that temperature through the existing fluid backend. Rating and iterative
Simulation continue re-evaluating the provider inside the wall solve.
Snapshot Simulation resolves the representative wall through a wall probe
even with bulk references. Hydraulic inlet/midpoint/outlet evaluations receive
`state.wall` from the backend whenever representative wall temperature is
known. Film `state.hydraulic` and `state.wall` are separate states, evaluated
at their respective temperatures; transport properties are never averaged.

Missing required wall properties/temperature raise
`EnhancementUnsupportedError` before provider invocation. The provider
returns its already corrected absolute `alpha_inside`; core does not apply
that correction again. Providers without the new capability retain their
existing optional-wall behavior. For clearance composition, correction mode
honors both active providers' requirements, while absolute mode honors the
replacement provider only.

### Sessions and explicit failure

An external provider may retain mutable private configuration, opaque model
identity and a non-serializable session/process/client. Core performs calls
serially within an operation and neither deep-copies, pickles, reconstructs,
compares nor hashes that provider object. Applications sharing a stateful
object across concurrent operations must coordinate access themselves.
Private adapters retain their runtime dependencies and session lifecycle;
they may use `source_access_basis="private"` or `"external"` and opaque
diagnostics/provenance without revealing private configuration to core.

`EnhancementProviderError` marks a fatal selected-provider failure.
`EnhancementUnsupportedError` remains catchable as `ValueError`;
`EnhancementTimeoutError` is also a `TimeoutError`. Adapters can raise the
generic marker for an unavailable backend or failed external protocol.
Core normalizes unmarked exceptions raised during provider evaluation and
result validation into marked failures, retaining the original exception as
`__cause__`. Validation failures retain `ValueError`/`TypeError` compatibility.

An explicitly selected provider never triggers physical-model fallback.
Fatal unsupported, timeout, backend and invalid-result failures also propagate
from endpoint wall-envelope probes and fail Rating/Simulation. Ordinary
numerical inability to estimate an optional envelope can still produce
`wall_temperature_probe_not_converged` warnings and nonconverged/NaN probe
diagnostics when the provider itself has not failed.

Direct distributed-gradient output remains deferred. The hydraulic contract
still requires coherent friction/reference normalization. Conversion of a
backend pressure drop to equivalent Darcy friction requires a documented
distributed straight-tube loss, applicable length and reference state; it
must exclude separately owned acceleration, entrances/exits and component
losses. An undivided whole-exchanger pressure drop cannot be assumed to meet
that condition.

This remains a 0D boundary. Local segment coordinates, cumulative development
lengths, pass indices and spatial wall/pressure fields remain future
distributed-model concerns.

Single-phase `liquid` or `gas` must be declared. Where a property backend
exposes authoritative phase data, core checks it, including wall states.
Transport-only providers rely on the declaration and cannot independently
prove phase. Quality inputs, active inside phase-change paths, and wet-gas
phase-capable inside providers are excluded. Gas support in the generic
contract does not extend the liquid-only public model.

## External contract regression coverage

`core/tests/external_enhancement_hardening_test.py` loads a synthetic fixture
from `tests/fixtures/external_enhancement_provider`, outside the core namespace.
It verifies structural compatibility, arbitrary private config, retained
mutable session state, repeated Rating/Simulation evaluation, shared deadlines,
required wall properties, opaque result details and strict endpoint failures.
The fixture rejects deepcopy, serialization, equality and hashing to expose
accidental assumptions about private session objects.

Existing provider tests continue checking independently calculated public
equation anchors, Darcy/Fanning normalization, reference geometry, warning
propagation, no fallback and unchanged smooth/default behavior. Synthetic
fixtures contain no proprietary physics or runtime dependencies.

## Source policy and private-model boundary

Built-in engineering correlations must be independently reproducible from
legally and freely accessible source material. Models requiring paywalled
publications, licensed datasets, proprietary software or manufacturer-confidential
information are integrated through external providers. This is a project
policy; see [CONTRIBUTING.md](../CONTRIBUTING.md).

Licensed reference data and software remain outside the public distribution.
No private package is required to use the public models. Sharing the generic
interface does not imply shared physics or implementation of an external model.

Synthetic fixtures in the public test tree verify dispatch, alpha-only
results, native Fanning factors, independent hydraulic references, typed
metadata, warnings, Rating and both Simulation modes. They contain no
manufacturer correlation. Public-model tests include independently
hand-calculated equation anchors, boundary/unsupported cases and end-to-end
laminar solves; the source's CFD comparison tables are not presented as
exact anchors for Eqs.21-22.
