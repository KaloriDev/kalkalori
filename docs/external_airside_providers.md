# Independent external outside providers (v0.9.10)

Intrinsic geometry and channel-specific states are covered by the
[reporting follow-up](elliptical_roughness_and_reporting.md). Optional
`OutsideReferenceQuantities.flow_area`, `flow_area_basis` and `mass_flux`
describe the provider's actual flow reference. A supplied area requires
its basis; omitted values remain unknown and no minimum area is inferred.

`BareTubeHeatExchanger` accepts `outside_thermal_provider`,
`outside_hydraulic_provider` and `outside_provider_policy`. Each channel is
optional and independent. An omitted channel retains its existing built-in
selection and applicability guards. Objects implement `evaluate(request)`;
there is no discovery registry or import of external implementations.

This API introduces **no public SHIFTED heat-transfer or pressure correlation**.
Passing a geometry through a generic boundary does not validate a correlation.
Public examples and tests use synthetic data only.

## Thermal contract

Import the contracts from `core.heat_transfer.outside_provider` (also exported
by `core.heat_transfer`). `OutsideProviderRequest` contains the unchanged
`TubeBundle`, mass flow, rho, mu, k, cp, frontal area, physical outside area,
face velocity and available absolute T/p. Units are SI. The bundle retains
dimensions, layout, lattice, physical rows, orientation, material and circuit.
Low-level explicit-property calls may leave T/p unavailable (`None`).

No Zhao throat, passage velocity or Reynolds definition is evaluated when an
external channel is selected. A provider needing those quantities must define
and validate its own physical reference. It may return optional
`OutsideReferenceQuantities` with velocity, characteristic length, Re, Nu and a
defined pressure coefficient. Unknown diagnostics remain NaN in legacy scalar
slots; the core does not invent a Reynolds number or characteristic diameter.

`OutsideThermalProviderResult.alpha_outside` is positive finite physical film h
[W/(m² K)] on `area_basis="physical_outside_area"`. It excludes fin efficiency,
surface enhancement, contact resistance, wall conduction and fouling. The
existing annular or 2D rectangular fin solver receives this h; the resistance
network calculates primary/fin contributions, effective outside h, U and UA.
Returning an apparent UA-derived or already fin-reduced coefficient violates
this contract even if its number passes validation.

Independent injection supports plain, solid circular-finned and individual
rectangular-finned surfaces. Serrated and continuous-sheet models retain their
explicit complete profiles and coupled fin definitions; generic film injection
there is rejected. Outside wet-surface execution and variable outside mass-flow
hydraulics are unsupported. Water vapour in a dry operating gas is allowed;
actual wet execution rejects external dry providers. Dry outside helpers used
by existing tube-side phase-change workflows share the same boundary.

## Applicability and provenance

Every result requires `OutsideProviderMetadata` with a provider name,
provenance, reference basis and typed `OutsideApplicability`. Supported exact
and interpolated states execute normally. `UNSUPPORTED` raises
`OutsideProviderApplicabilityError`, without fallback.
`ENGINEERING_ESTIMATE` requires declared limitations and
`OutsideProviderPolicy(allow_engineering_estimates=True)`.

Warnings are tuples of `ModelWarning`; metadata can carry diagnostics in a
mapping. Engineering limitations also become result warnings. Invalid h,
pressure, references, output types or metadata raise
`OutsideProviderResultError`. Exceptions from the selected provider propagate;
the core never substitutes a built-in model after failure.

Provenance is retained in the dispatch result's `external_result`,
`HXResult.outside_side_thermal.provider_result`, iterative thermal diagnostics'
`outside_provider_result`, and finned diagnostics' `heat_transfer_metadata`.
Hydraulic results expose `provider_result` and `metadata`; the final hydraulic
refresh preserves these in Rating and Simulation.

## Pressure contract and accounting

Import from `core.pressure_drop.outside_provider` (also exported by
`core.pressure_drop`). `OutsideBankPressureDropRequest` contains actual inlet,
arithmetic-temperature midpoint and outlet requests. The provider evaluates
the complete declared bank/scope once; it owns spatial integration and returns
`OutsideBankPressureDropResult`, including `delta_p` [Pa], typed `scope`, a
nonempty `scope_description`, metadata and optional references/warnings.
Positive static difference means inlet minus outlet pressure.

| Scope | Provider quantity | Core additions |
|---|---|---|
| `BANK_ONLY` | Nonnegative irreversible bank loss, excluding fittings and acceleration | Signed face acceleration once; explicit local paths may be added |
| `LOSS_WITH_ENTRANCE_EXIT` | Nonnegative irreversible loss including the specified `entrance` and/or `exit` contributions | Signed face acceleration once; additional local paths rejected |
| `COMPLETE_STATIC` | Signed complete static difference over the declared stations | None; additional local paths rejected |
| `UNRESOLVED_ENGINEERING` | Nonnegative engineering estimate whose experimental tapping scope is unresolved | None; additional local paths rejected |

For irreversible results the acceleration term is
`(mass_flow/frontal_area)**2 * (1/rho_out - 1/rho_in)` for equal frontal inlet
and outlet areas. A provider whose stations have other areas must supply a
complete static difference rather than requesting this face-area correction.
`included_contributions` must be empty for bank-only loss and specify entrance
and/or exit for the loss-with-ends scope; neither may include acceleration.

Unresolved scope requires both an `ENGINEERING_ESTIMATE` metadata status and
`OutsideProviderPolicy(allow_engineering_estimates=True,
allow_unresolved_pressure_scope=True)`. Its warnings and scope description
remain visible. It is not converted into bank-only irreversible loss.

Only bank-only results can enter an irreversible bank stage. Other scopes use
an undecomposed `ScopedOutsidePressureResult` with `pressure_basis`, `scope_description`,
`provider_result`, no fictitious stages and no added contributions. Its
`components_resolved` and `validated_bank_irreversible` flags are false.
The latter flag denotes the provider's declared bank basis, not experimental
certification by KalKalori. Legacy `dp_drag` slots on non-bank results retain
the supplied value for compatibility: inspect `pressure_scope` before treating
it as drag. Use `dp_total` for the supplied result plus any permitted acceleration.

## Selection and example

External providers take precedence only for their own selected channels;
conflicting per-call Euler or finned-provider selectors are rejected. An
`airside_model` may still supply the other channel. When Zhao2017 is selected,
its original geometry and source checks remain in `Zhao2017AirsideModel`.
An external provider cannot supply fin efficiency or modify the geometry.

Run the self-contained synthetic example from the repository:

```powershell
python -m examples.external_airside
```

It exercises ordinary Simulation and inverse Rating. The supplied constants
demonstrate integration and carry no predictive validity.
