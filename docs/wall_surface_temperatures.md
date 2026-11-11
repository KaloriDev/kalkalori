# Wall and exposed-surface temperatures (v0.9.7)

All core temperatures use K. These are 0D estimates unless a provider explicitly
supplies a native profile. Numerical verification is not experimental validation.
The [integrated v0.9.7 review](validation/v0.9.7_integrated_validation.md) traces
the temperatures used in the actual equations and records the conditional
experimental comparison and remaining model limitations.

| Physical quantity | Core diagnostic | Meaning |
| --- | --- | --- |
| Inner metal wall | `inside_wall_temperature`, envelope `inside_*` | Inner tube metal after inside film and fouling |
| Outer metal wall | `outside_wall_temperature`, envelope `outside_*` | Core tube outer metal before the outside root/contact/fouling path |
| Primary exposed surface | `primary_surface_mean`, probe `outside_primary_surface_temperature` | Gas-facing primary node; a bare probe retains its historical `None` convention |
| Fin base and tip | envelope `fin_base_*`, `fin_tip_*` | Separate conduction/contact nodes, available only if the model reconstructs them |
| Exposed fin extrema | `fin_surface_min`, `fin_surface_max` | Fin-only bounds, excluding the primary tube |
| Whole exposed surface | `outside_skin_min/max/mean` | Primary plus fin surface, with geometric area weighting for the mean |
| Inner exposed surface | `inside_surface_min/max/mean` | Fluid-facing inside fouling surface; coincides with metal for zero fouling |
| Condensate interface | wet `surface.interface_temperature_*` | Only supplied when the provider defines this boundary; never inferred from metal |
| Representative wet surface | wet `surface_temperature_wet_mean` | Temperature used for wet-zone equilibrium/drainage, distinct from the whole-surface mean |

The historical public `HXSimulationResult` / `HXRatingResult` wall-mean
properties retain their global thermal-state definitions. Envelope means now
come from a separate representative mean-bulk resistance calculation, replacing
the former midpoint of extrema. In particular, a Rating's specified terminal
program need not equal the achievable Simulation used by its property iteration.
Do not mix a global wall mean with an envelope of a different surface definition.
Active inside carrier-gas results with nonzero inside fouling now correctly
report the inner **metal** wall on the thermal state, rather than the gas-facing
interface. This is a correction to the documented metal meaning of that public
field. `inside_surface_*` and the phase result retain the interface definition.
The active inside envelope uses its own accepted wet states, not dry probes at
wet outlets, and honors the selected per-call flow arrangement.

## Dry envelope

Each representative bulk pair gets its own existing property/HTC iteration and
signed resistance balance. Counterflow uses inside-in/outside-out and
inside-out/outside-in. Cocurrent flow uses inlet/inlet and outlet/outlet.
The mean-bulk pair is an additional calculated state, with its own correlations.
Extrema include this physically evaluated state; no post-hoc temperature clamp
is applied. Sampling does not prove that all nonlinear interior extrema have
been resolved. The dry envelope is diagnostic and does not change epsilon-NTU
duty, installed geometry, source correlations or defaults.

Unresolved crossflow and counter-crossflow return explicitly identified
bulk-temperature rectangle bounds. These pairs are bounding states, not asserted
co-located row/circuit temperatures. An explicit equivalent counterflow model
on a multipass bundle does not reconstruct the actual pass connections.
`method`, `temperature_definition`, `spatial_assumption`, `applicability` and
`mean_method` record these distinctions. Probe failure leaves an incomplete
envelope and warnings; missing mean calculations remain unavailable.

Circular solid fins retain their existing annular conduction model and actual
root/contact topology. For uniform h, the fin's area-mean excess temperature is
eta times its base excess temperature. The combined mean weights primary and
fin geometric areas. Reversed heat flow uses signed branch rates throughout.
A fouled bare tube's gas-facing surface differs from outer metal by the signed
outside fouling temperature drop.

Continuous plate fins retain the v0.9.6 experimental reduced-HTC/Schmidt network.
Its mean-efficiency surface temperature is available; a finite-sheet local tip
profile and intrinsic contact/film split are not. Base/tip and exposed extrema
remain unavailable instead of receiving an invented radial profile. Corrected
serrated efficiencies without a temperature field retain the same limitation.
Existing applicability guards, including rejection of wet continuous sheets,
remain in force.

## Condensation and provider ownership

Dry onset compares the cold exposed surface with the water dew point at the
current inlet vapor partial pressure. Unresolved crossflow bounds only provide
a conservative possibility screen. Possibility, solver activation, positive
condensate, and nonzero wet area are distinct diagnostics.

The bulk-mean analogy's bare wet-area model closes representative interface
balances using current mean-property resistances and a uniform latent source.
Both stream terminal changes enter counter/cocurrent pairs. For unresolved
crossflow it retains the opposite-stream mean approximation: a conservative
rectangle bound cannot be interpreted as an area distribution. The inside
solver retains its established interface-iterate linearization for numerical
stability. At convergence that linearization closes the same latent balance.
The unchanged linear wet-area approximation uses the local mean-composition dew
point; saturation, latent heat and drain enthalpy use its wet-zone temperature.
This fraction is a `wet_surface_fraction_0d`, not an axial wet length.

Bulk-mean circular fins retain the native nonlinear radial solution and the
existing optional cold-zone equilibrium offset. An equilibrium offset is
reported separately and is not presented as a physical metal temperature.
The public active envelope now reports the accepted wet states, rather than
rerunning an unrelated dry envelope at wet outlets.

Elmahdy retains its source-profile solution, radial states, native interface
temperatures and quadrature means. The generic surface mean is the native
axial/radial area integral; an interface mean refers to its primary boundary.
No generic 0D profile overwrites it. Private external providers retain their
own definitions. `axial_wet_fraction`, radial wet fractions and aggregate wet
area fractions cannot be substituted for one another.

No liquid-film resistance or unsupported wet geometry is introduced. Gas
enthalpy inversion, active wet hydraulics, and a distributed v2.x model remain
separate work. See the [provider guide](wet_coil_providers.md) and
[implementation evidence](validation/v0.9.7_wall_temperature_envelope_patch.md).
