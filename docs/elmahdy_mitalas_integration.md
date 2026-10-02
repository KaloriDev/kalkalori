# Elmahdy-Mitalas public forward/inverse integration

The default outside water-condensing gas route uses the accepted production engine
through `core.phase_change.wet_coil_integration.forward_wet_process`.
Simulation and Rating both use installed geometry. Rating solves a positive
thermal area scale, with `A_required=thermal_scale*A_o`. Tube lengths, fin
geometry, topology, pitches, materials and hydraulic flow areas remain
installed. Velocities, Reynolds numbers and correlations are evaluated on
that geometry at the solved fluid states and flow rates.

Rating supports a specified gas outlet and known inside mass flow, predicting
inside outlet; or two specified outlets and an unknown inside mass flow,
solving bounded positive area scale and flow together. Fully specified
inconsistent processes are errors. Installed geometry never closes an unspecified active
wet gas outlet. The same physical forward solve supplies all wet outputs.

Selection is by direct `wet_coil_provider` object. Omission or `None` selects
Elmahdy for Simulation and Rating. Explicit historical Legacy selection is
Simulation-only; unsupported operations fail explicitly without fallback.
See [the provider contract](wet_coil_providers.md) for external implementations.

## AUTO and DISABLED routing

With the default Elmahdy provider, an eligible outside wet-gas/water-condensation
case in AUTO uses the same
accepted Elmahdy-Mitalas production engine in DRY, PARTIALLY_WET and FULLY_WET
regimes. Its caloric and surface definitions remain continuous through onset.
An AUTO result in DRY has zero condensate/drain and inactive phase change,
but retains production provenance and equivalent process UA reporting.

DISABLED retains the legacy sensible model and its established numeric
results. AUTO-DRY may therefore differ from DISABLED-DRY. This is an approved
model-path difference, not a regression or humidity-dependent interpolation.
No blending or clipping is used. Unrelated/unsupported fluid paths and inside
phase-change handling retain their appropriate existing routes.

## Thermal reserve versus physical hydraulics

Simulation applies `thermal_scale=1/(1+surface_margin)` to inside/outside
transfer areas, wall and root/contact conductances, and fin/primary thermal
responses. Physical hydraulic geometry is unchanged. Film HTCs and Reynolds
numbers at identical fluid states are unchanged. Different converged fluid
states can of course change viscosity, velocity and Reynolds number.
`Q_full` is a separate zero-margin physical forward result; `q` and
`Q_derated` are the actual reserved-area result. No flow or outlet is rescaled.

Rating uses the same thermal scaling mechanism, allowing scales below or above
one. Its hydraulic diagnostics always use installed geometry. Both `A_o` and
`final_result.A_o` are installed outside area; `A_required` is the required
thermal area. `final_result` combines the required thermal process with
hydraulics evaluated on installed geometry at that process's fluid states.

## Numeric equivalent UA reporting (approved process reduction)

Only AFTER the physical solution, set `Q_process=Q_liquid`, excluding drain:

```
Q_gas = md*(h_in-h_out) = Q_process + H_drain
C_hot_eq  = Q_process/(T_hot_in-T_hot_out)
C_cold_eq = Q_process/(T_cold_out-T_cold_in)
eps_eq = Q_process/(min(C_hot_eq,C_cold_eq)*(T_hot_in-T_cold_in))
UA_process_eq = ntu_from_effectiveness(eps_eq,...)*min(C_hot_eq,C_cold_eq)
EMTD_eq = Q_process/UA_process_eq
```

The existing inversion uses the declared flow arrangement (including explicit
inside/outside capacities for crossflow). A degenerate zero-duty/isothermal
sensible program or an unreachable conventional equivalent raises a controlled
error; arbitrary epsilon temperature changes are not inserted.

For wet Rating, `A_required` is required thermal outside area,
`UA_required=UA_process_eq`, `U_mean=UA_required/A_required`, and
`UA_actual=U_mean*A_o` at that SAME required-process reference. Consequently
`overdesign_factor=ua_margin=UA_actual/UA_required-1=A_o/A_required-1`.
The optional installed Simulation is a separate operating point and does not
supply Rating's `UA_actual`.

For wet Simulation, `A_thermal=A_o/(1+surface_margin)`,
`U_mean=UA_process_eq/A_thermal` and `UA_actual=U_mean*A_o`. Thus the numeric
UA ratio reproduces the requested reserve without changing the solved duty.

These values describe a conventional two-stream exchanger with secant process
capacities that reproduces the solved wet operating point. They are NOT native
Elmahdy-Mitalas temperature/enthalpy conductances and cannot feed back into
wet heat, moisture, wall or thermal-area solving. Results identify
`ua_reporting_basis="wet_equivalent_secant_capacity_ntu"` and
`ua_is_equivalent=True`. Native conductances, saturation secant, surfaces,
radial response, wet fraction and integrated drain remain separate in
`wet_coil_diagnostics`, including `process_profile` and radial `surface_states`. Legacy paths retain their numeric UA meanings; eligible AUTO-DRY uses the
same equivalent reduction as AUTO wet regimes.

## Process, energy and surface diagnostics

Gas mass and enthalpy use the configured dry-carrier basis. `H_drain` is the
coupled local-surface integral from the accepted source moisture profile.
The reporting split follows fixed-inlet-W sensible cooling followed by
isothermal moisture removal at gas outlet temperature; the moisture term
subtracts integrated drain enthalpy. It is an explicit thermodynamic path,
not a new effective wet HTC or a normalization to required duty.
The reported sensible/latent components partition the accepted liquid duty:
the net latent component retains the endpoint moisture enthalpy minus
integrated drain; sensible heat is the remaining Q_liquid. In exact energy
closure this equals fixed-inlet-W sensible cooling. Keeping the latent
formula avoids assigning an absolute solver residual to a vanishing onset
latent term. The independently evaluated
endpoint terms are retained as `endpoint_sensible_heat` and
`endpoint_latent_heat`, along with `sensible_latent_closure_residual`.
In the DRY boundary regime the whole accepted duty is sensible and latent
heat is exactly zero. This exposes the finite solver
residual while preserving the exact reporting identity Q_total=Q_sensible+
Q_latent. The separate endpoint gas/liquid/drain energy residual is unchanged;
no outlet, total duty, condensate or drain integral is adjusted for reporting.

Public wall probes sample the solved native process, including its warm end.
They are not a shifted dry envelope. Fin radial states and their conditional
condensate enthalpy weighting retain the accepted adapter definition.
Finned pressure drop remains the existing dry-bank reference evaluated with
wet gas properties and real physical geometry; no wet film or fewer-than-four-
row pressure-drop correction is introduced.

Rating sizing diagnostics include `required_area_scale`,
`required_thermal_outside_area`, `installed_physical_outside_area`,
`physical_surface_overdesign`, and `required_area_temperature_residual`.
Hydraulic length and flow-area diagnostics describe installed geometry.
Required tube lengths and required geometry are not Rating outputs. The
surface margin and reported wet UA margin share the same process reference.

Before the required-area correction, integration acceptance gates passed:
the complete public test collection, private project checks and four strict
notebook executions. Those runs do not validate the corrected Rating contract.
Release remains HOLD; numerical acceptance does not authorize a release.


## Numerical onset evaluation

The production source-profile moisture law is evaluated in its algebraically
identical deficit form: with bypass fraction B and effective saturated state
Teff, `Win-W=(1-B)*(Win-Wsat(Teff))*hv(Teff)/hv(Tgas)`. The property models are
affine in W. This avoids subtracting large absolute enthalpies to recover a
vanishing moisture difference. It does not replace the source-profile closure
with a local transport ODE, interpolate models, or clip W/condensate.
Property inversions and the wet-length root use tighter precision at onset.
The affine humidity/enthalpy slope uses a caloric-scale difference, avoiding
roundoff from a one-joule perturbation at large duties. Identical forcing
integrals are reused only within one unchanged coefficient iteration; the
cache is reset for each update. Neither change alters the profile equations.
The isolated frozen source-reference kernel and fixtures remain unchanged.


`wet_fraction` is the native axial wet-region fraction. `wet_surface_fraction`
and `wet_area` describe the actual wet exposed area, including partial radial
wetting of circular fins. Their area-weighted wet temperature is obtained
from the same radial states, not a linearly shifted legacy wall envelope.
The native wet fraction can reach one while fin tips remain dry.
For Simulation with thermal reserve, `wet_area` and `outside_total_area`
refer to the thermally active areas of that solve; installed physical area
and hydraulic geometry are reported separately and remain unchanged.

Legacy activation-band and outer-iteration controls do not redefine the
production source onset or its internal numerical acceptance criteria. They
remain applicable to their existing legacy paths. The Elmahdy-Mitalas source
moisture closure requires Lewis number one. Unsupported geometry/enhancement
paths retain their existing dispatch and controlled errors.


## General regime complementarity (decision 18C)

Production uses the general partial-wet interface construction at every wet
fraction. A stable dry cold surface gives f=0; otherwise the dry-limit
interface residual selects an interior root or the unstable dry-boundary
limit f=1. The separate all-wet hot-surface criterion no longer overrides
an interior partial solution. No resistance is fitted, blended or rescaled
for matching. The moisture and coupled-drain equations are unchanged.

At f=1 the general saturation secant uses the cold surface and inlet dewpoint.
The specialized source full-wet secant uses the actual cold/hot surfaces.
They are distinct away from their common boundary. Production follows the
general limit; the frozen EnergyPlus source kernel and pinned parity tests
remain unchanged. This approved production/source difference exists even
with drain disabled, independently of the property/surface adaptations.
For the pinned initialized-coefficient family at Win=0.016, no-drain source
Q=11540.116721 W, Tout=16.732164 C, Wout=0.0110801403, whereas the general
production limit gives Q=11330.223554 W, Tout=16.896544 C, Wout=0.0111793463.
It is a declared formulation difference, not an expanded parity tolerance.

## Native thermal-state provenance

`thermal_state` uses native production film correlations, their actual
Nusselt/correction breakdown, the converged property residual and native
profile wall temperatures. Optional wall-property objects are absent because
these mean-property correlations do not evaluate a wall-property correction.
U/UA remain the approved process-equivalent reporting quantities.
The hydraulic snapshot's independent dry thermal solution is discarded.

For AUTO circular fins, `finned_tube_diagnostics.thermal_reporting_basis` is
`native_dry_constitutive_network`: its efficiencies, resistances and U/UA
identify the actual dry constitutive network used by the production adapter
at the solved fluid state and physical area. They do not describe a scalar
resistance network for the latent process. The authoritative whole-process
U/UA are on the public result and thermal_state and retain their equivalent
basis flags. The generic outside film coefficient includes the common
root/contact resistance on its historical gross-area basis; native process
operators retain their own explicit common/parallel topology. Wet radial
states, conductances and effective areas remain nested in wet_coil_diagnostics.


## Exact property batching

Radial quadrature groups uncached saturation enthalpy queries on the same
quadrature points and finite-difference temperature stencil. The IF97
Region 1/2 coefficient order, saturation-pressure roundtrip and per-point
reductions remain the scalar backend operations; other regions use the
existing scalar path. No property interpolation or temperature rounding is
introduced. Exact scalar equality is checked separately from radial and
coupled-process convergence. This changes evaluation cost, not constitutive
properties or quadrature/admissibility tolerances.

The radial property batch spans all wet half-cells of a fin evaluation.
Each face retains its original wet-support nodes, derivative stencil and
integration order. Grouping the exact property rows avoids repeated small
array evaluations; scalar face integrals and their derivatives remain equal.


## Physical transition verification

The focused `wet_coil_transition_test.py` sweep holds geometry, dry-carrier
flow and liquid flow fixed and varies only inlet humidity. Both transitions
pass for BareTube and CircularFinnedTube. Humidity offsets of 1e-6 and 1e-8
on each side demonstrate shrinking differences; every fine difference is
less than one tenth of its coarse counterpart. Predeclared fine-pair limits
are: axial wet fraction 1e-4, liquid duty 0.1 W, both outlet temperatures
2e-4 K, outlet humidity 1e-7 kg/kg dry carrier, condensate 1e-7 kg/s, and
drain enthalpy 0.005 W. Wet fraction approaches zero/one from the interior.
Each point independently closes mass within 2e-10 kg/s and endpoint gas /
liquid / drain energy within 0.002 W; humidity and condensate signs are
checked directly. No branch interpolation or post-solve correction is used.


The earlier focused public contract group passed all 19 cases using the
superseded physical-length Rating contract. The corrected roundtrip uses
reserved-area Simulation -> required-area Rating -> reserved-area Simulation
on the same installed geometry. Coverage retains all three regimes on both
geometries, both geometry thermal reserves at 0/5/10%,
the joint unknown-inside-flow inverse, exact dry onset, explicit DISABLED
routing and the reporting identities. Larger integration regressions and the
complete public collection also passed: 1404 tests, no failures, errors or
skips. Coverage was reconciled against the complete collected node list after
an explicit interruption: 1227 completed results were preserved and the
remaining 177 cases were executed on unchanged source.

Seven private project cases and four clean notebook executions passed with
unchanged process inputs. Independent endpoint mass, profile mass, gas/liquid
energy and physical drain-integral checks retain their declared tolerances.
The enabled workbook preserves the standard sheets and numeric equivalent
thermal fields, identifies model/provider/basis provenance and contains no
run_metadata sheet. Private datasets, execution logs and outputs remain
outside version control.

Profile property derivatives use a fourth-order centered stencil with a
0.02 K initial step. This replaces cancellation-prone 0.001 K two-point
quotients in the unchanged source-profile moisture derivative. Property-domain
failures shrink the numerical stencil; all process, local-admissibility and
mass/energy tolerances remain unchanged. A captured multi-megawatt trial had
otherwise stationary temperatures but a noisy drain integral and exhausted
250 iterations. Fourth-order steps 0.01/0.02/0.05 K reproduced its duty within
0.000001 W and converged at the original gates. The source-reference kernel
and its pinned fixtures are unchanged.

When radial-drain quadrature is refined, the prior converged process and
radial drain correction provide the initial iterate on the finer grid. The
finer solve repeats every original convergence and independent balance check.
A large-bank comparison verifies that this initial guess reproduces a fresh
solve on the same final quadrature. Iteration diagnostics count the accepted
final-grid property solve, as before.

At the maximum axial order, the independent physical radial drain and the
profile drain use identical nodes. Any remaining discrepancy is therefore a
coupled constitutive iteration residual. The solver continues updating that
field until the original 0.0002 W whole-coil gate passes, within its existing
80-iteration bound; it does not increase the tolerance or alter condensate.

The Rating regime-pair fixture uses a common 360 K gas outlet target and
varies only coolant inlet temperature (20/60 C), retaining installed hydraulic
geometry. The fixture retains wet/dry and condensate assertions and also
checks both physical onset signs.

Caloric and saturation-enthalpy inversions use a 5e-14 K absolute root
tolerance, with the existing Brent relative tolerance. This reduces
inverse-root noise in the coupled multi-megawatt drain profile. No property law or profile/mass/energy acceptance
tolerance is relaxed; the frozen source-reference kernel remains unchanged.


## Applicability and computational cost

These gates verify the implementation, balances and forward/inverse
consistency. They do not establish experimental accuracy. Applicability
remains limited to the declared counterflow mean-property/secant model and
local correlation ranges.
Circular-finned pressure drop remains the existing dry-bank reference;
condensate-film hydraulics are not added. Large finned inverse cases require
repeated coupled profile and radial solves and can take substantial time.
VDI, Jeong and the future wet-model selector remain outside this integration.
