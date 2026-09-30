# Public API and thermal-state audit for v0.8.3 integration

Compared field declarations, defaults, properties and positional constructor
prefixes against both `v0.8.2` and starting HEAD
`c80e779fc61b35c3bc77751e5d6809a31129c0f0`. All existing fields retain their
names, types, defaults and positional order. No existing numeric field became
Optional. This audit describes the integration working tree, not a release.

| Public class / fields | v0.8.2 type and meaning | Integration type and meaning | Compatibility |
|---|---|---|---|
| HXRatingResult: 20 existing fields | Existing declarations | Same declarations and constructor prefix | Preserved |
| HXRatingResult: U_mean, UA_required/UA_process, UA_actual, EMTD | float, conventional thermal reduction | float, approved process-equivalent reduction for eligible outside AUTO | Explicit wet-path basis |
| HXRatingResult: A_o, A_required, overdesign_factor, ua_margin | float, installed/required area and reserve | float, installed/required physical area; equivalent UA and area ratios agree | Approved physical sizing contract |
| HXRatingResult: alfa_i, alfa_o, Q_required and pressure-drop accessors | Numeric corrected HTC, duty, physical hydraulics | Numeric native HTC, liquid duty, real-geometry hydraulics | No Optional substitution |
| HXSimulationResult: 33 existing fields | Existing declarations | Same declarations and constructor prefix | Preserved |
| HXSimulationResult: U_mean, UA/UA_actual/UA_process, EMTD | float, conventional thermal reduction | float, approved equivalent reduction of reserved-area wet process | Explicit wet-path basis |
| HXSimulationResult: q, Q_full, Q_derated, overdesign_factor | float, duty and reserve | float, direct wet forward solves at reserved/full area; numeric reserve | Physical area scaling, real hydraulics |
| PhaseChangeResult: 47 existing fields | Existing declarations | Same declarations and constructor prefix | Preserved |
| PhaseChangeResult: Q_total, Q_sensible, Q_latent, wall and wet-area diagnostics | Liquid duty and declared sensible/latent/surface approximations | Liquid duty; fixed-inlet-W cooling plus isothermal removal minus integrated drain; native profile/radial surfaces | Active outside-wet semantics documented |
| IterativeThermalState: 25 existing fields | Existing declarations, converged mean thermal state | Same declarations; native film/correction data and profile walls, equivalent U/UA | No fabricated legacy state |
| WallTemperatureEnvelope and probes | Existing types, four-point legacy approximation | Same types; native wet/dry profile and radial sampling under explicit method | Active AUTO path only |
| FinnedTubeDiagnostics: 49 existing fields | Dry constitutive network and physical hydraulics | Same types; native dry constitutive operator at solved wet fluid states, explicitly labelled | Wet process U/UA remain on main result/state |

Added fields, all defaulted and appended:

- HXRatingResult and HXSimulationResult: `ua_reporting_basis: str`,
  `ua_is_equivalent: bool`, `wet_coil_diagnostics: dict | None`.
- PhaseChangeResult: only `wet_coil_diagnostics: dict | None`.
- IterativeThermalState: `ua_reporting_basis: str`, `ua_is_equivalent: bool`.
- FinnedTubeDiagnostics: `thermal_reporting_basis: str`.

Added properties:

- HXSimulationResult: `ua_margin: float`, exact alias of overdesign_factor.
- PhaseChangeResult: `H_drain: float`, `regime: str | None`,
  `wet_fraction: float | None`, `global_wet_model: str | None`, forwarding
  to the nested diagnostic object. These do not expand its constructor.

Removed fields/properties: **NONE**. Renamed fields/properties: **NONE**.
Changed existing field types/defaults: **NONE**. No new required arguments.

DISABLED, unrelated fluids and inside phase-change paths retain their
established semantics and defaults. Eligible AUTO that resolves DRY uses
Elmahdy-Mitalas calorics and equivalent reporting, per approved variant 1;
its difference from legacy DISABLED-DRY is intentional.

## Thermal-state provenance and consumers

The legacy hydraulic snapshot previously also supplied dry thermal values
and fin-network diagnostics. The wet mapping now discards those thermal
values and uses the native adapter's inside/outside correlation objects,
actual corrections and property residual. The residual is the solver norm:
maximum outlet/interface temperature change and 1000 times humidity change;
it is not a fabricated zero wall-iteration residual. Profile wall temperatures come
from the accepted solution. Wall-probe iteration counts and residuals refer
explicitly to that same accepted native property solve; no constant zero
residual or invented one-iteration solve is reported. Optional wall-property
objects remain None when
no wall-property correction was evaluated; no dry values are inserted.

Finned resistance/efficiency diagnostics are rebuilt from the actual native
dry constitutive operator, with `thermal_reporting_basis` explicitly set to
`native_dry_constitutive_network`. That operator is a component of the
accepted model, not a separately solved legacy dry process. Its network U/UA
are not promoted to wet process U/UA. The latter use the approved equivalent
basis. Physical root/contact topology and wet radial states are preserved
in nested diagnostics.

Consumers include HXRatingResult/HXSimulationResult convenience properties,
physical-HTC and pressure-drop contract tests, and the private case-study
reporter's thermal/geometry rows. Their existing field access remains valid;
reports must carry the basis of fin efficiencies/resistances alongside the
process-equivalent U/UA to avoid identifying them as one scalar wet network.
No structural fallback to fabricated IterativeThermalState data is required.

Validation status: focused integration gates, all 1404 public tests, seven
private project cases and four strict notebook executions passed. Existing
public field declarations and constructor prefixes retain the audited
compatibility described above. Release remains HOLD.


Simulation temperature residuals are the actual last native outlet changes
for each stream; its relative-duty residual is the independently checked
gas/liquid/drain energy imbalance divided by liquid duty. Probe and
thermal_state residuals use the documented native maximum property norm.
These are measured native diagnostics, without fabricated zero residuals.

The existing optional `wet_finned_surface` accessor remains available but is
None on the native Elmahdy-Mitalas path: a single legacy surface solution is
not a representation of this axial/radial process. Actual fin surface states
are in `wet_coil_diagnostics["surface_states"]`. This is an explicit change
of the optional active-outside-wet diagnostic, not a removed field or a
change to its type. Legacy paths keep their existing optional diagnostics.
