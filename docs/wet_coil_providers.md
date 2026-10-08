# Global wet-coil model providers

`BareTubeHeatExchanger.simulate` and `.rate` accept a direct model object as
`wet_coil_provider`. Omission or explicit `None` selects
`ElmahdyMitalasWetCoilProvider` for both operations:

```python
from core import ElmahdyMitalasWetCoilProvider, WetCoilSolverOptions

result = hx.simulate(
    inside, outside,
    wet_coil_provider=ElmahdyMitalasWetCoilProvider(),
    wet_solver_options=WetCoilSolverOptions(timeout_s=300),
)
```

The public `WetCoilModelProvider` protocol declares stable `model_id`,
`model_name`, `source`, textual `applicability`, an `is_applicable` predicate,
and independent `supports_simulation` / `supports_rating` flags. A model can
support just one operation. Unsupported operation methods can raise
`WetCoilProviderUnsupportedError`; the dispatcher checks flags before calling
them. There is no registry, discovery, or string-based model selection.
`model_id` is diagnostic metadata, not a selection key. External provider
implementations may be supplied through the public provider API and are
outside the KalKalori distribution. They can implement the protocol
structurally in a separate package, using the public input/result types;
subclassing, registration and solver-routing changes are unnecessary.
The separate-package fixture in `tests/fixtures/external_wet_provider` and
`core/tests/external_wet_coil_provider_test.py` demonstrates this boundary.

Provider `simulate` takes the existing `HXSideInput` objects and returns
`HXSimulationResult`. Provider `rate` takes `BalanceSideSpec` objects and returns
`HXRatingResult`. Both receive the exchanger, phase-change `settings`, existing
operation keyword options, the exact `wet_solver_options` object and a
`WetCoilOperationContext`. This is the original whole-operation budget, started
before public preflight, with `options`, monotonic `started` and `deadline`, and
`check(**state)`. Providers must check it during iterative work, reuse it in
nested solves, and pass their provider selection to nested public calls.
The dispatcher also checks the budget before and after solving. It does not
create a fresh timeout for model dispatch or Rating trials.

Outside `PhaseChangeMode.DISABLED` bypasses the wet provider, including its
applicability predicate. AUTO uses the selected provider. An explicit provider
with unsupported operation or applicability raises
`WetCoilProviderUnsupportedError`. Exceptions and invalid results propagate;
they never trigger fallback to Elmahdy. When no provider is supplied, existing
dry and non-wet routing remains in effect.

Simulation uses installed physical geometry and hydraulics with existing
thermal surface-margin semantics. Rating solves required thermal area / area
scale with installed hydraulics; it does not infer required physical tube
length. The Elmahdy adapter retains the same equations and solver options. Its
BareTube numerical implementation uses exact property reuse and staged Rating
with a strict final solve; finned and Legacy behavior remain unchanged. See
[numerical performance](wet_numerical_performance.md).

`ElmahdyMitalasWetCoilProvider` remains the default for Simulation and Rating.
`LegacyBulkMeanWetCoilProvider` is available only by explicit selection and
supports Simulation only:

```python
from core import LegacyBulkMeanWetCoilProvider

result = hx.simulate(
    inside, outside,
    surface_margin=0.10,
    wet_coil_provider=LegacyBulkMeanWetCoilProvider(),
)
```

Legacy restores the historical KalKalori bulk-mean outside-condensation model
from `c80e779` (v0.8.2): a dry onset screen, whole-HX coupled enthalpy/water/wall
iteration, fully drained condensate, and Chilton-Colburn/Lewis mass transfer.
Its wall/wet-area envelope is a **0D estimate**, not an axial distribution.
CircularFinnedTube uses the historical nonlinear wet annular-fin FVM, including
its endpoint cold-zone fallback and dry-collapse behavior.

The supported outside phase change is H2O condensation from a carrier gas on
BareTube or CircularFinnedTube, with a sensible inside fluid (liquid or the
historically tested dry gas). Installed bundle flow semantics and existing
geometry/correlation guards apply; a conflicting per-call flow override is
rejected. Tube-side enhancement, inside condensation,
and Rating are unsupported. Simultaneously active phase-change sides retain
the existing guard. Frost retains the historical warning/dry behavior; no ice
physics is added. Unsupported condensables are not routed to another model.
There is **no automatic fallback** between providers.

The v0.8.4 patch extends this historical bulk-mean Simulation provider to
nonnegative thermal `surface_margin`. Active inside/outside thermal areas are
`A_process = A_actual / (1 + surface_margin)`; the corresponding absolute
inside-film, wall, fin/root and contact resistances scale inversely with active
area. Physical HTC correlations still use the installed geometry. Sensible and
latent duty, condensate and outlet/wall states are solved together on this
process surface. Zero-margin behavior remains numerically compatible with
v0.8.3. Legacy remains Simulation-only; Rating raises the existing controlled
provider error. Omission or `None` continues to select Elmahdy.

`final_result.A_i` / `A_o` retain installed physical areas. Legacy diagnostics
`actual_outside_area`, `thermal_outside_area`, `thermal_inside_area` and
`thermal_area_scale` distinguish installed and process surfaces. Native
`outside_phase_change.outside_total_area` / `wet_area` and the wet-fin
whole-surface areas describe the active thermal surface; wet fractions retain
their native definitions. `UA_actual` and `U_mean` use installed area at the
solved working state, while `UA_process = UA_actual / (1 + surface_margin)`
reports active conductance and `EMTD = abs(q) / UA_process`. This reporting
division does not derate duty again. Legacy's historical `Q_full` alias still
equals the achieved wet duty; it does not represent a separate zero-margin
simulation.

Legacy uses the existing `phase_change_*` convergence settings. Generic wet
energy/mass/outlet-temperature tolerances are not reinterpreted as legacy
residual tolerances. The same operation deadline is checked before/after the
dry baseline and during the global and radial iterations.
As historically, exhausted iterations return `converged=False`; callers must
check this flag. Increasing `phase_change_max_iterations` permits more work
without changing convergence tolerances or the operation deadline.

Hydraulics use installed geometry and the historical solved gas states. Wet
pressure-drop correction is not implemented; active circular-fin results
explicitly label outside dp as a dry/reference bank correlation.
Thermal margin does not change tube count, lengths, flow/frontal/free areas or
hydraulic diameter. Velocity, Reynolds number and calculated pressure drop can
change as solved temperatures, gas composition and transport properties change.

Results retain native diagnostics. `wet_coil_diagnostics["provider"]` contains
the selected model's identity, source, applicability and support flags; the
same diagnostics are attached to the outside phase-change result.

`wet_coil_diagnostics["surface"]` is the generic nested surface reporting
contract. Providers should supply available values only. The Elmahdy adapter
reports these aliases of its existing accepted solution:

- `surface_temperature_min`, `surface_temperature_max`,
  `surface_temperature_wet_mean`, `dew_point_in`, `dew_point_out` (K).
  The wet mean is `None` for a dry solution.
- `onset_margin` (K): inlet dew point minus the dry cold-surface temperature,
  matching `PhaseChangeResult.onset_margin_K`; positive means possible wetting.
- `wet_regime` and `axial_wet_fraction` (0–1).
- `metal_wall_temperature_min`, `metal_wall_temperature_max`,
  `metal_wall_temperature_mean` (K), plus matching tuples of
  `interface_temperature` and `metal_wall_temperature` at the existing wall
  envelope probes. These preserve distinct interface and metal temperatures.
- For finned geometry, `fin_base_temperature` and `fin_tip_temperature` tuples
  (K) at those same probes. Where native radial states exist,
  `radial_wet_fraction` contains axial `coordinate` / `fraction` samples of
  locally wet outside area divided by total thermal outside area. This includes
  the primary surface and fins. A radial wet-front position is not synthesized.

Missing diagnostics are omitted rather than estimated. Providers may preserve
additional native values alongside the generic surface dictionary.

Legacy preserves the historical method under `wet_coil_diagnostics["native_method"]`
before the dispatcher attaches its provider identity. Its `surface` dictionary
maps native wall minima/maxima/wet mean, inlet/outlet dew points and onset margin;
`wet_surface_fraction` retains its original area definition and is **not** named
`axial_wet_fraction`. `wall_temperature_mean` and `global_core_wall_temperature`
remain separate. When a native wet-fin state exists, fin base/tip temperatures,
wet/dry boundary radius (possibly `None`), fin wet fraction, primary surface,
core wall and root surface temperatures are also reported. A distinct
condensate-film interface temperature is not synthesized.
Finned extrema retain their radial exposed-surface meaning, including any
native cold-zone offset; the onset envelope remains identified as a 0D estimate.
`H_drain` retains the model's drained-condensate enthalpy rate (W).

## Wet-gas process phase fractions

Since v0.8.5, the side-scoped `PhaseChangeResult` exposes
`gas_phase_mass_fraction_in`, `gas_phase_mass_fraction_mid` and
`gas_phase_mass_fraction_out`, together with complementary
`liquid_phase_mass_fraction_in`, `liquid_phase_mass_fraction_mid` and
`liquid_phase_mass_fraction_out`.

The reference mass is the total inlet process stream, including water that
later drains as condensate. For conserved dry-carrier mass `md` and humidity
ratio `W` in kg water vapor / kg dry carrier:

```text
m_reference = md * (1 + W_in)
x_gas = (1 + W) / (1 + W_in)
x_liquid = (W_in - W) / (1 + W_in)
x_gas + x_liquid = 1
```

An inlet flow of 1000 kg/h with 100 kg/h cumulatively condensed therefore has
an outlet gas fraction of 0.9 and liquid fraction of 0.1. Without condensation,
the fractions are 1 and 0. The liquid fraction includes drained condensate;
it does not imply suspended liquid in the gas-property or hydraulic state.

The mean fraction uses the stored `W_mid`, preserving the solver's existing
mean-state convention. Missing humidity/reference data returns `None`.
Supported wet-gas models prohibit re-evaporation, and these derived fractions
are not clipped to conceal an inconsistent result. Component composition and
humidity ratio are unchanged. Pure-water/steam uses its existing whole-stream
vapor quality instead of the dry-carrier humidity basis.
