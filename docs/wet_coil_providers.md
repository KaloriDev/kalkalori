# Global wet-coil model providers

`BareTubeHeatExchanger.simulate` and `.rate` accept a direct model object as
`wet_coil_provider`. Omission selects `ElmahdyMitalasWetCoilProvider`:

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
length. The Elmahdy adapter calls the existing engine without moving equations
or altering solver options. Legacy and other wet models are not included.

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
