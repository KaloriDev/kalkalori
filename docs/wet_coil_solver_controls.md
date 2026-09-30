# Wet-coil numerical controls

```python
from core import WetCoilSolverOptions, WetCoilTimeoutError

controls = WetCoilSolverOptions(
    energy_tolerance_W=1.0,
    mass_tolerance_kg_s=5e-7,
    outlet_temperature_tolerance_K=0.01,
    timeout_s=300.0,
)
result = hx.rate(inside, outside, wet_solver_options=controls)
# The same keyword is accepted by hx.simulate(). Omission uses these defaults.
```

All four numbers must be positive and finite; only `timeout_s` also accepts
`None`, meaning unlimited execution. There are no named accuracy presets.
The immutable options object governs convergence, not physical admissibility.
Humidity, condensate, saturation, driving force, temperatures, provider
applicability and regime checks are unchanged. No output is clipped.

One monotonic budget starts on entry to a public operation. Every nested wet
trial, property iteration, profile iteration and regime root uses it. Optional
Rating Simulation and Simulation reserve evaluations share that same budget.
Non-wet routes do not acquire a wet timeout. Independent internal forward calls
create their own budget. A timeout raises `WetCoilTimeoutError` (a
`WetCoilConvergenceError` / `RuntimeError`), never an accepted last iterate.
Its `diagnostics` contain elapsed/configured seconds, attempted forward and
property evaluation counts, and available length, flow, residual and regime
information. It cannot be swallowed by inadmissible-trial `ValueError` handlers.
Checks occur between numerical evaluations; an individual property call is
not preempted.

Rating stops when the specified outlet residuals meet the selected tolerance.
Nearby accepted trials supply an initial property/drain iterate only. Successive
wet-fraction/property evaluations also reuse fixed-point coefficients as seeds. Every
new geometry/flow still solves the same equations and passes all checks. Exact
saturation-inverse roots narrow subsequent Brent brackets without property
interpolation; all trials share one operation-local thermodynamics evaluator.
`wet_coil_diagnostics["solver_options"]` records the selected object;
`solver_statistics` records operation-wide counts and elapsed seconds.

## Inventory before this change

This inventory refers to production code at
`7833409d5d938ab24998b073932c23bd84cb871d`, not the frozen reference kernel.

| Layer | Previous numerical control | Current treatment |
| --- | --- | --- |
| Wet-region fixed point | Mixed kW/K/W iterate change < 2e-8; drain density change < 2e-5 W | Separate energy gates, each energy tolerance / 10; keep 2e-8 K surface/secant stability ceiling, optionally tighter with requested temperature/energy accuracy |
| Profile quadrature and integral closure | 2e-4 W, 2e-10 kg/s | Public energy and mass tolerances |
| Production property fixed point | max(outlet/interface temperature changes, 1000 × humidity change) < 2e-7; radial drain correction < 0.002 J/kg | Temperature convergence derived from outlet and energy tolerances; radial specific-enthalpy stability control stays internal |
| Liquid-provider duty check | 0.002 W | Public energy tolerance |
| Independent radial drain integral | 2e-4 W; refine axial quadrature 10 → 20 → 32 | Public energy tolerance; same refinement |
| Rating outlet acceptance | 2e-5 K; installed-length shortcut 2e-6 K | Public outlet temperature tolerance |
| Rating length root | log-length xtol 2e-8, rtol 1e-12 | Retained internal root limits; stop once outlet accuracy is met |
| Joint length/flow root | outlet residuals 2e-5 K; optimizer xtol/ftol/gtol 1e-10; diff_step 1e-4; 80 function evaluations | Public outlet tolerance with early completion; optimizer safeguards retained; Jacobian uses an absolute 1e-4 log-space step so it does not collapse at a 1 m starting length |
| Specified duty consistency | max(0.002 W, 1e-7 × duty) | max(energy tolerance, 1e-7 × duty) |

Other internal limits remain: 250 wet-profile iterations with 0.6 drain
relaxation; 200 dry-profile iterations at 1e-10 K; 80 production property
iterations; 26 length-bracketing doublings, up to 18 admissible-boundary
bisections; joint log-length/log-flow bounds ±10. Wet-fraction Brent roots use
`min(2e-10, 1e-6 × onset fraction scale)` with a positive floating-point floor.
Brent's default evaluation limit is 100. Property enthalpy inversions use
5e-14 K, inlet dewpoint 1e-12 K; finite-difference derivatives use a 0.02 K
step with up to eight halvings. These preserve stability near regime boundaries.

The local annular-fin solve retains 64 radial cells, 60 Newton iterations,
16 damping attempts, per-cell residual `max(2e-10 W, |fin heat| × 2e-10)`, and
independent radial quadrature floors 2e-14 kg/s and 2e-8 W with 2e-8 relative
accuracy. These are local constitutive/discretization safeguards, distinct
from whole-coil balance accuracy. Radial dewpoint accuracy remains 1e-10 K.

The expensive nesting is Rating residual/Jacobian evaluation → new geometry
and flow → production property iteration → dry/full/partial regime tests →
wet-fraction root → coupled profile/drain fixed point → thermodynamic
inversions (plus radial-fin solves and quadrature refinement when applicable).

Tests requiring numerical regression precision pass explicit tighter options;
their expected accuracy is unchanged. Unlimited validation is intentional and
must be requested with `timeout_s=None`.
