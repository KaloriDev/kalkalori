# BareTube wet numerical performance

`ElmahdyMitalasWetCoilProvider` uses optimized BareTube wet Simulation/Rating
numerics with unchanged physical equations and public provider selection.
CircularFinnedTube wet calculations retain their existing native path; Legacy
is unchanged. Numerical equivalence is not empirical model validation: faster
convergence does not improve correlation accuracy or extend model applicability.

## Problem and development

Wet Rating nests area/flow trials, property iterations, wet-region fixed points,
wet-front roots, quadrature points and exact property inversions. Difficult
cases repeatedly paid for all those calculations.

Three private experiments established the design:

1. **Staged Rating:** COARSE/MEDIUM/STRICT search, property snapshots, refreshes
   and boundary promotions greatly improved difficult joint Rating, but kernel
   costs still caused inconsistent performance.
2. **Exact kernel reuse:** shared states, direct derivatives, exact property
   reuse, region/front continuation and safeguarded fixed-point acceleration
   reduced strict-forward and Rating work.
3. **Cold initialization and fused properties:** low-order initialization,
   fused exact IF97 values/derivatives, structural reuse and deferred outlet
   work produced the production design. An additional initial front estimator
   did not reduce total work and was rejected.

## Illustrative benchmark history

Measurements used Python 3.11, NumPy 2.4, SciPy 1.17, CoolProp 8 and iapws 1.5.5
on one development machine. These hardware/environment dependent examples are
not an API performance guarantee. Historical columns were measured at different
times and are not simultaneous speedup measurements.

| Representative BareTube Rating | Previous production | Stage 1 | Stage 2 | Accepted stage 3 |
| --- | ---: | ---: | ---: | ---: |
| Difficult PARTIAL joint Rating | 189 s | 53 s | 15.4 s | 12.4–13.9 s |
| PARTIAL Rating | 38 s | 45 s | 9.3 s | 8.0 s |
| FULL Rating A | 22 s | 15 s | 3.7 s | 3.3 s |
| FULL Rating B | 19 s | 38 s | 8.9 s | 7.8 s |

Three fresh-process paired strict-forward measurements at identical area/flow
gave stage-2 → stage-3 medians of **5.62 → 2.87 s cold** and **1.84 → 1.50 s
seeded**. Cold timing includes initialization. A subsequent boundary-fallback
confirmation gave 3.06 s cold / 1.47 s seeded. Historical native cold was about
14.5 s. Profiling and memory-instrumented runs were excluded from these timings.

Production-integration checks at the same saved inputs gave **3.37 s cold /
1.52 s seeded**, versus fresh native measurements of 17.84 / 8.44 s. Cold CPU
time was 2.81 s optimized versus 14.28 s native. The four integrated Rating
times were **15.05 / 7.68 / 3.46 / 8.20 s** in the table's order; their reported
strict states matched the accepted stage-3 states exactly. These single-sample
checks confirm integration behavior and remain subject to environment variation.

At identical physical input, area and flow, strict results remained equivalent
to native at floating-point/solver-convergence scale. Final energy, moisture/
component closure, humidity bounds, driving-force guards and wet-front
acceptance were retained. Whole Rating can stop at a different admissible
point within the requested outlet tolerance; fixed-state equivalence is a
separate check.

## Production design

`wet_coil_provider=None` still selects `ElmahdyMitalasWetCoilProvider`.
Direct provider-object injection is unchanged. There is no public numerical
selector or second Elmahdy provider.

- Common physical equations support optimized and internal native/reference
  numerical paths. Native forward/Rating functions remain regression/debugging
  oracles. Finned calculations use native numerics.
- An order-3 cold initializer only seeds a new strict solve. Approximate
  property/inverse caches stay separate and cannot supply an accepted output.
  Rejected initialization resumes generic strict initialization with the
  original operation deadline.
- Exact temperature keys and local bundles share the configured dry-carrier
  EOS, saturation pressure and IF97 liquid/vapor enthalpy work. Strict uses no
  rounded-state substitution or property interpolation. Reference datums and
  the native pressure/temperature unit roundtrip remain unchanged.
- Fused derivatives apply to validated installed IF97 Region1/2 saturation
  from 273.16 K to below 623.15 K. The actual pressure-based Region3 boundary
  is checked near the upper limit. Exactly 623.15 K, adjacent floats selecting
  Region3 and the 640 K upper inverse probe retain native properties; profile
  derivatives outside the validated domain use the native numerical stencil.
  Unknown Region4 coefficient layouts use native work. Backend upgrades
  require revalidation.
- Previous region coefficients, drain polynomial and interface/stability state
  can initialize nearby solves. Safeguarded one-step acceleration and wet-front
  corrections evaluate actual residuals. Rejected steps retain damping or
  native Brent on a physical bracket with unchanged acceptance gates.
- Outlet-point work that does not advance the fixed-point vector is deferred
  until convergence. Iterative quadrature/drain work, complete final outlet,
  independent quadrature and physical checks remain authoritative.
- Exact Gauss rules are cached read-only. Scalar state dictionaries grow within
  an operation and are released with it. IF97 hot-term LRUs cap at 512 records
  per object, inverse tables clear at 4096 entries per table, and enthalpy LRUs
  cap at 16384 per object. No new global cross-operation property cache is used.

Rating keeps the accepted COARSE → MEDIUM → STRICT strategy. Approximate stages
locate the solution and create warm starts from constitutive snapshots. Refresh
thresholds are fixed; boundaries and rejected approximate states promote to
higher fidelity. Only strict production states enter the final report, honoring
the original [solver controls](wet_coil_solver_controls.md), including tighter
requests. Optional installed Simulation shares the same deadline.

Numerical-path diagnostics describe evaluation within the same physical model,
not fallback between providers. Final physical inadmissibility, closure,
applicability and convergence failures propagate; native retries do not hide
them. Installed geometry governs hydraulics; required thermal area governs
sizing. Thermal sizing/reserve do not change installed lengths or flow areas.
Wet-fin performance, pressure-drop models and tube-side enhancements are outside
this optimization.

## Verification and limits

Focused tests compare native/optimized DRY/PARTIAL/FULL states, both sides of
both regime boundaries, cold/seeded solving, rejected initialization, IF97
boundaries, tighter controls, shared deadlines and installed hydraulics. They
also check finned routing, cache isolation/lifetime and structural call savings.
Normal pytest contains no fragile wall-clock assertions. Reproducible
engineering benchmarks and experimental evidence remain private.

An initializer can be rejected on a valid case, adding work before strict
solving. Large/pathological operations can grow scalar caches. The operation
deadline is a work budget, not a throughput guarantee. Physical applicability
remains as described in [the production model](elmahdy_mitalas_production.md).
