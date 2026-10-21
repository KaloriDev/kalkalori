# v0.9.2 elliptical API cleanup verification

This record covers the public engineering API cleanup on
`feature/v0.9.2-zhao2017-airside`, starting at
`8599e9ef9e0105ed84ed421397021f71f5fa3268`.
See [API rules and migration](../elliptical_api_v092.md).

## Audit and retained controls

Independent ellipse contours, fluid state and fin dimensions are physical
inputs. `TubeWallModel` and `Zhao2017AirsideModel` are advanced model controls.
`EllipticalFlowAssumptions` remains useful for direct analytical H1 and source
assumption verification; normal dispatch supplies model properties instead.
Legacy STAGE declarations are removed from the shared notebook builder.

The public and current private consumers were audited separately. Historical
archival snapshots retain their original meanings. Current private fin geometry,
builder, plotting, notebook examples and tests follow width=transverse and
height=streamwise. A read-only audit outside the sandbox completed successfully
for private files whose initial sandbox reads failed.

## Public regressions

The focused public command selected all elliptical geometry/internal/wall suites,
elliptical applicability, automatic dispatch, Zhao airside/integration,
rectangular geometry/network, 2D fin efficiency, airside correction and fin
turbulator metadata. Initial result: **315 passed in 12.87 s**.

The full run then exposed a malformed-geometry boundary regression: an object
merely declaring the elliptical family reached dimension access before actual
`EllipticalTube` validation. Early type validation was restored for HTC and
friction dispatch, retaining the original guard expectation. The corrected
dispatch passed **99 targeted tests in 7.47 s**. These focused groups overlap.

Coverage includes:

- Ordinary 36x14 mm outer / nominal 33x11 mm inner contours, automatic numerical
  conduction, incompatible forced exact rejection and unchanged inner-flow
  results when only outer contours/confocality change.
- Existing confocal/circle analytical comparisons and numerical wall convergence.
- Yang source endpoints, lower/upper source extrapolation within the turbulent
  regime, 500000 Re extrapolation, unchanged Eq.18 coefficients, strict rejection,
  laminar selection, unsupported transition and Rating/Simulation warnings.
- Width 26 mm / height 55 mm, clearances 0.67/5 mm at Pt=26.67/Pl=60 mm,
  unchanged source reconstruction and fin-mesh orientation.
- The previous synthetic Zhao integration remains
  Q=24058.39114704213 W, UA=481.30043424526707 W/K and
  outside source-domain pressure estimate=372.055553752331 Pa, at the original
  strict numerical regression tolerances. This verifies orientation preservation.

## Full public run

Equivalent serial command: `py -3.11 -m pytest core/tests tests -q --tb=short`.

The complete collection contains **2075 tests**. Final verification covered
all 2075 unique node IDs with no omissions, unexpected nodes or duplicates.
Whole modules ran across four isolated pytest processes. That phase recorded
**2073 passed and two deadline expiries in 2339.9095692 s wall time**.
Both expiries were existing wet Rating checks with a 300-second production
limit. The same unmodified tests then ran without competing test workers:
**2 passed in 471.43 s pytest time** (473.8614456 s wrapper wall time), with
individual calls of **212.76 s** and **254.78 s**. The original deadlines,
assertions, solver inputs, numerical tolerances and coefficients were unchanged.

Final result: **2075 unique tests verified passing**, zero unresolved failures,
zero errors/skips. There were 2077 attempts including the two documented
isolated retries. Test-execution phase wall time totals **2813.7710148 s
(46m 53.77s)**. This is a coverage-verified full run with explicit retries;
it is not a claim that the initial concurrent invocation had zero failures.
Raw and retry JUnit/log receipts remain private under ignored `.kon`.

An earlier sequential discovery run exposed the malformed-geometry boundary
regression described above and was stopped after diagnosis. The corrected
boundary test passed both its targeted rerun and the complete final verification.

## Scope

No calibration, coefficient tuning, new empirical correlation, circular fallback,
transition law or thin-wall approximation was introduced. Existing smoothness,
shape, source-coefficient, geometry-validation and numerical-convergence guards
remain. Development and thermal boundary conditions are visible model assumptions;
automatic selection does not establish experimental applicability for every duct.
Wet/phase-change rectangular fins, developing/transition internal physics,
bare-ellipse outside-bank physics and validated ellipse tube-sheet/local losses
remain outside this implementation's scope.

Private customer inputs, execution details and case results remain under ignored
`.kon`; they are not part of the public validation fixtures.
