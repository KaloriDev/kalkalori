# v0.9.5 Phase A validation

Validation date: 2026-10-27. Development branch:
`feature/v0.9.5-serrated-fins`. Exact base:
`ff4f9137554257af6fd1d7ea57a35f9169c30288`, the default-branch merge containing
finalized v0.9.4 and the optional Genic providers. Source version: **0.9.5**,
unreleased. No v0.9.4 history was rewritten.

## Public verification

Python 3.11 was used. All code and test sources remained fixed during the
complete public-suite run. The final source-hash comparison passed.

| Coverage | Result |
| --- | --- |
| Targeted serrated geometry/dispatch and solid-fin/provider regression | 471 PASS |
| New serrated invariants, reference geometry and dispatch guards | 124 PASS |
| Existing public collection from v0.9.4 | 2,328 PASS |
| Complete public collection, including new cases | **2,452 PASS** |

The full run reconciled all 2,452 collected test IDs with disjoint passing
reports: no missing IDs, failures, skips or xfails. Completed partition results
covered 1,344 IDs; four workers then covered the remaining 1,106 regular IDs,
using whole-module scheduling to avoid repeated expensive module fixtures.
The two original inverse-area default-deadline tests passed serially after all
workers finished (580.30 s combined). Their original 300-second solver limits,
assertions, tolerances and physics were unchanged.

Coverage includes existing circular-finned geometry and efficiency, outside
dispatch, Briggs-Young/Robinson-Briggs, Frass/ESCOA, Genic, dry/wet thermal and
hydraulic regression, and affected Rating/Simulation APIs. No solid-fin
correlation equation or numerical evaluation was changed.

Equivalent serial commands for the public collection and new cases are:

```powershell
py -3.11 -m pytest -q
py -3.11 -m pytest core/tests/serrated_finned_tube_geometry_test.py core/tests/serrated_finned_dispatch_test.py -q
```

The version/import smoke check passed for `core.__version__ == "0.9.5"`,
`SerratedFlowAreaProvider` and `SerratedFinEfficiencyProvider`.
`git diff --check` passed. Local execution logs and the scheduling helper
remain in ignored build storage; no private case output or source PDF is
part of this change.

## Scope of the result

The reference case gives fin volume **0.0003112338269 m3/m**, fin mass
**2.443185541 kg/m**, fin area **1.173311399 m2/m**, primary area
**0.06483461839 m2/m** and total external area **1.238146017 m2/m**.
The model is an engineering radial-sector-cut idealization, without
manufacturer CAD validation or physical product exactness claims.

Serrated A_min, fin efficiency, HTC and pressure-drop models are intentionally
unimplemented. Explicit and automatic solid-fin substitution is rejected,
including through supplied-film UA and dry/AUTO Rating/Simulation paths.
This record verifies the geometry/API foundation and regression consistency;
it does not establish empirical serrated thermal/hydraulic accuracy. See
[the geometry and future-provider guide](../serrated_fins.md).
