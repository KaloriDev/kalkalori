# v0.9.4 release integration validation

Final integration record: 2026-10-27. Branch:
`feature/v0.9.4-inline-round-finned`. Package and built-wheel version: **0.9.4**.
This is qualified engineering support; unresolved source conditions remain
visible and strict source-domain mode continues to reject them.

## Provider policy

| Round solid-fin bank | Thermal AUTO | Explicit thermal alternative | Hydraulic AUTO | Explicit hydraulic alternative |
| --- | --- | --- | --- | --- |
| INLINE | `FrassEscoaInlineSolidFinProvider` | Same provider, `frass_escoa_inline_solid_fin` | `FrassInlineSolidFinPressureDropProvider` | `Genic2006InlinePressureDropProvider` |
| STAGGERED | `BriggsYoung1963Provider` | `FrassEscoaStaggeredSolidFinProvider`, `frass_escoa_staggered_solid_fin` | `RobinsonBriggs1966Provider` | `Genic2006StaggeredPressureDropProvider` |

The STAGGERED ESCOA addition does not change AUTO. Briggs–Young and
Robinson–Briggs retain their equations and source/geometry/layout guards.
There is no automatic ESCOA fallback, blending or calibration.

INLINE thermal support uses ESCOA Eqs.(96)–(98), and explicit STAGGERED
support uses Eqs.(60)–(62), from Frass/Hofmann/Ponweiser's legal secondary
2015 reproduction. The original ESCOA 1979 manual was not inspected.
Layout-specific C3/C5 remain explicit, including the squared STAGGERED row
term. Shared mechanics preserve INLINE's numerical evaluation and physical
film-HTC output, with efficiency applied separately by the thermal network.
The existing mean-fin-temperature coupling is used by both providers.

INLINE hydraulics use native Eq.191 xi and whole-bank drag at mean outside
gas state. Eq.190 row treatment is retained for 2–8 rows; above eight rows
Kz=1 remains an explicit engineering asymptotic convention. One row is
rejected. Eq.143 is not applied. Acceleration and external losses remain
separate; no internal conversion to Robinson–Briggs is introduced.

The two [Genić 2006 hydraulics alternatives](../genic2006_finned_hydraulics.md)
are explicit only, with native equivalent-diameter Reynolds number and loss
basis, not blended or fitted to either AUTO provider. The supplied equation
reconstruction is closed; calibration provenance is partial, raw calibration
data were not recovered, and complete independently reproducible public source
access remains unverified. These are separate provenance and source-policy
questions. INLINE retains its open Reynolds interval and calibration-geometry
warning. Constant thickness and five or more rows are required.

## Final regression after Genić integration

The complete public collection was rerun after the optional Genić providers
and the documented STAGGERED Reynolds-domain guard were added. The four-worker
partition passed **2,326** tests; the two unchanged inverse-area deadline tests
then passed serially. The disjoint reports cover all **2,328** collected IDs:
2,328 passed, zero failed, skipped, xfailed or missing. Code and test hashes
were unchanged throughout the run.

The 0.9.4 wheel was built from a clean snapshot of the public source because the
local ignored build directory contained stale files. The wheel's Python files
match the working source, and its contents exclude private case files, source
PDFs and temporary diagnostic outputs. Isolated wheel imports and Genić
numerical smoke tests passed. Private user files remain ignored and untouched.

## Historical public regression coverage before Genić

The frozen collection contains **2,263 unique public test IDs**.
The complete suite was partitioned using the established four-worker
work-stealing strategy, reserving only the two default-deadline inverse-area
tests for isolated serial execution after the parallel workers completed.

| Partition | Result | Elapsed seconds |
| --- | --- | --- |
| Parallel, four workers, work stealing | 2,261 PASS | 2849.72 |
| Two original default-deadline tests, serial | 2 PASS | 414.04 |
| Unique collected IDs with PASS evidence | 2,263/2,263 | Disjoint union |

There are no missing IDs, skipped tests or failed test results in these final
partitions. Matching uses each collected test's JUnit class/name identity,
including parameterized cases. Both deadline tests retain their original
300-second limits, assertions, solver budgets and physics. The deadline-test
source is unchanged from the original thermal checkpoint. Code/test hashes
are unchanged across the complete release run. `git diff --check` passes.

The existing 359-test focused record and 40 new STAGGERED cases remain
preserved. Nineteen release integration cases additionally cover both
explicit ESCOA selectors through low-level calls, `solve_thermal_state`,
Rating and Simulation; public exports/error messages; the ten-entry
layout/provider accept/reject matrix; and unchanged AUTO selection. All are
included in the complete collection above. A fresh rerun of the 503
pre-refactor INLINE direct/coupled comparisons remains numerically identical.

## Layout/provider guard matrix

| Provider | Supported layout | Other layout |
| --- | --- | --- |
| ESCOA INLINE thermal | INLINE: PASS | STAGGERED: REJECT |
| ESCOA STAGGERED thermal | STAGGERED: PASS | INLINE: REJECT |
| Briggs–Young thermal | Valid triangular STAGGERED: PASS | INLINE: REJECT |
| Robinson–Briggs hydraulics | Valid triangular STAGGERED: PASS | INLINE: REJECT |
| Frass Eq.191 hydraulics | INLINE: PASS | STAGGERED: REJECT |

No tolerance or applicability guard was relaxed for these release checks.
Selecting ESCOA STAGGERED thermal does not bypass hydraulic eligibility.

## Private workflow validation

The four original private INLINE workflows were rerun from the original
production notebook configuration: **4/4 PASS**. Explicit ESCOA STAGGERED
production thermal-only comparisons also passed **4/4**. Inputs and design
fouling are unchanged, and no export was triggered. Square-pitch staggered
hydraulics were not forced through Robinson–Briggs. Private geometries,
properties, numerical outputs and input hashes remain under ignored `.kon`;
none are included in this public record.

These cases establish execution and reporting under the stated engineering
policy, not experimental model accuracy. The independent correlations are
not treated as ground truth. The
[public synthetic comparison](escoa_staggered_comparison_v094.md) retains its
production-path/source-range distinction and numerical interpretation.

## Package and source checks

`core.__version__`, setuptools' dynamically expanded project metadata and an
offline-built wheel all report 0.9.4. The wheel contains both ESCOA providers,
the shared helper and native Frass hydraulics, with no private input/worklog
files. Build artifacts remain ignored; no package was published.

Required source-range limits and the original ESCOA property convention
remain unresolved. Root-layer and taper mappings remain engineering
conventions. Hydraulic heated normalization and loss decomposition remain
partial. No source range or accuracy claim is added by release validation.

Implementation checkpoints are separate:

- `e374329`: INLINE thermal.
- `fa531d7`: INLINE hydraulics.
- `d727c21`: explicit ESCOA STAGGERED thermal.

The final integration commit adds the release API/guard tests and consistent
release documentation. No release tag or publication is part of this task.
