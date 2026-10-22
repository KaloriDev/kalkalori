# v0.9.3 public Chakroun–Al-Fahed provider verification

Implementation date: 2026-10-22. This records equation/software checks using
synthetic public inputs, without commercial-reference calibration or new
experimental validation. See [the model guide](../twisted_tape_chakroun1995.md)
for complete equations, usage and applicability.

## Repository boundary

- Initial branch: `main`; tracked tree and staged index clean.
- Recorded base HEAD and initial `main`:
  `5da3ab5a819bc67bd05561935b9013e184887005`.
- Implementation branch: `feature/v0.9.3-chakroun1995-twisted-tape`.
- Research archive: `aab39f409602a0d0c06ae70388f844801c3beeb2`, one local
  commit above the base, inspected read-only for engineering evidence.
- No merge, cherry-pick, push or tag. No research artifacts were copied.
- Private `.kon` material was neither read nor modified. No public module
  or test imports private/research modules, and no PDF, property tables from
  named oils, source illustrations or anonymous-reference outputs are added.
- Existing Yang2020, Rossi2017, enhancement contracts, integration adapters
  and solver implementation files have no changes from the base.

## Changed public files

- `core/enhancements/chakroun1995.py`: standalone public paired provider.
- `core/enhancements/__init__.py`: explicit public export.
- `core/__init__.py`: dynamic package version 0.9.3.
- `core/tests/chakroun1995_test.py`: synthetic anchors and solver integration.
- `docs/twisted_tape_chakroun1995.md`: provenance, equations and usage.
- `docs/tube_side_enhancements.md`: provider discovery.
- `docs/references.md`: 1995 authority and 1996 bibliographic context.
- `README.md`, `CHANGELOG.md`, `docs/roadmap.md`: release metadata and scope.
- `docs/validation/chakroun1995_v093.md`: this verification record.

## API and physical checks

`ChakrounAlFahed1995TwistedTapeProvider(strict_source_domain=False)` uses
existing `TwistedTapeGeometry`, `TubeSideEnhancement`, `EnhancementInput`
and `EnhancementResult`. No parallel solver or generic API extension was
needed; `requires_wall_state=True` activates the existing coupled wall path.
The provider is explicitly selected and does not change default dispatch.

Eq.17 uses `0.318*Sw^0.569*Pr_bulk^0.3*(mu_bulk/mu_wall)^0.14` and
`h=Nu*k_bulk/D`. Width prediction follows actual `Ac=A0-delta*W`, through
Va, Vs, Re_sw and Sw. The printed nominal Eq.11 is diagnostic only; its Sw
and actual-area Sw coincide at W=D. Eq.16 retains 15.767 and its native
Fanning convention. Source Darcy friction is exactly four times native
Fanning; `phi²*s³` then transforms to the engine's empty-bore axial basis.
Tests prove identical helical/axial dp at 0.25, 3 and 24 m for both source
and target geometries. No double length or factor-four conversion occurs.

The equation/convention anchor at Sw=1000, D=14 mm and delta=0.5 mm checks
`(f_F*Re)_sw=85.2156457` and `(f_D*Re)_sw=340.862583` to the precision of
those rounded anchors. They are not experimental fit data.

Known source limits are empty-bore Re=230–2300, Y=H180/D=3.6–7.1 and
W/D=10.8/14–13.2/14. Lower positive Re and out-of-range Y/width warn in
permissive mode and reject in strict mode. Re above 2300 is always rejected.
The inclusive reported source endpoint is retained at the regime boundary.
Only delta/D=0.5/14 was tested; a 1% relative reporting tolerance is an
implementation choice, not a validated thickness interval. The target
delta/D=0.9/21 warns (20% greater). Source Pr bounds remain unknown.

Cooling retains the equation and viscosity ratio unchanged, warns in
permissive mode, and rejects in strict mode. Informational diagnostics
describe the constant-wall-temperature heating source and incomplete
width/leakage representation. Similarity scaling, unknown Pr coverage,
other fluids, variable wall conditions and arbitrary-clearance performance
remain limitations; strict mode does not certify them. The 1996 full
equations were not independently audited.

## Synthetic Rating and Simulation

The distributable test fixture uses D=21 mm, W=19 mm, delta=0.9 mm and
H180=76 mm, 2 m physical axial length, and original high-Pr liquid states
in `TabulatedLiquidProvider`. There is no named-oil dataset. Both heating
and cooling run through normal Rating and Simulation (iterated and
noniterated), including the Rating-to-Simulation bridge. Tests check
coupled wall viscosity, finite h/dp, direction, warning propagation and
three-point distributed hydraulic quadrature.

Representative heating run (300 K liquid inlet, 340 K outside inlet):

| Quantity | Simulation | Rating |
| --- | --- | --- |
| h inside, W/(m² K) | 311.605838061 | 311.605838061 |
| Distributed inside dp, Pa | 7632.98044934 | 7632.98044934 |

Simulation and the Rating simulation bridge converge. Simulation gives
outlet 300.418682048 K, duty 5318.25803990 W, wall 304.309461627 K,
mu_bulk/mu_wall=1.09021107497 and Pr_bulk=353.906998106. These values
describe the software fixture, not measured validation or commercial targets.

## Test execution

Successful focused command under Python 3.11:

```powershell
py -3.11 -m pytest core/tests/chakroun1995_test.py core/tests/yang2020_twisted_tape_test.py core/tests/rossi2017_test.py core/tests/enhancement_integration_test.py -q
```

Result: **107 passed in 3.82 s; 0 failed, 0 errors, 0 skipped, 0 timeouts**.
This includes 38 new Chakroun cases, unchanged Yang/Rossi equations and
existing enhancement solver regression tests. An initial attempt with the pre-existing `.venv` Python
3.8 stopped during collection (4 errors in 1.05 s, no tests executed): the
repository requires Python >=3.10. No compatibility workaround or environment
modification was made; subsequent execution uses installed Python 3.11.

The full public suite is run once with:

```powershell
py -3.11 -m pytest core/tests tests -q --junitxml=$env:TEMP\kalkalori-v093-public-tests.xml
```

Result: **2113 passed in 4463.37 s (1:14:23); 0 failed, 0 errors,
0 skipped, 0 timeouts; exit code 0**. The complete public suite ran once,
without retries. JUnit confirms all 2113 test cases, including the 38 new
Chakroun cases, with no failure/error/skip records. Focused counts overlap
this full run and must not be added to it. Its temporary JUnit artifact stays
outside the public repository. No executable files changed during the full run.

For timing context, the preceding [Zhao verification](zhao2017_v092.md)
recorded a 4347.63 s serial public run, dominated by pre-existing wet/finned
and phase-transition cases (including a 674.79 s transition fixture setup).
Those tests and their tolerances are unchanged in this patch.

Final boundary checks: `git diff --check` and the staged whitespace check
pass. The staged file allowlist contains exactly the 11 public files listed
above, all Python or Markdown, with no `_validation_runs/`, `.kon/`, PDF or
private/source data. Base-to-feature diffs for Yang2020, Rossi2017, shared
enhancement contracts/adapters and solver code are empty. The commit uses
`Add Chakroun-Al-Fahed 1995 twisted-tape provider`; its resulting hash is
reported with the completed task. No push, merge or tag is performed.
