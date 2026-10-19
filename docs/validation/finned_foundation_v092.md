# v0.9.2 finned foundation validation

Implementation base: merged public `main` / tag `v0.9.1`,
`36031d2de6a8f2d747c6929b4d386bf68df8cfff` (PR #28).
The six subsequent experimental provider commits are not ancestors of this
implementation. The existing experimental branch and `.kon` were untouched.

Validation environment: Python 3.11.9, NumPy 2.4.2, SciPy 1.17.1, Windows.
Tests use analytical geometry and an independently integrated straight-fin
solution, with no paid/private data or software validation reference.

The user-supplied decision record
`Free_Source_Airside_Models_KalKalori_Audit_2026-10-19.md` was reviewed against
the finished implementation. Its P0 scope (2D fin conduction and independent
geometry/area accounting) and explicit HTC/hydraulic/row source gaps agree
with this patch. Its historical "no implementation" status and proposed
backlogs describe the audit, not additional implementation instructions.
Public physics/provenance references remain the two verified primary free
sources; no audit equations for other models or private registries were used.

## Insulated straight-fin analytical benchmark

L=0.020 m, width=0.008 m, t=0.0005 m, k=200 W/(m K), h=100 W/(m2 K).
The elementary ODE gives theta(x)=cosh(m(L-x))/cosh(mL),
eta=tanh(mL)/(mL)=**0.797799455468**. The same 2D assembly/solve kernel
is tested on a rectangular strip with one fixed-temperature edge and three
insulated edges; no 1D shortcut is used in the numerical solution.

| Axial cells | Numerical eta | Absolute eta error | Relative residual | Energy imbalance |
|---|---|---|---|---|
| 16 | 0.797926752368 | 1.2730e-4 | 1.61e-13 | 8.70e-15 |
| 32 | 0.797825917672 | 2.6462e-5 | 6.22e-13 | 1.47e-14 |
| 64 | 0.797805615873 | 6.1604e-6 | 1.75e-12 | 9.46e-14 |

## Rectangular fin with elliptical contact

Outer ellipse semi-axes a=0.012 m, b=0.006 m; rectangle 0.040 x 0.030 m;
t=0.0004 m, k=200 W/(m K), h=80 W/(m2 K). Mesh counts below describe
the solved first quadrant. Analytical metal area is used as the independent
reference for mesh-domain area error.

| Radial / angular cells per sector | Nodes | Triangles | Eta | Relative mesh-area error | Relative residual | Energy imbalance |
|---|---|---|---|---|---|---|
| 8 / 8 | 281 | 512 | 0.896659056553 | 4.4354e-4 | 6.29e-14 | 3.97e-16 |
| 16 / 16 | 1073 | 2048 | 0.896385123431 | 1.1094e-4 | 2.28e-13 | 1.45e-14 |
| 32 / 32 | 4193 | 8192 | 0.896316447430 | 2.7739e-5 | 9.36e-13 | 1.81e-14 |
| 64 / 64 | 16577 | 32768 | 0.896299251605 | 6.9349e-6 | 5.38e-12 | 1.37e-14 |

Successive eta differences decrease approximately fourfold. The default
converges at 32/32, with absolute eta difference 6.8676e-5. Integrated
full-fin base and convective heat rates are respectively
0.139657911090259 and 0.139657911090262 W/K of base excess temperature.
This is numerical convergence evidence, not experimental validation of a
surface-specific convection or hydraulic correlation.

## Network and regression checks

For 24 tubes, active length 1.8 m and pitch 0.003 m, the prescribed-film
example in the guide gives A_physical=31.028406481780 m2,
A_effective=27.314139716524 m2 and G_outside=2185.131177321921 W/K at
h=80 W/(m2 K). Physical h is unchanged; the adiabatic outer edge contributes
no convective conductance. Inside h=1200 W/(m2 K) and prescribed wall
R=0.001 K/W give UA=532.836020161614 W/K.

Focused runs passed: geometry regression (132 tests), fin/geometry
(37 tests), network/ellipse/circular-fin/topology regression (105 tests),
and internal/wall/Rating/Simulation/topology checks (150 tests).
These runs overlap; counts are not additive. Initial test discovery with
the local Python 3.8 environment failed because the project requires >=3.10;
all reported successful runs used Python 3.11.

The full public suite was executed once after implementation:

```text
py -3.11 -m pytest core/tests tests -q
1930 passed in 4341.24s (1:12:21)
```

No failures, skips, xfails or warnings were reported. This includes all
45 new tests and the existing public regression tests, with no edits to
established numerical baselines. Most runtime was in existing strict wet
Rating/Simulation regressions with disabled deadlines, not the new 2D solver.

Runtime sanity is informational only: a default evaluation with warm mesh
cache took about 0.015 s; 1000 cached evaluations including fixture creation
took about 0.038 s. No CI timing assertion is imposed.

Reproduce the focused numerical checks from the repository root:

```powershell
py -3.11 -m pytest core/tests/thin_plate_fin_test.py core/tests/rectangular_finned_geometry_test.py core/tests/rectangular_finned_network_test.py -q
```

See [model definitions and primary free sources](../rectangular_finned_ellipse.md).

## Changed files

| Purpose | Files |
|---|---|
| Geometry, public exports and surface discrimination | `core/geometry/rectangular_finned_tube.py`, `core/geometry/__init__.py`, `core/geometry/tube.py` |
| Applicability boundary | `core/geometry/applicability.py` |
| Fin solver and public exports | `core/heat_transfer/thin_plate_fin.py`, `core/heat_transfer/__init__.py` |
| Shared thermal network and outside dispatch | `core/heat_transfer/outside_dispatch.py` |
| Normal workflow entry guards | `core/models/bare_tube.py`, `core/models/rating.py`, `core/models/simulation.py` |
| New tests | `core/tests/rectangular_finned_geometry_test.py`, `core/tests/thin_plate_fin_test.py`, `core/tests/rectangular_finned_network_test.py` |
| Version | `core/__init__.py` (setuptools reads this version dynamically) |
| Public documentation | `README.md`, `CHANGELOG.md`, `docs/references.md`, `docs/roadmap.md`, `docs/rectangular_finned_ellipse.md`, `docs/validation/finned_foundation_v092.md` |

The common resistance network required only a new surface-specific area/fin
branch and an optional numerical configuration argument. No constitutive
provider framework was imported. Smooth circular/elliptical and existing
circular-finned formulas and regression baselines were not edited.
