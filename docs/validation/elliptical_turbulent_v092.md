# Explicit turbulent ellipse / nonconfocal wall validation

Checked 2026-10-20 with Python 3.11, on the existing v0.9.2 branch.
This is implementation verification under the documented assumptions,
not an experimental validation of exchanger performance. See
[equations, sources and limits](../elliptical_turbulent.md).

## Focused public tests

```text
py -3.11 -m pytest \
 core/tests/elliptical_turbulent_test.py \
 core/tests/elliptical_turbulent_integration_test.py \
 core/tests/elliptical_numerical_wall_test.py \
 core/tests/elliptical_internal_test.py \
 core/tests/elliptical_wall_test.py \
 core/tests/elliptical_applicability_test.py \
 core/tests/zhao2017_airside_test.py \
 core/tests/zhao2017_integration_test.py \
 core/tests/thin_plate_fin_test.py \
 core/tests/rectangular_finned_network_test.py \
 core/tests/circular_finned_tube_integration_test.py \
 core/tests/thermal_iteration_test.py \
 core/tests/rating_thermal_iteration_integration_test.py \
 core/tests/simulate_thermal_iteration_integration_test.py -q
```

Result: **211 passed in 11.22 s**. The continuation backslashes above are
for presentation; PowerShell accepts the command as one line.

Coverage includes source equation coefficients and unit-Pr reference values,
explicit assumptions, shape/Re/roughness guards, geometry-adjusted friction,
numerical wall convergence and analytical limits, normal public synthetic
Rating/Simulation energy/resistance consistency, and existing circular,
laminar H1, exact confocal, fin, Zhao and thermal iteration regressions.
The full public suite was not rerun for this bounded extension.

## Independent wall limits

One tube, active length 1.8 m, conductivity 40 W/(m K):

| Cross-section | Numerical R, K/W | Analytical R, K/W | Relative difference |
|---|---:|---:|---:|
| Circle: ri=0.010 m, ro=0.012 m | 0.000403012967802 | 0.000403019124944 | −1.528e−5 |
| Confocal: c=0.010 m, mu_i=1, mu_o=1.2 | 0.000442089790735 | 0.000442097064144 | −1.645e−5 |

Both accepted meshes have 32 radial × 128 angular cells per quarter,
area error 2.510e−5, balance error below 2.5e−15 and residual below 1.6e−15.
The analytical answers use log(ro/ri)/(2*pi*k*L) and
(mu_o-mu_i)/(2*pi*k*L), respectively. Tests additionally cover mu_i=0.5, 2,
nonconfocal sections, independent finer meshes and deliberate nonconvergence.

## Source interpretation

Yang Eq.18 and Table 4 are implemented without fitting project outputs or
inventing coefficient interpolation. The air table omits its Pr/transport
properties, so exact table reproduction is not asserted. Nikitin/Yakhot's
finest-grid DNS Fanning values are checked within 3% at the two source shapes
and Re=6000. The exact ellipse perimeter is retained; the paper's rounded
modified-Re numbers are not used to calibrate geometry. Outside those DNS
points the friction law explicitly warns of engineering extrapolation.

No source PDFs, figures, private data, private notebooks or external solver
implementation are included in the public validation fixtures.
