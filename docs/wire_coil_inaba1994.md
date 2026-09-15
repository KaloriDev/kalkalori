# Inaba1994 wire-coil provider

`core.enhancements.Inaba1994WireCoilProvider` is the public wire-coil model
selected for v0.8.2 development. It uses one coherent primary-source family
from Inaba, Ozaki and Kanaoka (1994): distributed friction Eq. (7), high-Re
heat transfer Eq. (10), and low-Re heat transfer Eq. (11). Only `P/e > 10`
is implemented. The different source equations for `P/e <= 10` are an
unsupported branch, not ordinary extrapolation.

The legal implementation authority is the official free-access
[J-STAGE article](https://www.jstage.jst.go.jp/article/kikaib1979/60/569/60_569_240/_article/-char/en),
[DOI 10.1299/kikaib.60.240](https://doi.org/10.1299/kikaib.60.240),
pages 240--247.

## Geometry and source references

`WireCoilGeometry(wire_diameter=e, pitch=P)` describes an ideal circular wire
coil contacting the inner tube wall. The tube inside diameter `d_i` remains
part of the ordinary tube geometry. Callers do not supply derived source
quantities. The provider calculates

```text
q = P/e
L_cL/L = sqrt(P^2 + pi^2 (d_i - e)^2) / P
A_0 = pi d_i L
A_s = pi d_i L + pi e L_cL
D_h = 4 [(pi/4) d_i^2 L] / [pi d_i L + pi e L_cL]
```

The `D_h` numerator is the nominal tube volume. Wire volume is not subtracted
and no blockage correction is added. The source mean axial velocity uses the
nominal bore area `A_flow = pi d_i^2/4`. Invalid primitive geometry (`e <= 0`,
`P <= 0`, or `e >= d_i`) is rejected independently of correlation limits.

## Properties and equations

The article states that water properties used for experimental data reduction
were evaluated at the mean of local heating-surface and bulk-mixed water
temperatures. KalKalori therefore evaluates its existing property backend at

```text
T_film = (T_bulk + T_wall)/2
```

and uses that state for source kinematic viscosity and Prandtl number. It does
not average already evaluated properties. The source Reynolds number is
`Re_D = U D_h / nu_film` and is never reused from the bare-tube diameter.

For the implemented high-pitch family:

```text
C_fD = 11.5 Re_D^-0.39 q^-0.87                         Eq. (7)
Nu_D = 0.225 Re_D^0.800 Pr_film^(1/3) q^-0.48          Eq. (11), Re_D <= 2000
Nu_D = 0.803 Re_D^0.630 Pr_film^(1/3) q^-0.48          Eq. (10), Re_D > 2000
```

The two Nu equations differ at `Re_D=2000`. KalKalori deliberately selects
Eq. (11) at and below 2000 and Eq. (10) above 2000. It does not interpolate,
average, smooth, or move this boundary.

## Hydraulic and heat-transfer normalization

The source `C_fD` is Fanning: its smooth laminar reference is `16/Re` and its
distributed definition gives

```text
delta_p/L = 4 C_fD rho U^2 / (2 D_h)
```

KalKalori's canonical engine reference uses the same nominal-area velocity
and `d_i`, so the returned Darcy factor is

```text
f_Darcy,di = 4 C_fD (d_i/D_h)
```

This preserves the source pressure gradient exactly. Local entrance, exit,
support and attachment losses are outside Eq. (7); existing local-loss
handling remains separate and no insert `K` value is invented.

The source `Nu_D` uses `D_h` and its coefficient acts on `A_s`. KalKalori's
inside coefficient acts on tube-wall area `A_0`, so heat flow is preserved:

```text
h_source = Nu_D k_film/D_h
h_engine = h_source (A_s/A_0)
Nu_engine,di = h_engine d_i/k_film
```

The implementation keeps the general area/length conversion. For this exact
geometry `A_s/A_0 = d_i/D_h`, but wire fin efficiency, separate conduction,
contact resistance, and attachment losses are not separately modelled.

## Applicability and diagnostics

The numerical public domain is `400 <= Re_D <= 6000`, `10 < q <= 50.3`, and
`4.21 <= Pr_film <= 8.12`. Source experiments used water heating, uniform
wall temperature, a horizontal copper tube with `d_i=16 mm`, `d_o=20 mm`,
and wire diameters 2.0, 2.5, and 3.0 mm (`e/d_i` 0.125, 0.15625, 0.1875).
These points do not establish universal scale independence or validity for
cooling, other wall conditions, oils, glycols, non-Newtonian fluids, or other
media.

Default `extrapolation_policy="error"` is strict. Explicit `"warn"` mode can
evaluate mathematically valid out-of-range Re, Pr, diameter scaling, and
geometry-ratio extrapolation. Ratio interpolation between tested wires is
identified separately. `q <= 10` and nonphysical geometry always fail.
Source-context warnings are always retained without brittle fluid-name tests.

Diagnostics expose the primitive and derived geometry, nominal flow and
velocity, film reference, source Re/Pr, selected Nu branch, source Fanning
factor and `Nu_D`, source `h`, canonical Darcy factor, canonical Nu/HTC,
applicability, warnings, and provenance. Rating and Simulation select this
same provider. Their v0.x use is a 0D engineering application of an average
experimental relationship, not a local axial-development model and not
physical validation of a complete apparatus.

```python
from core.enhancements import (
    Inaba1994WireCoilProvider, TubeSideEnhancement, WireCoilGeometry,
)

wire_coil = TubeSideEnhancement(
    provider=Inaba1994WireCoilProvider(),
    geometry=WireCoilGeometry(wire_diameter=0.002, pitch=0.040),
    fluid_phase="liquid",
)
```
