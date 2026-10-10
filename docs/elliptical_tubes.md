# Smooth elliptical tubes — v0.9.0

KalKalori 0.9.0 exposes geometry and standalone transport calculations for a
true mathematical ellipse. It supports exact fully developed smooth laminar
Darcy friction, fully developed H1 internal heat transfer and explicitly
selected confocal wall resistance. Automatic elliptical-bank Rating and
Simulation remain unsupported because outside-bank HTC and hydraulics have
not been established.

## Geometry and public constructor

```python
from core.geometry import EllipticalTube

EllipticalTube(
    outer_semi_major_axis, outer_semi_minor_axis,
    inner_semi_major_axis, inner_semi_minor_axis,
    length_total, length_effective, wall_k,
    orientation_angle=0.0,
    roughness_inner=None, roughness_outer=None,
    tube_orientation=None,
)
```

All axes are **semi-axes in metres**, lengths are metres, conductivity is
W/(m K), roughness is metres, and `orientation_angle` is radians. The geometry
is frozen and is a sibling of `BareTube`. Both contours share a centre and
major-axis orientation. For each contour `a >= b > 0`, and the outer axes must
strictly exceed their corresponding inner axes. Lengths and conductivity must
be finite and positive; active `length_effective` cannot exceed hydraulic
`length_total`. Roughness must be finite and nonnegative.

Inner and outer contours are specified independently. Subtracting the same
amount from both semi-axes does not define a constant-normal-thickness wall.
There is no inferred thickness, nor a `D_i`/`D_o` equivalent-diameter alias.

With `m = 1 - (b/a)**2` and `E = scipy.special.ellipe(m)`, each exact perimeter
is `P = 4*a*E`. SciPy takes the **parameter m**, whereas some references write
the elliptic integral in terms of modulus `sqrt(m)`.

| Quantity | Definition |
|---|---|
| `inside_aspect_ratio`, `outside_aspect_ratio` | `b_i/a_i`, `b_o/a_o` |
| `flow_area` | `pi*a_i*b_i` |
| `inside_wetted_perimeter`, `outside_perimeter` | Exact `P_i`, `P_o` |
| `hydraulic_diameter` | `4*flow_area/P_i` |
| `area_inner`, `area_outer` | `P_i*length_effective`, `P_o*length_effective` |
| `projected_blocking_area_per_length` | `2*sqrt(a_o**2*sin(theta)**2 + b_o**2*cos(theta)**2)` |
| `streamwise_extent` | `2*sqrt(a_o**2*cos(theta)**2 + b_o**2*sin(theta)**2)` |

At `theta=0`, the major axis lies along outside flow: projected width is
`2*b_o`, and streamwise extent is `2*a_o`. `tube_orientation` retains the
existing axial/gravity meaning and is independent of this rotation.
Projection is a geometric width; it does not establish a staggered-bank
throat or validate circular minimum-gap/reference-velocity equations.

## Explicit internal assumptions and example

The analytical solutions require a smooth straight duct, laminar fully
developed flow, a no-slip Newtonian fluid and constant properties. The caller
declares these conditions through `EllipticalFlowAssumptions`. The API does not
infer a universal elliptical transition Reynolds number or certify entrance
length from the declaration. Positive inner roughness, developing flow and
transitional/turbulent selections fail with an applicability error.

```python
import math
import core
from core.geometry import EllipticalTube
from core.properties.common import FluidTransportProperties
from core.heat_transfer import (
    EllipticalFlowAssumptions, TubeWallModel,
    elliptical_laminar_friction, elliptical_h1_heat_transfer,
    elliptical_distributed_pressure_drop, tube_wall_resistance_diagnostics,
)

# This particular outer contour is confocal with the inner contour.
ai, bi, ao = 0.010, 0.005, 0.012
bo = math.sqrt(ao**2 - (ai**2 - bi**2))
tube = EllipticalTube(ao, bo, ai, bi, 2.0, 1.8, 20.0)
props = FluidTransportProperties(rho=1000.0, mu=0.001, k=0.6, cp=4200.0)
assumptions = EllipticalFlowAssumptions(
    regime="laminar", development="fully_developed",
    thermal_boundary_condition="H1",
)

htc = elliptical_h1_heat_transfer(tube, m_dot=0.005, props=props,
                                   assumptions=assumptions)
friction = elliptical_laminar_friction(tube, htc.Re, assumptions=assumptions)
distributed = elliptical_distributed_pressure_drop(
    tube, m_dot=0.005, props=props, assumptions=assumptions,
)
wall = tube_wall_resistance_diagnostics(
    tube, n_tubes=1, model=TubeWallModel.CONFOCAL_EXACT,
)
assert core.__version__ == "0.9.0"
assert htc.model_metadata.thermal_boundary_condition == "H1"
assert distributed.local_losses_included is False
```

Mass flow is positive kg/s. By default, transport functions use one tube's
unblocked flow area; optional `flow_area` supplies an explicitly declared total
parallel flow area. Reynolds number uses true internal `Dh` and mean velocity,
`Re = rho*v*Dh/mu`. The H1 result retains the existing diagnostic fields
(`Re`, `Pr`, `Nu_corrected`, `alfa_corrected`) and adds model metadata. It applies
no circular entrance-length or wall-property correction.

## Friction and distributed pressure drop

For internal `beta=b_i/a_i` and `E=ellipe(1-beta**2)`:

```text
Po = f_D*Re_Dh = 8*(1+beta**2)*(pi/E)**2
f_D = Po/Re_Dh
dp/dz = f_D*rho*v**2/(2*Dh)
```

The returned factor is **Darcy**; `f_fanning` is `f_darcy/4`.
`elliptical_fd_laminar_darcy_exact` identifies the model, with provenance ID
`analytical_elliptical_poiseuille_v1`. Substituting the exact parabolic velocity
into the momentum equation independently gives
`dp/dz = 4*mu*v*(a_i**2+b_i**2)/(a_i**2*b_i**2)`.

`elliptical_distributed_pressure_drop` returns only fully developed distributed
friction, in Pa, over `length_total` or an explicit positive `length`. It
excludes entrances, exits, bends, manifolds and acceleration. The automatic
tube-bundle total-loss path rejects ellipse geometry because its circular
tube-sheet coefficients are not cleared for this shape. Generic local-loss
arithmetic remains subject to its existing user-supplied approximation basis.

## H1 thermal boundary condition

```text
Nu_H1 = (3*pi/E)**2 * (1+beta**2)*(1+6*beta**2+beta**4)
        / (17+98*beta**2+17*beta**4)
alpha_i = Nu_H1*k_fluid/Dh
```

`elliptical_fd_laminar_H1_exact` identifies the model, with provenance ID
`analytical_elliptical_energy_H1_v1`. It solves the fully developed energy
problem with negligible axial conduction and viscous dissipation.

**H1** means constant heat input per unit axial length and a wall temperature
uniform around each cross-section; local circumferential flux may vary.
**H2** instead prescribes uniform local wall heat flux around the perimeter.
**T** prescribes constant wall temperature. These are distinct conditions in
non-round ducts, and v0.9.0 accepts only the explicit string `"H1"` for HTC.

At `beta=1`, the H1 mathematics gives `Nu=48/11`. This does not change
`BareTube`'s existing circular constant-wall-temperature laminar default of
3.66. An `EllipticalTube` with equal axes retains elliptical applicability
rules; use `BareTube` for existing circular workflows.

## Confocal wall conduction

Explicitly select `TubeWallModel.CONFOCAL_EXACT` for ellipse wall calculations.
The model requires:

```text
a_o**2 - b_o**2 = a_i**2 - b_i**2
R_wall = log((a_o+b_o)/(a_i+b_i)) / (2*pi*k_wall*L_effective*N)
```

`N` is a positive integer tube count. The equality check scales both focal
squares by `a_o**2` and accepts an absolute normalized difference up to
`1e-12`. This is floating-point tolerance, not a manufacturing allowance.
The implementation evaluates the logarithm with `log1p` for narrow gaps.

The exact solution assumes steady conduction, constant conductivity,
isothermal inner/outer contour surfaces and negligible end effects. Concentric
contours need not be confocal. Non-confocal walls fail explicitly; there is no
thin-wall approximation because normal thickness has not been supplied.
`WallResistanceResult` records model ID `elliptical_confocal_wall_exact`,
references, assumptions and `approximation=False`. The equal-axis limit
recovers cylindrical resistance while the existing circular expression stays
unchanged.

This wall boundary supplies an absolute K/W resistance to the existing thermal
network. Area-specific fouling uses the actual ellipse areas,
`Rf_i=Rf_i''/Ai` and `Rf_o=Rf_o''/Ao`. Prescribed resistance-network arithmetic
can use these quantities; it does not provide missing outside-bank physics.

## Dimensions, dispatch and unsupported paths

`TubeGeometry` formalizes existing shared physical quantities.
`cross_section_family` identifies circular or elliptical sections separately
from `surface_type` (plain versus circular-finned). Geometry-aware internal
HTC/friction and wall boundaries select constitutive physics before equations
are evaluated. The new metadata records source access, geometry, regime,
development, boundary condition, reference dimension/velocity, area basis,
assumptions and approximation status.
Friction metadata records actual unblocked inside flow area; H1 metadata
records actual inside heat-transfer area.

| Dimension | Meaning |
|---|---|
| `Dh` | Internal `4*A/P`, used by these friction and H1 models |
| `2*a`, `2*b` | Full major/minor axes |
| `P/pi` | Perimeter-based dimension |
| Projected width | Orientation-dependent frontal blockage |

These quantities are different and must follow the selected correlation's
definition. No single interchangeable equivalent diameter is exposed.

Ellipse geometry is rejected before current circular equations in:

- automatic exchanger solve, wall-state solve, Rating and Simulation;
- outside-bank HTC, Euler/pressure drop and circular diagonal-gap velocity;
- tube-side enhancement routing, including twisted tape, wire coil and
  clearance corrections;
- steam condensation, water boiling and carrier-gas inside condensation;
- wet-coil routing/adapters and wet radial-fin calculations;
- `CircularFinnedTube` and annular-fin calculations;
- automatic circular entrance/exit total hydraulics.

Developing/turbulent internal models, T/H2, arbitrary walls, fins, inserts,
phase change and wet elliptical operation remain unsupported. No isolated
ellipse correlation is implemented, and an isolated-cylinder model would
require a separate standalone boundary; it cannot supply bank Rating.

## Public provenance and verification

The perimeter follows [NIST DLMF 19.30.5](https://dlmf.nist.gov/19.30#E5).
The internal equations are independent analytical reconstructions from the
continuum momentum/energy formulation and parabolic elliptical velocity field;
the public governing formulation is available in
[Su et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6843173/).
The confocal resistance is independently derived from steady Laplace
conduction using the coordinate mapping available in
[Khan, Culham and Yovanovich](https://www.mhtlab.uwaterloo.ca/pdf_papers/mhtl04-5.pdf).
These citations do not assert that each paper publishes every implemented
closed form. No private handbook or VDI text supplies public provenance.

Tests compare perimeter with numerical contour quadrature, pressure gradient
with the momentum equation, H1 with an independently solved polynomial energy
problem and velocity-weighted bulk integration, and wall resistance with
elliptical coordinates. They also check source-audit rounded values, scaling,
circle limits, actual fouling areas and explicit rejection of excluded paths.
These checks verify analytical implementation under the declared assumptions;
they do not establish general industrial ellipse-bank applicability.
