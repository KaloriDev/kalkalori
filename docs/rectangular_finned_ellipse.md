# Rectangular individual fins on elliptical tubes (v0.9.2)

This release provides geometry, analytical surface areas, deterministic 2D
fin conduction and a **Zhao-2017-based engineering air-side baseline** for
individual rectangular fins on elliptical tubes. Normal dry Rating/Simulation
uses physical film h, solves eta(h), and returns a finite pressure estimate.
The equations, inferred diameter and reconstructed passage normalization have
different evidence grades; none is silently promoted to an exact source
definition. This is not experimental validation of every rectangular-finned
ellipse. Prescribed-film network calculations remain available separately.

## Geometry and area definitions

`RectangularFinnedEllipticalTube(core_tube, fin)` composes the existing frozen
`EllipticalTube` with a frozen `RectangularIndividualFin`. Fin inputs are
`outer_width`, `outer_height`, `thickness`, `axial_pitch` [m] and
`thermal_conductivity` [W/(m K)]. Width is transverse (perpendicular to air
flow), height is streamwise (parallel at zero tube angle): the source physical
55x26 mm fin uses `outer_width=.026`, `outer_height=.055`. This is an intentional
pre-release semantic correction; migrate old positional and serialized values
by swapping them. Internal source axes use Wx=height and Wy=width. Pt=.02667
and Pl=.060 give .00067 transverse and .005 streamwise clearance.
See [API migration and CASE options](elliptical_api_v092.md).
All must be finite and positive; pitch must
exceed thickness. The rectangle must strictly enclose the tube's outer
ellipse with a positive ligament on both principal axes. Tangent, zero-ligament
domains are excluded. The rectangle follows the tube's crossflow rotation;
the hole is concentric and aligned with the rectangle.

Optional `fin_turbulator_count: int | None = None` describes the number of
disturbance features **per individual fin**. Omitted/None normalizes to 0 via
`effective_fin_turbulator_count`; zero means no explicit turbulators. Any
nonnegative integer is accepted, including 4, 8 and 10; booleans, floats,
strings and negative counts are rejected. Five-argument construction and
serialized definitions without the count field remain supported; width/height
values must follow the current engineering convention. Frozen dataclass
equality/hash includes the raw count; None and 0 have identical physics.

The count alone changes neither ideal-rectangle area/conduction nor the Zhao
correlations. Positive counts calculate with the same baseline and an
informational `fin_turbulators_present_uncorrected` diagnostic unless an
explicit provider changes h or pressure. Omission/None/0 adds no diagnostic.
No universal or source-validated count-to-HTC/pressure law is claimed for 4,
10 or other counts. Future external providers can inspect the fin metadata;
empirical factors remain model configuration, separate from geometry.

Let W, H, t, p denote the fin dimensions, a and b the outer ellipse semi-axes,
P its existing exact elliptic-integral perimeter and L the active tube length.
The periodic 0D convention uses the fractional fin count n = L/p. It excludes
end placement and end faces, consistently with `CircularFinnedTube`.

| Quantity | Definition for one active tube | API |
|---|---|---|
| Hole, one planar section | pi a b | `aperture_area_per_fin` |
| One broad metal face per fin | WH - pi a b | `fin_face_area_one_side` |
| Both broad faces, all fins | 2n(WH - pi a b) | `area_fin_faces` |
| Geometric outer edges | 2n(W+H)t | `area_fin_outer_edge` |
| Buried contact footprint | nPt | `area_contact` |
| Exposed base tube | PL(1-t/p) | `area_primary_outside` |
| Physical outside surface | exposed tube + both faces + edges | `area_outer`, `A_physical` |

The buried contact is counted once and is not exposed surface. An aperture is
missing metal; its planar area is not contact area. `area_fin` includes broad
faces and geometric outer edges. `TubeBundle` multiplies tube areas by the
actual physical tube count; tube-side pass count does not multiply surface.
Copying uses the existing frozen-dataclass composition. The core has no
application serialization layer; dataclass inspection/copying remains usable.

`projected_blocking_area_per_length` is periodic orthographic projected
blockage. It does not define A_min, a hydraulic equivalent diameter or an
applicable bank correlation. Existing circular-bank minimum-flow helpers
continue to reject this geometry.

## ThinPlateFinEfficiency2D

For uniform film coefficient h, conductivity k and thickness t, the model uses

```
theta = (T - T_inf)/(T_base - T_inf)
laplacian(theta) - m² theta = 0
m² = 2h/(kt)
theta = 1                on the elliptical tube contact
d(theta)/dn = 0          on the rectangular outer edge
eta_fin = integral(theta dA)/integral(dA)
```

The two broad faces convect. The model assumes ideal isothermal contact,
constant properties, uniform h and temperature through the thickness. It
excludes edge convection, contact resistance, radiation, variable k,
nonuniform h and 3D conduction. Formal h -> 0, k -> infinity and t -> infinity
limits give eta -> 1; the latter is a mathematical limit, not a claim that a
physically thick fin satisfies the thin-plate assumptions.

The independently written P1 finite-element scheme uses existing NumPy/SciPy.
It solves a first quadrant with natural symmetry boundaries. Two angular
sectors meet at the rectangle-corner ray; radial rings connect exact ellipse
nodes to the exact rectangle. Four triangles around each cell center preserve
reflection symmetry. Elliptical contact chords converge with refinement;
`mesh_area_relative_error` measures their difference from analytical metal
area. There is no staircase aperture or general CAD mesher.

Reaction mass is lumped. Solving the temperature deficit 1-theta stabilizes
small-h limits. Diagnostics expose eta, quadrant node/element counts,
resolution, free-node residual, discrete base/convective heat rates [W/K],
energy balance, quadrant coordinates and temperatures. Base heat is the
algebraic boundary reaction including boundary-node convection weights.
Heat diagnostics integrate the polygonal domain mirrored four times. The
network applies its computed mean eta to the **analytical** broad-face area;
the separately measured mesh-area error quantifies this discretization.

Defaults start at 16 radial cells and 16 angular cells per sector, doubling
both until successive eta differences and relative area error are <= 5e-4,
with at most three refinements. The eta difference is an observed convergence
estimate, not a guaranteed error bound. Failure raises
`FinEfficiencyConvergenceError`. Setting `max_refinements=0` returns a single
mesh with `converged=False` and no convergence estimate, for refinement
studies. Physical temperature bounds are checked rather than silently
repairing a nonphysical field.

Mesh caches (32 entries) and result caches (128 entries) are bounded. Fin
dimensions, h, k, thickness and numerical configuration determine solutions;
tube length, pitch, count, rotation and rows do not change the conduction
problem. Identical evaluations reuse results without remeshing.

## Effective surface and the common resistance network

Under the adiabatic outer-edge condition:

```
A_effective = A_exposed_tube + eta_fin * A_fin_faces
G_outside = h_film * A_effective
R_outside_clean = 1/G_outside
```

The edge belongs to `A_physical` but not `A_effective`, even for eta=1.
Physical film h is retained unchanged. The existing network separately
reports its gross-area equivalent coefficient; that reporting coefficient
includes area/topology effects and is not the physical film coefficient.
Existing inside/outside fouling retains its inside/gross-area convention.

The common `calculate_resistance_network` evaluates h -> eta(h) -> area ->
resistance once, without a second efficiency multiplier or a fixed eta.
Its `A_physical` and `A_effective` aliases expose the distinction. Detailed
fin mesh diagnostics are in `fin_efficiency_result`. A prescribed-film network
calculation does not establish predictive exchanger applicability.

```python
from core.geometry import (
    EllipticalTube, RectangularIndividualFin,
    RectangularFinnedEllipticalTube, TubeBundle,
)
from core.heat_transfer import ThinPlateFinEfficiency2D
from core.heat_transfer.outside_dispatch import calculate_resistance_network

core = EllipticalTube(.012, .006, .010, .004, 2., 1.8, 40.)
fin = RectangularIndividualFin(.030, .040, .0004, .003, 200.)
tube = RectangularFinnedEllipticalTube(core, fin)
bundle = TubeBundle(tube, 4, 6, .060, .060, "inline", 2)
result = ThinPlateFinEfficiency2D().evaluate(tube, h=80.)
network = calculate_resistance_network(
    bundle=bundle, alpha_inside=1200., outside_alpha_physical=80.,
    resistance_core_wall=.001,  # explicitly supplied resistance [K/W]
)
assert network.outside_alpha_physical == 80.
assert network.A_effective < network.A_physical
```

This is generic illustrative geometry, not a product preset. The specified
inside HTC, outside HTC and wall resistance above are illustrative inputs,
not predictions. Existing standalone elliptical internal/wall models may be
evaluated on `tube.core_tube` under their own explicit applicability rules.
Arbitrary nested ellipses are not assumed confocal.

## Primary sources and public scope

- Taler, D.; Marcinkowski, M. (2020), *Calculating the Efficiency of Complex
  Shaped Fins*, Chemical Engineering Transactions 81, 1333-1338,
  [DOI 10.3303/CET2081223](https://doi.org/10.3303/CET2081223),
  [free publisher PDF](https://www.aidic.it/cet/20/81/223.pdf).
- Marcinkowski, M.; Taler, D. (2021), *Calculating the Efficiency of
  Complex-Shaped Fins*, Energies 14(3), 577,
  [DOI/publisher article](https://www.mdpi.com/1996-1073/14/3/577), CC BY 4.0.

Both were accessible without payment on 2026-10-19. They support the governing
conduction problem, its assumptions and efficiency definition. KalKalori's
mesh, assembly, geometry and tests are independent GPL-3.0-only code; no
article text, figures, external source code or full articles are redistributed.
No paid/private coefficients, calibration data or experimental provider code
were used. Numerical evidence is recorded in
[the validation note](validation/finned_foundation_v092.md).

The [2026-10-19 free-source decision](validation/finned_ellipse_source_closure_v092.md)
remains a historical record. On 2026-10-20 the project explicitly authorized
implementation using supplied Zhao/Tongji equation transcriptions and visible
engineering reconstructions. Free access to the full Zhao paper is **not**
asserted. No article images, full papers or external code are redistributed.
The geometry/conduction sources above and their independent implementation
remain unchanged. See [implementation validation](validation/zhao2017_v092.md).

## Zhao 2017: equations and evidence grades

The [2026-10-27 lattice audit](validation/zhao_layout_audit_2026-10-27.md)
distinguishes conventional whole-row STAGGERED from per-site SHIFTED. Zhao
rejects SHIFTED (including unresolved phases) and INLINE in both domain modes.
Recovered Zhao2017 Fig. 2 and Zhao2016 Fig. 1 resolve the source identities:
Pt is transverse, Pl follows flow, and there is no Pt/Pl swap. The audit gives
the ROW/COLUMN coordinates and the legacy coordinate-helper indexing caveat.
YC SHIFTED uses the user-confirmed Zhao COLUMN design basis; the named COLUMN
preset retains SHIFTED identity and does not enable Zhao2017 coefficients.

Zhao, Gu, Gao, Yang (2017), *Numerical study on airside thermal-hydraulic
performance of rectangular finned elliptical tube heat exchanger with large
row number in turbulent flow regime*, IJHMT 114, 1314–1330,
[DOI 10.1016/j.ijheatmasstransfer.2017.06.049](https://doi.org/10.1016/j.ijheatmasstransfer.2017.06.049).
`Zhao2017AirsideModel` uses the supplied, visually verified transcriptions:

```text
Xt=(Pt-Dt)/Dt; Xl=(Pl-Dl)/Dl; Xp=Fp/de; Xf=Ft/de
j_eff = 1.0062*N**(-0.2045)*Xt**(-0.7268)*Xl**0.1319
        *Xp**0.1381*Xf**0.1495*Re**(-0.4408)       # Eq.18
f = 1.3549*N**(-0.0591)*Xt**(-0.9114)*Xl**0.4709
    *Xp**0.1296*Xf**0.0438*Re**(-0.3164)          # Eq.19
j_prime = 0.2*N**(-0.2160)*Xt**(-0.82)*Xl**0.1361
          *Xp**0.0043*Xf**(-0.0578)*Re**(-0.3326) # Eq.20
h_film = j_prime*rho*cp*u_m/Pr**(2/3)
```

Eq.20 excludes fin thermal resistance. Eq.18 includes it and is **diagnostic
only**. The source Colburn definition is inverted to obtain physical film h;
only the common resistance network applies eta and effective area.

The source nominal geometry is a 36×14 mm ellipse with a 55×26 mm fin envelope,
small holes, positioning tabs and clipped corners. Source tubes/fins were
carbon steel; Eq.18 diagnostics retain that source-material context. The public
geometry is a full ideal rectangle. Every calculation carries
`source_geometry_simplified_rectangular_fin`; positive fin conductivity is a
user input, never a carbon-steel whitelist. The canonical major axis and fin
height are parallel to air flow. Other cross-section orientations reject.
Fin envelopes must not overlap and tube gaps must be positive.

| Parameter | Source fit interval |
|---|---|
| Xt | 0.93–1.50 |
| Xl | 0.56–0.78 |
| Xp | 0.10–0.52 |
| Xf | 0.01–0.03 |
| Re_de | 5500–23800 |
| N_rows | 6–10 |

The supplied regression campaign used frontal CFD velocities 3.5–12.5 m/s.
The callable source-domain checks use the printed normalized ratios and Re/N
intervals above; this campaign description is not an independently validated
extension to other fluids, materials or property temperatures.

The default model calculates outside these intervals with explicit
`zhao2017_extrapolated_*` warnings and no clipping. N=2/4 is therefore an
**extrapolated baseline**, not a newly supported experimental row law.
`Zhao2017AirsideModel(strict_source_domain=True)` rejects out-of-domain ratios,
Re/N and nominal geometry/layout extrapolation. The simplified-fin caveat
remains even for nominal dimensions. No extra N multiplier is applied: N
already enters Eqs.18–20 and actual physical bundle area. Tube-side pass count
does not enter any air-side correlation or physical surface scaling.

### Source-inferred diameter and reconstructed velocity

The package identifies `de=4*Se/P_ellipse` as **source-derived/inferred**, not a
printed Zhao formula. `Se` is the actual elliptical aperture area; the existing
accurate outer perimeter is reused. For 36×14 mm, de=0.019218003478794744 m.

The average passage is an **engineering reconstruction** on one periodic cell:

```text
Sg = Pt*Fp
Savg = Sg - ((Fp-Ft)*Se + Ft*Wx*Wy)/Pl
sigma_avg = Savg/Sg
Aa = bundle.frontal_flow_area*sigma_avg
u_frontal = m_dot/(rho*bundle.frontal_flow_area)
u_m = m_dot/(rho*Aa)
Re_de = rho*u_m*de/mu; Pr = cp*mu/k
```

Wx=fin.outer_height and Wy=fin.outer_width are streamwise/transverse extents. Neither Aa nor de is replaced by
a circular-bank throat or a nominal circular diameter. The separate sensitivity
estimate `sigma_throat=(Pt*Fp-(Fp-Ft)*Dt-Ft*Wy)/(Pt*Fp)` and
`u_max_est=u_frontal/sigma_throat` do not drive the correlations.
Nonfinite/nonpositive passage area, properties or mass flow reject.

Zhao's CFD used constant air properties at an average flow-field temperature.
The normal 0D engine's mean bulk state approximates that convention. Each
thermal iteration recomputes h and eta from its current state; cached numerical
fin meshes/results remain in use. Hydraulic inlet/midpoint/outlet diagnostics
retain their respective properties, while the source pressure estimate uses
the arithmetic-temperature midpoint, without a new axial integration law.

### Pressure estimate and the existing result contract

The source definition `f=2*DeltaP*Aa/(rho*u_m**2*A)` gives:

```text
DeltaP_estimate = f*(rho*u_m**2/2)*(A_physical/Aa)
```

Physical exposed area, including geometric edges, is used, never eta-reduced
area. This is `source_domain_inlet_to_outlet_estimate`, with
`validated_bank_irreversible=False`. No separate acceleration, entrance/exit,
duct or extra row contribution is added. The input package reports an unresolved
normalization signal of roughly 735 Pa versus 404.49 Pa for one nine-row source
case; this is not silently fitted away or presented as a validated fixture.

Normal `outside_dp_total` returns the finite estimate.
`outside_tube_bank_hydraulic.midpoint.airside_result` exposes its raw/corrected
values and provenance. The legacy `dp_drag` slot aliases this estimate; it must
not be read as a separately validated form/irreversible drag.
`outside_side_pressure_drop.flow_path` is a
`SourceDomainPressureEstimateResult`, not an irreversible-loss stage path.
Its compatibility totals carry the estimate and `components_resolved=False`;
zero additional dynamic/local terms mean none was added, not a measured physical
decomposition. Combining this estimate with explicit irreversible local-loss
assemblies is rejected. Ordinary circular pressure-path semantics are unchanged.

## Normal dry Simulation and Rating

Automatic bare-ellipse bank models remain unavailable. The finned source
baseline automatically selects analytical laminar H1 or Yang turbulent T from
actual inner-contour Re; transition remains explicitly unsupported. Wall
selection is automatic for general and confocal contours. The
[turbulent T and numerical wall guide](elliptical_turbulent.md) describes source
extrapolation, strict policy and engineering friction. No circular
HTC/entrance/exit correlation is substituted. Tube-side pressure reports distributed friction and
bulk acceleration with a warning that tube-sheet losses were not evaluated.
Other contour families, internal developing/transition flow, inserts, phase change
and wet rectangular fins require separate models and remain unsupported.

```python
import math
from core.geometry import (
    EllipticalTube, RectangularIndividualFin, RectangularFinnedEllipticalTube,
    TubeBundle,
)
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.models.heat_balance import BalanceSideSpec
from core.properties import ConstantPropertyProvider
from core.properties.common import FluidTransportProperties

ai = .0175
bi = math.sqrt(ai**2 - (.018**2 - .007**2))  # explicitly confocal inner contour
core = EllipticalTube(.018, .007, ai, bi, 1., 1., 45., roughness_inner=0.)
tube = RectangularFinnedEllipticalTube(
    core, RectangularIndividualFin(.026, .055, .0003, .004, 45.))
bundle = TubeBundle(tube, 8, 6, .031, .060, 'staggered', 2,
                    flow_arrangement='counterflow')  # explicit 0D thermal topology
hx = BareTubeHeatExchanger(bundle)  # automatic internal/wall/airside selection
liquid = ConstantPropertyProvider(FluidTransportProperties(
    rho=1000., mu=.02, k=.6, cp=4200.))
air = ConstantPropertyProvider(FluidTransportProperties(
    rho=1.2, mu=1.85e-5, k=.026, cp=1006.))
inside = HXSideInput(provider=liquid, m_dot=1., T_in=360., p=300000.)
outside = HXSideInput(provider=air, m_dot=1.7, T_in=300., p=101325.)
simulation = hx.simulate(inside, outside)
rating = hx.rate(
    BalanceSideSpec(provider=liquid, m_dot=1., T_in=360.,
                    T_out=simulation.T_out_inside, p=300000.),
    BalanceSideSpec(provider=air, m_dot=1.7, T_in=300.,
                    T_out=simulation.T_out_outside, p=101325.),
    include_simulation=False,
)
d = simulation.finned_tube_diagnostics
print(simulation.q, simulation.UA, simulation.outside_dp_total)
print(d.outside_alpha_physical, d.eta_fin, d.A_outside_geometric, d.A_effective)
print(d.airside_result.flow, d.airside_result.extrapolated_parameters)
```

This is a synthetic numerical example with an explicit thermal-topology and
automatically recorded FD laminar H1 model assumption, not a reconstruction of a measured exchanger.
The geometry and operating point fit the source air-side ratios, with the
always-present simplified-fin and flow-normalization caveats.

## Optional neutral correction hook

Configure one dependency-injected object through `Zhao2017AirsideModel`:

```python
from core.heat_transfer import AirsideCorrection, Zhao2017AirsideModel

class ExampleCorrection:
    def correct(self, base_result, *, bundle, props):
        return AirsideCorrection(
            provenance='user_supplied_example', h_film_multiplier=1.05,
            pressure_drop_multiplier=1.10)

model = Zhao2017AirsideModel(correction_provider=ExampleCorrection())
# Advanced direct use: BareTubeHeatExchanger(bundle, airside_model=model).
# In notebook CASES: 'options': {'airside_model': model}.
```

Without a provider the public baseline works on its own. Each channel can have
one positive multiplier or replacement (`h_film_replacement`,
`pressure_drop_replacement`), never both. h is corrected **before** eta;
pressure is corrected on the final estimate. `h_film_base`, `h_film`,
`pressure_drop_base`, `pressure_drop` and `correction_provenance` remain visible.
Providers must be deterministic for a given state; exceptions/invalid responses
propagate without fallback. There is no runtime name discovery or global hook.
These example multipliers are API demonstrations, not calibration coefficients.

For constant independent factors, the generic immutable provider avoids a
custom adapter:

```python
from core.heat_transfer import MultiplicativeAirsideCorrection, Zhao2017AirsideModel

model = Zhao2017AirsideModel(correction_provider=MultiplicativeAirsideCorrection(
    h_film_multiplier=1.0, pressure_drop_multiplier=1.0,
    provenance='explicit_engineering_input'))
```

Both defaults are 1.0. Factors must be finite and positive; None, zero,
negative and boolean factors are rejected. They are independent engineering
inputs, never derived from count. Result diagnostics expose normalized
`fin_turbulator_count` and `correction_applied` (whether either base value
actually changed), alongside the existing base/corrected values/provenance.
Physical h is corrected before eta, and the pressure scope remains the
source-domain estimate. Contact-efficiency semantics are unchanged.
