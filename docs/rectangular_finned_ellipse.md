# Rectangular individual fins on elliptical tubes (v0.9.2)

This release provides a public physical foundation for a centered rectangular
transverse fin on a mathematical elliptical tube. It supports geometry,
analytical surface areas and deterministic 2D thin-plate fin efficiency.
It does not supply predictive outside HTC, bank pressure drop or N_rows
corrections for this surface. Normal Rating/Simulation rejects the geometry
with `TubeGeometryApplicabilityError` before selecting unrelated physics.

## Geometry and area definitions

`RectangularFinnedEllipticalTube(core_tube, fin)` composes the existing frozen
`EllipticalTube` with a frozen `RectangularIndividualFin`. Fin inputs are
`outer_width`, `outer_height`, `thickness`, `axial_pitch` [m] and
`thermal_conductivity` [W/(m K)]. All must be finite and positive; pitch must
exceed thickness. The rectangle must strictly enclose the tube's outer
ellipse with a positive ligament on both principal axes. Tangent, zero-ligament
domains are excluded. The rectangle follows the tube's crossflow rotation;
the hole is concentric and aligned with the rectangle.

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
fin = RectangularIndividualFin(.040, .030, .0004, .003, 200.)
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

Public predictive outside HTC, bank pressure loss and row effects remain
unsupported for this surface. No smooth-ellipse, circular-bank, equivalent-
diameter or arbitrary fin-factor substitute is allowed. Ribeiro prediction
is deferred pending source closure of flow-area, diameter and applicability
definitions. Contact resistance, convective edges, variable h/k, radiation,
3D and wet rectangular fins require separate documented model variants.

The [second-series targeted source closure](validation/finned_ellipse_source_closure_v092.md)
checked Tang--Du--Zeng (2019), Zhao--Yang (2016), Ribeiro--Andrade (2004),
the Ribeiro thesis record, and the audit's Chiu--Lin--Jang (2007) hydraulic
candidate on 2026-10-19. Tang's full free publisher PDF provides a physical-h
Nu equation and fixed N=2 scope, but its flow construction and mapping to
individual rectangular fins remain incomplete. No candidate closed a complete
predictive model for this geometry. The existing applicability refusals remain;
the requested full thermal-hydraulic v0.9.2 release is not complete. The note
records exact recovered equations, missing definitions and retrieval limits.
