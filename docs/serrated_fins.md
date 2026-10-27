# Serrated helical fins: v0.9.5 Phase A

**Phase A is a geometry/API foundation.** It supplies volume, fin mass and
external geometric area, with engineering-derivation provenance. It does not
enable physical serrated Rating/Simulation.

## Supported geometry and API

`core.geometry.SerratedCircularFinnedTube` composes a `BareTube` with a
constant-thickness, untwisted, high-frequency welded I-foot fin. The helical
winding is idealized as periodic annuli with radial-sector cuts. The model
is dry geometry; U-foot, L-foot, tapered serrated fins, twisted/deformed
segments and SHIFTED banks are unsupported.

```python
from core.geometry import BareTube, SerratedCircularFinnedTube

tube = SerratedCircularFinnedTube(
    core_tube=BareTube(
        D_i=0.020, D_o=0.0254, length_total=1.0,
        length_effective=1.0, wall_k=45.0,
    ),
    fin_k=45.0, D_root=0.0254, D_fin=0.0534,
    fin_thickness_root=0.0006, fin_pitch=0.0032,
    serration_depth=0.007, kerf_width_at_tip=0.00030,
    segments_per_turn=40, fin_density_material=7850.0,
    segment_tip_twist_angle=0.0, base_style="welded_i_foot",
)

volume = tube.fin_volume_per_length       # m3/m, fin wings
mass = tube.fin_mass_per_length           # kg/m, fin wings
fin_area = tube.fin_area_per_length       # m2/m
primary_area = tube.primary_area_per_length
external_area = tube.external_area_per_length
```

All core inputs use SI: dimensions/arc widths in m, conductivity in W/(m K),
material density in kg/m3, twist in radians. There are no implicit unit
conversions. `fin_thickness_tip=None` means constant thickness; an explicit
equal tip thickness is also accepted. Unequal taper is rejected.
`fin_density` retains the existing meaning of turns/m; it is distinct from
`fin_density_material`.

`segments_per_turn` is a required positive integer, equal to `slits_per_turn`.
There is no rounding or guessing of this count. Segment and kerf widths are
arc lengths at the outer radius, with periodic closure:

```text
segment_width_at_tip = pi*D_fin / segments_per_turn - kerf_width_at_tip
```

The segment width is derived when omitted. An explicit value must satisfy
closure within floating-point tolerance (relative 1e-9, absolute 1e-12 m).
When changing count, diameter or kerf with `dataclasses.replace`, pass
`segment_width_at_tip=None` to recompute the dependent width. A measured
chord width must be explicitly converted by the caller; it is not an arc
width. The geometry does not infer a slit opening from either width.

Core geometry, materials, roughness and orientation are delegated to the
existing `BareTube`. Existing fin/root validation and contact input semantics
(explicit resistance, contact efficiency, ideal default, in that order) are
reused through private composition with uncut annular geometry. No serrated
object inherits from `CircularFinnedTube`. A continuous root layer, if
`D_root > core_tube.D_o`, retains the existing geometric/contact convention;
its volume and mass are not part of the fin-wing mass above.

## Engineering geometry derivation

These equations are geometric integrations, not empirical correlations or
claims of manufacturer validation. Define:

```text
r0 = D_root/2       r1 = D_fin/2       H = r1-r0
t = fin_thickness_root               pf = fin_pitch
ds = serration_depth                 rs = r1-ds
n = slits_per_turn                   beta = kerf_width_at_tip/r1
```

Construction requires `0 <= ds <= H`, `n*beta < 2*pi`, positive finite
dimensions/material properties, `r1 > r0 >= core_tube.D_o/2 > 0`, and
`pf > t > 0`. The bundle retains the existing full fin-envelope clearance
checks for inline and staggered arrangements.

Integrating thickness over the annulus and the removed sectors gives:

```text
V_full_turn    = pi*t*(r1^2-r0^2)
V_removed      = n*beta*t*(r1^2-rs^2)/2
V_fin_per_turn = V_full_turn - V_removed
V_fin'         = V_fin_per_turn/pf
m_fin'         = fin_density_material*V_fin'
```

For area, both broad faces and the outer tip are counted. Removed faces
and tip are subtracted; two sides and a bottom are added per slit:

```text
A_full_turn = 2*pi*(r1^2-r0^2) + 2*pi*r1*t
delta_faces  = -n*beta*(r1^2-rs^2)
delta_tip    = -n*beta*r1*t
delta_sides  =  2*n*ds*t
delta_bottom =  n*beta*rs*t
A_cut_turn   = A_full_turn + delta_faces + delta_tip + delta_sides + delta_bottom
A_fin'      = A_cut_turn/pf
A_primary'  = pi*D_root*(pf-t)/pf
A_external' = A_fin' + A_primary'
```

`fin_area_deltas_per_turn` exposes the four corrections in the order above.
For `ds == 0`, all corrections are exactly zero: no cut exists. For
`kerf == 0` and positive depth, removed volume is exactly zero but two
limiting slit faces are counted. This is the declared idealized surface
convention; it does not establish that coincident or touching slit faces
are accessible to real flow. Under the audited bookkeeping, slit bottoms
remain in the fin-attributed area even at the full-depth endpoint `ds == H`;
the exposed-primary and contact-area conventions remain those of the uncut
root reference. No thermal/contact interpretation of that endpoint is enabled.

A zero-width slit may remove no material. Real winding may open gaps without
removing material. Opened gaps and removed kerf material are different;
twist, helix deformation and segment deformation are excluded. These are
reasons to retain the explicit idealization, rather than treating it as the
exact geometry of every commercial serrated fin.

The external area is derived by the object; it accepts no external-area
override. Geometric area can increase or decrease relative to an equivalent
solid fin. The model does not force either outcome.

## Audit example

For the inputs in the example above:

| Quantity per axial metre | Equivalent solid | Serrated |
| --- | ---: | ---: |
| Fin volume, m3/m | 0.0003249192202 | 0.0003112338269 |
| Fin mass, kg/m | 2.550615879 | 2.443185541 |
| Fin area, m2/m | 1.114519264 | 1.173311399 |
| Primary area, m2/m | 0.06483461839 | 0.06483461839 |
| Total external area, m2/m | 1.179353882 | 1.238146017 |

Mass changes by approximately -4.21%; total external area by +4.99%.
The primary-area convention matches the existing solid-fin implementation;
no formula was adjusted to match rounded audit values. The reference and
geometric invariants are executable in
`core/tests/serrated_finned_tube_geometry_test.py`.

## Physical execution and future providers

`TubeSurfaceType.SERRATED_HELICAL_FIN` routes separately. All automatic
outside-bank entry points, direct solid-fin HTC/pressure adapters, annular
fin-efficiency functions, physical UA networks and Rating/Simulation reject
serrated geometry with `SerratedFinPhysicsNotSupportedError`. This also
applies to zero-depth serrated objects and supplied-film attempts. Users who
intend a solid fin must explicitly construct a solid-fin geometry.

Phase A implements **no serrated A_min, fin efficiency, HTC or pressure-drop
model**. Actual 3D flow through serration gaps, source-specific area bases
and serrated conduction remain unresolved. Neither a mass ratio nor an area
ratio supplies a hydraulic free-area law. Annular Bessel efficiency,
Briggs-Young, Robinson-Briggs, Frass/ESCOA solid-fin and Genic solid-fin
providers are never substituted. There is no constant serration multiplier,
RB correction or Frass correction.

The typed future boundaries are:

- `core.geometry.SerratedFlowAreaProvider`, with explicit provider/source
  provenance and one of `SerratedFlowAreaBasis`'s three distinct bases:
  orthographic geometric blockage, source-equivalent flow area or hydraulic
  provider-specific A_min. No implementation is registered. Orthographic
  blocked area is not interchangeable with either open-area definition.
- `core.heat_transfer.serrated_fin_efficiency.SerratedFinEfficiencyProvider`,
  for future fin-area-based conduction efficiency with provider/source
  provenance, before the shared contact network is applied.
- Existing outside thermal/hydraulic request/result and Rating/Simulation
  contracts, extended later with source-closed serrated providers through
  the distinct surface route. No such execution is enabled in Phase A.

Geometry provenance is directly inspectable:
`geometry_model=IDEALIZED_RADIAL_SECTOR_CUT`,
`geometry_provenance=ENGINEERING_DERIVATION`,
`physical_product_exactness=NOT_CLAIMED`,
`validated_against_manufacturer_CAD=False`. Existing `ModelWarning`
diagnostics include `SERRATED_IDEALIZED_RADIAL_SECTOR_CUT`. Public tests
verify guards and unchanged solid routes; they establish implementation
consistency, not empirical validation.

See [the Phase A validation record](validation/serrated_geometry_v095.md)
for the complete public-suite result and exact branch base.
