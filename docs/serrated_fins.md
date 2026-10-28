# Serrated helical fins: v0.9.5 Phases A and B

**STAGGERED: explicit source-bounded execution is available. AUTO: unavailable.
INLINE and SHIFTED: rejected at bundle construction.**

Phase A supplies volume, fin mass and external geometric area, with engineering
derivation provenance. Phase B adds separate Holfeld source-equivalent flow
area, Hashizume serrated efficiency, and Holfeld physical-film HTC/native-Eu
pressure loss. Only an explicit compatible complete profile opens execution.
The source-equivalent surface used for physics is distinct from the unchanged
Phase A geometric surface. Independent matched hydraulic validation remains
incomplete; see the [Phase B validation record](validation/serrated_physics_v095.md).

## Supported geometry and API

`core.geometry.SerratedCircularFinnedTube` composes a `BareTube` with a
constant-thickness, untwisted, high-frequency welded I-foot fin. The helical
winding is idealized as periodic annuli with radial-sector cuts. The model
is dry geometry; U-foot, L-foot, tapered serrated fins, twisted/deformed
segments, INLINE banks and SHIFTED banks are unsupported. A serrated
`TubeBundle` requires `CONVENTIONAL_STAGGERED`; the legacy `staggered` spelling
maps to that identity. An explicit `TubeBankLattice` is authoritative over the
legacy label, so changing physical layout with `dataclasses.replace` requires
changing or clearing the lattice as well.

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
`pf > t > 0`. The bundle retains full fin-envelope clearance checks for the
supported conventional staggered arrangement.

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

## Phase B: explicit source-bounded execution

`TubeSurfaceType.SERRATED_HELICAL_FIN` has a distinct route. Select the complete
profile through the existing `airside_model` argument:

```python
from core.geometry import BareTube, SerratedCircularFinnedTube, TubeBundle
from core.heat_transfer import Holfeld2016SerratedStaggeredModel
from core.models.bare_tube import BareTubeHeatExchanger

tube = SerratedCircularFinnedTube(
    core_tube=BareTube(.025, .03175, 2.0, 2.0, 45.0),
    fin_k=45.0, D_root=.03175, D_fin=.06775,
    fin_thickness_root=.001, fin_pitch=1/277,
    serration_depth=.011, kerf_width_at_tip=0.0,
    segments_per_turn=32, fin_density_material=7850.0,
)
bank = TubeBundle(tube, 8, 8, .06975, .0604052719139646,
                  "staggered", 1, flow_arrangement="counterflow")
model = Holfeld2016SerratedStaggeredModel(source_segment_width=.0045)
hx = BareTubeHeatExchanger(bank, airside_model=model, strict_source_domain=True)
# hx.rate(...), hx.simulate(...), hx.solve_thermal_state(...)
```

The profile itself is the explicit selector. The generic finned-provider
arguments retain their default sentinel values; supplying a solid/bare selector
alongside the profile raises an incompatibility error. No string selector or
provider is registered with AUTO. Omitting `airside_model`, using an incomplete
component set, or requesting a solid-fin adapter still raises
`SerratedFinPhysicsNotSupportedError`, including for zero serration depth.
An explicit solid tube object is required for solid-fin physics.

The individual implementations remain public and inspectable:

| Component | Class | Source |
| --- | --- | --- |
| Flow area | `Holfeld2016SerratedFlowAreaProvider` | Eq. 5-9 |
| Serrated efficiency | `HashizumeSerratedFinEfficiencyProvider` | 1997 two-region theory; 2002 STAGGERED correction |
| Physical film HTC | `Holfeld2016SerratedStaggeredHeatTransferProvider` | Eq. 6-6 |
| Bank-section loss | `Holfeld2016SerratedStaggeredPressureDropProvider` | Eqs. 6-10, 4-8 |

The low-level HTC and DP classes consume a `SerratedTubeBankRequest` and return
the normal `FinnedTubeHeatTransferResult` / `FinnedTubePressureDropResult`.
These requests are separate from solid-fin requests. The profile exposes each
component as a named field and rejects missing/incompatible fields.

### Geometry and area contracts

`source_segment_width` is the physical rectangular strip-segment width [m].
It is required independently of Phase A's derived outer-radius tip arc width.
Neither width is rounded into a segment count. The native strip interpretation
and Phase A radial-sector idealization are different reference models; no exact
manufacturer CAD correspondence is claimed between them.

Initially the physical profile requires `D_root` equal to bare core OD,
zero removed kerf, constant thickness, no twist, and ideal HF-welded I-foot
contact. Geometry with finite kerf or a continuous root layer still works for
Phase A quantities but has no source-closed Phase B mapping. Nonideal contact,
fouling, wet gas, condensation, and phase-capable property-provider routes are
rejected. Dry-air transport properties are the source calibration contract;
constant/custom point properties are a caller assertion of that contract.
Single-phase liquid tube-side transport can be supplied independently.

Holfeld `sf` is **fin pitch**, not clear spacing. `Nf=1/fin_pitch` is turns/m.
The source flow area is:

```text
A_min_source = length_effective * transverse_flow_cells
             * (Pt - do - 2*Nf*hf*tf)
Vmax = mdot/(rho*A_min_source)
Re = mdot*do/(A_min_source*mu)
```

Its basis is `SOURCE_EQUIVALENT_FLOW_AREA`, exposed with `flow_area_source=
HOLFELD_2016`. It is not an orthographic blockage result, a mass-ratio correction
or a surface-area correction. Bundle geometry's automatic `minimum_free_flow_area`
property remains guarded; source providers supply the explicit area instead.

Holfeld Eqs. 2-17/2-19 supply the native `Ar` and source surface area. The
resistance network uses native primary and fin areas with physical film h:
`G_out = h*(A_primary_source + eta_corrected*A_fin_source)`. Eta enters once.
`area_outside_gross`/reported `A_o` use that source reference;
`area_outside_geometric` retains the Phase A geometric area. Native area can
be smaller than the geometric radial-sector area; no geometry override occurs.

Hashizume's theory couples a continuous annular root to straight segmented
fins with insulated tips/sides. Its theoretical denominator counts broad faces.
The empirical correction supplies physical-fin-area-based eta, including the
effects excluded by that theory. The implementation uses the explicit 2002
STAGGERED restatement because the 2000 main equation and its appendix disagree
on a symbol. Corrected mean eta does not define a corrected temperature field:
tip temperature and full fin-skin extrema remain unavailable. The ideal weld
base/core temperatures remain available. These outputs cannot support wet
onset calculations.

### Domain and pressure convention

Every component is guarded; complete execution uses their **intersection**,
even when the generic engine's `strict_source_domain` flag is false. The
complete Re interval is `[5000,50000)`, with at least four longitudinal rows;
the source reports no upper row count. Additional dimensional and ratio bounds
are machine-readable, including `df/do` 1.80–2.15, `hs/hf` .50–.86 in the
intersection, `hs/ws<=2.60`, `m*hf<=2.5`, `Pt/Pl<2`, and layout angle <45°.
Passing these few examples of bounds does not replace all the other checks.
Unknown source bounds remain explicit diagnostics. No row correction is added.

Failures identify `FLOW_AREA`, `HT`, `DP`, `ETA` or `PROFILE`. Where available,
preceding component statuses are retained, for example `HT_DOMAIN=PASS`,
`DP_DOMAIN=PASS`, `ETA_DOMAIN=FAIL`.

Pressure loss stays on Holfeld's native `Eu=2*dp/(Nl*rho*Vmax^2)` basis.
`dp_drag=Eu*Nl*rho*Vmax^2/2` uses one mean-temperature evaluation and mean
endpoint density. It is not converted through RB, Darcy/Fanning, Frass or
Genic coefficients; their `f`/`xi` aliases remain unavailable (NaN).
The correlated quantity is the adiabatic measured bundle-section pressure
difference treated as friction by the source after neglecting approximately
1% acceleration. Exact tap distances and an empty-channel subtraction are
unreported; it is not an isolated skin-friction measurement. No extra inlet/exit
loss coefficients are added. Signed acceleration uses face-area momentum
change separately, and `dp_total=dp_drag+dp_acceleration`.

Source equations, units, area/velocity/property/row mappings, access status,
symbol discrepancies and numerical fixture provenance are in the
[source mapping](validation/serrated_source_mapping_v095.md). Source-reported
regression coverage is metadata, not a prediction-uncertainty guarantee.
Kawaguchi remains validation-only; the qualified HTC equation comparison and
unresolved hydraulic comparison are recorded in the validation report.

Geometry provenance is directly inspectable:
`geometry_model=IDEALIZED_RADIAL_SECTOR_CUT`,
`geometry_provenance=ENGINEERING_DERIVATION`,
`physical_product_exactness=NOT_CLAIMED`,
`validated_against_manufacturer_CAD=False`. Existing `ModelWarning`
diagnostics include `SERRATED_IDEALIZED_RADIAL_SECTOR_CUT`. Public tests
verify guards and unchanged solid routes; they establish implementation
consistency, not empirical validation.

See the historical [Phase A validation record](validation/serrated_geometry_v095.md)
and current [Phase B validation record](validation/serrated_physics_v095.md).
