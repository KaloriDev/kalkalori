# Elliptical roughness and reference reporting — v0.9.10

The original smooth ellipse models remain the default. Rough hydraulics and
smooth-wall heat transfer on a rough tube require separate permissions:

```python
from core.heat_transfer import EllipticalRoughnessPolicy
from core.models.bare_tube import BareTubeHeatExchanger

hx = BareTubeHeatExchanger(
    bundle,
    elliptical_roughness_policy=EllipticalRoughnessPolicy(
        allow_hydraulic_roughness=True,
        allow_smooth_htc_on_rough_wall=True,
    ),
    # Keep outside models/providers appropriate for the actual bundle.
)
```

Set absolute equivalent sand roughness in the actual `EllipticalTube`'s
`roughness_inner` [m]. The rectangular-finned wrapper forwards it to
hydraulic inputs and diagnostics. No dimensions change. Direct internal HTC
and friction calls accept `roughness_policy`; Rating, Simulation, wall
iteration and endpoint probes share the exchanger policy.

## Sources and method

The existing [Nikitin–Yakhot smooth elliptical DNS model](https://doi.org/10.1017/S0022112005003964)
uses a Jones modified Reynolds number. Its DNS comparisons cover Re_Dh=6000
and minor/major ratios 0.67 and 0.5, not rough elliptical ducts.
[Jones's primary rectangular-duct study](https://pages.mtu.edu/~fmorriso/cm310/JonesOCpaper1976.pdf)
supports geometric Reynolds scaling for smooth turbulent friction; it does
not establish a universal rough-wall ellipse law.

For actual inner semi-axes `a,b`, perimeter `P` and mean unblocked velocity:

```text
Dh = 4*pi*a*b/P
Re_Dh = rho*v*Dh/mu
Re_star = Re_Dh * 8*a^2*b^2 / (Dh^2*(a^2+b^2))
1/sqrt(f_ellipse_smooth) = 2*log10(Re_star*sqrt(f_ellipse_smooth)) - 0.8
```

The opt-in extension is an engineering construction, **not a published or
validated rough-ellipse correlation**:

```text
r = epsilon/Dh
f_Darcy = f_ellipse_smooth * f_CW(Re_star,r) / f_CW(Re_star,0)
1/sqrt(f_CW) = -2*log10(r/3.7 + 2.51/(Re_star*sqrt(f_CW)))
dp_friction/dz = f_Darcy*rho*v^2/(2*Dh)
f_Fanning = f_Darcy/4
```

[Colebrook's pipe law](https://doi.org/10.1680/ijoti.1939.13150) supplies
the circular rough/smooth comparison. Normalizing at identical `Re_star`
retains shape dependence and avoids a discontinuity from the rounded
smooth-law constants. At epsilon=0 the existing smooth solver runs unchanged,
retaining its exact result. For fixed geometry/properties the ratio increases
with roughness. Actual Dh remains in relative roughness and pressure gradient;
no equivalent circular bore replaces the physical section. The existing
Colebrook solver is reused; [Haaland](https://doi.org/10.1115/1.3240948)
only seeds its iteration.

The [NETL primary rough-channel experiment](https://www.netl.doe.gov/projects/files/DOE.NETL-2021.2655SurfaceRoughnessEffectsonHeatTransfer_09092021.pdf)
tested square/rectangular sCO2 channels at epsilon/Dh=0.0029–0.0073. Friction
differed from Colebrook by about 8%; thermal correlations differed more.
This supports consideration of noncircular rough-flow engineering estimates,
but does not validate industrial elliptical water/oil tubes. The normalized
combination above is our engineering inference, not a method prescribed by
NETL or Jones.

Implementation scope: fully developed, constant-property, Newtonian no-slip
turbulence; Re_Dh>=4000; 1<=a/b<=5; epsilon/Dh<=0.01. The roughness ceiling
is a conservative implementation restriction, not an experimentally
established ellipse limit. Positive roughness in laminar flow remains an
error; transition Re 2300–4000 remains unsupported. Nonuniform corrosion,
ribs, deposits, altered flow area and anisotropic roughness are outside scope.
Equivalent sand roughness is not automatically equal to Ra.

Model ID `elliptical_jones_normalized_colebrook_engineering` carries sources,
`approximation=True`, relative roughness, modified Re and multiplier. Each
hydraulic point retains `friction_diagnostics`; warnings propagate independently
of HTC. Historical two-value friction callback usage is preserved.

## Independent heat-transfer applicability

[Yang et al. (2018)](https://www.techscience.com/fhmt/v11n1/53399) fit smooth
elliptical CFD heat transfer without a roughness parameter. Circular rough
friction/heat-transfer analogies cannot be inserted into this fit without
additional assumptions and validation. No rough HTC correlation is added.

`allow_smooth_htc_on_rough_wall=True` evaluates unchanged Yang HTC and emits
`elliptical_rough_wall_smooth_htc_approximation`. The thermal roughness effect
remains unquantified; unchanged HTC is not an asserted lower bound. Yang
shape, source-Re, development and boundary-condition restrictions remain.
Hydraulic permission alone enables direct friction; thermal permission alone
enables direct HTC approximation. A combined rough exchanger needs both.

## Local losses

Distributed friction uses actual mean tube velocity and hydraulic length
across passes. Bulk acceleration is a separate signed term. Ellipse tube-sheet
entrance/exit losses remain omitted with `tube_end_losses_not_evaluated`;
zero bookkeeping fields do not predict zero physical end loss. Pass count
does not invent return-bend/header K values.

Specified local paths remain explicit. Inlet, return and outlet assemblies
evaluate their own supplied sections/states separately from the core.
User-defined K uses the selected section's `rho*v^2/2`, with
`v=mdot/(rho*A)`; area-change dynamic-pressure terms remain separate from
irreversible loss. `CustomFlowSection` can carry an actual ellipse's area
and Dh for a justified supplied K. Circular-elbow formulas do not become
validated ellipse fittings by supplying Dh. No new ellipse fitting
coefficients are inferred.

## Geometry and state reporting

`core.geometry.reporting.rectangular_fin_geometry_metadata(tube,n_tubes=...)`
returns intrinsic values without consulting outside models:

| Quantity | Physical mapping |
|---|---|
| Pitch / density | `fin.axial_pitch` / its reciprocal |
| Conductivity | `fin.thermal_conductivity` |
| Root / tip thickness | Both actual `fin.thickness`; constant plate, no taper |
| Fin count | `length_effective / axial_pitch`, fractional periodic count |
| Physical fin area | Two faces excluding aperture, plus outer edge |
| Primary / gross / inside areas | Exposed core / primary plus physical fin / inner perimeter times active length |

Gross-area resistance bookkeeping includes geometric edge metal. The
adiabatic-edge fin solver convects only broad faces: effective area is
`A_primary + eta_fin*A_fin_faces`. Physical fin area is neither projected nor
efficiency weighted. Faces and edge, effective finned length and count/area
conventions are explicit. Tube ends/discrete end-fin placement remain excluded.

`minimum_free_flow_area_status(bundle)` returns availability and reason.
SHIFTED remains unavailable: no conventional throat or unresolved lattice
phase is inferred. Other guarded geometries retain their reason. Frontal
and provider reference area remain distinct. External
`OutsideReferenceQuantities` optionally supplies `flow_area`, `flow_area_basis`
and `mass_flux`; a supplied area requires a nonempty basis. Omitted reference
quantities remain unknown. Existing positional fields remain compatible.

`core.heat_transfer.reference_state.result_reference_states(result,bundle)`
exposes `inside_thermal`, `inside_hydraulic`, `outside_thermal` and
`outside_hydraulic` snapshots. Converged thermal diagnostics retain actual
correlation Re, property temperature/pressure, diameter, velocity, mass flux
and flow area/basis. Hydraulic snapshots use the final midpoint/property
method; ellipse friction supplies modified Re. Outside definitions stay
provider-specific. Missing references and unsupported equivalent/phase-zone
states remain unavailable. Snapshots do not reevaluate properties.

Temperatures identify the bulk-property reference used for Re; existing wall
temperature/correction diagnostics remain separate. They do not imply that
every property or wall correction was evaluated at one common temperature.

The private reporting adapter appends named state columns and intrinsic
geometry/availability columns to established schemas. Excel exports those
same DataFrames. Legacy `inside_Re`/`outside_Re` retain their driver-dependent
compatibility selection (`*_Re_mean` when present, otherwise final kernel
thermal aliases); use explicit channel columns for comparisons. Existing
round/plain/continuous/serrated columns retain their meanings.

Public tests use synthetic geometry/fluids. Private project inputs,
calibration curves and workbook evidence remain outside the public package.
