# Continuous plate fins: geometry and experimental dry profile

Status: **IMPLEMENTED_BUT_SOURCE_BLOCKED**. This feature branch is preparation
for v0.9.6, not a production v0.9.6 release. The package version stays 0.9.5.
Wang source closure and experimental validation remain incomplete. Do not use
the experimental profile as validated design physics. No provider is added to
AUTO. Geometry is independent of dry, wet or condensing thermal processes.

## Geometry and SI inputs

`TubeBundle(..., continuous_fin_geometry=ContinuousPlateFinGeometry(...))`
owns one shared sheet pack around the existing `BareTube` cores. Tube diameter,
length, material, lattice, row counts, passes and pitches remain authoritative
in their existing objects. The new object contains only:

- Integer sheet count, thickness, pitch and first-sheet leading axial position.
- Rectangular sheet height/depth and transverse/streamwise origins, relative
  to `bundle.tube_center_coordinates` (first physical tube at x=y=0).
- Measured collar outside diameter, collar height including sheet thickness,
  fin conductivity and mechanically expanded collar attachment identity.

The aperture fits the collar OD, the collar bore fits the expanded tube OD,
and collars extend from each sheet leading face in positive axial z. Bypass
outside the specified rectangular face is excluded: the active duct spans the
full effective tube length and actual sheet height. Sheet/collar intervals
must fit inside that length. Collars may touch at one fin pitch; they cannot
overlap. Every aperture must fit strictly inside the sheet and all passages
must remain open. Arbitrary resolved lattices are geometry inputs; each
physical provider separately restricts applicability.

`bundle.continuous_fin_surface` is a cached immutable ledger. With Nt actual
tubes, Nf sheets, sheet thickness t, collar height hc and effective length W:

```text
A_tube    = Nt pi Do (W - Nf hc)
A_collar  = Nt pi Dc Nf (hc - t)
A_faces   = exposed broad-face count * (H L - Nt pi Dc^2/4)
A_edges   = 2 Nf t H   # inlet/outlet edges; side-wall edges are sealed
A_lips    = exposed collar end count * Nt pi (Dc^2 - Do^2)/4
A_contact = Nt Nf pi Do hc
A_gross   = A_tube + A_collar + A_faces + A_edges + A_lips
```

Touching collar ends and sealed axial end faces are excluded from exposed
lip area. A broad face flush with an axial end wall and sheet edges against
the transverse duct walls are sealed, not gas-exposed; their areas remain
separately visible in the ledger. Aperture walls are covered by collars and are not counted again.
Each sheet is counted once; no equivalent annular fin diameter is invented.
The idealized construction excludes formed bends, collar flares, burrs,
variable thickness, tolerances and gaps in mechanical contact.

## Finite flow area and reference diameters

At each open axial layer, finite throat width is the shortest side-wall to
side-wall separator through the actual circular tube/collar obstacles. Its
clear segments include transverse and diagonal passages and actual side
margins. There is no periodic wrap at the sheet boundary. Dijkstra's algorithm
evaluates the nonnegative gap graph; independent exhaustive enumeration and
hand-calculated cases test that construction. This is a geometric local
throat-normal area, not a resolved velocity field or a source-validated
hydraulic model.

The bare and collar open axial lengths weight their respective throat widths;
sheet-thickness layers are closed. Actual frontal area is H W and sigma is
Ac/(H W). Legacy per-tube blockage and periodic-gap helpers raise for common
sheets. The experimental source adapter explicitly uses:

```text
Ao_candidate = A_tube + A_collar + A_faces
Dh_candidate = 4 Ac L / Ao_candidate
Gc = external mass flow / Ac
Re_Dc = Gc Dc / mu
```

Do is the metal tube OD, Dc the measured collar OD and Dh the declared compact
passage diameter. They are never substituted for one another. Acceptance of
the candidate surface/diameter convention by Wang remains an open source gate.
`total_outer_area` remains the full physical ledger including edges and lips.

## Explicit research API

```python
from core.geometry import BareTube, TubeBundle, ContinuousPlateFinGeometry
from core.heat_transfer import ExperimentalWang2000Model
from core.heat_transfer.airside_correction import MultiplicativeAirsideCorrection
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.properties.dry_air import DryAirPropertyProvider
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider

fin = ContinuousPlateFinGeometry(
    sheet_count=180, thickness=0.000115, pitch=0.002,
    first_sheet_position=0.0, height=0.2032, depth=0.05715,
    transverse_origin=-0.0127, streamwise_origin=-0.009525,
    collar_outer_diameter=0.00975, collar_height=0.002,
    conductivity=205.0,
)
bundle = TubeBundle(
    BareTube(0.008, 0.00952, 0.36, 0.36, 45.0),
    n_rows=3, n_tubes_per_row=7.5, pitch_transverse=0.0254,
    pitch_longitudinal=0.01905, layout="staggered", n_passes_tube=1,
    flow_arrangement="counterflow", continuous_fin_geometry=fin,
)
model = ExperimentalWang2000Model(
    acknowledge_source_limits=True,
    correction_provider=MultiplicativeAirsideCorrection(
        h_film_multiplier=1.0, pressure_drop_multiplier=1.0,
        provenance="explicit_research_identity",
    ),
)
hx = BareTubeHeatExchanger(bundle, airside_model=model)
inside = HXSideInput(
    provider=ConstantPropertyProvider(FluidTransportProperties(990., .0007, .63, 4200.)),
    m_dot=.1, T_in=360., p=101325.,
)
for mass_flow in (.1, .2):
    outside = HXSideInput(provider=DryAirPropertyProvider(),
        m_dot=mass_flow, T_in=290., p=101325.)
    result = hx.simulate(inside, outside, iterate=True)
    duty, ua = result.q, result.UA
    pressure_estimate = result.outside_dp_total
    diagnostics = result.finned_tube_diagnostics
```

The profile explicitly asserts dry air. Constant/custom transport providers
must satisfy that assertion; numbers alone do not establish gas identity.
Known external wet/condensing and liquid providers are rejected. Direct
`ContinuousFinRequest` also carries explicit fluid and surface identities.
Without a compatible profile, execution fails before a bare/individual-fin
fallback. Nondefault bare/individual-fin selectors are incompatible overrides.

Rating uses the existing `.rate(BalanceSideSpec, BalanceSideSpec, ...)` API;
it always performs the established iterative rating calculation. Direct
`.solve()` and `.solve_thermal_state()` also propagate the same profile.
Fan curves, selection and equipment orchestration belong outside this engine.

## Source and contact semantics

Separate `ExperimentalWang2000HTC` and `ExperimentalWang2000DP` providers use
`ContinuousFinRequest`. `ContinuousFinHTCProvider` and `ContinuousFinDPProvider`
define the independent interfaces. Their shared immutable source card records
DOI, equations, bases, source status, unknown Reynolds envelope, pressure scope,
and the P2/P5/F3/P6 transcription conflicts. The complete profile accepts only
the compatible component set, not an undocumented Kim/Wang hybrid.

Wang's nominal secondary-source restrictions are enforced: conventional
STAGGERED, N=1..6, Do=6.35..12.7 mm, Fp=1.19..8.7 mm, Pt=17.7..31.75 mm,
Pl=12.4..27.5 mm. No row clipping or industrial diameter extrapolation occurs.
The source Reynolds range remains **unknown**, including at Re=100; 300..20000
is not promoted from software documentation into a primary source domain.
`strict_source_domain=True` therefore raises. Re>1 is only the mathematical
logarithm guard. Each sealed axial end gap is restricted to at most one pitch.

The selected h is an **effective reduced empirical HTC including contact**.
The shared result field `alpha_physical` and existing correction field name
`h_film_multiplier` are compatibility names, not a claim of intrinsic gas-film
physics. `ContactPolicy.EMPIRICAL_INCLUDED` rejects additional resistance and
the relabelling of this correlation as ideal contact. Wall and fouling terms
remain independent physical resistances.

## Efficiency, corrections and resistance

The small Schmidt kernel is MIT-adapted from pinned python-hvac; see
[THIRD_PARTY.md](../THIRD_PARTY.md). It uses M=Pt/2 and
L=sqrt(Pl^2+(Pt/2)^2)/2, the distances to the two neighbour bisectors. Only an
equilateral triangular lattice has M=L. Cells outside the admitted staggered
topology, including repeated unshifted section boundaries, are rejected.

Finite edge cells remain unresolved by the periodic equivalent-cell model.
Every result explicitly warns about this B05 limitation. No exposed-tip t/2
extension is added at an adiabatic cell symmetry boundary. Exposed sheet edges
and collar lips are in physical gross area but adiabatic in this thermal
approximation. A terminal sheet with one sealed broad face uses the one-face
convection parameter m^2=h/(k t); interior sheets use m^2=2h/(k t). Face-area
weighting combines those efficiencies. Thus the overall efficiency is based on effective area divided
by gross area and need not approach one even when face efficiency does.

Each evaluation creates frozen baseline HTC/DP results, invokes the existing
`AirsideCorrectionProvider`, applies its independent channel corrections once,
then evaluates eta at corrected h. Reporting does not reapply corrections.
With Af denoting broad fin faces:

```text
Goutside = h_corrected (A_primary + eta_face Af)
Rtotal = 1/(hi Ai) + Rfi/Ai + Rwall + Rfo/A_gross + 1/Goutside
UA = 1/Rtotal
```

Baseline, correction, corrected h/friction estimate and efficiency are in
`finned_tube_diagnostics.continuous_result`. The final hydraulic evaluation
has its own fresh snapshot in `.hydraulic_result.continuous_result`.
External mass flow is retained by the hydraulic snapshot as
`reference_mass_flux * reference_area`, without a duplicate scalar column.

## Pressure scope and steam limitations

The candidate constant-density compact-HX estimate is
`f (Ao_candidate/Ac) Gc^2/(2 rho)`. There is no extra factor four. Three-state
evaluation uses mean-temperature transport and harmonic endpoint density for
the positive friction estimate. A separate signed source density term is
`Gc^2/(2 rho_in) (1+sigma^2) (rho_in/rho_out-1)`. Only the positive friction
estimate receives the hydraulic engineering correction. No second acceleration
term or undocumented entry/exit term is added.

Compatibility fields `dp_drag` and `dp_acceleration` carry these two research
terms, without establishing irreversible-loss decomposition. The bank reports
`validated_bank_irreversible=False`; path extraction returns the existing
`SourceDomainPressureEstimateResult` and refuses local irreversible aggregation.
Use `hydraulic_result.reference_density` and `.reference_velocity` for the
core estimate; `.reference_area` and `.reference_diameter` give its area and
collar diameter. Individual hydraulic points retain their own actual densities
and velocities. Pressure scope must be resolved before any validated fan pressure match.

IAPWS97/Shah steam thermal paths retain their orientation and applicability
guards. Saturated condensation, desuperheating/condensation and representative
three-zone operation are tested. Common-fin conduction between zones at
different temperatures is neglected. Some three-zone trials still fail the
existing parallel zone-allocation convergence guard; no tolerance is relaxed.
Two-phase tube-side DP remains unsupported and NaN, with its existing warning.
Mean Schmidt eta does not determine finite-sheet temperature extrema, so
those remain unavailable. Wet sheets, wet DP, industrial flue-gas condensation
and simultaneous active phase change on both sides are outside this profile.
Reported core-wall/equivalent base nodes are lumped network quantities, not
resolved physical tube/collar contact temperatures.

## Release gates

See [the implementation and validation record](validation/continuous_plate_fin_implementation_2026-10-31.md).
No release tag, package publication, version promotion, push or final PR is
authorized by passing arithmetic tests alone. The original Elmahdy-Mitalas wet
default is unchanged. No private reporter changes or dependencies are included.
