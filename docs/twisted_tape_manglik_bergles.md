# Tube-side enhancement: open-source basis and external providers

The original source-gap checkpoint `29c7183` is retained in history. The
public implementation policy below supersedes its requirement to obtain
Manglik-Bergles Part I/II before implementing any public model.
Version remains v0.7.10; this is unreleased v0.8 work.

## PUBLIC BUILT-IN MODEL

Selected provider: `Yang2020TwistedTapeProvider`, single conventional tape
(TT) specialization of Yang, Liu, Cao, Gao and Ding (2020), *Numerical
Analysis of FLiBe Laminar Convective Heat Transfer Characteristics in Tubes
Fitted With Coaxial Cross Twisted Tape Inserts*, Front. Energy Res. 8:178.
Implementation source **Y**: [publisher full text](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2020.00178/full),
[original PDF](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2020.00178/pdf),
DOI [10.3389/fenrg.2020.00178](https://doi.org/10.3389/fenrg.2020.00178).
Free access: publisher open access, CC BY. Equations visually checked in
original PDF pages 3, 11 and 12; no OCR reconstruction.

Implement Y's **new** Eqs. (21)-(22), with TT coefficients A=59.24,
B=3.81, n=1, m=2. Do not implement its historical M&B Eqs. (9)-(10).
This is a numerical-study correlation, not canonical M&B reproduction.

### Frozen equations (SI)

```text
D = inside diameter; delta = tape thickness; H = axial half-turn length
y = H/D
A_empty = pi*D**2/4
A_open = A_empty - delta*D
P_wetted = pi*D + 2*D - 2*delta
D_h = 4*A_open/P_wetted
phi = A_empty/A_open
psi = D/D_h
u = mass_flow_per_tube/(rho*A_empty)
Re = rho*u*D/mu
Pr = cp*mu/k
s = 1 + (pi/(2*y))**2
u_axial = u*phi
u_swirl = u*phi*sqrt(s)
Sw = Re*phi*sqrt(s/y)
Gz = mass_flow_per_tube*cp/(k*heated_length)
f_D = 59.24*phi*psi**2*(1 + 1.348e-3*Sw**1.09)**0.462*s/Re
Nu_base = 3.81*((1 + 0.2165*Gz**0.662)**0.251
               + 3.87e-2*(Sw*Pr**0.4)**0.431)**2.06
wall_factor = (mu_bulk/mu_wall)**0.14
Nu = Nu_base*wall_factor
alpha_inside = Nu*k/D
f_F = f_D/4
pressure_gradient = f_D*rho*u**2/(2*D)
```

Here `u`, `u_axial` and `u_swirl` denote velocities (the ASCII `u` is not
kinematic viscosity). The provider reports D_h, but pressure gradient uses
D and empty-tube u: blockage is already inside phi, psi and Sw.

### Source cross-check and classification

All references to Y below use the freely accessible DOI/PDF above.
`VERIFIED_OPEN` means the production expression is explicitly printed or
algebraically derived from the cited definition. Algebraic checks are not
claimed as independent experimental validation. Second-source confirmation
of the complete selected correlation has not been established.

| Quantity / exact meaning | Y location; free access | Cross-check / agreement | Status | KalKalori representation / limits / unresolved items |
| --- | --- | --- | --- | --- |
| D, inner tube diameter | p.3, Eq.1; open | nomenclature p.15 agrees | VERIFIED_OPEN | tube.D_i; strict studied 0.012 m |
| Tape width, clearance c | p.3 physical model, Eq.11; open | zero clearance and blocked area agree | VERIFIED_OPEN | tape_width=D, no loose-fit extrapolation |
| delta, tape thickness | p.3, Eqs.11-12; open | same parameter in geometry equations | VERIFIED_OPEN | tape_thickness=0.001 m |
| H, 180-degree axial twist length | p.3; open | nomenclature p.15 agrees | VERIFIED_OPEN | half_turn_length, never ambiguous pitch |
| y=H/D | p.3; open | Eq.13 uses same y | VERIFIED_OPEN | derived, 2 <= y <= 4 |
| Empty and open axial area | Eq.11; open | phi=A_empty/A_open | VERIFIED_OPEN | pi D^2/4 and minus delta D; no global bundle mutation |
| Wetted perimeter | Eq.12; open | D_h=4A/P | VERIFIED_OPEN | pi D+2D-2delta; geometric correlation approximation |
| Hydraulic diameter | Eqs.12,20; open | n=1,m=2 reduce identically | VERIFIED_OPEN | diagnostic D/psi, not pressure denominator |
| Reference velocity u | Eqs.1,14; open | Eq.14 distinguishes u from blocked/swirl velocity | VERIFIED_OPEN | empty-tube mass-flow velocity |
| Axial / swirl velocities | Eq.14; open | continuity with Eq.11 | VERIFIED_OPEN | u*phi and u*phi*sqrt(s) |
| Re | Eq.1; open | Eq.14/13 reference consistency | VERIFIED_OPEN | rho*u*D/mu, 100..1100 inclusive |
| Sw | final expression Eq.13; open | Eq.14 gives Sw=rho*u_swirl*D/(mu*sqrt(y)) | VERIFIED_OPEN | provider-specific diagnostic; no generic-contract requirement |
| Pr | nomenclature p.15; open | cp*mu/k | VERIFIED_OPEN | 7..900 inclusive |
| Gz | p.10, nomenclature; open | M is mass flow, not bundle-wide flow | VERIFIED_OPEN | per-tube M*cp/(k*L) |
| Nu, alpha | Eq.3 and Eq.22; open | Nu=hD/k | VERIFIED_OPEN | physical tube-wall film, no tape fin area |
| Thermal area | pp.3-4 boundary conditions; open | tape is adiabatic | VERIFIED_OPEN | physical inner wall pi D L |
| Distributed friction | Eqs.2,21; open | Eq.2 is Darcy-Weisbach directly | VERIFIED_OPEN | f_D native; f_F=f_D/4, no extra factor four |
| Bulk/wall correction | Eq.22; open | viscosity definitions p.10 | VERIFIED_OPEN | (mu_bulk/mu_wall)^0.14, applied once |
| Liquid applicability | pp.3-4 and p.12; open | water/oil/FLiBe study | VERIFIED_OPEN | single-phase incompressible forced-convection liquid heating |
| Laminar range | p.12; open | stated Re,Pr,y intervals | VERIFIED_OPEN | only above intervals, explicit error outside |
| Length and geometry range | p.3; open | no published broad size validation | VERIFIED_OPEN | strict L=0.3 m, D=0.012 m, delta=0.001 m; no size extrapolation |
| Buoyancy / radiation | pp.3,11; open | neglected in model | VERIFIED_OPEN | forced-convection assumption warning, caller applicability responsibility |
| Turbulent branch | not specified for selected model | unavailable | OPEN_SOURCE_GAP | unsupported, never smooth fallback |
| Transition rule | not specified for selected model | unavailable | OPEN_SOURCE_GAP | unsupported; no invented interpolation |
| Gas / cooling / supercritical extension | not validated by selected model | unavailable | OPEN_SOURCE_GAP | unsupported |
| M&B-specific Ra/Re_ax correction | excluded historical Eq.10 | unused | NOT_REQUIRED | no hidden completion from Part I/II |
| Insert local losses | no separately validated model | unused | NOT_REQUIRED | excluded; existing local losses unchanged |

Other candidates audited: MDPI [10.3390/fluids9120293](https://doi.org/10.3390/fluids9120293)
and SciELO *A New Correlation for Single and Two-Phase Flow Pressure Drop
in Round Tubes with Twisted-Tape Inserts*. A complete additional paired
thermal/hydraulic model was not verified from those candidates in this
freeze. They are not implementation authorities. Broader regime coverage
is deferred rather than assembled from different sources or conventions.

### Integration interpretation and validation

Keep one provider instance/configuration for thermal and hydraulic paths.
The source uses inlet-reference properties and an isothermal-wall study.
Application at KalKalori's iterated mean bulk/wall state and at three
hydraulic quadrature states is an explicit 0D engineering approximation,
reported as a warning; it is not a new spatially resolved validation.
Every evaluated state must stay inside the provider's limits. Gz uses the
single-tube heated length, never the total multi-pass hydraulic length.
Only full-length heated tapes with length_total=length_effective are enabled
by the built-in provider. Local losses and signed acceleration retain their
existing reference geometry and separate accounting.

When no wall state exists during initialization/hydraulic-only evaluation,
use unity correction with an explicit diagnostic. Iterative thermal output
must use authoritative wall viscosity. No gas/length smooth-tube correction
may be multiplied onto the enhanced result. Cooling, unsupported phases,
regimes and geometries must raise a controlled error.

Independent equation-level anchors will encode numerical constants from
hand calculations of the printed equations (not implementation calls).
Y's Tables 1-2 validate CFD against other correlations/experiments; they
are not exact anchors for the new Eqs.21-22 and must not be mislabelled.
Tests also lock Darcy/Fanning, blockage references, limits and smooth defaults.

## PRIVATE M&B PROVIDER - FUTURE

Manglik and Bergles (1993), Part I, DOI 10.1115/1.2911383, and Part II,
DOI 10.1115/1.2911384, are reserved as the basis for a future private external
provider from legitimately acquired full sources. Neither is an authority
for public built-in physics under current project policy. The earlier OCR
and Scribd material must not complete missing definitions or equations.
No private M&B implementation is present or claimed in the GPL repository.

A private package can implement the common coherent provider contract with
its own reference geometry, Nu/alpha, Darcy friction, regime, applicability,
provenance and additional typed diagnostics. The solver must not depend on
private equation semantics. Supplier tables, licensed software and proprietary
manufacturer adapters use that same interface, without implying that
CALGAVIN/hiTRAN and twisted tapes share physics. No manufacturer correlation
or proprietary source material is included.

## Audit of the existing production paths

The audit baseline is main commit `42b9da4`, after the v0.7.10 merge.

- `core/geometry/bundle.py`: `internal_flow_area_per_pass` multiplies the
  effective tubes per pass by the tube's flow area;
  `internal_hydraulic_diameter` delegates to the tube. These quantities
  also serve existing hydraulic/local-loss paths. Do not globally reduce
  them for tape blockage. Keep physical inner-wall heat-transfer area.
- `core/heat_transfer/internal_flow.py`: smooth thermal evaluation owns
  laminar/Gnielinski Nu, its Re = 2300...4000 blend, optional gas wall
  correction and finite heated-length correction. An enhancement result
  must bypass those correlation-specific operations.
- `core/heat_transfer/thermal_iteration.py`: `_evaluate_local_wall_state`
  evaluates authoritative bulk and wall properties, calls the internal
  diagnostic correlation, and collects `ModelWarning` values. It is shared
  by the iterative thermal state and wall-temperature probes. Dispatch here
  must pass bulk/wall states to the selected enhancement provider once,
  with no subsequent smooth-tube wall or length multiplier.
- `core/models/bare_tube.py`: `BareTubeHeatExchanger.solve` calls internal
  heat transfer and `calculate_tube_bundle_hydraulics` separately. A shared
  enhancement configuration must govern both paths, including the
  constant-property route.
- `core/pressure_drop/internal_pressure_drop.py`: inlet, midpoint and
  outlet hydraulic points currently use Darcy friction factors. Straight
  friction uses Simpson integration of f/rho (or f*G^2/rho for differing
  point flows). Signed acceleration is a separate endpoint momentum term.
  The provider's friction must be integrated with its own verified
  reference area, velocity and diameter, without a second blockage factor.
  Preserve nozzle, chamber, tube-sheet and return-loss models separately.
- `core/models/rating.py` and `core/models/simulation.py`: authoritative
  iterative thermal states coexist with a `solve` snapshot for hydraulics.
  Both must expose the same provider identity and applicable diagnostics.
  Preserve `core/models/surface_margin.py` semantics and physical areas.
- `core/common/warnings.py` provides typed `ModelWarning` and applicability
  helpers. Provider warnings must reach standard result `warnings`.
  Existing controlled phase-boundary errors in `core/phase_change/capability.py`
  provide a pattern for explicit rejection, not silent smooth fallback.
