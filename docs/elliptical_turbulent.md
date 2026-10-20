# Turbulent ellipse flow and nonconfocal walls — v0.9.2

These models extend the explicit dry rectangular-finned ellipse workflow.
They retain the actual inner/outer contours and parallel flow area. Selection
does not change dimensions, fluids, required duty or mass flows. Existing
laminar H1, exact confocal and circular models retain their previous equations.

```python
from core.heat_transfer import EllipticalFlowAssumptions, TubeWallModel
from core.models.bare_tube import BareTubeHeatExchanger

hx = BareTubeHeatExchanger(
    bundle,
    elliptical_flow_assumptions=EllipticalFlowAssumptions(
        regime='turbulent', development='fully_developed',
        thermal_boundary_condition='T'),
    tube_wall_model=TubeWallModel.ELLIPTICAL_NUMERICAL,
)
```

Normal `rate` and `simulate` use these selections. No circular HTC, circular
entrance/exit law or thin-wall formula is substituted. The outside model and
its source/reconstruction limits are independent of this internal extension.

## Internal heat transfer

Yang, Wang, Wang, Li and Zhang (2018), *Correlation for turbulent convection
heat transfer in elliptical tubes by numerical simulations*, FHMT 11, 7,
[DOI 10.5098/hmt.11.7](https://www.techscience.com/fhmt/v11n1/53399),
CC BY 4.0, Eq.18 and Table 4:

```text
Dh = 4*pi*ai*bi / P_inner
Re = rho*v_mean*Dh/mu
Pr = cp*mu/k_fluid
Nu = C*(0.02979*Re**0.8 - 8.5626)*Pr**0.4
h_inside = Nu*k_fluid/Dh
```

Here `P_inner` means the actual inner wetted perimeter. The implemented Reynolds
interval is 10,000–120,000, inclusive, with only 1e-12 relative arithmetic
tolerance at endpoints. The higher-Re Eq.19 and transition blending are not
implemented. Table 4 gives:

| Inner ai/bi | C |
|---|---:|
| 1.5 | 1.00 |
| 2 | 1.27 |
| 2.5 | 1.17 |
| 3–5 | 0.94 |

The source labels the last group `>=3`; this implementation stops at 5, the
largest studied ratio. Below 3 only the listed ratios are accepted (1e-12
relative tolerance); no interpolation is invented between discrete entries.

The source is a smooth, constant-property Newtonian CFD study with constant
wall temperature **T** and fully developed flow. It uses length 100 Dh and
discusses development after about 60 Dh. The implementation warns for heated
length below 60 Dh; a longer duct does not certify a fully developed inlet or
make a developing average equivalent to the source's local developed HTC.
The caller must justify the declared developed approximation. No entrance
or wall-property correction is applied. The source discusses water and checks
other fluids, but supplies no universal Pr applicability interval. Positive
Pr is required, and this lack of validation bounds is explicitly diagnosed.

The result identifies `elliptical_yang2018_turbulent_T` and carries source,
reference area/dimension, assumptions and `approximation=True`. Normal thermal
iteration preserves this metadata in
`result.thermal_state.diagnostics.inside_model_metadata`. A constant-wall CFD
fit used at representative bulk properties in a 0D exchanger remains an
engineering approximation, including when cooling water in a nonuniform wall.

## Distributed friction

Nikitin and Yakhot (2005), *Direct numerical simulation of turbulent flow in
elliptical ducts*, JFM 532, 141–164,
[DOI 10.1017/S0022112005003964](https://doi.org/10.1017/S0022112005003964).
The [author-hosted full paper](https://www.researchgate.net/publication/231908645_Direct_numerical_simulation_of_turbulent_flow_in_elliptical_ducts)
provides Eqs.3.2 and 3.5. The latter applies Jones's geometric Reynolds
adjustment to an ellipse:

```text
Re_modified = Re_Dh * 8*ai**2*bi**2 / (Dh**2*(ai**2+bi**2))
            = Re_Dh * 64/Po_ellipse
1/sqrt(f_Darcy) = 2*log10(Re_modified*sqrt(f_Darcy)) - 0.8
dp/dz = f_Darcy*rho*v_mean**2/(2*Dh)
```

The implicit equation is solved independently by a bracketed scalar root;
actual Dh still appears in the pressure gradient. The factor returned is
Darcy, with Fanning `f_Darcy/4`. This is geometry dependent and does not invoke
the existing circular friction implementation.

**Source DNS only covers Re_Dh=6000 and bi/ai=0.67, 0.5.** Application to other
Re or axis ratios is an explicitly warned engineering extrapolation. The API
is bounded to Re_Dh 4,000–120,000 and ai/bi 1–5; these are explicit
implementation limits, not a new validation claim or universal transition
criterion. Positive roughness and undeclared/developing flow remain errors.
The model ID is `elliptical_jones_prandtl_turbulent_engineering` and metadata
always says `approximation=True`.

Only straight distributed friction is established by this law. The existing
engine may also report signed bulk acceleration. Elliptical tube-sheet
entrance/exit and return/bend losses remain omitted with their existing scope
warning. A successful calculation does not validate those missing components.

## Numerical wall resistance

`ELLIPTICAL_NUMERICAL` solves steady two-dimensional Laplace conduction between
the supplied concentric coaxial ellipse contours. It does not require them
to be confocal or infer a constant normal wall thickness. Conductivity is
constant, both contour surfaces are isothermal, and axial/end effects are
neglected, as in the existing exact resistance model.

An independently constructed quarter-annulus mesh interpolates between the
actual contours. Four P1 triangles per quadrilateral preserve symmetry. The
weak diffusion kernel is shared with the existing fin solver, without changing
fin physics. Inner/outer normalized temperatures are 1/0; principal axes have
natural zero flux. The [public Poisson weak-form tutorial](https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html)
documents the variational formulation. No tutorial code or FEniCS dependency
is incorporated.

For the full-section dimensionless conductance `G` obtained from boundary
reactions, `R_wall = 1/(G*k_wall*L_effective*N_tubes)`, in K/W. Meshes start
with 8 radial and 32 angular cells per quarter. Both directions double until
successive resistance change and area error are <=1e-4; balance and algebraic
residual must each be <1e-8. Refinement/memory limits raise an error rather
than returning an unverified answer. Geometry-normalized immutable scalar
solutions use a bounded cache; k, length and physical tube count scale R.

```python
from core.heat_transfer import tube_wall_resistance_diagnostics

wall = tube_wall_resistance_diagnostics(
    core_elliptical_tube, n_tubes=bundle.n_tubes_total,
    model=TubeWallModel.ELLIPTICAL_NUMERICAL)
print(wall.resistance, wall.model_id, wall.convergence)
```

The result carries `elliptical_wall_laplace_P1`, `approximation=True`, mesh
resolution, measured relative resistance change, area error, residual and heat
balance error. Refinement change is an error indicator, not a rigorous bound
or an allowance for unknown physical contact conditions. `CONFOCAL_EXACT`
continues to reject nonconfocal contours; no implicit numerical fallback exists.

## Verification and limits

Tests check the transcribed Yang coefficients, independent reference values,
shape/Re/roughness/assumption guards and parallel flow-area consistency.
They do not tune missing source-table transport properties: Yang's air table
does not report Pr. Friction is checked against the independent laminar
Poiseuille identity, the implicit equation and finest-grid source DNS Fanning
factors (within 3%). The paper's printed modified-Re values are not forced by
altering the exact perimeter.

Numerical wall tests compare circle and confocal limits with their independent
analytical solutions, require finer-mesh agreement, test multiplicity/k/length/
cross-section scale/rotation invariance and reject failed convergence. Public
synthetic normal Rating/Simulation tests check energy closure, resistance
coupling, friction selection, metadata and warnings without private inputs.

These checks establish implementation consistency under the stated models.
They do not constitute experimental exchanger validation. Bare-ellipse outside
bank physics, internal transition/development, inserts, phase change, wet fins
and nonisothermal circumferential contact require separate models.

See the [validation record](validation/elliptical_turbulent_v092.md) for the
focused test command and analytical wall comparisons.
