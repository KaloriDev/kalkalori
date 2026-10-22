# Chakroun–Al-Fahed 1995 twisted tape (v0.9.3)

`ChakrounAlFahed1995TwistedTapeProvider(strict_source_domain=False)` is an
explicitly selected public GPLv3 implementation of paired laminar liquid
heat transfer and distributed friction. It shares the existing enhancement,
coupled wall-temperature, Rating and Simulation paths. Other providers and
automatic/default selection retain their existing behavior.

## Authority and provenance

W. M. Chakroun and S. F. Al-Fahed, *The Effect of Twisted Tape Width on Heat
Transfer and Pressure Drop for Fully Developed Laminar Flow*, ASME 95-GT-042
(1995), [DOI 10.1115/95-GT-042](https://doi.org/10.1115/95-GT-042), Eqs.16–17
and the explicit area/velocity definitions provide the equation authority.
The 1995 source was audited before this implementation. The later journal
publication in *Journal of Engineering for Gas Turbines and Power* (1996),
[DOI 10.1115/1.2816688](https://doi.org/10.1115/1.2816688), is bibliographic
context only; its full equations have not been independently audited here.

This release follows explicit authorization to implement the audited model
publicly, a scoped exception to the repository's free-source-only policy.
Free full-text access to the primary paper is not asserted. The contract's
`source_access_basis="open"` identifies the public independent implementation;
it does not assert an open-access license for the publication. Only original
code, documentation and synthetic equation tests are distributed. There is
no runtime or test dependency on the archived research branch, private
providers, named-oil property tables, reference-software outputs or PDFs.

## Geometry and equations

All inputs use SI units. `D` is bore diameter, `W` actual tape width, `delta`
tape thickness, and `H180` the **axial length of a half turn**, not a full turn.

```text
Y = H180/D
A0 = pi*D²/4
Ac = A0 - delta*W
phi = A0/Ac
s = sqrt(1 + (pi/(2*Y))²)
V0 = m_dot_tube/(rho_bulk*A0)
Va = m_dot_tube/(rho_bulk*Ac)
Vs = Va*s
Re_empty = rho_bulk*V0*D/mu_bulk
Re_sw = rho_bulk*Vs*D/mu_bulk
Sw = Re_sw/sqrt(Y)
Pr_bulk = cp_bulk*mu_bulk/k_bulk

Nu = 0.318*Sw^0.569*Pr_bulk^0.3*(mu_bulk/mu_wall)^0.14   [Eq.17]
h = Nu*k_bulk/D

(f_Fanning*Re)_sw =                                      [Eq.16]
  15.767*((pi+2-2*delta/D)/(pi-4*delta/D))²
  *(1+10^-6*Sw^2.55)^(1/6)
f_Darcy_swirl = 4*f_Fanning_swirl
```

Eq.17 uses bulk rho, cp, mu, k and Pr; only wall viscosity comes from the
coupled wall state. Standalone requests must supply that state. No film
Pr/k, second viscosity correction, thermal entrance multiplier, tape fin
area, leakage multiplier or width-dependent hydraulic diameter is added.
Hydraulic nodes return friction without evaluating Nu or h.

The prediction uses **actual `Ac(W)`**. The printed expanded Eq.11 instead
implies `phi_nominal=A0/(A0-delta*D)` and
`Sw_printed_nominal=Re_empty*phi_nominal*s/sqrt(Y)`. That value is exposed
only as a diagnostic alongside `Sw_actual_area`. There is no selector,
averaging or reference-based calibration. At `W=D` the readings coincide.
Eq.16's nominal geometric factor is retained as printed, without inventing
a width-dependent perimeter correction.

## Hydraulic convention

Native Eq.16 friction is **Fanning**. The coefficient 15.767 is unchanged;
conversion to Darcy happens exactly once. Source pressure drop uses
`Vs`, `Ls=L*s` and bore `D`:

```text
dp = f_Darcy_swirl*(L*s/D)*rho_bulk*Vs²/2
   = f_Darcy_axial_equivalent*(L/D)*rho_bulk*V0²/2
f_Darcy_axial_equivalent = 4*f_Fanning_swirl*phi²*s³
```

`EnhancementResult.f_darcy` and `.reference` use the second, empty-bore axial
basis, so the solver uses its physical axial hydraulic length unchanged.
`friction_factor_native` is swirl Fanning; `friction_basis="fanning"` and
`friction_normalization=phi²*s³` preserve both conversions explicitly in the
existing contract. `result.f_fanning` is the equivalent axial Fanning factor;
the diagnostic `f_Fanning_swirl` is the source-native value. Neither helical
length nor the factor four is applied again. Insert-specific local losses
are outside this distributed model; existing solver local losses remain
separate. Tests verify both dp expressions for multiple lengths/geometries.

## Applicability

| Quantity | Source evidence and runtime treatment |
| --- | --- |
| Fluid/regime | Shell Tellus R5 oil; fully developed laminar liquid flow. Gas and phase change unsupported. Other liquid properties do not establish direct experimental validation. |
| Empty-bore Re | 230–2300 inclusive; lower positive Re calculates with `chakroun1995_re_below_source_range`. Re above 2300 always raises, including permissive mode; no transition/turbulent continuation. The reported source endpoint 2300 is retained at the engine's regime boundary. |
| Twist | `Y=H180/D`, 3.6–7.1; outside warns `chakroun1995_twist_ratio_extrapolation`. |
| Width | `W/D=10.8/14..13.2/14` (about 0.771–0.943); outside warns `chakroun1995_width_ratio_extrapolation`. Width above D is invalid. |
| Clearance | Nominal centered edge clearance `(D-W)/2`; ratio `(D-W)/(2D)`. No eccentricity/bypass model. |
| Thickness | Single tested `delta/D=0.5/14`; **no source thickness interval**. More than 1% relative difference triggers `chakroun1995_thickness_ratio_extrapolation`. The 1% is an implementation reporting tolerance, not a validated source range. |
| Diameter | Source D=14 mm; use at other sizes is a similarity assumption, not experimentally proven size independence. |
| Pr | Source range **unknown**; actual `Pr_bulk` is exposed, with no invented rejection limits. Strict mode cannot certify Pr coverage. |
| Heat direction | Heating demonstrated. Cooling calculates with `chakroun1995_cooling_extrapolation`; the unchanged viscosity ratio normally lowers Nu. Cooling is rejected in strict mode. |
| Wall boundary | Approximately constant wall temperature source; always reports informational `chakroun1995_constant_wall_temperature_source`. A coupled 0D wall does not reproduce that boundary exactly and is not rejected solely for this difference. |

`strict_source_domain=True` rejects the listed Re, Y, width and material
thickness extrapolations and cooling. Permissive mode marks their results
`applicability="extrapolated"`; informational source-scope warnings alone
do not change that flag. Strict mode is a check of the known source bounds,
not a claim of validation for every fluid, Pr, size or boundary condition.

The experiments showed additional width/leakage behavior not fully resolved
by the final correlation. Width dependence here is limited to
`W -> Ac -> Va -> Vs -> Re_sw -> Sw`. This provider is **not a universal
arbitrary-clearance correlation**. Entrance effects, tape conduction and
arbitrary roughness corrections are excluded.

## Public usage

```python
from core.enhancements import (
    ChakrounAlFahed1995TwistedTapeProvider,
    TubeSideEnhancement, TwistedTapeGeometry,
)
from core.models.bare_tube import BareTubeHeatExchanger

# bundle must have a circular bore D_i=0.021 m.
enhancement = TubeSideEnhancement(
    provider=ChakrounAlFahed1995TwistedTapeProvider(),
    geometry=TwistedTapeGeometry(
        half_turn_length=0.076, tape_width=0.019, tape_thickness=0.0009,
    ),
    fluid_phase="liquid",
)
hx = BareTubeHeatExchanger(bundle, tube_side_enhancement=enhancement)
simulation = hx.simulate(inside, outside)
rating = hx.rate(inside_balance, outside_balance, include_simulation=True)
```

The synthetic target has `W/D=0.9047619`, `clearance/D=0.0476190`,
`delta/D=0.0428571`, `Y=3.6190476`. Its thickness ratio is 20% above the
single source value and calculates with a warning. Public tests build
original, temperature-dependent high-Pr liquid states and verify heating
and cooling, wall viscosity, Rating, Simulation, hydraulic quadrature and
warning propagation. These are software/equation checks, not experimental
cooling validation.

`result.tube_side_enhancement` exposes provider identity, provenance,
applicability, warnings and scalar diagnostics. Diagnostics include Re_empty,
Re_sw, Pr_bulk, Y, W_over_D, clearance_over_D, delta_over_D, Ac, phi,
helical_factor, both Sw readings, the three friction factors, mu_bulk,
mu_wall, mu_bulk_over_wall and thermal Nu/h when evaluated. The three
hydraulic property points each expose their own paired provider result.
