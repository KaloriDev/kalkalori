# INLINE round solid-fin hydraulics

Implementation/validation record: 2026-10-26, on
`feature/v0.9.4-inline-round-finned`. The completed thermal implementation is
checkpointed separately at `e374329`. The original implementation was released
in v0.9.4 after private engineering workflows and public regression checks passed.

`core.pressure_drop.FrassInlineSolidFinPressureDropProvider` implements Frass
Eq. (191) with native per-row xi and returns whole-bank drag. Its explicit
name is `"frass_inline_solid_fin_191"`.

## v0.9.11 review

INLINE AUTO retains Frass Eq. (191). Both Frass and the independent Genić
constant-fin and authorized tapered-fin providers remain explicitly selectable.
The review does not establish either model's experimental superiority: no
independent pressure dataset with fully matched geometry, reference state,
measurement scope and supported row count was recovered for both providers.

Eq. (191) explicitly includes fin center pitch (equivalently inverse fin
density), fin height and both bank pitches. Fin thickness affects minimum
free-flow area and therefore velocity and Reynolds number; there is no
separate thickness, clear-fin-spacing or extended-surface-area factor.
Equal projected blockage gives the same Frass prediction for constant and
linear-taper fins. This is a limitation of the empirical model, not evidence
that their measured losses are equal.

Genić's separate INLINE fit used six bundles and 91 regimes, with seven rows,
32 mm root diameter, 50 mm fin diameter, 6 mm fin pitch and 1.3 mm thickness.
Its reported 4.9% relative RMS deviation is an in-sample fit statistic.
The scalar limits `6000 < Re_de < 200000`, `0.5 <= Ri <= 2` and at least five
rows do not validate arbitrary fin geometry or a fin-pitch sweep. The primary
publication recommends at least five rows; its INLINE calibration used seven.
See the [Genić guide](genic2006_finned_hydraulics.md) and
[primary publication](https://doi.org/10.1016/j.applthermaleng.2005.07.017).

Independent synthetic fixtures check controlled 2–8 mm fin-pitch changes,
constant/tapered projected area and analytical local sensitivities. At fixed
mass flow, gas properties and other geometry, define
`beta = 2*h*t_mean/(Fp*(Pt-d_A)-2*h*t_mean)`. Frass then has
`d ln(dp_bank)/d ln(Fp) = -0.7 - 1.66*beta`; the velocity exponent is 1.66.
At fixed minimum-passage velocity instead, the pitch exponent is -0.7.
These are numerical/model checks, not experimental accuracy claims.

Frass selection metadata now reports `SOURCE_RANGE_UNRESOLVED`,
`engineering_estimate=True` and `pressure_scope_source_closed=False`, including
constant-fin results within the approximate Reynolds evidence range.
`pressure_scope="BANK_ONLY"` records the existing integration policy;
`pressure_scope_policy` preserves its qualification. This policy must not be
read as source verification of the pressure-tap or heated-flow decomposition.
Thermal equations, hydraulic coefficients and numerical predictions are unchanged.

## Source and conditional release policy

Implementation source: Frass/Hofmann/Ponweiser, *Principles of Finned-Tube
Heat Exchanger Design*, 2nd edition, 2015, pp. 86/88/108/118-119,
Eqs. (139), (144), (190)-(191).
[Legal publisher PDF](https://www.wseas.us/e-library/conferences/2015/Frass/Frass.pdf).
The [hydraulic audit](validation/inline_round_finned_hydraulic_audit_v094.md)
records the evidence and unresolved normalization/loss decomposition.

The user's revised implementation policy explicitly permits qualified
non-strict use despite partial heated-flow and loss-scope provenance. It
supersedes the earlier gate requiring those questions to be source-closed
before any production implementation. That authorization does not resolve
the missing source facts. In particular, the fitted constant's heated/isothermal
state remains unknown. Eq. (143) is **not applied**.

## Equations and physical contract

```text
d_A = bare tube OD                 # ordinary constant-thickness solid fins
h = (D_fin-d_A)/2
G_min = m_dot/A_min
w_E = G_min/rho_mean
nu_mean = mu_mean/rho_mean
Re = w_E*d_A/nu_mean

xi = 4.534*Re**(-0.34)*(Pt/d_A)**(-0.5)*(Fp/d_A)**(-0.7)
     *(h/d_A)**0.5*(Pl/d_A)**0.4

dp_row = xi*rho_mean*w_E**2/2
bank_delta_p = n_rows*Kz*dp_row
```

`Fp` is fin center pitch. `Pt` and `Pl` are independent transverse and
longitudinal center pitches. There is no square-pitch or triangular identity.
One-row hydraulics are rejected. Layout and physical fin-tip clearance are
checked; serrated, segmented and nonround families are unsupported.

| Rows | Kz | Provenance/status |
| --- | --- | --- |
| 2/4/6/8 | `1-0.1*(8-n_rows)/6` | Reported row anchors |
| 3/5/7 | Same analytic equation | `CORRELATION_INTERPOLATION`, dedicated warning |
| >8, non-strict | Exactly 1 | `ENGINEERING_ASYMPTOTIC_CONVENTION`, dedicated warning |
| 1 | Rejected | Unsupported in v0.9.4 |

Eq. (190) is never extrapolated to Kz above unity. Physical row count and Kz
each enter the bank pressure exactly once. The factor-of-four relationship
to Robinson-Briggs is checked only as a diagnostic regression: at matching
bases and Kz=1, equivalent `f_RB=xi/4`. No such conversion is used in production.
The explicit Robinson-Briggs INLINE rejection is unchanged.

## Verification of the live minimum area

The existing geometry gives

```text
blockage = D_root + (D_fin-D_root)*(tf_root+tf_tip)/(2*Fp)
A_min = n_tubes_per_row*length_effective*(Pt-blockage)    # INLINE
```

For constant thickness and ordinary tubes this reduces identically to

```text
A_min = n_tubes_per_row*length_effective*(Pt-d_A-2*h*tf/Fp)
```

Nine live geometry checks cover three fin pitches and three unequal/equal
pitch pairs. No geometry formula or empirical blockage correction was added.
The physical circular-fin overlap guards remain unchanged.

For a continuous root layer, non-strict calculations use gas-facing `D_root`
as equivalent `d_A`, with actual height `(D_fin-D_root)/2`. Both metal OD and
root diameter are reported with `ENGINEERING_CONVENTION` provenance and a
dedicated warning. Strict mode rejects this mapping.

Linear taper retains the existing exact projected geometry for A_min. Eq. (191)
has no empirical thickness term: no mean thickness or altered clear-fin-spacing
term is introduced. Results report actual root/tip thicknesses and an
`UNVALIDATED_GEOMETRY_EXTENSION` warning. Legacy correlation-thickness/spacing
fields are NaN for this provider. Strict mode rejects the taper extension.

## Mean gas state and separate pressure components

The provider is marked `hydraulic_state_basis="mean_outside_gas"`. The bank
adapter evaluates it **once**, at the arithmetic inlet/outlet mean temperature,
using mean-state properties (or explicitly supplied midpoint properties).
It does not Simpson-average Frass drag over three temperatures or use an
enthalpy-derived midpoint. The existing Robinson-Briggs three-state Simpson
calculation is unchanged.

The native result's `bank_delta_p` is assigned directly to bank `dp_drag`.
Under the explicit engineering policy this represents empirical aerodynamic
bank drag. No generic entrance/exit, duct, local or acceleration term is
included in the provider. Existing flow-path aggregation remains responsible
for these separately specified contributions.

The inlet/outlet states still supply the normal signed acceleration term:

```text
dp_acceleration = G_face_out**2/rho_out - G_face_in**2/rho_in
dp_total = bank_delta_p + dp_acceleration
```

This retains cooling pressure recovery as well as heating acceleration.
Tests verify both directions, the isothermal limit and nonzero external local
losses without double counting. The ordinary three-state result structure is
retained for compatibility; endpoint native coefficient/local-drag fields are
NaN because the model is evaluated only at the mean state.

## API, dispatch and diagnostics

```python
from core.pressure_drop import calculate_finned_tube_bank_hydraulics

bank = calculate_finned_tube_bank_hydraulics(
    m_dot, bundle, property_provider=gas_provider,
    temperature_in=T_in, temperature_out=T_out, pressure=p,
    pressure_drop_provider="auto",
)
native = bank.native_result
# native.xi, native.Kz, native.dp_row, native.bank_delta_p
# bank.dp_drag, bank.dp_acceleration, bank.dp_total
```

Automatic round solid-fin hydraulic selection is layout-specific:
INLINE selects Frass; in v0.9.11, conventional staggered geometry selects
source-bounded Genić as described in the [AUTO guide](finned_hydraulic_auto.md).
Robinson–Briggs remains an explicit STAGGERED reference. Explicit provider
guards remain active. Plain-tube and Zhao ellipse physics are unchanged;
their previously inactive default string literals remain accepted.

The shared pressure-result contract now supports both explicitly named bases.
The new provider's coefficient is **native xi**, with basis
`2*delta_p_row/(rho_mean*w_E^2)`. The legacy `f`/`mean_f_over_rho` aliases
return NaN for native results rather than exposing xi as Robinson-Briggs f.
They consult coefficient-basis metadata, preserving legacy custom providers
that use descriptive coefficient labels. Existing provider defaults remain compatible.
Additive request fields carry endpoint temperatures, metal OD and strict-domain
state. Additive diagnostics/defaulted result fields preserve existing request
construction and frozen-result hashing.

`FinnedTubeBankHydraulicResult.native_result` retains the complete native
result. `FinnedTubeDiagnostics.hydraulic_result` preserves it through final
Rating/Simulation diagnostics and diagnostic merging. Reported quantities
include xi, Kz, dp_row, bank pressure, Re, velocity, dynamic pressure, density,
viscosity, source dimensions, mean-state policy, row status, source uncertainty
and engineering geometry mappings.

## Source-domain behavior

Every non-strict result reports:

```text
heated_normalization = NOT_APPLIED
heated_normalization_source_closed = False
loss_scope_source_closed = False
source_range_status = SOURCE_RANGE_UNRESOLVED
```

Warnings include `frass_inline_heated_normalization_unresolved` and
`frass_inline_loss_scope_partial`. They remain visible even for prescribed
isothermal states because the regression normalization itself is unresolved.

Approximate Re 7000-25000 is experimental evidence, not a complete verified
multidimensional domain. Values outside it emit a dedicated evidence
extrapolation warning. Missing diameter, height, fin pitch/thickness, bank pitch,
pitch-ratio and accuracy limits remain unresolved; no bounds are borrowed
from thermal equations or unrelated figures.

Strict mode rejects nonisothermal use, rows above eight, root-layer mapping,
taper, and any other unresolved required source conditions. Ordinary isothermal
geometry still has unresolved required limits/loss provenance, so there is no
claimed fully verified strict-domain success envelope. `hx.strict_source_domain`
reaches both direct solve and the Simulation hydraulic refresh.

## Hydraulic-checkpoint validation

The counts below belong to the completed hydraulic implementation checkpoint.
Later ESCOA STAGGERED and release API tests extend the public collection.
See [final release validation](validation/v094_release_validation.md) for the
complete release test-ID coverage. STAGGERED AUTO thermal remains Briggs–Young;
the [explicit ESCOA STAGGERED alternative](staggered_round_finned_thermal.md)
does not alter Robinson–Briggs hydraulics or its geometry/layout guards.

The final focused thermal/geometry/hydraulic/regression suite passed **314 tests
in 30.20 s**, including all 48 new hydraulic tests. Three additional legacy
wet-coil/steam hydraulic adapter checks passed after final strict-state
propagation. The final provider-contract check passed 146 tests in 17.36 s,
covering custom coefficient labels and both cross-family type guards.
Independent 60-digit Decimal fixtures verify xi,
Kz, dp_row and bank pressure, including unequal pitches and Re evidence points.
The parallel public regression run passed **2,202 tests in 3,535.66 s** using
four workers and work stealing (`-n 4 --dist worksteal core/tests tests -k
"not test_inverse_area_rating_with_default_deadline"`). The two reserved
default-deadline tests passed serially with their original 300-second limits:
**2 passed in 485.02 s**. All **2,204 public test IDs at that checkpoint** therefore have
passing results across the complete partitioned suite and final focused
contract/adapter checks. No tolerances, solver budgets or physics were changed
to obtain those results. The version update followed these checks; source
attribute and setuptools dynamic metadata were checked separately afterward.

Four private Rating/Simulation targets completed using their original notebook
configuration. All used the existing thermal HTC and native Frass hydraulics. The original
`execute_cases` executor also completed all four cases and constructed its
in-memory reports successfully. No notebook inputs,
coefficients or topology were changed and no exports were triggered. Numerical
case diagnostics are retained only in the ignored private worklog. Public
documentation records completion and qualifications without copying case data.

The private diagnostics include low-Re evidence extrapolation, the above-eight
row convention and root-layer/taper engineering warnings where applicable.
These cases verify execution and reporting, not experimental correlation
accuracy or calibration. The normalization and loss-scope questions remain
future source-verification tasks under the accepted conditional policy.
