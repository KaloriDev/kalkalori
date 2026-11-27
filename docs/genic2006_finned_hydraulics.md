# Genić 2006 solid-fin bank hydraulics

Two explicit public providers implement Genić, Jaćimović and Latinović,
*Research on air pressure drop in helically-finned tube heat exchangers*,
Applied Thermal Engineering 26 (2006), 478–485,
[DOI 10.1016/j.applthermaleng.2005.07.017](https://doi.org/10.1016/j.applthermaleng.2005.07.017).
v0.9.11 verifies Eqs.(1)–(6), (8)–(10), (12) and (15), coefficient
normalization, explicit limits and the calibration summary against the
[primary-publication text](https://www.academia.edu/70435875/Research_on_air_pressure_drop_in_helically_finned_tube_heat_exchangers).
Calibration provenance remains **PARTIAL**, and original raw calibration data
were **not recovered**. This closes the previous equation-verification gap;
it does not establish a complete joint accuracy envelope. The copyrighted
article's availability is not an open implementation license. Source and
geometry kernels are independently written mathematics under the project's
GPL terms; no external implementation, article figures, tables or prose are
copied into the package.

| Public class (`core.pressure_drop`) | Explicit selector | Equation |
| --- | --- | --- |
| `Genic2006StaggeredPressureDropProvider` | `genic2006_staggered` | Normal staggered Eq.(10) |
| `Genic2006InlinePressureDropProvider` | `genic2006_inline` | Inline Eq.(12) |
| `Genic2006StaggeredPressureDropProvider(allow_linear_taper=True)` | `genic2006_staggered_tapered` | Eq.(10), unvalidated geometry extension |
| `Genic2006InlinePressureDropProvider(allow_linear_taper=True)` | `genic2006_inline_tapered` | Eq.(12), unvalidated geometry extension |

Use `provider=` in `calculate_finned_tube_pressure_drop`, or
`pressure_drop_provider=` in `calculate_finned_tube_bank_hydraulics`.
Outside dispatch, Rating and Simulation accept
`finned_pressure_drop_provider="genic2006_inline"` (or the staggered selector).
Provider instances are also accepted. From v0.9.11, STAGGERED hydraulic AUTO
selects Genić within its supported scalar domain; INLINE AUTO remains Frass
Eq.191. See the [exact AUTO and authorization rules](finned_hydraulic_auto.md).
Thermal selection is unchanged.

## Geometry and native coefficient basis

For one fin-pitch segment, let `d=D_root`, `D=D_fin`, `sf=fin_pitch`,
`delta=constant fin thickness`, `Pt=transverse pitch`, `Pl=longitudinal pitch`.
All dimensions use metres. The implementation evaluates:

```text
sd = sqrt(Pl² + (Pt/2)²)
af = (Pt-d)(sf-delta) + delta(Pt-D)
ad = 2[(sd-d)(sf-delta) + delta(sd-D)]
asurf = pi*d*(sf-delta) + pi/2*(D²-d²) + pi*D*delta
Rd = ad/af; Ri = (Pl-d)/(Pt-d); W = asurf/af
epsilon = 1 - pi*[d²(sf-delta)+D²delta]/(4*Pt*Pl*sf)
Sv = asurf/(Pt*Pl*sf)
de = 4*epsilon/Sv
w_min = m_dot/(rho*A_min)
Re_de = rho*w_min*de/mu
```

For constant thickness the existing projected throat is exactly `af/sf`.
Because `A_face=n_tubes_per_row*L*Pt`, its bank area equals
`A_face*af/(Pt*sf)`. For normal staggered banks, `Rd>=1` makes that frontal
passage controlling. Each evaluation checks the supplied area against this
identity; a mismatch raises an error. No area correction is fitted.

The staggered and inline coefficients are respectively:

```text
xi_S = (0.43 + 17.7/Re_de^0.3)*W^-0.8*Rd^0.3
xi_I = (0.4 + 8.5/Re_de^0.3)*W^-0.75*Ri^0.7
bank_delta_p = n_rows*xi*(Pl/de)*rho*w_min²/2          [Eq.(15)]
```

`result.xi` is native Genić xi, whose basis is
`2*delta_p*de/(n_rows*Pl*rho*w_min²)`. It is neither Frass's per-row xi nor
Robinson–Briggs f. `result.f`, hydraulic-point `f` and `mean_f_over_rho`
return NaN. Diagnostics expose `zeta_row=xi*Pl/de`, `dp_row` and whole-bank
drag. No factor-of-four conversion enters this model.

## Applicability and engineering extensions

Both providers require at least five physical rows. Fewer rows are rejected
in all modes: no unclosed short-bank correction is approximated.
Eq.(10) always rejects `Rd<1`; the separate dense-packing Eq.(11) is absent.
It checks `Pt/d<=3.5`, `Pl/d<=3.5` and the open Reynolds interval
`550<Re_de<130000`; the cited open-access staggered reproduction reports
these bounds. Eq.(12) checks inclusive
`0.5<=Ri<=2` and the open interval `6000<Re_de<200000`.
Floating-point boundary comparisons allow only numerical roundoff (relative
`1e-14`); open Reynolds endpoints remain excluded.

Pitch/Ri/Re violations produce `genic2006_source_domain_extrapolation` in
non-strict mode and raise in strict mode. Wrong layout/family, unauthorized taper,
invalid geometry/area, dense packing and short banks always raise. There is
no automatic fallback. `SOURCE_DOMAIN_PASS` in diagnostics means the
**audited explicit equation limits** pass; it does not establish a complete
multidimensional accuracy envelope or calibration coverage.

The original selectors require constant-thickness fins. A gas-facing root layer can use
`D_root` in non-strict mode with an explicit
`genic2006_root_diameter_engineering_mapping` warning; both metal OD and
root diameter are retained. Strict mode rejects that mapping.

### Explicit linear taper geometry extension

Select `genic2006_staggered_tapered` or `genic2006_inline_tapered`, or pass
the corresponding class with `allow_linear_taper=True`. Actual taper also
requires `allow_engineering_estimates=True` in low-level functions or
`OutsideProviderPolicy(allow_engineering_estimates=True)` on the exchanger.
This is an explicit
engineering choice, not a source-validated tapered-fin correlation.
`strict_source_domain=True` rejects actual taper even when scalar equation
limits pass. Constant fins on the opt-in provider retain exactly the original
arithmetic, method name and result.

For root/tip thicknesses `tr`, `tt`, height `H=(D-d)/2`, and one pitch `sf`,
linear thickness gives these independently derived geometric quantities:

```text
t_projected = (tr+tt)/2
af = (Pt-d)*sf - (D-d)*t_projected
ad = 2*((sd-d)*sf - (D-d)*t_projected)
slope = sqrt(1+((tr-tt)/(D-d))^2)
A_primary = pi*d*(sf-tr)
A_fin = pi/2*(D^2-d^2)*slope + pi*D*tt
V_fin = 2*pi*((d/2)*H*(tr+tt)/2 + H^2*(tr+2*tt)/6)
V_metal = pi*d^2*sf/4 + V_fin
V_cell = Pt*Pl*sf
asurf = A_primary+A_fin
epsilon = 1-V_metal/V_cell; Sv = asurf/V_cell
```

The projected mean thickness is used only for blockage. Primary exposure uses
root thickness, the tip surface uses tip thickness, and metal volume integrates
the radial thickness profile. This is not a constant mean-thickness substitute.
The unchanged fitted equations receive `de=4*epsilon/Sv`, `W=asurf/af` and
`Rd=ad/af`. This reuse of empirical fits for taper is unvalidated.

Diagnostics expose both thicknesses, both clear spacings, surface and volume
components, `geometry_mapping`, `source_geometry_mapping_closed=False`, and
`prediction_status="UNVALIDATED_ENGINEERING_EXTENSION"`. The warning is
`genic2006_linear_taper_geometry_extension`; scalar source-domain PASS alone
does not validate this extension. No single correlation thickness or spacing
exists for this mapping, so those legacy result slots are NaN. No fin
conduction or thermal formula changes accompany hydraulic selection.

The bank adapter evaluates native Eq.(15) once at the arithmetic mean
outside-gas temperature (or explicitly supplied midpoint properties).
Endpoint properties serve the existing separate acceleration calculation.
This choice is an engineering extension for nonisothermal operation,
reported as `genic2006_mean_state_nonisothermal_extension` and rejected by
strict mode. No heated-flow multiplier, external duct loss or generic
entrance/exit loss enters the provider. Non-air use also warns and is
rejected in strict mode. These qualifications are separate from the
audited scalar equation limits.

## Calibration coverage

The primary publication reports 13 normal staggered bundles / 189 regimes for
Eq.(10). Its Table 1 lists four tabulated Kays–London bundles and eight Yudin
bundles, which do not fully reconcile with that stated fitting count. Nine
diagram-only bundles are comparison evidence. Diagnostics retain the published
fitting count; original data are needed to reconcile this discrepancy and
recover the joint test matrix. None of these descriptors establishes tapered-fin
calibration.

INLINE diagnostics identify Yudin 1982 via Genić: six bundles and 91 regimes,
with recovered geometry `d=32 mm`, `D=50 mm`, `sf=6 mm`, `delta=1.3 mm`,
and seven rows. These are calibration descriptors, **not hard source bounds**.
Geometry or rows outside that recovered family emit
`genic2006_inline_calibration_geometry_extrapolation`, including in strict
mode when the equation limits otherwise pass. Equality is checked only to
floating-point tolerance; no percentage band for acceptable accuracy is
invented. The actual-to-calibration ratios are exposed for review.

## Validation

Independent high-precision fixtures verify the audited reference case,
including `de`, Reynolds number, xi, per-row equivalent loss and Eq.(15).
The diagnostic reference produces 353.986049 Pa staggered and 175.455772 Pa
inline (ratio 0.495657). This verifies arithmetic, not industrial accuracy.
Additional tests cover source boundaries, hard exclusions, exact live area
equivalence, separate acceleration, native adapters and public workflows.
The focused and adjacent hydraulic/flow-path/geometry regression run passes
**488 tests**, including **59 new Genić cases** and existing Frass,
Robinson–Briggs, thermal, layout and release tests. An additional 300 legacy
direct/three-state hydraulic API comparisons match the preceding commit
exactly; 503 existing INLINE direct/coupled thermal comparisons remain
identical. The final complete public regression after the optional-provider
change passed **2,328 tests**, with no failures or skips; see the
[release integration record](validation/v094_release_validation.md).

Four original private ECO workflows complete with unchanged input hashes.
Their Genić INLINE Reynolds numbers fall below the stated source interval,
so their numerical layout comparisons are diagnostic extrapolations. Tapered
geometry in v0.9.11 instead has the explicit, independently derived geometry
extension above. Complete engineering tables, domain labels and operating
data remain outside public documentation and fixtures. The
[v0.9.11 comparison](validation/v0.9.11_extended_surface_hydraulics.md) uses
independent synthetic geometry and distinguishes arithmetic verification from
experimental validation.
