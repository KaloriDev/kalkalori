# Optional Genić 2006 solid-fin bank hydraulics

Two explicit public providers implement Genić, Jaćimović and Latinović,
*Research on air pressure drop in helically-finned tube heat exchangers*,
Applied Thermal Engineering 26 (2006), 478–485,
[DOI 10.1016/j.applthermaleng.2005.07.017](https://doi.org/10.1016/j.applthermaleng.2005.07.017).
The user-supplied audit and equation reconstruction close the implemented
equations and definitions. Calibration provenance remains **PARTIAL**, and
original raw calibration data were **not recovered**. Independent legal,
freely accessible reproduction of the complete equation set and its original
definitions has not been established in the public record. The publisher's
article is copyrighted; public availability does not establish an open license.
The implementation contains independently written mathematics, without article
figures, tables or prose. A later [open-access experimental article](https://www.mdpi.com/1996-1073/12/3/556)
reproduces the normal staggered expression, but does not close the full model.
This user-directed optional implementation therefore requires that remaining
public-source reproducibility gap to be considered in review against the
project's source policy.

| Public class (`core.pressure_drop`) | Explicit selector | Equation |
| --- | --- | --- |
| `Genic2006StaggeredPressureDropProvider` | `genic2006_staggered` | Normal staggered Eq.(10) |
| `Genic2006InlinePressureDropProvider` | `genic2006_inline` | Inline Eq.(12) |

Use `provider=` in `calculate_finned_tube_pressure_drop`, or
`pressure_drop_provider=` in `calculate_finned_tube_bank_hydraulics`.
Outside dispatch, Rating and Simulation accept
`finned_pressure_drop_provider="genic2006_inline"` (or the staggered selector).
Provider instances are also accepted. AUTO remains Robinson–Briggs for
STAGGERED and Frass Eq.191 for INLINE. Thermal selection is unchanged.

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
non-strict mode and raise in strict mode. Wrong layout/family, taper,
invalid geometry/area, dense packing and short banks always raise. There is
no automatic fallback. `SOURCE_DOMAIN_PASS` in diagnostics means the
**audited explicit equation limits** pass; it does not establish a complete
multidimensional accuracy envelope or calibration coverage.

Constant-thickness fins are required. A gas-facing root layer can use
`D_root` in non-strict mode with an explicit
`genic2006_root_diameter_engineering_mapping` warning; both metal OD and
root diameter are retained. Strict mode rejects that mapping. No public
mean-thickness surrogate for tapered fins is provided.

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
geometry additionally requires a private, explicitly unvalidated surrogate;
the public provider continues to reject it. Complete private tables, domain
labels and operating data remain outside public documentation and fixtures.
