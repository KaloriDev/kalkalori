# Explicit ESCOA staggered round solid-fin thermal model

`core.heat_transfer.FrassEscoaStaggeredSolidFinProvider` is an explicit thermal
alternative to Briggs–Young for resolved STAGGERED banks of round tubes with
solid individual radial/spiral fins. Select it with
`finned_heat_transfer_provider="frass_escoa_staggered_solid_fin"` in exchanger
thermal, Rating or Simulation calls, or with `provider=` in
`calculate_finned_tube_outside_heat_transfer`.

AUTO remains Briggs–Young for staggered banks and Frass/ESCOA for inline banks.
This addition changes no hydraulic equation or hydraulic layout guard.
The [v0.9.4 release validation](validation/v094_release_validation.md) records
explicit-selector workflow and layout-guard checks for both ESCOA providers.

## Implementation source

The implementation uses the **legal secondary reproduction** in
Frass/Hofmann/Ponweiser (2015), *Principles of Finned-Tube Heat Exchanger
Design*, second edition, printed pp.17–18, Eqs.(60)–(62).
[Publisher full text](https://www.wseas.us/e-library/conferences/2015/Frass/Frass.pdf).
The original ESCOA 1979 manual was not inspected. The source page was visually
checked before implementation, including the superscript on `n_rows²`.

```text
C3 = 0.35 + 0.65 exp(-0.25 h/(Fp-tf))
C5 = 0.7 + (0.7 - 0.8 exp(-0.15 n_rows²)) exp(-Pl/Pt)
Nu = 0.25 Re^0.65 Pr^(1/3) (T_g_mean/T_fin_mean)^0.25
     * (D_fin/d_A)^0.5 C3 C5
h_air = Nu k/d_A
```

`d_A` is the source bare/root diameter; `h=(D_fin-d_A)/2` is radial fin height;
`Fp` is fin centre pitch, `tf` correlation material thickness, and `Fp-tf`
clear spacing. `Pt` and `Pl` are resolved transverse and longitudinal pitches.
`n_rows` counts consecutive physical tube rows in crossflow. No additional
row correction is applied. These are solid-fin equations, not the source's
serrated-fin Eq.(63), INLINE Eqs.(96)–(98), or later TURB-X coefficients.

## Physical output and coupling

The result is physical film HTC before fin efficiency, surface efficiency,
contact/root/fouling resistance and UA. Fin efficiency is applied separately
by the existing thermal resistance network.

The existing provider-neutral circular-finned adapter solves
`h = provider(T_fin_mean(h))`. Its annular-fin/network callback uses the actual
fin conduction and contact/fouling topology. The mean fin temperature is the
area mean over the fin faces and convective tip, in kelvin:

```text
T_fin_mean = T_g_mean + eta_fin (T_base-T_g_mean)
```

It is not wall, root, overall-surface or gas temperature. The adapter retains
its existing relaxation, convergence tolerance and failure behaviour.
Low-level callers may instead prescribe a physically determined mean fin
temperature explicitly. Such results report zero coupling iterations with
`coupling_method="prescribed_mean_fin_temperature"`.

INLINE and STAGGERED keep separate explicit factor equations and metadata.
`_escoa_common.py` shares geometry mapping, supplied-property mechanics,
Nu/result assembly, source-domain policy and diagnostics. INLINE's numerical
equations and evaluation order are preserved.

## Geometry and source qualifications

Only the project's resolved staggered round solid-fin geometry is supported.
INLINE, arbitrary shifted/offset banks, plain tubes, ellipses and serrated,
segmented or louvered fins are rejected. The provider does not impose a
Briggs–Young equilateral identity on the independent ESCOA pitch factors.
That does not establish a verified arbitrary-pitch experimental domain.

For ordinary fins `d_A` is metal tube OD. A continuous root layer may use
gas-facing `D_root` in non-strict calculations as an `ENGINEERING_CONVENTION`.
Diagnostics retain metal tube OD, `D_root`, `D_fin`, the mapping and
`source_geometry_mapping_closed=False`.

For linear taper, only the empirical `h/(Fp-tf)` term uses arithmetic mean
root/tip thickness. Actual taper stays in area, projected blockage and fin
conduction. Actual thicknesses and correlation thickness are diagnosed
separately. This is an unvalidated geometry extension, with warnings.

The original ESCOA property-state convention is not fully source-closed.
Request-supplied properties are used, normally outside mean-bulk properties
in exchanger workflows:

```text
G_reference = m_dot/A_min
u_reference = G_reference/rho
Re = G_reference*d_A/mu
Pr = cp*mu/k
```

`A_min` is the actual resolved staggered minimum free-flow area, including
physical fin blockage and the transverse/diagonal throat comparison. No
INLINE aperture identity is substituted inside the provider.

Missing Re, diameter, height, thickness, fin/tube pitch and accuracy limits
remain `SOURCE_RANGE_UNRESOLVED`. No bounds are borrowed from Briggs–Young,
INLINE, unrelated figures or later revisions. Non-strict calculations warn;
strict source-domain mode rejects unresolved required conditions, including
the root/taper/property mappings. An empty metadata applicability tuple
means missing evidence, not unlimited validity.

## Diagnostics and comparison limits

Results expose provider/source, layout/rows, all source dimensions and actual
thicknesses, areas/velocities, Re/Pr, gas/mean-fin temperatures, temperature and
pitch ratios, C3/C5, Nu/physical HTC, mapping/property/source status, warnings
and coupling iterations/residual. `coupling_residual` aliases the existing
relative constitutive residual for coupled results.

Full Rating/Simulation still requires applicable hydraulics. Selecting this
thermal model does not make Robinson–Briggs accept a square-pitch staggered
bank. Such geometry can use the normal thermal-only solve and qualified
resistance/UA comparisons; a complete hydraulic run must retain its existing
guard. No hydraulic override accompanies this provider.

Independent high-precision fixtures cover rows 1, 2, 4, 8 and 40, several
pitches/heights/Re, and specifically distinguish `n_rows²` from `n_rows`.
These verify equations and integration, not experimental accuracy. A
[public synthetic model comparison](validation/escoa_staggered_comparison_v094.md)
separates production-path eligibility from unresolved source range. Private
ECO inputs and comparison outputs remain outside the public repository.
