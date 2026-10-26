# INLINE round solid-finned thermal support (v0.9.4)

Package version is **0.9.4**. INLINE AUTO combines this ESCOA thermal model
with [native Frass Eq.191 hydraulics](inline_round_finned_hydraulics.md), under
the qualified non-strict policy. STAGGERED AUTO remains Briggs–Young;
[ESCOA STAGGERED](staggered_round_finned_thermal.md) is an explicit alternative.
Source-range, property and geometry qualifications remain visible, and strict
mode rejects unresolved required conditions. See the
[release integration validation](validation/v094_release_validation.md).

The implementation began with the separate thermal checkpoint `e374329` on
2026-10-25. The final section preserves its historical validation record,
including the hydraulic blocker that existed then. Native Frass hydraulics
subsequently resolved that implementation blocker under the revised policy;
the underlying source uncertainties remain. The thermal equations are unchanged.

## Provider and provenance

`core.heat_transfer.FrassEscoaInlineSolidFinProvider` implements the old ESCOA
1979 solid-fin INLINE thermal model from its complete algebraic reproduction
in Frass/Hofmann/Ponweiser, *Principles of Finned-Tube Heat Exchanger Design*,
2nd edition, 2015, p. 39 Eqs. (96)-(98), with temperature definitions from
p. 18 Eq. (60). [Legal publisher full text](https://www.wseas.us/e-library/conferences/2015/Frass/Frass.pdf).
The primary 1979 manual was not inspected. The provider's name and metadata
identify the secondary implementation source.

The [forensic reconstruction](validation/inline_round_finned_source_audit_v094.md)
records the citation chain, ITE evidence and unresolved quantities. Eq. (100)
remains a reference/future candidate; it is not implemented as the default or
combined with an ESCOA row multiplier. No TURB-X 1993 coefficients are imported.

## Exact expression and physical output

```text
C3 = 0.20 + 0.65 exp(-0.25*h/(Fp-tf))
C5 = 1.1 - (0.75 - 1.5 exp(-0.70*n_rows)) exp(-2*Pl/Pt)
Nu = 0.25 Re^0.65 Pr^(1/3) (T_g_mean/T_fin_mean)^0.25
     * ((d_A+2*h)/d_A)^0.5 C3 C5
h_air = Nu*lambda_gas/d_A
```

In the source, `d_A` is bare tube OD; `h` radial fin height; `Fp` center pitch; `tf` fin
thickness; `Pt`/`Pl` transverse/longitudinal center pitches. Temperatures are
absolute kelvin. The `0.20` is the **INLINE SOLID** constant. `n_rows` is the
number of consecutive physical rows, not a current-row index. No additional
first-row, asymptotic, Stasiulevicius or Eq. (100) correction is applied.

The returned `alpha`/`h` is physical film HTC. The physical sequence remains

```text
physical film h -> annular-fin efficiency -> overall surface efficiency
                -> outside resistance -> exchanger UA
```

Fin conduction is used inside the temperature coupling to determine the mean
fin temperature. It does not reduce the provider's physical HTC or multiply
surface efficiency into that coefficient. The resistance network applies
surface efficiency exactly once.

## Mean fin temperature and convergence

The annular-fin model uses uniform film HTC on the actual two faces and the
convective tip. With `theta=T-T_gas`:

```text
q_fin = h_air * integral(theta, dA)
      = eta_fin * h_air * A_fin * (T_base-T_gas)
T_fin_mean = T_gas + eta_fin*(T_base-T_gas)
```

Thus `FinEfficiencyResult.mean_fin_temperature(base_temperature, fluid_temperature)`
is the **fin-only** area mean, including the tip. It uses neither overall
surface efficiency nor any gross-area override. This identity follows the
existing fin model; it does not assume an isothermal fin at the root or gas
temperature. Equal root/gas temperatures give that common temperature exactly.

`make_mean_fin_temperature_model(...)` in `outside_dispatch` returns a generic
`h -> T_fin_mean` callback. For each trial HTC it builds the existing full
resistance network, computes the signed heat rate, and obtains the fin-base
node from `fin_surface_temperatures`. Outside fouling, fin-only contact and
common-root/contact topology use the same node calculations as wall iteration.
The original import path for `thermal_iteration.fin_surface_temperatures`
remains usable through an import alias.

The circular-finned adapter iterates `h = provider(T_fin_mean(h))` with 0.5
relaxation. Its starting value, 100 W/(m2 K), is a numerical guess, not an
empirical coefficient. A result requires relative constitutive residual
`abs(h_new-h_old)/max(h_new,h_old) <= 1e-10`, within 100 iterations. Failure
raises an error; no last-iterate or temperature substitution is returned.
Diagnostics retain iterations, residual and convergence. The existing outer
bulk/wall iteration continues to update transport properties and inside HTC.

## Property convention and source domain

The exact 1979 property-state instruction remains unresolved. The implementation
reuses request-supplied `rho`, `mu`, `cp` and `k`; normal exchanger workflows
supply outside mean-bulk properties:

```text
G_reference = m_dot/A_min
u_reference = G_reference/rho_supplied
Re = G_reference*d_A/mu_supplied
Pr = cp_supplied*mu_supplied/k_supplied
```

This is explicitly an **ENGINEERING_CONVENTION**, not source-exact closure.
`property_convention_provenance="ENGINEERING_CONVENTION"` and
`source_property_convention_closed=False` are returned with a structured warning.
The source temperature-ratio term remains present; this property convention
does not replace mean fin temperature with a bulk/root temperature.

Re, diameter, height, thickness, fin pitch, both bank pitches/pitch ratio and
accuracy/deviation limits are unresolved. Every non-strict result carries
`SOURCE_RANGE_UNRESOLVED` and a list of unresolved dimensions. An empty
`applicability` tuple is **not** a positive in-domain assertion. No Eq. (100),
TURB-X or unrelated staggered limits are borrowed.

`strict_source_domain=True` rejects because compliance with the required source
limits/property convention cannot be established. It does not convert unknown
bounds into unrestricted acceptance. `hx.strict_source_domain` is threaded to
both the iterative thermal state and the fixed-property solve adapter.

## Geometry and selection

Automatic circular-finned thermal selection now defaults to `"auto"`:

| Geometry/layout | Thermal selection |
| --- | --- |
| Round, individual solid fins, INLINE | `FrassEscoaInlineSolidFinProvider` |
| Round, individual solid fins, staggered equilateral triangular | `BriggsYoung1963Provider` |
| Explicit ESCOA STAGGERED on supported staggered round solid fins | `FrassEscoaStaggeredSolidFinProvider`; source ranges unresolved |
| Explicit Briggs-Young with INLINE | Rejected as before |
| Explicit ESCOA INLINE with staggered/shifted | Rejected |
| Serrated/segmented/louvered/nonround families | Rejected by the applicable family guard |

The explicit built-in name is `"frass_escoa_inline_solid_fin"`. Low-level
requests prescribe `gas_temperature` and `mean_fin_temperature`; the bundle
adapter also accepts `fin_temperature_model`. Exchanger thermal iteration and
fixed-property solve construct that callback from actual inside/outside states.
Calls lacking the necessary temperature state fail explicitly.

The implementation accepts constant-thickness fins and the existing linear-taper
geometry with an explicitly qualified empirical thickness mapping. For a continuous root layer
over a smaller metal core, the user approved an explicit non-strict engineering
mapping: the gas-facing root cylinder `D_root` becomes the equivalent source
`d_A`. Actual radial height remains `(D_fin-D_root)/2`, preserving the source's
identity `D_fin=d_A+2h`. The metal core OD remains separate in wall/root/contact
conduction; the physical geometry and areas are unchanged. Results show both
diameters, `diameter_mapping_provenance="ENGINEERING_CONVENTION"`,
`source_geometry_mapping_closed=False`, and a dedicated
`escoa_root_diameter_engineering_mapping` warning. Strict mode still rejects
the unresolved domain/mapping. This does not establish experimental validation
of root-layer tubes.

For a linear taper, empirical `tf=(tf_root+tf_tip)/2` and clear spacing
`Fp-tf` follow the existing circular-finned correlation convention. This mapping
is applied only to the empirical terms. Geometry, exact projected blockage,
physical surface area and the finite-volume conduction model retain the real
root/tip thicknesses. Both thicknesses and the mapped value are reported, with
`thickness_mapping_provenance="ENGINEERING_CONVENTION"` and the dedicated
`escoa_tapered_fin_engineering_mapping` warning. This does not claim source-exact
ESCOA validation of tapered fins; strict mode continues to reject.

Positive `Fp-tf`, physical transverse clearance,
positive longitudinal pitch and integer `n_rows>=1` are required. Existing
bundle geometry/fin-tip clearance rules remain unchanged. Independent `Pt`
and `Pl` are retained; no square or triangular identity is imposed on INLINE.

A one-row request preserves the source caveat: the two layout expressions can
disagree (about 15% at small height/clear spacing), and a longitudinal-pitch
dependence remains although one row has no physical longitudinal pitch.
Computability is not experimental single-row validation.

## Diagnostics and compatibility

`FinnedTubeHeatTransferResult.diagnostics` includes provider, provenance,
layout/rows, all source dimensions, `Pl/Pt`, reference velocity, Re/Pr,
gas/mean-fin temperatures, ratio, C3/C5, Nu/physical HTC, property convention,
source-range status, warnings and coupled-solve convergence. Nonapplicable
triangular-pitch result fields are NaN for INLINE.

`FinnedTubeDiagnostics.heat_transfer_result` retains that complete thermal
result through workflow diagnostics. The request's additive temperature,
diameter and strict-domain fields and the result's additive diagnostics field
have defaults; existing custom providers remain compatible. Briggs-Young
equation coefficients, source bounds, row warnings and explicit INLINE guard
are unchanged. Its existing numerical fixtures remain authoritative.

## Historical thermal-checkpoint validation and engineering-case progress

The results below describe the original thermal-only checkpoint, when package
version was 0.9.3 and INLINE hydraulics had not been implemented. They are not
the current release status. The [release validation](validation/v094_release_validation.md)
records the final public suite and completed original INLINE workflows.

Focused command:

```text
py -3.11 -m pytest -q --tb=short core/tests/escoa_inline_test.py core/tests/finned_tube_correlations_test.py core/tests/circular_finned_tube_geometry_test.py core/tests/circular_fin_efficiency_test.py core/tests/circular_finned_tube_integration_test.py core/tests/finned_tube_dry_release_regression_test.py core/tests/pressure_drop_flow_path_test.py core/tests/fin_contact_efficiency_test.py core/tests/elliptical_automatic_models_test.py
```

The final focused run passed **266 tests in 29.93 s**, including 43 new ESCOA
tests. Independent 60-digit
decimal fixtures cover C3/C5/Nu/HTC at rows 1/2/4/8 and pitch ratios .75/1/1.25.
A separately calculated SciPy Bessel/Brent fixture checks the coupled fin/network
solution (no new SciPy runtime/test dependency). Coupling checks include both
heat directions, equal temperatures, contact, fouling, area-override invariance
and nonconvergence, plus explicit root-layer and linear-taper mappings. Synthetic thermal-state, Rating and Simulation checks show
thermal closure and the unchanged hydraulic stop.

The full public suite was run with four pytest-xdist workers and work stealing
(`-n 4 --dist worksteal core/tests tests`): **2,152 passed; 2 failed in
2,702.08 s**. Both failures were the existing parametrizations of
`test_inverse_area_rating_with_default_deadline`, each hitting its unchanged
300-second timeout. Their isolated serial rerun passed **2 tests in 443.37 s**,
with the original deadlines and numerical assertions unchanged. This supports
parallel CPU contention as the explanation; no solver budget or physics was
changed. The earlier serial full-suite attempt was interrupted before this
parallel run. pytest-xdist 3.8.0 and execnet 2.1.2 were installed only into a
temporary tools directory, with no project dependency change.

Two ESCOA root-layer/taper tests and the diagnostics hash-compatibility check
were finalized during that long run and are covered by the final 266-test
focused run. Matching the combined passing JUnit results against the final
collection confirms **all 2,156 current test IDs have a passing result** across
the full run, final focused run and isolated deadline rerun. This is combined
coverage, not a claim that one final full-suite invocation passed cleanly.
Existing Briggs-Young equations/warnings, annular-fin solver functions and
finned pressure-drop modules were also checked unchanged against the branch
baseline; `git diff --check` passed.

All four requested private ECO cases were rerun from the original notebook
configuration, without editing/exporting that notebook or copying its inputs
into public files:

| Case | Stage | Reached result |
| --- | --- | --- |
| eco_nominal | ECO_LT_B1 | Gas-facing root-diameter and arithmetic-mean thickness engineering mappings; ESCOA physical HTC computed; stops at Robinson-Briggs INLINE hydraulic rejection |
| eco_max | ECO_LT_B1 | Same |
| eco_nominal | ECO_LT_B2 | ESCOA physical HTC computed; stops at Robinson-Briggs INLINE hydraulic rejection |
| eco_max | ECO_LT_B2 | Same |

The private local validation record retains the numerical thermal diagnostics;
public documentation records only progress/blocker status. There is no completed
INLINE bank pressure-drop or complete exchanger result. Hydraulics were not
changed, bypassed with staggered geometry, or substituted with a smooth-tube
model. Eq. (191)'s loss-scope/heated normalization audit remains open, so no
native xi/f_RB integration was attempted.
