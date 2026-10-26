# INLINE hydraulic integration audit after the thermal checkpoint

Audit date: 2026-10-26. Branch: `feature/v0.9.4-inline-round-finned`.
Thermal checkpoint: `e374329 feat: add inline round solid-finned thermal model`.
The working tree was clean immediately after that commit. Package version was
0.9.3 at that historical checkpoint; current release integration uses 0.9.4.
At the initial audit stage this document contained hydraulic
investigation only, without provider/dispatch/loss/release changes. The revised
policy and subsequent implementation are distinguished below. See the
[final release validation](v094_release_validation.md) for current completion
and regression status; the source findings below remain qualified.

## Current decision under the revised user policy

The user subsequently superseded the earlier production gate with an explicit
conditional implementation policy. The [native-xi implementation guide](../inline_round_finned_hydraulics.md)
describes the resulting provider: evaluate Eq.191 once at mean gas state,
apply Eq.190 for 2..8 rows, use Kz=1 above eight only as an engineering
convention, leave Eq.143 unapplied, and keep acceleration/external losses separate.
Non-strict empirical bank-drag use is authorized with structured provenance
warnings. Strict mode rejects unresolved required conditions. Neither this
authorization nor implementation changes the source findings below.

```text
INLINE ROUND SOLID-FIN HYDRAULIC MODEL = FRASS EQ.191
PUBLIC IMPLEMENTATION POSSIBLE = CONDITIONAL
RECOMMENDED PROVIDER ARCHITECTURE = NATIVE XI-BASED
NATIVE XI -> KALKALORI DP MAPPING = CLOSED
HEATED PRESSURE-DROP TREATMENT = PARTIAL
LOSS SCOPE = PARTIAL
READY FOR v0.9.4 IMPLEMENTATION = CONDITIONAL
```

## Historical implementation decision before that policy change

**Production integration remains blocked by the earlier source-audit condition.**
The thermal implementation request, section 14, requires closure of native
normalization, velocity, row multiplication, heated/isothermal normalization
and irreversible bank-loss scope before promoting Frass Eq. (191).
The coefficient conversion can be derived. The last two conditions cannot be
established for Eq. (191) from the inspected material. Missing information is
not replaced by an implicit engineering convention or by a staggered provider.

## Evidence inspected

- Frass/Hofmann/Ponweiser, *Principles of Finned-Tube Heat Exchanger Design*,
  2nd ed., 2015: sections 5.2-5.4, Eqs. (139), (143), (144), (185),
  (190)-(191), Figs. 75-76/108 and the experimental-facility appendix.
  [Legal publisher PDF](https://www.wseas.us/e-library/conferences/2015/Frass/Frass.pdf).
  References below use bottom-of-page chapter pagination; add 20 for the
  one-based PDF page. Rendered pages 107-108 were checked against extraction.
- Hofmann, Frasz, Ponweiser, *Heat Transfer and Pressure Drop Performance
  Comparison of Finned-Tube Bundles in Forced Convection*, 2007, p. 77,
  Eqs. (17)-(21). [Publisher PDF](https://wseas.us/e-library/transactions/heat/2007/25-669N.pdf).
- Hofmann, *Experimental and Numerical Gas-Side Performance Evaluation of
  Finned-Tube Heat Exchangers*, TU Wien dissertation, 2009, pp. 32, 39, 115,
  Eqs. (2.28)-(2.29). [Institutional PDF](https://repositum.tuwien.at/bitstream/20.500.12708/12978/2/Hofmann%20Rene%20-%202009%20-%20Experimental%20and%20numerical%20gas-side%20performance%20evaluation...pdf).
  The existing local full PDF was inspected; a fresh web extraction exceeded
  the web tool's file-size limit. No inaccessible manual or Scribd scan was used.

## Native normalization and the integration algebra

Frass Eq. (139) defines `xi=2*delta_p/(rho_m*w_E**2)`; Eq. (144)
identifies per-row loss and multiplication by consecutive rows. Section 5.4
explicitly refers INLINE calculations to Eq. (139). `w_E` uses minimum area
and mean gas density. Eq. (191)'s dimensionless inputs are:

```text
xi_191 = 4.534*Re**(-0.34)*(Pt/dA)**(-0.5)*(Fp/dA)**(-0.7)
         *(h/dA)**0.5*(Pl/dA)**0.4
```

Here `Fp` is fin center pitch. The parent Waermeatlas Eq. (185) uses bare-tube
OD and minimum-section velocity for Re. Transferring that basis to its stated
Eq. (191) extension is a reconstruction; a model-specific property-state
instruction is not restated with Eq. (191).

For any selected effective per-row coefficient `xi_effective`, matching the
current core contract is exact algebra:

```text
q_min = rho*w_E**2/2 = G_min**2/(2*rho)
delta_p_source = n_rows*xi_effective*q_min
delta_p_core   = 2*n_rows*f_RB*rho*w_E**2
f_RB_equivalent = xi_effective/4
```

This does not establish that the source loss is the core's irreversible drag.
That physical equivalence must be established separately before using the map.

The source proposes `Kz=1-0.1*(8-n_rows)/6` below eight rows. It describes
three geometries at rows 2/4/6/8 and approximate Re 7000-25000, with an imperfect
row trend. These are experimental context, not a complete applicability box.
Do not extend Kz upward, invent single-row validation or infer missing geometry
limits. If Kz is selected, it enters `xi_effective` once, before the division
by four; it is not another multiplication by physical row count.

An independent 60-digit Decimal calculation checks the conversion algebra for
`Re=20000`, `dA=.038 m`, `Pt=.080 m`, `Pl=.090 m`, `Fp=1/150 m`,
`h=.016 m`, `mu=2e-5 Pa s`, `rho=1 kg/m3`:

```text
xi_191 = 0.3338430036837449
G_min  = 10.5263157894736842 kg/(m2 s)
q_min  = 55.4016620498614958 Pa
```

| Rows | Kz | Equivalent f with Kz | Algebraic bank pressure, Pa |
| --- | --- | --- | --- |
| 2 | 0.9000000000000000 | 0.0751146758288426 | 33.2918230820355040 |
| 4 | 0.9333333333333333 | 0.0778967008595405 | 69.0497071331106749 |
| 6 | 0.9666666666666667 | 0.0806787258902384 | 107.2736521532255128 |
| 8 | 1.0000000000000000 | 0.0834607509209362 | 147.9636581423800177 |

These are equation/conversion fixtures, not validated hydraulic predictions.
Returning native xi on the existing f_RB basis would quadruple these pressures.

## Heated/isothermal normalization remains unresolved

Section 5.2 p. 87 first converts heat-transfer measurements using `Tg/Twall`,
then recommends `Twall/Tg` for the same conversion. Eq. (143) prints
`xi_W/xi_a=Twall/Tg`. The inconsistency is visible in the rendered source.
Eq. (191) does not identify which normalization its fitted constant contains.

Even choosing the printed equation over the contradictory prose would not
identify the regression state. Applying a temperature multiplier to an already
converted fit could count the conversion twice. Omitting it could omit a
required correction. The thermal model's mean fin temperature does not resolve
this source question or establish the hydraulic mean wall-temperature definition.

## Irreversible bank-loss scope remains unresolved

The later ITE publications provide useful **related primary evidence**, rather
than an Eq. (191) reduction recipe. H07 Eq. (20) explicitly removes channel and
density-change contributions for its serrated bundle. H09 Eqs. (2.28)-(2.29)
also distinguish measured test-section pressure, channel friction and density
variation; p. 115 identifies the discussed results as staggered.

Neither source attaches those reductions to the three INLINE geometries or the
4.534 regression. Their different density-change expressions should not be
silently transferred. The monograph's appendix describes the facility but does
not close Eq. (191)'s treatment of these contributions or entry/exit effects.
This is absence of demonstrated linkage, not a claim that the experimenters
failed to make appropriate corrections.

The existing core separately integrates drag with three property states,
adds signed acceleration on face sections, and aggregates external losses in
the flow path. A measured bank/test-section static-pressure coefficient cannot
be assigned to that drag slot until its included losses are known. The same
problem prevents claiming that an isothermal-only implementation closes scope:
zero acceleration does not establish removal of channel/transition losses.

## Remaining source-verification tasks (no longer a non-strict implementation gate)

Obtain a legal source or documented source clarification attaching to Eq. (191):

1. The regression's heated/isothermal state and the direction of any required
   conversion, including the wall-temperature definition.
2. The pressure-reduction recipe: channel friction, density-change/acceleration,
   and entry/exit/transition treatment.

The following paragraph and status block record the historical pre-authorization
decision, not the current runtime state. Unknown geometry/accuracy limits may retain unresolved-range diagnostics under
the accepted policy; they cannot resolve an unknown output quantity. Once the
two physical-contract questions are closed, implement native xi diagnostics,
one explicit row treatment, the derived coefficient mapping or native-basis
integration, and independent single-/three-state and flow-path tests.

```text
THERMAL CHECKPOINT = e374329
COEFFICIENT CONVERSION ALGEBRA = CLOSED
MINIMUM-AREA VELOCITY BASIS = CLOSED
PHYSICAL ROW MULTIPLICATION = CLOSED
HEATED/ISOTHERMAL REGRESSION STATE = OPEN
IRREVERSIBLE BANK-LOSS SCOPE = OPEN
INLINE PRODUCTION HYDRAULICS = NOT IMPLEMENTED
PACKAGE VERSION = 0.9.3
```
