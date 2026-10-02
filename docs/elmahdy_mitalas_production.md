# Elmahdy-Mitalas production adaptation (internal engine)

This stage implements the source-profile moisture closure selected for
KalKalori. Simulation and area-based Rating are connected through
`ElmahdyMitalasWetCoilProvider`, the public default for both operations.
The v0.8.3 release candidate has completed public and engineering validation;
publication awaits user release review.
The original reference kernel, native tooling, fixtures and pinned EnergyPlus
v25.2.0 revision `cf7368216c73c43181e057fa33b479c4e0c86df0` are unchanged.
See [source provenance and tolerances](elmahdy_mitalas.md).

## Internal boundary and units

`wet_coil.py` contains one global dry/partial/full-wet process engine.
`wet_coil_adapters.py` supplies configured gas properties, sensible inside-fluid capacity /
inside-wall resistance, and BareTube or CircularFinnedTube surface quantities.
The public API accepts direct `wet_coil_provider` objects; `None` selects
Elmahdy. The explicit historical `LegacyBulkMeanWetCoilProvider` supports
Simulation only. Unsupported explicit operations raise
`WetCoilProviderUnsupportedError`; there is no silent fallback, registry,
discovery or string-based selection. External implementations may satisfy the
[public provider contract](wet_coil_providers.md) without registration.
The unvalidated profile candidate is an
internal iterate of the property solve; only the final validated result can
leave either solving entry point. This is also why intermediate coefficient
iterations do not incorrectly terminate at a temporarily negative onset margin.

Production temperatures are K. `md` is kg dry carrier/s; `W` is kg H2O/kg dry
carrier; gas h and cp are J/kg dry carrier and J/(kg dry carrier K). The gas
enthalpy uses the existing `WetGasEnthalpyEvaluator`: configured dry composition
through CoolProp, water vapor through IAPWS-IF97. Drained liquid uses the SAME
IAPWS water datum. Conserved dry-gas mass makes its separate reference cancel.
A solve-local CoolProp state optionally avoids rebuilding the same EOS for
each temperature query; strict high-level/low-level enthalpy parity is tested.
Existing callers retain their previous evaluation path.
The existing saturation-vapor-at-gas-temperature engineering approximation is
retained. Air psychrometrics are not substituted for a different carrier.

Inside transport comes from the configured sensible-liquid provider. Native
enthalpy differences are used where available, otherwise a cp integral supplies
the sensible enthalpy rise. Mean liquid capacity is iterated against that rise.
The internal-flow HTC dispatch and exact cylindrical core-wall resistance are
used. No source water correlation or 5e-5 m2 K/W fouling constant is introduced.
Production fouling is zero. Unsupported phase / cocurrent inputs fail explicitly.

## Source-profile moisture and drain equations

Coordinate x is whole-coil outside-area fraction, increasing from the cold
liquid inlet/gas outlet (0) to the warm wet/dry interface (f). In a wet region
`hs(Ts)=a+b*Ts`, `Ri` is K/W and `Rw` is s/kg dry carrier. Define

```
K = 1/(Rw + b*Ri)
D = h - a - b*Tl
j = md*dW/dx                  [kg/s per unit x]
d = h_liquid(Ts)*j            [W per unit x]
qg = K*(D + b*Ri*d)
ql = qg - d
md*h' = qg; Cw*Tl' = ql
lambda = K*(1/md - b/Cw)
gamma = K*b*(Ri/md + Rw/Cw)
D(x) = D(0)*exp(lambda*x)
       + gamma*integral_0^x exp(lambda*(x-s))*d(s) ds
```

Integrating this expression gives continuous h/Tl profiles and
`Ts=(Rw*Tl+Ri*(h-a)-Ri*Rw*d)/(Rw+b*Ri)`. Stable `expm1(z)/z` evaluation covers
exactly equal enthalpy capacities and vanishing lengths. The dry-region
counterflow transfer and the wet endpoint enthalpy are solved together.
The saturation secant, inside resistance and drain forcing are iterated.
When secant endpoints coalesce within 1e-3 K, its centered derivative limit
avoids subtracting large absolute enthalpies over a vanishing interval.
Convergence is checked on physical states and drain density; independent
energy and mass tolerances are unchanged.

For every remaining wet length, the approved source construction defines W:

```
y = exp(-(f-x)/(Rw*md))
heff = h_interface - (h_interface-h(x))/(1-y)
Teff = hs_inverse(heff)
Tgas = Teff + (T_interface-Teff)*y       # constant-cp source limit
W(x) = humidity(Tgas,h(x))
```

The differentiated SAME W(x) supplies j and the drain integral. No independent
axial moisture transport ODE, fitted mass correction or output clipping is used.
In the physical gas-flow direction this is `dM_cond=-md*dW`.

For variable KalKalori cp, the sensible interpolation is performed in the
fixed-inlet-W enthalpy coordinate instead of a temperature coordinate. Its
inverse supplies Tgas. This is the caloric-property adapter of the source
constant-cp relation: affine h(T,Win) recovers exactly the formula above.
The wet sensible cp is the secant between interface gas temperature and inlet
dewpoint, so dry and wet fluxes agree at zero condensation. The dry-region cp
is iterated on its own inlet/interface enthalpy difference. This avoids creating
an artificial moisture source solely from a nonlinear property conversion.

Only d(x) is represented at Gauss/Chebyshev quadrature points. There are no
independent axial cell temperatures or enthalpies. A second quadrature of the
continuous profile checks mass and drainage; integral checks are distinct from
nonlinear residuals. The coupled solve enforces

```
md*(hin-hout) = Qliquid + Hdrain
Mcond = md*(Win-Wout) = integral j dx
```

Drain is never subtracted after solving or used to normalize an outlet.

## Surface adapters

BareTube uses the existing Zukauskas dispatch, wet-gas physical mass flow
`md*(1+W)`, and actual outside area. Both effective areas equal that area.

CircularFinnedTube uses Briggs-Young and actual annular/taper geometry.
The validated radial conduction mesh and tridiagonal primitive are reused.
Only the continuous radial wet-front source quadrature from `d8b38b8` was
recovered, adapted to explicit dry-basis cp. Radial mass, latent heat and drain
use the same wet faces, taper areas and physical convecting tip. A higher-order
source quadrature checks the radial solution independently of its Newton solve.
No legacy whole-coil LUMPED/REGIONAL model is included.

For a continuous root layer, common root/contact resistance joins the inside
path and the global surface coordinate denotes the exposed root temperature.
For fins attached at Do, the exposed primary tube bypasses the fin contact
resistance, which remains in the fin branch. Contact uses the existing input
precedence and resistance conversion. Diagnostic wall/core/base/tip states and
separate dry/wet effective areas accompany the result.

The global source profile fixes axial condensate removal. The local radial
response supplies its *conditional enthalpy distribution*: primary and fin
condensation weights determine `hbar_liquid = integral hl(T)*jm dA / integral
jm dA`. Then the global drain density is `j_source(x)*hbar_liquid(x)`. The radial
mass magnitude is diagnostic, not a replacement for source-profile condensate.
This is an explicit reduced surface assumption; it is not a claim that the
source whole-coil moisture approximation equals a local transport model.
The radial enthalpy distribution is iterated inside the global solve and checked
again at additional physical radial evaluations. Its zero-mass limit is the
base-surface liquid enthalpy. No completed Q/W/condensate result is rescaled.

The wet enthalpy resistance is the radial heat/mass response at the quadrature
mean wet-region state, consistent with the global mean-property approximation.
Local correlation applicability and radial response diagnostics are separate
from global model provenance; original EnergyPlus flat-fin/Re limits do not
validate these industrial adapters.

## Acceptance and limits

A dry result is accepted when the cold dry exposed surface is at/above inlet
dewpoint. Wet candidates require nonnegative total/local removal, nonnegative
wet driving force (including the warm endpoint), admissible bulk vapor, and
independently consistent energy/mass integrals. If the dry criterion fails and
the wet candidate is invalid, `WetCoilModelError` reports the reason, onset
margin, f and condensate. An invalid wet candidate is never clipped to dry.

Numerical acceptance uses `WetCoilSolverOptions`: defaults are 1 W energy,
5e-7 kg/s mass, 0.01 K outlet accuracy and a 300 s whole-operation deadline.
Profile and independent drain integrals and the inside-provider energy balance
use the configured tolerances. Strict regressions request tighter controls;
see [the numerical-control guide](wet_coil_solver_controls.md).
W admissibility remains 1e-9 kg/kg (roundoff allowance, not output clipping);
radial specific-enthalpy stability remains .002 J/kg. Synthetic cases cover
dry, both sides of onset, partial, full, exactly equal enthalpy capacities,
vanishing dry region, configured gas/water/glycol and both surface adapters.
These numerical controls do not represent physical model uncertainty.
The predeclared native source tolerances remain unchanged.

Source-limit tests exercise the actual production region equations with source
properties/correlations and drain disabled. Source candidates violating local
admissibility remain rejected at the production acceptance boundary even when
the source kernel accepts their endpoints. There is no source-only/test-only
branch in the production equations. Reference mode still reproduces the pinned
native procedure and retains its documented near-onset limitations.

Supplemental validation data remains outside the public distribution and is
not a production dependency. Public reference fixtures use synthetic inputs.

The integrated provider remains a counterflow, mean-property/secant
model with the documented source outlet approximation and reduced annular
surface mapping. Frost, two-phase inside flow and cocurrent flow are outside
this implementation. Public result mapping and Simulation/Rating routing are
implemented; engineering case acceptance remains separate. Rating sizes
`A_required` / thermal area scale only, while installed hydraulics always use
installed geometry. See [the integration semantics](elmahdy_mitalas_integration.md).

## v0.8.3 release-candidate validation

All 1487 public tests passed on Python 3.11.9. Seven engineering notebook
representatives (17 required cases) passed with their original physical checks,
covering dry operation, both public wet providers, thermal-area Rating,
zero/nonzero margins and partial/full wet regimes. Existing tube-side
enhancement regressions passed in the full suite. Numerical acceptance does
not establish experimental accuracy. Engineering inputs and outputs remain
outside the public distribution.

## Historical production-engine validation record

Focused coverage passed in both environments: 203 distinct tests per
interpreter, across grouped runs. The initial 201-test run covered reference,
production and directly affected property/surface tests. After the coalescing
secant fix, all 37 then-current reference/production tests were rerun; the
additional finned dry test passed separately. Both final model runs include
the real-property onset continuity regression.

Environments: Python 3.11.9 / IAPWS 1.5.5 / CoolProp 8.0.0 and notebook
Python 3.12.8 / IAPWS 1.5.4 / CoolProp 6.6.0. The latter emits upstream
IAPWS/NumPy scalar-array deprecation warnings; there are no failed tests.

All eight pinned source-limit cases pass the existing comparison tolerances.
This is equation parity, with physical acceptance still enforced separately.
The frozen native source was not rebuilt in this stage. This historical
record does not establish engineering-case or experimental acceptance.
