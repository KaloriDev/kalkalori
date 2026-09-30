# Elmahdy-Mitalas equation interface (verification stage)

This page documents the isolated, frozen source-reference kernel.
Current production routing is described in the separate
[Rating/Simulation integration](elmahdy_mitalas_integration.md).
No process or notebook acceptance is claimed. Release remains on hold.
The separate [production adaptation](elmahdy_mitalas_production.md) records the
approved source-profile moisture closure, coupled drain equations and adapters.
This page describes the frozen reference and its historical verification limits.

## Authority and licensing

Baseline: EnergyPlus **v25.2.0**, commit
`cf7368216c73c43181e057fa33b479c4e0c86df0`,
[CalcDetailFlatFinCoolingCoil](https://github.com/NatLabRockies/EnergyPlus/blob/cf7368216c73c43181e057fa33b479c4e0c86df0/src/EnergyPlus/WaterCoils.cc#L2924),
object `Coil:Cooling:Water:DetailedGeometry`. The same revision's
[Engineering Reference](https://github.com/NatLabRockies/EnergyPlus/blob/cf7368216c73c43181e057fa33b479c4e0c86df0/doc/engineering-reference/src/simulation-models-encyclopedic-reference-001/coils.tex#L551)
provides the equations. The original 1977 paper has NOT been independently
verified. This is neither the simple design-UA object nor its sizing algorithm.

The pinned license has four conditions, including a naming condition; it is
not assumed to be ordinary BSD-3-Clause or relicensed GPL. Production Python
is an original implementation of the documented mathematical equations under
GPL-3.0-only. The separately compiled native verification source retains the
upstream notices and is not linked or distributed with the Python package.
File hashes are recorded in `references/elmahdy_mitalas_source.json`.

## Equation and interface map

Temperatures here are Celsius; the eventual public adapter must convert Kelvin.
`md` and `W` use kg dry carrier/s and kg water/kg dry carrier. Moist gas mass
flux for the source air correlation is `md*(1+Win)/Amin`. Gas enthalpy `h` and
`cp` are J/kg dry carrier and J/(kg dry carrier K). Source C++ uses kJ/kW
internally; this kernel uses SI J/W. The liquid capacity is `Cw=mw*cpw`.

| Equation/variable | Meaning and implementation |
|---|---|
| `hs(T)=a+b*T` | Iterated saturation-enthalpy secant, `b` in J/(kg K); full wet uses both wet wall endpoints, partial uses cold wet wall and inlet dew point. |
| `Rd=1/(alpha_d*Aeff_d)` | Whole-surface dry outside resistance, K/W. |
| `Rw=cp/(alpha_w*Aeff_w)` | Whole-surface wet enthalpy resistance, s/kg; NOT K/W. |
| `Ri(T)` | Whole-surface liquid + metal (+ explicitly configured fouling) resistance, K/W. |
| `UA_d=(1-f)/(Ri+Rd)` | Dry temperature conductance, W/K. |
| `K_w=f/(b*Ri+Rw)` | Wet enthalpy conductance, kg/s. Multiplying by the SAME `b` gives W/K on that specific reference only. |
| `Qd=Ca*(Ta,in-Ta,int)=Cw*(Tw,out-Tw,int)` | Dry balance; standard counterflow epsilon-NTU is algebraically the documented LMTD solution. |
| `Qw=md*(h_int-h_out)=Cw*(Tw,int-Tw,in)` | Source wet LMHD balance. Counterflow capacities are `md` and `Cw/b`; the interface is solved jointly with the dry balance. |
| `Ts=(Rw*Tw+Ri*(h-a))/(Rw+b*Ri)` | Wet surface endpoints; no isothermal wall constraint. |
| `Ts,int=Tw,int+(Ta,int-Tw,int)*Ri/(Ri+Rd)` | Dry-side interface; partial wet solves `Ts,int=Tdew,in`. |
| `y=exp(-f/(Rw*md))` | Source outlet bypass approximation. `h_eff=h_int-(h_int-h_out)/(1-y)`, `T_eff=hs^-1(h_eff)`, `Ta,out=T_eff+(Ta,int-T_eff)*y`. |
| `Wout=W(Ta,out,h_out)` | Bulk outlet is NOT forced saturated. `mcond=md*(Win-Wout)`. |

The kernel brackets the interface and iterates the secant/properties to tight
numerical tolerances; equal capacity rates and vanishing dry area use analytic
limits. C++ uses at most 8 full-wet/40 partial iterations, 0.01 K full-wet and
0.00002 K interface criteria and exponent clipping. These differences are
numerical, not claims of better physical accuracy. A failed solve raises.

## Source conventions and adapter restrictions

Source psychrometrics: `cp=1004.84+1858.95*max(W,1e-5)`;
`h=1004.84*T+max(W,1e-5)*(2500940+1858.95*T)`; saturation humidity uses
0.62198 and a 1000 Pa denominator guard. The native enthalpy-to-saturation
inversion uses a polynomial near standard pressure; exact inversion is an
identified numerical difference. No production flue-gas substitution by air
is authorized by this verification interface.

Source air viscosity is 1.846e-5 Pa s; inverse Pr^(2/3) is 1.23. The source
flat-fin wet HTC multiplier is `1.425-0.00051*Re+0.000000263*Re^2`, documented
for Re=400..1500. Inside water HTC is `1429*(1+0.0146*T)*v^0.8*Di^-0.2`,
documented for Re>3100. Wall conduction uses thickness/(k*Ai), with a fixed
5e-5 m2 K/W water fouling allowance. Dry fin efficiency uses a polynomial fit
to constant-thickness annular-fin Bessel equations with equivalent flat-fin
area. C++ wet fins use a separate clipped empirical correlation (not the
same dry expression suggested by the prose). These are source test assumptions,
NOT approved bare/annular/glycol adapters. Source circuit count equals tubes
per row; counterflow approximates cross-counterflow with at least four rows.
Schedules, fan cycling and sizing input resets are outside this kernel.

## Energy and required-area gate

The source solves `md*(hin-hout)=Qliquid` and neglects drained-liquid enthalpy.
`heat` is explicitly labeled with that approximation. A production adaptation
must solve `md*(hin-hout)=Qliquid+H_drain` on common datums; adding a reporting
term afterward cannot repair it. Drain temperature/transport must be resolved
before process integration. No legacy condensate closure is attached here.

Rating will size effective tube length with fixed topology and preserved end
allowance; it must call the eventual SAME forward kernel at every trial.
Physical required area differs from conductance-equivalent area. Published
`overdesign_factor=UA_actual/UA_required-1` and `A_required=UA_required/U_mean`
are not silently redefined. Reference-state conversions, physical area and
length belong in explicit diagnostics; their approved mapping is documented in the integration page linked above.

## Predeclared native comparison tolerances

Before reference cases: Q relative 0.2%; each outlet temperature absolute
0.08 K; humidity 4e-5 kg/kg; condensate 2e-5 kg/s; wet fraction absolute
0.002; applicable wall/interface temperatures 0.08 K. These account for native
stopping criteria and psychrometric inversion, and assess implementation
agreement only. Dry/full/partial synthetic cases must all pass. Uninitialized
or stale native local variables are not treated as physical reference values.


## Verified source agreement and unresolved physical limits

Eight native comparisons cover dry, partial and full wet conditions in two
families: directly supplied initialized coefficients, and native dry-fin
fitting with source geometry coefficients. All eight meet the predeclared
tolerances. Constant liquid rho/cp injections are shared assumptions; no
claim of complete EnergyPlus object/plant validation follows. The native
fixture also records two near-onset counterexamples, not accepted outputs.

For the native-fin synthetic case at Win=0.0067, the original procedure returns
Wout=0.00670166231229, wet fraction=0.0582528303969 and a negative condensate
flow of -8.31156e-7 kg/s, with no convergence warning. Exact inversion of
saturation enthalpy also produces Wout>Win. Therefore replacing the native
saturation-temperature polynomial alone does not resolve this case. The
isolated kernel raises `SourceClosureError` with computed diagnostics; it
does not clip W, discard latent energy or fall back to another model.

The source general solution can also reach f=1 while retaining its
cold-wall/dewpoint secant; that limit is represented explicitly and tested.
Dry operation allows the legitimate counterflow outlet-temperature cross.
Source agreement is established, but physical onset acceptance is NOT met.
A consistent humidity/condensate-energy adaptation must be agreed before
connecting this kernel to either public solving mode or any project workflow.

Reproduce the independent reference with separately obtained pinned files:

```text
python tools/verify_elmahdy_mitalas_native.py --source-dir SOURCE --output-dir SCRATCH --compiler CXX
```

For Zig 0.13.0, use its executable as CXX and add `--zig`. This tool verifies
source hashes, extracts the native coil and relevant psychrometric/fin
routines, retains upstream notices, compiles and runs them independently,
and writes `native_results.json`. It never imports the Python kernel.
The generated native files are validation-only, outside the distribution.

Current verification: 17 focused tests in both Python 3.11 and 3.12. These
include expected rejection of source-limit violations, NOT physical acceptance
of those conditions. Geometry adapters, shared Rating/Simulation, process
acceptance and full public regression remain pending. No prior experimental
wet orchestration has been recovered. Candidate property/transport patches
remain deferred until the accepted adaptation specifies their dependencies.
