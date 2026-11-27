# Circular solid-fin hydraulic selection in v0.9.11

For circular tubes with individual solid fins in a conventional STAGGERED
bank, `auto` selects Genić 2006. This intentionally changes pressure predictions
from v0.9.10. Thermal selection and equations are unchanged.

Nir 1991 remains deferred: legitimate secondary sources recover the plain-fin
equation and its piecewise geometry factor, but pressure normalization, physical
row interpretation, property reduction and joint applicability are insufficiently
closed for a public provider. There is no public Nir selector in this release.

AUTO follows these rules:

1. Require circular individual solid fins and conventional staggered geometry.
   At least five physical rows, `Rd >= 1`, `Pt/D_root <= 3.5`,
   `Pl/D_root <= 3.5` and `550 < Re_de < 130000` must pass. Actual projected
   passage area must agree with the source geometry mapping.
2. For constant thickness, evaluate `genic2006_staggered`. For actual linear
   taper, evaluate `genic2006_staggered_tapered`, retaining actual projection,
   inclined surface and metal volume in the unchanged empirical fit.
3. Actual taper, a gas-facing root layer, non-air gas and nonisothermal mean-state
   reduction require explicit engineering authorization. Taper remains an
   **unvalidated engineering extension**. Strict source mode rejects these
   extensions even when authorized.
4. Reject scalar source-limit failures, unsupported geometry, invalid outputs
   and execution errors. Authorization does not relax AUTO's scalar limits.
   There is no fallback to Robinson–Briggs or substitution after failure.

`SOURCE_DOMAIN_PASS` describes the implemented scalar limits, without claiming
complete joint calibration coverage. Genić's raw calibration data remain
unavailable. Read its [source and applicability guide](genic2006_finned_hydraulics.md).

For low-level pressure functions, authorize with
`allow_engineering_estimates=True`. Rating and Simulation receive the policy
on the exchanger:

```python
from core.heat_transfer import OutsideProviderPolicy
from core.models.bare_tube import BareTubeHeatExchanger

hx = BareTubeHeatExchanger(
    bundle,
    outside_provider_policy=OutsideProviderPolicy(allow_engineering_estimates=True),
)
# hx.rate(...) and hx.simulate(...) use hydraulic AUTO independently of HTC.
# Explicit reference: hx.rate(..., finned_pressure_drop_provider="robinson_briggs_1966")
```

Explicit Robinson–Briggs remains available with its original equations,
equilateral geometry guard, range warnings, three-state integration and
numerics. Explicit constant-fin Genić selectors preserve their previous
arithmetic and warning/strict extrapolation policy. Actual taper additionally
requires authorization even with an explicit tapered selector.

INLINE circular solid-fin AUTO remains Frass Eq.191. Serrated, continuous plate,
elliptical and bare-tube dispatch is unchanged. Hydraulic AUTO requires constant
gas mass flow; an unclosed changing-mass wet-flow mapping raises a diagnostic.

Direct result diagnostics and bank `selection_metadata` record requested and
resolved provider, AUTO reason, engineering status, scalar applicability and
pressure scope. Native Genić reports bank loss at mean gas state and adds signed
face-flux acceleration separately. External scoped estimates retain their
declared scope and policy; unresolved estimates are not certified bank losses.
Reference area, diameter, velocity, Reynolds number and state basis remain
provider-specific. Known non-air identity now reaches Briggs–Young and
Robinson–Briggs warnings without changing their numerical equations.

The [synthetic example](../examples/extended_surface_hydraulics.py) exposes
AUTO provenance alongside explicit predictions. No model is selected because
it predicts the highest pressure.
