# Property Model Selection Guide

This document explains how to select property models in KalKalori `v0.7.0`.

The goal is to avoid hidden assumptions. KalKalori does not automatically decide whether a fluid should be treated as classical moist air, dry gas, wet gas, condensing gas, steam, or water. The user must select the appropriate property path.

---

## 1. Available Property Paths in v0.7.0

| Property path              | Main API                                       | Intended use                                                           | Phase-change support              |
| -------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------- |
| Dry air | `dry_air_props()`, `DryAirPropertyProvider` | Standard dry air properties for sensible gas-side calculations | No |
| Psychrometric moist air    | `MoistAirState`, PsychroLib adapter            | Classical moist air, RH, dew point, saturation, HVAC-like calculations | Onset / limit helpers only        |
| Moist-air transport helper | `moist_air_transport_props_from_state()`       | Transport properties of moist air in normal engineering range          | No full wet HX solver             |
| Water/steam                | `water_steam_props_iapws97()`, `IAPWS97WaterSteamProvider` | Water, saturated water, wet or superheated steam | Inside cooling/condensation and heating/evaporation |
| CoolProp pure fluid        | `CoolPropFluidProvider`                        | Pure fluids and pseudo-pure fluids                                     | Single-phase Water only; no pure-water phase change |
| Explicit gas mixture       | `GasMixtureSpec`, `GasMixturePropertyProvider` | Dry gases, flue gas, hot humid gas with H2O as gas-phase component     | Wet-gas H2O condensation; pure-H2O phase change unsupported |
| Constant properties        | `ConstantPropertyProvider`                     | Debugging, reference calculations, fixed-property cases                | No                                |
| Tabulated liquid           | `LiquidPropertyPoint`, `TabulatedLiquidProvider` | Manually entered single-phase liquid (e.g. from a datasheet), with or without T-dependency | No                                |

### 1.1 Tabulated Liquid Provider (v0.7.7)

```python
from core.properties import LiquidPropertyPoint, TabulatedLiquidProvider
```

For a manually entered single-phase liquid (e.g. from a datasheet):

```text
1 point   -> constant rho/cp/mu/k at every positive temperature
2+ points -> T-dependent interpolation, with the queried T restricted to
             the supplied table (outside it raises ValueError; this
             provider never extrapolates)

rho, cp, k -> linear interpolation versus T
mu         -> log-linear interpolation versus T
h          -> exact integral of the interpolated cp(T), h = 0 at the
              lowest supplied T (or the single point's T for one point)

pressure -> accepted for interface compatibility only; ignored
```

This replaces the practical need for a separate manual constant-property
liquid implementation, while `ConstantPropertyProvider` remains available
unchanged for fixed-property cases that do not need temperature dependency
or enthalpy.

---

## 2. Recommended Decision Tree

### Step 1 — Is the medium water or steam?

Use:

```python
water_steam_props_iapws97(...)
```

Typical cases:

```text
liquid water
pressurized water
saturated liquid
saturated steam
superheated steam
```

Supported input modes:

```python
water_steam_props_iapws97(T=..., p=...)
water_steam_props_iapws97(p=..., x=...)
water_steam_props_iapws97(p=..., h=...)
```

Use this path for tube-side water/steam, not the gas-mixture path.

---

### Step 2 — Is the medium classical dry air?

KalKalori provides a dedicated dry-air property path:

```python
from core.properties import DryAirPropertyProvider, dry_air_props

props = dry_air_props(T=303.15, p=101325.0)
```

or:

```python
provider = DryAirPropertyProvider()
props = provider.at(T=303.15, p=101325.0)
```

This path is for dry air only.

Use it for:

```text
dry air in tubes
dry air outside tubes
dry gas-gas sensible heat exchangers
pressure-drop calculations with dry air
heat-transfer calculations with dry air
```

Do not use it for:

```text
relative humidity
humidity ratio
dew point
saturation
condensation
wet-surface calculations
```

For humid air, use the psychrometric moist-air path.  
For hot gases with water vapor as a gas-phase component, use `GasMixtureSpec`
explicitly.

The default dry-air provider tries CoolProp `"Air"` first. If CoolProp is not
available, it uses a simple ideal-gas / Sutherland fallback. The fallback is
intended for tests and preliminary calculations, not for high-accuracy property
work.

### Step 3 — Is the medium classical moist air?

Use the psychrometric moist-air path when the input is naturally defined by:

```text
T + RH
T + W
T + dew point
saturation humidity ratio
dew-point margin
condensation onset
```

Typical examples:

```text
ambient air
ventilation air
drying air in moderate temperature range
air preheaters below the high-temperature psychrometric limit
```

Use:

```python
moist_air_state_from_t_rh(T=T, RH=RH, p=p)
moist_air_state_from_t_w(T=T, W=W, p=p)
moist_air_state_from_t_w_g_per_kg_da(T=T, W_g_per_kg_da=W_g, p=p)
```

Recommended range for KalKalori `v0.4.x`:

| Parameter                       |                          Recommended range | Hard/implementation notes                    |
| ------------------------------- | -----------------------------------------: | -------------------------------------------- |
| Dry-bulb temperature            |                             `-40 … 100 °C` | safest practical psychrometric range         |
| Extended dry-bulb temperature   |                            `-100 … 200 °C` | PsychroLib limit for SI dry-bulb temperature |
| Pressure                        | near atmospheric, typically `80 … 120 kPa` | wider range may work but must be validated   |
| Relative humidity               |                                    `0 … 1` | only if physically valid at given `T,p`      |
| Humidity ratio `W`              |                        `0 … 0.10 kg/kg_da` | above `0.10` KalKalori should warn           |
| High humidity `W`               |                          `> 0.10 kg/kg_da` | possible but requires caution                |
| Transport helper temperature    |                              `250 … 450 K` | current recommended warning range            |
| Transport helper humidity ratio |                        `0 … 0.10 kg/kg_da` | current recommended warning range            |

PsychroLib itself is based on ASHRAE psychrometric formulations and its SI implementation restricts dry-bulb temperature to `-100 … 200 °C`. Therefore, a case such as `400 °C` must not be passed through the PsychroLib moist-air state path.

---

### Step 4 — Is the gas hot, dry, or explicitly defined by composition?

Use the gas-mixture path:

```python
GasMixtureSpec(...)
GasMixturePropertyProvider(...)
```

Typical examples:

```text
dry air treated as N2/O2
dry flue gas
process gas
hot humid air represented as N2/O2/H2O
hot flue gas represented as N2/O2/CO2/H2O
gas inside tubes
gas outside tubes
```

Supported composition bases:

```python
basis="mole"
basis="volume"
basis="mass"
```

For gases, volume fractions may be treated as mole fractions.

Example — dry air as explicit gas mixture:

```python
dry_air = GasMixtureSpec(
    components={
        "N2": 0.79,
        "O2": 0.21,
    },
    basis="volume",
    backend="HEOS",
    imposed_phase="gas",
)
```

Example — dry flue gas:

```python
dry_flue_gas = GasMixtureSpec(
    components={
        "N2": 0.74,
        "O2": 0.04,
        "CO2": 0.22,
    },
    basis="volume",
    backend="HEOS",
    imposed_phase="gas",
)
```

Example — hot humid air as gas mixture:

```python
hot_humid_air = GasMixtureSpec(
    components={
        "N2": 0.698,
        "O2": 0.186,
        "H2O": 0.116,
    },
    basis="mole",
    backend="HEOS",
    imposed_phase="gas",
)
```

Example — hot wet flue gas as gas mixture:

```python
hot_wet_flue_gas = GasMixtureSpec(
    components={
        "N2": 0.68,
        "O2": 0.04,
        "CO2": 0.16,
        "H2O": 0.12,
    },
    basis="mole",
    backend="HEOS",
    imposed_phase="gas",
)
```

Recommended gas-mixture path range in `v0.4.x`:

| Parameter            |         Recommended range | Notes                                         |
| -------------------- | ------------------------: | --------------------------------------------- |
| Temperature          |            `200 … 800 °C` | intended for gas-phase engineering cases      |
| Pressure             |       `0.7 … 2.0 bar abs` | best suited for low-pressure gas-side HX work |
| Water vapor fraction |              user-defined | H2O is treated as gas-phase component         |
| CO2 fraction         |              user-defined | backend support must be checked               |
| SO2 / acid gases     | user-defined with caution | acid dew point is not modelled                |
| Phase                |                  gas only | use `imposed_phase="gas"`                     |
| Condensation         | provider evaluation only | use the section 15 HX solver for wet-surface balance |

CoolProp supports phase imposition through input keys such as `T|gas`, which can avoid automatic PT-flash problems for known gas-phase states. KalKalori uses this for gas-mixture calculations when `imposed_phase="gas"` is selected.

---

## 3. How to Represent Hot Humid Air

For hot humid air, especially above `200 °C`, do not use:

```python
moist_air_state_from_t_w_g_per_kg_da(...)
```

Instead, represent the medium explicitly as a gas mixture containing water vapor.

Given:

```text
W = kg water / kg dry air
```

take a basis of:

```text
1 kg dry air + W kg water vapor
```

For dry air approximated as:

```text
79% N2
21% O2
```

convert the dry-air part to component masses, then add water vapor mass. This gives mass fractions suitable for:

```python
GasMixtureSpec(..., basis="mass")
```

Example for `W = 0.120 kg/kg_da`:

```python
hot_humid_air = GasMixtureSpec(
    components={
        "N2": 0.681,   # approximate mass fraction
        "O2": 0.199,   # approximate mass fraction
        "H2O": 0.120 / 1.120,
    },
    basis="mass",
    backend="HEOS",
    imposed_phase="gas",
)
```

Better implementation pattern:

```python
def humid_air_mass_components_from_W(W):
    # 1 kg dry air + W kg water vapor
    # split dry air into N2/O2 first
    ...
```

Important:

```text
This is not psychrometric moist air anymore.
This is an explicit gas-phase mixture.
RH is not evaluated.
Dew point is not evaluated.
Saturation humidity ratio is not evaluated.
Condensation is not modelled.
```

---

## 4. How to Represent Wet Flue Gas

For flue gas, the preferred `v0.4.5` approach is explicit composition.

Example input if composition is known on wet volume basis:

```python
wet_flue_gas = GasMixtureSpec(
    components={
        "N2": 0.70,
        "O2": 0.04,
        "CO2": 0.14,
        "H2O": 0.12,
    },
    basis="volume",
    backend="HEOS",
    imposed_phase="gas",
)
```

Example input if composition is known on dry basis and water is known as `W`:

```text
dry gas: N2/O2/CO2
water: W kg water / kg dry gas
```

Procedure:

```text
1. Convert dry gas mole or volume composition to dry-gas molar mass.
2. Convert dry-gas mole fractions to dry-gas mass fractions.
3. Take 1 kg dry gas.
4. Add W kg water vapor.
5. Normalize total mass to get wet gas mass fractions.
6. Use GasMixtureSpec(..., basis="mass").
```

Important:

```text
GasMixtureSpec treats H2O as vapor-phase component.
It does not remove water when temperature drops below dew point.
It does not calculate latent heat.
It does not calculate acid dew point.
```

---

## 5. Explicit Water Vapor in GasMixtureSpec

For hot gas-phase calculations with water vapor, do not introduce a separate
humid-gas model in `v0.4.5`. Use `GasMixtureSpec` explicitly.

If a dry gas composition is known and water is specified as:

```text
W = kg water / kg dry gas
```

use:

```python
from core.properties import gas_mixture_from_dry_composition_and_water_ratio

spec = gas_mixture_from_dry_composition_and_water_ratio(
    dry_components={
        "N2": 0.79,
        "O2": 0.21,
    },
    dry_basis="mole",
    water_ratio=0.120,
    backend="HEOS",
    imposed_phase="gas",
)
```

This returns a normal `GasMixtureSpec` with `basis="mass"` and with water vapor
as an explicit gas-phase component.

Important limitations:

```text
- no condensation
- no latent heat balance
- no wet-surface heat transfer
- no water removal from gas composition
- no acid dew point
```

This helper only performs composition bookkeeping.

---

## 6. Dry Gas Cases

Use `GasMixtureSpec` for dry gas when:

```text
there is no water vapor,
water vapor is negligible,
or water vapor is intentionally excluded from the model.
```

Examples:

```python
dry_air = GasMixtureSpec(
    components={
        "N2": 0.79,
        "O2": 0.21,
    },
    basis="volume",
    backend="HEOS",
    imposed_phase="gas",
)
```

```python
dry_flue_gas = GasMixtureSpec(
    components={
        "N2": 0.75,
        "O2": 0.05,
        "CO2": 0.20,
    },
    basis="volume",
    backend="HEOS",
    imposed_phase="gas",
)
```

Recommended use:

```text
gas-gas sensible heat exchanger
dry air heater
dry flue-gas cooler above dew point
process-gas heater/cooler without condensation
```

---

## 7. Non-Condensing Humid Gas Cases

Use `GasMixtureSpec` with H2O as gas component when:

```text
gas contains water vapor,
the complete heat exchanger is safely above dew point,
the calculation is sensible-only,
water vapor remains in gas phase,
composition is assumed constant.
```

Examples:

```text
hot humid air at 400 °C
wet flue gas above water dew point
process gas with water vapor but no condensation
```

Use:

```python
GasMixtureSpec(
    components={
        "N2": ...,
        "O2": ...,
        "CO2": ...,
        "H2O": ...,
    },
    basis="mole",  # or "mass"
    backend="HEOS",
    imposed_phase="gas",
)
```

Do not use this path if:

```text
the gas cools below water dew point,
liquid water may form,
latent heat is significant,
tube surface is wet,
condensate removal changes gas composition,
acid dew point is relevant.
```

---

## 8. Condensation / Wet-Surface Cases

The psychrometric `MoistAirState` path alone provides onset/limit helpers,
not a full wet exchanger solve. For the current 0D wet-gas solver based on
an explicit `GasMixtureSpec`, see section 15.

### 8.1 What is available in the psychrometric helper path

Current helpers can support:

```text
dew point calculation in psychrometric range,
condensation onset check,
surface saturation limit,
condensable water limit,
enthalpy-drop limit to surface saturation.
```

These are helper calculations, not a complete heat exchanger solver.

Available conceptual tools:

```python
check_condensation_onset(...)
wet_surface_process_limit(...)
```

Use them only when the gas can be represented by the psychrometric moist-air path.

### 8.2 What is not available in that helper path

Not implemented by the psychrometric helpers themselves:

```text
segment-by-segment wet heat exchanger calculation,
condensation rate along tube rows,
latent heat balance along the bundle,
change of humidity ratio along the heat exchanger,
change of gas composition due to water removal,
condensate film resistance,
wet fin/tube surface response (the v0.7.5 solver requires `GasMixtureSpec`),
acid dew point,
sulfuric acid condensation,
mixed water/acid condensation,
fog/mist formation,
two-phase gas/liquid pressure drop,
drainage model.
```

### 8.3 When condensation is suspected

Use this decision rule:

| Situation                                               | Recommended action                                           |
| ------------------------------------------------------- | ----------------------------------------------------------- |
| Moist air below `200 °C`, no condensation expected      | Use `MoistAirState` + moist-air transport                   |
| Moist air below `200 °C`, condensation onset check only | Use `check_condensation_onset()`                            |
| Moist air below `200 °C`, wet process rough limit       | Use `wet_surface_process_limit()`                           |
| Hot humid gas above `200 °C`, clearly no condensation   | Use `GasMixtureSpec` with H2O gas component                 |
| Hot humid gas may cool through dew point                | Do not rely on `GasMixtureSpec` alone                       |
| Flue gas may condense water                             | Wet economizer solver required; not in v0.4.5               |
| Flue gas may reach acid dew point                       | Not supported in v0.4.5                                     |
| Condensation is central to duty                         | Do not use sensible-only gas mixture result as final design |

---

## 9. Temperature Ranges and Recommended Model Choice

| Temperature range | Gas type    | Recommended path                   | Notes                                       |
| ----------------: | ----------- | ---------------------------------- | ------------------------------------------- |
|        `< -40 °C` | moist air   | PsychroLib with caution            | check validity carefully                    |
|    `-40 … 100 °C` | moist air   | PsychroLib / `MoistAirState`       | normal practical range                      |
|    `100 … 200 °C` | moist air   | PsychroLib only with caution       | RH/saturation may become physically limited |
|        `> 200 °C` | humid air   | explicit `GasMixtureSpec`          | do not use PsychroLib state                 |
|    `200 … 800 °C` | dry/wet gas | explicit `GasMixtureSpec`          | sensible gas-phase only                     |
|        `> 800 °C` | gas         | explicit mixture, validate backend | property accuracy must be verified          |

Important physical note:

At atmospheric pressure, saturation-based psychrometric concepts become problematic above the boiling point of water. A gas at `120 °C` and `1 atm` may contain water vapor, but a saturated moist-air state at that same `T,p` is not necessarily defined in the same way as ordinary HVAC psychrometrics. Therefore, high-temperature water-containing gases should usually be treated as explicit gas mixtures with H2O vapor as a component.

---

## 10. Humidity Ranges and Recommended Model Choice

| Moisture input                |   Approximate range | Recommended path                    |
| ----------------------------- | ------------------: | ----------------------------------- |
| `RH = 0 … 1`, moderate T      |         normal HVAC | PsychroLib                          |
| `W = 0 … 0.03 kg/kg_da`       |    common moist air | PsychroLib                          |
| `W = 0.03 … 0.10 kg/kg_da`    |       high humidity | PsychroLib with warning / check     |
| `W > 0.10 kg/kg_da`           |  very high humidity | prefer explicit gas mixture if hot  |
| `H2O` as mole/volume fraction | any gas composition | `GasMixtureSpec`                    |
| `H2O` as mass fraction        | any gas composition | `GasMixtureSpec(..., basis="mass")` |

For `v0.4.5`, `W > 0.10 kg/kg_da` should be treated as a warning threshold, not an absolute physical limit.

---

## 11. CoolProp, HEOS, and REFPROP Notes

### HEOS

`HEOS` is the default open CoolProp backend.

Use it for:

```text
pure fluids,
pseudo-pure fluids,
selected gas mixtures,
development and open-source testing.
```

Caution:

```text
User-defined mixtures may fail for some component combinations, state points,
or transport-property calls.
```

CoolProp mixture calculations use mixture models that depend on available binary interaction data. User-defined mixtures can require binary-pair data and may fail if data or solver convergence is insufficient.

For gas mixtures, KalKalori uses:

```python
imposed_phase="gas"
```

This avoids some automatic PT-flash failures for known gas-phase engineering states.

### REFPROP

`REFPROP` is optional.

Use it when:

```text
higher accuracy is required,
HEOS fails for an important mixture,
validated industrial property data is required,
commercial licensing is acceptable.
```

REFPROP is not distributed with KalKalori. It must be installed, licensed, and configured locally by the user. CoolProp can interface with REFPROP if the local REFPROP installation is available.

---

## 12. Practical Rules for Heat Exchanger Cases

### Case A — Dry air in tubes, humid air outside, no condensation

Use:

```text
inside: GasMixtureSpec for dry air or CoolProp Air
outside: MoistAirState if T <= 200 °C
outside: GasMixtureSpec with H2O if T > 200 °C
```

### Case B — Dry gas in tubes, hot wet gas outside, no condensation

Use:

```text
inside: GasMixtureSpec
outside: GasMixtureSpec with H2O
```

Assume:

```text
gas phase only,
constant composition,
sensible heat transfer only.
```

### Case C — Gas cools close to water dew point

Use `GasMixtureSpec` with explicit H2O and the default
`PhaseChangeMode.AUTO`. The solver checks the minimum wall temperature;
the bulk outlet need not cool below the dew point.

Recommended procedure:

```text
1. Build the wet gas from its dry composition and H2O content.
2. Run Simulation or Rating with `PhaseChangeMode.AUTO`.
3. Inspect the side-scoped `PhaseChangeResult` and its assumptions/warnings.
```

### Case D — Condensing economizer / wet surface

Water condensation from wet gas is supported inside tubes with a dry outside
surface and outside bare-tube banks since v0.6.1. v0.7.5 also supports the
outside surface of `CircularFinnedTube`, including independent primary/root
and radial annular-fin condensation, with these 0D limits:

```text
one active side,
partial H2O condensation,
fully drained condensate,
gas-phase-only hydraulics.
```

### Case E — Flue gas with possible acid dew point

Not supported.

Do not use current water-only moist-air helpers to assess acid condensation.

---

## 13. Recommended User Responsibility

KalKalori intentionally keeps property-model selection explicit.

The user must decide:

```text
Is this classical moist air?
Is this dry gas?
Is this gas with water vapor but no condensation?
Is this a condensing case?
Is this water/steam?
Is this a real-gas mixture requiring REFPROP?
```

KalKalori should provide clear tools and warnings, but it should not silently convert one physical model into another.

---

## 14. Summary

Use `MoistAirState` / PsychroLib when:

```text
the case is classical psychrometric moist air,
temperature is within PsychroLib range,
RH/dew point/saturation are meaningful,
classical moist-air state calculations are needed.
```

Use `GasMixtureSpec` when:

```text
the gas composition is known,
the gas is dry,
the gas is hot,
the gas contains water vapor but remains gas-phase,
the calculation is sensible-only,
gas properties are needed inside or outside tubes.
```

Use `GasMixtureSpec` together with the wet-gas phase-change solver when:

```text
H2O condenses from a gas with a non-condensable carrier,
latent heat and condensate removal must be included,
the wet gas flows inside tubes with a dry outside surface or outside bare or
circular-finned tube banks.
```

Pure-water phase change inside tubes uses the separate v0.6.2/v0.6.3
water/steam models described in sections 16 and 17. Acid dew points,
multiple condensables and liquid-inventory evaporation carried by a gas
require later models; do not represent them by silently changing the
H2O-only wet-gas model.

---

## 15. v0.6.1 — Wet-Gas Water Condensation

The wet-process / wet-economizer solver referenced as future work in
sections 8 and 14 above is available for **partial H2O condensation from a
`GasMixtureSpec`-based gas mixture flowing either inside tubes with a dry
outside surface or outside a bare-tube bank** since v0.6.1. v0.7.5 extends the
outside route to a **`CircularFinnedTube` primary/root surface and annular
fins**, solved by
`BareTubeHeatExchanger.simulate()` / `.rate()`. It does not replace the
property-path decision tree above --
you still choose `GasMixtureSpec` (with H2O as a component) exactly as
sections 4 and 7 describe. What changes is what the *heat-exchanger
solver* does with that composition once the wall runs below the dew point.

For the v0.7.5 finned outside route, the dry Briggs--Young physical film HTC
still defines sensible convection and the existing Chilton--Colburn path
defines H2O mass transfer. A deterministic nonlinear radial finite-volume
solve resolves both fin faces and tip as dry, partially wet or fully wet while
the exposed primary/root cylinder condenses independently. The nested
`WetFinnedSurfaceResult` reports primary and fin sensible/latent/total duties,
condensate rates, wet areas, temperatures, contact topology, convergence and
split residuals. The generic `outside_phase_change` result remains
authoritative for the whole-side water and energy balance. A separately named
`outside_alpha_wet_effective_gross_core_basis` is a total-duty reconstruction;
`outside_alpha_physical` is never redefined to include latent heat. Equations,
area bases, wet-boundary convention and references are documented in
[`finned_tube_model.md`](finned_tube_model.md#wet-annular-fin-condensation-v075).

### 15.1 `imposed_phase` vs. `PhaseChangeMode` -- two different knobs

These control two unrelated things and must not be confused:

- **`GasMixtureSpec.imposed_phase`** (default `"gas"`) tells the CoolProp
  backend how to evaluate *properties at one (T, p) state point* -- it
  avoids PT-flash failures for known gas-phase states (section 2 above).
  It says nothing about whether the exchanger solver removes water from
  the stream.
- **`PhaseChangeMode`** (`core.phase_change.types`, set per side via
  `HXSideInput.phase_change_mode` / `BalanceSideSpec.phase_change_mode`,
  default `AUTO`) tells the *heat-exchanger solver* whether it is allowed
  to solve an active condensation case.

Setting `imposed_phase="gas"` does **not** disable solver-level
condensation, and setting `PhaseChangeMode.DISABLED` does **not** change
how `imposed_phase` is used for property evaluation. A capable medium
(H2O present) with `imposed_phase="gas"` and `PhaseChangeMode.AUTO` (the
default for both) is exactly the intended v0.6.1 configuration.

### 15.2 Capability vs. possibility vs. activity

`core.phase_change.capability.detect_phase_change_capability` is the
single place that decides whether a provider is phase-change *capable*:
currently, a `GasMixturePropertyProvider` whose spec has a positive H2O
mole/volume/mass fraction (with a nonzero non-condensable dry-gas
remainder). Capability alone never changes solver behavior.

Given a capable side, the solver runs an ordinary sensible-only ("dry")
baseline first, and only from that baseline's wall temperature vs. the
medium's own dew point decides whether condensation is **possible** at
this operating point. Only when it is possible *and* `PhaseChangeMode.AUTO`
is set does the solver actually make phase change **active** and run the
coupled side-specific solve. A capable
medium that never reaches its dew point produces an ordinary sensible-only
result under `AUTO`, bit-for-bit identical to not having this feature at
all. A small activation band straddling the dew-point margin resolves to a
**near-onset** result (`possible=True`, `active=False`,
`near_onset=True`) instead of oscillating between the dry and wet solve on
repeated calls close to onset.

`active=False` -- whether dry, near-onset, or a not-capable side -- is
always a valid, converged result, never a calculation failure. Its
`PhaseChangeResult.Q_sensible`/`Q_total` equal the real exchanger duty
(`Q_latent = 0`, `m_dot_condensate = 0`) exactly like `DISABLED` below, so a
caller never has to check `active` before reading them.

### 15.3 `PhaseChangeMode.DISABLED`

Forces the sensible-only ("dry") result on that side even when
condensation would be possible: H2O is not removed from the stream,
`m_dot_condensate = 0`, `Q_latent = 0`, the outlet water content equals the
inlet water content. The solver still checks whether condensation *would*
have been possible from the dry baseline and attaches a
`PHASE_CHANGE_DISABLED_BUT_POSSIBLE` warning when it would -- this is a
forced approximation, not a "this case has no condensation" statement; the
true duty and outlet temperature may differ from the dry result.

### 15.35 Rating with a single unknown non-condensing-side variable (v0.7.5)

Active outside condensation does not inherently require every one of the
non-condensing (typically inside) side's `m_dot`/`T_in`/`T_out` to be
explicit. Exactly as for a dry `Rating` call, `BalanceSideSpec` may still
leave a single supported unknown --  `m_dot` (with `T_out` known) or
`T_out` (with `m_dot` known) -- for the solver to close, provided the wet
outside side is itself fully specified (`m_dot`, `T_in`, `T_out`).

Unlike the dry case, this cannot be recovered by one more algebraic
`close_heat_balance` step: the wet outside stream's own duty is not pinned
down by its `(T_in, T_out, m_dot)` alone once condensation is active (how
much water actually condenses is an extra physical degree of freedom, set
only by the mass-transfer-coefficient-driven physics at the real wall
temperature, which itself depends on the unknown flow through the inside
film coefficient). `core.phase_change.rating_integration` instead runs an
outer, bounded, bracketed scalar root search over the single unknown,
re-evaluating the whole Rating call at each trial and comparing the
resulting nonlinear wet-fin surface's own physically-driven duty against
the duty the trial's inside energy balance requires. This is currently
implemented only for a `CircularFinnedTube` outside surface (a bare-tube
outside surface does not yet expose an equivalent independent raw duty and
raises a clear, typed error if this closure is attempted on one).

Each trial may resolve dry, near-onset or actively condensing -- AUTO
legitimately crosses regime boundaries during the search, and a dry/near-
onset trial is not treated as a search failure. A genuinely underdetermined
side (both `m_dot` and `T_out` missing) is still rejected, and a target
that cannot be reached by any physically valid flow/temperature raises a
diagnostic `RatingClosureError` (no fabricated "closest" result is
returned). This search is materially more expensive than a plain Rating
call (each trial re-runs the full nonlinear wet-fin solve); it is not
attempted unless exactly one supported unknown triggers it.

### 15.4 Scope (v0.6.1/v0.7.5)

`GasMixturePropertyProvider` represents the gas phase, not a pre-existing
liquid-water inventory transported with that gas.  Public Simulation and
Rating calls therefore compare the specified inlet water ratio with the
equilibrium vapor capacity before evaluating gas-mixture properties.  An
exactly saturated inlet (within `1e-10 kg/kg` absolute plus `1e-8` relative
tolerance) is accepted; a composition above that capacity is rejected with
`LIQUID_WATER_IN_GAS_INLET_NOT_SUPPORTED`.  Droplet, mist, wall-film and
  other carried-liquid evaporation are not included in the wet-gas model.

- Only H2O condenses from a wet gas with a non-condensable dry carrier;
  either the **inside** or **outside** stream may be active.
- This wet-gas solver supports only **partial** condensation
  (0 <= W_out <= W_in). Full pure-steam condensation, vapor quality and
  condensate subcooling use the separate inside-tube model in section 16.
  Pure-steam condensation outside tubes is not in the planned scope.
- At most **one** active phase-changing side per call
  (`MULTIPLE_PHASE_CHANGE_SIDES_NOT_SUPPORTED` otherwise).
- Onset uses the **minimum** side-wall temperature. The bulk outlet gas and
  the mean wall do not need to reach the dew point for local surface
  condensation to be active.
- Condensate is treated as **fully drained**, leaving as saturated liquid
  at the representative wet-wall temperature -- it is never re-added to
  the gas-phase stream or its hydraulics.
- The model remains **0D**: no axial/segmented resolution. `.simulate()`
  requires `iterate=True` for an active condensation case (the
  `iterate=False` single-pass escape hatch cannot represent condensation).
- For a bare surface, `wet_surface_fraction` remains the linear 0D wall-
  envelope estimate: sensible convection uses the full side area while
  latent/mass transfer uses the estimated `A_wet` and representative wet-wall
  temperature. For a circular-finned outside surface, the fraction instead
  comes from the nonlinear 160-cell radial fin field plus the independently
  solved primary surface. Its typed state is `DRY`, `PARTIALLY_WET` or
  `FULLY_WET`. If endpoint onset is active but that one bulk-mean radial field
  is wholly dry, a declared fallback applies the existing linear 0D endpoint
  fraction and representative cold-zone temperature to mass/latent transfer;
  it does not add axial marching or use rows/passes as thermal segments. See
  [`finned_tube_model.md`](finned_tube_model.md#wet-annular-fin-condensation-v075).
- The Chilton-Colburn heat/mass-transfer analogy uses `lewis_number=1.0`
  by default -- a configurable first-model assumption, not a universal
  constant (see `docs/references.md`).
- Condensate film thermal resistance, film hydraulics, carryover/
  re-entrainment, evaporation, and freezing/frost are not modelled (frost
  conditions are detected and rejected with `FROSTING_NOT_SUPPORTED` rather
  than silently clamped).
- Two-phase gas/liquid pressure drop and condensate-film resistance are not
  modelled. Inside and bare-outside hydraulic point states use only the
  remaining gas-phase `m_dot`; friction/acceleration therefore reflect gas
  composition, density and gas-flow changes without adding liquid-film
  momentum. During circular-finned outside condensation, the existing dry
  finned-bank pressure drop is retained only as `outside_dp_dry_reference`;
  `wet_pressure_drop_supported` is false, `outside_dp_reference_only` is true,
  and the
  `circular_finned_tube_wet_pressure_drop_reference_only` warning is emitted.

See `core/phase_change/` module docstrings for the equilibrium, enthalpy,
and heat/mass-transfer model details, and
`core/tests/outside_water_condensation_examples.ipynb` and
`core/tests/inside_water_condensation_examples.ipynb` for worked examples.

---

## 16. v0.6.2 — Pure Water/Steam Cooling Inside Tubes

Pure H2O uses pressure/enthalpy phase equilibrium, not the wet-gas water
ratio, dew-point or heat/mass-transfer model. The public
`BareTubeHeatExchanger.simulate()` and `.rate()` paths accept an
`IAPWS97WaterSteamProvider` on the inside side with exactly one inlet state
specification:

```python
HXSideInput(provider=steam, p=p, T_in=T, ...)
HXSideInput(provider=steam, p=p, quality_in=x, ...)
HXSideInput(provider=steam, p=p, h_in=h, ...)
```

`T_in + p` is rejected on the saturation line because it cannot distinguish
saturated liquid, saturated vapor and a two-phase mixture. Use `quality_in`
or `h_in` there. Quality is the vapor mass fraction and is defined only in
the saturation dome: `x=0` is saturated liquid, `x=1` is saturated vapor,
and `0<x<1` is wet steam. Superheated vapor and subcooled liquid have
`quality=None`. IAPWS T+p and p+h states remain available above the critical
pressure when IF97 supports the state; p+x and saturation helpers are
rejected at and above the critical point.

`IAPWS97WaterSteamProvider` is the supported provider for pure-water phase
change. `CoolPropFluidProvider("Water")` remains a CoolProp calculation for
single-phase water, without backend substitution. Likewise, a pure-H2O
`GasMixturePropertyProvider` is never routed through the wet-gas model. If
either non-IAPWS representation would need a phase crossing or a quality/
enthalpy phase-state input, the call raises the controlled
`PURE_WATER_PHASE_CHANGE_PROVIDER_NOT_SUPPORTED` error instead of silently
switching to IAPWS.

For Rating, the equivalent outlet fields are `T_out`, `quality_out` and
`h_out`; an explicit `Q` or a fully specified opposing-side temperature
program may also define duty. Effectiveness-only steam Rating is not
supported because the phase-changing stream has no single sensible capacity
rate. The steam-side cooling direction may contain any non-empty subset
of the ordered 0D zones:

```text
SUPERHEAT -> CONDENSATION -> SUBCOOLING
```

This order describes the **tube-side thermodynamic path only**. Since v0.7.6,
the intended crossflow steam-air-heater geometry couples the outside stream to
those zones in parallel. Every active zone receives the same global outside
inlet state; no zone receives the outlet state of the preceding tube-side
phase zone. The branch outlets are mixed after the zone calculations.

For geometry that is uniform along tube length, the geometric allocation is
the converged required gross-outside-area fraction, equivalently the
represented tube-length and frontal-area fraction:

```text
f_z             = A_required,z / sum(A_required,z)
m_dot_outside,z = f_z * m_dot_outside,total
A_frontal,z     = f_z * A_frontal,total
L_z / L_total   = f_z
```

Thus `m_dot_outside,z / A_frontal,z` equals the full-exchanger face mass flux.
The whole-bank outside correlation and its physical film coefficient are
preserved; partitioning the air flow does not spuriously lower local velocity
or outside HTC. The branch mass flow is used with its corresponding frontal
fraction for its energy balance and sensible capacity rate.

The fractions are required-area fractions, not heat-duty fractions. The
solver uses a deterministic bounded fixed-point allocation: it assigns branch
flow from trial fractions, evaluates all zones from the common outside inlet,
normalizes their required areas, and iterates until the flow and area
fractions agree. Iteration and residual limits are explicit, and an
unconverged allocation is not returned as successful. With one active zone,
`f_z = 1` and the method reduces to the unchanged single-zone calculation.

Each branch outlet uses the existing outside-property convention. The mixed
outlet is recovered with the same convention from the total sensible-energy
balance, with branch and global closure checked as

```text
sum(m_dot_outside,z * delta_h_outside,z)
    ~= m_dot_outside,total * delta_h_outside,mixed
    = Q_total
```

For a constant-`cp` outside fluid, this is the usual mass-weighted outlet
temperature. The steam-zone correction does not introduce a competing fluid
enthalpy model.

Each zone reports its own duty, outer-reference area, area/tube-length
fraction, outside mass flow and fraction, frontal area, face mass flux or
velocity, outside inlet/outlet temperatures, inside/outside HTC, `U`, and
`UA`. `zone_alpha_condensation` is the physical Shah (2009) condensation
coefficient and remains the value used to validate the condensation
correlation. `inside_alpha_area_weighted` is the arithmetic area-weighted
mean of the zone HTCs. It is a descriptive statistic only and is not used to
reconstruct `U`, `UA`, or wall temperatures.

`inside_alpha_equivalent` is the resistance-consistent whole-exchanger HTC.
It is obtained by inverting the same authoritative-area resistance network
used for every zone:

```text
R_i,equivalent = 1/(U_equivalent A_outside,gross)
                 - R_wall - R_outside
inside_alpha_equivalent = 1/(R_i,equivalent A_inside)
R_outside = 1/(outside_alpha_effective_gross A_outside,gross)
```

Here `U_equivalent = UA_total / A_total`, while the authoritative conductance
remains `UA_total = sum(U_zone * A_zone)`. For a plain tube this reduces to
the historical diameter-basis expression. For a dry circular-finned outside,
`R_outside` includes the physical film, fin efficiency, contact topology and
root layer through the shared resistance network. The historical top-level
`alfa_i`/`inside_alfa_mean` fields now expose `inside_alpha_equivalent` for
multi-zone steam so that they reconstruct `U_equivalent`. Their semantics are
unchanged for sensible-only calculations, wet-gas condensation, and
non-water fluids. The zone allocation, zone alphas, zone `U` values, and zone
`UA` values do not depend on either top-level alpha diagnostic.

Steam wall-temperature diagnostics use `inside_alpha_equivalent` in their
inside resistance split. The wall-temperature envelope remains a four-point
0D inlet/outlet estimate, not a local segmented solution. If an inside
Nusselt diagnostic can be formed from representative transport properties,
it is the dimensionless equivalent HTC for reporting; it must not be read as
a local Shah value or as the actual Nusselt number of a particular zone.

The transport-only Shah equations live in
`core.heat_transfer.condensation_inside_shah2009`; the IAPWS saturation
adapter and eight-point, area-consistent harmonic quality integration live
in `core.phase_change.steam_condensation`. The equivalent zone coefficient
used inside the condensation zone does not integrate a local `1/U` profile;
that existing zone-level limitation is distinct from the whole-exchanger
`inside_alpha_equivalent` described above.

Tube orientation is geometry, supplied as `BareTube(tube_orientation=...)`
with a `TubeOrientation` value. It is required only when the accepted result
actually contains a condensation zone, not for superheated-to-superheated or
saturated-liquid-to-subcooled calculations. Supported values are horizontal,
vertical downward, and downward inclined by at least 15 degrees.

The result is a typed `WaterSteamPhaseChangeResult`. Endpoint properties
come from the final pressure/enthalpy solution and contain at least `T`, `p`,
`h`, phase and quality; a two-phase endpoint does not invent single-phase
transport properties. It exposes stable physics and convergence fields, not
the internal `SteamHeaterSolution` object or wall-clock runtime. Steam
Simulation and Rating construct their diagnostics directly from the shared
zone solution. v0.7.6 diagnostics identify the allocation as
`parallel_by_geometry`, report its iteration count, convergence and residual,
sum the zone area fractions and air mass flows, and expose the mixed outside
outlet and air-energy residual. The accepted outside temperature program is
evaluated by the neutral `core.heat_transfer.outside_side` helper, while every
trial duty in the steam solver still updates and caches its own outside mean
properties and HTC. No fake tube-side fluid or sensible full-HX solve is used.

`PhaseChangeMode.DISABLED` is allowed while the result stays in one phase,
but raises the controlled
`PHASE_CHANGE_DISABLED_BUT_REQUIRED` error if the required solution crosses
the saturation dome.

Cooling-model limits:

- this v0.6.2 path covers cooling/condensation; the heating direction uses
  the v0.6.3 model in section 17;
- the opposing outside surface may be bare or circular-finned while that side
  remains dry; simultaneous active outside wet-gas condensation is unsupported
  because only one exchanger side may have active phase change;
- pure-steam phase change is supported only inside tubes and is outside the
  planned scope on the outside side;
- one active phase-changing side per exchanger call;
- the model remains global multi-zone 0D: converged area/tube-length fractions
  allocate parallel outside branches but are not resolved axial phase-front
  positions, row-by-row temperatures, or longitudinal 1D segmentation;
- no falling-film, liquid-level, drainage-hydraulic, flooded-pipe or general
  1D steam-condensation model is included;
- the Shah (2009) applicability diagnostics warn without clipping or
  calibration outside the published range;
- two-phase pressure drop is not supported and is returned explicitly as
  unavailable rather than as a partial single-phase tube-side pressure drop.

See `core/tests/steam_condensation_examples.ipynb` for public Simulation
examples covering saturated, wet, superheated, subcooled and low-mass-flux
cases. See `docs/steam_heater_zone_driving_force.md` for the detailed zone
driving-force, parallel-air allocation, outlet-mixing and 0D-scope contract.

---

## 17. v0.6.3 — Pure-Water Heating and Evaporation Inside Tubes

The public `BareTubeHeatExchanger.simulate()` and `.rate()` APIs support
pure water heated inside tubes, including a dry circular-finned outside
surface, through the ordered constant-pressure
p-h zones:

```text
PREHEAT -> EVAPORATION -> SUPERHEAT
```

Any non-empty subset is allowed: subcooled water may remain liquid, reach a
partial quality, evaporate completely to `x=1`, or continue into superheated
vapor. A wet inlet with `0 <= x < 1` may leave at a higher quality or as
superheated vapor. Total pure-water mass flow is conserved and
`Q_total = m_dot * (h_out - h_in)`; excess heat after `x=1` is never clipped.

The state and provider rules are shared with section 16. Inlet states use
exactly one of `T_in+p`, `h_in+p`, or `quality_in+p`; Rating targets use one
of `T_out+p`, `h_out+p`, or `quality_out+p`, or receive duty from explicit
`Q` or the opposing temperature program. `T+p` exactly on the saturation
line remains ambiguous and is rejected. `p+x` and saturation helpers remain
invalid at and above the critical pressure, while supported supercritical
`T+p` and `p+h` states stay single-phase. Pure-water phase change is
supported only with `IAPWS97WaterSteamProvider`; CoolProp Water and pure-H2O
gas-mixture providers are not silently replaced.

Simulation routes to the evaporator only when the full geometry can reach
saturated liquid. A preheat-only liquid case retains the established
sensible result and tube hydraulics. For the IAPWS path, `capable=True`
describes the medium; `possible=True` means the full geometry can reach
evaporation; `active=True` means the accepted (including surface-margin
derated) result contains an EVAPORATION zone. Thus a strongly derated result
may be capable and possible but not active. `PhaseChangeMode.DISABLED`
allows a final single-phase result, but raises
`PHASE_CHANGE_DISABLED_BUT_REQUIRED` rather than extrapolating liquid `cp`
through a required boiling zone.

The transport boundary is
`core.heat_transfer.evaporation_inside_shah1982`. It implements Shah's
(1982) saturated, pre-dryout flow-boiling correlation for horizontal and
vertical-upward tubes. Pressure, critical pressure, inner diameter, mass
flux, quality, inside-wetted-area heat flux, endpoint densities, liquid
transport data and latent heat are explicit SI inputs. Orientation remains
a tube-geometry property and is required only for an active evaporation
zone. Downward and inclined-downward boiling are rejected because those
variants are not supplied by the selected primary equations.

`core.phase_change.water_evaporation` obtains one IAPWS saturation snapshot
and uses eight-point Gauss-Legendre nodes strictly inside each quality
interval, so the singular `x=0` and `x=1` endpoints are never sent to the
local correlation. The zone HTC is the duty/quality-weighted harmonic value:

```text
alpha_zone = 1 / mean(1 / alpha_local)
```

This follows `dA = dQ / (alpha * deltaT)` and is not an arithmetic HTC.
Because Shah (1982) depends on boiling number, the multi-zone solver obtains
the inside-area heat flux from a bounded fixed point against the wall and
outside resistances. It reports iterations, residual and convergence; no
arbitrary heat-flux correction factor is used.

Each zone reports duty, inside/outside HTC, outer-area `U`, outer area and
`UA`. The authoritative totals are:

```text
UA_total = sum(U_zone * A_zone)
U_equivalent = UA_total / A_total
```

`zone_alpha_evaporation` is the physical boiling-zone value used to validate
Shah (1982). Top-level `inside_alpha_equivalent` is instead obtained by
inverting the complete outer-area resistance network so it reconstructs
`U_equivalent`; `inside_alpha_area_weighted` is descriptive only. Every
trial duty updates the opposing outlet/mean temperature, current properties,
flow coefficient and hydraulics through the neutral outside-side evaluator.
Repeated IAPWS and outside states are cached.

Rating uses the same p-h partition, correlation, quality integration,
heat-flux solve and zone-U calculation as Simulation. It reports required
area/UA, actual scaled UA, overdesign and UA margin without recalculating
physics through an arithmetic alpha. Effectiveness-only Rating is rejected
for an active phase-changing stream because no unique sensible capacity rate
defines its duty.

Current limits:

- pure-water evaporation is supported only inside tubes;
- the opposing outside surface may be bare or circular-finned while that side
  remains dry; simultaneous active outside wet-gas condensation is unsupported
  because only one exchanger side may have active phase change;
- at most one side may have active phase change;
- Shah (1982) does not predict CHF, dryout quality or post-dryout heat
  transfer; warnings expose those applicability limits without clipping;
- two-phase tube pressure drop is unsupported: active boiling returns no
  complete tube-side hydraulic result and generic inside-dp fields are NaN;
- the model is 0D; zone fractions are thermal area allocations, not resolved
  axial phase-front locations;
- gas-mixture providers describe gas phase only. A supersaturated inlet is
  rejected by `LIQUID_WATER_IN_GAS_INLET_NOT_SUPPORTED`; droplet/mist,
  wall-film, condensate re-evaporation, drainage, carryover and re-entrainment
  are not modelled.

See `core/tests/water_steam_evaporation_examples.ipynb` for executed public
Simulation, Rating and duty-controlled examples.
