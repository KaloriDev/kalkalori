# Property Model Selection Guide

This document explains how to select property models in KalKalori `v0.4.x`.

The goal is to avoid hidden assumptions. KalKalori does not automatically decide whether a fluid should be treated as classical moist air, dry gas, wet gas, condensing gas, steam, or water. The user must select the appropriate property path.

---

## 1. Available Property Paths in v0.4.x

| Property path              | Main API                                       | Intended use                                                           | Condensation support              |
| -------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------- |
| Dry air | `dry_air_props()`, `DryAirPropertyProvider` | Standard dry air properties for sensible gas-side calculations | No |
| Psychrometric moist air    | `MoistAirState`, PsychroLib adapter            | Classical moist air, RH, dew point, saturation, HVAC-like calculations | Onset / limit helpers only        |
| Moist-air transport helper | `moist_air_transport_props_from_state()`       | Transport properties of moist air in normal engineering range          | No full wet HX solver             |
| Water/steam                | `water_steam_props_iapws97()`                  | Water, saturated water, steam, saturated steam                         | Water/steam phase input supported |
| CoolProp pure fluid        | `CoolPropFluidProvider`                        | Pure fluids and pseudo-pure fluids                                     | Backend dependent                 |
| Explicit gas mixture       | `GasMixtureSpec`, `GasMixturePropertyProvider` | Dry gases, flue gas, hot humid gas with H2O as gas-phase component     | No condensation                   |
| Constant properties        | `ConstantPropertyProvider`                     | Debugging, reference calculations, fixed-property cases                | No                                |

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
water_steam_props_iapws97(T=..., x=...)
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
| Condensation         |             not supported | do not use for wet-surface heat balance       |

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

KalKalori `v0.4.5` does not yet provide a full wet economizer solver.

### 8.1 What is available

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

### 8.2 What is not available yet

Not implemented in `v0.4.5`:

```text
segment-by-segment wet heat exchanger calculation,
condensation rate along tube rows,
latent heat balance along the bundle,
change of humidity ratio along the heat exchanger,
change of gas composition due to water removal,
condensate film resistance,
wet fin/tube surface correction,
acid dew point,
sulfuric acid condensation,
mixed water/acid condensation,
fog/mist formation,
two-phase gas/liquid pressure drop,
drainage model.
```

### 8.3 When condensation is suspected

Use this decision rule:

| Situation                                               | Recommended action in v0.4.5                                |
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

Use `v0.4.5` only for preliminary checks.

Recommended procedure:

```text
1. Calculate sensible-only case using GasMixtureSpec.
2. Estimate outlet temperature.
3. Independently check whether water dew point may be reached.
4. If dew point may be reached, do not accept sensible-only result as final.
5. Use future wet economizer solver or external validated method.
```

### Case D — Condensing economizer / wet surface

Not supported as final design in `v0.4.5`.

Allowed only:

```text
onset checks,
rough limits,
development tests,
comparison against external validated tools.
```

### Case E — Flue gas with possible acid dew point

Not supported in `v0.4.5`.

Do not use current water-only moist-air helpers to assess acid condensation.

---

## 13. Recommended User Responsibility

KalKalori `v0.4.5` intentionally keeps property-model selection explicit.

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
condensation is only checked as onset or limit.
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

Do not use `GasMixtureSpec` alone when:

```text
water condenses,
latent heat dominates,
composition changes due to condensate removal,
surface is wet,
acid dew point matters.
```

Do not add a misleading `HumidGasProvider` in `v0.4.5`.

High-temperature humid gas should remain an explicit user-defined gas mixture until a real wet-process / wet-economizer solver is implemented.

---

## 15. v0.6.0 — Outside Water Condensation

The wet-process / wet-economizer solver referenced as future work in
sections 8 and 14 above is now available, scoped as follows: **partial
H2O condensation from a `GasMixtureSpec`-based gas mixture flowing outside
a bare-tube bank**, solved by `BareTubeHeatExchanger.simulate()` /
`.rate()`. It does not replace the property-path decision tree above --
you still choose `GasMixtureSpec` (with H2O as a component) exactly as
sections 4 and 7 describe. What changes is what the *heat-exchanger
solver* does with that composition once the wall runs below the dew point.

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
default for both) is exactly the intended v0.6.0 configuration.

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
coupled solve (`core.phase_change.outside_condensation_solver`). A capable
medium that never reaches its dew point produces an ordinary sensible-only
result under `AUTO`, bit-for-bit identical to not having this feature at
all.

### 15.3 `PhaseChangeMode.DISABLED`

Forces the sensible-only ("dry") result on that side even when
condensation would be possible: H2O is not removed from the stream,
`m_dot_condensate = 0`, `Q_latent = 0`, the outlet water content equals the
inlet water content. The solver still checks whether condensation *would*
have been possible from the dry baseline and attaches a
`PHASE_CHANGE_DISABLED_BUT_POSSIBLE` warning when it would -- this is a
forced approximation, not a "this case has no condensation" statement; the
true duty and outlet temperature may differ from the dry result.

### 15.4 Scope (v0.6.0)

- Only H2O condenses; only the **outside** stream may have an active
  phase-changing side (inside condensation is detected but rejected with
  `INSIDE_CONDENSATION_NOT_SUPPORTED` under `AUTO` -- see the roadmap for
  v0.6.1).
- Only **partial** condensation (0 <= W_out <= W_in); full steam
  condensation is out of scope.
- At most **one** active phase-changing side per call
  (`MULTIPLE_PHASE_CHANGE_SIDES_NOT_SUPPORTED` otherwise).
- Condensate is treated as **fully drained**, leaving as saturated liquid
  at a representative interface temperature -- it is never re-added to the
  gas-phase stream or its hydraulics.
- The model remains **0D**: no axial/segmented resolution. `.simulate()`
  requires `iterate=True` for an active outside-condensation case (the
  `iterate=False` single-pass escape hatch cannot represent condensation).
- `wet_surface_fraction` is a 0D endpoint estimate (compares the existing
  four-probe wall-temperature envelope against the local dew point), not a
  spatially resolved wetted-area fraction.
- The Chilton-Colburn heat/mass-transfer analogy uses `lewis_number=1.0`
  by default -- a configurable first-model assumption, not a universal
  constant (see `docs/references.md`).
- Condensate film thermal resistance, film hydraulics, carryover/
  re-entrainment, evaporation, and freezing/frost are not modelled (frost
  conditions are detected and rejected with `FROSTING_NOT_SUPPORTED` rather
  than silently clamped).
- Two-phase gas/liquid pressure drop is not modelled; the outside gas-phase
  hydraulics contract gains an optional per-point gas-phase mass-flow input
  (`core.heat_transfer.outside_flow.calculate_outside_tube_bank_hydraulics`,
  `m_dot_inlet`/`m_dot_midpoint`/`m_dot_outlet`) so the signed acceleration
  term reflects the reduced gas mass flow once condensate has been removed,
  without adding any liquid-phase hydraulics.

See `core/phase_change/` module docstrings for the equilibrium, enthalpy,
and heat/mass-transfer model details, and
`core/tests/outside_water_condensation_examples.ipynb` for a worked
example.
