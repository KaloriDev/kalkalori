# Thermal fouling resistance

`BareTubeHeatExchanger` accepts two independent optional exchanger parameters:

```python
hx = BareTubeHeatExchanger(
    bundle,
    fouling_resistance_inside=0.0002,   # m² K/W
    fouling_resistance_outside=None,    # m² K/W; None = 0
)
```

Both are conventional area-specific thermal resistances, **m² K/W**.
Omitting either parameter, supplying `None`, or supplying `0.0` gives the
same clean circuit. Negative values, NaN and infinity raise `ValueError`.
The normalized values are read-only exchanger properties, shared by
`solve`, `solve_thermal_state`, `simulate` and `rate`; they are not fluid
properties or dimensionless correction factors.

## Area and resistance basis

The common whole-exchanger thermal circuit uses absolute resistance [K/W]:

```text
inside bulk -> inside convection -> inside fouling -> cylindrical core wall
            -> outside fouling -> outside thermal path -> outside bulk

Rf_i = fouling_resistance_inside  / bundle.total_inner_area
Rf_o = fouling_resistance_outside / bundle.total_outer_area

1/UA = 1/(alpha_i * Ai) + Rf_i + Rwall + Routside_clean + Rf_o
U    = UA / Ao
```

`Ai` and `Ao` are independent. For plain tubes they use effective heated
length and the inner and outer diameters respectively, times installed tube
count. The wall term remains cylindrical conduction:
`ln(Do/Di) / (2*pi*wall_k*length_effective*n_tubes_total)`.
Expressed per outer area, inside convection and fouling acquire `Ao/Ai`;
the plain-tube wall term becomes `Do*ln(Do/Di)/(2*wall_k)`.
Area-specific fouling is never added directly to a K/W sum.

For circular-finned tubes, `Ao` is the existing authoritative **gross outside
thermal area**, including the primary surface and fin-attributed area. An
explicit `external_area_per_length` override remains authoritative. Outside
fouling does not use bare-core outer area or the fin-efficiency-adjusted area.
Fin efficiency, physical outside film HTC, primary/fin parallel conductances,
root conduction and contact topology retain their existing definitions.

Outside finned fouling is a **lumped gross-area series approximation**, common
to the outside path before the root/contact/primary/fin circuit. It shifts the
surface-driving temperature without changing fin geometry or applying a
different deposit resistance to each radial fin cell. It does not represent
a spatial coating or compute fouling-dependent fin material conductivity.
Generic `alfa_o` / `outside_alpha_effective_gross` includes this series term;
`outside_alpha_physical` remains the correlation's physical film HTC. The
common network's `resistance_inside` includes inside convection and inside
fouling; its `resistance_outside` includes the complete outside path and
outside fouling. `resistance_core_wall` remains wall conduction alone.

## Simulation, Rating and condensation

Dry Simulation uses the fouled circuit in the actual iterative U/UA and
effectiveness solve. Dry Rating uses it when finding required thermal area.
For ordinary fixed-inlet/target cases, fouling lowers achievable duty and
increases required area respectively.

Elmahdy wet Simulation and Rating include **both fouling contributions in the
surface-to-inside-fluid resistance operator**. The air film / enthalpy mass
transfer operator stays exposed to the gas-facing surface. Saturation,
condensation onset, wet fraction, surface temperature, duty, outlet humidity,
condensate and outlet temperatures therefore respond within the same solved
process. Thermal-area trials and Simulation surface margin scale fouling
resistances with their associated active areas, alongside the existing film
and wall terms. No reported-duty derating is applied after a clean wet solve.

The optimized BareTube numerical path, staged Rating search, strict final
energy/mass gates and native/reference fallback remain in use. Circular-fin
wet calculations retain their radial heat/mass response downstream of the
lumped fouling node; metal wall temperatures are distinguished from the
exposed surface temperatures.

The explicitly selected `LegacyBulkMeanWetCoilProvider` supports both
fouling inputs in its existing coupled Simulation network, including circular
fins. Its bulk-mean physics, convergence tolerances and Simulation-only
restriction remain unchanged. Zero fouling retains historical Legacy results.
Existing inside wet-gas condensation and pure-water tube-side steam
condensation / evaporation paths also use the fouled circuit. Their phase
correlations are unchanged; equivalent inside HTC reconstruction removes the
explicit inside fouling contribution to avoid counting it twice.

## Hydraulics and diagnostics

This is a **thermal-only** model. Installed diameters, tube lengths, fin
geometry, free-flow areas, roughness and hydraulic topology remain unchanged.
It models no blockage, deposit thickness, roughness increase or time-dependent
fouling growth. Velocities, Reynolds numbers and pressure drop are evaluated
on installed geometry; they may respond to different solved fluid properties.
At fixed transport properties, hydraulic velocities, Reynolds numbers and
losses are unchanged.

The exchanger and `HXResult` (`final_result` in Simulation) expose
`fouling_resistance_inside/outside` [m² K/W] and
`resistance_fouling_inside/outside` [K/W] on installed areas. The common
`ThermalResistanceNetwork` and finned diagnostics expose the same explicit
decomposition. Wet-coil diagnostics additionally expose the applied values
and process-area resistance contributions. Public wet U/UA retains its
existing, explicitly marked equivalent process reporting convention.
