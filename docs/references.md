# Engineering References

KalKalori uses open engineering literature, public standards, and open-source
property libraries as implementation references.

This document lists the main references used by the current computational core.
Short reference notes are also included directly in the relevant source files.

---

## Heat Exchanger Theory and ε-NTU Method

- Incropera, F. P., DeWitt, D. P., Bergman, T. L., Lavine, A. S.  
  *Fundamentals of Heat and Mass Transfer.*

- Shah, R. K., Sekulić, D. P.  
  *Fundamentals of Heat Exchanger Design.*

- Kays, W. M., London, A. L.  
  *Compact Heat Exchangers.*

Used for:
- heat exchanger fundamentals,
- ε-NTU method,
- overall heat-transfer resistance,
- standard heat exchanger modelling assumptions.

---

## Internal Flow and Pressure Drop

- White, F. M.  
  *Fluid Mechanics.*

- Idelchik, I. E.  
  *Handbook of Hydraulic Resistance.*

- Crane Co.  
  *Flow of Fluids Through Valves, Fittings, and Pipe*, Technical Paper No. 410.

- Rennels, D. C., Hudson, H. M.  
  *Pipe Flow: A Practical and Comprehensive Guide.*

- Colebrook, C. F. (1939).  
  "Turbulent Flow in Pipes, with Particular Reference to the Transition
  Region Between the Smooth and Rough Pipe Laws", *Journal of the ICE*.

- Haaland, S. E. (1983).  
  "Simple and Explicit Formulas for the Friction Factor in Turbulent Pipe
  Flow", *ASME J. Fluids Eng.* — used only as an explicit initial guess for
  the iterative Colebrook-White solve, not as a replacement for it.

Used for:
- Reynolds number and flow-regime interpretation,
- pipe friction (smooth and rough, tube-side pressure drop),
- local loss coefficients,
- inlet/outlet/pass pressure-drop components,
- tube-sheet entrance and exit losses.

---

## Local Pressure-Drop Fittings (explicit local pressure-drop paths)

- Gibson, A. H.
  Gradual enlargement and contraction loss-coefficient correlations, as
  reproduced in Crane TP-410 and standard fluid-mechanics references.

- Borda, J.-C., Carnot, L.
  Sudden-expansion pressure loss (the limiting case of the Gibson
  expansion form at a 180-degree included angle).

- Crane Co.
  *Flow of Fluids Through Valves, Fittings, and Pipe*, Technical Paper
  No. 410 -- gradual/sudden area changes and the equivalent-length (Le/D)
  convention for fittings.

- Idelchik, I. E.
  *Handbook of Hydraulic Resistance* -- high-Reynolds-number
  screen-equivalent blockage model for flat planar obstructions; general
  reference for elbow and fitting resistance where an automatic
  geometry-based correlation is actually implemented.

- Miller, D. S.
  *Internal Flow Systems* -- elbow/bend loss-coefficient reference
  (background only; no automatic geometry-based elbow correlation is
  implemented from this commit's elbow taxonomy -- see the module
  docstring of `core.pressure_drop.direction_changes`).

- ASHRAE Handbook -- Fundamentals.
  Duct fitting loss-coefficient conventions (terminology and structure
  only; the proprietary ASHRAE duct fitting database tables are not
  reproduced in this repository).

Pressure-result semantics for the explicitly invoked local paths:

- `dp_irreversible` is the non-negative irreversible mechanical-energy loss,
  equivalently the total-pressure or stagnation-pressure loss. It is the
  quantity summed as hydraulic resistance.
- `delta_dynamic_pressure = q_out - q_in`, where `q = rho * V**2 / 2`.
  This signed quantity is positive for acceleration and negative for
  deceleration.
- `dp_static = p_in - p_out = dp_irreversible + delta_dynamic_pressure`.
  It is a signed static-pressure difference, not an irreversible loss.
- For a complete explicit path, `dp_core`, `dp_local`, and `dp_total` mean
  irreversible total-pressure loss in the core, local stages, and their sum,
  respectively. Dynamic-pressure changes and static-pressure differences are
  aggregated and reported separately.

A diffuser can therefore have a positive `dp_irreversible` while
`delta_dynamic_pressure < 0` and `dp_static < 0`. The negative static-pressure
difference means that static pressure rises through the diffuser; it is
static-pressure recovery, not negative hydraulic resistance or negative
total-pressure loss.

The standard `solve()`/`simulate()`/`rate()` workflows do not evaluate
explicit local-path geometry and continue to report `dp_local = 0`. Their
legacy tube-bundle and tube-bank static-pressure-difference outputs remain
numerically unchanged; the separate semantics above apply to the explicit
path result API.

Used for:
- straight local-section Darcy-Weisbach pressure drop (nozzles, ducts,
  headers),
- gradual/sudden expansion and contraction pressure change,
- circular and rectangular elbow pressure drop via user-defined K or
  equivalent-length methods,
- flat planar obstruction (bird net/wire mesh/grille/louver) pressure drop,
- user-defined fixed or K-based local losses.

---

## External Crossflow over Tube Banks

- Incropera, F. P., DeWitt, D. P., Bergman, T. L., Lavine, A. S.  
  *Fundamentals of Heat and Mass Transfer.*

- VDI Heat Atlas.  
  Tube banks in crossflow.

- Žukauskas, A.  
  Heat transfer from tubes in crossflow / tube-bank correlations.

- Gaddis, E. S., Gnielinski, V.  
  Pressure drop and heat-transfer correlations for tube banks.

Used for:
- tube-bank heat transfer,
- inline and staggered layouts,
- Reynolds and Prandtl number interpretation,
- outside-side pressure-drop modelling through Euler-number methods.

### Circular-finned tubes (v0.7.0)

- Briggs, D. E.; Young, E. H. (1963).
  "Convection Heat Transfer and Pressure Drop of Air Flowing Across
  Triangular Pitch Banks of Finned Tubes", *Chemical Engineering Progress
  Symposium Series*, 59(41), 1-10.

- Robinson, K. K.; Briggs, D. E. (1966).
  "Pressure Drop of Air Flowing Across Triangular Pitch Banks of Finned
  Tubes", *Chemical Engineering Progress Symposium Series*, 62(64), 177-184.

- Gardner, K. A. (1945).
  "Efficiency of Extended Surfaces", *Transactions of the ASME*, 67,
  621-631.

- Schmidt, T. E. (1949).
  "Heat Transfer Calculation for Extended Surfaces", *Refrigerating
  Engineering*, 57, 351-357.

- VDI Heat Atlas, chapter M1, "Heat Transfer to Finned Tubes".

- Kraus, A. D.; Aziz, A.; Welty, J. (2001).
  *Extended Surface Heat Transfer*. Wiley.

- Genić, S. B. et al. (2006).
  "Research on Air Pressure Drop in Helically-Finned Tube Heat Exchangers",
  *Applied Thermal Engineering*, 26, 478-485.
  DOI: `10.1016/j.applthermaleng.2005.07.017`.

- Nir, A. (1991).
  "Heat Transfer and Friction Factor Correlations for Crossflow over
  Staggered Finned Tube Banks", *Heat Transfer Engineering*, 12(1), 43-58.
  DOI: `10.1080/01457639108939746`.

The Genić DOI above corrects the commonly repeated but incorrect
`10.1016/j.applthermaleng.2005.06.020`. It is used only as a comparative
validation reference. ESDU 86022 may likewise be used externally for
validation; no closed ESDU tables, data or code are included in the GPL core.
Schmidt, VDI, Genić and Nir were reviewed as extended-surface or comparative
references; their correlations are not blended into the two implemented
Briggs--Young/Robinson--Briggs empirical models.

Used for the v0.7.0 implementation described in
[`finned_tube_model.md`](finned_tube_model.md):

- physical outside film HTC separate from fin efficiency;
- root-diameter and minimum-free-flow Reynolds definitions;
- source-defined Robinson--Briggs per-row friction coefficient and its
  conversion to Pa;
- annular-fin conduction and applicability diagnostics.

---

## Psychrometrics and Moist Air

- ASHRAE Handbook — Fundamentals.  
  Psychrometrics chapter.

- PsychroLib.  
  Open-source implementation of psychrometric equations based on ASHRAE formulations.

Used for:
- humidity ratio,
- relative humidity,
- dew point,
- moist-air enthalpy,
- moist-air density,
- saturation state checks.

---

## Water and Steam Properties

- IAPWS.  
  *Revised Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam*.

- Wagner, W., Kretzschmar, H.-J.  
  *International Steam Tables — Properties of Water and Steam Based on the Industrial Formulation IAPWS-IF97.*

Used for:
- water and steam thermodynamic properties,
- saturated liquid / saturated vapor states,
- compressed liquid and superheated steam states.

---

## General Fluid and Gas-Mixture Properties

- CoolProp documentation and source references.

Used for:
- optional pure-fluid property calculations,
- optional gas-mixture property calculations,
- future flue-gas / process-gas property backends.

CoolProp is used as an optional backend, not as a mandatory dependency of the
base property layer.

---

## Pure Water/Steam Condensation Inside Tubes (v0.6.2)

- Shah, M. M. (2009).
  "An Improved and Extended General Correlation for Heat Transfer During
  Condensation in Plain Tubes", *HVAC&R Research*, 15(5), 889-913.
  DOI: `10.1080/10789669.2009.10390871`.

Used as the default production in-tube pure-fluid condensation correlation, including
its forced-flow and gravity-driven film terms, orientation-specific regime
maps, and published applicability diagnostics. Two-phase pressure drop is not
part of this model.

---

## Pure-Water Flow Boiling Inside Tubes (v0.6.3)

- Shah, M. M. (1982).
  "Chart Correlation for Saturated Boiling Heat Transfer: Equations and
  Further Study", *ASHRAE Transactions*, 88(1), paper 2673, 165-196.
  No DOI was assigned in the publication.

The implementation uses equations 1-14: the Dittus-Boelter liquid-only
coefficient, boiling number, convection number, liquid Froude number,
orientation-dependent `N`, and the published nucleate, bubble-suppression,
and convective selections. Heat flux is explicitly referenced to tube inside
wetted area. The primary source reports verification against about 3000 data
points for 12 fluids, reduced pressure through 0.89, tube diameters through
41 mm, and both horizontal and vertical tubes.

Gungor, K. E.; Winterton, R. H. S. (1986), "A General Correlation for Flow
Boiling in Tubes and Annuli", *International Journal of Heat and Mass
Transfer*, 29(3), 351-358, DOI `10.1016/0017-9310(86)90205-X`, was also
considered. Its larger database is attractive, but Shah (1982) was selected
because the complete primary equations and orientation rules could be
verified directly and implemented without reconstructing constants from a
secondary source.

Shah (1982) is a saturated, pre-dryout, subcritical-heat-flux correlation. It
does not calculate CHF, dryout quality, or post-dryout heat transfer. The
published paper does not state one universal mass-flux or heat-flux interval
for all fluids; the implementation therefore reports equation-level
applicability diagnostics (including the underlying turbulent
Dittus-Boelter Reynolds range) without inventing or clipping a global range.
KalKalori integrates the local coefficient harmonically over quality and
solves the correlation's inside-area heat flux as a bounded fixed point in
each v0.6.3 evaporation zone.

---

## Gas-Mixture Transport Approximations

- Wilke, C. R.  
  “A Viscosity Equation for Gas Mixtures.”  
  *Journal of Chemical Physics*, 1950.

- Incropera et al.  
  *Fundamentals of Heat and Mass Transfer.*

Used for:
- simple gas-mixture viscosity estimates,
- Sutherland-type gas transport approximations,
- MVP moist-air transport properties.

---

## Outside Water Condensation (v0.6.0)

- IAPWS.
  *Revised Release on the IAPWS Industrial Formulation 1997 for the
  Thermodynamic Properties of Water and Steam.* Used for the water
  saturation curve (dew point, saturated liquid/vapor enthalpy, latent
  heat) via `core.properties.water`; the sole source of water-property
  data used by `core.phase_change`.

- Chilton, T. H., Colburn, A. P. (1934).
  "Mass Transfer (Absorption) Coefficients", *Ind. Eng. Chem.*, 26(11).
  Heat/mass-transfer analogy used by
  `core.phase_change.mass_heat_transfer` to derive a condensation
  mass-transfer coefficient from the existing dry outside heat-transfer
  coefficient.

- Lewis, W. K. (1922).
  Relation between heat- and mass-transfer coefficients for air-water
  systems ("Lewis relation"). The `lewis_number=1.0` default in
  `core.phase_change` is this classic air-water simplification, exposed as
  a **configurable** assumption (`lewis_number`), not asserted as a
  universal constant for arbitrary gas mixtures.

- Bosnjakovic, F.; Merkel, F.
  Enthalpy-potential ("Merkel") method for combined heat/mass transfer in
  the presence of a non-condensable carrier gas, as used in cooling-tower
  and wet-cooling-coil literature. Background for expressing the
  condensation driving force directly on the W = kg H2O / kg dry-carrier
  basis used throughout `core.phase_change`, rather than a molar/volume
  concentration basis.

- Incropera et al.
  *Fundamentals of Heat and Mass Transfer* -- heat-mass transfer analogy
  chapter (Chilton-Colburn / Lewis-relation background), and the same
  Zukauskas tube-bank correlation already used for dry outside heat
  transfer (`core.heat_transfer.outside_flow`), reused unchanged as
  `alfa_outside_dry` -- v0.6.0 does not introduce or calibrate a separate
  wet-surface heat-transfer correlation.

Used for:
- H2O dew point and saturated water content of a general (non-air) carrier
  gas mixture,
- the wet-gas specific-enthalpy balance (`core.phase_change.
  wet_gas_enthalpy`),
- the condensation mass-transfer coefficient
  (`core.phase_change.mass_heat_transfer`).

---

## Notes on Applicability

The references listed here are not a substitute for validation.

KalKalori correlations and property helpers should always be used with:
- explicit applicability checks,
- engineering judgment,
- vendor data or experimental validation where available,
- standards required for the specific equipment or project.
