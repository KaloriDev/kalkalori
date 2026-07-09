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

Used for:
- Reynolds number and flow-regime interpretation,
- pipe friction,
- local loss coefficients,
- inlet/outlet/pass pressure-drop components.

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

## Notes on Applicability

The references listed here are not a substitute for validation.

KalKalori correlations and property helpers should always be used with:
- explicit applicability checks,
- engineering judgment,
- vendor data or experimental validation where available,
- standards required for the specific equipment or project.
