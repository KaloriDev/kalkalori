# Engineering References

KalKalori uses open engineering literature, public standards, and open-source
property libraries as implementation references.

This document lists the main references used by the current computational core.
Short reference notes are also included directly in the relevant source files.

The v0.9.4 inline round-finned correlations have a separate
[forensic source reconstruction](validation/inline_round_finned_source_audit_v094.md),
including the ITE evidence matrix and a conditional ESCOA thermal recommendation.
The [INLINE thermal provider guide](inline_round_finned_thermal.md) documents
the implemented ESCOA 1979 model from Frass's legal secondary reproduction,
its engineering property convention and unresolved domain. The later
[native INLINE hydraulic provider](inline_round_finned_hydraulics.md) implements
Frass Eq.191 under an explicit conditional engineering policy; normalization
and loss-scope provenance remain partial. v0.9.4 provides qualified non-strict
support with strict rejection of unresolved required source conditions.
The [explicit ESCOA STAGGERED guide](staggered_round_finned_thermal.md) covers
Eqs.(60)–(62) from the same legal secondary source. STAGGERED AUTO remains
Briggs–Young, INLINE AUTO remains ESCOA, and staggered hydraulics remain
Robinson–Briggs. The [release validation](validation/v094_release_validation.md)
records integration checks without expanding the source domain.

## v0.9.2 thin-plate fin foundation: free primary sources

- Taler, D.; Marcinkowski, M. (2020), *Calculating the Efficiency of Complex
  Shaped Fins*, Chemical Engineering Transactions 81, 1333-1338,
  [DOI 10.3303/CET2081223](https://doi.org/10.3303/CET2081223).
  [Free publisher PDF](https://www.aidic.it/cet/20/81/223.pdf), accessible
  without payment on 2026-10-19. Copyright AIDIC Servizi S.r.l.; no article,
  prose, figures or external code are redistributed.
- Marcinkowski, M.; Taler, D. (2021), *Calculating the Efficiency of
  Complex-Shaped Fins*, Energies 14(3), 577,
  [publisher article / DOI 10.3390/en14030577](https://www.mdpi.com/1996-1073/14/3/577).
  Open access under CC BY 4.0, verified on 2026-10-19. Sections 2-3 provide
  the governing thin-plate problem, assumptions and area-average efficiency.

KalKalori's numerical implementation and geometric accounting are independent
GPL-3.0-only code. These sources support fin conduction, **not** a predictive
outside-film HTC, pressure drop or bank-row correction for the new surface.
The sources cited by these articles were not imported as implementation
requirements. See [scope](rectangular_finned_ellipse.md).

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

For tube-side enhancement correlations introduced in v0.8.0, see the dedicated
[source audit](twisted_tape_manglik_bergles.md) and
[provider guide](tube_side_enhancements.md). The public built-in model uses
Yang, Liu, Cao, Gao and Ding (2020), "Numerical Analysis of FLiBe Laminar
Convective Heat Transfer Characteristics in Tubes Fitted With Coaxial Cross
Twisted Tape Inserts", *Frontiers in Energy Research*,
[DOI 10.3389/fenrg.2020.00178](https://doi.org/10.3389/fenrg.2020.00178),
freely available from the publisher under CC BY. Only the conventional
single-tape specialization of the authors' Eqs.21-22 is implemented; it is
not a canonical reproduction of Manglik-Bergles (1993) Parts I/II.

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

## Smooth elliptical foundation (v0.9.0)

- [NIST DLMF, 19.30.5](https://dlmf.nist.gov/19.30#E5): exact ellipse
  perimeter using the complete elliptic integral of the second kind. SciPy
  evaluates it with parameter `m = 1 - (b/a)**2`.
- [Su et al. (2019), open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6843173/):
  continuum governing equations and elliptical velocity formulation.
- [Khan, Culham and Yovanovich, open author-hosted paper](https://www.mhtlab.uwaterloo.ca/pdf_papers/mhtl04-5.pdf):
  elliptical-coordinate mapping used to derive the restricted confocal wall
  solution.

The implemented laminar Darcy, H1 and confocal-wall expressions are independent
analytical reconstructions from their stated momentum, energy and Laplace
boundary-value problems. These sources support the governing formulation or
coordinate definitions; the code does not claim they all publish the resulting
closed forms. No empirical bank, turbulent or private handbook model is
implemented. See [equations and applicability](elliptical_tubes.md).

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

The explicit v0.9.4 ESCOA staggered solid-fin alternative uses the legal
secondary reproduction in Frass/Hofmann/Ponweiser (2015), printed pp.17–18,
Eqs.(60)–(62), including `exp(-0.15*n_rows^2)`.
[Publisher full text](https://www.wseas.us/e-library/conferences/2015/Frass/Frass.pdf).
The original ESCOA 1979 manual was not inspected; complete source-domain,
property-state and accuracy closure is not claimed. See the
[staggered model guide](staggered_round_finned_thermal.md). This model is
explicit only and changes neither Briggs–Young AUTO nor hydraulics.

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
`10.1016/j.applthermaleng.2005.06.020`. It is now implemented as an
[explicit optional hydraulic model](genic2006_finned_hydraulics.md); equation
definitions are closed, calibration provenance is partial, and original raw
calibration data were not recovered. ESDU 86022 may be used externally for
validation; no closed ESDU tables, data or code are included in the GPL core.
Schmidt, VDI and Nir remain extended-surface or comparative references.
Genić is explicit only; no blending or change to Briggs--Young,
Robinson--Briggs or Frass AUTO selection is introduced.

The Briggs--Young source experiments used six-row banks, while later
references recommend applying the correlation to banks with at least four
rows. In v0.7.4, one- to three-row banks remain calculable as explicitly
unvalidated extrapolations of the unchanged equation. No empirical row-count
correction is applied.

Used for the v0.7.0 implementation described in
[`finned_tube_model.md`](finned_tube_model.md):

- physical outside film HTC separate from fin efficiency;
- root-diameter and minimum-free-flow Reynolds definitions;
- source-defined Robinson--Briggs per-row friction coefficient and its
  conversion to Pa;
- annular-fin conduction and applicability diagnostics.

### Wet annular fins (v0.7.5)

- Sharqawy, M. H.; Moinuddin, S. M.; Zubair, S. M. (2012).
  "Heat and mass transfer from annular fins of different cross-sectional
  area. Part I. Temperature distribution and fin efficiency",
  *International Journal of Refrigeration*, 35(2), 365--376.
  DOI: [`10.1016/j.ijrefrig.2011.11.004`](https://doi.org/10.1016/j.ijrefrig.2011.11.004).

- Sharqawy, M. H.; Zubair, S. M. (2007).
  "Efficiency and optimization of an annular fin with combined heat and mass
  transfer -- An analytical solution", *International Journal of
  Refrigeration*, 30(5), 751--757.
  DOI: [`10.1016/j.ijrefrig.2006.12.008`](https://doi.org/10.1016/j.ijrefrig.2006.12.008).

- Rosario, L.; Rahman, M. M. (1999).
  "Analysis of heat transfer in a partially wet radial fin assembly during
  dehumidification", *International Journal of Heat and Fluid Flow*, 20(6),
  642--648.
  DOI: [`10.1016/S0142-727X(99)00057-0`](https://doi.org/10.1016/S0142-727X(99)00057-0).

These sources support the combined sensible/latent radial-fin energy balance,
the emergence of fully dry, partially wet and fully wet states, and treatment
of a radial dew-point crossing. KalKalori uses their public physical framework,
not proprietary data or an undocumented calibration. Its H2O saturation,
latent heat and drained saturated-liquid enthalpy remain supplied by IAPWS,
and its dry-carrier mass-transfer coefficient remains the existing
Chilton--Colburn implementation described below.

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

### Open-secondary twisted-tape implementation

Rossi, R., Cattani, L., Mocerino, A., Bozzoli, F., Rainieri, S., Caminati, R.,
and Pagliarini, G. (2017). *Numerical analysis of flow resistance and heat
transfer in the transitional regime of pipe flow with twisted-tape turbulators*.
Journal of Physics: Conference Series 923, 012033.
[doi:10.1088/1742-6596/923/1/012033](https://doi.org/10.1088/1742-6596/923/1/012033).
Open article, CC BY 3.0. Eqs.9/10 are the implementation authority for
`Rossi2017TwistedTapeProvider`; Agarwal & Raja Rao (1996) is historical
attribution, not independently verified primary authority.
See [equations, property references and limitations](twisted_tape_rossi2017.md).

### Primary-source wire-coil implementation

Inaba, H., Ozaki, K., and Kanaoka, S. (1994). *A Fundamental Study of
Heat-Transfer Enhancement and Flow-Drag Reduction in Tubes by Means of Wire
Coil Insert: 1st Report, Characteristics of Flow Resistance and Heat Transfer
in Tubes with Wire Coil Insert*. Transactions of the Japan Society of
Mechanical Engineers Series B, 60(569), 240--247.
[doi:10.1299/kikaib.60.240](https://doi.org/10.1299/kikaib.60.240).
Official free-access [J-STAGE record](https://www.jstage.jst.go.jp/article/kikaib1979/60/569/60_569_240/_article/-char/en).
Eqs. (7), (10), and (11) are the primary authority for the restricted
`P/e > 10` `Inaba1994WireCoilProvider`. See the
[model definitions, normalization and limits](wire_coil_inaba1994.md).

The references listed here are not a substitute for validation.

KalKalori correlations and property helpers should always be used with:
- explicit applicability checks,
- engineering judgment,
- vendor data or experimental validation where available,
- standards required for the specific equipment or project.

---

## Equal-UA counter-crossflow sections (v0.9.1)

- Jaeger, F. A., Beshore, D. G., Miller, F. M., and Gartner, E. M. (1978).
  *Applications of thermal energy storage in the cement industry*,
  NASA-CR-159399 / CONS/5084-1, appendix ENTU, printed pp.161-162
  (PDF pp.169-170). [NASA NTRS record and public report](https://ntrs.nasa.gov/citations/19780025594).
  Local mixed/unmixed crossflow and identical-stage countercurrent series law.
- Delnero, C. (2000). *Neural Networks and PI Control using Steady State
  Prediction Applied to a Heating Coil*. Colorado State University,
  Eq.4.33, printed p.31 (PDF p.45).
  [University-hosted thesis](https://projects-web.engr.colostate.edu/nnhvac/papers/chris_thesis.pdf).
  Independent authority for identical-stage global composition only.

Implementation is original algebra and stable numerical evaluation of these
public equations, with exact capacity-ratio limits and an independently
derived inverse. No upstream code or source PDFs are redistributed.
See [assumptions, equations and validation](thermal_topology.md).

## Rectangular-finned ellipse: Zhao engineering baseline and source grades

Zhao, L.; Gu, X.; Gao, L.; Yang, Z. (2017). *Numerical study on airside
thermal-hydraulic performance of rectangular finned elliptical tube heat
exchanger with large row number in turbulent flow regime*. International
Journal of Heat and Mass Transfer 114, 1314–1330.
[DOI 10.1016/j.ijheatmasstransfer.2017.06.049](https://doi.org/10.1016/j.ijheatmasstransfer.2017.06.049).

The project explicitly authorized the supplied equation transcriptions on
2026-10-20. Eqs.18–20 and definitions were provided with visual evidence;
the diameter inference and passage-area reconstruction remain separately
identified project assumptions. Free full-text access is not asserted.
No source figures, article text, PDFs or external implementation are included.
See [equations, source grades and limits](rectangular_finned_ellipse.md) and
[numerical evidence](validation/zhao2017_v092.md). This baseline is not a
claim that every rectangular-finned ellipse has experimentally validated
thermal/hydraulic support.

### Earlier free-source candidates

The [targeted v0.9.2 source-closure record](validation/finned_ellipse_source_closure_v092.md)
links the free primary material checked on 2026-10-19: Tang, Du and Zeng
(2019), DOI 10.11949/0438-1157.20190527; Zhao and Yang (2016), DOI
10.11908/j.issn.0253-374x.2016.02.020; Ribeiro Jr. and Andrade (2004), DOI
10.1590/S0104-66322004000200025; the UFMG Ribeiro thesis catalogue; and
Chiu, Lin and Jang (Clima 2007). It distinguishes retrieved full papers from
abstract/catalogue-only evidence and records each missing model definition.
Those candidates did not close a complete predictive model under the earlier
free-source-only decision. The current Zhao baseline follows the separate
2026-10-20 authorization; geometry/conduction authorities remain unchanged.

## Turbulent ellipse and nonconfocal wall extension

The explicitly selected turbulent ellipse extension uses Yang et al. (2018),
FHMT 11, 7, [DOI 10.5098/hmt.11.7](https://www.techscience.com/fhmt/v11n1/53399)
(CC BY 4.0), Eq.18/Table 4; and Nikitin & Yakhot (2005), JFM 532, 141–164,
[DOI 10.1017/S0022112005003964](https://doi.org/10.1017/S0022112005003964),
Eqs.3.2/3.5, read in the
[author-hosted full text](https://www.researchgate.net/publication/231908645_Direct_numerical_simulation_of_turbulent_flow_in_elliptical_ducts).
Source equations, bounded implementation and the friction extrapolation are
separately described in [the model guide](elliptical_turbulent.md).
The nonconfocal wall solver independently implements the standard Laplace
weak form documented by the
[DOLFINx tutorial](https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html)
(CC BY 4.0); it does not reuse tutorial implementation or require FEniCS.
No source PDF, figure, external implementation or private geometry is included.

## Chakroun–Al-Fahed laminar twisted tape

Chakroun, W. M.; Al-Fahed, S. F. (1995), *The Effect of Twisted Tape Width
on Heat Transfer and Pressure Drop for Fully Developed Laminar Flow*, ASME
95-GT-042, [DOI 10.1115/95-GT-042](https://doi.org/10.1115/95-GT-042).
The public v0.9.3 implementation follows the separately audited 1995 equations
and explicit public-implementation authorization. This is a scoped exception
to the free-source-only project policy, without asserting free full-text
access. No PDF, copied source prose, figures or tables are redistributed.
See [equations, conventions and limitations](twisted_tape_chakroun1995.md).

The later Chakroun and Al-Fahed (1996) journal publication, *Journal of
Engineering for Gas Turbines and Power*,
[DOI 10.1115/1.2816688](https://doi.org/10.1115/1.2816688), is recorded
bibliographically; its complete equations were not independently audited.

## Private Reference Sources

Project source policy: built-in engineering correlations must be independently
reproducible from legally and freely accessible source material. Models
requiring paywalled publications, licensed datasets, proprietary software or
manufacturer-confidential information are integrated through external
providers. This is a project rule, not an interpretation of copyright law.
The explicitly authorized Zhao-2017 transcription/reconstruction described
above is a scoped project-policy exception; it does not establish free access
or authorize importing other inaccessible source models.
M&B Part I/II are reserved as sources for a future private external provider;
neither their implementation nor manufacturer physics is included in the
new enhancement module. Existing bibliographic entries below and elsewhere
do not authorize new built-in models from inaccessible sources.

This document and the source-file comments it points to store **bibliographic
references only** -- author, title, publisher/journal, edition/chapter, and a
DOI or publisher URL where one is already known and verified. They do not
imply that the cited publication's full text is, or should be, present in
this repository.

Commercial books, licensed standards, and publisher/journal PDFs (for
example the VDI Heat Atlas) must not be committed to KalKalori. If you keep
a personal or company-licensed copy for local reference while developing,
place it under `docs/private_sources/` (already excluded via `.gitignore`)
or anywhere outside the repository -- never in a tracked path. Obtain such
source material through your own legitimate personal, company, or library
access.
