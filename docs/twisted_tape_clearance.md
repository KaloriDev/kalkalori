# Twisted-tape edge clearance

`TwistedTapeGeometry` keeps width independent of bore diameter. Its
`clearance_for(D)` returns `TwistedTapeClearanceGeometry`, with SI properties
`diametral_clearance = D - tape_width` and `radial_clearance = (D - tape_width)/2`.
This describes centered nominal edge geometry, not an eccentric installation or
finite-thickness corner fit. Width must be positive, finite and no greater than D.
Nominal equality uses only relative 1e-12 roundoff tolerance, not a manufacturing
gap allowance. No literature-specific c/D is stored in the physical tape.

## Model selection and contract

```python
selection = TubeSideEnhancement(
    provider=base_provider,
    geometry=TwistedTapeGeometry(.066, .021, .001),
    fluid_phase="liquid",
    clearance_provider=clearance_model,
)
result = exchanger.rate(inside, outside, tube_side_enhancement=selection)
```

`clearance_provider=None` delegates actual width to the base; it never widens
the tape. Nominal-only providers can use `require_nominal_twisted_tape`, which
raises `twisted_tape_clearance_model_required` for a finite gap. An external
provider may instead own documented internal clearance physics without the
optional composition layer. `tube_side_enhancement=None` remains the smooth path.

There is **no built-in clearance correlation and no AUTO selector**. The audit
below did not establish a complete applicable production definition. Synthetic
tests are not engineering correlations. Explicit errors propagate; no default
air-to-liquid transfer, interpolation or smooth fallback exists.

`TwistedTapeClearanceProvider` declares `provider_id`, `mode` (an explicit
`ClearanceModelMode`) and `evaluate(geometry, state, base_result)`. It receives
the actual tape and authoritative `EnhancementInput`, including bulk/wall state,
phase, heat-flow direction, lengths and roughness, at every thermal/hydraulic
evaluation. `TwistedTapeClearanceResult` declares the source's ratio definition
and value. Fluid, boundary condition, reference convention and geometry limits
are the model's responsibility.

- `CORRECTION` additionally implements `base_geometry_for(geometry, state)`:
  its source-supported normalization geometry. Core evaluates the base at that
  geometry using the unchanged state. `ClearanceCorrection` names exact
  compatible base correlation IDs and matching regime, separate positive
  thermal/friction factors, provenance, applicability and warnings. Core
  multiplies complete alpha/Nu and Darcy/native f once. Base references and
  wall correction remain unchanged. Replacement references or wall physics
  require absolute mode.
- `ABSOLUTE` supplies a complete `EnhancementResult` in `absolute`. The base
  is never evaluated, including when it would reject the state. The absolute
  model owns Nu/alpha, friction convention, reference geometry, wall correction,
  scope and provenance. Its provider identity is active; `replaced_base_provider`
  records the unused selection. No before values or invented factors are needed.

Core validates Nu/alpha, area/mass-flow/velocity, Darcy/native friction, mode,
provider identity and consistency between thermal/hydraulic models. Correction
provenance includes both sources; private input cannot become open provenance.
Base diagnostics receive a `base.` prefix; final geometry and performance have
explicit names. Warnings and extrapolation propagate. A fit singular at zero
gap is never normalized against a nominal model by core.

Only heat transfer and distributed friction are composed. The existing solver
integrates final Darcy f with its declared velocity and friction diameter.
There is no second blockage, wall correction or friction multiplier. Acceleration,
tube entrances/exits, nozzles, chambers and return losses retain separate
definitions; thermal changes may indirectly change them through outlet properties.

## Focused freely accessible source audit, 2026-09-10–11

This is a bounded access/definition audit, not proof that no suitable source
exists anywhere. Unverified fields are not filled from memory. No candidate
was implemented. Source fit accuracy is unverified unless stated below.

| Candidate | Verified accessible information | Missing production definition / decision |
|---|---|---|
| Chakroun and Al-Fahed (1996), DOI 10.1115/1.2816688 | Laminar width-study lead; publisher full text could not be retrieved. | Complete coefficients, reference conventions, fluid/Pr/Re and geometry ranges not verified from free primary text. No implementation from a bibliographic lead. |
| Al-Fahed and Chakroun (1996), DOI 10.1016/0142-727X(95)00096-9 | Publisher abstract: turbulent horizontal isothermal tube, 15 tapes, qualitative width effects. | Paired Nu/f equations, friction and Re/Nu reference conventions, ranges and accuracy not established. [Publisher](https://www.sciencedirect.com/science/article/pii/0142727X95000969). |
| Patil (2000), DOI 10.1115/1.521448 | Author upload: laminar power-law fluids, reduced widths, uniform-wall-temperature heating, pitch/width convention. | Complete coefficient/reference/rheology/range set suitable for Newtonian cooling not established. A scalar Newtonian viscosity cannot silently replace a power-law constitutive model. [Author upload](https://www.researchgate.net/publication/237900526_Laminar_Flow_Heat_Transfer_and_Pressure_Drop_Characteristics_of_Power-Law_Fluids_Inside_Tubes_With_Varying_Width_Twisted_Tape_Inserts). |
| Bas and Ozceyhan (2012), DOI 10.1016/j.expthermflusci.2012.03.008 | Author institution: air, uniform heat flux, Re5132–24989, y/D2–4, c/D .0178/.0357, nominal comparator. | Complete primary equations, radial/diametral convention, friction basis, Nu/Re references, Pr limits and accuracy unverified. No transfer to laminar oil. [Author institution](https://avesis.erciyes.edu.tr/yayin/df16bef4-9f73-4de1-a030-0dc9332dc7fa/heat-transfer-enhancement-in-a-tube-with-twisted-tape-inserts-placed-separately-from-the-tube-wall). |
| Naga Sarada et al. (2012), DOI 10.15282/ijame.6.2012.11.0065 | Open primary pp.797–810: air, Re6000–13500, D27.5 mm, widths10–26 mm, H/D3–5; absolute Eqs.15–16. Insert Re/Nu use hydraulic diameter. Reported average fit deviations: Nu6.246%, f2.216%. | Printed friction formulas have conflicting conventions; De in Eq.16 is undefined in nomenclature. Pr limits and complete thickness/reference construction unverified. No silent repair of De/Dh or friction basis. [Publisher PDF](https://ijame.umpsa.edu.my/images/Volume%206/11%20Naga%20Sarada%20et%20al.pdf). |
| Wang, Ni and Xi (2021), DOI 10.1038/s41598-021-86285-0 | Open primary CFD: helium, Re2600–8760, Pr.68, D20 mm, widths12–20 mm, H/width2 for 180-degree pitch; centered versus wall-attached placement. | No complete deployable paired clearance fit identified. CFD plots do not justify creating an oil correlation. [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7994558/). |

The open comparative paper by Gürlek and Özbalta (2013),
[pp.57–58](https://dergipark.org.tr/tr/download/article-file/105357), reproduces
absolute loose-fit equations with negative c/D exponents. This is a lead to
the primary model, not resolution of the missing conventions or authority for
a zero-gap relative normalization.

**OPEN CLEARANCE CORRELATION GAP — Newtonian viscous-liquid cooling at low
and transitional reference Reynolds numbers.** No verified transition model
or interpolation was established. Production implementation requires complete
equations, reference conventions, clearance/twist definitions, fluid and
boundary-condition envelope, geometry and accuracy evidence together.

## Validation and release

`core/tests/twisted_tape_clearance_test.py` uses synthetic external fixtures for
geometry, both modes, compatibility, warnings/errors, public-base composition,
real Rating/Simulation, wall iteration, separate losses and independent friction
integration. Existing smooth and external-provider regressions remain required.
The composition API is included in v0.8.0. No built-in clearance correlation
is included; the source gaps and model limits described above still apply.
