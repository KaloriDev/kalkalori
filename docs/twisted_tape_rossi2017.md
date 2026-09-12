# Rossi 2017 open-secondary twisted-tape comparator

`core.enhancements.Rossi2017TwistedTapeProvider` is an explicit-selection,
native public provider. It is **OPEN_SECONDARY_RECONSTRUCTED**, not a
primary-source-verified Agarwal model. No pychemqt code or numerical results
are used. Smooth defaults and Yang2020 selection are unchanged.

## Implementation authority

R. Rossi, L. Cattani, A. Mocerino, F. Bozzoli, S. Rainieri, R. Caminati and
G. Pagliarini (2017), *Numerical analysis of flow resistance and heat transfer
in the transitional regime of pipe flow with twisted-tape turbulators*,
Journal of Physics: Conference Series **923**, 012033.
[DOI and open paper](https://doi.org/10.1088/1742-6596/923/1/012033),
[publisher PDF](https://iopscience.iop.org/article/10.1088/1742-6596/923/1/012033/pdf).
The article grants CC BY 3.0 on its first page. Equations were reconstructed
independently from this publication; the PDF is not bundled with core.

Article page 7 reproduces the following equations, attributing them to
Agarwal & Raja Rao (1996), reference [5]. That attribution is historical
provenance only, not authority for additional equations or validity limits.

```text
Eq. 9:  f_Darcy = 19.48 Re^(-0.6519) y^(-0.6281)
Eq. 10: Nu = 0.725 Re^(0.568) y^(-0.788) Pr_film^(1/3)
y = H180 / D
Re = rho_bulk v_bulk D / mu_bulk
v_bulk = mass_flow_per_tube / (rho_bulk pi D^2/4)
T_film = (T_bulk + T_wall)/2
Pr_film = cp_film mu_film / k_film
alpha_inside = Nu k_film / D
```

D is the tube bore, not tape width or a blocked hydraulic diameter.
H180 is axial distance for a 180-degree rotation (article page 3).
The paper uses the smooth laminar reference f=64/Re (page 4): Eq.9 is
therefore used directly as **Darcy**, with no factor-of-four conversion.
No viscosity-ratio correction or heating/cooling branch is added.

The paper specifies film-temperature Pr. The arithmetic mean of bulk and
wall temperatures is the project's explicit engineering convention for
reconstructing that reference. The fluid backend is evaluated at this
temperature: transport values are never averaged. Flow/Re/friction retain
the ordinary local bulk hydraulic reference; no film hydraulic correction
is inferred from the source's isothermal pressure-drop comparison.

## Minimal property-reference integration

Providers may declare `thermal_property_reference = "bulk"`, `"wall"` or
`"film"`; absence preserves the legacy bulk convention and protocol.
The solver owns backend evaluation and supplies `EnhancementInput.thermal`.
The result declares the same reference, so generic validation checks Nu
against that state's conductivity instead of unconditionally using bulk k.

Rossi requires a wall and a film state for standalone thermal evaluation.
Rating/Simulation use the existing wall/resistance-network iteration. Its
first wall guess is the mean of the two bulk temperatures, not an assumed
fixed operating wall temperature. Fixed-bulk snapshots, including
`simulate(iterate=False)`, still resolve a local wall for this provider;
`iterate=False` skips the overall outlet/mean-bulk iteration, not this
physically required wall calculation. Missing backend/temperatures or a
nonconverged snapshot wall probe fail explicitly.

Hydraulic inlet/midpoint/outlet nodes evaluate Eq.9 only and return
`alpha_inside=None`, `nusselt=None`. Their reported Pr is local bulk Pr,
not film Pr or a thermal prediction. Their diagnostics label thermal
evaluation as absent. Thermal requests still require a positive alpha.
Existing paired providers continue returning their usual results.

## Applicability and limitations

Default `extrapolation_policy="error"` checks Rossi's direct comparison:
Re 210–3100, thermal Pr_film 44–51, **y=4.44 only** (roundoff tolerance,
not a twist-ratio interval). `extrapolation_policy="warn"` explicitly
permits each departure with named warnings and `applicability="extrapolated"`.
Hydraulic-only requests check Re and y, not an unavailable film Pr.
`within_range` means these dimensionless checks pass, not universal validation.

Rossi's context is 40% water/ethylene glycol, a horizontal 13.5 mm pipe,
approximately uniform wall heat flux, fully developed modelling and an
adiabatic tape. The scope warning always preserves these qualifications.
Rossi secondarily reports Servotherm oil, Pr 195–375 and Re up to 4000
for the historical experiments. Those are not this provider's limits;
the original lower Re and y interval were not verified from Rossi.

This is a nominal/full-width sensitivity model. Tape width, clearance and
thickness in `TwistedTapeGeometry` are retained and warned as unrepresented,
not used to adjust area, Nu or friction. Width greater than bore remains
invalid geometry. Roughness is ignored with a diagnostic. There is no
clearance correlation, tape conduction, buoyancy or insert local-loss model.
Do not combine it with an arbitrary clearance correction. Solver-owned
distributed path length, local losses and acceleration remain separate.

```python
from core.enhancements import Rossi2017TwistedTapeProvider, TubeSideEnhancement, TwistedTapeGeometry

comparator = TubeSideEnhancement(
    Rossi2017TwistedTapeProvider(extrapolation_policy="warn"),
    TwistedTapeGeometry(half_turn_length=0.152, tape_width=0.019, tape_thickness=0.0009),
    fluid_phase="liquid",
)
# hx.simulate(inside, outside, tube_side_enhancement=comparator)
```

Such a finite-width, high-Pr, off-y calculation is an extrapolated nominal
comparator, never a validated real-clearance prediction.
