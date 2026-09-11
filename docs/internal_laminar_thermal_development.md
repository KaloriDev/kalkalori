# Smooth circular-tube thermal development

When a positive heated length is available and Re < 2300, the internal HTC
uses the Hausen mean constant-wall-temperature thermal-entry expression:

    Gz = Re Pr D / L_heated
    Nu = 3.66 + 0.0668 Gz / (1 + 0.04 Gz^(2/3))

The open, independently reproducible source is the `ht` project's
[Hausen documentation](https://ht.readthedocs.io/en/release/ht.conv_internal.html#ht.conv_internal.laminar_entry_thermal_Hausen).
It identifies Hausen (1943), Kays (1953), the constant-temperature asymptote,
and the high-Pr extension to developing velocity profiles. The implementation
here is independently written from the equation; no external code is copied.

This is a mean thermal-entry model with bulk properties and a developed
velocity profile. At high Pr the velocity entrance region is short relative
to thermal development. A real exchanger's variable wall temperature is
approximated by this boundary condition. No wall-viscosity multiplier or
buoyancy treatment is appended without a separate source basis. A visible
informational assumption warning accompanies its use. This does not certify
the physics of an arbitrary installation or mixed-convection liquid flow.

`L_heated` is the active straight length of one pass, not the total hydraulic
path. The diagnostic length factor is Nu/3.66 so existing diagnostic identities
remain valid; it is not the turbulent length formula. Omitted length retains
exact Nu=3.66 behavior. The existing transition blend and turbulent correction
are unchanged. Hydraulics still use their own Darcy model and path length.

Tests include Gz=1000 giving Nu=17.02, the long-length asymptote, high Pr,
legacy no-length behavior, and production Rating/Simulation with two passes.
