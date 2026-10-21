# Elliptical v0.9.2 API cleanup and migration

This pre-release cleanup intentionally changes rectangular-fin dimension
semantics. Migrate all saved configurations together: `outer_width` (private
builder `fin_outer_width`) is transverse, perpendicular to air flow;
`outer_height` (`fin_outer_height`) is streamwise, parallel to air flow at
zero tube orientation. Swap the previous field values, including positional
`RectangularIndividualFin(width, height, ...)` arguments. There is no silent
legacy interpretation or alias.

The previous correctly oriented 55/26 mm fin becomes width 26 mm, height 55 mm.
Its physical orientation, surface areas, projection, Zhao reconstruction and
fin-efficiency mesh are unchanged. Internal source coordinates use
`Wx = fin_streamwise_extent = fin.outer_height` and
`Wy = fin_transverse_extent = fin.outer_width`. With Pt=26.67 mm and Pl=60 mm,
nominal envelope clearances are 0.67 mm transverse and 5 mm streamwise.
The aligned rectangle and ellipse rotate together for nonzero tube angle;
Zhao's existing reconstruction requires zero angle modulo pi.

## Geometry and wall

Inner and outer ellipses are independent, concentric and coaxial. They need
not be confocal. Internal velocity, Re, Dh, friction and HTC depend on the
actual inner contour and fluid state, independently of the outer contour.

```python
from core.geometry import EllipticalTube

tube = EllipticalTube.from_nominal_wall(
    outer_semi_major_axis=.018, outer_semi_minor_axis=.007,
    nominal_wall_thickness=.0015,
    length_total=.419, length_effective=.405, wall_k=45.)
```

This gives 36x14 mm outer / 33x11 mm inner full axes. Subtracting the same
nominal thickness from each semi-axis is an accepted engineering approximation
to constant nominal thickness. It is not exact geometric normal distance:
an exact constant-normal offset of an ellipse is not itself an ellipse.
Explicit independent inner axes remain available through `EllipticalTube`.

Omitted wall selection uses the existing converged numerical Laplace solver
for general contours, or exact confocal conduction when focal squares agree
within the existing arithmetic tolerance. No geometry is changed. Advanced
`CONFOCAL_EXACT` overrides reject nonconfocal inputs. No new thin-wall formula
or empirical conduction law is introduced.

## Internal model and applicability policy

Normal geometry-based dispatch uses actual Re_Dh:

| Reynolds interval | Selected model |
|---|---|
| 0 < Re < 2300 | Existing analytical fully developed laminar H1 |
| 2300 <= Re < 4000 | Explicit unsupported transition error |
| Re >= 4000 | Yang 2018 turbulent T, with existing shape/smoothness guards |

These conservative dispatch boundaries do not establish a universal ellipse
transition criterion. No transition correlation or circular fallback is added.
The selected model metadata exposes regime, development, thermal BC, references,
area and velocity bases, assumptions and approximation status. H1 and T are
model assumptions; automatic selection does not certify the actual boundary
condition or development of every exchanger.

Yang Eq.18/Table 4 coefficients are unchanged. Its source interval remains
10000 <= Re_Dh <= 120000. Outside that interval, within the implemented turbulent
regime, default evaluation uses actual Re and emits `yang2018_re_extrapolation`
with actual Re, source bounds and direction. No clipping or boundary substitution
occurs. Short heated length (<60 Dh) retains the development warning.
`strict_source_domain=True` rejects Yang source-Re exceedance. Geometry shape,
smoothness and physical-regime guards remain errors. Engineering turbulent
friction remains explicitly warned and has no artificial 120000 upper cap.

`EllipticalFlowAssumptions` remains public for useful advanced direct-model
verification of laminar H1, turbulent T and constitutive assumptions. It is
not a normal STAGE input. Direct analytical functions retain their exact-law
verification use; geometry-based dispatch checks regime compatibility.

## Configuration

The private shared notebook builder uses the existing CASE `options` mapping:

```python
STAGES = {'TEST_HEATER2': {'geometry': 'hx_finned_ellipse_3010'}}

case = {
    'name': 'example', 'mode': 'simulate', 'stage': 'TEST_HEATER2',
    'inside': {...}, 'outside': {...},
    'options': {
        'tube_wall_model': TubeWallModel.ELLIPTICAL_NUMERICAL,
        'airside_model': Zhao2017AirsideModel(correction_provider=my_correction),
        'strict_source_domain': False,
    },
}
```

All three options are optional. `tube_wall_model` accepts `TubeWallModel` or its
string value, `airside_model` accepts `Zhao2017AirsideModel` or None, and
`strict_source_domain` is boolean and controls Yang source-Re applicability.
Zhao's independent source policy is set on `Zhao2017AirsideModel` itself.
Other existing solver options continue through the same mapping. Options are
copied and constructor controls consumed before solver invocation; shared stages
are never mutated. Unknown legacy STAGE model fields fail explicitly.
Existing low-level exchanger constructor controls remain available.

Rectangular-finned ellipses default to Zhao 2017. Explicit correction provenance
and base/corrected values reach result diagnostics. Existing tube turbulator
factors default to 1.0/1.0; conflicting correction providers are rejected.
This refactor performs no calibration and changes no empirical coefficients.
