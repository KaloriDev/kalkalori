# SPDX-License-Identifier: GPL-3.0-only
"""Mathematical elliptical tube contours; independent SI semi-axis inputs.

Exact perimeter: NIST DLMF 19.30.5, https://dlmf.nist.gov/19.30#E5.
SciPy ellipe takes m=1-(b/a)^2, not the elliptic-integral modulus.
Inner/outer contours are concentric and coaxial, and need not be confocal.
Nominal wall thickness is an engineering convention, not exact normal distance.
"""
from dataclasses import dataclass
import math
from typing import ClassVar
from scipy.special import ellipe
from core.geometry.tube import BaseTube, TubeCrossSectionFamily, TubeOrientation, TubeSurfaceType, _validate_roughness


@dataclass(frozen=True)
class EllipticalTube(BaseTube):
    """Smooth tube with independently specified elliptical inner/outer contours.

    All axes are SEMI-AXES [m]. ``orientation_angle`` [rad] is measured from
    the major axis to external flow in the cross-sectional plane: zero means
    major axis parallel to flow. ``tube_orientation`` describes axial/gravity
    orientation and is independent of this rotation. The two lengths keep
    the existing hydraulic versus active heat-transfer meanings.
    """

    outer_semi_major_axis: float
    outer_semi_minor_axis: float
    inner_semi_major_axis: float
    inner_semi_minor_axis: float
    length_total: float
    length_effective: float
    wall_k: float
    orientation_angle: float = 0.0
    roughness_inner: float | None = None
    roughness_outer: float | None = None
    tube_orientation: TubeOrientation | None = None

    surface_type: ClassVar[TubeSurfaceType] = TubeSurfaceType.PLAIN
    cross_section_family: ClassVar[TubeCrossSectionFamily] = TubeCrossSectionFamily.ELLIPTICAL

    @classmethod
    def from_nominal_wall(cls, *, outer_semi_major_axis, outer_semi_minor_axis,
                          nominal_wall_thickness, **kwargs):
        """Subtract nominal thickness from both semi-axes [m].

        36 x 14 x 1.5 mm gives inner full axes 33 x 11 mm. These nested
        ellipses approximate an engineering constant-thickness tube; an exact
        constant-normal offset of an ellipse is not itself an ellipse.
        """
        if (isinstance(nominal_wall_thickness, bool)
                or not math.isfinite(nominal_wall_thickness)
                or nominal_wall_thickness <= 0):
            raise ValueError('nominal_wall_thickness must be positive and finite [m].')
        return cls(outer_semi_major_axis=outer_semi_major_axis,
                   outer_semi_minor_axis=outer_semi_minor_axis,
                   inner_semi_major_axis=outer_semi_major_axis-nominal_wall_thickness,
                   inner_semi_minor_axis=outer_semi_minor_axis-nominal_wall_thickness,
                   **kwargs)

    def __post_init__(self):
        for name in ('outer_semi_major_axis', 'outer_semi_minor_axis',
                     'inner_semi_major_axis', 'inner_semi_minor_axis',
                     'length_total', 'length_effective', 'wall_k'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite [SI].')
        if not math.isfinite(self.orientation_angle):
            raise ValueError('orientation_angle must be finite [rad].')
        ao, bo = self.outer_semi_major_axis, self.outer_semi_minor_axis
        ai, bi = self.inner_semi_major_axis, self.inner_semi_minor_axis
        if bo > ao or bi > ai:
            raise ValueError('Each semi-major axis must be >= its semi-minor axis.')
        if ao <= ai or bo <= bi:
            raise ValueError('Outer semi-axes must strictly exceed corresponding inner semi-axes.')
        if self.length_effective > self.length_total:
            raise ValueError('length_effective must not exceed length_total.')
        _validate_roughness(self.roughness_inner, 'roughness_inner')
        _validate_roughness(self.roughness_outer, 'roughness_outer')
        if self.tube_orientation is not None and not isinstance(self.tube_orientation, TubeOrientation):
            raise ValueError('tube_orientation must be a TubeOrientation value or None.')
        for value in (self.flow_area, self.inside_wetted_perimeter, self.outside_perimeter,
                      self.hydraulic_diameter, self.area_inner, self.area_outer):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Derived ellipse geometry must be positive and numerically representable.')

    @property
    def inside_aspect_ratio(self) -> float:
        return self.inner_semi_minor_axis / self.inner_semi_major_axis

    @property
    def outside_aspect_ratio(self) -> float:
        return self.outer_semi_minor_axis / self.outer_semi_major_axis

    @property
    def flow_area(self) -> float:
        """Unblocked inside cross-section [m2]."""
        return math.pi * self.inner_semi_major_axis * self.inner_semi_minor_axis

    @property
    def inside_wetted_perimeter(self) -> float:
        return float(4 * self.inner_semi_major_axis * ellipe(1 - self.inside_aspect_ratio**2))

    @property
    def outside_perimeter(self) -> float:
        return float(4 * self.outer_semi_major_axis * ellipe(1 - self.outside_aspect_ratio**2))

    @property
    def hydraulic_diameter(self) -> float:
        return 4 * self.flow_area / self.inside_wetted_perimeter

    @property
    def area_inner(self) -> float:
        return self.inside_wetted_perimeter * self.length_effective

    @property
    def area_outer(self) -> float:
        return self.outside_perimeter * self.length_effective

    @property
    def projected_blocking_area_per_length(self) -> float:
        """Exact orthographic flow-normal width [m2/m], not a bank throat.

        Derived from the support function of the rotated mathematical ellipse.
        """
        return 2 * math.hypot(self.outer_semi_major_axis * math.sin(self.orientation_angle),
                              self.outer_semi_minor_axis * math.cos(self.orientation_angle))

    @property
    def streamwise_extent(self) -> float:
        """Exact orthographic extent along external flow [m]."""
        return 2 * math.hypot(self.outer_semi_major_axis * math.cos(self.orientation_angle),
                              self.outer_semi_minor_axis * math.sin(self.orientation_angle))
