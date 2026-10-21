# SPDX-License-Identifier: GPL-3.0-only
"""Individual transverse rectangular fins on a mathematical elliptical tube.

Independent geometric construction, SI throughout. The rectangle is centered
on and aligned with the tube's major/minor axes; both rotate together relative
to the external flow. Periodic counts use active length / axial pitch, just as
CircularFinnedTube does. Tube ends and discrete end-fin placement are excluded.
No convection, bank-throat or pressure-loss correlation is implied.
"""
from dataclasses import dataclass
import math
from typing import ClassVar

from core.geometry.elliptical_tube import EllipticalTube
from core.geometry.tube import BaseTube, TubeCrossSectionFamily, TubeSurfaceType


@dataclass(frozen=True)
class RectangularIndividualFin:
    """Dimensions [m], conductivity [W/m/K].

    Width is transverse to air flow; height is streamwise at zero tube angle.
    The rectangle and ellipse rotate together for nonzero orientation_angle.
    """

    outer_width: float
    outer_height: float
    thickness: float
    axial_pitch: float
    thermal_conductivity: float
    fin_turbulator_count: int | None = None

    def __post_init__(self):
        for name in ('outer_width', 'outer_height', 'thickness',
                     'axial_pitch', 'thermal_conductivity'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite [SI].")
        if self.axial_pitch <= self.thickness:
            raise ValueError("Fin axial_pitch must exceed thickness.")
        count = self.fin_turbulator_count
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            raise ValueError('fin_turbulator_count must be None or a nonnegative integer.')

    @property
    def effective_fin_turbulator_count(self) -> int:
        """Count per individual fin; unspecified means smooth, without correction."""
        return 0 if self.fin_turbulator_count is None else self.fin_turbulator_count


@dataclass(frozen=True)
class RectangularFinnedEllipticalTube(BaseTube):
    """Composition with ideal tube contact and a strictly positive fin ligament.

    ``area_outer`` / ``A_physical`` include the geometric outer edge. The
    adiabatic-edge thin-plate model uses only ``area_fin_faces`` for fin
    convection; it never adds ``area_fin_outer_edge`` to effective area.
    Contact area is buried, and aperture area is missing metal, not surface.
    """

    core_tube: EllipticalTube
    fin: RectangularIndividualFin
    surface_type: ClassVar[TubeSurfaceType] = TubeSurfaceType.RECTANGULAR_INDIVIDUAL_FINNED
    cross_section_family: ClassVar[TubeCrossSectionFamily] = TubeCrossSectionFamily.ELLIPTICAL

    def __post_init__(self):
        if not isinstance(self.core_tube, EllipticalTube):
            raise TypeError("core_tube must be an EllipticalTube.")
        if not isinstance(self.fin, RectangularIndividualFin):
            raise TypeError("fin must be a RectangularIndividualFin.")
        if (self.fin.outer_height <= 2 * self.core_tube.outer_semi_major_axis
                or self.fin.outer_width <= 2 * self.core_tube.outer_semi_minor_axis):
            raise ValueError("Fin rectangle must strictly enclose the ellipse with a positive ligament.")
        for value in (self.area_outer, self.area_primary_outside, self.area_fin_faces,
                      self.area_contact, self.fin_face_area_one_side):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Derived fin areas must be positive and numerically representable.")

    @property
    def length_total(self):
        return self.core_tube.length_total

    @property
    def length_effective(self):
        return self.core_tube.length_effective

    @property
    def wall_k(self):
        return self.core_tube.wall_k

    @property
    def flow_area(self):
        return self.core_tube.flow_area

    @property
    def hydraulic_diameter(self):
        return self.core_tube.hydraulic_diameter

    @property
    def inside_wetted_perimeter(self):
        return self.core_tube.inside_wetted_perimeter

    @property
    def area_inner(self):
        return self.core_tube.area_inner

    @property
    def effective_fin_count(self):
        return self.length_effective / self.fin.axial_pitch

    @property
    def aperture_area_per_fin(self):
        """Exact elliptical hole area, one planar section [m2]."""
        return math.pi * self.core_tube.outer_semi_major_axis * self.core_tube.outer_semi_minor_axis

    @property
    def fin_face_area_one_side(self):
        return self.fin.outer_width * self.fin.outer_height - self.aperture_area_per_fin

    @property
    def area_primary_outside(self):
        """Exposed base tube between fins [m2]."""
        return (self.core_tube.outside_perimeter * self.length_effective
                * (1 - self.fin.thickness / self.fin.axial_pitch))

    @property
    def area_contact(self):
        """Buried tube/fin footprint, counted once [m2]."""
        return self.core_tube.outside_perimeter * self.fin.thickness * self.effective_fin_count

    @property
    def area_fin_faces(self):
        return 2 * self.fin_face_area_one_side * self.effective_fin_count

    @property
    def area_fin_outer_edge(self):
        return (2 * (self.fin.outer_width + self.fin.outer_height)
                * self.fin.thickness * self.effective_fin_count)

    @property
    def area_fin(self):
        """Geometric fin surface, including its adiabatic outer edge [m2]."""
        return self.area_fin_faces + self.area_fin_outer_edge

    @property
    def area_fin_geometric(self):
        return self.area_fin

    @property
    def area_outer(self):
        return self.area_primary_outside + self.area_fin

    @property
    def area_outside_geometric(self):
        return self.area_outer

    @property
    def A_physical(self):
        return self.area_outer

    def effective_outside_area(self, eta_fin: float) -> float:
        """Adiabatic-edge area: exposed tube + eta * both broad fin faces."""
        if not math.isfinite(eta_fin) or not 0 <= eta_fin <= 1:
            raise ValueError("eta_fin must be finite and in [0, 1].")
        return self.area_primary_outside + eta_fin * self.area_fin_faces

    @property
    def projected_blocking_area_per_length(self):
        """Exact periodic orthographic blockage, not a bank minimum flow area."""
        angle = self.core_tube.orientation_angle
        fin_streamwise_extent = self.fin.outer_height
        fin_transverse_extent = self.fin.outer_width
        rectangle_width = (fin_streamwise_extent * abs(math.sin(angle))
                           + fin_transverse_extent * abs(math.cos(angle)))
        fraction = self.fin.thickness / self.fin.axial_pitch
        return ((1 - fraction) * self.core_tube.projected_blocking_area_per_length
                + fraction * rectangle_width)
