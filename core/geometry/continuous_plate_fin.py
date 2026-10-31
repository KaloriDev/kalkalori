# SPDX-License-Identifier: GPL-3.0-only
"""Finite, rectangular common sheets around round tubes; all dimensions SI.

The sheet aperture fits the collar OD; the collar bore fits the expanded tube.
Collar height includes the sheet thickness and extends in positive axial z.
The active pack is sealed at z=0 and z=tube.length_effective. Outside bypass
is excluded. No correlation or thermal-process identity belongs here.
"""
from dataclasses import dataclass
from enum import Enum
import heapq
import math

from core.geometry.tube import BareTube


class FinAttachment(str, Enum):
    MECHANICALLY_EXPANDED_COLLAR = "mechanically_expanded_collar"


@dataclass(frozen=True)
class ContinuousPlateFinGeometry:
    sheet_count: int
    thickness: float
    pitch: float
    first_sheet_position: float
    height: float
    depth: float
    transverse_origin: float
    streamwise_origin: float
    collar_outer_diameter: float
    collar_height: float
    conductivity: float
    attachment: FinAttachment = FinAttachment.MECHANICALLY_EXPANDED_COLLAR

    def __post_init__(self):
        if isinstance(self.sheet_count, bool) or not isinstance(self.sheet_count, int) or self.sheet_count < 1:
            raise ValueError("sheet_count must be a positive integer.")
        for name in ("thickness", "pitch", "height", "depth", "collar_outer_diameter", "collar_height", "conductivity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive [SI].")
        for name in ("first_sheet_position", "transverse_origin", "streamwise_origin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite [m].")
        if self.first_sheet_position < 0 or not self.thickness <= self.collar_height <= self.pitch or self.thickness >= self.pitch:
            raise ValueError("Require position >= 0 and thickness <= collar_height <= pitch, with positive fin spacing.")
        object.__setattr__(self, "attachment", FinAttachment(self.attachment))

    @property
    def sheet_positions(self):
        """Leading axial faces, never a fractional manufactured sheet count."""
        return tuple(self.first_sheet_position + i*self.pitch for i in range(self.sheet_count))

    @property
    def pack_extent(self):
        return (self.first_sheet_position, self.sheet_positions[-1] + self.collar_height)

    def evaluate(self, bundle):
        if type(bundle.tube) is not BareTube:
            raise ValueError("Continuous sheets require round BareTube cores, without individual fins.")
        tube = bundle.tube
        length = tube.length_effective
        if self.pack_extent[1] > length + 1e-12:
            raise ValueError("Finite collar/fin pack extends beyond effective tube length.")
        dc, do = self.collar_outer_diameter, tube.D_o
        if dc <= do:
            raise ValueError("Collar OD must exceed tube OD; collar bore is the expanded tube OD.")
        rows = bundle.tube_center_coordinates
        centers = tuple(p for row in rows for p in row)
        xmin, ymin = self.streamwise_origin, self.transverse_origin
        xmax, ymax = xmin+self.depth, ymin+self.height
        margins = tuple((x-xmin-dc/2, xmax-x-dc/2, y-ymin-dc/2, ymax-y-dc/2) for x, y in centers)
        if any(min(m) <= 0 for m in margins):
            raise ValueError("Every collar aperture must lie strictly inside the finite sheet outline.")
        if any(math.dist(a, b) <= dc for i, a in enumerate(centers) for b in centers[i+1:]):
            raise ValueError("Collars overlap or leave no positive tube-to-tube opening.")
        count, nt = self.sheet_count, len(centers)
        if nt != bundle.n_tubes_total:
            raise ValueError("Lattice coordinates disagree with authoritative bundle tube count.")
        faces = 2*count*(self.height*self.depth - nt*math.pi*dc**2/4)
        edges = 2*count*self.thickness*(self.height+self.depth)
        bare_length = max(0., length-count*self.collar_height)
        collar_length = count*(self.collar_height-self.thickness)
        tube_area = nt*math.pi*do*bare_length
        collar_area = nt*math.pi*dc*collar_length
        # Exposed axial annuli at free collar ends; coincident collars touch.
        touching = math.isclose(self.collar_height, self.pitch, rel_tol=0., abs_tol=1e-12)
        ends = 2 if touching else 2*count
        if self.first_sheet_position == 0.:  # sealed end is not gas-facing
            ends -= 1
        if math.isclose(self.pack_extent[1], length, rel_tol=0., abs_tol=1e-12):
            ends -= 1
        lips = nt*ends*math.pi*(dc**2-do**2)/4
        contact = nt*count*math.pi*do*self.collar_height
        bare_cut = _finite_throat_width(centers, ymin, ymax, do)
        collar_cut = _finite_throat_width(centers, ymin, ymax, dc)
        area_min = bare_length*bare_cut + collar_length*collar_cut
        if area_min <= 0 or not math.isfinite(area_min):
            raise ValueError("Finite pack leaves no positive external free-flow area.")
        return ContinuousPlateFinSurface(
            tube_area, collar_area, faces, edges, lips, contact,
            self.height*length, area_min, self.depth, margins,
            bare_cut, collar_cut, nt*count,
        )


def _finite_throat_width(centers, lower, upper, diameter):
    """Shortest finite separator between side walls through circular obstacles.

    A graph edge costs its clear surface-to-surface distance. Wall edges retain
    actual margins; disk chains include transverse and diagonal passages and
    do not wrap periodically. The returned width is a throat normal to the
    local passage, not an orthographic projection onto an arbitrary row plane.
    """
    distances = [y-lower-diameter/2 for x, y in centers]
    pending = [(d, i) for i, d in enumerate(distances)]
    heapq.heapify(pending)
    result = upper-lower
    while pending:
        distance, i = heapq.heappop(pending)
        if distance != distances[i]:
            continue
        result = min(result, distance+upper-centers[i][1]-diameter/2)
        for j, center in enumerate(centers):
            if i == j:
                continue
            candidate = distance+math.dist(centers[i], center)-diameter
            if candidate < distances[j]:
                distances[j] = candidate
                heapq.heappush(pending, (candidate, j))
    return result


@dataclass(frozen=True)
class ContinuousPlateFinSurface:
    exposed_tube_area: float
    exposed_collar_area: float
    fin_face_area: float
    fin_edge_area: float
    collar_lip_area: float
    contact_area: float
    frontal_area: float
    minimum_free_flow_area: float
    flow_depth: float
    aperture_margins: tuple
    bare_throat_width: float
    collar_throat_width: float
    aperture_count: int
    flow_area_basis: str = "finite_wall_to_wall_transverse_or_diagonal_separator"

    @property
    def primary_area(self):
        return self.exposed_tube_area+self.exposed_collar_area

    @property
    def fin_area(self):
        return self.fin_face_area+self.fin_edge_area+self.collar_lip_area

    @property
    def total_area(self):
        return self.primary_area+self.fin_area

    @property
    def source_face_cylinder_area(self):
        """Explicit candidate compact-HX basis, excluding edges and collar lips.

        This definition is geometric; acceptance by a correlation is a source
        gate. It must not be described as a closed Wang source convention.
        """
        return self.primary_area+self.fin_face_area

    @property
    def free_flow_ratio(self):
        return self.minimum_free_flow_area/self.frontal_area

    @property
    def compact_hydraulic_diameter(self):
        """4 Ac L / Ao, using the declared face-and-cylinder area [m]."""
        return 4*self.minimum_free_flow_area*self.flow_depth/self.source_face_cylinder_area
