# KalKalori — Heat Exchanger Open Engine
# Copyright (C) 2025  KalKalori Project Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# NOTE ON UNITS
# -------------
# All dimensions are expressed in SI units [m].

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.geometry.tube import BaseTube


class TubePathType(str, Enum):
    """How successive tube passes are physically connected (v0.5.6).

    ``STRAIGHT``: each pass is a separate set of straight tubes connected
    through chambers, headers, collectors, or external return devices
    (none of those return devices are modeled in this commit). Each pass
    therefore has its own tube-sheet entrance and its own tube-sheet exit:
    ``entrance_count = exit_count = n_passes_tube``.

    ``U_TUBE``: successive straight legs are connected by integral
    U-bends -- the same pressure-drop element family as a 180-degree
    elbow (``core.geometry.pressure_drop_stages.DirectionChangeType.
    U_BEND``/``ELBOW_180``; not calculated in this commit). The complete
    continuous tube path has only one initial tube-sheet entrance and one
    final tube-sheet exit: ``entrance_count = exit_count = 1``.
    Intermediate U-bend boundaries do not create additional tube-sheet
    entrance or exit losses.
    """

    STRAIGHT = "straight"
    U_TUBE = "u_tube"


@dataclass(frozen=True)
class TubeBundle:
    """
    Tube bundle geometry (bare tube MVP).

    Parameters
    ----------
    tube : BaseTube
        Tube geometry.
    n_rows : int
        Number of tube rows in outside flow direction.
    n_tubes_per_row : int
        Number of tubes per row (across the face).
    pitch_transverse : float
        Transverse pitch [m] (tube center-to-center spacing across the face).
    pitch_longitudinal : float
        Longitudinal pitch [m] (tube spacing in flow direction).
    layout : str
        "inline" or "staggered" (used later for refined outside correlations).
    n_passes_tube : int
        Number of tube-side passes (biegów).
    tube_path_type : TubePathType
        How successive passes are connected; see ``TubePathType``. Defaults
        to ``STRAIGHT``, preserving the interpretation used by models that
        do not specify it.

    Notes
    -----
    - No defaults are imposed for geometric definition; all key geometric parameters
      must be provided explicitly.
    - MVP assumes equal partitioning of tubes among passes.
      Future: support explicit pass partition map (unequal passes).
    """

    tube: BaseTube
    n_rows: int
    n_tubes_per_row: int
    pitch_transverse: float
    pitch_longitudinal: float
    layout: str
    n_passes_tube: int
    flow_arrangement: str
    tube_path_type: TubePathType = TubePathType.STRAIGHT

    def __post_init__(self) -> None:
        if self.n_rows <= 0 or self.n_tubes_per_row <= 0:
            raise ValueError("n_rows and n_tubes_per_row must be positive integers.")
        if self.pitch_transverse <= 0.0 or self.pitch_longitudinal <= 0.0:
            raise ValueError("pitch_transverse and pitch_longitudinal must be positive.")
        if self.n_passes_tube <= 0:
            raise ValueError("n_passes_tube must be a positive integer.")
        if self.layout.lower() not in ("inline", "staggered"):
            raise ValueError("layout must be 'inline' or 'staggered'.")
        if self.flow_arrangement.lower() not in ("crossflow", "counterflow", "cocurrentflow"):
            raise ValueError("flow_arrangement must be 'crossflow', 'counterflow', or 'cocurrentflow'.")

        if self.n_tubes_total % self.n_passes_tube != 0:
            raise ValueError(
                "For MVP, total tube count must be divisible by n_passes_tube "
                "(equal tube partitioning per pass)."
            )

        if self.tube_path_type == TubePathType.U_TUBE and self.n_passes_tube < 2:
            raise ValueError(
                "tube_path_type=U_TUBE requires at least two tube passes "
                "(a single-pass bundle has no U-bend). Bend arc length and "
                "bend pressure loss are not yet included in any tube path type."
            )

    # -----------------------
    # Tube counts
    # -----------------------

    @property
    def n_tubes_total(self) -> int:
        return self.n_rows * self.n_tubes_per_row

    @property
    def n_tubes_per_pass(self) -> int:
        """Number of tubes in parallel within a single pass (MVP equal split)."""
        return self.n_tubes_total // self.n_passes_tube

    @property
    def n_turns(self) -> int:
        """Number of 180° turns for n_passes (MVP): turns = passes - 1."""
        return max(self.n_passes_tube - 1, 0)

    # -----------------------
    # Heat transfer areas (effective)
    # -----------------------

    @property
    def total_inner_area(self) -> float:
        return self.n_tubes_total * self.tube.area_inner

    @property
    def total_outer_area(self) -> float:
        return self.n_tubes_total * self.tube.area_outer

    # -----------------------
    # Internal flow geometry (per pass)
    # -----------------------

    @property
    def internal_flow_area_per_pass(self) -> float:
        """Total internal flow area within a single pass [m^2]."""
        return self.n_tubes_per_pass * self.tube.flow_area

    @property
    def internal_hydraulic_diameter(self) -> float:
        return self.tube.hydraulic_diameter

    @property
    def internal_length_total(self) -> float:
        """
        Total hydraulic length experienced by the fluid on the tube side [m].
        Includes multiple passes.
        """
        tube = self.tube
        if not hasattr(tube, "length_total"):
            raise ValueError("Tube object must provide length_total.")
        return self.n_passes_tube * float(getattr(tube, "length_total"))

    # -----------------------
    # Outside flow geometry (effective)
    # -----------------------

    @property
    def frontal_flow_area(self) -> float:
        """
        Frontal (approach) flow area for outside crossflow [m^2].

        MVP:
        - height = n_tubes_per_row * pitch_transverse
        - width  = tube.length_effective

        Blockage by tubes is neglected (to be refined later).
        """
        tube = self.tube
        if not hasattr(tube, "length_effective"):
            raise ValueError("Tube object must provide length_effective.")
        height = self.n_tubes_per_row * self.pitch_transverse
        width = float(getattr(tube, "length_effective"))
        return height * width
