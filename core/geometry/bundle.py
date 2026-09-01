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
import math

from core.common.warnings import ModelWarning, make_warning
from core.geometry.finned_tube import CircularFinnedTube
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
    flow_arrangement : str
        Global 0D flow arrangement. ``"auto"`` (the default) resolves the
        arrangement from the tube circuit topology; explicit ``"crossflow"``,
        ``"counterflow"``, and ``"cocurrentflow"`` values remain authoritative.
    tube_path_type : TubePathType
        How successive passes are connected; see ``TubePathType``. Defaults
        to ``STRAIGHT``, preserving the interpretation used by models that
        do not specify it.
    n_passes_transverse : int | None
        Number of tube-side passes contained within one longitudinal
        section of the tube bank (v0.7.3). For example
        ``n_passes_tube=18, n_passes_transverse=6`` describes 3
        longitudinal sections of 6 transverse passes each. ``None`` (the
        default) means the topology is not separately partitioned into
        longitudinal sections: all tube-side passes are treated as
        belonging to one effective longitudinal section, which preserves
        every pre-v0.7.3 bundle's interpretation and numerical results.

    Notes
    -----
    - No defaults are imposed for geometric definition; all key geometric parameters
      must be provided explicitly.
    - Exact integer tube partitioning among passes is not required or modeled.
      The current 0D model uses the effective average parallel tube count per
      pass, ``n_tubes_total / n_passes_tube`` (see ``n_tubes_per_pass_effective``),
      which is authoritative for tube-side flow area, velocity, Reynolds
      number, heat-transfer coefficient, and pressure drop. Individual real
      passes may contain slightly different integer tube counts; an explicit
      circuit/pass map remains future scope.
    - ``n_passes_transverse`` describes circuit topology only (how the total
      tube-side passes are grouped into longitudinal sections of the outside
      tube bank). It does not alter total heat-transfer area or outside-flow
      geometry. Row counts need not divide evenly between sections; the 0D
      model uses the effective average ``n_rows / n_sections_longitudinal``.
    - ``flow_arrangement="auto"`` maps a single longitudinal section to
      crossflow and multiple single-transverse-pass sections to counterflow.
      Intermediate multi-pass/multi-section circuits use global crossflow as
      a conservative 0D approximation; section-wise coupling remains future
      segmented/distributed-model scope.
    """

    tube: BaseTube
    n_rows: int
    n_tubes_per_row: int
    pitch_transverse: float
    pitch_longitudinal: float
    layout: str
    n_passes_tube: int
    flow_arrangement: str = "auto"
    tube_path_type: TubePathType = TubePathType.STRAIGHT
    n_passes_transverse: int | None = None

    def __post_init__(self) -> None:
        if self.n_rows <= 0 or self.n_tubes_per_row <= 0:
            raise ValueError("n_rows and n_tubes_per_row must be positive integers.")
        if (
            not math.isfinite(self.pitch_transverse)
            or not math.isfinite(self.pitch_longitudinal)
            or self.pitch_transverse <= 0.0
            or self.pitch_longitudinal <= 0.0
        ):
            raise ValueError(
                "pitch_transverse and pitch_longitudinal must be positive and finite."
            )
        if self.n_passes_tube <= 0:
            raise ValueError("n_passes_tube must be a positive integer.")
        if self.layout.lower() not in ("inline", "staggered"):
            raise ValueError("layout must be 'inline' or 'staggered'.")
        if (
            not isinstance(self.flow_arrangement, str)
            or self.flow_arrangement.lower()
            not in ("auto", "crossflow", "counterflow", "cocurrentflow")
        ):
            raise ValueError(
                "flow_arrangement must be 'auto', 'crossflow', "
                "'counterflow', or 'cocurrentflow'."
            )

        if self.tube_path_type == TubePathType.U_TUBE and self.n_passes_tube < 2:
            raise ValueError(
                "tube_path_type=U_TUBE requires at least two tube passes "
                "(a single-pass bundle has no U-bend). Bend arc length and "
                "bend pressure loss are not yet included in any tube path type."
            )

        if self.n_passes_transverse is not None:
            if self.n_passes_transverse <= 0:
                raise ValueError("n_passes_transverse must be a positive integer.")
            if self.n_passes_transverse > self.n_passes_tube:
                raise ValueError(
                    "n_passes_transverse must not exceed n_passes_tube "
                    f"(got n_passes_transverse={self.n_passes_transverse}, "
                    f"n_passes_tube={self.n_passes_tube})."
                )
            if self.n_passes_tube % self.n_passes_transverse != 0:
                raise ValueError(
                    "n_passes_tube must be an exact integer multiple of "
                    "n_passes_transverse (equal transverse pass count in "
                    f"every longitudinal section); got n_passes_tube="
                    f"{self.n_passes_tube}, n_passes_transverse="
                    f"{self.n_passes_transverse}."
                )

        if isinstance(self.tube, CircularFinnedTube):
            self._validate_circular_fin_clearance()

    # -----------------------
    # Tube counts
    # -----------------------

    @property
    def n_tubes_total(self) -> int:
        return self.n_rows * self.n_tubes_per_row

    @property
    def n_tubes_per_pass(self) -> float:
        """Effective average number of parallel tubes per tube-side pass."""
        return self.n_tubes_per_pass_effective

    @property
    def n_tubes_per_pass_effective(self) -> float:
        """Effective average parallel tube count per pass, ``total / passes``.

        This is the authoritative 0D approximation for tube-side flow area,
        velocity, Reynolds number, heat-transfer coefficient, and pressure
        drop (v0.7.3). Real circuiting may assign slightly different integer
        tube counts to individual passes; that exact distribution is not
        modeled here (see ``pass_partition_is_exact``).
        """
        return self.n_tubes_total / self.n_passes_tube

    @property
    def pass_partition_is_exact(self) -> bool:
        """Whether ``n_tubes_total`` divides evenly by ``n_passes_tube``.

        Diagnostic only: an unequal (non-integer) partition is an accepted,
        intentional 0D approximation and never rejects bundle construction.
        """
        return self.n_tubes_total % self.n_passes_tube == 0

    @property
    def n_passes_transverse_resolved(self) -> int:
        """Transverse passes per longitudinal section, with legacy default.

        ``n_passes_transverse=None`` means the topology is not separately
        partitioned into longitudinal sections: it resolves to
        ``n_passes_tube``, i.e. one effective longitudinal section, which is
        exactly the pre-v0.7.3 interpretation.
        """
        if self.n_passes_transverse is None:
            return self.n_passes_tube
        return self.n_passes_transverse

    @property
    def n_sections_longitudinal(self) -> int:
        """Number of longitudinal sections the tube-side passes are grouped into."""
        return self.n_passes_tube // self.n_passes_transverse_resolved

    @property
    def n_rows_per_section_effective(self) -> float:
        """Effective average outside-flow rows per longitudinal section."""
        return self.n_rows / self.n_sections_longitudinal

    @property
    def n_rows_per_section(self) -> float:
        """Compatibility alias for ``n_rows_per_section_effective``."""
        return self.n_rows_per_section_effective

    @property
    def rows_partition_is_exact(self) -> bool:
        """Whether rows divide evenly between longitudinal sections.

        Diagnostic only: a non-integer partition is an accepted effective 0D
        average and never rejects bundle construction.
        """
        return self.n_rows % self.n_sections_longitudinal == 0

    @property
    def flow_arrangement_resolved(self) -> str:
        """Resolve the global 0D flow arrangement from circuit topology.

        Explicit arrangements always win. AUTO maps one longitudinal section
        to crossflow, multiple sections with one transverse pass each to
        counterflow, and intermediate multi-pass/multi-section circuits to a
        conservative global crossflow approximation.
        """
        arrangement = self.flow_arrangement.lower()
        if arrangement != "auto":
            return self.flow_arrangement
        if self.n_sections_longitudinal == 1:
            return "crossflow"
        if self.n_passes_transverse_resolved == 1:
            return "counterflow"
        return "crossflow"

    @property
    def topology_warnings(self) -> tuple[ModelWarning, ...]:
        """Topology limitations exposed through the standard warning model."""
        if (
            self.flow_arrangement.lower() == "auto"
            and 1 < self.n_passes_transverse_resolved < self.n_passes_tube
        ):
            return (
                make_warning(
                    code="FLOW_ARRANGEMENT_AUTO_MULTIPASS_APPROXIMATION",
                    message=(
                        "TubeBundle circuit contains multiple transverse passes "
                        "and multiple longitudinal sections; the current 0D "
                        "model uses global crossflow. Exact section-wise coupling "
                        "requires a future segmented/distributed model."
                    ),
                    source="tube_bundle_geometry",
                    severity="warning",
                ),
            )
        return ()

    @property
    def geometry_warnings(self) -> tuple[ModelWarning, ...]:
        """Geometry-diagnostics alias for ``topology_warnings``."""
        return self.topology_warnings

    @property
    def warnings(self) -> tuple[ModelWarning, ...]:
        """Compatibility alias for bundle-level ``topology_warnings``."""
        return self.topology_warnings

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

    @property
    def total_primary_outside_area(self) -> float:
        """Exposed primary outside area for all tubes [m2]."""
        area = getattr(self.tube, "area_primary_outside", self.tube.area_outer)
        return self.n_tubes_total * float(area)

    @property
    def total_fin_area(self) -> float:
        """Fin-attributed gross solver area for all tubes [m2]."""
        return self.n_tubes_total * float(getattr(self.tube, "area_fin", 0.0))

    @property
    def total_fin_geometric_area(self) -> float:
        return self.n_tubes_total * float(
            getattr(self.tube, "area_fin_geometric", 0.0)
        )

    @property
    def total_outer_geometric_area(self) -> float:
        area = getattr(self.tube, "area_outside_geometric", self.tube.area_outer)
        return self.n_tubes_total * float(area)

    @property
    def root_conduction_resistance(self) -> float:
        """Equivalent root-layer resistance of all tubes in parallel [K/W]."""
        if not isinstance(self.tube, CircularFinnedTube):
            return 0.0
        return self.tube.root_conduction_resistance_for_tubes(self.n_tubes_total)

    @property
    def fin_contact_thermal_resistance(self) -> float:
        """Equivalent declared fin/root contact resistance [K/W]."""
        if not isinstance(self.tube, CircularFinnedTube):
            return 0.0
        return self.tube.contact_thermal_resistance_for_tubes(self.n_tubes_total)

    @property
    def contact_thermal_resistance(self) -> float:
        return self.fin_contact_thermal_resistance

    # -----------------------
    # Internal flow geometry (per pass)
    # -----------------------

    @property
    def internal_flow_area_per_pass(self) -> float:
        """Total internal flow area within a single pass [m^2]."""
        return self.n_tubes_per_pass_effective * self.tube.flow_area

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

    @property
    def projected_blocking_area_per_length(self) -> float:
        """Per-tube flow-normal periodic blockage [m2/m]."""
        return float(self.tube.projected_blocking_area_per_length)

    @property
    def projected_blocking_area_per_row(self) -> float:
        """Flow-normal blockage at a row's controlling plane [m2]."""
        return (
            self.n_tubes_per_row
            * self.projected_blocking_area_per_length
            * float(getattr(self.tube, "length_effective"))
        )

    @property
    def diagonal_pitch(self) -> float:
        return math.sqrt(
            self.pitch_longitudinal * self.pitch_longitudinal
            + (0.5 * self.pitch_transverse) ** 2
        )

    @property
    def transverse_free_flow_gap(self) -> float:
        return self.pitch_transverse - self.projected_blocking_area_per_length

    @property
    def diagonal_free_flow_gap(self) -> float:
        return 2.0 * (
            self.diagonal_pitch - self.projected_blocking_area_per_length
        )

    @property
    def minimum_free_flow_gap(self) -> float:
        if self.layout.lower() == "inline":
            gap = self.transverse_free_flow_gap
        else:
            gap = min(
                self.transverse_free_flow_gap,
                self.diagonal_free_flow_gap,
            )
        if not math.isfinite(gap) or gap <= 0.0:
            raise ValueError(
                "Tube-bank geometry leaves no positive minimum free-flow gap."
            )
        return gap

    @property
    def minimum_free_flow_area(self) -> float:
        """Minimum periodic open cross-section, face-area compatible [m2]."""
        area = (
            self.n_tubes_per_row
            * float(getattr(self.tube, "length_effective"))
            * self.minimum_free_flow_gap
        )
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("minimum_free_flow_area must be positive and finite.")
        return area

    @property
    def reference_flow_area(self) -> float:
        """Reference-area alias used by finned-tube correlation requests."""
        return self.minimum_free_flow_area

    @property
    def maximum_to_face_velocity_ratio(self) -> float:
        return self.frontal_flow_area / self.minimum_free_flow_area

    def face_velocity(self, m_dot: float, rho: float) -> float:
        """Outside approach velocity for mass flow ``m_dot`` [m/s]."""
        _validate_flow_state(m_dot=m_dot, rho=rho)
        return m_dot / (rho * self.frontal_flow_area)

    def reference_velocity(self, m_dot: float, rho: float) -> float:
        """Velocity on ``minimum_free_flow_area`` [m/s]."""
        _validate_flow_state(m_dot=m_dot, rho=rho)
        return m_dot / (rho * self.minimum_free_flow_area)

    def maximum_gap_velocity(self, m_dot: float, rho: float) -> float:
        return self.reference_velocity(m_dot=m_dot, rho=rho)

    def _validate_circular_fin_clearance(self) -> None:
        """Reject physical fin overlap independently of averaged blockage."""
        D_fin = self.tube.D_fin
        if self.pitch_transverse <= D_fin:
            raise ValueError(
                "pitch_transverse must be greater than D_fin to avoid "
                "overlap of neighboring finned tubes in the same row."
            )

        if self.layout.lower() == "inline":
            if self.pitch_longitudinal <= D_fin:
                raise ValueError(
                    "pitch_longitudinal must be greater than D_fin for an "
                    "inline circular-finned tube bank."
                )
        else:
            if self.diagonal_pitch <= D_fin:
                raise ValueError(
                    "Diagonal tube-center spacing must be greater than D_fin for "
                    "a staggered circular-finned tube bank."
                )
            # Rows separated by two longitudinal pitches are aligned again in
            # a staggered bank. Check that second-neighbour spacing as well as
            # the nearest diagonal spacing above.
            if self.n_rows >= 3 and 2.0 * self.pitch_longitudinal <= D_fin:
                raise ValueError(
                    "Twice pitch_longitudinal must be greater than D_fin to "
                    "avoid overlap of aligned rows in a staggered circular-"
                    "finned tube bank."
                )


def _validate_flow_state(*, m_dot: float, rho: float) -> None:
    if not math.isfinite(m_dot) or m_dot <= 0.0:
        raise ValueError("m_dot must be positive and finite.")
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be positive and finite.")
