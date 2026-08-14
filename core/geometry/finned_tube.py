# KalKalori - Heat Exchanger Open Engine
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

"""Geometry for periodic full circular-finned tubes.

The core-tube dimensions and material data remain owned by ``BareTube``.
This module adds only the fin/root geometry and its derived quantities.
Bundle-level free-flow geometry deliberately remains in
``core.geometry.bundle`` so periodic blockage has one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

from core.common.warnings import ModelWarning, make_warning
from core.geometry.tube import (
    BaseTube,
    BareTube,
    TubeOrientation,
    TubeSurfaceType,
)


EXTERNAL_AREA_OVERRIDE_RELATIVE_WARNING_THRESHOLD = 0.05


@dataclass(frozen=True)
class CircularFinnedTube(BaseTube):
    """Circular core tube with periodic full annular fins.

    A continuous spiral fin is represented by an equivalent periodic train
    of complete annular fins.  The periodic 0D model intentionally uses the
    fractional count ``length_effective / fin_pitch`` and therefore has no
    discontinuities when the effective length changes.

    The core tube is composed, rather than copied: all internal geometry,
    wall data, roughness and orientation come directly from ``core_tube``.
    ``external_area_per_length`` can override the gross outside area used by
    a solver, but never changes fin shape, flow blockage or fin efficiency.
    """

    core_tube: BareTube
    fin_k: float
    D_fin: float
    D_root: float
    fin_thickness_root: float
    fin_pitch: float
    fin_contact_resistance: float | None = None
    fin_thickness_tip: float | None = None
    external_area_per_length: float | None = None

    surface_type: ClassVar[TubeSurfaceType] = TubeSurfaceType.CIRCULAR_FINNED

    def __post_init__(self) -> None:
        if not isinstance(self.core_tube, BareTube):
            raise TypeError("core_tube must be a BareTube instance.")

        for name, value in (
            ("fin_k", self.fin_k),
            ("D_fin", self.D_fin),
            ("D_root", self.D_root),
            ("fin_thickness_root", self.fin_thickness_root),
            ("fin_pitch", self.fin_pitch),
        ):
            _validate_positive_finite(value, name)

        if self.D_root < self.D_o:
            raise ValueError("D_root must be greater than or equal to core_tube.D_o.")
        if self.D_fin <= self.D_root:
            raise ValueError("D_fin must be greater than D_root.")
        if self.fin_pitch <= self.fin_thickness_root:
            raise ValueError("fin_pitch must be greater than fin_thickness_root.")

        tip = self.fin_thickness_tip_effective
        _validate_positive_finite(tip, "fin_thickness_tip")
        if tip > self.fin_thickness_root:
            raise ValueError(
                "fin_thickness_tip must be less than or equal to fin_thickness_root."
            )

        if self.fin_contact_resistance is not None:
            if (
                not math.isfinite(self.fin_contact_resistance)
                or self.fin_contact_resistance < 0.0
            ):
                raise ValueError(
                    "fin_contact_resistance must be None or a finite, non-negative value."
                )

        if self.external_area_per_length is not None:
            _validate_positive_finite(
                self.external_area_per_length, "external_area_per_length"
            )
            if self.external_area_per_length < self.primary_outside_area_per_length:
                raise ValueError(
                    "external_area_per_length must not be smaller than the "
                    "geometrically exposed primary outside area per length."
                )

        if self.outside_area_used_per_length <= 0.0:
            raise ValueError("Used outside area per length must be positive.")

    # ------------------------------------------------------------------
    # Delegated core-tube public contract
    # ------------------------------------------------------------------

    @property
    def D_i(self) -> float:
        return self.core_tube.D_i

    @property
    def D_o(self) -> float:
        return self.core_tube.D_o

    @property
    def length_total(self) -> float:
        return self.core_tube.length_total

    @property
    def length_effective(self) -> float:
        return self.core_tube.length_effective

    @property
    def wall_k(self) -> float:
        return self.core_tube.wall_k

    @property
    def roughness_inner(self) -> float | None:
        return self.core_tube.roughness_inner

    @property
    def roughness_outer(self) -> float | None:
        return self.core_tube.roughness_outer

    @property
    def tube_orientation(self) -> TubeOrientation | None:
        return self.core_tube.tube_orientation

    @property
    def relative_roughness_inner(self) -> float | None:
        return self.core_tube.relative_roughness_inner

    @property
    def relative_roughness_outer(self) -> float | None:
        return self.core_tube.relative_roughness_outer

    @property
    def flow_area(self) -> float:
        return self.core_tube.flow_area

    @property
    def hydraulic_diameter(self) -> float:
        return self.core_tube.hydraulic_diameter

    @property
    def area_inner(self) -> float:
        return self.core_tube.area_inner

    # ------------------------------------------------------------------
    # Periodic fin geometry (single tube)
    # ------------------------------------------------------------------

    @property
    def fin_thickness_tip_effective(self) -> float:
        """Resolved tip thickness; ``None`` means constant thickness."""
        if self.fin_thickness_tip is None:
            return self.fin_thickness_root
        return self.fin_thickness_tip

    @property
    def effective_fin_thickness_tip(self) -> float:
        """Compatibility spelling for ``fin_thickness_tip_effective``."""
        return self.fin_thickness_tip_effective

    @property
    def fin_height(self) -> float:
        return 0.5 * (self.D_fin - self.D_root)

    @property
    def root_radial_thickness(self) -> float:
        return 0.5 * (self.D_root - self.D_o)

    @property
    def fin_density(self) -> float:
        return 1.0 / self.fin_pitch

    @property
    def effective_fin_count(self) -> float:
        return self.length_effective / self.fin_pitch

    @property
    def clear_spacing_root(self) -> float:
        return self.fin_pitch - self.fin_thickness_root

    @property
    def inner_area_per_length(self) -> float:
        return math.pi * self.D_i

    @property
    def primary_outside_area_per_length(self) -> float:
        """Exposed root/tube cylinder between consecutive fin roots [m2/m]."""
        return (
            math.pi
            * self.D_root
            * self.clear_spacing_root
            / self.fin_pitch
        )

    @property
    def exposed_primary_area_per_length(self) -> float:
        return self.primary_outside_area_per_length

    @property
    def fin_side_slope_factor(self) -> float:
        """Area multiplier for either sloping face of a linear taper."""
        delta_t = self.fin_thickness_tip_effective - self.fin_thickness_root
        return math.sqrt(1.0 + (delta_t / (2.0 * self.fin_height)) ** 2)

    @property
    def fin_one_side_area_per_fin(self) -> float:
        r_root = 0.5 * self.D_root
        r_fin = 0.5 * self.D_fin
        return (
            math.pi
            * (r_fin * r_fin - r_root * r_root)
            * self.fin_side_slope_factor
        )

    @property
    def fin_both_sides_area_per_fin(self) -> float:
        return 2.0 * self.fin_one_side_area_per_fin

    @property
    def fin_side_area_per_fin(self) -> float:
        """Combined area of both exposed fin faces [m2/fin]."""
        return self.fin_both_sides_area_per_fin

    @property
    def fin_tip_area_per_fin(self) -> float:
        return math.pi * self.D_fin * self.fin_thickness_tip_effective

    @property
    def fin_area_per_fin(self) -> float:
        """Both faces plus outer tip; the root edge is not exposed."""
        return self.fin_both_sides_area_per_fin + self.fin_tip_area_per_fin

    @property
    def fin_area_geometric_per_length(self) -> float:
        return self.fin_area_per_fin / self.fin_pitch

    @property
    def fin_area_per_length(self) -> float:
        return self.fin_area_geometric_per_length

    @property
    def outside_area_geometric_per_length(self) -> float:
        return (
            self.primary_outside_area_per_length
            + self.fin_area_geometric_per_length
        )

    @property
    def geometric_external_area_per_length(self) -> float:
        return self.outside_area_geometric_per_length

    @property
    def outside_area_used_per_length(self) -> float:
        if self.external_area_per_length is not None:
            return self.external_area_per_length
        return self.outside_area_geometric_per_length

    @property
    def area_primary_outside(self) -> float:
        return self.primary_outside_area_per_length * self.length_effective

    @property
    def area_fin_geometric(self) -> float:
        return self.fin_area_geometric_per_length * self.length_effective

    @property
    def area_fin(self) -> float:
        """Fin-attributed gross area used by a thermal solver [m2]."""
        return self.area_outside_gross - self.area_primary_outside

    @property
    def area_outside_geometric(self) -> float:
        return self.outside_area_geometric_per_length * self.length_effective

    @property
    def area_outside_gross(self) -> float:
        return self.outside_area_used_per_length * self.length_effective

    @property
    def area_outer(self) -> float:
        """Authoritative gross outside area used by bundle-level solvers."""
        return self.area_outside_gross

    @property
    def bare_outside_area(self) -> float:
        return math.pi * self.D_o * self.length_effective

    @property
    def outside_area_ratio_to_bare(self) -> float:
        return self.area_outside_gross / self.bare_outside_area

    @property
    def geometric_outside_area_ratio_to_bare(self) -> float:
        return self.area_outside_geometric / self.bare_outside_area

    @property
    def fin_volume_per_fin(self) -> float:
        """Exact volume of one linearly tapered annular fin [m3]."""
        r_root = 0.5 * self.D_root
        height = self.fin_height
        t_root = self.fin_thickness_root
        t_tip = self.fin_thickness_tip_effective
        radial_integral = (
            r_root * height * (t_root + t_tip) / 2.0
            + height * height * (t_root + 2.0 * t_tip) / 6.0
        )
        return 2.0 * math.pi * radial_integral

    @property
    def fin_volume_per_length(self) -> float:
        return self.fin_volume_per_fin / self.fin_pitch

    @property
    def root_layer_volume_per_length(self) -> float:
        return math.pi * (self.D_root**2 - self.D_o**2) / 4.0

    @property
    def projected_blocking_area_per_length(self) -> float:
        """Flow-normal blocked area per axial tube length [m2/m].

        For the permitted non-increasing linear thickness profile this is
        the exact periodic orthographic blockage of the root cylinder plus
        the annular fin wings.
        """
        return self.D_root + (
            (self.D_fin - self.D_root)
            * (self.fin_thickness_root + self.fin_thickness_tip_effective)
            / (2.0 * self.fin_pitch)
        )

    @property
    def projected_blocking_area(self) -> float:
        return self.projected_blocking_area_per_length * self.length_effective

    # ------------------------------------------------------------------
    # Override/contact diagnostics and standalone resistance helpers
    # ------------------------------------------------------------------

    @property
    def external_area_relative_difference(self) -> float | None:
        if self.external_area_per_length is None:
            return None
        return (
            self.external_area_per_length - self.outside_area_geometric_per_length
        ) / self.outside_area_geometric_per_length

    @property
    def contact_resistance_used(self) -> float:
        """Areal contact resistance used by calculations [m2 K/W]."""
        return 0.0 if self.fin_contact_resistance is None else self.fin_contact_resistance

    @property
    def contact_area_per_tube(self) -> float:
        """Physical 0D area basis for the declared contact resistance [m2].

        A welded fin with ``D_root == D_o`` contacts the core only beneath
        its periodic root footprint.  A geometry with ``D_root > D_o`` has a
        continuous root layer, so its core/root interface is the complete
        outside cylinder of the core tube.
        """
        if self.D_root == self.D_o:
            return (
                math.pi
                * self.D_root
                * self.fin_thickness_root
                * self.effective_fin_count
            )
        return math.pi * self.D_o * self.length_effective

    @property
    def contact_area_basis(self) -> str:
        if self.D_root == self.D_o:
            return "periodic_fin_root_footprints"
        return "complete_core_root_layer_interface"

    @property
    def contact_topology(self) -> str:
        if self.D_root == self.D_o:
            return "fin_branch_only"
        return "series_before_primary_and_fin_parallel_branches"

    @property
    def contact_thermal_resistance_per_tube(self) -> float:
        return self.contact_resistance_used / self.contact_area_per_tube

    @property
    def root_conduction_resistance_per_tube(self) -> float:
        if self.D_root == self.D_o:
            return 0.0
        return math.log(self.D_root / self.D_o) / (
            2.0 * math.pi * self.fin_k * self.length_effective
        )

    def contact_thermal_resistance_for_tubes(self, n_tubes: int) -> float:
        _validate_positive_integer(n_tubes, "n_tubes")
        return self.contact_thermal_resistance_per_tube / float(n_tubes)

    def root_conduction_resistance_for_tubes(self, n_tubes: int) -> float:
        _validate_positive_integer(n_tubes, "n_tubes")
        return self.root_conduction_resistance_per_tube / float(n_tubes)

    @property
    def geometry_warnings(self) -> tuple[ModelWarning, ...]:
        warnings: list[ModelWarning] = []
        if self.fin_contact_resistance is None:
            warnings.append(
                make_warning(
                    code="circular_finned_tube_contact_resistance_unspecified",
                    message=(
                        "CircularFinnedTube: fin_contact_resistance is None; "
                        "the resistance helper assumes ideal contact (0 m2 K/W)."
                    ),
                    source="circular_finned_tube_geometry",
                    severity="warning",
                )
            )

        relative_difference = self.external_area_relative_difference
        if (
            relative_difference is not None
            and abs(relative_difference)
            > EXTERNAL_AREA_OVERRIDE_RELATIVE_WARNING_THRESHOLD
            and not math.isclose(
                abs(relative_difference),
                EXTERNAL_AREA_OVERRIDE_RELATIVE_WARNING_THRESHOLD,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            warnings.append(
                make_warning(
                    code="circular_finned_tube_external_area_override_difference",
                    message=(
                        "CircularFinnedTube: external_area_per_length differs "
                        f"from geometric area by {relative_difference:+.2%}; "
                        "the supplied value remains authoritative for gross area."
                    ),
                    source="circular_finned_tube_geometry",
                    severity="warning",
                )
            )
        return tuple(warnings)

    @property
    def warnings(self) -> tuple[ModelWarning, ...]:
        return self.geometry_warnings


def _validate_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


__all__ = [
    "CircularFinnedTube",
    "EXTERNAL_AREA_OVERRIDE_RELATIVE_WARNING_THRESHOLD",
]
