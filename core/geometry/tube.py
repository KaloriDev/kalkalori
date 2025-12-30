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
import math


@dataclass(frozen=True)
class BaseTube:
    """
    Base tube interface (geometry-only).

    Implementations must provide:
    - flow_area [m^2]
    - hydraulic_diameter [m]
    - area_inner [m^2] (heat transfer area, effective length)
    - area_outer [m^2] (heat transfer area, effective length)
    """
    pass


@dataclass(frozen=True)
class BareTube(BaseTube):
    """
    Smooth (bare) circular tube.

    Parameters
    ----------
    D_i : float
        Inner diameter [m]
    D_o : float
        Outer diameter [m]
    length_total : float
        Total hydraulic length for tube-side pressure drop calculations [m]
    length_effective : float
        Effective length participating in heat transfer and outside flow exposure [m]

    Notes
    -----
    The split between length_total and length_effective supports practical designs where:
    - a portion of tube length is in headers / inactive region for heat exchange,
    - but still contributes to tube-side frictional losses.

    Constraints
    -----------
    0 < length_effective <= length_total
    D_o > D_i > 0
    """

    D_i: float
    D_o: float
    length_total: float
    length_effective: float

    def __post_init__(self) -> None:
        if self.D_i <= 0.0 or self.D_o <= 0.0:
            raise ValueError("Diameters must be positive.")
        if self.D_o <= self.D_i:
            raise ValueError("Outer diameter must be greater than inner diameter.")
        if self.length_total <= 0.0:
            raise ValueError("length_total must be positive.")
        if self.length_effective <= 0.0:
            raise ValueError("length_effective must be positive.")
        if self.length_effective > self.length_total:
            raise ValueError("length_effective must not exceed length_total.")

    @property
    def flow_area(self) -> float:
        """Internal flow cross-sectional area [m^2]."""
        return math.pi * (self.D_i ** 2) / 4.0

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter for a circular tube equals D_i [m]."""
        return self.D_i

    @property
    def area_inner(self) -> float:
        """Inner heat transfer area using length_effective [m^2]."""
        return math.pi * self.D_i * self.length_effective

    @property
    def area_outer(self) -> float:
        """Outer heat transfer area using length_effective [m^2]."""
        return math.pi * self.D_o * self.length_effective


# TODO class FinnedTube(BaseTube):
#     ...
