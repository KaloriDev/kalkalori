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

# -------------------------------------------------------------------------
# CIRCULAR FINNED-TUBE BANK FREE-FLOW GEOMETRY (v0.7.x)
# -------------------------------------------------------------------------
#
# Purely geometric (no fluid properties, no correlation): the minimum
# free-flow area / maximum-velocity ratio for a bank of circular finned
# tubes, generalizing
# core.heat_transfer.outside_flow.vmax_ratio_min_freeflow by accounting
# for periodic blockage from the root/fin metal, not just the bare tube
# diameter. Lives alongside the tube geometry (not in heat_transfer)
# because it depends only on tube + bundle-pitch geometry.
# -------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Literal

from core.geometry.finned_tube import CircularFinnedTube


class FinnedTubeGeometryOverlapError(ValueError):
    """Raised when adjacent tubes in the same row would have overlapping fins."""


def validate_no_fin_row_overlap(tube: CircularFinnedTube, tube_pitch_transverse: float) -> None:
    """Reject a geometry where fins from adjacent tubes in the same row would overlap."""
    if tube.D_fin >= tube_pitch_transverse:
        raise FinnedTubeGeometryOverlapError(
            f"Fin outer diameter D_fin={tube.D_fin!r} m is not smaller than the "
            f"transverse tube pitch S_T={tube_pitch_transverse!r} m; adjacent "
            "tubes in the same row would have overlapping fins."
        )


def finned_blocked_equivalent_diameter(tube: CircularFinnedTube) -> float:
    """Equivalent diameter representing the periodic root+fin blockage,
    per unit axial tube length [m]:

        D_blocked = D_root + (D_fin - D_root) * fin_density * fin_thickness_avg
                  = D_root + 2 * fin_height * fin_density * fin_thickness_avg

    ``fin_thickness_avg = (fin_thickness_root + fin_thickness_tip_used)/2``
    is the documented, controlled approximation used for a tapered fin's
    axial footprint (see docs/finned_tube_model.md). This mirrors the
    standard "minimum free-flow area accounting for fin metal projected
    area" construction used broadly in finned-tube-bank design practice
    (cross-checked, not copied, against the open-source (MIT)
    ``fluids.geometry.AirCooledExchanger.A_min`` formula).
    """
    fin_thickness_avg = (tube.fin_thickness_root + tube.fin_thickness_tip_used) / 2.0
    return tube.D_root + 2.0 * tube.fin_height * tube.fin_density * fin_thickness_avg


def finned_min_free_flow_area_per_length(
    tube: CircularFinnedTube,
    S_T: float,
    S_L: float,
    layout: Literal["inline", "staggered"],
) -> float:
    """Minimum free-flow width per unit axial tube length [m] (i.e. area
    per unit length), accounting for fin-blockage, for the governing
    (transverse or diagonal) gap."""
    if S_T <= 0.0 or S_L <= 0.0:
        raise ValueError("S_T and S_L must be positive.")
    validate_no_fin_row_overlap(tube, S_T)

    D_blocked = finned_blocked_equivalent_diameter(tube)
    if S_T <= D_blocked:
        raise ValueError(
            "S_T must exceed the fin-blockage-aware equivalent diameter "
            f"({D_blocked!r} m) to have a valid transverse flow gap; got S_T={S_T!r} m."
        )

    A_T = S_T - D_blocked

    if layout == "inline":
        return A_T
    if layout == "staggered":
        S_D = math.sqrt(S_L * S_L + (S_T * 0.5) ** 2)
        if S_D <= D_blocked:
            raise ValueError("Invalid geometry: diagonal pitch S_D must exceed the blocked diameter.")
        A_D = 2.0 * (S_D - D_blocked)
        return min(A_T, A_D)
    raise ValueError("layout must be 'inline' or 'staggered'.")


def finned_vmax_ratio_min_freeflow(
    tube: CircularFinnedTube,
    S_T: float,
    S_L: float,
    layout: Literal["inline", "staggered"],
) -> float:
    """Ratio (V_max / V_face) for a circular finned-tube bank (see
    ``finned_min_free_flow_area_per_length``)."""
    A_min = finned_min_free_flow_area_per_length(tube, S_T, S_L, layout)
    return S_T / A_min
