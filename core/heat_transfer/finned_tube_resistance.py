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
# CIRCULAR FINNED-TUBE RESISTANCE NETWORK (v0.7.x)
# -------------------------------------------------------------------------
#
# Explicitly separates, per unit exchanger (all tubes):
#   R_inside_convection  : 1/(alfa_i * A_inside)
#   R_wall_conduction     : ln(D_o/D_i) / (2*pi*wall_k*L_eff*N)          (core tube wall)
#   R_root_conduction     : ln(D_root/D_o) / (2*pi*fin_k*L_eff*N)        (0 when D_root==D_o)
#   R_contact             : fin_contact_resistance / A_contact           (0 when ideal/None)
#   R_primary_convection  : 1/(alfa_o_physical * A_primary)
#   R_fin_convection      : 1/(alfa_o_physical * fin_efficiency * A_fin_used)
#
# R_primary_convection and R_fin_convection are in parallel (both paths
# start at the same base/root surface temperature and end at the same
# bulk outside fluid temperature); their parallel combination is
# R_outside_convection. Fin efficiency and contact resistance are each
# applied exactly once.
# -------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import math

from core.common.warnings import ModelWarning, make_warning
from core.geometry.finned_tube import (
    EXTERNAL_AREA_OVERRIDE_WARNING_THRESHOLD,
    CircularFinnedTube,
)
from core.heat_transfer.fin_efficiency import (
    fin_efficiency_for_tube,
    overall_surface_efficiency,
)


@dataclass(frozen=True)
class FinnedTubeResistanceNetwork:
    """One evaluation of the finned-tube resistance network for a bundle
    of ``n_tubes`` identical tubes.

    All resistances are total (already divided by ``n_tubes`` where the
    per-tube contributions are in parallel), i.e. directly summable /
    invertible into a bundle UA, matching the convention already used by
    ``BareTubeHeatExchanger._tube_wall_resistance``.
    """

    n_tubes: int
    alfa_i: float
    alfa_o_physical: float
    fin_efficiency: float
    overall_surface_efficiency: float
    A_inside: float
    A_primary: float
    A_fin_used: float
    A_outside_used: float
    contact_resistance_used: float
    contact_resistance_unknown: bool
    R_inside_convection: float
    R_wall_conduction: float
    R_root_conduction: float
    R_contact: float
    R_primary_convection: float
    R_fin_convection: float

    @property
    def R_outside_convection(self) -> float:
        """Parallel combination of the primary-surface and fin-surface convection paths."""
        return 1.0 / (1.0 / self.R_primary_convection + 1.0 / self.R_fin_convection)

    @property
    def R_total(self) -> float:
        return (
            self.R_inside_convection
            + self.R_wall_conduction
            + self.R_root_conduction
            + self.R_contact
            + self.R_outside_convection
        )

    @property
    def UA(self) -> float:
        return 1.0 / self.R_total

    @property
    def alfa_o_gross_basis(self) -> float:
        """Equivalent HTC referenced to A_outside_used such that
        ``1/(alfa_o_gross_basis * A_outside_used) == R_outside_convection``.

        This is the "overall-surface-efficiency-weighted" coefficient
        used to plug a finned tube into the existing generic
        ``R_o = 1/(alfa_o * A_o)`` formula (see
        ``core.heat_transfer.thermal_iteration`` / ``core.models.bare_tube``)
        unchanged, with ``A_o`` taken as ``CircularFinnedTube.area_outer``
        (== ``A_outside_used``, gross-area basis).
        """
        return 1.0 / (self.R_outside_convection * self.A_outside_used)

    def U_outside_basis(self) -> float:
        """Overall U referenced to the used total outside area [W/(m^2*K)]."""
        return 1.0 / (self.R_total * self.A_outside_used)

    def U_inside_basis(self) -> float:
        """Overall U referenced to the inside area [W/(m^2*K)]."""
        return 1.0 / (self.R_total * self.A_inside)


@dataclass(frozen=True)
class ConductionAndContactResistance:
    """The alfa-independent (pure conduction/contact) part of the finned-tube
    resistance network -- reused by both
    ``build_finned_tube_resistance_network`` and
    ``BareTubeHeatExchanger.tube_wall_resistance`` so the two never drift
    apart."""

    R_wall_conduction: float
    R_root_conduction: float
    R_contact: float
    contact_resistance_used: float
    contact_resistance_unknown: bool

    @property
    def R_total(self) -> float:
        return self.R_wall_conduction + self.R_root_conduction + self.R_contact


def conduction_and_contact_resistance(
    tube: CircularFinnedTube, *, n_tubes: int
) -> ConductionAndContactResistance:
    """Core-tube wall conduction + root/foot-layer conduction + contact
    resistance for ``n_tubes`` identical finned tubes. Independent of
    alfa_i/alfa_o (pure geometry + material properties).
    """
    if n_tubes <= 0:
        raise ValueError("n_tubes must be positive.")

    R_wall_conduction = math.log(tube.D_o / tube.D_i) / (
        2.0 * math.pi * tube.wall_k * tube.length_effective * n_tubes
    )

    if tube.D_root > tube.D_o:
        R_root_conduction = math.log(tube.D_root / tube.D_o) / (
            2.0 * math.pi * tube.fin_k * tube.length_effective * n_tubes
        )
    else:
        R_root_conduction = 0.0

    contact_resistance_unknown = tube.fin_contact_resistance is None
    contact_resistance_used = (
        0.0 if contact_resistance_unknown else float(tube.fin_contact_resistance)
    )
    if contact_resistance_used == 0.0:
        R_contact = 0.0
    else:
        A_contact = n_tubes * math.pi * tube.D_o * tube.length_effective
        R_contact = contact_resistance_used / A_contact

    return ConductionAndContactResistance(
        R_wall_conduction=R_wall_conduction,
        R_root_conduction=R_root_conduction,
        R_contact=R_contact,
        contact_resistance_used=contact_resistance_used,
        contact_resistance_unknown=contact_resistance_unknown,
    )


def build_finned_tube_resistance_network(
    tube: CircularFinnedTube,
    *,
    n_tubes: int,
    alfa_i: float,
    alfa_o_physical: float,
) -> tuple[FinnedTubeResistanceNetwork, list[ModelWarning]]:
    """Build the resistance network for ``n_tubes`` identical finned tubes.

    ``alfa_o_physical`` is the *physical* outside heat-transfer
    coefficient (e.g. from Briggs & Young), referenced to the true
    finned-surface area -- not yet weighted by fin efficiency.
    """
    if alfa_i <= 0.0 or not math.isfinite(alfa_i):
        raise ValueError("alfa_i must be finite and positive.")
    if alfa_o_physical <= 0.0 or not math.isfinite(alfa_o_physical):
        raise ValueError("alfa_o_physical must be finite and positive.")

    warnings: list[ModelWarning] = []

    eta_fin = fin_efficiency_for_tube(tube, alfa_o_physical)
    eta_o = overall_surface_efficiency(
        A_primary=tube.A_primary, A_fin=tube.A_fin_used, fin_efficiency=eta_fin
    )

    A_inside = n_tubes * tube.area_inner
    A_primary = n_tubes * tube.A_primary
    A_fin_used = n_tubes * tube.A_fin_used
    A_outside_used = n_tubes * tube.A_outside_gross

    conduction = conduction_and_contact_resistance(tube, n_tubes=n_tubes)
    R_wall_conduction = conduction.R_wall_conduction
    R_root_conduction = conduction.R_root_conduction
    R_contact = conduction.R_contact
    contact_resistance_used = conduction.contact_resistance_used
    contact_resistance_unknown = conduction.contact_resistance_unknown

    R_inside_convection = 1.0 / (alfa_i * A_inside)

    if contact_resistance_unknown:
        warnings.append(
            make_warning(
                code="finned_tube_contact_resistance_unknown",
                message=(
                    "finned_tube_resistance: fin_contact_resistance was not "
                    "supplied; assuming ideal (zero) contact between the "
                    "core tube and the fin root/foot. Supply an explicit "
                    "value (0.0 for genuinely ideal contact) to remove this "
                    "warning."
                ),
                source="finned_tube_resistance",
                severity="warning",
            )
        )

    R_primary_convection = 1.0 / (alfa_o_physical * A_primary)
    R_fin_convection = 1.0 / (alfa_o_physical * eta_fin * A_fin_used)

    network = FinnedTubeResistanceNetwork(
        n_tubes=n_tubes,
        alfa_i=alfa_i,
        alfa_o_physical=alfa_o_physical,
        fin_efficiency=eta_fin,
        overall_surface_efficiency=eta_o,
        A_inside=A_inside,
        A_primary=A_primary,
        A_fin_used=A_fin_used,
        A_outside_used=A_outside_used,
        contact_resistance_used=contact_resistance_used,
        contact_resistance_unknown=contact_resistance_unknown,
        R_inside_convection=R_inside_convection,
        R_wall_conduction=R_wall_conduction,
        R_root_conduction=R_root_conduction,
        R_contact=R_contact,
        R_primary_convection=R_primary_convection,
        R_fin_convection=R_fin_convection,
    )

    if tube.external_area_override_exceeds_threshold:
        diff = tube.external_area_relative_difference
        warnings.append(
            make_warning(
                code="finned_tube_external_area_override_large_difference",
                message=(
                    "finned_tube_resistance: external_area_per_length differs "
                    f"from the geometrically computed total outside area by "
                    f"{diff * 100.0:.1f}%, which exceeds the documented "
                    f"{EXTERNAL_AREA_OVERRIDE_WARNING_THRESHOLD * 100:.0f}% "
                    "sanity threshold; double-check the supplied value."
                ),
                source="finned_tube_resistance",
                severity="warning",
            )
        )

    return network, warnings
