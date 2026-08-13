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

# -----------------------------------------------------------------------
# CIRCULAR FINNED TUBE GEOMETRY (v0.7.x)
# -----------------------------------------------------------------------
#
# Scope (see docs/finned_tube_model.md for the full write-up):
#   - full (non-segmented, non-serrated) circular fins, modeled as an
#     equivalent periodic array of full annular fins,
#   - a full helical/spiral fin is approximated as this periodic ring
#     array; this is an explicit, documented approximation, not an
#     attempt to resolve the true helical geometry,
#   - constant-thickness OR linearly-tapered (root-to-tip) fin profile.
#
# Out of scope: segmented/serrated fins, wavy fins, continuous plate
# lamella spanning multiple tubes, elliptical/flattened tubes, wet/
# condensing finned surfaces (see core.phase_change guard).
#
# NOTE ON UNITS
# -------------
# All dimensions are expressed in SI units [m].
# -----------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import math

from core.geometry.tube import BareTube, BaseTube, TubeSurfaceType

# Relative-difference threshold above which an explicitly supplied
# ``external_area_per_length`` triggers a warning against the
# geometrically computed total outer area. Documented, not derived from
# any literature source: a plain, conservative "these two disagree by a
# lot, please double check" sanity threshold.
EXTERNAL_AREA_OVERRIDE_WARNING_THRESHOLD = 0.15


@dataclass(frozen=True)
class CircularFinnedTube(BaseTube):
    """Full circular (annular) finned round tube.

    Composes an existing :class:`~core.geometry.tube.BareTube` as
    ``core_tube`` -- the single source of truth for ``D_i``, ``D_o``,
    ``length_total``, ``length_effective``, ``wall_k``, roughness and
    ``tube_orientation``. This class never copies or re-validates those
    fields; it only adds the fin/root geometry layered on top of them.

    A full helical/spiral fin is modeled as an equivalent periodic
    sequence of full annular (ring) fins of pitch ``fin_pitch`` -- this is
    an explicit, documented approximation (see module docstring and
    ``docs/finned_tube_model.md``); the true helical lead angle and its
    (usually small) effect on local flow/conduction are not resolved.

    Parameters
    ----------
    core_tube : BareTube
        Core (root) tube geometry; ``core_tube.D_o`` is the tube outer
        diameter before any fin/root collar is added.
    fin_k : float
        Thermal conductivity of the fin material [W/(m*K)].
    D_fin : float
        Outer (tip) diameter of the fin [m].
    D_root : float
        Outer diameter of the fin root/collar [m]. ``D_o <= D_root <
        D_fin``. For a fin welded directly onto the bare tube (no
        additional collar/foot layer), ``D_root == core_tube.D_o``. For
        an extruded/integral fin with a foot layer, ``D_root >
        core_tube.D_o``; the radial foot-layer thickness
        ``root_radial_thickness = (D_root - D_o) / 2`` is a derived
        conduction path (see ``core.heat_transfer.finned_tube_resistance``).
    fin_thickness_root : float
        Axial fin thickness at the root [m].
    fin_pitch : float
        Axial center-to-center fin pitch [m] (repeat distance between
        corresponding points on consecutive fins).
    fin_thickness_tip : float | None, optional
        Axial fin thickness at the tip [m]. ``None`` means
        ``fin_thickness_tip = fin_thickness_root`` (constant-thickness
        fin). Otherwise ``0 < fin_thickness_tip <= fin_thickness_root``
        (only a root-to-tip taper is supported; the model never silently
        collapses both thicknesses to a single mean value).
    fin_contact_resistance : float | None, optional
        Area-basis contact resistance between the core tube and the
        root/foot layer [m^2*K/W], evaluated at the ``core_tube.D_o``
        interface.

        - ``0.0``: explicitly ideal (perfect) contact.
        - positive value: included in the resistance network.
        - ``None``: unknown. The resistance network treats this as ideal
          contact (0.0) but a caller building that network must raise an
          explicit warning; it is never inferred from the fin
          manufacturing technology (welded/extruded/wrapped/...).
    external_area_per_length : float | None, optional
        Optional, explicitly supplied *authoritative* total outside
        heat-transfer area per unit axial tube length [m^2/m], overriding
        the geometrically computed value for thermal use. ``None`` means
        the geometric value is used directly. The geometric value is
        always still computed and exposed (``A_outside_geometric``) so
        the relative difference can be inspected
        (``external_area_relative_difference``). This override never
        changes flow geometry, V_max, Reynolds number, or correlation
        applicability -- only the "used" heat-transfer area basis
        (``A_outside_gross`` / ``area_outer``).

    Notes
    -----
    Internal geometry and hydraulics (``D_i``, ``area_inner``,
    ``flow_area``, ``hydraulic_diameter``) are taken from ``core_tube``
    unchanged, so a ``CircularFinnedTube`` behaves exactly like its
    ``core_tube`` on the tube side.
    """

    core_tube: BareTube
    fin_k: float
    D_fin: float
    D_root: float
    fin_thickness_root: float
    fin_pitch: float
    fin_thickness_tip: float | None = None
    fin_contact_resistance: float | None = None
    external_area_per_length: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.core_tube, BareTube):
            raise TypeError("core_tube must be a BareTube instance.")

        for name in ("fin_k", "D_fin", "D_root", "fin_thickness_root", "fin_pitch"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        if self.fin_thickness_tip is not None:
            if not math.isfinite(self.fin_thickness_tip) or self.fin_thickness_tip <= 0.0:
                raise ValueError("fin_thickness_tip must be None or finite and positive.")
            if self.fin_thickness_tip > self.fin_thickness_root:
                raise ValueError(
                    "fin_thickness_tip must not exceed fin_thickness_root "
                    "(only a root-to-tip taper is supported)."
                )

        D_o = self.core_tube.D_o
        if not (D_o <= self.D_root < self.D_fin):
            raise ValueError(
                "Geometry must satisfy D_o <= D_root < D_fin "
                f"(got D_o={D_o!r}, D_root={self.D_root!r}, D_fin={self.D_fin!r})."
            )
        if self.fin_pitch <= self.fin_thickness_root:
            raise ValueError("fin_pitch must be strictly greater than fin_thickness_root.")

        if self.fin_contact_resistance is not None:
            if not math.isfinite(self.fin_contact_resistance) or self.fin_contact_resistance < 0.0:
                raise ValueError(
                    "fin_contact_resistance must be None or finite and non-negative."
                )

        if self.external_area_per_length is not None:
            if not math.isfinite(self.external_area_per_length) or self.external_area_per_length <= 0.0:
                raise ValueError("external_area_per_length must be None or finite and positive.")
            primary_per_length = self._A_primary_per_length()
            if self.external_area_per_length < primary_per_length:
                raise ValueError(
                    "external_area_per_length must not be smaller than the "
                    f"geometrically exposed primary (root) surface per unit "
                    f"length ({primary_per_length!r} m^2/m)."
                )

    # -----------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------

    @property
    def surface_type(self) -> TubeSurfaceType:
        return TubeSurfaceType.CIRCULAR_FINNED

    # -----------------------------------------------------------------
    # Delegated core-tube geometry (single source of truth: core_tube)
    # -----------------------------------------------------------------

    @property
    def D_i(self) -> float:
        return self.core_tube.D_i

    @property
    def D_o(self) -> float:
        """Core (root) tube outer diameter [m] -- the conduction/Re basis."""
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
    def tube_orientation(self):
        return self.core_tube.tube_orientation

    @property
    def flow_area(self) -> float:
        """Internal flow cross-sectional area [m^2] -- identical to core_tube."""
        return self.core_tube.flow_area

    @property
    def hydraulic_diameter(self) -> float:
        """Internal hydraulic diameter [m] -- identical to core_tube."""
        return self.core_tube.hydraulic_diameter

    @property
    def area_inner(self) -> float:
        """Inner heat-transfer area [m^2] -- identical to core_tube."""
        return self.core_tube.area_inner

    # -----------------------------------------------------------------
    # Derived fin/root geometry
    # -----------------------------------------------------------------

    @property
    def fin_thickness_tip_used(self) -> float:
        """Resolved tip thickness: ``fin_thickness_tip`` or, if unset, ``fin_thickness_root``."""
        return (
            self.fin_thickness_root
            if self.fin_thickness_tip is None
            else self.fin_thickness_tip
        )

    @property
    def fin_height(self) -> float:
        """Radial fin height, (D_fin - D_root) / 2 [m]."""
        return (self.D_fin - self.D_root) / 2.0

    @property
    def root_radial_thickness(self) -> float:
        """Radial thickness of the root/foot collar layer, (D_root - D_o) / 2 [m].

        Zero for a fin welded directly onto the bare tube (``D_root ==
        core_tube.D_o``); positive for an extruded/integral fin with an
        additional foot layer.
        """
        return (self.D_root - self.D_o) / 2.0

    @property
    def fin_density(self) -> float:
        """Fins per unit axial length, 1 / fin_pitch [1/m]."""
        return 1.0 / self.fin_pitch

    @property
    def effective_fin_count(self) -> float:
        """Non-integer fin count over ``length_effective`` (0D model; not rounded).

        Using this continuous quantity (rather than ``floor``/``round``)
        avoids discontinuous jumps in the 0D thermal result for a small
        change in tube length. A rounded integer count may additionally
        be inspected as pure diagnostics via
        ``nominal_fin_count_diagnostic``, but must never feed back into
        the thermal calculation.
        """
        return self.length_effective / self.fin_pitch

    @property
    def nominal_fin_count_diagnostic(self) -> int:
        """Rounded fin count, diagnostics only -- never used in the 0D thermal result."""
        return round(self.effective_fin_count)

    @property
    def clear_spacing_root(self) -> float:
        """Clear axial gap between adjacent fin roots, fin_pitch - fin_thickness_root [m]."""
        return self.fin_pitch - self.fin_thickness_root

    @property
    def fin_slant_length(self) -> float:
        """Slant length of the (possibly tapered) fin side profile [m].

        Exact for a linear thickness taper: the fin side face is the
        lateral surface of a conical frustum between r_root and r_tip
        with axial half-thickness varying linearly from
        ``fin_thickness_root/2`` to ``fin_thickness_tip_used/2``. Reduces
        to ``fin_height`` exactly for a constant-thickness fin.
        """
        half_thickness_drop = (self.fin_thickness_root - self.fin_thickness_tip_used) / 2.0
        return math.hypot(self.fin_height, half_thickness_drop)

    # -----------------------------------------------------------------
    # Surface areas (per unit axial tube length, then totals)
    # -----------------------------------------------------------------

    def _A_primary_per_length(self) -> float:
        """Exposed root/base surface between fins, per unit axial length [m^2/m].

        The fin footprint on the root cylinder has axial width
        ``fin_thickness_root`` (the root is not exposed to flow under a
        fin base); the remaining ``clear_spacing_root`` per fin pitch is
        exposed bare root surface.
        """
        return math.pi * self.D_root * (self.clear_spacing_root / self.fin_pitch)

    @property
    def A_primary(self) -> float:
        """Total exposed root/base surface over length_effective [m^2]."""
        return self._A_primary_per_length() * self.length_effective

    @property
    def A_fin_side_per_fin(self) -> float:
        """Both flat/sloped side faces of a single fin [m^2].

        Exact frustum-lateral-area formula (see ``fin_slant_length``);
        reduces to the classical flat annular-fin area
        ``2*pi*(r_tip^2 - r_root^2)`` for a constant-thickness fin.
        """
        r_root = self.D_root / 2.0
        r_tip = self.D_fin / 2.0
        return 2.0 * math.pi * (r_root + r_tip) * self.fin_slant_length

    @property
    def A_fin_tip_per_fin(self) -> float:
        """Outer (tip) edge area of a single fin [m^2]; uses fin_thickness_tip_used."""
        return math.pi * self.D_fin * self.fin_thickness_tip_used

    @property
    def A_fin_per_fin(self) -> float:
        """Total area (both sides + tip edge) of a single fin [m^2]."""
        return self.A_fin_side_per_fin + self.A_fin_tip_per_fin

    @property
    def A_fin(self) -> float:
        """Total fin area over all fins on length_effective [m^2]."""
        return self.effective_fin_count * self.A_fin_per_fin

    @property
    def A_outside_geometric(self) -> float:
        """Geometric total outside area, always computed from geometry [m^2]."""
        return self.A_primary + self.A_fin

    @property
    def A_outside_gross(self) -> float:
        """Total outside area actually used for heat transfer [m^2].

        Equal to ``A_outside_geometric`` unless ``external_area_per_length``
        was supplied, in which case that authoritative value is used
        instead (see class docstring).
        """
        if self.external_area_per_length is None:
            return self.A_outside_geometric
        return self.external_area_per_length * self.length_effective

    @property
    def external_area_relative_difference(self) -> float | None:
        """Relative difference (override - geometric) / geometric, or None if no override."""
        if self.external_area_per_length is None:
            return None
        geometric = self.A_outside_geometric
        return (self.A_outside_gross - geometric) / geometric

    @property
    def external_area_override_exceeds_threshold(self) -> bool:
        """True if an override was supplied and disagrees with geometry beyond the documented threshold."""
        diff = self.external_area_relative_difference
        return diff is not None and abs(diff) > EXTERNAL_AREA_OVERRIDE_WARNING_THRESHOLD

    @property
    def A_fin_used(self) -> float:
        """Fin-area component of ``A_outside_gross`` (root surface held fixed; see docs)."""
        return self.A_outside_gross - self.A_primary

    @property
    def area_outer(self) -> float:
        """Outer heat-transfer area [m^2] -- BaseTube contract; equals A_outside_gross."""
        return self.A_outside_gross

    @property
    def area_outer_to_bare_ratio(self) -> float:
        """Ratio of used outside area to the equivalent bare-tube outer area (pi*D_o*L_eff)."""
        bare_area = math.pi * self.D_o * self.length_effective
        return self.A_outside_gross / bare_area

    # -----------------------------------------------------------------
    # Fin metal volume
    # -----------------------------------------------------------------

    @property
    def fin_volume_per_fin(self) -> float:
        """Metal volume of a single fin [m^3].

        Exact closed form for a linear radial thickness taper (derived
        from ``V = 2*pi * integral[r_root..r_tip] r*t(r) dr`` with
        ``t(r)`` linear in ``r``); reduces to
        ``pi*t*(r_tip^2 - r_root^2)`` for constant thickness. Cross-
        checked against direct numerical quadrature in the test suite.
        """
        r_root = self.D_root / 2.0
        H = self.fin_height
        t_root = self.fin_thickness_root
        t_tip = self.fin_thickness_tip_used
        integral = H * r_root * (t_root + t_tip) / 2.0 + (H * H) * (t_root + 2.0 * t_tip) / 6.0
        return 2.0 * math.pi * integral

    @property
    def fin_volume_per_length(self) -> float:
        """Fin metal volume per unit axial tube length [m^3/m]."""
        return self.fin_volume_per_fin * self.fin_density

    # -----------------------------------------------------------------
    # Diagnostics helpers
    # -----------------------------------------------------------------

    def describe(self) -> str:
        return (
            f"CircularFinnedTube(D_i={self.D_i:.5g}, D_o={self.D_o:.5g}, "
            f"D_root={self.D_root:.5g}, D_fin={self.D_fin:.5g}, "
            f"fin_pitch={self.fin_pitch:.5g}, "
            f"fin_thickness_root={self.fin_thickness_root:.5g}, "
            f"fin_thickness_tip={self.fin_thickness_tip_used:.5g})"
        )
