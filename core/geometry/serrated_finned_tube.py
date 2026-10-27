# SPDX-License-Identifier: GPL-3.0-only
"""Engineering radial-sector cut geometry, SI units throughout.

The equations are independent geometric integrals, documented in
docs/serrated_fins.md. A helical winding is replaced by periodic annuli;
neither deformation nor commercial product exactness is claimed.
"""
from dataclasses import dataclass, field
import math
from typing import ClassVar

from core.common.warnings import ModelWarning, make_warning
from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.tube import BaseTube, BareTube, TubeCrossSectionFamily, TubeSurfaceType


class SerratedFinPhysicsNotSupportedError(NotImplementedError):
    """The requested route lacks an explicit compatible serrated model."""


def reject_serrated_physics(tube: BaseTube, operation: str) -> None:
    """Guard physical entry points, including explicitly selected solid models."""
    if getattr(tube, "surface_type", None) is TubeSurfaceType.SERRATED_HELICAL_FIN:
        raise SerratedFinPhysicsNotSupportedError(
            f"{operation}: serrated fins support geometry only without an explicit "
            "complete Phase B profile. Serrated A_min, fin efficiency, HTC and pressure-drop "
            "providers must all be selected compatibly; no solid-fin fallback is permitted."
        )


@dataclass(frozen=True)
class SerratedCircularFinnedTube(BaseTube):
    """Untwisted, constant-thickness, HF-welded I-foot serrated fin.

    Dimensions and tip *arc* widths are in m, fin_k in W/(m K), material
    density in kg/m3, twist in radians. ``segments_per_turn`` is an explicit
    positive integer (also the slit count); it is never rounded from widths.
    Tip segment width is derived as pi*D_fin/n - kerf when omitted. An
    explicitly supplied width must close the same periodic circumference.

    The private uncut geometry reuses validation, core delegation and contact
    bookkeeping only. This object is deliberately NOT a CircularFinnedTube;
    it cannot satisfy any solid-fin provider's geometry type check.
    """

    core_tube: BareTube
    fin_k: float
    D_fin: float
    D_root: float
    fin_thickness_root: float
    fin_pitch: float
    serration_depth: float
    kerf_width_at_tip: float
    segments_per_turn: int
    fin_density_material: float
    segment_width_at_tip: float | None = None
    segment_tip_twist_angle: float = 0.0
    base_style: str = "welded_i_foot"
    fin_thickness_tip: float | None = None
    fin_contact_resistance: float | None = None
    fin_contact_efficiency: float | None = None
    _uncut: CircularFinnedTube = field(init=False, repr=False, compare=False)

    surface_type: ClassVar[TubeSurfaceType] = TubeSurfaceType.SERRATED_HELICAL_FIN
    cross_section_family: ClassVar[TubeCrossSectionFamily] = TubeCrossSectionFamily.CIRCULAR
    geometry_model: ClassVar[str] = "IDEALIZED_RADIAL_SECTOR_CUT"
    geometry_provenance: ClassVar[str] = "ENGINEERING_DERIVATION"
    physical_product_exactness: ClassVar[str] = "NOT_CLAIMED"
    validated_against_manufacturer_CAD: ClassVar[bool] = False

    def __post_init__(self) -> None:
        uncut = CircularFinnedTube(
            core_tube=self.core_tube, fin_k=self.fin_k, D_fin=self.D_fin,
            D_root=self.D_root, fin_thickness_root=self.fin_thickness_root,
            fin_pitch=self.fin_pitch, fin_thickness_tip=self.fin_thickness_tip,
            fin_contact_resistance=self.fin_contact_resistance,
            fin_contact_efficiency=self.fin_contact_efficiency,
        )
        object.__setattr__(self, "_uncut", uncut)
        if uncut.fin_thickness_tip_effective != self.fin_thickness_root:
            raise NotImplementedError("Tapered serrated geometry is not source-closed in Phase A.")
        if self.base_style != "welded_i_foot":
            raise NotImplementedError("Only HF-welded welded_i_foot geometry is supported.")
        if not math.isfinite(self.segment_tip_twist_angle):
            raise ValueError("segment_tip_twist_angle must be finite [rad].")
        if self.segment_tip_twist_angle != 0.0:
            raise NotImplementedError("Segment twist/deformation is unsupported in Phase A.")
        for name in ("serration_depth", "kerf_width_at_tip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative [m].")
        if self.serration_depth > self.fin_height:
            raise ValueError("serration_depth must not exceed fin height.")
        n = self.segments_per_turn
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("segments_per_turn must be a positive integer; no width rounding is used.")
        if not math.isfinite(self.fin_density_material) or self.fin_density_material <= 0:
            raise ValueError("fin_density_material must be positive and finite [kg/m3].")
        if (n * self.angular_kerf_width >= 2.0 * math.pi
                or self.kerf_width_at_tip >= math.pi * self.D_fin / n):
            raise ValueError("Overlapping kerfs: segments_per_turn * beta must be < 2*pi.")
        width = math.pi * self.D_fin / n - self.kerf_width_at_tip
        if self.segment_width_at_tip is not None:
            if not math.isfinite(self.segment_width_at_tip) or self.segment_width_at_tip <= 0:
                raise ValueError("segment_width_at_tip must be positive and finite [m].")
            if not math.isclose(self.segment_width_at_tip, width, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Tip arc widths must satisfy n*(segment_width_at_tip + kerf_width_at_tip) = pi*D_fin.")
        object.__setattr__(self, "segment_width_at_tip", width)

    @property
    def angular_kerf_width(self) -> float:
        """beta [rad]; input kerf is narrow arc length at D_fin/2 only."""
        return self.kerf_width_at_tip / (self.D_fin / 2.0)

    @property
    def slits_per_turn(self) -> int:
        return self.segments_per_turn

    @property
    def fin_thickness(self) -> float:
        return self.fin_thickness_root

    @property
    def fin_volume_removed_per_turn(self) -> float:
        r1 = self.D_fin / 2.0
        rs = r1 - self.serration_depth
        return self.slits_per_turn * self.angular_kerf_width * self.fin_thickness * (r1**2 - rs**2) / 2.0

    @property
    def fin_volume_per_fin(self) -> float:
        return self._uncut.fin_volume_per_fin - self.fin_volume_removed_per_turn

    @property
    def fin_volume_per_length(self) -> float:
        return self.fin_volume_per_fin / self.fin_pitch

    @property
    def fin_mass_per_length(self) -> float:
        """Fin wings only; core tube and any continuous root layer excluded."""
        return self.fin_density_material * self.fin_volume_per_length

    @property
    def fin_area_deltas_per_turn(self) -> tuple[float, float, float, float]:
        """Face, tip, slit-side and bottom areas [m2/turn], in that order.

        A zero-depth cut has no effect. A zero-kerf slit of positive depth
        has two limiting exposed faces, but removes no material. Opening
        by deformation is a different quantity and is not modeled.
        """
        if self.serration_depth == 0.0:
            return (0.0, 0.0, 0.0, 0.0)
        n, beta, t = self.slits_per_turn, self.angular_kerf_width, self.fin_thickness
        r1 = self.D_fin / 2.0
        rs = r1 - self.serration_depth
        return (-n * beta * (r1**2 - rs**2), -n * beta * r1 * t,
                2.0 * n * self.serration_depth * t, n * beta * rs * t)

    @property
    def fin_area_per_fin(self) -> float:
        return self._uncut.fin_area_per_fin + sum(self.fin_area_deltas_per_turn)

    @property
    def fin_area_per_length(self) -> float:
        return self.fin_area_per_fin / self.fin_pitch

    @property
    def fin_area_geometric_per_length(self) -> float:
        return self.fin_area_per_length

    @property
    def primary_area_per_length(self) -> float:
        return self._uncut.primary_outside_area_per_length

    @property
    def primary_outside_area_per_length(self) -> float:
        return self.primary_area_per_length

    @property
    def external_area_per_length(self) -> float:
        """Derived geometric area [m2/m]; no override is accepted."""
        return self.primary_area_per_length + self.fin_area_per_length

    @property
    def outside_area_geometric_per_length(self) -> float:
        return self.external_area_per_length

    @property
    def geometric_external_area_per_length(self) -> float:
        return self.external_area_per_length

    @property
    def outside_area_used_per_length(self) -> float:
        return self.external_area_per_length

    @property
    def area_primary_outside(self) -> float:
        return self.primary_area_per_length * self.length_effective

    @property
    def area_fin(self) -> float:
        return self.fin_area_per_length * self.length_effective

    @property
    def area_fin_geometric(self) -> float:
        return self.area_fin

    @property
    def area_outer(self) -> float:
        return self.external_area_per_length * self.length_effective

    @property
    def area_outside_geometric(self) -> float:
        return self.area_outer

    @property
    def area_outside_gross(self) -> float:
        return self.area_outer

    @property
    def projected_blocking_area_per_length(self) -> float:
        reject_serrated_physics(self, "Serrated orthographic geometric blockage / A_min")

    @property
    def geometry_warnings(self) -> tuple[ModelWarning, ...]:
        return self._uncut.geometry_warnings + (make_warning(
            code="SERRATED_IDEALIZED_RADIAL_SECTOR_CUT",
            message="Engineering geometry only; helix deformation, opened gaps and twist excluded; manufacturer CAD validation not claimed.",
            source="geometry", severity="info",
        ),)

    # Shared core and contact quantities retain the existing annular geometry
    # conventions; none of these delegation points evaluates a correlation.
    @property
    def D_i(self):
        return self.core_tube.D_i

    @property
    def D_o(self):
        return self.core_tube.D_o

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
    def roughness_inner(self):
        return self.core_tube.roughness_inner

    @property
    def roughness_outer(self):
        return self.core_tube.roughness_outer

    @property
    def tube_orientation(self):
        return self.core_tube.tube_orientation

    @property
    def relative_roughness_inner(self):
        return self.core_tube.relative_roughness_inner

    @property
    def relative_roughness_outer(self):
        return self.core_tube.relative_roughness_outer

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
    def fin_thickness_tip_effective(self):
        return self._uncut.fin_thickness_tip_effective

    @property
    def fin_height(self):
        return self._uncut.fin_height

    @property
    def fin_density(self):
        """Axial turns per m, distinct from material density [kg/m3]."""
        return self._uncut.fin_density

    @property
    def effective_fin_count(self):
        return self._uncut.effective_fin_count

    @property
    def clear_spacing_root(self):
        return self._uncut.clear_spacing_root

    @property
    def root_layer_volume_per_length(self):
        return self._uncut.root_layer_volume_per_length

    @property
    def contact_area_per_tube(self):
        return self._uncut.contact_area_per_tube

    @property
    def contact_area_basis(self):
        return self._uncut.contact_area_basis

    @property
    def contact_topology(self):
        return self._uncut.contact_topology

    @property
    def contact_input_mode(self):
        return self._uncut.contact_input_mode

    @property
    def contact_resistance_used(self):
        return self._uncut.contact_resistance_used

    @property
    def contact_thermal_resistance_per_tube(self):
        return self._uncut.contact_thermal_resistance_per_tube

    def contact_thermal_resistance_for_tubes(self, n_tubes: int) -> float:
        return self._uncut.contact_thermal_resistance_for_tubes(n_tubes)

    @property
    def root_conduction_resistance_per_tube(self):
        return self._uncut.root_conduction_resistance_per_tube

    def root_conduction_resistance_for_tubes(self, n_tubes: int) -> float:
        return self._uncut.root_conduction_resistance_for_tubes(n_tubes)
