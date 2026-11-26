# SPDX-License-Identifier: GPL-3.0-only
"""Intrinsic geometry snapshots. No correlation or property-provider calls."""
from dataclasses import dataclass

from core.geometry.rectangular_finned_tube import RectangularFinnedEllipticalTube


@dataclass(frozen=True)
class FlowAreaAvailability:
    area: float | None
    reason: str
    basis: str


def minimum_free_flow_area_status(bundle):
    """Preserve guarded/unresolved geometry as unavailable, with its reason.

    In particular this never reconstructs a SHIFTED lattice, invokes Zhao,
    or substitutes a provider's frontal/average passage reference area.
    """
    if getattr(bundle,'layout','').lower() == 'shifted':
        return FlowAreaAvailability(None,
            'SHIFTED minimum passage area is not established; conventional throat formulas '
            'do not apply and no lattice phase is inferred.', 'unavailable')
    try:
        return FlowAreaAvailability(bundle.minimum_free_flow_area, '',
                                    'bundle_minimum_free_flow_area')
    except (ValueError, NotImplementedError, AttributeError) as exc:
        return FlowAreaAvailability(None, str(exc), 'unavailable')


def rectangular_fin_geometry_metadata(tube, *, n_tubes=1):
    """Physical rectangular-fin values, including adiabatic edge metal.

    The network's gross area includes edges; its adiabatic-edge fin solver
    convects only the two broad faces. Both are exposed so physical area
    cannot be confused with projected or efficiency-weighted area.
    """
    if not isinstance(tube, RectangularFinnedEllipticalTube):
        raise TypeError('Expected RectangularFinnedEllipticalTube.')
    fin = tube.fin
    return dict(
        fin_pitch_mm=1000 * fin.axial_pitch,
        fin_density_1_m=1 / fin.axial_pitch,
        fin_k_W_mK=fin.thermal_conductivity,
        fin_thickness_root_mm=1000 * fin.thickness,
        fin_thickness_tip_mm=1000 * fin.thickness,
        fin_outer_width_mm=1000 * fin.outer_width,
        fin_outer_height_mm=1000 * fin.outer_height,
        fin_length_effective_m=tube.length_effective,
        fin_count_per_tube=tube.effective_fin_count,
        fin_count_total=n_tubes * tube.effective_fin_count,
        fin_count_convention='periodic effective length / axial pitch; fractional count; no end-placement correction',
        fin_thickness_convention='constant thickness; root and tip both equal actual plate thickness',
        A_fin_m2=n_tubes * tube.area_fin,
        A_fin_faces_m2=n_tubes * tube.area_fin_faces,
        A_fin_outer_edge_m2=n_tubes * tube.area_fin_outer_edge,
        fin_area_convention='physical two faces minus aperture plus outer edge; contact buried; adiabatic edge excluded from effective area',
        A_primary_outside_m2=n_tubes * tube.area_primary_outside,
        A_outside_geometric_m2=n_tubes * tube.area_outer,
        A_outside_gross_m2=n_tubes * tube.area_outer,
        bundle_total_inner_area_m2=n_tubes * tube.area_inner,
        bundle_total_outer_area_m2=n_tubes * tube.area_outer,
        physical_outside_area_m2=n_tubes * tube.area_outer,
    )
