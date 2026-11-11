# SPDX-License-Identifier: GPL-3.0-only
"""Explicit experimental Schmidt equivalent-cell approximation.

The small Schmidt kernel is adapted from python-hvac PlainContinuousFin,
commit 94247560a667905bfbf67e45a58965a58254fce8, fins.py. Copyright (c) 2023
Tom Christiaens. MIT notice: core/heat_transfer/python_hvac_MIT.txt.
Scientific provenance: Schmidt (1949), Heat Transfer Calculations for Extended
Surfaces; Thulukkanam (2013), Heat Exchanger Design Handbook, section 4.7.2.2.

M and L are distances to the transverse/diagonal perpendicular bisectors.
Finite sheet edge cells are NOT resolved by this periodic approximation.
That limitation is explicit and remains part of release gate B05. The physical
surface ledger is always finite. Sheet edges/lips are adiabatic in this model.
"""
from dataclasses import dataclass
import math

from core.geometry.lattice import LayoutClass
from core.common.warnings import make_warning
from core.heat_transfer.continuous_fin_source import ContinuousFinSourceError


@dataclass(frozen=True)
class ContinuousFinEfficiencyResult:
    fin_efficiency: float
    overall_surface_efficiency: float
    effective_area: float
    equivalent_radius: float
    M: float
    L: float
    cell_basis: str = "periodic_staggered_bisector_hexagon; finite_edge_cells_unresolved"
    tip_temperature_ratio: None = None
    warnings: tuple = ()
    two_face_efficiency: float = 1.
    single_face_efficiency: float = 1.
    single_face_sheet_count: int = 0


@dataclass(frozen=True)
class SchmidtEquivalentCell:
    """Research-only hexagonal cells, with both tube pitches retained.

    The admitted topology has six Voronoi neighbours (Pl/Pt >= 1/2).
    No claim is made for unresolved shifted/inline or degenerate cells.
    """
    def evaluate(self, bundle, h):
        if isinstance(h, bool) or not math.isfinite(h) or h < 0:
            raise ValueError("h must be finite and nonnegative.")
        if bundle.continuous_fin_geometry is None or bundle.tube_bank_lattice.layout_class is not LayoutClass.CONVENTIONAL_STAGGERED:
            raise ContinuousFinSourceError("Schmidt continuous cell requires explicit conventional staggered sheets.")
        pt, pl = bundle.pitch_transverse, bundle.pitch_longitudinal
        if pl < pt/2:
            raise ContinuousFinSourceError("Schmidt six-neighbour cell requires Pl/Pt >= 1/2; no regular-hexagon substitution.")
        # Section-parity resets can create adjacent unshifted rows. Such a bank
        # has no single conventional periodic cell even though its enum is STAGGERED.
        rows = bundle.tube_center_coordinates
        if any(math.isclose(rows[j][0][1], rows[j-1][0][1], abs_tol=1e-12) for j in range(1, len(rows))):
            raise ContinuousFinSourceError("Schmidt requires alternating physical rows, including section boundaries.")
        fin, surface = bundle.continuous_fin_geometry, bundle.continuous_fin_surface
        r = fin.collar_outer_diameter/2
        M, L = pt/2, math.hypot(pl, pt/2)/2
        radius_ratio = 1.27*(M/r)*math.sqrt(L/M-.3)
        if radius_ratio <= 1:
            raise ContinuousFinSourceError("Schmidt equivalent cell must extend beyond collar radius.")
        radius = radius_ratio*r
        # Adiabatic cell boundary: unlike upstream's exposed-tip correction,
        # do not append t/2 at a symmetry boundary shared with the next cell.
        corrected_length = (radius-r)*(1+.35*math.log(radius_ratio))
        def face_efficiency(exposed_faces):
            # Thin-sheet energy balance: m^2 = exposed_faces*h/(k*t).
            argument = math.sqrt(exposed_faces*h/(fin.conductivity*fin.thickness))*corrected_length
            return 1. if argument == 0 else math.tanh(argument)/argument
        eta_two, eta_one = face_efficiency(2), face_efficiency(1)
        one_face_area = fin.height*fin.depth-bundle.n_tubes_total*math.pi*r*r
        single_sheets = round(surface.sealed_fin_face_area/one_face_area)
        double_sheets = fin.sheet_count-single_sheets
        exposed_faces = 2*double_sheets+single_sheets
        eta = (2*double_sheets*eta_two+single_sheets*eta_one)/exposed_faces
        active = surface.primary_area+eta*surface.fin_face_area
        warning = make_warning(code="CONTINUOUS_FIN_EDGE_CELL_APPROXIMATION",
            message="Schmidt uses pitch-dependent periodic cells; finite boundary cells and inter-zone common-sheet conduction remain unvalidated. Edges/lips are adiabatic.",
            source="schmidt_equivalent_cell", severity="warning")
        return ContinuousFinEfficiencyResult(eta, active/surface.total_area, active, radius, M, L,
            warnings=(warning,), two_face_efficiency=eta_two, single_face_efficiency=eta_one,
            single_face_sheet_count=single_sheets)
