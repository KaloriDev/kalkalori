# SPDX-License-Identifier: GPL-3.0-only
"""Physical wall resistance [K/W], selected independently of surface fins."""
import math
from dataclasses import dataclass
from enum import Enum
from core.geometry.applicability import require_circular_tube, tube_cross_section_family, TubeGeometryApplicabilityError
from core.geometry.tube import TubeCrossSectionFamily
from core.heat_transfer.elliptical_wall import EllipticalWallConvergence

# Relative error in focal-square equality, scaled by the largest semi-axis
# squared. This is floating-point tolerance, not a manufacturing allowance.
CONFOCAL_RELATIVE_TOLERANCE = 1.0e-12


class TubeWallModel(str, Enum):
    CYLINDRICAL = "cylindrical"
    CONFOCAL_EXACT = "confocal_exact"
    ELLIPTICAL_NUMERICAL = "elliptical_numerical"


@dataclass(frozen=True)
class WallResistanceResult:
    resistance: float  # total [K/W]
    model_id: str
    geometry_family: str
    source_references: tuple[str, ...]
    assumptions: tuple[str, ...]
    approximation: bool = False


@dataclass(frozen=True, kw_only=True)
class EllipticalNumericalWallResult(WallResistanceResult):
    convergence: 'EllipticalWallConvergence'


def _numerical_wall_result(tube, n_tubes):
    from core.heat_transfer.elliptical_wall import elliptical_wall_solution
    solution=elliptical_wall_solution(tube)
    resistance=1/(solution.unit_conductance*tube.wall_k*tube.length_effective*n_tubes)
    if not math.isfinite(resistance) or resistance<=0:
        raise ValueError('Numerical wall resistance must be positive and representable [K/W].')
    return EllipticalNumericalWallResult(
        resistance=resistance,model_id='elliptical_wall_laplace_P1',geometry_family='elliptical',
        source_references=('https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html',),
        assumptions=('independent P1 weak Laplace solve; concentric coaxial ellipse contours',
                     'constant conductivity; isothermal inner/outer contours',
                     'quarter-section symmetry; negligible axial conduction and end effects',
                     'measured mesh convergence, not a certified continuum error bound'),
        approximation=True,convergence=solution)


def tube_wall_resistance_diagnostics(tube, *, n_tubes=1, model=None):
    resistance=tube_wall_resistance(tube,n_tubes=n_tubes,model=model)
    if model == TubeWallModel.ELLIPTICAL_NUMERICAL:
        return _numerical_wall_result(tube,n_tubes)
    family=tube_cross_section_family(tube)
    is_ellipse=family is TubeCrossSectionFamily.ELLIPTICAL
    return WallResistanceResult(
        resistance=resistance,
        model_id='elliptical_confocal_wall_exact' if is_ellipse else 'cylindrical_wall_exact',
        geometry_family=family.value,
        source_references=('https://www.mhtlab.uwaterloo.ca/pdf_papers/mhtl04-5.pdf',) if is_ellipse else (),
        assumptions=('steady Laplace conduction', 'constant conductivity', 'isothermal contour surfaces',
                     'negligible end effects', 'coaxial confocal contours' if is_ellipse else 'circular contours'),
    )


def tube_wall_resistance(tube, *, n_tubes=1, model=None):
    if isinstance(n_tubes, bool) or not isinstance(n_tubes, int) or n_tubes <= 0:
        raise ValueError("n_tubes must be a positive integer.")
    if tube_cross_section_family(tube) is TubeCrossSectionFamily.ELLIPTICAL:
        if model == TubeWallModel.ELLIPTICAL_NUMERICAL:
            return _numerical_wall_result(tube,n_tubes).resistance
        if model != TubeWallModel.CONFOCAL_EXACT:
            raise TubeGeometryApplicabilityError(
                "Elliptical wall conduction requires explicit CONFOCAL_EXACT or "
                "ELLIPTICAL_NUMERICAL selection; thin-wall conduction is unsupported."
            )
        return confocal_wall_resistance(tube, n_tubes=n_tubes)
    require_circular_tube(tube, "Cylindrical wall conduction")
    if model not in (None, TubeWallModel.CYLINDRICAL):
        raise TubeGeometryApplicabilityError("Select cylindrical wall conduction for circular tubes.")
    k = getattr(tube, "wall_k", None)
    if k is None:
        return 0.0
    if k <= 0.0:
        raise ValueError("wall_k must be positive.")
    Di = float(getattr(tube, "D_i"))
    Do = float(getattr(tube, "D_o"))
    L_eff = float(getattr(tube, "length_effective"))
    if Do <= Di:
        raise ValueError("Tube outer diameter must exceed inner diameter.")
    # Preserve the historical circular expression AND evaluation order.
    # Cylindrical steady Laplace solution; no equivalent diameter substitution.
    return math.log(Do / Di) / (2.0 * math.pi * k * L_eff * n_tubes)


def confocal_wall_resistance(tube, *, n_tubes=1):
    """Exact coaxial confocal steady wall R [K/W], active axial length.

    Constant conductivity, negligible end effects and isothermal inner/outer
    surfaces. Analytical Laplace solution in elliptical coordinates:
    a=c*cosh(mu), b=c*sinh(mu), Q=2*pi*k*L*dT/(mu_o-mu_i).
    Public coordinate mapping: Khan, Culham & Yovanovich (2004),
    https://www.mhtlab.uwaterloo.ca/pdf_papers/mhtl04-5.pdf.
    The steady resistance is an independent derivation, not a published fit.
    """
    from core.geometry.elliptical_tube import EllipticalTube
    if not isinstance(tube, EllipticalTube):
        raise TubeGeometryApplicabilityError('Confocal wall model requires EllipticalTube geometry.')
    if isinstance(n_tubes, bool) or not isinstance(n_tubes, int) or n_tubes <= 0:
        raise ValueError('n_tubes must be a positive integer.')
    ao,bo=tube.outer_semi_major_axis,tube.outer_semi_minor_axis
    ai,bi=tube.inner_semi_major_axis,tube.inner_semi_minor_axis
    # Normalize first: avoid overflow, underflow and a unit-dependent abs_tol.
    scale=ao
    outer=(ao/scale-bo/scale)*(ao/scale+bo/scale)
    inner=(ai/scale-bi/scale)*(ai/scale+bi/scale)
    if abs(outer-inner) > CONFOCAL_RELATIVE_TOLERANCE:
        raise TubeGeometryApplicabilityError(
            'CONFOCAL_EXACT requires equal inner/outer focal squares; concentric is not confocal.'
        )
    # log1p retains the log((ao+bo)/(ai+bi)) solution for a very small wall gap.
    log_ratio=math.log1p(((ao-ai)+(bo-bi))/(ai+bi))
    resistance=log_ratio/(2.0*math.pi*tube.wall_k*tube.length_effective*n_tubes)
    if not math.isfinite(resistance) or resistance <= 0:
        raise ValueError('Confocal wall resistance must be positive and representable [K/W].')
    return resistance
