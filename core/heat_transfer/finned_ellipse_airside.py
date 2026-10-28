# SPDX-License-Identifier: GPL-3.0-only
"""Zhao et al. (2017), IJHMT 114, 1314-1330, Eqs.18-20.

DOI 10.1016/j.ijheatmasstransfer.2017.06.049. Equations transcribed from the
user-authorized 2026-10-20 source package; no claim of free full-text access.
Eq.20 excludes fin conduction. Eq.18 is diagnostic only. Flow normalization
is a separately identified engineering reconstruction, not a printed equation.
"""
from dataclasses import dataclass, replace
import math

from core.common.warnings import ModelWarning, make_warning
from core.geometry.applicability import TubeGeometryApplicabilityError
from core.geometry.rectangular_finned_tube import RectangularFinnedEllipticalTube
from core.geometry.lattice import LayoutClass
from core.heat_transfer.airside_correction import AirsideCorrection, AirsideCorrectionProvider


ZHAO2017_CITATION = (
    'Zhao, Gu, Gao, Yang (2017), IJHMT 114, 1314-1330; '
    'doi:10.1016/j.ijheatmasstransfer.2017.06.049')
ZHAO2017_IMPLEMENTED_LATTICE = 'conventional_staggered:whole_row_half_pitch:x=flow,y=transverse'
# Recovered Fig. 2 resolves the lattice and Pt/Pl axes; normalization below
# remains a separately graded engineering reconstruction.
ZHAO2017_SOURCE_LATTICE_GRADE = 'SOURCE_EXACT:ZHAO2017_FIG2;Pt=TRANSVERSE;Pl=LONGITUDINAL'
ZHAO2017_DOMAIN = (
    ('Xt', .93, 1.50), ('Xl', .56, .78), ('Xp', .10, .52),
    ('Xf', .01, .03), ('Re', 5500., 23800.), ('N', 6., 10.))


def _positive(**values):
    for name, value in values.items():
        try:
            valid = not isinstance(value,bool) and math.isfinite(value) and value>0
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f'{name} must be positive and finite [SI].')


def _extrapolation(values, strict):
    outside = tuple(name for name, lower, upper in ZHAO2017_DOMAIN
                    if not lower <= values[name] <= upper)
    if outside and strict:
        raise TubeGeometryApplicabilityError(
            f'Zhao2017 source domain exceeded: {", ".join(outside)}')
    warnings = tuple(make_warning(
        code=f'zhao2017_extrapolated_{name}', source='zhao2017',
        message=f'Zhao2017 extrapolation: {name}={values[name]:.12g} is outside its source domain; no clipping.')
        for name in outside)
    return outside, warnings


@dataclass(frozen=True)
class Zhao2017Factors:
    """Dimensionless equation outputs, independent of flow reconstruction."""
    j_effective: float
    j_prime: float
    friction_factor: float
    extrapolated_parameters: tuple[str, ...]
    warnings: tuple[ModelWarning, ...]


def zhao2017_factors(*, n_rows, reynolds, Xt, Xl, Xp, Xf,
                     strict_source_domain=False):
    """Evaluate source Eqs.18/19/20 exactly; never silently clamp inputs.

    The source fit uses rows 6..10. Other positive integer counts are an
    explicit extrapolation, not a new measured row law.
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
        raise ValueError('n_rows must be a positive integer.')
    _positive(reynolds=reynolds, Xt=Xt, Xl=Xl, Xp=Xp, Xf=Xf)
    if not isinstance(strict_source_domain, bool):
        raise TypeError('strict_source_domain must be boolean.')
    outside, warnings = _extrapolation(
        dict(N=n_rows, Re=reynolds, Xt=Xt, Xl=Xl, Xp=Xp, Xf=Xf),
        strict_source_domain)
    j_effective = (1.0062*n_rows**(-.2045)*Xt**(-.7268)*Xl**.1319
                   *Xp**.1381*Xf**.1495*reynolds**(-.4408))
    friction = (1.3549*n_rows**(-.0591)*Xt**(-.9114)*Xl**.4709
                *Xp**.1296*Xf**.0438*reynolds**(-.3164))
    j_prime = (.2*n_rows**(-.2160)*Xt**(-.82)*Xl**.1361
               *Xp**.0043*Xf**(-.0578)*reynolds**(-.3326))
    _positive(j_effective=j_effective, j_prime=j_prime, friction_factor=friction)
    return Zhao2017Factors(j_effective, j_prime, friction, outside, warnings)


@dataclass(frozen=True)
class Zhao2017Flow:
    reference_diameter: float
    frontal_area: float
    average_passage_area: float
    periodic_gross_section: float
    periodic_average_open_section: float
    sigma_average: float
    sigma_throat: float
    face_velocity: float
    reference_velocity: float
    maximum_velocity_estimate: float
    reynolds_number: float
    prandtl_number: float
    Xt: float
    Xl: float
    Xp: float
    Xf: float
    geometry_extrapolation: tuple[str, ...]
    diameter_basis: str = 'ellipse_4area_over_perimeter_source_inferred'
    passage_area_basis: str = 'periodic_streamwise_volume_average_engineering_reconstruction'


def validate_zhao2017_geometry(bundle):
    """Reject unsupported physical geometry before any property evaluation."""
    tube = bundle.tube
    if not isinstance(tube, RectangularFinnedEllipticalTube):
        raise TubeGeometryApplicabilityError('Zhao2017 requires RectangularFinnedEllipticalTube.')
    core, fin = tube.core_tube, tube.fin
    if not math.isclose(math.remainder(core.orientation_angle, math.pi), 0., abs_tol=1e-12):
        raise TubeGeometryApplicabilityError('Zhao2017 reconstruction requires major axis parallel to flow.')
    if bundle.tube_bank_lattice.layout_class != LayoutClass.CONVENTIONAL_STAGGERED:
        raise TubeGeometryApplicabilityError(
            'Zhao2017 requires its conventional whole-row staggered lattice; INLINE and SHIFTED are unsupported, including unresolved phases.')
    Dl, Dt = 2*core.outer_semi_major_axis, 2*core.outer_semi_minor_axis
    Pt, Pl = bundle.pitch_transverse, bundle.pitch_longitudinal
    fin_streamwise_extent, fin_transverse_extent = fin.outer_height, fin.outer_width
    Wx, Wy, Ft, Fp = fin_streamwise_extent, fin_transverse_extent, fin.thickness, fin.axial_pitch
    if Pt <= Dt or Pl <= Dl or Pt < Wy or Pl < Wx:
        raise TubeGeometryApplicabilityError('Zhao2017 reconstruction requires positive tube gaps and nonoverlapping fin envelopes.')
    return tube


def zhao2017_flow(bundle, m_dot, props):
    """Reconstruct average open passage from actual geometry, with continuity.

    This is NOT the source's printed free-area formula or a circular-bank
    throat. Source Fs/pitch and fin details follow the authorized package.
    No pass count, tube length or total tube count enters dimensionless ratios.
    """
    tube=validate_zhao2017_geometry(bundle)
    core,fin=tube.core_tube,tube.fin
    Dl,Dt=2*core.outer_semi_major_axis,2*core.outer_semi_minor_axis
    Pt,Pl=bundle.pitch_transverse,bundle.pitch_longitudinal
    fin_streamwise_extent,fin_transverse_extent=fin.outer_height,fin.outer_width
    Wx,Wy,Ft,Fp=fin_streamwise_extent,fin_transverse_extent,fin.thickness,fin.axial_pitch
    _positive(m_dot=m_dot, rho=props.rho, mu=props.mu, cp=props.cp, k=props.k,
              frontal_area=bundle.frontal_flow_area)
    Se = tube.aperture_area_per_fin
    de = 4*Se/core.outside_perimeter
    Sg = Pt*Fp
    Savg = Sg - ((Fp-Ft)*Se + Ft*Wx*Wy)/Pl
    sigma = Savg/Sg
    sigma_throat = (Sg - (Fp-Ft)*Dt - Ft*Wy)/Sg
    if not (math.isfinite(sigma) and 0 < sigma <= 1
            and math.isfinite(sigma_throat) and 0 < sigma_throat <= 1):
        raise TubeGeometryApplicabilityError('Zhao2017 reconstruction leaves no positive finite open flow area.')
    Aa = bundle.frontal_flow_area*sigma
    u_fr = m_dot/(props.rho*bundle.frontal_flow_area)
    um = m_dot/(props.rho*Aa)
    Re = props.rho*um*de/props.mu
    Pr = props.cp*props.mu/props.k
    _positive(de=de, Aa=Aa, u_frontal=u_fr, u_m=um, Re=Re, Pr=Pr)
    mismatches = tuple(name for name, value, source in (
        ('tube_major',Dl,.036),('tube_minor',Dt,.014),
        ('fin_height',Wx,.055),('fin_width',Wy,.026))
        if not math.isclose(value,source,rel_tol=1e-10))
    if bundle.layout.lower() != 'staggered':
        mismatches += ('layout',)
    return Zhao2017Flow(de,bundle.frontal_flow_area,Aa,Sg,Savg,sigma,sigma_throat,
        u_fr,um,u_fr/sigma_throat,Re,Pr,(Pt-Dt)/Dt,(Pl-Dl)/Dl,Fp/de,Ft/de,mismatches)


@dataclass(frozen=True)
class Zhao2017AirsideResult:
    """Physical h and source-domain pressure estimate; no eta is applied here."""
    flow: Zhao2017Flow
    factors: Zhao2017Factors
    h_film_base: float
    h_film: float
    pressure_drop_base: float
    pressure_drop: float
    warnings: tuple[ModelWarning, ...]
    physical_area: float
    extrapolated_parameters: tuple[str, ...]
    correction_provenance: str = 'none'
    method: str = 'zhao2017_reconstructed'
    citation: str = ZHAO2017_CITATION
    equations: tuple[int, ...] = (18,19,20)
    equation_basis: str = 'source_exact_user_supplied_transcription'
    property_basis: str = 'caller_bulk_state; engine_mean_approximates_CFD_average_flow_field'
    pressure_scope: str = 'source_domain_inlet_to_outlet_estimate'
    pressure_model: str = 'zhao2017_source_normalization_reconstructed'
    validated_bank_irreversible: bool = False
    fin_turbulator_count: int = 0
    implemented_lattice: str = ZHAO2017_IMPLEMENTED_LATTICE
    source_lattice_grade: str = ZHAO2017_SOURCE_LATTICE_GRADE

    @property
    def correction_applied(self) -> bool:
        """Whether an explicit provider changed either physical result channel."""
        return self.h_film != self.h_film_base or self.pressure_drop != self.pressure_drop_base

    @property
    def within_source_domain(self):
        """Ratio/envelope check within the conventional-staggered provider.

        Source lattice/axes are resolved separately in source_lattice_grade.
        Passing this check does not validate engineering flow normalization
        or extend applicability to staggered-column / SHIFTED geometry.
        """
        return not self.extrapolated_parameters


@dataclass(frozen=True)
class Zhao2017AirsideModel:
    """Authorized engineering baseline; extrapolate with warnings by default."""
    strict_source_domain: bool = False
    correction_provider: AirsideCorrectionProvider | None = None

    def __post_init__(self):
        if not isinstance(self.strict_source_domain, bool):
            raise TypeError('strict_source_domain must be boolean.')
        if self.correction_provider is not None and not callable(getattr(self.correction_provider,'correct',None)):
            raise TypeError('correction_provider must implement correct(base_result, *, bundle, props).')

    def evaluate(self, bundle, m_dot, props):
        flow = zhao2017_flow(bundle,m_dot,props)
        f = zhao2017_factors(n_rows=bundle.n_rows, reynolds=flow.reynolds_number,
            Xt=flow.Xt,Xl=flow.Xl,Xp=flow.Xp,Xf=flow.Xf,
            strict_source_domain=self.strict_source_domain)
        if flow.geometry_extrapolation and self.strict_source_domain:
            raise TubeGeometryApplicabilityError('Zhao2017 source geometry exceeded: '+', '.join(flow.geometry_extrapolation))
        warnings = list(f.warnings)
        warnings.append(make_warning(code='source_geometry_simplified_rectangular_fin',source='zhao2017',
            message='Source fin has holes, positioning tabs and clipped corners; calculation uses an ideal full rectangle.'))
        warnings.append(make_warning(code='zhao2017_reconstructed_normalization',source='zhao2017',
            message='Diameter is source-inferred 4A/P; passage area/velocity are an engineering reconstruction; caller bulk properties approximate the CFD mean state.'))
        warnings.append(make_warning(code='zhao2017_pressure_scope',source='zhao2017',
            message='Pressure is a reconstructed source-domain inlet-to-outlet estimate; bank-only irreversible equivalence and entry/exit/acceleration decomposition are not established. No separate losses are added.'))
        if flow.geometry_extrapolation:
            warnings.append(make_warning(code='zhao2017_geometry_extrapolation',source='zhao2017',
                message='Zhao2017 geometry extrapolation: '+', '.join(flow.geometry_extrapolation)))
        h = f.j_prime*props.rho*props.cp*flow.reference_velocity/flow.prandtl_number**(2/3)
        dp = f.friction_factor*props.rho*flow.reference_velocity**2/2*(bundle.total_outer_area/flow.average_passage_area)
        _positive(h_film=h, pressure_estimate=dp)
        baseline = Zhao2017AirsideResult(flow,f,h,h,dp,dp,tuple(warnings),bundle.total_outer_area,
            f.extrapolated_parameters+flow.geometry_extrapolation,
            fin_turbulator_count=bundle.tube.fin.effective_fin_turbulator_count)
        result = baseline
        if self.correction_provider is not None:
            correction = self.correction_provider.correct(baseline,bundle=bundle,props=props)
            if not isinstance(correction,AirsideCorrection):
                raise TypeError('Airside correction provider must return AirsideCorrection.')
            corrected_h,corrected_dp=correction.apply(h,dp)
            notice=make_warning(code='airside_correction_applied',source='airside_correction',severity='info',
                message='Explicit airside correction: '+correction.provenance)
            result=replace(baseline,h_film=corrected_h,pressure_drop=corrected_dp,
                correction_provenance=correction.provenance,warnings=baseline.warnings+(notice,))
        if result.fin_turbulator_count > 0 and not result.correction_applied:
            notice=make_warning(code='fin_turbulators_present_uncorrected',source='fin_geometry',severity='info',
                message='Fin turbulator count is metadata only; the unchanged Zhao baseline has no count-specific correction or validation.')
            result=replace(result,warnings=result.warnings+(notice,))
        return result
