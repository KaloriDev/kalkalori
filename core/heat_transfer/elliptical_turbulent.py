# SPDX-License-Identifier: GPL-3.0-only
"""Explicit smooth, fully developed turbulent ellipse engineering models.

HTC: Yang et al. (2018), Eq.18 and Table 4, doi:10.5098/hmt.11.7
(CC BY 4.0). The 10,000--120,000 source branch extrapolates with warnings
within the implemented turbulent regime (Re >= 4000).
Friction: Nikitin & Yakhot (2005), Eqs.3.2,3.5,
doi:10.1017/S0022112005003964. Their DNS checks b/a=.67,.5 at Re=6000;
other operating points are engineering extrapolations, not DNS validation.
"""
from dataclasses import dataclass
import math
from scipy.optimize import brentq

from core.common.warnings import ModelWarning, make_warning
from core.geometry.applicability import TubeGeometryApplicabilityError
from core.geometry.elliptical_tube import EllipticalTube
from core.heat_transfer.internal_model_contract import EllipticalFlowAssumptions, InternalModelMetadata

HTC_MODEL_ID = 'elliptical_yang2018_turbulent_T'
FRICTION_MODEL_ID = 'elliptical_jones_prandtl_turbulent_engineering'
YANG_REFERENCE = 'https://www.techscience.com/fhmt/v11n1/53399'
FRICTION_REFERENCE = 'https://doi.org/10.1017/S0022112005003964'


@dataclass(frozen=True)
class YangReExtrapolationWarning(ModelWarning):
    """Source-domain diagnostics, including machine-readable actual inputs."""
    reynolds: float
    source_lower: float
    source_upper: float
    direction: str


def _positive(**values):
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite [SI].')


def _in_range(value, lower, upper):
    # Only arithmetic roundoff at inclusive bounds, never extrapolation.
    return (lower <= value <= upper or math.isclose(value,lower,rel_tol=1e-12)
            or math.isclose(value,upper,rel_tol=1e-12))


def _validate(tube, assumptions, *, htc=False):
    if not isinstance(tube, EllipticalTube) or not isinstance(assumptions, EllipticalFlowAssumptions):
        raise TubeGeometryApplicabilityError('Turbulent ellipse models require EllipticalTube and explicit EllipticalFlowAssumptions.')
    if assumptions.regime != 'turbulent' or assumptions.development != 'fully_developed':
        raise TubeGeometryApplicabilityError('Declare turbulent, fully_developed ellipse flow explicitly.')
    if not all(v is True for v in (assumptions.constant_properties, assumptions.newtonian, assumptions.no_slip)):
        raise TubeGeometryApplicabilityError('Turbulent ellipse models require constant-property Newtonian no-slip flow.')
    if tube.roughness_inner not in (None, 0.):
        raise TubeGeometryApplicabilityError('Turbulent ellipse models are smooth-only.')
    if htc and (assumptions.thermal_boundary_condition != 'T' or
                assumptions.axial_conduction is not False or assumptions.viscous_dissipation is not False):
        raise TubeGeometryApplicabilityError('Yang 2018 requires T (constant wall temperature), negligible axial conduction and viscous dissipation.')


def _metadata(htc):
    return InternalModelMetadata(
        model_id=HTC_MODEL_ID if htc else FRICTION_MODEL_ID,
        provenance_id='yang2018_eq18_table4' if htc else 'nikitin_yakhot2005_eq3_2_eq3_5_engineering',
        source_references=(YANG_REFERENCE,) if htc else (FRICTION_REFERENCE,),
        source_access_basis='public primary full text; independently implemented equations',
        geometry_family='elliptical', regime='turbulent', development='fully_developed',
        thermal_boundary_condition='T' if htc else None,
        reference_dimension='inside_hydraulic_diameter', reference_velocity='mean_unblocked_inside_velocity',
        area_basis='actual_inside_perimeter_times_active_length' if htc else 'actual_unblocked_inside_flow_area',
        applicability_assumptions=(
            'smooth straight duct; constant-property Newtonian no-slip flow',
            'fully developed source assumption; no entrance correction',
            *(('10000 <= Re_Dh <= 120000', 'a/b = 1.5, 2, 2.5 or 3 <= a/b <= 5',
               'constant wall temperature CFD correlation; no universal Pr bounds established by source') if htc else
              ('Re_Dh >= 4000 and 1 <= a/b <= 5: implementation bounds, not validation',
               'DNS comparison only at Re_Dh=6000, b/a=.67,.5; other points extrapolate',
               'Darcy factor uses Jones modified Re; pressure gradient retains actual Dh')),
        ), approximation=True,
    )


def _shape_factor(ratio):
    # Table 4 supplies discrete coefficients below 3, not an interpolation law.
    for shape, factor in ((1.5, 1.), (2., 1.27), (2.5, 1.17)):
        if math.isclose(ratio, shape, rel_tol=1e-12):
            return factor
    if 3.-1e-12 <= ratio <= 5.+1e-12:
        return .94
    raise TubeGeometryApplicabilityError('Yang 2018 Table 4 supports a/b=1.5,2,2.5 or 3<=a/b<=5; no coefficient interpolation is inferred.')


@dataclass(frozen=True)
class EllipticalTurbulentFrictionResult:
    f_darcy: float
    reynolds: float
    modified_reynolds: float
    hydraulic_diameter: float
    metadata: InternalModelMetadata
    warnings: tuple

    @property
    def f_fanning(self):
        return self.f_darcy / 4.


def elliptical_turbulent_friction(tube, reynolds, *, assumptions):
    """Darcy factor; shape-adjusted Re in Prandtl, actual Dh in dp/dz."""
    _validate(tube, assumptions)
    _positive(reynolds=reynolds)
    ratio=1./tube.inside_aspect_ratio
    if reynolds < 4000 or ratio > 5.+1e-12:
        raise TubeGeometryApplicabilityError('Turbulent ellipse friction implementation requires Re_Dh>=4000 and 1<=a/b<=5.')
    # Dimensionless version of Eq.3.5, avoiding squared dimensional axes.
    beta=tube.inside_aspect_ratio
    dh_a=tube.hydraulic_diameter/tube.inner_semi_major_axis
    modified=reynolds*8*beta**2/(dh_a**2*(1+beta**2))
    f=brentq(lambda f: 1/math.sqrt(f)-2*math.log10(modified*math.sqrt(f))+.8,
             1e-5, 1., xtol=1e-14)
    warnings=(make_warning(
        code='elliptical_turbulent_friction_engineering', source=FRICTION_MODEL_ID, severity='warning',
        message='Jones-modified-Re Prandtl friction is an engineering approximation. Source DNS covers only Re_Dh=6000, b/a=0.67 and 0.5; other conditions extrapolate. Distributed straight friction only; no ellipse entrance/exit/bend validation.'),)
    return EllipticalTurbulentFrictionResult(f,reynolds,modified,tube.hydraulic_diameter,_metadata(False),warnings)


def elliptical_turbulent_heat_transfer(tube, m_dot, props, *, assumptions=None, flow_area=None,
                                      L_heated=None, strict_source_domain=False):
    """Yang Eq.18; no circular length/wall-property corrections or fallback."""
    if not isinstance(strict_source_domain, bool):
        raise TypeError('strict_source_domain must be boolean.')
    if assumptions is None:
        assumptions = EllipticalFlowAssumptions('turbulent', 'fully_developed', 'T')
    _validate(tube, assumptions, htc=True)
    from core.heat_transfer.internal_flow import InternalHeatTransferDiagnostics
    area=tube.flow_area if flow_area is None else flow_area
    length=tube.length_effective if L_heated is None else L_heated
    _positive(m_dot=m_dot,flow_area=area,rho=props.rho,mu=props.mu,k=props.k,cp=props.cp,length=length)
    v=m_dot/(props.rho*area)
    dh=tube.hydraulic_diameter
    re=props.rho*v*dh/props.mu
    pr=props.cp*props.mu/props.k
    if re < 4000:
        raise TubeGeometryApplicabilityError('Yang 2018 Eq.18 requires turbulent Re_Dh>=4000; laminar/transition flow is unsupported by this provider.')
    extrapolated = not _in_range(re,10000,120000)
    if extrapolated and strict_source_domain:
        raise TubeGeometryApplicabilityError('Yang 2018 Eq.18 strict source domain requires 10000<=Re_Dh<=120000.')
    c=_shape_factor(1./tube.inside_aspect_ratio)
    nu=c*(.02979*re**.8-8.5626)*pr**.4
    alpha=nu*props.k/dh
    warnings=[make_warning(
        code='elliptical_yang2018_CFD_assumptions', source=HTC_MODEL_ID, severity='warning',
        message='Yang 2018 is a fully developed constant-wall-temperature CFD fit, evaluated at bulk properties. Its use in a 0D exchanger is an engineering approximation; source does not establish universal Pr bounds. No entrance or variable-wall correction is applied.')]
    if extrapolated:
        direction = 'below' if re < 10000 else 'above'
        warnings.append(YangReExtrapolationWarning(code='yang2018_re_extrapolation', source=HTC_MODEL_ID,
            severity='warning', reynolds=re, source_lower=10000., source_upper=120000., direction=direction,
            message=f'Yang 2018 turbulent elliptical-tube correlation extrapolated {direction} source Reynolds range: Re_Dh={re:.12g}, source lower=10000, source upper=120000; no clipping.'))
    if length/dh < 60:
        warnings.append(make_warning(code='elliptical_yang2018_short_duct',source=HTC_MODEL_ID,severity='warning',
            message='Heated length is below 60 Dh, the source development guidance; the fully developed model assumption may be weak/extrapolated.'))
    # Carry hydraulic approximation scope into normal combined result warnings.
    warnings.extend(elliptical_turbulent_friction(tube,re,assumptions=assumptions).warnings)
    return InternalHeatTransferDiagnostics(
        v=v,Re=re,Pr=pr,Nu_base=nu,Nu_corrected=nu,
        length_correction=1.,wall_temperature_correction=1.,combined_correction=1.,
        alfa_base=alpha,alfa_corrected=alpha,warnings=warnings,model_metadata=_metadata(True))
