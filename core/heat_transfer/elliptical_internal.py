# SPDX-License-Identifier: GPL-3.0-only
"""Exact, constant-property, fully developed smooth elliptical duct models.

Public governing continuum problem and elliptical parabolic velocity:
Su et al. (2019), https://pmc.ncbi.nlm.nih.gov/articles/PMC6843173/.
Perimeter: https://dlmf.nist.gov/19.30#E5.
The closed friction/H1 expressions below are independent analytical
reconstructions, not an empirical fit or a quotation from a private handbook.
"""
from dataclasses import dataclass
import math
from scipy.special import ellipe
from core.common.warnings import make_warning
from core.geometry.applicability import TubeGeometryApplicabilityError
from core.geometry.elliptical_tube import EllipticalTube
from core.heat_transfer.internal_model_contract import EllipticalFlowAssumptions, InternalModelMetadata


FRICTION_MODEL_ID = 'elliptical_fd_laminar_darcy_exact'
H1_MODEL_ID = 'elliptical_fd_laminar_H1_exact'
_REFERENCES = ('https://pmc.ncbi.nlm.nih.gov/articles/PMC6843173/',
               'https://dlmf.nist.gov/19.30#E5')


def _validate_assumptions(tube, assumptions, *, htc=False):
    if not isinstance(tube, EllipticalTube):
        raise TubeGeometryApplicabilityError('Exact elliptical models require EllipticalTube geometry.')
    if not isinstance(assumptions, EllipticalFlowAssumptions):
        raise TubeGeometryApplicabilityError('Elliptical models require explicit EllipticalFlowAssumptions.')
    if assumptions.regime != 'laminar' or assumptions.development != 'fully_developed':
        raise TubeGeometryApplicabilityError('Only explicitly declared fully_developed laminar ellipse flow is supported.')
    if not all(value is True for value in (assumptions.constant_properties, assumptions.newtonian, assumptions.no_slip)):
        raise TubeGeometryApplicabilityError('Exact ellipse models require constant-property Newtonian no-slip flow.')
    if tube.roughness_inner not in (None, 0.):
        raise TubeGeometryApplicabilityError('Exact elliptical internal models are smooth-only; positive roughness is unsupported.')
    if htc and (assumptions.thermal_boundary_condition != 'H1' or
                assumptions.axial_conduction is not False or assumptions.viscous_dissipation is not False):
        raise TubeGeometryApplicabilityError('Only H1 with negligible axial conduction and viscous dissipation is supported; T/H2 are not.')


def _metadata(htc=False):
    return InternalModelMetadata(
        model_id=H1_MODEL_ID if htc else FRICTION_MODEL_ID,
        provenance_id='analytical_elliptical_energy_H1_v1' if htc else 'analytical_elliptical_poiseuille_v1',
        source_references=_REFERENCES, source_access_basis='public governing problem; independent analytical reconstruction',
        geometry_family='elliptical', regime='laminar', development='fully_developed',
        thermal_boundary_condition='H1' if htc else None,
        reference_dimension='inside_hydraulic_diameter', reference_velocity='mean_unblocked_inside_velocity',
        area_basis=('actual_inside_perimeter_times_active_length' if htc
                    else 'actual_unblocked_inside_flow_area'),
        applicability_assumptions=('0 < inside b/a <= 1', 'smooth no-slip Newtonian fluid',
            'constant properties', 'laminar and fully developed declared by caller; not inferred from Re',
            *(('H1: constant axial heat input, uniform wall temperature around each section',
              'negligible axial conduction and viscous dissipation') if htc else ())),
    )


@dataclass(frozen=True)
class EllipticalFrictionResult:
    f_darcy: float
    reynolds: float
    poiseuille_number: float
    hydraulic_diameter: float
    metadata: InternalModelMetadata

    @property
    def f_fanning(self):
        return self.f_darcy / 4.


def elliptical_laminar_friction(tube, reynolds, *, assumptions):
    """Darcy f with Re based on true Dh and mean unblocked inside velocity.

    u=2*u_mean*(1-x²/a²-y²/b²) -> dp/dz=4*mu*u_mean*(a²+b²)/(a²*b²).
    Darcy-Weisbach and Dh=pi*b/E give f_D*Re_Dh=8*(1+beta²)*(pi/E)².
    """
    _validate_assumptions(tube, assumptions)
    _positive(reynolds=reynolds)
    beta=tube.inside_aspect_ratio
    E=float(ellipe(1-beta**2))  # SciPy parameter m, not modulus sqrt(m).
    poiseuille=8*(1+beta**2)*(math.pi/E)**2
    f=poiseuille/reynolds
    _positive(f_darcy=f)
    return EllipticalFrictionResult(f,reynolds,poiseuille,tube.hydraulic_diameter,_metadata())


def elliptical_h1_heat_transfer(tube, m_dot, props, *, assumptions, flow_area=None):
    """Exact H1 HTC [W/(m2 K)] for the declared ideal fully developed state.

    Matching k*Laplacian(T)=rho*cp*u*dTb/dz with
    Tw-T=(1-X²-Y²)*(C0+Cx*X²+Cy*Y²), then velocity-weighting the
    bulk temperature, gives the polynomial ratio below (X=x/a, Y=y/b).
    H1 is not local uniform flux H2 and not constant wall temperature T.
    """
    _validate_assumptions(tube, assumptions, htc=True)
    from core.heat_transfer.internal_flow import InternalHeatTransferDiagnostics, mean_velocity
    area=tube.flow_area if flow_area is None else flow_area
    _positive(m_dot=m_dot,flow_area=area,rho=props.rho,mu=props.mu,k=props.k,cp=props.cp)
    v=mean_velocity(m_dot,props.rho,area)
    dh=tube.hydraulic_diameter
    Re=props.rho*v*dh/props.mu
    Pr=props.cp*props.mu/props.k
    beta2=tube.inside_aspect_ratio**2
    E=float(ellipe(1-beta2))
    Nu=(3*math.pi/E)**2*((1+beta2)*(1+6*beta2+beta2**2))/(17+98*beta2+17*beta2**2)
    alpha=Nu*props.k/dh
    _positive(velocity=v,reynolds=Re,prandtl=Pr,alpha=alpha)
    return InternalHeatTransferDiagnostics(
        v=v,Re=Re,Pr=Pr,Nu_base=Nu,Nu_corrected=Nu,
        length_correction=1.,wall_temperature_correction=1.,combined_correction=1.,
        alfa_base=alpha,alfa_corrected=alpha,
        warnings=[make_warning(code='elliptical_fd_H1_assumptions',source=H1_MODEL_ID,severity='info',
            message='Laminar, fully developed H1 and constant-property assumptions are declared by the caller; no transition/entrance criterion is inferred.')],
        model_metadata=_metadata(htc=True),
    )


@dataclass(frozen=True)
class EllipticalDistributedPressureDrop:
    pressure_drop: float  # [Pa], distributed friction only
    pressure_gradient: float  # [Pa/m]
    velocity: float
    mass_flux: float
    friction: EllipticalFrictionResult
    length: float
    calculation_scope: str = 'distributed_fully_developed_friction_only'
    local_losses_included: bool = False


def elliptical_distributed_pressure_drop(tube, m_dot, props, *, assumptions, length=None, flow_area=None):
    """Exact FD constant-property straight friction; no inlet/exit/bend claims.

    ``flow_area`` may explicitly supply total parallel-tube flow area. The
    default is one tube. Length defaults to the tube's total hydraulic length.
    """
    _validate_assumptions(tube,assumptions)
    area=tube.flow_area if flow_area is None else flow_area
    L=tube.length_total if length is None else length
    _positive(m_dot=m_dot,flow_area=area,rho=props.rho,mu=props.mu,length=L)
    G=m_dot/area
    velocity=G/props.rho
    reynolds=G*tube.hydraulic_diameter/props.mu
    # The same geometry-aware law is selected before Darcy integration.
    from core.heat_transfer.internal_dispatch import internal_friction
    f,model=internal_friction(tube=tube,reynolds=reynolds,assumptions=assumptions)
    friction=elliptical_laminar_friction(tube,reynolds,assumptions=assumptions)
    gradient=f*props.rho*velocity**2/(2*tube.hydraulic_diameter)
    _positive(pressure_gradient=gradient,pressure_drop=gradient*L)
    return EllipticalDistributedPressureDrop(gradient*L,gradient,velocity,G,friction,L)


def _positive(**values):
    for name,value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite [SI].')
