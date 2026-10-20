# SPDX-License-Identifier: GPL-3.0-only
"""Select internal constitutive physics before interpreting reference lengths.

Existing scalar circular APIs retain their meanings. Geometry-based callers
use this boundary so a hydraulic diameter never certifies circular physics.
"""
from core.geometry.applicability import require_circular_tube, tube_cross_section_family
from core.geometry.tube import TubeCrossSectionFamily
from core.heat_transfer.internal_flow import heat_transfer_coefficient_internal_diagnostics
from core.pressure_drop.straight_sections import darcy_friction_factor, darcy_friction_factor_method


def internal_heat_transfer_diagnostics(*, tube, m_dot, flow_area, props,
                                     T_bulk=None, T_wall=None, L_heated=None,
                                     assumptions=None):
    from core.geometry.rectangular_finned_tube import RectangularFinnedEllipticalTube
    if isinstance(tube,RectangularFinnedEllipticalTube):
        tube=tube.core_tube
    if tube_cross_section_family(tube) is TubeCrossSectionFamily.ELLIPTICAL:
        if getattr(assumptions, 'regime', None) == 'turbulent':
            from core.heat_transfer.elliptical_turbulent import elliptical_turbulent_heat_transfer
            return elliptical_turbulent_heat_transfer(
                tube,m_dot,props,assumptions=assumptions,flow_area=flow_area,L_heated=L_heated)
        from core.heat_transfer.elliptical_internal import elliptical_h1_heat_transfer
        return elliptical_h1_heat_transfer(tube,m_dot,props,assumptions=assumptions,flow_area=flow_area)
    require_circular_tube(tube, "Internal circular HTC")
    return heat_transfer_coefficient_internal_diagnostics(
        m_dot=m_dot, tube_inner_diameter=tube.hydraulic_diameter,
        flow_area=flow_area, props=props, T_bulk=T_bulk, T_wall=T_wall,
        L_heated=L_heated,
    )


def internal_heat_transfer_coefficient(**kwargs):
    result = internal_heat_transfer_diagnostics(**kwargs)
    return result.v, result.Re, result.Pr, result.alfa_corrected, result.warnings


def internal_friction(*, tube, reynolds, relative_roughness=None, assumptions=None):
    from core.geometry.rectangular_finned_tube import RectangularFinnedEllipticalTube
    if isinstance(tube,RectangularFinnedEllipticalTube):
        tube=tube.core_tube
    if tube_cross_section_family(tube) is TubeCrossSectionFamily.ELLIPTICAL:
        from core.geometry.applicability import TubeGeometryApplicabilityError
        if relative_roughness not in (None,0.):
            raise TubeGeometryApplicabilityError('Elliptical friction is smooth-only.')
        if getattr(assumptions, 'regime', None) == 'turbulent':
            from core.heat_transfer.elliptical_turbulent import elliptical_turbulent_friction
            result=elliptical_turbulent_friction(tube,reynolds,assumptions=assumptions)
            return result.f_darcy,result.metadata.model_id
        from core.heat_transfer.elliptical_internal import elliptical_laminar_friction
        result=elliptical_laminar_friction(tube,reynolds,assumptions=assumptions)
        return result.f_darcy,result.metadata.model_id
    require_circular_tube(tube, "Internal circular friction")
    return (darcy_friction_factor(reynolds, relative_roughness),
            darcy_friction_factor_method(reynolds, relative_roughness))


def straight_friction_evaluator(tube, *, assumptions=None):
    """Bind geometry once; hydraulic points evaluate the selected law first."""
    if tube_cross_section_family(tube) is TubeCrossSectionFamily.ELLIPTICAL and assumptions is not None:
        return lambda reynolds, roughness: internal_friction(
            tube=tube,reynolds=reynolds,relative_roughness=roughness,assumptions=assumptions)
    require_circular_tube(tube, "Automatic tube-bundle friction and entrance/exit losses")
    return lambda reynolds, roughness: internal_friction(
        tube=tube, reynolds=reynolds, relative_roughness=roughness)
