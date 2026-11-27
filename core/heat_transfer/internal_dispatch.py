# SPDX-License-Identifier: GPL-3.0-only
"""Select internal constitutive physics before interpreting reference lengths.

Existing scalar circular APIs retain their meanings. Geometry-based callers
use this boundary so a hydraulic diameter never certifies circular physics.
"""
from core.geometry.applicability import require_circular_tube, tube_cross_section_family
from core.geometry.tube import TubeCrossSectionFamily
from core.heat_transfer.internal_flow import heat_transfer_coefficient_internal_diagnostics
from core.pressure_drop.straight_sections import darcy_friction_factor, darcy_friction_factor_method


class _StraightFrictionEvaluation(tuple):
    """Keep the historical two-tuple API while retaining source diagnostics."""
    def __new__(cls, diagnostics):
        value = super().__new__(cls, (diagnostics.f_darcy, diagnostics.metadata.model_id))
        value.diagnostics = diagnostics
        return value

    @property
    def f_darcy(self): return self[0]

    @property
    def metadata(self): return self.diagnostics.metadata


def _require_ellipse(tube):
    from core.geometry.elliptical_tube import EllipticalTube
    from core.geometry.applicability import TubeGeometryApplicabilityError
    if not isinstance(tube, EllipticalTube):
        raise TubeGeometryApplicabilityError('Elliptical dispatch requires EllipticalTube geometry.')


def _ellipse_assumptions(reynolds, assumptions):
    """Conservative regime dispatch; no transition law or circular fallback.

    Advanced declarations can verify source assumptions but cannot override
    the physical regime in geometry-based Rating/Simulation dispatch.
    """
    import math
    from core.geometry.applicability import TubeGeometryApplicabilityError
    from core.heat_transfer.internal_model_contract import EllipticalFlowAssumptions
    if not math.isfinite(reynolds) or reynolds <= 0:
        raise ValueError('Elliptical Reynolds number must be positive and finite.')
    if 2300 <= reynolds < 4000:
        raise TubeGeometryApplicabilityError('Elliptical transition flow (2300<=Re_Dh<4000) has no implemented provider.')
    regime = 'laminar' if reynolds < 2300 else 'turbulent'
    if assumptions is not None:
        if not isinstance(assumptions, EllipticalFlowAssumptions):
            raise TypeError('assumptions must be EllipticalFlowAssumptions or None.')
        if assumptions.regime != regime:
            raise TubeGeometryApplicabilityError(f'Elliptical Re_Dh={reynolds:.12g} selects {regime}; incompatible declared regime.')
        return assumptions
    return EllipticalFlowAssumptions(regime, 'fully_developed', 'H1' if regime == 'laminar' else 'T')


def internal_heat_transfer_diagnostics(*, tube, m_dot, flow_area, props,
                                     T_bulk=None, T_wall=None, L_heated=None,
                                     assumptions=None, strict_source_domain=False, roughness_policy=None):
    from core.geometry.rectangular_finned_tube import RectangularFinnedEllipticalTube
    if isinstance(tube,RectangularFinnedEllipticalTube):
        tube=tube.core_tube
    if tube_cross_section_family(tube) is TubeCrossSectionFamily.ELLIPTICAL:
        _require_ellipse(tube)
        assumptions = _ellipse_assumptions(m_dot*tube.hydraulic_diameter/(flow_area*props.mu), assumptions)
        if getattr(assumptions, 'regime', None) == 'turbulent':
            from core.heat_transfer.elliptical_turbulent import elliptical_turbulent_heat_transfer
            return elliptical_turbulent_heat_transfer(
                tube,m_dot,props,assumptions=assumptions,flow_area=flow_area,L_heated=L_heated,
                strict_source_domain=strict_source_domain, roughness_policy=roughness_policy)
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


def _elliptical_friction(tube, reynolds, relative_roughness, assumptions, roughness_policy):
    from core.geometry.applicability import TubeGeometryApplicabilityError
    from core.heat_transfer.elliptical_turbulent import _roughness_policy
    import math
    assumptions = _ellipse_assumptions(reynolds, assumptions)
    policy = _roughness_policy(roughness_policy)
    if relative_roughness is not None:
        if (isinstance(relative_roughness, bool) or not math.isfinite(relative_roughness)
                or relative_roughness < 0):
            raise ValueError('relative_roughness must be finite and nonnegative.')
        if relative_roughness > 0 and not policy.allow_hydraulic_roughness:
            raise TubeGeometryApplicabilityError('Elliptical friction is smooth-only without hydraulic roughness authorization.')
        physical = (tube.roughness_inner or 0.) / tube.hydraulic_diameter
        if not math.isclose(relative_roughness, physical, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError('relative_roughness must match actual tube roughness_inner / Dh.')
    if assumptions.regime == 'turbulent':
        from core.heat_transfer.elliptical_turbulent import elliptical_turbulent_friction
        return elliptical_turbulent_friction(tube,reynolds,assumptions=assumptions,roughness_policy=policy)
    from core.heat_transfer.elliptical_internal import elliptical_laminar_friction
    return elliptical_laminar_friction(tube,reynolds,assumptions=assumptions)


def internal_friction(*, tube, reynolds, relative_roughness=None, assumptions=None, roughness_policy=None):
    from core.geometry.rectangular_finned_tube import RectangularFinnedEllipticalTube
    if isinstance(tube,RectangularFinnedEllipticalTube):
        tube=tube.core_tube
    if tube_cross_section_family(tube) is TubeCrossSectionFamily.ELLIPTICAL:
        _require_ellipse(tube)
        result = _elliptical_friction(tube,reynolds,relative_roughness,assumptions,roughness_policy)
        return result.f_darcy,result.metadata.model_id
    require_circular_tube(tube, "Internal circular friction")
    return (darcy_friction_factor(reynolds, relative_roughness),
            darcy_friction_factor_method(reynolds, relative_roughness))


def straight_friction_evaluator(tube, *, assumptions=None, roughness_policy=None):
    """Bind geometry once; hydraulic points evaluate the selected law first."""
    if tube_cross_section_family(tube) is TubeCrossSectionFamily.ELLIPTICAL:
        _require_ellipse(getattr(tube, 'core_tube', tube))
        return lambda reynolds, roughness: _StraightFrictionEvaluation(_elliptical_friction(
            getattr(tube, 'core_tube', tube),reynolds,roughness,assumptions,roughness_policy))
    require_circular_tube(tube, "Automatic tube-bundle friction and entrance/exit losses")
    return lambda reynolds, roughness: internal_friction(
        tube=tube, reynolds=reynolds, relative_roughness=roughness)
