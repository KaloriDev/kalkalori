# SPDX-License-Identifier: GPL-3.0-only
"""Select internal constitutive physics before interpreting reference lengths.

Existing scalar circular APIs retain their meanings. Geometry-based callers
use this boundary so a hydraulic diameter never certifies circular physics.
"""
from core.geometry.applicability import require_circular_tube
from core.heat_transfer.internal_flow import heat_transfer_coefficient_internal_diagnostics
from core.pressure_drop.straight_sections import darcy_friction_factor, darcy_friction_factor_method


def internal_heat_transfer_diagnostics(*, tube, m_dot, flow_area, props,
                                     T_bulk=None, T_wall=None, L_heated=None,
                                     assumptions=None):
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
    require_circular_tube(tube, "Internal circular friction")
    return (darcy_friction_factor(reynolds, relative_roughness),
            darcy_friction_factor_method(reynolds, relative_roughness))


def straight_friction_evaluator(tube):
    """Bind geometry once; hydraulic points evaluate the selected law first."""
    require_circular_tube(tube, "Automatic tube-bundle friction and entrance/exit losses")
    return lambda reynolds, roughness: internal_friction(
        tube=tube, reynolds=reynolds, relative_roughness=roughness)
