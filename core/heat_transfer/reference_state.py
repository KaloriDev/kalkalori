# SPDX-License-Identifier: GPL-3.0-only
"""Reporting snapshots of actual correlation states, never a second solve."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FlowReferenceState:
    temperature: float | None = None
    pressure: float | None = None
    reynolds: float | None = None
    diameter: float | None = None
    velocity: float | None = None
    mass_flux: float | None = None
    flow_area: float | None = None
    area_basis: str = ''
    reference_basis: str = ''
    face_velocity: float | None = None
    modified_reynolds: float | None = None


def _finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def outside_reference_state(*, bundle, temperature, pressure, reynolds, velocity,
                            face_velocity, external=None, airside=None,
                            reference_area=None, reference_mass_flux=None, reference_diameter=None,
                            finned=None):
    """Respect a supplied provider definition; missing fields stay unknown."""
    if external is not None:
        ref = external.reference
        return FlowReferenceState(temperature,pressure,ref.reynolds_number,
            ref.characteristic_length,ref.velocity,ref.mass_flux,ref.flow_area,
            ref.flow_area_basis,external.metadata.reference_basis,face_velocity)
    flow = getattr(airside, 'flow', None)
    if flow is not None:
        return FlowReferenceState(temperature,pressure,flow.reynolds_number,
            flow.reference_diameter,flow.reference_velocity,None,flow.average_passage_area,
            flow.passage_area_basis,flow.diameter_basis,flow.face_velocity)
    if finned is not None:
        return FlowReferenceState(temperature,pressure,finned.reynolds_number,
            finned.reference_diameter_value,finned.reference_velocity,finned.reference_mass_flux,
            finned.minimum_free_flow_area,'minimum_free_flow_area',
            finned.metadata.reynolds_basis,finned.face_velocity)
    # Circular built-ins use the real minimum passage and outside diameter.
    # Continuous/other providers must supply their own snapshots instead.
    from core.geometry.tube import TubeCrossSectionFamily
    if bundle.continuous_fin_geometry is None and bundle.tube.cross_section_family is TubeCrossSectionFamily.CIRCULAR:
        from core.geometry.reporting import minimum_free_flow_area_status
        area = minimum_free_flow_area_status(bundle).area if reference_area is None else reference_area
        core = getattr(bundle.tube, 'core_tube', bundle.tube)
        basis = ('frontal_area' if area == bundle.frontal_flow_area else 'minimum_free_flow_area')
        return FlowReferenceState(temperature,pressure,_finite(reynolds),
            core.D_o if reference_diameter is None else reference_diameter,
            _finite(velocity),reference_mass_flux,area,basis,
            'provider diameter and reference velocity; '+basis,_finite(face_velocity))
    return FlowReferenceState(temperature=temperature,pressure=pressure,
        reynolds=_finite(reynolds),velocity=_finite(velocity),face_velocity=_finite(face_velocity),
        reference_basis='provider-specific; unreported dimensions remain unavailable')


def result_reference_states(result, bundle):
    """Four separate references for normal dry Rating/Simulation results.

    Thermal fields come from the converged correlation evaluation; hydraulic
    fields come from the final hydraulic midpoint, including its property
    midpoint method. Missing/phase-zone equivalent states remain unavailable.
    """
    thermal = getattr(result, 'thermal_state', None)
    diag = getattr(thermal, 'diagnostics', None)
    states = {side+'_thermal':getattr(diag,side+'_reference_state',None)
              for side in ('inside','outside')}
    final = getattr(result, 'final_result', result)
    inside = getattr(final, 'tube_bundle_hydraulic', None)
    if inside is not None:
        point = inside.midpoint
        friction = getattr(point,'friction_diagnostics',None)
        states['inside_hydraulic'] = FlowReferenceState(point.temperature,point.pressure,
            point.reynolds,bundle.tube.hydraulic_diameter,point.velocity,point.mass_flux,
            bundle.internal_flow_area_per_pass,'actual_inside_flow_area_per_pass',
            'actual Dh; mean unblocked tube velocity; '+inside.midpoint_method,
            modified_reynolds=getattr(friction,'modified_reynolds',None))
    outside = getattr(final, 'outside_tube_bank_hydraulic', None)
    if outside is not None:
        point = outside.midpoint
        states['outside_hydraulic'] = outside_reference_state(bundle=bundle,
            temperature=point.temperature,pressure=point.pressure,
            reynolds=getattr(point,'reynolds_number',getattr(point,'reynolds',None)),
            velocity=getattr(point,'reference_velocity',None),face_velocity=getattr(point,'face_velocity',None),
            external=getattr(outside,'provider_result',None),airside=getattr(point,'airside_result',None),
            reference_area=getattr(point,'reference_area',None),
            reference_mass_flux=getattr(point,'reference_mass_flux',None),
            reference_diameter=getattr(outside,'reference_diameter',None))
    return states
