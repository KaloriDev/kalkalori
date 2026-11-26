# SPDX-License-Identifier: GPL-3.0-only
"""Explicit assumptions and reference metadata for bounded internal models."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EllipticalRoughnessPolicy:
    """Separate opt-ins for unvalidated rough hydraulics and smooth-wall HTC.

    Neither permission certifies a rough elliptical correlation. Laminar and
    transition guards are unaffected. Absolute roughness is equivalent sand
    roughness, not an arithmetic surface-finish measurement.
    """
    allow_hydraulic_roughness: bool = False
    allow_smooth_htc_on_rough_wall: bool = False

    def __post_init__(self):
        if any(not isinstance(value, bool) for value in (
                self.allow_hydraulic_roughness, self.allow_smooth_htc_on_rough_wall)):
            raise TypeError('Elliptical roughness permissions must be boolean.')


@dataclass(frozen=True)
class EllipticalFlowAssumptions:
    """Advanced direct-model verification; normal dispatch supplies these.

    Supply development='fully_developed'. The exact laminar HTC requires
    regime='laminar', thermal_boundary_condition='H1'. Yang 2018 requires
    regime='turbulent', thermal_boundary_condition='T' and enforces its own
    shape/source policy. Geometry-based dispatch also checks physical regime.
    """
    regime: str
    development: str
    thermal_boundary_condition: str | None = None
    constant_properties: bool = True
    newtonian: bool = True
    no_slip: bool = True
    axial_conduction: bool = False
    viscous_dissipation: bool = False


@dataclass(frozen=True)
class InternalModelMetadata:
    model_id: str
    provenance_id: str
    source_references: tuple[str, ...]
    source_access_basis: str
    geometry_family: str
    regime: str
    development: str
    thermal_boundary_condition: str | None
    reference_dimension: str
    reference_velocity: str
    area_basis: str
    applicability_assumptions: tuple[str, ...]
    approximation: bool = False
