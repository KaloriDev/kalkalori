# SPDX-License-Identifier: GPL-3.0-only
"""Separate-package synthetic transport fixture; no engineering correlation."""
from dataclasses import dataclass
import math

from core.enhancements import (
    EnhancementDiagnostic, EnhancementReferenceState, EnhancementResult,
)
from core.common.warnings import make_warning


@dataclass(frozen=True)
class PrivateConfig:
    revision: str = "synthetic-1"


@dataclass(frozen=True)
class PrivateResult(EnhancementResult):
    private_revision: str = "synthetic-1"


class ExternalProvider:
    provider_id = "external_contract_fixture"
    thermal_property_reference = "bulk"
    hydraulic_property_reference = "bulk"
    requires_wall_state = False

    def __init__(self):
        self.config = PrivateConfig()
        self.session = object()
        self.requests = []
        self.remaining = []

    def __deepcopy__(self, memo):
        raise AssertionError("Provider session must not be deep-copied")

    def __reduce_ex__(self, protocol):
        raise AssertionError("Provider session must not be pickled")

    def __eq__(self, other):
        raise AssertionError("Provider identity must not require equality")

    def __hash__(self):
        raise AssertionError("Provider session must not require hashing")

    def evaluate(self, geometry, state):
        assert geometry is self.config
        if self.requires_wall_state:
            assert state.wall is not None and math.isfinite(state.wall.temperature)
        self.requests.append(state)
        if state.operation_context is not None:
            state.operation_context.check_deadline()
            self.remaining.append(state.operation_context.remaining_time())
        area = state.base_flow_area_per_tube
        if area is None:
            area = math.pi * state.tube_inner_diameter**2 / 4
        velocity = state.mass_flow_per_tube / (state.bulk.rho * area)
        reference = EnhancementReferenceState(
            area, velocity, state.bulk.rho * velocity * state.tube_inner_diameter / state.bulk.mu,
            state.bulk.cp * state.bulk.mu / state.bulk.k,
            state.tube_inner_diameter, state.tube_inner_diameter, state.tube_inner_diameter,
        )
        # Synthetic wall sensitivity, not a physical enhancement equation.
        correction = (1.0 if state.wall is None else
                      1.0 + .001 * (state.wall.temperature - state.bulk.temperature))
        return PrivateResult(
            provider_id=self.provider_id, correlation_id="synthetic_transport",
            source_references=("private:synthetic_fixture",), source_access_basis="private",
            reference=reference, alpha_inside=1000.0 * correction,
            f_darcy=.08, friction_factor_native=.02, friction_basis="fanning",
            regime="synthetic", wall_correction=correction,
            warnings=(make_warning(code="external_fixture_notice", message="Synthetic fixture",
                                   source=self.provider_id),),
            diagnostics=(EnhancementDiagnostic("revision", geometry.revision),),
            private_revision=geometry.revision,
        )
