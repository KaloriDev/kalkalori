# SPDX-License-Identifier: GPL-3.0-only
"""Separate experimental compact-HX friction and pressure-estimate contract."""
from dataclasses import dataclass
from typing import Protocol

from core.heat_transfer.continuous_fin_source import (
    ContinuousFinRequest, ContinuousFinSourceCard, WANG2000_SOURCE,
    kernel_arguments, positive, source_warnings, wang2000_f,
)


@dataclass(frozen=True)
class ContinuousFinDPResult:
    coefficient: float
    core_friction_estimate: float
    metadata: ContinuousFinSourceCard = WANG2000_SOURCE
    coefficient_definition: str = "dp_core_reference=f*(Ao/Ac)*Gc^2/(2*rho); compact-HX, not Darcy"
    warnings: tuple = ()


class ContinuousFinDPProvider(Protocol):
    def evaluate(self, request: ContinuousFinRequest) -> ContinuousFinDPResult: ...


@dataclass(frozen=True)
class ExperimentalWang2000DP:
    def evaluate(self, request):
        f = positive(wang2000_f(**kernel_arguments(request)), "compact-HX f")
        flow = request.flow
        dp = f*(flow.source_face_cylinder_area/flow.minimum_free_flow_area)*request.mass_flux**2/(2*request.props.rho)
        return ContinuousFinDPResult(f, positive(dp, "core friction estimate"), warnings=source_warnings())


def wang_density_change_estimate(*, mass_flux, sigma, density_in, density_out):
    """Signed source-form density term, separate from the core friction estimate.

    Not a separately validated irreversible loss or a generic momentum term.
    Entry/exit terms are unavailable. Never apply a positive drag correction to
    this signed contribution and never append another acceleration term.
    """
    for name, value in (("mass_flux", mass_flux), ("density_in", density_in), ("density_out", density_out), ("sigma", sigma)):
        positive(value, name)
    if sigma > 1:
        raise ValueError("sigma must not exceed one.")
    return mass_flux**2/(2*density_in)*(1+sigma**2)*(density_in/density_out-1)
