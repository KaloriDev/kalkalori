# SPDX-License-Identifier: GPL-3.0-only
"""Explicit research profile, deliberately absent from production/AUTO routing."""
from dataclasses import dataclass, field

from core.heat_transfer.airside_correction import AirsideCorrection, AirsideCorrectionProvider
from core.heat_transfer.continuous_fin_source import (
    ContinuousFinRequest, ContinuousFinSourceError, ContactPolicy,
    ExperimentalWang2000HTC, WANG2000_SOURCE,
)
from core.heat_transfer.continuous_fin_efficiency import SchmidtEquivalentCell
from core.pressure_drop.continuous_fin import ExperimentalWang2000DP


@dataclass(frozen=True)
class ContinuousFinBaseline:
    heat_transfer: object
    pressure_drop: object


@dataclass(frozen=True)
class ContinuousAirsideResult:
    baseline: ContinuousFinBaseline
    correction: AirsideCorrection
    alpha: float
    core_friction_estimate: float
    efficiency: object
    metadata: object = WANG2000_SOURCE

    @property
    def warnings(self):
        return self.baseline.heat_transfer.warnings+self.efficiency.warnings


@dataclass(frozen=True)
class ExperimentalWang2000Model:
    """Research opt-in, not a validated engineering prediction provider.

    ``acknowledge_source_limits=True`` is mandatory. Strict source validation
    still raises. Geometry remains process-neutral; this profile supports only
    a dry air surface. Internal supported steam models remain independently
    selectable. Empirically included contact cannot take an extra penalty.
    """
    acknowledge_source_limits: bool = False
    correction_provider: AirsideCorrectionProvider | None = None
    contact_policy: ContactPolicy = ContactPolicy.EMPIRICAL_INCLUDED
    contact_resistance: float = 0.
    heat_transfer_provider: ExperimentalWang2000HTC = field(default_factory=ExperimentalWang2000HTC)
    pressure_drop_provider: ExperimentalWang2000DP = field(default_factory=ExperimentalWang2000DP)
    fin_efficiency_provider: SchmidtEquivalentCell = field(default_factory=SchmidtEquivalentCell)

    def __post_init__(self):
        if self.acknowledge_source_limits is not True:
            raise ContinuousFinSourceError("Experimental Wang requires acknowledge_source_limits=True; B01-B05 remain open.")
        if self.contact_policy != ContactPolicy.EMPIRICAL_INCLUDED or self.contact_resistance != 0.:
            raise ContinuousFinSourceError("Wang EMPIRICAL_INCLUDED contact prevents additional contact resistance.")
        for name, expected in (("heat_transfer_provider", ExperimentalWang2000HTC),
                ("pressure_drop_provider", ExperimentalWang2000DP), ("fin_efficiency_provider", SchmidtEquivalentCell)):
            if type(getattr(self, name)) is not expected:
                raise ContinuousFinSourceError(f"Explicit Wang profile requires compatible {name}; no undocumented hybrid.")

    def validate_bundle(self, bundle):
        from core.properties.common import FluidTransportProperties
        ContinuousFinRequest(bundle, 1., FluidTransportProperties(1., 2e-5, .026, 1000.))
        fin = bundle.continuous_fin_geometry
        if fin.first_sheet_position > fin.pitch or bundle.tube.length_effective-fin.pack_extent[1] > fin.pitch:
            raise ContinuousFinSourceError("Wang pack has unsupported unfinned axial bypass regions; each sealed end gap must not exceed one fin pitch.")
        self.fin_efficiency_provider.evaluate(bundle, 0.)

    def guard_property_provider(self, provider):
        if provider is None:
            return  # SI point properties assert dry air in this explicit profile.
        from core.phase_change.capability import detect_phase_change_capability
        cap = detect_phase_change_capability(provider)
        if cap.capable or cap.provider_kind:
            raise ContinuousFinSourceError("Continuous-fin dry-air profile rejects external wet/condensing property providers.")
        from core.properties.tabulated_liquid import TabulatedLiquidProvider
        if isinstance(provider, TabulatedLiquidProvider):
            raise ContinuousFinSourceError("Wang source fluid is dry air, not a tabulated liquid.")
        # Named fluid providers retain their physical identity; custom constants
        # explicitly assert dry air. Do not reinterpret a known liquid as air.
        fluid = getattr(provider, "fluid", None)
        if fluid is not None and str(getattr(fluid, "value", fluid)).lower() not in ("air", "dry_air"):
            raise ContinuousFinSourceError("Wang source fluid is dry air; selected outside fluid is unsupported.")

    def evaluate(self, bundle, m_dot, props, *, strict_source_domain=False):
        request = ContinuousFinRequest(bundle, m_dot, props, strict_source_domain=strict_source_domain)
        ht = self.heat_transfer_provider.evaluate(request)
        dp = self.pressure_drop_provider.evaluate(request)
        baseline = ContinuousFinBaseline(ht, dp)
        correction = (AirsideCorrection() if self.correction_provider is None else
            self.correction_provider.correct(baseline, bundle=bundle, props=props))
        if not isinstance(correction, AirsideCorrection):
            raise TypeError("correction_provider must return AirsideCorrection.")
        h, friction = correction.apply(ht.alpha, dp.core_friction_estimate)
        efficiency = self.fin_efficiency_provider.evaluate(bundle, h)
        return ContinuousAirsideResult(baseline, correction, h, friction, efficiency)


def require_continuous_profile(bundle, model):
    if not isinstance(model, ExperimentalWang2000Model):
        raise ContinuousFinSourceError("Continuous plate fins require an explicit experimental continuous-fin profile; AUTO, bare and individual-fin fallback are unavailable.")
    model.validate_bundle(bundle)


def guard_continuous_execution(hx, *, outside_provider=None):
    if hx.bundle.continuous_fin_geometry is None:
        return
    require_continuous_profile(hx.bundle, hx.airside_model)
    if hx.strict_source_domain:
        raise ContinuousFinSourceError("Strict source validation unavailable: Wang Reynolds/source/validation gates remain open.")
    hx.airside_model.guard_property_provider(outside_provider)
