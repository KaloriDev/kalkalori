# SPDX-License-Identifier: GPL-3.0-only
"""Explicit compatible component set; no serrated AUTO registration."""
from dataclasses import dataclass, field, replace

from core.geometry.serrated_finned_tube import SerratedFinPhysicsNotSupportedError
from core.geometry.serrated_flow_area import Holfeld2016SerratedFlowAreaProvider
from core.geometry.serrated_source import (
    HolfeldSerratedSourceGeometry, SerratedTubeBankRequest, SerratedSourceDomainError,
    positive, require_source_tube,
)
from core.heat_transfer.serrated_fin_efficiency import HashizumeSerratedFinEfficiencyProvider
from core.heat_transfer.serrated_holfeld import Holfeld2016SerratedStaggeredHeatTransferProvider
from core.pressure_drop.serrated_holfeld import Holfeld2016SerratedStaggeredPressureDropProvider


@dataclass(frozen=True)
class Holfeld2016SerratedStaggeredModel:
    """Explicit source-bounded preset, requiring a physical strip width [m].

    Native source surface is distinct from the Phase A radial-sector area.
    Correlations are calibrated with dry air; constant/custom transport inputs
    explicitly assert that dry-air contract. Known wet/phase-change providers
    are rejected even when the caller disables phase change.
    """

    source_segment_width: float
    flow_area_provider: Holfeld2016SerratedFlowAreaProvider = field(default_factory=Holfeld2016SerratedFlowAreaProvider)
    fin_efficiency_provider: HashizumeSerratedFinEfficiencyProvider = field(default_factory=HashizumeSerratedFinEfficiencyProvider)
    heat_transfer_provider: Holfeld2016SerratedStaggeredHeatTransferProvider = field(default_factory=Holfeld2016SerratedStaggeredHeatTransferProvider)
    pressure_drop_provider: Holfeld2016SerratedStaggeredPressureDropProvider = field(default_factory=Holfeld2016SerratedStaggeredPressureDropProvider)
    provider_name = "holfeld2016_serrated_staggered"

    def __post_init__(self):
        positive(self.source_segment_width, "source_segment_width [m]")
        for name, expected in (
            ("flow_area_provider", Holfeld2016SerratedFlowAreaProvider),
            ("fin_efficiency_provider", HashizumeSerratedFinEfficiencyProvider),
            ("heat_transfer_provider", Holfeld2016SerratedStaggeredHeatTransferProvider),
            ("pressure_drop_provider", Holfeld2016SerratedStaggeredPressureDropProvider),
        ):
            if type(getattr(self, name)) is not expected:
                raise SerratedFinPhysicsNotSupportedError(
                    f"Explicit complete serrated profile requires compatible {name}; no solid-fin fallback.")

    def validate_bundle(self, bundle):
        require_source_tube(bundle.tube, "GEOMETRY")
        failures = []
        if bundle.layout != "staggered":
            failures.append("complete execution requires STAGGERED; INLINE/SHIFTED unsupported")
        if bundle.n_rows < 4:
            failures.append("Holfeld requires at least four longitudinal rows")
        t = bundle.tube
        if t.fin_contact_resistance not in (None, 0.) or t.fin_contact_efficiency not in (None, 1.):
            failures.append("initial source profile requires ideal HF-weld contact")
        if failures:
            raise SerratedSourceDomainError("PROFILE", tuple(failures))

    def source_geometry(self, bundle):
        self.validate_bundle(bundle)
        return HolfeldSerratedSourceGeometry(bundle, self.source_segment_width)

    def request(self, bundle, m_dot, props):
        self.validate_bundle(bundle)
        return SerratedTubeBankRequest(bundle, m_dot, props, self.source_segment_width,
                                      self.flow_area_provider.evaluate(bundle))

    def evaluate_thermal(self, bundle, m_dot, props):
        """Return physical film, eta, and all component-domain labels.

        Individual low-level providers retain their separate source envelopes;
        a complete profile enforces their intersection unconditionally.
        """
        request = self.request(bundle, m_dot, props)
        ht = self.heat_transfer_provider.evaluate(request)
        statuses = {"FLOW_AREA_DOMAIN": "PASS", "HT_DOMAIN": "PASS"}
        try:
            self.pressure_drop_provider.evaluate(request)
            statuses["DP_DOMAIN"] = "PASS"
            eta = self.fin_efficiency_provider.evaluate(bundle.tube, ht.alpha,
                reynolds_number=ht.reynolds_number, source_segment_width=self.source_segment_width,
                layout=bundle.layout)
        except SerratedSourceDomainError as exc:
            exc.diagnostics.update(statuses)
            raise
        diagnostics = {**ht.diagnostics, **statuses, "ETA_DOMAIN": "PASS",
            "complete_domain_status": "INTERSECTION_PASS", "profile": self.provider_name,
            "eta_fin_theoretical": eta.fin_efficiency_theoretical,
            "eta_fin_corrected": eta.fin_efficiency,
            "fin_efficiency_provider": eta.diagnostics,
            "pressure_drop_provider": self.pressure_drop_provider.provider_name}
        return replace(ht, diagnostics=diagnostics), eta

    def guard_fouling(self, inside=0., outside=0.):
        if inside or outside:
            raise SerratedSourceDomainError("PROFILE", ("fouling correction is unsupported",))

    def guard_property_provider(self, provider, side="outside"):
        if provider is None:
            return  # low-level point properties are an explicit caller assertion
        from core.phase_change.capability import detect_phase_change_capability
        capability = detect_phase_change_capability(provider)
        if capability.capable or capability.provider_kind:
            raise SerratedSourceDomainError("PROFILE", (f"{side}: dry single-phase provider required; wet/condensing/steam execution unsupported",))


def guard_serrated_execution(hx, *, inside_provider=None, outside_provider=None):
    """Reject unsupported modes before generic phase/wet dispatch can run."""
    if not isinstance(hx.airside_model, Holfeld2016SerratedStaggeredModel):
        return
    model = hx.airside_model
    model.validate_bundle(hx.bundle)
    model.guard_fouling(hx.fouling_resistance_inside, hx.fouling_resistance_outside)
    model.guard_property_provider(inside_provider, "inside")
    model.guard_property_provider(outside_provider, "outside")
