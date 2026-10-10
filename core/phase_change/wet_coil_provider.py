# SPDX-License-Identifier: GPL-3.0-only
"""Direct-object GLOBAL wet model contract; no registry or model discovery."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from core.heat_transfer.wet_coil_solver import WetCoilSolverOptions, _solve_budget
from core.phase_change.types import PhaseChangeMode

if TYPE_CHECKING:
    from core.models.bare_tube import BareTubeHeatExchanger
    from core.models.heat_balance import BalanceSideSpec
    from core.models.rating import HXRatingResult
    from core.models.simulation import HXSideInput, HXSimulationResult
    from core.phase_change.integration import PhaseChangeSettings


class WetCoilProviderUnsupportedError(ValueError):
    """Selected provider cannot perform the requested operation or case."""


class WetCoilOperationContext(Protocol):
    """The existing whole-operation budget, shared by all nested solves.

    Providers must call ``check`` during work and pass this same context to
    their nested solves. Times are monotonic seconds; deadline None is unlimited.
    """
    options: WetCoilSolverOptions
    started: float
    deadline: float | None

    def check(self, **state: object) -> None: ...


class WetCoilModelProvider(Protocol):
    """Global model using the existing HX inputs/results and SI units.

    Unsupported methods may raise WetCoilProviderUnsupportedError; the dispatcher
    checks their support flag before calling. Applicability is evaluated only
    for enabled, supported operations.
    Simulation retains installed geometry/hydraulics and surface-margin
    semantics. Rating solves thermal area with installed hydraulic geometry.
    """
    model_id: str
    model_name: str
    source: str
    applicability: str
    supports_simulation: bool
    supports_rating: bool

    def is_applicable(
        self, hx: BareTubeHeatExchanger,
        inside: HXSideInput | BalanceSideSpec,
        outside: HXSideInput | BalanceSideSpec,
        *, settings: PhaseChangeSettings, **options: object,
    ) -> bool: ...

    def simulate(
        self, hx: BareTubeHeatExchanger, inside: HXSideInput, outside: HXSideInput,
        *, settings: PhaseChangeSettings, wet_solver_options: WetCoilSolverOptions,
        context: WetCoilOperationContext, **options: object,
    ) -> HXSimulationResult: ...

    def rate(
        self, hx: BareTubeHeatExchanger, inside: BalanceSideSpec, outside: BalanceSideSpec,
        *, settings: PhaseChangeSettings, wet_solver_options: WetCoilSolverOptions,
        context: WetCoilOperationContext, **options: object,
    ) -> HXRatingResult: ...


@dataclass(frozen=True)
class ElmahdyMitalasWetCoilProvider:
    """Thin adapter; all equations and inverse solving remain in the engine."""
    model_id: str = "elmahdy_mitalas_energyplus_v25_2_adapted"
    model_name: str = "Elmahdy-Mitalas (source-profile adaptation)"
    source: str = "Elmahdy-Mitalas 1977; EnergyPlus v25.2; docs/elmahdy_mitalas.md"
    applicability: str = (
        "Outside wet gas cooled by inside liquid; counterflow mean-property/"
        "secant approximation; Lewis number 1; no tube-side enhancement."
    )
    supports_simulation: bool = True
    supports_rating: bool = True

    def is_applicable(self, hx, inside, outside, *, settings, **options):
        from core.phase_change.wet_coil_integration import _eligible
        flow = options.get("flow_arrangement") or hx.bundle.flow_arrangement_resolved
        return (_eligible(inside, outside) and hx.tube_side_enhancement is None
                and flow != "cocurrentflow" and settings.lewis_number == 1.0)

    def simulate(self, hx, inside, outside, *, settings, wet_solver_options, context, **options):
        from core.phase_change.wet_coil_integration import _run_elmahdy
        # This engine has no dry baseline; these controls belong only to
        # providers that run the historical sensible Simulation first.
        for name in ("max_iter", "temperature_tolerance_K", "relative_duty_tolerance",
                     "relaxation_factor", "relative_alfa_tolerance"):
            options.pop(name, None)
        return _run_elmahdy(hx, inside, outside, mode="simulation", settings=settings,
                            wet_solver_options=wet_solver_options, _budget=context, **options)

    def rate(self, hx, inside, outside, *, settings, wet_solver_options, context, **options):
        from core.phase_change.wet_coil_integration import _run_elmahdy
        return _run_elmahdy(hx, inside, outside, mode="rating", settings=settings,
                            wet_solver_options=wet_solver_options, _budget=context, **options)


@dataclass(frozen=True)
class LegacyBulkMeanWetCoilProvider:
    """Historical KalKalori outside-condensation Simulation, explicitly selected."""

    model_id: str = "legacy_outside_condensation_0d_bulk_mean"
    model_name: str = "KalKalori legacy bulk-mean outside condensation"
    source: str = "KalKalori c80e779 (v0.8.2); docs/wet_coil_providers.md"
    applicability: str = (
        "Outside H2O in a carrier gas; sensible inside fluid; BareTube or "
        "CircularFinnedTube; installed bundle flow; no tube-side enhancement; "
        "non-negative thermal surface_margin; no frost or simultaneous active phase-change sides."
    )
    supports_simulation: bool = True
    supports_rating: bool = False

    def is_applicable(self, hx, inside, outside, *, settings, **options):
        from core.geometry.tube import TubeSurfaceType
        from core.phase_change.capability import detect_phase_change_capability

        cap = detect_phase_change_capability(outside.provider)
        flow = options.get("flow_arrangement") or hx.bundle.flow_arrangement_resolved
        return (
            cap.capable and cap.component == "H2O" and cap.provider_kind == "gas_mixture"
            and hx.bundle.tube.surface_type in (TubeSurfaceType.PLAIN, TubeSurfaceType.CIRCULAR_FINNED)
            and hx.tube_side_enhancement is None
            and flow == hx.bundle.flow_arrangement_resolved
        )

    def simulate(self, hx, inside, outside, *, settings, wet_solver_options, context, **options):
        from core.phase_change.legacy_wet_coil_integration import run_legacy_simulation

        return run_legacy_simulation(hx, inside, outside, settings=settings,
                                     context=context, **options)

    def rate(self, *args, **kwargs):
        raise WetCoilProviderUnsupportedError("Legacy wet provider does not support rating")


def dispatch_wet_coil(hx, inside, outside, *, mode, settings,
                      wet_coil_provider=None, force_candidate=False, **options):
    """Select once; unsupported selections and provider failures never fall back."""
    if outside.phase_change_mode is PhaseChangeMode.DISABLED:
        return None
    if mode not in ("simulation", "rating"):
        raise ValueError(f"Unknown wet-coil operation: {mode}")
    provider = wet_coil_provider
    if provider is None:
        # Preserve historical default routing, including the later forced
        # candidate path. Model-specific validation remains in the adapter.
        from core.phase_change.wet_coil_integration import _eligible
        flow = options.get("flow_arrangement") or hx.bundle.flow_arrangement_resolved
        if not force_candidate and (not _eligible(inside, outside)
                or hx.tube_side_enhancement is not None or flow == "cocurrentflow"):
            return None
        provider = ElmahdyMitalasWetCoilProvider()
    else:
        if not getattr(provider, f"supports_{mode}", False):
            raise WetCoilProviderUnsupportedError(f"Selected wet provider does not support {mode}")
        if not provider.is_applicable(hx, inside, outside, settings=settings, **options):
            raise WetCoilProviderUnsupportedError(
                f"Wet provider {provider.model_id} is not applicable: {provider.applicability}")
    budget = _solve_budget(options.pop("wet_solver_options", None))
    budget.used = True
    budget.check()
    method = provider.simulate if mode == "simulation" else provider.rate
    result = method(hx, inside, outside, settings=settings,
                    wet_solver_options=budget.options, context=budget, **options)
    budget.check()
    from core.models.rating import HXRatingResult
    from core.models.simulation import HXSimulationResult
    expected = HXSimulationResult if mode == "simulation" else HXRatingResult
    if not isinstance(result, expected):
        raise TypeError(f"Wet provider {provider.model_id} must return {expected.__name__}")
    return _attach_provider_diagnostics(result, provider)


def _attach_provider_diagnostics(result, provider):
    """Also cover the adapter's cached Rating-to-Simulation report."""
    diagnostics = dict(result.wet_coil_diagnostics or {})
    # Applied exchanger parameters also accompany Legacy/custom provider
    # reports. Preserve Elmahdy's process-area resistance decomposition.
    snapshot = getattr(result, "final_result", None)
    if snapshot is not None:
        for name in ("fouling_resistance_inside", "fouling_resistance_outside"):
            diagnostics.setdefault(name, getattr(snapshot, name))
        for name in ("resistance_fouling_inside", "resistance_fouling_outside"):
            diagnostics.setdefault(name, getattr(snapshot, name))
    diagnostics["global_wet_model"] = provider.model_id
    diagnostics["provider"] = dict(
        model_id=provider.model_id, model_name=provider.model_name,
        source=provider.source, applicability=provider.applicability,
        supports_simulation=provider.supports_simulation, supports_rating=provider.supports_rating,
    )
    phase = result.outside_phase_change
    if phase is not None:
        phase = replace(phase, method=provider.model_id, wet_coil_diagnostics=diagnostics)
    updates = dict(wet_coil_diagnostics=diagnostics, outside_phase_change=phase)
    simulation = getattr(result, "simulation", None)
    if simulation is not None:
        updates["simulation"] = _attach_provider_diagnostics(simulation, provider)
    return replace(result, **updates)
