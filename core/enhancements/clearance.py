# KalKalori - GNU GPL v3 only
"""Source-neutral composition of tape clearance models; no built-in fit or AUTO.

Only heat transfer and distributed friction are composed. No local-loss,
acceleration, flow-blockage or extra wall-correction formula lives here.
See docs/twisted_tape_clearance.md for the provider contract and source audit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Protocol, runtime_checkable

from core.common.warnings import ModelWarning
from .base import (
    EnhancementDiagnostic, EnhancementInput, EnhancementResult,
    EnhancementUnsupportedError, TwistedTapeGeometry, _positive, _validate_result,
)


class ClearanceModelMode(str, Enum):
    CORRECTION = "correction"
    ABSOLUTE = "absolute"


@dataclass(frozen=True)
class ClearanceCorrection:
    """Documented relative factors on the complete base Nu/alpha and Darcy f.

    The base's references and wall correction are retained. Factors must not
    reapply a correction already owned by the base. If a source needs different
    reference geometry or replacement wall physics, use ABSOLUTE instead.
    Compatibility must name exact base correlation IDs, not a wildcard.
    """
    correlation_id: str
    source_references: tuple[str, ...]
    source_access_basis: str
    compatible_base_correlation_ids: tuple[str, ...]
    heat_transfer_factor: float
    friction_factor_factor: float
    regime: str
    applicability: str = "within_range"
    warnings: tuple[ModelWarning, ...] = ()

    def __post_init__(self):
        _positive(heat_transfer_factor=self.heat_transfer_factor,
                  friction_factor_factor=self.friction_factor_factor)
        if not self.correlation_id or not self.regime:
            raise ValueError("Clearance correlation and regime identifiers are required.")
        for name in ("source_references", "compatible_base_correlation_ids"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not value or any(not isinstance(v, str) or not v or v == "*" for v in value):
                raise ValueError(f"{name} must contain explicit identifiers.")
        if self.source_access_basis not in ("open", "private", "external"):
            raise ValueError("Unknown clearance source access basis.")
        if not isinstance(self.warnings, tuple) or any(not isinstance(w, ModelWarning) for w in self.warnings):
            raise TypeError("Clearance warnings must be a tuple of ModelWarning.")
        if self.applicability not in ("within_range", "extrapolated"):
            raise EnhancementUnsupportedError("Unsupported clearance applicability.")
        if self.applicability == "extrapolated" and not self.warnings:
            raise ValueError("Clearance extrapolation requires a warning.")


@dataclass(frozen=True)
class TwistedTapeClearanceResult:
    """Exactly one mode payload, with the source's explicit ratio definition.

    A ratio definition may describe radial/D, diametral/D, width/D, etc.
    It is model-owned metadata, never silently inferred by the solver.
    ABSOLUTE owns a complete EnhancementResult, including wall and reference
    conventions. It has no invented base values or multiplicative factors.
    """
    mode: ClearanceModelMode
    clearance_ratio: float
    clearance_ratio_definition: str
    correction: ClearanceCorrection | None = None
    absolute: EnhancementResult | None = None

    def __post_init__(self):
        if not isinstance(self.mode, ClearanceModelMode):
            raise TypeError("Clearance mode must be ClearanceModelMode.")
        if not math.isfinite(self.clearance_ratio) or self.clearance_ratio < 0:
            raise ValueError("Clearance ratio must be nonnegative and finite.")
        if not isinstance(self.clearance_ratio_definition, str) or not self.clearance_ratio_definition.strip():
            raise ValueError("An explicit source clearance-ratio definition is required.")
        if self.mode is ClearanceModelMode.CORRECTION:
            if not isinstance(self.correction, ClearanceCorrection) or self.absolute is not None:
                raise TypeError("Correction mode requires only a ClearanceCorrection.")
        elif not isinstance(self.absolute, EnhancementResult) or self.correction is not None:
            raise TypeError("Absolute mode requires only an EnhancementResult.")


@runtime_checkable
class TwistedTapeClearanceProvider(Protocol):
    provider_id: str
    mode: ClearanceModelMode

    def evaluate(self, geometry: TwistedTapeGeometry, state: EnhancementInput,
                 base_result: EnhancementResult | None) -> TwistedTapeClearanceResult:
        """Evaluate actual geometry and authoritative state, or raise unsupported.

        CORRECTION providers additionally implement base_geometry_for(geometry,
        state): the source-documented geometry at which to evaluate the base.
        ABSOLUTE receives base_result=None; the base provider is never called.
        """
        ...


def validate_clearance_provider(provider):
    if not isinstance(provider, TwistedTapeClearanceProvider) or not provider.provider_id:
        raise TypeError("Clearance provider needs provider_id, mode and evaluate.")
    if not isinstance(provider.mode, ClearanceModelMode):
        raise TypeError("Clearance provider must declare an explicit ClearanceModelMode.")
    if provider.mode is ClearanceModelMode.CORRECTION and not callable(getattr(provider, "base_geometry_for", None)):
        raise TypeError("Correction provider must declare source-compatible base_geometry_for.")


def require_nominal_twisted_tape(geometry: TwistedTapeGeometry, diameter: float):
    """Shared safeguard for base models which do not themselves model edge gaps."""
    if not geometry.clearance_for(diameter).nominal_full_width:
        raise EnhancementUnsupportedError("twisted_tape_clearance_model_required")


def evaluate_with_clearance(configuration, state):
    provider, geometry = configuration.clearance_provider, configuration.geometry
    validate_clearance_provider(provider)
    physical = geometry.clearance_for(state.tube_inner_diameter)
    base = None
    if provider.mode is ClearanceModelMode.CORRECTION:
        # Only the model may select a nominal base geometry. Core never widens
        # a tape to hide an unsupported finite gap or normalizes a singular fit.
        base_geometry = provider.base_geometry_for(geometry, state)
        if not isinstance(base_geometry, TwistedTapeGeometry):
            raise TypeError("Clearance base geometry must be TwistedTapeGeometry.")
        base_geometry.clearance_for(state.tube_inner_diameter)
        base = _validate_result(configuration.provider.evaluate(base_geometry, state),
                                configuration.provider.provider_id, state,
                                getattr(configuration.provider, "thermal_property_reference", "bulk"))
    model = provider.evaluate(geometry, state, base)
    if not isinstance(model, TwistedTapeClearanceResult) or model.mode is not provider.mode:
        raise TypeError("Clearance result must match the selected provider mode.")
    fields = [
        ("tube_inner_diameter", physical.tube_inner_diameter, "m"),
        ("tape_width", physical.tape_width, "m"),
        ("diametral_clearance", physical.diametral_clearance, "m"),
        ("radial_clearance", physical.radial_clearance, "m"),
        ("twist_ratio_diameter", geometry.twist_ratio_for(state.tube_inner_diameter), "-"),
        ("clearance_model_id", provider.provider_id, "-"),
        ("clearance_model_mode", model.mode.value, "-"),
        ("clearance_model_active", 1, "-"),
        ("clearance_ratio", model.clearance_ratio, "-"),
        ("clearance_ratio_definition", model.clearance_ratio_definition, "-"),
    ]
    if model.mode is ClearanceModelMode.ABSOLUTE:
        expected_thermal_reference = getattr(provider, "thermal_property_reference", "bulk")
        result = _validate_result(model.absolute, provider.provider_id, state, expected_thermal_reference)
        fields.extend((("active_model", result.correlation_id, "-"),
                       ("replaced_base_provider", configuration.provider.provider_id, "-")))
    else:
        correction = model.correction
        # A relative correction retains its validated base property reference;
        # an absolute model above owns its own declaration instead.
        expected_thermal_reference = base.thermal_property_reference
        if base.correlation_id not in correction.compatible_base_correlation_ids:
            raise EnhancementUnsupportedError("clearance_correction_base_incompatible")
        if correction.regime != base.regime:
            raise EnhancementUnsupportedError("clearance_correction_regime_incompatible")
        cn, cf = correction.heat_transfer_factor, correction.friction_factor_factor
        basis = ("private" if "private" in (base.source_access_basis, correction.source_access_basis)
                 else "external" if "external" in (base.source_access_basis, correction.source_access_basis) else "open")
        result = replace(base, provider_id=provider.provider_id,
            correlation_id=f"{base.correlation_id}+{correction.correlation_id}",
            source_references=tuple(dict.fromkeys(base.source_references+correction.source_references)),
            source_access_basis=basis,
            alpha_inside=None if base.alpha_inside is None else base.alpha_inside*cn,
            nusselt=None if base.nusselt is None else base.nusselt*cn,
            f_darcy=base.f_darcy*cf, friction_factor_native=base.friction_factor_native*cf,
            applicability="extrapolated" if "extrapolated" in (base.applicability, correction.applicability) else "within_range",
            warnings=base.warnings+correction.warnings,
            diagnostics=tuple(replace(d, name="base."+d.name) for d in base.diagnostics))
        fields.extend((
            ("base_enhancement_provider", base.provider_id, "-"),
            ("base_enhancement_model", base.correlation_id, "-"),
            ("active_model", result.correlation_id, "-"),
            ("base_tape_width", base_geometry.tape_width, "m"),
            ("f_darcy_before_clearance", base.f_darcy, "-"),
            ("heat_transfer_factor", cn, "-"), ("friction_factor_factor", cf, "-"),
        ))
        if base.alpha_inside is not None:
            fields.append(("alpha_before_clearance", base.alpha_inside, "W/(m2 K)"))
        if base.nusselt is not None:
            fields.append(("Nu_before_clearance", base.nusselt, "-"))
    fields.append(("f_darcy_after_clearance", result.f_darcy, "-"))
    if result.alpha_inside is not None:
        fields.append(("alpha_after_clearance", result.alpha_inside, "W/(m2 K)"))
    if result.nusselt is not None:
        fields.append(("Nu_after_clearance", result.nusselt, "-"))
    names = {name for name, _, _ in fields}
    result = replace(result, diagnostics=tuple(d for d in result.diagnostics if d.name not in names)
                     + tuple(EnhancementDiagnostic(*f) for f in fields))
    return _validate_result(result, provider.provider_id, state, expected_thermal_reference)
