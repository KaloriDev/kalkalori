# KalKalori - GNU GPL v3 only
"""SI-only enhancement boundary; no correlation or property-provider imports.

One configuration owns thermal and distributed hydraulic performance.
Reference areas are PER TUBE, never bundle totals. Local losses and momentum
acceleration are outside this contract. Source policy: CONTRIBUTING.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from core.common.warnings import ModelWarning

if TYPE_CHECKING:
    from .clearance import TwistedTapeClearanceProvider


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")


class EnhancementUnsupportedError(ValueError):
    """Explicitly requested enhancement cannot describe the supplied state."""


@dataclass(frozen=True)
class TwistedTapeClearanceGeometry:
    """Nominal centered edge gaps [m], derived from bore and tape width.

    Radial clearance is half the diametral difference, not a claim about
    eccentric installation or the minimum gap around finite-thickness corners.
    No correlation-specific clearance-ratio convention is imposed here.
    """
    tube_inner_diameter: float
    tape_width: float

    def __post_init__(self) -> None:
        _positive(tube_inner_diameter=self.tube_inner_diameter, tape_width=self.tape_width)
        if self.tape_width > self.tube_inner_diameter:
            raise ValueError("Tape width must not exceed tube inside diameter.")

    @property
    def diametral_clearance(self) -> float:
        return self.tube_inner_diameter - self.tape_width

    @property
    def radial_clearance(self) -> float:
        return self.diametral_clearance / 2

    @property
    def nominal_full_width(self) -> bool:
        # Floating-point tolerance only; never a manufacturing-gap allowance.
        return math.isclose(self.tape_width, self.tube_inner_diameter, rel_tol=1e-12)


@dataclass(frozen=True)
class TwistedTapeGeometry:
    """Continuous full-length single tape, SI metres, no tape fin conduction.

    ``half_turn_length`` is axial length for 180 degrees (Yang 2020, p.3).
    Width is independent of tube diameter; providers own clearance limits.
    Tube-diameter twist ratio is derived with ``twist_ratio_for`` so it cannot
    be confused with the width-based convention of another provider.
    """
    half_turn_length: float
    tape_width: float
    tape_thickness: float

    def __post_init__(self) -> None:
        _positive(half_turn_length=self.half_turn_length,
                  tape_width=self.tape_width, tape_thickness=self.tape_thickness)
        if self.tape_thickness >= self.tape_width:
            raise ValueError("Tape thickness must be smaller than tape width.")

    def twist_ratio_for(self, tube_inner_diameter: float) -> float:
        _positive(tube_inner_diameter=tube_inner_diameter)
        return self.half_turn_length / tube_inner_diameter

    def clearance_for(self, tube_inner_diameter: float) -> TwistedTapeClearanceGeometry:
        return TwistedTapeClearanceGeometry(tube_inner_diameter, self.tape_width)


@dataclass(frozen=True)
class EnhancementState:
    """Authoritative transport state; T [K] and p [Pa] may be unavailable."""
    rho: float
    mu: float
    k: float
    cp: float
    temperature: float | None = None
    pressure: float | None = None

    def __post_init__(self) -> None:
        _positive(rho=self.rho, mu=self.mu, k=self.k, cp=self.cp)
        for name in ("temperature", "pressure"):
            value = getattr(self, name)
            if value is not None:
                _positive(**{name: value})

    @classmethod
    def from_properties(cls, props, temperature=None, pressure=None):
        return cls(props.rho, props.mu, props.k, props.cp, temperature, pressure)


@dataclass(frozen=True)
class EnhancementInput:
    mass_flow_per_tube: float
    tube_inner_diameter: float
    heated_length: float
    tube_length: float
    bulk: EnhancementState
    wall: EnhancementState | None = None
    fluid_phase: str = "unknown"
    position: str = "thermal"
    roughness_inner: float = 0.0
    heat_flow_direction: str = "unknown"
    # Solver-owned unblocked geometry. Optional for standalone evaluations;
    # the exchanger adapters always supply both, independently of the
    # provider-owned blocked/reference geometry in EnhancementResult.
    base_flow_area_per_tube: float | None = None
    hydraulic_length_total: float | None = None

    def __post_init__(self) -> None:
        _positive(mass_flow_per_tube=self.mass_flow_per_tube,
                  tube_inner_diameter=self.tube_inner_diameter,
                  heated_length=self.heated_length, tube_length=self.tube_length)
        for name in ("base_flow_area_per_tube", "hydraulic_length_total"):
            value = getattr(self, name)
            if value is not None:
                _positive(**{name: value})
        if self.heat_flow_direction not in ("heating", "cooling", "unknown"):
            raise ValueError("Unknown tube-side heat-flow direction.")
        if not math.isfinite(self.roughness_inner) or self.roughness_inner < 0:
            raise ValueError("roughness_inner must be nonnegative and finite.")
        if not isinstance(self.bulk, EnhancementState):
            raise TypeError("bulk must be an EnhancementState.")
        if self.wall is not None and not isinstance(self.wall, EnhancementState):
            raise TypeError("wall must be an EnhancementState.")

    @property
    def base_mass_flux(self) -> float | None:
        """Unblocked per-tube mass flux [kg/(m2 s)], when area is available."""
        if self.base_flow_area_per_tube is None:
            return None
        return self.mass_flow_per_tube / self.base_flow_area_per_tube


@dataclass(frozen=True)
class EnhancementDiagnostic:
    """Provider-specific scalar with explicit units; solver never interprets it."""
    name: str
    value: float | str
    units: str = "-"

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.value, (str, int, float)):
            raise ValueError("Diagnostic needs a name and scalar value.")
        if not isinstance(self.value, str) and not math.isfinite(self.value):
            raise ValueError("Diagnostic numeric values must be finite.")


@dataclass(frozen=True)
class EnhancementReferenceState:
    """Exact reference quantities used by the returned friction and Nu.

    ``friction_diameter`` and ``velocity`` define dp/L=f_D*rho*v^2/(2*D).
    ``hydraulic_diameter`` is separately reported, not substituted by core.
    ``flow_area`` is the per-tube reference area corresponding to velocity.
    Nu (if supplied) uses ``nusselt_length``. Reynolds need not use either
    diameter: an external provider owns its dimensionless-group definitions.
    """
    flow_area: float
    velocity: float
    reynolds: float
    prandtl: float
    friction_diameter: float
    hydraulic_diameter: float
    nusselt_length: float

    def __post_init__(self) -> None:
        _positive(**vars(self))


@dataclass(frozen=True)
class EnhancementResult:
    provider_id: str
    correlation_id: str
    source_references: tuple[str, ...]
    source_access_basis: str
    reference: EnhancementReferenceState
    alpha_inside: float
    f_darcy: float
    friction_factor_native: float
    friction_basis: str
    regime: str
    applicability: str = "within_range"
    nusselt: float | None = None
    wall_correction: float = 1.0
    warnings: tuple[ModelWarning, ...] = ()
    diagnostics: tuple[EnhancementDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _positive(alpha_inside=self.alpha_inside, f_darcy=self.f_darcy,
                  friction_factor_native=self.friction_factor_native,
                  wall_correction=self.wall_correction)
        if self.nusselt is not None:
            _positive(nusselt=self.nusselt)
        if not self.provider_id or not self.correlation_id or not self.regime:
            raise ValueError("Provider, correlation and regime identifiers are required.")
        if self.source_access_basis not in ("open", "private", "external"):
            raise ValueError("Unknown source access basis.")
        if self.applicability not in ("within_range", "extrapolated"):
            raise EnhancementUnsupportedError("Unsupported enhancement applicability.")
        if not isinstance(self.reference, EnhancementReferenceState):
            raise TypeError("reference must be an EnhancementReferenceState.")
        for name, kind in (("source_references", str), ("warnings", ModelWarning),
                           ("diagnostics", EnhancementDiagnostic)):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(v, kind) for v in value):
                raise TypeError(f"{name} must be an immutable tuple of {kind.__name__}.")
        if not self.source_references or any(not r for r in self.source_references):
            raise ValueError("Source/provenance references are required.")
        if self.applicability == "extrapolated" and not self.warnings:
            raise ValueError("Extrapolation requires a warning.")
        if self.friction_basis not in ("darcy", "fanning"):
            raise ValueError("Friction basis must be darcy or fanning.")
        factor = 4.0 if self.friction_basis == "fanning" else 1.0
        if not math.isclose(self.f_darcy, factor*self.friction_factor_native, rel_tol=1e-12):
            raise ValueError("Inconsistent Darcy/native friction convention.")

    @property
    def f_fanning(self) -> float:
        return self.f_darcy / 4.0

    def pressure_gradient(self, density: float) -> float:
        _positive(density=density)
        r = self.reference
        value = self.f_darcy*density*r.velocity**2/(2*r.friction_diameter)
        _positive(pressure_gradient=value)
        return value


@runtime_checkable
class TubeSideEnhancementProvider(Protocol):
    provider_id: str

    def evaluate(self, geometry: object, state: EnhancementInput) -> EnhancementResult:
        """Return paired thermal/hydraulic performance or raise unsupported."""
        ...


@dataclass(frozen=True)
class TubeSideEnhancement:
    provider: TubeSideEnhancementProvider
    geometry: object
    fluid_phase: str
    clearance_provider: TwistedTapeClearanceProvider | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, TubeSideEnhancementProvider):
            raise TypeError("Enhancement provider must implement evaluate and provider_id.")
        if not self.provider.provider_id or self.geometry is None:
            raise ValueError("A provider identifier and geometry are required.")
        if self.fluid_phase not in ("liquid", "gas"):
            raise EnhancementUnsupportedError("Declare single-phase liquid or gas explicitly.")
        if self.clearance_provider is not None:
            from .clearance import validate_clearance_provider
            if not isinstance(self.geometry, TwistedTapeGeometry):
                raise TypeError("A clearance provider requires TwistedTapeGeometry.")
            validate_clearance_provider(self.clearance_provider)


def evaluate_enhancement(configuration: TubeSideEnhancement | None,
                         state: EnhancementInput) -> EnhancementResult | None:
    """None is the exact legacy-dispatch sentinel. Never catch provider errors."""
    if configuration is None:
        return None
    if state.fluid_phase != configuration.fluid_phase:
        raise EnhancementUnsupportedError("Enhancement fluid phase does not match configuration.")
    if isinstance(configuration.geometry, TwistedTapeGeometry):
        configuration.geometry.clearance_for(state.tube_inner_diameter)
    if configuration.clearance_provider is not None:
        from .clearance import evaluate_with_clearance
        return evaluate_with_clearance(configuration, state)
    result = configuration.provider.evaluate(configuration.geometry, state)
    return _validate_result(result, configuration.provider.provider_id, state)


def _validate_result(result, expected_provider_id, state):
    """Check the same physical contract for native, corrected and absolute results."""
    if not isinstance(result, EnhancementResult):
        raise TypeError("Provider must return a coherent EnhancementResult.")
    if result.provider_id != expected_provider_id:
        raise ValueError("Returned provider identity does not match selected provider.")
    r = result.reference
    velocity = state.mass_flow_per_tube/(state.bulk.rho*r.flow_area)
    if not math.isclose(velocity, r.velocity, rel_tol=1e-10):
        raise ValueError("Provider reference area and mass-flow velocity are inconsistent.")
    if result.nusselt is not None and not math.isclose(
            result.alpha_inside, result.nusselt*state.bulk.k/r.nusselt_length, rel_tol=1e-10):
        raise ValueError("Provider Nu and alpha reference bases are inconsistent.")
    return result
