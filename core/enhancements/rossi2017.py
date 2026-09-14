# KalKalori - GNU GPL v3 only
"""Open-secondary reconstruction of Rossi et al. (2017), Eqs.9 and 10.

Implementation authority: doi:10.1088/1742-6596/923/1/012033, article pp.3-7
(CC BY 3.0). Historical attribution: Agarwal & Raja Rao (1996), not a
primary-source verification. See docs/twisted_tape_rossi2017.md.
"""
from dataclasses import dataclass
import math

from core.common.warnings import make_warning
from .base import (
    EnhancementDiagnostic, EnhancementInput, EnhancementReferenceState,
    EnhancementResult, EnhancementUnsupportedError, TwistedTapeGeometry,
)


@dataclass(frozen=True)
class Rossi2017TwistedTapeProvider:
    """Nominal tape comparator; explicit ``'warn'`` permits extrapolation.

    Thermal properties must be supplied at (T_bulk + T_wall)/2 by the
    integration layer. Re and distributed friction use the local bulk flow
    state. Hydraulic-only nodes do not evaluate Eq.10.
    """
    extrapolation_policy: str = "error"
    provider_id = "rossi_2017_twisted_tape"
    thermal_property_reference = "film"

    def __post_init__(self):
        if self.extrapolation_policy not in ("error", "warn"):
            raise ValueError("extrapolation_policy must be 'error' or 'warn'.")

    def evaluate(self, geometry: object, state: EnhancementInput) -> EnhancementResult:
        if not isinstance(geometry, TwistedTapeGeometry):
            raise EnhancementUnsupportedError("rossi2017_geometry_unsupported")
        if state.fluid_phase != "liquid":
            raise EnhancementUnsupportedError("rossi2017_phase_unsupported: single-phase liquid only")
        D = state.tube_inner_diameter
        clearance = geometry.clearance_for(D)
        y = geometry.twist_ratio_for(D)
        area = math.pi*D**2/4
        velocity = state.mass_flow_per_tube/(state.bulk.rho*area)
        re = state.bulk.rho*velocity*D/state.bulk.mu
        thermal = state.position == "thermal"
        film = state.thermal
        if thermal:
            if (film is None or state.wall is None or state.bulk.temperature is None
                    or state.wall.temperature is None or film.temperature is None):
                raise EnhancementUnsupportedError("rossi2017_film_state_required")
            if not math.isclose(film.temperature, (state.bulk.temperature + state.wall.temperature)/2, rel_tol=1e-12):
                raise EnhancementUnsupportedError("rossi2017_film_temperature_inconsistent")
        pr = (film.cp*film.mu/film.k if thermal else state.bulk.cp*state.bulk.mu/state.bulk.k)
        warnings = []

        def warning(code, message, severity="warning"):
            warnings.append(make_warning(code=code, message=message, severity=severity, source=self.provider_id))

        outside = []
        for name, value, lower, upper in (("reynolds", re, 210, 3100),
                                         ("twist_ratio", y, 4.44, 4.44)) + (
                                             (("prandtl_film", pr, 44, 51),) if thermal else ()):
            if ((value < lower and not math.isclose(value, lower, rel_tol=1e-12))
                    or (value > upper and not math.isclose(value, upper, rel_tol=1e-12))):
                outside.append(name)
                warning(f"rossi2017_{name}_extrapolated",
                        f"action=extrapolated; parameter={name}; actual_value={value:.9g}; "
                        f"Rossi direct comparison={lower}..{upper}. A single y=4.44 is not a universal range.")
        if outside and self.extrapolation_policy == "error":
            raise EnhancementUnsupportedError("rossi2017_extrapolation_required: " + ", ".join(outside))
        warning("rossi2017_scope",
                "OPEN_SECONDARY_RECONSTRUCTED: Rossi 2017 Eq9/10, historically attributed to Agarwal & Raja Rao 1996. "
                "Rossi comparison: 40% water/ethylene glycol, horizontal 13.5 mm pipe, uniform wall heat flux, "
                "adiabatic tape, fully developed model. Dimensionless agreement is not universal validation. "
                "No buoyancy, insert conduction or insert-specific local losses.", "info")
        warning("rossi2017_width_clearance_not_represented",
                f"action=ignored; tape_width={geometry.tape_width:.9g} m; radial_clearance={clearance.radial_clearance:.9g} m. "
                "Nominal-tape comparator only; finite width / clearance are not represented. Not a real-clearance model.")
        warning("rossi2017_thickness_ignored",
                f"action=ignored; tape_thickness={geometry.tape_thickness:.9g} m; no thickness or blockage correction.", "info")
        if state.roughness_inner:
            warning("rossi2017_roughness_ignored",
                    f"action=ignored; roughness_inner={state.roughness_inner:.9g} m; no additional smooth friction correction.", "info")
        # Rossi Eq.9, Darcy: source smooth reference is 64/Re (article p4).
        friction = 19.48*re**(-.6519)*y**(-.6281)
        # Rossi Eq.10. No viscosity-ratio multiplier or heating/cooling branch.
        nu = .725*re**.568*y**(-.788)*pr**(1/3) if thermal else None
        alpha = nu*film.k/D if thermal else None
        values = [
            ("verification_classification", "OPEN_SECONDARY_RECONSTRUCTED", "-"),
            ("historical_correlation_attribution", "Agarwal & Raja Rao 1996", "-"),
            ("twist_ratio_diameter", y, "-"),
            ("tape_width", geometry.tape_width, "m"),
            ("radial_clearance_not_modelled", clearance.radial_clearance, "m"),
            ("thermal_evaluation", "film" if thermal else "not_evaluated_hydraulic_node", "-"),
        ]
        if thermal:
            values.extend((("T_bulk", state.bulk.temperature, "K"),
                           ("T_wall", state.wall.temperature, "K"),
                           ("T_film", film.temperature, "K"),
                           ("Pr_film", pr, "-"), ("k_film", film.k, "W/(m K)")))
        return EnhancementResult(
            provider_id=self.provider_id, correlation_id="rossi2017_eq9_eq10",
            source_references=("https://doi.org/10.1088/1742-6596/923/1/012033",),
            source_access_basis="open",
            reference=EnhancementReferenceState(area, velocity, re, pr, D, D, D),
            alpha_inside=alpha, nusselt=nu, f_darcy=friction,
            friction_factor_native=friction, friction_basis="darcy",
            regime="laminar_transitional_comparator", applicability="extrapolated" if outside else "within_range",
            warnings=tuple(warnings), diagnostics=tuple(EnhancementDiagnostic(*v) for v in values),
            thermal_property_reference="film",
        )
