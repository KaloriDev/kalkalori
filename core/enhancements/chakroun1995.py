# KalKalori - GNU GPL v3 only
"""Chakroun & Al-Fahed, ASME 95-GT-042 (1995), Eqs.16 and 17.

Independent implementation from the audited primary-source equations.
Actual Ac(W) is predictive; the printed nominal Eq.11 is diagnostic only.
See docs/twisted_tape_chakroun1995.md for provenance and source limitations.
"""
from dataclasses import dataclass
import math

from core.common.warnings import make_warning
from .base import (
    EnhancementDiagnostic, EnhancementInput, EnhancementReferenceState,
    EnhancementResult, EnhancementUnsupportedError, TwistedTapeGeometry,
)


@dataclass(frozen=True)
class ChakrounAlFahed1995TwistedTapeProvider:
    """Paired laminar liquid model; optional strict demonstrated source domain.

    Requires the coupled wall state, using bulk properties and wall viscosity.
    Returned friction uses empty-bore velocity and physical axial path length.
    Width is represented through blockage, without an additional leakage fit.
    """
    strict_source_domain: bool = False
    provider_id = "chakroun_al_fahed_1995_twisted_tape"
    requires_wall_state = True
    thermal_property_reference = "bulk"

    def __post_init__(self):
        if not isinstance(self.strict_source_domain, bool):
            raise TypeError("strict_source_domain must be a bool.")

    def evaluate(self, geometry: object, state: EnhancementInput) -> EnhancementResult:
        if not isinstance(geometry, TwistedTapeGeometry):
            raise EnhancementUnsupportedError("chakroun1995_geometry_unsupported")
        if state.fluid_phase != "liquid":
            raise EnhancementUnsupportedError("chakroun1995_phase_unsupported: single-phase liquid only")
        if state.wall is None or state.wall.temperature is None:
            raise EnhancementUnsupportedError("enhancement_wall_state_required")
        D = state.tube_inner_diameter
        clearance = geometry.clearance_for(D)
        delta = geometry.tape_thickness
        y = geometry.twist_ratio_for(D)
        bulk = state.bulk
        area = math.pi*D**2/4
        ac = area - delta*geometry.tape_width
        nominal_ac = area - delta*D
        if ac <= 0 or nominal_ac <= 0:
            raise EnhancementUnsupportedError("chakroun1995_blockage_unsupported: nonpositive flow area")
        phi = area/ac
        helical = math.sqrt(1 + (math.pi/(2*y))**2)
        v0 = state.mass_flow_per_tube/(bulk.rho*area)
        va = state.mass_flow_per_tube/(bulk.rho*ac)
        vs = va*helical
        re = bulk.rho*v0*D/bulk.mu
        re_sw = bulk.rho*vs*D/bulk.mu
        sw = re_sw/math.sqrt(y)
        sw_nominal = re*(area/nominal_ac)*helical/math.sqrt(y)
        pr = bulk.cp*bulk.mu/bulk.k
        ratio = bulk.mu/state.wall.mu
        warnings = []
        outside = []

        def warning(code, message, severity="warning"):
            warnings.append(make_warning(code="chakroun1995_" + code, message=message,
                                         source=self.provider_id, severity=severity))

        def extrapolation(code, message):
            outside.append(code)
            warning(code, "action=extrapolated; " + message)

        def in_range(value, lower, upper):
            return ((value >= lower or math.isclose(value, lower, rel_tol=1e-12))
                    and (value <= upper or math.isclose(value, upper, rel_tol=1e-12)))

        # Source applicability uses empty-bore Re, not Re_sw. Retain the
        # reported inclusive 2300 endpoint; do not extend into transition.
        if re > 2300 and not math.isclose(re, 2300, rel_tol=1e-12):
            raise EnhancementUnsupportedError(
                "chakroun1995_regime_unsupported: empty-bore Re > 2300; no transition/turbulence model")
        if not in_range(re, 230, 2300):
            extrapolation("re_below_source_range", f"Re_empty={re:.9g}; source 230..2300.")
        if not in_range(y, 3.6, 7.1):
            extrapolation("twist_ratio_extrapolation", f"Y=H180/D={y:.9g}; source 3.6..7.1.")
        if not in_range(geometry.tape_width/D, 10.8/14, 13.2/14):
            extrapolation("width_ratio_extrapolation",
                          f"W/D={geometry.tape_width/D:.9g}; source 10.8/14..13.2/14.")
        # One source thickness only. 1% is a reporting tolerance, not an
        # experimentally established thickness interval.
        if not math.isclose(delta/D, .5/14, rel_tol=.01):
            extrapolation("thickness_ratio_extrapolation",
                          f"delta/D={delta/D:.9g}; single source value 0.5/14; no thickness sweep.")
        cooling = (state.heat_flow_direction == "cooling"
                   or (bulk.temperature is not None and state.wall.temperature < bulk.temperature))
        if cooling:
            extrapolation("cooling_extrapolation",
                          "source demonstrated heating only; Eq.17 retains (mu_bulk/mu_wall)^0.14.")
        if outside and self.strict_source_domain:
            raise EnhancementUnsupportedError("chakroun1995_source_domain_unsupported: " + ", ".join(outside))

        warning("source_width_interpretation",
                "Predictive Sw uses actual Ac=A0-delta*W; printed Eq.11 uses nominal W=D. "
                "No averaging, interpretation switch or empirical leakage correction.", "info")
        warning("fanning_to_darcy_conversion",
                "Eq.16 is native Fanning: f_D,sw=4*f_F,sw once; axial normalization=phi^2*s^3.", "info")
        warning("constant_wall_temperature_source",
                "Source: oil heating under approximately constant wall temperature. "
                "KalKalori uses a coupled 0D wall; no local axial or arbitrary-clearance validation.", "info")
        warning("source_scope",
                "Source Pr range unknown; source D=14 mm, delta=0.5 mm. Additional measured "
                "width/leakage behavior is not fully resolved by the final correlation. "
                "Tape conduction, entry effects and insert local losses are outside this model.", "info")
        if state.roughness_inner:
            warning("roughness_ignored", "No separate roughness friction correction is applied.", "info")

        # ASME 95-GT-042 Eq.16, native Fanning. The nominal geometric
        # factor is retained as printed; no W-based perimeter is invented.
        f_re_fanning = (15.767*((math.pi + 2 - 2*delta/D)/(math.pi - 4*delta/D))**2
                        * (1 + 1e-6*sw**2.55)**(1/6))
        f_fanning = f_re_fanning/re_sw
        f_darcy_swirl = 4*f_fanning
        # dp = f_D,sw * (L*s/D) * rho*(V0*phi*s)^2/2.
        # Core multiplies by physical axial L, so this includes s exactly once
        # for length and twice for velocity, independent of the Fanning factor.
        normalization = phi**2*helical**3
        f_darcy_axial = f_darcy_swirl*normalization
        thermal = state.position == "thermal"
        correction = ratio**.14
        # Eq.17: all thermal properties bulk except mu_wall. No extra
        # viscosity, length, clearance or tape-area correction follows.
        nu = .318*sw**.569*pr**.3*correction if thermal else None
        alpha = nu*bulk.k/D if thermal else None
        values = [
            ("Re_empty", re, "-"), ("Re_sw", re_sw, "-"), ("Pr_bulk", pr, "-"),
            ("Y", y, "-"), ("W_over_D", geometry.tape_width/D, "-"),
            ("clearance", clearance.radial_clearance, "m"),
            ("clearance_over_D", clearance.radial_clearance/D, "-"),
            ("delta_over_D", delta/D, "-"), ("A0", area, "m2"), ("Ac", ac, "m2"),
            ("phi", phi, "-"), ("helical_factor", helical, "-"),
            ("V0", v0, "m/s"), ("Va", va, "m/s"), ("Vs", vs, "m/s"),
            ("Sw_actual_area", sw, "-"), ("Sw_printed_nominal", sw_nominal, "-"),
            ("f_Fanning_swirl", f_fanning, "-"), ("f_Darcy_swirl", f_darcy_swirl, "-"),
            ("f_Darcy_axial_equivalent", f_darcy_axial, "-"),
            ("mu_bulk", bulk.mu, "Pa s"), ("mu_wall", state.wall.mu, "Pa s"),
            ("mu_bulk_over_wall", ratio, "-"), ("T_wall", state.wall.temperature, "K"),
            ("source_pr_range", "unknown", "-"),
            ("provenance", "Chakroun & Al-Fahed 1995; ASME 95-GT-042; Eqs.16/17", "-"),
        ]
        if bulk.temperature is not None:
            values.append(("T_bulk", bulk.temperature, "K"))
        if thermal:
            values.extend((("Nu", nu, "-"), ("h", alpha, "W/(m2 K)")))
        return EnhancementResult(
            provider_id=self.provider_id, correlation_id="chakroun1995_eq16_eq17_actual_area",
            source_references=("https://doi.org/10.1115/95-GT-042",),
            source_access_basis="open",
            reference=EnhancementReferenceState(area, v0, re, pr, D, D, D),
            alpha_inside=alpha, nusselt=nu, wall_correction=correction,
            f_darcy=f_darcy_axial, friction_factor_native=f_fanning,
            friction_basis="fanning", friction_normalization=normalization,
            regime="laminar", applicability="extrapolated" if outside else "within_range",
            warnings=tuple(warnings), diagnostics=tuple(EnhancementDiagnostic(*v) for v in values),
        )
