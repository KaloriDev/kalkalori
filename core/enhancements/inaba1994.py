# KalKalori - GNU GPL v3 only
"""Primary-source Inaba, Ozaki & Kanaoka (1994) wire-coil model.

Only the coherent P/e > 10 subset is public: distributed friction Eq. (7),
high-Re heat transfer Eq. (10), and low-Re heat transfer Eq. (11).
"""
from dataclasses import dataclass
import math

from core.common.warnings import make_warning
from .base import (
    EnhancementDiagnostic, EnhancementInput, EnhancementReferenceState,
    EnhancementResult, EnhancementUnsupportedError, WireCoilGeometry,
)


_DOI = "https://doi.org/10.1299/kikaib.60.240"
_OFFICIAL = "https://www.jstage.jst.go.jp/article/kikaib1979/60/569/60_569_240/_article/-char/en"
_TESTED_DIAMETER = 0.016
_TESTED_RATIOS = (0.125, 0.15625, 0.1875)


@dataclass(frozen=True)
class Inaba1994WireCoilProvider:
    """Water wire-coil correlation with explicit warning-mode extrapolation."""
    extrapolation_policy: str = "error"
    provider_id = "inaba_1994_wire_coil"
    thermal_property_reference = "film"
    hydraulic_property_reference = "film"

    def __post_init__(self) -> None:
        if self.extrapolation_policy not in ("error", "warn"):
            raise ValueError("extrapolation_policy must be 'error' or 'warn'.")

    def evaluate(self, geometry: object, state: EnhancementInput) -> EnhancementResult:
        if not isinstance(geometry, WireCoilGeometry):
            raise EnhancementUnsupportedError("inaba1994_geometry_unsupported")
        if state.fluid_phase != "liquid":
            raise EnhancementUnsupportedError("inaba1994_phase_unsupported: single-phase liquid only")
        D = state.tube_inner_diameter
        geometry.validate_for(D)
        q = geometry.pitch_ratio
        if q <= 10.0:
            raise EnhancementUnsupportedError(
                "inaba1994_pitch_ratio_unsupported_branch: v0.8.2 implements P/e > 10 only")

        e, pitch = geometry.wire_diameter, geometry.pitch
        helix_ratio = math.sqrt(pitch**2 + math.pi**2*(D-e)**2)/pitch
        area_ratio = 1.0 + e*helix_ratio/D
        source_hydraulic_diameter = D/area_ratio
        flow_area = math.pi*D**2/4.0
        velocity = state.mass_flow_per_tube/(state.bulk.rho*flow_area)

        hydraulic = state.hydraulic
        if hydraulic is None or state.bulk.temperature is None or hydraulic.temperature is None:
            raise EnhancementUnsupportedError("inaba1994_hydraulic_film_state_required")
        if state.wall is not None and state.wall.temperature is not None:
            expected_film = (state.bulk.temperature + state.wall.temperature)/2.0
            if not math.isclose(hydraulic.temperature, expected_film, rel_tol=1e-12):
                raise EnhancementUnsupportedError("inaba1994_hydraulic_film_temperature_inconsistent")
        source_reynolds = velocity*source_hydraulic_diameter/(hydraulic.mu/hydraulic.rho)
        source_prandtl = hydraulic.cp*hydraulic.mu/hydraulic.k

        thermal_request = state.position == "thermal"
        film = state.thermal
        if thermal_request:
            if (film is None or state.wall is None or state.wall.temperature is None
                    or film.temperature is None):
                raise EnhancementUnsupportedError("inaba1994_thermal_film_state_required")
            expected_film = (state.bulk.temperature + state.wall.temperature)/2.0
            if not math.isclose(film.temperature, expected_film, rel_tol=1e-12):
                raise EnhancementUnsupportedError("inaba1994_thermal_film_temperature_inconsistent")
            source_prandtl = film.cp*film.mu/film.k

        warnings = []
        outside = []

        def warning(code, message, severity="warning"):
            warnings.append(make_warning(code=code, message=message,
                                         severity=severity, source=self.provider_id))

        def check(name, value, lower, upper):
            if ((value < lower and not math.isclose(value, lower, rel_tol=1e-12))
                    or (value > upper and not math.isclose(value, upper, rel_tol=1e-12))):
                outside.append(name)
                warning(f"inaba1994_{name}_extrapolated",
                        f"action=extrapolated; parameter={name}; actual_value={value:.9g}; "
                        f"source range={lower}..{upper}.")

        check("reynolds_source", source_reynolds, 400.0, 6000.0)
        check("pitch_ratio", q, 10.0, 50.3)
        if thermal_request:
            check("prandtl_film", source_prandtl, 4.21, 8.12)

        ratio = e/D
        ratio_match = next((r for r in _TESTED_RATIOS
                            if math.isclose(ratio, r, rel_tol=1e-12, abs_tol=1e-15)), None)
        diameter_match = math.isclose(D, _TESTED_DIAMETER, rel_tol=1e-12)
        if ratio < _TESTED_RATIOS[0] and not math.isclose(ratio, _TESTED_RATIOS[0], rel_tol=1e-12):
            outside.append("wire_diameter_ratio")
            warning("inaba1994_wire_diameter_ratio_extrapolated",
                    f"action=extrapolated; e/d_i={ratio:.9g}; tested range="
                    f"{_TESTED_RATIOS[0]}..{_TESTED_RATIOS[-1]}.")
            ratio_status = "geometry_extrapolation"
        elif ratio > _TESTED_RATIOS[-1] and not math.isclose(ratio, _TESTED_RATIOS[-1], rel_tol=1e-12):
            outside.append("wire_diameter_ratio")
            warning("inaba1994_wire_diameter_ratio_extrapolated",
                    f"action=extrapolated; e/d_i={ratio:.9g}; tested range="
                    f"{_TESTED_RATIOS[0]}..{_TESTED_RATIOS[-1]}.")
            ratio_status = "geometry_extrapolation"
        elif ratio_match is None:
            ratio_status = "geometry_interpolation"
            warning("inaba1994_geometry_interpolation",
                    f"e/d_i={ratio:.9g} lies between tested ratios but is not a tested geometry.", "info")
        else:
            ratio_status = "tested_ratio"

        if not diameter_match:
            outside.append("tube_diameter_scaling")
            warning("inaba1994_tube_diameter_scaling",
                    f"action=scale_transfer; d_i={D:.9g} m; source tube d_i=0.016 m. "
                    "Similar dimensionless geometry is not proof of scale independence.")
            scale_status = "diameter_scaling"
        else:
            scale_status = "source_diameter"

        if outside and self.extrapolation_policy == "error":
            raise EnhancementUnsupportedError("inaba1994_extrapolation_required: " + ", ".join(outside))

        warning("inaba1994_source_context",
                "PRIMARY_SOURCE: Inaba, Ozaki & Kanaoka (1994), J-STAGE, Eqs. (7), (10), (11); "
                "water heating in a horizontal 16/20 mm tube under uniform wall temperature, "
                "tested wire diameters 2.0, 2.5 and 3.0 mm. Cooling, other wall conditions, "
                "other fluids and universal diameter scaling are not physically validated. "
                "The average correlation is used in KalKalori's 0D framework; it is not local Nu(x).", "info")
        if state.heat_flow_direction != "heating":
            warning("inaba1994_heating_context_unmatched",
                    f"source context is heating; declared direction={state.heat_flow_direction}.")
        warning("inaba1994_local_losses_outside_model",
                "Coil entrance, exit, support and attachment local losses are outside Eq. (7) and remain separate.", "info")
        if state.roughness_inner:
            warning("inaba1994_roughness_ignored",
                    f"action=ignored; roughness_inner={state.roughness_inner:.9g} m; "
                    "no separate roughness correction is added.", "info")

        source_fanning = 11.5*source_reynolds**(-0.39)*q**(-0.87)
        normalization = D/source_hydraulic_diameter
        canonical_darcy = 4.0*source_fanning*normalization

        source_nusselt = None
        source_alpha = None
        canonical_nusselt = None
        canonical_alpha = None
        branch = "not_evaluated_hydraulic_node"
        if thermal_request:
            if source_reynolds <= 2000.0:
                branch = "eq11_low_re"
                source_nusselt = (0.225*source_reynolds**0.800
                                  * source_prandtl**(1.0/3.0)*q**(-0.48))
            else:
                branch = "eq10_high_re"
                source_nusselt = (0.803*source_reynolds**0.630
                                  * source_prandtl**(1.0/3.0)*q**(-0.48))
            source_alpha = source_nusselt*film.k/source_hydraulic_diameter
            canonical_alpha = source_alpha*area_ratio
            canonical_nusselt = canonical_alpha*D/film.k

        source_area = math.pi*D*state.heated_length
        values = [
            ("tube_inner_diameter", D, "m"),
            ("wire_diameter", e, "m"),
            ("coil_pitch", pitch, "m"),
            ("pitch_ratio", q, "-"),
            ("helix_length_ratio", helix_ratio, "-"),
            ("source_area_A0", source_area, "m2/tube"),
            ("source_area_ratio_As_A0", area_ratio, "-"),
            ("source_hydraulic_diameter", source_hydraulic_diameter, "m"),
            ("source_flow_area", flow_area, "m2/tube"),
            ("source_volumetric_flow", state.mass_flow_per_tube/state.bulk.rho, "m3/s/tube"),
            ("source_velocity", velocity, "m/s"),
            ("source_reynolds", source_reynolds, "-"),
            ("hydraulic_property_reference", "film", "-"),
            ("hydraulic_reference_temperature", hydraulic.temperature, "K"),
            ("source_prandtl", source_prandtl, "-"),
            ("nusselt_equation_branch", branch, "-"),
            ("source_friction_factor", source_fanning, "-"),
            ("source_friction_convention", "FANNING", "-"),
            ("friction_reference_normalization", normalization, "-"),
            ("canonical_darcy_friction_factor", canonical_darcy, "-"),
            ("geometry_ratio_status", ratio_status, "-"),
            ("geometry_scale_status", scale_status, "-"),
            ("provenance", "Inaba, Ozaki & Kanaoka 1994; primary J-STAGE; q>10 subset", "-"),
        ]
        if thermal_request:
            values.extend((
                ("thermal_reference_temperature", film.temperature, "K"),
                ("source_nusselt", source_nusselt, "-"),
                ("source_alpha", source_alpha, "W/(m2 K)"),
                ("canonical_nusselt", canonical_nusselt, "-"),
                ("canonical_alpha_inside", canonical_alpha, "W/(m2 K)"),
            ))

        return EnhancementResult(
            provider_id=self.provider_id,
            correlation_id="inaba1994_eq7_eq10_eq11_q_gt_10",
            source_references=(_DOI, _OFFICIAL), source_access_basis="open",
            reference=EnhancementReferenceState(
                flow_area, velocity, source_reynolds, source_prandtl,
                D, source_hydraulic_diameter, D),
            alpha_inside=canonical_alpha, nusselt=canonical_nusselt,
            f_darcy=canonical_darcy, friction_factor_native=source_fanning,
            friction_basis="fanning", friction_normalization=normalization,
            regime=branch,
            applicability="extrapolated" if outside else "within_range",
            warnings=tuple(warnings),
            diagnostics=tuple(EnhancementDiagnostic(*value) for value in values),
            thermal_property_reference="film",
        )
