# KalKalori - GNU GPL v3 only
"""Yang et al. (2020), ordinary single-tape specialization of Eqs.21-22.

Open implementation source: doi:10.3389/fenrg.2020.00178, publisher PDF
pp.3,11,12, Eqs.1-3,11-14,19-22. NOT canonical Manglik-Bergles Part I/II.
See docs/twisted_tape_manglik_bergles.md for the frozen source and limitations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from core.common.warnings import make_warning
from .base import (
    EnhancementDiagnostic, EnhancementInput, EnhancementReferenceState,
    EnhancementResult, EnhancementUnsupportedError, TwistedTapeGeometry,
)


def _within(value: float, lower: float, upper: float, name: str) -> None:
    # Roundoff tolerance only, not an extrapolation interval.
    if value < lower and not math.isclose(value, lower, rel_tol=1e-12):
        raise EnhancementUnsupportedError(f"yang2020_{name}_unsupported: {value}; range {lower}..{upper}.")
    if value > upper and not math.isclose(value, upper, rel_tol=1e-12):
        raise EnhancementUnsupportedError(f"yang2020_{name}_unsupported: {value}; range {lower}..{upper}.")


@dataclass(frozen=True)
class Yang2020TwistedTapeProvider:
    """Laminar, single-phase liquid heating; no geometry/regime extrapolation.

    Native friction is DARCY (source Eq.2). No separate smooth-tube wall,
    length, roughness, blockage or friction multiplier may be applied.
    """
    provider_id = "yang_2020_twisted_tape"

    def evaluate(self, geometry: object, state: EnhancementInput) -> EnhancementResult:
        if not isinstance(geometry, TwistedTapeGeometry):
            raise EnhancementUnsupportedError("yang2020_geometry_unsupported: requires a single continuous tape.")
        if state.fluid_phase != "liquid":
            raise EnhancementUnsupportedError("yang2020_phase_unsupported: single-phase liquid only.")
        if state.heat_flow_direction == "cooling":
            raise EnhancementUnsupportedError("yang2020_cooling_unsupported: liquid heating only.")
        D = state.tube_inner_diameter
        delta = geometry.tape_thickness
        # Strict measured dimensions: the paper supplies no broad size range.
        for name, value, required in (
            ("diameter", D, .012), ("thickness", delta, .001),
            ("width", geometry.tape_width, D),
            ("heated_length", state.heated_length, .3),
        ):
            _within(value, required, required, name)
        y = geometry.twist_ratio_for(D)
        _within(y, 2, 4, "twist_ratio")
        bulk = state.bulk
        area_empty = math.pi*D**2/4
        area_open = area_empty - delta*D  # Eq.11; correlation geometry.
        perimeter = math.pi*D + 2*D - 2*delta  # Eq.12.
        dh = 4*area_open/perimeter
        phi = area_empty/area_open
        psi = D/dh
        velocity = state.mass_flow_per_tube/(bulk.rho*area_empty)
        re = bulk.rho*velocity*D/bulk.mu
        pr = bulk.cp*bulk.mu/bulk.k
        _within(re, 100, 1100, "reynolds")
        _within(pr, 7, 900, "prandtl")
        swirl_squared = 1 + (math.pi/(2*y))**2
        sw = re*phi*math.sqrt(swirl_squared/y)  # Eq.13 final expression.
        gz = state.mass_flow_per_tube*bulk.cp/(bulk.k*state.heated_length)
        # Eq.21, A=59.24 for TT, native Darcy per Eq.2.
        f_darcy = 59.24*phi*psi**2*(1 + 1.348e-3*sw**1.09)**.462*swirl_squared/re
        # Eq.22, B=3.81 for TT. No Eq.9/10 or inaccessible source completion.
        nu_base = 3.81*((1 + .2165*gz**.662)**.251
                        + 3.87e-2*(sw*pr**.4)**.431)**2.06
        warnings = [make_warning(
            code="enhancement_yang2020_scope",
            message=("Yang2020: numerical laminar liquid-heating correlation; "
                     "forced convection, negligible buoyancy/radiation, adiabatic tape; "
                     "insert-specific local losses excluded. Source fit deviations: Nu 20%, f 12%."),
            source=self.provider_id, severity="info",
        )]
        if not math.isclose(state.tube_length, state.heated_length, rel_tol=1e-12):
            warnings.append(make_warning(
                code="yang2020_physical_length_ignored", source=self.provider_id, severity="info",
                message=(f"action=ignored; parameter=tube_length; actual_value={state.tube_length:.9g} m. "
                         f"The correlation uses heated_length={state.heated_length:.9g} m for thermal "
                         "development. Physical length is not an equation input; the hydraulic path remains solver-owned."),
            ))
        if state.roughness_inner > 0:
            warnings.append(make_warning(
                code="yang2020_roughness_ignored", source=self.provider_id, severity="info",
                message=(f"action=ignored; parameter=roughness_inner; actual_value={state.roughness_inner:.9g} m. "
                         "The selected correlation contains no roughness term. Enhanced distributed "
                         "friction uses the provider model; no additional roughness correction is applied."),
            ))
        correction = 1.0
        if state.wall is None:
            warnings.append(make_warning(
                code="enhancement_wall_state_unavailable",
                message="Yang2020: no wall state; unity viscosity correction for this evaluation.",
                source=self.provider_id, severity="info",
            ))
        else:
            if bulk.temperature is None or state.wall.temperature is None:
                raise EnhancementUnsupportedError("yang2020_temperature_required: cannot verify heating direction.")
            if state.wall.temperature < bulk.temperature:
                raise EnhancementUnsupportedError("yang2020_cooling_unsupported: liquid heating only.")
            correction = (bulk.mu/state.wall.mu)**.14
        nu = nu_base*correction
        ref = EnhancementReferenceState(area_empty, velocity, re, pr, D, dh, D)
        diagnostics = tuple(EnhancementDiagnostic(name, value, units) for name, value, units in (
            ("empty_tube_area", area_empty, "m2"), ("open_axial_area", area_open, "m2"),
            ("wetted_perimeter", perimeter, "m"), ("phi", phi, "-"), ("psi", psi, "-"),
            ("twist_ratio_diameter", y, "-"), ("swirl_parameter", sw, "-"),
            ("graetz", gz, "-"), ("axial_velocity", velocity*phi, "m/s"),
            ("swirl_velocity", velocity*phi*math.sqrt(swirl_squared), "m/s"),
            ("nusselt_base", nu_base, "-"),
        ))
        return EnhancementResult(
            provider_id=self.provider_id, correlation_id="yang2020_eq21_eq22_single_tape",
            source_references=("https://doi.org/10.3389/fenrg.2020.00178",),
            source_access_basis="open", reference=ref, alpha_inside=nu*bulk.k/D,
            f_darcy=f_darcy, friction_factor_native=f_darcy, friction_basis="darcy",
            regime="laminar", nusselt=nu, wall_correction=correction,
            warnings=tuple(warnings), diagnostics=diagnostics,
        )
