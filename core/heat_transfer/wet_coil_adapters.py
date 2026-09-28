# SPDX-License-Identifier: GPL-3.0-only
"""Internal KalKalori property and geometry boundary for the wet-coil engine.

No public model selector, Rating or Simulation wiring lives here. All
production temperatures are kelvin and gas enthalpy/cp use kg dry carrier.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from math import log, pi

from numpy.polynomial.legendre import leggauss
from numpy.polynomial import chebyshev as cheb
import numpy as np
from scipy.optimize import brentq

from core.geometry.finned_tube import CircularFinnedTube
from core.heat_transfer.elmahdy_mitalas import CoilInput, SOURCE_REVISION
from core.heat_transfer.internal_flow import (
    heat_transfer_coefficient_internal_diagnostics,
)
from core.heat_transfer.outside_dispatch import (
    calculate_resistance_network,
    evaluate_outside_thermal,
)
from core.heat_transfer.wet_coil import (
    _solve_profile_candidate,
    validate_wet_coil,
    WetCoilModelError,
)
from core.phase_change.wet_gas_enthalpy import WetGasEnthalpyEvaluator
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.water_equilibrium import saturated_water_ratio, water_dew_point
from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props
from core.properties.water import (
    water_saturation_liquid_enthalpy,
    water_saturation_vapor_enthalpy,
    water_saturation_temperature,
)


class WetGasThermodynamics:
    """Existing dry-carrier CoolProp + water IAPWS datum, without air substitution."""

    def __init__(self, pressure, capability):
        if not capability.capable or capability.component != "H2O":
            raise ValueError("The wet coil requires a configured H2O/dry-carrier gas")
        self.pressure, self.capability = pressure, capability
        self.evaluator = WetGasEnthalpyEvaluator(
            pressure, capability, reuse_dry_backend_state=True
        )
        self.lower = 273.16
        self.saturation_upper = water_saturation_temperature(pressure) - 1e-3

    def enthalpy(self, T, W):
        return self.evaluator.enthalpy(T, W)

    @lru_cache(maxsize=16384)
    def saturation_humidity(self, T):
        if T >= self.saturation_upper + 1e-3:
            return float("inf")  # No finite saturated vapor/dry-carrier ratio.
        return saturated_water_ratio(
            p_total=self.pressure,
            T=T,
            M_dry=self.capability.M_dry,
            M_h2o=self.capability.M_condensable,
        )

    @lru_cache(maxsize=16384)
    def saturation_enthalpy(self, T):
        return self.enthalpy(T, self.saturation_humidity(T))

    def saturation_temperature(self, h):
        return brentq(
            lambda T: self.saturation_enthalpy(T) - h,
            self.lower,
            self.saturation_upper,
            xtol=2e-9,
        )

    def humidity(self, T, h):
        return (h - self.enthalpy(T, 0.0)) / water_saturation_vapor_enthalpy(T=T)

    def temperature(self, h, W):
        return brentq(lambda T: self.enthalpy(T, W) - h, self.lower, 640.0, xtol=2e-9)

    @lru_cache(maxsize=16384)
    def condensate_enthalpy(self, T):
        return water_saturation_liquid_enthalpy(T=T)

    def dewpoint(self, W):
        c = self.capability
        partial = self.pressure * W * c.M_dry / (c.M_condensable + W * c.M_dry)
        return water_dew_point(partial, tolerance_K=1e-8)

    def cp(self, T, W):
        return (self.enthalpy(T + 0.005, W) - self.enthalpy(T - 0.005, W)) / 0.01

    def secant_cp(self, T1, T2, W):
        return (
            self.cp((T1 + T2) / 2, W)
            if abs(T1 - T2) < 1e-4
            else (self.enthalpy(T1, W) - self.enthalpy(T2, W)) / (T1 - T2)
        )


@dataclass(frozen=True)
class InsideWallAdapter:
    bundle: object
    provider: object
    mass_flow: float
    pressure: float

    def enthalpy_difference(self, T1, T2):
        """Sensible liquid enthalpy rise; native h where provided, cp integral otherwise."""
        if hasattr(self.provider, "specific_enthalpy"):
            return self.provider.specific_enthalpy(
                T2, self.pressure
            ) - self.provider.specific_enthalpy(T1, self.pressure)
        if hasattr(self.provider, "full_at"):
            return (
                self.provider.full_at(T2, self.pressure).h
                - self.provider.full_at(T1, self.pressure).h
            )
        nodes, weights = leggauss(8)
        return (
            (T2 - T1)
            / 2
            * sum(
                w * self.provider.at(T1 + (n + 1) * (T2 - T1) / 2, self.pressure).cp
                for n, w in zip(nodes, weights)
            )
        )

    def capacity(self, T1, T2):
        cp = (
            self.provider.at((T1 + T2) / 2, self.pressure).cp
            if abs(T2 - T1) < 1e-5
            else self.enthalpy_difference(T1, T2) / (T2 - T1)
        )
        return self.mass_flow * cp

    def evaluate(self, T):
        b = self.bundle
        core = b.tube.core_tube if isinstance(b.tube, CircularFinnedTube) else b.tube
        if hasattr(self.provider, "full_at"):
            full = self.provider.full_at(T, self.pressure)
            phase = getattr(full.phase, "value", full.phase)
            incompressible = (
                getattr(self.provider, "fluid", "").upper().startswith("INCOMP::")
            )
            if not incompressible and str(phase).lower() not in (
                "liquid",
                "subcooled_liquid",
                "supercritical_liquid",
            ):
                raise ValueError(
                    "Inside wet-coil adapter requires a sensible liquid state"
                )
        props = self.provider.at(T, self.pressure)
        inside = heat_transfer_coefficient_internal_diagnostics(
            self.mass_flow,
            b.internal_hydraulic_diameter,
            b.internal_flow_area_per_pass,
            to_internal_fluid_props(props),
            T_bulk=T,
            L_heated=core.length_effective,
        )
        wall = log(core.D_o / core.D_i) / (
            2 * pi * core.wall_k * core.length_effective * b.n_tubes_total
        )
        film = 1 / (inside.alfa_corrected * b.total_inner_area)
        return film + wall, dict(
            film_resistance=film,
            wall_resistance=wall,
            fouling_resistance=0.0,
            htc=inside.alfa_corrected,
            reynolds=inside.Re,
            inside_htc_model=(
                "Hausen thermal entry"
                if inside.Re < 2300
                else (
                    "Gnielinski"
                    if inside.Re > 4000
                    else "laminar/Gnielinski transition blend"
                )
            ),
            property_provider=type(self.provider).__name__,
            warnings=inside.warnings,
        )


@dataclass(frozen=True)
class BareTubeAdapter:
    bundle: object
    thermodynamics: WetGasThermodynamics

    def evaluate(self, T, W, md, inside):
        if isinstance(self.bundle.tube, CircularFinnedTube):
            raise TypeError("BareTubeAdapter requires a bare tube")
        provider = wet_gas_provider_at_water_ratio(self.thermodynamics.capability, W)
        props = provider.at(T, self.thermodynamics.pressure)
        outside = evaluate_outside_thermal(
            bundle=self.bundle, m_dot=md * (1 + W), props=to_outside_fluid_props(props)
        )
        ri, idiag = inside
        alpha = outside.alpha_physical
        area = self.bundle.total_outer_area
        return 1 / (alpha * area), dict(
            geometry_adapter="BareTube",
            outside_htc_model="Zukauskas",
            outside_alpha_physical=alpha,
            dry_effective_area=area,
            wet_effective_area=area,
            physical_area=area,
            outside_reynolds=outside.reynolds_number,
            outside_warnings=outside.warnings,
        )


def solve_production_coil(
    *,
    bundle,
    thermodynamics,
    inside,
    air_in,
    liquid_in,
    humidity_in,
    dry_mass_flow,
    drain_enabled=True,
    radial_cells=64,
    quadrature_order=10,
):
    """Internal production adapter; not a public solving-mode entry point."""
    if inside.bundle != bundle:
        raise ValueError("Inside/outside adapters must use the same bundle")
    if bundle.flow_arrangement == "cocurrentflow":
        raise ValueError("Elmahdy-Mitalas requires a counterflow process arrangement")
    finned = isinstance(bundle.tube, CircularFinnedTube)
    outside = (
        CircularFinnedTubeAdapter(bundle, thermodynamics, radial_cells)
        if finned
        else BareTubeAdapter(bundle, thermodynamics)
    )
    previous = None
    drain_correction = np.array([0.0])
    taout, twout, tint = air_in, liquid_in, air_in
    Wout = humidity_in
    for iteration in range(80):
        cp = thermodynamics.secant_cp(air_in, tint, humidity_in)
        capacity = inside.capacity(liquid_in, twout)
        ri, idiag = inside.evaluate((liquid_in + twout) / 2)
        rd, odiag = outside.evaluate(
            (air_in + taout) / 2, (humidity_in + Wout) / 2, dry_mass_flow, (ri, idiag)
        )
        if finned:
            ri += odiag["common_root_contact_resistance"]
        rw = (
            thermodynamics.secant_cp(
                tint, thermodynamics.dewpoint(humidity_in), humidity_in
            )
            * rd
        )
        if finned and previous is not None and previous.profile:
            points = previous.profile[1:]
            weights = leggauss(len(points))[1] / 2
            ts = sum(p.surface_temperature * v for p, v in zip(points, weights))
            tg = sum(p.gas_temperature * v for p, v in zip(points, weights))
            w = sum(p.humidity * v for p, v in zip(points, weights))
            response = outside.response(ts, tg, w, odiag)
            rw = (
                thermodynamics.enthalpy(tg, w) - thermodynamics.saturation_enthalpy(ts)
            ) / response["heat_gas"]
            if rw <= 0:
                raise WetCoilModelError("invalid wet annular response")
            odiag["representative_radial_response"] = response
            odiag["wet_effective_area"] = cp / (odiag["outside_alpha_physical"] * rw)
        surface_enthalpy = (
            (
                lambda u, Ts, Tg, W: thermodynamics.condensate_enthalpy(Ts)
                + cheb.chebval(2 * u - 1, drain_correction)
            )
            if finned
            else None
        )
        x = CoilInput(
            air_in,
            liquid_in,
            humidity_in,
            dry_mass_flow,
            capacity,
            cp,
            thermodynamics.enthalpy(air_in, humidity_in),
            thermodynamics.dewpoint(humidity_in),
            rd,
            rw,
            lambda T: ri,
            thermodynamics.saturation_enthalpy,
            thermodynamics.saturation_temperature,
            thermodynamics.humidity,
        )
        r = _solve_profile_candidate(
            x,
            liquid_enthalpy=thermodynamics.condensate_enthalpy,
            saturation_humidity=thermodynamics.saturation_humidity,
            drain_enabled=drain_enabled,
            quadrature_order=quadrature_order,
            sensible_coordinate=lambda T: air_in
            + (thermodynamics.enthalpy(T, humidity_in) - x.gas_h_in) / cp,
            temperature_from_coordinate=lambda u: thermodynamics.temperature(
                x.gas_h_in + cp * (u - air_in), humidity_in
            ),
            surface_enthalpy=surface_enthalpy,
        )
        correction_error = 0.0
        if finned and r.profile:
            points = r.profile[1:]
            coordinates = [2 * p.coordinate / r.wet_fraction - 1 for p in points]
            correction = []
            surface_states = []
            for point in points:
                response = outside.response(
                    point.surface_temperature,
                    point.gas_temperature,
                    point.humidity,
                    odiag,
                )
                correction.append(
                    response["drain_enthalpy_per_mass"]
                    - thermodynamics.condensate_enthalpy(point.surface_temperature)
                )
                qlocal = (point.surface_temperature - point.liquid_temperature) / ri
                surface_states.append(
                    dict(
                        coordinate=point.coordinate,
                        inside_wall_temperature=point.liquid_temperature
                        + qlocal * idiag["film_resistance"],
                        core_wall_temperature=point.liquid_temperature
                        + qlocal
                        * (idiag["film_resistance"] + idiag["wall_resistance"]),
                        surface_base_temperature=point.surface_temperature,
                        fin_base_temperature=response["fin_base"],
                        fin_tip_temperature=response["fin_tip"],
                        radial_wet_area=response["wet_area"],
                    )
                )
            correction_error = max(
                abs(a - b)
                for a, b in zip(correction, cheb.chebval(coordinates, drain_correction))
            )
            drain_correction = cheb.chebfit(coordinates, correction, len(points) - 1)
            odiag["surface_states"] = tuple(surface_states)
            odiag["radial_drain_enthalpy_iteration_error_J_kg"] = correction_error
            odiag["drain_distribution"] = (
                "source axial removal times radial conditional enthalpy; radial local mass determines weights only"
            )
        previous = r
        nxt = r.air_out if r.regime == "DRY" else r.interface_air
        error = max(
            abs(r.air_out - taout),
            abs(r.liquid_out - twout),
            abs(nxt - tint),
            1e3 * abs(r.humidity_out - Wout),
        )
        taout, twout, tint, Wout = r.air_out, r.liquid_out, nxt, r.humidity_out
        if error < 2e-7 and correction_error < 0.002:
            odiag.setdefault("wet_effective_area", odiag["dry_effective_area"])
            qactual = inside.mass_flow * inside.enthalpy_difference(
                liquid_in, r.liquid_out
            )
            if abs(qactual - r.heat_liquid) > 2e-3:
                raise WetCoilModelError(
                    "liquid provider energy mismatch", residual=qactual - r.heat_liquid
                )
            validate_wet_coil(r)
            if finned and r.profile and drain_enabled:
                nodes, weights = leggauss(min(32, 2 * quadrature_order))
                integral = 0.0
                for node, weight in zip(nodes, weights):
                    point = r._region.point(
                        float((node + 1) * r.wet_fraction / 2),
                        thermodynamics.condensate_enthalpy,
                        False,
                    )
                    physical = outside.response(
                        point.surface_temperature,
                        point.gas_temperature,
                        point.humidity,
                        odiag,
                    )
                    integral += (
                        weight
                        * point.condensate_density
                        * physical["drain_enthalpy_per_mass"]
                        * r.wet_fraction
                        / 2
                    )
                odiag["independent_radial_drain_integral_error_W"] = (
                    integral - r.drain_enthalpy
                )
                if abs(integral - r.drain_enthalpy) > 2e-4:
                    if quadrature_order >= 32:
                        raise WetCoilModelError(
                            "physical radial drain quadrature unresolved",
                            error=integral - r.drain_enthalpy,
                        )
                    return solve_production_coil(
                        bundle=bundle,
                        thermodynamics=thermodynamics,
                        inside=inside,
                        air_in=air_in,
                        liquid_in=liquid_in,
                        humidity_in=humidity_in,
                        dry_mass_flow=dry_mass_flow,
                        drain_enabled=drain_enabled,
                        radial_cells=radial_cells,
                        quadrature_order=min(32, 2 * quadrature_order),
                    )
            return replace(
                r,
                diagnostics=dict(
                    r.diagnostics,
                    **odiag,
                    **idiag,
                    property_iterations=iteration + 1,
                    global_model="ElmahdyMitalas1977/source-profile moisture/drain-coupled",
                    process_source_revision=SOURCE_REVISION,
                    global_flow_assumption="counterflow",
                    configured_flow_arrangement=bundle.flow_arrangement,
                    source_cross_counterflow_row_count_applicable=bundle.n_rows >= 4,
                    gas_property_provider="configured dry carrier CoolProp + water IAPWS-IF97",
                    dry_composition=dict(thermodynamics.capability.dry_mole_fractions),
                    applicability="Counterflow mean-property/secant process approximation; local HTC ranges reported independently.",
                ),
            )
    raise WetCoilModelError("production property iteration did not converge")


@dataclass(frozen=True)
class CircularFinnedTubeAdapter:
    bundle: object
    thermodynamics: WetGasThermodynamics
    radial_cells: int = 64

    def evaluate(self, T, W, md, inside):
        if not isinstance(self.bundle.tube, CircularFinnedTube):
            raise TypeError("CircularFinnedTubeAdapter requires annular geometry")
        provider = wet_gas_provider_at_water_ratio(self.thermodynamics.capability, W)
        props = provider.at(T, self.thermodynamics.pressure)
        outside = evaluate_outside_thermal(
            bundle=self.bundle, m_dot=md * (1 + W), props=to_outside_fluid_props(props)
        )
        ri, idiag = inside
        network = calculate_resistance_network(
            bundle=self.bundle,
            alpha_inside=idiag["htc"],
            outside_alpha_physical=outside.alpha_physical,
            resistance_core_wall=idiag["wall_resistance"],
        )
        tube = self.bundle.tube
        common = (
            (network.resistance_root + network.resistance_contact)
            if tube.D_root > tube.D_o
            else 0.0
        )
        return network.resistance_outside - common, dict(
            geometry_adapter="CircularFinnedTube",
            outside_htc_model="Briggs-Young 1963",
            outside_alpha_physical=outside.alpha_physical,
            dry_effective_area=1
            / (outside.alpha_physical * (network.resistance_outside - common)),
            physical_area=network.area_outside_gross,
            common_root_contact_resistance=common,
            network=network,
            outside_reynolds=outside.reynolds_number,
            outside_warnings=outside.warnings,
        )

    def response(self, Ts, Tg, W, diag):
        """Physical radial heat/mass shape, with exact common/parallel contact paths."""
        from core.heat_transfer.wet_coil_surface import annular_response
        from core.properties.water import water_latent_heat_of_vaporization

        t = self.thermodynamics
        tube = self.bundle.tube
        cap = t.capability
        alpha = diag["outside_alpha_physical"]
        cp = t.secant_cp(Tg, Ts, W)
        count = self.bundle.total_fin_area / tube.fin_area_per_fin
        network = diag["network"]

        def fin(base):
            return annular_response(
                tube,
                base_temperature=base,
                gas_temperature=Tg,
                humidity=W,
                alpha=alpha,
                cp_dry=cp,
                pressure=t.pressure,
                M_dry=cap.M_dry,
                M_water=cap.M_condensable,
                radial_cells=self.radial_cells,
            )

        if tube.D_root == tube.D_o and network.resistance_contact > 0:
            # Contact is in the fin branch only; the exposed tube bypasses it.
            def residual(base):
                r = fin(base)
                return base - Ts - network.resistance_contact * count * r.heat_liquid

            base = brentq(residual, Ts, Tg - 1e-8, xtol=2e-8)
            fr = fin(base)
        else:
            fr = fin(Ts)
        primary = self.bundle.total_primary_outside_area
        mass = alpha / cp * max(W - t.saturation_humidity(Ts), 0.0) * primary
        drain = mass * t.condensate_enthalpy(Ts) + count * fr.drain_enthalpy
        q = (
            alpha * primary * (Tg - Ts)
            + mass * water_latent_heat_of_vaporization(T=Ts)
            + count * fr.heat_liquid
        )
        mass += count * fr.condensate
        return dict(
            heat_liquid=q,
            heat_gas=q + drain,
            local_condensate_response=mass,
            drain_enthalpy_per_mass=(
                drain / mass if mass > 0 else t.condensate_enthalpy(Ts)
            ),
            fin_base=fr.base_temperature,
            fin_tip=fr.tip_temperature,
            wet_area=fr.wet_area * count
            + (primary if W > t.saturation_humidity(Ts) else 0.0),
            radial_energy_residual=fr.energy_residual * count,
        )
