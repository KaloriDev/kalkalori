# SPDX-License-Identifier: GPL-3.0-only
"""One outside-wet forward process, public forward/inverse orchestration.

Rating solves thermal area; Simulation applies thermal reserve scaling.
Both retain installed hydraulic geometry. Equivalent UA follows the solve.
"""
from dataclasses import replace
import logging
from math import exp, isfinite, log

import numpy as np
from scipy.optimize import brentq, least_squares
from numpy.polynomial.legendre import leggauss

from core.geometry.finned_tube import CircularFinnedTube
from core.heat_transfer.wet_coil import WetCoilModelError
from core.heat_transfer.wet_coil_solver import _solve_budget
from core.heat_transfer.wet_coil_adapters import (
    InsideWallAdapter,
    WetGasThermodynamics,
    solve_production_coil,
)
from core.heat_transfer.outside_dispatch import (
    DEFAULT_FINNED_HT_PROVIDER,
    DEFAULT_FINNED_DP_PROVIDER,
    evaluate_outside_hydraulics,
    build_finned_tube_diagnostics,
)
from core.heat_transfer.thermal_iteration import (
    IterativeThermalState,
    ThermalIterationDiagnostics,
    WallTemperatureEnvelope,
    WallTemperatureProbe,
)
from core.heat_transfer.streams import SensibleHeatStream
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.rating_integration import RatingClosureError
from core.phase_change.types import (
    PhaseChangeMode,
    PhaseChangeDirection,
    PhaseChangeResult,
)
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.phase_change.wet_coil_reporting import (
    equivalent_wet_process,
    GLOBAL_WET_MODEL,
    UA_REPORTING_BASIS,
)
from core.properties.adapters import to_internal_fluid_props, to_outside_fluid_props
from core.properties.water import water_saturation_vapor_enthalpy

# Numerical implementation tolerances, independent of model uncertainty.
AREA_ROOT_TOLERANCE = 2e-8


class WetRatingIncompatibilityError(RatingClosureError):
    """No physically admissible member of the declared family realizes the target."""


class WetRatingMultipleSolutionsError(ValueError):
    """A nonunique physical sizing problem requires an explicit user decision."""


def _eligible(inside, outside):
    if outside.phase_change_mode is not PhaseChangeMode.AUTO:
        return False
    if inside.T_in is None or outside.T_in is None or outside.T_in <= inside.T_in:
        return False
    for name in ("water_steam_state", "water_steam_outlet_state"):
        state = getattr(inside, name, None)
        if (
            state is not None
            and getattr(state.phase, "value", state.phase) != "subcooled_liquid"
        ):
            return False
    cap = detect_phase_change_capability(outside.provider)
    if not cap.capable or cap.W_in is None or cap.W_in <= 0:
        return False
    # A wet inside gas still follows the established one-active-side guard.
    icap = detect_phase_change_capability(inside.provider)
    if (
        icap.capable
        and icap.W_in is not None
        and inside.phase_change_mode is PhaseChangeMode.AUTO
    ):
        return False
    from core.phase_change.integration import _dew_point_for

    dew = _dew_point_for(cap, p=outside.p)
    return dew is not None


def forward_wet_process(
    hx,
    inside,
    outside,
    *,
    wet_solver_options=None,
    _budget=None,
    _initial_state=None,
    _thermodynamics=None,
    _area_scale=1.0,
    surface_margin=0.0,
    flow_arrangement=None,
    finned_heat_transfer_provider=DEFAULT_FINNED_HT_PROVIDER,
):
    """Sole physical wet dispatch used by Simulation and every Rating trial."""
    budget = _solve_budget(wet_solver_options, _budget)
    budget.check()
    if not isfinite(surface_margin) or surface_margin < 0:
        raise ValueError("surface_margin must be a non-negative finite value")
    if not isfinite(_area_scale) or _area_scale <= 0:
        raise ValueError("area_scale must be positive and finite")
    if hx.tube_side_enhancement is not None:
        from core.enhancements import EnhancementUnsupportedError

        raise EnhancementUnsupportedError(
            "enhancement_phase_change_unsupported: outside wet-coil adapter"
        )
    flow = flow_arrangement or hx.bundle.flow_arrangement_resolved
    bundle = replace(hx.bundle, flow_arrangement=flow)
    cap = detect_phase_change_capability(outside.provider)
    thermo = _thermodynamics or WetGasThermodynamics(outside.p, cap)
    if thermo.pressure != outside.p or thermo.capability != cap:
        raise ValueError("Wet thermodynamics must match the configured outside state")
    adapter = InsideWallAdapter(
        bundle,
        inside.provider,
        inside.m_dot,
        inside.p,
        thermal_scale=_area_scale / (1 + surface_margin),
    )
    r = solve_production_coil(
        wet_solver_options=budget.options, _budget=budget,
        _initial_state=_initial_state,
        # Keep the grid that resolved a Rating warm start's drain polynomial.
        quadrature_order=(10 if _initial_state is None else
                          _initial_state[0].diagnostics.get("quadrature_order", 10)),
        bundle=bundle,
        thermodynamics=thermo,
        inside=adapter,
        air_in=outside.T_in,
        liquid_in=inside.T_in,
        humidity_in=cap.W_in,
        dry_mass_flow=outside.m_dot / (1 + cap.W_in),
        finned_heat_transfer_provider=finned_heat_transfer_provider,
    )
    return r, thermo, adapter


def _envelope(r, thermo, adapter, inside, outside, alpha):
    """Quadrature of native wet/dry profile and physical radial surface states."""
    from core.heat_transfer.wet_coil import _exprel
    from core.heat_transfer.wet_coil_adapters import CircularFinnedTubeAdapter

    d = r.diagnostics
    ri = d["inner_resistance"]
    film, wall = d["film_resistance"], d["wall_resistance"]
    radial = {p["coordinate"]: p for p in d.get("surface_states", ())}
    finned = isinstance(adapter.bundle.tube, CircularFinnedTube)
    surface = (
        CircularFinnedTubeAdapter(
            adapter.bundle, thermo, thermal_scale=adapter.thermal_scale
        )
        if finned
        else None
    )
    samples = []

    def append(z, tl, tg, ts, W, weight):
        q = (ts - tl) / ri
        ti, tc = tl + q * film, tl + q * (film + wall)
        if min(ti, tc, ts) < tl - 2e-7:
            raise WetCoilModelError(
                "surface below local cold stream",
                coordinate=z,
                liquid_temperature=tl,
                surface_temperature=ts,
            )
        v = radial.get(z)
        if finned and v is None:
            fr = surface.response(ts, tg, W, d)
            fb, ft, mean = (
                fr["fin_base"],
                fr["fin_tip"],
                fr["surface_temperature_area_mean"],
            )
        elif finned:
            fb, ft, mean = (
                v["fin_base_temperature"],
                v["fin_tip_temperature"],
                v["surface_temperature_area_mean"],
            )
        else:
            fb = ft = None
            mean = ts
        samples.append((tl, tg, ti, tc, ts, fb, ft, mean, weight))

    weights = (leggauss(len(r.profile) - 1)[1] * r.wet_fraction / 2
               if r.profile else ())
    for n, p in enumerate(r.profile):
        append(
            p.coordinate,
            p.liquid_temperature,
            p.gas_temperature,
            p.surface_temperature,
            p.humidity,
            0.0 if n == 0 else weights[n - 1],
        )
    f = r.wet_fraction
    if f < 1.0:
        cg, cw = d["dry_gas_capacity"], d["liquid_capacity"]
        conductance = 1 / (ri + d["dry_air_resistance"])
        lam = conductance * (1 / cg - 1 / cw)
        delta = r.interface_air - r.interface_liquid
        nodes, weights = leggauss(8)
        lengths = [0.0, *((nodes + 1) * (1 - f) / 2), 1 - f]
        weights = [0.0, *(weights * (1 - f) / 2), 0.0]
        for length, weight in zip(lengths, weights):
            integral = delta * length * _exprel(lam * length)
            tl = r.interface_liquid + conductance / cw * integral
            tg = r.interface_air + conductance / cg * integral
            ts = tl + (tg - tl) * ri / (ri + d["dry_air_resistance"])
            append(f + length, tl, tg, ts, thermo.capability.W_in, weight)
    else:
        append(
            1.0, r.liquid_out, outside.T_in, r.hot_surface, thermo.capability.W_in, 0.0
        )
    probes = tuple(
        WallTemperatureProbe(
            inside_bulk_temperature=tl,
            outside_bulk_temperature=tg,
            inside_wall_temperature=ti,
            outside_wall_temperature=tc,
            alfa_i=d["htc"],
            alfa_o=alpha,
            converged=True,
            iterations=d["property_iterations"],
            residual=d["property_residual_K"],
            outside_alpha_physical=d["outside_alpha_physical"],
            outside_alpha_effective_gross=alpha,
            outside_primary_surface_temperature=ts,
            fin_base_temperature=fb,
            fin_tip_temperature=ft,
            outside_skin_temperature_min=ts,
            outside_skin_temperature_max=max(ts, ft) if ft is not None else ts,
        )
        for tl, tg, ti, tc, ts, fb, ft, mean, weight in samples
    )
    fins = [p for p in probes if p.fin_base_temperature is not None]
    envelope = WallTemperatureEnvelope(
        min(p.inside_wall_temperature for p in probes),
        max(p.inside_wall_temperature for p in probes),
        min(p.outside_wall_temperature for p in probes),
        max(p.outside_wall_temperature for p in probes),
        probes,
        method="elmahdy_mitalas_source_profile",
        inside_mean=sum(p[2] * p[8] for p in samples),
        outside_mean=sum(p[3] * p[8] for p in samples),
        outside_skin_min=min(p.outside_skin_temperature_min for p in probes),
        outside_skin_max=max(p.outside_skin_temperature_max for p in probes),
        fin_base_min=min((p.fin_base_temperature for p in fins), default=float("nan")),
        fin_base_max=max((p.fin_base_temperature for p in fins), default=float("nan")),
        fin_tip_min=min((p.fin_tip_temperature for p in fins), default=float("nan")),
        fin_tip_max=max((p.fin_tip_temperature for p in fins), default=float("nan")),
    )
    return envelope, sum(p[7] * p[8] for p in samples)


def _public_simulation(
    hx,
    inside,
    outside,
    physical,
    *,
    _area_scale=1.0,
    surface_margin=0.0,
    flow_arrangement=None,
    euler_provider="zukauskas",
    finned_heat_transfer_provider=DEFAULT_FINNED_HT_PROVIDER,
    finned_pressure_drop_provider=DEFAULT_FINNED_DP_PROVIDER,
    Q_full=None,
    K_inlet=0.5,
    K_outlet=1.0,
    K_turn=1.5,
):
    """Map converged physics and existing physical-geometry hydraulics."""
    from core.models.simulation import HXSimulationResult
    from core.phase_change.integration import capability_only_result
    from core.pressure_drop.flow_path import build_outside_pressure_drop_result

    r, thermo, adapter = physical
    d = dict(r.diagnostics)
    cap = thermo.capability
    md, wi, wo = outside.m_dot / (1 + cap.W_in), cap.W_in, r.humidity_out
    wm = (wi + wo) / 2
    ti, to = (inside.T_in + r.liquid_out) / 2, (outside.T_in + r.air_out) / 2
    op = wet_gas_provider_at_water_ratio(cap, wm)
    iprops, oprops = inside.provider.at(ti, inside.p), op.at(to, outside.p)
    flow = flow_arrangement or hx.bundle.flow_arrangement_resolved
    eq = equivalent_wet_process(
        heat=r.heat_liquid,
        hot_in=outside.T_in,
        hot_out=r.air_out,
        cold_in=inside.T_in,
        cold_out=r.liquid_out,
        flow_arrangement=flow,
    )
    area = hx.bundle.total_outer_area
    thermal_area = area / (1 + surface_margin) * _area_scale
    u, ua = eq.ua / thermal_area, eq.ua * (1 + surface_margin) / _area_scale
    # Generic outside HTC includes root/contact on its historical gross-area
    # basis; the process keeps common root/contact in the inside operator.
    alpha = 1 / ((d["dry_air_resistance"]
                  + d.get("common_root_contact_resistance", 0.0)) * thermal_area)
    # A snapshot after the wet solve supplies existing hydraulic models only.
    # Its independent dry heat calculation never feeds the accepted process.
    snapshot = hx.solve(
        SensibleHeatStream(eq.hot_capacity, outside.T_in),
        SensibleHeatStream(eq.cold_capacity, inside.T_in),
        K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
        m_dot_tube_side=inside.m_dot,
        tube_side_props=to_internal_fluid_props(iprops),
        tube_side_provider=inside.provider,
        tube_side_temperature_in=inside.T_in,
        tube_side_temperature_out=r.liquid_out,
        tube_side_pressure=inside.p,
        m_dot_outside=md * (1 + wm),
        outside_props=to_outside_fluid_props(oprops),
        outside_provider=op,
        outside_temperature_in=outside.T_in,
        outside_temperature_out=r.air_out,
        outside_pressure=outside.p,
        flow_arrangement=flow,
        euler_provider=euler_provider,
        finned_heat_transfer_provider=finned_heat_transfer_provider,
        finned_pressure_drop_provider=finned_pressure_drop_provider,
    )
    bank = evaluate_outside_hydraulics(
        bundle=hx.bundle,
        m_dot=md * (1 + wm),
        inlet_props=outside.provider.at(outside.T_in, outside.p),
        midpoint_props=oprops,
        outlet_props=wet_gas_provider_at_water_ratio(cap, wo).at(r.air_out, outside.p),
        temperature_in=outside.T_in,
        temperature_out=r.air_out,
        pressure=outside.p,
        euler_provider=euler_provider,
        finned_pressure_drop_provider=finned_pressure_drop_provider,
        m_dot_inlet=outside.m_dot,
        m_dot_midpoint=md * (1 + wm),
        m_dot_outlet=md * (1 + wo),
    )
    bank = replace(bank, midpoint_method="arithmetic_temperature_and_water_ratio")
    # Only hydraulics are retained from the legacy snapshot. Thermal
    # diagnostics come from the coefficients actually used by the wet solve.
    oc = d["outside_correlation"]
    ic = d["inside_correlation"]
    fins = (
        build_finned_tube_diagnostics(network=d["network"], thermal=oc, hydraulic=bank)
        if "network" in d else None
    )
    process_warnings = list(snapshot.warnings or ())
    wet_warnings = []
    if fins is not None and r.regime != "DRY":
        from core.common.warnings import make_warning
        from core.phase_change.warning_codes import CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY
        warning = make_warning(
            code=CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY,
            message="Circular-finned pressure drop uses the existing dry-bank model at the solved wet-gas states; condensate-film drag is not modeled.",
            source="wet_coil_integration",
        )
        wet_warnings.append(warning)
        process_warnings.append(warning)
    if fins is not None:
        fins = replace(
            fins,
            thermal_reporting_basis="native_dry_constitutive_network",
            outside_dp_dry_reference=bank.dp_total,
            wet_pressure_drop_supported=False,
            outside_dp_reference_only=r.regime != "DRY",
            warnings=tuple(fins.warnings) + tuple(wet_warnings),
            outside_dp_drag=bank.dp_drag,
            outside_dp_acceleration=bank.dp_acceleration,
            outside_dp_total=bank.dp_total,
        )
    snapshot = replace(
        snapshot,
        UA=ua,
        eps=eq.effectiveness,
        Q=r.heat_liquid,
        T_hot_out=r.air_out,
        T_cold_out=r.liquid_out,
        tube_side_thermal=replace(
            snapshot.tube_side_thermal, v=ic.v, Re=ic.Re, Pr=ic.Pr,
            alfa=ic.alfa_corrected,
        ),
        outside_side_thermal=replace(
            snapshot.outside_side_thermal, v=oc.face_velocity,
            Re=oc.reynolds_number, Pr=oc.prandtl_number,
            alfa=alpha,
        ),
        outside_side_hydraulic=replace(
            snapshot.outside_side_hydraulic,
            dp_total=bank.dp_total,
            Re=bank.midpoint.reynolds,
            v=bank.midpoint.face_velocity,
            tube_bank=bank,
        ),
        outside_side_pressure_drop=replace(
            snapshot.outside_side_pressure_drop,
            tube_bank=bank,
            flow_path=build_outside_pressure_drop_result(bank),
        ),
        finned_tube_diagnostics=fins,
        warnings=process_warnings,
    )
    envelope, surface_mean = _envelope(r, thermo, adapter, inside, outside, alpha)
    diagnostics = ThermalIterationDiagnostics(
        ic.Nu_base,
        ic.Nu_corrected,
        ic.length_correction,
        ic.wall_temperature_correction,
        ic.combined_correction,
        ic.alfa_base,
        ic.alfa_corrected,
        oc.nusselt_number_base,
        oc.nusselt_number,
        oc.wall_property_correction,
    )
    state = IterativeThermalState(
        ti,
        to,
        envelope.inside_mean,
        envelope.outside_mean,
        iprops,
        None,
        oprops,
        None,
        d["htc"],
        alpha,
        u,
        ua,
        d["property_iterations"],
        True,
        d["property_residual_K"],
        diagnostics,
        outside_alpha_physical=d["outside_alpha_physical"],
        outside_alpha_effective_gross=alpha,
        inside_provider_name=type(inside.provider).__name__,
        outside_provider_name=type(outside.provider).__name__,
        finned_tube_diagnostics=fins,
        ua_reporting_basis=UA_REPORTING_BASIS,
        ua_is_equivalent=True,
    )
    mass_error = md * wi - md * wo - r.condensate
    gas_heat = md * (thermo.enthalpy(outside.T_in, wi) - thermo.enthalpy(r.air_out, wo))
    liquid_heat = inside.m_dot * adapter.enthalpy_difference(inside.T_in, r.liquid_out)
    energy_error = gas_heat - liquid_heat - r.drain_enthalpy
    # Partition the accepted liquid duty on the fixed-inlet-W cooling path.
    # Preserve the endpoint moisture heat minus drain, including its vanishing
    # onset limit; obtain sensible heat as the remaining liquid duty. The
    # independent gas sensible term and finite closure residual stay visible.
    # No process state, total duty, moisture or drain is adjusted.
    sensible = md * (thermo.enthalpy(outside.T_in, wi) - thermo.enthalpy(r.air_out, wi))
    endpoint_latent = (
        md * (wi - wo) * water_saturation_vapor_enthalpy(T=r.air_out) - r.drain_enthalpy
    )
    split_residual = sensible + endpoint_latent - r.heat_liquid
    endpoint_sensible = sensible
    latent = endpoint_latent
    sensible = r.heat_liquid - latent
    d.update(
        global_wet_model=GLOBAL_WET_MODEL,
        ua_reporting_basis=UA_REPORTING_BASIS,
        ua_is_equivalent=True,
        ua_process_eq=eq.ua,
        equivalent_process=eq,
        physical_installed_outside_area=area,
        thermal_outside_area=thermal_area,
        thermal_inside_area=hx.bundle.total_inner_area / (1 + surface_margin) * _area_scale,
        hydraulic_effective_length=hx.bundle.tube.length_effective,
        hydraulic_total_length=hx.bundle.tube.length_total,
        hydraulic_inner_flow_area=hx.bundle.internal_flow_area_per_pass,
        hydraulic_frontal_area=hx.bundle.frontal_flow_area,
        thermal_state_basis="native_profile_walls_and_correlations_with_equivalent_process_UA",
        finned_thermal_reporting_basis=(None if fins is None else fins.thermal_reporting_basis),
        independent_mass_residual=mass_error,
        independent_energy_residual=energy_error,
        endpoint_sensible_heat=endpoint_sensible,
        endpoint_latent_heat=endpoint_latent,
        sensible_latent_closure_residual=split_residual,
        Q_gas=gas_heat,
        Q_process=r.heat_liquid,
        H_drain=r.drain_enthalpy,
        process_profile=r.profile,
        regime=r.regime,
        wet_fraction=r.wet_fraction,
        sensible_latent_basis="fixed_inlet_W_cooling_then_isothermal_moisture_removal_minus_drain",
    )
    axial_weights = (leggauss(len(r.profile) - 1)[1] * r.wet_fraction / 2
                     if r.profile else ())
    if r.regime == "DRY":
        wet_area, wet_mean = 0.0, None
    elif "surface_states" in d:
        wet_area = sum(
            w * p["radial_wet_area"] for w, p in zip(axial_weights, d["surface_states"])
        )
        wet_mean = (
            sum(
                w * p["wet_temperature_area_integral"]
                for w, p in zip(axial_weights, d["surface_states"])
            )
            / wet_area
        )
    else:
        wet_area = thermal_area * r.wet_fraction
        wet_mean = (
            float(np.dot(axial_weights, [p.surface_temperature for p in r.profile[1:]]))
            / r.wet_fraction
        )
    d["physical_wet_area"] = wet_area
    d["physical_wet_surface_fraction"] = wet_area / thermal_area
    # Generic reporting aliases of the accepted native solution, in K.
    # Keep the gas/interface surface and the underlying metal wall distinct.
    d["surface"] = dict(
        surface_temperature_min=envelope.outside_skin_min,
        surface_temperature_max=envelope.outside_skin_max,
        surface_temperature_wet_mean=wet_mean,
        dew_point_in=thermo.dewpoint(wi),
        dew_point_out=thermo.dewpoint(wo),
        onset_margin=-r.onset_margin,  # dew point minus dry cold surface [K]
        wet_regime=r.regime,
        axial_wet_fraction=r.wet_fraction,
        metal_wall_temperature_min=envelope.outside_min,
        metal_wall_temperature_max=envelope.outside_max,
        metal_wall_temperature_mean=envelope.outside_mean,
        interface_temperature=tuple(p.outside_primary_surface_temperature
                                    for p in envelope.probes),
        metal_wall_temperature=tuple(p.outside_wall_temperature for p in envelope.probes),
    )
    if fins is not None:
        d["surface"].update(
            fin_base_temperature=tuple(p.fin_base_temperature for p in envelope.probes),
            fin_tip_temperature=tuple(p.fin_tip_temperature for p in envelope.probes),
        )
        if "surface_states" in d:
            d["surface"]["radial_wet_fraction"] = tuple(
                dict(coordinate=p["coordinate"], fraction=p["radial_wet_area"] / thermal_area)
                for p in d["surface_states"]
            )
    pc = PhaseChangeResult(
        side="outside",
        mode=outside.phase_change_mode,
        direction=(PhaseChangeDirection.NONE if r.regime == "DRY"
                   else PhaseChangeDirection.CONDENSATION),
        component="H2O",
        capable=True,
        possible=bool(r.onset_margin < 0),
        active=r.regime != "DRY",
        iterations=d["property_iterations"],
        method=GLOBAL_WET_MODEL,
        onset_margin_K=-r.onset_margin,
        onset_wall_temperature=thermo.dewpoint(wi) + r.onset_margin,
        onset_temperature_method="elmahdy_mitalas_dry_cold_surface",
        W_in=wi,
        W_mid=wm,
        W_out=wo,
        m_dot_dry_carrier=md,
        m_dot_water_vapor_in=md * wi,
        m_dot_water_vapor_out=md * wo,
        m_dot_condensate=r.condensate,
        m_dot_gas_in=outside.m_dot,
        m_dot_gas_out=md * (1 + wo),
        dew_point_in=thermo.dewpoint(wi),
        dew_point_out=thermo.dewpoint(wo),
        wall_temperature_mean=surface_mean,
        wall_temperature_min=envelope.outside_skin_min,
        wall_temperature_max=envelope.outside_skin_max,
        wall_temperature_wet_mean=wet_mean,
        W_sat_wet_surface=(None if wet_mean is None
                           else thermo.saturation_humidity(wet_mean)),
        wet_surface_fraction=wet_area / thermal_area,
        wet_surface_fraction_method="elmahdy_mitalas_source_profile",
        wet_area=wet_area,
        outside_total_area=thermal_area,
        alfa_dry=d["outside_alpha_physical"],
        alfa_effective=alpha,
        Q_sensible=sensible,
        Q_latent=latent,
        Q_total=r.heat_liquid,
        mass_balance_error=mass_error,
        energy_balance_error=energy_error,
        residuals={
            key: float(d[key]) for key in (
                "energy_residual", "mass_residual", "property_residual_K",
                "drain_quadrature_error", "mass_quadrature_error",
                "independent_radial_drain_integral_error_W",
            ) if key in d
        },
        warnings=tuple(wet_warnings),
        wet_coil_diagnostics=d,
        assumptions=(
            "source_profile_moisture",
            "fully_drained_liquid_at_local_surface",
            "counterflow_mean_property_process",
            "wet_equivalent_UA_reporting_only",
        ),
    )
    ipc = capability_only_result(
        "inside",
        inside.phase_change_mode,
        detect_phase_change_capability(inside.provider),
        r.heat_liquid,
    )
    return HXSimulationResult(
        converged=True,
        iterations=d["property_iterations"],
        residual_q_rel=abs(energy_error) / r.heat_liquid,
        residual_T_inside_K=d["property_change_liquid_out_K"],
        residual_T_outside_K=d["property_change_air_out_K"],
        T_mean_inside=ti,
        T_mean_outside=to,
        inside_props_mean=iprops,
        outside_props_mean=oprops,
        inside_velocity_mean=snapshot.tube_side_thermal.v,
        outside_velocity_mean=snapshot.outside_side_thermal.v,
        inside_Re_mean=snapshot.tube_side_thermal.Re,
        outside_Re_mean=snapshot.outside_side_thermal.Re,
        inside_Pr_mean=snapshot.tube_side_thermal.Pr,
        outside_Pr_mean=snapshot.outside_side_thermal.Pr,
        inside_alfa_mean=d["htc"],
        outside_alfa_mean=alpha,
        U_mean=u,
        UA=ua,
        EMTD=eq.emtd,
        q=r.heat_liquid,
        T_out_inside=r.liquid_out,
        T_out_outside=r.air_out,
        surface_margin=surface_margin,
        overdesign_factor=ua / eq.ua - 1,
        Q_full=r.heat_liquid if Q_full is None else Q_full,
        Q_derated=r.heat_liquid,
        final_result=snapshot,
        thermal_state=state,
        wall_temperature_envelope=envelope,
        warnings=snapshot.warnings,
        inside_phase_change=ipc,
        outside_phase_change=pc,
        ua_reporting_basis=UA_REPORTING_BASIS,
        ua_is_equivalent=True,
        wet_coil_diagnostics=d,
    )


def route_outside_wet(hx, inside, outside, *, mode, settings, **options):
    """Global model selection shared by the public and deferred wet paths."""
    from core.phase_change.wet_coil_provider import dispatch_wet_coil
    return dispatch_wet_coil(hx, inside, outside, mode=mode, settings=settings, **options)


def _run_elmahdy(hx, inside, outside, *, mode, settings, _budget=None, **options):
    """Existing Elmahdy orchestration, invoked by its thin provider adapter."""
    if settings.lewis_number != 1.0:
        raise ValueError(
            "Elmahdy-Mitalas source-profile closure requires Lewis number 1"
        )
    budget = _solve_budget(options.pop("wet_solver_options", None), _budget)
    budget.used = True
    budget.check()
    if mode == "rating":
        return _rate(hx, inside, outside, settings=settings, wet_solver_options=budget.options, _budget=budget, **options)
    margin = options.get("surface_margin", 0.0)
    physical = forward_wet_process(
        hx,
        inside,
        outside,
        wet_solver_options=budget.options, _budget=budget,
        surface_margin=margin,
        flow_arrangement=options.get("flow_arrangement"),
        finned_heat_transfer_provider=options.get(
            "finned_heat_transfer_provider", DEFAULT_FINNED_HT_PROVIDER
        ),
    )
    if not options.pop("iterate", True):
        raise ValueError("Active outside wet Simulation requires iterate=True")
    full = None
    if margin:
        full = forward_wet_process(
            hx,
            inside,
            outside,
            wet_solver_options=budget.options, _budget=budget,
            flow_arrangement=options.get("flow_arrangement"),
            finned_heat_transfer_provider=options.get(
                "finned_heat_transfer_provider", DEFAULT_FINNED_HT_PROVIDER
            ),
        )[0].heat_liquid
    return _public_simulation(hx, inside, outside, physical, Q_full=full, **options)


def _rate(
    hx,
    inside,
    outside,
    *,
    settings,
    wet_solver_options=None,
    _budget=None,
    Q=None,
    effectiveness=None,
    include_simulation=False,
    over_specified_tolerance=1e-3,
    **options,
):
    from core.models.rating import HXRatingResult
    from core.models.heat_balance import ClosedBalance, ClosedBalanceSide

    budget = _solve_budget(wet_solver_options, _budget)
    controls = budget.options
    outlet_tolerance = controls.outlet_temperature_tolerance_K
    if outside.T_out is None:
        raise WetRatingIncompatibilityError(
            "Active outside-wet Rating requires outside.T_out; installed geometry cannot close an unspecified process"
        )
    if outside.m_dot is None or outside.m_dot <= 0:
        raise WetRatingIncompatibilityError(
            "Active outside-wet Rating requires outside.m_dot"
        )
    if inside.m_dot is None and inside.T_out is None:
        raise WetRatingIncompatibilityError(
            "Specify inside.m_dot or inside.T_out for wet Rating"
        )
    if not inside.T_in < outside.T_out < outside.T_in:
        raise WetRatingIncompatibilityError(
            "Outside outlet target must lie between the two inlet temperatures"
        )
    base = 1.0
    cache = {}
    thermodynamics = WetGasThermodynamics(
        outside.p, detect_phase_change_capability(outside.provider),
        _reuse_inverse_state=True,
    )

    def trial(area_scale, mass):
        budget.check(required_area_scale=area_scale, inside_mass_flow=mass)
        key = (float(area_scale), float(mass))
        if key not in cache:
            from core.models.simulation import HXSideInput

            side = HXSideInput(
                inside.provider,
                mass,
                inside.T_in,
                inside.p,
                phase_change_mode=inside.phase_change_mode,
            )
            log_trial = logging.getLogger(__name__)
            log_trial.debug("Rating trial area_scale=%.12g m_inside=%.12g", area_scale, mass)
            # Reuse only an initial property/drain iterate from the nearest
            # accepted area/flow trial. No result or physical gate is reused.
            initial = None
            if cache:
                nearest = min(cache, key=lambda k: abs(log(area_scale / k[0])) + abs(log(mass / k[1])))
                previous_result = cache[nearest][1][0]
                initial = (previous_result, np.asarray(previous_result.diagnostics[
                    "radial_drain_correction_coefficients"]))
            try:
                result = forward_wet_process(
                    hx,
                    side,
                    outside,
                    wet_solver_options=controls, _budget=budget,
                    _initial_state=initial,
                    _thermodynamics=thermodynamics,
                    _area_scale=area_scale,
                    flow_arrangement=options.get("flow_arrangement"),
                    finned_heat_transfer_provider=options.get(
                        "finned_heat_transfer_provider", DEFAULT_FINNED_HT_PROVIDER
                    ),
                )
            except ValueError as exc:
                log_trial.debug(
                    "Rating trial rejected: %s; %s", exc,
                    getattr(exc, "diagnostics", {}),
                )
                raise
            log_trial.debug(
                "Rating trial accepted regime=%s Tout=%.12g", result[0].regime,
                result[0].air_out,
            )
            budget.check(last_regime=result[0].regime, wet_fraction=result[0].wet_fraction,
                         outside_outlet_residual_K=result[0].air_out - outside.T_out,
                         inside_outlet_residual_K=(None if inside.T_out is None else result[0].liquid_out - inside.T_out))
            cache[key] = (side, result)
        return cache[key]

    if inside.m_dot is not None:
        mass = inside.m_dot

        def residual(z):
            return trial(exp(z), mass)[1][0].air_out - outside.T_out

        origin = log(base)
        try:
            value = residual(origin)
        except (WetCoilModelError, OverflowError, ValueError) as initial_error:
            # Installed geometry need not itself be an admissible wet process.
            # Locate a valid trial in the declared family before bracketing.
            found = False
            for step in range(1, 27):
                for direction in (-1, 1):
                    candidate = log(base) + direction * step * log(2)
                    try:
                        value = residual(candidate)
                    except (WetCoilModelError, OverflowError, ValueError):
                        continue
                    origin, found = candidate, True
                    break
                if found:
                    break
            if not found:
                raise WetRatingIncompatibilityError(
                    "No admissible required-area trial at installed geometry"
                ) from initial_error
        if abs(value) <= outlet_tolerance:
            area_scale = exp(origin)
        else:
            direction = 1 if value > 0 else -1
            previous = (origin, value)
            bracket = None
            for n in range(1, 27):
                z = origin + direction * n * log(2)
                try:
                    v = residual(z)
                except (WetCoilModelError, OverflowError, ValueError) as exc:
                    # Resolve the admissible boundary before declaring the target impossible.
                    invalid = z
                    for _ in range(18):
                        z = (previous[0] + invalid) / 2
                        try:
                            v = residual(z)
                        except (WetCoilModelError, OverflowError, ValueError):
                            invalid = z
                            continue
                        if v * previous[1] <= 0:
                            bracket = sorted((previous[0], z))
                            break
                        previous = (z, v)
                    if bracket is None:
                        raise WetRatingIncompatibilityError(
                            "Required-area target lies beyond the admissible forward-model domain"
                        ) from exc
                    break
                if v * previous[1] <= 0:
                    bracket = sorted((previous[0], z))
                    break
                # A local nonmonotone step alone does not prove multiple
                # physical roots. Continue constructing a sign-changing bracket.
                previous = (z, v)
            if bracket is None:
                raise WetRatingIncompatibilityError(
                    "Cannot bracket a required thermal area"
                )
            class AreaTargetMet(Exception):
                def __init__(self, z):
                    self.z = z

            def root_residual(z):
                value = residual(z)
                if abs(value) <= outlet_tolerance:
                    raise AreaTargetMet(z)
                return value

            try:
                area_scale = exp(brentq(root_residual, *bracket,
                                    xtol=AREA_ROOT_TOLERANCE, rtol=1e-12))
            except AreaTargetMet as solved:
                area_scale = exp(solved.z)
    else:
        if not inside.T_in < inside.T_out < outside.T_in:
            raise WetRatingIncompatibilityError(
                "Inside outlet target must lie between the inlet temperatures"
            )
        cap = detect_phase_change_capability(outside.provider)
        t = WetGasThermodynamics(outside.p, cap)
        sensible = (
            outside.m_dot
            / (1 + cap.W_in)
            * (t.enthalpy(outside.T_in, cap.W_in) - t.enthalpy(outside.T_out, cap.W_in))
        )
        mass0 = sensible / (
            inside.provider.at((inside.T_in + inside.T_out) / 2, inside.p).cp
            * (inside.T_out - inside.T_in)
        )

        class TargetsMet(Exception):
            def __init__(self, z):
                self.z = z

        def residuals(z):
            budget.count("joint_solver_evaluations")
            r = trial(exp(z[0]), exp(z[1]))[1][0]
            values = [r.air_out - outside.T_out, r.liquid_out - inside.T_out]
            if max(abs(v) for v in values) <= outlet_tolerance:
                raise TargetsMet(z)
            return values

        lower_bounds = np.array([log(base) - 10, log(mass0) - 10])
        upper_bounds = np.array([log(base) + 10, log(mass0) + 10])
        jacobian_state = None

        def jacobian(z):
            nonlocal jacobian_state
            base_values = np.asarray(residuals(z))
            if jacobian_state is not None:
                previous_z, previous_values, matrix = jacobian_state
                delta = z - previous_z
                norm_squared = float(np.dot(delta, delta))
                if norm_squared > 0:
                    # Broyden's secant update reuses accepted outlet responses.
                    # The bounded optimizer and physical acceptance gates are
                    # unchanged; avoid two new forward solves per Jacobian.
                    matrix = matrix + np.outer(
                        base_values - previous_values - matrix @ delta, delta
                    ) / norm_squared
                jacobian_state = (np.array(z, copy=True), base_values, matrix)
                return matrix
            # A relative step collapses near log(area_scale)=0 (installed area),
            # measuring fixed-point noise instead of the physical slope.
            # Use the existing 1e-4 scale as an absolute log-space step.
            columns = []
            for axis in range(2):
                shifted = np.array(z, copy=True)
                step = 1e-4 if z[axis] + 1e-4 <= upper_bounds[axis] else -1e-4
                shifted[axis] += step
                columns.append((np.asarray(residuals(shifted)) - base_values) / step)
            matrix = np.column_stack(columns)
            jacobian_state = (np.array(z, copy=True), base_values, matrix)
            return matrix

        try:
            fit = least_squares(
                residuals,
                [log(base), log(mass0)],
                bounds=(lower_bounds, upper_bounds),
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                jac=jacobian,
                max_nfev=80,
            )
            if max(abs(v) for v in fit.fun) > outlet_tolerance:
                raise WetRatingIncompatibilityError(
                    "Bounded area-scale/mass-flow solve did not reproduce both outlet targets"
                )
            area_scale, mass = exp(fit.x[0]), exp(fit.x[1])
        except TargetsMet as solved:
            area_scale, mass = exp(solved.z[0]), exp(solved.z[1])
    side, physical = trial(area_scale, mass)
    r = physical[0]
    if abs(r.air_out - outside.T_out) > outlet_tolerance:
        raise WetRatingIncompatibilityError(
            "Required-area outlet residual exceeds numerical tolerance"
        )
    if (
        inside.T_out is not None
        and abs(r.liquid_out - inside.T_out) > outlet_tolerance
    ):
        raise WetRatingIncompatibilityError(
            "Fully specified outlet temperatures and mass flows are incompatible with the forward wet model"
        )
    if Q is not None and abs(Q - r.heat_liquid) > max(controls.energy_tolerance_W, abs(Q) * 1e-7):
        raise WetRatingIncompatibilityError(
            "Specified duty is incompatible with the physical outlet solution"
        )
    forward = _public_simulation(
        hx, side, outside, physical, _area_scale=area_scale, **options
    )
    eq = forward.wet_coil_diagnostics["equivalent_process"]
    if effectiveness is not None and abs(effectiveness - eq.effectiveness) > 1e-7:
        raise WetRatingIncompatibilityError(
            "Specified effectiveness is incompatible with the solved wet process"
        )
    area = hx.bundle.total_outer_area
    required = area * area_scale
    u = eq.ua / required
    actual = u * area
    margin = area / required - 1
    d = dict(
        forward.wet_coil_diagnostics,
        required_area_scale=area_scale,
        required_thermal_outside_area=required,
        installed_physical_outside_area=area,
        physical_surface_overdesign=margin,
        required_inside_mass_flow=mass,
        rating_forward_evaluations=len(cache),
        solver_statistics=dict(budget.diagnostics),
        required_area_temperature_residual=r.air_out - outside.T_out,
    )
    cb = ClosedBalance(
        ClosedBalanceSide(
            inside.provider,
            inside.p,
            mass,
            inside.T_in,
            r.liquid_out,
            eq.cold_capacity / mass,
            eq.cold_capacity,
        ),
        ClosedBalanceSide(
            outside.provider,
            outside.p,
            outside.m_dot,
            outside.T_in,
            r.air_out,
            eq.hot_capacity / outside.m_dot,
            eq.hot_capacity,
        ),
        False,
        r.heat_liquid,
        r.heat_liquid / eq.effectiveness,
        eq.effectiveness,
    )
    state = replace(forward.thermal_state, U=u, UA=actual)
    simulation = None
    if include_simulation:
        from core.models.simulation import HXSideInput
        installed = cache.get((1.0, mass))
        if installed is not None:
            # This is already the validated full-installed-area process with
            # the same inlet states, flow and controls. Only map its report.
            from core.heat_transfer import wet_coil_solver

            simulation = _public_simulation(hx, installed[0], outside, installed[1], **options)
            simulation.wet_coil_diagnostics["solver_statistics"] = dict(
                budget.diagnostics,
                elapsed_s=wet_coil_solver.monotonic() - budget.started,
            )
        else:
            simulation = hx.simulate(
                HXSideInput(
                    inside.provider, mass, inside.T_in, inside.p,
                    phase_change_mode=inside.phase_change_mode,
                ),
                HXSideInput(
                    outside.provider, outside.m_dot, outside.T_in, outside.p,
                    phase_change_mode=outside.phase_change_mode,
                ),
                wet_solver_options=controls,
                **options,
            )
    budget.check()
    d["solver_statistics"] = dict(budget.diagnostics)
    return HXRatingResult(
        overdesign_factor=margin,
        ua_margin=margin,
        A_o=area,
        A_required=required,
        UA_required=eq.ua,
        UA_actual=actual,
        U_mean=u,
        EMTD=eq.emtd,
        alfa_i=state.alfa_i,
        alfa_o=state.alfa_o,
        Q_required=r.heat_liquid,
        Q_achievable=None if simulation is None else simulation.q,
        closed_balance=cb,
        final_result=forward.final_result,
        simulation=simulation,
        thermal_state=state,
        wall_temperature_envelope=forward.wall_temperature_envelope,
        warnings=forward.warnings,
        inside_phase_change=forward.inside_phase_change,
        outside_phase_change=replace(
            forward.outside_phase_change, wet_coil_diagnostics=d
        ),
        ua_reporting_basis=UA_REPORTING_BASIS,
        ua_is_equivalent=True,
        wet_coil_diagnostics=d,
    )
