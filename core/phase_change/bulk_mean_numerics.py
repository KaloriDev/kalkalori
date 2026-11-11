# SPDX-License-Identifier: GPL-3.0-only
"""Bounded residual solve of the existing Bulk-Mean physical equations."""
import numpy as np
from scipy.optimize import least_squares

from core.phase_change.capability import pure_water_saturation_temperature
from core.phase_change.water_equilibrium import saturated_water_ratio
from core.phase_change.wet_gas_enthalpy import WetGasEnthalpyError, WetGasEnthalpyEvaluator
from core.properties.water import WATER_TRIPLE_POINT_TEMPERATURE_K, water_latent_heat_of_vaporization


def solve_bounded_outlets(hx, options, evaluate, initial):
    cap = options["outside_capability"]
    md, mi = options["m_dot_dry_carrier"], options["m_dot_inside"]
    cold, hot = options["T_in_inside"], options["T_in_outside"]
    p = options["p_outside"]
    # Pressure/composition are invariant across residual trials. Reuse exact
    # component snapshots rather than rebuilding CoolProp states per probe.
    options = dict(options, _enthalpy_evaluator=WetGasEnthalpyEvaluator(
        p,cap,reuse_dry_backend_state=True))
    coolant_upper = hot
    sat = pure_water_saturation_temperature(options["inside_provider"], p=options["p_inside"])
    if sat is not None and cold < sat:
        coolant_upper = min(coolant_upper, sat-1e-6)
    def water_limit(T):
        # At/above the pure-water boiling temperature no finite saturation
        # ratio limits the available vapor. Preserve the inlet-water bound.
        from core.properties.water import water_saturation_pressure
        if water_saturation_pressure(T) >= p:
            return cap.W_in
        return min(cap.W_in, saturated_water_ratio(p_total=p,T=T,M_dry=cap.M_dry))
    cp_scale = cap.W_in * md * water_latent_heat_of_vaporization(T=WATER_TRIPLE_POINT_TEMPERATURE_K)
    # State variables: gas temperature, coolant temperature, fraction of the
    # admissible vapor inventory, and latent source divided by its upper bound.
    def decode(x):
        return (float(x[1]),float(x[0]),float(x[2]*water_limit(x[0])),float(x[3]*cp_scale),
                float(x[4]),float(x[5]))
    if initial is None:
        initial = (options["T_out_inside_init"], options["T_out_outside_init"],cap.W_in,0.,
                   (cold+options["T_out_inside_init"])/2,
                   (cold+options["T_out_inside_init"])/2)
    tg = min(hot-1e-7,max(cold+1e-7,initial[1]))
    x0 = [tg,min(coolant_upper-1e-7,max(cold+1e-7,initial[0])),
          min(1-1e-7,max(1e-7,initial[2]/water_limit(tg))),
          min(1-1e-7,max(1e-7,initial[3]/cp_scale)),
          min(coolant_upper-1e-7,max(cold+1e-7,initial[4])),
          min(hot-1e-7,max(cold+1e-7,initial[5]))]
    count = 0
    def residual(x):
        nonlocal count
        count += 1
        if options.get("check") is not None:
            options["check"]()
        r = evaluate(hx, **options, _trial_state=decode(x))
        return np.array([r["gas_residual"]/(md*r["cp_gas"]),
            r["coolant_residual"]/(mi*r["cp_other"]),
            r["mass_residual"]/(md*cap.W_in)*100.,
            r["latent_residual"]/(md*r["cp_gas"]),
            r["inside_wall_residual"],r["outside_wall_residual"]])
    fit = least_squares(residual, x0, bounds=([cold,cold,0.,0.,cold,cold],
        [hot,coolant_upper,1.,1.,coolant_upper,hot]), x_scale="jac", diff_step=1e-5,
        ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=options.get("max_iterations",50))
    state = decode(fit.x)
    r = evaluate(hx, **options, _trial_state=state)
    mass_tol = options.get("condensate_tolerance_kg_s",1e-8)
    qtol = options.get("relative_Q_tolerance",1e-4)
    wall_tol = options.get("wall_temperature_tolerance_K",.05)
    # Use stricter physical closure gates than the historical successive-step
    # test, without changing any configured convergence tolerance.
    energy_tol = min(qtol*max(abs(r["Q_total"]),1.), .01)
    source_tol = min(qtol*max(abs(r["Q_latent"]),1.),.01)
    if (abs(r["mass_residual"]) > mass_tol
        or abs(r["gas_residual"]) > energy_tol
        or abs(r["coolant_residual"]) > energy_tol
        or abs(r["latent_residual"]) > source_tol
        or abs(r["inside_wall_residual"]) > wall_tol
        or abs(r["outside_wall_residual"]) > wall_tol):
        raise WetGasEnthalpyError("numerical_stagnation",
            f"Bounded Bulk-Mean residual solve did not close: {r}; evaluations={count}, "
            f"optimizer={fit.message}. No physical prediction is accepted.",
            diagnostics=dict(outlet_state=state,residuals=r,active_bounds=fit.active_mask.tolist(),
                             evaluations=count))
    return state, dict(gas_energy_W=r["gas_residual"],coolant_energy_W=r["coolant_residual"],
        mass_balance_kg_s=abs(r["mass_residual"]),latent_source_W=r["latent_residual"],
        bounded_residual_evaluations=float(count))
