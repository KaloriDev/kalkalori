# SPDX-License-Identifier: GPL-3.0-only
"""Expose the accepted bulk-mean wet states without rerunning a dry model."""
import math

from core.heat_transfer.thermal_iteration import WallTemperatureEnvelope, WallTemperatureProbe


def outside_surface_envelope(hx, solution, *, inside_mean, outside_mean, surface_margin):
    """Provider-specific metal, interface and radial diagnostics, all in K.

    Bare bounds are the exact final wet-area input states. Their mean differs
    from the solved mean only by the reported outer fixed-point residual.
    Finned bounds are native radial metal states at the 0D mean operating point;
    an equilibrium cold-zone offset is never presented as actual metal.
    """
    scale = 1 + surface_margin
    ri = (1 / (solution.alfa_i * hx.bundle.total_inner_area)
          + hx.resistance_fouling_inside) * scale
    rfo = hx.resistance_fouling_outside * scale
    rw = hx.tube_wall_resistance() * scale
    wet = solution.wet_finned_surface
    probes = []
    if wet is None:
        downstream = ri + rw + rfo
        for ti, to, ts in solution.surface_temperature_samples:
            q = (ts - ti) / downstream
            probes.append(WallTemperatureProbe(
                ti, to, ti + q * ri, ts - q * rfo,
                solution.alfa_i, solution.alfa_o_effective, solution.converged,
                solution.iterations, heat_rate_probe=-q,
                outside_primary_surface_temperature=ts,
                outside_skin_temperature_min=ts, outside_skin_temperature_max=ts,
                outside_surface_temperature_mean=ts,
            ))
        ts = solution.T_wall_outside
        metal = ts - solution.Q_total * rfo
        primary, fb, ft, skin_min, skin_max, surface_mean = ts, None, None, ts, ts, ts
        assumption = "topology_paired_0d_interface_balance_uniform_latent_source"
    else:
        ts, metal = wet.primary_surface_temperature, wet.core_wall_temperature
        primary, fb, ft = ts, wet.fin_base_temperature, wet.fin_tip_temperature
        skin_min, skin_max = min(primary, fb, ft), max(primary, fb, ft)
        surface_mean = wet.outside_surface_temperature_area_mean
        assumption = "native_radial_fin_solution_at_0d_mean_bulk_state"
    probes.append(WallTemperatureProbe(
        inside_mean, outside_mean, solution.T_wall_inside, metal,
        solution.alfa_i, solution.alfa_o_effective, solution.converged,
        solution.iterations, heat_rate_probe=-solution.Q_total,
        outside_primary_surface_temperature=primary,
        fin_base_temperature=fb, fin_tip_temperature=ft,
        outside_skin_temperature_min=skin_min, outside_skin_temperature_max=skin_max,
        outside_surface_temperature_mean=surface_mean,
    ))
    return WallTemperatureEnvelope(
        min(p.inside_wall_temperature for p in probes),
        max(p.inside_wall_temperature for p in probes),
        min(p.outside_wall_temperature for p in probes),
        max(p.outside_wall_temperature for p in probes), tuple(probes),
        method="bulk_mean_analogy_native_surface_envelope",
        inside_mean=solution.T_wall_inside, outside_mean=metal,
        outside_skin_min=min(p.outside_skin_temperature_min for p in probes),
        outside_skin_max=max(p.outside_skin_temperature_max for p in probes),
        outside_skin_mean=surface_mean, primary_surface_mean=primary,
        fin_base_min=fb if fb is not None else math.nan,
        fin_base_max=fb if fb is not None else math.nan,
        fin_tip_min=ft if ft is not None else math.nan,
        fin_tip_max=ft if ft is not None else math.nan,
        fin_base_mean=fb if fb is not None else math.nan,
        fin_tip_mean=ft if ft is not None else math.nan,
        fin_surface_min=min(fb, ft) if fb is not None else math.nan,
        fin_surface_max=max(fb, ft) if fb is not None else math.nan,
        temperature_definition=("gas_facing_interface_separate_from_metal_wall" if wet is None
                                else "native_radial_exposed_metal; equilibrium_offset_separate"),
        spatial_assumption=assumption,
        applicability="bulk_mean_provider_native_0d_states",
        mean_method="global_interface_balance" if wet is None else "native_radial_area_integral",
    )
