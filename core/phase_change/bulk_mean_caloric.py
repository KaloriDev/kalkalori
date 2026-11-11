# SPDX-License-Identifier: GPL-3.0-only
"""Common advected-energy basis for Bulk-Mean dry and wet transport.

Transport correlations still consume their original bulk cp. Capacity rates
use finite stream enthalpy increments, so the zero-condensate branch does not
switch caloric models or backends at activation.
"""
from core.heat_transfer.wet_coil_adapters import InsideWallAdapter
from core.phase_change.capability import detect_phase_change_capability, _PureWaterSinglePhaseGuardProvider
from core.phase_change.wet_gas_enthalpy import WetGasEnthalpyEvaluator
from core.properties.water import WATER_TRIPLE_POINT_TEMPERATURE_K, WATER_CRITICAL_TEMPERATURE_K
from core.phase_change.bulk_mean_closure import uniform_source_bulk_means
from core.geometry._thermal_topology import resolve_thermal_topology


def gas_capacity(evaluator, mass_dry, inlet, outlet, humidity, fallback):
    if not (WATER_TRIPLE_POINT_TEMPERATURE_K <= min(inlet,outlet)
            and max(inlet,outlet) < WATER_CRITICAL_TEMPERATURE_K):
        # Keep dry-only calculations outside the wet caloric model's domain
        # on their established sensible path. Active wet guards remain strict.
        return fallback
    if abs(inlet-outlet) < 1e-5:
        # One-sided derivative remains within the same water caloric domain.
        other = inlet+1e-4 if inlet+1e-4 < WATER_CRITICAL_TEMPERATURE_K else inlet-1e-4
        return mass_dry*(evaluator.enthalpy(other,humidity)-evaluator.enthalpy(inlet,humidity))/(other-inlet)
    return mass_dry*(evaluator.enthalpy(inlet,humidity)-evaluator.enthalpy(outlet,humidity))/(inlet-outlet)


def bulk_mean_caloric_rates(hx,inside,outside):
    cap = detect_phase_change_capability(outside.provider)
    if not cap.capable or cap.provider_kind != "gas_mixture":
        return None
    evaluator = WetGasEnthalpyEvaluator(outside.p,cap,reuse_dry_backend_state=True)
    provider = inside.provider
    raw = provider.provider if isinstance(provider,_PureWaterSinglePhaseGuardProvider) else provider
    coolant = InsideWallAdapter(hx.bundle,raw,inside.m_dot,inside.p)
    def rates(ti,to,ci,co):
        provider.at(T=ti,p=inside.p)  # Preserve the configured phase guard.
        return (coolant.capacity(inside.T_in,ti),
                gas_capacity(evaluator,outside.m_dot/(1+cap.W_in),outside.T_in,to,cap.W_in,co))
    def wall_means(network,ci,co,scale):
        flow = resolve_thermal_topology(hx.bundle)
        if flow not in ("counterflow","cocurrentflow","crossflow"):
            return None  # Existing dry-only N-section topology stays separate.
        mean = uniform_source_bulk_means(UA=network.UA*scale,C_gas=co,C_other=ci,
            T_gas_in=outside.T_in,T_other_in=inside.T_in,
            gas_film_fraction=network.resistance_outside/network.resistance_total,
            Q_latent=0.,flow_arrangement=flow)
        inner = mean.other_temperature+mean.heat_total*network.resistance_inside/scale
        outer = inner+mean.heat_total*network.resistance_core_wall/scale
        return inner,outer
    def outside_property_temperature(inner,outer):
        if inner is None or outer is None:
            return outer
        # Property correction belongs to the exposed gas-facing boundary.
        # Signed wall conduction recovers its fouling drop without changing
        # the separately reported metal temperature.
        return outer+(outer-inner)/hx.tube_wall_resistance()*hx.resistance_fouling_outside
    rates.wall_means = wall_means
    rates.outside_property_temperature = outside_property_temperature
    return rates
