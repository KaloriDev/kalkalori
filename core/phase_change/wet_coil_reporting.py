# SPDX-License-Identifier: GPL-3.0-only
"""Post-solve equivalent reporting; never a wet-process constitutive closure."""
from dataclasses import dataclass
from math import isfinite

from core.heat_transfer.ntu import ntu_from_effectiveness

UA_REPORTING_BASIS = "wet_equivalent_secant_capacity_ntu"
GLOBAL_WET_MODEL = "elmahdy_mitalas_energyplus_v25_2_adapted"


@dataclass(frozen=True)
class WetEquivalentProcess:
    heat: float
    hot_capacity: float
    cold_capacity: float
    effectiveness: float
    ntu: float
    ua: float
    emtd: float


def equivalent_wet_process(
    *, heat, hot_in, hot_out, cold_in, cold_out, flow_arrangement
):
    """Reduce a solved Q_liquid/temperature program on its secant capacities.

    Zero duty or an exactly isothermal sensible stream does not determine
    both secant capacities. Reject that degenerate reporting state explicitly;
    do not invent a small temperature difference or a conductance.
    """
    if not all(isfinite(v) for v in (heat, hot_in, hot_out, cold_in, cold_out)):
        raise ValueError("Wet equivalent process requires finite solved states")
    dh, dc, span = hot_in - hot_out, cold_out - cold_in, hot_in - cold_in
    if heat <= 0 or dh <= 0 or dc <= 0 or span <= 0:
        raise ValueError(
            "Wet equivalent process is undefined for zero duty or nonpositive temperature changes"
        )
    ch, cc = heat / dh, heat / dc
    cmin = min(ch, cc)
    eps = heat / (cmin * span)
    ntu = ntu_from_effectiveness(
        eps, ch, cc, flow_arrangement=flow_arrangement, C_inside=cc, C_outside=ch
    )
    ua = ntu * cmin
    if not isfinite(ua) or ua <= 0:
        raise ValueError(
            "Wet equivalent process has no finite positive reporting conductance"
        )
    return WetEquivalentProcess(heat, ch, cc, eps, ntu, ua, heat / ua)
