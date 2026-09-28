# SPDX-License-Identifier: GPL-3.0-only
"""Isolated Elmahdy-Mitalas counterflow equations; not yet a public HX adapter.

Original equation implementation. See docs/elmahdy_mitalas.md for source,
units and the deliberate source approximation Q_gas == Q_liquid (no drain).
Temperatures at this internal equation interface are Celsius. Conductances
and enthalpies use W, J, kg dry carrier and seconds throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, isfinite
from typing import Callable

SOURCE_REVISION = "cf7368216c73c43181e057fa33b479c4e0c86df0"
MODEL_ID = "ElmahdyMitalas1977"


class SourceClosureError(ValueError):
    """The source approximation produced a non-condensing wet solution."""

    def __init__(self, humidity_in, humidity_out, wet_fraction, effective_surface,
                 cold_surface, hot_surface):
        self.diagnostics = dict(humidity_in=humidity_in, humidity_out=humidity_out,
                                wet_fraction=wet_fraction, effective_surface=effective_surface,
                                cold_surface=cold_surface, hot_surface=hot_surface)
        super().__init__("Wet source closure predicts an invalid humidity: "
                         f"Win={humidity_in:.12g}, Wout={humidity_out:.12g}, "
                         f"wet_fraction={wet_fraction:.12g}")


@dataclass(frozen=True)
class CoilInput:
    air_in: float
    liquid_in: float
    humidity_in: float
    dry_mass_flow: float
    liquid_capacity: float  # m_liquid * cp_liquid, W/K
    gas_cp: float  # J/(kg dry carrier K), at unchanged inlet W
    gas_h_in: float  # J/kg dry carrier
    dewpoint: float
    dry_air_resistance: float  # K/W, whole physical surface
    wet_air_resistance: float  # s/kg dry carrier, cp/(h_wet A_eff)
    inner_resistance: Callable[[float], float]  # K/W incl wall, at liquid T
    saturation_enthalpy: Callable[[float], float]  # J/kg dry carrier
    temperature_at_saturation_enthalpy: Callable[[float], float]
    humidity_at_temperature_enthalpy: Callable[[float, float], float]


@dataclass(frozen=True)
class CoilResult:
    heat: float  # source approximation, gas removal == liquid gain, W
    air_out: float
    liquid_out: float
    humidity_out: float
    condensate: float
    wet_fraction: float
    interface_air: float
    interface_liquid: float
    interface_surface: float
    cold_surface: float
    hot_surface: float
    outlet_enthalpy: float
    wet_enthalpy_conductance: float  # kg dry carrier/s, whole surface
    source_revision: str = SOURCE_REVISION
    model: str = MODEL_ID
    energy_convention: str = "source_no_drain_enthalpy"


def _counterflow_transfer(ua: float, c1: float, c2: float) -> float:
    """epsilon*Cmin, including equal-capacity and zero-area limits."""
    if ua == 0.0:
        return 0.0
    cmin, cmax = min(c1, c2), max(c1, c2)
    ratio = cmin / cmax
    ntu = ua / cmin
    if abs(1.0 - ratio) < 1e-10:
        return cmin * ntu / (1.0 + ntu)
    decay = exp(-ntu * (1.0 - ratio))
    return cmin * (-expm1(-ntu * (1.0 - ratio))) / (1.0 - ratio * decay)


def solve_source_coil(x: CoilInput) -> CoilResult:
    """Solve the documented dry/wet LM differences with a bounded interface.

    The callbacks separate thermodynamic properties and local resistance from
    the global equations. This entry point intentionally preserves the source
    energy approximation. It MUST NOT be used for process acceptance until the
    drain-enthalpy adaptation and geometry adapters have been verified.
    """
    values = (x.air_in, x.liquid_in, x.humidity_in, x.dry_mass_flow,
              x.liquid_capacity, x.gas_cp, x.gas_h_in, x.dewpoint,
              x.dry_air_resistance, x.wet_air_resistance)
    if not all(isfinite(v) for v in values):
        raise ValueError("Coil inputs must be finite")
    if min(x.dry_mass_flow, x.liquid_capacity, x.gas_cp,
           x.dry_air_resistance, x.wet_air_resistance) <= 0:
        raise ValueError("Flow, capacity and resistances must be positive")
    if x.humidity_in < 0 or x.air_in <= x.liquid_in:
        raise ValueError("Source cooling kernel requires nonnegative W and hot gas")
    md, cw = x.dry_mass_flow, x.liquid_capacity
    ca = md * x.gas_cp
    ta, tw, hin = x.air_in, x.liquid_in, x.gas_h_in

    def resistance(t):
        value = x.inner_resistance(t)
        if not isfinite(value) or value <= 0:
            raise ValueError("Inner resistance must be finite and positive")
        return value

    # Solve the dry limit on its own temperature basis.
    toutw = tw
    for _ in range(200):
        ri = resistance((tw + toutw) / 2)
        transfer = _counterflow_transfer(1 / (ri + x.dry_air_resistance), ca, cw)
        qdry = transfer * (ta - tw)
        nextw = tw + qdry / cw
        if abs(nextw - toutw) < 1e-9:
            break
        toutw = nextw
    else:
        raise RuntimeError("Dry liquid property iteration did not converge")
    touta = ta - qdry / ca
    cold = tw + (touta - tw) * ri / (ri + x.dry_air_resistance)
    hot = nextw + (ta - nextw) * ri / (ri + x.dry_air_resistance)
    if cold >= x.dewpoint:
        return CoilResult(qdry, touta, nextw, x.humidity_in, 0., 0.,
                          touta, tw, cold, cold, hot, hin - qdry / md, 0.)

    def region(f, *, boundary_secant=False):
        # Linear saturation secant and water resistance converge together.
        low = tw
        high = max(tw + .01, min(x.dewpoint, ta))
        b = (x.saturation_enthalpy(high) - x.saturation_enthalpy(low)) / (high - low)
        a = x.saturation_enthalpy(low) - b * low
        tintw, outw = tw, tw
        for _ in range(400):
            riw = resistance((tw + tintw) / 2)
            rid = resistance((tintw + outw) / 2)
            ua_d = (1 - f) / (rid + x.dry_air_resistance)
            dry_effectiveness = _counterflow_transfer(ua_d, ca, cw) / ca
            kw = 1 / (b * riw + x.wet_air_resistance)
            transfer_w = _counterflow_transfer(f * kw, md, cw / b)
            denom = cw - transfer_w * x.gas_cp * dry_effectiveness
            if denom <= 0:
                raise ValueError("Singular wet/dry interface capacity balance")
            new_tintw = (cw * tw + transfer_w *
                        (hin - a - b * tw - x.gas_cp * dry_effectiveness * ta)) / denom
            tinta = ta - dry_effectiveness * (ta - new_tintw)
            hd = hin - x.gas_cp * (ta - tinta)
            hout = hd - cw * (new_tintw - tw) / md
            heat = md * (hin - hout)
            new_outw = tw + heat / cw
            cold = (x.wet_air_resistance * tw + riw * (hout - a)) / (x.wet_air_resistance + b * riw)
            wet_hot = (x.wet_air_resistance * new_tintw + riw * (hd - a)) / (x.wet_air_resistance + b * riw)
            surf = new_tintw + (tinta - new_tintw) * rid / (rid + x.dry_air_resistance)
            high = wet_hot if f == 1 and not boundary_secant else x.dewpoint
            if abs(high - cold) < 1e-4:
                high = cold + 1e-4
            new_b = (x.saturation_enthalpy(high) - x.saturation_enthalpy(cold)) / (high - cold)
            new_a = x.saturation_enthalpy(cold) - new_b * cold
            error = max(abs(new_tintw - tintw), abs(new_outw - outw),
                        abs(new_b - b) / max(1., abs(b)), abs(new_a - a) / max(1., abs(b)))
            tintw, outw = new_tintw, new_outw
            if error < 1e-9:
                return heat, tinta, tintw, surf, cold, wet_hot, hout, kw, hd
            a, b = new_a, new_b
            if not isfinite(b) or b <= 0:
                raise ValueError("Saturation enthalpy slope must be positive")
        raise RuntimeError("Wet saturation/property iteration did not converge")

    full = region(1.)
    if full[5] < x.dewpoint:
        f, solved = 1., full
    else:
        # The source's general wet/dry solution includes its f=1 bound.
        # It keeps the cold-wall/dewpoint secant in that limiting branch.
        wet_bound = region(1., boundary_secant=True)
        if wet_bound[3] <= x.dewpoint:
            f, solved = 1., wet_bound
        else:
            lo, hi = 0., 1.
            for _ in range(70):
                f = (lo + hi) / 2
                solved = region(f, boundary_secant=True)
                residual = solved[3] - x.dewpoint
                if abs(residual) < 1e-8 and hi - lo < 1e-8:
                    break
                if residual > 0:
                    hi = f
                else:
                    lo = f
            else:
                raise RuntimeError("Wet/dry interface is not bracketed or did not converge")
    q, tint_air, tint_water, surf, cold, hot, hout, kw, hint = solved
    bypass = exp(-f / (x.wet_air_resistance * md))
    fraction = -expm1(-f / (x.wet_air_resistance * md))
    effective_surface_h = hint - (hint - hout) / fraction
    effective_surface_t = x.temperature_at_saturation_enthalpy(effective_surface_h)
    outa = effective_surface_t + (tint_air - effective_surface_t) * bypass
    wout = x.humidity_at_temperature_enthalpy(outa, hout)
    if not isfinite(wout) or wout < 0 or wout > x.humidity_in + 1e-10:
        raise SourceClosureError(x.humidity_in, wout, f, effective_surface_t, cold, hot)
    return CoilResult(q, outa, tw + q/cw, wout, md*(x.humidity_in-wout), f,
                      tint_air, tint_water, hot if f == 1 else surf,
                      cold, hot, hout, kw)
