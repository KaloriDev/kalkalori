# SPDX-License-Identifier: GPL-3.0-only
"""Internal drain-coupled Elmahdy-Mitalas profile (not public orchestration).

The frozen reference kernel is deliberately separate. Here the source outlet
construction is applied to each remaining wet length, defining W(x). No local
moisture transport ODE is substituted. x increases from liquid inlet/gas outlet.
Only the drain forcing is interpolated at quadrature nodes; h and liquid T are
continuous analytical Green-function profiles, not axial cell unknowns.
"""
from dataclasses import dataclass, field
from math import exp, expm1, isfinite
from typing import Callable

import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from core.heat_transfer.elmahdy_mitalas import CoilInput, _counterflow_transfer


class WetCoilModelError(ValueError):
    def __init__(self, reason, **diagnostics):
        self.diagnostics = dict(rejection_reason=reason, **diagnostics)
        super().__init__(f"Elmahdy-Mitalas candidate rejected: {reason}")


@dataclass(frozen=True)
class ProfilePoint:
    coordinate: float
    gas_enthalpy: float
    gas_temperature: float
    humidity: float
    liquid_temperature: float
    surface_temperature: float
    condensate_density: float  # kg/s per unit whole-coil area fraction
    drain_density: float  # W per unit whole-coil area fraction


@dataclass(frozen=True)
class WetCoilResult:
    regime: str
    heat_liquid: float
    heat_gas: float
    drain_enthalpy: float
    condensate: float
    air_out: float
    liquid_out: float
    humidity_out: float
    outlet_enthalpy: float
    wet_fraction: float
    onset_margin: float  # cold dry surface minus inlet dewpoint, K
    interface_air: float
    interface_liquid: float
    cold_surface: float
    hot_surface: float
    profile: tuple[ProfilePoint, ...]
    diagnostics: dict
    _region: object = field(default=None, repr=False, compare=False)


def _exprel(z):
    if abs(z) < 1e-6:
        return 1 + z / 2 + z * z / 6 + z**3 / 24 + z**4 / 120
    return expm1(z) / z


class _Region:
    """A wet-region analytic profile and its bounded drain quadrature."""

    def __init__(
        self,
        x,
        f,
        liquid_enthalpy,
        *,
        drain,
        order,
        boundary_secant=False,
        sensible_coordinate=lambda T: T,
        temperature_from_coordinate=lambda u: u,
        surface_enthalpy=None,
    ):
        self.x, self.f = x, f
        self.surface_enthalpy = surface_enthalpy
        self.sensible_coordinate = sensible_coordinate
        self.temperature_from_coordinate = temperature_from_coordinate
        md, cw = x.dry_mass_flow, x.liquid_capacity
        cp, rw, tw, ta = x.gas_cp, x.wet_air_resistance, x.liquid_in, x.air_in
        nodes, weights = leggauss(order)
        self.z = (nodes + 1) * f / 2
        self.weights = weights * f / 2
        self.qnodes, self.qweights = leggauss(max(12, order))
        density = np.zeros(order)
        low, high = tw, max(tw + 0.01, min(x.dewpoint, ta))
        b = (x.saturation_enthalpy(high) - x.saturation_enthalpy(low)) / (high - low)
        a = x.saturation_enthalpy(low) - b * low
        tintw = outw = tw
        previous = None
        for iteration in range(250):
            ri = x.inner_resistance((tw + tintw) / 2)
            rid = x.inner_resistance((tintw + outw) / 2)
            if not all(isfinite(v) and v > 0 for v in (ri, rid, b)):
                raise WetCoilModelError("invalid resistance or saturation slope")
            k = 1 / (rw + b * ri)
            lam = k * (1 / md - b / cw)
            gamma = k * b * (ri / md + rw / cw)
            self.a, self.b, self.ri, self.k, self.lam, self.gamma = (
                a,
                b,
                ri,
                k,
                lam,
                gamma,
            )
            self.poly = cheb.chebfit(nodes, density, order - 1)
            # Particular solution for a zero homogeneous driving force.
            ih, it = self._forcing(f)
            J = f * _exprel(lam * f)
            Ah = 1 + k / md * J
            ch = -k / md * (a + b * tw) * J + ih
            At = k / cw * J
            ct = tw - k / cw * (a + b * tw) * J + it
            transfer = _counterflow_transfer(
                (1 - f) / (rid + x.dry_air_resistance), md * cp, cw
            )
            denom = Ah - transfer / md * At
            if denom <= 0:
                raise WetCoilModelError("singular wet/dry capacity balance")
            self.h0 = (x.gas_h_in - transfer / md * (ta - ct) - ch) / denom
            self.D0 = self.h0 - a - b * tw
            self.hint = Ah * self.h0 + ch
            tintw = At * self.h0 + ct
            self.tint = self.temperature_from_coordinate(
                self.sensible_coordinate(ta) - (x.gas_h_in - self.hint) / cp
            )
            outw = tintw + transfer * (ta - tintw) / cw
            self.interface_surface = tintw + (self.tint - tintw) * rid / (
                rid + x.dry_air_resistance
            )
            cold = self.thermal(0)[2]
            hot = self.thermal(f)[2]
            high = x.dewpoint if boundary_secant else hot
            if abs(high - cold) < 1e-3:
                # The coalescing secant is the derivative at its midpoint.
                # A finite symmetric property interval avoids cancellation of
                # large absolute enthalpy datums at vanishing wet lengths.
                middle = (high + cold) / 2
                nb = (
                    x.saturation_enthalpy(middle + 1e-3)
                    - x.saturation_enthalpy(middle - 1e-3)
                ) / 2e-3
            else:
                nb = (x.saturation_enthalpy(high) - x.saturation_enthalpy(cold)) / (
                    high - cold
                )
            na = x.saturation_enthalpy(cold) - nb * cold
            points = [self.point(float(z), liquid_enthalpy, drain) for z in self.z]
            nd = np.array([p.drain_density for p in points])
            hout = self.point(0.0, liquid_enthalpy, drain)
            current = np.array(
                [
                    self.h0 * md / 1000,
                    tintw,
                    outw,
                    cold,
                    hot,
                    float(np.dot(self.weights, nd)),
                ]
            )
            err = (
                float("inf")
                if previous is None
                else float(np.max(np.abs(current - previous)))
            )
            derr = float(np.max(np.abs(nd - density)))
            if err < 2e-8 and derr < 2e-5:
                self.cold, self.hot, self.tintw, self.outw = cold, hot, tintw, outw
                self.points, self.outlet, self.iterations = (
                    tuple(points),
                    hout,
                    iteration + 1,
                )
                self.drain = float(np.dot(self.weights, nd))
                self.mass_integral = float(
                    np.dot(self.weights, [p.condensate_density for p in points])
                )
                self.heat_liquid = cw * (outw - tw)
                self.heat_gas = md * (x.gas_h_in - self.h0)
                return
            previous = current
            a, b = na, nb
            density = 0.4 * density + 0.6 * nd
        raise WetCoilModelError(
            "wet profile iteration did not converge",
            wet_fraction=f,
            error=err,
            drain_error=derr,
        )

    def _density(self, z):
        return cheb.chebval(2 * np.asarray(z) / self.f - 1, self.poly)

    def _forcing(self, z):
        # D'=lambda D+gamma*d. Analytically integrate its exponential kernel.
        if z == 0:
            return 0.0, 0.0
        ss = (self.qnodes + 1) * z / 2
        ds = self._density(ss)
        spans = z - ss
        kernels = np.array([v * _exprel(self.lam * v) for v in spans])
        intd = float(np.dot(self.qweights, ds) * z / 2)
        intD = self.gamma * float(np.dot(self.qweights, kernels * ds) * z / 2)
        k, ri, b, rw = self.k, self.ri, self.b, self.x.wet_air_resistance
        return (
            k / self.x.dry_mass_flow * (intD + b * ri * intd),
            k / self.x.liquid_capacity * (intD - rw * intd),
        )

    def thermal(self, z):
        md, cw = self.x.dry_mass_flow, self.x.liquid_capacity
        ih, it = self._forcing(z)
        integral = self.D0 * z * _exprel(self.lam * z)
        h = self.h0 + self.k / md * integral + ih
        tl = self.x.liquid_in + self.k / cw * integral + it
        d = float(self._density(z))
        rw = self.x.wet_air_resistance
        ts = (rw * tl + self.ri * (h - self.a) - self.ri * rw * d) / (
            rw + self.b * self.ri
        )
        hp = (h - self.a - self.b * ts) / (rw * md)
        return h, tl, ts, hp

    def point(self, z, hl, drain):
        x = self.x
        h, tl, ts, hp = self.thermal(z)
        rate = 1 / (x.wet_air_resistance * x.dry_mass_flow)
        length = self.f - z
        if length <= 0:
            raise ValueError("Moisture derivative is evaluated inside the wet region")
        fraction = -expm1(-rate * length)
        bypass = exp(-rate * length)
        # Integrating the derivative avoids subtracting nearly equal endpoint
        # enthalpies at onset and at the warm end of the wet region.
        dh = (
            self.k
            / x.dry_mass_flow
            * self.D0
            * exp(self.lam * z)
            * length
            * _exprel(self.lam * length)
            + self._forcing(self.f)[0]
            - self._forcing(z)[0]
        )
        heff = self.hint - dh / fraction
        teff = x.temperature_at_saturation_enthalpy(heff)
        seff = self.sensible_coordinate(teff)
        sint = self.sensible_coordinate(self.tint)
        gas = self.temperature_from_coordinate(seff + (sint - seff) * bypass)
        W = x.humidity_at_temperature_enthalpy(gas, h)
        step = 1e-3
        hsprime = (
            x.saturation_enthalpy(teff + step) - x.saturation_enthalpy(teff - step)
        ) / (2 * step)
        effprime = (hp / fraction - rate * bypass * dh / fraction**2) / hsprime
        sp_eff = (
            self.sensible_coordinate(teff + step)
            - self.sensible_coordinate(teff - step)
        ) / (2 * step)
        sp_gas = (
            self.sensible_coordinate(gas + step) - self.sensible_coordinate(gas - step)
        ) / (2 * step)
        gasprime = (
            fraction * sp_eff * effprime + rate * bypass * (sint - seff)
        ) / sp_gas
        wh = x.humidity_at_temperature_enthalpy(gas, h + 1) - W
        wt = (
            x.humidity_at_temperature_enthalpy(gas + step, h)
            - x.humidity_at_temperature_enthalpy(gas - step, h)
        ) / (2 * step)
        j = x.dry_mass_flow * (wh * hp + wt * gasprime)
        hcond = (
            hl(ts)
            if self.surface_enthalpy is None
            else self.surface_enthalpy(z / self.f, ts, gas, W)
        )
        return ProfilePoint(z, h, gas, W, tl, ts, j, hcond * j if drain else 0.0)


def _solve_profile_candidate(
    x: CoilInput,
    *,
    liquid_enthalpy: Callable[[float], float],
    saturation_humidity: Callable[[float], float],
    drain_enabled=True,
    quadrature_order=10,
    sensible_coordinate=lambda T: T,
    temperature_from_coordinate=lambda u: u,
    surface_enthalpy=None,
) -> WetCoilResult:
    """Internal cooling solve with source-profile moisture and coupled drain.

    Temperatures use one consistent scale supplied by the adapter (K in
    production, Celsius in reference-equation comparisons). Resistances and
    cp are whole-coil SI dry-carrier quantities as in CoilInput.
    """
    md, cw, cp = x.dry_mass_flow, x.liquid_capacity, x.gas_cp
    values = (
        md,
        cw,
        cp,
        x.dry_air_resistance,
        x.wet_air_resistance,
        x.air_in,
        x.liquid_in,
        x.humidity_in,
        x.gas_h_in,
        x.dewpoint,
    )
    if not all(isfinite(v) for v in values) or x.humidity_in < 0:
        raise ValueError("Coil states must be finite with nonnegative humidity")
    if not isinstance(quadrature_order, int) or not 6 <= quadrature_order <= 32:
        raise ValueError("Wet profile quadrature order must be an integer from 6 to 32")
    if x.humidity_in > saturation_humidity(x.air_in) + 1e-9:
        raise WetCoilModelError("supersaturated inlet vapor")
    if (
        min(md, cw, cp, x.dry_air_resistance, x.wet_air_resistance) <= 0
        or x.air_in <= x.liquid_in
    ):
        raise ValueError("Positive flows/resistances and a hot gas inlet are required")
    twout = x.liquid_in
    for _ in range(200):
        ri = x.inner_resistance((x.liquid_in + twout) / 2)
        if not isfinite(ri) or ri <= 0:
            raise ValueError("Invalid inside resistance")
        q = _counterflow_transfer(1 / (ri + x.dry_air_resistance), md * cp, cw) * (
            x.air_in - x.liquid_in
        )
        nextw = x.liquid_in + q / cw
        if abs(nextw - twout) < 1e-10:
            break
        twout = nextw
    else:
        raise WetCoilModelError("dry property iteration did not converge")
    tout = x.air_in - q / (md * cp)
    cold = x.liquid_in + (tout - x.liquid_in) * ri / (ri + x.dry_air_resistance)
    hot = nextw + (x.air_in - nextw) * ri / (ri + x.dry_air_resistance)
    margin = cold - x.dewpoint
    if margin >= 0:
        return WetCoilResult(
            "DRY",
            q,
            q,
            0.0,
            0.0,
            tout,
            nextw,
            x.humidity_in,
            x.gas_h_in - q / md,
            0.0,
            margin,
            tout,
            x.liquid_in,
            cold,
            hot,
            (),
            dict(rejection_reason=None, energy_residual=0.0, mass_residual=0.0),
        )

    def region(f, bound=False):
        try:
            return _Region(
                x,
                f,
                liquid_enthalpy,
                drain=drain_enabled,
                order=quadrature_order,
                boundary_secant=bound,
                sensible_coordinate=sensible_coordinate,
                temperature_from_coordinate=temperature_from_coordinate,
                surface_enthalpy=surface_enthalpy,
            )
        except WetCoilModelError as exc:
            diagnostics = dict(exc.diagnostics)
            reason = diagnostics.pop("rejection_reason")
            diagnostics.setdefault("onset_margin", margin)
            diagnostics.setdefault("wet_fraction", f)
            raise WetCoilModelError(reason, **diagnostics) from exc

    wet = region(1.0)
    if wet.hot < x.dewpoint:
        f = 1.0
    else:
        wet = region(1.0, True)
        if wet.interface_surface <= x.dewpoint:
            f = 1.0
        else:
            candidates = {}

            def boundary(f):
                if f == 0:
                    return margin
                candidate = region(f, True)
                candidates[f] = candidate
                return candidate.interface_surface - x.dewpoint

            f = brentq(boundary, 0.0, 1.0, xtol=2e-10)
            wet = candidates[f] if f in candidates else region(f, True)
    check_nodes, check_weights = leggauss(min(32, 2 * quadrature_order))
    checked = [
        wet.point(float((v + 1) * f / 2), liquid_enthalpy, drain_enabled)
        for v in check_nodes
    ]
    mass_check = float(
        np.dot(check_weights, [p.condensate_density for p in checked]) * f / 2
    )
    drain_check = float(
        np.dot(check_weights, [p.drain_density for p in checked]) * f / 2
    )
    if (
        abs(drain_check - wet.drain) > 2e-4
        or abs(mass_check - wet.mass_integral) > 2e-10
    ):
        raise WetCoilModelError(
            "wet profile quadrature unresolved",
            drain_error=drain_check - wet.drain,
            mass_error=mass_check - wet.mass_integral,
        )
    mass = md * (x.humidity_in - wet.outlet.humidity)
    energy_error = wet.heat_gas - wet.heat_liquid - wet.drain
    mass_error = mass - wet.mass_integral
    sample = (wet.outlet,) + wet.points
    force = min(
        x.humidity_in - saturation_humidity(wet.hot),
        min(p.humidity - saturation_humidity(p.surface_temperature) for p in sample),
    )
    vapor = min(saturation_humidity(p.gas_temperature) - p.humidity for p in sample)
    diagnostics = dict(
        onset_margin=margin,
        wet_fraction=f,
        condensate=mass,
        min_wet_driving_force=force,
        min_vapor_margin=vapor,
        energy_residual=energy_error,
        mass_residual=mass_error,
        quadrature_order=quadrature_order,
        iterations=wet.iterations,
        drain_quadrature_error=drain_check - wet.drain,
        mass_quadrature_error=mass_check - wet.mass_integral,
        rejection_reason=None,
    )
    reasons = []
    if mass < 0:
        reasons.append("negative condensate")
    if min(p.condensate_density for p in sample) < -1e-11:
        reasons.append("negative local removal")
    if force < -1e-9:
        reasons.append("negative wet driving force")
    if vapor < -1e-9 or min(p.humidity for p in sample) < 0:
        reasons.append("inadmissible vapor state")
    if abs(energy_error) > 2e-4 or abs(mass_error) > 2e-10:
        reasons.append("inconsistent integral balance")
    diagnostics["rejection_reason"] = ", ".join(reasons) if reasons else None
    return WetCoilResult(
        "FULLY_WET" if f == 1 else "PARTIALLY_WET",
        wet.heat_liquid,
        wet.heat_gas,
        wet.drain,
        mass,
        wet.outlet.gas_temperature,
        wet.outw,
        wet.outlet.humidity,
        wet.h0,
        f,
        margin,
        wet.tint,
        wet.tintw,
        wet.cold,
        wet.hot,
        sample,
        diagnostics,
        _region=wet,
    )


def validate_wet_coil(result):
    """Accept only after all constitutive/property iterations have converged."""
    if result.diagnostics.get("rejection_reason"):
        diagnostics = dict(result.diagnostics)
        reason = diagnostics.pop("rejection_reason")
        raise WetCoilModelError(reason, **diagnostics)
    return result


def solve_wet_coil(x: CoilInput, **kwargs) -> WetCoilResult:
    """Solve and validate a coil with already fixed constitutive coefficients."""
    return validate_wet_coil(_solve_profile_candidate(x, **kwargs))
