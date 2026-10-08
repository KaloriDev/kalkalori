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
from core.heat_transfer._wet_numerics import gauss as leggauss
from scipy.optimize import brentq

from core.heat_transfer.wet_coil_solver import _solve_budget
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
    _full_region: object = field(default=None, repr=False, compare=False)


def _exprel(z):
    if abs(z) < 1e-6:
        return 1 + z / 2 + z * z / 6 + z**3 / 24 + z**4 / 120
    return expm1(z) / z


def _profile_derivative(function, temperature):
    """Fourth-order property derivative without small-step datum cancellation.

    The stencil changes numerical evaluation only, not the source-profile
    moisture law. Halving keeps probes inside a callback's property domain;
    failure at every step is propagated rather than accepting a derivative.
    """
    step = 0.02
    for attempt in range(8):
        try:
            near = function(temperature + step) - function(temperature - step)
            far = function(temperature + 2 * step) - function(temperature - 2 * step)
            return (8 * near - far) / (12 * step)
        except ValueError:
            if attempt == 7:
                raise
            step /= 2


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
        wet_solver_options=None,
        _budget=None,
        _initial_region=None,
    ):
        budget = _solve_budget(wet_solver_options, _budget)
        options = budget.options
        self._budget = budget
        _count(budget, 'kernel_region_solves')
        self.x, self.f = x, f
        self.surface_enthalpy = surface_enthalpy
        self.sensible_coordinate = sensible_coordinate
        self.temperature_from_coordinate = temperature_from_coordinate
        thermo = getattr(x.saturation_enthalpy, '__self__', None)
        continuation = getattr(getattr(thermo, 'options', None), 'region_continuation', False)
        accelerate = getattr(getattr(thermo, 'options', None), 'region_acceleration', False)
        defer_outlet = (getattr(getattr(thermo,'options',None),'defer_outlet',False)
                        and surface_enthalpy is None)
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
        if _initial_region is not None:
            # Initial fixed-point values only; solve the new coefficients and
            # drain field to all the same convergence/admissibility gates.
            a, b = _initial_region.a, _initial_region.b
            density = cheb.chebval(nodes, _initial_region.poly)
            tintw, outw = _initial_region.tintw, _initial_region.outw
        previous = None
        previous_fixed = previous_residual = None
        previous_residual_norm = None
        if continuation and _initial_region is not None:
            # Previous converged *full* vector is a predictor. The new region
            # must evaluate its own coefficients, drain and residual gates.
            previous = getattr(_initial_region, '_convergence_vector', None)
        for iteration in range(250):
            _count(budget, 'kernel_region_iterations')
            budget.check(wet_fraction=f)
            fixed = np.r_[a, b, tintw, outw, density]
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
            # Coefficients are fixed for this iterate. Surface points reuse
            # the same endpoint and local forcing integrals many times.
            self._forcing_cache = {}
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
            self._sensible_interface = (self.sensible_coordinate(self.tint)
                if getattr(getattr(thermo,'options',None),'point_reuse',False) else None)
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
            # The outlet moisture/removal does not enter the fixed-point
            # vector or quadrature drain density. Evaluate it only for the
            # converged region; all final closure/admissibility checks remain.
            hout = None if defer_outlet else self.point(0.0, liquid_enthalpy, drain)
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
            # Separate the old mixed kW/K/W vector into dimensional gates.
            # Retain the surface/secant stability ceiling near the wet front;
            # loosening it can violate the unchanged driving-force guard.
            delta = None if previous is None else np.abs(current - previous)
            fidelity = getattr(budget, 'fidelity', None)
            ceiling = 2e-8 if fidelity is None or f < 1 else fidelity.region_temperature_K
            temperature_gate = min(ceiling, options.outlet_temperature_tolerance_K / 10,
                                   options.energy_tolerance_W / (10 * max(cw, md * cp)))
            converged = (delta is not None
                         and delta[0] * 1000 < options.energy_tolerance_W / 10
                         and max(delta[1:5]) < temperature_gate
                         and delta[5] < options.energy_tolerance_W / 10
                         and derr < options.energy_tolerance_W / 10)
            if converged:
                if hout is None:
                    hout = self.point(0.0,liquid_enthalpy,drain)
                self._convergence_vector = np.array(current, copy=True)
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
            if defer_outlet:
                _count(budget,'outlet_point_evaluations_deferred')
            previous = current
            target = np.r_[na, nb, tintw, outw, nd]
            if accelerate:
                # One-step Anderson continuation of the SAME fixed point.
                # Only new physical iterates can satisfy the unchanged gates.
                scale = np.r_[max(abs(x.gas_h_in), 1.), max(abs(b), 1.),
                              max(abs(ta), 1.), max(abs(ta), 1.),
                              np.full(order, max(abs(x.gas_h_in*md), 1.))]
                residual = target-fixed
                norm = np.linalg.norm(residual/scale)
                candidate = None
                if (previous_residual is not None and previous_residual_norm is not None
                        and norm <= 2*previous_residual_norm):
                    difference = (residual-previous_residual)/scale
                    denominator = np.dot(difference,difference)
                    if denominator > 1e-30:
                        coefficient = np.dot(difference,residual/scale)/denominator
                        if abs(coefficient) <= 2:
                            candidate = target-coefficient*(target-previous_fixed)
                previous_fixed, previous_residual = target, residual
                previous_residual_norm = norm
                if (candidate is not None and np.all(np.isfinite(candidate)) and candidate[1] > 0
                        and np.max(np.abs(candidate[2:4]-target[2:4])) < 10
                        and np.min(candidate[4:]) >= 0):
                    a, b, tintw, outw = candidate[:4]
                    density = candidate[4:]
                    _count(budget, 'kernel_region_accelerations')
                    continue
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
        if z in self._forcing_cache:
            return self._forcing_cache[z]
        ss = (self.qnodes + 1) * z / 2
        ds = self._density(ss)
        spans = z - ss
        kernels = np.array([v * _exprel(self.lam * v) for v in spans])
        intd = float(np.dot(self.qweights, ds) * z / 2)
        intD = self.gamma * float(np.dot(self.qweights, kernels * ds) * z / 2)
        k, ri, b, rw = self.k, self.ri, self.b, self.x.wet_air_resistance
        result = (
            k / self.x.dry_mass_flow * (intD + b * ri * intd),
            k / self.x.liquid_capacity * (intD - rw * intd),
        )
        self._forcing_cache[z] = result
        return result

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
        _count(self._budget, 'kernel_profile_points')
        x = self.x
        thermo = getattr(x.saturation_enthalpy, '__self__', None)
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
        sint = (self._sensible_interface if self._sensible_interface is not None
                else self.sensible_coordinate(self.tint))
        gas = self.temperature_from_coordinate(seff + (sint - seff) * bypass)
        if getattr(getattr(thermo,'options',None),'point_reuse',False):
            from core.heat_transfer._wet_if97 import supported
            if supported(teff) and supported(gas):
                effective,carrier = thermo.derivative_bundle(teff),thermo.derivative_bundle(gas)
                ws = (effective.saturation_humidity if effective.saturation_humidity is not None
                      else thermo.saturation_humidity(teff))
                # Keep the native affine inverse's floating-point arithmetic.
                weff = (effective.enthalpy(ws)-effective.dry_enthalpy)/effective.vapor_enthalpy
                wh,wh_eff = 1/carrier.vapor_enthalpy,1/effective.vapor_enthalpy
                W = x.humidity_in-fraction*(x.humidity_in-weff)*wh/wh_eff
                p,dp = effective.pressure_slope
                c = thermo.capability
                dw = c.M_condensable/c.M_dry*thermo.pressure*dp/(thermo.pressure-p)**2
                hsprime = effective.dry_cp+ws*effective.vapor_slope+dw*effective.vapor_enthalpy
                effprime = (hp/fraction-rate*bypass*dh/fraction**2)/hsprime
                sp_eff = (effective.dry_cp+x.humidity_in*effective.vapor_slope)/x.gas_cp
                sp_gas = (carrier.dry_cp+x.humidity_in*carrier.vapor_slope)/x.gas_cp
                gasprime = (fraction*sp_eff*effprime+rate*bypass*(sint-seff))/sp_gas
                base = (h-carrier.dry_enthalpy)/carrier.vapor_enthalpy
                wt = -(carrier.dry_cp+base*carrier.vapor_slope)/carrier.vapor_enthalpy
                j = x.dry_mass_flow*(wh*hp+wt*gasprime)
                hcond = (hl(ts) if self.surface_enthalpy is None
                         else self.surface_enthalpy(z/self.f,ts,gas,W))
                _count(self._budget,'point_bundle_reuses')
                return ProfilePoint(z,h,gas,W,tl,ts,j,hcond*j if drain else 0.)
        # Algebraic source-profile moisture construction, anchored at Win.
        # h(T,W) is affine in W for both supported property formulations.
        # With F=1-bypass and h_eff=h_sat(T_eff), the unchanged source law is
        #   Win-W = F * (Win-Wsat(T_eff)) * h_v(T_eff)/h_v(T_gas).
        # Evaluating this deficit avoids recovering an O(f**2) moisture
        # change by subtracting large absolute enthalpies at dry onset.
        # There is no clipping, blending, transport ODE or alternate closure.
        base_humidity = x.humidity_at_temperature_enthalpy(gas, h)
        # The humidity inverse is affine in enthalpy. A caloric-scale
        # interval computes its exact slope without differencing W at a
        # one-joule perturbation, which causes drain-density roundoff.
        enthalpy_span = 1e6
        wh = (
            x.humidity_at_temperature_enthalpy(gas, h + enthalpy_span) - base_humidity
        ) / enthalpy_span
        hs = x.saturation_enthalpy(teff)
        weff = x.humidity_at_temperature_enthalpy(teff, hs)
        wh_eff = (
            x.humidity_at_temperature_enthalpy(teff, hs + enthalpy_span) - weff
        ) / enthalpy_span
        W = x.humidity_in - fraction * (x.humidity_in - weff) * wh / wh_eff
        thermo = getattr(x.saturation_enthalpy, '__self__', None)
        analytic = (getattr(getattr(thermo, 'options', None), 'analytic_derivatives', False)
                    and max(teff, gas) <= 623.15)
        if analytic and getattr(thermo.options,'shared_water',False):
            from core.heat_transfer._wet_if97 import supported
            analytic = supported(teff) and supported(gas)
        if analytic:
            # h(T,W) is affine in W: these partial derivatives are exact.
            wh = 1/thermo.state(gas).vapor_enthalpy
            wh_eff = 1/thermo.state(teff).vapor_enthalpy
            W = x.humidity_in - fraction * (x.humidity_in - weff) * wh / wh_eff
            hsprime = thermo.saturation_enthalpy_slope(teff)
        else:
            if getattr(thermo, 'options', None) is not None:
                _count(self._budget, 'native_derivative_fallbacks')
            hsprime = _profile_derivative(x.saturation_enthalpy, teff)
        effprime = (hp / fraction - rate * bypass * dh / fraction**2) / hsprime
        if analytic:
            sp_eff = thermo.enthalpy_slope(teff, x.humidity_in)/x.gas_cp
            sp_gas = thermo.enthalpy_slope(gas, x.humidity_in)/x.gas_cp
        else:
            sp_eff = _profile_derivative(self.sensible_coordinate, teff)
            sp_gas = _profile_derivative(self.sensible_coordinate, gas)
        gasprime = (
            fraction * sp_eff * effprime + rate * bypass * (sint - seff)
        ) / sp_gas
        wt = (thermo.humidity_temperature_slope(gas, h) if analytic else
              _profile_derivative(lambda T: x.humidity_at_temperature_enthalpy(T, h), gas))
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
    wet_solver_options=None,
    _budget=None,
    _initial_region=None,
    _initial_full_region=None,
    _reuse_interface_bracket=False,
) -> WetCoilResult:
    """Internal cooling solve with source-profile moisture and coupled drain.

    Temperatures use one consistent scale supplied by the adapter (K in
    production, Celsius in reference-equation comparisons). Resistances and
    cp are whole-coil SI dry-carrier quantities as in CoilInput.
    """
    budget = _solve_budget(wet_solver_options, _budget)
    options = budget.options
    _count(budget, 'kernel_profile_solves')
    budget.check()
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
    predictor = getattr(getattr(budget, 'fidelity', None), 'name', None) == 'initial'
    minimum_order = 3 if predictor else 6
    if not isinstance(quadrature_order, int) or not minimum_order <= quadrature_order <= 32:
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
        budget.check()
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

    region_seed = _initial_region

    def region(f, bound=False):
        nonlocal region_seed
        try:
            candidate = _Region(
                x,
                f,
                liquid_enthalpy,
                drain=drain_enabled,
                order=quadrature_order,
                boundary_secant=bound,
                sensible_coordinate=sensible_coordinate,
                temperature_from_coordinate=temperature_from_coordinate,
                surface_enthalpy=surface_enthalpy,
                wet_solver_options=options, _budget=budget,
                _initial_region=(
                    _initial_full_region
                    if f == 1.0 and _initial_full_region is not None
                    else region_seed
                ),
            )
            region_seed = candidate
            return candidate
        except WetCoilModelError as exc:
            diagnostics = dict(exc.diagnostics)
            reason = diagnostics.pop("rejection_reason")
            diagnostics.setdefault("onset_margin", margin)
            diagnostics.setdefault("wet_fraction", f)
            raise WetCoilModelError(reason, **diagnostics) from exc

    # One general partial-wet formulation supplies both boundary limits.
    # The specialized all-wet hot surface is not a complementarity test:
    # it can cross the dewpoint while a stable interior dry region remains.
    wet = region(1.0, True)
    full_region = wet
    full_interface_margin = wet.interface_surface - x.dewpoint
    if full_interface_margin <= 0:
        f = 1.0
    else:
        candidates = {1.0: wet} if _reuse_interface_bracket else {}

        def boundary(f):
            _count(budget, 'kernel_wet_fraction_evaluations')
            if f == 0:
                return margin
            if _reuse_interface_bracket and f in candidates:
                return candidates[f].interface_surface - x.dewpoint
            candidate = region(f, True)
            candidates[f] = candidate
            return candidate.interface_surface - x.dewpoint

        # Resolve the physical root relatively near f=0, without an onset
        # threshold, clipping, or interpolation between different models.
        fraction_scale = -margin / (full_interface_margin - margin)
        root_tolerance = max(np.nextafter(0.0, 1.0), min(2e-10, 1e-6 * fraction_scale))
        lower, upper = 0.0, 1.0
        predicted_root = None
        thermo = getattr(x.saturation_enthalpy, '__self__', None)
        front_enabled = getattr(getattr(thermo, 'options', None), 'front_continuation', False)
        slope = getattr(_initial_region, '_front_slope', None)
        if (front_enabled and _initial_region is not None and slope is not None
                and isfinite(slope) and slope > 0 and 0 < _initial_region.f < 1):
            probe = _initial_region.f
            region_seed = _initial_region
            previous_probe = previous_value = None
            for correction in range(4):
                value = boundary(probe)
                _count(budget, 'kernel_front_corrections')
                if value <= 0:
                    lower = probe
                else:
                    upper = probe
                if (abs(value) <= 2e-10 and abs(value/slope) <= root_tolerance/20):
                    predicted_root = probe
                    _count(budget, 'kernel_front_predictor_successes')
                    break
                if previous_probe is not None and probe != previous_probe:
                    new_slope = (value-previous_value)/(probe-previous_probe)
                    if not isfinite(new_slope) or new_slope <= 0:
                        break
                    slope = new_slope
                new_probe = probe-value/slope
                if not lower < new_probe < upper:
                    break
                previous_probe, previous_value, probe = probe, value, new_probe
        if (_reuse_interface_bracket and _initial_region is not None
                and 0.0 < _initial_region.f < 1.0 and predicted_root is None):
            # Rebracket the unchanged interface equation near the accepted
            # prior root. Every sample solves the new coefficients; no old
            # residual or physical gate is accepted as a new result.
            seed = _initial_region.f
            region_seed = _initial_region
            value = boundary(seed)
            direction = -1 if value > 0 else 1
            step = max(1e-6, 0.01 * min(seed, 1.0 - seed))
            while value != 0:
                probe = seed + direction * step
                if probe <= 0.0:
                    probe = 0.0
                elif probe >= 1.0:
                    probe = 1.0
                if boundary(probe) * value <= 0:
                    lower, upper = sorted((seed, probe))
                    break
                step *= 2
            if value == 0:
                lower = upper = seed
        if front_enabled and predicted_root is None:
            _count(budget, 'kernel_front_brent_fallbacks')
        f = (predicted_root if predicted_root is not None else lower if lower == upper else
             brentq(boundary, lower, upper, xtol=root_tolerance))
        wet = candidates[f] if f in candidates else region(f, True)
        if front_enabled:
            neighbors = sorted((abs(v-f), v, r.interface_surface-x.dewpoint)
                               for v, r in candidates.items() if abs(v-f) >= 1e-5)
            if neighbors and abs(neighbors[0][1]-f) > 1e-12:
                _, other_f, other_r = neighbors[0]
                wet._front_slope = ((wet.interface_surface-x.dewpoint)-other_r)/(f-other_f)
            elif slope is not None:
                wet._front_slope = slope
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
        abs(drain_check - wet.drain) > options.energy_tolerance_W
        or abs(mass_check - wet.mass_integral) > options.mass_tolerance_kg_s
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
        regime_formulation="general_partial_wet_boundary_limits",
        full_interface_margin=full_interface_margin,
        interface_margin=wet.interface_surface - x.dewpoint,
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
    if abs(energy_error) > options.energy_tolerance_W or abs(mass_error) > options.mass_tolerance_kg_s:
        reasons.append("inconsistent integral balance")
    diagnostics["rejection_reason"] = ", ".join(reasons) if reasons else None
    budget.check(last_regime="FULLY_WET" if f == 1 else "PARTIALLY_WET", wet_fraction=f)
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
        _full_region=full_region,
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


def _count(budget, key):
    budget.diagnostics[key] = budget.diagnostics.get(key, 0) + 1
