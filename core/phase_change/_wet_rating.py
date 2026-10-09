# SPDX-License-Identifier: GPL-3.0-only
"""Private staged BareTube Rating: approximate search, authoritative strict solve.

Policies are frozen from the accepted wet-performance checkpoints. Approximate
properties and inverse caches stay in their own objects; installed geometry
continues to govern every hydraulic report.
"""
from dataclasses import dataclass, replace
from math import exp, log
from time import monotonic

import numpy as np
from scipy.optimize import brentq, least_squares

from core.heat_transfer._wet_numerics import COARSE, MEDIUM
from core.heat_transfer._wet_thermodynamics import _OptimizedWetThermodynamics
from core.heat_transfer.elmahdy_mitalas import CoilInput
from core.heat_transfer.wet_coil import _solve_profile_candidate, validate_wet_coil, WetCoilModelError
from core.heat_transfer.wet_coil_adapters import InsideWallAdapter, BareTubeAdapter
from core.heat_transfer.wet_coil_solver import WetCoilConvergenceError
from core.heat_transfer.outside_dispatch import DEFAULT_FINNED_HT_PROVIDER
from core.models.simulation import HXSideInput
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change import wet_coil_integration as integration

@dataclass(frozen=True)
class _RatingPolicy:
    """Validated refresh thresholds, not physical admissibility allowances."""
    gas_refresh_K: float = 5.0
    liquid_refresh_K: float = 2.0
    humidity_refresh: float = 0.005
    mass_refresh_log: float = 0.10
    boundary_band_K: float = 0.5
    boundary_fraction: float = 0.02
    coarse: object = COARSE
    medium: object = MEDIUM

    def __post_init__(self):
        for name in ("gas_refresh_K", "liquid_refresh_K", "humidity_refresh",
                     "mass_refresh_log", "boundary_band_K", "boundary_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

@dataclass
class _Snapshot:
    cp: float
    capacity: float
    ri: float
    rd: float
    rw: float
    mass: float
    gas_mean: float
    liquid_mean: float
    humidity_mean: float
    regime: str | None
    generation: int

class _Operation:
    def __init__(self, hx, inside, outside, budget, config, options):
        self.hx, self.inside, self.outside = hx, inside, outside
        self.budget, self.config, self.options = budget, config, options
        self.cap = detect_phase_change_capability(outside.provider)
        self.bundle = replace(hx.bundle, flow_arrangement=options.get("flow_arrangement") or hx.bundle.flow_arrangement_resolved)
        self.md = outside.m_dot / (1 + self.cap.W_in)
        self.stats = budget.diagnostics
        for key in ("coarse_forward_solves", "medium_forward_solves", "strict_forward_solves",
                    "snapshot_refreshes", "boundary_promotions", "admissibility_promotions",
                    "region_solves", "region_iterations", "profile_solves", "profile_points",
                    "wet_fraction_evaluations", "enthalpy_calls", "enthalpy_cache_misses",
                    "inverse_root_calls", "inverse_cache_hits", "caloric_corrections",
                    "approximate_cache_hits", "strict_cache_hits"):
            self.stats[key] = 0
        self.thermo = {f.name: _OptimizedWetThermodynamics(outside.p, self.cap, self.stats, f)
                       for f in (config.coarse, config.medium)}
        self.thermo["strict"] = _OptimizedWetThermodynamics(outside.p, self.cap, self.stats)
        self.history = []
        self.strict_cache = {}
        self.approximate_cache = {}
        self.snapshots = {}
        self.trials = []
        self.last_jacobian = None

    def nearest(self, area, mass):
        if not self.history:
            return None
        return min(self.history, key=lambda item: abs(log(area / item[0])) + abs(log(mass / item[1])))[2]

    def refresh(self, fidelity, mass, seed):
        th = self.thermo[fidelity.name]
        a, b, wi = self.inside, self.outside, self.cap.W_in
        if seed is None:
            ti = (a.T_in + a.T_out) / 2 if a.T_out else a.T_in + 5.0
            tg, wm, tint = (b.T_in + b.T_out) / 2, wi, b.T_out
            twout = 2 * ti - a.T_in
            regime = None
        else:
            tg, ti = (b.T_in + seed.air_out) / 2, (a.T_in + seed.liquid_out) / 2
            wm, tint = (wi + seed.humidity_out) / 2, seed.interface_air
            twout, regime = seed.liquid_out, seed.regime
        ia = InsideWallAdapter(self.bundle, a.provider, mass, a.p,
            fouling_resistance_inside=self.hx.fouling_resistance_inside,
            fouling_resistance_outside=self.hx.fouling_resistance_outside)
        oa = BareTubeAdapter(self.bundle, th, heat_transfer_provider=self.options.get("finned_heat_transfer_provider", DEFAULT_FINNED_HT_PROVIDER))
        cp = th.secant_cp(b.T_in, tint, wi)
        capacity = ia.capacity(a.T_in, twout)
        ri, idiag = ia.evaluate(ti)
        rd, _ = oa.evaluate(tg, wm, self.md, (ri, idiag))
        rw = th.secant_cp(tint, th.dewpoint(wi), wi) * rd
        self.stats["snapshot_refreshes"] += 1
        self.snapshots[fidelity.name] = _Snapshot(cp, capacity, ri, rd, rw, mass, tg, ti, wm, regime, self.stats["snapshot_refreshes"])
        return self.snapshots[fidelity.name]

    def snapshot(self, fidelity, area, mass):
        seed = self.nearest(area, mass)
        snap = self.snapshots.get(fidelity.name)
        c = self.config
        refresh = snap is None or abs(log(mass / snap.mass)) > c.mass_refresh_log
        if snap is not None and seed is not None:
            refresh |= (abs((self.outside.T_in + seed.air_out) / 2 - snap.gas_mean) > c.gas_refresh_K
                        or abs((self.inside.T_in + seed.liquid_out) / 2 - snap.liquid_mean) > c.liquid_refresh_K
                        or abs((self.cap.W_in + seed.humidity_out) / 2 - snap.humidity_mean) > c.humidity_refresh
                        or (snap.regime is not None and seed.regime != snap.regime))
        return self.refresh(fidelity, mass, seed) if refresh else snap

    def strict(self, area, mass):
        self.budget.check(required_area_scale=area, inside_mass_flow=mass)
        key = (float(area), float(mass))
        if key in self.strict_cache:
            self.stats["strict_cache_hits"] += 1
            return self.strict_cache[key]
        seed = self.nearest(area, mass)
        initial = None if seed is None else (seed, np.asarray([0.0]))
        # Never carry a low-order grid or a predictor inverse cache into FINAL.
        if seed is not None:
            initial = (replace(seed, diagnostics=dict(seed.diagnostics, quadrature_order=10)), np.asarray([0.0]))
        side = HXSideInput(self.inside.provider, mass, self.inside.T_in, self.inside.p,
                           phase_change_mode=self.inside.phase_change_mode)
        self.stats["strict_forward_solves"] += 1
        started = monotonic()
        physical = integration.forward_wet_process(self.hx, side, self.outside,
            wet_solver_options=self.budget.options, _budget=self.budget,
            _initial_state=initial, _thermodynamics=self.thermo["strict"], _area_scale=area,
            flow_arrangement=self.options.get("flow_arrangement"),
            finned_heat_transfer_provider=self.options.get("finned_heat_transfer_provider", DEFAULT_FINNED_HT_PROVIDER))
        self.trials.append(dict(stage="strict", area=area, mass=mass, elapsed_s=monotonic()-started))
        self.strict_cache[key] = (side, physical)
        self.history.append((area, mass, physical[0]))
        return side, physical

    def approximate(self, area, mass, fidelity):
        self.budget.check(required_area_scale=area, inside_mass_flow=mass)
        snap = self.snapshot(fidelity, area, mass)
        key = (fidelity.name, snap.generation, float(area), float(mass))
        if key in self.approximate_cache:
            self.stats["approximate_cache_hits"] += 1
            return self.approximate_cache[key]
        th, seed = self.thermo[fidelity.name], self.nearest(area, mass)
        cp, cw = snap.cp, snap.capacity * mass / snap.mass
        x = CoilInput(self.outside.T_in, self.inside.T_in, self.cap.W_in, self.md,
            cw, cp, th.enthalpy(self.outside.T_in, self.cap.W_in), th.dewpoint(self.cap.W_in),
            snap.rd / area, snap.rw / area, lambda T: snap.ri / area,
            th.saturation_enthalpy, th.saturation_temperature, th.humidity)
        child = fidelity.budget(self.budget)
        self.stats[fidelity.name + "_forward_solves"] += 1
        started = monotonic()
        try:
            for correction in range(12):
                r = _solve_profile_candidate(x, liquid_enthalpy=th.condensate_enthalpy,
                    saturation_humidity=th.saturation_humidity, quadrature_order=fidelity.quadrature,
                    sensible_coordinate=lambda T: x.air_in + (th.enthalpy(T, x.humidity_in)-x.gas_h_in)/cp,
                    temperature_from_coordinate=lambda u: th.temperature(x.gas_h_in+cp*(u-x.air_in), x.humidity_in),
                    wet_solver_options=child.options, _budget=child,
                    _initial_region=None if seed is None else seed._region,
                    _initial_full_region=None if seed is None else seed._full_region,
                    _reuse_interface_bracket=True)
                try:
                    validate_wet_coil(r)
                    break
                except WetCoilModelError as exc:
                    if exc.diagnostics["rejection_reason"] != "negative wet driving force" or correction == 11:
                        raise
                    # Gas secant cp also maps the nonlinear sensible enthalpy
                    # coordinate. Freezing it can break source-profile moisture
                    # admissibility. Correct this caloric mapping while leaving
                    # the expensive transport/inside snapshot fixed.
                    self.stats["caloric_corrections"] += 1
                    cp = th.secant_cp(self.outside.T_in, r.interface_air, self.cap.W_in)
                    rw = th.secant_cp(r.interface_air, th.dewpoint(self.cap.W_in), self.cap.W_in) * snap.rd / area
                    x = replace(x, gas_cp=cp, wet_air_resistance=rw)
                    seed = r
        except WetCoilModelError as exc:
            # Physical gates are never relaxed to rescue a cheap trial.
            self.stats["admissibility_promotions"] += 1
            self.trials.append(dict(stage=fidelity.name, area=area, mass=mass,
                elapsed_s=monotonic()-started, promoted=True, reason=exc.diagnostics))
            return self.approximate(area, mass, self.config.medium) if fidelity.name == "coarse" else self.strict(area, mass)[1][0]
        self.trials.append(dict(stage=fidelity.name, area=area, mass=mass,
            elapsed_s=monotonic()-started, regime=r.regime))
        self.budget.check(last_regime=r.regime, wet_fraction=r.wet_fraction)
        self.history.append((area, mass, r))
        band, fraction = self.config.boundary_band_K, self.config.boundary_fraction
        near = (abs(r.onset_margin) < band
                or abs(r.diagnostics.get("full_interface_margin", float("inf"))) < band
                or (r.regime != "DRY" and (r.wet_fraction < fraction or 1-fraction < r.wet_fraction < 1)))
        if near:
            self.stats["boundary_promotions"] += 1
            return self.approximate(area, mass, self.config.medium) if fidelity.name == "coarse" else self.strict(area, mass)[1][0]
        self.approximate_cache[key] = r
        return r

    def solve(self, fidelity, z, tolerance):
        known_mass = self.inside.m_dot is not None
        def result(v):
            area, mass = exp(v[0]), self.inside.m_dot if known_mass else exp(v[1])
            return self.strict(area, mass)[1][0] if fidelity is None else self.approximate(area, mass, fidelity)
        if known_mass:
            samples = {}
            class Done(Exception):
                pass
            def residual(v):
                r = result([v])
                value = r.air_out-self.outside.T_out
                samples[v] = value
                if abs(value) <= tolerance:
                    raise Done(v)
                return value
            try:
                center, value = z[0], residual(z[0])
                direction = 1 if value > 0 else -1
                last = center
                for n in range(26):
                    distance = 0.08 * 2**min(n, 3) + max(0, n-3) * log(2)
                    probe = center + direction * distance
                    v = residual(probe)
                    if v * value <= 0:
                        return np.array([brentq(residual, *sorted((last, probe)), xtol=2e-8, rtol=1e-12)])
                    last, value = probe, v
                raise WetCoilConvergenceError("Wet Rating area bracket not found")
            except Done as solved:
                return np.array([solved.args[0]])
        class Done(Exception):
            pass
        def residual(v):
            self.stats["joint_solver_evaluations"] += 1
            r = result(v)
            values = np.array([r.air_out-self.outside.T_out, r.liquid_out-self.inside.T_out])
            if np.max(np.abs(values)) <= tolerance:
                raise Done(np.array(v, copy=True))
            return values
        matrix_state = None
        def jac(v):
            nonlocal matrix_state
            values = residual(v)
            if matrix_state is not None:
                old_v, old_values, matrix = matrix_state
                delta = v-old_v
                norm = np.dot(delta, delta)
                if norm > 0:
                    matrix += np.outer(values-old_values-matrix@delta, delta)/norm
            elif self.last_jacobian is not None:
                # Predictor sensitivities seed a correction; only new strict
                # outlet residuals determine acceptance and Broyden updates.
                matrix = np.array(self.last_jacobian, copy=True)
            else:
                columns = []
                for i in range(2):
                    shifted = np.array(v, copy=True)
                    shifted[i] += 1e-3
                    columns.append((residual(shifted)-values)/1e-3)
                matrix = np.column_stack(columns)
            matrix_state = (np.array(v, copy=True), values, matrix)
            self.last_jacobian = np.array(matrix, copy=True)
            return matrix
        try:
            fit = least_squares(residual, z, jac=jac, bounds=self.joint_bounds,
                xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=30)
        except Done as solved:
            return solved.args[0]
        if np.max(np.abs(fit.fun)) > tolerance:
            raise WetCoilConvergenceError("Wet Rating joint outlet solve did not converge")
        return fit.x

    def run(self):
        th = self.thermo["strict"]
        known = self.inside.m_dot is not None
        if not known:
            sensible = self.md * (th.enthalpy(self.outside.T_in, self.cap.W_in)-th.enthalpy(self.outside.T_out, self.cap.W_in))
            mass = sensible / (self.inside.provider.at((self.inside.T_in+self.inside.T_out)/2, self.inside.p).cp
                               * (self.inside.T_out-self.inside.T_in))
        else:
            mass = self.inside.m_dot
        z = np.array([0.0] if known else [0.0, log(mass)])
        self.joint_bounds = (z-10, z+10)
        z = self.solve(self.config.coarse, z, self.config.coarse.outlet_K)
        # A refreshed MEDIUM constitutive state corrects the frozen predictor.
        self.refresh(self.config.medium, mass if known else exp(z[1]), self.nearest(exp(z[0]), mass if known else exp(z[1])))
        z = self.solve(self.config.medium, z, self.config.medium.outlet_K)
        # The strict kernel recomputes all coefficients and physical gates.
        z = self.solve(None, z, self.budget.options.outlet_temperature_tolerance_K)
        area, mass = exp(z[0]), self.inside.m_dot if known else exp(z[1])
        result = integration._assemble_rating(self.hx, self.inside, self.outside, area_scale=area,
            mass=mass, trial=self.strict, cache=self.strict_cache, budget=self.budget, **self.options)
        result.wet_coil_diagnostics["numerical_rating"] = dict(config=self.config,
            trials=self.trials, statistics=dict(self.stats),
            enthalpy_cache_hits=self.stats["enthalpy_calls"]-self.stats["enthalpy_cache_misses"],
            dry_backend_updates=sum(len(t.evaluator._dry_enthalpy_by_temperature) for t in self.thermo.values()),
            vapor_property_misses=sum(len(t.evaluator._water_vapor_enthalpy_by_temperature) for t in self.thermo.values()),
            dewpoint_evaluations=sum(t._dewpoint.cache_info().misses for t in self.thermo.values()),
            final_solver="strict production forward_wet_process",
            final_grid=10, approximate_inverse_caches_shared_with_final=False,
            kernel_thermodynamics={name: dict(t.statistics) for name,t in self.thermo.items()})
        result.wet_coil_diagnostics["rating_total_area_trials"] = sum(
            self.stats[k] for k in ("coarse_forward_solves", "medium_forward_solves", "strict_forward_solves"))
        return result


def _rate_optimized(hx, inside, outside, *, budget, **options):
    integration._validate_rating_inputs(inside, outside)
    return _Operation(hx, inside, outside, budget, _RatingPolicy(), options).run()
