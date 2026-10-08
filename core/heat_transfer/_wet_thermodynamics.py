# SPDX-License-Identifier: GPL-3.0-only
"""Operation-local exact states for the optimized BareTube numerical path.

Configured dry-carrier EOS, IF97 saturation definitions and reference datums
are unchanged. Approximate inverse roots are isolated by fidelity; strict
inversion delegates to the reference implementation with its original xtol.
"""
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite

from scipy.optimize import brentq

from core.heat_transfer._wet_if97 import IF97WaterCache, supported
from core.heat_transfer._wet_numerics import _WetNumerics
from core.heat_transfer.wet_coil_adapters import WetGasThermodynamics
from core.properties.water import (
    _saturation_enthalpies_at_temperature,
    water_saturation_liquid_enthalpy,
    water_saturation_vapor_enthalpy,
)


@dataclass
class _WetThermoPoint:
    temperature: float
    pressure: float
    dry_enthalpy: float | None = None
    vapor_enthalpy: float | None = None
    liquid_enthalpy: float | None = None
    saturation_humidity: float | None = None
    dry_cp: float | None = None
    vapor_slope: float | None = None
    pressure_slope: tuple | None = None

    def enthalpy(self, humidity):
        return self.dry_enthalpy + humidity * self.vapor_enthalpy


class _OptimizedWetThermodynamics(WetGasThermodynamics):
    def __init__(self, pressure, capability, stats=None, fidelity=None, *, options=None):
        super().__init__(pressure, capability, _reuse_inverse_state=True)
        self.options = options or _WetNumerics()
        if fidelity is not None and stats is None:
            stats = dict(enthalpy_calls=0, enthalpy_cache_misses=0,
                         inverse_cache_hits=0, inverse_root_calls=0)
        self.stats, self.fidelity = stats, fidelity
        self.statistics = dict(state_requests=0, state_cache_hits=0, state_cache_misses=0,
                               enthalpy_calls=0, humidity_calls=0, saturation_calls=0, liquid_calls=0)
        self.states, self.children = {}, []
        self.water = IF97WaterCache(self.statistics)
        self._dewpoint = lru_cache(maxsize=32)(super().dewpoint)
        self._enthalpy = lru_cache(maxsize=16384)(self._evaluate_enthalpy)
        self._saturation_enthalpy = lru_cache(maxsize=16384)(
            lambda T: self.enthalpy(T, self.saturation_humidity(T)))

    def state(self, T, *, caloric=True):
        self.statistics['state_requests'] += 1
        point = self.states.get(T)
        if point is None:
            self.statistics['state_cache_misses'] += 1
            point = self.states[T] = _WetThermoPoint(T, self.pressure)
        else:
            self.statistics['state_cache_hits'] += 1
        if caloric and point.dry_enthalpy is None:
            self._water(point)
            backend = self.evaluator._dry_state
            backend.update(self.evaluator._dry_pt_inputs, self.pressure, T)
            point.dry_enthalpy, point.dry_cp = backend.hmass(), backend.cpmass()
            self.evaluator._dry_enthalpy_by_temperature[T] = point.dry_enthalpy
            self.evaluator._water_vapor_enthalpy_by_temperature[T] = point.vapor_enthalpy
        return point

    def _water(self, point):
        if point.vapor_enthalpy is not None:
            return
        self.water.count('water_bundle_evaluations')
        if supported(point.temperature):
            pair = self.water.get(point.temperature, values=True)
            point.liquid_enthalpy, point.vapor_enthalpy = pair.liquid, pair.vapor
        else:
            self.water.count('native_water_fallbacks')
            pair = _saturation_enthalpies_at_temperature(point.temperature)
            if pair is not None:
                point.liquid_enthalpy, point.vapor_enthalpy = pair
            else:
                point.vapor_enthalpy = water_saturation_vapor_enthalpy(T=point.temperature)

    def _evaluate_enthalpy(self, T, W):
        if self.stats is not None:
            self.stats['enthalpy_cache_misses'] += 1
        if not isfinite(W) or W < 0:
            raise ValueError('Humidity must be nonnegative and finite')
        return self.state(T).enthalpy(W)

    def enthalpy(self, T, W):
        self.statistics['enthalpy_calls'] += 1
        if self.stats is not None:
            self.stats['enthalpy_calls'] += 1
        return self._enthalpy(T, W)

    def humidity(self, T, h):
        self.statistics['humidity_calls'] += 1
        point = self.state(T)
        return (h - point.dry_enthalpy) / point.vapor_enthalpy

    def condensate_enthalpy(self, T):
        self.statistics['liquid_calls'] += 1
        point = self.state(T, caloric=False)
        self._water(point)
        if point.liquid_enthalpy is None:
            point.liquid_enthalpy = water_saturation_liquid_enthalpy(T=T)
        return point.liquid_enthalpy

    def saturation_humidity(self, T):
        self.statistics['saturation_calls'] += 1
        point = self.state(T, caloric=False)
        if point.saturation_humidity is None:
            if not supported(T):
                # Execute the native definition without its global self-keyed LRU.
                point.saturation_humidity = WetGasThermodynamics.saturation_humidity.__wrapped__(self, T)
                self.water.count('native_equilibrium_fallbacks')
            elif T >= self.saturation_upper + 1e-3:
                point.saturation_humidity = float('inf')
            else:
                pressure = self.water.get(T).pressure
                c = self.capability
                point.saturation_humidity = (c.M_condensable / c.M_dry) * pressure / (self.pressure - pressure)
        return point.saturation_humidity

    def saturation_enthalpy(self, T):
        return self._saturation_enthalpy(T)

    def dewpoint(self, W):
        return self._dewpoint(W)

    def derivative_bundle(self, T):
        if not supported(T):
            raise ValueError('Analytic wet derivatives require the supported IF97 Region1/2 domain')
        point = self.state(T)
        if point.vapor_slope is None:
            water = self.water.get(T, values=True)
            dp, point.vapor_slope = self.water.derivatives(water)
            point.pressure_slope = water.pressure, dp
            self.water.count('analytic_vapor_derivatives')
        return point

    def enthalpy_slope(self, T, W):
        point = self.derivative_bundle(T)
        return point.dry_cp + W * point.vapor_slope

    def saturation_enthalpy_slope(self, T):
        point = self.derivative_bundle(T)
        W = self.saturation_humidity(T)
        pressure, dp = point.pressure_slope
        c = self.capability
        dw = c.M_condensable / c.M_dry * self.pressure * dp / (self.pressure - pressure)**2
        return point.dry_cp + W * point.vapor_slope + dw * point.vapor_enthalpy

    def humidity_temperature_slope(self, T, h):
        point = self.derivative_bundle(T)
        W = (h - point.dry_enthalpy) / point.vapor_enthalpy
        return -(point.dry_cp + W * point.vapor_slope) / point.vapor_enthalpy

    def _inverse(self, h, roots, function, upper):
        """Approximate inverse used only by this object's predictor fidelity."""
        index = bisect_left(roots, (h,))
        if index < len(roots) and roots[index][0] == h:
            self.stats['inverse_cache_hits'] += 1
            return roots[index][1]
        self.stats['inverse_root_calls'] += 1
        lower = self.lower if index == 0 else roots[index - 1][1]
        high = upper if index == len(roots) else roots[index][1]
        if function(lower) > h:
            lower = self.lower
        if function(high) < h:
            high = upper
        value = brentq(lambda T: function(T) - h, lower, high, xtol=self.fidelity.inverse_temperature_K)
        if len(roots) >= 4096:
            roots.clear()
            index = 0
        roots.insert(index, (h, value))
        return value

    def _count_inverse(self, h, roots):
        index = bisect_left(roots, (h,))
        hit = index < len(roots) and roots[index][0] == h
        self.water.count('inverse_requests')
        self.water.count('inverse_exact_hits' if hit else 'inverse_brent_roots')
        if self.stats is not None and self.fidelity is None:
            self.stats['inverse_cache_hits' if hit else 'inverse_root_calls'] += 1

    def saturation_temperature(self, h):
        self._count_inverse(h, self._saturation_inverse_roots)
        if self.fidelity is None:
            return super().saturation_temperature(h)
        return self._inverse(h, self._saturation_inverse_roots,
                             self.saturation_enthalpy, self.saturation_upper)

    def temperature(self, h, W):
        self._count_inverse(h, self._temperature_inverse_roots.get(W, []))
        if self.fidelity is None:
            return super().temperature(h, W)
        return self._inverse(h, self._temperature_inverse_roots.setdefault(W, []),
                             lambda T: self.enthalpy(T, W), 640.0)
