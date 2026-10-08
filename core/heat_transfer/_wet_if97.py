# SPDX-License-Identifier: GPL-3.0-only
"""Shared installed IF97 value/derivative intermediates, owned by one operation.

Values retain native Region4 and Region1/2 arithmetic and coefficient order.
Gradients reuse those intermediates; no pressure-path or datum substitution.
Only a bounded hot set retains arrays; scalar exact values remain in bundles.
"""
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
from iapws import iapws97 as if97
from core.properties.water import _validate_saturation_temperature


def _region4_coefficients():
    # Read the installed definitions' constants, rather than maintain another
    # coefficient set. Unknown future function layouts use the native path.
    def coefficients(fn):
        return next((n for n in fn.__code__.co_consts
                     if isinstance(n,tuple) and len(n)==11 and n[0]==0),None)
    pressure,temperature = coefficients(if97._PSat_T),coefficients(if97._TSat_P)
    return pressure if pressure == temperature else None


N = _region4_coefficients()
C = if97.Const
IDEAL_NJ = C.Region2_cp0_no*C.Region2_cp0_Jo
IDEAL_J = C.Region2_cp0_Jo-1
if hasattr(C,'Region2_n'):
    RESIDUAL_NJ,RESIDUAL_I,RESIDUAL_J = C.Region2_n*C.Region2_Lj,C.Region2_Li,C.Region2_Lj_less_1
else:
    RESIDUAL_NJ,RESIDUAL_I,RESIDUAL_J = C.Region2_nr_Jr_product,C.Region2_Ir,C.Region2_Jr_less_1


def supported(T):
    # The native pressure roundtrip can select Region3 at exactly 623.15 K.
    if N is None or not 273.16 <= T < 623.15:
        return False
    # It can also cross that pressure boundary a few ulps below the nominal
    # temperature boundary. Probe the actual native arithmetic in this narrow
    # boundary strip; common-domain requests do not repeat pressure work.
    return T < 623.149 or pressure_terms(T)[0]/1e6 <= if97.Ps_623


def pressure_terms(T):
    _validate_saturation_temperature(T)
    n = N
    theta = T+n[9]/(T-n[10])
    a = theta**2+n[1]*theta+n[2]
    b = n[3]*theta**2+n[4]*theta+n[5]
    c = n[6]*theta**2+n[7]*theta+n[8]
    radical = (b**2-4*a*c)**.5
    denominator = -b+radical
    ratio = 2*c/denominator
    pressure_pa = float(ratio**4)*1e6
    return pressure_pa,(theta,a,b,c,radical,denominator,ratio)


def temperature_terms(p):
    n = N
    beta = p**.25
    e = beta**2+n[3]*beta+n[6]
    f = n[1]*beta**2+n[4]*beta+n[7]
    g = n[2]*beta**2+n[5]*beta+n[8]
    radical = (f**2-4*e*g)**.5
    denominator = -f-radical
    d = 2*g/denominator
    radical_t = ((n[10]+d)**2-4*(n[9]+n[10]*d))**.5
    t = (n[10]+d-radical_t)/2
    return t,(beta,e,f,g,radical,denominator,d,radical_t)


@dataclass(slots=True)
class IF97WaterBundle:
    temperature: float
    pressure: float
    saturation_temperature: float | None = None
    liquid: float | None = None
    vapor: float | None = None
    pressure_derivative: float | None = None
    vapor_derivative: float | None = None


class IF97WaterCache:
    """Exact scalar bundles; at most 512 hot pressure/Region2 term records."""
    def __init__(self,stats,capacity=512):
        self.stats,self.capacity = stats,capacity
        self.bundles = {}
        self.terms = OrderedDict()

    def count(self,key):
        self.stats[key] = self.stats.get(key,0)+1

    def store(self,T,terms):
        self.terms[T] = terms
        self.terms.move_to_end(T)
        if len(self.terms)>self.capacity:
            self.terms.popitem(last=False)
        self.stats['water_terms_peak'] = max(self.stats.get('water_terms_peak',0),len(self.terms))

    def get(self,T,*,values=False):
        point = self.bundles.get(T)
        if point is None:
            p,terms = pressure_terms(T)
            if p/1e6 > if97.Ps_623:
                raise NotImplementedError('native pressure-based Region3 boundary')
            point = self.bundles[T] = IF97WaterBundle(T,p)
            self.count('if97_pressure_evaluations')
            self.store(T,dict(pressure=terms))
        if values and point.vapor is None:
            self.values(point)
        return point

    def intermediates(self,point):
        T = point.temperature
        terms = self.terms.get(T)
        if terms is None:
            _,pressure = pressure_terms(T)
            terms = dict(pressure=pressure)
            self.count('if97_term_rebuilds')
            self.store(T,terms)
        else:
            self.terms.move_to_end(T)
            self.count('if97_term_hits')
        return terms

    def values(self,point):
        terms = self.intermediates(point)
        p = point.pressure/1e6
        t,tterms = temperature_terms(p)
        tau = 1386/t
        pi = p/16.53
        gt = np.sum(C.Region1_n*C.Region1_Lj*(7.1-pi)**C.Region1_Li
                    *(tau-1.222)**C.Region1_Lj_less_1)
        liquid = float(tau*gt*if97.R*t)*1000.
        tau = 540/t
        ideal = IDEAL_NJ*tau**IDEAL_J
        residual = RESIDUAL_NJ*p**RESIDUAL_I*(tau-.5)**RESIDUAL_J
        got,grt = np.sum(ideal),np.sum(residual)
        vapor = float(tau*(got+grt)*if97.R*t)*1000.
        point.saturation_temperature,point.liquid,point.vapor = t,liquid,vapor
        terms.update(temperature=tterms,ideal=ideal,residual=residual)
        self.count('if97_pair_evaluations')

    def derivatives(self,point):
        if point.vapor_derivative is not None:
            return point.pressure_derivative,point.vapor_derivative
        terms = self.intermediates(point)
        if 'ideal' not in terms:
            # Eviction removes expensive arrays, never an exact scalar value.
            self.values(point)
            terms = self.terms[point.temperature]
        n = N
        theta,a,b,c,radical,denominator,ratio = terms['pressure']
        dtheta = 1-n[9]/(point.temperature-n[10])**2
        da,db,dc = (2*theta+n[1])*dtheta,(2*n[3]*theta+n[4])*dtheta,(2*n[6]*theta+n[7])*dtheta
        dradical = (2*b*db-4*(da*c+a*dc))/(2*radical)
        ddenominator = -db+dradical
        dratio = 2*(dc*denominator-c*ddenominator)/denominator**2
        dp_pa = 4*ratio**3*dratio*1e6
        p,dp = point.pressure/1e6,dp_pa/1e6
        beta,e,f,g,radical,denominator,d,radical_t = terms['temperature']
        dbeta = .25*p**(-.75)*dp
        de,df,dg = (2*beta+n[3])*dbeta,(2*n[1]*beta+n[4])*dbeta,(2*n[2]*beta+n[5])*dbeta
        dradical = (2*f*df-4*(de*g+e*dg))/(2*radical)
        dd = 2*(dg*denominator-g*(-df-dradical))/denominator**2
        dt = (dd-(2*(n[10]+d)*dd-4*n[10]*dd)/(2*radical_t))/2
        t = point.saturation_temperature
        tau,dtau = 540/t,-540*dt/t**2
        ideal_slope = np.sum(terms['ideal']*IDEAL_J)*dtau/tau
        residual_slope = np.sum(terms['residual']*(RESIDUAL_I*dp/p+RESIDUAL_J*dtau/(tau-.5)))
        point.pressure_derivative = dp_pa
        point.vapor_derivative = float(1000*if97.R*540*(ideal_slope+residual_slope))
        self.count('if97_derivative_evaluations')
        return point.pressure_derivative,point.vapor_derivative
