"""Public equation and independent port-balance checks for equal-UA sections."""
import math

import numpy as np
import pytest

from core.geometry._thermal_topology import CounterCrossflowTopology
from core.heat_transfer.counter_crossflow import (
    effectiveness_counter_crossflow as forward,
    ntu_counter_crossflow as inverse, counter_crossflow_ceiling as ceiling,
)
from core.heat_transfer.ntu import effectiveness_ntu, ntu_from_effectiveness


@pytest.mark.parametrize("mixed_min", [True, False])
@pytest.mark.parametrize("cr", [.01, .7, 1.])
def test_single_section_preserves_legacy_evaluation(cr, mixed_min):
    ci, co = (1/cr, 1.) if mixed_min else (1., 1/cr)
    kw = dict(C_inside=ci, C_outside=co)
    for x in (.01, .1, 1., 5.):
        legacy = effectiveness_ntu(ci, co, x, flow_arrangement="crossflow", **kw)
        assert effectiveness_ntu(ci, co, x, flow_arrangement=CounterCrossflowTopology(1), **kw) == legacy
        assert forward(x, cr, 1, outside_is_cmin=mixed_min) == pytest.approx(legacy, abs=2e-14)
        assert ntu_from_effectiveness(legacy, ci, co, flow_arrangement=CounterCrossflowTopology(1), **kw) == ntu_from_effectiveness(legacy, ci, co, flow_arrangement="crossflow", **kw)


@pytest.mark.parametrize("mixed_min", [True, False])
@pytest.mark.parametrize("cr", [0., 1e-12, .1, .7, 1-1e-12, 1.])
@pytest.mark.parametrize("n", [1, 2, 8, 64])
def test_roundtrip_limits_monotonicity_and_ceiling(cr, n, mixed_min):
    old = -1.
    limit = ceiling(cr, n, outside_is_cmin=mixed_min)
    for x in (1e-8, .001, .1, 1., 5., 20., 100.):
        e = forward(x, cr, n, outside_is_cmin=mixed_min)
        assert old <= e <= limit
        old = e
        if e < limit-1e-9:
            assert inverse(e, cr, n, outside_is_cmin=mixed_min) == pytest.approx(x, rel=1e-7)
    for bad in (limit, limit+.01, -1., math.nan, math.inf):
        with pytest.raises(ValueError, match="unreachable"):
            inverse(bad, cr, n, outside_is_cmin=mixed_min)
    assert forward(1e6, cr, n, outside_is_cmin=mixed_min) == pytest.approx(limit, abs=2e-14)


@pytest.mark.parametrize("mixed_min", [True, False])
@pytest.mark.parametrize("cr", [.1, .7, 1.])
def test_n_progression_and_countercurrent_limit(cr, mixed_min):
    for x in (.01, .1, 1., 5., 20.):
        vals = [forward(x, cr, n, outside_is_cmin=mixed_min) for n in (1, 2, 4, 8, 64, 1024, 100000)]
        t = -math.expm1(-(1-cr)*x)
        counter = x/(1+x) if cr == 1 else t/(1-cr+cr*t)
        assert all(a <= b+1e-13 for a, b in zip(vals, vals[1:]))
        assert vals[-1] <= counter+1e-13
        assert vals[-1] == pytest.approx(counter, abs=1e-8)


@pytest.mark.parametrize("mixed_min", [True, False])
@pytest.mark.parametrize("hot_min", [True, False])
@pytest.mark.parametrize("cr,n", [(.1,2), (.7,2), (.7,8), (1.,3)])
def test_independent_simultaneous_port_balances(cr, n, mixed_min, hot_min):
    # Each cell uses only its published local law. Solve all connected ports
    # simultaneously, independently of the production series composition.
    x = 2/n
    ep = (-math.expm1(math.expm1(-cr*x)/cr) if mixed_min else
          -math.expm1(cr*math.expm1(-x))/cr)
    ch, cc = (1.,1/cr) if hot_min else (1/cr,1.)
    a = np.zeros((2*(n+1), 2*(n+1))); b = np.zeros(2*(n+1))
    h = lambda j: j
    c = lambda j: n+1+j
    a[0,h(0)] = 1; b[0] = 1; a[1,c(n)] = 1
    for j in range(n):
        r = 2+2*j
        a[r,h(j+1)] = 1; a[r,h(j)] = -(1-ep/ch); a[r,c(j+1)] = -ep/ch
        a[r+1,c(j)] = 1; a[r+1,h(j)] = -ep/cc; a[r+1,c(j+1)] = -(1-ep/cc)
    ports = np.linalg.solve(a,b)
    qh, qc = ch*(1-ports[h(n)]), cc*ports[c(0)]
    assert qh == pytest.approx(qc, abs=1e-13)
    assert qh == pytest.approx(forward(2.,cr,n,outside_is_cmin=mixed_min), abs=1e-13)


def test_prescribed_dimensionless_diagnostic_without_calibration():
    target, cr = 2/3, .7
    counter = ntu_from_effectiveness(target, 1., 1/cr)
    n2 = inverse(target, cr, 2, outside_is_cmin=True)
    assert n2 == pytest.approx(1.6720745671944721, abs=2e-13)
    assert counter/n2 == pytest.approx(.9369670437492145, abs=2e-13)
    assert 40/n2 == pytest.approx(23.922378095323165, abs=2e-12)
    assert inverse(target, cr, 1, outside_is_cmin=True) == pytest.approx(2.0935162780018226, abs=2e-13)
    assert counter == pytest.approx(1.5666787641524522, abs=2e-13)


@pytest.mark.parametrize("n", [0, -1, 2.5, True])
def test_fractional_or_invalid_sections_fail(n):
    with pytest.raises(ValueError, match="positive integer"):
        forward(1., .7, n, outside_is_cmin=True)


@pytest.mark.parametrize("x,cr", [(-1.,.7), (math.nan,.7), (math.inf,.7), (1.,-1.), (1.,1.1)])
def test_invalid_numeric_inputs_fail(x, cr):
    with pytest.raises(ValueError):
        forward(x, cr, 2, outside_is_cmin=True)


def test_descriptor_requires_physical_side_identity():
    with pytest.raises(ValueError, match="capacity rates"):
        effectiveness_ntu(1., 2., 1., flow_arrangement=CounterCrossflowTopology(2))
