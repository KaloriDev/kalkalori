"""Independent ODE and dry-topology limits of the uniform-source balance."""
import numpy as np
import pytest
from scipy.integrate import solve_bvp, simpson

from core.heat_transfer.ntu import effectiveness_ntu
from core.phase_change.bulk_mean_closure import uniform_source_bulk_means


@pytest.mark.parametrize("flow", ["counterflow", "cocurrentflow", "crossflow"])
@pytest.mark.parametrize("side", ["inside", "outside"])
@pytest.mark.parametrize("capacities", [(200., 300.), (300., 200.), (200., 200.)])
@pytest.mark.parametrize("ntu", [1e-5, 0.5, 5., 100.])
def test_dry_limit_uses_existing_physical_topology(flow, side, capacities, ntu):
    h, c = capacities
    g = ntu * min(h, c)
    r = uniform_source_bulk_means(UA=g, C_gas=h, C_other=c,
        T_gas_in=360., T_other_in=290., gas_film_fraction=.7,
        Q_latent=0., flow_arrangement=flow, gas_side=side)
    expected = effectiveness_ntu(h, c, g, flow_arrangement=flow,
        C_inside=h if side == "inside" else c,
        C_outside=h if side == "outside" else c) * min(h, c) * 70
    assert r.heat_total == pytest.approx(expected, rel=2e-11)
    assert g * (r.gas_temperature - r.other_temperature) == pytest.approx(expected, rel=2e-9)
    assert 290 <= r.other_temperature <= r.gas_temperature <= 360


@pytest.mark.parametrize("flow", ["counterflow", "cocurrentflow"])
@pytest.mark.parametrize("h,c", [(200., 300.), (300., 200.), (200., 200.), (200., 200.000001)])
@pytest.mark.parametrize("latent", [0., 1e-5, 2000.])
def test_uniform_source_against_independent_boundary_value_problem(flow, h, c, latent):
    g, b = 800., .7
    sign = -1 if flow == "counterflow" else 1
    def ode(x, y):
        sensible = g * (y[0] - y[1]) - (1-b) * latent
        return np.array([-sensible/h, sign*(sensible+latent)/c])
    def boundaries(left, right):
        return [left[0]-360, (right[1] if sign < 0 else left[1])-290]
    x = np.linspace(0, 1, 101)
    solved = solve_bvp(ode, boundaries, x, np.array([360-30*x, 320-30*x]),
                       tol=1e-9, max_nodes=10000)
    assert solved.success
    x = np.linspace(0, 1, 1001)
    y = solved.sol(x)
    r = uniform_source_bulk_means(UA=g, C_gas=h, C_other=c,
        T_gas_in=360, T_other_in=290, gas_film_fraction=b,
        Q_latent=latent, flow_arrangement=flow)
    assert r.heat_total == pytest.approx(h*(360-y[0,-1])+latent, abs=1e-7)
    assert r.gas_temperature == pytest.approx(simpson(y[0], x=x), abs=1e-8)
    assert r.other_temperature == pytest.approx(simpson(y[1], x=x), abs=1e-8)
