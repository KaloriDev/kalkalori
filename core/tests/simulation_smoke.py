# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""
Smoke test for BareTubeHeatExchanger.simulate (v0.5.3: wall/length-corrected
iterative thermal state by default; iterate=False uncorrected escape hatch).

Runs WITHOUT CoolProp / IAPWS / PsychroLib: it uses a tiny linear
temperature-dependent provider plus ConstantPropertyProvider so the default
corrected path, the iterate=False escape hatch, convergence diagnostics,
non-convergence handling, and surface-margin derating can all be exercised
in a bare Python environment.

Note (v0.5.3): a ``ConstantPropertyProvider`` no longer forces a single-pass
shortcut under the default ``iterate=True`` path -- constant bulk properties
do not make the wall-temperature correction trivial, so genuine iteration
(usually a handful of passes) is still required. Only ``iterate=False``
remains a guaranteed single pass.

Run:
    python -m core.tests.simulation_smoke
"""

from __future__ import annotations

from dataclasses import dataclass

from core.geometry.tube import BareTube
from core.geometry.bundle import TubeBundle
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput


@dataclass(frozen=True)
class LinearGasProvider:
    """Toy provider with linear T-dependence, evaluated about a reference T.

    Not physical: it only needs to make properties move with temperature so the
    mean-property iteration produces a visibly different result from an
    inlet-only evaluation. Ignores pressure.
    """

    rho0: float
    mu0: float
    k0: float
    cp0: float
    T_ref: float = 300.0
    rho_slope: float = -1.0 / 300.0   # rho falls with T (ideal-gas-like)
    mu_slope: float = 1.5e-3
    k_slope: float = 1.5e-3
    cp_slope: float = 2.0e-4

    def at(self, T: float, p: float) -> FluidTransportProperties:
        dT = T - self.T_ref
        rho = max(self.rho0 * (1.0 + self.rho_slope * dT), 1e-3)
        mu = max(self.mu0 * (1.0 + self.mu_slope * dT), 1e-7)
        k = max(self.k0 * (1.0 + self.k_slope * dT), 1e-4)
        cp = max(self.cp0 * (1.0 + self.cp_slope * dT), 1.0)
        return FluidTransportProperties(rho=rho, mu=mu, k=k, cp=cp)


def build_bundle() -> TubeBundle:
    """Geometry from the v0.4.5 gas-gas test case."""
    tube = BareTube(
        D_i=25e-3 - 2 * 1.5e-3,
        D_o=25e-3,
        length_total=2.8,
        length_effective=2.8,
        wall_k=50.0,
    )
    return TubeBundle(
        tube=tube,
        n_rows=36,
        n_tubes_per_row=56,
        pitch_transverse=35e-3,
        pitch_longitudinal=35e-3,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement="counterflow",
    )


def c_to_k(t_c: float) -> float:
    return t_c + 273.15


def kgh_to_kgs(m: float) -> float:
    return m / 3600.0


def main() -> None:
    hx = BareTubeHeatExchanger(build_bundle())

    inside_var = HXSideInput(
        provider=LinearGasProvider(rho0=1.13, mu0=1.9e-5, k0=0.027, cp0=1007.0),
        m_dot=kgh_to_kgs(18_220.0), T_in=c_to_k(30.0), p=101_325.0,
    )
    outside_var = HXSideInput(
        provider=LinearGasProvider(rho0=0.50, mu0=3.1e-5, k0=0.052, cp0=1180.0),
        m_dot=kgh_to_kgs(28_380.0), T_in=c_to_k(400.0), p=101_325.0,
    )

    print("=" * 70)
    print("CASE 1 - default simulate(): mean-property iteration (variable props)")
    print("=" * 70)
    res = hx.simulate(inside_var, outside_var)
    print(f"converged / iterations : {res.converged} / {res.iterations}")
    print(f"residual_q_rel         : {res.residual_q_rel:.3e}")
    print(f"q [kW]                 : {res.q / 1e3:.2f}")
    print(f"UA [W/K] / U_mean      : {res.UA:.1f} / {res.U_mean:.2f}")
    print(f"T_mean in/out  [degC]  : {res.T_mean_inside - 273.15:.2f} / {res.T_mean_outside - 273.15:.2f}")
    print(f"T_out  in/out  [degC]  : {res.T_out_inside - 273.15:.2f} / {res.T_out_outside - 273.15:.2f}")

    print()
    print("=" * 70)
    print("CASE 2 - ConstantPropertyProvider still iterates for wall temperature (v0.5.3)")
    print("=" * 70)
    inside_c = HXSideInput(
        provider=ConstantPropertyProvider(
            FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0)
        ),
        m_dot=kgh_to_kgs(18_220.0), T_in=c_to_k(30.0), p=101_325.0,
    )
    outside_c = HXSideInput(
        provider=ConstantPropertyProvider(
            FluidTransportProperties(rho=0.50, mu=3.1e-5, k=0.052, cp=1180.0)
        ),
        m_dot=kgh_to_kgs(28_380.0), T_in=c_to_k(400.0), p=101_325.0,
    )
    res2 = hx.simulate(inside_c, outside_c)
    print(f"converged / iterations : {res2.converged} / {res2.iterations}")
    print(f"q [kW]                 : {res2.q / 1e3:.2f}")
    print(f"T_out in/out [degC]    : {res2.T_out_inside - 273.15:.2f} / {res2.T_out_outside - 273.15:.2f}")

    print()
    print("=" * 70)
    print("CASE 3 - iterate=False escape hatch (variable props -> inlet-only, 1 pass)")
    print("=" * 70)
    res_inlet = hx.simulate(inside_var, outside_var, iterate=False)
    print(f"converged / iterations : {res_inlet.converged} / {res_inlet.iterations}")
    print(f"q [kW]                 : {res_inlet.q / 1e3:.2f}")
    dq = (res.q - res_inlet.q) / res_inlet.q * 100.0
    print(f"mean-property vs inlet-only duty shift: {dq:+.2f}%")

    print()
    print("=" * 70)
    print("CASE 4 - force non-convergence (max_iter=3) -> warning, no hang")
    print("=" * 70)
    res4 = hx.simulate(inside_var, outside_var, max_iter=3, relaxation_factor=0.2)
    print(f"converged / iterations : {res4.converged} / {res4.iterations}")
    for w in (res4.warnings or []):
        if w.source == "simulation":
            print(f"warning[{w.severity}] {w.code}")

    print()
    print("=" * 70)
    print("CASE 5 - surface_margin: 0.0 must reproduce case 2 bit-for-bit")
    print("=" * 70)
    res_margin0 = hx.simulate(inside_c, outside_c, surface_margin=0.0)
    print(f"q [kW]                 : {res_margin0.q / 1e3:.2f}")
    print(f"Q_full == Q_derated    : {res_margin0.Q_full == res_margin0.Q_derated}")

    print()
    print("=" * 70)
    print("CASE 6 - surface_margin: 0.2 and 0.5 must derate Q monotonically")
    print("=" * 70)
    res_margin_low = hx.simulate(inside_c, outside_c, surface_margin=0.2)
    res_margin_high = hx.simulate(inside_c, outside_c, surface_margin=0.5)
    print(f"q [kW] margin=0.0/0.2/0.5 : {res_margin0.q / 1e3:.2f} / {res_margin_low.q / 1e3:.2f} / {res_margin_high.q / 1e3:.2f}")

    # ---- assertions --------------------------------------------------------
    assert res.converged and res.iterations > 1, "case 1 should iterate and converge"
    # v0.5.3: ConstantPropertyProvider no longer forces a single pass under
    # the default iterate=True path -- the wall-temperature correction still
    # needs genuine iteration even with constant bulk properties.
    assert res2.converged and res2.iterations > 1, "case 2 must still iterate (wall correction)"
    assert res_inlet.converged and res_inlet.iterations == 1, "iterate=False -> single pass"
    assert not res4.converged and res4.iterations == 3, "case 4 must not converge"

    # v0.5.3: the default (corrected) path always carries a thermal_state;
    # the iterate=False escape hatch never does.
    for sim_result in (res, res2, res4, res_margin0, res_margin_low, res_margin_high):
        assert sim_result.thermal_state is not None, sim_result
        assert sim_result.inside_alfa_mean == sim_result.thermal_state.alfa_i
        assert sim_result.outside_alfa_mean == sim_result.thermal_state.alfa_o
        assert sim_result.U_mean == sim_result.thermal_state.U
        assert sim_result.UA == sim_result.thermal_state.UA
    assert res_inlet.thermal_state is None

    # energy balance on the converged case
    q_in = inside_var.m_dot * res.inside_props_mean.cp * (res.T_out_inside - inside_var.T_in)
    q_out = outside_var.m_dot * res.outside_props_mean.cp * (outside_var.T_in - res.T_out_outside)
    print()
    print(f"energy-balance: q_in={q_in/1e3:.2f} kW  q_out={q_out/1e3:.2f} kW  q={res.q/1e3:.2f} kW")
    assert abs(q_in - res.q) / res.q < 0.02
    assert abs(q_out - res.q) / res.q < 0.02

    # surface_margin=0.0 regression: must be bit-for-bit identical to the
    # margin-less single-pass case (case 2).
    assert res_margin0.q == res2.q, "surface_margin=0.0 must reproduce q bit-for-bit"
    assert res_margin0.T_out_inside == res2.T_out_inside
    assert res_margin0.T_out_outside == res2.T_out_outside
    assert res_margin0.Q_full == res_margin0.Q_derated == res_margin0.q

    # Simulation does not calculate Rating overdesign; the shared reporting
    # value is explicitly zero for every simulation, independent of derating.
    for sim_result in (res, res2, res_inlet, res4, res_margin0, res_margin_low, res_margin_high):
        assert sim_result.overdesign_factor == 0.0

    # surface_margin > 0 must strictly derate duty, monotonically with margin.
    assert res_margin_low.q < res_margin0.q, "margin=0.2 must derate duty below margin=0"
    assert res_margin_high.q < res_margin_low.q, "margin=0.5 must derate duty further than 0.2"
    assert res_margin_high.Q_full > res_margin_high.Q_derated, "Q_full must exceed Q_derated when margin > 0"
    assert res_margin_low.Q_full > res_margin_low.Q_derated

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
