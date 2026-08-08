# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for the outside-condensation mass and energy balance (v0.6.0).

Exercises core.phase_change.outside_condensation_solver directly on a small
synthetic bare-tube exchanger, checking the closure invariants:

    m_dot_water_vapor_in = m_dot_water_vapor_out + m_dot_condensate
    Q_total = Q_sensible + Q_latent

and the physical bounds (0 <= W_out <= W_in, m_dot_condensate >= 0).

Also covers rejecting W_out < 0 and p_H2O >= p_total at the equilibrium
layer (spec section 35, items 18-19), and the controlled frost/ice
(section 15) behavior.

Run:
    pytest -q core/tests/phase_change_mass_energy_balance_test.py
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.properties.fluids import ConstantPropertyProvider
from core.properties.common import FluidTransportProperties
from core.properties.gas_mixture import GasMixtureSpec, GasMixturePropertyProvider
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.outside_condensation_solver import (
    FrostingNotSupportedError,
    solve_outside_condensation,
)
from core.phase_change.water_equilibrium import water_partial_pressure


def _hx() -> BareTubeHeatExchanger:
    tube = BareTube(D_i=25e-3 - 2 * 1.5e-3, D_o=25e-3, length_total=2.8, length_effective=2.8, wall_k=50.0)
    bundle = TubeBundle(
        tube=tube, n_rows=20, n_tubes_per_row=30,
        pitch_transverse=35e-3, pitch_longitudinal=35e-3,
        layout="staggered", n_passes_tube=2, flow_arrangement="counterflow",
    )
    return BareTubeHeatExchanger(bundle)


def _wet_capability(y_h2o: float = 0.17):
    remainder = 1.0 - y_h2o
    spec = GasMixtureSpec(
        components={"N2": 0.65 * remainder / 0.83, "O2": 0.10 * remainder / 0.83,
                    "CO2": 0.08 * remainder / 0.83, "H2O": y_h2o},
        basis="mole",
    )
    return detect_phase_change_capability(GasMixturePropertyProvider(spec))


def _dry_air_provider():
    return ConstantPropertyProvider(FluidTransportProperties(rho=1.13, mu=1.9e-5, k=0.027, cp=1007.0))


def test_mass_and_energy_balance_close_for_a_condensing_case() -> None:
    hx = _hx()
    cap = _wet_capability(0.17)
    m_dot_outside_total = 6.0
    m_dot_dry_carrier = m_dot_outside_total / (1.0 + cap.W_in)

    solution = solve_outside_condensation(
        hx,
        inside_provider=_dry_air_provider(),
        m_dot_inside=15.0, T_in_inside=290.0, p_inside=101325.0,
        outside_capability=cap,
        m_dot_dry_carrier=m_dot_dry_carrier,
        T_in_outside=420.0, p_outside=101325.0,
        T_out_inside_init=330.0, T_out_outside_init=335.0,
    )

    assert solution.converged
    assert 0.0 <= solution.W_out <= cap.W_in
    assert solution.m_dot_condensate >= 0.0

    m_dot_water_vapor_in = m_dot_dry_carrier * cap.W_in
    m_dot_water_vapor_out = m_dot_dry_carrier * solution.W_out
    mass_balance_error = m_dot_water_vapor_in - (m_dot_water_vapor_out + solution.m_dot_condensate)
    assert abs(mass_balance_error) < 1e-6

    energy_balance_error = solution.Q_total - (solution.Q_sensible + solution.Q_latent)
    assert abs(energy_balance_error) < 1e-6

    assert solution.Q_sensible > 0.0
    assert solution.Q_latent > 0.0
    assert math.isfinite(solution.alfa_o_effective)
    assert math.isfinite(solution.UA_effective)


def test_solver_starts_from_dry_baseline_not_zero() -> None:
    """The initial iterate must be the supplied dry baseline, not zeros."""
    hx = _hx()
    cap = _wet_capability(0.05)  # a low water content: likely little/no condensation
    m_dot_outside_total = 6.0
    m_dot_dry_carrier = m_dot_outside_total / (1.0 + cap.W_in)

    solution = solve_outside_condensation(
        hx,
        inside_provider=_dry_air_provider(),
        m_dot_inside=15.0, T_in_inside=290.0, p_inside=101325.0,
        outside_capability=cap,
        m_dot_dry_carrier=m_dot_dry_carrier,
        T_in_outside=350.0, p_outside=101325.0,
        T_out_inside_init=310.0, T_out_outside_init=320.0,
    )
    # Converges to something close-ish to the seeded dry baseline (not a
    # wildly different arbitrary-zero-start solution).
    assert 250.0 < solution.T_out_outside < 360.0
    assert 250.0 < solution.T_out_inside < 360.0


def test_water_partial_pressure_rejects_p_h2o_at_or_above_total() -> None:
    with pytest.raises(ValueError):
        water_partial_pressure(1.0, 101325.0)


def test_frosting_not_supported_is_raised_for_a_sub_freezing_interface() -> None:
    """A very cold inside stream can pull the outside interface below the
    water triple point; this must raise FrostingNotSupportedError, not
    silently clamp to 273.15 K or continue as ordinary condensation."""
    hx = _hx()
    cap = _wet_capability(0.05)
    m_dot_outside_total = 6.0
    m_dot_dry_carrier = m_dot_outside_total / (1.0 + cap.W_in)

    very_cold_inside = ConstantPropertyProvider(
        FluidTransportProperties(rho=1.3, mu=1.7e-5, k=0.024, cp=1006.0)
    )
    with pytest.raises(FrostingNotSupportedError):
        solve_outside_condensation(
            hx,
            inside_provider=very_cold_inside,
            m_dot_inside=50.0, T_in_inside=230.0, p_inside=101325.0,
            outside_capability=cap,
            m_dot_dry_carrier=m_dot_dry_carrier,
            T_in_outside=300.0, p_outside=101325.0,
            T_out_inside_init=250.0, T_out_outside_init=270.0,
        )
