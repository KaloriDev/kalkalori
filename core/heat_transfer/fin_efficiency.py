# KalKalori — Heat Exchanger Open Engine
# Copyright (C) 2025  KalKalori Project Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# -------------------------------------------------------------------------
# ANNULAR FIN EFFICIENCY (v0.7.x)
# -------------------------------------------------------------------------
#
# Two independent models are provided, both solving the standard 1D radial
# conduction/convection fin equation:
#
#     d/dr [ k_fin * A_c(r) * dtheta/dr ] - h_o * P(r) * theta = 0
#
# with A_c(r) = 2*pi*r*t(r) (radial conduction cross-section) and P(r)
# accounting for convection from both flat faces of the fin, a base
# Dirichlet condition (theta(r_root) = theta_base) and a convective tip
# condition (-k_fin*t(r_tip)*dtheta/dr|_tip = h_o*t(r_tip)*theta(r_tip)).
#
# 1) fin_efficiency_constant_thickness: closed-form Bessel-function
#    solution for a fin of constant axial thickness (Gardner, 1945; see
#    also Kern, D.Q.; Kraus, A.D., "Extended Surface Heat Transfer",
#    McGraw-Hill, 1972, and Kraus, A.D.; Aziz, A.; Welty, J., "Extended
#    Surface Heat Transfer", Wiley, 2001).
#
# 2) fin_efficiency_tapered: deterministic numerical solution for a
#    linearly root-to-tip tapered fin, via linear-superposition shooting
#    (the governing ODE and both boundary conditions are linear in
#    theta, so exactly two RK4 integrations plus one linear solve give
#    the exact solution of the discretized problem -- no root-finding
#    iteration, no new production dependency). This reduces to the
#    constant-thickness closed form (verified in the test suite) when
#    fin_thickness_tip == fin_thickness_root.
#
# Both are validated in core/tests/finned_tube_fin_efficiency_test.py
# against each other (tapered solver vs. closed form in the
# constant-thickness limit) and against known qualitative and limiting
# behavior (m*L -> 0 gives eta -> 1; higher h_o or taller fins reduce
# eta; higher fin_k increases eta).
# -------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import math

from core.heat_transfer.modified_bessel import bessel_i0, bessel_i1, bessel_k0, bessel_k1


def fin_efficiency_constant_thickness(
    *,
    D_root: float,
    D_fin: float,
    fin_thickness: float,
    fin_k: float,
    h_o: float,
) -> float:
    """Gardner (1945) / Kern-Kraus closed-form efficiency of a constant-
    thickness annular (circular) fin.

        m = sqrt(2*h_o / (fin_k*fin_thickness))
        r_o = D_root/2, r_e = D_fin/2 + fin_thickness/2   (corrected tip radius)

        eta = [2*r_o / (m*(r_e^2 - r_o^2))] *
              [I1(m*r_e)*K1(m*r_o) - K1(m*r_e)*I1(m*r_o)] /
              [I0(m*r_o)*K1(m*r_e) + I1(m*r_e)*K0(m*r_o)]

    This Bessel-function solution (Gardner 1945; Kern & Kraus 1972;
    Incropera et al., "Fundamentals of Heat and Mass Transfer", Table
    3.5) is the classical solution of an annular fin with an *adiabatic*
    tip. The standard corrected-radius technique
    (r_e -> r_e + fin_thickness/2, "Lc = L + t/2") is applied so that it
    approximates a convective tip instead, consistent with the exact
    convective-tip boundary condition used by ``fin_efficiency_tapered``
    for the general (tapered) case, and with the convective-tip
    requirement in docs/finned_tube_model.md. The corrected-radius
    approximation is cross-checked in
    core/tests/finned_tube_fin_efficiency_test.py against the exact
    numerical convective-tip solution (``fin_efficiency_tapered`` with
    fin_thickness_tip == fin_thickness_root): agreement is within
    ~1e-4 relative for the finned-tube geometry ranges this feature
    targets.

    Assumptions: 1D radial conduction, steady state, no radiation,
    temperature-independent fin_k, uniform h_o over both fin faces,
    uniform (constant) base temperature, no fin-base contact resistance
    (contact resistance is handled separately in the resistance network,
    not inside this efficiency), constant surrounding fluid temperature.
    """
    if D_root <= 0.0 or D_fin <= D_root:
        raise ValueError("Require 0 < D_root < D_fin.")
    if fin_thickness <= 0.0:
        raise ValueError("fin_thickness must be positive.")
    if fin_k <= 0.0:
        raise ValueError("fin_k must be positive.")
    if h_o <= 0.0:
        raise ValueError("h_o must be positive.")

    r_o = D_root / 2.0
    r_e = D_fin / 2.0 + fin_thickness / 2.0
    m = math.sqrt(2.0 * h_o / (fin_k * fin_thickness))

    m_ro = m * r_o
    m_re = m * r_e

    numerator = bessel_i1(m_re) * bessel_k1(m_ro) - bessel_k1(m_re) * bessel_i1(m_ro)
    denominator = bessel_i0(m_ro) * bessel_k1(m_re) + bessel_i1(m_re) * bessel_k0(m_ro)

    eta = (2.0 * r_o / (m * (r_e * r_e - r_o * r_o))) * (numerator / denominator)

    # Numerical guardrail: the closed form is bounded in (0, 1] by
    # construction; small floating point overshoot just above 1.0 can
    # occur for very small m*(r_e-r_o) (near-isothermal fin).
    return min(max(eta, 0.0), 1.0)


@dataclass(frozen=True)
class TaperedFinSolution:
    """Diagnostics from the numerical tapered-fin ODE solve."""

    efficiency: float
    theta_tip: float
    base_slope: float
    n_steps: int


def fin_efficiency_tapered(
    *,
    D_root: float,
    D_fin: float,
    fin_thickness_root: float,
    fin_thickness_tip: float,
    fin_k: float,
    h_o: float,
    n_steps: int = 400,
) -> TaperedFinSolution:
    """Numerical efficiency of a linearly root-to-tip tapered annular fin.

    Solves d/dr[r*t(r)*dtheta/dr] = (2*h_o/fin_k)*r*theta with
    theta(r_root)=1 and the convective tip condition
    -fin_k*dtheta/dr(r_tip) = h_o*theta(r_tip), via RK4 integration of
    two linearly independent solutions and a linear superposition
    (exact for a linear ODE + linear BCs -- no iterative root-finding).
    """
    if D_root <= 0.0 or D_fin <= D_root:
        raise ValueError("Require 0 < D_root < D_fin.")
    if fin_thickness_root <= 0.0 or fin_thickness_tip <= 0.0:
        raise ValueError("fin thicknesses must be positive.")
    if fin_k <= 0.0:
        raise ValueError("fin_k must be positive.")
    if h_o <= 0.0:
        raise ValueError("h_o must be positive.")
    if n_steps < 10:
        raise ValueError("n_steps must be at least 10.")

    r_root = D_root / 2.0
    r_tip = D_fin / 2.0
    dr_dt = (fin_thickness_tip - fin_thickness_root) / (r_tip - r_root)

    def thickness(r: float) -> float:
        return fin_thickness_root + dr_dt * (r - r_root)

    def rhs(r: float, theta: float, dtheta: float) -> tuple[float, float]:
        t = thickness(r)
        d2theta = ((2.0 * h_o / fin_k) * r * theta - (t + r * dr_dt) * dtheta) / (r * t)
        return dtheta, d2theta

    step = (r_tip - r_root) / n_steps

    def integrate(theta0: float, dtheta0: float) -> tuple[float, float]:
        r = r_root
        theta = theta0
        dtheta = dtheta0
        for _ in range(n_steps):
            k1_t, k1_d = rhs(r, theta, dtheta)
            k2_t, k2_d = rhs(r + step / 2.0, theta + step / 2.0 * k1_t, dtheta + step / 2.0 * k1_d)
            k3_t, k3_d = rhs(r + step / 2.0, theta + step / 2.0 * k2_t, dtheta + step / 2.0 * k2_d)
            k4_t, k4_d = rhs(r + step, theta + step * k3_t, dtheta + step * k3_d)
            theta = theta + (step / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
            dtheta = dtheta + (step / 6.0) * (k1_d + 2.0 * k2_d + 2.0 * k3_d + k4_d)
            r += step
        return theta, dtheta

    theta_tip_0, dtheta_tip_0 = integrate(1.0, 0.0)
    theta_tip_1, dtheta_tip_1 = integrate(0.0, 1.0)

    # Tip BC residual R(s) = fin_k*dtheta_tip(s) + h_o*theta_tip(s) = 0,
    # R linear in s = dtheta/dr at the root: R(s) = R0 + s*R1.
    R0 = fin_k * dtheta_tip_0 + h_o * theta_tip_0
    R1 = fin_k * dtheta_tip_1 + h_o * theta_tip_1
    if abs(R1) < 1.0e-300:
        raise ValueError("Tapered fin ODE is numerically degenerate (R1 ~ 0).")
    s = -R0 / R1

    theta_tip = theta_tip_0 + s * theta_tip_1
    base_slope = s  # dtheta/dr at r_root, with theta(r_root) = 1

    q_actual = -fin_k * (2.0 * math.pi * r_root * fin_thickness_root) * base_slope

    slant = math.hypot(r_tip - r_root, (fin_thickness_root - fin_thickness_tip) / 2.0)
    a_side = 2.0 * math.pi * (r_root + r_tip) * slant
    a_tip = math.pi * (2.0 * r_tip) * fin_thickness_tip
    q_ideal = h_o * (a_side + a_tip)

    eta = q_actual / q_ideal
    eta = min(max(eta, 0.0), 1.0)

    return TaperedFinSolution(
        efficiency=eta,
        theta_tip=theta_tip,
        base_slope=base_slope,
        n_steps=n_steps,
    )


def fin_efficiency_for_tube(tube, h_o: float, *, n_steps: int = 400) -> float:
    """Dispatch: closed form for a constant-thickness fin, numerical
    tapered solver otherwise. ``tube`` is a
    ``core.geometry.finned_tube.CircularFinnedTube``.
    """
    t_root = tube.fin_thickness_root
    t_tip = tube.fin_thickness_tip_used
    if t_root == t_tip:
        return fin_efficiency_constant_thickness(
            D_root=tube.D_root,
            D_fin=tube.D_fin,
            fin_thickness=t_root,
            fin_k=tube.fin_k,
            h_o=h_o,
        )
    return fin_efficiency_tapered(
        D_root=tube.D_root,
        D_fin=tube.D_fin,
        fin_thickness_root=t_root,
        fin_thickness_tip=t_tip,
        fin_k=tube.fin_k,
        h_o=h_o,
        n_steps=n_steps,
    ).efficiency


def overall_surface_efficiency(*, A_primary: float, A_fin: float, fin_efficiency: float) -> float:
    """Overall (weighted) surface efficiency eta_o = (A_primary + eta_fin*A_fin) / (A_primary+A_fin).

    Applied once, to the total (primary + fin) outside surface -- never
    combined a second time with a separately fin-efficiency-weighted
    area (see core.heat_transfer.finned_tube_resistance).
    """
    if A_primary < 0.0 or A_fin < 0.0:
        raise ValueError("Areas must be non-negative.")
    total = A_primary + A_fin
    if total <= 0.0:
        raise ValueError("A_primary + A_fin must be positive.")
    if not (0.0 <= fin_efficiency <= 1.0):
        raise ValueError("fin_efficiency must be in [0, 1].")
    return (A_primary + fin_efficiency * A_fin) / total
