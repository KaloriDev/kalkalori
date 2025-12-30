# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

# NOTE ON UNITS
# -------------
# SI units:
# - m_dot [kg/s], A [m^2], D [m]
# - rho [kg/m^3], mu [Pa*s], k [W/(m*K)], cp [J/(kg*K)]
# - v [m/s], h [W/(m^2*K)], dp [Pa]

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FluidProps:
    rho: float  # [kg/m^3]
    mu: float   # [Pa*s]
    k: float    # [W/(m*K)]
    cp: float   # [J/(kg*K)]


def reynolds_number(rho: float, v: float, D: float, mu: float) -> float:
    if rho <= 0.0 or mu <= 0.0 or D <= 0.0 or v <= 0.0:
        raise ValueError("rho, mu, D, v must be positive.")
    return rho * v * D / mu


def prandtl_number(cp: float, mu: float, k: float) -> float:
    if cp <= 0.0 or mu <= 0.0 or k <= 0.0:
        raise ValueError("cp, mu, k must be positive.")
    return cp * mu / k


def nusselt_zukauskas(Re: float, Pr: float, n_rows: int) -> float:
    if Re <= 0.0 or Pr <= 0.0:
        raise ValueError("Re and Pr must be positive.")
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    if Re < 1e2:
        C, m = 0.90, 0.40
    elif Re < 1e3:
        C, m = 0.52, 0.50
    elif Re < 2e5:
        C, m = 0.27, 0.63
    else:
        C, m = 0.021, 0.84

    Nu = C * (Re ** m) * (Pr ** 0.36)

    if n_rows < 20:
        Nu *= (n_rows / 20.0) ** 0.20

    return Nu


def outside_flow_from_mass_flow(
    m_dot: float,
    frontal_area: float,
    tube_outer_diameter: float,
    n_rows: int,
    props: FluidProps,
    *,
    zeta_dp: float = 1.2,
) -> tuple[float, float, float, float, float]:
    """
    Compute outside forced convection from mass flow rate.

    Returns
    -------
    v : float
        Approach (characteristic) velocity [m/s]
    Re : float
    Pr : float
    h_o : float
    dp_o : float
    """
    if m_dot <= 0.0:
        raise ValueError("m_dot must be positive.")
    if frontal_area <= 0.0:
        raise ValueError("frontal_area must be positive.")
    if tube_outer_diameter <= 0.0:
        raise ValueError("tube_outer_diameter must be positive.")
    if zeta_dp <= 0.0:
        raise ValueError("zeta_dp must be positive.")

    v = m_dot / (props.rho * frontal_area)

    Re = reynolds_number(props.rho, v, tube_outer_diameter, props.mu)
    Pr = prandtl_number(props.cp, props.mu, props.k)

    Nu = nusselt_zukauskas(Re, Pr, n_rows)
    h_o = Nu * props.k / tube_outer_diameter

    dp_o = zeta_dp * n_rows * (props.rho * v * v / 2.0)

    return v, Re, Pr, h_o, dp_o
