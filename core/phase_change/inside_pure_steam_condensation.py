# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""In-tube pure water/steam condensation-zone physics (v0.6.2).

This module is the pure-H2O counterpart of
``core.phase_change.inside_condensation_solver`` (v0.6.1, wet-gas H2O
condensation) but is built on entirely different physics and does not
share its solver:

- v0.6.1 (wet-gas): H2O vapor condensing out of a non-condensable carrier
  gas -- dew point, humidity ratio W, mass diffusion, Chilton-Colburn/
  Lewis analogy.
- v0.6.2 (pure steam, this module): direct vapor-liquid phase equilibrium
  of pure H2O, parameterized by vapor quality x. There is no carrier gas
  and no diffusion resistance.

This module only solves the physics of a single condensing zone (latent
energy balance + quality-averaged condensation HTC via
``core.heat_transfer.condensation_inside``, the Shah (1979) correlation).
It does not yet allocate heat-transfer surface area to the zone or combine
it with desuperheating/subcooling zones -- see
``core.phase_change.inside_pure_steam_zones`` (v0.6.2, multi-zone solver)
for that.

Ref: IAPWS-IF97 (via ``core.properties.water``); Shah, M.M. (1979),
Int. J. Heat Mass Transfer, 22(4), 547-556 (via
``core.heat_transfer.condensation_inside``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.common.warnings import ModelWarning, make_warning
from core.heat_transfer.condensation_inside import (
    CORRELATION_NAME,
    condensation_zone_alpha_effective,
)
from core.phase_change.warning_codes import TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED
from core.properties.water import WATER_CRITICAL_PRESSURE_PA, water_steam_state

SOURCE = "phase_change.inside_pure_steam_condensation"


@dataclass(frozen=True)
class InsideCondensationZoneResult:
    """Self-contained physics of one in-tube pure-steam condensing zone.

    Covers quality decreasing from `x_in` to `x_out` (`0 <= x_out < x_in
    <= 1`) at a fixed nominal pressure `p` (see the module docstring of
    ``core.properties.water`` and the v0.6.2 pressure-modelling notes in
    ``docs/references.md`` -- two-phase pressure drop is not modelled, so
    `T_sat` is constant across the zone).

    Attributes:
        x_in: Zone inlet vapor quality [-].
        x_out: Zone outlet vapor quality [-].
        h_in: Specific enthalpy at `x_in` [J/kg].
        h_out: Specific enthalpy at `x_out` [J/kg].
        T_sat: Saturation temperature at `p` [K].
        h_f: Saturated-liquid specific enthalpy at `p` [J/kg].
        h_g: Saturated-vapor specific enthalpy at `p` [J/kg].
        h_fg: Latent heat of vaporization at `p` [J/kg].
        Q_condensation: Latent duty released by the zone [W],
            `m_dot_total * (h_in - h_out) = m_dot_total * h_fg * (x_in -
            x_out)`. Always computed from the exact enthalpy difference,
            never `m_dot * cp * dT` (cp is not defined in this region).
        alpha_condensation_effective: Quality-averaged condensation heat
            transfer coefficient over the zone [W/(m2*K)] (Shah, 1979,
            5-point Gauss-Legendre quadrature).
        m_dot_total: Total (vapor + liquid) tube-side mass flow [kg/s],
            constant through condensation.
        m_dot_vapor_in, m_dot_vapor_out: Vapor mass flow at the zone inlet/
            outlet [kg/s], `x * m_dot_total`.
        m_dot_liquid_in, m_dot_liquid_out: Liquid (condensate) mass flow at
            the zone inlet/outlet [kg/s], `(1-x) * m_dot_total`.
        G: Total mass flux through the tube-side flow area [kg/(m2*s)].
        correlation: Name of the condensation HTC correlation used.
        two_phase_pressure_drop_supported: Always `False` in v0.6.2 -- see
            `warnings` for the corresponding diagnostic.
        warnings: Applicability / limitation warnings.
    """

    x_in: float
    x_out: float
    h_in: float
    h_out: float
    T_sat: float
    h_f: float
    h_g: float
    h_fg: float
    Q_condensation: float
    alpha_condensation_effective: float
    m_dot_total: float
    m_dot_vapor_in: float
    m_dot_vapor_out: float
    m_dot_liquid_in: float
    m_dot_liquid_out: float
    G: float
    correlation: str
    two_phase_pressure_drop_supported: bool
    warnings: list[ModelWarning]


def solve_inside_condensation_zone(
    *,
    p: float,
    x_in: float,
    x_out: float,
    m_dot_total: float,
    D_i: float,
    flow_area: float,
) -> InsideCondensationZoneResult:
    """Solve one in-tube pure-steam condensing zone.

    Args:
        p: Nominal tube-side pressure [Pa], used for phase equilibrium
            (constant across the zone in this 0D model -- two-phase
            pressure drop is not modelled in v0.6.2).
        x_in: Zone inlet vapor quality [-].
        x_out: Zone outlet vapor quality [-]. Must satisfy
            `0 <= x_out < x_in <= 1` (quality strictly decreases across a
            condensing zone; a zero-size zone should not be constructed by
            the caller).
        m_dot_total: Total (vapor + liquid) tube-side mass flow [kg/s].
            Unlike v0.6.1 wet-gas condensation, this does not change
            through the zone -- only the vapor/liquid split does.
        D_i: Tube inner diameter [m].
        flow_area: Total tube-side internal flow area [m2], used with
            `m_dot_total` to obtain the mass flux `G` for the HTC
            correlation.

    Returns:
        InsideCondensationZoneResult.
    """
    if not (0.0 <= x_out < x_in <= 1.0):
        raise ValueError(
            "solve_inside_condensation_zone requires 0 <= x_out < x_in <= 1 "
            f"(got x_in={x_in}, x_out={x_out})."
        )
    for name, value in (
        ("m_dot_total", m_dot_total),
        ("D_i", D_i),
        ("flow_area", flow_area),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")

    saturated_liquid = water_steam_state(p=p, x=0.0)
    T_sat = saturated_liquid.T_sat
    h_f = saturated_liquid.h_f
    h_g = saturated_liquid.h_g
    h_fg = saturated_liquid.h_fg
    mu_L = saturated_liquid.mu
    k_L = saturated_liquid.k
    cp_L = saturated_liquid.cp

    h_in = h_f + x_in * h_fg
    h_out = h_f + x_out * h_fg
    Q_condensation = m_dot_total * (h_in - h_out)

    G = m_dot_total / flow_area

    htc = condensation_zone_alpha_effective(
        x_in=x_in,
        x_out=x_out,
        p=p,
        p_critical=WATER_CRITICAL_PRESSURE_PA,
        G=G,
        D_i=D_i,
        mu_L=mu_L,
        k_L=k_L,
        cp_L=cp_L,
    )

    warnings = list(htc.warnings)
    warnings.append(
        make_warning(
            code=TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED,
            message=(
                "inside: two-phase condensation-zone pressure drop is not "
                "modelled in v0.6.2; single-phase tube-side dp components "
                "do not represent the full condensing-zone dp."
            ),
            source=SOURCE,
            severity="info",
        )
    )

    return InsideCondensationZoneResult(
        x_in=x_in,
        x_out=x_out,
        h_in=h_in,
        h_out=h_out,
        T_sat=T_sat,
        h_f=h_f,
        h_g=h_g,
        h_fg=h_fg,
        Q_condensation=Q_condensation,
        alpha_condensation_effective=htc.alpha_effective,
        m_dot_total=m_dot_total,
        m_dot_vapor_in=x_in * m_dot_total,
        m_dot_vapor_out=x_out * m_dot_total,
        m_dot_liquid_in=(1.0 - x_in) * m_dot_total,
        m_dot_liquid_out=(1.0 - x_out) * m_dot_total,
        G=G,
        correlation=CORRELATION_NAME,
        two_phase_pressure_drop_supported=False,
        warnings=warnings,
    )
