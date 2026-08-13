# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Neutral outer-area tube-wall resistance helpers for multi-zone solvers."""

from __future__ import annotations

import math


def overall_u_outer_basis(
    *,
    alpha_inside: float,
    alpha_outside: float,
    D_i: float,
    D_o: float,
    wall_k: float,
) -> float:
    """Return U referenced to tube outer area [W/(m2*K)]."""
    return 1.0 / sum(
        outer_basis_resistance_components(
            alpha_inside=alpha_inside,
            alpha_outside=alpha_outside,
            D_i=D_i,
            D_o=D_o,
            wall_k=wall_k,
        )
    )


def outer_basis_resistance_components(
    *,
    alpha_inside: float,
    alpha_outside: float,
    D_i: float,
    D_o: float,
    wall_k: float,
) -> tuple[float, float, float]:
    """Return inside, cylindrical-wall and outside resistances per Ao."""
    if not math.isfinite(alpha_inside) or alpha_inside <= 0.0:
        raise ValueError("alpha_inside must be positive and finite.")
    wall_resistance, outside_resistance = fixed_outer_basis_resistances(
        alpha_outside=alpha_outside,
        D_i=D_i,
        D_o=D_o,
        wall_k=wall_k,
    )
    inside_resistance = D_o / (D_i * alpha_inside)
    if not math.isfinite(inside_resistance) or inside_resistance <= 0.0:
        raise ValueError("Inside thermal resistance must be positive and finite.")
    return inside_resistance, wall_resistance, outside_resistance


def fixed_outer_basis_resistances(
    *,
    alpha_outside: float,
    D_i: float,
    D_o: float,
    wall_k: float,
) -> tuple[float, float]:
    """Return wall and outside resistance terms shared by forward/inverse U."""
    for name, value in (
        ("alpha_outside", alpha_outside),
        ("D_i", D_i),
        ("D_o", D_o),
        ("wall_k", wall_k),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    if D_o <= D_i:
        raise ValueError("D_o must be greater than D_i for a tube wall.")
    values = (
        D_o * math.log(D_o / D_i) / (2.0 * wall_k),
        1.0 / alpha_outside,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Outer-basis thermal resistances must be positive and finite.")
    return values


def equivalent_inside_alpha_outer_basis(
    *,
    U_equivalent: float,
    alpha_outside: float,
    D_i: float,
    D_o: float,
    wall_k: float,
) -> float:
    """Invert the outer-area resistance network for an equivalent inside HTC."""
    if not math.isfinite(U_equivalent) or U_equivalent <= 0.0:
        raise ValueError("U_equivalent must be positive and finite.")
    wall_resistance, outside_resistance = fixed_outer_basis_resistances(
        alpha_outside=alpha_outside,
        D_i=D_i,
        D_o=D_o,
        wall_k=wall_k,
    )
    inside_resistance = 1.0 / U_equivalent - wall_resistance - outside_resistance
    if not math.isfinite(inside_resistance) or inside_resistance <= 0.0:
        raise ValueError(
            "U_equivalent leaves no positive finite inside thermal resistance."
        )
    alpha_inside = (D_o / D_i) / inside_resistance
    if not math.isfinite(alpha_inside) or alpha_inside <= 0.0:
        raise ValueError("Equivalent inside HTC must be positive and finite.")
    reconstructed = overall_u_outer_basis(
        alpha_inside=alpha_inside,
        alpha_outside=alpha_outside,
        D_i=D_i,
        D_o=D_o,
        wall_k=wall_k,
    )
    if not math.isclose(
        reconstructed, U_equivalent, rel_tol=2.0e-12, abs_tol=1.0e-12
    ):
        raise ValueError("Equivalent inside HTC does not reconstruct U_equivalent.")
    return alpha_inside
