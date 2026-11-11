# SPDX-License-Identifier: GPL-3.0-only
"""Area means of the constant-property exchanger with a uniform latent source.

The source is the existing Bulk-Mean approximation, not a local condensation
profile. Film, fouling and wall resistances remain unchanged. These analytic
means replace terminal arithmetic means in the transport balance only; the
property/correlation evaluation retains its existing bulk-state convention.
See docs/validation/v0.9.8_wet_closure_numerical_patch.md for the derivation.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BulkMeanClosure:
    heat_total: float
    gas_temperature: float
    other_temperature: float


def _exponential_integrals(z):
    """Return integral exp(z*x), (1-x)*exp(z*x), (1-x)^2/2*exp(z*x)."""
    if abs(z) < 0.05:
        # Analytic Taylor sums avoid cancellation at equal capacity rates.
        e, j, m = 1.0, 0.5, 1.0 / 6.0
        term = 1.0
        for n in range(1, 14):
            term *= z / n
            e += term / (n + 1)
            j += term / ((n + 1) * (n + 2))
            m += term / ((n + 1) * (n + 2) * (n + 3))
        return e, j, m
    e = math.expm1(z) / z
    j = (e - 1.0) / z
    return e, j, (j - 0.5) / z


def uniform_source_bulk_means(*, UA, C_gas, C_other, T_gas_in,
                              T_other_in, gas_film_fraction, Q_latent,
                              flow_arrangement, gas_side="outside"):
    """Solve the linear two-stream transport balance without axial segments.

    Let b=R_gas_film/R_total, a=1-b and L=Q_latent. Per unit area coordinate,
    q_sensible=UA*(Tg-Tc)-a*L, q_total=UA*(Tg-Tc)+b*L.
    Gas sensible and other-stream energy equations, with their own inlet
    boundary conditions, determine their actual area means. At L=0 this is
    exactly the existing epsilon-NTU topology, including outside-mixed crossflow.
    """
    for name, value in (("UA", UA), ("C_gas", C_gas), ("C_other", C_other)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not all(math.isfinite(v) for v in (T_gas_in, T_other_in, Q_latent,
                                         gas_film_fraction)):
        raise ValueError("Bulk-mean boundaries and source must be finite")
    if not 0 <= gas_film_fraction <= 1 or Q_latent < 0:
        raise ValueError("Film fraction must be in [0,1]; latent source nonnegative")
    if gas_side not in ("inside", "outside"):
        raise ValueError("gas_side must be inside or outside")
    g, h, c, L = UA, C_gas, C_other, Q_latent
    b, a = gas_film_fraction, 1 - gas_film_fraction
    delta = T_gas_in - T_other_in

    if flow_arrangement == "cocurrentflow":
        e, j, _ = _exponential_integrals(-g * (1 / h + 1 / c))
        difference = delta * e + L * (a / h - b / c) * j
        Q = g * difference + b * L
        total = h * T_gas_in + c * T_other_in + L / 2
        gas = (total + c * difference) / (h + c)
        other = gas - difference
    elif flow_arrangement == "counterflow":
        k = g * (1 / c - 1 / h)
        f = L * (a / h + b / c)
        # Always integrate the decaying exponential, so high NTU cannot
        # overflow. Equal capacities use the same analytic limiting equation.
        if k <= 0:
            e, j, m = _exponential_integrals(k)
            Q = (g * e * delta + g * f * j + b * L) / (1 + g * e / c)
            difference_in = delta - Q / c
            moment = difference_in * j + f * m
        else:
            e, j, m = _exponential_integrals(-k)
            Q = (g * e * (delta + L / h) - g * f * j + b * L) / (1 + g * e / h)
            difference_out = delta - (Q - L) / h
            moment = difference_out * (e - j) - f * (j - m)
        gas = T_gas_in - g * moment / h + a * L / (2 * h)
        other = gas - (Q - b * L) / g
    elif flow_arrangement == "crossflow":
        if gas_side == "outside":
            # Gas mixed transversely; each other-stream path is unmixed.
            A = _exponential_integrals(-g / c)[0]
            K = g * A / h
            E = _exponential_integrals(-K)[0]
            source = (1 - b * A) * L / (g * A)
            gas = T_other_in + delta * E + source * (1 - E)
            other = T_other_in * A + (gas + b * L / g) * (1 - A)
            G = -h * math.expm1(-K)
            Q = G * delta + (1 - G / g * (1 / A - b)) * L
        else:
            # Other stream is the mixed outside stream.
            A = _exponential_integrals(-g / h)[0]
            K = g * A / c
            E = _exponential_integrals(-K)[0]
            source = (1 - a * A) * L / (g * A)
            other = T_gas_in - delta * E + source * (1 - E)
            gas = T_gas_in * A + (other + a * L / g) * (1 - A)
            G = -c * math.expm1(-K)
            Q = G * delta + G / g * (1 / A - a) * L
    else:
        raise ValueError(f"Unsupported wet bulk-mean topology: {flow_arrangement}")
    return BulkMeanClosure(Q, gas, other)
