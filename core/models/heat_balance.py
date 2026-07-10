# KalKalori - Heat Exchanger Open Engine
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

# NOTE ON UNITS
# -------------
# All calculations in the KalKalori core engine are performed using
# plain Python floats in consistent SI base units:
#   T [K], p [Pa], rho [kg/m3], mu [Pa*s], k [W/(m*K)],
#   cp [J/(kg*K)], m_dot [kg/s].

"""
v0.5.1 - Heat-balance closure.

This is the pre-step for Rating (``core.models.rating``): flexible
"input-side" closure of a two-stream heat balance. The invariant is
``Q = m_dot * cp_mean * dT`` on each side, with ``Q_inside == Q_outside == Q``
(no losses). ``cp`` is always evaluated at ``T_mean = 0.5*(T_in+T_out)``.

Each side is described by a ``BalanceSideSpec`` where ``m_dot``, ``T_in``,
``T_out`` may be partially unknown (``None``); ``close_heat_balance`` fills in
whatever is missing from ``Q`` (given directly, implied by a fully-specified
side, or implied by an ``effectiveness`` target) plus the invariant above.

Both sides' ``T_in`` are always required -- this closure does not attempt to
solve cases with no known inlet temperature on a side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.properties.fluids import PropertyProvider
from core.properties.averaging import mean_temperature
from core.common.warnings import ModelWarning, make_warning


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BalanceSideSpec:
    """Per-side input for heat-balance closure.

    Unlike ``HXSideInput`` (Simulation), fields other than ``provider``/``p``
    are optional: whichever of ``m_dot``/``T_in``/``T_out`` is left ``None``
    is a candidate for ``close_heat_balance`` to solve.

    Attributes:
        provider: Point-property provider exposing ``at(T, p)``.
        p: Bulk pressure used for property evaluation [Pa].
        m_dot: Mass flow through this side [kg/s], or None if unknown.
        T_in: Inlet bulk temperature [K]. Required by ``close_heat_balance``.
        T_out: Outlet bulk temperature [K], or None if unknown.
    """

    provider: PropertyProvider
    p: float
    m_dot: float | None = None
    T_in: float | None = None
    T_out: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.p) or self.p <= 0.0:
            raise ValueError("p must be a positive finite value [Pa].")
        for name, value in (("m_dot", self.m_dot), ("T_in", self.T_in), ("T_out", self.T_out)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be a positive finite value when provided.")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClosedBalanceSide:
    """Fully resolved per-side state after heat-balance closure."""

    provider: PropertyProvider
    p: float
    m_dot: float
    T_in: float
    T_out: float
    cp_mean: float
    C: float  # [W/K] = m_dot * cp_mean

    def to_hx_side_input(self):
        """Bridge to ``core.models.simulation.HXSideInput`` (T_in/m_dot/provider/p only).

        Lets a closed balance be re-run through Simulation (e.g. to compare
        "required" vs. "achievable" duty in Rating).
        """
        from core.models.simulation import HXSideInput

        return HXSideInput(provider=self.provider, m_dot=self.m_dot, T_in=self.T_in, p=self.p)


@dataclass(frozen=True)
class ClosedBalance:
    """Fully resolved two-stream heat balance."""

    inside: ClosedBalanceSide
    outside: ClosedBalanceSide
    hot_is_inside: bool
    Q: float           # [W]
    Q_max: float        # [W] = C_min * (T_hot_in - T_cold_in)
    effectiveness: float  # [-] = Q / Q_max
    warnings: list[ModelWarning] | None = None


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------
def close_heat_balance(
    inside: BalanceSideSpec,
    outside: BalanceSideSpec,
    *,
    Q: float | None = None,
    effectiveness: float | None = None,
    over_specified_tolerance: float = 1e-3,
    max_T_out_iter: int = 4,
) -> ClosedBalance:
    """Close a two-stream heat balance from partial per-side information.

    Algorithm (see module docstring for the invariant):

    1. Resolve ``Q``: explicit ``Q`` wins; else a fully-specified side
       (``m_dot+T_in+T_out``) implies it; else ``effectiveness`` with both
       ``m_dot`` and both ``T_in`` known gives ``Q = eff*C_min*(Th_in-Tc_in)``;
       else raise ``ValueError`` ("under-specified").
    2. Resolve each side from ``Q``: both ``m_dot``/``T_out`` present ->
       verify implied duty against ``Q`` (warn if mismatched beyond
       ``over_specified_tolerance``, code ``heat_balance_over_specified``);
       only ``T_out`` missing -> small fixed-point iteration; only ``m_dot``
       missing -> direct solve; both missing -> raise ``ValueError``.

    Raises:
        ValueError: if the balance is under-specified, a side cannot be
            resolved, or inputs are otherwise invalid.
    """
    if inside.T_in is None or outside.T_in is None:
        raise ValueError("close_heat_balance requires T_in on both sides.")

    hot_is_inside = inside.T_in >= outside.T_in
    hot_spec, cold_spec = (inside, outside) if hot_is_inside else (outside, inside)

    def cp_at(spec: BalanceSideSpec, T: float) -> float:
        return spec.provider.at(T=T, p=spec.p).cp

    def is_complete(spec: BalanceSideSpec) -> bool:
        return spec.m_dot is not None and spec.T_out is not None

    def side_implied_Q(spec: BalanceSideSpec) -> float:
        T_mean = mean_temperature(spec.T_in, spec.T_out)
        cp_mean = cp_at(spec, T_mean)
        return spec.m_dot * cp_mean * abs(spec.T_out - spec.T_in)

    warnings_list: list[ModelWarning] = []

    if Q is not None:
        Q_final = Q
    elif is_complete(inside):
        Q_final = side_implied_Q(inside)
    elif is_complete(outside):
        Q_final = side_implied_Q(outside)
    elif effectiveness is not None:
        if inside.m_dot is None or outside.m_dot is None:
            raise ValueError(
                "close_heat_balance: under-specified. The effectiveness path "
                "requires m_dot on both sides (plus T_in on both sides)."
            )
        C_hot = hot_spec.m_dot * cp_at(hot_spec, hot_spec.T_in)
        C_cold = cold_spec.m_dot * cp_at(cold_spec, cold_spec.T_in)
        C_min = min(C_hot, C_cold)
        Q_final = effectiveness * C_min * (hot_spec.T_in - cold_spec.T_in)
    else:
        raise ValueError(
            "close_heat_balance: under-specified. Provide Q explicitly, a "
            "fully specified side (m_dot + T_in + T_out), or effectiveness "
            "with m_dot and T_in known on both sides."
        )

    if not math.isfinite(Q_final) or Q_final <= 0.0:
        raise ValueError("close_heat_balance: resolved Q must be positive and finite.")

    def resolve_side(spec: BalanceSideSpec, *, is_hot: bool) -> ClosedBalanceSide:
        has_m = spec.m_dot is not None
        has_T_out = spec.T_out is not None

        if has_m and has_T_out:
            T_mean = mean_temperature(spec.T_in, spec.T_out)
            cp_mean = cp_at(spec, T_mean)
            C = spec.m_dot * cp_mean
            Q_side = C * abs(spec.T_out - spec.T_in)
            if abs(Q_side - Q_final) > over_specified_tolerance * max(abs(Q_final), 1e-9):
                warnings_list.append(
                    make_warning(
                        code="heat_balance_over_specified",
                        message=(
                            "heat_balance: a fully specified side implies a "
                            f"duty of {Q_side:.6g} W, which does not match "
                            f"the resolved balance duty of {Q_final:.6g} W "
                            f"within tolerance={over_specified_tolerance:.3g}."
                        ),
                        source="heat_balance",
                        severity="warning",
                    )
                )
            return ClosedBalanceSide(
                provider=spec.provider, p=spec.p, m_dot=spec.m_dot,
                T_in=spec.T_in, T_out=spec.T_out, cp_mean=cp_mean, C=C,
            )

        if has_m and not has_T_out:
            T_out = spec.T_in
            cp_mean = cp_at(spec, spec.T_in)
            for _ in range(max_T_out_iter):
                C = spec.m_dot * cp_mean
                T_out = spec.T_in - Q_final / C if is_hot else spec.T_in + Q_final / C
                cp_mean = cp_at(spec, mean_temperature(spec.T_in, T_out))
            C = spec.m_dot * cp_mean
            return ClosedBalanceSide(
                provider=spec.provider, p=spec.p, m_dot=spec.m_dot,
                T_in=spec.T_in, T_out=T_out, cp_mean=cp_mean, C=C,
            )

        if has_T_out and not has_m:
            T_mean = mean_temperature(spec.T_in, spec.T_out)
            cp_mean = cp_at(spec, T_mean)
            dT = abs(spec.T_out - spec.T_in)
            if dT <= 0.0:
                raise ValueError(
                    "close_heat_balance: cannot solve m_dot with zero "
                    "T_in/T_out difference."
                )
            m_dot = Q_final / (cp_mean * dT)
            C = m_dot * cp_mean
            return ClosedBalanceSide(
                provider=spec.provider, p=spec.p, m_dot=m_dot,
                T_in=spec.T_in, T_out=spec.T_out, cp_mean=cp_mean, C=C,
            )

        raise ValueError(
            "close_heat_balance: under-specified side (missing both m_dot "
            "and T_out)."
        )

    inside_resolved = resolve_side(inside, is_hot=hot_is_inside)
    outside_resolved = resolve_side(outside, is_hot=not hot_is_inside)

    C_min = min(inside_resolved.C, outside_resolved.C)
    Th_in = hot_spec.T_in
    Tc_in = cold_spec.T_in
    Q_max = C_min * (Th_in - Tc_in)
    eff = Q_final / Q_max if Q_max > 0.0 else math.nan

    return ClosedBalance(
        inside=inside_resolved,
        outside=outside_resolved,
        hot_is_inside=hot_is_inside,
        Q=Q_final,
        Q_max=Q_max,
        effectiveness=eff,
        warnings=warnings_list if warnings_list else None,
    )
