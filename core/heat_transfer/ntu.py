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

"""
ε–NTU method for heat exchangers.

Literature (primary references)
-------------------------------
1) Incropera, DeWitt, Bergman, Lavine:
   "Fundamentals of Heat and Mass Transfer", Wiley, Heat Exchangers chapter
2) Shah, Sekulić:
   "Fundamentals of Heat Exchanger Design", Wiley
3) Kays, London:
   "Compact Heat Exchangers"

Notes on C_min / C_max
----------------------
Heat capacity rates are:
    C = m_dot * c_p  [W/K]

In general, C_hot != C_cold.
C_min limits maximum possible heat transfer:

    Q_max = C_min * (T_hot,in - T_cold,in)

This follows directly from energy balance.
"""

from __future__ import annotations

import math
from core.heat_transfer.streams import EnergyStream


def effectiveness_ntu(
    C_hot: float,
    C_cold: float,
    UA: float,
    *,
    flow_arrangement: str = "counterflow",
) -> float:
    """
    Compute effectiveness ε using ε–NTU.

    Supported flow arrangements (MVP):
    - "counterflow"
    - "cocurrentflow"
    - "crossflow"  (default assumption: both fluids mixed)

    Crossflow remark (MVP)
    ----------------------
    With both fluids treated as perfectly mixed (lumped-parameter),
    crossflow performance reduces to a form equivalent to cocurrent flow
    in effectiveness-only modeling (0D).

    More detailed crossflow models (mixed/unmixed, finite rows, etc.) are
    intentionally deferred until segmentation / higher-fidelity modeling.
    """

    if C_hot <= 0.0 or C_cold <= 0.0:
        raise ValueError("C_hot and C_cold must be positive.")
    if UA <= 0.0:
        raise ValueError("UA must be positive.")

    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    C_r = C_min / C_max

    NTU = UA / C_min

    fa = flow_arrangement.lower()

    # Counterflow
    # Ref: Incropera (standard ε–NTU counterflow relation)
    if fa == "counterflow":
        if abs(1.0 - C_r) < 1e-9:
            eps = NTU / (1.0 + NTU)
        else:
            eps = (1.0 - math.exp(-NTU * (1.0 - C_r))) / (1.0 - C_r * math.exp(-NTU * (1.0 - C_r)))

    # Cocurrentflow (parallel)
    # Ref: Incropera (standard ε–NTU parallel relation)
    elif fa == "cocurrentflow":
        eps = (1.0 - math.exp(-NTU * (1.0 + C_r))) / (1.0 + C_r)

    # Crossflow (both fluids mixed, MVP)
    # Ref concept: lumped mixing removes counterflow advantage; use cocurrent-like form in 0D.
    elif fa == "crossflow":
        eps = (1.0 - math.exp(-NTU * (1.0 + C_r))) / (1.0 + C_r)

    else:
        raise ValueError(f"Unsupported flow_arrangement: {flow_arrangement}")

    return eps


def ntu_from_effectiveness(
    effectiveness: float,
    C_hot: float,
    C_cold: float,
    *,
    flow_arrangement: str = "counterflow",
) -> float:
    """
    Invert eps-NTU: compute NTU required to achieve a target effectiveness.

    This is the inverse of ``effectiveness_ntu`` and is used by Rating
    (``core.models.rating``) to find the NTU/UA/area required to close a known
    heat balance. Supports the same three arrangements.

    Counterflow (Cr != 1):
        NTU = 1/(Cr-1) * ln((eps-1)/(eps*Cr-1))
    Counterflow (Cr == 1):
        NTU = eps/(1-eps)
    Cocurrent/crossflow (lumped, both mixed):
        NTU = -ln(1 - eps*(1+Cr)) / (1+Cr), valid for eps < 1/(1+Cr)

    Raises:
        ValueError: if C_hot/C_cold are not positive, or if ``effectiveness``
            is at or above the thermodynamic maximum achievable by the given
            flow arrangement (unreachable regardless of area).
    """
    if C_hot <= 0.0 or C_cold <= 0.0:
        raise ValueError("C_hot and C_cold must be positive.")
    if not (0.0 <= effectiveness < 1.0):
        raise ValueError("effectiveness must be in [0, 1).")

    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    C_r = C_min / C_max

    fa = flow_arrangement.lower()

    if fa == "counterflow":
        if abs(1.0 - C_r) < 1e-9:
            NTU = effectiveness / (1.0 - effectiveness)
        else:
            arg = (effectiveness - 1.0) / (effectiveness * C_r - 1.0)
            if arg <= 0.0:
                raise ValueError(
                    "ntu_from_effectiveness: requested effectiveness is "
                    "unreachable for a counterflow arrangement with this "
                    "capacity-rate ratio, regardless of area."
                )
            NTU = 1.0 / (C_r - 1.0) * math.log(arg)

    elif fa in ("cocurrentflow", "crossflow"):
        eps_max = 1.0 / (1.0 + C_r)
        if effectiveness >= eps_max:
            raise ValueError(
                f"ntu_from_effectiveness: requested effectiveness "
                f"{effectiveness:.6g} is unreachable for flow_arrangement="
                f"{flow_arrangement!r} (thermodynamic maximum eps_max="
                f"{eps_max:.6g} for C_r={C_r:.6g}), regardless of area."
            )
        NTU = -math.log(1.0 - effectiveness * (1.0 + C_r)) / (1.0 + C_r)

    else:
        raise ValueError(f"Unsupported flow_arrangement: {flow_arrangement}")

    return NTU


def heat_duty_from_effectiveness(
    eps: float,
    hot_stream: EnergyStream,
    cold_stream: EnergyStream,
) -> tuple[float, float, float]:
    """
    Compute heat duty and outlet temperatures from ε.

    Ref: Incropera, ε–NTU method (Q = ε * Q_max).
    """
    if not (0.0 <= eps <= 1.0):
        raise ValueError("eps must be between 0 and 1.")

    C_hot = hot_stream.capacity_rate()
    C_cold = cold_stream.capacity_rate()

    C_min = min(C_hot, C_cold)

    Q_max = C_min * (hot_stream.inlet_temperature() - cold_stream.inlet_temperature())
    Q = eps * Q_max

    T_hot_out = hot_stream.inlet_temperature() - Q / C_hot
    T_cold_out = cold_stream.inlet_temperature() + Q / C_cold

    return Q, T_hot_out, T_cold_out
