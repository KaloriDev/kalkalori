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
