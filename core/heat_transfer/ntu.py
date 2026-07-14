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
   (Table 11.4: crossflow, one fluid mixed / one unmixed).
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

Crossflow: mixed / unmixed (v0.5.2)
------------------------------------
KalKalori's public flow-arrangement values are fixed to
``"counterflow"``, ``"cocurrentflow"``, ``"crossflow"`` -- there is no public
option to select mixed/unmixed behaviour. For the current bare-tube
crossflow model, the physical interpretation is fixed:

    - outside / inter-tube side = mixed (transverse mixing between
      parallel flow paths across the bundle),
    - inside / tube side        = unmixed (each tube carries its own
      stream with no transverse mixing).

"Mixed" and "unmixed" describe this transverse-mixing topology; they are
unrelated to laminar vs. turbulent flow regimes.

Because the crossflow closed-form effectiveness depends on *which physical
stream* (mixed outside vs. unmixed inside) carries C_min -- not merely on
which one is hot or cold -- ``effectiveness_ntu``/``ntu_from_effectiveness``
accept optional ``C_inside``/``C_outside`` to resolve the correct branch.
Reducing capacity rates to anonymous C_hot/C_cold too early would lose
exactly this information.
"""

from __future__ import annotations

import math
from core.heat_transfer.streams import EnergyStream


# Below this Cr, both crossflow branches converge to the same Cr->0 limit
# (eps = 1 - exp(-NTU), same as one fluid isothermal); handled directly to
# avoid a 0/0 numerical form.
_CR_ZERO_EPS = 1e-9


def effectiveness_ntu(
    C_hot: float,
    C_cold: float,
    UA: float,
    *,
    flow_arrangement: str = "counterflow",
    C_inside: float | None = None,
    C_outside: float | None = None,
) -> float:
    """
    Compute effectiveness ε using ε–NTU.

    Supported flow arrangements:
    - "counterflow"
    - "cocurrentflow"
    - "crossflow"  (outside stream mixed, tube/inside stream unmixed; see
      module docstring)

    Parameters
    ----------
    C_hot, C_cold : float
        Heat capacity rates [W/K] of the hot/cold streams.
    UA : float
        Overall conductance [W/K].
    flow_arrangement : str
        "counterflow", "cocurrentflow", or "crossflow".
    C_inside, C_outside : float, optional
        Heat capacity rates [W/K] of the physical inside (tube) and outside
        (bundle) streams. Required only for ``flow_arrangement="crossflow"``,
        where the correct closed-form branch depends on which physical
        stream (mixed outside vs. unmixed inside) carries C_min. Ignored for
        "counterflow"/"cocurrentflow" (those formulas depend only on
        C_min/C_max, not on which physical side is which).
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

    # Crossflow: outside stream mixed, inside (tube) stream unmixed.
    #
    # For the current bare-tube crossflow model:
    # - the outside stream is treated as mixed,
    # - the tube-side stream is treated as unmixed.
    #
    # Mixed/unmixed describes transverse mixing between parallel flow paths.
    # It is unrelated to laminar or turbulent flow regimes.
    #
    # Which closed form applies depends on which physical stream carries
    # C_min (Incropera Table 11.4):
    elif fa == "crossflow":
        if C_inside is None or C_outside is None:
            raise ValueError(
                "flow_arrangement='crossflow' requires C_inside and "
                "C_outside to resolve which physical stream (mixed outside "
                "vs. unmixed inside) carries C_min; see module docstring."
            )
        if not (math.isfinite(C_inside) and math.isfinite(C_outside)):
            raise ValueError("C_inside and C_outside must be finite.")
        if C_inside <= 0.0 or C_outside <= 0.0:
            raise ValueError("C_inside and C_outside must be positive.")

        if C_outside <= C_inside:
            # Case A: outside (mixed) stream is C_min.
            # eps = 1 - exp( -(1 - exp(-Cr*NTU)) / Cr )
            if C_r < _CR_ZERO_EPS:
                eps = 1.0 - math.exp(-NTU)
            else:
                eps = 1.0 - math.exp(-(1.0 - math.exp(-C_r * NTU)) / C_r)
        else:
            # Case B: inside (unmixed) stream is C_min.
            # eps = (1/Cr) * (1 - exp( -Cr*(1 - exp(-NTU)) ))
            if C_r < _CR_ZERO_EPS:
                eps = 1.0 - math.exp(-NTU)
            else:
                eps = (1.0 / C_r) * (1.0 - math.exp(-C_r * (1.0 - math.exp(-NTU))))

    else:
        raise ValueError(f"Unsupported flow_arrangement: {flow_arrangement}")

    return eps


def ntu_from_effectiveness(
    effectiveness: float,
    C_hot: float,
    C_cold: float,
    *,
    flow_arrangement: str = "counterflow",
    C_inside: float | None = None,
    C_outside: float | None = None,
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
    Cocurrentflow:
        NTU = -ln(1 - eps*(1+Cr)) / (1+Cr), valid for eps < 1/(1+Cr)
    Crossflow (outside mixed / inside unmixed; see ``effectiveness_ntu``):
        Both branches below have a stable closed-form inverse (derived by
        solving the corresponding forward relation for NTU), so no root
        solver is required.

        Case A (outside/mixed is C_min):
            NTU = -ln(1 + Cr*ln(1-eps)) / Cr,  valid for eps < 1 - exp(-1/Cr)
        Case B (inside/unmixed is C_min):
            NTU = -ln(1 + ln(1-Cr*eps)/Cr),    valid for eps < (1-exp(-Cr))/Cr

    Parameters
    ----------
    C_inside, C_outside : float, optional
        Required for ``flow_arrangement="crossflow"``; see
        ``effectiveness_ntu``.

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

    elif fa == "cocurrentflow":
        eps_max = 1.0 / (1.0 + C_r)
        if effectiveness >= eps_max:
            raise ValueError(
                f"ntu_from_effectiveness: requested effectiveness "
                f"{effectiveness:.6g} is unreachable for flow_arrangement="
                f"{flow_arrangement!r} (thermodynamic maximum eps_max="
                f"{eps_max:.6g} for C_r={C_r:.6g}), regardless of area."
            )
        NTU = -math.log(1.0 - effectiveness * (1.0 + C_r)) / (1.0 + C_r)

    elif fa == "crossflow":
        if C_inside is None or C_outside is None:
            raise ValueError(
                "flow_arrangement='crossflow' requires C_inside and "
                "C_outside to resolve which physical stream (mixed outside "
                "vs. unmixed inside) carries C_min; see module docstring."
            )
        if not (math.isfinite(C_inside) and math.isfinite(C_outside)):
            raise ValueError("C_inside and C_outside must be finite.")
        if C_inside <= 0.0 or C_outside <= 0.0:
            raise ValueError("C_inside and C_outside must be positive.")

        if C_outside <= C_inside:
            # Case A (outside/mixed is C_min): NTU = -ln(1 + Cr*ln(1-eps)) / Cr
            if C_r < _CR_ZERO_EPS:
                NTU = -math.log(1.0 - effectiveness)
            else:
                eps_max = 1.0 - math.exp(-1.0 / C_r)
                if effectiveness >= eps_max:
                    raise ValueError(
                        f"ntu_from_effectiveness: requested effectiveness "
                        f"{effectiveness:.6g} is unreachable for crossflow "
                        f"with the outside (mixed) stream as C_min "
                        f"(eps_max={eps_max:.6g} for C_r={C_r:.6g}), "
                        "regardless of area."
                    )
                NTU = -math.log(1.0 + C_r * math.log(1.0 - effectiveness)) / C_r
        else:
            # Case B (inside/unmixed is C_min): NTU = -ln(1 + ln(1-Cr*eps)/Cr)
            if C_r < _CR_ZERO_EPS:
                NTU = -math.log(1.0 - effectiveness)
            else:
                eps_max = (1.0 - math.exp(-C_r)) / C_r
                if effectiveness >= eps_max:
                    raise ValueError(
                        f"ntu_from_effectiveness: requested effectiveness "
                        f"{effectiveness:.6g} is unreachable for crossflow "
                        f"with the inside (unmixed) stream as C_min "
                        f"(eps_max={eps_max:.6g} for C_r={C_r:.6g}), "
                        "regardless of area."
                    )
                NTU = -math.log(1.0 + math.log(1.0 - C_r * effectiveness) / C_r)

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
