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

# NOTE ON UNITS
# -------------
# All calculations in the KalKalori core engine are performed using
# plain Python floats in consistent SI units.

from __future__ import annotations

import math
from dataclasses import dataclass

from core.geometry.bundle import TubeBundle

from core.heat_transfer.internal_flow import (
    FluidProps as InternalFlowFluidProps,
    heat_transfer_coefficient_internal,
)

from core.heat_transfer.internal_pressure_drop import (
    FluidProps as InternalDPFluidProps,
    pressure_drop_internal_total,
)

from core.heat_transfer.outside_flow import (
    FluidProps as OutsideFlowFluidProps,
    outside_flow_from_mass_flow,
)

from core.heat_transfer.ntu import (
    effectiveness_ntu,
    heat_duty_from_effectiveness,
)

from core.heat_transfer.streams import EnergyStream


@dataclass(frozen=True)
class HXOutSideThermalResults:
    v: float
    Re: float
    Pr: float
    h: float


@dataclass(frozen=True)
class HXTubeSideHydraulicResults:
    dp_total: float
    dp_tubes: float
    dp_inlet: float
    dp_outlet: float
    dp_turns: float
    Re: float
    f: float
    v: float


@dataclass(frozen=True)
class HXOutSideHydraulicResults:
    dp_total: float
    Re: float
    v: float


@dataclass(frozen=True)
class HXResult:
    # Geometry / areas
    A_i: float          # [m^2] inner heat transfer area (effective)
    A_o: float          # [m^2] outer heat transfer area (effective)
    A_frontal: float    # [m^2] frontal area for outside approach flow (effective)

    # Thermal performance
    UA: float           # [W/K]
    eps: float          # [-]
    Q: float            # [W]
    T_hot_out: float    # [K]
    T_cold_out: float   # [K]

    # Diagnostics
    tube_side_thermal: HXOutSideThermalResults
    tube_side_hydraulic: HXTubeSideHydraulicResults

    outside_side_thermal: HXOutSideThermalResults
    outside_side_hydraulic: HXOutSideHydraulicResults


class BareTubeHeatExchanger:
    """
    Bare (smooth) tube heat exchanger model (MVP).

    Implemented features (current stage)
    ------------------------------------
    - Tube-side: internal convection correlation -> h_i
    - Tube-side: component-based pressure drop:
        dp_total = dp_tubes + dp_inlet + dp_outlet + dp_turns
    - Outside side: forced flow from mass flow rate -> h_o and dp_o (MVP)
    - Overall thermal duty: ε–NTU with flow_arrangement:
        "counterflow", "cocurrentflow", "crossflow"

    Literature anchors in code
    --------------------------
    - ε–NTU: Incropera et al., Heat Exchangers chapter
    - Darcy–Weisbach & K-loss decomposition: White; Idelchik; Crane TP-410
    """

    def __init__(self, bundle: TubeBundle, wall_k: float | None = None):
        self.bundle = bundle
        self.wall_k = wall_k

    def _tube_wall_resistance(self) -> float:
        """
        Cylindrical wall conduction resistance (total) [K/W].

        R_wall = ln(Do/Di) / (2*pi*k*L_eff*N_tubes)

        Ref: standard conduction through cylindrical wall (heat transfer textbooks).
        """
        if self.wall_k is None:
            return 0.0
        if self.wall_k <= 0.0:
            raise ValueError("wall_k must be positive.")

        tube = self.bundle.tube
        if not all(hasattr(tube, a) for a in ("D_i", "D_o", "length_effective")):
            raise ValueError("Tube must provide D_i, D_o, length_effective for wall resistance.")

        Di = float(getattr(tube, "D_i"))
        Do = float(getattr(tube, "D_o"))
        L_eff = float(getattr(tube, "length_effective"))
        if Do <= Di:
            raise ValueError("Tube outer diameter must exceed inner diameter.")

        N = self.bundle.n_tubes_total
        return math.log(Do / Di) / (2.0 * math.pi * self.wall_k * L_eff * N)

    def solve(
        self,
        hot_stream: EnergyStream,
        cold_stream: EnergyStream,
        *,
        # Tube-side (total across exchanger):
        m_dot_tube_side: float,
        tube_side_props: InternalFlowFluidProps,

        # Outside-side (preferred path):
        m_dot_outside: float | None = None,
        outside_props: OutsideFlowFluidProps | None = None,
        zeta_dp_outside: float = 1.2,

        # Tube-side DP coefficients (MVP defaults exist in DP module; caller may override):
        K_inlet: float = 0.5,
        K_outlet: float = 1.0,
        K_turn: float = 1.5,

        # Outside h override (validation / calibration):
        h_o: float | None = None,

        # Thermal flow arrangement:
        flow_arrangement: str | None = None,

    ) -> HXResult:
        # Use bundle's flow_arrangement if not provided
        if flow_arrangement is None:
            flow_arrangement = self.bundle.flow_arrangement

        if m_dot_tube_side <= 0.0:
            raise ValueError("m_dot_tube_side must be positive.")

        # --- Areas (effective) ---
        A_i = self.bundle.total_inner_area
        A_o = self.bundle.total_outer_area
        A_frontal = self.bundle.frontal_flow_area

        # --------------------------------------------------------------
        # Tube-side: thermal (per pass flow area)
        # --------------------------------------------------------------
        # m_dot_tube_side is total across exchanger. In a multi-pass bundle,
        # the flow is distributed among tubes within a pass.
        flow_area_pass = self.bundle.internal_flow_area_per_pass
        D_h = self.bundle.internal_hydraulic_diameter

        v_i, Re_i, Pr_i, h_i = heat_transfer_coefficient_internal(
            m_dot=m_dot_tube_side,
            tube_inner_diameter=D_h,
            flow_area=flow_area_pass,
            props=tube_side_props,
        )
        tube_thermal = HXOutSideThermalResults(v=v_i, Re=Re_i, Pr=Pr_i, h=h_i)

        # --------------------------------------------------------------
        # Tube-side: hydraulic (component-based, includes passes)
        # --------------------------------------------------------------
        dp_total, dp_tubes, dp_in, dp_out, dp_turns, Re_dp, f, v_dp = pressure_drop_internal_total(
            m_dot=m_dot_tube_side,
            flow_area=flow_area_pass,
            hydraulic_diameter=D_h,
            flow_length=self.bundle.internal_length_total,
            props=InternalDPFluidProps(rho=tube_side_props.rho, mu=tube_side_props.mu),
            n_turns=self.bundle.n_turns,
            K_in=K_inlet,
            K_out=K_outlet,
            K_turn=K_turn,
        )

        tube_hyd = HXTubeSideHydraulicResults(
            dp_total=dp_total,
            dp_tubes=dp_tubes,
            dp_inlet=dp_in,
            dp_outlet=dp_out,
            dp_turns=dp_turns,
            Re=Re_dp,
            f=f,
            v=v_dp,
        )

        # --------------------------------------------------------------
        # Outside-side: compute from mass flow unless overridden
        # --------------------------------------------------------------
        have_outside = (m_dot_outside is not None) and (outside_props is not None)

        if have_outside:
            v_o, Re_o, Pr_o, h_o_calc, dp_o = outside_flow_from_mass_flow(
                m_dot=m_dot_outside,
                frontal_area=A_frontal,
                tube_outer_diameter=float(getattr(self.bundle.tube, "D_o")),
                n_rows=self.bundle.n_rows,
                props=outside_props,
                zeta_dp=zeta_dp_outside,
            )
        else:
            v_o, Re_o, Pr_o, h_o_calc, dp_o = float("nan"), float("nan"), float("nan"), None, float("nan")

        if h_o is not None:
            if h_o <= 0.0:
                raise ValueError("h_o must be positive when provided.")
            h_o_used = h_o
        else:
            if h_o_calc is None:
                raise ValueError(
                    "Outside side not specified. Provide either:\n"
                    "- (m_dot_outside and outside_props) to compute h_o, or\n"
                    "- h_o directly as an override."
                )
            h_o_used = h_o_calc

        outside_thermal = HXOutSideThermalResults(v=v_o, Re=Re_o, Pr=Pr_o, h=h_o_used)
        outside_hyd = HXOutSideHydraulicResults(dp_total=dp_o, Re=Re_o, v=v_o)

        # --------------------------------------------------------------
        # Overall UA
        # --------------------------------------------------------------
        R_i = 1.0 / (h_i * A_i)
        R_o = 1.0 / (h_o_used * A_o)
        R_w = self._tube_wall_resistance()

        R_tot = R_i + R_w + R_o
        if R_tot <= 0.0:
            raise ValueError("Invalid total thermal resistance.")

        UA = 1.0 / R_tot

        # --------------------------------------------------------------
        # ε–NTU thermal duty
        # --------------------------------------------------------------
        eps = effectiveness_ntu(
            C_hot=hot_stream.capacity_rate(),
            C_cold=cold_stream.capacity_rate(),
            UA=UA,
            flow_arrangement=flow_arrangement,
        )

        Q, T_hot_out, T_cold_out = heat_duty_from_effectiveness(
            eps=eps,
            hot_stream=hot_stream,
            cold_stream=cold_stream,
        )

        return HXResult(
            A_i=A_i,
            A_o=A_o,
            A_frontal=A_frontal,
            UA=UA,
            eps=eps,
            Q=Q,
            T_hot_out=T_hot_out,
            T_cold_out=T_cold_out,
            tube_side_thermal=tube_thermal,
            tube_side_hydraulic=tube_hyd,
            outside_side_thermal=outside_thermal,
            outside_side_hydraulic=outside_hyd,
        )
