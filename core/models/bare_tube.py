# KalKalori â€” Heat Exchanger Open Engine
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
from dataclasses import dataclass, replace

from core.geometry.bundle import TubeBundle

from core.heat_transfer.internal_flow import (
    FluidProps as InternalFlowFluidProps,
    heat_transfer_coefficient_internal,
)

from core.pressure_drop.internal_pressure_drop import (
    TubeBundleHydraulicResult,
    TubeSideHydraulicPoint,
    calculate_tube_bundle_hydraulics,
)

from core.heat_transfer.outside_flow import (
    FluidProps as OutsideFlowFluidProps,
    OutsideHydraulicPoint,
    OutsideTubeBankHydraulicResult,
    calculate_outside_tube_bank_hydraulics,
    outside_flow_from_mass_flow,
)

from core.pressure_drop.outside_pressure_drop import (
    EulerProvider,
)

from core.pressure_drop.flow_path import (
    PressureDropPathResult,
    build_tube_side_pressure_drop_result,
    build_outside_pressure_drop_result,
)

from core.heat_transfer.ntu import (
    effectiveness_ntu,
    heat_duty_from_effectiveness,
)

from core.heat_transfer.streams import EnergyStream

from core.common.warnings import (
    ModelWarning,
    make_warning,
)
from core.properties.common import FluidTransportProperties
from core.properties.fluids import PropertyProvider


@dataclass(frozen=True)
class HXOutSideThermalResults:
    v: float
    Re: float
    Pr: float
    alfa: float


@dataclass(frozen=True)
class HXTubeSideHydraulicResults:
    """Tube-side tube-bundle hydraulics and compatibility accessors.

    Since v0.5.6 the nested result contains distributed straight-tube
    friction, signed acceleration pressure change, and tube-sheet entrance/
    exit losses (see ``dp_straight_tubes`` for the pre-v0.5.6 friction +
    acceleration-only scope).  ``dp_inlet``, ``dp_outlet`` and ``dp_turns``
    remain as zero-valued compatibility accessors for the older,
    unused K_inlet/K_outlet/K_turn helpers; they are unrelated to the new
    tube-sheet entrance/exit calculation.
    """

    tube_bundle: TubeBundleHydraulicResult

    @property
    def dp_total(self) -> float:
        return self.tube_bundle.dp_tube_bundle

    @property
    def dp_tubes(self) -> float:
        return self.tube_bundle.dp_friction

    @property
    def dp_inlet(self) -> float:
        return 0.0

    @property
    def dp_outlet(self) -> float:
        return 0.0

    @property
    def dp_turns(self) -> float:
        return 0.0

    @property
    def Re(self) -> float:
        return self.tube_bundle.midpoint.reynolds

    @property
    def f(self) -> float:
        return self.tube_bundle.midpoint.friction_factor

    @property
    def v(self) -> float:
        return self.tube_bundle.midpoint.velocity

    @property
    def dp_friction(self) -> float:
        return self.tube_bundle.dp_friction

    @property
    def dp_acceleration(self) -> float:
        return self.tube_bundle.dp_acceleration

    @property
    def dp_straight_tubes(self) -> float:
        return self.tube_bundle.dp_straight_tubes

    @property
    def dp_tube_entrances(self) -> float:
        return self.tube_bundle.dp_tube_entrances

    @property
    def dp_tube_exits(self) -> float:
        return self.tube_bundle.dp_tube_exits

    @property
    def dp_tube_bundle(self) -> float:
        return self.tube_bundle.dp_tube_bundle

    @property
    def tube_path_type(self):
        return self.tube_bundle.tube_path_type

    @property
    def entrance_count(self) -> int:
        return self.tube_bundle.entrance_count

    @property
    def exit_count(self) -> int:
        return self.tube_bundle.exit_count

    @property
    def pass_boundary_method(self) -> str:
        return self.tube_bundle.pass_boundary_method

    @property
    def pass_boundary_states(self):
        return self.tube_bundle.pass_boundary_states

    @property
    def entrance_results(self):
        return self.tube_bundle.entrance_results

    @property
    def exit_results(self):
        return self.tube_bundle.exit_results

    @property
    def inlet(self):
        return self.tube_bundle.inlet

    @property
    def midpoint(self):
        return self.tube_bundle.midpoint

    @property
    def outlet(self):
        return self.tube_bundle.outlet


@dataclass(frozen=True)
class HXOutSideHydraulicResults:
    dp_total: float
    Re: float
    v: float
    tube_bank: OutsideTubeBankHydraulicResult | None = None

    @property
    def dp_drag(self) -> float:
        return float("nan") if self.tube_bank is None else self.tube_bank.dp_drag

    @property
    def dp_acceleration(self) -> float:
        return float("nan") if self.tube_bank is None else self.tube_bank.dp_acceleration

    @property
    def dp_outside_total(self) -> float:
        return self.dp_total

    @property
    def outside_dp_drag(self) -> float:
        return self.dp_drag

    @property
    def outside_dp_acceleration(self) -> float:
        return self.dp_acceleration

    @property
    def outside_dp(self) -> float:
        """Compatibility alias for total bare-bank pressure change."""
        return self.dp_total

    @property
    def outside_pressure_drop(self) -> float:
        return self.dp_total

    @property
    def inlet(self):
        return None if self.tube_bank is None else self.tube_bank.inlet

    @property
    def midpoint(self):
        return None if self.tube_bank is None else self.tube_bank.midpoint

    @property
    def outlet(self):
        return None if self.tube_bank is None else self.tube_bank.outlet


@dataclass(frozen=True)
class HXTubeSidePressureDropResults:
    """Tube-side complete pressure-drop result (v0.5.6 architecture).

    ``tube_bundle`` is the existing calculated straight tube-bundle result,
    decomposed in ``flow_path`` into irreversible core loss and signed
    dynamic-pressure change. With no explicitly calculated local-loss path,
    ``dp_local`` is ``0.0`` and ``dp_static_total`` equals the legacy signed
    ``tube_bundle.dp_tube_bundle`` pressure difference exactly.
    """

    tube_bundle: TubeBundleHydraulicResult
    flow_path: PressureDropPathResult

    @property
    def dp_core(self) -> float:
        return self.flow_path.dp_core

    @property
    def dp_local(self) -> float:
        return self.flow_path.dp_local

    @property
    def dp_total(self) -> float:
        return self.flow_path.dp_total

    @property
    def delta_dynamic_pressure_core(self) -> float:
        return self.flow_path.delta_dynamic_pressure_core

    @property
    def delta_dynamic_pressure_local(self) -> float:
        return self.flow_path.delta_dynamic_pressure_local

    @property
    def delta_dynamic_pressure_total(self) -> float:
        return self.flow_path.delta_dynamic_pressure_total

    @property
    def dp_static_core(self) -> float:
        return self.flow_path.dp_static_core

    @property
    def dp_static_local(self) -> float:
        return self.flow_path.dp_static_local

    @property
    def dp_static_total(self) -> float:
        return self.flow_path.dp_static_total


@dataclass(frozen=True)
class HXOutsidePressureDropResults:
    """Outside-side complete pressure-drop result (v0.5.6 architecture).

    ``tube_bank`` is the existing calculated bare tube-bank result (``None``
    when the outside side was not specified), decomposed in ``flow_path``
    into irreversible drag and signed dynamic-pressure change. With no
    explicitly calculated local-loss path, ``dp_local`` is ``0.0`` and
    ``dp_static_total`` equals the legacy signed ``tube_bank.dp_total``
    pressure difference exactly.
    """

    tube_bank: OutsideTubeBankHydraulicResult | None
    flow_path: PressureDropPathResult

    @property
    def dp_core(self) -> float:
        return self.flow_path.dp_core

    @property
    def dp_local(self) -> float:
        return self.flow_path.dp_local

    @property
    def dp_total(self) -> float:
        return self.flow_path.dp_total

    @property
    def delta_dynamic_pressure_core(self) -> float:
        return self.flow_path.delta_dynamic_pressure_core

    @property
    def delta_dynamic_pressure_local(self) -> float:
        return self.flow_path.delta_dynamic_pressure_local

    @property
    def delta_dynamic_pressure_total(self) -> float:
        return self.flow_path.delta_dynamic_pressure_total

    @property
    def dp_static_core(self) -> float:
        return self.flow_path.dp_static_core

    @property
    def dp_static_local(self) -> float:
        return self.flow_path.dp_static_local

    @property
    def dp_static_total(self) -> float:
        return self.flow_path.dp_static_total


@dataclass(frozen=True)
class HXResult:
    """Single-pass exchanger result and its solver-owned diagnostics.

    ``inside_properties_*`` and ``outside_properties_*`` expose the existing
    inlet/midpoint/outlet hydraulic points. Each point contains the exact
    temperature, representative pressure, transport properties, and Prandtl
    number used by the hydraulic calculation. The midpoint definition is
    recorded on ``tube_bundle_hydraulic.midpoint_method`` or
    ``outside_tube_bank_hydraulic.midpoint_method``.

    These point states are distinct from representative 0D thermal
    properties reported by higher-level Simulation/Rating results.
    """

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

    # Complete pressure-drop flow-path results (v0.5.6 architecture):
    # stage-by-stage/grouped irreversible losses plus separate signed
    # dynamic- and static-pressure changes for each side.
    tube_side_pressure_drop: HXTubeSidePressureDropResults
    outside_side_pressure_drop: HXOutsidePressureDropResults

    # Warnings and applicability diagnostics
    warnings: list[ModelWarning] | None = None

    @property
    def tube_bundle_hydraulic(self) -> TubeBundleHydraulicResult:
        return self.tube_side_hydraulic.tube_bundle

    @property
    def inside_properties_inlet(self) -> TubeSideHydraulicPoint:
        return self.tube_bundle_hydraulic.inlet

    @property
    def inside_properties_midpoint(self) -> TubeSideHydraulicPoint:
        return self.tube_bundle_hydraulic.midpoint

    @property
    def inside_properties_outlet(self) -> TubeSideHydraulicPoint:
        return self.tube_bundle_hydraulic.outlet

    @property
    def outside_properties_inlet(self) -> OutsideHydraulicPoint | None:
        hydraulic = self.outside_tube_bank_hydraulic
        return None if hydraulic is None else hydraulic.inlet

    @property
    def outside_properties_midpoint(self) -> OutsideHydraulicPoint | None:
        hydraulic = self.outside_tube_bank_hydraulic
        return None if hydraulic is None else hydraulic.midpoint

    @property
    def outside_properties_outlet(self) -> OutsideHydraulicPoint | None:
        hydraulic = self.outside_tube_bank_hydraulic
        return None if hydraulic is None else hydraulic.outlet

    @property
    def inside_dp_friction(self) -> float:
        return self.tube_side_hydraulic.dp_friction

    @property
    def inside_dp_acceleration(self) -> float:
        return self.tube_side_hydraulic.dp_acceleration

    @property
    def inside_dp_tube_bundle(self) -> float:
        return self.tube_side_hydraulic.dp_tube_bundle

    @property
    def inside_dp_total(self) -> float:
        """Compatibility alias for the straight tube-bundle pressure change."""
        return self.inside_dp_tube_bundle

    # -- Canonical tube-side pressure-drop field names (v0.5.6) -----------
    # The tube-side core now covers distributed straight-tube friction,
    # signed acceleration pressure change, and tube-sheet entrance/exit
    # losses (dp_tube_bundle = dp_straight_tubes + dp_tube_entrances +
    # dp_tube_exits). ``inside_dp_local`` remains 0.0 in this commit: no
    # inlet/return/outlet local-loss path is automatically invoked (see
    # core.pressure_drop.flow_path). ``inside_dp_tube_bundle`` ==
    # ``inside_dp_total``.

    @property
    def inside_dp_tube_bundle_friction(self) -> float:
        return self.tube_side_hydraulic.dp_friction

    @property
    def inside_dp_tube_bundle_acceleration(self) -> float:
        return self.tube_side_hydraulic.dp_acceleration

    @property
    def inside_dp_straight_tube_friction(self) -> float:
        return self.tube_side_hydraulic.dp_friction

    @property
    def inside_dp_straight_tube_acceleration(self) -> float:
        return self.tube_side_hydraulic.dp_acceleration

    @property
    def inside_dp_straight_tubes(self) -> float:
        return self.tube_side_hydraulic.dp_straight_tubes

    @property
    def inside_dp_tube_entrances(self) -> float:
        return self.tube_side_hydraulic.dp_tube_entrances

    @property
    def inside_dp_tube_exits(self) -> float:
        return self.tube_side_hydraulic.dp_tube_exits

    @property
    def tube_path_type(self):
        return self.tube_side_hydraulic.tube_path_type

    @property
    def entrance_count(self) -> int:
        return self.tube_side_hydraulic.entrance_count

    @property
    def exit_count(self) -> int:
        return self.tube_side_hydraulic.exit_count

    @property
    def pass_boundary_method(self) -> str:
        return self.tube_side_hydraulic.pass_boundary_method

    @property
    def pass_boundary_states(self):
        return self.tube_side_hydraulic.pass_boundary_states

    @property
    def entrance_results(self):
        return self.tube_side_hydraulic.entrance_results

    @property
    def exit_results(self):
        return self.tube_side_hydraulic.exit_results

    @property
    def inside_dp_local(self) -> float:
        return self.tube_side_pressure_drop.dp_local

    @property
    def outside_tube_bank_hydraulic(self) -> OutsideTubeBankHydraulicResult | None:
        return self.outside_side_hydraulic.tube_bank

    @property
    def outside_dp_drag(self) -> float:
        return self.outside_side_hydraulic.dp_drag

    @property
    def outside_dp_acceleration(self) -> float:
        return self.outside_side_hydraulic.dp_acceleration

    @property
    def outside_dp_total(self) -> float:
        return self.outside_side_hydraulic.dp_outside_total

    @property
    def outside_dp(self) -> float:
        return self.outside_dp_total

    @property
    def outside_pressure_drop(self) -> float:
        return self.outside_dp_total

    # -- Canonical outside-side pressure-drop field names (v0.5.6) --------
    # The outside result covers only bare tube-bank Euler drag and signed
    # acceleration pressure change; ``outside_dp_local`` is 0.0 and
    # ``outside_dp_tube_bank`` == ``outside_dp_total`` because the standard
    # solver does not invoke the separately available local-path models.

    @property
    def outside_dp_tube_bank_drag(self) -> float:
        return self.outside_side_hydraulic.dp_drag

    @property
    def outside_dp_tube_bank_acceleration(self) -> float:
        return self.outside_side_hydraulic.dp_acceleration

    @property
    def outside_dp_tube_bank(self) -> float:
        return self.outside_dp_total

    @property
    def outside_dp_local(self) -> float:
        return self.outside_side_pressure_drop.dp_local


class BareTubeHeatExchanger:
    """
    Bare (smooth) tube heat exchanger model (MVP).

    The thermal conductivity of the tube wall is no longer passed to the
    exchanger; it is expected to be supplied on the ``bundle.tube``
    geometry object (``BareTube.wall_k``).

    Implemented features (current stage)
    ------------------------------------
    - Tube-side: internal convection correlation -> alfa_i
    - Tube-side: universal three-state straight-bundle hydraulics plus
      tube-sheet entrance/exit losses (v0.5.6):
        dp_tube_bundle = dp_straight_tubes + dp_tube_entrances + dp_tube_exits
        dp_straight_tubes = dp_friction + dp_acceleration
      with no nozzle, chamber, return, bend, header, or collector
      local-loss coefficients (those remain in the optional, explicitly
      invoked pressure-drop path architecture; see core.pressure_drop.flow_path).
    - Outside side: forced flow from mass flow rate -> alfa_o and universal
      three-state tube-bank drag plus signed acceleration pressure change
    - Overall thermal duty: Îµâ€“NTU with flow_arrangement:
        "counterflow", "cocurrentflow", "crossflow"

    Literature anchors in code
    --------------------------
    - Îµâ€“NTU: Incropera et al., Heat Exchangers chapter
    - Darcyâ€“Weisbach & K-loss decomposition: White; Idelchik; Crane TP-410
    """

    def __init__(self, bundle: TubeBundle):
        # conductivity is stored on the tube geometry itself
        self.bundle = bundle

    def tube_wall_resistance(self) -> float:
        """Public accessor for the cylindrical tube-wall conduction resistance [K/W].

        Exposed so orchestration layers (e.g.
        ``core.heat_transfer.thermal_iteration``) can reuse the existing
        wall-resistance model without duplicating it.
        """
        return self._tube_wall_resistance()

    def _tube_wall_resistance(self) -> float:
        """
        Cylindrical wall conduction resistance (total) [K/W].

        R_wall = ln(Do/Di) / (2*pi*k*L_eff*N_tubes)

        Ref: standard conduction through cylindrical wall (heat transfer textbooks).
        """
        # conductivity should be defined on the tube object; if absent we
        # assume a negligible wall resistance.
        tube = self.bundle.tube
        k = getattr(tube, "wall_k", None)
        if k is None:
            return 0.0
        if k <= 0.0:
            raise ValueError("wall_k must be positive.")

        # tube geometry validation is handled by the BareTube class, but we
        # keep a couple of defensive checks in case a different implementation
        # is used.
        Di = float(getattr(tube, "D_i"))
        Do = float(getattr(tube, "D_o"))
        L_eff = float(getattr(tube, "length_effective"))
        if Do <= Di:
            raise ValueError("Tube outer diameter must exceed inner diameter.")

        N = self.bundle.n_tubes_total
        return math.log(Do / Di) / (2.0 * math.pi * k * L_eff * N)

    def solve(
        self,
        hot_stream: EnergyStream,
        cold_stream: EnergyStream,
        *,
        # Tube-side (total across exchanger):
        m_dot_tube_side: float,
        tube_side_props: InternalFlowFluidProps,
        tube_side_provider: PropertyProvider | None = None,
        tube_side_temperature_in: float | None = None,
        tube_side_temperature_out: float | None = None,
        tube_side_pressure: float | None = None,

        # Outside-side (preferred path):
        m_dot_outside: float | None = None,
        outside_props: OutsideFlowFluidProps | None = None,
        outside_provider: PropertyProvider | None = None,
        outside_temperature_in: float | None = None,
        outside_temperature_out: float | None = None,
        outside_pressure: float | None = None,

        # Retained for public-call compatibility. Local losses are outside the
        # v0.5.5 straight-tube-bundle scope and these values are not applied.
        K_inlet: float = 0.5,
        K_outlet: float = 1.0,
        K_turn: float = 1.5,

        # Outside alfa override (validation / calibration):
        alfa_o: float | None = None,

        # Thermal flow arrangement:
        flow_arrangement: str | None = None,

        # Outside pressure-drop provider selection:
        euler_provider: str | EulerProvider = "zukauskas",

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

        v_i, Re_i, Pr_i, alfa_i, internal_ht_warnings = heat_transfer_coefficient_internal(
            m_dot=m_dot_tube_side,
            tube_inner_diameter=D_h,
            flow_area=flow_area_pass,
            props=tube_side_props,
        )
        tube_thermal = HXOutSideThermalResults(v=v_i, Re=Re_i, Pr=Pr_i, alfa=alfa_i)

        # --------------------------------------------------------------
        # Tube-side: universal three-state straight-bundle hydraulics.  The
        # representative pressure is held constant for all three provider
        # evaluations; no pressure-distribution iteration is introduced here.
        # --------------------------------------------------------------
        if tube_side_provider is None:
            hydraulic_props = FluidTransportProperties(
                rho=tube_side_props.rho,
                mu=tube_side_props.mu,
                k=tube_side_props.k,
                cp=tube_side_props.cp,
            )
            tube_bundle_hydraulic = calculate_tube_bundle_hydraulics(
                m_dot=m_dot_tube_side,
                flow_area_per_pass=flow_area_pass,
                hydraulic_diameter=D_h,
                hydraulic_length_total=self.bundle.internal_length_total,
                n_tube_passes=self.bundle.n_passes_tube,
                tube_path_type=self.bundle.tube_path_type,
                roughness_inner=getattr(self.bundle.tube, "roughness_inner", None),
                inlet_props=hydraulic_props,
                temperature_in=tube_side_temperature_in,
                temperature_out=tube_side_temperature_out,
                pressure=tube_side_pressure,
            )
        else:
            T_in_hyd = 300.0 if tube_side_temperature_in is None else tube_side_temperature_in
            T_out_hyd = T_in_hyd if tube_side_temperature_out is None else tube_side_temperature_out
            p_hyd = 101325.0 if tube_side_pressure is None else tube_side_pressure
            tube_bundle_hydraulic = calculate_tube_bundle_hydraulics(
                m_dot=m_dot_tube_side,
                flow_area_per_pass=flow_area_pass,
                hydraulic_diameter=D_h,
                hydraulic_length_total=self.bundle.internal_length_total,
                n_tube_passes=self.bundle.n_passes_tube,
                tube_path_type=self.bundle.tube_path_type,
                roughness_inner=getattr(self.bundle.tube, "roughness_inner", None),
                provider=tube_side_provider,
                temperature_in=T_in_hyd,
                temperature_out=T_out_hyd,
                pressure=p_hyd,
            )

        tube_hyd = HXTubeSideHydraulicResults(
            tube_bundle=tube_bundle_hydraulic,
        )

        # --------------------------------------------------------------
        # Outside-side: compute from mass flow unless overridden
        # --------------------------------------------------------------
        have_outside = (m_dot_outside is not None) and (outside_props is not None)

        if have_outside:
            # Keep the existing outside heat-transfer calculation and its
            # face-velocity/Re/Pr contract, but bypass its historical
            # one-state pressure-drop calculation.  Outside pressure is
            # authoritative only from the three-state bank result below.
            v_o, Re_o, Pr_o, alfa_o_calc, _legacy_dp_o, outside_warnings, _outside_euler_result = outside_flow_from_mass_flow(
                m_dot=m_dot_outside,
                frontal_area=A_frontal,
                n_tubes_per_row=self.bundle.n_tubes_per_row,
                tube_outer_diameter=float(getattr(self.bundle.tube, "D_o")),
                tube_pitch_transverse=self.bundle.pitch_transverse,
                tube_pitch_longitudinal=self.bundle.pitch_longitudinal,
                layout=self.bundle.layout,
                n_rows=self.bundle.n_rows,
                props=outside_props,
                euler_provider=euler_provider,
                calculate_pressure_drop=False,
            )
            outside_bank_hydraulic = calculate_outside_tube_bank_hydraulics(
                m_dot=m_dot_outside,
                face_area=A_frontal,
                tube_outer_diameter=float(getattr(self.bundle.tube, "D_o")),
                tube_pitch_transverse=self.bundle.pitch_transverse,
                tube_pitch_longitudinal=self.bundle.pitch_longitudinal,
                layout=self.bundle.layout,
                n_rows=self.bundle.n_rows,
                n_tubes_per_row=self.bundle.n_tubes_per_row,
                provider=outside_provider,
                temperature_in=outside_temperature_in,
                temperature_out=outside_temperature_out,
                pressure=outside_pressure,
                euler_provider=euler_provider,
                inlet_props=outside_props if outside_provider is None else None,
            )
            dp_o = outside_bank_hydraulic.dp_total
        else:
            v_o, Re_o, Pr_o, alfa_o_calc, dp_o, outside_warnings, _outside_euler_result = (
                float("nan"),
                float("nan"),
                float("nan"),
                None,
                float("nan"),
                [],
                None,
            )
            outside_bank_hydraulic = None

        if alfa_o is not None:
            if alfa_o <= 0.0:
                raise ValueError("alfa_o must be positive when provided.")
            alfa_o_used = alfa_o
        else:
            if alfa_o_calc is None:
                raise ValueError(
                    "Outside side not specified. Provide either:\n"
                    "- (m_dot_outside and outside_props) to compute alfa_o, or\n"
                    "- alfa_o directly as an override."
                )
            alfa_o_used = alfa_o_calc

        outside_thermal = HXOutSideThermalResults(v=v_o, Re=Re_o, Pr=Pr_o, alfa=alfa_o_used)
        outside_hyd = HXOutSideHydraulicResults(
            dp_total=dp_o,
            Re=(
                outside_bank_hydraulic.midpoint.reynolds
                if outside_bank_hydraulic is not None
                else Re_o
            ),
            v=(
                outside_bank_hydraulic.midpoint.face_velocity
                if outside_bank_hydraulic is not None
                else v_o
            ),
            tube_bank=outside_bank_hydraulic,
        )

        # --------------------------------------------------------------
        # Overall UA
        # --------------------------------------------------------------
        R_i = 1.0 / (alfa_i * A_i)
        R_o = 1.0 / (alfa_o_used * A_o)
        R_w = self._tube_wall_resistance()

        R_tot = R_i + R_w + R_o
        if R_tot <= 0.0:
            raise ValueError("Invalid total thermal resistance.")

        UA = 1.0 / R_tot

        # --------------------------------------------------------------
        # Îµâ€“NTU thermal duty
        # --------------------------------------------------------------
        # C_inside/C_outside are derived from the physical tube-side and
        # outside-side inputs directly (not from hot_stream/cold_stream),
        # so their *physical* identity (which is mixed, which is unmixed)
        # is preserved regardless of which one happens to be hot or cold.
        # This is required for the crossflow branch below; see
        # core.heat_transfer.ntu module docstring.
        C_inside = m_dot_tube_side * tube_side_props.cp
        C_outside = (
            m_dot_outside * outside_props.cp
            if (have_outside and outside_props is not None)
            else None
        )
        if flow_arrangement.lower() == "crossflow" and C_outside is None:
            raise ValueError(
                "flow_arrangement='crossflow' requires the outside side to "
                "be specified via (m_dot_outside and outside_props) so that "
                "C_outside can be resolved; the alfa_o override path alone "
                "does not carry enough information to identify the mixed "
                "(outside) stream's capacity rate."
            )

        eps = effectiveness_ntu(
            C_hot=hot_stream.capacity_rate(),
            C_cold=cold_stream.capacity_rate(),
            UA=UA,
            flow_arrangement=flow_arrangement,
            C_inside=C_inside,
            C_outside=C_outside,
        )

        Q, T_hot_out, T_cold_out = heat_duty_from_effectiveness(
            eps=eps,
            hot_stream=hot_stream,
            cold_stream=cold_stream,
        )

        # --------------------------------------------------------------
        # Collect warnings
        # --------------------------------------------------------------
        warnings_list: list[ModelWarning] = []

        # Internal heat-transfer applicability warnings (e.g. gas
        # wall-property correction, only non-empty when T_bulk/T_wall are
        # supplied to heat_transfer_coefficient_internal).
        warnings_list.extend(internal_ht_warnings)
        warnings_list.extend(tube_bundle_hydraulic.warnings)

        # Add outside flow applicability warnings
        if have_outside:
            warnings_list.extend(outside_warnings)
            existing_outside_warning_ids = {
                (warning.source, warning.code) for warning in warnings_list
            }
            for warning in outside_bank_hydraulic.warnings:
                if (warning.source, warning.code) not in existing_outside_warning_ids:
                    warnings_list.append(warning)
                    existing_outside_warning_ids.add((warning.source, warning.code))

        # Tube-side regime diagnostics (most internal correlations assume turbulence).
        if Re_i < 2300.0:
            warnings_list.append(
                make_warning(
                    code="tube_ht_laminar_regime",
                    message="tube_ht: tube-side Reynolds number indicates laminar flow while turbulent behavior is expected by the selected model.",
                    source="tube_ht",
                    severity="warning",
                )
            )
        elif 2300.0 <= Re_i <= 4000.0:
            warnings_list.append(
                make_warning(
                    code="tube_ht_transition_regime",
                    message="tube_ht: tube-side Reynolds number is in the transition region; heat-transfer prediction uncertainty is increased.",
                    source="tube_ht",
                    severity="warning",
                )
            )

        # Outside-side regime diagnostics when outside flow is computed.
        if have_outside and not math.isnan(Re_o):
            if Re_o < 2300.0:
                warnings_list.append(
                    make_warning(
                        code="outside_ht_laminar_regime",
                        message="outside_ht: outside-side Reynolds number indicates laminar flow while turbulent behavior is expected by the selected model.",
                        source="outside_ht",
                        severity="warning",
                    )
                )
            elif 2300.0 <= Re_o <= 4000.0:
                warnings_list.append(
                    make_warning(
                        code="outside_ht_transition_regime",
                        message="outside_ht: outside-side Reynolds number is in the transition region; heat-transfer prediction uncertainty is increased.",
                        source="outside_ht",
                        severity="warning",
                    )
                )

        # NTU diagnostic for very high NTU where effectiveness approaches unity.
        C_hot = hot_stream.capacity_rate()
        C_cold = cold_stream.capacity_rate()
        C_min = min(C_hot, C_cold)
        if C_min > 0.0:
            ntu = UA / C_min
            if ntu > 10.0:
                warnings_list.append(
                    make_warning(
                        code="ntu_extreme_high",
                        message="ntu: NTU is very high (>10), so effectiveness is near unity and additional area may provide diminishing returns.",
                        source="ntu",
                        severity="info",
                    )
                )

        # --------------------------------------------------------------
        # Pressure-drop flow-path aggregation (v0.5.6 architecture). The
        # standard solver never accepts or evaluates local-loss path
        # geometry (nozzles, ducts, screens, elbows, external pipework,
        # inlet/outlet/return assemblies): this is always the thin,
        # core-only wrapper around the existing tube-bundle/tube-bank
        # results, so dp_local==0. The new path dp_core/dp_total fields are
        # irreversible loss; dp_static_core/dp_static_total reproduce the
        # existing signed core pressure difference exactly. Local pressure-
        # drop paths are calculated only by explicitly invoking
        # core.pressure_drop.flow_path.calculate_tube_side_pressure_drop_path
        # or calculate_outside_pressure_drop_path.
        # --------------------------------------------------------------
        tube_side_pressure_drop = HXTubeSidePressureDropResults(
            tube_bundle=tube_bundle_hydraulic,
            flow_path=build_tube_side_pressure_drop_result(
                tube_bundle_hydraulic,
                n_tube_passes=self.bundle.n_passes_tube,
            ),
        )
        outside_side_pressure_drop = HXOutsidePressureDropResults(
            tube_bank=outside_bank_hydraulic,
            flow_path=build_outside_pressure_drop_result(
                outside_bank_hydraulic,
            ),
        )
        existing_warning_ids = {(warning.source, warning.code) for warning in warnings_list}
        for warning in (
            tuple(tube_side_pressure_drop.flow_path.warnings)
            + tuple(outside_side_pressure_drop.flow_path.warnings)
        ):
            if (warning.source, warning.code) not in existing_warning_ids:
                warnings_list.append(warning)
                existing_warning_ids.add((warning.source, warning.code))

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
            tube_side_pressure_drop=tube_side_pressure_drop,
            outside_side_pressure_drop=outside_side_pressure_drop,
            warnings=warnings_list if warnings_list else None,
        )



    def simulate(
        self,
        inside: "HXSideInput",
        outside: "HXSideInput",
        *,
        surface_margin: float = 0.0,
        iterate: bool = True,
        flow_arrangement: str | None = None,
        K_inlet: float = 0.5,
        K_outlet: float = 1.0,
        K_turn: float = 1.5,
        euler_provider: str = "zukauskas",
        max_iter: int = 30,
        temperature_tolerance_K: float = 0.05,
        relative_duty_tolerance: float = 1e-4,
        relaxation_factor: float = 0.5,
        relative_alfa_tolerance: float = 1e-3,
        phase_change_onset_tolerance_K: float = 0.0,
        phase_change_activation_band_K: float = 0.5,
        lewis_number: float = 1.0,
        phase_change_max_iterations: int = 50,
        phase_change_temperature_tolerance_K: float = 0.05,
        phase_change_relative_Q_tolerance: float = 1e-4,
        phase_change_water_ratio_tolerance: float = 1e-6,
        phase_change_condensate_tolerance_kg_s: float = 1e-8,
        phase_change_wall_temperature_tolerance_K: float = 0.05,
        phase_change_wet_fraction_tolerance: float = 1e-3,
        phase_change_relaxation_factor: float = 0.5,
    ) -> "HXSimulationResult":
        """Simulate this exchanger, converging the wall/length-corrected
        iterative thermal state by default (v0.5.3).

        This is the intended default entry point for Simulation: given
        geometry, both inlet temperatures, and both flow rates, compute the
        achievable outlet temperatures (and duty). By default (``iterate=
        True``) this calls ``solve_iterative_thermal_state`` -- the same
        function used by ``.rate(...)`` -- so ``inside_alfa_mean``/
        ``outside_alfa_mean``/``U_mean``/``UA`` are wall/length-corrected,
        for any property provider (including ``ConstantPropertyProvider``:
        unlike earlier versions, constant bulk properties no longer skip
        wall-temperature iteration, since the wall correction still needs to
        converge).

        ``surface_margin`` (default ``0.0``, "on the nose") derates the
        full-geometry ``UA`` before computing duty/outlet temperatures; see
        ``core.models.simulation`` for the exact relation.

        ``iterate=False`` is an explicit escape hatch: a single, fast,
        *uncorrected* ``solve()`` pass at the inlet state (``converged=True``,
        ``iterations=1``, ``thermal_state=None`` on the result) -- no wall-
        temperature correction is applied.

        For Rating (closing a known heat balance to get overdesign/margin),
        see ``.rate(...)``.

        See ``core.models.simulation.run_simulation`` for the full algorithm,
        arguments and result fields.

        Phase change (v0.6.1)
        ----------------------
        After the sensible-only dry baseline above, this method calls
        ``core.phase_change.integration.apply_phase_change``, which detects
        H2O phase-change capability (``inside.provider``/``outside.provider``)
        and, when ``phase_change_mode=PhaseChangeMode.AUTO`` (the default,
        set per side on ``HXSideInput``) shows a side's minimum wall
        temperature below its water dew point, solves partial wet-gas H2O
        condensation inside or outside. Only one side can be active.
        See that module and ``core.phase_change.types.PhaseChangeMode`` for
        the full AUTO/DISABLED semantics, and ``docs/property_models.md``
        for the physical model. The ``phase_change_*`` keyword arguments
        here control only the phase-change onset/iteration; they have no
        effect for a call where neither side is phase-change capable.
        """
        from core.models.simulation import run_simulation
        from core.phase_change.integration import PhaseChangeSettings, apply_phase_change
        settings = PhaseChangeSettings(
            onset_tolerance_K=phase_change_onset_tolerance_K,
            activation_band_K=phase_change_activation_band_K,
            lewis_number=lewis_number,
            max_iterations=phase_change_max_iterations,
            temperature_tolerance_K=phase_change_temperature_tolerance_K,
            relative_Q_tolerance=phase_change_relative_Q_tolerance,
            water_ratio_tolerance=phase_change_water_ratio_tolerance,
            condensate_tolerance_kg_s=phase_change_condensate_tolerance_kg_s,
            wall_temperature_tolerance_K=phase_change_wall_temperature_tolerance_K,
            wet_fraction_tolerance=phase_change_wet_fraction_tolerance,
            relaxation_factor=phase_change_relaxation_factor,
        )
        from core.phase_change.steam_integration import (
            apply_water_steam_simulation,
            is_inside_water_steam_case,
            reject_unsupported_outside_pure_steam,
            translate_saturation_crossing_error,
        )
        from core.phase_change.capability import (
            guard_pure_water_single_phase_provider,
            reject_unsupported_pure_water_phase_crossing,
        )

        reject_unsupported_outside_pure_steam(outside)
        if is_inside_water_steam_case(inside):
            return apply_water_steam_simulation(
                self, inside, outside,
                surface_margin=surface_margin,
                iterate=iterate,
                flow_arrangement=flow_arrangement,
                K_inlet=K_inlet,
                K_outlet=K_outlet,
                K_turn=K_turn,
                euler_provider=euler_provider,
                max_iter=max_iter,
                temperature_tolerance_K=temperature_tolerance_K,
                relative_duty_tolerance=relative_duty_tolerance,
                relaxation_factor=relaxation_factor,
                relative_alfa_tolerance=relative_alfa_tolerance,
                settings=settings,
            )

        guarded_inside_provider = guard_pure_water_single_phase_provider(
            inside.provider, T_in=inside.T_in, p=inside.p
        )
        guarded_outside_provider = guard_pure_water_single_phase_provider(
            outside.provider, T_in=outside.T_in, p=outside.p
        )
        dry_inside = (
            inside
            if guarded_inside_provider is inside.provider
            else replace(inside, provider=guarded_inside_provider)
        )
        dry_outside = (
            outside
            if guarded_outside_provider is outside.provider
            else replace(outside, provider=guarded_outside_provider)
        )

        try:
            dry_result = run_simulation(
                self,
                dry_inside,
                dry_outside,
                surface_margin=surface_margin,
                iterate=iterate,
                flow_arrangement=flow_arrangement,
                K_inlet=K_inlet,
                K_outlet=K_outlet,
                K_turn=K_turn,
                euler_provider=euler_provider,
                max_iter=max_iter,
                temperature_tolerance_K=temperature_tolerance_K,
                relative_duty_tolerance=relative_duty_tolerance,
                relaxation_factor=relaxation_factor,
                relative_alfa_tolerance=relative_alfa_tolerance,
            )
        except ValueError as exc:
            translate_saturation_crossing_error(inside, exc)
        reject_unsupported_pure_water_phase_crossing(
            inside.provider,
            T_in=inside.T_in,
            T_out=dry_result.T_out_inside,
            p=inside.p,
        )
        reject_unsupported_pure_water_phase_crossing(
            outside.provider,
            T_in=outside.T_in,
            T_out=dry_result.T_out_outside,
            p=outside.p,
        )
        return apply_phase_change(
            self, inside, outside, dry_result,
            iterate=iterate, euler_provider=euler_provider, settings=settings,
        )

    def rate(
        self,
        inside: "BalanceSideSpec",
        outside: "BalanceSideSpec",
        *,
        Q: float | None = None,
        effectiveness: float | None = None,
        flow_arrangement: str | None = None,
        K_inlet: float = 0.5,
        K_outlet: float = 1.0,
        K_turn: float = 1.5,
        euler_provider: str = "zukauskas",
        include_simulation: bool = False,
        over_specified_tolerance: float = 1e-3,
        max_iterations: int = 25,
        wall_temperature_tolerance_K: float = 0.05,
        relative_alfa_tolerance: float = 1e-3,
        relaxation_factor: float = 0.5,
        phase_change_onset_tolerance_K: float = 0.0,
        phase_change_activation_band_K: float = 0.5,
        lewis_number: float = 1.0,
        phase_change_max_iterations: int = 50,
        phase_change_temperature_tolerance_K: float = 0.05,
        phase_change_relative_Q_tolerance: float = 1e-4,
        phase_change_water_ratio_tolerance: float = 1e-6,
        phase_change_condensate_tolerance_kg_s: float = 1e-8,
        phase_change_wall_temperature_tolerance_K: float = 0.05,
        phase_change_wet_fraction_tolerance: float = 1e-3,
        phase_change_relaxation_factor: float = 0.5,
    ) -> "HXRatingResult":
        """Rate this exchanger against a closed heat balance (overdesign).

        This is the Rating entry point (v0.5.1, thermal state wiring since
        v0.5.3): given geometry and a *closed* heat balance (duty, both
        temperature programs, both flow rates -- with some fields optionally
        left for ``close_heat_balance`` to solve), report how much surface
        margin / overdesign the geometry provides (``overdesign_factor``,
        ``ua_margin``).

        ``inside``/``outside`` are ``BalanceSideSpec`` (fields may be left
        ``None`` for the closure to solve); ``Q`` or ``effectiveness`` may be
        supplied directly instead of implying duty from a fully specified
        side. See ``core.models.heat_balance.close_heat_balance`` for the
        closure algorithm and ``core.models.rating.run_rating`` for the
        overdesign algorithm and the wall/length-corrected ``U``/``UA_actual``
        source.

        ``max_iterations``/``wall_temperature_tolerance_K``/
        ``relative_alfa_tolerance``/``relaxation_factor`` control the
        underlying wall-temperature iteration (same meaning as
        ``.solve_thermal_state(...)``).

        For Simulation (computing achievable outlet temperatures from known
        inlets), see ``.simulate(...)``.

        Phase change (v0.6.1)
        ----------------------
        See
        ``core.phase_change.rating_integration.apply_phase_change_to_rating``
        for inside/outside wet-gas H2O condensation Rating and its scope.
        The active wet side requires a specified mass flow and temperature
        program; duty must come from explicit ``Q`` or the fully specified
        non-condensing side. ``phase_change_mode`` is per ``BalanceSideSpec``.
        """
        from core.phase_change.rating_integration import apply_phase_change_to_rating
        from core.phase_change.integration import PhaseChangeSettings

        settings = PhaseChangeSettings(
            onset_tolerance_K=phase_change_onset_tolerance_K,
            activation_band_K=phase_change_activation_band_K,
            lewis_number=lewis_number,
            max_iterations=phase_change_max_iterations,
            temperature_tolerance_K=phase_change_temperature_tolerance_K,
            relative_Q_tolerance=phase_change_relative_Q_tolerance,
            water_ratio_tolerance=phase_change_water_ratio_tolerance,
            condensate_tolerance_kg_s=phase_change_condensate_tolerance_kg_s,
            wall_temperature_tolerance_K=phase_change_wall_temperature_tolerance_K,
            wet_fraction_tolerance=phase_change_wet_fraction_tolerance,
            relaxation_factor=phase_change_relaxation_factor,
        )
        from core.phase_change.steam_integration import (
            apply_water_steam_rating,
            is_inside_water_steam_case,
            reject_unsupported_outside_pure_steam,
            translate_saturation_crossing_error,
        )

        reject_unsupported_outside_pure_steam(outside)
        if is_inside_water_steam_case(inside):
            return apply_water_steam_rating(
                self, inside, outside,
                Q=Q, effectiveness=effectiveness,
                flow_arrangement=flow_arrangement,
                K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
                euler_provider=euler_provider,
                include_simulation=include_simulation,
                over_specified_tolerance=over_specified_tolerance,
                max_iterations=max_iterations,
                wall_temperature_tolerance_K=wall_temperature_tolerance_K,
                relative_alfa_tolerance=relative_alfa_tolerance,
                relaxation_factor=relaxation_factor,
                settings=settings,
            )
        try:
            return apply_phase_change_to_rating(
                self, inside, outside,
                Q=Q, effectiveness=effectiveness,
                flow_arrangement=flow_arrangement,
                K_inlet=K_inlet, K_outlet=K_outlet, K_turn=K_turn,
                euler_provider=euler_provider,
                include_simulation=include_simulation,
                over_specified_tolerance=over_specified_tolerance,
                max_iterations=max_iterations,
                wall_temperature_tolerance_K=wall_temperature_tolerance_K,
                relative_alfa_tolerance=relative_alfa_tolerance,
                relaxation_factor=relaxation_factor,
                settings=settings,
            )
        except ValueError as exc:
            translate_saturation_crossing_error(inside, exc)

    def solve_thermal_state(
        self,
        *,
        m_dot_inside: float,
        m_dot_outside: float,
        inside_provider: "PropertyProvider",
        outside_provider: "PropertyProvider",
        T_in_inside: float,
        T_in_outside: float,
        p_inside: float,
        p_outside: float,
        flow_arrangement: str | None = None,
        euler_provider: str = "zukauskas",
        max_iterations: int = 25,
        wall_temperature_tolerance_K: float = 0.05,
        relative_alfa_tolerance: float = 1e-3,
        relaxation_factor: float = 0.5,
    ) -> "IterativeThermalState":
        """Iterative mean-property and wall-temperature thermal state (v0.5.2).

        Refines alfa_i/alfa_o with separate inside/outside wall-property
        corrections and both tube-wall surface temperatures, on top of the
        mean-bulk-property iteration. This is a physics-kernel capability
        alongside ``.solve()``/``.simulate()``/``.rate()`` -- it is not a new
        application calculation mode, and present/future modes may reuse it.

        See ``core.heat_transfer.thermal_iteration.solve_iterative_thermal_state``
        for the full algorithm, convergence controls and result fields.
        """
        from core.heat_transfer.thermal_iteration import solve_iterative_thermal_state

        return solve_iterative_thermal_state(
            self,
            m_dot_inside=m_dot_inside,
            m_dot_outside=m_dot_outside,
            inside_provider=inside_provider,
            outside_provider=outside_provider,
            T_in_inside=T_in_inside,
            T_in_outside=T_in_outside,
            p_inside=p_inside,
            p_outside=p_outside,
            flow_arrangement=flow_arrangement,
            euler_provider=euler_provider,
            max_iterations=max_iterations,
            wall_temperature_tolerance_K=wall_temperature_tolerance_K,
            relative_alfa_tolerance=relative_alfa_tolerance,
            relaxation_factor=relaxation_factor,
        )
