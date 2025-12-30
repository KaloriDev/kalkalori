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
# plain Python floats in consistent SI base units.
#
# Unit handling (e.g. pint.Quantity) must be applied ONLY at API boundaries
# and is intentionally excluded from this module.

"""
Energy stream abstractions for heat exchanger calculations.

This module defines generic representations of energy-carrying streams
used in NTU/LMTD-based heat exchanger models.

The purpose of these abstractions is to:
- support single-phase heat transfer,
- support phase-change processes (e.g. steam condensation),
- allow future extension to moist air and enthalpy-based models,
without modifying the core NTU formulation.

References
----------
- Incropera et al., Fundamentals of Heat and Mass Transfer
- Kays & London, Compact Heat Exchangers
- Shah & Sekulić, Fundamentals of Heat Exchanger Design
"""


class EnergyStream:
    """
    Abstract representation of an energy-carrying stream
    participating in heat exchange.
    """

    def capacity_rate(self) -> float:
        """
        Effective heat capacity rate [W/K].

        For phase-change processes this may be infinite or
        defined in an effective sense.
        """
        raise NotImplementedError

    def inlet_temperature(self) -> float:
        """
        Inlet temperature of the stream [K].
        """
        raise NotImplementedError

    def outlet_temperature(self, Q: float) -> float:
        """
        Outlet temperature after heat transfer Q [W].

        Parameters
        ----------
        Q : float
            Heat transfer rate [W], positive when heat is removed
            from this stream.
        """
        raise NotImplementedError


class SensibleHeatStream(EnergyStream):
    """
    Single-phase stream with sensible heat transfer only.

    Examples
    --------
    - liquid water
    - dry air
    - oil
    """

    def __init__(self, C: float, T_in: float):
        """
        Parameters
        ----------
        C : float
            Heat capacity rate [W/K].
        T_in : float
            Inlet temperature [K].
        """
        if C <= 0.0:
            raise ValueError("Heat capacity rate must be positive.")

        self._C = C
        self._T_in = T_in

    def capacity_rate(self) -> float:
        return self._C

    def inlet_temperature(self) -> float:
        return self._T_in

    def outlet_temperature(self, Q: float) -> float:
        return self._T_in - Q / self._C


class CondensingSteamStream(EnergyStream):
    """
    Condensing steam stream at saturation temperature.

    Assumptions
    -----------
    - condensation occurs at constant saturation temperature,
    - pressure drop effects on saturation temperature are neglected,
    - subcooling of condensate is neglected.

    This model is suitable for:
    - steam heaters,
    - reboilers,
    - condensers with isothermal hot side.
    """

    def __init__(self, T_sat: float):
        """
        Parameters
        ----------
        T_sat : float
            Saturation temperature of steam [K].
        """
        self._T_sat = T_sat

    def capacity_rate(self) -> float:
        # Infinite capacity rate represents isothermal behavior
        return float("inf")

    def inlet_temperature(self) -> float:
        return self._T_sat

    def outlet_temperature(self, Q: float) -> float:
        # Temperature remains constant during condensation
        return self._T_sat

class MoistAirStream(EnergyStream):
    """
    Moist air energy stream with possible condensation.

    This stream is enthalpy-based and intended for use in
    segmented or iterative heat exchanger models.

    Notes
    -----
    - Heat transfer includes sensible and latent components.
    - Effective heat capacity rate is not constant and must be
      provided or updated externally.
    - Psychrometric relations are intentionally NOT implemented here.
    """

    def __init__(
        self,
        m_dot: float,
        h_in: float,
        T_in: float,
        C_eff: float | None = None,
    ):
        """
        Parameters
        ----------
        m_dot : float
            Dry air mass flow rate [kg/s].
        h_in : float
            Inlet specific enthalpy of moist air [J/kg_dry_air].
        T_in : float
            Inlet dry-bulb temperature [K].
        C_eff : float, optional
            Effective heat capacity rate [W/K] used for NTU coupling.
            If None, the stream must be updated iteratively.
        """

        if m_dot <= 0.0:
            raise ValueError("Dry air mass flow rate must be positive.")

        self.m_dot = m_dot
        self.h_in = h_in
        self.T_in = T_in
        self.C_eff = C_eff

    def capacity_rate(self) -> float:
        """
        Effective heat capacity rate [W/K].

        For moist air with condensation, this value is an approximation
        and should be updated by the calling heat exchanger model.
        """
        if self.C_eff is None:
            raise RuntimeError(
                "Effective heat capacity rate not set for MoistAirStream."
            )
        return self.C_eff

    def inlet_temperature(self) -> float:
        return self.T_in

    def outlet_temperature(self, Q: float) -> float:
        """
        Outlet temperature after heat transfer.

        This method only updates enthalpy. Temperature must be
        determined by psychrometric relations at a higher level.
        """
        h_out = self.h_in - Q / self.m_dot

        # Temperature cannot be determined without psychrometrics
        # Returning inlet temperature as placeholder
        return self.T_in

    def outlet_enthalpy(self, Q: float) -> float:
        """
        Outlet specific enthalpy [J/kg_dry_air].
        """
        return self.h_in - Q / self.m_dot
