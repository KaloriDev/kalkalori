"""
Moist-air transport property helpers.

This module converts a psychrometric `MoistAirState` into common transport
properties required by existing heat-transfer correlations.

Returned `FluidTransportProperties` use the following basis:
- rho: kg_moist_air/m3
- mu: Pa*s
- k: W/(m*K)
- cp: J/(kg_moist_air*K)

Important:
    Psychrometric enthalpy in KalKalori is stored per kg dry air, but the
    transport-property cp returned here is per kg moist air. This is intentional
    because existing heat-transfer correlations use cp in the conventional
    fluid-property sense, e.g. for Pr = mu * cp / k.

The current implementation is an MVP engineering approximation:
- cp is calculated from dry-air and water-vapor mass fractions,
- dry-air and water-vapor viscosity use Sutherland-type approximations,
- mixture viscosity uses a Wilke-type gas-mixture rule,
- thermal conductivity uses simple component approximations and the same
  Wilke-type structure as a practical MVP.

Refs:
- ASHRAE Fundamentals, moist air and psychrometrics.
- Incropera et al., Fundamentals of Heat and Mass Transfer, gas properties.
- Wilke, C. R. (1950), A viscosity equation for gas mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from core.common.warnings import ModelWarning
from core.properties.common import FluidTransportProperties
from core.psychrometrics.moist_air import (
    MoistAirState,
    moist_air_state_from_t_rh,
    moist_air_state_from_t_w,
)


SOURCE = "properties.moist_air_transport"

# Molar masses [kg/mol]
M_DRY_AIR = 28.96546e-3
M_WATER_VAPOR = 18.01528e-3

# Constant-pressure specific heat capacities [J/(kg*K)].
# MVP constants around ordinary HVAC / low-temperature heat-recovery range.
CP_DRY_AIR = 1006.0
CP_WATER_VAPOR = 1860.0

# Recommended MVP applicability range for simple transport approximations.
T_MIN_RECOMMENDED = 250.0
T_MAX_RECOMMENDED = 450.0
W_HIGH_RECOMMENDED = 0.10


@dataclass(frozen=True)
class MoistAirTransportResult:
    """Moist-air transport properties with applicability warnings.

    Attributes:
        props:
            Common transport properties. The cp value is per kg moist air.
        warnings:
            Applicability and consistency warnings.
    """

    props: FluidTransportProperties
    warnings: list[ModelWarning] = field(default_factory=list)


@dataclass(frozen=True)
class MoistAirTransportProvider:
    """Property provider for moist air with fixed humidity ratio.

    This provider fits the existing `PropertyProvider.at(T, p)` shape by
    treating humidity ratio W as a fixed stream composition.

    This is usually the preferred provider form for dry-side sensible cooling
    where water content is conserved over a temperature change.

    Attributes:
        W: Humidity ratio [kg_water/kg_dry_air].
    """

    W: float

    @classmethod
    def from_t_rh(
        cls,
        T: float,
        RH: float,
        p: float,
    ) -> "MoistAirTransportProvider":
        """Create provider from a reference T/RH/p moist-air state.

        Args:
            T: Reference dry-bulb temperature [K].
            RH: Reference relative humidity [-].
            p: Reference pressure [Pa].

        Returns:
            MoistAirTransportProvider with fixed W.
        """
        air = moist_air_state_from_t_rh(T=T, RH=RH, p=p)
        return cls(W=air.W)

    def __post_init__(self) -> None:
        _validate_humidity_ratio(self.W)

    def at(self, T: float, p: float) -> FluidTransportProperties:
        """Return moist-air transport properties at T [K] and p [Pa].

        The humidity ratio W is fixed by the provider instance.
        """
        air = moist_air_state_from_t_w(T=T, W=self.W, p=p)
        return moist_air_transport_props_from_state(air)


def moist_air_transport_props_from_state(
    air: MoistAirState,
) -> FluidTransportProperties:
    """Return moist-air transport properties from a psychrometric state.

    Args:
        air: Moist-air psychrometric state.

    Returns:
        FluidTransportProperties with cp per kg moist air.
    """
    return moist_air_transport_result_from_state(air).props


def moist_air_transport_result_from_state(
    air: MoistAirState,
) -> MoistAirTransportResult:
    """Return moist-air transport properties and applicability warnings.

    Args:
        air: Moist-air psychrometric state.

    Returns:
        MoistAirTransportResult.

    Notes:
        This helper is intended to provide `rho`, `mu`, `k`, and `cp` for
        existing outside-flow heat-transfer and pressure-drop correlations.
    """
    _validate_moist_air_state(air)

    warnings: list[ModelWarning] = []

    if air.T < T_MIN_RECOMMENDED or air.T > T_MAX_RECOMMENDED:
        warnings.append(
            _warning(
                code="MOIST_AIR_TRANSPORT_T_RANGE",
                message=(
                    "Moist-air transport approximations are intended mainly "
                    f"for {T_MIN_RECOMMENDED:g} K <= T <= {T_MAX_RECOMMENDED:g} K."
                ),
            )
        )

    if air.W > W_HIGH_RECOMMENDED:
        warnings.append(
            _warning(
                code="MOIST_AIR_HIGH_HUMIDITY_RATIO",
                message=(
                    "Humidity ratio is high for the current moist-air transport "
                    "MVP approximation. Results should be treated with caution."
                ),
            )
        )

    if air.RH > 1.0:
        warnings.append(
            _warning(
                code="MOIST_AIR_SUPERSATURATED_STATE",
                message=(
                    "Relative humidity is above 1.0. Supersaturated / fog conditions "
                    "are outside the current moist-air transport MVP assumptions."
                ),
            )
        )

    mass_fraction_da, mass_fraction_v = _mass_fractions_from_humidity_ratio(air.W)
    mole_fraction_da, mole_fraction_v = _mole_fractions_from_humidity_ratio(air.W)

    cp = _moist_air_cp_mass_basis(
        mass_fraction_da=mass_fraction_da,
        mass_fraction_v=mass_fraction_v,
    )

    mu_da = _dry_air_viscosity_sutherland(air.T)
    mu_v = _water_vapor_viscosity_sutherland(air.T)

    k_da = _dry_air_thermal_conductivity(air.T)
    k_v = _water_vapor_thermal_conductivity(air.T)

    mu = _wilke_binary_mixture_property(
        x_1=mole_fraction_da,
        x_2=mole_fraction_v,
        prop_1=mu_da,
        prop_2=mu_v,
        M_1=M_DRY_AIR,
        M_2=M_WATER_VAPOR,
    )

    k = _wilke_binary_mixture_property(
        x_1=mole_fraction_da,
        x_2=mole_fraction_v,
        prop_1=k_da,
        prop_2=k_v,
        M_1=M_DRY_AIR,
        M_2=M_WATER_VAPOR,
    )

    props = FluidTransportProperties(
        rho=air.rho,
        mu=mu,
        k=k,
        cp=cp,
    )

    return MoistAirTransportResult(
        props=props,
        warnings=warnings,
    )


def moist_air_transport_result_from_t_rh(
    T: float,
    RH: float,
    p: float,
) -> MoistAirTransportResult:
    """Return moist-air transport properties from T/RH/p.

    Args:
        T: Dry-bulb temperature [K].
        RH: Relative humidity [-].
        p: Pressure [Pa].

    Returns:
        MoistAirTransportResult.
    """
    air = moist_air_state_from_t_rh(T=T, RH=RH, p=p)
    return moist_air_transport_result_from_state(air)


def moist_air_transport_props_from_t_rh(
    T: float,
    RH: float,
    p: float,
) -> FluidTransportProperties:
    """Return moist-air transport properties from T/RH/p."""
    return moist_air_transport_result_from_t_rh(T=T, RH=RH, p=p).props


def _mass_fractions_from_humidity_ratio(W: float) -> tuple[float, float]:
    """Return dry-air and water-vapor mass fractions from W."""
    _validate_humidity_ratio(W)

    denominator = 1.0 + W
    return 1.0 / denominator, W / denominator


def _mole_fractions_from_humidity_ratio(W: float) -> tuple[float, float]:
    """Return dry-air and water-vapor mole fractions from W."""
    _validate_humidity_ratio(W)

    n_ratio_v_to_da = W * M_DRY_AIR / M_WATER_VAPOR
    denominator = 1.0 + n_ratio_v_to_da

    x_da = 1.0 / denominator
    x_v = n_ratio_v_to_da / denominator

    return x_da, x_v


def _moist_air_cp_mass_basis(
    mass_fraction_da: float,
    mass_fraction_v: float,
) -> float:
    """Return moist-air cp per kg moist air."""
    return mass_fraction_da * CP_DRY_AIR + mass_fraction_v * CP_WATER_VAPOR


def _dry_air_viscosity_sutherland(T: float) -> float:
    """Return dry-air viscosity using Sutherland approximation.

    Args:
        T: Temperature [K].

    Returns:
        Dynamic viscosity [Pa*s].

    Ref: Incropera et al., dry air property approximation.
    """
    T_ref = 273.15
    mu_ref = 1.716e-5
    S = 111.0

    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


def _water_vapor_viscosity_sutherland(T: float) -> float:
    """Return water-vapor viscosity using a Sutherland-type approximation.

    Args:
        T: Temperature [K].

    Returns:
        Dynamic viscosity [Pa*s].
    """
    T_ref = 273.15
    mu_ref = 9.20e-6
    S = 1064.0

    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


def _dry_air_thermal_conductivity(T: float) -> float:
    """Return dry-air thermal conductivity approximation.

    Args:
        T: Temperature [K].

    Returns:
        Thermal conductivity [W/(m*K)].
    """
    T_ref = 273.15
    k_ref = 0.0241
    S = 194.0

    return k_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


def _water_vapor_thermal_conductivity(T: float) -> float:
    """Return water-vapor thermal conductivity approximation.

    Args:
        T: Temperature [K].

    Returns:
        Thermal conductivity [W/(m*K)].
    """
    T_C = T - 273.15

    # Simple engineering approximation around low-temperature steam / water vapor.
    return 0.0167 + 5.7e-5 * T_C


def _wilke_binary_mixture_property(
    x_1: float,
    x_2: float,
    prop_1: float,
    prop_2: float,
    M_1: float,
    M_2: float,
) -> float:
    """Return binary gas-mixture property using Wilke-type mixing structure.

    Args:
        x_1: Mole fraction of component 1 [-].
        x_2: Mole fraction of component 2 [-].
        prop_1: Pure-component transport property of component 1.
        prop_2: Pure-component transport property of component 2.
        M_1: Molar mass of component 1 [kg/mol].
        M_2: Molar mass of component 2 [kg/mol].

    Returns:
        Mixture transport property in the same units as prop_1 and prop_2.

    Ref: Wilke (1950), gas-mixture viscosity rule.
    """
    if x_2 == 0.0:
        return prop_1
    if x_1 == 0.0:
        return prop_2

    phi_12 = _wilke_phi(prop_i=prop_1, prop_j=prop_2, M_i=M_1, M_j=M_2)
    phi_21 = _wilke_phi(prop_i=prop_2, prop_j=prop_1, M_i=M_2, M_j=M_1)

    term_1 = x_1 * prop_1 / (x_1 + x_2 * phi_12)
    term_2 = x_2 * prop_2 / (x_2 + x_1 * phi_21)

    return term_1 + term_2


def _wilke_phi(
    prop_i: float,
    prop_j: float,
    M_i: float,
    M_j: float,
) -> float:
    """Return Wilke interaction parameter."""
    numerator = (
        1.0
        + math.sqrt(prop_i / prop_j) * (M_j / M_i) ** 0.25
    ) ** 2

    denominator = math.sqrt(8.0) * math.sqrt(1.0 + M_i / M_j)

    return numerator / denominator


def _validate_moist_air_state(air: MoistAirState) -> None:
    _validate_temperature(air.T)
    _validate_pressure(air.p)
    _validate_humidity_ratio(air.W)

    if not math.isfinite(air.RH):
        raise ValueError("Relative humidity must be finite [-].")
    if air.RH < 0.0:
        raise ValueError("Relative humidity must be non-negative [-].")

    if not math.isfinite(air.rho):
        raise ValueError("Moist-air density must be finite [kg/m3].")
    if air.rho <= 0.0:
        raise ValueError("Moist-air density must be positive [kg/m3].")


def _validate_temperature(T: float) -> None:
    if not math.isfinite(T):
        raise ValueError("Temperature must be finite [K].")
    if T <= 0.0:
        raise ValueError("Temperature must be above absolute zero [K].")


def _validate_pressure(p: float) -> None:
    if not math.isfinite(p):
        raise ValueError("Pressure must be finite [Pa].")
    if p <= 0.0:
        raise ValueError("Pressure must be positive [Pa].")


def _validate_humidity_ratio(W: float) -> None:
    if not math.isfinite(W):
        raise ValueError("Humidity ratio must be finite [kg_water/kg_dry_air].")
    if W < 0.0:
        raise ValueError("Humidity ratio must be non-negative [kg_water/kg_dry_air].")


def _warning(code: str, message: str, severity: str = "warning") -> ModelWarning:
    return ModelWarning(
        code=code,
        message=message,
        severity=severity,
        source=SOURCE,
    )