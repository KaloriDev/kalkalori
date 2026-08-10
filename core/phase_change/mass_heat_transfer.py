# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Heat/mass-transfer analogy for H2O condensation from a non-condensable
carrier gas on either exchanger side (v0.6.1).

Uses the Chilton-Colburn heat/mass-transfer analogy with a configurable
Lewis number (default ``lewis_number=1.0``, the classic simplifying
assumption that reduces the analogy to the "enthalpy potential"/Lewis
relation widely used for air-water systems -- but it is a *configurable
first-model assumption*, not asserted here as a universal constant; see
``docs/references.md``).

Chilton-Colburn analogy: ``St * Pr^(2/3) = St_m * Sc^(2/3)`` with
``St = alfa/(rho*u*cp)``, ``St_m = h_m/u``, giving

    h_m = (alfa / (rho*cp)) * Le^(-2/3),   Le = Sc/Pr = alpha_thermal/D_mass

With the driving force expressed directly in the W = kg H2O / kg dry
carrier basis used throughout this package (rather than a molar/volume
concentration), the corresponding mass-transfer coefficient on a
dry-carrier-gas mass-flux basis is:

    mass_transfer_coefficient = (alfa_dry / cp_gas) * Le^(-2/3)   [kg dry gas / (m^2*s)]
    m_dot_condensation = mass_transfer_coefficient * (W_bulk - W_sat_surface) * A_wet   [kg H2O/s]

only when ``W_bulk > W_sat_surface`` (surface below the local dew point);
otherwise no condensation occurs (v0.6.0 does not model re-evaporation of
the gas-phase vapor -- see ``docs/roadmap.md``, v0.6.2 for droplet/mist
evaporation).

Ref:
- Chilton, T.H. & Colburn, A.P. (1934), "Mass Transfer (Absorption)
  Coefficients", Ind. Eng. Chem.
- Incropera et al., Fundamentals of Heat and Mass Transfer, heat-mass
  analogy chapter.
- Bosnjakovic / Merkel enthalpy-potential method (cooling-tower and
  wet-cooling-coil literature) for the Le=1 W-basis simplification.
"""

from __future__ import annotations

import math


def mass_transfer_coefficient(
    alfa_dry: float,
    cp_gas: float,
    *,
    lewis_number: float = 1.0,
) -> float:
    """Return the Chilton-Colburn mass-transfer coefficient [kg dry gas/(m2*s)].

    Args:
        alfa_dry: Dry (sensible-only) heat-transfer coefficient for the
            condensing side [W/(m2*K)], from its existing correlation.
        cp_gas: Specific heat of the bulk gas mixture [J/(kg*K)].
        lewis_number: Le = Sc/Pr [-]. Must be positive. Default 1.0 (see
            module docstring).
    """
    if not math.isfinite(alfa_dry) or alfa_dry <= 0.0:
        raise ValueError("alfa_dry must be a positive finite value [W/(m2*K)].")
    if not math.isfinite(cp_gas) or cp_gas <= 0.0:
        raise ValueError("cp_gas must be a positive finite value [J/(kg*K)].")
    if not math.isfinite(lewis_number) or lewis_number <= 0.0:
        raise ValueError("lewis_number must be a positive finite value.")

    return (alfa_dry / cp_gas) * lewis_number ** (-2.0 / 3.0)


def condensation_mass_flux(
    *,
    alfa_dry: float,
    cp_gas: float,
    W_bulk: float,
    W_sat_surface: float,
    lewis_number: float = 1.0,
) -> float:
    """Return the local condensation mass flux [kg H2O/(m2*s)], >= 0.

    Zero when the surface is not below the local saturation content
    (``W_bulk <= W_sat_surface``): v0.6.0 does not model re-evaporation, so
    a negative driving force is clamped to zero here (this is a physical
    floor on a one-directional process, not a sign-error mask -- see
    ``condensation_rate`` for the corresponding total-rate guard against
    over-condensing beyond the water actually present in the stream).
    """
    driving_force = W_bulk - W_sat_surface
    if driving_force <= 0.0:
        return 0.0
    h_m = mass_transfer_coefficient(alfa_dry, cp_gas, lewis_number=lewis_number)
    return h_m * driving_force


def condensation_rate(
    *,
    alfa_dry: float,
    cp_gas: float,
    W_bulk: float,
    W_sat_surface: float,
    A_wet: float,
    m_dot_water_vapor_available: float,
    lewis_number: float = 1.0,
) -> float:
    """Return the condensation rate [kg H2O/s], bounded to the water
    actually available in the stream.

    ``A_wet`` should be positive and is the active mass-transfer area,
    ``A_side * wet_surface_fraction`` for the partial-condensation model.
    It does not scale the full-area sensible duty.
    """
    if not math.isfinite(A_wet) or A_wet <= 0.0:
        raise ValueError("A_wet must be a positive finite value [m2].")
    if not math.isfinite(m_dot_water_vapor_available) or m_dot_water_vapor_available < 0.0:
        raise ValueError("m_dot_water_vapor_available must be a non-negative finite value [kg/s].")

    flux = condensation_mass_flux(
        alfa_dry=alfa_dry,
        cp_gas=cp_gas,
        W_bulk=W_bulk,
        W_sat_surface=W_sat_surface,
        lewis_number=lewis_number,
    )
    rate = flux * A_wet
    # Numerical guard against a single-iteration overshoot condensing more
    # water than is present in the stream; this bounds the per-iteration
    # step, it does not mask a sign error (the flux above is already
    # clamped to >= 0 by condensation_mass_flux).
    return min(rate, m_dot_water_vapor_available)
