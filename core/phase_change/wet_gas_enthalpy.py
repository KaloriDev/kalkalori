# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""Wet-gas specific enthalpy on a per-kg-dry-gas basis.

``h_wet_gas_dry_basis(T, p, W, capability)`` returns the specific enthalpy
of a wet-gas stream, referenced to 1 kg of the *dry* (non-condensable)
carrier gas -- i.e. the enthalpy of (1 kg dry gas + W kg water vapor).

Reference-state consistency (why this is a sum of two independently-
evaluated terms, not one CoolProp mixture flash)
--------------------------------------------------------------------------
The overall outside-stream energy balance (``core.phase_change.
outside_condensation_solver``) is

    Q = H_gas_in - H_gas_out - H_drained_condensate

which subtracts enthalpies drawn from *two different property backends*:
the dry-gas part from CoolProp (``core.properties.gas_mixture``), and both
the vapor-in-gas part and the drained-liquid-condensate part from
IAPWS-IF97 (``core.properties.water``). CoolProp and IAPWS-IF97 do not
share a common absolute enthalpy reference state, so directly differencing
a whole-mixture CoolProp flash enthalpy against an IAPWS-IF97 condensate
enthalpy would introduce a hidden, backend-dependent offset into the
balance.

This module avoids that by *never* asking CoolProp to evaluate the
enthalpy of the water component: it evaluates

    h_dry_basis(T, p, W) = h_dry_gas(T, p) + W * h_h2o_vapor(T)

where ``h_dry_gas`` comes from a CoolProp gas-mixture evaluated on the
*dry* composition only (no water component at all -- see
``core.phase_change.wet_gas_composition.dry_gas_spec``), and
``h_h2o_vapor(T)`` comes from
``core.properties.water.water_saturation_vapor_enthalpy`` (IAPWS-IF97).
Every water-referenced quantity in the balance -- vapor remaining in the
gas phase, and the drained liquid condensate
(``core.properties.water.water_saturation_liquid_enthalpy``) -- therefore
comes from the *same* IAPWS-IF97 reference state, and every dry-gas-
referenced quantity comes from the same CoolProp reference state. Only
dry-gas mass is conserved end to end (``m_dot_dry_carrier`` constant), so
differencing ``H_dry_in - H_dry_out`` never mixes backends either.

Ideal-gas assumptions (v0.6.0)
-------------------------------
- The dry-gas enthalpy is evaluated as a function of (T, p) via CoolProp
  but is not itself corrected for the presence of water vapor (Gibbs-
  Dalton ideal-gas-mixture rule: each component's partial enthalpy equals
  its pure-component enthalpy at the mixture temperature).
- Water-vapor-in-gas enthalpy uses the saturated-vapor value at the gas
  temperature T as an engineering approximation for superheated vapor at
  low partial pressure (standard psychrometric convention, e.g. ASHRAE
  Fundamentals; see ``core.properties.water.water_saturation_vapor_
  enthalpy`` docstring). This is accurate to a fraction of a percent for
  the low-water-content, near-atmospheric-pressure gas streams this model
  targets, and keeps the water-vapor term on the same IAPWS-IF97 basis as
  the condensate term (see above).

Ref:
- Gibbs-Dalton ideal-gas mixture rule (any thermodynamics textbook).
- ASHRAE Fundamentals, moist-air enthalpy.
- IAPWS-IF97 (via ``core.properties.water``).
"""

from __future__ import annotations

import math

from core.properties.gas_mixture import GasMixturePropertyProvider
from core.properties.water import water_saturation_vapor_enthalpy
from core.phase_change.types import PhaseChangeCapability
from core.phase_change.wet_gas_composition import dry_gas_spec

# Default search bracket for temperature_from_h_wet_gas_dry_basis. Wide
# enough for typical process-gas exchanger duties; callers with unusual
# temperature ranges may override it. Bounded by the water saturation-curve
# validity range (see core.properties.water): below the triple point,
# h_h2o_vapor is undefined (v0.6.0 does not model frost/ice); above the
# critical temperature, there is no saturated-vapor state to approximate
# low-partial-pressure superheated vapor with.
DEFAULT_TEMPERATURE_BRACKET_K = (274.0, 640.0)


def dry_gas_enthalpy(T: float, p: float, capability: PhaseChangeCapability) -> float:
    """Return the dry-gas-only specific enthalpy [J/kg dry gas] at (T, p).

    Evaluated on the capability's dry composition (no water component),
    via CoolProp. See the module docstring for why this is kept separate
    from the water-vapor enthalpy term.
    """
    provider = GasMixturePropertyProvider(dry_gas_spec(capability))
    return provider.full_at(T=T, p=p).h


def h_wet_gas_dry_basis(
    T: float,
    p: float,
    W: float,
    capability: PhaseChangeCapability,
) -> float:
    """Return wet-gas specific enthalpy [J/(kg dry gas)] at (T, p, W).

    ``h = h_dry_gas(T, p) + W * h_h2o_vapor(T)``. See the module docstring
    for the reference-state rationale.
    """
    if not math.isfinite(W) or W < 0.0:
        raise ValueError(f"W must be a non-negative finite value [kg/kg], got {W!r}.")
    h_dry = dry_gas_enthalpy(T, p, capability)
    h_vapor = water_saturation_vapor_enthalpy(T=T)
    return h_dry + W * h_vapor


def temperature_from_h_wet_gas_dry_basis(
    h_target: float,
    p: float,
    W: float,
    capability: PhaseChangeCapability,
    *,
    T_bracket: tuple[float, float] = DEFAULT_TEMPERATURE_BRACKET_K,
    tolerance_K: float = 1e-4,
    max_iterations: int = 200,
) -> float:
    """Invert ``h_wet_gas_dry_basis`` for temperature, by bisection.

    An interval (bisection) method is used deliberately, not an unbounded
    Newton iteration: ``h_wet_gas_dry_basis`` is monotonic increasing in T
    over any physically sensible bracket (positive dry-gas cp, positive
    d(h_g)/dT on the water saturation curve), so bisection is robust and
    bounded, and cannot diverge outside the declared temperature range.

    Raises:
        ValueError: if ``h_target`` is not bracketed by
            ``h_wet_gas_dry_basis`` evaluated at the bracket endpoints.
    """
    T_lo, T_hi = T_bracket
    h_lo = h_wet_gas_dry_basis(T_lo, p, W, capability)
    h_hi = h_wet_gas_dry_basis(T_hi, p, W, capability)
    if not (h_lo <= h_target <= h_hi):
        raise ValueError(
            f"temperature_from_h_wet_gas_dry_basis: h_target={h_target:.6g} J/kg "
            f"is not bracketed by [{h_lo:.6g}, {h_hi:.6g}] J/kg over T in "
            f"[{T_lo:.6g}, {T_hi:.6g}] K. Widen T_bracket."
        )

    for _ in range(max_iterations):
        T_mid = 0.5 * (T_lo + T_hi)
        h_mid = h_wet_gas_dry_basis(T_mid, p, W, capability)
        if h_mid < h_target:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if (T_hi - T_lo) < tolerance_K:
            break
    return 0.5 * (T_lo + T_hi)
