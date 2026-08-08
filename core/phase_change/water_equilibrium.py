# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""H2O vapor-gas equilibrium helpers for a general (non-air) carrier gas.

This module composes ``core.properties.water`` (IAPWS-IF97 saturation
curve) with ideal-gas mixture partial-pressure relations. It does not
duplicate any IAPWS-IF97 access: every saturation-pressure/-temperature
lookup goes through ``core.properties.water``, so there is a single place
in KalKalori that talks to the ``iapws`` package for water saturation
states (``core.properties.water.water_saturation_pressure`` /
``water_saturation_temperature``).

Assumptions (v0.6.0, see also the module docstring of
``core.phase_change.wet_gas_enthalpy``):
- Ideal-gas partial pressures (Dalton's law): ``p_H2O = y_H2O * p_total``.
- Non-condensable ("dry") components remain entirely in the gas phase.
- Only H2O condenses; the dry-gas composition (relative mole fractions) is
  unchanged by condensation.
- ``core.psychrometrics`` (PsychroLib-based) is dry-air-specific and is not
  reused here; the classic psychrometric humidity-ratio formula
  (``W_sat = 0.622 * p_sat/(p - p_sat)`` for dry air, ``M_air`` = 28.97
  g/mol) is the special case of ``saturated_water_ratio`` below at
  ``M_dry = M_air`` -- see the dry-air regression test in
  ``core/tests/phase_change_water_equilibrium_test.py``.

Ref:
- IAPWS-IF97 (via ``core.properties.water``).
- Ideal-gas partial-pressure / Dalton's law (any thermodynamics textbook).
- ASHRAE Fundamentals, moist-air / dew-point relations (generalized here to
  an arbitrary dry-gas molar mass instead of dry air's).
"""

from __future__ import annotations

import math

from core.properties.water import (
    WATER_CRITICAL_TEMPERATURE_K,
    WATER_TRIPLE_POINT_TEMPERATURE_K,
    water_saturation_pressure,
    water_saturation_temperature,
)

WATER_MOLAR_MASS_KG_PER_MOL = 18.01528e-3


def water_partial_pressure(y_h2o: float, p_total: float) -> float:
    """Return H2O partial pressure [Pa] = y_H2O * p_total (Dalton's law).

    Raises:
        ValueError: if ``y_h2o`` is not in [0, 1), ``p_total`` is not
            positive/finite, or the implied partial pressure would be
            >= the total pressure (physically impossible for a mixture
            that still contains a non-condensable carrier gas).
    """
    if not math.isfinite(y_h2o) or not (0.0 <= y_h2o < 1.0):
        raise ValueError(f"y_h2o must be in [0, 1), got {y_h2o!r}.")
    if not math.isfinite(p_total) or p_total <= 0.0:
        raise ValueError("p_total must be a positive finite value [Pa].")

    p_h2o = y_h2o * p_total
    if p_h2o >= p_total:
        raise ValueError(
            f"Computed water partial pressure ({p_h2o:.6g} Pa) must be less "
            f"than the total pressure ({p_total:.6g} Pa)."
        )
    return p_h2o


def is_frost_regime(p_h2o: float) -> bool:
    """Return True if the equilibrium dew point for ``p_h2o`` would fall
    below the water triple point (i.e. frost/ice, not liquid condensation).

    Callers must check this *before* calling ``water_dew_point`` when
    ``p_h2o`` might be very low, and handle it via the
    ``FROSTING_NOT_SUPPORTED`` path (see ``core.phase_change.regime`` /
    ``core.phase_change.outside_condensation_solver``) rather than letting
    the bisection in ``water_dew_point`` fail.
    """
    if not math.isfinite(p_h2o) or p_h2o <= 0.0:
        raise ValueError("p_h2o must be a positive finite value [Pa].")
    return p_h2o < water_saturation_pressure(WATER_TRIPLE_POINT_TEMPERATURE_K)


def water_dew_point(p_h2o: float, *, tolerance_K: float = 1e-6, max_iterations: int = 200) -> float:
    """Return the dew-point temperature [K] at which p_sat_H2O(T) = p_h2o.

    Solved by bisection (interval method) on
    ``core.properties.water.water_saturation_pressure``, which is monotonic
    increasing in T over the saturation line.

    Raises:
        ValueError: if ``p_h2o`` is outside the achievable saturation-curve
            range [p_sat(triple point), p_sat(critical point)]. A ``p_h2o``
            below the triple-point saturation pressure means the true
            equilibrium is frost/ice, not liquid water; check
            ``is_frost_regime(p_h2o)`` first and handle that case via
            ``FROSTING_NOT_SUPPORTED`` instead of calling this function.
    """
    if not math.isfinite(p_h2o) or p_h2o <= 0.0:
        raise ValueError("p_h2o must be a positive finite value [Pa].")

    p_lo = water_saturation_pressure(WATER_TRIPLE_POINT_TEMPERATURE_K)
    p_hi = water_saturation_pressure(WATER_CRITICAL_TEMPERATURE_K - 1e-6)
    if p_h2o < p_lo:
        raise ValueError(
            f"water_dew_point: p_h2o={p_h2o:.6g} Pa is below the triple-point "
            f"saturation pressure ({p_lo:.6g} Pa); the equilibrium state is "
            "frost/ice, not liquid water. Check is_frost_regime(p_h2o) first."
        )
    if p_h2o > p_hi:
        raise ValueError(
            f"water_dew_point: p_h2o={p_h2o:.6g} Pa is above the achievable "
            f"saturation-curve range (critical pressure limit {p_hi:.6g} Pa)."
        )

    T_lo, T_hi = WATER_TRIPLE_POINT_TEMPERATURE_K, WATER_CRITICAL_TEMPERATURE_K - 1e-6
    for _ in range(max_iterations):
        T_mid = 0.5 * (T_lo + T_hi)
        p_mid = water_saturation_pressure(T_mid)
        if p_mid < p_h2o:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if (T_hi - T_lo) < tolerance_K:
            break
    return 0.5 * (T_lo + T_hi)


def dry_gas_average_molar_mass(
    dry_mole_fractions: dict[str, float],
    *,
    molar_masses: dict[str, float] | None = None,
) -> float:
    """Return the mole-fraction-weighted average molar mass of a dry-gas
    composition [kg/mol]."""
    from core.properties.gas_mixture import component_molar_mass

    total = sum(dry_mole_fractions.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("dry_mole_fractions must sum to a positive value.")
    return sum(
        (fraction / total) * component_molar_mass(name, molar_masses=molar_masses)
        for name, fraction in dry_mole_fractions.items()
    )


def saturated_water_ratio(
    *,
    p_total: float,
    T: float,
    M_dry: float,
    M_h2o: float = WATER_MOLAR_MASS_KG_PER_MOL,
) -> float:
    """Return the saturated water content W_sat(T, p_total) [kg H2O / kg dry gas].

    General ideal-gas form of the classic psychrometric humidity-ratio
    formula:

        W_sat = (M_h2o / M_dry) * p_sat(T) / (p_total - p_sat(T))

    For dry air (``M_dry`` = 28.97 g/mol), ``M_h2o/M_dry`` = 0.622 and this
    reduces to the familiar ``W_sat = 0.622 * p_sat/(p - p_sat)`` -- see the
    dry-air regression test.

    Raises:
        ValueError: if ``p_sat(T) >= p_total`` (no dry carrier headroom
            left; T is at or above the boiling point of water at
            ``p_total``), or if ``M_dry``/``M_h2o`` are not positive.
    """
    if not math.isfinite(M_dry) or M_dry <= 0.0:
        raise ValueError("M_dry must be a positive finite value [kg/mol].")
    if not math.isfinite(M_h2o) or M_h2o <= 0.0:
        raise ValueError("M_h2o must be a positive finite value [kg/mol].")
    if not math.isfinite(p_total) or p_total <= 0.0:
        raise ValueError("p_total must be a positive finite value [Pa].")

    p_sat = water_saturation_pressure(T)
    if p_sat >= p_total:
        raise ValueError(
            f"saturated_water_ratio: p_sat(T={T:.6g} K)={p_sat:.6g} Pa >= "
            f"p_total={p_total:.6g} Pa; no saturated gas-phase state exists "
            "at this temperature and pressure."
        )
    return (M_h2o / M_dry) * p_sat / (p_total - p_sat)


def water_mole_fraction_from_ratio(W: float, *, M_dry: float, M_h2o: float = WATER_MOLAR_MASS_KG_PER_MOL) -> float:
    """Return the H2O mole fraction y_H2O implied by W [kg H2O/kg dry gas].

    Inverse of the mixing relation used in
    ``core.phase_change.capability._detect_gas_mixture_capability``:
    ``W = (y/(1-y)) * (M_h2o/M_dry)``.
    """
    if not math.isfinite(W) or W < 0.0:
        raise ValueError("W must be a non-negative finite value [kg/kg].")
    n_h2o_per_kg_dry = W / M_h2o
    n_dry_per_kg_dry = 1.0 / M_dry
    return n_h2o_per_kg_dry / (n_h2o_per_kg_dry + n_dry_per_kg_dry)
