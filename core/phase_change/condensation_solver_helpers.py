# KalKalori â€” Heat Exchanger Open Engine
# GNU GPL v3 only

"""Side-neutral numerical helpers for wet-gas H2O condensation.

The helpers in this module contain the physical equations shared by the
inside- and outside-condensation solvers.  Geometry, heat-transfer
correlations and hydraulic integration remain side-specific in their
orchestration layers.
"""

from __future__ import annotations

import math

from core.phase_change.mass_heat_transfer import condensation_rate
from core.phase_change import warning_codes as WC
from core.phase_change.types import PhaseChangeCapability
from core.phase_change.water_equilibrium import (
    is_frost_regime,
    saturated_water_ratio,
    water_dew_point,
    water_mole_fraction_from_ratio,
    water_partial_pressure,
)
from core.phase_change.wet_gas_enthalpy import (
    WetGasEnthalpyEvaluator,
    temperature_from_h_wet_gas_dry_basis,
)
from core.properties.water import (
    WATER_TRIPLE_POINT_TEMPERATURE_K,
    water_latent_heat_of_vaporization,
    water_saturation_liquid_enthalpy,
)


class FrostingNotSupportedError(RuntimeError):
    """Raised when liquid condensation would enter the frost/ice regime."""


class CondensateStateInconsistentError(RuntimeError):
    """Raised for positive condensate flow without a valid wet surface."""

    warning_code = WC.CONDENSATE_STATE_INCONSISTENT


def condensate_enthalpy_flow(
    *,
    m_dot_condensate: float,
    condensation_mass_tolerance: float,
    wet_surface_fraction: float,
    wet_area: float,
    wall_temperature_wet_mean: float | None,
) -> float:
    """Return drained-liquid enthalpy flow [W] with a strict wet-state invariant.

    A zero/tolerance-level condensate flow carries zero enthalpy flow and does
    not require a fictitious liquid state.  A material condensate flow must
    have a positive wet fraction and area plus a finite representative wet
    wall temperature; otherwise the coupled state is internally inconsistent.
    """
    numeric = (
        ("m_dot_condensate", m_dot_condensate),
        ("condensation_mass_tolerance", condensation_mass_tolerance),
        ("wet_surface_fraction", wet_surface_fraction),
        ("wet_area", wet_area),
    )
    for name, value in numeric:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if condensation_mass_tolerance < 0.0:
        raise ValueError("condensation_mass_tolerance must be non-negative.")
    if m_dot_condensate < 0.0:
        raise ValueError("m_dot_condensate must be non-negative.")
    if m_dot_condensate <= condensation_mass_tolerance:
        return 0.0

    if (
        wet_surface_fraction <= 0.0
        or wet_area <= 0.0
        or wall_temperature_wet_mean is None
        or not math.isfinite(wall_temperature_wet_mean)
    ):
        raise CondensateStateInconsistentError(
            "Positive condensate flow was obtained without a valid wet-"
            "surface state (wet fraction, wet area, and representative wet-"
            "wall temperature are required)."
        )
    return m_dot_condensate * water_saturation_liquid_enthalpy(
        T=wall_temperature_wet_mean
    )


def sensible_only_wall_temperature(
    *,
    T_bulk_wet_gas: float,
    T_bulk_other: float,
    R_wet_film: float,
    R_downstream: float,
) -> float:
    """Return the wet-side wall temperature from a dry resistance split."""
    q = (T_bulk_wet_gas - T_bulk_other) / (R_wet_film + R_downstream)
    return T_bulk_wet_gas - q * R_wet_film


def local_dew_point_or_triple_point(
    capability: PhaseChangeCapability,
    W: float,
    *,
    p_wet_gas: float,
) -> float:
    """Return the current dew point, with a bounded frost diagnostic floor."""
    if W <= 0.0:
        return WATER_TRIPLE_POINT_TEMPERATURE_K
    y = water_mole_fraction_from_ratio(
        W,
        M_dry=capability.M_dry,
        M_h2o=capability.M_condensable,
    )
    p_h2o = water_partial_pressure(y, p_wet_gas)
    if is_frost_regime(p_h2o):
        return WATER_TRIPLE_POINT_TEMPERATURE_K
    return water_dew_point(p_h2o)


def solve_condensing_interface_state(
    *,
    alfa_dry: float,
    A_total: float,
    A_wet: float,
    T_wall_wet_mean: float | None,
    T_bulk_wet_gas: float,
    T_bulk_other: float,
    R_downstream: float,
    cp_gas: float,
    W_bulk: float,
    p_wet_gas: float,
    M_dry: float,
    m_dot_water_vapor_available: float,
    lewis_number: float,
    tolerance_K: float = 1e-7,
    max_iterations: int = 100,
) -> tuple[float, float, float, float, float | None]:
    """Close the wet-side wall balance for either exchanger side.

    Returns ``(T_wall_mean, Q_sensible, Q_latent, m_dot_condensate,
    W_sat_wet_surface)``.  Sensible convection always uses ``A_total``;
    only mass/latent transfer uses ``A_wet``.
    """
    W_sat_wet_surface: float | None = None
    m_dot_condensate = 0.0
    Q_latent = 0.0
    if A_wet > 0.0 and T_wall_wet_mean is not None:
        if not math.isfinite(T_wall_wet_mean):
            raise ValueError("T_wall_wet_mean must be finite when supplied.")
        if T_wall_wet_mean <= WATER_TRIPLE_POINT_TEMPERATURE_K:
            raise FrostingNotSupportedError(
                "representative wet-surface temperature "
                f"{T_wall_wet_mean:.3f} K is at/below the water triple "
                f"point ({WATER_TRIPLE_POINT_TEMPERATURE_K} K)."
            )
        W_sat_wet_surface = saturated_water_ratio(
            p_total=p_wet_gas,
            T=T_wall_wet_mean,
            M_dry=M_dry,
        )
        m_dot_condensate = condensation_rate(
            alfa_dry=alfa_dry,
            cp_gas=cp_gas,
            W_bulk=W_bulk,
            W_sat_surface=W_sat_wet_surface,
            A_wet=A_wet,
            m_dot_water_vapor_available=m_dot_water_vapor_available,
            lewis_number=lewis_number,
        )
        if m_dot_condensate > 0.0:
            Q_latent = m_dot_condensate * water_latent_heat_of_vaporization(
                T=T_wall_wet_mean
            )

    def evaluate(T_wall_mean: float) -> tuple[float, float]:
        Q_sensible = alfa_dry * A_total * (T_bulk_wet_gas - T_wall_mean)
        Q_removed = (T_wall_mean - T_bulk_other) / R_downstream
        return Q_sensible + Q_latent - Q_removed, Q_sensible

    T_lo, T_hi = sorted((T_bulk_other, T_bulk_wet_gas))
    f_lo, _ = evaluate(T_lo)
    f_hi, _ = evaluate(T_hi)
    bracket_step = max(T_hi - T_lo, 1.0)
    for _ in range(60):
        if f_hi <= 0.0:
            break
        T_hi += bracket_step
        bracket_step *= 2.0
        f_hi, _ = evaluate(T_hi)

    if f_lo < 0.0 or f_hi > 0.0:
        raise ValueError(
            "wet-gas condensation wall-temperature balance is not bracketed "
            f"on [{T_lo:.6g}, {T_hi:.6g}] K (f_lo={f_lo:.6g}, "
            f"f_hi={f_hi:.6g}); the wet gas must be the hotter, cooling side."
        )

    for _ in range(max_iterations):
        T_mid = 0.5 * (T_lo + T_hi)
        f_mid, _ = evaluate(T_mid)
        if f_mid > 0.0:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if T_hi - T_lo < tolerance_K:
            break

    T_wall_mean = 0.5 * (T_lo + T_hi)
    _, Q_sensible = evaluate(T_wall_mean)
    return (
        T_wall_mean,
        Q_sensible,
        Q_latent,
        m_dot_condensate,
        W_sat_wet_surface,
    )


def invert_wet_gas_enthalpy(
    *,
    h_target: float,
    p_wet_gas: float,
    W: float,
    capability: PhaseChangeCapability,
    T_wall_mean: float,
    T_in_wet_gas: float,
    evaluator: WetGasEnthalpyEvaluator | None = None,
) -> float:
    """Invert the shared dry-basis wet-gas enthalpy with bounded brackets."""
    T_lo = max(
        WATER_TRIPLE_POINT_TEMPERATURE_K + 0.5,
        min(T_wall_mean, T_in_wet_gas) - 10.0,
    )
    T_hi = T_in_wet_gas + 5.0
    try:
        return temperature_from_h_wet_gas_dry_basis(
            h_target,
            p_wet_gas,
            W,
            capability,
            T_bracket=(T_lo, T_hi),
            tolerance_K=1e-6,
            evaluator=evaluator,
        )
    except ValueError:
        return temperature_from_h_wet_gas_dry_basis(
            h_target,
            p_wet_gas,
            W,
            capability,
            T_bracket=(
                WATER_TRIPLE_POINT_TEMPERATURE_K + 0.5,
                T_in_wet_gas + 10.0,
            ),
            tolerance_K=1e-6,
            evaluator=evaluator,
        )
