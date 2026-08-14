# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only

"""Efficiency of full circular (annular) fins with optional linear taper.

The fin model solves the standard one-dimensional radial balance

    d/dr [k_fin A_c(r) d(theta)/dr] - h_o P_s(r) theta = 0

where ``A_c(r) = 2*pi*r*t(r)`` and ``t(r)`` varies linearly from the root
thickness to the tip thickness.  ``P_s(r)`` includes both sloping faces of
the fin.  The root temperature is prescribed and the outside edge has a
convective boundary condition.  No average fin thickness is substituted.

Constant-thickness fins use the exact modified-Bessel solution at the physical
tip radius, including the convective condition ``-k*d(theta)/dr = h*theta``.
Exponentially scaled Bessel functions avoid overflow in otherwise finite
function products.  No corrected-radius/adiabatic-tip approximation is used.

Linearly tapered fins retain the conservative cell-centred finite-volume
solver and local Thomas tridiagonal solver.  The last half-cell conduction
resistance and tip convection resistance are combined in series. Integrated
convection is used for FVM efficiency because it remains well conditioned in
the isothermal-fin limit; the separately exposed root-flux diagnostic can lose
relative precision there through subtraction of nearly equal temperatures.

The geometry represents a continuous spiral fin as an equivalent periodic
train of full annular fins; see
:class:`core.geometry.finned_tube.CircularFinnedTube`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from core.geometry.finned_tube import CircularFinnedTube
from core.heat_transfer.modified_bessel import (
    scaled_bessel_i0,
    scaled_bessel_i1,
    scaled_bessel_k0,
    scaled_bessel_k1,
)


DEFAULT_FIN_RADIAL_CELLS = 400
_ANALYTICAL_METHOD = (
    "analytical_modified_bessel_constant_annular_fin_convective_tip"
)
_FINITE_VOLUME_METHOD = "conservative_finite_volume_linear_taper_annular_fin"
_BESSEL_APPROXIMATION_RELATIVE_TOLERANCE = 2.0e-7


@dataclass(frozen=True)
class FinEfficiencyResult:
    """Thermal effectiveness and area diagnostics for one finned tube.

    ``fin_heat_rate_per_base_temperature`` is the stable integrated convective
    heat rate for one geometric annular fin and a unit root-to-fluid
    temperature difference [W/K].  Gross/effective
    areas refer to one tube over ``length_effective``.  If the geometry has
    an external-area override, its excess over primary area is treated as
    fin-attributed area for the overall surface-efficiency calculation; the
    differential-equation solution itself always uses physical fin geometry.
    """

    fin_efficiency: float
    overall_surface_efficiency: float
    fin_heat_rate_per_base_temperature: float
    primary_outside_area: float
    fin_area: float
    outside_area_gross: float
    outside_area_geometric: float
    outside_area_effective: float
    outside_htc: float
    radial_cells: int
    tip_temperature_ratio: float
    root_heat_rate_per_base_temperature: float
    convective_heat_rate_per_base_temperature: float
    energy_balance_relative_error: float
    method: str = _FINITE_VOLUME_METHOD


def annular_fin_efficiency(
    tube: CircularFinnedTube,
    outside_htc: float,
    *,
    radial_cells: int = DEFAULT_FIN_RADIAL_CELLS,
) -> FinEfficiencyResult:
    """Return annular-fin efficiency and tube-area metrics.

    Constant thickness dispatches to the exact analytical modified-Bessel
    solution. A genuine linear taper dispatches to the finite-volume solver.
    Callers therefore do not choose the physical solver explicitly.

    Parameters
    ----------
    tube:
        Validated periodic circular-finned-tube geometry.
    outside_htc:
        Physical dry outside convection coefficient ``h_o`` [W/(m2 K)],
        before fin efficiency is applied.
    radial_cells:
        Number of uniform cell-centred finite volumes over the fin height for
        the tapered/reference path. It remains validated for every call for
        public API compatibility, but is not used by the analytical path.
    """
    _validate_fin_efficiency_inputs(tube, outside_htc, radial_cells)
    if tube.fin_thickness_root == tube.fin_thickness_tip_effective:
        return _annular_fin_efficiency_analytical(tube, outside_htc)
    return _annular_fin_efficiency_fvm(
        tube,
        outside_htc,
        radial_cells=radial_cells,
    )


def _annular_fin_efficiency_analytical(
    tube: CircularFinnedTube,
    outside_htc: float,
) -> FinEfficiencyResult:
    """Solve a constant-thickness annular fin with an exact convective tip.

    For ``theta = T - T_inf`` the solution of

    ``d/dr(r*dtheta/dr) - m**2*r*theta = 0``

    is a combination of ``I0(m*r)`` and ``K0(m*r)``.  The coefficients below
    impose ``theta(r_root) = 1`` and the physical-radius tip condition
    ``-k*dtheta/dr = h*theta`` at ``r_tip``.  Exponentially scaled Bessel
    functions and a common exponential cancellation are used throughout.
    """
    r_root = 0.5 * tube.D_root
    r_tip = 0.5 * tube.D_fin
    thickness = tube.fin_thickness_root
    k_fin = tube.fin_k

    m = math.sqrt(2.0 * outside_htc / (k_fin * thickness))
    x_root = m * r_root
    x_tip = m * r_tip
    tip_biot_over_m = outside_htc / (m * k_fin)
    argument_delta = x_tip - x_root

    i0_root = scaled_bessel_i0(x_root)
    i1_root = scaled_bessel_i1(x_root)
    k0_root = scaled_bessel_k0(x_root)
    k1_root = scaled_bessel_k1(x_root)
    i0_tip = scaled_bessel_i0(x_tip)
    i1_tip = scaled_bessel_i1(x_tip)
    k0_tip = scaled_bessel_k0(x_tip)
    k1_tip = scaled_bessel_k1(x_tip)

    # If unscaled functions were used, the corresponding coefficients would
    # be A = I1(x_tip) + beta*I0(x_tip) and
    # C = K1(x_tip) - beta*K0(x_tip), beta = h/(m*k).
    coefficient_i = i1_tip + tip_biot_over_m * i0_tip
    coefficient_k = k1_tip - tip_biot_over_m * k0_tip
    attenuation = math.exp(-2.0 * argument_delta)

    denominator = (
        coefficient_i * k0_root
        + attenuation * coefficient_k * i0_root
    )
    numerator = (
        coefficient_i * k1_root
        - attenuation * coefficient_k * i1_root
    )
    if not (
        math.isfinite(denominator)
        and math.isfinite(numerator)
        and denominator > 0.0
        and numerator > 0.0
    ):
        raise ValueError(
            "Analytical annular-fin solution is numerically degenerate."
        )

    dimensionless_root_gradient = numerator / denominator
    root_heat_rate_raw = (
        2.0
        * math.pi
        * k_fin
        * r_root
        * thickness
        * m
        * dimensionless_root_gradient
    )
    ideal_fin_heat_rate = outside_htc * tube.fin_area_per_fin
    efficiency = _validated_analytical_efficiency(
        root_heat_rate_raw / ideal_fin_heat_rate
    )
    # Preserve q = eta*h*A exactly if a roundoff-sized overshoot was normalized.
    root_heat_rate = efficiency * ideal_fin_heat_rate

    # At the physical tip the two common exponential factors cancel; only
    # exp(-(x_tip-x_root)) remains relative to the normalized root solution.
    tip_numerator = (
        coefficient_k * i0_tip + coefficient_i * k0_tip
    )
    tip_temperature = (
        math.exp(-argument_delta) * tip_numerator / denominator
    )
    if not (
        math.isfinite(tip_temperature)
        and 0.0 <= tip_temperature <= (
            1.0 + _BESSEL_APPROXIMATION_RELATIVE_TOLERANCE
        )
    ):
        raise ValueError(
            "Analytical annular-fin tip temperature is outside its physical range."
        )
    tip_temperature = min(tip_temperature, 1.0)

    return _build_fin_efficiency_result(
        tube=tube,
        outside_htc=outside_htc,
        efficiency=efficiency,
        radial_cells=0,
        tip_temperature_ratio=tip_temperature,
        root_heat_rate=root_heat_rate,
        convective_heat_rate=root_heat_rate,
        energy_balance_relative_error=0.0,
        method=_ANALYTICAL_METHOD,
    )


def _validated_analytical_efficiency(efficiency: float) -> float:
    """Allow only the documented modified-Bessel approximation error."""
    upper_bound = 1.0 + _BESSEL_APPROXIMATION_RELATIVE_TOLERANCE
    if not (
        math.isfinite(efficiency)
        and efficiency > 0.0
        and efficiency <= upper_bound
    ):
        raise ValueError(
            "Computed analytical annular-fin efficiency is outside its "
            "physical range beyond the modified-Bessel approximation tolerance."
        )
    # This is deliberately not a broad [0, 1] clip: negative values and any
    # overshoot beyond the helper's stated ~2e-7 error are rejected above.
    return min(efficiency, 1.0)


def _annular_fin_efficiency_fvm(
    tube: CircularFinnedTube,
    outside_htc: float,
    *,
    radial_cells: int = DEFAULT_FIN_RADIAL_CELLS,
) -> FinEfficiencyResult:
    """Reference finite-volume solver, used publicly for tapered fins."""
    _validate_fin_efficiency_inputs(tube, outside_htc, radial_cells)

    r_root = 0.5 * tube.D_root
    r_tip = 0.5 * tube.D_fin
    height = r_tip - r_root
    dr = height / float(radial_cells)
    k_fin = tube.fin_k
    slope_factor = tube.fin_side_slope_factor

    lower = [0.0] * radial_cells
    diagonal = [0.0] * radial_cells
    upper = [0.0] * radial_cells
    rhs = [0.0] * radial_cells
    side_conductances = [0.0] * radial_cells

    root_conductance = 0.0
    tip_effective_conductance = 0.0

    for index in range(radial_cells):
        r_west = r_root + index * dr
        r_east = r_west + dr

        # The exact sloping-face area in this radial control volume.
        side_area = (
            2.0
            * math.pi
            * slope_factor
            * (r_east * r_east - r_west * r_west)
        )
        side_conductance = outside_htc * side_area
        side_conductances[index] = side_conductance

        if index == 0:
            root_area = _radial_conduction_area(tube, r_root)
            west_conductance = k_fin * root_area / (0.5 * dr)
            root_conductance = west_conductance
            rhs[index] = west_conductance  # theta_root / theta_root = 1
        else:
            west_area = _radial_conduction_area(tube, r_west)
            west_conductance = k_fin * west_area / dr
            lower[index] = -west_conductance

        if index == radial_cells - 1:
            tip_area = tube.fin_tip_area_per_fin
            half_cell_resistance = (0.5 * dr) / (k_fin * tip_area)
            convection_resistance = 1.0 / (outside_htc * tip_area)
            east_conductance = 1.0 / (
                half_cell_resistance + convection_resistance
            )
            tip_effective_conductance = east_conductance
        else:
            east_area = _radial_conduction_area(tube, r_east)
            east_conductance = k_fin * east_area / dr
            upper[index] = -east_conductance

        diagonal[index] = (
            west_conductance + east_conductance + side_conductance
        )

    temperatures = _solve_tridiagonal(lower, diagonal, upper, rhs)

    root_heat_rate = root_conductance * (1.0 - temperatures[0])
    side_heat_rate = sum(
        conductance * temperature
        for conductance, temperature in zip(side_conductances, temperatures)
    )
    tip_heat_rate = tip_effective_conductance * temperatures[-1]
    convective_heat_rate = side_heat_rate + tip_heat_rate

    ideal_fin_heat_rate = outside_htc * tube.fin_area_per_fin
    # Integrated convection is numerically stable even in the isothermal-fin
    # limit.  Root flux is a subtraction of two nearly equal temperatures and
    # is therefore retained as a diagnostic rather than used to calculate eta.
    efficiency = convective_heat_rate / ideal_fin_heat_rate

    energy_scale = max(abs(convective_heat_rate), 1.0e-300)
    balance_error = abs(root_heat_rate - convective_heat_rate) / energy_scale

    # Recover the actual edge temperature from the last-cell value through
    # the convective tip boundary; this is a reporting diagnostic only.
    tip_temperature = tip_heat_rate / (outside_htc * tube.fin_tip_area_per_fin)

    return _build_fin_efficiency_result(
        tube=tube,
        outside_htc=outside_htc,
        efficiency=efficiency,
        radial_cells=radial_cells,
        tip_temperature_ratio=tip_temperature,
        root_heat_rate=root_heat_rate,
        convective_heat_rate=convective_heat_rate,
        energy_balance_relative_error=balance_error,
        method=_FINITE_VOLUME_METHOD,
    )


def _build_fin_efficiency_result(
    *,
    tube: CircularFinnedTube,
    outside_htc: float,
    efficiency: float,
    radial_cells: int,
    tip_temperature_ratio: float,
    root_heat_rate: float,
    convective_heat_rate: float,
    energy_balance_relative_error: float,
    method: str,
) -> FinEfficiencyResult:
    """Assemble shared physical-area metrics for either solution method."""
    if not math.isfinite(efficiency) or not 0.0 < efficiency <= 1.0 + 1.0e-10:
        raise ValueError(
            "Computed annular-fin efficiency is outside its physical range."
        )
    # Normalize only a roundoff-sized overshoot after rejecting larger errors.
    efficiency = min(efficiency, 1.0)

    gross_area = tube.area_outside_gross
    primary_area = tube.area_primary_outside
    fin_area = tube.area_fin
    effective_area = primary_area + efficiency * fin_area
    overall_efficiency = effective_area / gross_area
    if not (
        math.isfinite(effective_area)
        and math.isfinite(overall_efficiency)
        and effective_area > 0.0
        and 0.0 < overall_efficiency <= 1.0 + 1.0e-10
    ):
        raise ValueError(
            "Computed effective outside area is outside its physical range."
        )
    overall_efficiency = min(overall_efficiency, 1.0)

    return FinEfficiencyResult(
        fin_efficiency=efficiency,
        overall_surface_efficiency=overall_efficiency,
        fin_heat_rate_per_base_temperature=convective_heat_rate,
        primary_outside_area=primary_area,
        fin_area=fin_area,
        outside_area_gross=gross_area,
        outside_area_geometric=tube.area_outside_geometric,
        outside_area_effective=effective_area,
        outside_htc=outside_htc,
        radial_cells=radial_cells,
        tip_temperature_ratio=tip_temperature_ratio,
        root_heat_rate_per_base_temperature=root_heat_rate,
        convective_heat_rate_per_base_temperature=convective_heat_rate,
        energy_balance_relative_error=energy_balance_relative_error,
        method=method,
    )


def _validate_fin_efficiency_inputs(
    tube: CircularFinnedTube,
    outside_htc: float,
    radial_cells: int,
) -> None:
    if not isinstance(tube, CircularFinnedTube):
        raise TypeError("tube must be a CircularFinnedTube instance.")
    if not math.isfinite(outside_htc) or outside_htc <= 0.0:
        raise ValueError("outside_htc must be positive and finite.")
    if (
        isinstance(radial_cells, bool)
        or not isinstance(radial_cells, int)
        or radial_cells < 4
    ):
        raise ValueError("radial_cells must be an integer greater than or equal to 4.")


def calculate_fin_efficiency(
    tube: CircularFinnedTube,
    outside_htc: float,
    *,
    radial_cells: int = DEFAULT_FIN_RADIAL_CELLS,
) -> FinEfficiencyResult:
    """Public descriptive alias for :func:`annular_fin_efficiency`."""
    return annular_fin_efficiency(
        tube,
        outside_htc,
        radial_cells=radial_cells,
    )


def fin_efficiency(
    tube: CircularFinnedTube,
    outside_htc: float,
    *,
    radial_cells: int = DEFAULT_FIN_RADIAL_CELLS,
) -> float:
    """Return only the dimensionless annular-fin efficiency."""
    return annular_fin_efficiency(
        tube,
        outside_htc,
        radial_cells=radial_cells,
    ).fin_efficiency


def overall_surface_efficiency(
    tube: CircularFinnedTube,
    fin_efficiency_value: float,
) -> float:
    """Return ``(A_primary + eta_fin*A_fin) / A_outside_gross``."""
    if not isinstance(tube, CircularFinnedTube):
        raise TypeError("tube must be a CircularFinnedTube instance.")
    if (
        not math.isfinite(fin_efficiency_value)
        or fin_efficiency_value <= 0.0
        or fin_efficiency_value > 1.0
    ):
        raise ValueError("fin_efficiency_value must be finite and in (0, 1].")
    return (
        tube.area_primary_outside + fin_efficiency_value * tube.area_fin
    ) / tube.area_outside_gross


def effective_outside_area(
    tube: CircularFinnedTube,
    fin_efficiency_value: float,
) -> float:
    """Return fin-efficiency-weighted outside area for one tube [m2]."""
    return (
        overall_surface_efficiency(tube, fin_efficiency_value)
        * tube.area_outside_gross
    )


def _linear_thickness(tube: CircularFinnedTube, radius: float) -> float:
    r_root = 0.5 * tube.D_root
    fraction = (radius - r_root) / tube.fin_height
    return tube.fin_thickness_root + fraction * (
        tube.fin_thickness_tip_effective - tube.fin_thickness_root
    )


def _radial_conduction_area(
    tube: CircularFinnedTube,
    radius: float,
) -> float:
    return 2.0 * math.pi * radius * _linear_thickness(tube, radius)


def _solve_tridiagonal(
    lower: list[float],
    diagonal: list[float],
    upper: list[float],
    rhs: list[float],
) -> list[float]:
    """Solve a strictly diagonally dominant tridiagonal system."""
    count = len(diagonal)
    if not (len(lower) == len(upper) == len(rhs) == count) or count == 0:
        raise ValueError("Invalid tridiagonal system dimensions.")

    c_prime = [0.0] * count
    d_prime = [0.0] * count

    pivot = diagonal[0]
    if not math.isfinite(pivot) or pivot <= 0.0:
        raise ValueError("Invalid first pivot in fin-efficiency solver.")
    c_prime[0] = upper[0] / pivot
    d_prime[0] = rhs[0] / pivot

    for index in range(1, count):
        pivot = diagonal[index] - lower[index] * c_prime[index - 1]
        if not math.isfinite(pivot) or pivot <= 0.0:
            raise ValueError("Non-positive pivot in fin-efficiency solver.")
        c_prime[index] = upper[index] / pivot if index < count - 1 else 0.0
        d_prime[index] = (
            rhs[index] - lower[index] * d_prime[index - 1]
        ) / pivot

    result = [0.0] * count
    result[-1] = d_prime[-1]
    for index in range(count - 2, -1, -1):
        result[index] = d_prime[index] - c_prime[index] * result[index + 1]

    if any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0 + 1.0e-10
        for value in result
    ):
        raise ValueError("Fin-efficiency temperature solution is non-physical.")
    return result


__all__ = [
    "DEFAULT_FIN_RADIAL_CELLS",
    "FinEfficiencyResult",
    "annular_fin_efficiency",
    "calculate_fin_efficiency",
    "fin_efficiency",
    "overall_surface_efficiency",
    "effective_outside_area",
]
