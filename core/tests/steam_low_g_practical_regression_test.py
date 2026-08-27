"""Practical low-mass-flux steam-heater regression for Shah (2009)."""

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.water import (
    IAPWS97WaterSteamProvider,
    WATER_CRITICAL_PRESSURE_PA,
    WaterSteamPhase,
    water_saturation_snapshot,
)


P_STEAM = 1.4e6
M_DOT_STEAM = 900.81 / 3600.0
NODES = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
)
WEIGHTS = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
)


def _heater():
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.015,
                D_o=0.018,
                length_total=1.5,
                length_effective=1.45,
                wall_k=50.0,
                tube_orientation=TubeOrientation.HORIZONTAL,
            ),
            n_rows=10,
            n_tubes_per_row=40,
            pitch_transverse=0.025,
            pitch_longitudinal=0.025,
            layout="inline",
            n_passes_tube=1,
            flow_arrangement="crossflow",
        )
    )


def _outside():
    provider = ConstantPropertyProvider(
        FluidTransportProperties(
            rho=1.184,
            mu=1.85e-5,
            k=0.0263,
            cp=1006.0,
        )
    )
    return HXSideInput(
        provider=provider,
        m_dot=28_785.0 / 3600.0,
        T_in=298.15,
        p=101_325.0,
    )


def _independent_horizontal_alpha(*, quality, mass_flux, diameter, saturation):
    """Independent direct formula, intentionally not calling production Shah code."""
    liquid = saturation.saturated_liquid.transport
    vapor = saturation.saturated_vapor.transport
    p_r = saturation.p / WATER_CRITICAL_PRESSURE_PA
    Re_LT = mass_flux * diameter / liquid.mu
    Re_LS = mass_flux * (1.0 - quality) * diameter / liquid.mu
    Pr_l = liquid.cp * liquid.mu / liquid.k
    Z = ((1.0 - quality) / quality) ** 0.8 * p_r**0.4
    J_g = quality * mass_flux / math.sqrt(
        9.80665 * diameter * vapor.rho * (liquid.rho - vapor.rho)
    )
    h_LT = 0.023 * Re_LT**0.8 * Pr_l**0.4 * liquid.k / diameter
    n = 0.0058 + 0.557 * p_r
    h_I = h_LT * (liquid.mu / (14.0 * vapor.mu)) ** n * (
        (1.0 - quality) ** 0.8
        + 3.8
        * quality**0.76
        * (1.0 - quality) ** 0.04
        / p_r**0.38
    )
    h_Nu = 1.32 * Re_LS ** (-1.0 / 3.0) * (
        liquid.rho
        * (liquid.rho - vapor.rho)
        * 9.80665
        * liquid.k**3
        / liquid.mu**2
    ) ** (1.0 / 3.0)
    boundary = 0.98 * (Z + 0.263) ** (-0.62)
    return h_I if J_g >= boundary else h_I + h_Nu


def _independent_harmonic_zone_alpha(*, quality_out, mass_flux, diameter):
    saturation = water_saturation_snapshot(P_STEAM)
    midpoint = 0.5 * (1.0 + quality_out)
    half_range = 0.5 * (1.0 - quality_out)
    local_alphas = (
        _independent_horizontal_alpha(
            quality=midpoint + half_range * node,
            mass_flux=mass_flux,
            diameter=diameter,
            saturation=saturation,
        )
        for node in NODES
    )
    mean_inverse_alpha = sum(
        weight / alpha for weight, alpha in zip(WEIGHTS, local_alphas)
    ) / 2.0
    return 1.0 / mean_inverse_alpha


@pytest.fixture(scope="module")
def practical_result():
    hx = _heater()
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(),
        m_dot=M_DOT_STEAM,
        T_in=593.15,
        p=P_STEAM,
    )
    return hx, inside, hx.simulate(inside, _outside())


def test_practical_low_g_condensation_matches_independent_shah_reference(
    practical_result,
):
    hx, inside, result = practical_result
    steam = result.inside_phase_change
    mass_flux = inside.m_dot / hx.bundle.internal_flow_area_per_pass
    assert mass_flux == pytest.approx(3.539959612015067, rel=2.0e-12)
    assert mass_flux < 4.0
    assert steam.phase_in is WaterSteamPhase.SUPERHEATED_VAPOR
    assert steam.phase_out is WaterSteamPhase.TWO_PHASE
    assert 0.0 < steam.quality_out < 1.0
    expected = _independent_harmonic_zone_alpha(
        quality_out=steam.quality_out,
        mass_flux=mass_flux,
        diameter=float(hx.bundle.tube.D_i),
    )
    assert expected == pytest.approx(15390.902436060176, rel=2.0e-10)
    assert steam.zone_alpha_condensation == pytest.approx(expected, rel=2.0e-12)
    warning_codes = {warning.code for warning in steam.warnings}
    assert "STEAM_CONDENSATION_SHAH_2009_OUTSIDE_RANGE" in warning_codes
    assert "STEAM_CONDENSATION_HORIZONTAL_LOW_RE_GT_UNVERIFIED" in warning_codes


def test_practical_low_g_public_invariants_and_diagnostics(practical_result):
    hx, inside, result = practical_result
    steam = result.inside_phase_change
    assert steam.mass_flow_total == inside.m_dot
    assert steam.m_dot_condensate == pytest.approx(
        inside.m_dot * (1.0 - steam.quality_out)
    )
    assert steam.Q_total == pytest.approx(
        inside.m_dot * (steam.h_in - steam.h_out)
    )
    assert steam.Q_total == pytest.approx(
        steam.Q_desuperheat + steam.Q_condensation + steam.Q_subcooling
    )
    assert steam.UA_total == pytest.approx(
        steam.zone_UA_desuperheat
        + steam.zone_UA_condensation
        + steam.zone_UA_subcooling
    )
    assert steam.A_total == pytest.approx(
        steam.A_desuperheat + steam.A_condensation + steam.A_subcooling
    )
    assert steam.A_total == pytest.approx(hx.bundle.total_outer_area, rel=2.0e-8)
    assert steam.zone_fraction_desuperheat + steam.zone_fraction_condensation + steam.zone_fraction_subcooling == pytest.approx(1.0)
    assert min(
        steam.A_desuperheat, steam.A_condensation, steam.A_subcooling
    ) >= 0.0
    assert result.tube_side_hydraulic is None
    assert result.tube_side_pressure_drop is None
    assert math.isnan(result.inside_dp_total)
    assert result.outside_tube_bank_hydraulic is not None
    assert math.isfinite(result.outside_dp_total)
    assert all(
        math.isfinite(value)
        for value in (
            result.q,
            result.UA,
            result.final_result.eps,
            result.T_out_inside,
            result.T_out_outside,
            result.inside_wall_temperature_mean,
            result.outside_wall_temperature_mean,
        )
    )
    assert not hasattr(steam, "solution")
    assert not hasattr(steam, "runtime_s")
    assert steam.zone_allocation_method == "parallel_by_geometry"
    assert steam.zone_allocation_converged is True
    assert steam.zone_allocation_residual <= 1.0e-8
    # The phase-state solve now includes the bounded parallel-allocation loop.
    assert steam.property_evaluations < 350
