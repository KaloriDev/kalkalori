import math

import pytest

from core.geometry.tube import TubeOrientation
from core.phase_change.water_evaporation import (
    local_water_evaporation_alpha,
    solve_water_evaporation_zone,
    water_mass_flux,
)
from core.properties.water import (
    WATER_CRITICAL_PRESSURE_PA,
    water_saturation_snapshot,
)


P = 1.0e6


def _zone(**overrides):
    inputs = dict(
        p=P,
        mass_flow_total=1.0,
        flow_area_per_pass=math.pi * 0.020**2 / 4.0,
        tube_inner_diameter=0.020,
        quality_in=0.0,
        quality_out=1.0,
        heat_flux_inner=30_000.0,
        orientation=TubeOrientation.VERTICAL_UPWARD,
    )
    inputs.update(overrides)
    return solve_water_evaporation_zone(**inputs)


def test_mass_flux_uses_total_flow_and_per_pass_area():
    assert water_mass_flux(0.5, 0.002) == pytest.approx(250.0)


@pytest.mark.parametrize("mass_flow, area", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0)])
def test_mass_flux_rejects_nonpositive_inputs(mass_flow, area):
    with pytest.raises(ValueError):
        water_mass_flux(mass_flow, area)


def test_full_quality_interval_is_integrated_without_endpoint_evaluation():
    result = _zone()
    assert result.quality_in == 0.0
    assert result.quality_out == 1.0
    assert result.quadrature_order == 8
    assert all(0.0 < local.quality < 1.0 for local in result.local_results)
    assert all(math.isfinite(local.alpha) and local.alpha > 0.0 for local in result.local_results)


def test_effective_alpha_is_the_quality_weighted_harmonic_mean():
    result = _zone(quality_in=0.2, quality_out=0.85)
    # Gauss-Legendre weights integrate a constant to two; the interval
    # scaling cancels in the harmonic mean.
    weights = (
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763,
    )
    expected = 2.0 / sum(
        weight / local.alpha for weight, local in zip(weights, result.local_results)
    )
    assert result.zone_alpha_evaporation == pytest.approx(expected, rel=2e-14)
    assert result.zone_alpha_evaporation != pytest.approx(
        sum(local.alpha for local in result.local_results) / 8.0,
        rel=1e-3,
    )


@pytest.mark.parametrize("quality_in, quality_out", [(0.0, 0.1), (0.45, 0.55), (0.9, 1.0)])
def test_zone_mass_and_enthalpy_balances(quality_in, quality_out):
    sat = water_saturation_snapshot(P)
    result = _zone(quality_in=quality_in, quality_out=quality_out)
    expected_mass = result.mass_flow_total * (quality_out - quality_in)
    assert result.m_dot_evaporated == pytest.approx(expected_mass)
    assert result.Q_evaporation == pytest.approx(expected_mass * sat.hfg)
    assert result.mass_flux == pytest.approx(
        result.mass_flow_total / (math.pi * 0.020**2 / 4.0)
    )
    assert result.two_phase_pressure_drop_supported is False


def test_adapter_maps_one_saturation_snapshot_to_the_transport_boundary():
    sat = water_saturation_snapshot(P)
    result = local_water_evaporation_alpha(
        p=P,
        mass_flux=500.0,
        tube_inner_diameter=0.020,
        quality=0.5,
        heat_flux_inner=30_000.0,
        orientation=TubeOrientation.HORIZONTAL,
        saturation=sat,
    )
    liquid = sat.saturated_liquid.transport
    vapor = sat.saturated_vapor.transport
    assert liquid is not None and vapor is not None
    assert result.liquid_reynolds == pytest.approx(
        500.0 * (1.0 - 0.5) * 0.020 / liquid.mu
    )
    assert result.boiling_number == pytest.approx(30_000.0 / (500.0 * sat.hfg))
    assert result.convection_number == pytest.approx(
        ((1.0 / 0.5) - 1.0) ** 0.8 * (vapor.rho / liquid.rho) ** 0.5
    )


@pytest.mark.parametrize(
    "quality_in, quality_out",
    [(-1e-6, 0.5), (0.5, 0.5), (0.7, 0.2), (0.5, 1.000001)],
)
def test_zone_rejects_invalid_quality_intervals(quality_in, quality_out):
    with pytest.raises(ValueError):
        _zone(quality_in=quality_in, quality_out=quality_out)


def test_local_adapter_rejects_endpoints_and_supercritical_pressure():
    common = dict(
        mass_flux=500.0,
        tube_inner_diameter=0.020,
        heat_flux_inner=30_000.0,
        orientation=TubeOrientation.HORIZONTAL,
    )
    with pytest.raises(ValueError, match="0 < x < 1"):
        local_water_evaporation_alpha(p=P, quality=0.0, **common)
    with pytest.raises(ValueError, match="critical"):
        local_water_evaporation_alpha(
            p=WATER_CRITICAL_PRESSURE_PA, quality=0.5, **common
        )


def test_supplied_saturation_snapshot_must_match_pressure():
    with pytest.raises(ValueError, match="does not match"):
        local_water_evaporation_alpha(
            p=P,
            mass_flux=500.0,
            tube_inner_diameter=0.020,
            quality=0.5,
            heat_flux_inner=30_000.0,
            orientation=TubeOrientation.HORIZONTAL,
            saturation=water_saturation_snapshot(2.0e6),
        )
