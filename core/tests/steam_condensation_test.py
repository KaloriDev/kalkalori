import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.phase_change.steam_condensation import (
    SHAH_2009_CORRELATION,
    SteamTubeOrientation,
    local_steam_condensation_alpha,
    phase_enthalpy_flow,
    solve_steam_condensation_zone,
    steam_mass_flux,
)
from core.properties.water import water_saturation_snapshot


P = 1.0e6
D = 0.020


def _local(G=100.0, x=0.5, orientation=SteamTubeOrientation.VERTICAL_DOWNWARD):
    return local_steam_condensation_alpha(
        p=P, mass_flux=G, tube_inner_diameter=D, quality=x,
        orientation=orientation,
    )


def test_shah_2009_equation_reference_point_is_stable():
    result = _local(G=100.0, x=0.5)
    assert result.correlation == SHAH_2009_CORRELATION
    assert result.regime == "I"
    assert result.alpha == pytest.approx(11568.207007, rel=2e-6)
    assert result.h_forced == pytest.approx(11568.207007, rel=2e-6)
    assert result.h_gravity == pytest.approx(3286.039661, rel=2e-6)


@pytest.mark.parametrize("G", [4.0, 100.0, 800.0])
def test_low_transition_and_high_mass_flux_htc_is_finite_positive(G):
    result = _local(G=G, x=0.5)
    assert math.isfinite(result.alpha)
    assert result.alpha > 0.0
    assert result.regime in {"I", "II", "III"}


def test_vertical_transition_regime_combines_forced_and_gravity_terms():
    result = _local(G=20.0, x=0.5)
    assert result.regime == "II"
    assert result.alpha == pytest.approx(result.h_forced + result.h_gravity)


@pytest.mark.parametrize("quality", [1.0e-6, 0.5, 1.0 - 1.0e-6])
def test_local_quality_near_endpoints_is_finite_and_bounded(quality):
    result = _local(G=20.0, x=quality)
    assert math.isfinite(result.alpha)
    assert result.alpha > 0.0
    assert 0.0 < result.quality < 1.0


def test_mass_flux_uses_tubes_per_pass_not_all_tubes():
    tube = BareTube(
        D_i=0.020, D_o=0.024, length_total=4.0,
        length_effective=4.0, wall_k=16.0,
    )
    bundle = TubeBundle(
        tube=tube, n_rows=20, n_tubes_per_row=10,
        pitch_transverse=0.04, pitch_longitudinal=0.04,
        layout="inline", n_passes_tube=4, flow_arrangement="crossflow",
    )
    m_dot = 2.0
    expected = m_dot / (bundle.n_tubes_per_pass * math.pi * D**2 / 4.0)
    assert bundle.n_tubes_per_pass == 50
    assert steam_mass_flux(m_dot, bundle.internal_flow_area_per_pass) == pytest.approx(expected)


def test_zone_conserves_total_water_and_exact_latent_energy():
    saturation = water_saturation_snapshot(P)
    zone = solve_steam_condensation_zone(
        p=P, mass_flow_total=2.5, flow_area_per_pass=0.05,
        tube_inner_diameter=D, quality_in=0.9, quality_out=0.2,
        orientation=SteamTubeOrientation.VERTICAL_DOWNWARD,
        saturation=saturation,
    )
    assert zone.mass_vapor_in + zone.mass_liquid_in == pytest.approx(2.5)
    assert zone.mass_vapor_out + zone.mass_liquid_out == pytest.approx(2.5)
    assert zone.mass_vapor_out < zone.mass_vapor_in
    assert zone.Q_condensation == pytest.approx(2.5 * saturation.hfg * 0.7)
    assert zone.zone_alpha_condensation > 0.0
    assert zone.two_phase_pressure_drop_supported is False
    assert all(0.0 < item.quality < 1.0 for item in zone.local_results)


def test_zone_rejects_quality_increase_or_out_of_range():
    for quality_in, quality_out in ((0.2, 0.3), (1.1, 0.5), (0.5, -0.1)):
        with pytest.raises(ValueError, match="Condensation requires"):
            solve_steam_condensation_zone(
                p=P, mass_flow_total=1.0, flow_area_per_pass=0.01,
                tube_inner_diameter=D, quality_in=quality_in,
                quality_out=quality_out,
                orientation=SteamTubeOrientation.VERTICAL_DOWNWARD,
            )


def test_applicability_warnings_are_returned_without_clipping():
    result = _local(G=3.0, x=0.5)
    assert result.mass_flux == 3.0
    assert any(w.code == "STEAM_CONDENSATION_SHAH_2009_OUTSIDE_RANGE" for w in result.warnings)


def test_horizontal_low_flow_has_explicit_unverified_warning():
    result = _local(G=4.0, x=0.5, orientation=SteamTubeOrientation.HORIZONTAL)
    if result.regime == "II":
        assert any(
            w.code == "STEAM_CONDENSATION_HORIZONTAL_LOW_RE_GT_UNVERIFIED"
            for w in result.warnings
        )


def test_absent_phase_uses_zero_enthalpy_flow_without_fake_specific_enthalpy():
    assert phase_enthalpy_flow(0.0, None) == 0.0
    with pytest.raises(ValueError, match="requires a thermodynamic phase state"):
        phase_enthalpy_flow(0.1, None)
    vapor = water_saturation_snapshot(P).saturated_vapor
    assert phase_enthalpy_flow(0.1, vapor) == pytest.approx(0.1 * vapor.h)


def test_low_g_industrial_heater_retains_gravity_film_heat_transfer():
    result = _local(G=4.0, x=0.5)
    assert result.h_gravity > 1000.0
    assert result.alpha > 1000.0
    assert result.alpha >= min(result.h_forced, result.h_gravity)


def test_orientation_is_explicit_and_validated():
    with pytest.raises(ValueError, match="explicit"):
        local_steam_condensation_alpha(
            p=P, mass_flux=20.0, tube_inner_diameter=D, quality=0.5,
            orientation="vertical_downward",
        )
