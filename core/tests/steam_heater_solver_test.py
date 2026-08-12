import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.phase_change.steam_condensation import SteamTubeOrientation
from core.phase_change.steam_heater import (
    SteamEvaporationNotSupportedError,
    SteamHeaterZoneKind,
    _equivalent_inside_alpha_outer_basis,
    rate_steam_heater,
    solve_steam_heater,
)
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.water import water_steam_props_iapws97


P = 1.0e6
OUTSIDE = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1005.0)
)


def _hx(n_rows=10, n_tubes_per_row=10):
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.020, D_o=0.024, length_total=4.0,
                length_effective=4.0, wall_k=16.0,
            ),
            n_rows=n_rows, n_tubes_per_row=n_tubes_per_row,
            pitch_transverse=0.04, pitch_longitudinal=0.04,
            layout="inline", n_passes_tube=1, flow_arrangement="crossflow",
        )
    )


def _rate(inlet, outlet, *, hx=None, m_dot=1.0):
    return rate_steam_heater(
        hx or _hx(), inlet_state=inlet, outlet_state=outlet,
        mass_flow_steam=m_dot, outside_provider=OUTSIDE,
        mass_flow_outside=30.0, T_in_outside=300.0, p_outside=101325.0,
        orientation=SteamTubeOrientation.VERTICAL_DOWNWARD,
    )


@pytest.mark.parametrize(
    ("inlet", "outlet", "expected_kinds"),
    [
        (water_steam_props_iapws97(T=520.0, p=P), water_steam_props_iapws97(T=480.0, p=P), (SteamHeaterZoneKind.SUPERHEAT,)),
        (water_steam_props_iapws97(T=520.0, p=P), water_steam_props_iapws97(p=P, x=0.5), (SteamHeaterZoneKind.SUPERHEAT, SteamHeaterZoneKind.CONDENSATION)),
        (water_steam_props_iapws97(T=520.0, p=P), water_steam_props_iapws97(p=P, x=0.0), (SteamHeaterZoneKind.SUPERHEAT, SteamHeaterZoneKind.CONDENSATION)),
        (water_steam_props_iapws97(T=520.0, p=P), water_steam_props_iapws97(T=350.0, p=P), (SteamHeaterZoneKind.SUPERHEAT, SteamHeaterZoneKind.CONDENSATION, SteamHeaterZoneKind.SUBCOOLING)),
        (water_steam_props_iapws97(p=P, x=1.0), water_steam_props_iapws97(p=P, x=0.4), (SteamHeaterZoneKind.CONDENSATION,)),
        (water_steam_props_iapws97(p=P, x=1.0), water_steam_props_iapws97(p=P, x=0.0), (SteamHeaterZoneKind.CONDENSATION,)),
        (water_steam_props_iapws97(p=P, x=1.0), water_steam_props_iapws97(T=350.0, p=P), (SteamHeaterZoneKind.CONDENSATION, SteamHeaterZoneKind.SUBCOOLING)),
        (water_steam_props_iapws97(p=P, x=0.8), water_steam_props_iapws97(p=P, x=0.3), (SteamHeaterZoneKind.CONDENSATION,)),
        (water_steam_props_iapws97(p=P, x=0.8), water_steam_props_iapws97(p=P, x=0.0), (SteamHeaterZoneKind.CONDENSATION,)),
        (water_steam_props_iapws97(p=P, x=0.8), water_steam_props_iapws97(T=350.0, p=P), (SteamHeaterZoneKind.CONDENSATION, SteamHeaterZoneKind.SUBCOOLING)),
        (water_steam_props_iapws97(p=P, x=0.0), water_steam_props_iapws97(T=350.0, p=P), (SteamHeaterZoneKind.SUBCOOLING,)),
    ],
)
def test_rating_supports_representative_zone_transitions(inlet, outlet, expected_kinds):
    result = _rate(inlet, outlet)
    assert tuple(zone.kind for zone in result.zones) == expected_kinds
    assert result.state_out.h == pytest.approx(outlet.h)
    assert all(zone.Q > 0.0 and zone.area > 0.0 for zone in result.zones)
    assert math.isfinite(result.UA_total)


def test_reverse_evaporation_is_unsupported():
    with pytest.raises(SteamEvaporationNotSupportedError):
        _rate(
            water_steam_props_iapws97(p=P, x=0.0),
            water_steam_props_iapws97(p=P, x=0.5),
        )


def test_zone_area_energy_and_ua_balances_are_exact():
    inlet = water_steam_props_iapws97(T=520.0, p=P)
    outlet = water_steam_props_iapws97(T=350.0, p=P)
    result = _rate(inlet, outlet)
    assert sum(zone.area for zone in result.zones) == pytest.approx(result.A_total)
    assert sum(zone.Q for zone in result.zones) == pytest.approx(result.Q_total)
    assert result.Q_total == pytest.approx(result.mass_flow_steam * (inlet.h - outlet.h))
    assert sum(zone.U * zone.area for zone in result.zones) == pytest.approx(result.UA_total)
    assert result.UA_total != pytest.approx(result.U_equivalent * _hx().bundle.total_outer_area)


@pytest.mark.parametrize(
    ("inlet", "outlet"),
    [
        (water_steam_props_iapws97(p=P, x=1.0), water_steam_props_iapws97(p=P, x=0.4)),
        (water_steam_props_iapws97(T=520.0, p=P), water_steam_props_iapws97(p=P, x=0.5)),
        (water_steam_props_iapws97(p=P, x=1.0), water_steam_props_iapws97(T=350.0, p=P)),
        (water_steam_props_iapws97(T=520.0, p=P), water_steam_props_iapws97(T=350.0, p=P)),
    ],
)
def test_equivalent_inside_alpha_reconstructs_outer_basis_resistance(inlet, outlet):
    hx = _hx()
    result = _rate(inlet, outlet, hx=hx)
    tube = hx.bundle.tube
    reconstructed_u = 1.0 / (
        tube.D_o / (tube.D_i * result.inside_alpha_equivalent)
        + tube.D_o * math.log(tube.D_o / tube.D_i) / (2.0 * tube.wall_k)
        + 1.0 / result.outside_alpha
    )
    assert reconstructed_u == pytest.approx(result.U_equivalent, rel=2.0e-12)
    assert result.UA_total == pytest.approx(
        result.U_equivalent * result.A_total, rel=2.0e-12
    )


@pytest.mark.parametrize(
    ("inlet", "outlet", "zone_kind"),
    [
        (
            water_steam_props_iapws97(p=P, x=1.0),
            water_steam_props_iapws97(p=P, x=0.4),
            SteamHeaterZoneKind.CONDENSATION,
        ),
        (
            water_steam_props_iapws97(T=520.0, p=P),
            water_steam_props_iapws97(T=480.0, p=P),
            SteamHeaterZoneKind.SUPERHEAT,
        ),
    ],
)
def test_single_zone_equivalent_and_area_weighted_alpha_equal_zone_alpha(
    inlet, outlet, zone_kind
):
    result = _rate(inlet, outlet)
    assert tuple(zone.kind for zone in result.zones) == (zone_kind,)
    zone_alpha = result.zones[0].alpha_inside
    assert result.inside_alpha_equivalent == pytest.approx(zone_alpha, rel=2.0e-12)
    assert result.inside_alpha_area_weighted == pytest.approx(zone_alpha, rel=2.0e-12)
    assert result.inside_alfa_mean == result.inside_alpha_equivalent


def test_three_zone_alphas_remain_physical_and_equivalent_is_distinct():
    result = _rate(
        water_steam_props_iapws97(T=520.0, p=P),
        water_steam_props_iapws97(T=350.0, p=P),
    )
    assert tuple(zone.kind for zone in result.zones) == (
        SteamHeaterZoneKind.SUPERHEAT,
        SteamHeaterZoneKind.CONDENSATION,
        SteamHeaterZoneKind.SUBCOOLING,
    )
    assert result.zone_alpha_desuperheat == pytest.approx(205.2660302122064)
    assert result.zone_alpha_condensation == pytest.approx(8826.095321215491)
    assert result.zone_alpha_subcooling == pytest.approx(307.89897058408707)
    assert math.isfinite(result.inside_alpha_equivalent)
    assert math.isfinite(result.inside_alpha_area_weighted)
    assert result.inside_alpha_equivalent > 0.0
    assert result.inside_alpha_area_weighted > 0.0
    assert result.inside_alpha_equivalent != pytest.approx(
        result.inside_alpha_area_weighted
    )


@pytest.mark.parametrize(
    ("inlet", "outlet", "expected_kinds"),
    [
        (
            water_steam_props_iapws97(T=520.0, p=P),
            water_steam_props_iapws97(p=P, x=0.5),
            (SteamHeaterZoneKind.SUPERHEAT, SteamHeaterZoneKind.CONDENSATION),
        ),
        (
            water_steam_props_iapws97(p=P, x=1.0),
            water_steam_props_iapws97(T=350.0, p=P),
            (SteamHeaterZoneKind.CONDENSATION, SteamHeaterZoneKind.SUBCOOLING),
        ),
    ],
)
def test_two_zone_equivalent_alpha_is_positive_and_not_area_weighted(
    inlet, outlet, expected_kinds
):
    result = _rate(inlet, outlet)
    assert tuple(zone.kind for zone in result.zones) == expected_kinds
    assert math.isfinite(result.inside_alpha_equivalent)
    assert math.isfinite(result.inside_alpha_area_weighted)
    assert result.inside_alpha_equivalent > 0.0
    assert result.inside_alpha_area_weighted > 0.0
    assert result.inside_alpha_equivalent != pytest.approx(
        result.inside_alpha_area_weighted
    )


def test_impossible_equivalent_alpha_resistance_raises_without_fallback():
    with pytest.raises(ValueError, match="no positive finite inside thermal resistance"):
        _equivalent_inside_alpha_outer_basis(
            U_equivalent=1.0e6,
            alpha_outside=250.0,
            D_i=0.020,
            D_o=0.024,
            wall_k=16.0,
        )
    with pytest.raises(ValueError, match="U_equivalent must be positive and finite"):
        _equivalent_inside_alpha_outer_basis(
            U_equivalent=0.0,
            alpha_outside=250.0,
            D_i=0.020,
            D_o=0.024,
            wall_k=16.0,
        )


@pytest.mark.parametrize(
    ("x_in", "x_out"),
    [(1.0, 1.0 - 1.0e-6), (1.0e-6, 0.0)],
)
def test_phase_boundary_continuity_near_quality_endpoints(x_in, x_out):
    result = _rate(
        water_steam_props_iapws97(p=P, x=x_in),
        water_steam_props_iapws97(p=P, x=x_out),
    )
    assert result.zone_alpha_condensation is not None
    assert math.isfinite(result.zone_alpha_condensation)
    assert result.Q_condensation > 0.0
    assert 0.0 <= result.quality_out <= result.quality_in <= 1.0


def test_simulation_allocates_all_available_area_and_conserves_energy():
    hx = _hx()
    inlet = water_steam_props_iapws97(T=520.0, p=P)
    result = solve_steam_heater(
        hx, inlet_state=inlet, mass_flow_steam=1.0,
        outside_provider=OUTSIDE, mass_flow_outside=30.0,
        T_in_outside=300.0, p_outside=101325.0,
        orientation=SteamTubeOrientation.VERTICAL_DOWNWARD,
    )
    assert result.converged
    assert result.A_total == pytest.approx(hx.bundle.total_outer_area, rel=2e-8)
    assert result.Q_total == pytest.approx(inlet.h - result.state_out.h)
    assert result.Q_total == pytest.approx(
        result.Q_desuperheat + result.Q_condensation + result.Q_subcooling
    )
    assert result.UA_total == pytest.approx(sum(zone.UA for zone in result.zones))
    assert result.property_evaluations < 300


def test_low_g_simulation_remains_finite_with_many_parallel_tubes():
    hx = _hx(n_rows=20, n_tubes_per_row=20)
    inlet = water_steam_props_iapws97(p=P, x=1.0)
    result = solve_steam_heater(
        hx, inlet_state=inlet, mass_flow_steam=0.5,
        outside_provider=OUTSIDE, mass_flow_outside=20.0,
        T_in_outside=300.0, p_outside=101325.0,
        orientation=SteamTubeOrientation.VERTICAL_DOWNWARD,
    )
    expected_G = 0.5 / hx.bundle.internal_flow_area_per_pass
    condensation = next(zone.condensation for zone in result.zones if zone.condensation is not None)
    assert condensation.mass_flux == pytest.approx(expected_G)
    assert expected_G < 5.0
    assert result.zone_alpha_condensation > 1000.0
    assert math.isfinite(result.Q_total)
    assert all(math.isfinite(value) for value in (result.A_total, result.UA_total, result.U_equivalent))
