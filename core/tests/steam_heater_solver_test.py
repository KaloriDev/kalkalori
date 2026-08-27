import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, CircularFinnedTube
from core.heat_transfer.ntu import ntu_from_effectiveness
from core.models.bare_tube import BareTubeHeatExchanger
from core.phase_change.steam_condensation import SteamTubeOrientation
from core.phase_change.steam_heater import (
    SteamEvaporationNotSupportedError,
    SteamHeaterZoneKind,
    _equivalent_inside_alpha_outer_basis,
    _log_mean_temperature_difference,
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


def _rate(inlet, outlet, *, hx=None, m_dot=1.0, mass_flow_outside=100.0):
    # The generous outside flow keeps every parallel branch away from its
    # thermal pinch in the deeply subcooled fixtures used below.
    return rate_steam_heater(
        hx or _hx(), inlet_state=inlet, outlet_state=outlet,
        mass_flow_steam=m_dot, outside_provider=OUTSIDE,
        mass_flow_outside=mass_flow_outside, T_in_outside=300.0, p_outside=101325.0,
        orientation=SteamTubeOrientation.VERTICAL_DOWNWARD,
    )


def _assert_parallel_air_allocation(
    result,
    *,
    hx,
    mass_flow_outside=100.0,
    T_in_outside=300.0,
):
    """Check geometry allocation and constant-cp outlet mixing invariants."""
    cp_outside = OUTSIDE.props.cp
    frontal_area = hx.bundle.frontal_flow_area
    expected_face_mass_flux = mass_flow_outside / frontal_area
    expected_face_velocity = expected_face_mass_flux / result.outside_props_mean.rho

    assert result.zone_allocation_method == "parallel_by_geometry"
    assert result.zone_allocation_converged is True
    assert 1 <= result.zone_allocation_iterations <= 80
    assert 0.0 <= result.zone_allocation_residual <= 1.0e-8
    assert result.sum_zone_area_fraction == pytest.approx(1.0, abs=2.0e-12)
    assert result.sum_zone_air_mass_flow == pytest.approx(mass_flow_outside)
    assert result.mixed_outside_T_out == pytest.approx(result.T_out_outside)
    assert result.Q_zone_sum == pytest.approx(result.Q_total)
    assert abs(result.mixed_air_energy_residual) <= max(1.0e-6, 1.0e-12 * result.Q_total)

    for zone in result.zones:
        assert zone.outside_T_in == pytest.approx(T_in_outside)
        assert zone.outside_T_out > zone.outside_T_in
        assert zone.area_fraction == pytest.approx(zone.area / result.A_total)
        assert zone.tube_length_fraction == pytest.approx(zone.area_fraction)
        assert zone.outside_mass_flow == pytest.approx(
            mass_flow_outside * zone.outside_mass_flow_fraction
        )
        assert zone.outside_mass_flow_fraction == pytest.approx(
            zone.area_fraction, abs=2.0e-8
        )
        assert zone.outside_frontal_area == pytest.approx(
            frontal_area * zone.outside_mass_flow_fraction
        )
        assert zone.outside_face_mass_flux == pytest.approx(expected_face_mass_flux)
        assert zone.outside_mass_flow / zone.outside_frontal_area == pytest.approx(
            expected_face_mass_flux
        )
        assert zone.outside_velocity == pytest.approx(expected_face_velocity)
        assert zone.alpha_outside == pytest.approx(result.outside_alpha)
        assert zone.alpha_outside_physical == pytest.approx(
            result.outside_alpha_physical
        )
        assert zone.Q == pytest.approx(
            zone.outside_mass_flow
            * cp_outside
            * (zone.outside_T_out - zone.outside_T_in),
            rel=2.0e-12,
            abs=1.0e-6,
        )

    area_residual = max(
        abs(zone.area_fraction - zone.outside_mass_flow_fraction)
        for zone in result.zones
    )
    assert result.zone_allocation_residual == pytest.approx(
        area_residual, rel=0.0, abs=2.0e-15
    )
    assert sum(zone.outside_mass_flow_fraction for zone in result.zones) == pytest.approx(
        1.0
    )
    mixed_temperature = sum(
        zone.outside_mass_flow * zone.outside_T_out for zone in result.zones
    ) / mass_flow_outside
    assert result.T_out_outside == pytest.approx(mixed_temperature, rel=2.0e-12)
    assert result.Q_total == pytest.approx(
        mass_flow_outside
        * cp_outside
        * (result.T_out_outside - T_in_outside),
        rel=2.0e-12,
        abs=1.0e-6,
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


def test_zero_duty_boundary_zone_is_omitted_before_parallel_normalization():
    hx = _hx()
    result = _rate(
        water_steam_props_iapws97(T=520.0, p=P),
        water_steam_props_iapws97(p=P, x=1.0),
        hx=hx,
    )
    assert tuple(zone.kind for zone in result.zones) == (
        SteamHeaterZoneKind.SUPERHEAT,
    )
    _assert_parallel_air_allocation(result, hx=hx)


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
    (
        "inlet",
        "outlet",
        "zone_kind",
        "expected_area",
        "expected_UA",
        "expected_outside_T_out",
        "expected_alpha_inside",
    ),
    [
        (
            water_steam_props_iapws97(p=P, x=1.0),
            water_steam_props_iapws97(p=P, x=0.0),
            SteamHeaterZoneKind.CONDENSATION,
            28.661144675225202,
            14108.76015346184,
            320.04414620248014,
            8826.095321215491,
        ),
        (
            water_steam_props_iapws97(T=520.0, p=P),
            water_steam_props_iapws97(T=480.0, p=P),
            SteamHeaterZoneKind.SUPERHEAT,
            3.592158897826218,
            460.1422854716331,
            300.9105525423216,
            203.00640157303883,
        ),
        (
            water_steam_props_iapws97(p=P, x=0.0),
            water_steam_props_iapws97(T=350.0, p=P),
            SteamHeaterZoneKind.SUBCOOLING,
            28.43958784441023,
            4909.416618347276,
            304.379916602776,
            307.89897058408707,
        ),
    ],
)
def test_single_zone_parallel_limit_preserves_numerical_regression(
    inlet,
    outlet,
    zone_kind,
    expected_area,
    expected_UA,
    expected_outside_T_out,
    expected_alpha_inside,
):
    hx = _hx()
    result = _rate(inlet, outlet, hx=hx)
    _assert_parallel_air_allocation(result, hx=hx)
    assert tuple(zone.kind for zone in result.zones) == (zone_kind,)
    zone_alpha = result.zones[0].alpha_inside
    assert result.A_total == pytest.approx(expected_area, rel=2.0e-12)
    assert result.UA_total == pytest.approx(expected_UA, rel=2.0e-12)
    assert result.T_out_outside == pytest.approx(expected_outside_T_out, rel=2.0e-12)
    assert zone_alpha == pytest.approx(expected_alpha_inside, rel=2.0e-12)
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
    # The nested phase-state and parallel-allocation solves remain bounded.
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


# ---------------------------------------------------------------------------
# Zone driving-force model (corrected steam-heater zone driving forces)
# ---------------------------------------------------------------------------

def _finned_hx(n_rows=10, n_tubes_per_row=10):
    core = BareTube(
        D_i=0.020, D_o=0.024, length_total=4.0, length_effective=4.0, wall_k=16.0,
    )
    finned = CircularFinnedTube(
        core_tube=core, fin_k=200.0, D_fin=0.045, D_root=0.024,
        fin_thickness_root=0.0005, fin_pitch=0.0024, fin_contact_resistance=None,
    )
    pitch_transverse = 0.06
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=finned, n_rows=n_rows, n_tubes_per_row=n_tubes_per_row,
            pitch_transverse=pitch_transverse,
            pitch_longitudinal=pitch_transverse * math.sqrt(3.0) / 2.0,
            layout="staggered", n_passes_tube=1, flow_arrangement="crossflow",
        )
    )


def test_pure_condensation_zone_matches_independent_terminal_lmtd():
    """T1: the condensation-zone driving force is the exact terminal LMTD.

    The expected value is computed here from first principles, not by
    calling the production ``_log_mean_temperature_difference`` helper.
    """
    result = _rate(
        water_steam_props_iapws97(p=P, x=1.0),
        water_steam_props_iapws97(p=P, x=0.0),
    )
    assert tuple(zone.kind for zone in result.zones) == (SteamHeaterZoneKind.CONDENSATION,)
    zone = result.zones[0]
    Tsat = result.saturation.Tsat
    dT1 = Tsat - zone.outside_T_in
    dT2 = Tsat - zone.outside_T_out
    lmtd = (dT1 - dT2) / math.log(dT1 / dT2)
    assert zone.driving_force_method == "isothermal_lmtd"
    assert zone.effective_mean_temperature_difference == pytest.approx(lmtd, rel=1.0e-12)
    assert zone.UA == pytest.approx(zone.Q / lmtd, rel=1.0e-12)
    assert zone.area == pytest.approx(zone.UA / zone.U, rel=1.0e-12)


@pytest.mark.parametrize(
    ("delta_T_in", "delta_T_out"),
    [(50.0, 50.0), (50.0, 50.0 + 1.0e-7), (50.0 - 1.0e-8, 50.0)],
)
def test_lmtd_helper_equal_terminal_difference_limit(delta_T_in, delta_T_out):
    """T2: no NaN / division-by-zero at (or near) equal terminal differences."""
    value = _log_mean_temperature_difference(delta_T_in, delta_T_out)
    assert math.isfinite(value)
    assert value == pytest.approx(0.5 * (delta_T_in + delta_T_out), rel=1.0e-6)


def test_lmtd_helper_rejects_non_positive_terminal_difference():
    with pytest.raises(ValueError):
        _log_mean_temperature_difference(0.0, 10.0)
    with pytest.raises(ValueError):
        _log_mean_temperature_difference(10.0, -1.0)


def test_condensation_zone_non_positive_approach_rejects_finite_solution():
    """T3: Tsat <= T_air_zone_out must reject a finite solution, not fake one."""
    inlet = water_steam_props_iapws97(p=P, x=1.0)
    outlet = water_steam_props_iapws97(p=P, x=0.0)
    Tsat = water_steam_props_iapws97(p=P, x=0.5).T
    with pytest.raises(ValueError, match="positive zone temperature difference"):
        rate_steam_heater(
            _hx(), inlet_state=inlet, outlet_state=outlet,
            mass_flow_steam=1.0, outside_provider=OUTSIDE,
            # A small outside mass flow drives the sole branch outlet above Tsat.
            mass_flow_outside=0.05, T_in_outside=Tsat - 5.0, p_outside=101325.0,
            orientation=SteamTubeOrientation.VERTICAL_DOWNWARD,
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
            water_steam_props_iapws97(p=P, x=0.8),
            water_steam_props_iapws97(T=350.0, p=P),
            (SteamHeaterZoneKind.CONDENSATION, SteamHeaterZoneKind.SUBCOOLING),
        ),
        (
            water_steam_props_iapws97(T=520.0, p=P),
            water_steam_props_iapws97(T=350.0, p=P),
            (
                SteamHeaterZoneKind.SUPERHEAT,
                SteamHeaterZoneKind.CONDENSATION,
                SteamHeaterZoneKind.SUBCOOLING,
            ),
        ),
    ],
)
def test_multizone_air_branches_share_inlet_and_converge_by_geometry(
    inlet, outlet, expected_kinds
):
    """T4: parallel branches allocate by area and mix by constant-cp energy."""
    hx = _hx()
    result = _rate(inlet, outlet, hx=hx)
    assert tuple(zone.kind for zone in result.zones) == expected_kinds
    _assert_parallel_air_allocation(result, hx=hx)
    assert any(
        zone.area_fraction != pytest.approx(zone.Q / result.Q_total, rel=1.0e-4)
        for zone in result.zones
    )
    assert sum(zone.Q for zone in result.zones) == pytest.approx(result.Q_total)
    assert sum(zone.UA for zone in result.zones) == pytest.approx(result.UA_total)
    assert result.Q_total / result.UA_total == pytest.approx(result.EMTD, rel=1.0e-12)


def test_steam_rating_emtd_invariant_holds():
    """Section 4E invariant: result.EMTD == result.Q_required / result.UA_required."""
    result = _rate(
        water_steam_props_iapws97(T=520.0, p=P),
        water_steam_props_iapws97(T=350.0, p=P),
    )
    assert result.EMTD == pytest.approx(result.Q_total / result.UA_total, rel=1.0e-12)


def test_sensible_zone_ua_matches_public_ntu_contract():
    """T5: cross-check the SUPERHEAT zone UA against the public NTU API."""
    result = _rate(
        water_steam_props_iapws97(T=520.0, p=P),
        water_steam_props_iapws97(T=480.0, p=P),
    )
    assert tuple(zone.kind for zone in result.zones) == (SteamHeaterZoneKind.SUPERHEAT,)
    zone = result.zones[0]
    C_inside = zone.Q / abs(zone.T_in - zone.T_out)
    C_outside = zone.Q / abs(zone.outside_T_out - zone.outside_T_in)
    C_min = min(C_inside, C_outside)
    eps = zone.Q / (C_min * (zone.T_in - zone.outside_T_in))
    NTU = ntu_from_effectiveness(
        eps, C_inside, C_outside,
        flow_arrangement="crossflow", C_inside=C_inside, C_outside=C_outside,
    )
    assert zone.driving_force_method == "epsilon_ntu_crossflow"
    assert zone.UA == pytest.approx(NTU * C_min, rel=1.0e-9)
    assert zone.area == pytest.approx(zone.UA / zone.U, rel=1.0e-9)


def test_bare_vs_finned_geometry_changes_parallel_allocation_not_inside_states():
    """T6: geometry changes branch allocation, not prescribed steam states."""
    inlet = water_steam_props_iapws97(T=520.0, p=P)
    outlet = water_steam_props_iapws97(T=350.0, p=P)
    bare_hx = _hx()
    finned_hx = _finned_hx()
    bare_result = _rate(inlet, outlet, hx=bare_hx)
    finned_result = _rate(inlet, outlet, hx=finned_hx)
    _assert_parallel_air_allocation(bare_result, hx=bare_hx)
    _assert_parallel_air_allocation(finned_result, hx=finned_hx)
    assert tuple(z.kind for z in bare_result.zones) == tuple(z.kind for z in finned_result.zones)
    for bare_zone, finned_zone in zip(bare_result.zones, finned_result.zones):
        assert bare_zone.T_in == pytest.approx(finned_zone.T_in)
        assert bare_zone.T_out == pytest.approx(finned_zone.T_out)
        assert bare_zone.outside_T_in == pytest.approx(finned_zone.outside_T_in)
        assert bare_zone.driving_force_method == finned_zone.driving_force_method
        assert bare_zone.alpha_inside == pytest.approx(finned_zone.alpha_inside)
        assert bare_zone.U != pytest.approx(finned_zone.U)
        assert bare_zone.area != pytest.approx(finned_zone.area)
    assert any(
        bare_zone.outside_mass_flow_fraction
        != pytest.approx(finned_zone.outside_mass_flow_fraction)
        for bare_zone, finned_zone in zip(bare_result.zones, finned_result.zones)
    )
    assert any(
        bare_zone.outside_T_out != pytest.approx(finned_zone.outside_T_out)
        for bare_zone, finned_zone in zip(bare_result.zones, finned_result.zones)
    )
