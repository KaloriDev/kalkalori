import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.phase_change.steam_condensation import SteamTubeOrientation
from core.phase_change.steam_heater import rate_steam_heater
from core.phase_change.water_evaporator import (
    WaterCondensationRequiredError,
    WaterEvaporatorZoneKind,
    rate_water_evaporator,
    solve_water_evaporator,
)
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.water import water_steam_props_iapws97


P = 1.0e6
OUTSIDE = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1005.0)
)


def _hx(n_rows=10, n_tubes_per_row=10, *, orientation=TubeOrientation.VERTICAL_UPWARD):
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.020,
                D_o=0.024,
                length_total=4.0,
                length_effective=4.0,
                wall_k=16.0,
                tube_orientation=orientation,
            ),
            n_rows=n_rows,
            n_tubes_per_row=n_tubes_per_row,
            pitch_transverse=0.04,
            pitch_longitudinal=0.04,
            layout="inline",
            n_passes_tube=1,
            flow_arrangement="crossflow",
        )
    )


def _rate(inlet, outlet, *, hx=None, mass_flow_water=1.0, outside_provider=OUTSIDE,
          orientation=TubeOrientation.VERTICAL_UPWARD):
    return rate_water_evaporator(
        hx or _hx(),
        inlet_state=inlet,
        outlet_state=outlet,
        mass_flow_water=mass_flow_water,
        outside_provider=outside_provider,
        mass_flow_outside=30.0,
        T_in_outside=700.0,
        p_outside=101325.0,
        orientation=orientation,
    )


@pytest.mark.parametrize(
    ("inlet", "outlet", "expected_kinds"),
    [
        (
            water_steam_props_iapws97(T=350.0, p=P),
            water_steam_props_iapws97(T=400.0, p=P),
            (WaterEvaporatorZoneKind.PREHEAT,),
        ),
        (
            water_steam_props_iapws97(T=350.0, p=P),
            water_steam_props_iapws97(p=P, x=0.5),
            (WaterEvaporatorZoneKind.PREHEAT, WaterEvaporatorZoneKind.EVAPORATION),
        ),
        (
            water_steam_props_iapws97(p=P, x=0.0),
            water_steam_props_iapws97(p=P, x=0.5),
            (WaterEvaporatorZoneKind.EVAPORATION,),
        ),
        (
            water_steam_props_iapws97(p=P, x=0.2),
            water_steam_props_iapws97(p=P, x=0.8),
            (WaterEvaporatorZoneKind.EVAPORATION,),
        ),
        (
            water_steam_props_iapws97(p=P, x=0.2),
            water_steam_props_iapws97(p=P, x=1.0),
            (WaterEvaporatorZoneKind.EVAPORATION,),
        ),
        (
            water_steam_props_iapws97(T=350.0, p=P),
            water_steam_props_iapws97(p=P, x=1.0),
            (WaterEvaporatorZoneKind.PREHEAT, WaterEvaporatorZoneKind.EVAPORATION),
        ),
        (
            water_steam_props_iapws97(T=350.0, p=P),
            water_steam_props_iapws97(T=520.0, p=P),
            (
                WaterEvaporatorZoneKind.PREHEAT,
                WaterEvaporatorZoneKind.EVAPORATION,
                WaterEvaporatorZoneKind.SUPERHEAT,
            ),
        ),
        (
            water_steam_props_iapws97(p=P, x=1.0e-7),
            water_steam_props_iapws97(p=P, x=0.1),
            (WaterEvaporatorZoneKind.EVAPORATION,),
        ),
        (
            water_steam_props_iapws97(p=P, x=0.9),
            water_steam_props_iapws97(p=P, x=1.0 - 1.0e-7),
            (WaterEvaporatorZoneKind.EVAPORATION,),
        ),
    ],
)
def test_rating_supports_physical_heating_transitions(inlet, outlet, expected_kinds):
    result = _rate(inlet, outlet)
    assert tuple(zone.kind for zone in result.zones) == expected_kinds
    assert result.state_out.h == pytest.approx(outlet.h, rel=2e-12)
    assert all(zone.Q > 0.0 and zone.area > 0.0 and zone.UA > 0.0 for zone in result.zones)
    assert all(math.isfinite(zone.alpha_inside) for zone in result.zones)


def test_zone_energy_area_and_ua_balances_are_exact():
    inlet = water_steam_props_iapws97(T=350.0, p=P)
    outlet = water_steam_props_iapws97(T=520.0, p=P)
    result = _rate(inlet, outlet)
    assert result.Q_total == pytest.approx(result.mass_flow_water * (outlet.h - inlet.h))
    assert result.Q_total == pytest.approx(sum(zone.Q for zone in result.zones))
    assert result.A_total == pytest.approx(sum(zone.area for zone in result.zones))
    assert result.UA_total == pytest.approx(sum(zone.U * zone.area for zone in result.zones))
    assert result.UA_total == pytest.approx(result.U_equivalent * result.A_total)
    assert result.Q_total == pytest.approx(
        result.Q_preheat + result.Q_evaporation + result.Q_superheat
    )


def test_equivalent_inside_alpha_reconstructs_outer_basis_resistance():
    hx = _hx()
    result = _rate(
        water_steam_props_iapws97(T=350.0, p=P),
        water_steam_props_iapws97(T=520.0, p=P),
        hx=hx,
    )
    tube = hx.bundle.tube
    reconstructed_u = 1.0 / (
        tube.D_o / (tube.D_i * result.inside_alpha_equivalent)
        + tube.D_o * math.log(tube.D_o / tube.D_i) / (2.0 * tube.wall_k)
        + 1.0 / result.outside_alpha
    )
    assert reconstructed_u == pytest.approx(result.U_equivalent, rel=2e-12)
    assert result.inside_alpha_equivalent != pytest.approx(
        result.inside_alpha_area_weighted
    )


@pytest.mark.parametrize("quality_in, quality_out", [(0.0, 0.5), (0.2, 0.8), (0.7, 1.0)])
def test_boiling_mass_and_self_consistent_inside_area_heat_flux(quality_in, quality_out):
    hx = _hx()
    result = _rate(
        water_steam_props_iapws97(p=P, x=quality_in),
        water_steam_props_iapws97(p=P, x=quality_out),
        hx=hx,
    )
    assert result.m_dot_evaporated == pytest.approx(quality_out - quality_in)
    assert result.heat_flux_converged is True
    assert result.heat_flux_iterations > 0
    assert math.isfinite(result.heat_flux_residual)
    assert result.heat_flux_residual <= 1.0e-8
    inside_area = result.A_evaporation * hx.bundle.tube.D_i / hx.bundle.tube.D_o
    assert result.heat_flux_inner_evaporation == pytest.approx(
        result.Q_evaporation / inside_area, rel=2.0e-8
    )
    assert result.heat_flux_outer_evaporation == pytest.approx(
        result.Q_evaporation / result.A_evaporation, rel=2.0e-12
    )
    assert result.two_phase_pressure_drop_supported is False
    assert result.two_phase_pressure_drop_status == "not_supported"


def test_single_phase_case_does_not_require_orientation_or_disable_hydraulics():
    result = _rate(
        water_steam_props_iapws97(T=350.0, p=P),
        water_steam_props_iapws97(T=400.0, p=P),
        hx=_hx(orientation=None),
        orientation=None,
    )
    assert result.Q_evaporation == 0.0
    assert result.two_phase_pressure_drop_supported is True
    assert result.two_phase_pressure_drop_status == "not_applicable_single_phase"


@pytest.mark.parametrize(
    "orientation",
    [None, TubeOrientation.VERTICAL_DOWNWARD, TubeOrientation.DOWNWARD_INCLINED_15_PLUS],
)
def test_active_boiling_requires_a_supported_explicit_orientation(orientation):
    with pytest.raises(ValueError, match="orientation"):
        _rate(
            water_steam_props_iapws97(p=P, x=0.0),
            water_steam_props_iapws97(p=P, x=0.5),
            hx=_hx(orientation=orientation),
            orientation=orientation,
        )


def test_horizontal_boiling_is_supported():
    result = _rate(
        water_steam_props_iapws97(p=P, x=0.0),
        water_steam_props_iapws97(p=P, x=0.5),
        hx=_hx(orientation=TubeOrientation.HORIZONTAL),
        orientation=TubeOrientation.HORIZONTAL,
    )
    assert result.Q_evaporation > 0.0


@pytest.mark.parametrize(
    ("rows", "expected_kinds"),
    [
        (2, (WaterEvaporatorZoneKind.PREHEAT,)),
        (10, (WaterEvaporatorZoneKind.PREHEAT, WaterEvaporatorZoneKind.EVAPORATION)),
        (
            50,
            (
                WaterEvaporatorZoneKind.PREHEAT,
                WaterEvaporatorZoneKind.EVAPORATION,
                WaterEvaporatorZoneKind.SUPERHEAT,
            ),
        ),
    ],
)
def test_simulation_allocates_available_area_across_ordered_zones(rows, expected_kinds):
    hx = _hx(n_rows=rows)
    result = solve_water_evaporator(
        hx,
        inlet_state=water_steam_props_iapws97(T=350.0, p=P),
        mass_flow_water=1.0,
        outside_provider=OUTSIDE,
        mass_flow_outside=30.0,
        T_in_outside=700.0,
        p_outside=101325.0,
        orientation=TubeOrientation.VERTICAL_UPWARD,
    )
    assert result.converged is True
    assert tuple(zone.kind for zone in result.zones) == expected_kinds
    assert result.A_total == pytest.approx(hx.bundle.total_outer_area, rel=1e-8)
    assert result.Q_total == pytest.approx(
        result.mass_flow_water * (result.state_out.h - result.state_in.h)
    )
    assert result.root_iterations > 0


def test_simulation_requires_orientation_only_if_available_area_reaches_boiling():
    liquid = solve_water_evaporator(
        _hx(n_rows=2, orientation=None),
        inlet_state=water_steam_props_iapws97(T=350.0, p=P),
        mass_flow_water=1.0,
        outside_provider=OUTSIDE,
        mass_flow_outside=30.0,
        T_in_outside=700.0,
        p_outside=101325.0,
        orientation=None,
    )
    assert tuple(zone.kind for zone in liquid.zones) == (WaterEvaporatorZoneKind.PREHEAT,)
    with pytest.raises(ValueError, match="orientation"):
        solve_water_evaporator(
            _hx(n_rows=10, orientation=None),
            inlet_state=water_steam_props_iapws97(T=350.0, p=P),
            mass_flow_water=1.0,
            outside_provider=OUTSIDE,
            mass_flow_outside=30.0,
            T_in_outside=700.0,
            p_outside=101325.0,
            orientation=None,
        )


def test_reverse_direction_is_routed_away_from_evaporator_and_condenser_still_works():
    hot = water_steam_props_iapws97(p=P, x=0.8)
    cold = water_steam_props_iapws97(p=P, x=0.2)
    with pytest.raises(WaterCondensationRequiredError):
        _rate(hot, cold)
    condensed = rate_steam_heater(
        _hx(orientation=TubeOrientation.VERTICAL_DOWNWARD),
        inlet_state=hot,
        outlet_state=cold,
        mass_flow_steam=1.0,
        outside_provider=OUTSIDE,
        mass_flow_outside=30.0,
        T_in_outside=300.0,
        p_outside=101325.0,
        orientation=SteamTubeOrientation.VERTICAL_DOWNWARD,
    )
    assert condensed.Q_condensation > 0.0


def test_low_mass_flux_remains_finite_and_reports_applicability():
    result = _rate(
        water_steam_props_iapws97(p=P, x=0.2),
        water_steam_props_iapws97(p=P, x=0.8),
        mass_flow_water=0.05,
    )
    assert result.mass_flux < 200.0
    assert all(math.isfinite(value) and value > 0.0 for value in (
        result.Q_total, result.A_total, result.UA_total,
        result.inside_alpha_equivalent, result.zone_alpha_evaporation,
    ))
    assert result.warnings


class _TemperatureDependentOutsideProvider:
    def __init__(self):
        self.calls = []

    def at(self, T, p):
        self.calls.append((T, p))
        return FluidTransportProperties(
            rho=1.2 * 500.0 / T,
            mu=1.4e-5 + 1.0e-8 * T,
            k=0.018 + 2.0e-5 * T,
            cp=900.0 + 0.25 * T,
        )


def test_current_outside_properties_are_used_for_the_trial_duty():
    provider = _TemperatureDependentOutsideProvider()
    result = solve_water_evaporator(
        _hx(n_rows=10),
        inlet_state=water_steam_props_iapws97(T=350.0, p=P),
        mass_flow_water=1.0,
        outside_provider=provider,
        mass_flow_outside=30.0,
        T_in_outside=700.0,
        p_outside=101325.0,
        orientation=TubeOrientation.VERTICAL_UPWARD,
    )
    T_mean = 0.5 * (700.0 + result.T_out_outside)
    expected = provider.at(T_mean, 101325.0)
    assert result.outside_props_mean == expected
    assert result.outside_evaluation.T_mean == pytest.approx(T_mean)
    assert len({round(T, 8) for T, _ in provider.calls}) > 10


def test_rating_duty_resolves_the_final_ph_state_without_effective_cp():
    inlet = water_steam_props_iapws97(T=350.0, p=P)
    target = water_steam_props_iapws97(p=P, x=0.65)
    duty = 0.4 * (target.h - inlet.h)
    result = rate_water_evaporator(
        _hx(), inlet_state=inlet, Q_total=duty, mass_flow_water=0.4,
        outside_provider=OUTSIDE, mass_flow_outside=30.0,
        T_in_outside=700.0, p_outside=101325.0,
        orientation=TubeOrientation.VERTICAL_UPWARD,
    )
    assert result.state_out.h == pytest.approx(target.h)
    assert result.state_out.quality == pytest.approx(0.65)
    assert result.Q_total == pytest.approx(duty)


def test_solver_caches_repeated_iapws_and_outside_states():
    result = _rate(
        water_steam_props_iapws97(T=350.0, p=P),
        water_steam_props_iapws97(T=520.0, p=P),
    )
    assert result.cache_hits > 0
    assert 0 < result.property_evaluations < 20


def test_public_result_extension_keeps_condensation_defaults_and_convenience_semantics():
    from core.phase_change.types import (
        PhaseChangeDirection,
        PhaseChangeMode,
        WaterSteamPhaseChangeResult,
    )

    state = water_steam_props_iapws97(T=350.0, p=P)
    values = WaterSteamPhaseChangeResult(
        side="inside", mode=PhaseChangeMode.AUTO,
        direction=PhaseChangeDirection.EVAPORATION, component="H2O",
        capable=True, possible=True, active=True, converged=True,
        method="water_evaporator", state_in=state, state_midpoint=state,
        state_out=state, phase_in=state.phase, phase_out=state.phase,
        T_in=state.T, T_out=state.T, Tsat=453.0, p=P,
        h_in=state.h, h_out=state.h, quality_in=0.0, quality_out=0.5,
        Q_desuperheat=0.0, Q_condensation=0.0, Q_subcooling=0.0,
        Q_total=30.0, A_desuperheat=0.0, A_condensation=0.0,
        A_subcooling=0.0, A_total=1.0, zone_fraction_desuperheat=0.0,
        zone_fraction_condensation=0.0, zone_fraction_subcooling=0.0,
        zone_alpha_desuperheat=None, zone_alpha_condensation=None,
        zone_alpha_subcooling=None, zone_U_desuperheat=None,
        zone_U_condensation=None, zone_U_subcooling=None,
        zone_UA_desuperheat=0.0, zone_UA_condensation=0.0,
        zone_UA_subcooling=0.0, UA_total=1.0, mass_flow_total=1.0,
        mass_flux=1.0, m_dot_condensate=0.0,
        two_phase_pressure_drop_supported=False,
        two_phase_pressure_drop_status="not_supported", iterations=1,
        root_iterations=0, property_evaluations=1,
        Q_preheat=10.0, Q_evaporation=20.0, Q_superheat=0.0,
        m_dot_evaporated=0.5,
    )
    assert values.Q_sensible == 10.0
    assert values.Q_latent == 20.0
    assert values.is_evaporating is True
    assert values.is_condensing is False
