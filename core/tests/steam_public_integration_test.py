import ast
import inspect
import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput, run_simulation
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change import warning_codes as WC
from core.phase_change.integration import MultiplePhaseChangeSidesError
from core.phase_change.steam_integration import (
    PhaseChangeDisabledButRequiredError,
    PureSteamOutsideNotSupportedError,
)
import core.phase_change.steam_integration as steam_integration
from core.phase_change.types import PhaseChangeMode, WaterSteamPhaseChangeResult
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import IAPWS97WaterSteamProvider, WaterSteamPhase


P = 1.0e6
ORIENTATION = TubeOrientation.VERTICAL_DOWNWARD
OUTSIDE_PROVIDER = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1005.0)
)


class TemperatureDependentOutsideProvider:
    def at(self, T, p):
        return FluidTransportProperties(
            rho=1.2 * 300.0 / T,
            mu=1.8e-5 * (T / 300.0) ** 0.7,
            k=0.026 * (T / 300.0) ** 0.8,
            cp=1005.0 + 0.25 * (T - 300.0),
        )


def _hx(n_rows=10, n_tubes_per_row=10, *, orientation=ORIENTATION):
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=BareTube(
                D_i=0.020, D_o=0.024, length_total=4.0,
                length_effective=4.0, wall_k=16.0,
                tube_orientation=orientation,
            ),
            n_rows=n_rows, n_tubes_per_row=n_tubes_per_row,
            pitch_transverse=0.04, pitch_longitudinal=0.04,
            layout="inline", n_passes_tube=1, flow_arrangement="crossflow",
        )
    )


def _outside_sim(m_dot=30.0):
    return HXSideInput(
        provider=OUTSIDE_PROVIDER, m_dot=m_dot,
        T_in=300.0, p=101325.0,
    )


def _inside_sim(*, m_dot=1.0, **state):
    return HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=m_dot, p=P,
        **state,
    )


@pytest.mark.parametrize(
    ("inside", "expected_phase"),
    [
        (_inside_sim(quality_in=1.0), WaterSteamPhase.TWO_PHASE),
        (_inside_sim(quality_in=0.8), WaterSteamPhase.TWO_PHASE),
        (_inside_sim(T_in=520.0), WaterSteamPhase.TWO_PHASE),
        (_inside_sim(quality_in=1.0, m_dot=0.3), WaterSteamPhase.SUBCOOLED_LIQUID),
        (_inside_sim(T_in=520.0, m_dot=0.25), WaterSteamPhase.SUBCOOLED_LIQUID),
    ],
)
def test_simulation_public_transitions_use_typed_steam_result(inside, expected_phase):
    result = _hx().simulate(inside, _outside_sim())
    steam = result.inside_phase_change
    assert isinstance(steam, WaterSteamPhaseChangeResult)
    assert steam.phase_out is expected_phase
    assert steam.h_out == pytest.approx(inside.h_in - result.q / inside.m_dot)
    assert steam.Q_total == pytest.approx(
        steam.Q_desuperheat + steam.Q_condensation + steam.Q_subcooling
    )
    assert steam.A_total == pytest.approx(_hx().bundle.total_outer_area, rel=2e-8)
    assert steam.UA_total == pytest.approx(
        steam.zone_UA_desuperheat + steam.zone_UA_condensation + steam.zone_UA_subcooling
    )
    assert result.UA == pytest.approx(steam.UA_total)
    assert result.inside_properties_inlet is steam.state_in
    assert result.inside_properties_outlet is steam.state_out
    assert result.inside_properties_outlet.quality == steam.quality_out
    assert steam.p == P
    assert result.tube_side_hydraulic is None
    assert result.tube_side_pressure_drop is None
    assert math.isnan(result.inside_dp_total)
    assert result.outside_tube_bank_hydraulic is not None
    assert math.isfinite(result.outside_dp_total)
    assert steam.two_phase_pressure_drop_status == "not_supported"
    assert WC.STEAM_TWO_PHASE_PRESSURE_DROP_NOT_SUPPORTED in {
        warning.code for warning in steam.warnings
    }


def test_superheated_steam_remaining_superheated_is_detected_without_condensation():
    result = _hx(n_rows=2, n_tubes_per_row=2, orientation=None).simulate(
        _inside_sim(T_in=600.0, m_dot=10.0), _outside_sim()
    )
    steam = result.inside_phase_change
    assert steam.phase_out is WaterSteamPhase.SUPERHEATED_VAPOR
    assert steam.active is False
    assert steam.Q_condensation == 0.0


def test_superheated_to_superheated_rating_does_not_require_orientation():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        T_in=600.0, T_out=550.0,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=30.0, T_in=300.0,
    )
    result = _hx(orientation=None).rate(inside, outside)
    assert result.inside_phase_change.active is False
    assert result.inside_phase_change.Q_condensation == 0.0


def test_disabled_mode_allows_a_same_phase_superheated_result():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=10.0, p=P,
        T_in=600.0,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )
    result = _hx(n_rows=2, n_tubes_per_row=2, orientation=None).simulate(
        inside, _outside_sim()
    )
    assert result.inside_phase_change.phase_out is WaterSteamPhase.SUPERHEATED_VAPOR
    assert result.inside_phase_change.active is False


def test_surface_margin_derates_public_steam_simulation():
    hx = _hx()
    baseline = hx.simulate(_inside_sim(quality_in=1.0), _outside_sim())
    derated = hx.simulate(
        _inside_sim(quality_in=1.0), _outside_sim(), surface_margin=0.25
    )
    assert derated.surface_margin == 0.25
    assert derated.q < baseline.q
    assert derated.Q_full == pytest.approx(baseline.q)
    assert derated.Q_derated == derated.q
    assert derated.inside_phase_change.A_total == pytest.approx(
        hx.bundle.total_outer_area / 1.25, rel=2e-8
    )


def test_outside_properties_and_alpha_follow_final_trial_temperature():
    provider = TemperatureDependentOutsideProvider()
    cold = HXSideInput(provider=provider, m_dot=30.0, T_in=280.0, p=101325.0)
    warm = HXSideInput(provider=provider, m_dot=30.0, T_in=340.0, p=101325.0)
    cold_result = _hx().simulate(_inside_sim(quality_in=1.0), cold)
    warm_result = _hx().simulate(_inside_sim(quality_in=1.0), warm)
    for result, outside in ((cold_result, cold), (warm_result, warm)):
        T_mean = 0.5 * (outside.T_in + result.T_out_outside)
        assert result.outside_props_mean == provider.at(T_mean, outside.p)
        assert result.outside_alfa_mean == pytest.approx(
            result.final_result.outside_side_thermal.alfa
        )
    assert cold_result.outside_alfa_mean != pytest.approx(warm_result.outside_alfa_mean)


def test_steam_adapter_has_no_fake_provider_or_private_wet_gas_imports():
    source = inspect.getsource(steam_integration)
    assert "ConstantPropertyProvider" not in source
    tree = ast.parse(source)
    private_imports = [
        name.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {
            "core.phase_change.integration",
            "core.phase_change.rating_integration",
        }
        for name in node.names
        if name.name.startswith("_")
    ]
    assert private_imports == []


def test_saturated_liquid_subcooling_does_not_require_orientation():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0, p=P,
        quality_in=0.0,
    )
    result = _hx(orientation=None).simulate(inside, _outside_sim())
    assert result.inside_phase_change.phase_in is WaterSteamPhase.SATURATED_LIQUID
    assert result.inside_phase_change.phase_out is WaterSteamPhase.SUBCOOLED_LIQUID
    assert result.inside_phase_change.Q_condensation == 0.0


@pytest.mark.parametrize(
    ("inlet_kwargs", "outlet_kwargs", "expected_phase"),
    [
        ({"quality_in": 1.0}, {"quality_out": 0.4}, WaterSteamPhase.TWO_PHASE),
        ({"quality_in": 1.0}, {"quality_out": 0.0}, WaterSteamPhase.SATURATED_LIQUID),
        ({"quality_in": 1.0}, {"T_out": 350.0}, WaterSteamPhase.SUBCOOLED_LIQUID),
        ({"quality_in": 0.8}, {"quality_out": 0.2}, WaterSteamPhase.TWO_PHASE),
        ({"T_in": 520.0}, {"quality_out": 0.5}, WaterSteamPhase.TWO_PHASE),
        ({"T_in": 520.0}, {"T_out": 350.0}, WaterSteamPhase.SUBCOOLED_LIQUID),
    ],
)
def test_rating_reuses_same_zone_physics(inlet_kwargs, outlet_kwargs, expected_phase):
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        **inlet_kwargs, **outlet_kwargs,
    )
    # The generous outside flow keeps every parallel branch away from its
    # thermal pinch in the T_out=350.0 fixtures above.
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=100.0, T_in=300.0,
    )
    result = _hx().rate(inside, outside)
    steam = result.inside_phase_change
    assert isinstance(steam, WaterSteamPhaseChangeResult)
    assert steam.phase_out is expected_phase
    assert result.UA_required == pytest.approx(steam.UA_total)
    assert result.UA_actual == pytest.approx(
        result.UA_required * result.A_o / result.A_required
    )
    assert result.inside_properties_outlet.phase is expected_phase
    assert math.isnan(result.inside_dp_total)


def test_steam_rating_rejects_effectiveness_only_target():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        quality_in=1.0,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=30.0, T_in=300.0,
    )
    with pytest.raises(ValueError, match="effectiveness-only"):
        _hx().rate(inside, outside, effectiveness=0.5)


def test_steam_rating_rejects_inconsistent_over_specified_duties():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        quality_in=1.0, quality_out=0.5,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=30.0, T_in=300.0,
    )
    with pytest.raises(ValueError, match="Over-specified"):
        _hx().rate(inside, outside, Q=1.0)


# ---------------------------------------------------------------------------
# v0.7.1 -- derived pure-steam mass flow (inside.m_dot=None) in Rating
# ---------------------------------------------------------------------------
def _outside_program(m_dot=30.0, T_in=300.0, T_out=340.0):
    return BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=m_dot, T_in=T_in, T_out=T_out,
    )


def _expected_outside_duty(outside: BalanceSideSpec) -> float:
    """Independent reference duty using the documented Rating cp_mean convention."""
    props = outside.provider.at(T=0.5 * (outside.T_in + outside.T_out), p=outside.p)
    return outside.m_dot * props.cp * (outside.T_out - outside.T_in)


def test_unknown_steam_mass_flow_rating_explicit_full_condensation():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=None,
        T_in=600.0, quality_out=0.0,
    )
    outside = _outside_program()
    result = _hx().rate(inside, outside)

    Q_outside = _expected_outside_duty(outside)
    h_in = inside.water_steam_state.h
    h_f = IAPWS97WaterSteamProvider().state(x=0.0, p=P).h
    expected_m_dot = Q_outside / (h_in - h_f)

    assert result.Q_required == pytest.approx(Q_outside)
    assert result.closed_balance.inside.m_dot == pytest.approx(expected_m_dot)
    assert result.inside_phase_change.mass_flow_total == pytest.approx(expected_m_dot)
    assert result.inside_phase_change.quality_out == pytest.approx(0.0, abs=1e-9)
    assert inside.m_dot is None


def test_unknown_steam_mass_flow_rating_defaults_to_saturated_liquid():
    """No T_out/h_out/quality_out on the steam side -> default x_out=0."""
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=None, T_in=600.0,
    )
    outside = _outside_program()
    result = _hx().rate(inside, outside)

    Q_outside = _expected_outside_duty(outside)
    h_in = inside.water_steam_state.h
    h_f = IAPWS97WaterSteamProvider().state(x=0.0, p=P).h
    expected_m_dot = Q_outside / (h_in - h_f)

    assert result.inside_phase_change.quality_out == pytest.approx(0.0, abs=1e-9)
    assert result.closed_balance.inside.m_dot == pytest.approx(expected_m_dot)
    assert result.inside_phase_change.mass_flow_total == pytest.approx(expected_m_dot)
    # The superheated inlet is never force-assigned quality_in=1; desuperheating
    # is included in the zone solution instead.
    assert inside.quality_in is None
    assert inside.m_dot is None


def test_unknown_steam_mass_flow_rating_partial_condensation():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=None,
        T_in=600.0, quality_out=0.2,
    )
    outside = _outside_program()
    result = _hx().rate(inside, outside)

    Q_outside = _expected_outside_duty(outside)
    h_in = inside.water_steam_state.h
    h_target = IAPWS97WaterSteamProvider().state(x=0.2, p=P).h
    expected_m_dot = Q_outside / (h_in - h_target)

    assert result.closed_balance.inside.m_dot == pytest.approx(expected_m_dot)
    assert result.inside_phase_change.quality_out == pytest.approx(0.2, abs=1e-6)


def test_unknown_steam_mass_flow_rating_explicit_h_out():
    h_target = IAPWS97WaterSteamProvider().state(T=400.0, p=P).h
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=None,
        T_in=600.0, h_out=h_target,
    )
    outside = _outside_program()
    result = _hx().rate(inside, outside)

    Q_outside = _expected_outside_duty(outside)
    h_in = inside.water_steam_state.h
    expected_m_dot = Q_outside / (h_in - h_target)

    assert result.closed_balance.inside.m_dot == pytest.approx(expected_m_dot)
    assert result.inside_phase_change.h_out == pytest.approx(h_target)


def test_unknown_steam_mass_flow_rating_uses_explicit_Q():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=None,
        T_in=600.0, quality_out=0.0,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=30.0, T_in=300.0,
    )
    h_in = inside.water_steam_state.h
    h_f = IAPWS97WaterSteamProvider().state(x=0.0, p=P).h
    Q = 5.0e5
    expected_m_dot = Q / (h_in - h_f)

    result = _hx().rate(inside, outside, Q=Q)
    assert result.closed_balance.inside.m_dot == pytest.approx(expected_m_dot)
    assert result.Q_required == pytest.approx(Q)


def test_unknown_steam_mass_flow_rating_over_specified_duty_consistency():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=None,
        T_in=600.0, quality_out=0.0,
    )
    outside = _outside_program()
    Q_consistent = _expected_outside_duty(outside)

    result = _hx().rate(inside, outside, Q=Q_consistent)
    assert result.Q_required == pytest.approx(Q_consistent)

    with pytest.raises(ValueError, match="Over-specified"):
        _hx().rate(inside, outside, Q=Q_consistent * 2.0)


def test_unknown_steam_mass_flow_rating_without_independent_duty_is_rejected():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=None,
        T_in=600.0, quality_out=0.0,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=30.0, T_in=300.0,
    )
    with pytest.raises(ValueError, match="requires explicit Q"):
        _hx().rate(inside, outside)


def test_unknown_steam_mass_flow_rating_rejects_non_condensing_target():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=None,
        T_in=600.0, T_out=650.0,
    )
    outside = _outside_program()
    with pytest.raises(ValueError, match="lower specific enthalpy"):
        _hx().rate(inside, outside)


def test_unknown_steam_mass_flow_rating_requires_known_outside_mass_flow():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=None,
        T_in=600.0, quality_out=0.0,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=None, T_in=300.0, T_out=340.0,
    )
    with pytest.raises(ValueError, match="unknown steam mass flow requires a known"):
        _hx().rate(inside, outside, Q=1.0e6)


def test_known_steam_mass_flow_rating_does_not_default_outlet_to_saturated_liquid():
    """Regression: known m_dot + explicit Q must NOT acquire the x_out=0 default."""
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0, T_in=600.0,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=30.0, T_in=300.0,
    )
    result = _hx().rate(inside, outside, Q=2.0e5)
    assert result.inside_phase_change.quality_out is None
    assert result.inside_phase_change.phase_out is WaterSteamPhase.SUPERHEATED_VAPOR
    assert result.closed_balance.inside.m_dot == pytest.approx(1.0)


def test_unknown_steam_mass_flow_rating_include_simulation_uses_resolved_flow():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=None,
        T_in=600.0, quality_out=0.0,
    )
    outside = _outside_program()
    result = _hx().rate(inside, outside, include_simulation=True)

    resolved_m_dot = result.closed_balance.inside.m_dot
    assert result.simulation is not None
    assert result.Q_achievable == pytest.approx(result.simulation.q)

    direct_simulation = _hx().simulate(
        HXSideInput(
            provider=IAPWS97WaterSteamProvider(), m_dot=resolved_m_dot,
            T_in=600.0, p=P,
        ),
        HXSideInput(provider=OUTSIDE_PROVIDER, m_dot=30.0, T_in=300.0, p=101325.0),
    )
    assert result.simulation.q == pytest.approx(direct_simulation.q)
    assert inside.m_dot is None


def test_public_steam_result_exposes_equivalent_and_area_weighted_alpha():
    result = _hx().simulate(_inside_sim(T_in=520.0, m_dot=0.25), _outside_sim())
    steam = result.inside_phase_change
    assert steam.zone_alpha_condensation > 0.0
    assert steam.inside_alpha_equivalent > 0.0
    assert steam.inside_alpha_area_weighted > 0.0
    assert steam.inside_alpha_equivalent != pytest.approx(
        steam.inside_alpha_area_weighted
    )
    assert result.inside_alfa_mean == steam.inside_alpha_equivalent
    assert result.thermal_state.alfa_i == steam.inside_alpha_equivalent
    assert result.final_result.tube_side_thermal.alfa == steam.inside_alpha_equivalent
    assert result.thermal_state.diagnostics.inside_alfa_base == steam.inside_alpha_equivalent
    assert result.thermal_state.diagnostics.inside_alfa_corrected == steam.inside_alpha_equivalent
    assert result.UA == pytest.approx(steam.UA_total)
    assert steam.zone_alpha_condensation != pytest.approx(steam.inside_alpha_equivalent)
    assert not hasattr(steam, "solution")
    assert not hasattr(steam, "runtime_s")


def test_public_steam_rating_uses_equivalent_alpha_without_internal_leakage():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        T_in=520.0, T_out=350.0,
    )
    # The generous outside flow keeps every parallel branch away from its
    # thermal pinch in this deeply subcooled fixture.
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=100.0, T_in=300.0,
    )
    result = _hx().rate(inside, outside)
    steam = result.inside_phase_change
    assert result.alfa_i == steam.inside_alpha_equivalent
    assert result.thermal_state.alfa_i == steam.inside_alpha_equivalent
    assert result.final_result.tube_side_thermal.alfa == steam.inside_alpha_equivalent
    assert steam.inside_alpha_area_weighted != pytest.approx(result.alfa_i)
    assert steam.zone_alpha_condensation == pytest.approx(8826.095321215491)
    assert not hasattr(steam, "solution")
    assert not hasattr(steam, "runtime_s")


def test_public_steam_result_exposes_raw_zone_driving_force_diagnostics():
    from core.phase_change.steam_heater import SteamHeaterZoneKind

    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        T_in=520.0, T_out=350.0,
    )
    outside = BalanceSideSpec(
        provider=OUTSIDE_PROVIDER, p=101325.0, m_dot=100.0, T_in=300.0,
    )
    hx = _hx()
    result = hx.rate(inside, outside)
    steam = result.inside_phase_change
    cp_outside = OUTSIDE_PROVIDER.props.cp
    frontal_area = hx.bundle.frontal_flow_area
    face_mass_flux = outside.m_dot / frontal_area
    assert tuple(zone.kind for zone in steam.zones) == (
        SteamHeaterZoneKind.SUPERHEAT,
        SteamHeaterZoneKind.CONDENSATION,
        SteamHeaterZoneKind.SUBCOOLING,
    )
    assert steam.zone_allocation_method == "parallel_by_geometry"
    assert steam.zone_allocation_converged is True
    assert 1 <= steam.zone_allocation_iterations <= 80
    assert 0.0 <= steam.zone_allocation_residual <= 1.0e-8
    assert steam.sum_zone_area_fraction == pytest.approx(1.0, abs=2.0e-12)
    assert steam.sum_zone_air_mass_flow == pytest.approx(outside.m_dot)
    assert steam.Q_zone_sum == pytest.approx(steam.Q_total)
    assert steam.mixed_outside_T_out == pytest.approx(
        result.closed_balance.outside.T_out
    )
    assert abs(steam.mixed_air_energy_residual) <= max(
        1.0e-6, 1.0e-12 * steam.Q_total
    )
    for zone in steam.zones:
        assert math.isfinite(zone.outside_T_in)
        assert math.isfinite(zone.outside_T_out)
        assert math.isfinite(zone.delta_T_terminal_in)
        assert math.isfinite(zone.delta_T_terminal_out)
        assert zone.outside_T_in == pytest.approx(outside.T_in)
        assert zone.area_fraction == pytest.approx(zone.area / steam.A_total)
        assert zone.tube_length_fraction == pytest.approx(zone.area_fraction)
        assert zone.outside_mass_flow == pytest.approx(
            outside.m_dot * zone.outside_mass_flow_fraction
        )
        assert zone.outside_mass_flow_fraction == pytest.approx(
            zone.area_fraction, abs=2.0e-8
        )
        assert zone.outside_frontal_area == pytest.approx(
            frontal_area * zone.outside_mass_flow_fraction
        )
        assert zone.outside_face_mass_flux == pytest.approx(face_mass_flux)
        assert zone.outside_mass_flow / zone.outside_frontal_area == pytest.approx(
            face_mass_flux
        )
        assert zone.outside_velocity == pytest.approx(
            result.final_result.outside_side_thermal.v
        )
        assert zone.Q == pytest.approx(
            zone.outside_mass_flow
            * cp_outside
            * (zone.outside_T_out - zone.outside_T_in),
            rel=2.0e-12,
            abs=1.0e-6,
        )
        assert zone.driving_force_method in (
            "isothermal_lmtd", "epsilon_ntu_crossflow",
        )
    allocation_residual = max(
        abs(zone.area_fraction - zone.outside_mass_flow_fraction)
        for zone in steam.zones
    )
    assert steam.zone_allocation_residual == pytest.approx(
        allocation_residual, rel=0.0, abs=2.0e-15
    )
    mixed_temperature = sum(
        zone.outside_mass_flow * zone.outside_T_out for zone in steam.zones
    ) / outside.m_dot
    assert steam.mixed_outside_T_out == pytest.approx(mixed_temperature, rel=2.0e-12)
    assert steam.Q_total == pytest.approx(
        outside.m_dot
        * cp_outside
        * (steam.mixed_outside_T_out - outside.T_in),
        rel=2.0e-12,
        abs=1.0e-6,
    )
    assert sum(zone.UA for zone in steam.zones) == pytest.approx(steam.UA_total, rel=1.0e-9)
    assert result.EMTD == pytest.approx(steam.Q_total / steam.UA_total, rel=1.0e-9)


def test_multizone_wall_diagnostics_use_equivalent_resistance_and_remain_0d():
    hx = _hx()
    result = hx.simulate(_inside_sim(T_in=520.0, m_dot=0.25), _outside_sim())
    steam = result.inside_phase_change
    tube = hx.bundle.tube
    envelope = result.wall_temperature_envelope

    values = (
        envelope.inside_min,
        envelope.inside_max,
        envelope.outside_min,
        envelope.outside_max,
        envelope.inside_mean,
        envelope.outside_mean,
        result.thermal_state.inside_wall_temperature,
        result.thermal_state.outside_wall_temperature,
    )
    assert all(math.isfinite(value) for value in values)
    assert result.thermal_state.alfa_i == steam.inside_alpha_equivalent
    assert result.thermal_state.diagnostics.inside_Nu_base is None
    assert result.thermal_state.diagnostics.inside_Nu_corrected is None

    wall_resistance = tube.D_o * math.log(tube.D_o / tube.D_i) / (2.0 * tube.wall_k)
    for probe in envelope.probes:
        heat_flux = result.U_mean * (
            probe.inside_bulk_temperature - probe.outside_bulk_temperature
        )
        assert probe.alfa_i == steam.inside_alpha_equivalent
        assert probe.inside_nusselt is None
        assert probe.inside_bulk_temperature - probe.inside_wall_temperature == pytest.approx(
            heat_flux * tube.D_o / (tube.D_i * steam.inside_alpha_equivalent)
        )
        assert probe.inside_wall_temperature - probe.outside_wall_temperature == pytest.approx(
            heat_flux * wall_resistance
        )
        assert probe.outside_wall_temperature - probe.outside_bulk_temperature == pytest.approx(
            heat_flux / result.outside_alfa_mean
        )

    warning = next(
        item for item in envelope.warnings
        if item.code == "wall_temperature_envelope_0d_estimate"
    )
    assert "0D" in warning.message
    assert "not local extrema" in warning.message


def test_disabled_mode_rejects_required_saturation_crossing():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0, p=P,
        quality_in=1.0,
        phase_change_mode=PhaseChangeMode.DISABLED,
    )
    with pytest.raises(PhaseChangeDisabledButRequiredError) as caught:
        _hx().simulate(inside, _outside_sim())
    assert caught.value.warning_code == WC.PHASE_CHANGE_DISABLED_BUT_REQUIRED


def test_missing_geometry_orientation_is_rejected_only_when_condensation_is_active():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0, p=P,
        quality_in=1.0,
    )
    with pytest.raises(ValueError, match="tube_orientation on BareTube"):
        _hx(orientation=None).simulate(inside, _outside_sim())


def test_pure_steam_outside_is_controlled_unsupported_scope():
    outside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0, p=P,
        quality_in=1.0,
    )
    inside = HXSideInput(
        provider=OUTSIDE_PROVIDER, m_dot=5.0, T_in=300.0, p=101325.0,
    )
    with pytest.raises(PureSteamOutsideNotSupportedError) as caught:
        _hx().simulate(inside, outside)
    assert caught.value.warning_code == WC.PURE_STEAM_OUTSIDE_NOT_SUPPORTED


def test_rating_rejects_second_auto_phase_changing_wet_gas_side():
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=P, m_dot=1.0,
        quality_in=1.0, quality_out=0.5,
    )
    wet = GasMixturePropertyProvider(
        GasMixtureSpec(
            components={"N2": 0.75, "O2": 0.15, "CO2": 0.02, "H2O": 0.08},
            basis="mole",
        )
    )
    outside = BalanceSideSpec(
        provider=wet, p=101325.0, m_dot=10.0, T_in=340.0,
    )
    with pytest.raises(MultiplePhaseChangeSidesError):
        _hx().rate(inside, outside)


def test_iapws_provider_is_reported_as_pure_steam_capable():
    capability = detect_phase_change_capability(IAPWS97WaterSteamProvider())
    assert capability.capable
    assert capability.provider_kind == "pure_water_steam"
    assert capability.W_in is None


def test_subcooled_sensible_water_keeps_existing_simulation_path():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=5.0,
        T_in=300.0, p=P,
    )
    outside = HXSideInput(
        provider=OUTSIDE_PROVIDER, m_dot=20.0,
        T_in=500.0, p=101325.0,
    )
    expected = run_simulation(_hx(), inside, outside)
    actual = _hx().simulate(inside, outside)
    assert actual.q == expected.q
    assert actual.UA == expected.UA
    assert actual.inside_dp_total == expected.inside_dp_total


def test_public_water_heating_routes_to_evaporation_solver():
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(), m_dot=1.0,
        T_in=300.0, p=P,
    )
    outside = HXSideInput(
        provider=OUTSIDE_PROVIDER, m_dot=30.0,
        T_in=700.0, p=101325.0,
    )
    result = _hx(
        n_rows=20,
        n_tubes_per_row=20,
        orientation=TubeOrientation.VERTICAL_UPWARD,
    ).simulate(inside, outside)
    assert result.inside_phase_change.is_evaporating is True
    assert result.inside_phase_change.Q_evaporation > 0.0


def test_public_low_g_case_keeps_finite_condensation_zone_htc():
    hx = _hx(n_rows=20, n_tubes_per_row=20)
    result = hx.simulate(
        _inside_sim(quality_in=1.0, m_dot=0.5),
        _outside_sim(m_dot=20.0),
    )
    steam = result.inside_phase_change
    assert steam.mass_flux < 5.0
    assert steam.zone_alpha_condensation > 1000.0
    assert all(math.isfinite(value) for value in (result.q, result.UA, steam.A_total))
