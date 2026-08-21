"""Focused active-wet CircularFinnedTube Rating integration tests."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, CircularFinnedTube
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec, close_heat_balance
from core.models.rating import run_rating
from core.phase_change import warning_codes as WC
from core.phase_change.types import PhaseChangeMode
from core.phase_change.wet_finned_surface import WetFinnedSurfaceResult
from core.pressure_drop.finned_tube_pressure_drop import (
    RobinsonBriggs1966Provider,
)
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import (
    GasMixturePropertyProvider,
    GasMixtureSpec,
)


P = 101_325.0


def _exchanger() -> BareTubeHeatExchanger:
    core_tube = BareTube(
        D_i=0.022,
        D_o=0.025,
        length_total=2.8,
        length_effective=2.8,
        wall_k=50.0,
    )
    tube = CircularFinnedTube(
        core_tube=core_tube,
        fin_k=200.0,
        D_fin=0.050,
        D_root=0.028,
        fin_thickness_root=0.0005,
        fin_thickness_tip=0.0003,
        fin_pitch=0.0024,
        fin_contact_efficiency=0.95,
    )
    pitch_transverse = 0.060
    bundle = TubeBundle(
        tube=tube,
        n_rows=6,
        n_tubes_per_row=8,
        pitch_transverse=pitch_transverse,
        pitch_longitudinal=math.sqrt(3.0) * pitch_transverse / 2.0,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement="counterflow",
    )
    return BareTubeHeatExchanger(bundle)


def _inside_provider() -> ConstantPropertyProvider:
    return ConstantPropertyProvider(
        FluidTransportProperties(
            rho=900.0,
            mu=5.0e-4,
            k=0.60,
            cp=4000.0,
        )
    )


def _wet_provider() -> GasMixturePropertyProvider:
    return GasMixturePropertyProvider(
        GasMixtureSpec(
            components={
                "N2": 0.65,
                "O2": 0.10,
                "CO2": 0.08,
                "H2O": 0.17,
            },
            basis="mole",
        )
    )


def _dry_provider() -> GasMixturePropertyProvider:
    return GasMixturePropertyProvider(
        GasMixtureSpec(
            components={"N2": 0.79, "O2": 0.21},
            basis="mole",
        )
    )


def _rating_sides(*, wet_outside: bool) -> tuple[BalanceSideSpec, BalanceSideSpec]:
    inside = BalanceSideSpec(
        provider=_inside_provider(),
        p=P,
        m_dot=8.0,
        T_in=290.0,
        T_out=311.0,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    outside = BalanceSideSpec(
        provider=_wet_provider() if wet_outside else _dry_provider(),
        p=P,
        m_dot=6.0,
        T_in=420.0,
        T_out=333.0 if wet_outside else 350.0,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    return inside, outside


class _TaggedPressureDropProvider:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        result = RobinsonBriggs1966Provider().evaluate(request)
        return replace(
            result,
            metadata=replace(
                result.metadata,
                method="wet_rating_tagged_robinson_briggs",
            ),
        )


def test_active_auto_rating_uses_converged_wet_finned_surface() -> None:
    hx = _exchanger()
    inside, outside = _rating_sides(wet_outside=True)
    pressure_provider = _TaggedPressureDropProvider()

    result = hx.rate(
        inside,
        outside,
        finned_pressure_drop_provider=pressure_provider,
        include_simulation=True,
    )

    phase_change = result.outside_phase_change
    wet = result.wet_finned_surface
    assert phase_change is not None and phase_change.active is True
    assert phase_change.converged is True
    assert phase_change.method == "outside_condensation_rating_wet_annular_fin_fvm"
    assert isinstance(wet, WetFinnedSurfaceResult)
    assert wet is phase_change.wet_finned_surface
    assert wet is result.thermal_state.finned_tube_diagnostics.wet_surface
    assert wet is result.final_result.finned_tube_diagnostics.wet_surface

    assert wet.m_dot_condensate > 0.0
    assert wet.Q_sensible > 0.0
    assert wet.Q_latent > 0.0
    assert wet.Q_total == pytest.approx(
        wet.Q_sensible + wet.Q_latent, rel=1.0e-12
    )
    assert wet.Q_total == pytest.approx(phase_change.Q_total, rel=1.0e-12)
    assert wet.m_dot_condensate == pytest.approx(
        phase_change.m_dot_condensate, rel=1.0e-12
    )
    assert wet.Q_primary_total + wet.Q_fin_total == pytest.approx(
        wet.Q_total, rel=1.0e-12
    )
    assert wet.Q_primary_sensible + wet.Q_fin_sensible == pytest.approx(
        wet.Q_sensible, rel=1.0e-12
    )
    assert wet.Q_primary_latent + wet.Q_fin_latent == pytest.approx(
        wet.Q_latent, rel=1.0e-12
    )
    assert (
        wet.m_dot_condensate_primary + wet.m_dot_condensate_fin
        == pytest.approx(wet.m_dot_condensate, rel=1.0e-12)
    )
    assert abs(phase_change.mass_balance_error) < 1.0e-6
    assert abs(phase_change.energy_balance_error) < 1.0e-6
    assert abs(wet.mass_balance_error) < 1.0e-12
    assert abs(wet.energy_balance_error) < 1.0e-6
    assert wet.wet_area == pytest.approx(
        wet.wet_primary_area + wet.wet_fin_area, rel=1.0e-12
    )
    assert wet.wet_surface_fraction == pytest.approx(
        wet.wet_area / wet.outside_total_area, rel=1.0e-12
    )
    assert wet.wall_temperature_wet_mean == pytest.approx(
        phase_change.wall_temperature_wet_mean, rel=1.0e-12
    )
    exposed_temperatures = (
        wet.primary_surface_temperature,
        wet.fin_base_temperature,
        wet.fin_tip_temperature,
    )
    assert phase_change.wall_temperature_min == pytest.approx(
        min(exposed_temperatures)
    )
    assert phase_change.wall_temperature_max == pytest.approx(
        max(exposed_temperatures)
    )
    assert phase_change.wall_temperature_mean == pytest.approx(
        wet.outside_surface_temperature_area_mean
    )
    assert (
        phase_change.wall_temperature_min
        <= phase_change.wall_temperature_mean
        <= phase_change.wall_temperature_max
    )
    assert "rating_raw_surface_Q_total_W" in wet.residuals
    assert "rating_surface_Q_gap_W" in wet.residuals
    assert "rating_raw_surface_condensate_kg_s" in wet.residuals
    assert "rating_surface_condensate_gap_kg_s" in wet.residuals
    assert (
        "rating_closed_balance_normalized_primary_fin_distribution"
        in wet.assumptions
    )
    assert (
        "rating_closed_balance_normalized_primary_fin_distribution"
        in phase_change.assumptions
    )
    assert wet.outside_alpha_wet_effective_basis == (
        "gross_outside_area_and_bulk_gas_to_core_wall_temperature_difference_"
        "using_raw_radial_transport_duty_before_rating_distribution_"
        "normalization"
    )
    assert wet.outside_alpha_wet_effective_gross_core_basis == pytest.approx(
        result.alfa_o
    )
    assert (
        result.thermal_state.outside_alpha_wet_effective_gross_core_basis
        == pytest.approx(wet.outside_alpha_wet_effective_gross_core_basis)
    )
    assert result.thermal_state.outside_alpha_wet_effective_basis == (
        wet.outside_alpha_wet_effective_basis
    )

    diagnostics = result.finned_tube_diagnostics
    assert diagnostics is not None
    assert diagnostics.outside_alpha_physical == pytest.approx(
        result.thermal_state.outside_alpha_physical
    )
    # Established FinnedTubeDiagnostics thermal fields retain the dry
    # resistance-network meaning.  The latent-inclusive wet coefficient is
    # separately and unambiguously named on the nested wet result above.
    assert diagnostics.outside_alpha_effective_gross == pytest.approx(
        1.0
        / (diagnostics.resistance_outside * diagnostics.area_outside_gross)
    )
    assert result.thermal_state.outside_alpha_effective_gross == pytest.approx(
        diagnostics.outside_alpha_effective_gross
    )
    assert diagnostics.UA == pytest.approx(1.0 / diagnostics.resistance_total)
    assert diagnostics.U == pytest.approx(
        diagnostics.UA / diagnostics.area_outside_gross
    )

    expected_wet_UA = 1.0 / (
        1.0
        / (
            result.thermal_state.alfa_i
            * result.final_result.A_i
        )
        + hx.tube_wall_resistance()
        + 1.0
        / (
            wet.outside_alpha_wet_effective_gross_core_basis
            * result.A_o
        )
    )
    assert result.UA_actual == pytest.approx(expected_wet_UA)
    assert result.U_mean == pytest.approx(expected_wet_UA / result.A_o)
    assert result.thermal_state.UA == pytest.approx(result.UA_actual)
    assert result.thermal_state.U == pytest.approx(result.U_mean)
    assert result.A_required == pytest.approx(
        result.UA_required / result.U_mean
    )
    assert result.overdesign_factor == pytest.approx(
        result.A_o / result.A_required - 1.0
    )
    assert result.ua_margin == pytest.approx(
        result.UA_actual / result.UA_required - 1.0
    )
    assert math.isfinite(result.overdesign_factor)

    assert result.wet_pressure_drop_supported is False
    assert diagnostics.outside_dp_reference_only is True
    assert result.outside_dp_dry_reference == pytest.approx(
        result.outside_dp_total
    )
    warning_sources = (
        list(phase_change.warnings)
        + list(diagnostics.warnings)
        + list(result.warnings or [])
    )
    assert any(
        warning.code
        == WC.CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY
        for warning in warning_sources
    )
    assert pressure_provider.calls > 0

    # The optional achievable bridge must itself use the public phase-aware
    # Simulation path, not the sensible-only internal rating driver.
    achievable = result.simulation
    assert achievable is not None
    assert achievable.outside_phase_change is not None
    assert achievable.outside_phase_change.active is True
    assert achievable.wet_finned_surface is not None
    assert achievable.wet_finned_surface is (
        achievable.outside_phase_change.wet_finned_surface
    )
    assert result.Q_achievable == pytest.approx(achievable.q)

    hydraulic = result.outside_tube_bank_hydraulic
    assert hydraulic.midpoint_method == "arithmetic_temperature_and_water_ratio"
    for nested_diagnostics in (
        result.thermal_state.finned_tube_diagnostics,
        result.final_result.finned_tube_diagnostics,
        diagnostics,
    ):
        assert nested_diagnostics is not None
        assert nested_diagnostics.pressure_drop_coefficient == pytest.approx(
            hydraulic.midpoint.coefficient
        )
        assert nested_diagnostics.pressure_drop_coefficient_definition == (
            hydraulic.coefficient_definition
        )
        assert nested_diagnostics.outside_dp_drag == pytest.approx(
            hydraulic.dp_drag
        )
        assert nested_diagnostics.outside_dp_acceleration == pytest.approx(
            hydraulic.dp_acceleration
        )
        assert nested_diagnostics.outside_dp_total == pytest.approx(
            hydraulic.dp_total
        )
        assert (
            nested_diagnostics.pressure_drop_metadata.method
            == "wet_rating_tagged_robinson_briggs"
        )
    point_mass_flows = tuple(
        state.face_mass_flux * hydraulic.face_area
        for state in (
            hydraulic.inlet,
            hydraulic.midpoint,
            hydraulic.outlet,
        )
    )
    assert point_mass_flows[0] == pytest.approx(
        phase_change.m_dot_gas_in, rel=1.0e-12
    )
    assert point_mass_flows[1] == pytest.approx(
        phase_change.m_dot_dry_carrier * (1.0 + phase_change.W_mid),
        rel=1.0e-12,
    )
    assert point_mass_flows[2] == pytest.approx(
        phase_change.m_dot_gas_out, rel=1.0e-12
    )


def test_dry_finned_rating_keeps_the_legacy_result_exactly() -> None:
    hx = _exchanger()
    inside, outside = _rating_sides(wet_outside=False)
    closed = close_heat_balance(inside, outside)

    expected = run_rating(hx, closed)
    actual = hx.rate(inside, outside)

    assert actual.outside_phase_change.active is False
    assert actual.wet_finned_surface is None
    for name in (
        "Q_required",
        "UA_required",
        "UA_actual",
        "U_mean",
        "A_required",
        "overdesign_factor",
        "ua_margin",
        "alfa_i",
        "alfa_o",
    ):
        assert getattr(actual, name) == getattr(expected, name)
    assert actual.thermal_state == expected.thermal_state
    assert actual.wall_temperature_envelope == expected.wall_temperature_envelope
