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


@pytest.mark.parametrize("unknown_flow", [False, True])
def test_inverse_area_rating_with_default_deadline(unknown_flow) -> None:
    """Production accuracy and deadline, independently of strict regressions."""
    from core import WetCoilSolverOptions

    hx = _exchanger()
    inside, outside = _rating_sides(wet_outside=True)
    inside = replace(inside, m_dot=None) if unknown_flow else replace(inside, T_out=None)
    result = hx.rate(inside, outside, include_simulation=not unknown_flow)
    controls = WetCoilSolverOptions()
    diagnostics = result.wet_coil_diagnostics
    assert diagnostics["solver_options"] == controls
    assert diagnostics["solver_statistics"]["elapsed_s"] < controls.timeout_s
    assert abs(result.closed_balance.outside.T_out - outside.T_out) <= controls.outlet_temperature_tolerance_K
    if unknown_flow:
        assert abs(result.closed_balance.inside.T_out - inside.T_out) <= controls.outlet_temperature_tolerance_K
        assert result.closed_balance.inside.m_dot > 0
    else:
        assert result.simulation is not None
    assert abs(result.outside_phase_change.energy_balance_error) <= controls.energy_tolerance_W
    assert abs(result.outside_phase_change.mass_balance_error) <= controls.mass_tolerance_kg_s
    assert result.A_required == pytest.approx(result.A_o * diagnostics["required_area_scale"])
    assert result.overdesign_factor == pytest.approx(result.A_o / result.A_required - 1)
    assert diagnostics["hydraulic_effective_length"] == hx.bundle.tube.length_effective
    assert result.outside_tube_bank_hydraulic.face_area == hx.bundle.frontal_flow_area


def test_active_auto_rating_uses_installed_geometry_and_native_radial_profile() -> None:
    from numpy.polynomial.legendre import leggauss
    from core import WetCoilSolverOptions
    from core.tests.wet_coil_public_test import check_equivalent
    hx = _exchanger()
    inside, outside = _rating_sides(wet_outside=True)
    inside = replace(inside, T_out=None)
    pressure_provider = _TaggedPressureDropProvider()
    result = hx.rate(inside, outside, finned_pressure_drop_provider=pressure_provider,
                     include_simulation=True,
                     wet_solver_options=WetCoilSolverOptions(
                         energy_tolerance_W=2e-4, timeout_s=None))
    phase_change = result.outside_phase_change
    native = result.wet_coil_diagnostics
    assert phase_change.active and phase_change.converged
    assert phase_change.method == "elmahdy_mitalas_energyplus_v25_2_adapted"
    assert phase_change.m_dot_condensate > 0
    assert phase_change.Q_sensible > 0 and phase_change.Q_latent > 0
    assert phase_change.Q_total == pytest.approx(
        phase_change.Q_sensible+phase_change.Q_latent,rel=1e-12)
    assert abs(phase_change.mass_balance_error) < 1e-6
    assert abs(phase_change.energy_balance_error) < 1e-6
    assert phase_change.wet_area == pytest.approx(
        phase_change.outside_total_area*phase_change.wet_surface_fraction,rel=1e-12)
    profile = native["process_profile"]
    radial = native["surface_states"]
    weights=leggauss(len(profile)-1)[1]*phase_change.wet_fraction/2
    assert len(radial) == len(profile)-1
    assert phase_change.H_drain == pytest.approx(
        sum(w*p.drain_density for w,p in zip(weights,profile[1:])),abs=1e-8)
    assert phase_change.wet_area == pytest.approx(
        sum(w*p["radial_wet_area"] for w,p in zip(weights,radial)),abs=1e-12)
    for axial, surface in zip(profile[1:],radial):
        assert axial.liquid_temperature <= surface["inside_wall_temperature"]
        assert surface["inside_wall_temperature"] <= surface["core_wall_temperature"]
        assert surface["surface_base_temperature"] <= surface["fin_tip_temperature"]
    assert phase_change.wall_temperature_min <= phase_change.wall_temperature_mean <= phase_change.wall_temperature_max
    check_equivalent(result)
    assert result.A_required == native["required_thermal_outside_area"]
    assert native["hydraulic_effective_length"] == hx.bundle.tube.length_effective
    assert result.final_result.A_o == result.A_o == hx.bundle.total_outer_area
    diagnostics = result.finned_tube_diagnostics
    assert diagnostics.outside_alpha_physical == pytest.approx(native["outside_alpha_physical"])
    assert result.wet_pressure_drop_supported is False
    assert diagnostics.outside_dp_reference_only is True
    assert result.outside_dp_dry_reference == pytest.approx(result.outside_dp_total)
    assert WC.CIRCULAR_FINNED_TUBE_WET_PRESSURE_DROP_REFERENCE_ONLY in {
        w.code for w in phase_change.warnings}
    assert pressure_provider.calls > 0
    achievable=result.simulation
    assert achievable is not None
    assert achievable.outside_phase_change.global_wet_model == phase_change.global_wet_model
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


def _capable_auto_rating(inside_T_in_C: float, *, outside_T_out_K: float = 380.0):
    """Rating with a genuinely H2O-capable outside gas, AUTO on both sides.

    Sweeping ``inside_T_in_C`` alone (all else fixed) crosses the AUTO
    onset threshold: colder inside temperatures pull the outside wall
    below the dew point (active wet condensation); warmer ones hold it
    dry/near-onset. Mirrors the Simulation-side transition covered in
    ``phase_change_auto_regime_transition_test.py``, but for Rating.
    """
    hx = _exchanger()
    inside_T_in = inside_T_in_C + 273.15
    inside = BalanceSideSpec(
        provider=_inside_provider(), p=P, m_dot=8.0,
        T_in=inside_T_in, T_out=None,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    outside = BalanceSideSpec(
        provider=_wet_provider(), p=P, m_dot=6.0,
        T_in=420.0, T_out=outside_T_out_K,
        phase_change_mode=PhaseChangeMode.AUTO,
    )
    return hx.rate(inside, outside)


def test_auto_rating_dry_regime_is_a_valid_production_result() -> None:
    """Spec section 5/6/17.24: Rating's AUTO must also legitimately resolve
    to a converged near-onset/dry result, and that result must expose the
    real sensible duty rather than a hardcoded zero."""
    result = _capable_auto_rating(60.0)
    pc = result.outside_phase_change

    assert pc.capable is True
    assert pc.active is False
    assert pc.regime == "DRY"
    assert pc.possible is False
    assert result.ua_is_equivalent
    assert result.wet_finned_surface is None
    assert pc.m_dot_condensate == 0.0
    assert pc.Q_latent == 0.0
    assert pc.Q_sensible == pytest.approx(result.Q_required)
    assert pc.Q_total == pytest.approx(result.Q_required)
    assert math.isfinite(result.UA_actual)
    assert math.isfinite(result.overdesign_factor)


def test_crossing_auto_rating_onset_changes_regime_not_exception() -> None:
    # Use the same outlet target to compare cold and warm coolant at
    # installed hydraulic geometry.
    wet = _capable_auto_rating(20.0, outside_T_out_K=360.0)
    dry = _capable_auto_rating(60.0, outside_T_out_K=360.0)

    assert wet.outside_phase_change.active is True
    assert wet.outside_phase_change.m_dot_condensate > 0.0
    assert wet.outside_phase_change.onset_margin_K > 0.0
    assert dry.outside_phase_change.active is False
    assert dry.outside_phase_change.regime == "DRY"
    assert dry.outside_phase_change.onset_margin_K < 0.0
    assert math.isfinite(wet.Q_required) and math.isfinite(dry.Q_required)
