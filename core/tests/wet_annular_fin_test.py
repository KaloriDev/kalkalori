# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Focused deterministic tests for the nonlinear wet annular-fin surface."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.tube import BareTube
from core.heat_transfer.fin_efficiency import _annular_fin_efficiency_fvm
from core.heat_transfer.outside_dispatch import calculate_resistance_network
from core.phase_change.condensation_solver_helpers import (
    solve_condensing_interface_state,
)
from core.phase_change.mass_heat_transfer import condensation_mass_flux
from core.phase_change.water_equilibrium import saturated_water_ratio
from core.phase_change.wet_finned_surface import (
    DEFAULT_WET_FIN_RADIAL_CELLS,
    WetFinConvergenceError,
    WetFinState,
    _latent_heat_at_temperature,
    _liquid_enthalpy_at_temperature,
    _saturated_ratio_if_gas_phase_exists,
    _total_flux_temperature_derivative,
    solve_wet_annular_fin,
    solve_wet_finned_surface,
)
from core.properties.water import (
    water_latent_heat_of_vaporization,
    water_saturation_liquid_enthalpy,
)


P = 101_325.0
M_DRY = 0.029
CP_GAS = 1010.0
H_OUTSIDE = 75.0


def _tube(**changes) -> CircularFinnedTube:
    values = dict(
        core_tube=BareTube(
            D_i=0.020,
            D_o=0.025,
            length_total=1.8,
            length_effective=1.8,
            wall_k=45.0,
        ),
        fin_k=180.0,
        D_fin=0.052,
        D_root=0.025,
        fin_thickness_root=0.0008,
        fin_thickness_tip=0.00045,
        fin_pitch=0.0035,
        fin_contact_resistance=0.0,
    )
    values.update(changes)
    return CircularFinnedTube(**values)


def _bundle(tube: CircularFinnedTube) -> TubeBundle:
    return TubeBundle(
        tube=tube,
        n_rows=4,
        n_tubes_per_row=6,
        pitch_transverse=0.065,
        pitch_longitudinal=0.055,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement="counterflow",
    )


def _network(tube: CircularFinnedTube):
    return calculate_resistance_network(
        bundle=_bundle(tube),
        alpha_inside=850.0,
        outside_alpha_physical=H_OUTSIDE,
        resistance_core_wall=4.0e-4,
    )


def _W_at_dew_point(temperature: float) -> float:
    return saturated_water_ratio(p_total=P, T=temperature, M_dry=M_DRY)


def _solve_fin(
    tube: CircularFinnedTube,
    *,
    W_bulk: float,
    radial_cells: int | None = None,
):
    kwargs = {}
    if radial_cells is not None:
        kwargs["radial_cells"] = radial_cells
    return solve_wet_annular_fin(
        tube,
        fin_base_temperature=295.0,
        gas_bulk_temperature=330.0,
        outside_alpha_physical=H_OUTSIDE,
        cp_gas=CP_GAS,
        W_bulk=W_bulk,
        p_total=P,
        M_dry=M_DRY,
        **kwargs,
    )


def test_zero_humidity_reproduces_the_existing_dry_fvm() -> None:
    tube = _tube()
    radial_cells = 120
    wet_solver = _solve_fin(tube, W_bulk=0.0, radial_cells=radial_cells)
    dry_reference = _annular_fin_efficiency_fvm(
        tube,
        H_OUTSIDE,
        radial_cells=radial_cells,
    )

    expected_heat = (
        dry_reference.fin_efficiency
        * H_OUTSIDE
        * tube.fin_area_per_fin
        * (330.0 - 295.0)
    )
    expected_tip = 330.0 + dry_reference.tip_temperature_ratio * (295.0 - 330.0)
    assert wet_solver.fin_wet_state is WetFinState.DRY
    assert wet_solver.wet_fin_area == 0.0
    assert wet_solver.m_dot_condensate_fin == 0.0
    assert wet_solver.Q_fin_latent == 0.0
    assert wet_solver.Q_fin_sensible == pytest.approx(expected_heat, rel=2.0e-11)
    assert wet_solver.fin_tip_temperature == pytest.approx(expected_tip, rel=2.0e-11)


def test_partially_wet_fin_resolves_a_radial_dew_point_crossing() -> None:
    tube = _tube(
        fin_k=12.0,
        D_fin=0.072,
        fin_thickness_root=0.00055,
        fin_thickness_tip=0.00030,
    )
    result = _solve_fin(tube, W_bulk=_W_at_dew_point(312.0), radial_cells=160)

    assert result.fin_wet_state is WetFinState.PARTIALLY_WET
    assert 0.0 < result.fin_wet_fraction < 1.0
    assert result.wet_dry_boundary_radius is not None
    assert 0.5 * tube.D_root < result.wet_dry_boundary_radius < 0.5 * tube.D_fin
    assert result.fin_base_temperature < 312.0 < result.fin_tip_temperature
    assert result.m_dot_condensate_fin > 0.0
    assert result.Q_fin_latent > 0.0


def test_fully_wet_high_conductivity_fin_approaches_isothermal_limit() -> None:
    tube = _tube(fin_k=2.0e8)
    W_bulk = _W_at_dew_point(325.0)
    result = _solve_fin(tube, W_bulk=W_bulk, radial_cells=80)

    expected_sensible = H_OUTSIDE * tube.fin_area_per_fin * (330.0 - 295.0)
    expected_mass = tube.fin_area_per_fin * condensation_mass_flux(
        alfa_dry=H_OUTSIDE,
        cp_gas=CP_GAS,
        W_bulk=W_bulk,
        W_sat_surface=_W_at_dew_point(295.0),
    )
    expected_latent = expected_mass * water_latent_heat_of_vaporization(T=295.0)
    assert result.fin_wet_state is WetFinState.FULLY_WET
    assert result.fin_wet_fraction == 1.0
    assert result.fin_tip_temperature == pytest.approx(295.0, abs=2.0e-4)
    assert result.Q_fin_sensible == pytest.approx(expected_sensible, rel=1.0e-5)
    assert result.m_dot_condensate_fin == pytest.approx(expected_mass, rel=2.0e-5)
    assert result.Q_fin_latent == pytest.approx(expected_latent, rel=2.0e-5)


def test_welded_contact_is_applied_only_to_fin_branch_in_correct_direction() -> None:
    ideal_tube = _tube(fin_contact_resistance=0.0)
    finite_tube = replace(ideal_tube, fin_contact_resistance=8.0e-4)
    W_bulk = _W_at_dew_point(318.0)

    def solve(tube: CircularFinnedTube):
        return solve_wet_finned_surface(
            _bundle(tube),
            _network(tube),
            gas_bulk_temperature=334.0,
            inside_bulk_temperature=288.0,
            cp_gas=CP_GAS,
            W_bulk=W_bulk,
            p_total=P,
            M_dry=M_DRY,
            m_dot_water_vapor_available=1.0,
            radial_cells=80,
        )

    ideal = solve(ideal_tube)
    finite = solve(finite_tube)
    assert ideal.contact_topology == finite.contact_topology == "fin_branch_only"
    assert ideal.fin_base_temperature == pytest.approx(ideal.core_wall_temperature)
    assert finite.fin_base_temperature > finite.core_wall_temperature
    assert finite.primary_surface_temperature == pytest.approx(
        finite.core_wall_temperature
    )
    assert finite.Q_fin_total < ideal.Q_fin_total


def test_continuous_root_primary_and_fin_splits_close_exactly() -> None:
    tube = _tube(D_root=0.029, fin_contact_resistance=2.0e-4)
    bundle = _bundle(tube)
    result = solve_wet_finned_surface(
        bundle,
        _network(tube),
        gas_bulk_temperature=334.0,
        inside_bulk_temperature=288.0,
        cp_gas=CP_GAS,
        W_bulk=_W_at_dew_point(320.0),
        p_total=P,
        M_dry=M_DRY,
        m_dot_water_vapor_available=1.0,
        radial_cells=100,
    )

    assert result.contact_topology == (
        "series_before_primary_and_fin_parallel_branches"
    )
    assert result.primary_surface_temperature == pytest.approx(
        result.fin_base_temperature
    )
    assert result.root_surface_temperature > result.core_wall_temperature
    assert result.Q_fin_total == pytest.approx(
        result.Q_fin_sensible + result.Q_fin_latent, abs=1.0e-10
    )
    assert result.Q_primary_total == pytest.approx(
        result.Q_primary_sensible + result.Q_primary_latent, abs=1.0e-10
    )
    assert result.Q_total == pytest.approx(
        result.Q_primary_total + result.Q_fin_total, abs=1.0e-9
    )
    assert result.m_dot_condensate == pytest.approx(
        result.m_dot_condensate_primary + result.m_dot_condensate_fin,
        abs=1.0e-14,
    )
    assert result.wet_area == pytest.approx(
        result.wet_primary_area + result.wet_fin_area, abs=1.0e-12
    )
    assert result.energy_balance_error < 1.0e-6
    assert result.outside_alpha_physical == H_OUTSIDE
    assert result.outside_surface_temperature_area_mean == pytest.approx(
        334.0
        - result.Q_sensible
        / (result.outside_alpha_physical * result.outside_total_area)
    )
    assert result.outside_alpha_wet_effective_gross_core_basis == pytest.approx(
        result.Q_total
        / (
            result.outside_total_area
            * (334.0 - result.core_wall_temperature)
        )
    )
    assert result.outside_alpha_wet_effective_basis == (
        "gross_outside_area_and_bulk_gas_to_core_wall_temperature_difference"
    )


def test_authoritative_fin_area_override_scales_the_representative_fin() -> None:
    geometric = _tube()
    overridden = replace(
        geometric,
        external_area_per_length=(
            geometric.primary_outside_area_per_length
            + 0.5 * geometric.fin_area_geometric_per_length
        ),
    )
    bundle = _bundle(overridden)
    network = _network(overridden)
    result = solve_wet_finned_surface(
        bundle,
        network,
        gas_bulk_temperature=334.0,
        inside_bulk_temperature=288.0,
        cp_gas=CP_GAS,
        W_bulk=_W_at_dew_point(320.0),
        p_total=P,
        M_dry=M_DRY,
        m_dot_water_vapor_available=1.0,
        radial_cells=60,
    )

    expected_count = network.area_fin / overridden.fin_area_per_fin
    assert result.equivalent_fin_count == pytest.approx(expected_count)
    assert result.annular_fin is not None
    assert result.Q_fin_total == pytest.approx(
        result.annular_fin.Q_fin_total * expected_count, rel=1.0e-12
    )
    assert result.primary_area + result.fin_area == pytest.approx(
        result.outside_total_area
    )


def test_available_water_cap_is_shared_by_primary_and_fin_sources() -> None:
    tube = _tube()
    available = 2.0e-4
    result = solve_wet_finned_surface(
        _bundle(tube),
        _network(tube),
        gas_bulk_temperature=334.0,
        inside_bulk_temperature=288.0,
        cp_gas=CP_GAS,
        W_bulk=_W_at_dew_point(328.0),
        p_total=P,
        M_dry=M_DRY,
        m_dot_water_vapor_available=available,
        radial_cells=60,
    )

    assert result.water_availability_scale < 1.0
    assert result.m_dot_condensate == pytest.approx(available, rel=2.0e-9)
    assert result.m_dot_condensate_primary > 0.0
    assert result.m_dot_condensate_fin > 0.0


@pytest.mark.parametrize(
    ("fin_base_temperature", "gas_bulk_temperature"),
    ((390.0, 410.0), (650.0, 670.0)),
)
def test_surface_above_boiling_at_total_pressure_is_dry_not_an_error(
    fin_base_temperature: float,
    gas_bulk_temperature: float,
) -> None:
    tube = _tube()
    result = solve_wet_annular_fin(
        tube,
        fin_base_temperature=fin_base_temperature,
        gas_bulk_temperature=gas_bulk_temperature,
        outside_alpha_physical=H_OUTSIDE,
        cp_gas=CP_GAS,
        W_bulk=_W_at_dew_point(300.0),
        p_total=P,
        M_dry=M_DRY,
        radial_cells=40,
    )

    assert result.fin_wet_state is WetFinState.DRY
    assert result.m_dot_condensate_fin == 0.0
    assert result.Q_fin_latent == 0.0


def test_shifted_derivative_uses_one_sided_sample_at_triple_point() -> None:
    derivative = _total_flux_temperature_derivative(
        283.1601,
        gas_bulk_temperature=330.0,
        outside_alpha_physical=H_OUTSIDE,
        cp_gas=CP_GAS,
        W_bulk=_W_at_dew_point(315.0),
        p_total=P,
        M_dry=M_DRY,
        M_h2o=0.01801528,
        lewis_number=1.0,
        availability_scale=1.0,
        condensation_area_fraction=0.25,
        condensation_temperature_offset_K=-10.0,
    )

    assert math.isfinite(derivative)


def test_nonconvergence_raises_with_last_residuals_instead_of_returning_iterate() -> None:
    tube = _tube(fin_k=20.0, D_fin=0.070)
    with pytest.raises(WetFinConvergenceError) as caught:
        solve_wet_annular_fin(
            tube,
            fin_base_temperature=295.0,
            gas_bulk_temperature=330.0,
            outside_alpha_physical=H_OUTSIDE,
            cp_gas=CP_GAS,
            W_bulk=_W_at_dew_point(318.0),
            p_total=P,
            M_dry=M_DRY,
            radial_cells=40,
            max_iterations=1,
        )

    assert caught.value.iterations == 1
    assert caught.value.residuals["equation_residual_W"] > 0.0


def test_off_grid_water_property_interpolation_tracks_authoritative_helpers() -> None:
    """Guard the documented 0.25 K acceleration approximation explicitly."""

    temperature = 312.123
    interpolated_W_sat = _saturated_ratio_if_gas_phase_exists(
        p_total=P,
        temperature=temperature,
        M_dry=M_DRY,
        M_h2o=0.01801528,
    )
    assert interpolated_W_sat is not None
    exact_W_sat = saturated_water_ratio(
        p_total=P,
        T=temperature,
        M_dry=M_DRY,
    )
    exact_h_fg = water_latent_heat_of_vaporization(T=temperature)
    exact_h_liquid = water_saturation_liquid_enthalpy(T=temperature)

    assert interpolated_W_sat == pytest.approx(exact_W_sat, rel=3.0e-5)
    assert _latent_heat_at_temperature(temperature) == pytest.approx(
        exact_h_fg, rel=1.0e-8
    )
    assert _liquid_enthalpy_at_temperature(temperature) == pytest.approx(
        exact_h_liquid, rel=1.0e-8
    )


def test_partial_wet_refinement_is_deterministic_through_320_cells() -> None:
    tube = _tube(
        fin_k=12.0,
        D_fin=0.072,
        fin_thickness_root=0.00055,
        fin_thickness_tip=0.00030,
    )
    W_bulk = _W_at_dew_point(312.0)
    results = {}
    for radial_cells in (40, 80, 160, 320):
        first = _solve_fin(
            tube, W_bulk=W_bulk, radial_cells=radial_cells
        )
        second = _solve_fin(
            tube, W_bulk=W_bulk, radial_cells=radial_cells
        )
        assert first == second
        assert first.fin_wet_state is WetFinState.PARTIALLY_WET
        results[radial_cells] = first

    reference = results[320]
    selected_default = results[DEFAULT_WET_FIN_RADIAL_CELLS]
    assert DEFAULT_WET_FIN_RADIAL_CELLS == 160
    assert selected_default.Q_fin_total == pytest.approx(
        reference.Q_fin_total, rel=2.0e-4
    )
    assert selected_default.m_dot_condensate_fin == pytest.approx(
        reference.m_dot_condensate_fin, rel=5.0e-4
    )
    assert selected_default.fin_wet_fraction == pytest.approx(
        reference.fin_wet_fraction, rel=2.0e-4
    )
    assert selected_default.wet_dry_boundary_radius == pytest.approx(
        reference.wet_dry_boundary_radius,
        abs=0.25 * (tube.D_fin - tube.D_root) / 320.0,
    )

    implicit_default = _solve_fin(tube, W_bulk=W_bulk)
    assert implicit_default.radial_cells == DEFAULT_WET_FIN_RADIAL_CELLS
    assert implicit_default == selected_default


def test_humidity_sweep_transitions_continuously_dry_partial_and_full() -> None:
    tube = _tube(fin_k=40.0, D_fin=0.060)
    dew_points = (294.0, 300.0, 306.0, 312.0, 318.0, 324.0, 328.0)
    results = tuple(
        _solve_fin(
            tube,
            W_bulk=_W_at_dew_point(dew_point),
            radial_cells=60,
        )
        for dew_point in dew_points
    )

    assert results[0].fin_wet_state is WetFinState.DRY
    assert results[3].fin_wet_state is WetFinState.PARTIALLY_WET
    assert results[-1].fin_wet_state is WetFinState.FULLY_WET
    fractions = tuple(result.fin_wet_fraction for result in results)
    condensate = tuple(result.m_dot_condensate_fin for result in results)
    assert fractions == tuple(sorted(fractions))
    assert condensate == tuple(sorted(condensate))
    assert all(
        right - left < 0.30
        for left, right in zip(fractions, fractions[1:])
    )


def test_colder_fin_base_monotonically_increases_wet_fraction() -> None:
    tube = _tube(fin_k=40.0, D_fin=0.060)
    results = tuple(
        solve_wet_annular_fin(
            tube,
            fin_base_temperature=base_temperature,
            gas_bulk_temperature=330.0,
            outside_alpha_physical=H_OUTSIDE,
            cp_gas=CP_GAS,
            W_bulk=_W_at_dew_point(312.0),
            p_total=P,
            M_dry=M_DRY,
            radial_cells=60,
        )
        for base_temperature in (305.0, 300.0, 295.0, 290.0, 285.0)
    )

    fractions = tuple(result.fin_wet_fraction for result in results)
    condensate = tuple(result.m_dot_condensate_fin for result in results)
    # Inputs are ordered from warm to cold, so both sequences must increase.
    assert fractions == tuple(sorted(fractions))
    assert condensate == tuple(sorted(condensate))


def test_efficiency_contact_mode_matches_ideal_at_one_and_reduces_fin_duty() -> None:
    explicit_ideal = _tube(fin_contact_resistance=0.0)
    efficiency_ideal = _tube(
        fin_contact_resistance=None,
        fin_contact_efficiency=1.0,
    )
    efficiency_finite = replace(
        efficiency_ideal,
        fin_contact_efficiency=0.8,
    )

    def solve(tube: CircularFinnedTube):
        return solve_wet_finned_surface(
            _bundle(tube),
            _network(tube),
            gas_bulk_temperature=334.0,
            inside_bulk_temperature=288.0,
            cp_gas=CP_GAS,
            W_bulk=_W_at_dew_point(322.0),
            p_total=P,
            M_dry=M_DRY,
            m_dot_water_vapor_available=1.0,
            radial_cells=60,
        )

    explicit = solve(explicit_ideal)
    ideal = solve(efficiency_ideal)
    finite = solve(efficiency_finite)
    assert ideal.contact_input_mode == "contact_efficiency"
    assert ideal.contact_resistance_used == 0.0
    assert ideal.Q_total == pytest.approx(explicit.Q_total, rel=1.0e-12)
    assert ideal.fin_base_temperature == pytest.approx(
        explicit.fin_base_temperature,
        abs=1.0e-12,
    )
    assert finite.contact_input_mode == "contact_efficiency"
    assert finite.contact_resistance_used > 0.0
    assert finite.fin_base_temperature > finite.core_wall_temperature
    assert finite.Q_fin_total < ideal.Q_fin_total
    assert finite.Q_total < ideal.Q_total


def test_zero_authoritative_fin_area_reduces_to_bare_wet_interface() -> None:
    geometric = _tube()
    primary_only = replace(
        geometric,
        external_area_per_length=geometric.primary_outside_area_per_length,
    )
    bundle = _bundle(primary_only)
    network = _network(primary_only)
    W_bulk = _W_at_dew_point(322.0)
    wet = solve_wet_finned_surface(
        bundle,
        network,
        gas_bulk_temperature=334.0,
        inside_bulk_temperature=288.0,
        cp_gas=CP_GAS,
        W_bulk=W_bulk,
        p_total=P,
        M_dry=M_DRY,
        m_dot_water_vapor_available=1.0,
        radial_cells=60,
    )
    bare = solve_condensing_interface_state(
        alfa_dry=H_OUTSIDE,
        A_total=network.area_primary_outside,
        A_wet=network.area_primary_outside,
        T_wall_wet_mean=wet.primary_surface_temperature,
        T_bulk_wet_gas=334.0,
        T_bulk_other=288.0,
        R_downstream=(
            network.resistance_inside + network.resistance_core_wall
        ),
        cp_gas=CP_GAS,
        W_bulk=W_bulk,
        p_wet_gas=P,
        M_dry=M_DRY,
        m_dot_water_vapor_available=1.0,
        lewis_number=1.0,
    )

    assert network.area_fin == 0.0
    assert wet.annular_fin is None
    assert wet.Q_fin_total == 0.0
    assert wet.m_dot_condensate_fin == 0.0
    assert wet.primary_surface_temperature == pytest.approx(
        bare[0],
        abs=4.0e-4,
    )
    assert wet.Q_sensible == pytest.approx(bare[1], rel=3.0e-5)
    assert wet.Q_latent == pytest.approx(bare[2], rel=3.0e-5)
    assert wet.m_dot_condensate == pytest.approx(bare[3], rel=3.0e-5)


def test_0d_endpoint_wet_zone_activates_an_otherwise_dry_mean_surface() -> None:
    """The AUTO integration fallback is area weighted, not a fake radius."""

    tube = _tube()
    common = dict(
        gas_bulk_temperature=334.0,
        inside_bulk_temperature=288.0,
        cp_gas=CP_GAS,
        W_bulk=_W_at_dew_point(315.0),
        p_total=P,
        M_dry=M_DRY,
        m_dot_water_vapor_available=1.0,
        radial_cells=60,
    )
    mean_surface = solve_wet_finned_surface(
        _bundle(tube),
        _network(tube),
        **common,
    )
    cold_zone = solve_wet_finned_surface(
        _bundle(tube),
        _network(tube),
        condensation_area_fraction=0.35,
        condensation_temperature_offset_K=-10.0,
        **common,
    )

    assert mean_surface.fin_wet_state is WetFinState.DRY
    assert mean_surface.m_dot_condensate == 0.0
    assert cold_zone.fin_wet_state is WetFinState.PARTIALLY_WET
    assert cold_zone.fin_wet_fraction == pytest.approx(0.35)
    # The representative cold-zone fin is radially wet to the tip. Its
    # aggregate partial area comes from the endpoint envelope, so no radial
    # crossing exists and no axial segmentation is implied.
    assert cold_zone.wet_dry_boundary_radius is None
    assert cold_zone.wet_area == pytest.approx(
        0.35 * cold_zone.outside_total_area,
        rel=1.0e-12,
    )
    assert cold_zone.m_dot_condensate_primary > 0.0
    assert cold_zone.m_dot_condensate_fin > 0.0
    assert cold_zone.Q_latent > 0.0
    assert cold_zone.energy_balance_error < 1.0e-6
    assert (
        "endpoint_envelope_wet_zone_0d_linear_weighting"
        in cold_zone.assumptions
    )


def test_zero_condensation_area_ignores_the_inactive_temperature_offset() -> None:
    """A zero-area wet-zone request is the exact dry sensible limit."""

    tube = _tube()
    common = dict(
        gas_bulk_temperature=334.0,
        inside_bulk_temperature=288.0,
        cp_gas=CP_GAS,
        p_total=P,
        M_dry=M_DRY,
        m_dot_water_vapor_available=1.0,
        radial_cells=60,
    )
    zero_area = solve_wet_finned_surface(
        _bundle(tube),
        _network(tube),
        W_bulk=_W_at_dew_point(315.0),
        condensation_area_fraction=0.0,
        # This would be below the water triple point if it were evaluated;
        # with no condensation area the offset has no physical meaning.
        condensation_temperature_offset_K=-100.0,
        **common,
    )
    dry_reference = solve_wet_finned_surface(
        _bundle(tube),
        _network(tube),
        W_bulk=0.0,
        **common,
    )

    assert zero_area.fin_wet_state is WetFinState.DRY
    assert zero_area.m_dot_condensate == 0.0
    assert zero_area.Q_latent == 0.0
    assert zero_area.wet_area == 0.0
    assert zero_area.Q_sensible == pytest.approx(
        dry_reference.Q_sensible,
        rel=1.0e-12,
    )
