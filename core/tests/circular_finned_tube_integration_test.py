"""Integration coverage for dry circular-finned-tube exchanger workflows."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, CircularFinnedTube, TubeSurfaceType
from core.heat_transfer.finned_tube_outside import BriggsYoung1963Provider
from core.heat_transfer.internal_flow import FluidProps as InternalFluidProps
from core.heat_transfer.outside_dispatch import (
    IncompatibleOutsideCorrelationProviderError,
    calculate_resistance_network,
    evaluate_outside_hydraulics,
    evaluate_outside_thermal,
)
from core.heat_transfer.outside_flow import (
    FluidProps as OutsideFluidProps,
    calculate_outside_tube_bank_hydraulics,
    outside_flow_from_mass_flow,
)
from core.heat_transfer.streams import SensibleHeatStream
from core.models import FinnedTubeDiagnostics
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput, run_simulation
from core.phase_change.finned_tube_guard import (
    CircularFinnedTubeWetSurfaceNotSupportedError,
)
from core.phase_change import (
    CircularFinnedTubeWetSurfaceNotSupportedError as PublicWetSurfaceError,
)
from core.phase_change.regime import OnsetDecision
from core.phase_change.types import PhaseChangeCapability, PhaseChangeMode
from core.pressure_drop.finned_tube_pressure_drop import RobinsonBriggs1966Provider
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.water import IAPWS97WaterSteamProvider


P = 101_325.0
M_DOT_INSIDE = 1.5
M_DOT_OUTSIDE = 2.0
T_INSIDE = 360.0
T_OUTSIDE = 300.0

INSIDE_TRANSPORT = FluidTransportProperties(
    rho=900.0, mu=5.0e-4, k=0.60, cp=4000.0
)
OUTSIDE_TRANSPORT = FluidTransportProperties(
    rho=1.10, mu=1.90e-5, k=0.028, cp=1007.0
)


def _core_tube() -> BareTube:
    return BareTube(
        D_i=0.021,
        D_o=0.025,
        length_total=2.0,
        length_effective=2.0,
        wall_k=45.0,
    )


def _finned_tube(*, contact: float | None = None) -> CircularFinnedTube:
    return CircularFinnedTube(
        core_tube=_core_tube(),
        fin_k=200.0,
        D_fin=0.050,
        D_root=0.028,
        fin_thickness_root=0.0005,
        fin_pitch=0.0024,
        fin_contact_resistance=contact,
    )


def _bundle(*, finned: bool, layout: str = "staggered") -> TubeBundle:
    pitch_transverse = 0.060
    pitch_longitudinal = (
        pitch_transverse * math.sqrt(3.0) / 2.0
        if layout == "staggered"
        else 0.060
    )
    return TubeBundle(
        tube=_finned_tube() if finned else _core_tube(),
        n_rows=6,
        n_tubes_per_row=8,
        pitch_transverse=pitch_transverse,
        pitch_longitudinal=pitch_longitudinal,
        layout=layout,
        n_passes_tube=2,
        flow_arrangement="counterflow",
    )


def _providers() -> tuple[ConstantPropertyProvider, ConstantPropertyProvider]:
    return (
        ConstantPropertyProvider(INSIDE_TRANSPORT),
        ConstantPropertyProvider(OUTSIDE_TRANSPORT),
    )


def _side_inputs(
    *, mode: PhaseChangeMode = PhaseChangeMode.AUTO
) -> tuple[HXSideInput, HXSideInput]:
    inside_provider, outside_provider = _providers()
    return (
        HXSideInput(
            provider=inside_provider,
            m_dot=M_DOT_INSIDE,
            T_in=T_INSIDE,
            p=P,
            phase_change_mode=mode,
        ),
        HXSideInput(
            provider=outside_provider,
            m_dot=M_DOT_OUTSIDE,
            T_in=T_OUTSIDE,
            p=P,
            phase_change_mode=mode,
        ),
    )


def _solve(hx: BareTubeHeatExchanger, **kwargs):
    inside_provider, outside_provider = _providers()
    return hx.solve(
        hot_stream=SensibleHeatStream(
            C=M_DOT_INSIDE * INSIDE_TRANSPORT.cp, T_in=T_INSIDE
        ),
        cold_stream=SensibleHeatStream(
            C=M_DOT_OUTSIDE * OUTSIDE_TRANSPORT.cp, T_in=T_OUTSIDE
        ),
        m_dot_tube_side=M_DOT_INSIDE,
        tube_side_props=InternalFluidProps(**INSIDE_TRANSPORT.__dict__),
        tube_side_provider=inside_provider,
        tube_side_temperature_in=T_INSIDE,
        tube_side_temperature_out=350.0,
        tube_side_pressure=P,
        m_dot_outside=M_DOT_OUTSIDE,
        outside_props=OutsideFluidProps(**OUTSIDE_TRANSPORT.__dict__),
        outside_provider=outside_provider,
        outside_temperature_in=T_OUTSIDE,
        outside_temperature_out=320.0,
        outside_pressure=P,
        **kwargs,
    )


def _assert_finned_diagnostics(result) -> None:
    diagnostics = result.finned_tube_diagnostics
    assert isinstance(diagnostics, FinnedTubeDiagnostics)
    assert diagnostics.tube_surface_type is TubeSurfaceType.CIRCULAR_FINNED
    for value in (
        diagnostics.A_inside,
        diagnostics.A_primary_outside,
        diagnostics.A_fin,
        diagnostics.A_outside_gross,
        diagnostics.A_outside_geometric,
        diagnostics.A_outside_effective,
        diagnostics.eta_fin,
        diagnostics.eta_overall,
        diagnostics.outside_htc,
        diagnostics.UA,
        diagnostics.U,
        diagnostics.face_velocity,
        diagnostics.reference_velocity,
        diagnostics.Re,
        diagnostics.Nu,
        diagnostics.j,
        diagnostics.f,
        diagnostics.dp,
    ):
        assert math.isfinite(value) and value > 0.0
    assert 0.0 < diagnostics.eta_fin <= 1.0
    assert 0.0 < diagnostics.eta_overall <= 1.0
    assert diagnostics.A_outside_effective == pytest.approx(
        diagnostics.A_primary_outside
        + diagnostics.eta_fin * diagnostics.A_fin
    )
    assert diagnostics.contact_resistance_used == 0.0
    assert diagnostics.resistance_contact == 0.0
    assert diagnostics.UA == pytest.approx(1.0 / diagnostics.resistance_total)
    assert diagnostics.U == pytest.approx(
        diagnostics.UA / diagnostics.A_outside_gross
    )
    assert diagnostics.heat_transfer_metadata.method == "briggs_young_1963"
    assert diagnostics.pressure_drop_metadata.method == "robinson_briggs_1966"
    assert diagnostics.pressure_drop_coefficient_definition == (
        "delta_p/(2*n_rows*rho*V_max^2)"
    )
    assert any(
        warning.code == "circular_finned_tube_contact_resistance_unspecified"
        for warning in diagnostics.warnings
    )
    assert not any(
        warning.code.endswith(("_below_range", "_above_range"))
        for warning in diagnostics.warnings
    )


def test_dry_solve_simulate_and_rate_expose_consistent_finned_results() -> None:
    assert PublicWetSurfaceError is CircularFinnedTubeWetSurfaceNotSupportedError
    hx = BareTubeHeatExchanger(_bundle(finned=True))
    solve_result = _solve(hx)

    assert solve_result.tube_surface_type is TubeSurfaceType.CIRCULAR_FINNED
    assert solve_result.UA > 0.0
    assert solve_result.Q > 0.0
    assert solve_result.outside_dp_total > 0.0
    assert solve_result.outside_tube_bank_hydraulic.metadata.method == (
        "robinson_briggs_1966"
    )
    bank_group = solve_result.outside_side_pressure_drop.flow_path.group("tube_bank")
    assert bank_group.stages[0].stage_type == "circular_finned_tube_bank"
    assert not any(
        warning.code in {"outside_ht_laminar_regime", "outside_ht_transition_regime"}
        for warning in (solve_result.warnings or [])
    )
    _assert_finned_diagnostics(solve_result)

    inside, outside = _side_inputs()
    simulation = hx.simulate(inside, outside)
    assert simulation.phase_change_active is False
    assert simulation.finned_tube == simulation.finned_tube_diagnostics
    _assert_finned_diagnostics(simulation)
    q_inside = M_DOT_INSIDE * INSIDE_TRANSPORT.cp * (
        T_INSIDE - simulation.T_out_inside
    )
    q_outside = M_DOT_OUTSIDE * OUTSIDE_TRANSPORT.cp * (
        simulation.T_out_outside - T_OUTSIDE
    )
    assert simulation.q == pytest.approx(q_inside, rel=2.0e-9)
    assert simulation.q == pytest.approx(q_outside, rel=2.0e-9)
    assert simulation.UA == pytest.approx(
        simulation.finned_tube_diagnostics.UA
    )
    assert simulation.outside_properties_midpoint.Pr > 0.0

    rating = hx.rate(
        BalanceSideSpec(
            provider=inside.provider,
            p=P,
            m_dot=M_DOT_INSIDE,
            T_in=T_INSIDE,
            T_out=simulation.T_out_inside,
        ),
        BalanceSideSpec(
            provider=outside.provider,
            p=P,
            m_dot=M_DOT_OUTSIDE,
            T_in=T_OUTSIDE,
            T_out=simulation.T_out_outside,
        ),
    )
    assert rating.phase_change_active is False
    assert rating.UA_actual > 0.0
    assert math.isfinite(rating.overdesign_factor)
    assert rating.finned_tube == rating.finned_tube_diagnostics
    _assert_finned_diagnostics(rating)
    assert rating.UA_actual == pytest.approx(rating.finned_tube_diagnostics.UA)


class _CountingHeatTransferProvider:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        return BriggsYoung1963Provider().evaluate(request)


class _CountingPressureDropProvider:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        return RobinsonBriggs1966Provider().evaluate(request)


class _CustomMethodPressureDropProvider:
    def evaluate(self, request):
        result = RobinsonBriggs1966Provider().evaluate(request)
        return replace(
            result,
            metadata=replace(result.metadata, method="custom_finned_pressure_drop"),
        )


def test_public_workflows_thread_dedicated_finned_providers() -> None:
    hx = BareTubeHeatExchanger(_bundle(finned=True))
    inside, outside = _side_inputs(mode=PhaseChangeMode.DISABLED)
    heat_transfer_provider = _CountingHeatTransferProvider()
    pressure_drop_provider = _CountingPressureDropProvider()
    simulation = hx.simulate(
        inside,
        outside,
        finned_heat_transfer_provider=heat_transfer_provider,
        finned_pressure_drop_provider=pressure_drop_provider,
    )
    assert simulation.finned_tube_diagnostics is not None
    assert heat_transfer_provider.calls > 0
    assert pressure_drop_provider.calls > 0

    heat_transfer_calls = heat_transfer_provider.calls
    pressure_drop_calls = pressure_drop_provider.calls
    rating = hx.rate(
        BalanceSideSpec(
            provider=inside.provider,
            p=P,
            m_dot=M_DOT_INSIDE,
            T_in=T_INSIDE,
            T_out=simulation.T_out_inside,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        BalanceSideSpec(
            provider=outside.provider,
            p=P,
            m_dot=M_DOT_OUTSIDE,
            T_in=T_OUTSIDE,
            T_out=simulation.T_out_outside,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        finned_heat_transfer_provider=heat_transfer_provider,
        finned_pressure_drop_provider=pressure_drop_provider,
    )
    assert rating.finned_tube_diagnostics is not None
    assert heat_transfer_provider.calls > heat_transfer_calls
    assert pressure_drop_provider.calls > pressure_drop_calls


def test_finned_rejects_bare_provider_alpha_override_and_inline_solver() -> None:
    hx = BareTubeHeatExchanger(_bundle(finned=True))
    with pytest.raises(IncompatibleOutsideCorrelationProviderError):
        _solve(hx, euler_provider="gaddis_gnielinski")
    with pytest.raises(ValueError, match="does not accept the bare alfa_o override"):
        _solve(hx, alfa_o=50.0)
    with pytest.raises(IncompatibleOutsideCorrelationProviderError):
        _solve(hx, finned_heat_transfer_provider=RobinsonBriggs1966Provider())
    with pytest.raises(IncompatibleOutsideCorrelationProviderError):
        _solve(hx, finned_pressure_drop_provider=BriggsYoung1963Provider())

    inline_hx = BareTubeHeatExchanger(_bundle(finned=True, layout="inline"))
    with pytest.raises(ValueError, match="staggered"):
        _solve(inline_hx)


def test_flow_path_identifies_finned_result_by_type_with_custom_method() -> None:
    result = _solve(
        BareTubeHeatExchanger(_bundle(finned=True)),
        finned_pressure_drop_provider=_CustomMethodPressureDropProvider(),
    )
    bank_group = result.outside_side_pressure_drop.flow_path.group("tube_bank")
    stage = bank_group.stages[0]
    assert stage.stage_type == "circular_finned_tube_bank"
    assert stage.method == (
        "finned_tube_bank_hydraulics_custom_finned_pressure_drop"
    )


def test_plain_dispatch_is_exact_legacy_regression() -> None:
    bundle = _bundle(finned=False)
    props = OutsideFluidProps(**OUTSIDE_TRANSPORT.__dict__)
    dispatched = evaluate_outside_thermal(
        bundle=bundle,
        m_dot=M_DOT_OUTSIDE,
        props=props,
    )
    legacy = outside_flow_from_mass_flow(
        m_dot=M_DOT_OUTSIDE,
        frontal_area=bundle.frontal_flow_area,
        n_tubes_per_row=bundle.n_tubes_per_row,
        tube_outer_diameter=bundle.tube.D_o,
        tube_pitch_transverse=bundle.pitch_transverse,
        tube_pitch_longitudinal=bundle.pitch_longitudinal,
        layout=bundle.layout,
        n_rows=bundle.n_rows,
        props=props,
        calculate_pressure_drop=False,
    )
    assert (
        dispatched.face_velocity,
        dispatched.reynolds_number,
        dispatched.prandtl_number,
        dispatched.alpha,
        list(dispatched.warnings),
    ) == (legacy[0], legacy[1], legacy[2], legacy[3], legacy[5])

    outside_provider = _providers()[1]
    dispatched_hydraulics = evaluate_outside_hydraulics(
        bundle=bundle,
        m_dot=M_DOT_OUTSIDE,
        property_provider=outside_provider,
        temperature_in=T_OUTSIDE,
        temperature_out=320.0,
        pressure=P,
    )
    legacy_hydraulics = calculate_outside_tube_bank_hydraulics(
        m_dot=M_DOT_OUTSIDE,
        face_area=bundle.frontal_flow_area,
        tube_outer_diameter=bundle.tube.D_o,
        tube_pitch_transverse=bundle.pitch_transverse,
        tube_pitch_longitudinal=bundle.pitch_longitudinal,
        layout=bundle.layout,
        n_rows=bundle.n_rows,
        n_tubes_per_row=bundle.n_tubes_per_row,
        provider=outside_provider,
        temperature_in=T_OUTSIDE,
        temperature_out=320.0,
        pressure=P,
    )
    assert dispatched_hydraulics == legacy_hydraulics
    assert _solve(BareTubeHeatExchanger(bundle)).finned_tube_diagnostics is None
    with pytest.raises(IncompatibleOutsideCorrelationProviderError):
        evaluate_outside_thermal(
            bundle=bundle,
            m_dot=M_DOT_OUTSIDE,
            props=props,
            finned_heat_transfer_provider="robinson_briggs_1966",
        )
    with pytest.raises(IncompatibleOutsideCorrelationProviderError):
        evaluate_outside_hydraulics(
            bundle=bundle,
            m_dot=M_DOT_OUTSIDE,
            inlet_props=OUTSIDE_TRANSPORT,
            finned_pressure_drop_provider="briggs_young_1963",
        )


def _active_outside_onset(*, side: str, **_kwargs):
    if side == "inside":
        return None, None, None, None, None
    return OnsetDecision(5.0, True, True, False), 310.0, 305.0, 307.0, 309.0


def _capability(provider, outside_provider):
    if provider is not outside_provider:
        return PhaseChangeCapability(capable=False)
    return PhaseChangeCapability(
        capable=True,
        component="H2O",
        provider_kind="gas_mixture",
        dry_mole_fractions={"Air": 1.0},
        M_dry=0.029,
        M_condensable=0.018,
        W_in=0.01,
    )


def test_active_wet_simulation_and_rating_are_rejected_but_dry_disabled_runs(
    monkeypatch,
) -> None:
    hx = BareTubeHeatExchanger(_bundle(finned=True))
    inside, outside = _side_inputs()
    dry_result = run_simulation(hx, inside, outside)

    import core.phase_change.integration as simulation_integration

    monkeypatch.setattr(
        simulation_integration,
        "detect_phase_change_capability",
        lambda provider: _capability(provider, outside.provider),
    )
    monkeypatch.setattr(
        simulation_integration, "evaluate_side_onset", _active_outside_onset
    )
    with pytest.raises(
        CircularFinnedTubeWetSurfaceNotSupportedError,
        match="dry single-phase operation.*unsupported",
    ):
        simulation_integration.apply_phase_change(
            hx, inside, outside, dry_result, iterate=True
        )

    import core.phase_change.rating_integration as rating_integration

    monkeypatch.setattr(
        rating_integration,
        "detect_phase_change_capability",
        lambda provider: _capability(provider, outside.provider),
    )
    monkeypatch.setattr(
        rating_integration, "evaluate_side_onset", _active_outside_onset
    )
    with pytest.raises(
        CircularFinnedTubeWetSurfaceNotSupportedError,
        match="dry single-phase operation.*unsupported",
    ):
        hx.rate(
            BalanceSideSpec(
                provider=inside.provider,
                p=P,
                m_dot=M_DOT_INSIDE,
                T_in=T_INSIDE,
                T_out=dry_result.T_out_inside,
            ),
            BalanceSideSpec(
                provider=outside.provider,
                p=P,
                m_dot=M_DOT_OUTSIDE,
                T_in=T_OUTSIDE,
                T_out=dry_result.T_out_outside,
            ),
        )

    dry_inside, dry_outside = _side_inputs(mode=PhaseChangeMode.DISABLED)
    disabled = hx.simulate(dry_inside, dry_outside)
    assert disabled.phase_change_active is False
    assert disabled.finned_tube_diagnostics is not None


def test_finned_subcooled_water_preheat_uses_surface_dispatch_not_bare_probe(
    monkeypatch,
) -> None:
    hx = BareTubeHeatExchanger(_bundle(finned=True))
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(),
        m_dot=0.1,
        T_in=300.0,
        p=1.0e6,
    )
    outside = HXSideInput(
        provider=ConstantPropertyProvider(OUTSIDE_TRANSPORT),
        m_dot=2.0,
        T_in=500.0,
        p=P,
    )

    def bare_water_evaporator_probe_must_not_run(*_args, **_kwargs):
        raise AssertionError("legacy bare water-evaporator precheck was entered")

    monkeypatch.setattr(
        "core.phase_change.steam_integration.rate_water_evaporator",
        bare_water_evaporator_probe_must_not_run,
    )
    result = hx.simulate(inside, outside)
    assert result.phase_change_active is False
    assert result.T_out_inside < inside.provider.state(x=0.0, p=inside.p).T
    assert result.finned_tube_diagnostics is not None

    crossing_inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(),
        m_dot=0.01,
        T_in=300.0,
        p=1.0e6,
    )
    with pytest.raises(CircularFinnedTubeWetSurfaceNotSupportedError):
        hx.simulate(crossing_inside, outside)


def test_surface_margin_is_used_by_finned_water_phase_boundary_probe() -> None:
    hx = BareTubeHeatExchanger(_bundle(finned=True))
    inside = HXSideInput(
        provider=IAPWS97WaterSteamProvider(),
        m_dot=0.07,
        T_in=300.0,
        p=1.0e6,
    )
    outside = HXSideInput(
        provider=ConstantPropertyProvider(OUTSIDE_TRANSPORT),
        m_dot=2.0,
        T_in=500.0,
        p=P,
    )

    result = hx.simulate(inside, outside, surface_margin=1.0)
    assert result.phase_change_active is False
    assert result.T_out_inside < inside.provider.state(x=0.0, p=inside.p).T
    assert result.finned_tube_diagnostics is not None


def test_public_workflow_warnings_are_deduplicated() -> None:
    hx = BareTubeHeatExchanger(_bundle(finned=True))
    inside, outside = _side_inputs()
    simulation = hx.simulate(inside, outside)
    warning_keys = [
        (warning.source, warning.code) for warning in (simulation.warnings or [])
    ]
    assert len(warning_keys) == len(set(warning_keys))

    rating = hx.rate(
        BalanceSideSpec(
            provider=inside.provider,
            p=P,
            m_dot=M_DOT_INSIDE,
            T_in=T_INSIDE,
            T_out=simulation.T_out_inside,
        ),
        BalanceSideSpec(
            provider=outside.provider,
            p=P,
            m_dot=M_DOT_OUTSIDE,
            T_in=T_OUTSIDE,
            T_out=simulation.T_out_outside,
        ),
    )
    rating_warning_keys = [
        (warning.source, warning.code) for warning in (rating.warnings or [])
    ]
    assert len(rating_warning_keys) == len(set(rating_warning_keys))


def test_welded_contact_resistance_affects_only_the_parallel_fin_branch() -> None:
    core = _core_tube()
    welded = CircularFinnedTube(
        core_tube=core,
        fin_k=200.0,
        D_fin=0.047,
        D_root=core.D_o,
        fin_thickness_root=0.0005,
        fin_pitch=0.0024,
        fin_contact_resistance=1.0e6,
    )
    pitch_transverse = 0.060
    bundle = TubeBundle(
        tube=welded,
        n_rows=6,
        n_tubes_per_row=8,
        pitch_transverse=pitch_transverse,
        pitch_longitudinal=math.sqrt(3.0) * pitch_transverse / 2.0,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement="counterflow",
    )
    hx = BareTubeHeatExchanger(bundle)
    network = calculate_resistance_network(
        bundle=bundle,
        alpha_inside=1000.0,
        alpha_outside=50.0,
        resistance_core_wall=hx.tube_wall_resistance(),
    )

    primary_only_limit = 1.0 / (
        network.resistance_inside
        + network.resistance_core_wall
        + 1.0 / (50.0 * network.area_primary_outside)
    )
    assert network.contact_topology == "fin_branch_only"
    assert network.contact_area_basis == "periodic_fin_root_footprints"
    assert network.conductance_primary_outside > 0.0
    assert network.conductance_fin_outside > 0.0
    assert network.UA == pytest.approx(primary_only_limit, rel=1.0e-8)
    assert network.UA > 0.0


def test_primary_only_area_override_is_valid_for_welded_topology() -> None:
    core = _core_tube()
    welded = CircularFinnedTube(
        core_tube=core,
        fin_k=200.0,
        D_fin=0.047,
        D_root=core.D_o,
        fin_thickness_root=0.0005,
        fin_pitch=0.0024,
        fin_contact_resistance=2.0e-4,
    )
    welded = replace(
        welded,
        external_area_per_length=welded.primary_outside_area_per_length,
    )
    pitch_transverse = 0.060
    bundle = TubeBundle(
        tube=welded,
        n_rows=6,
        n_tubes_per_row=8,
        pitch_transverse=pitch_transverse,
        pitch_longitudinal=math.sqrt(3.0) * pitch_transverse / 2.0,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement="counterflow",
    )

    network = calculate_resistance_network(
        bundle=bundle,
        alpha_inside=800.0,
        alpha_outside=50.0,
        resistance_core_wall=0.001,
    )

    assert network.area_fin == 0.0
    assert network.conductance_fin_outside == 0.0
    assert network.resistance_outside == pytest.approx(
        1.0 / (50.0 * network.area_primary_outside)
    )


def test_extruded_root_and_positive_contact_are_common_series_terms() -> None:
    tube = _finned_tube(contact=2.0e-4)
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
    network = calculate_resistance_network(
        bundle=bundle,
        alpha_inside=800.0,
        alpha_outside=50.0,
        resistance_core_wall=0.001,
    )

    expected_outside = 1.0 / (
        50.0
        * (
            network.area_primary_outside
            + network.fin_efficiency * network.area_fin
        )
    )
    expected_total = (
        network.resistance_inside
        + network.resistance_core_wall
        + network.resistance_root
        + network.resistance_contact
        + expected_outside
    )
    assert network.contact_topology == (
        "series_before_primary_and_fin_parallel_branches"
    )
    assert network.contact_area_basis == "complete_core_root_layer_interface"
    assert network.resistance_root > 0.0
    assert network.resistance_contact > 0.0
    assert network.resistance_outside == pytest.approx(expected_outside)
    assert network.resistance_total == pytest.approx(expected_total)
