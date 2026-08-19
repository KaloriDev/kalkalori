"""Focused coverage for fin-surface / outside-skin temperature diagnostics.

These tests exercise the low-level ``fin_surface_temperatures`` helper
directly (for exact control over the core-wall/bulk/heat-rate inputs used by
each contact topology) and the full public ``Simulation`` result surface
(for end-to-end wiring: ``FinnedTubeDiagnostics.fin_tip_temperature_ratio``
and ``WallTemperatureEnvelope`` skin/fin-base/fin-tip aggregation).
"""

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, CircularFinnedTube, TubeOrientation
from core.heat_transfer.fin_efficiency import annular_fin_efficiency
from core.heat_transfer.outside_dispatch import calculate_resistance_network
from core.heat_transfer.thermal_iteration import fin_surface_temperatures
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


P = 101_325.0
M_DOT_INSIDE = 1.5
M_DOT_OUTSIDE = 2.0
INSIDE_TRANSPORT = FluidTransportProperties(rho=900.0, mu=5.0e-4, k=0.60, cp=4000.0)
OUTSIDE_TRANSPORT = FluidTransportProperties(rho=1.10, mu=1.90e-5, k=0.028, cp=1007.0)


def _core_tube() -> BareTube:
    return BareTube(
        D_i=0.021, D_o=0.025, length_total=2.0, length_effective=2.0,
        wall_k=45.0, tube_orientation=TubeOrientation.VERTICAL_UPWARD,
    )


def _finned_tube(*, contact: float | None, D_root: float) -> CircularFinnedTube:
    return CircularFinnedTube(
        core_tube=_core_tube(), fin_k=200.0, D_fin=0.050, D_root=D_root,
        fin_thickness_root=0.0005, fin_pitch=0.0024, fin_contact_resistance=contact,
    )


def _bundle(tube) -> TubeBundle:
    pitch_transverse = 0.060
    return TubeBundle(
        tube=tube, n_rows=6, n_tubes_per_row=8,
        pitch_transverse=pitch_transverse,
        pitch_longitudinal=pitch_transverse * math.sqrt(3.0) / 2.0,
        layout="staggered", n_passes_tube=1, flow_arrangement="counterflow",
    )


def _hx(tube) -> BareTubeHeatExchanger:
    return BareTubeHeatExchanger(_bundle(tube))


def _sides(*, T_inside=360.0, T_outside=300.0):
    return (
        HXSideInput(provider=ConstantPropertyProvider(INSIDE_TRANSPORT), m_dot=M_DOT_INSIDE, T_in=T_inside, p=P),
        HXSideInput(provider=ConstantPropertyProvider(OUTSIDE_TRANSPORT), m_dot=M_DOT_OUTSIDE, T_in=T_outside, p=P),
    )


def _network(tube, *, alpha_inside=500.0, alpha_outside=60.0, resistance_core_wall=0.0):
    return calculate_resistance_network(
        bundle=_bundle(tube), alpha_inside=alpha_inside,
        outside_alpha_physical=alpha_outside, resistance_core_wall=resistance_core_wall,
    )


# ---------------------------------------------------------------------------
# S1: existing fin ratio pass-through (end-to-end through the public result)
# ---------------------------------------------------------------------------

def test_fin_tip_ratio_matches_production_fin_solver():
    tube = _finned_tube(contact=1.0e-4, D_root=0.028)
    hx = _hx(tube)
    inside, outside = _sides()
    result = hx.simulate(inside, outside)
    diagnostics = result.final_result.finned_tube_diagnostics
    assert diagnostics is not None
    direct = annular_fin_efficiency(tube, diagnostics.outside_alpha_physical)
    assert diagnostics.fin_tip_temperature_ratio == pytest.approx(
        direct.tip_temperature_ratio, rel=1.0e-12
    )


# ---------------------------------------------------------------------------
# S2: ideal welded contact (D_root == D_o, R_contact = 0)
# ---------------------------------------------------------------------------

def test_ideal_welded_contact_fin_base_equals_core_wall():
    tube = _finned_tube(contact=0.0, D_root=0.025)  # D_root == D_o
    network = _network(tube)
    assert network.contact_topology == "fin_branch_only"
    T_core, T_bulk, heat_rate = 350.0, 300.0, 1000.0
    primary, fin_base, fin_tip, skin_min, skin_max = fin_surface_temperatures(
        network=network, outside_wall_temperature=T_core,
        outside_bulk_temperature=T_bulk, heat_rate=heat_rate,
    )
    assert primary == pytest.approx(T_core, rel=1.0e-12)
    assert fin_base == pytest.approx(T_core, rel=1.0e-12)
    expected_tip = T_bulk + network.fin_efficiency_result.tip_temperature_ratio * (fin_base - T_bulk)
    assert fin_tip == pytest.approx(expected_tip, rel=1.0e-12)


# ---------------------------------------------------------------------------
# S3: finite welded contact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("contact", [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
def test_finite_welded_contact_ordering_hot_inside(contact):
    tube = _finned_tube(contact=contact, D_root=0.025)
    network = _network(tube)
    T_core, T_bulk, heat_rate = 350.0, 300.0, 1000.0
    tol = 1.0e-9
    primary, fin_base, fin_tip, _, _ = fin_surface_temperatures(
        network=network, outside_wall_temperature=T_core,
        outside_bulk_temperature=T_bulk, heat_rate=heat_rate,
    )
    assert primary == pytest.approx(T_core, rel=1.0e-12)
    assert T_core + tol >= fin_base >= fin_tip >= T_bulk - tol


def test_increasing_contact_resistance_moves_fin_toward_outside_bulk():
    T_core, T_bulk, heat_rate = 350.0, 300.0, 1000.0
    tube_small = _finned_tube(contact=1.0e-5, D_root=0.025)
    tube_large = _finned_tube(contact=1.0e-2, D_root=0.025)
    _, base_small, tip_small, _, _ = fin_surface_temperatures(
        network=_network(tube_small), outside_wall_temperature=T_core,
        outside_bulk_temperature=T_bulk, heat_rate=heat_rate,
    )
    _, base_large, tip_large, _, _ = fin_surface_temperatures(
        network=_network(tube_large), outside_wall_temperature=T_core,
        outside_bulk_temperature=T_bulk, heat_rate=heat_rate,
    )
    assert base_large < base_small
    assert tip_large < tip_small


# ---------------------------------------------------------------------------
# S4: continuous root-layer topology (D_root > D_o)
# ---------------------------------------------------------------------------

def test_root_layer_topology_matches_formula():
    tube = _finned_tube(contact=1.0e-4, D_root=0.028)  # D_root > D_o (0.025)
    network = _network(tube)
    assert network.contact_topology == "series_before_primary_and_fin_parallel_branches"
    T_core, T_bulk, heat_rate = 350.0, 300.0, 1000.0
    primary, fin_base, fin_tip, _, _ = fin_surface_temperatures(
        network=network, outside_wall_temperature=T_core,
        outside_bulk_temperature=T_bulk, heat_rate=heat_rate,
    )
    R_common = network.resistance_root + network.resistance_contact
    expected_root_surface = T_core - heat_rate * R_common
    assert primary == pytest.approx(expected_root_surface, rel=1.0e-12)
    assert fin_base == pytest.approx(expected_root_surface, rel=1.0e-12)
    assert R_common > 0.0  # sanity: the root layer actually contributes


# ---------------------------------------------------------------------------
# S5: reverse heat direction (outside hotter than inside)
# ---------------------------------------------------------------------------

def test_reverse_heat_direction_ordering_reverses():
    tube = _finned_tube(contact=1.0e-4, D_root=0.025)
    network = _network(tube)
    tol = 1.0e-9

    _, base_hot_inside, tip_hot_inside, _, _ = fin_surface_temperatures(
        network=network, outside_wall_temperature=350.0,
        outside_bulk_temperature=300.0, heat_rate=1000.0,
    )
    assert 350.0 + tol >= base_hot_inside >= tip_hot_inside >= 300.0 - tol

    _, base_hot_outside, tip_hot_outside, _, _ = fin_surface_temperatures(
        network=network, outside_wall_temperature=300.0,
        outside_bulk_temperature=350.0, heat_rate=-1000.0,
    )
    assert 300.0 - tol <= base_hot_outside <= tip_hot_outside <= 350.0 + tol


# ---------------------------------------------------------------------------
# S6: skin extrema
# ---------------------------------------------------------------------------

def test_finned_skin_extrema_are_min_max_of_exposed_temperatures():
    tube = _finned_tube(contact=5.0e-4, D_root=0.025)
    network = _network(tube)
    primary, fin_base, fin_tip, skin_min, skin_max = fin_surface_temperatures(
        network=network, outside_wall_temperature=350.0,
        outside_bulk_temperature=300.0, heat_rate=1000.0,
    )
    assert skin_min == pytest.approx(min(primary, fin_base, fin_tip))
    assert skin_max == pytest.approx(max(primary, fin_base, fin_tip))


def test_bare_tube_skin_equals_existing_outside_wall_envelope():
    tube = _core_tube()
    hx = _hx(tube)
    inside, outside = _sides()
    result = hx.simulate(inside, outside)
    envelope = result.wall_temperature_envelope
    assert envelope.outside_skin_min == pytest.approx(envelope.outside_min, rel=1.0e-12)
    assert envelope.outside_skin_max == pytest.approx(envelope.outside_max, rel=1.0e-12)
    assert math.isnan(envelope.fin_base_min)
    assert math.isnan(envelope.fin_base_max)
    assert math.isnan(envelope.fin_tip_min)
    assert math.isnan(envelope.fin_tip_max)
    for probe in envelope.probes:
        assert probe.outside_primary_surface_temperature is None
        assert probe.fin_base_temperature is None
        assert probe.fin_tip_temperature is None
        assert probe.outside_skin_temperature_min == pytest.approx(probe.outside_wall_temperature)
        assert probe.outside_skin_temperature_max == pytest.approx(probe.outside_wall_temperature)
    assert result.final_result.finned_tube_diagnostics is None


# ---------------------------------------------------------------------------
# S7: no thermal-result regression -- new diagnostics are purely additive
# ---------------------------------------------------------------------------

def test_new_diagnostics_do_not_disturb_existing_thermal_fields():
    tube = _finned_tube(contact=1.0e-4, D_root=0.025)
    hx = _hx(tube)
    inside, outside = _sides()
    result = hx.simulate(inside, outside)
    diagnostics = result.final_result.finned_tube_diagnostics
    reference = _network(
        tube,
        alpha_inside=result.thermal_state.alfa_i,
        alpha_outside=diagnostics.outside_alpha_physical,
        resistance_core_wall=hx.tube_wall_resistance(),
    )
    assert diagnostics.UA == pytest.approx(reference.UA, rel=1.0e-9)
    assert diagnostics.fin_efficiency == pytest.approx(reference.fin_efficiency, rel=1.0e-9)
    assert diagnostics.outside_alpha_effective_gross == pytest.approx(
        reference.outside_alpha_effective_gross, rel=1.0e-9
    )
    assert math.isfinite(result.q)
    assert math.isfinite(result.outside_dp_total)


# ---------------------------------------------------------------------------
# Public Simulation/Rating properties (section 7I)
# ---------------------------------------------------------------------------

def test_public_simulation_properties_expose_skin_and_fin_estimates():
    tube = _finned_tube(contact=1.0e-4, D_root=0.025)
    hx = _hx(tube)
    inside, outside = _sides()
    result = hx.simulate(inside, outside)
    envelope = result.wall_temperature_envelope
    assert result.outside_skin_temperature_min_estimate == pytest.approx(envelope.outside_skin_min)
    assert result.outside_skin_temperature_max_estimate == pytest.approx(envelope.outside_skin_max)
    assert result.fin_base_temperature_min_estimate == pytest.approx(envelope.fin_base_min)
    assert result.fin_base_temperature_max_estimate == pytest.approx(envelope.fin_base_max)
    assert result.fin_tip_temperature_min_estimate == pytest.approx(envelope.fin_tip_min)
    assert result.fin_tip_temperature_max_estimate == pytest.approx(envelope.fin_tip_max)
    # Existing endpoint envelope semantics stay untouched by the new fields.
    assert result.outside_wall_temperature_min_estimate == pytest.approx(envelope.outside_min)
    assert result.outside_wall_temperature_max_estimate == pytest.approx(envelope.outside_max)


def test_public_simulation_properties_are_none_for_bare_tube_fin_fields():
    tube = _core_tube()
    hx = _hx(tube)
    inside, outside = _sides()
    result = hx.simulate(inside, outside)
    assert result.fin_base_temperature_min_estimate is None
    assert result.fin_base_temperature_max_estimate is None
    assert result.fin_tip_temperature_min_estimate is None
    assert result.fin_tip_temperature_max_estimate is None
    assert result.outside_skin_temperature_min_estimate == pytest.approx(
        result.outside_wall_temperature_min_estimate
    )
    assert result.outside_skin_temperature_max_estimate == pytest.approx(
        result.outside_wall_temperature_max_estimate
    )


def test_steam_heater_rating_exposes_fin_surface_diagnostics_on_finned_tube():
    from core.models.heat_balance import BalanceSideSpec
    from core.properties.water import IAPWS97WaterSteamProvider

    core = BareTube(
        D_i=0.021, D_o=0.025, length_total=2.0, length_effective=2.0,
        wall_k=45.0, tube_orientation=TubeOrientation.VERTICAL_DOWNWARD,
    )
    tube = CircularFinnedTube(
        core_tube=core, fin_k=200.0, D_fin=0.050, D_root=0.025,
        fin_thickness_root=0.0005, fin_pitch=0.0024, fin_contact_resistance=1.0e-4,
    )
    bundle = TubeBundle(
        tube=tube, n_rows=6, n_tubes_per_row=8,
        pitch_transverse=0.060, pitch_longitudinal=0.060 * math.sqrt(3.0) / 2.0,
        layout="staggered", n_passes_tube=1, flow_arrangement="crossflow",
    )
    hx = BareTubeHeatExchanger(bundle)
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(), p=6.0e5, m_dot=0.05,
        quality_in=1.0, quality_out=0.0,
    )
    outside = BalanceSideSpec(
        provider=ConstantPropertyProvider(OUTSIDE_TRANSPORT), p=P, m_dot=10.0, T_in=300.0,
    )
    result = hx.rate(inside, outside)
    envelope = result.wall_temperature_envelope
    assert math.isfinite(envelope.fin_base_min)
    assert math.isfinite(envelope.fin_tip_min)
    assert result.fin_base_temperature_min_estimate is not None
    assert result.fin_tip_temperature_min_estimate is not None
    assert result.outside_skin_temperature_min_estimate <= result.outside_skin_temperature_max_estimate
