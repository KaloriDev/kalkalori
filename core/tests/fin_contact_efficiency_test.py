# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Focused coverage for the v0.7.2 dimensionless fin-contact-efficiency input.

Test IDs T1-T12 below correspond to the v0.7.2 task specification. Expected
values for T3, T7 and T8 are computed here from first principles (not by
calling ``calculate_resistance_network`` for the expectation itself), so
these are independent analytical checks rather than circular ones.
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.tube import BareTube, TubeOrientation
from core.heat_transfer.outside_dispatch import (
    calculate_resistance_network,
    evaluate_outside_hydraulics,
    evaluate_outside_thermal,
)
from core.heat_transfer.outside_flow import FluidProps as OutsideFluidProps
from core.heat_transfer.thermal_iteration import _evaluate_local_wall_state
from core.models.bare_tube import BareTubeHeatExchanger
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


ALPHA_INSIDE = 800.0
ALPHA_OUTSIDE = 50.0
RESISTANCE_CORE_WALL = 0.001

INSIDE_TRANSPORT = FluidTransportProperties(rho=900.0, mu=5.0e-4, k=0.60, cp=4000.0)
OUTSIDE_TRANSPORT = FluidTransportProperties(rho=1.10, mu=1.90e-5, k=0.028, cp=1007.0)


def _core_tube() -> BareTube:
    return BareTube(
        D_i=0.021,
        D_o=0.025,
        length_total=2.0,
        length_effective=2.0,
        wall_k=45.0,
        tube_orientation=TubeOrientation.VERTICAL_UPWARD,
    )


def _welded_tube(
    *, resistance: float | None = None, efficiency: float | None = None
) -> CircularFinnedTube:
    core = _core_tube()
    return CircularFinnedTube(
        core_tube=core,
        fin_k=200.0,
        D_fin=0.047,
        D_root=core.D_o,
        fin_thickness_root=0.0005,
        fin_pitch=0.0024,
        fin_contact_resistance=resistance,
        fin_contact_efficiency=efficiency,
    )


def _extruded_tube(
    *, resistance: float | None = None, efficiency: float | None = None
) -> CircularFinnedTube:
    return CircularFinnedTube(
        core_tube=_core_tube(),
        fin_k=200.0,
        D_fin=0.050,
        D_root=0.028,
        fin_thickness_root=0.0005,
        fin_pitch=0.0024,
        fin_contact_resistance=resistance,
        fin_contact_efficiency=efficiency,
    )


def _bundle(tube: CircularFinnedTube) -> TubeBundle:
    pitch_transverse = 0.060
    return TubeBundle(
        tube=tube,
        n_rows=6,
        n_tubes_per_row=8,
        pitch_transverse=pitch_transverse,
        pitch_longitudinal=math.sqrt(3.0) * pitch_transverse / 2.0,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement="counterflow",
    )


def _network(tube: CircularFinnedTube):
    return calculate_resistance_network(
        bundle=_bundle(tube),
        alpha_inside=ALPHA_INSIDE,
        outside_alpha_physical=ALPHA_OUTSIDE,
        resistance_core_wall=RESISTANCE_CORE_WALL,
    )


# ---------------------------------------------------------------------------
# T1 - default omitted contact inputs give ideal contact, no legacy warning
# ---------------------------------------------------------------------------


def test_t1_default_omitted_contact_gives_ideal_contact_no_warning() -> None:
    for factory in (_welded_tube, _extruded_tube):
        tube = factory()
        assert tube.geometry_warnings == ()
        network = _network(tube)
        assert network.resistance_contact == 0.0
        assert network.contact_input_mode == "ideal_default"
        assert network.fin_contact_efficiency_effective == 1.0
        assert network.contact_resistance_equivalent_areal == 0.0
        assert not any(
            w.code == "circular_finned_tube_contact_resistance_unspecified"
            for w in tube.geometry_warnings
        )


# ---------------------------------------------------------------------------
# T2 - explicit fin_contact_resistance=0.0 preserves ideal contact exactly
# ---------------------------------------------------------------------------


def test_t2_explicit_zero_resistance_matches_ideal_contact_exactly() -> None:
    for factory in (_welded_tube, _extruded_tube):
        ideal = _network(factory())
        explicit_zero = _network(factory(resistance=0.0))
        assert explicit_zero.resistance_contact == 0.0
        assert explicit_zero.contact_input_mode == "explicit_resistance"
        assert explicit_zero.resistance_outside == ideal.resistance_outside
        assert explicit_zero.UA == ideal.UA


# ---------------------------------------------------------------------------
# T3 - explicit positive resistance regression reproduces v0.7.1 equations
# ---------------------------------------------------------------------------


def test_t3_explicit_positive_resistance_matches_v071_welded_equations() -> None:
    resistance_area = 2.0e-4
    network = _network(_welded_tube(resistance=resistance_area))

    G_fin_ideal = ALPHA_OUTSIDE * network.fin_efficiency * network.area_fin
    R_contact_abs = resistance_area / network.contact_area
    G_fin_expected = 1.0 / (R_contact_abs + 1.0 / G_fin_ideal)
    G_primary = ALPHA_OUTSIDE * network.area_primary_outside
    R_outside_expected = 1.0 / (G_primary + G_fin_expected)

    assert network.conductance_fin_outside == pytest.approx(G_fin_expected)
    assert network.resistance_outside == pytest.approx(R_outside_expected)
    assert network.contact_resistance_used == resistance_area
    assert network.contact_resistance_equivalent_areal == resistance_area


def test_t3_explicit_positive_resistance_matches_v071_continuous_root_equations() -> None:
    resistance_area = 2.0e-4
    network = _network(_extruded_tube(resistance=resistance_area))

    R_root = network.resistance_root
    G_primary = ALPHA_OUTSIDE * network.area_primary_outside
    G_fin_ideal = ALPHA_OUTSIDE * network.fin_efficiency * network.area_fin
    R_branches = 1.0 / (G_primary + G_fin_ideal)
    R_contact_abs = resistance_area / network.contact_area
    R_outside_expected = R_contact_abs + R_root + R_branches

    assert network.resistance_outside_branches == pytest.approx(R_branches)
    assert network.resistance_outside == pytest.approx(R_outside_expected)
    assert network.contact_resistance_used == resistance_area


# ---------------------------------------------------------------------------
# T4 - efficiency=1.0 equals explicit zero resistance, both topologies
# ---------------------------------------------------------------------------


def test_t4_efficiency_one_equals_explicit_zero_resistance_welded() -> None:
    zero = _network(_welded_tube(resistance=0.0))
    eff_one = _network(_welded_tube(efficiency=1.0))
    assert eff_one.resistance_contact == zero.resistance_contact == 0.0
    assert eff_one.contact_input_mode == "contact_efficiency"
    assert eff_one.resistance_outside == pytest.approx(zero.resistance_outside)
    assert eff_one.UA == pytest.approx(zero.UA)


def test_t4_efficiency_one_equals_explicit_zero_resistance_continuous_root() -> None:
    zero = _network(_extruded_tube(resistance=0.0))
    eff_one = _network(_extruded_tube(efficiency=1.0))
    assert eff_one.resistance_contact == zero.resistance_contact == 0.0
    assert eff_one.resistance_outside == pytest.approx(zero.resistance_outside)
    assert eff_one.UA == pytest.approx(zero.UA)


# ---------------------------------------------------------------------------
# T5 - welded topology: efficiency scales only the fin branch
# ---------------------------------------------------------------------------


def test_t5_welded_topology_efficiency_scales_only_fin_branch() -> None:
    ideal = _network(_welded_tube())
    degraded = _network(_welded_tube(efficiency=0.8))
    G_fin_ideal = ALPHA_OUTSIDE * ideal.fin_efficiency * ideal.area_fin

    assert degraded.conductance_fin_outside == pytest.approx(0.8 * G_fin_ideal)
    assert degraded.conductance_primary_outside == pytest.approx(
        ideal.conductance_primary_outside
    )


# ---------------------------------------------------------------------------
# T6 - continuous-root topology: efficiency scales the whole downstream path
# ---------------------------------------------------------------------------


def test_t6_continuous_root_efficiency_scales_whole_downstream_path() -> None:
    ideal = _network(_extruded_tube())
    degraded = _network(_extruded_tube(efficiency=0.8))

    R_path_ideal = ideal.resistance_root + ideal.resistance_outside_branches
    G_path_ideal = 1.0 / R_path_ideal
    # The path before adding the derived contact resistance is unaffected by
    # contact efficiency: only the extra series term changes.
    assert degraded.resistance_root == pytest.approx(ideal.resistance_root)
    assert degraded.resistance_outside_branches == pytest.approx(
        ideal.resistance_outside_branches
    )
    R_path_actual = (
        degraded.resistance_root
        + degraded.resistance_contact
        + degraded.resistance_outside_branches
    )
    G_path_actual = 1.0 / R_path_actual
    assert G_path_actual == pytest.approx(0.8 * G_path_ideal)


# ---------------------------------------------------------------------------
# T7 - welded efficiency mode: independent analytical R_contact_equiv
# ---------------------------------------------------------------------------


def test_t7_welded_efficiency_matches_independent_contact_resistance_formula() -> None:
    eta = 0.8
    ideal = _network(_welded_tube())
    degraded = _network(_welded_tube(efficiency=eta))
    G_fin_ideal = ALPHA_OUTSIDE * ideal.fin_efficiency * ideal.area_fin

    expected_R_contact_equiv = (1.0 / eta - 1.0) / G_fin_ideal
    assert degraded.resistance_contact == pytest.approx(expected_R_contact_equiv)


# ---------------------------------------------------------------------------
# T8 - continuous-root efficiency mode: independent analytical R_contact_equiv
# ---------------------------------------------------------------------------


def test_t8_continuous_root_efficiency_matches_independent_contact_resistance_formula() -> None:
    eta = 0.8
    ideal = _network(_extruded_tube())
    degraded = _network(_extruded_tube(efficiency=eta))

    R_path_ideal = ideal.resistance_root + ideal.resistance_outside_branches
    expected_R_contact_equiv = (1.0 / eta - 1.0) * R_path_ideal
    assert degraded.resistance_contact == pytest.approx(expected_R_contact_equiv)


# ---------------------------------------------------------------------------
# T9 - explicit zero beats efficiency; efficiency reported as ignored
# ---------------------------------------------------------------------------


def test_t9_explicit_zero_beats_efficiency_and_is_reported_ignored() -> None:
    for factory in (_welded_tube, _extruded_tube):
        tube = factory(resistance=0.0, efficiency=0.8)
        assert tube.contact_input_mode == "explicit_resistance"
        assert [w.code for w in tube.geometry_warnings] == [
            "circular_finned_tube_contact_efficiency_ignored"
        ]
        assert tube.geometry_warnings[0].severity == "info"

        network = _network(tube)
        ideal = _network(factory())
        assert network.resistance_contact == 0.0 == ideal.resistance_contact
        assert network.UA == pytest.approx(ideal.UA)
        assert network.fin_contact_efficiency_input == 0.8
        assert network.fin_contact_efficiency_effective == 1.0


# ---------------------------------------------------------------------------
# T10 - both inputs with positive explicit resistance reproduce it exactly
# ---------------------------------------------------------------------------


def test_t10_positive_explicit_resistance_beats_efficiency_exactly() -> None:
    resistance_area = 3.0e-4
    for factory in (_welded_tube, _extruded_tube):
        both = _network(factory(resistance=resistance_area, efficiency=0.5))
        explicit_only = _network(factory(resistance=resistance_area))
        assert both.resistance_contact == explicit_only.resistance_contact
        assert both.UA == explicit_only.UA
        tube_both = factory(resistance=resistance_area, efficiency=0.5)
        assert [w.code for w in tube_both.geometry_warnings] == [
            "circular_finned_tube_contact_efficiency_ignored"
        ]


# ---------------------------------------------------------------------------
# T11 - invalid fin_contact_efficiency values reject cleanly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, -0.1, 1.0000001, math.nan, math.inf, -math.inf])
def test_t11_invalid_fin_contact_efficiency_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="fin_contact_efficiency"):
        _welded_tube(efficiency=value)


def test_t11_invalid_fin_contact_efficiency_is_not_silently_clamped_from_percent() -> None:
    # 95 ("95%") must not be silently interpreted as 0.95.
    with pytest.raises(ValueError, match="fin_contact_efficiency"):
        _welded_tube(efficiency=95)


# ---------------------------------------------------------------------------
# T12 - efficiency affects only contact-mediated quantities
# ---------------------------------------------------------------------------


def test_t12_efficiency_leaves_physical_htc_areas_and_fin_efficiency_unchanged() -> None:
    for factory in (_welded_tube, _extruded_tube):
        ideal_tube = factory()
        degraded_tube = factory(efficiency=0.6)

        ideal_dispatch = evaluate_outside_thermal(
            bundle=_bundle(ideal_tube),
            m_dot=2.0,
            props=OutsideFluidProps(**OUTSIDE_TRANSPORT.__dict__),
        )
        degraded_dispatch = evaluate_outside_thermal(
            bundle=_bundle(degraded_tube),
            m_dot=2.0,
            props=OutsideFluidProps(**OUTSIDE_TRANSPORT.__dict__),
        )
        # outside_alpha_physical (and the Briggs-Young Re/Nu/j it is built
        # from) must not depend on contact efficiency at all.
        assert degraded_dispatch.alpha_physical == ideal_dispatch.alpha_physical
        assert degraded_dispatch.reynolds_number == ideal_dispatch.reynolds_number
        assert degraded_dispatch.nusselt_number == ideal_dispatch.nusselt_number
        assert (
            degraded_dispatch.finned_result.colburn_j_factor
            == ideal_dispatch.finned_result.colburn_j_factor
        )

        ideal_network = _network(ideal_tube)
        degraded_network = _network(degraded_tube)
        assert degraded_network.area_outside_geometric == pytest.approx(
            ideal_network.area_outside_geometric
        )
        assert degraded_network.area_primary_outside == pytest.approx(
            ideal_network.area_primary_outside
        )
        assert degraded_network.area_fin == pytest.approx(ideal_network.area_fin)
        assert degraded_network.fin_efficiency == pytest.approx(ideal_network.fin_efficiency)

        # Contact efficiency must, however, move UA/effective gross alpha.
        assert degraded_network.UA < ideal_network.UA
        assert (
            degraded_network.outside_alpha_effective_gross
            < ideal_network.outside_alpha_effective_gross
        )

        # Pressure drop is a pure bundle/hydraulic-provider computation with
        # no dependence on the contact input at all.
        ideal_hydraulics = evaluate_outside_hydraulics(
            bundle=_bundle(ideal_tube),
            m_dot=2.0,
            property_provider=ConstantPropertyProvider(OUTSIDE_TRANSPORT),
            temperature_in=300.0,
            temperature_out=320.0,
            pressure=101_325.0,
        )
        degraded_hydraulics = evaluate_outside_hydraulics(
            bundle=_bundle(degraded_tube),
            m_dot=2.0,
            property_provider=ConstantPropertyProvider(OUTSIDE_TRANSPORT),
            temperature_in=300.0,
            temperature_out=320.0,
            pressure=101_325.0,
        )
        assert degraded_hydraulics.dp_total == pytest.approx(ideal_hydraulics.dp_total)


def test_t12_efficiency_changes_fin_base_and_fin_tip_temperatures() -> None:
    for factory in (_welded_tube, _extruded_tube):
        results = {}
        for efficiency in (1.0, 0.6):
            tube = factory(efficiency=efficiency)
            bundle = _bundle(tube)
            hx = BareTubeHeatExchanger(bundle)
            inside_provider = ConstantPropertyProvider(INSIDE_TRANSPORT)
            outside_provider = ConstantPropertyProvider(OUTSIDE_TRANSPORT)
            local = _evaluate_local_wall_state(
                hx,
                m_dot_inside=1.5,
                m_dot_outside=2.0,
                inside_provider=inside_provider,
                outside_provider=outside_provider,
                inside_bulk_temperature=360.0,
                outside_bulk_temperature=300.0,
                p_inside=101_325.0,
                p_outside=101_325.0,
                inside_wall_temperature=None,
                outside_wall_temperature=None,
                euler_provider="zukauskas",
            )
            results[efficiency] = local

        assert results[1.0].fin_base_temperature != pytest.approx(
            results[0.6].fin_base_temperature
        )
        assert results[1.0].fin_tip_temperature != pytest.approx(
            results[0.6].fin_tip_temperature
        )
