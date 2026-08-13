# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Resistance-network tests for circular finned tubes (v0.7.x)."""

from __future__ import annotations

import math

import pytest

from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.tube import BareTube
from core.heat_transfer.finned_tube_resistance import build_finned_tube_resistance_network


def _core_tube() -> BareTube:
    return BareTube(D_i=0.020, D_o=0.025, length_total=3.0, length_effective=3.0, wall_k=45.0)


def _welded_tube(**overrides) -> CircularFinnedTube:
    kwargs = dict(
        core_tube=_core_tube(),
        fin_k=200.0,
        D_fin=0.057,
        D_root=0.025,
        fin_thickness_root=0.0004,
        fin_pitch=0.0023,
    )
    kwargs.update(overrides)
    return CircularFinnedTube(**kwargs)


def test_network_all_resistances_positive_and_finite():
    tube = _welded_tube()
    net, warnings = build_finned_tube_resistance_network(tube, n_tubes=20, alfa_i=3000.0, alfa_o_physical=80.0)
    for value in (
        net.R_inside_convection,
        net.R_wall_conduction,
        net.R_primary_convection,
        net.R_fin_convection,
        net.R_outside_convection,
        net.R_total,
        net.UA,
    ):
        assert math.isfinite(value) and value > 0.0
    assert net.R_root_conduction == 0.0  # welded fin: D_root == D_o
    assert net.R_contact == 0.0  # contact resistance unknown -> ideal fallback


def test_contact_resistance_none_emits_warning():
    tube = _welded_tube()
    _, warnings = build_finned_tube_resistance_network(tube, n_tubes=20, alfa_i=3000.0, alfa_o_physical=80.0)
    codes = [w.code for w in warnings]
    assert "finned_tube_contact_resistance_unknown" in codes


def test_explicit_zero_contact_resistance_emits_no_warning():
    tube = _welded_tube(fin_contact_resistance=0.0)
    _, warnings = build_finned_tube_resistance_network(tube, n_tubes=20, alfa_i=3000.0, alfa_o_physical=80.0)
    codes = [w.code for w in warnings]
    assert "finned_tube_contact_resistance_unknown" not in codes


def test_positive_contact_resistance_increases_r_contact_and_total():
    tube_ideal = _welded_tube(fin_contact_resistance=0.0)
    tube_real = _welded_tube(fin_contact_resistance=0.0001)
    net_ideal, _ = build_finned_tube_resistance_network(tube_ideal, n_tubes=20, alfa_i=3000.0, alfa_o_physical=80.0)
    net_real, _ = build_finned_tube_resistance_network(tube_real, n_tubes=20, alfa_i=3000.0, alfa_o_physical=80.0)
    assert net_real.R_contact > net_ideal.R_contact == 0.0
    assert net_real.R_total > net_ideal.R_total
    assert net_real.UA < net_ideal.UA


def test_extruded_root_adds_root_conduction_resistance():
    tube = _welded_tube(D_root=0.0265)
    net, _ = build_finned_tube_resistance_network(tube, n_tubes=20, alfa_i=3000.0, alfa_o_physical=80.0)
    assert net.R_root_conduction > 0.0


def test_fin_efficiency_and_contact_resistance_not_double_counted():
    """R_outside_convection must equal the exact parallel combination of
    primary and fin paths -- not a further eta_o-weighted duplication."""
    tube = _welded_tube()
    net, _ = build_finned_tube_resistance_network(tube, n_tubes=20, alfa_i=3000.0, alfa_o_physical=80.0)
    expected = 1.0 / (1.0 / net.R_primary_convection + 1.0 / net.R_fin_convection)
    assert net.R_outside_convection == pytest.approx(expected, rel=1e-12)
    # UA computed directly from alfa_o_gross_basis on the gross area must
    # reproduce the same R_outside_convection (single conversion, no
    # double efficiency weighting)
    reconstructed_R_o = 1.0 / (net.alfa_o_gross_basis * net.A_outside_used)
    assert reconstructed_R_o == pytest.approx(net.R_outside_convection, rel=1e-9)


def test_ua_matches_direct_series_sum_of_resistances():
    tube = _welded_tube()
    net, _ = build_finned_tube_resistance_network(tube, n_tubes=20, alfa_i=3000.0, alfa_o_physical=80.0)
    R_total = (
        net.R_inside_convection
        + net.R_wall_conduction
        + net.R_root_conduction
        + net.R_contact
        + net.R_outside_convection
    )
    assert net.UA == pytest.approx(1.0 / R_total, rel=1e-12)


def test_external_area_override_large_difference_warns():
    tube = _welded_tube()
    geometric_per_length = tube.A_outside_geometric / tube.length_effective
    tube_override = _welded_tube(external_area_per_length=geometric_per_length * 1.5)
    _, warnings = build_finned_tube_resistance_network(tube_override, n_tubes=20, alfa_i=3000.0, alfa_o_physical=80.0)
    codes = [w.code for w in warnings]
    assert "finned_tube_external_area_override_large_difference" in codes


def test_rejects_nonpositive_alfa():
    tube = _welded_tube()
    with pytest.raises(ValueError):
        build_finned_tube_resistance_network(tube, n_tubes=20, alfa_i=0.0, alfa_o_physical=80.0)
    with pytest.raises(ValueError):
        build_finned_tube_resistance_network(tube, n_tubes=20, alfa_i=3000.0, alfa_o_physical=0.0)
