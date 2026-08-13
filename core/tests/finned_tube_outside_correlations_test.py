# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Dry outside HTC/pressure-drop correlation tests for circular finned
tube banks (v0.7.x): Briggs & Young (1963) Nusselt correlation and the
Robinson & Briggs (1966) pressure-drop provider contract/blocker."""

from __future__ import annotations

import math

import pytest

from core.geometry import BareTube, CircularFinnedTube
from core.heat_transfer.outside_flow import FluidProps
from core.heat_transfer.outside_flow_finned import (
    HTC_CONTRACT,
    RE_MAX,
    RE_MIN,
    FinnedTubeUnsupportedLayoutError,
    check_finned_outside_ht_applicability,
    finned_outside_flow_from_mass_flow,
    nusselt_briggs_young,
)
from core.pressure_drop.outside_pressure_drop import (
    EulerRequest,
    GaddisGnielinskiEulerProvider,
    RobinsonBriggsEulerProvider,
    ZukauskasEulerProvider,
    evaluate_euler,
)


def _tube(**overrides) -> CircularFinnedTube:
    core = BareTube(D_i=0.020, D_o=0.025, length_total=3.0, length_effective=3.0, wall_k=45.0)
    kwargs = dict(
        core_tube=core,
        fin_k=200.0,
        D_fin=0.057,
        D_root=0.025,
        fin_thickness_root=0.0004,
        fin_pitch=0.0023,
    )
    kwargs.update(overrides)
    return CircularFinnedTube(**kwargs)


def _air_props() -> FluidProps:
    return FluidProps(rho=1.15, mu=1.9e-5, k=0.028, cp=1007.0)


# -------------------------------------------------------------------
# Briggs & Young (1963) Nusselt correlation -- independent reference point
# -------------------------------------------------------------------

def test_nusselt_briggs_young_matches_hand_computed_reference_point():
    Re, Pr = 5000.0, 0.7
    clear_spacing_root = 0.0023 - 0.0004  # fin_pitch - fin_thickness_root
    fin_height = 0.016
    fin_thickness_root = 0.0004

    # Computed independently of the production nusselt_briggs_young
    # implementation, directly from the documented equation:
    #   Nu = 0.134 * Re^0.681 * Pr^(1/3) * (s/l)^0.2 * (s/b)^0.1134
    expected = (
        0.134
        * (Re ** 0.681)
        * (Pr ** (1.0 / 3.0))
        * ((clear_spacing_root / fin_height) ** 0.2)
        * ((clear_spacing_root / fin_thickness_root) ** 0.1134)
    )
    actual = nusselt_briggs_young(
        Re, Pr,
        clear_spacing_root=clear_spacing_root,
        fin_height=fin_height,
        fin_thickness_root=fin_thickness_root,
    )
    assert actual == pytest.approx(expected, rel=1e-12)
    # sanity: known order of magnitude for this Re/geometry combination
    assert 20.0 < actual < 60.0


def test_nusselt_briggs_young_requires_positive_inputs():
    with pytest.raises(ValueError):
        nusselt_briggs_young(0.0, 0.7, clear_spacing_root=0.0019, fin_height=0.016, fin_thickness_root=0.0004)
    with pytest.raises(ValueError):
        nusselt_briggs_young(5000.0, 0.7, clear_spacing_root=-0.001, fin_height=0.016, fin_thickness_root=0.0004)


def test_higher_reynolds_number_increases_nusselt():
    kwargs = dict(clear_spacing_root=0.0019, fin_height=0.016, fin_thickness_root=0.0004)
    lo = nusselt_briggs_young(1500.0, 0.7, **kwargs)
    hi = nusselt_briggs_young(6000.0, 0.7, **kwargs)
    assert hi > lo


# -------------------------------------------------------------------
# Re / velocity definition (root diameter, fin-blockage-aware V_max)
# -------------------------------------------------------------------

def test_reynolds_uses_root_diameter_and_max_gap_velocity_not_face_velocity():
    tube = _tube()
    props = _air_props()
    result = finned_outside_flow_from_mass_flow(
        m_dot=5.0, frontal_area=2.0, tube=tube,
        tube_pitch_transverse=0.065, tube_pitch_longitudinal=0.055,
        layout="staggered", n_rows=6, n_tubes_per_row=10, props=props,
        calculate_pressure_drop=False,
    )
    assert result.V_max > result.v  # blockage must accelerate the flow
    expected_Re = props.rho * result.V_max * tube.D_root / props.mu
    assert result.Re == pytest.approx(expected_Re, rel=1e-12)


def test_contract_reports_root_diameter_and_vmax_basis():
    assert HTC_CONTRACT.reference_diameter == "D_root"
    assert "max" in HTC_CONTRACT.velocity_basis.lower()
    assert HTC_CONTRACT.geometry_family == "circular_finned_tube_bank"
    assert "Briggs" in HTC_CONTRACT.source and "1963" in HTC_CONTRACT.source


# -------------------------------------------------------------------
# Applicability diagnostics
# -------------------------------------------------------------------

def test_applicability_flags_reynolds_out_of_range():
    tube = _tube()
    warnings = check_finned_outside_ht_applicability(tube, Re=500.0, S_T=0.065, layout="staggered")
    codes = [w.code for w in warnings]
    assert "outside_ht_finned_re_out_of_range" in codes


def test_applicability_silent_for_in_range_geometry_and_reynolds():
    tube = _tube()
    mid_re = (RE_MIN + RE_MAX) / 2.0
    warnings = check_finned_outside_ht_applicability(tube, Re=mid_re, S_T=0.065, layout="staggered")
    codes = [w.code for w in warnings]
    assert "outside_ht_finned_re_out_of_range" not in codes
    assert "outside_ht_finned_d_root_out_of_range" not in codes


def test_applicability_flags_tapered_fin_as_outside_source_dataset():
    tube = _tube(D_root=0.0265, fin_thickness_root=0.0006, fin_thickness_tip=0.0002)
    warnings = check_finned_outside_ht_applicability(tube, Re=5000.0, S_T=0.065, layout="staggered")
    codes = [w.code for w in warnings]
    assert "outside_ht_finned_tapered_fin_not_in_source_dataset" in codes


# -------------------------------------------------------------------
# Layout support: staggered only; inline is a controlled rejection
# -------------------------------------------------------------------

def test_inline_layout_is_a_controlled_unsupported_rejection():
    tube = _tube()
    props = _air_props()
    with pytest.raises(FinnedTubeUnsupportedLayoutError):
        finned_outside_flow_from_mass_flow(
            m_dot=5.0, frontal_area=2.0, tube=tube,
            tube_pitch_transverse=0.065, tube_pitch_longitudinal=0.055,
            layout="inline", n_rows=4, n_tubes_per_row=10, props=props,
        )


def test_staggered_layout_is_accepted():
    tube = _tube()
    props = _air_props()
    result = finned_outside_flow_from_mass_flow(
        m_dot=5.0, frontal_area=2.0, tube=tube,
        tube_pitch_transverse=0.065, tube_pitch_longitudinal=0.055,
        layout="staggered", n_rows=4, n_tubes_per_row=10, props=props,
        calculate_pressure_drop=False,
    )
    assert math.isfinite(result.alfa_o_physical) and result.alfa_o_physical > 0.0


# -------------------------------------------------------------------
# Pressure drop: documented Robinson-Briggs blocker, controlled not
# silent, never a bare-tube correlation reused for finned geometry
# -------------------------------------------------------------------

def test_robinson_briggs_provider_rejects_bare_tube_request():
    request = EulerRequest(Re=5000.0, ST_over_D=2.6, SL_over_D=2.2, layout="staggered", n_rows=4, is_finned=False)
    with pytest.raises(ValueError):
        RobinsonBriggsEulerProvider().evaluate(request)


def test_robinson_briggs_provider_rejects_inline_layout():
    request = EulerRequest(Re=5000.0, ST_over_D=2.6, SL_over_D=2.2, layout="inline", n_rows=4, is_finned=True)
    with pytest.raises(ValueError):
        RobinsonBriggsEulerProvider().evaluate(request)


def test_robinson_briggs_provider_is_a_documented_not_implemented_blocker():
    request = EulerRequest(Re=5000.0, ST_over_D=2.6, SL_over_D=2.2, layout="staggered", n_rows=4, is_finned=True)
    with pytest.raises(NotImplementedError):
        RobinsonBriggsEulerProvider().evaluate(request)


def test_bare_tube_providers_reject_finned_requests():
    request = EulerRequest(Re=5000.0, ST_over_D=2.6, SL_over_D=2.2, layout="staggered", n_rows=6, is_finned=True)
    with pytest.raises(ValueError):
        ZukauskasEulerProvider().evaluate(request)
    with pytest.raises(ValueError):
        GaddisGnielinskiEulerProvider().evaluate(request)


def test_robinson_briggs_resolves_by_name_through_dispatcher():
    request = EulerRequest(Re=5000.0, ST_over_D=2.6, SL_over_D=2.2, layout="staggered", n_rows=4, is_finned=True)
    with pytest.raises(NotImplementedError):
        evaluate_euler(request, euler_provider="robinson_briggs")


def test_finned_outside_flow_reports_dp_unavailable_not_silent_wrong_value():
    tube = _tube()
    props = _air_props()
    result = finned_outside_flow_from_mass_flow(
        m_dot=5.0, frontal_area=2.0, tube=tube,
        tube_pitch_transverse=0.065, tube_pitch_longitudinal=0.055,
        layout="staggered", n_rows=4, n_tubes_per_row=10, props=props,
        calculate_pressure_drop=True,
    )
    assert result.dp_available is False
    assert math.isnan(result.dp_o)
    codes = [w.code for w in result.warnings]
    assert "outside_dp_robinson_briggs_not_implemented" in codes
    assert "outside_dp_finned_unavailable" in codes
    # HTC/UA-relevant results must still be usable even though dp is not
    assert math.isfinite(result.alfa_o_physical) and result.alfa_o_physical > 0.0


def test_finned_outside_flow_can_skip_pressure_drop_entirely():
    tube = _tube()
    props = _air_props()
    result = finned_outside_flow_from_mass_flow(
        m_dot=5.0, frontal_area=2.0, tube=tube,
        tube_pitch_transverse=0.065, tube_pitch_longitudinal=0.055,
        layout="staggered", n_rows=4, n_tubes_per_row=10, props=props,
        calculate_pressure_drop=False,
    )
    assert result.euler_result is None
    assert not any("robinson_briggs" in w.code for w in result.warnings)
