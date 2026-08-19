# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Frozen dry single-phase release regressions for circular finned tubes.

The case intentionally sits inside the declared Briggs-Young and
Robinson-Briggs applicability envelopes.  It is suitable for carrying the
same inputs into an independent reference-calculation comparison: compare a
film HTC with ``outside_alpha_physical``, never the extended-surface effective
coefficient or overall U.
"""

from __future__ import annotations

import math

import pytest

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.models.bare_tube import BareTubeHeatExchanger
from core.models import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.phase_change.types import PhaseChangeMode
from core.properties.dry_air import DryAirPropertyProvider
from core.properties.water import IAPWS97WaterSteamProvider


P_INSIDE = 1.0e6
P_OUTSIDE = 101_325.0
M_DOT_INSIDE = 1.5
T_INSIDE = 360.0
T_OUTSIDE = 300.0

# Each expectation was generated with deterministic IAPWS liquid-water and
# in-repository dry-air property models (``prefer_coolprop=False``), after the
# effective-gross HTC semantics were made explicit.  Values cover lower,
# nominal, and upper Reynolds numbers within both correlation envelopes.
EXPECTED_BY_OUTSIDE_FLOW = {
    1.8: {
        "Q": 77_808.86990112283,
        "Q_rating": 77_808.0863577339,
        "T_out_inside": 347.63217933980644,
        "T_out_outside": 342.86330582281425,
        "UA": 2_823.220439577648,
        "outside_alpha_physical": 36.05648835393788,
        "outside_alpha_effective_gross": 33.96071727276357,
        "eta_fin": 0.9384658704212169,
        "eta_overall": 0.9418753412533745,
        "U_gross": 24.896359064542132,
        "A_outside_effective": 106.80765440710047,
        "V_face": 1.7065087872513536,
        "V_max": 3.3570664667239742,
        "Re_D_root": 4_740.59023761467,
        "dp_outside": 43.52444757390779,
    },
    3.6: {
        "Q": 118_213.97353932956,
        "Q_rating": 118_213.17616830448,
        "T_out_inside": 341.1977760810158,
        "T_out_outside": 332.57221412283803,
        "UA": 3_780.0653080565917,
        "outside_alpha_physical": 57.566832185435466,
        "outside_alpha_effective_gross": 52.43695588318964,
        "eta_fin": 0.9056612305458943,
        "eta_overall": 0.9108883343498674,
        "U_gross": 33.33422423467391,
        "A_outside_effective": 103.29376102917638,
        "V_face": 3.359491665543631,
        "V_max": 6.60883606336452,
        "Re_D_root": 9_595.511022940624,
        "dp_outside": 137.02541569787903,
    },
    5.4: {
        "Q": 143_409.85781260842,
        "Q_rating": 143_409.33145217362,
        "T_out_inside": 337.18180627088026,
        "T_out_outside": 326.34868392424005,
        "UA": 4_366.498082102598,
        "outside_alpha_physical": 75.67773315656855,
        "outside_alpha_effective_gross": 67.0992829331716,
        "eta_fin": 0.8799958220107768,
        "eta_overall": 0.8866449896741868,
        "U_gross": 38.5056379525659,
        "A_outside_effective": 100.54459172154107,
        "V_face": 4.9902468509060425,
        "V_max": 9.816879050962704,
        "Re_D_root": 14_500.63779992951,
        "dp_outside": 267.7399934189248,
    },
}


def _release_case_exchanger() -> BareTubeHeatExchanger:
    """Welded constant fin; all nondimensional inputs are in range."""
    core = BareTube(
        D_i=0.021,
        D_o=0.025,
        length_total=2.0,
        length_effective=2.0,
        wall_k=45.0,
    )
    tube = CircularFinnedTube(
        core_tube=core,
        fin_k=200.0,
        D_fin=0.052,
        D_root=core.D_o,
        fin_thickness_root=0.0005,
        fin_thickness_tip=0.0005,
        fin_pitch=0.003,
        fin_contact_resistance=0.0,
    )
    pitch_transverse = 0.060
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=tube,
            n_rows=6,
            n_tubes_per_row=8,
            pitch_transverse=pitch_transverse,
            pitch_longitudinal=math.sqrt(3.0) * pitch_transverse / 2.0,
            layout="staggered",
            n_passes_tube=2,
            flow_arrangement="crossflow",
        )
    )


def _dry_inputs(m_dot_outside: float) -> tuple[HXSideInput, HXSideInput]:
    return (
        HXSideInput(
            provider=IAPWS97WaterSteamProvider(),
            m_dot=M_DOT_INSIDE,
            T_in=T_INSIDE,
            p=P_INSIDE,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
        HXSideInput(
            provider=DryAirPropertyProvider(prefer_coolprop=False),
            m_dot=m_dot_outside,
            T_in=T_OUTSIDE,
            p=P_OUTSIDE,
            phase_change_mode=PhaseChangeMode.DISABLED,
        ),
    )


@pytest.mark.parametrize(
    ("m_dot_outside", "expected"),
    list(EXPECTED_BY_OUTSIDE_FLOW.items()),
    ids=["low_flow", "nominal_flow", "high_flow"],
)
def test_dry_single_phase_release_case_is_frozen_across_outside_flows(
    m_dot_outside: float,
    expected: dict[str, float],
) -> None:
    """Regression for the geometry -> HTC -> fin -> network -> UA chain."""
    exchanger = _release_case_exchanger()
    inside, outside = _dry_inputs(m_dot_outside)
    result = exchanger.simulate(inside, outside)
    diagnostics = result.finned_tube_diagnostics
    assert diagnostics is not None

    tube = exchanger.bundle.tube
    assert isinstance(tube, CircularFinnedTube)
    assert tube.D_root == tube.D_o
    assert tube.fin_thickness_root == tube.fin_thickness_tip_effective
    assert result.phase_change_active is False
    assert diagnostics.heat_transfer_metadata.method == "briggs_young_1963"
    assert diagnostics.pressure_drop_metadata.method == "robinson_briggs_1966"

    # Full HX result and the extended-surface reporting contract.
    assert result.q == pytest.approx(expected["Q"], rel=2.0e-7)
    assert result.T_out_inside == pytest.approx(expected["T_out_inside"], rel=2.0e-7)
    assert result.T_out_outside == pytest.approx(expected["T_out_outside"], rel=2.0e-7)
    assert result.UA == pytest.approx(expected["UA"], rel=2.0e-7)
    assert diagnostics.UA == pytest.approx(expected["UA"], rel=2.0e-7)
    assert diagnostics.outside_alpha_physical == pytest.approx(
        expected["outside_alpha_physical"], rel=2.0e-7
    )
    assert diagnostics.outside_alpha_effective_gross == pytest.approx(
        expected["outside_alpha_effective_gross"], rel=2.0e-7
    )
    assert diagnostics.eta_fin == pytest.approx(expected["eta_fin"], rel=2.0e-7)
    assert diagnostics.eta_overall == pytest.approx(expected["eta_overall"], rel=2.0e-7)
    assert diagnostics.U == pytest.approx(expected["U_gross"], rel=2.0e-7)
    assert result.outside_alfa_mean == pytest.approx(
        diagnostics.outside_alpha_effective_gross
    )
    assert result.thermal_state is not None
    assert result.thermal_state.alfa_o == pytest.approx(
        diagnostics.outside_alpha_effective_gross
    )
    assert result.thermal_state.outside_alpha_physical == pytest.approx(
        diagnostics.outside_alpha_physical
    )

    # Areas are total-bank values; geometry never reuses a per-tube fallback.
    assert diagnostics.A_inside == pytest.approx(48.0 * math.pi * 0.021 * 2.0)
    assert diagnostics.A_primary_outside == pytest.approx(6.283185307179587)
    assert diagnostics.A_fin == pytest.approx(107.11574311679757)
    assert diagnostics.A_outside_gross == pytest.approx(113.39892842397715)
    assert diagnostics.A_outside_geometric == pytest.approx(
        diagnostics.A_outside_gross
    )
    assert diagnostics.A_outside_effective == pytest.approx(
        expected["A_outside_effective"], rel=2.0e-7
    )
    assert diagnostics.A_outside_effective == pytest.approx(
        diagnostics.A_primary_outside + diagnostics.eta_fin * diagnostics.A_fin
    )
    assert diagnostics.eta_overall == pytest.approx(
        diagnostics.A_outside_effective / diagnostics.A_outside_gross
    )
    assert diagnostics.resistance_outside == pytest.approx(
        1.0
        / (
            diagnostics.outside_alpha_effective_gross
            * diagnostics.A_outside_gross
        )
    )
    assert diagnostics.UA == pytest.approx(1.0 / diagnostics.resistance_total)
    assert diagnostics.U == pytest.approx(
        diagnostics.UA / diagnostics.A_outside_gross
    )

    # Both velocity bases, Re_Droot, and Robinson-Briggs bank pressure loss.
    assert diagnostics.face_velocity == pytest.approx(expected["V_face"], rel=2.0e-7)
    assert diagnostics.reference_velocity == pytest.approx(expected["V_max"], rel=2.0e-7)
    assert diagnostics.Re == pytest.approx(expected["Re_D_root"], rel=2.0e-7)
    assert diagnostics.dp == pytest.approx(expected["dp_outside"], rel=2.0e-7)
    assert diagnostics.dp == pytest.approx(diagnostics.outside_dp_total)

    warning_codes = {warning.code for warning in (result.warnings or ())}
    assert {
        "finned_tube_bank_hydraulics_midpoint_temperature_fallback",
        "wall_temperature_envelope_0d_estimate",
    } <= warning_codes
    assert not any(
        code.endswith(("_below_range", "_above_range"))
        for code in warning_codes
    )


@pytest.mark.parametrize(
    ("m_dot_outside", "expected"),
    list(EXPECTED_BY_OUTSIDE_FLOW.items()),
    ids=["low_flow", "nominal_flow", "high_flow"],
)
def test_dry_single_phase_release_case_rating_preserves_finned_contract(
    m_dot_outside: float,
    expected: dict[str, float],
) -> None:
    """Rating closes the same dry water/air program on the effective basis."""
    exchanger = _release_case_exchanger()
    inside, outside = _dry_inputs(m_dot_outside)
    simulation = exchanger.simulate(inside, outside)
    rating = exchanger.rate(
        BalanceSideSpec(
            provider=inside.provider,
            p=P_INSIDE,
            m_dot=M_DOT_INSIDE,
            T_in=T_INSIDE,
            T_out=simulation.T_out_inside,
        ),
        BalanceSideSpec(
            provider=outside.provider,
            p=P_OUTSIDE,
            m_dot=m_dot_outside,
            T_in=T_OUTSIDE,
            T_out=simulation.T_out_outside,
        ),
    )
    diagnostics = rating.finned_tube_diagnostics
    assert diagnostics is not None

    assert rating.Q_required == pytest.approx(expected["Q_rating"], rel=2.0e-7)
    assert rating.UA_actual == pytest.approx(expected["UA"], rel=2.0e-7)
    assert rating.alfa_o == pytest.approx(
        diagnostics.outside_alpha_effective_gross
    )
    assert diagnostics.outside_alpha_physical == pytest.approx(
        expected["outside_alpha_physical"], rel=2.0e-7
    )
    assert diagnostics.outside_alpha_effective_gross == pytest.approx(
        expected["outside_alpha_effective_gross"], rel=2.0e-7
    )
    assert diagnostics.UA == pytest.approx(expected["UA"], rel=2.0e-7)
    assert diagnostics.U == pytest.approx(expected["U_gross"], rel=2.0e-7)
    assert diagnostics.dp == pytest.approx(expected["dp_outside"], rel=2.0e-7)
