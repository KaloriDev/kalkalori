# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Low-mass-flux in-tube steam-condensation regression tests (v0.6.2 patch).

Manual validation of a real steam-heater case (many parallel horizontal
tubes, 14 bara steam, superheated inlet, D_i ~ 15 mm) found
``zone_alpha_condensation`` around a few hundred W/(m2*K) -- physically far
too low for condensing steam (practice: order a few kW/(m2*K)). Root cause:
the legacy Shah (1979) correlation has no gravity-film branch and
systematically underpredicts once the liquid-only reference Reynolds
number falls into the laminar/transitional range (which happens well
before its own documented mass-flux floor at low G). This module is a
dedicated regression suite built around that exact case geometry/
conditions, using the production ``BareTubeHeatExchanger.simulate``/
``.rate`` entry points (not the correlation module directly -- see
``core/tests/condensation_inside_shah2009_htc_test.py`` for that).

Run:
    pytest -q core/tests/inside_pure_steam_low_flow_regression_test.py
"""

from __future__ import annotations

import math

import pytest

from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.heat_transfer.condensation_inside import shah_condensation_alpha_local
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import IAPWS97WaterSteamProvider, WATER_CRITICAL_PRESSURE_PA

P_STEAM = 14.0e5  # 14 bara, matches the real steam-heater case


def _dry_air_provider() -> GasMixturePropertyProvider:
    return GasMixturePropertyProvider(GasMixtureSpec(components={"N2": 0.79, "O2": 0.21}, basis="mole"))


def _low_flow_hx(orientation: TubeOrientation = TubeOrientation.HORIZONTAL) -> BareTubeHeatExchanger:
    """Many parallel tubes at low per-tube mass flux -- the steam_heater_calc geometry."""
    tube = BareTube(
        D_o=18e-3, D_i=15e-3, length_total=1.5, length_effective=1.45, wall_k=50.0,
        tube_orientation=orientation,
    )
    bundle = TubeBundle(
        tube=tube, n_rows=10, n_tubes_per_row=40,
        pitch_transverse=25e-3, pitch_longitudinal=25e-3,
        layout="inline", n_passes_tube=1, flow_arrangement="crossflow",
    )
    return BareTubeHeatExchanger(bundle)


def _high_flow_hx(orientation: TubeOrientation = TubeOrientation.HORIZONTAL) -> BareTubeHeatExchanger:
    """Few tubes at high per-tube mass flux -- inside Shah (1979)'s own range."""
    tube = BareTube(
        D_o=25e-3, D_i=20e-3, length_total=3.0, length_effective=3.0, wall_k=50.0,
        tube_orientation=orientation,
    )
    bundle = TubeBundle(
        tube=tube, n_rows=2, n_tubes_per_row=2,
        pitch_transverse=40e-3, pitch_longitudinal=40e-3,
        layout="staggered", n_passes_tube=1, flow_arrangement="crossflow",
    )
    return BareTubeHeatExchanger(bundle)


# ---------------------------------------------------------------------------
# The dedicated low-flow case (spec section 16): pure steam, real pressure,
# real D_i, many parallel tubes, low G, superheated inlet -> full
# condensation (a large air flow guarantees full condensation within the
# available area).
# ---------------------------------------------------------------------------
def test_low_flow_case_selects_gravity_aware_regime_not_forced_convection_only() -> None:
    hx = _low_flow_hx()
    inside = HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=900.81 / 3600.0, T_in=320.0 + 273.15, p=P_STEAM)
    outside = HXSideInput(provider=_dry_air_provider(), m_dot=28_785.0 / 3600.0, T_in=25.0 + 273.15, p=101_325.0)
    result = hx.simulate(inside, outside)
    pc = result.inside_phase_change

    D_i = hx.bundle.tube.D_i
    G = inside.m_dot / hx.bundle.internal_flow_area_per_pass
    assert G < 10.8  # below Shah (1979)'s own documented mass-flux floor

    assert pc.Q_condensation > 0.0
    assert pc.zone_alpha_condensation is not None
    assert math.isfinite(pc.zone_alpha_condensation)

    # Independent legacy-1979 local alpha at the same local G/D/properties,
    # evaluated at the condensation zone's own midpoint quality, as the
    # "what the old forced-convection-only model would have said" reference.
    from core.properties.water import water_steam_state

    sat_liquid = water_steam_state(p=P_STEAM, x=0.0)
    legacy = shah_condensation_alpha_local(
        0.5, p=P_STEAM, p_critical=WATER_CRITICAL_PRESSURE_PA, G=G, D_i=D_i,
        mu_L=sat_liquid.mu, k_L=sat_liquid.k, cp_L=sat_liquid.cp,
    )
    # The new zone-averaged condensation HTC must be dramatically higher
    # than the legacy forced-convection-only local estimate at this G --
    # this is the direct evidence of the fix, with no calibration factor.
    assert pc.zone_alpha_condensation > 5.0 * legacy.alpha

    # Order-of-magnitude sanity (not a calibration target -- see spec
    # section 18): a few kW/(m2*K) is the practically expected range for
    # condensing steam; assert only that the fix lands in a physically
    # sane band, not any specific number.
    assert 1_000.0 < pc.zone_alpha_condensation < 100_000.0


def test_low_flow_case_energy_balance_closes() -> None:
    hx = _low_flow_hx()
    inside = HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=900.81 / 3600.0, T_in=320.0 + 273.15, p=P_STEAM)
    outside = HXSideInput(provider=_dry_air_provider(), m_dot=28_785.0 / 3600.0, T_in=25.0 + 273.15, p=101_325.0)
    result = hx.simulate(inside, outside)
    pc = result.inside_phase_change

    assert pc.Q_total == pytest.approx(pc.Q_desuperheat + pc.Q_condensation + pc.Q_subcooling, rel=1e-8)
    assert pc.Q_total == pytest.approx(inside.m_dot * (pc.h_in - pc.h_out), rel=1e-6)
    total_fraction = pc.f_desuperheat + pc.f_condensation + pc.f_subcooling
    assert total_fraction == pytest.approx(1.0, abs=1e-6)
    assert pc.A_desuperheat >= 0.0 and pc.A_condensation >= 0.0 and pc.A_subcooling >= 0.0
    for value in (pc.Q_total, pc.T_out, pc.h_out, result.inside_alfa_mean, result.UA):
        assert math.isfinite(value)


def test_low_flow_case_reports_applicability_extrapolation_warning() -> None:
    # Section 10: an explicit warning, never a silently clipped/extrapolated
    # G and never a calibration factor.
    hx = _low_flow_hx()
    inside = HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=900.81 / 3600.0, T_in=320.0 + 273.15, p=P_STEAM)
    outside = HXSideInput(provider=_dry_air_provider(), m_dot=28_785.0 / 3600.0, T_in=25.0 + 273.15, p=101_325.0)
    result = hx.simulate(inside, outside)
    pc = result.inside_phase_change
    codes = {w.code for w in pc.warnings}
    assert "SHAH_2009_CONDENSATION_OUT_OF_RANGE" in codes


# ---------------------------------------------------------------------------
# Regime coverage (spec section 17): high-flow (A), low-flow (B), x near 1
# (E), x near 0 (F), water at a representative steam-heater pressure (G).
# Transition (C) and partial quality range (D) are covered directly at the
# correlation level in condensation_inside_shah2009_htc_test.py.
# ---------------------------------------------------------------------------
def test_high_flow_case_stays_within_forced_convective_order_of_magnitude() -> None:
    hx = _high_flow_hx()
    inside = HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=0.19, x_in=1.0, p=P_STEAM)
    outside = HXSideInput(provider=_dry_air_provider(), m_dot=5.0, T_in=25.0 + 273.15, p=101_325.0)
    result = hx.simulate(inside, outside)
    pc = result.inside_phase_change

    G = inside.m_dot / hx.bundle.internal_flow_area_per_pass
    assert 10.8 <= G <= 210.6  # inside Shah (1979)'s own documented range too
    assert pc.zone_alpha_condensation is not None
    assert math.isfinite(pc.zone_alpha_condensation)
    assert pc.zone_alpha_condensation > 0.0


def test_saturated_vapor_inlet_x_near_one_finite() -> None:
    hx = _low_flow_hx()
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=900.81 / 3600.0, x_in=1.0, p=P_STEAM),
        HXSideInput(provider=_dry_air_provider(), m_dot=28_785.0 / 3600.0, T_in=25.0 + 273.15, p=101_325.0),
    )
    pc = result.inside_phase_change
    assert pc.zone_alpha_condensation is not None
    assert math.isfinite(pc.zone_alpha_condensation)


def test_near_complete_condensation_x_near_zero_finite() -> None:
    hx = _low_flow_hx()
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=200.0 / 3600.0, x_in=1.0, p=P_STEAM),
        HXSideInput(provider=_dry_air_provider(), m_dot=28_785.0 / 3600.0, T_in=25.0 + 273.15, p=101_325.0),
    )
    pc = result.inside_phase_change
    assert pc.phase_out.value in ("saturated_liquid", "subcooled_liquid")
    assert pc.zone_alpha_condensation is not None
    assert math.isfinite(pc.zone_alpha_condensation)


def test_vertical_downflow_orientation_selected_at_low_flow() -> None:
    # Vertical/inclined tubes are validated by Shah (2009) at all flow
    # rates (Regime III, the pure-gravity Nusselt limit); this must not
    # raise and must produce a finite, positive condensation HTC.
    hx = _low_flow_hx(orientation=TubeOrientation.VERTICAL_DOWNFLOW)
    result = hx.simulate(
        HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=900.81 / 3600.0, T_in=320.0 + 273.15, p=P_STEAM),
        HXSideInput(provider=_dry_air_provider(), m_dot=28_785.0 / 3600.0, T_in=25.0 + 273.15, p=101_325.0),
    )
    pc = result.inside_phase_change
    assert pc.zone_alpha_condensation is not None
    assert math.isfinite(pc.zone_alpha_condensation)
    assert pc.zone_alpha_condensation > 0.0


# ---------------------------------------------------------------------------
# Explicit tube orientation is required (spec section 9): never assumed
# silently.
# ---------------------------------------------------------------------------
def test_missing_tube_orientation_raises_on_simulate() -> None:
    tube = BareTube(D_o=18e-3, D_i=15e-3, length_total=1.5, length_effective=1.45, wall_k=50.0)
    bundle = TubeBundle(
        tube=tube, n_rows=10, n_tubes_per_row=40,
        pitch_transverse=25e-3, pitch_longitudinal=25e-3,
        layout="inline", n_passes_tube=1, flow_arrangement="crossflow",
    )
    hx = BareTubeHeatExchanger(bundle)
    with pytest.raises(ValueError, match="tube_orientation"):
        hx.simulate(
            HXSideInput(provider=IAPWS97WaterSteamProvider(), m_dot=900.81 / 3600.0, T_in=320.0 + 273.15, p=P_STEAM),
            HXSideInput(provider=_dry_air_provider(), m_dot=28_785.0 / 3600.0, T_in=25.0 + 273.15, p=101_325.0),
        )


def test_missing_tube_orientation_raises_on_rate() -> None:
    from core.models.heat_balance import BalanceSideSpec

    tube = BareTube(D_o=18e-3, D_i=15e-3, length_total=1.5, length_effective=1.45, wall_k=50.0)
    bundle = TubeBundle(
        tube=tube, n_rows=10, n_tubes_per_row=40,
        pitch_transverse=25e-3, pitch_longitudinal=25e-3,
        layout="inline", n_passes_tube=1, flow_arrangement="crossflow",
    )
    hx = BareTubeHeatExchanger(bundle)
    with pytest.raises(ValueError, match="tube_orientation"):
        hx.rate(
            BalanceSideSpec(provider=IAPWS97WaterSteamProvider(), p=P_STEAM, m_dot=900.81 / 3600.0, T_in=320.0 + 273.15, T_out=250.0 + 273.15),
            BalanceSideSpec(provider=_dry_air_provider(), p=101_325.0, m_dot=28_785.0 / 3600.0, T_in=25.0 + 273.15, T_out=60.0 + 273.15),
        )
