# SPDX-License-Identifier: GPL-3.0-only
"""Public synthetic onset/partial/intensive cases; no private fixture imports."""
import pytest
from iapws import IAPWS97

from core import BulkMeanAnalogyWetCoilProvider
from core.geometry.bundle import TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.types import PhaseChangeMode
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.properties.gas_mixture import GasMixturePropertyProvider, gas_mixture_from_dry_composition_and_water_ratio
from core.properties.water import IAPWS97WaterSteamProvider


@pytest.mark.parametrize("humidity,regime", [
    (.004, "DRY"), (.0175, "DRY"), (.0183, "PARTIALLY_WET"),
    (.04, "FULLY_WET"), (.08, "FULLY_WET"),
])
def test_surface_onset_and_independent_whole_exchanger_balances(humidity, regime):
    tube = BareTube(.021, .025, 2., 2., 45., tube_orientation=TubeOrientation.HORIZONTAL)
    hx = BareTubeHeatExchanger(TubeBundle(tube, 6, 8, .06, .05, "staggered", 2,
                                        flow_arrangement="counterflow"))
    gas = GasMixturePropertyProvider(gas_mixture_from_dry_composition_and_water_ratio(
        {"N2":.77, "O2":.15, "CO2":.08}, water_ratio=humidity, dry_basis="mole"))
    water = IAPWS97WaterSteamProvider()
    water_in = 303.15 if humidity == .004 else 293.15
    a = HXSideInput(water, .4, water_in, 600000., phase_change_mode=PhaseChangeMode.DISABLED)
    b = HXSideInput(gas, .5*(1+humidity), 333.15, 101325.)
    # The intensive case needs more outer iterations, as in the established
    # benchmark controls. No tolerance or global default is changed.
    r = hx.simulate(a, b, wet_coil_provider=BulkMeanAnalogyWetCoilProvider(),
                    phase_change_max_iterations=200)
    pc, e = r.outside_phase_change, r.wall_temperature_envelope
    surface = r.wet_coil_diagnostics["surface"]
    assert r.converged and pc.converged
    assert surface["wet_regime"] == regime
    assert e.outside_skin_min <= e.outside_skin_mean <= e.outside_skin_max
    assert surface["surface_temperature_mean"] == e.outside_skin_mean
    assert pc.m_dot_condensate == pytest.approx(.5*(humidity-pc.W_out), abs=1e-8)
    if regime == "DRY":
        assert not pc.active and pc.m_dot_condensate == 0.
        qgas = b.m_dot*(gas.full_at(T=b.T_in, p=b.p).h
                       - gas.full_at(T=r.T_out_outside, p=b.p).h)
    else:
        assert pc.active and pc.possible and pc.m_dot_condensate > 0.
        assert pc.wet_area > 0. and 0. < pc.wet_surface_fraction <= 1.
        if regime == "PARTIALLY_WET":
            assert pc.wet_surface_fraction < 1.
        # Independent endpoint evaluation with the declared dry-carrier and
        # saturated-vapor/liquid references, rather than solver duty helpers.
        dry = wet_gas_provider_at_water_ratio(detect_phase_change_capability(gas), 0.)
        def enthalpy(T, W):
            return dry.full_at(T=T, p=b.p).h + W*1000*IAPWS97(T=T, x=1).h
        qgas = .5*(enthalpy(b.T_in, humidity)-enthalpy(r.T_out_outside, pc.W_out))
        qgas -= r.wet_coil_diagnostics["H_drain"]
    qwater = .4*1000*(IAPWS97(T=r.T_out_inside, P=.6).h-IAPWS97(T=water_in, P=.6).h)
    # Independent finite-enthalpy increments need not exactly equal the 0D
    # mean-cp coolant closure. This is a 0.1% numerical engineering check.
    assert qgas == pytest.approx(r.q, rel=1e-3, abs=1.)
    assert qwater == pytest.approx(r.q, rel=1e-3, abs=1.)
