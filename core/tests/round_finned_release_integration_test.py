# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Release API coverage for explicit selectors and the layout guard matrix."""
import math

import pytest

import core.heat_transfer as heat_transfer
from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.heat_transfer.finned_tube_outside import calculate_finned_tube_outside_heat_transfer
from core.models import BalanceSideSpec
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.phase_change.types import PhaseChangeMode
from core.pressure_drop.finned_tube_pressure_drop import calculate_finned_tube_pressure_drop
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider


INLINE = "frass_escoa_inline_solid_fin"
STAGGERED = "frass_escoa_staggered_solid_fin"
PROPS = FluidTransportProperties(rho=1., mu=2e-5, k=.03, cp=1050.)


def bundle(layout):
    core = BareTube(D_i=.032, D_o=.038, length_total=1., length_effective=1., wall_k=45.)
    tube = CircularFinnedTube(core_tube=core, D_root=.038, D_fin=.060,
        fin_k=45., fin_pitch=.004, fin_thickness_root=.00075)
    return TubeBundle(tube=tube, n_rows=6, n_tubes_per_row=3,
        pitch_transverse=.08, pitch_longitudinal=math.sqrt(3)/2*.08,
        layout=layout, n_passes_tube=1, flow_arrangement="counterflow")


@pytest.mark.parametrize("layout,selector", [("inline", INLINE), ("staggered", STAGGERED)])
@pytest.mark.parametrize("api", ["low_level", "thermal_state", "rating", "simulation"])
def test_explicit_thermal_selectors_reach_every_public_workflow(layout, selector, api):
    bank = bundle(layout)
    if api == "low_level":
        result = calculate_finned_tube_outside_heat_transfer(1.,bank,PROPS,
            provider=selector,gas_temperature=600.,mean_fin_temperature=450.)
        assert result.method == selector
        return
    hx = BareTubeHeatExchanger(bank)
    inside = ConstantPropertyProvider(FluidTransportProperties(rho=900.,mu=.0005,k=.6,cp=4000.))
    outside = ConstantPropertyProvider(PROPS)
    if api == "thermal_state":
        state = hx.solve_thermal_state(m_dot_inside=1.5,m_dot_outside=1.,
            inside_provider=inside,outside_provider=outside,T_in_inside=400.,T_in_outside=600.,
            p_inside=101325.,p_outside=101325.,finned_heat_transfer_provider=selector)
    else:
        cls = BalanceSideSpec if api == "rating" else HXSideInput
        i = cls(provider=inside,m_dot=1.5,T_in=400.,p=101325.,
            phase_change_mode=PhaseChangeMode.DISABLED,
            **({"T_out":410.} if api == "rating" else {}))
        o = cls(provider=outside,m_dot=1.,T_in=600.,p=101325.,
            phase_change_mode=PhaseChangeMode.DISABLED)
        result = getattr(hx,"rate" if api == "rating" else "simulate")(
            i,o,finned_heat_transfer_provider=selector)
        state = result.thermal_state
        expected_dp = "frass_inline_solid_fin_191" if layout == "inline" else "robinson_briggs_1966"
        assert result.outside_tube_bank_hydraulic.metadata.method == expected_dp
    assert state.converged
    thermal = state.finned_tube_diagnostics.heat_transfer_result
    assert thermal.method == selector
    assert thermal.diagnostics["coupling_converged"]


@pytest.mark.parametrize("layout", ["inline", "staggered"])
@pytest.mark.parametrize("kind,selector,supported_layout", [
    ("thermal", INLINE, "inline"),
    ("thermal", STAGGERED, "staggered"),
    ("thermal", "briggs_young_1963", "staggered"),
    ("hydraulic", "robinson_briggs_1966", "staggered"),
    ("hydraulic", "frass_inline_solid_fin_191", "inline"),
])
def test_release_layout_provider_guard_matrix(kind, selector, supported_layout, layout):
    bank = bundle(layout)
    def calculate():
        if kind == "thermal":
            return calculate_finned_tube_outside_heat_transfer(1.,bank,PROPS,
                provider=selector,gas_temperature=600.,mean_fin_temperature=450.)
        return calculate_finned_tube_pressure_drop(1.,bank,PROPS,provider=selector)
    if layout == supported_layout:
        assert calculate().method == selector
    else:
        with pytest.raises(ValueError,match="INLINE|STAGGERED|inline|staggered"):
            calculate()


def test_public_exports_selector_errors_and_auto_policy():
    assert heat_transfer.FrassEscoaInlineSolidFinProvider
    assert heat_transfer.FrassEscoaStaggeredSolidFinProvider
    assert "FrassEscoaInlineSolidFinProvider" in heat_transfer.__all__
    assert "FrassEscoaStaggeredSolidFinProvider" in heat_transfer.__all__
    with pytest.raises(ValueError) as exc:
        calculate_finned_tube_outside_heat_transfer(1.,bundle("inline"),PROPS,provider="unknown")
    assert INLINE in str(exc.value) and STAGGERED in str(exc.value)
    for layout,expected in (("inline",INLINE),("staggered","briggs_young_1963")):
        result = calculate_finned_tube_outside_heat_transfer(1.,bundle(layout),PROPS,
            gas_temperature=600.,mean_fin_temperature=450.)
        assert result.method == expected
