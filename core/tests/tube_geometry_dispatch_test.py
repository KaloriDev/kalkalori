# SPDX-License-Identifier: GPL-3.0-only
"""Family gates and unchanged circular constitutive/reference arithmetic."""
import math
from types import SimpleNamespace
import pytest

from core.geometry import BareTube, TubeBundle, TubeCrossSectionFamily
from core.geometry.applicability import TubeGeometryApplicabilityError
from core.heat_transfer.internal_flow import FluidProps, heat_transfer_coefficient_internal_diagnostics
from core.heat_transfer.internal_dispatch import internal_heat_transfer_diagnostics, straight_friction_evaluator
from core.heat_transfer.outside_dispatch import evaluate_outside_thermal, evaluate_outside_hydraulics
from core.heat_transfer.wall_conduction import tube_wall_resistance
from core.models.bare_tube import BareTubeHeatExchanger
from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
from core.properties.fluids import FluidTransportProperties


def bundle(tube):
    return TubeBundle(tube, 4, 6, .05, .05, 'inline', 2)


def test_circular_constructor_wall_and_htc_are_unchanged():
    tube = BareTube(.02, .024, 2., 1.8, 20.)
    assert tube.cross_section_family is TubeCrossSectionFamily.CIRCULAR
    assert repr(tube) == ('BareTube(D_i=0.02, D_o=0.024, length_total=2.0, '
                         'length_effective=1.8, wall_k=20.0, roughness_inner=None, '
                         'roughness_outer=None, tube_orientation=None)')
    props = FluidProps(998., .001, .6, 4200.)
    common = dict(m_dot=.02, flow_area=tube.flow_area, props=props,
                  T_bulk=310., T_wall=305., L_heated=tube.length_effective)
    assert internal_heat_transfer_diagnostics(tube=tube, **common) == heat_transfer_coefficient_internal_diagnostics(
        tube_inner_diameter=tube.D_i, **common)
    assert tube_wall_resistance(tube, n_tubes=24) == math.log(tube.D_o/tube.D_i)/(2*math.pi*tube.wall_k*tube.length_effective*24)


def test_nonround_geometry_is_rejected_before_any_diameter_access():
    tube = SimpleNamespace(cross_section_family=TubeCrossSectionFamily.ELLIPTICAL)
    bank = bundle(tube)
    hx = BareTubeHeatExchanger(bank)
    calls = [
        lambda: internal_heat_transfer_diagnostics(tube=tube, m_dot=1, flow_area=1, props=None),
        lambda: straight_friction_evaluator(tube),
        lambda: tube_wall_resistance(tube),
        lambda: evaluate_outside_thermal(bundle=bank, m_dot=1, props=None),
        lambda: evaluate_outside_hydraulics(bundle=bank, m_dot=1),
        lambda: bank.minimum_free_flow_gap,
        lambda: hx.rate(None, None),
        lambda: hx.simulate(None, None),
    ]
    for call in calls:
        with pytest.raises(TubeGeometryApplicabilityError, match='circular|outside-bank|explicit|EllipticalTube'):
            call()


def test_selected_friction_is_evaluated_before_circular_model(monkeypatch):
    import core.pressure_drop.internal_pressure_drop as hydraulics
    monkeypatch.setattr(hydraulics, 'darcy_friction_factor', lambda *args: pytest.fail('Circular fallback ran'))
    seen=[]
    def selected(reynolds, roughness):
        seen.append((reynolds,roughness))
        return .023, 'explicit_test_model'
    result=calculate_tube_bundle_hydraulics(
        m_dot=.2, flow_area_per_pass=.001, hydraulic_diameter=.02,
        hydraulic_length_total=2., n_tube_passes=1,
        inlet_props=FluidTransportProperties(1000., 4200., .001, .6),
        straight_friction_evaluator=selected,
    )
    assert len(seen)==3
    assert all(point.friction_factor==.023 for point in (result.inlet,result.midpoint,result.outlet))
    assert result.dp_straight_tube_friction == pytest.approx(.023*2/.02*1000*.2**2/2)
