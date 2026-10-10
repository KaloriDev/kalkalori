# SPDX-License-Identifier: GPL-3.0-only
"""Excluded geometry is rejected before circular equations/diameter access."""
import inspect
import pytest
from core.geometry import EllipticalTube, TubeBundle, CircularFinnedTube, TubeGeometryApplicabilityError
from core.models.bare_tube import BareTubeHeatExchanger
from core.heat_transfer.outside_dispatch import evaluate_outside_thermal, evaluate_outside_hydraulics, calculate_resistance_network
from core.heat_transfer.fin_efficiency import annular_fin_efficiency
from core.heat_transfer.wet_coil_surface import annular_response
from core.heat_transfer.wet_coil_adapters import InsideWallAdapter, BareTubeAdapter
from core.phase_change.wet_finned_surface import solve_wet_annular_fin, solve_wet_finned_surface
from core.phase_change.wet_coil_provider import dispatch_wet_coil, ElmahdyMitalasWetCoilProvider, LegacyBulkMeanWetCoilProvider
from core.phase_change.wet_coil_integration import forward_wet_process
from core.phase_change.inside_condensation_solver import solve_inside_condensation
from core.phase_change.steam_heater import solve_steam_heater, rate_steam_heater
from core.phase_change.water_evaporator import solve_water_evaporator, rate_water_evaporator
from core.pressure_drop.internal_pressure_drop import calculate_tube_bundle_hydraulics
from core.enhancements import TubeSideEnhancement, TwistedTapeGeometry, WireCoilGeometry
from core.enhancements.yang2020 import Yang2020TwistedTapeProvider
from core.enhancements.inaba1994 import Inaba1994WireCoilProvider
from core.enhancements.integration import evaluate_for_bundle


def context(beta):
    tube=EllipticalTube(.012,.012,.01,.01*beta,2.,1.8,20.)
    bank=TubeBundle(tube,4,6,.05,.05,'inline',2)
    return tube,bank,BareTubeHeatExchanger(bank)


def invoke_with_required_arguments(fn,**overrides):
    kwargs={name:None for name,p in inspect.signature(fn).parameters.items()
            if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL,p.VAR_KEYWORD)}
    kwargs.update(overrides)
    return fn(**kwargs)


@pytest.mark.parametrize('beta',[.5,1.])
@pytest.mark.parametrize('fn',[evaluate_outside_thermal,evaluate_outside_hydraulics,solve_inside_condensation,
    solve_steam_heater,rate_steam_heater,solve_water_evaporator,rate_water_evaporator,dispatch_wet_coil,forward_wet_process])
def test_excluded_model_boundaries_are_descriptive(fn,beta):
    tube,bank,hx=context(beta)
    overrides={'bundle':bank} if 'bundle' in inspect.signature(fn).parameters else {'hx':hx}
    with pytest.raises(TubeGeometryApplicabilityError,match='circular|outside-bank'):
        invoke_with_required_arguments(fn,**overrides)


@pytest.mark.parametrize('beta',[.5,1.])
def test_full_workflows_and_automatic_total_hydraulics_rejected(beta):
    tube,bank,hx=context(beta)
    for fn in (hx.solve,hx.solve_thermal_state,hx.rate,hx.simulate):
        with pytest.raises(TubeGeometryApplicabilityError,match='outside-bank'):
            invoke_with_required_arguments(fn)
    with pytest.raises(TubeGeometryApplicabilityError,match='entrance/exit'):
        invoke_with_required_arguments(calculate_tube_bundle_hydraulics,tube=tube)


@pytest.mark.parametrize('beta',[.5,1.])
def test_fins_and_direct_wet_adapters_cannot_accept_an_ellipse(beta):
    tube,bank,hx=context(beta)
    for fn,kw in ((CircularFinnedTube,dict(core_tube=tube)),(annular_fin_efficiency,dict(tube=tube)),
                  (solve_wet_annular_fin,dict(tube=tube)),(solve_wet_finned_surface,dict(bundle=bank)),
                  (annular_response,dict(tube=tube))):
        with pytest.raises((TubeGeometryApplicabilityError,TypeError),match='circular|Circular|BareTube'):
            invoke_with_required_arguments(fn,**kw)
    for adapter in (InsideWallAdapter(bundle=bank,provider=None,pressure=1.,mass_flow=1.),BareTubeAdapter(bank,None)):
        with pytest.raises(TubeGeometryApplicabilityError): invoke_with_required_arguments(adapter.evaluate)


@pytest.mark.parametrize('beta',[.5,1.])
@pytest.mark.parametrize('provider,geometry',[(Yang2020TwistedTapeProvider(),TwistedTapeGeometry(.05,.019,.001)),
                                            (Inaba1994WireCoilProvider(),WireCoilGeometry(.001,.01))])
def test_current_insert_provider_integration_rejects_ellipse(provider,geometry,beta):
    tube,bank,hx=context(beta)
    configuration=TubeSideEnhancement(provider,geometry,'liquid')
    with pytest.raises(TubeGeometryApplicabilityError,match='inserts'):
        evaluate_for_bundle(configuration,bank,1.,None)


def test_fouling_uses_actual_ellipse_areas_without_a_fictitious_outside_solution():
    tube,bank,hx=context(.5)
    hx=BareTubeHeatExchanger(bank,fouling_resistance_inside=.002,fouling_resistance_outside=.003)
    assert hx.resistance_fouling_inside==.002/bank.total_inner_area
    assert hx.resistance_fouling_outside==.003/bank.total_outer_area
    # A prescribed resistance-network calculation is shape-neutral arithmetic,
    # not an artificial Rating/Simulation or an outside-bank correlation.
    values=dict(bundle=bank,alpha_inside=1200.,outside_alpha_physical=80.,resistance_core_wall=.001)
    clean=calculate_resistance_network(**values)
    dirty=calculate_resistance_network(**values,fouling_resistance_inside=.002,fouling_resistance_outside=.003)
    assert dirty.resistance_fouling_inside==hx.resistance_fouling_inside
    assert dirty.resistance_fouling_outside==hx.resistance_fouling_outside
    assert 1/dirty.UA==pytest.approx(1/clean.UA+hx.resistance_fouling_inside+hx.resistance_fouling_outside)
