"""Independent synthetic INLINE sensitivities and qualified pressure reporting."""
from dataclasses import replace
import math

import pytest

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.pressure_drop import calculate_finned_tube_pressure_drop as pressure
from core.properties.common import FluidTransportProperties

GAS = FluidTransportProperties(rho=1.1, mu=2.0e-5, k=.029, cp=1010.)


def synthetic_inline(taper=False, pitch=.004):
    core = BareTube(D_i=.026, D_o=.030, length_total=2.15,
                    length_effective=2.1, wall_k=35.)
    tube = CircularFinnedTube(core_tube=core, fin_k=165., D_root=.030,
        D_fin=.042, fin_pitch=pitch,
        fin_thickness_root=.0006 if taper else .0004,
        fin_thickness_tip=.0002 if taper else .0004)
    return TubeBundle(tube=tube, n_rows=8, n_tubes_per_row=10,
        pitch_transverse=.068, pitch_longitudinal=.074, layout='inline',
        n_passes_tube=2, flow_arrangement='counterflow')


def log_slope(evaluate, step=1e-5):
    return (math.log(evaluate(math.exp(step)))
            - math.log(evaluate(math.exp(-step)))) / (2*step)


@pytest.mark.parametrize('taper', [False, True])
@pytest.mark.parametrize('pitch', [.002, .004, .008])
def test_frass_pitch_elasticity_includes_explicit_pitch_and_actual_blockage(taper, pitch):
    b = synthetic_inline(taper, pitch)
    t = b.tube
    projected_fin = (t.D_fin-t.D_root)*(t.fin_thickness_root+t.fin_thickness_tip_effective)/(2*pitch)
    beta = projected_fin/(b.pitch_transverse-t.D_root-projected_fin)
    measured = log_slope(lambda factor: pressure(12., replace(b,
        tube=replace(t, fin_pitch=pitch*factor)), GAS).dp_drag)
    assert measured == pytest.approx(-.7-1.66*beta, rel=2e-8)
    # Holding passage velocity fixed isolates the explicit fin-density term.
    velocity = 14.
    fixed_velocity = log_slope(lambda factor: (
        lambda bank: pressure(velocity*GAS.rho*bank.minimum_free_flow_area,
            bank, GAS).dp_drag)(replace(b,tube=replace(t,fin_pitch=pitch*factor))))
    assert fixed_velocity == pytest.approx(-.7, abs=2e-9)


@pytest.mark.parametrize('taper', [False, True])
def test_frass_height_thickness_and_velocity_have_distinct_sensitivities(taper):
    b = synthetic_inline(taper)
    t = b.tube
    fin_blockage = (t.D_fin-t.D_root)*(t.fin_thickness_root+t.fin_thickness_tip_effective)/(2*t.fin_pitch)
    beta = fin_blockage/(b.pitch_transverse-t.D_root-fin_blockage)
    height = log_slope(lambda f: pressure(12.,replace(b,
        tube=replace(t,D_fin=t.D_root+(t.D_fin-t.D_root)*f)),GAS).dp_drag)
    thickness = log_slope(lambda f: pressure(12.,replace(b,tube=replace(t,
        fin_thickness_root=t.fin_thickness_root*f,
        fin_thickness_tip=t.fin_thickness_tip_effective*f)),GAS).dp_drag)
    velocity = log_slope(lambda f: pressure(12.*f,b,GAS).dp_drag)
    assert height == pytest.approx(.5+1.66*beta, rel=2e-8)
    assert thickness == pytest.approx(1.66*beta, rel=2e-8)
    assert velocity == pytest.approx(1.66, abs=2e-9)


def test_constant_and_equal_mean_taper_share_frass_blockage_not_genic_surface():
    constant, tapered = synthetic_inline(), synthetic_inline(True)
    assert constant.minimum_free_flow_area == tapered.minimum_free_flow_area
    assert pressure(12.,constant,GAS).dp_drag == pressure(12.,tapered,GAS).dp_drag
    c = pressure(12.,constant,GAS,provider='genic2006_inline')
    t = pressure(12.,tapered,GAS,provider='genic2006_inline_tapered',
                 allow_engineering_estimates=True)
    assert c.diagnostics['source_domain_status'] == t.diagnostics['source_domain_status'] == 'SOURCE_DOMAIN_PASS'
    assert c.diagnostics['asurf'] != t.diagnostics['asurf']
    assert c.reference_diameter_value != t.reference_diameter_value
    assert c.dp_drag != t.dp_drag


def test_inline_auto_reports_qualified_scope_even_without_geometry_extrapolation():
    b = synthetic_inline()
    result = pressure(12.,b,GAS)
    assert result.method == 'frass_inline_solid_fin_191'
    assert 7000 < result.Re < 25000
    assert result.diagnostics['row_status'] == 'SOURCE_ROW_ANCHOR'
    assert result.diagnostics['engineering_estimate'] is True
    assert result.diagnostics['source_applicability'] == 'SOURCE_RANGE_UNRESOLVED'
    assert result.diagnostics['pressure_scope'] == 'BANK_ONLY'
    assert result.diagnostics['pressure_scope_source_closed'] is False
    assert result.diagnostics['pressure_scope_policy'] == 'EMPIRICAL_AERODYNAMIC_BANK_DRAG_ENGINEERING_POLICY'
    q = GAS.rho*result.reference_velocity**2/2
    assert result.dp_drag == pytest.approx(b.n_rows*result.Kz*result.xi*q)
    assert result.reference_velocity == 12./(GAS.rho*b.minimum_free_flow_area)
    assert result.reference_diameter_value == b.tube.D_root
