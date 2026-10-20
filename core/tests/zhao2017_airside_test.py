# SPDX-License-Identifier: GPL-3.0-only
"""Equation-only regressions, not experimental validation."""
import pytest
from dataclasses import replace

from core.geometry import TubeGeometryApplicabilityError
from core.heat_transfer.finned_ellipse_airside import zhao2017_factors, ZHAO2017_DOMAIN
from core.heat_transfer.finned_ellipse_airside import zhao2017_flow, Zhao2017AirsideModel
from core.tests.zhao2017_fixtures import source_bundle, AIR


def factors(**overrides):
    inputs = dict(n_rows=8, reynolds=13000., Xt=(.031-.014)/.014,
                  Xl=(.060-.036)/.036, Xp=.004/.019218004, Xf=.0003/.019218004)
    inputs.update(overrides)
    return zhao2017_factors(**inputs)


def test_authorized_source_package_algebra_fixture():
    f = factors()
    assert f.j_effective == pytest.approx(.003596209097641707, rel=2e-14)
    assert f.j_prime == pytest.approx(.005572441209273412, rel=2e-14)
    assert f.friction_factor == pytest.approx(.028160566614707627, rel=2e-14)
    assert not f.extrapolated_parameters and not f.warnings


@pytest.mark.parametrize('name,lower,upper', ZHAO2017_DOMAIN)
def test_source_bounds_and_explicit_extrapolation(name, lower, upper):
    key = {'N': 'n_rows', 'Re': 'reynolds'}.get(name, name)
    for bound in (lower, upper):
        value = int(bound) if name=='N' else bound
        assert name not in factors(**{key:value}, strict_source_domain=True).extrapolated_parameters
    for value in ((int(lower)-1, int(upper)+1) if name=='N' else (lower*.99, upper*1.01)):
        extrapolated = factors(**{key:value})
        assert name in extrapolated.extrapolated_parameters
        assert any(w.code==f'zhao2017_extrapolated_{name}' for w in extrapolated.warnings)
        with pytest.raises(TubeGeometryApplicabilityError, match=name):
            factors(**{key:value}, strict_source_domain=True)


@pytest.mark.parametrize('key,value', [('n_rows',0),('n_rows',True),('n_rows',2.5),
    ('reynolds',float('nan')),('Xt',-1),('Xf',0),('Xp',float('inf'))])
def test_nonphysical_equation_inputs_reject(key,value):
    with pytest.raises(ValueError): factors(**{key:value})


def test_diameter_continuity_and_pressure_normalization():
    b = source_bundle()
    r = Zhao2017AirsideModel().evaluate(b,1.7,AIR)
    q = r.flow
    assert q.reference_diameter == pytest.approx(.019218004,abs=1e-9)
    assert q.sigma_average == pytest.approx(.7454824603355429)
    assert 0 < q.sigma_average < 1 and 0 < q.sigma_throat < 1
    assert q.reference_velocity*q.average_passage_area*AIR.rho == pytest.approx(1.7)
    assert q.reynolds_number == pytest.approx(1.7/q.average_passage_area*q.reference_diameter/AIR.mu)
    assert r.h_film == pytest.approx(r.factors.j_prime*AIR.rho*AIR.cp*q.reference_velocity/q.prandtl_number**(2/3))
    assert r.pressure_drop == pytest.approx(r.factors.friction_factor*.5*AIR.rho*q.reference_velocity**2*b.total_outer_area/q.average_passage_area)
    assert r.validated_bank_irreversible is False
    assert r.pressure_scope == 'source_domain_inlet_to_outlet_estimate'
    assert not r.extrapolated_parameters
    assert {w.code for w in r.warnings} >= {'source_geometry_simplified_rectangular_fin','zhao2017_pressure_scope'}


def test_area_and_velocity_sensitivities_and_pass_independence():
    b = source_bundle()
    q = zhao2017_flow(b,1.7,AIR)
    tighter = zhao2017_flow(replace(b,pitch_transverse=.029),1.7,AIR)
    thicker_tube = replace(b.tube,fin=replace(b.tube.fin,thickness=.0006))
    thicker = zhao2017_flow(replace(b,tube=thicker_tube),1.7,AIR)
    assert tighter.reference_velocity > q.reference_velocity
    assert thicker.average_passage_area < q.average_passage_area
    assert zhao2017_flow(replace(b,n_passes_tube=1),1.7,AIR) == q
    model = Zhao2017AirsideModel()
    # More length/tubes changes exchanger scaling; same face mass flux keeps h/f.
    long_core=replace(b.tube.core_tube,length_total=2.,length_effective=2.)
    long=replace(b,tube=replace(b.tube,core_tube=long_core))
    a,c=model.evaluate(b,1.7,AIR),model.evaluate(long,3.4,AIR)
    assert c.h_film == pytest.approx(a.h_film)
    assert c.pressure_drop == pytest.approx(a.pressure_drop)


@pytest.mark.parametrize('n',[2,4,11])
def test_low_row_and_high_row_extrapolate_without_clamping(n):
    r=Zhao2017AirsideModel().evaluate(source_bundle(n),1.7,AIR)
    assert 'N' in r.extrapolated_parameters and r.pressure_drop>0
    with pytest.raises(TubeGeometryApplicabilityError,match='N'):
        Zhao2017AirsideModel(strict_source_domain=True).evaluate(source_bundle(n),1.7,AIR)


def test_invalid_geometry_and_orientation_are_hard_errors():
    b=source_bundle()
    for bad in (replace(b,pitch_transverse=.025),replace(b,pitch_longitudinal=.040),
                replace(b,tube=replace(b.tube,core_tube=replace(b.tube.core_tube,orientation_angle=.2)))):
        with pytest.raises(TubeGeometryApplicabilityError): Zhao2017AirsideModel().evaluate(bad,1.7,AIR)


def test_conductivity_changes_no_airside_correlation():
    b=source_bundle(); modified=replace(b,tube=replace(b.tube,fin=replace(b.tube.fin,thermal_conductivity=200.)))
    assert Zhao2017AirsideModel().evaluate(b,1.7,AIR)==Zhao2017AirsideModel().evaluate(modified,1.7,AIR)
