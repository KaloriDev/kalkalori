# SPDX-License-Identifier: GPL-3.0-only
"""Equation-only regressions, not experimental validation."""
import pytest

from core.geometry import TubeGeometryApplicabilityError
from core.heat_transfer.finned_ellipse_airside import zhao2017_factors, ZHAO2017_DOMAIN


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
