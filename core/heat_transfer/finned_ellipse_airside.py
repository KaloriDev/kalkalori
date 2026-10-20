# SPDX-License-Identifier: GPL-3.0-only
"""Zhao et al. (2017), IJHMT 114, 1314-1330, Eqs.18-20.

DOI 10.1016/j.ijheatmasstransfer.2017.06.049. Equations transcribed from the
user-authorized 2026-10-20 source package; no claim of free full-text access.
Eq.20 excludes fin conduction. Eq.18 is diagnostic only. Flow normalization
is a separately identified engineering reconstruction, not a printed equation.
"""
from dataclasses import dataclass
import math

from core.common.warnings import ModelWarning, make_warning
from core.geometry.applicability import TubeGeometryApplicabilityError


ZHAO2017_CITATION = (
    'Zhao, Gu, Gao, Yang (2017), IJHMT 114, 1314-1330; '
    'doi:10.1016/j.ijheatmasstransfer.2017.06.049')
ZHAO2017_DOMAIN = (
    ('Xt', .93, 1.50), ('Xl', .56, .78), ('Xp', .10, .52),
    ('Xf', .01, .03), ('Re', 5500., 23800.), ('N', 6., 10.))


def _positive(**values):
    for name, value in values.items():
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite [SI].')


def _extrapolation(values, strict):
    outside = tuple(name for name, lower, upper in ZHAO2017_DOMAIN
                    if not lower <= values[name] <= upper)
    if outside and strict:
        raise TubeGeometryApplicabilityError(
            f'Zhao2017 source domain exceeded: {", ".join(outside)}')
    warnings = tuple(make_warning(
        code=f'zhao2017_extrapolated_{name}', source='zhao2017',
        message=f'Zhao2017 extrapolation: {name}={values[name]:.12g} is outside its source domain; no clipping.')
        for name in outside)
    return outside, warnings


@dataclass(frozen=True)
class Zhao2017Factors:
    """Dimensionless equation outputs, independent of flow reconstruction."""
    j_effective: float
    j_prime: float
    friction_factor: float
    extrapolated_parameters: tuple[str, ...]
    warnings: tuple[ModelWarning, ...]


def zhao2017_factors(*, n_rows, reynolds, Xt, Xl, Xp, Xf,
                     strict_source_domain=False):
    """Evaluate source Eqs.18/19/20 exactly; never silently clamp inputs.

    The source fit uses rows 6..10. Other positive integer counts are an
    explicit extrapolation, not a new measured row law.
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
        raise ValueError('n_rows must be a positive integer.')
    _positive(reynolds=reynolds, Xt=Xt, Xl=Xl, Xp=Xp, Xf=Xf)
    if not isinstance(strict_source_domain, bool):
        raise TypeError('strict_source_domain must be boolean.')
    outside, warnings = _extrapolation(
        dict(N=n_rows, Re=reynolds, Xt=Xt, Xl=Xl, Xp=Xp, Xf=Xf),
        strict_source_domain)
    j_effective = (1.0062*n_rows**(-.2045)*Xt**(-.7268)*Xl**.1319
                   *Xp**.1381*Xf**.1495*reynolds**(-.4408))
    friction = (1.3549*n_rows**(-.0591)*Xt**(-.9114)*Xl**.4709
                *Xp**.1296*Xf**.0438*reynolds**(-.3164))
    j_prime = (.2*n_rows**(-.2160)*Xt**(-.82)*Xl**.1361
               *Xp**.0043*Xf**(-.0578)*reynolds**(-.3326))
    _positive(j_effective=j_effective, j_prime=j_prime, friction_factor=friction)
    return Zhao2017Factors(j_effective, j_prime, friction, outside, warnings)
