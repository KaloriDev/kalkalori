# KalKalori — Heat Exchanger Open Engine
# Copyright (C) 2025  KalKalori Project Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# -------------------------------------------------------------------------
# MODIFIED BESSEL FUNCTIONS I0, I1, K0, K1 (rational/polynomial approximation)
# -------------------------------------------------------------------------
#
# The annular-fin efficiency solution (Gardner 1945; see
# core.heat_transfer.fin_efficiency) requires the modified Bessel
# functions of the first kind (I0, I1) and second kind (K0, K1). The
# project intentionally avoids adding scipy as a production dependency
# for this single use (see core.heat_transfer.fin_efficiency module
# docstring / docs/finned_tube_model.md: "no unjustified new production
# dependency"), so these are implemented directly from the well-known
# Abramowitz & Stegun polynomial/rational approximations:
#
#   Abramowitz, M.; Stegun, I.A. (1964), "Handbook of Mathematical
#   Functions", National Bureau of Standards, Applied Mathematics Series
#   55, sections 9.8.1-9.8.8. (Public domain, U.S. government
#   publication.)
#
# Advertised accuracy of these approximations is ~1.6e-7 to 1.9e-7
# relative error. Correctness here is verified in
# core/tests/finned_tube_fin_efficiency_test.py against exact reference
# values and, independently of any transcription of the coefficients
# below, against the modified-Bessel Wronskian identity, which holds
# exactly for the true functions:
#
#   I0(x)*K1(x) + I1(x)*K0(x) = 1/x   for all x > 0
#
# -------------------------------------------------------------------------

from __future__ import annotations

import math


def bessel_i0(x: float) -> float:
    """Modified Bessel function of the first kind, order 0."""
    ax = abs(x)
    if ax < 3.75:
        t = (x / 3.75) ** 2
        return (
            1.0
            + t
            * (
                3.5156229
                + t
                * (
                    3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))
                )
            )
        )
    t = 3.75 / ax
    poly = (
        0.39894228
        + t
        * (
            0.01328592
            + t
            * (
                0.00225319
                + t
                * (
                    -0.00157565
                    + t
                    * (
                        0.00916281
                        + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))
                    )
                )
            )
        )
    )
    return (math.exp(ax) / math.sqrt(ax)) * poly


def bessel_i1(x: float) -> float:
    """Modified Bessel function of the first kind, order 1."""
    ax = abs(x)
    if ax < 3.75:
        t = (x / 3.75) ** 2
        result = ax * (
            0.5
            + t
            * (
                0.87890594
                + t
                * (0.51498869 + t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))
            )
        )
    else:
        t = 3.75 / ax
        poly = (
            0.39894228
            + t
            * (
                -0.03988024
                + t
                * (
                    -0.00362018
                    + t
                    * (
                        0.00163801
                        + t
                        * (
                            -0.01031555
                            + t * (0.02282967 + t * (-0.02895312 + t * (0.01787654 + t * -0.00420059)))
                        )
                    )
                )
            )
        )
        result = (math.exp(ax) / math.sqrt(ax)) * poly
    return -result if x < 0.0 else result


def bessel_k0(x: float) -> float:
    """Modified Bessel function of the second kind, order 0. Requires x > 0."""
    if x <= 0.0:
        raise ValueError("bessel_k0 requires x > 0.")
    if x <= 2.0:
        t = (x / 2.0) ** 2
        poly = (
            -0.57721566
            + t
            * (
                0.42278420
                + t
                * (0.23069756 + t * (0.03488590 + t * (0.00262698 + t * (0.00010750 + t * 0.00000740))))
            )
        )
        return -math.log(x / 2.0) * bessel_i0(x) + poly
    t = 2.0 / x
    poly = (
        1.25331414
        + t
        * (
            -0.07832358
            + t * (0.02189568 + t * (-0.01062446 + t * (0.00587872 + t * (-0.00251540 + t * 0.00053208))))
        )
    )
    return (math.exp(-x) / math.sqrt(x)) * poly


def bessel_k1(x: float) -> float:
    """Modified Bessel function of the second kind, order 1. Requires x > 0."""
    if x <= 0.0:
        raise ValueError("bessel_k1 requires x > 0.")
    if x <= 2.0:
        t = (x / 2.0) ** 2
        poly = (
            1.0
            + t
            * (
                0.15443144
                + t
                * (
                    -0.67278579
                    + t * (-0.18156897 + t * (-0.01919402 + t * (-0.00110404 + t * -0.00004686)))
                )
            )
        )
        return math.log(x / 2.0) * bessel_i1(x) + poly / x
    t = 2.0 / x
    poly = (
        1.25331414
        + t
        * (
            0.23498619
            + t * (-0.03655620 + t * (0.01504268 + t * (-0.00780353 + t * (0.00325614 + t * -0.00068245))))
        )
    )
    return (math.exp(-x) / math.sqrt(x)) * poly
