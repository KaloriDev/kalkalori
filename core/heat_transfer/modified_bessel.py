# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only

"""Dependency-free modified Bessel functions used by annular-fin models.

The approximations below are the classical polynomial/asymptotic forms for
``I0``, ``I1``, ``K0`` and ``K1`` tabulated from Abramowitz and Stegun,
sections 9.8.1--9.8.8.  Their relative accuracy is approximately ``2e-7``.

Exponentially scaled variants are provided because an annular-fin solution
contains products of exponentially growing ``I`` and exponentially decaying
``K`` functions.  Evaluating those products from the unscaled functions can
overflow even when their ratio, and therefore the physical result, is finite.
"""

from __future__ import annotations

import math


_I_SWITCH = 3.75
_K_SWITCH = 2.0


def bessel_i0(x: float) -> float:
    """Return the modified Bessel function ``I0(x)`` for finite real ``x``."""
    _validate_finite(x, "bessel_i0")
    absolute_x = abs(x)
    if absolute_x < _I_SWITCH:
        return _i0_small(absolute_x)
    return math.exp(absolute_x) * _i0e_large(absolute_x)


def bessel_i1(x: float) -> float:
    """Return the modified Bessel function ``I1(x)`` for finite real ``x``."""
    _validate_finite(x, "bessel_i1")
    absolute_x = abs(x)
    if absolute_x < _I_SWITCH:
        result = _i1_small(absolute_x)
    else:
        result = math.exp(absolute_x) * _i1e_large(absolute_x)
    return -result if x < 0.0 else result


def bessel_k0(x: float) -> float:
    """Return the modified Bessel function ``K0(x)`` for finite ``x > 0``."""
    _validate_positive_finite(x, "bessel_k0")
    if x <= _K_SWITCH:
        return _k0_small(x)
    return math.exp(-x) * _k0e_large(x)


def bessel_k1(x: float) -> float:
    """Return the modified Bessel function ``K1(x)`` for finite ``x > 0``."""
    _validate_positive_finite(x, "bessel_k1")
    if x <= _K_SWITCH:
        return _k1_small(x)
    return math.exp(-x) * _k1e_large(x)


def scaled_bessel_i0(x: float) -> float:
    """Return ``exp(-abs(x)) * I0(x)`` without large-argument overflow."""
    _validate_finite(x, "scaled_bessel_i0")
    absolute_x = abs(x)
    if absolute_x < _I_SWITCH:
        return math.exp(-absolute_x) * _i0_small(absolute_x)
    return _i0e_large(absolute_x)


def scaled_bessel_i1(x: float) -> float:
    """Return ``exp(-abs(x)) * I1(x)`` without large-argument overflow."""
    _validate_finite(x, "scaled_bessel_i1")
    absolute_x = abs(x)
    if absolute_x < _I_SWITCH:
        result = math.exp(-absolute_x) * _i1_small(absolute_x)
    else:
        result = _i1e_large(absolute_x)
    return -result if x < 0.0 else result


def scaled_bessel_k0(x: float) -> float:
    """Return ``exp(x) * K0(x)`` without large-argument underflow."""
    _validate_positive_finite(x, "scaled_bessel_k0")
    if x <= _K_SWITCH:
        return math.exp(x) * _k0_small(x)
    return _k0e_large(x)


def scaled_bessel_k1(x: float) -> float:
    """Return ``exp(x) * K1(x)`` without large-argument underflow."""
    _validate_positive_finite(x, "scaled_bessel_k1")
    if x <= _K_SWITCH:
        return math.exp(x) * _k1_small(x)
    return _k1e_large(x)


def _i0_small(x: float) -> float:
    y = (x / _I_SWITCH) ** 2
    return 1.0 + y * (
        3.5156229
        + y
        * (
            3.0899424
            + y
            * (
                1.2067492
                + y * (0.2659732 + y * (0.0360768 + y * 0.0045813))
            )
        )
    )


def _i0e_large(x: float) -> float:
    y = _I_SWITCH / x
    polynomial = 0.39894228 + y * (
        0.01328592
        + y
        * (
            0.00225319
            + y
            * (
                -0.00157565
                + y
                * (
                    0.00916281
                    + y
                    * (
                        -0.02057706
                        + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))
                    )
                )
            )
        )
    )
    return polynomial / math.sqrt(x)


def _i1_small(x: float) -> float:
    y = (x / _I_SWITCH) ** 2
    return x * (
        0.5
        + y
        * (
            0.87890594
            + y
            * (
                0.51498869
                + y
                * (
                    0.15084934
                    + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))
                )
            )
        )
    )


def _i1e_large(x: float) -> float:
    y = _I_SWITCH / x
    polynomial = 0.39894228 + y * (
        -0.03988024
        + y
        * (
            -0.00362018
            + y
            * (
                0.00163801
                + y
                * (
                    -0.01031555
                    + y
                    * (
                        0.02282967
                        + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059))
                    )
                )
            )
        )
    )
    return polynomial / math.sqrt(x)


def _k0_small(x: float) -> float:
    y = (x / _K_SWITCH) ** 2
    polynomial = -0.57721566 + y * (
        0.42278420
        + y
        * (
            0.23069756
            + y
            * (
                0.03488590
                + y * (0.00262698 + y * (0.00010750 + y * 0.00000740))
            )
        )
    )
    return -math.log(x / _K_SWITCH) * _i0_small(x) + polynomial


def _k0e_large(x: float) -> float:
    y = _K_SWITCH / x
    polynomial = 1.25331414 + y * (
        -0.07832358
        + y
        * (
            0.02189568
            + y
            * (
                -0.01062446
                + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208))
            )
        )
    )
    return polynomial / math.sqrt(x)


def _k1_small(x: float) -> float:
    y = (x / _K_SWITCH) ** 2
    polynomial = 1.0 + y * (
        0.15443144
        + y
        * (
            -0.67278579
            + y
            * (
                -0.18156897
                + y * (-0.01919402 + y * (-0.00110404 - y * 0.00004686))
            )
        )
    )
    return math.log(x / _K_SWITCH) * _i1_small(x) + polynomial / x


def _k1e_large(x: float) -> float:
    y = _K_SWITCH / x
    polynomial = 1.25331414 + y * (
        0.23498619
        + y
        * (
            -0.03655620
            + y
            * (
                0.01504268
                + y * (-0.00780353 + y * (0.00325614 - y * 0.00068245))
            )
        )
    )
    return polynomial / math.sqrt(x)


def _validate_finite(x: float, function_name: str) -> None:
    if not math.isfinite(x):
        raise ValueError(f"{function_name} requires a finite real argument.")


def _validate_positive_finite(x: float, function_name: str) -> None:
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError(
            f"{function_name} requires a finite argument greater than zero."
        )


__all__ = [
    "bessel_i0",
    "bessel_i1",
    "bessel_k0",
    "bessel_k1",
    "scaled_bessel_i0",
    "scaled_bessel_i1",
    "scaled_bessel_k0",
    "scaled_bessel_k1",
]
