"""Transport-only regression coverage for the Shah (2009) correlation."""

import pytest

from core.geometry.tube import TubeOrientation
from core.heat_transfer.condensation_inside_shah2009 import (
    SHAH_2009_CORRELATION,
    shah2009_condensation_alpha_local,
)


# Fixed saturated-water transport snapshot at p=1 MPa. The correlation test
# intentionally has no dependency on IAPWS or any heat-exchanger object.
PLAIN_SI_INPUTS = dict(
    p=1.0e6,
    pcritical=22.064e6,
    tube_inner_diameter=0.020,
    orientation=TubeOrientation.VERTICAL_DOWNWARD,
    liquid_density=887.1274516747791,
    vapor_density=5.145385853182684,
    liquid_viscosity=0.00015048492650911248,
    vapor_viscosity=1.4981316222701132e-05,
    liquid_conductivity=0.6713377268728457,
    liquid_specific_heat=4405.112049727733,
)


@pytest.mark.parametrize(
    ("mass_flux", "quality", "regime", "alpha"),
    [
        (4.0, 0.9, "III", 16430.19830525919),
        (4.0, 0.5, "III", 9608.438257386568),
        (4.0, 0.1, "II", 8250.170609835557),
        (20.0, 0.9, "III", 9608.438257386568),
        (20.0, 0.5, "II", 8811.248447595906),
        (20.0, 0.1, "II", 5892.530573144826),
        (100.0, 0.9, "I", 15921.854888307133),
        (100.0, 0.5, "I", 11568.207006548264),
        (100.0, 0.1, "I", 4614.220063163214),
    ],
)
def test_plain_si_low_medium_high_flux_quality_grid_is_stable(
    mass_flux, quality, regime, alpha
):
    result = shah2009_condensation_alpha_local(
        mass_flux=mass_flux,
        quality=quality,
        **PLAIN_SI_INPUTS,
    )
    assert result.correlation == SHAH_2009_CORRELATION
    assert result.regime == regime
    assert result.alpha == pytest.approx(alpha, rel=2.0e-12)
    assert result.h_I == result.h_forced
    assert result.h_Nu == result.h_gravity
    assert result.p_r == result.reduced_pressure
    assert result.n == result.exponent_n
