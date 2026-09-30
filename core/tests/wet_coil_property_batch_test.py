# SPDX-License-Identifier: GPL-3.0-only
"""Grouped native property queries must retain the scalar IF97 answers."""
import numpy as np
import pytest

from core.properties.water import (
    _saturation_enthalpy_pairs_batch,
    water_saturation_liquid_enthalpy,
    water_saturation_vapor_enthalpy,
)
from core.heat_transfer.wet_coil_radial import SurfaceProperties


def test_batched_enthalpies_preserve_scalar_backend_and_region_boundaries():
    temperatures = [*np.linspace(273.16, 623.14, 1024), 623.15, 623.16, 640.0]
    expected = [
        (water_saturation_liquid_enthalpy(T=T), water_saturation_vapor_enthalpy(T=T))
        for T in temperatures
    ]
    assert _saturation_enthalpy_pairs_batch(()) == ()
    for start in range(0, len(temperatures), 12):
        actual = _saturation_enthalpy_pairs_batch(temperatures[start:start + 12])
        np.testing.assert_array_equal(actual, expected[start:start + 12])


@pytest.mark.parametrize("derivatives", [False, True])
def test_radial_prefetch_preserves_values_and_derivatives(derivatives):
    temperatures = [273.16, 273.16005, 280.0, 300.125, 350.0, 390.0]
    scalar = SurfaceProperties(101325.0, 0.029, 0.01801528, 0.016)
    batched = SurfaceProperties(101325.0, 0.029, 0.01801528, 0.016)
    batched.prefetch(temperatures, with_derivatives=derivatives)
    for T in temperatures:
        assert batched.values(T) == scalar.values(T)
        if derivatives:
            np.testing.assert_array_equal(batched.derivatives(T), scalar.derivatives(T))
