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


@pytest.mark.parametrize("derivatives", [False, True])
def test_whole_fin_batch_matches_scalar_face_integrals(derivatives, monkeypatch):
    from types import SimpleNamespace
    from core.heat_transfer import wet_coil_radial as radial
    from core.phase_change.wet_finned_surface import (
        _build_fin_mesh, _build_prescribed_base_chain,
    )
    from core.tests.wet_coil_test import production_context

    bundle, _, _ = production_context(finned=True)
    chain = _build_prescribed_base_chain(_build_fin_mesh(bundle.tube, 64), 280.0)
    temperatures = np.linspace(280.0, 300.0, len(chain.surface_areas))
    scalar = SurfaceProperties(101325.0, 0.029, 0.01801528, 0.016)
    batched = SurfaceProperties(101325.0, 0.029, 0.01801528, 0.016)
    arguments = dict(dew=scalar.dew(), alpha=50.0, cp=1050.0, W=0.016,
                     lewis=1.0, with_derivatives=derivatives)
    expected = radial.integrate_fin_cells(
        chain, temperatures,
        properties=SimpleNamespace(values=scalar.values, derivatives=scalar.derivatives),
        **arguments,
    )
    batches = []
    original = radial._saturation_enthalpy_pairs_batch

    def record(temperatures):
        batches.append(len(temperatures))
        return original(temperatures)

    monkeypatch.setattr(radial, "_saturation_enthalpy_pairs_batch", record)
    actual = radial.integrate_fin_cells(
        chain, temperatures, properties=batched, **arguments,
    )
    assert actual == expected
    assert len(batches) == 1
    assert batches[0] > 64


def test_radial_initial_guess_preserves_converged_surface():
    from core.heat_transfer.wet_coil_surface import annular_response
    from core.tests.wet_coil_test import production_context

    bundle, _, _ = production_context(finned=True)
    arguments = dict(gas_temperature=300.0, humidity=0.016, alpha=50.0,
                     cp_dry=1050.0, pressure=101325.0, M_dry=0.029,
                     M_water=0.01801528)
    previous = annular_response(bundle.tube, base_temperature=280.0, **arguments)
    initial = tuple(280.1 + (300.0 - 280.1) * (T - 280.0) / 20.0
                    for T in previous.radial_temperatures)
    fresh = annular_response(bundle.tube, base_temperature=280.1, **arguments)
    reused = annular_response(bundle.tube, base_temperature=280.1,
                              _initial_temperatures=initial, **arguments)
    assert reused.iterations < fresh.iterations
    np.testing.assert_allclose(reused.radial_temperatures, fresh.radial_temperatures,
                               rtol=0, atol=1e-8)
    for field in ("heat_liquid", "drain_enthalpy", "wet_area", "heat_sensible"):
        assert getattr(reused, field) == pytest.approx(getattr(fresh, field), abs=1e-8)
    assert reused.condensate == pytest.approx(fresh.condensate, abs=2e-14)
    assert abs(reused.energy_residual) < 2e-8


def test_converged_radial_seed_needs_only_residual_and_independent_quadrature(monkeypatch):
    from core.heat_transfer import wet_coil_surface as surface
    from core.tests.wet_coil_test import production_context

    bundle, _, _ = production_context(finned=True)
    arguments = dict(base_temperature=280.0, gas_temperature=300.0, humidity=0.016,
                     alpha=50.0, cp_dry=1050.0, pressure=101325.0,
                     M_dry=0.029, M_water=0.01801528)
    accepted = surface.annular_response(bundle.tube, **arguments)
    calls = []
    original = surface.integrate_fin_cells

    def integrate(*args, **kwargs):
        calls.append((kwargs["order"], kwargs["with_derivatives"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(surface, "integrate_fin_cells", integrate)
    reused = surface.annular_response(
        bundle.tube, _initial_temperatures=accepted.radial_temperatures, **arguments)
    assert reused.iterations == 1
    assert calls == [(4, False), (8, False)]
    assert reused.heat_liquid == accepted.heat_liquid
    assert reused.drain_enthalpy == accepted.drain_enthalpy
    assert reused.condensate == accepted.condensate
