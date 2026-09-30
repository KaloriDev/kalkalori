# SPDX-License-Identifier: GPL-3.0-only
"""18C physical onset sweeps with unchanged geometry and carrier/liquid flows."""
import pytest

from core.heat_transfer.wet_coil_adapters import solve_production_coil
from core.tests.wet_coil_test import production_context


# Declared numerical continuity tolerances, not model-accuracy allowances.
CONTINUITY = {
    "wet_fraction": 1e-4,
    "heat_liquid": 0.1,
    "air_out": 2e-4,
    "liquid_out": 2e-4,
    "humidity_out": 1e-7,
    "condensate": 1e-7,
    "drain_enthalpy": 0.005,
}


@pytest.mark.parametrize("finned,transition,center", [
    (False, "dry_partial", 0.0068219864228406),
    (False, "partial_full", 0.0077057271400947),
    (True, "dry_partial", 0.0077400821613445),
    (True, "partial_full", 0.0153463928),
])
def test_physical_humidity_sweep_through_general_boundary(finned, transition, center):
    """Centers locate test inputs only; no fitted constant enters the solver."""
    bundle, thermo, inside = production_context(finned)
    results = []
    for width in (1e-6, 1e-8):
        pair = []
        for humidity in (center - width, center + width):
            r = solve_production_coil(
                bundle=bundle,
                thermodynamics=thermo,
                inside=inside,
                air_in=300.15,
                liquid_in=280.15,
                humidity_in=humidity,
                dry_mass_flow=0.5,
            )
            assert r.humidity_out <= humidity
            assert r.condensate >= 0
            assert 0.5 * (humidity - r.humidity_out) == pytest.approx(
                r.condensate, abs=2e-10
            )
            gas_heat = 0.5 * (
                thermo.enthalpy(300.15, humidity)
                - thermo.enthalpy(r.air_out, r.humidity_out)
            )
            liquid_heat = inside.mass_flow * inside.enthalpy_difference(
                280.15, r.liquid_out
            )
            assert gas_heat == pytest.approx(
                liquid_heat + r.drain_enthalpy, abs=0.002
            )
            pair.append(r)
        expected = (
            ("DRY", "PARTIALLY_WET") if transition == "dry_partial"
            else ("PARTIALLY_WET", "FULLY_WET")
        )
        assert tuple(r.regime for r in pair) == expected
        results.append(pair)
    coarse, fine = results
    for name, tolerance in CONTINUITY.items():
        near = abs(getattr(fine[1], name) - getattr(fine[0], name))
        far = abs(getattr(coarse[1], name) - getattr(coarse[0], name))
        assert near < tolerance, (name, near, tolerance)
        assert near < 0.1 * far, (name, near, far)
    if transition == "dry_partial":
        assert 0 < fine[1].wet_fraction < CONTINUITY["wet_fraction"]
    else:
        assert 1 - CONTINUITY["wet_fraction"] < fine[0].wet_fraction < 1
