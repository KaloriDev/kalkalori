# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic acceptance of the source-profile/drain equations and adapters."""
from dataclasses import replace
import json
from pathlib import Path
from math import log, pi

import pytest
from scipy.optimize import brentq

from core.heat_transfer.elmahdy_mitalas import solve_source_coil, _counterflow_transfer
from core.heat_transfer.wet_coil import solve_wet_coil, _Region, WetCoilModelError
from core.tests.elmahdy_mitalas_test import source_case, _pressure


def ws(T):
    p = _pressure(T)
    return 0.62198 * p / (101325 - p)


def physical_case(W):
    x = source_case(W)
    # Synthetic bare surface: same physical HTC/area and inside resistance on
    # both sides of onset. No source empirical wet multiplier or fin fit.
    return replace(
        x,
        wet_air_resistance=x.gas_cp * x.dry_air_resistance,
        inner_resistance=lambda T: 0.0001,
    )


def solve(x, **kwargs):
    return solve_wet_coil(
        x, liquid_enthalpy=lambda T: 4180 * T, saturation_humidity=ws, **kwargs
    )


@pytest.mark.parametrize(
    "W,regime", [(0.004, "DRY"), (0.008, "PARTIALLY_WET"), (0.016, "FULLY_WET")]
)
def test_drain_enabled_energy_mass_and_local_admissibility(W, regime):
    x = physical_case(W)
    r = solve(x)
    assert r.regime == regime
    # Independent endpoint balances, not simply the diagnostic residual.
    assert x.dry_mass_flow * (x.gas_h_in - r.outlet_enthalpy) == pytest.approx(
        x.liquid_capacity * (r.liquid_out - x.liquid_in) + r.drain_enthalpy, abs=2e-4
    )
    assert r.condensate == pytest.approx(
        x.dry_mass_flow * (W - r.humidity_out), abs=2e-10
    )
    assert r.humidity_out <= W
    assert r.condensate >= 0
    for p in r.profile:
        assert p.humidity >= ws(p.surface_temperature) - 1e-9
        assert p.humidity <= ws(p.gas_temperature) + 1e-9
        assert p.condensate_density >= -1e-11
    if regime != "DRY":
        assert r.drain_enthalpy > 0
        assert r.diagnostics["mass_residual"] == pytest.approx(0, abs=2e-10)
        no_drain = solve(x, drain_enabled=False)
        assert abs(no_drain.liquid_out - r.liquid_out) > 1e-5


def onset_margin(W):
    x = physical_case(W)
    ri = x.inner_resistance(x.liquid_in)
    q = _counterflow_transfer(
        1 / (ri + x.dry_air_resistance), x.dry_mass_flow * x.gas_cp, x.liquid_capacity
    ) * (x.air_in - x.liquid_in)
    ta = x.air_in - q / (x.dry_mass_flow * x.gas_cp)
    return (
        x.liquid_in + (ta - x.liquid_in) * ri / (ri + x.dry_air_resistance) - x.dewpoint
    )


def test_continuous_physical_onset_without_output_clipping():
    onset = brentq(onset_margin, 0.005, 0.008, xtol=1e-14)
    dry = solve(physical_case(onset - 1e-7))
    assert dry.regime == "DRY"
    wet = [solve(physical_case(onset + delta)) for delta in (1e-5, 1e-6, 1e-7)]
    assert all(r.regime == "PARTIALLY_WET" for r in wet)
    assert 0 < wet[2].condensate < wet[1].condensate < wet[0].condensate
    assert wet[2].condensate < 1e-10
    assert wet[2].heat_liquid == pytest.approx(dry.heat_liquid, abs=0.02)
    assert wet[2].air_out == pytest.approx(dry.air_out, abs=1e-4)


@pytest.mark.parametrize("W", [0.00665, 0.0067, 0.008])
def test_source_inadmissibility_is_reported_not_clipped(W):
    with pytest.raises(WetCoilModelError) as caught:
        solve(source_case(W))
    d = caught.value.diagnostics
    assert d["onset_margin"] < 0  # Dry fallback would be physically wrong.
    assert d["min_wet_driving_force"] < 0
    assert "negative" in d["rejection_reason"]


def test_reference_limit_uses_same_region_equations_and_existing_tolerances():
    from core.heat_transfer.wet_coil import _solve_profile_candidate, validate_wet_coil

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/elmahdy_mitalas_native.json").read_text()
    )
    for case in fixture["cases"]:
        x = source_case(
            case["W_in"], upstream_fins=case.get("input_family") == "native_fins"
        )
        reference = solve_source_coil(x)
        # Exactly the candidate function used by the real property iteration.
        # It solves f itself; no source result is supplied to the process solve.
        r = _solve_profile_candidate(
            x,
            liquid_enthalpy=lambda T: 4180 * T,
            saturation_humidity=ws,
            drain_enabled=False,
        )
        assert r.heat_liquid == pytest.approx(case["Q_W"], rel=0.002)
        assert r.air_out == pytest.approx(case["air_out_C"], abs=0.08)
        assert r.liquid_out == pytest.approx(case["water_out_C"], abs=0.08)
        assert r.humidity_out == pytest.approx(case["W_out"], abs=4e-5)
        assert r.condensate == pytest.approx(case["condensate_kg_s"], abs=2e-5)
        assert r.wet_fraction == pytest.approx(case["wet_fraction"], abs=0.002)
        assert r.interface_air == pytest.approx(case["interface_air_C"], abs=0.08)
        assert r.outlet_enthalpy == pytest.approx(reference.outlet_enthalpy, abs=2e-4)
        if r.diagnostics["rejection_reason"]:
            # Source approximations can violate local wet admissibility despite
            # valid-looking outlet endpoints. Production never bypasses it.
            with pytest.raises(WetCoilModelError):
                validate_wet_coil(r)
        else:
            assert validate_wet_coil(r) is r


def test_drain_quadrature_convergence_and_equal_capacity_limit():
    x = physical_case(0.016)
    # Cw/b = md is the actual equal-capacity limit of the enthalpy equations.
    slope = (x.saturation_enthalpy(15) - x.saturation_enthalpy(8)) / 7
    x = replace(x, liquid_capacity=x.dry_mass_flow * slope)
    results = [solve(x, quadrature_order=n) for n in (8, 12)]
    assert results[0].heat_liquid == pytest.approx(results[1].heat_liquid, abs=2e-4)
    assert results[0].drain_enthalpy == pytest.approx(
        results[1].drain_enthalpy, abs=2e-4
    )
    assert results[0].humidity_out == pytest.approx(results[1].humidity_out, abs=2e-10)
    assert all(
        r.diagnostics["energy_residual"] == pytest.approx(0, abs=2e-4) for r in results
    )


def test_vanishing_dry_region_continuity():
    def boundary(W):
        x = physical_case(W)
        return (
            _Region(
                x, 1.0, lambda T: 4180 * T, drain=True, order=8, boundary_secant=True
            ).interface_surface
            - x.dewpoint
        )

    transition = brentq(boundary, 0.008, 0.012, xtol=1e-11)
    a, b = [solve(physical_case(transition + d)) for d in (-1e-7, 1e-7)]
    assert a.regime == "PARTIALLY_WET"
    assert b.regime == "FULLY_WET"
    assert a.wet_fraction > 0.999
    assert a.air_out == pytest.approx(b.air_out, abs=0.001)
    assert a.heat_liquid == pytest.approx(b.heat_liquid, abs=0.2)


def production_context(finned=False):
    pytest.importorskip("CoolProp")
    from core.tests.circular_finned_tube_integration_test import _bundle
    from core.properties.gas_mixture import (
        GasMixturePropertyProvider,
        gas_mixture_from_dry_composition_and_water_ratio,
    )
    from core.phase_change.capability import detect_phase_change_capability
    from core.properties.water import IAPWS97WaterSteamProvider
    from core.heat_transfer.wet_coil_adapters import (
        WetGasThermodynamics,
        InsideWallAdapter,
    )

    bundle = _bundle(finned=finned)
    cap = detect_phase_change_capability(
        GasMixturePropertyProvider(
            gas_mixture_from_dry_composition_and_water_ratio(
                {"N2": 0.77, "O2": 0.15, "CO2": 0.08},
                water_ratio=0.016,
                dry_basis="mole",
            )
        )
    )
    thermo = WetGasThermodynamics(101325.0, cap)
    inside = InsideWallAdapter(bundle, IAPWS97WaterSteamProvider(), 0.4, 2e5)
    return bundle, thermo, inside


def test_configured_thermodynamics_and_cylindrical_wall():
    b, t, i = production_context()
    assert t.capability.dry_mole_fractions["CarbonDioxide"] == pytest.approx(0.08)
    for T, W in ((285, 0.006), (310, 0.016), (350, 0.02)):
        h = t.enthalpy(T, W)
        assert t.humidity(T, h) == pytest.approx(W, abs=1e-12)
        assert t.temperature(h, W) == pytest.approx(T, abs=1e-7)
    assert t.saturation_temperature(t.saturation_enthalpy(290)) == pytest.approx(
        290, abs=1e-7
    )
    ri, d = i.evaluate(285)
    assert d["fouling_resistance"] == 0
    assert d["wall_resistance"] == pytest.approx(
        log(b.tube.D_o / b.tube.D_i)
        / (2 * pi * b.tube.wall_k * b.tube.length_effective * b.n_tubes_total)
    )
    assert ri == pytest.approx(d["wall_resistance"] + d["film_resistance"])


def test_bare_adapter_actual_composition_and_energy():
    from core.heat_transfer.wet_coil_adapters import solve_production_coil

    b, t, i = production_context()
    r = solve_production_coil(
        bundle=b,
        thermodynamics=t,
        inside=i,
        air_in=300.15,
        liquid_in=280.15,
        humidity_in=0.016,
        dry_mass_flow=0.5,
    )
    assert r.regime == "FULLY_WET"
    assert 0.5 * (
        t.enthalpy(300.15, 0.016) - t.enthalpy(r.air_out, r.humidity_out)
    ) == pytest.approx(
        i.mass_flow * i.enthalpy_difference(280.15, r.liquid_out) + r.drain_enthalpy,
        abs=0.002,
    )
    d = r.diagnostics
    assert d["dry_effective_area"] == d["wet_effective_area"] == b.total_outer_area
    assert d["outside_htc_model"] == "Zukauskas"
    assert d["property_provider"] == "IAPWS97WaterSteamProvider"


def test_annular_surface_continuous_wet_front_and_radial_convergence():
    from core.tests.circular_finned_tube_integration_test import _finned_tube
    from core.heat_transfer.wet_coil_surface import annular_response

    tube = replace(_finned_tube(), fin_thickness_tip=0.00025)
    kwargs = dict(
        base_temperature=285.0,
        gas_temperature=300.0,
        humidity=0.01,
        alpha=40.0,
        cp_dry=1035.0,
        pressure=101325.0,
        M_dry=0.029,
        M_water=0.01801528,
    )
    a, b = [annular_response(tube, radial_cells=n, **kwargs) for n in (64, 128)]
    assert 0 < a.wet_area < tube.fin_area_per_fin
    assert a.condensate == pytest.approx(b.condensate, rel=0.002)
    assert a.drain_enthalpy == pytest.approx(b.drain_enthalpy, rel=0.002)
    assert a.heat_liquid == pytest.approx(b.heat_liquid, rel=0.001)
    assert abs(a.energy_residual) < 1e-7


@pytest.mark.parametrize(
    "humidity,expected", [(0.004, "DRY"), (0.007, "PARTIALLY_WET")]
)
def test_real_property_dry_and_partial_branches(humidity, expected):
    from core.heat_transfer.wet_coil_adapters import solve_production_coil

    b, t, i = production_context()
    r = solve_production_coil(
        bundle=b,
        thermodynamics=t,
        inside=i,
        air_in=300.15,
        liquid_in=280.15,
        humidity_in=humidity,
        dry_mass_flow=0.5,
    )
    assert r.regime == expected
    assert 0.5 * (
        t.enthalpy(300.15, humidity) - t.enthalpy(r.air_out, r.humidity_out)
    ) == pytest.approx(
        i.mass_flow * i.enthalpy_difference(280.15, r.liquid_out) + r.drain_enthalpy,
        abs=0.002,
    )
    if expected == "PARTIALLY_WET":
        assert r.diagnostics["min_wet_driving_force"] >= -1e-9
        assert r.condensate > 0


@pytest.mark.parametrize(
    "humidity,expected", [(0.01, "PARTIALLY_WET"), (0.016, "FULLY_WET")]
)
def test_finned_adapter_uses_same_engine_and_radial_drain(humidity, expected):
    from core.heat_transfer.wet_coil_adapters import solve_production_coil

    b, t, i = production_context(finned=True)
    r = solve_production_coil(
        bundle=b,
        thermodynamics=t,
        inside=i,
        air_in=300.15,
        liquid_in=280.15,
        humidity_in=humidity,
        dry_mass_flow=0.5,
    )
    assert r.regime == expected
    assert 0.5 * (
        t.enthalpy(300.15, humidity) - t.enthalpy(r.air_out, r.humidity_out)
    ) == pytest.approx(
        i.mass_flow * i.enthalpy_difference(280.15, r.liquid_out) + r.drain_enthalpy,
        abs=0.002,
    )
    assert r.diagnostics["geometry_adapter"] == "CircularFinnedTube"
    assert r.diagnostics["outside_htc_model"] == "Briggs-Young 1963"
    assert r.diagnostics["radial_drain_enthalpy_iteration_error_J_kg"] < 0.002
    assert r.diagnostics["wet_effective_area"] <= b.total_outer_area
    for p in r.diagnostics["surface_states"]:
        assert (
            p["inside_wall_temperature"]
            <= p["core_wall_temperature"]
            <= p["surface_base_temperature"]
            <= p["fin_tip_temperature"]
        )


def test_glycol_and_parallel_fin_contact_are_actual_adapters():
    from core.heat_transfer.wet_coil_adapters import (
        InsideWallAdapter,
        CircularFinnedTubeAdapter,
    )
    from core.properties.coolprop_backend import CoolPropFluidProvider

    b, t, i = production_context(finned=True)
    glycol = InsideWallAdapter(b, CoolPropFluidProvider("INCOMP::MEG-30%"), 0.4, 2e5)
    ri, d = glycol.evaluate(285.0)
    assert ri > 0
    assert d["property_provider"] == "CoolPropFluidProvider"
    assert glycol.capacity(280.0, 290.0) != pytest.approx(
        i.capacity(280.0, 290.0), rel=0.01
    )
    assert glycol.mass_flow * glycol.enthalpy_difference(280.0, 290.0) == pytest.approx(
        glycol.capacity(280.0, 290.0) * 10, rel=1e-12
    )
    tube = replace(b.tube, D_root=b.tube.D_o, fin_contact_resistance=2e-4)
    b = replace(b, tube=tube)
    i = replace(i, bundle=b)
    outside = CircularFinnedTubeAdapter(b, t)
    _, diag = outside.evaluate(300.0, 0.016, 0.5, i.evaluate(285.0))
    assert diag["common_root_contact_resistance"] == 0
    response = outside.response(285.0, 300.0, 0.016, diag)
    assert response["fin_base"] > 285.0
    assert response["fin_tip"] > response["fin_base"]
    assert response["local_condensate_response"] > 0


def test_exact_enthalpy_capacity_equality_with_drain():
    from core.heat_transfer.elmahdy_mitalas import CoilInput

    x = CoilInput(
        30.0,
        5.0,
        0.014,
        0.5,
        1000.0,
        1000.0,
        65000.0,
        27.5,
        0.002,
        2.0,
        lambda T: 0.0001,
        lambda T: 7500.0 + 2000.0 * T,
        lambda h: (h - 7500.0) / 2000.0,
        lambda T, h: (h - 1000.0 * T) / 2.5e6,
    )

    def run(cw):
        return solve_wet_coil(
            replace(x, liquid_capacity=cw),
            liquid_enthalpy=lambda T: 4180 * T,
            saturation_humidity=lambda T: 0.003 + 0.0004 * T,
        )

    mid = run(1000.0)
    assert mid.regime == "FULLY_WET"
    for cw in (1000.0 - 1e-5, 1000.0 + 1e-5):
        nearby = run(cw)
        assert nearby.heat_liquid == pytest.approx(mid.heat_liquid, abs=1e-4)
        assert nearby.humidity_out == pytest.approx(mid.humidity_out, abs=1e-9)
    assert mid.heat_gas == pytest.approx(mid.heat_liquid + mid.drain_enthalpy, abs=2e-4)


def test_real_property_onset_is_continuous():
    from core.heat_transfer.wet_coil_adapters import solve_production_coil

    b, t, i = production_context()

    def run(W):
        return solve_production_coil(
            bundle=b,
            thermodynamics=t,
            inside=i,
            air_in=300.15,
            liquid_in=280.15,
            humidity_in=W,
            dry_mass_flow=0.5,
        )

    def criterion(W):
        try:
            return run(W).onset_margin
        except WetCoilModelError as exc:
            # Locating the dry criterion need not accept a numerically
            # unresolved wet candidate at exactly zero condensate.
            return exc.diagnostics["onset_margin"]

    onset = brentq(criterion, 0.006, 0.0075, xtol=1e-9)
    dry = run(onset - 5e-7)
    near = run(onset + 5e-7)
    farther = run(onset + 2e-6)
    assert dry.regime == "DRY"
    assert near.regime == farther.regime == "PARTIALLY_WET"
    assert 0 < near.condensate < farther.condensate
    assert near.condensate < 1e-10
    assert near.air_out == pytest.approx(dry.air_out, abs=0.002)
    assert near.heat_liquid == pytest.approx(dry.heat_liquid, abs=0.2)


def test_finned_dry_limit_uses_physical_dry_network():
    from core.heat_transfer.wet_coil_adapters import solve_production_coil

    b, t, i = production_context(finned=True)
    r = solve_production_coil(
        bundle=b,
        thermodynamics=t,
        inside=i,
        air_in=300.15,
        liquid_in=280.15,
        humidity_in=0.004,
        dry_mass_flow=0.5,
    )
    assert r.regime == "DRY"
    assert r.condensate == r.drain_enthalpy == 0.0
    assert r.onset_margin > 0
    assert r.diagnostics["wet_effective_area"] == r.diagnostics["dry_effective_area"]
    assert 0.5 * (
        t.enthalpy(300.15, 0.004) - t.enthalpy(r.air_out, 0.004)
    ) == pytest.approx(
        i.mass_flow * i.enthalpy_difference(280.15, r.liquid_out),
        abs=0.002,
    )
