# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Independent and directional tests for annular-fin efficiency."""

from __future__ import annotations

import math

import pytest

from core.geometry import BareTube, CircularFinnedTube
from core.heat_transfer.fin_efficiency import (
    _annular_fin_efficiency_fvm,
    _validated_analytical_efficiency,
    annular_fin_efficiency,
    calculate_fin_efficiency,
    effective_outside_area,
    fin_efficiency,
    overall_surface_efficiency,
)


def _tube(**changes) -> CircularFinnedTube:
    values = dict(
        core_tube=BareTube(
            D_i=0.020,
            D_o=0.025,
            length_total=2.0,
            length_effective=1.8,
            wall_k=45.0,
        ),
        fin_k=200.0,
        D_fin=0.050,
        D_root=0.025,
        fin_thickness_root=0.001,
        fin_thickness_tip=0.001,
        fin_pitch=0.004,
        fin_contact_resistance=0.0,
    )
    values.update(changes)
    return CircularFinnedTube(**values)


def _constant_tube(
    D_root: float,
    D_fin: float,
    thickness: float,
    fin_k: float,
) -> CircularFinnedTube:
    """Build an independent constant-fin geometry for numerical sweeps."""
    return CircularFinnedTube(
        core_tube=BareTube(
            D_i=0.70 * D_root,
            D_o=0.80 * D_root,
            length_total=1.0,
            length_effective=1.0,
            wall_k=45.0,
        ),
        fin_k=fin_k,
        D_fin=D_fin,
        D_root=D_root,
        fin_thickness_root=thickness,
        fin_thickness_tip=thickness,
        fin_pitch=max(0.004, 2.0 * thickness),
        fin_contact_resistance=0.0,
    )


def test_constant_thickness_matches_independent_classical_bessel_reference() -> None:
    # Frozen independent reference calculated from the classical analytical
    # annular-fin solution with modified Bessel I/K functions and a convective
    # tip boundary. Inputs are those in _tube(), h=50 W/(m2 K). SciPy was used
    # once to generate this number; neither production nor this test imports it.
    expected_efficiency = 0.9613978674253748
    result = annular_fin_efficiency(_tube(), 50.0)
    assert result.fin_efficiency == pytest.approx(expected_efficiency, rel=5.0e-7)
    assert result.method == (
        "analytical_modified_bessel_constant_annular_fin_convective_tip"
    )
    assert result.radial_cells == 0
    assert result.energy_balance_relative_error == 0.0
    assert result.root_heat_rate_per_base_temperature == (
        result.convective_heat_rate_per_base_temperature
    )
    assert result.fin_heat_rate_per_base_temperature == pytest.approx(
        result.fin_efficiency * 50.0 * _tube().fin_area_per_fin,
        rel=2.0e-15,
    )


@pytest.mark.parametrize("outside_htc", [10.0, 25.0, 50.0, 100.0, 200.0])
@pytest.mark.parametrize(
    "geometry",
    [
        (0.025, 0.050, 0.0010, 200.0),
        (0.020, 0.060, 0.0003, 120.0),
        (0.030, 0.070, 0.0003, 80.0),
    ],
    ids=["typical", "thin-tall", "low-k-tall"],
)
def test_constant_analytical_path_matches_fvm_reference_across_alpha_sweep(
    geometry: tuple[float, float, float, float],
    outside_htc: float,
) -> None:
    tube = _constant_tube(*geometry)
    analytical = annular_fin_efficiency(tube, outside_htc)
    reference = _annular_fin_efficiency_fvm(
        tube,
        outside_htc,
        radial_cells=800,
    )
    assert analytical.fin_efficiency == pytest.approx(
        reference.fin_efficiency,
        rel=2.0e-5,
    )
    assert analytical.tip_temperature_ratio == pytest.approx(
        reference.tip_temperature_ratio,
        rel=2.0e-5,
    )


def test_constant_analytical_path_uses_exact_physical_convective_tip() -> None:
    # A deliberately thick, low-k fin makes the common corrected-radius,
    # adiabatic-tip approximation differ by about 3.5%.  The exact physical-tip
    # Bessel solution must instead converge to the matching FVM boundary value.
    tube = _constant_tube(0.010, 0.020, 0.004, 5.0)
    analytical = annular_fin_efficiency(tube, 1000.0)
    reference = _annular_fin_efficiency_fvm(
        tube,
        1000.0,
        radial_cells=1600,
    )
    assert analytical.fin_efficiency == pytest.approx(
        0.3444266116811683,
        rel=2.0e-6,
    )
    assert analytical.fin_efficiency == pytest.approx(
        reference.fin_efficiency,
        rel=1.0e-5,
    )
    assert analytical.tip_temperature_ratio == pytest.approx(
        reference.tip_temperature_ratio,
        rel=1.0e-5,
    )


@pytest.mark.parametrize(
    "geometry, outside_htc",
    [
        ((0.020, 0.020002, 0.000001, 100.0), 0.1),
        ((0.025, 0.0252, 0.0010, 2.0e6), 20.0),
        ((0.010, 0.1500, 0.0001, 15.0), 1000.0),
        ((0.050, 0.3000, 0.0001, 10.0), 2000.0),
        ((0.030, 0.0450, 0.0015, 400.0), 10.0),
    ],
    ids=[
        "very-short",
        "near-isothermal",
        "thin-tall",
        "large-bessel-argument",
        "thick-short",
    ],
)
def test_constant_analytical_path_is_finite_over_wide_physical_range(
    geometry: tuple[float, float, float, float],
    outside_htc: float,
) -> None:
    tube = _constant_tube(*geometry)
    result = annular_fin_efficiency(tube, outside_htc)
    assert math.isfinite(result.fin_efficiency)
    assert 0.0 < result.fin_efficiency <= 1.0
    assert math.isfinite(result.tip_temperature_ratio)
    assert 0.0 <= result.tip_temperature_ratio <= 1.0
    assert result.fin_heat_rate_per_base_temperature == pytest.approx(
        result.fin_efficiency * outside_htc * tube.fin_area_per_fin,
        rel=2.0e-15,
    )


def test_constant_dispatch_is_independent_of_reference_cell_count() -> None:
    tube = _tube(fin_thickness_tip=None)
    minimum = annular_fin_efficiency(tube, 70.0, radial_cells=4)
    refined = annular_fin_efficiency(tube, 70.0, radial_cells=1600)
    assert minimum == refined
    assert minimum.radial_cells == 0


@pytest.mark.parametrize("outside_htc", [10.0, 25.0, 50.0, 100.0, 200.0])
def test_public_dispatch_keeps_constant_and_linear_taper_paths_distinct(
    outside_htc: float,
) -> None:
    """The public API, not callers, selects the validated numerical path."""
    constant = annular_fin_efficiency(_tube(fin_thickness_tip=None), outside_htc)
    tapered = annular_fin_efficiency(_tube(fin_thickness_tip=0.00045), outside_htc)

    assert constant.method == (
        "analytical_modified_bessel_constant_annular_fin_convective_tip"
    )
    assert constant.radial_cells == 0
    assert constant.energy_balance_relative_error == 0.0
    assert tapered.method == "conservative_finite_volume_linear_taper_annular_fin"
    assert tapered.radial_cells == 400
    assert tapered.energy_balance_relative_error < 1.0e-9
    for result in (constant, tapered):
        assert 0.0 < result.fin_efficiency <= 1.0
        assert 0.0 <= result.tip_temperature_ratio <= 1.0
        assert result.root_heat_rate_per_base_temperature == pytest.approx(
            result.convective_heat_rate_per_base_temperature,
            rel=1.0e-9,
        )


def test_analytical_range_guard_only_normalizes_bessel_roundoff() -> None:
    assert _validated_analytical_efficiency(1.0 + 1.0e-8) == 1.0
    for invalid in (0.0, -1.0, math.nan, math.inf, 1.0 + 3.0e-7):
        with pytest.raises(ValueError, match="approximation tolerance"):
            _validated_analytical_efficiency(invalid)


def test_finite_volume_balance_is_conservative_and_deterministic() -> None:
    tube = _tube(fin_thickness_tip=0.00045)
    first = annular_fin_efficiency(tube, 85.0)
    second = annular_fin_efficiency(tube, 85.0)
    assert first == second
    assert first.fin_efficiency == pytest.approx(
        0.9291494496892444,
        rel=1.0e-12,
    )
    assert first.method == "conservative_finite_volume_linear_taper_annular_fin"
    assert first.radial_cells == 400
    assert first.energy_balance_relative_error < 1.0e-9
    assert first.root_heat_rate_per_base_temperature == pytest.approx(
        first.convective_heat_rate_per_base_temperature,
        rel=1.0e-9,
    )
    assert 0.0 < first.tip_temperature_ratio < 1.0


def test_refinement_is_stable_for_linearly_tapered_fin() -> None:
    tube = _tube(fin_thickness_tip=0.00035)
    coarse = annular_fin_efficiency(tube, 120.0, radial_cells=100)
    fine = annular_fin_efficiency(tube, 120.0, radial_cells=800)
    assert coarse.fin_efficiency == pytest.approx(fine.fin_efficiency, rel=5.0e-6)
    assert math.isfinite(fine.fin_efficiency)
    assert 0.0 < fine.fin_efficiency < 1.0


def test_expected_efficiency_trends() -> None:
    baseline = _tube()
    low_k = annular_fin_efficiency(_tube(fin_k=80.0), 70.0).fin_efficiency
    high_k = annular_fin_efficiency(_tube(fin_k=300.0), 70.0).fin_efficiency
    low_h = annular_fin_efficiency(baseline, 30.0).fin_efficiency
    high_h = annular_fin_efficiency(baseline, 140.0).fin_efficiency
    short = annular_fin_efficiency(_tube(D_fin=0.038), 70.0).fin_efficiency
    tall = annular_fin_efficiency(_tube(D_fin=0.060), 70.0).fin_efficiency

    assert high_k > low_k
    assert low_h > high_h
    assert short > tall


def test_small_fin_or_very_high_conductivity_approaches_unity() -> None:
    small_high_k = _tube(D_fin=0.0252, fin_k=2.0e6)
    result = annular_fin_efficiency(small_high_k, 20.0)
    assert result.fin_efficiency > 0.999999
    assert result.fin_heat_rate_per_base_temperature == (
        result.convective_heat_rate_per_base_temperature
    )


def test_gross_effective_and_overall_area_identities_with_override() -> None:
    geometric = _tube()
    tube = _tube(
        external_area_per_length=1.1 * geometric.outside_area_geometric_per_length
    )
    result = calculate_fin_efficiency(tube, 60.0)
    expected_effective = (
        tube.area_primary_outside + result.fin_efficiency * tube.area_fin
    )
    assert result.outside_area_gross == tube.area_outside_gross
    assert result.outside_area_geometric == tube.area_outside_geometric
    assert result.outside_area_effective == pytest.approx(expected_effective)
    assert result.overall_surface_efficiency == pytest.approx(
        expected_effective / tube.area_outside_gross
    )
    assert overall_surface_efficiency(tube, result.fin_efficiency) == pytest.approx(
        result.overall_surface_efficiency
    )
    assert effective_outside_area(tube, result.fin_efficiency) == pytest.approx(
        result.outside_area_effective
    )
    assert fin_efficiency(tube, 60.0) == result.fin_efficiency


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_invalid_outside_htc_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="outside_htc"):
        annular_fin_efficiency(_tube(), bad)


@pytest.mark.parametrize("bad", [True, 0, 3, 4.5])
def test_invalid_radial_cell_count_is_rejected(bad) -> None:
    with pytest.raises(ValueError, match="radial_cells"):
        annular_fin_efficiency(_tube(), 50.0, radial_cells=bad)


def test_wrong_geometry_family_is_rejected() -> None:
    with pytest.raises(TypeError, match="CircularFinnedTube"):
        annular_fin_efficiency(_tube().core_tube, 50.0)
