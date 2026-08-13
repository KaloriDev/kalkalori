# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Independent numerical checks for circular-finned bank correlations."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.heat_transfer.finned_tube_outside import (
    BriggsYoung1963Provider,
    FinnedTubeHeatTransferRequest,
    calculate_finned_tube_outside_heat_transfer,
)
from core.pressure_drop.finned_tube_pressure_drop import (
    ROBINSON_BRIGGS_COEFFICIENT_BASIS,
    FinnedTubePressureDropRequest,
    RobinsonBriggs1966Provider,
    calculate_finned_tube_bank_hydraulics,
    calculate_finned_tube_pressure_drop,
)
from core.properties.common import FluidTransportProperties


INCH = 0.0254
FOOT = 0.3048
LBM = 0.45359237
BTU_IT = 1055.05585262


def _benchmark_tube(**changes) -> CircularFinnedTube:
    core = BareTube(
        D_i=0.60 * INCH,
        D_o=0.70 * INCH,
        length_total=FOOT,
        length_effective=FOOT,
        wall_k=45.0,
    )
    values = dict(
        core_tube=core,
        fin_k=200.0,
        D_fin=1.625 * INCH,
        D_root=0.750 * INCH,
        fin_thickness_root=0.019 * INCH,
        fin_thickness_tip=0.019 * INCH,
        fin_pitch=(1.0 / 9.0) * INCH,
        fin_contact_resistance=0.0,
    )
    values.update(changes)
    return CircularFinnedTube(**values)


def _benchmark_bundle(
    *,
    tube=None,
    layout: str = "staggered",
    n_rows: int = 5,
    pitch_longitudinal: float | None = None,
) -> TubeBundle:
    pitch_transverse = 1.875 * INCH
    if pitch_longitudinal is None:
        pitch_longitudinal = math.sqrt(3.0) * pitch_transverse / 2.0
    return TubeBundle(
        tube=_benchmark_tube() if tube is None else tube,
        n_rows=n_rows,
        n_tubes_per_row=1,
        pitch_transverse=pitch_transverse,
        pitch_longitudinal=pitch_longitudinal,
        layout=layout,
        n_passes_tube=1,
        flow_arrangement="crossflow",
    )


def _benchmark_state() -> tuple[TubeBundle, FluidTransportProperties, float]:
    bundle = _benchmark_bundle()
    rho = 0.0765 * LBM / FOOT**3
    mu = 0.0439 * LBM / (FOOT * 3600.0)
    k = 0.015 * BTU_IT / (3600.0 * FOOT * (5.0 / 9.0))
    cp = 0.702 * k / mu
    props = FluidTransportProperties(rho=rho, mu=mu, k=k, cp=cp)
    face_velocity = 600.0 * FOOT / 60.0
    m_dot = rho * face_velocity * bundle.frontal_flow_area
    return bundle, props, m_dot


def _heat_request(
    bundle: TubeBundle,
    props: FluidTransportProperties,
    m_dot: float,
) -> FinnedTubeHeatTransferRequest:
    tube = bundle.tube
    assert isinstance(tube, CircularFinnedTube)
    return FinnedTubeHeatTransferRequest(
        m_dot=m_dot,
        face_area=bundle.frontal_flow_area,
        minimum_free_flow_area=bundle.minimum_free_flow_area,
        rho=props.rho,
        mu=props.mu,
        k=props.k,
        cp=props.cp,
        D_root=tube.D_root,
        D_fin=tube.D_fin,
        fin_pitch=tube.fin_pitch,
        fin_thickness_root=tube.fin_thickness_root,
        fin_thickness_tip=tube.fin_thickness_tip_effective,
        pitch_transverse=bundle.pitch_transverse,
        pitch_longitudinal=bundle.pitch_longitudinal,
        layout=bundle.layout,
        n_rows=bundle.n_rows,
    )


def _pressure_request(
    bundle: TubeBundle,
    props: FluidTransportProperties,
    m_dot: float,
) -> FinnedTubePressureDropRequest:
    tube = bundle.tube
    assert isinstance(tube, CircularFinnedTube)
    return FinnedTubePressureDropRequest(
        m_dot=m_dot,
        face_area=bundle.frontal_flow_area,
        minimum_free_flow_area=bundle.minimum_free_flow_area,
        rho=props.rho,
        mu=props.mu,
        D_root=tube.D_root,
        D_fin=tube.D_fin,
        fin_pitch=tube.fin_pitch,
        fin_thickness_root=tube.fin_thickness_root,
        fin_thickness_tip=tube.fin_thickness_tip_effective,
        pitch_transverse=bundle.pitch_transverse,
        pitch_longitudinal=bundle.pitch_longitudinal,
        layout=bundle.layout,
        n_rows=bundle.n_rows,
    )


def test_independent_1960s_benchmark_uses_physical_a_min_re_and_pa() -> None:
    bundle, props, m_dot = _benchmark_state()
    htc = calculate_finned_tube_outside_heat_transfer(m_dot, bundle, props)
    pressure = calculate_finned_tube_pressure_drop(m_dot, bundle, props)

    expected_a_min = 11.7045 * INCH**2
    expected_g = m_dot / expected_a_min
    expected_v_max = expected_g / props.rho
    expected_re = expected_g * (0.750 * INCH) / props.mu

    assert bundle.minimum_free_flow_area == pytest.approx(
        expected_a_min, rel=2.0e-12
    )
    assert htc.minimum_free_flow_area == pytest.approx(expected_a_min)
    assert htc.reference_mass_flux == pytest.approx(expected_g)
    assert htc.reference_velocity == pytest.approx(expected_v_max)
    assert htc.reynolds_number == pytest.approx(7537.183438, rel=2.0e-10)
    assert htc.reynolds_number == pytest.approx(expected_re)
    assert htc.alpha == pytest.approx(62.1011, rel=8.0e-7)

    assert pressure.coefficient == pytest.approx(
        0.241008389, rel=2.0e-9
    )
    expected_dp = (
        2.0
        * bundle.n_rows
        * pressure.coefficient
        * props.rho
        * expected_v_max**2
    )
    assert pressure.dp_drag == pytest.approx(expected_dp)
    assert pressure.dp_drag == pytest.approx(101.392, rel=8.0e-7)


def test_metadata_states_diameter_velocity_area_and_row_bases() -> None:
    bundle, props, m_dot = _benchmark_state()
    htc = calculate_finned_tube_outside_heat_transfer(m_dot, bundle, props)
    pressure = calculate_finned_tube_pressure_drop(m_dot, bundle, props)

    assert htc.method == "briggs_young_1963"
    assert "minimum free-flow" in htc.velocity_basis
    assert "D_root" in htc.reynolds_basis
    assert "fin-root" in htc.reference_diameter
    assert "fin efficiency separately" in htc.area_basis
    assert "six-row" in htc.row_basis
    assert htc.applicability

    assert pressure.method == "robinson_briggs_1966"
    assert pressure.coefficient_definition == ROBINSON_BRIGGS_COEFFICIENT_BASIS
    assert "minimum free-flow" in pressure.velocity_basis
    assert "D_root" in pressure.reynolds_basis
    assert "fin-root" in pressure.reference_diameter
    assert "A_min" in pressure.area_basis
    assert "per-row" in pressure.row_basis
    assert pressure.applicability


def test_briggs_young_hard_rejects_inline_fewer_rows_and_isosceles() -> None:
    bundle, props, m_dot = _benchmark_state()
    request = _heat_request(bundle, props, m_dot)
    provider = BriggsYoung1963Provider()

    with pytest.raises(ValueError, match="staggered"):
        provider.evaluate(replace(request, layout="inline"))
    with pytest.raises(ValueError, match="at least four"):
        provider.evaluate(replace(request, n_rows=3))
    with pytest.raises(ValueError, match="equilateral"):
        provider.evaluate(
            replace(request, pitch_longitudinal=1.40 * INCH)
        )

    # Common rounded nominal inch pitches remain an equilateral engineering
    # geometry (relative diagonal mismatch about 5.6e-4).
    rounded = provider.evaluate(
        replace(request, pitch_longitudinal=1.625 * INCH)
    )
    assert rounded.equilateral_relative_deviation < 1.0e-3


def test_robinson_briggs_rejects_inline_and_fewer_rows_but_flags_isosceles() -> None:
    bundle, props, m_dot = _benchmark_state()
    request = _pressure_request(bundle, props, m_dot)
    provider = RobinsonBriggs1966Provider()

    with pytest.raises(ValueError, match="staggered"):
        provider.evaluate(replace(request, layout="inline"))
    with pytest.raises(ValueError, match="at least four"):
        provider.evaluate(replace(request, n_rows=3))

    result = provider.evaluate(
        replace(request, pitch_longitudinal=1.40 * INCH)
    )
    assert result.coefficient > 0.0
    assert any(
        warning.code == "robinson_briggs_1966_isosceles_triangular_geometry"
        for warning in result.warnings
    )


def test_high_level_adapters_reject_a_bare_tube_bundle() -> None:
    bundle, props, m_dot = _benchmark_state()
    finned = bundle.tube
    assert isinstance(finned, CircularFinnedTube)
    bare_bundle = replace(bundle, tube=finned.core_tube)

    with pytest.raises(TypeError, match="CircularFinnedTube"):
        calculate_finned_tube_outside_heat_transfer(m_dot, bare_bundle, props)
    with pytest.raises(TypeError, match="CircularFinnedTube"):
        calculate_finned_tube_pressure_drop(m_dot, bare_bundle, props)


def test_real_taper_geometry_is_kept_and_mean_is_only_correlation_mapping() -> None:
    tube = _benchmark_tube(fin_thickness_tip=0.011 * INCH)
    bundle = _benchmark_bundle(tube=tube)
    _, props, reference_m_dot = _benchmark_state()
    face_velocity = reference_m_dot / (
        props.rho * _benchmark_bundle().frontal_flow_area
    )
    m_dot = props.rho * face_velocity * bundle.frontal_flow_area

    htc = calculate_finned_tube_outside_heat_transfer(m_dot, bundle, props)
    expected_mean = 0.5 * (0.019 + 0.011) * INCH
    expected_blockage = tube.D_root + (
        (tube.D_fin - tube.D_root) * expected_mean / tube.fin_pitch
    )
    expected_a_min = (
        bundle.n_tubes_per_row
        * tube.length_effective
        * (bundle.pitch_transverse - expected_blockage)
    )

    assert htc.correlation_fin_thickness == pytest.approx(expected_mean)
    assert bundle.minimum_free_flow_area == pytest.approx(expected_a_min)
    assert any("mean_thickness_mapping" in w.code for w in htc.warnings)
    assert tube.fin_thickness_root != tube.fin_thickness_tip_effective


def test_published_range_extrapolation_is_reported_not_silently_clamped() -> None:
    bundle, props, m_dot = _benchmark_state()
    htc = calculate_finned_tube_outside_heat_transfer(
        0.05 * m_dot, bundle, props
    )
    pressure = calculate_finned_tube_pressure_drop(
        0.05 * m_dot, bundle, props
    )

    assert htc.reynolds_number < 1100.0
    assert any(w.code.endswith("reynolds_below_range") for w in htc.warnings)
    assert pressure.reynolds_number < 2000.0
    assert any(
        w.code.endswith("reynolds_below_range") for w in pressure.warnings
    )


def test_three_state_constant_properties_reproduce_single_state_drag() -> None:
    bundle, props, m_dot = _benchmark_state()
    single = calculate_finned_tube_pressure_drop(m_dot, bundle, props)
    result = calculate_finned_tube_bank_hydraulics(
        m_dot,
        bundle,
        inlet_props=props,
        temperature_in=300.0,
        temperature_out=350.0,
        pressure=101325.0,
    )

    assert [
        result.inlet.position,
        result.midpoint.position,
        result.outlet.position,
    ] == ["inlet", "midpoint", "outlet"]
    assert result.dp_drag == pytest.approx(single.dp_drag)
    assert result.dp_acceleration == pytest.approx(0.0, abs=1.0e-15)
    assert result.dp_total == pytest.approx(single.dp_drag)
    assert result.inlet.local_dp_drag == pytest.approx(single.dp_drag)
    assert result.coefficient_definition == ROBINSON_BRIGGS_COEFFICIENT_BASIS


def test_three_state_acceleration_uses_face_mass_flux_and_is_signed() -> None:
    bundle, props, m_dot = _benchmark_state()
    outlet = replace(props, rho=0.8 * props.rho)
    result = calculate_finned_tube_bank_hydraulics(
        m_dot,
        bundle,
        inlet_props=props,
        midpoint_props=props,
        outlet_props=outlet,
    )
    expected = (m_dot / bundle.frontal_flow_area) ** 2 * (
        1.0 / outlet.rho - 1.0 / props.rho
    )
    assert result.dp_acceleration == pytest.approx(expected)
    assert result.dp_acceleration > 0.0
    assert result.dp_total == pytest.approx(result.dp_drag + expected)


def test_explicit_endpoint_properties_require_an_explicit_midpoint() -> None:
    bundle, props, m_dot = _benchmark_state()
    outlet = replace(props, rho=0.6 * props.rho, mu=1.4 * props.mu)

    with pytest.raises(ValueError, match="midpoint_props must be supplied"):
        calculate_finned_tube_bank_hydraulics(
            m_dot,
            bundle,
            inlet_props=props,
            outlet_props=outlet,
            temperature_in=300.0,
            temperature_out=400.0,
            pressure=101325.0,
        )


def test_public_package_exports_are_available() -> None:
    import types

    import core.heat_transfer as heat_transfer
    import core.pressure_drop as pressure_drop
    import core.pressure_drop.finned_tube_pressure_drop as pressure_module

    assert heat_transfer.BriggsYoung1963Provider is BriggsYoung1963Provider
    assert (
        pressure_drop.RobinsonBriggs1966Provider
        is RobinsonBriggs1966Provider
    )
    assert callable(heat_transfer.calculate_finned_tube_outside_heat_transfer)
    assert callable(pressure_drop.calculate_finned_tube_bank_hydraulics)
    assert isinstance(pressure_module, types.ModuleType)
    assert pressure_module.RobinsonBriggs1966Provider is RobinsonBriggs1966Provider
