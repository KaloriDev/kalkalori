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


def _published_range_tube(*, fin_thickness_tip: float = 0.0005) -> CircularFinnedTube:
    """Geometry deliberately inside both published correlation envelopes."""
    core = BareTube(
        D_i=0.021,
        D_o=0.025,
        length_total=2.0,
        length_effective=2.0,
        wall_k=45.0,
    )
    return CircularFinnedTube(
        core_tube=core,
        fin_k=200.0,
        D_fin=0.052,
        D_root=0.025,
        fin_thickness_root=0.0005,
        fin_thickness_tip=fin_thickness_tip,
        fin_pitch=0.003,
        fin_contact_resistance=0.0,
    )


def _published_range_bundle(
    *,
    n_rows: int = 6,
    fin_thickness_tip: float = 0.0005,
) -> TubeBundle:
    pitch_transverse = 0.060
    return TubeBundle(
        tube=_published_range_tube(fin_thickness_tip=fin_thickness_tip),
        n_rows=n_rows,
        n_tubes_per_row=8,
        pitch_transverse=pitch_transverse,
        pitch_longitudinal=math.sqrt(3.0) * pitch_transverse / 2.0,
        layout="staggered",
        n_passes_tube=2,
        flow_arrangement="crossflow",
    )


def _published_range_properties() -> FluidTransportProperties:
    return FluidTransportProperties(rho=1.10, mu=1.90e-5, k=0.028, cp=1007.0)


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


@pytest.mark.parametrize(
    "m_dot",
    [1.0, 3.0, 6.0],
    ids=["low_in_range", "medium_in_range", "high_in_range"],
)
def test_briggs_young_matches_independent_equation_across_published_re_range(
    m_dot: float,
) -> None:
    """Keep the physical HTC equation independent of production helpers."""
    bundle = _published_range_bundle()
    props = _published_range_properties()
    request = _heat_request(bundle, props, m_dot)
    result = BriggsYoung1963Provider().evaluate(request)

    t_mean = 0.5 * (request.fin_thickness_root + request.fin_thickness_tip)
    spacing = request.fin_pitch - t_mean
    height = 0.5 * (request.D_fin - request.D_root)
    reference_mass_flux = m_dot / request.minimum_free_flow_area
    reynolds = reference_mass_flux * request.D_root / request.mu
    prandtl = request.cp * request.mu / request.k
    expected_j = (
        0.134
        * reynolds ** -0.319
        * (spacing / height) ** 0.2
        * (spacing / t_mean) ** 0.1134
    )
    expected_nusselt = expected_j * reynolds * prandtl ** (1.0 / 3.0)
    expected_alpha = expected_nusselt * request.k / request.D_root

    assert 1100.0 < reynolds < 18000.0
    assert result.reference_mass_flux == pytest.approx(reference_mass_flux)
    assert result.reference_velocity == pytest.approx(reference_mass_flux / request.rho)
    assert result.reynolds_number == pytest.approx(reynolds)
    assert result.colburn_j_factor == pytest.approx(expected_j)
    assert result.nusselt_number == pytest.approx(expected_nusselt)
    assert result.alpha == pytest.approx(expected_alpha)
    assert result.warnings == ()


@pytest.mark.parametrize(
    "m_dot",
    [1.0, 3.0, 6.0],
    ids=["low_in_range", "medium_in_range", "high_in_range"],
)
def test_robinson_briggs_matches_independent_equation_across_published_re_range(
    m_dot: float,
) -> None:
    """Freeze the verified triangular-bank drag equation and its Vmax basis."""
    bundle = _published_range_bundle()
    props = _published_range_properties()
    request = _pressure_request(bundle, props, m_dot)
    result = RobinsonBriggs1966Provider().evaluate(request)

    reference_mass_flux = m_dot / request.minimum_free_flow_area
    reference_velocity = reference_mass_flux / request.rho
    reynolds = reference_mass_flux * request.D_root / request.mu
    diagonal_pitch = math.hypot(
        request.pitch_longitudinal,
        0.5 * request.pitch_transverse,
    )
    expected_coefficient = (
        9.465
        * reynolds ** -0.316
        * (request.pitch_transverse / request.D_root) ** -0.927
        * (request.pitch_transverse / diagonal_pitch) ** 0.515
    )
    expected_dp_drag = (
        2.0 * request.n_rows * expected_coefficient * request.rho * reference_velocity**2
    )

    assert 2000.0 < reynolds < 50000.0
    assert result.reference_mass_flux == pytest.approx(reference_mass_flux)
    assert result.reference_velocity == pytest.approx(reference_velocity)
    assert result.reynolds_number == pytest.approx(reynolds)
    assert result.coefficient == pytest.approx(expected_coefficient)
    assert result.dp_drag == pytest.approx(expected_dp_drag)
    assert result.coefficient_definition == ROBINSON_BRIGGS_COEFFICIENT_BASIS
    assert result.warnings == ()


@pytest.mark.parametrize(
    ("m_dot", "suffix"),
    [(0.2, "below_range"), (20.0, "above_range")],
    ids=["low", "high"],
)
def test_correlation_reynolds_range_warnings_are_reported_without_clamping(
    m_dot: float,
    suffix: str,
) -> None:
    bundle = _published_range_bundle()
    props = _published_range_properties()
    htc = BriggsYoung1963Provider().evaluate(_heat_request(bundle, props, m_dot))
    pressure = RobinsonBriggs1966Provider().evaluate(
        _pressure_request(bundle, props, m_dot)
    )

    htc_codes = {warning.code for warning in htc.warnings}
    pressure_codes = {warning.code for warning in pressure.warnings}
    assert f"briggs_young_1963_reynolds_{suffix}" in htc_codes
    assert f"robinson_briggs_1966_reynolds_{suffix}" in pressure_codes
    assert htc.alpha > 0.0
    assert pressure.coefficient > 0.0
    assert pressure.dp_drag > 0.0


def test_correlation_row_extensions_are_explicit_and_drag_scales_per_row() -> None:
    props = _published_range_properties()
    six_rows = _published_range_bundle(n_rows=6)
    four_rows = _published_range_bundle(n_rows=4)
    m_dot = 3.0
    heat_six = BriggsYoung1963Provider().evaluate(_heat_request(six_rows, props, m_dot))
    heat_four = BriggsYoung1963Provider().evaluate(_heat_request(four_rows, props, m_dot))
    pressure_six = RobinsonBriggs1966Provider().evaluate(
        _pressure_request(six_rows, props, m_dot)
    )
    pressure_four = RobinsonBriggs1966Provider().evaluate(
        _pressure_request(four_rows, props, m_dot)
    )

    assert heat_four.alpha == pytest.approx(heat_six.alpha)
    assert pressure_four.coefficient == pytest.approx(pressure_six.coefficient)
    assert pressure_four.dp_drag == pytest.approx(pressure_six.dp_drag * 4.0 / 6.0)
    assert pressure_four.n_rows_effective == 4
    assert {
        warning.code for warning in heat_four.warnings
    } >= {"briggs_young_1963_row_count_secondary_extension"}
    assert {
        warning.code for warning in pressure_four.warnings
    } >= {"robinson_briggs_1966_row_count_secondary_extension"}


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
    assert pressure.metadata.supported_layouts == (
        "staggered_equilateral_triangular",
    )
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


def test_robinson_briggs_hard_rejects_inline_fewer_rows_and_isosceles() -> None:
    bundle, props, m_dot = _benchmark_state()
    request = _pressure_request(bundle, props, m_dot)
    provider = RobinsonBriggs1966Provider()

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
    pressure = calculate_finned_tube_pressure_drop(m_dot, bundle, props)
    expected_mean = 0.5 * (0.019 + 0.011) * INCH
    expected_spacing = tube.fin_pitch - expected_mean
    expected_blockage = tube.D_root + (
        (tube.D_fin - tube.D_root) * expected_mean / tube.fin_pitch
    )
    expected_a_min = (
        bundle.n_tubes_per_row
        * tube.length_effective
        * (bundle.pitch_transverse - expected_blockage)
    )

    assert htc.correlation_fin_thickness == pytest.approx(expected_mean)
    assert htc.correlation_fin_spacing == pytest.approx(expected_spacing)
    assert pressure.correlation_fin_thickness == pytest.approx(expected_mean)
    assert pressure.correlation_fin_spacing == pytest.approx(expected_spacing)
    assert bundle.minimum_free_flow_area == pytest.approx(expected_a_min)
    assert any("mean_thickness_mapping" in w.code for w in htc.warnings)
    assert any("mean_thickness_mapping" in w.code for w in pressure.warnings)
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
