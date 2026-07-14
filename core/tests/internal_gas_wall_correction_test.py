# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""
Smoke test for the turbulent-gas wall-property correction in
core.heat_transfer.internal_flow (v0.5.3: constant-exponent gas-heating
formula, n=0.45; n=0 for gas cooling).

Run:
    python -m core.tests.internal_gas_wall_correction_test
"""

from __future__ import annotations

import math

from core.heat_transfer.internal_flow import (
    FluidProps,
    gas_wall_temperature_correction,
    heat_transfer_coefficient_internal,
    nusselt_gnielinski,
    reynolds_number,
    prandtl_number,
)


# A turbulent-regime gas-like case (air-ish properties), Re > 4000.
_PROPS = FluidProps(rho=1.0, mu=2.0e-5, k=0.03, cp=1007.0)
_D = 0.02
_FLOW_AREA = math.pi * _D * _D / 4.0
_M_DOT = 0.05  # -> v ~ 3.2 m/s, Re ~ 3200... bump m_dot for solidly turbulent Re


def _turbulent_case():
    # Pick m_dot so Re is comfortably > 4000.
    m_dot = 0.15
    v = m_dot / (_PROPS.rho * _FLOW_AREA)
    Re = reynolds_number(_PROPS.rho, v, _D, _PROPS.mu)
    assert Re > 4000.0, Re
    return m_dot


def test_wall_equals_bulk_preserves_base_result() -> None:
    print("gas wall correction: T_wall == T_bulk -> factor == 1 (no change)")
    T_bulk = 400.0
    factor, warnings = gas_wall_temperature_correction(T_bulk, T_bulk)
    assert math.isclose(factor, 1.0, abs_tol=1e-12), factor
    assert warnings == []


def test_missing_wall_state_is_backward_compatible() -> None:
    print("gas wall correction: omitting T_bulk/T_wall reproduces the base Gnielinski result")
    m_dot = _turbulent_case()

    v, Re, Pr, alfa_base, warns_base = heat_transfer_coefficient_internal(
        m_dot=m_dot, tube_inner_diameter=_D, flow_area=_FLOW_AREA, props=_PROPS,
    )
    assert warns_base == []

    Nu_expected = nusselt_gnielinski(Re, Pr)
    alfa_expected = Nu_expected * _PROPS.k / _D
    assert math.isclose(alfa_base, alfa_expected, rel_tol=1e-9)


def test_heated_gas_reduces_nu() -> None:
    print("gas wall correction: heating (T_wall > T_bulk) reduces Nu relative to base")
    m_dot = _turbulent_case()
    T_bulk = 400.0
    T_wall = 500.0  # ratio 1.25 -> within applicability, heating branch

    v, Re, Pr, alfa_corrected, warns = heat_transfer_coefficient_internal(
        m_dot=m_dot, tube_inner_diameter=_D, flow_area=_FLOW_AREA, props=_PROPS,
        T_bulk=T_bulk, T_wall=T_wall,
    )
    alfa_base = nusselt_gnielinski(Re, Pr) * _PROPS.k / _D

    assert alfa_corrected < alfa_base, (alfa_corrected, alfa_base)
    assert not any(w.severity == "critical" for w in warns)


def test_cooled_gas_no_reduction() -> None:
    print("gas wall correction: cooling (T_wall < T_bulk) applies n=0 (no reduction)")
    m_dot = _turbulent_case()
    T_bulk = 400.0
    T_wall = 350.0  # ratio 0.875 -> cooling branch, n=0

    v, Re, Pr, alfa_corrected, warns = heat_transfer_coefficient_internal(
        m_dot=m_dot, tube_inner_diameter=_D, flow_area=_FLOW_AREA, props=_PROPS,
        T_bulk=T_bulk, T_wall=T_wall,
    )
    alfa_base = nusselt_gnielinski(Re, Pr) * _PROPS.k / _D

    assert math.isclose(alfa_corrected, alfa_base, rel_tol=1e-9), (alfa_corrected, alfa_base)


def test_applicability_warning_on_extreme_heating() -> None:
    print("gas wall correction: Tw/Tb far above 2.4 triggers an applicability warning")
    factor, warnings = gas_wall_temperature_correction(300.0, 900.0)  # ratio 3.0
    codes = {w.code for w in warnings}
    assert "tube_ht_gas_wall_correction_applicability_exceeded" in codes, codes
    assert 0.0 < factor < 1.0


def test_unavailable_wall_properties_warns_and_skips() -> None:
    print("gas wall correction: T_wall without T_bulk -> skipped with an info warning")
    m_dot = _turbulent_case()
    v, Re, Pr, alfa, warns = heat_transfer_coefficient_internal(
        m_dot=m_dot, tube_inner_diameter=_D, flow_area=_FLOW_AREA, props=_PROPS,
        T_wall=500.0,  # T_bulk intentionally omitted
    )
    codes = {w.code for w in warns}
    assert "tube_ht_gas_wall_correction_unavailable" in codes, codes
    alfa_base = nusselt_gnielinski(Re, Pr) * _PROPS.k / _D
    assert math.isclose(alfa, alfa_base, rel_tol=1e-9)


def test_laminar_regime_skips_correction() -> None:
    print("gas wall correction: laminar regime (Re<=4000) skips correction with an info note")
    m_dot = 0.0003  # small flow -> laminar
    v = m_dot / (_PROPS.rho * _FLOW_AREA)
    Re = reynolds_number(_PROPS.rho, v, _D, _PROPS.mu)
    assert Re <= 4000.0, Re

    v, Re, Pr, alfa, warns = heat_transfer_coefficient_internal(
        m_dot=m_dot, tube_inner_diameter=_D, flow_area=_FLOW_AREA, props=_PROPS,
        T_bulk=400.0, T_wall=500.0,
    )
    codes = {w.code for w in warns}
    assert "tube_ht_gas_wall_correction_not_applicable_regime" in codes, codes


# ---------------------------------------------------------------------------
# v0.5.3 formula audit: constant exponent (n=0.45 heating / n=0 cooling),
# not the Re/ratio-dependent exponent used in v0.5.2.
# ---------------------------------------------------------------------------
def test_heated_gas_matches_constant_exponent_0p45_formula() -> None:
    print("gas wall correction: heated dry-air fixture (T_bulk/T_wall ~ 0.66-0.71) matches (Tb/Tw)^0.45")
    # This is the diagnostic fixture from the v0.5.3 audit: heated dry air
    # inside tubes was observed with T_bulk/T_wall in [0.66, 0.71]. The
    # expected interval below (~0.83-0.86) is DERIVED from the formula here,
    # not hardcoded independently of it.
    for ratio_bulk_over_wall in (0.66, 0.685, 0.71):
        T_bulk = 400.0
        T_wall = T_bulk / ratio_bulk_over_wall
        factor, warnings = gas_wall_temperature_correction(T_bulk, T_wall)
        expected = ratio_bulk_over_wall ** 0.45
        assert math.isclose(factor, expected, rel_tol=1e-9), (ratio_bulk_over_wall, factor, expected)
        # Sanity band matching the audit's "approximately 0.83-0.86" note,
        # itself just a consequence of the exponent formula above.
        assert 0.80 < factor < 0.87, (ratio_bulk_over_wall, factor)
        # Must NOT match the old v0.5.2 variable-exponent formula (~0.90),
        # which used n = -log10(T_wall/T_bulk)/4 + 0.3.
        old_n = -math.log10(1.0 / ratio_bulk_over_wall) / 4.0 + 0.3
        old_factor = ratio_bulk_over_wall ** old_n
        assert abs(factor - old_factor) > 0.02, (factor, old_factor)


def test_exponent_is_exactly_0p45_for_heating() -> None:
    print("gas wall correction: heating exponent is exactly 0.45 (constant, not Re/ratio-dependent)")
    T_bulk = 350.0
    for T_wall in (351.0, 400.0, 600.0, 800.0):  # several heating ratios
        factor, _ = gas_wall_temperature_correction(T_bulk, T_wall)
        expected = (T_bulk / T_wall) ** 0.45
        assert math.isclose(factor, expected, rel_tol=1e-12), (T_wall, factor, expected)


def test_reversed_exchanger_direction_uses_inside_wall_vs_inside_bulk_only() -> None:
    print("gas wall correction: direction is decided from T_wall vs T_bulk only, not hot/cold stream identity")
    # "Reversed" exchanger: here the inside (tube/gas) stream is the COLD
    # stream overall (outside runs hotter), yet the tube wall can still run
    # hotter than the inside gas bulk (heating from the gas's perspective) --
    # the correction must still apply the heating branch (n=0.45) based
    # purely on T_wall_inside vs T_bulk_inside, regardless of which stream
    # is "hot_stream"/"cold_stream" for the exchanger as a whole.
    T_bulk_inside = 310.0
    T_wall_inside = 340.0  # wall hotter than inside bulk -> heating, from the gas's own perspective
    factor, warnings = gas_wall_temperature_correction(T_bulk_inside, T_wall_inside)
    expected = (T_bulk_inside / T_wall_inside) ** 0.45
    assert math.isclose(factor, expected, rel_tol=1e-12)
    assert factor < 1.0

    # And the reverse: inside gas bulk hotter than its own wall -> cooling,
    # n=0, factor=1, again independent of overall hot/cold stream identity.
    factor2, _ = gas_wall_temperature_correction(340.0, 310.0)
    assert math.isclose(factor2, 1.0, abs_tol=1e-12)


def test_temperatures_are_absolute_kelvin() -> None:
    print("gas wall correction: ratio uses absolute K, not a relative/Celsius-like offset")
    # Two cases with the SAME 50 K bulk-to-wall difference but different
    # absolute bases must give DIFFERENT correction factors, because the
    # formula depends on the *ratio* T_bulk/T_wall (absolute), not on the
    # magnitude of the temperature difference.
    factor_low_base, _ = gas_wall_temperature_correction(300.0, 350.0)
    factor_high_base, _ = gas_wall_temperature_correction(900.0, 950.0)
    assert not math.isclose(factor_low_base, factor_high_base, rel_tol=1e-6)
    assert math.isclose(factor_low_base, (300.0 / 350.0) ** 0.45, rel_tol=1e-9)
    assert math.isclose(factor_high_base, (900.0 / 950.0) ** 0.45, rel_tol=1e-9)


def test_equal_bulk_and_wall_gives_unity() -> None:
    print("gas wall correction: T_bulk == T_wall -> factor == 1 for any absolute level")
    for T in (250.0, 300.0, 800.0):
        factor, warnings = gas_wall_temperature_correction(T, T)
        assert math.isclose(factor, 1.0, abs_tol=1e-12)
        assert warnings == []


def test_invalid_wall_temperatures_are_rejected() -> None:
    print("gas wall correction: non-physical T_bulk/T_wall (<=0, nan, inf) warn and skip (factor=1)")
    for T_bulk, T_wall in (
        (0.0, 400.0),
        (400.0, 0.0),
        (-300.0, 400.0),
        (float("nan"), 400.0),
        (400.0, float("inf")),
    ):
        factor, warnings = gas_wall_temperature_correction(T_bulk, T_wall)
        assert factor == 1.0, (T_bulk, T_wall, factor)
        codes = {w.code for w in warnings}
        assert "tube_ht_gas_wall_correction_invalid_temperature" in codes, (T_bulk, T_wall, codes)


def main() -> None:
    test_wall_equals_bulk_preserves_base_result()
    test_missing_wall_state_is_backward_compatible()
    test_heated_gas_reduces_nu()
    test_cooled_gas_no_reduction()
    test_applicability_warning_on_extreme_heating()
    test_unavailable_wall_properties_warns_and_skips()
    test_laminar_regime_skips_correction()
    test_heated_gas_matches_constant_exponent_0p45_formula()
    test_exponent_is_exactly_0p45_for_heating()
    test_reversed_exchanger_direction_uses_inside_wall_vs_inside_bulk_only()
    test_temperatures_are_absolute_kelvin()
    test_equal_bulk_and_wall_gives_unity()
    test_invalid_wall_temperatures_are_rejected()

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
