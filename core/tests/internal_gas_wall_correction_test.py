# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""
Smoke test for the v0.5.2 turbulent-gas wall-property correction
(Petukhov 1970) in core.heat_transfer.internal_flow.

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


def main() -> None:
    test_wall_equals_bulk_preserves_base_result()
    test_missing_wall_state_is_backward_compatible()
    test_heated_gas_reduces_nu()
    test_cooled_gas_no_reduction()
    test_applicability_warning_on_extreme_heating()
    test_unavailable_wall_properties_warns_and_skips()
    test_laminar_regime_skips_correction()

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
