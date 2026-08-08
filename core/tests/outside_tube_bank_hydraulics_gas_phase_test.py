# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Tests for the v0.6.0 gas-phase m_dot_inlet/midpoint/outlet extension to
core.heat_transfer.outside_flow.calculate_outside_tube_bank_hydraulics
(spec section 29).

Confirms:
- the default (unset) path is bit-for-bit identical to not having the
  parameters at all (dp_drag, dp_acceleration, Reynolds, euler_basis,
  reference_velocity_definition all unchanged),
- when given, dp_acceleration reflects the remaining-gas-phase inlet/outlet
  mass flow (not the condensate), while dp_drag/Reynolds/euler basis stay on
  the existing reference-mass-flux basis (unchanged contract).

Run:
    pytest -q core/tests/outside_tube_bank_hydraulics_gas_phase_test.py
"""

from __future__ import annotations

from core.heat_transfer.outside_flow import calculate_outside_tube_bank_hydraulics
from core.properties.common import FluidTransportProperties


def _kwargs() -> dict:
    props = FluidTransportProperties(rho=0.9, mu=2.5e-5, k=0.045, cp=1100.0)
    return dict(
        face_area=1.0, tube_outer_diameter=0.025,
        tube_pitch_transverse=0.035, tube_pitch_longitudinal=0.035,
        layout="staggered", n_rows=10, n_tubes_per_row=10,
        inlet_props=props, temperature_in=400.0, temperature_out=350.0,
        pressure=101_325.0,
    )


def test_default_path_is_bit_for_bit_identical_without_gas_phase_overrides() -> None:
    base = calculate_outside_tube_bank_hydraulics(m_dot=5.0, **_kwargs())
    same = calculate_outside_tube_bank_hydraulics(
        m_dot=5.0, m_dot_inlet=5.0, m_dot_midpoint=5.0, m_dot_outlet=5.0, **_kwargs()
    )

    assert base.dp_drag == same.dp_drag
    assert base.dp_acceleration == same.dp_acceleration
    assert base.dp_total == same.dp_total
    assert base.inlet.reynolds == same.inlet.reynolds
    assert base.midpoint.reynolds == same.midpoint.reynolds
    assert base.outlet.reynolds == same.outlet.reynolds
    assert base.euler_basis == same.euler_basis
    assert base.reference_velocity_definition == same.reference_velocity_definition
    assert base.face_mass_flux == same.face_mass_flux
    assert base.reference_mass_flux == same.reference_mass_flux


def test_gas_phase_override_leaves_drag_and_reynolds_on_reference_basis() -> None:
    """dp_drag/Reynolds/euler_basis stay on the bulk reference mass flux;
    only the reported per-point face flux and dp_acceleration change."""
    base = calculate_outside_tube_bank_hydraulics(m_dot=5.0, **_kwargs())
    reduced_outlet = calculate_outside_tube_bank_hydraulics(
        m_dot=5.0, m_dot_inlet=5.0, m_dot_outlet=4.7, **_kwargs()
    )

    assert reduced_outlet.dp_drag == base.dp_drag
    assert reduced_outlet.inlet.reynolds == base.inlet.reynolds
    assert reduced_outlet.outlet.reynolds == base.outlet.reynolds
    assert reduced_outlet.euler_basis == base.euler_basis

    assert reduced_outlet.outlet.face_mass_flux < base.outlet.face_mass_flux
    assert reduced_outlet.inlet.face_mass_flux == base.inlet.face_mass_flux
    assert reduced_outlet.dp_acceleration != base.dp_acceleration


def test_dp_acceleration_generalizes_to_per_point_face_mass_flux() -> None:
    result = calculate_outside_tube_bank_hydraulics(
        m_dot=5.0, m_dot_inlet=5.2, m_dot_midpoint=5.1, m_dot_outlet=5.0, **_kwargs()
    )
    expected = (
        result.outlet.face_mass_flux**2 / result.outlet.props.rho
        - result.inlet.face_mass_flux**2 / result.inlet.props.rho
    )
    assert result.dp_acceleration == expected
