# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Unit tests for core.phase_change.capability (v0.6.0).

Covers H2O detection across all three composition bases (mole/volume/mass),
correct dry-composition extraction and proportion preservation, W_in, and
the "no H2O" / "not a gas mixture" negative cases.

Run:
    pytest -q core/tests/phase_change_capability_test.py
"""

from __future__ import annotations

import pytest

from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import GasMixtureSpec, GasMixturePropertyProvider
from core.phase_change.capability import detect_phase_change_capability


def _wet_mole_spec() -> GasMixtureSpec:
    return GasMixtureSpec(components={"N2": 0.75, "O2": 0.15, "CO2": 0.02, "H2O": 0.08}, basis="mole")


def test_detects_h2o_on_mole_basis() -> None:
    cap = detect_phase_change_capability(GasMixturePropertyProvider(_wet_mole_spec()))
    assert cap.capable is True
    assert cap.component == "H2O"
    assert cap.W_in is not None and cap.W_in > 0.0


def test_detects_h2o_on_volume_basis() -> None:
    spec = GasMixtureSpec(components={"N2": 0.75, "O2": 0.15, "CO2": 0.02, "H2O": 0.08}, basis="volume")
    cap = detect_phase_change_capability(GasMixturePropertyProvider(spec))
    assert cap.capable is True
    # volume basis is treated as mole basis for gas mixtures -> same W_in.
    cap_mole = detect_phase_change_capability(GasMixturePropertyProvider(_wet_mole_spec()))
    assert cap.W_in == pytest.approx(cap_mole.W_in, rel=1e-12)


def test_detects_h2o_on_mass_basis() -> None:
    from core.properties.gas_mixture import mole_fractions_to_mass_fractions

    mass_fractions = mole_fractions_to_mass_fractions(_wet_mole_spec().components)
    spec = GasMixtureSpec(components=mass_fractions, basis="mass")
    cap = detect_phase_change_capability(GasMixturePropertyProvider(spec))
    assert cap.capable is True
    cap_mole = detect_phase_change_capability(GasMixturePropertyProvider(_wet_mole_spec()))
    # Round-tripping mole -> mass -> mole should reproduce W_in closely.
    assert cap.W_in == pytest.approx(cap_mole.W_in, rel=1e-6)


def test_dry_composition_extraction_and_relative_proportions_preserved() -> None:
    cap = detect_phase_change_capability(GasMixturePropertyProvider(_wet_mole_spec()))
    assert "Water" not in cap.dry_mole_fractions
    assert sum(cap.dry_mole_fractions.values()) == pytest.approx(1.0, rel=1e-9)

    # Original wet mole fractions: N2:O2:CO2 = 0.75:0.15:0.02 (before H2O
    # normalization); the dry-only renormalization must preserve this ratio.
    assert cap.dry_mole_fractions["Nitrogen"] / cap.dry_mole_fractions["Oxygen"] == pytest.approx(
        0.75 / 0.15, rel=1e-9
    )
    assert cap.dry_mole_fractions["Oxygen"] / cap.dry_mole_fractions["CarbonDioxide"] == pytest.approx(
        0.15 / 0.02, rel=1e-9
    )


def test_W_in_matches_ideal_gas_mixing_relation() -> None:
    cap = detect_phase_change_capability(GasMixturePropertyProvider(_wet_mole_spec()))
    y_h2o = 0.08
    expected_W = (y_h2o / (1.0 - y_h2o)) * (cap.M_condensable / cap.M_dry)
    assert cap.W_in == pytest.approx(expected_W, rel=1e-9)


def test_dry_gas_average_molar_mass_reported() -> None:
    cap = detect_phase_change_capability(GasMixturePropertyProvider(_wet_mole_spec()))
    expected_M_dry = (
        0.75 * 28.0134e-3 + 0.15 * 31.9988e-3 + 0.02 * 44.0095e-3
    ) / (0.75 + 0.15 + 0.02)
    assert cap.M_dry == pytest.approx(expected_M_dry, rel=1e-9)


def test_no_h2o_component_is_not_capable() -> None:
    spec = GasMixtureSpec(components={"N2": 0.79, "O2": 0.21}, basis="mole")
    cap = detect_phase_change_capability(GasMixturePropertyProvider(spec))
    assert cap.capable is False
    assert cap.W_in is None


def test_pure_water_vapor_stream_is_not_capable() -> None:
    """No non-condensable carrier gas -> W is undefined; out of scope."""
    spec = GasMixtureSpec(components={"H2O": 1.0}, basis="mole")
    cap = detect_phase_change_capability(GasMixturePropertyProvider(spec))
    assert cap.capable is False


def test_non_gas_mixture_provider_is_not_capable() -> None:
    provider = ConstantPropertyProvider(
        FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1006.0)
    )
    cap = detect_phase_change_capability(provider)
    assert cap.capable is False
    assert cap.component is None
