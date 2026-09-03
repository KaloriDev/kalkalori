# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only
"""Deterministic regressions for core.properties.gas_mixture.

core/properties/gas_mixture.py's composition-handling helpers (alias
canonicalization, mass/mole-fraction conversion, CoolProp mixture-string
construction) previously had no .py pytest coverage and were exercised only
manually through a notebook (since removed). None of these checks require
CoolProp to be installed -- they only cover the pure-Python composition
math ahead of any backend call.
"""

from __future__ import annotations

import pytest

from core.properties.gas_mixture import (
    GasMixtureSpec,
    canonicalize_component_names,
    component_molar_mass,
    mass_fractions_to_mole_fractions,
)
from core.properties.coolprop_backend import build_coolprop_mixture_string


def test_canonicalize_component_names_maps_aliases_to_coolprop_names() -> None:
    canonical = canonicalize_component_names(
        {"N2": 0.75, "O2": 0.23, "CO2": 0.01, "H2O": 0.01}
    )

    assert "Nitrogen" in canonical
    assert "Oxygen" in canonical
    assert "CarbonDioxide" in canonical
    assert "Water" in canonical


def test_canonicalize_component_names_sums_alias_and_preferred_name() -> None:
    canonical = canonicalize_component_names({"CO2": 0.1, "CarbonDioxide": 0.2})

    assert canonical == pytest.approx({"CarbonDioxide": 0.3})


def test_component_molar_mass_is_positive_for_common_components() -> None:
    assert component_molar_mass("N2") > 0.0
    assert component_molar_mass("CO2") > 0.0
    assert component_molar_mass("H2O") > 0.0


def test_mass_fractions_to_mole_fractions_normalizes_to_one() -> None:
    mole_fractions = mass_fractions_to_mole_fractions(
        {"N2": 0.75, "O2": 0.23, "CO2": 0.02}
    )

    assert abs(sum(mole_fractions.values()) - 1.0) < 1e-12


def test_mole_and_volume_basis_are_equivalent_for_gas_mixtures() -> None:
    components = {"N2": 0.78, "O2": 0.21, "Ar": 0.01}

    mole_spec = GasMixtureSpec(components=components, basis="mole")
    volume_spec = GasMixtureSpec(components=components, basis="volume")

    assert mole_spec.to_mole_fractions() == volume_spec.to_mole_fractions()


def test_mass_basis_mole_fractions_differ_from_raw_mass_fractions() -> None:
    mass_spec = GasMixtureSpec(components={"N2": 0.75, "O2": 0.23, "CO2": 0.02}, basis="mass")

    mass_as_mole = mass_spec.to_mole_fractions()

    assert abs(sum(mass_as_mole.values()) - 1.0) < 1e-12
    assert mass_as_mole != mass_spec.normalized_components()


def test_build_coolprop_mixture_string_uses_heos_prefix_and_component_names() -> None:
    mixture_string = build_coolprop_mixture_string(
        {"Nitrogen": 0.78, "Oxygen": 0.21, "CarbonDioxide": 0.01},
        backend="HEOS",
    )

    assert mixture_string.startswith("HEOS::")
    assert "Nitrogen" in mixture_string
    assert "CarbonDioxide" in mixture_string
