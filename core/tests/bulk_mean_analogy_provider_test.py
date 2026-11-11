# SPDX-License-Identifier: GPL-3.0-only
"""Intentional provider API/identity replacement, separate from wet physics."""
import pytest

import core
from core import BulkMeanAnalogyWetCoilProvider, WetCoilProviderUnsupportedError
from core.phase_change.integration import PhaseChangeSettings


def test_descriptive_identity_and_no_deprecated_class_alias():
    provider = BulkMeanAnalogyWetCoilProvider()
    assert provider.model_id == "bulk_mean_analogy"
    assert provider.model_name == "Bulk-Mean Heat/Mass Transfer Analogy"
    assert provider.supports_simulation and not provider.supports_rating
    assert not hasattr(core, "LegacyBulkMeanWetCoilProvider")
    with pytest.raises(WetCoilProviderUnsupportedError, match="rating"):
        provider.rate()


def test_rename_keeps_the_existing_implementation_and_arguments(monkeypatch):
    from core.phase_change import legacy_wet_coil_integration as implementation
    calls = []
    sentinel = object()
    def existing(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    monkeypatch.setattr(implementation, "run_legacy_simulation", existing)
    provider = BulkMeanAnalogyWetCoilProvider()
    settings = PhaseChangeSettings(lewis_number=1.23)
    # No geometry guard in this adapter is bypassed: its historical engine
    # still owns validation. There is no continuous-sheet geometry here.
    from types import SimpleNamespace
    hx = SimpleNamespace(bundle=SimpleNamespace(continuous_fin_geometry=None))
    assert provider.simulate(hx, "inside", "outside", settings=settings,
        wet_solver_options=None, context="budget", surface_margin=.1) is sentinel
    args, kwargs = calls[0]
    assert args == (hx, "inside", "outside")
    assert kwargs == dict(settings=settings, context="budget", surface_margin=.1)


def test_renamed_provider_is_selected_with_native_diagnostics():
    from core.tests.wet_coil_public_test import context
    hx, a, b = context()
    result = hx.simulate(a, b, wet_coil_provider=BulkMeanAnalogyWetCoilProvider())
    assert result.wet_coil_diagnostics["provider"]["model_id"] == "bulk_mean_analogy"
    assert result.outside_phase_change.method == "bulk_mean_analogy"
    assert result.wet_coil_diagnostics["native_method"].startswith("outside_condensation_0d_")
