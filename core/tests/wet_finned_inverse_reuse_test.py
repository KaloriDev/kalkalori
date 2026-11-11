"""Strict exact-state reuse preserves native finned process equations and gates."""
import pytest

from core import WetCoilSolverOptions
from core.heat_transfer._wet_numerics import _WetNumerics
from core.heat_transfer._wet_thermodynamics import _OptimizedWetThermodynamics
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.wet_coil_integration import forward_wet_process, _forward_wet_process_native
from core.tests.wet_coil_public_test import context


@pytest.mark.parametrize("W", [.004,.007,.016])
def test_finned_fixed_state_matches_native_oracle(W):
    hx,a,b = context(finned=True,W=W)
    controls = WetCoilSolverOptions(energy_tolerance_W=2e-4,
        mass_tolerance_kg_s=2e-10,outlet_temperature_tolerance_K=2e-7,timeout_s=None)
    native = _forward_wet_process_native(hx,a,b,wet_solver_options=controls)[0]
    thermo = _OptimizedWetThermodynamics(b.p,detect_phase_change_capability(b.provider),
                                        options=_WetNumerics(cold_start=False))
    reused = forward_wet_process(hx,a,b,wet_solver_options=controls,_thermodynamics=thermo)[0]
    assert thermo.fidelity is None and not thermo.children
    assert reused.regime == native.regime
    for field, tolerance in (("heat_liquid",.03),("air_out",1e-5),
        ("liquid_out",1e-5),("humidity_out",3e-8),("wet_fraction",2e-6)):
        assert getattr(reused,field) == pytest.approx(getattr(native,field),abs=tolerance,rel=0)
    assert abs(reused.diagnostics["energy_residual"]) <= controls.energy_tolerance_W
    assert abs(reused.diagnostics["mass_residual"]) <= controls.mass_tolerance_kg_s
    assert reused.diagnostics["solver_options"] is controls
