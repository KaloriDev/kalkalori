"""Admissible conditional high-NTU reconstructions, not predictive validation."""
import json
from pathlib import Path

import CoolProp.CoolProp as CP
from iapws import IAPWS97
import pytest

from core import BulkMeanAnalogyWetCoilProvider, WetCoilSolverOptions
from core.geometry.bundle import RowParityMode, TubeBundle
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.types import PhaseChangeMode
from core.phase_change.water_equilibrium import saturated_water_ratio
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from core.properties.coolprop_backend import CoolPropFluidProvider
from core.properties.gas_mixture import (
    GasMixturePropertyProvider, gas_mixture_from_dry_composition_and_water_ratio,
)


@pytest.mark.parametrize("point_id,humidity",[("M01","0.02"),("M04","0")])
def test_conditional_high_ntu_closes_independent_energy_and_outlet_phase(point_id,humidity):
    data = json.loads((Path(__file__).resolve().parents[2]
        / "docs/validation/v097_mihara_inputs.json").read_text(encoding="utf-8"))
    point = next(p for p in data["points"] if p["point_id"] == point_id)
    s = point["gas_scenarios"][humidity]
    tube = BareTube(.0081,.0105,.2,.2,15.,tube_orientation=TubeOrientation.HORIZONTAL)
    hx = BareTubeHeatExchanger(TubeBundle(tube,40,9.5,.0205,.0205,"staggered",40,
        flow_arrangement="counterflow",n_passes_transverse=1,
        row_parity_mode=RowParityMode.CONTINUOUS))
    water = HXSideInput(CoolPropFluidProvider("Water"),float(point["water_kg_h"])/3600,
        float(point["water_in_C"])+273.15,101325.,phase_change_mode=PhaseChangeMode.DISABLED)
    gas = HXSideInput(GasMixturePropertyProvider(gas_mixture_from_dry_composition_and_water_ratio(
        {"CO2":s["xco2"],"O2":s["xo2"],"N2":s["xn2"]},
        water_ratio=s["w"],dry_basis="mole")),s["wet"]/3600,
        float(point["gas_in_C"])+273.15,101325.)
    r = hx.simulate(water,gas,wet_coil_provider=BulkMeanAnalogyWetCoilProvider(),
        phase_change_max_iterations=200,wet_solver_options=WetCoilSolverOptions(timeout_s=180.))
    pc = r.outside_phase_change
    cap = detect_phase_change_capability(gas.provider)
    dry = wet_gas_provider_at_water_ratio(cap,0.)
    def h(T,W):
        return dry.full_at(T=T,p=gas.p).h+W*1000*IAPWS97(T=T,x=1).h
    qgas = pc.m_dot_dry_carrier*(h(gas.T_in,pc.W_in)-h(r.T_out_outside,pc.W_out))
    qgas -= r.wet_coil_diagnostics["H_drain"]
    qwater = water.m_dot*(CP.PropsSI("H","T",r.T_out_inside,"P",water.p,"Water")
                         -CP.PropsSI("H","T",water.T_in,"P",water.p,"Water"))
    limit = saturated_water_ratio(p_total=gas.p,T=r.T_out_outside,M_dry=cap.M_dry)
    assert r.converged and pc.converged and pc.m_dot_condensate > 0
    assert 0 <= pc.W_out <= min(pc.W_in,limit)
    assert abs(r.q-qgas) < 1.
    assert abs(r.q-qwater) < 1.
    assert abs(pc.m_dot_dry_carrier*(pc.W_in-pc.W_out)-pc.m_dot_condensate) < 1e-8
