"""Audit the historical endpoint-finned regression against outlet saturation."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source",type=Path)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--inside",action="store_true")
args = parser.parse_args()
source = (args.source or Path(__file__).resolve().parents[2]).resolve()
sys.path.insert(0,str(source))
from core.models.bare_tube import BareTubeHeatExchanger
from core.tests.legacy_wet_coil_provider_test import test_endpoint_onset_uses_bounded_0d_wet_zone_when_mean_fin_is_dry
if args.inside:
    from core.tests import inside_water_condensation_integration_test as inside_tests
    scenario = getattr(inside_tests, "test_large_inside_condensing_geometry_relaxes_before_enthalpy_inversion",
                       None) or inside_tests.test_large_inside_condensing_geometry_rejects_supersaturated_fixed_point
else:
    scenario = test_endpoint_onset_uses_bounded_0d_wet_zone_when_mean_fin_is_dry
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.water_equilibrium import saturated_water_ratio

receipt = dict(source=str(source))
original = BareTubeHeatExchanger.simulate
def measured(hx,a,b,**kw):
    r = original(hx,a,b,**kw)
    p = r.inside_phase_change if args.inside else r.outside_phase_change
    gas = a if args.inside else b
    gas_out = r.T_out_inside if args.inside else r.T_out_outside
    cap = detect_phase_change_capability(gas.provider)
    limit = saturated_water_ratio(p_total=gas.p,T=gas_out,M_dry=cap.M_dry)
    receipt.update(status="RETURNED",converged=r.converged,Q_W=r.q,
        gas_out_K=gas_out,water_out_K=r.T_out_outside if args.inside else r.T_out_inside,
        W_out=p.W_out,W_sat_outlet=limit,condensate_kg_s=p.m_dot_condensate,
        saturation_excess=p.W_out-limit,wet_fraction=p.wet_surface_fraction,
        residuals=p.residuals)
    return r
BareTubeHeatExchanger.simulate = measured
try:
    scenario()
except Exception as exc:
    receipt.update(error=type(exc).__name__,message=str(exc),
        failure_classification=getattr(exc,"reason",None),
        diagnostics=getattr(exc,"diagnostics",{}))
args.output.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
print(json.dumps(receipt),flush=True)
