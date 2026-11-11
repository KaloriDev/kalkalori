"""Measure the exact public finned inverse case, optionally against a source oracle.

The default uses the unchanged production 300-second controls. --oracle alone
disables the validation process deadline to obtain an independent numerical
reference; it is never counted as a passing deadline gate.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--oracle", action="store_true")
parser.add_argument("--known-flow", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
source = (args.source or root).resolve()
sys.path.insert(0, str(source))
import core
from core.tests.wet_finned_rating_test import _exchanger, _rating_sides
from core.phase_change import wet_coil_integration as integration

assert Path(core.__file__).resolve() == source / "core/__init__.py"
inside, outside = _rating_sides(wet_outside=True)
inside = replace(inside, T_out=None) if args.known_flow else replace(inside, m_dot=None)
controls = core.WetCoilSolverOptions(timeout_s=None) if args.oracle else core.WetCoilSolverOptions()
receipt = dict(source=str(source), oracle=args.oracle, python=sys.version,
    platform=platform.platform(), machine=platform.machine(),
    controls=asdict(controls), forwards=[])
original = integration.forward_wet_process
def measured(*a, **kw):
    start = time.perf_counter()
    r = original(*a, **kw)
    row = dict(area_scale=kw.get("_area_scale"), inside_mass=a[1].m_dot,
        elapsed_s=time.perf_counter()-start, Q_W=r[0].heat_liquid,
        gas_out_K=r[0].air_out, water_out_K=r[0].liquid_out)
    receipt["forwards"].append(row)
    print(json.dumps(row), flush=True)
    return r
integration.forward_wet_process = measured
start = time.perf_counter()
try:
    r = _exchanger().rate(inside, outside, include_simulation=args.known_flow,
                          wet_solver_options=controls)
    p, d = r.outside_phase_change, r.wet_coil_diagnostics
    receipt.update(status="PASS", Q_W=r.Q_required, A_required=r.A_required,
        mass_inside=r.closed_balance.inside.m_dot, gas_out_K=r.closed_balance.outside.T_out,
        water_out_K=r.closed_balance.inside.T_out, condensate_kg_s=p.m_dot_condensate,
        energy_error_W=p.energy_balance_error, mass_error_kg_s=p.mass_balance_error,
        statistics=d["solver_statistics"], numerical_path=d["numerical_path"])
except Exception as exc:
    receipt.update(status="ERROR", error=type(exc).__name__, message=str(exc),
        diagnostics=getattr(exc,"diagnostics",{}))
receipt["elapsed_s"] = time.perf_counter()-start
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print(json.dumps({k:v for k,v in receipt.items() if k not in ("forwards",)}),flush=True)
if receipt["status"] != "PASS":
    sys.exit(1)
