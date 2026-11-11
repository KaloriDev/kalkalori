"""v0.9.8 conditional Mihara/onset probes with independent energy/mass checks.

Use --source .pytest_cache/v098-base for the exact merged v0.9.7 oracle.
Conditional reconstructions and numerical fixtures are not experiments.
Results are saved after each case; existing receipts must match source hashes.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback
import hashlib

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("mihara", "onset", "synthetic"))
parser.add_argument("--source", type=Path)
parser.add_argument("--provider", choices=("both","bulk","elmahdy"),default="both")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--point", help="Optional single Mihara point for a pilot")
parser.add_argument("--refined", action="store_true",
                    help="Separate tighter diagnostic controls; production defaults are unchanged")
parser.add_argument("--water-backend", choices=("coolprop", "iapws"), default="coolprop",
                    help="Mihara only; CoolProp uses the existing liquid-only inlet API")
args = parser.parse_args()
source = (args.source or ROOT).resolve()
if not (source / "core/__init__.py").is_file():
    raise FileNotFoundError(f"Missing source snapshot: {source}")
sys.path.insert(0, str(source))

import core
if Path(core.__file__).resolve() != (source / "core/__init__.py").resolve():
    raise RuntimeError("Imported core does not match the requested snapshot")
from core.geometry.bundle import TubeBundle, RowParityMode
from core.geometry.tube import BareTube, TubeOrientation
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.phase_change.types import PhaseChangeMode
from core.properties.gas_mixture import (
    GasMixturePropertyProvider, gas_mixture_from_dry_composition_and_water_ratio,
)
from core.properties.water import IAPWS97WaterSteamProvider
from core.properties.coolprop_backend import CoolPropFluidProvider
from core.properties.fluids import ConstantPropertyProvider
from core.phase_change.capability import detect_phase_change_capability
from core.phase_change.wet_gas_composition import wet_gas_provider_at_water_ratio
from iapws import IAPWS97

bulk_class = core.BulkMeanAnalogyWetCoilProvider
providers = {"BulkMeanAnalogy": bulk_class(),
             "Elmahdy": core.ElmahdyMitalasWetCoilProvider()}
if args.provider != "both":
    providers = {k:v for k,v in providers.items() if k == ("BulkMeanAnalogy" if args.provider=="bulk" else "Elmahdy")}
water = IAPWS97WaterSteamProvider()
source_hashes = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((source/"core").rglob("*.py"))}
output = dict(source=str(source),source_hashes=source_hashes,python=sys.version,rows=[])
if args.output.exists():
    output = json.loads(args.output.read_text(encoding="utf-8"))
    if output["source_hashes"] != source_hashes:
        raise ValueError("Output belongs to a different source snapshot")
    if args.mode == "mihara" and any(
        row.get("water_backend", "iapws") != args.water_backend for row in output["rows"]
    ):
        raise ValueError("Output belongs to a different water-backend scenario")


def execute(hx, a, b, provider):
    controls = {}
    if args.refined:
        controls = dict(max_iter=150, temperature_tolerance_K=1e-7,
            relative_alfa_tolerance=1e-8, relative_duty_tolerance=1e-8,
            phase_change_temperature_tolerance_K=1e-7,
            phase_change_relative_Q_tolerance=1e-8,
            phase_change_water_ratio_tolerance=1e-10,
            phase_change_condensate_tolerance_kg_s=1e-12,
            phase_change_wall_temperature_tolerance_K=1e-7,
            phase_change_wet_fraction_tolerance=1e-7)
    r = hx.simulate(a, b, wet_coil_provider=provider,
                    phase_change_max_iterations=200,
                    wet_solver_options=core.WetCoilSolverOptions(timeout_s=180.), **controls)
    p, e = r.outside_phase_change, r.wall_temperature_envelope
    d = r.wet_coil_diagnostics or {}
    drain = d.get("H_drain", 0.) or 0.
    common_caloric = (source / "core/phase_change/bulk_mean_caloric.py").exists()
    if p.active or provider.model_id.startswith("elmahdy") or common_caloric:
        dry = wet_gas_provider_at_water_ratio(detect_phase_change_capability(b.provider), 0.)
        def hg(T, W):
            return dry.full_at(T=T, p=b.p).h + W * 1000 * IAPWS97(T=T, x=1).h
        qg = p.m_dot_dry_carrier * (hg(b.T_in, p.W_in) - hg(r.T_out_outside, p.W_out)) - drain
    else:
        qg = b.m_dot * (b.provider.full_at(T=b.T_in, p=b.p).h
                       - b.provider.full_at(T=r.T_out_outside, p=b.p).h)
    if isinstance(a.provider,CoolPropFluidProvider):
        import CoolProp.CoolProp as CP
        qw = a.m_dot*(CP.PropsSI('H','T',r.T_out_inside,'P',a.p,'Water')
                     -CP.PropsSI('H','T',a.T_in,'P',a.p,'Water'))
    elif isinstance(a.provider,ConstantPropertyProvider):
        qw = a.m_dot*a.provider.at((a.T_in+r.T_out_inside)/2,a.p).cp*(r.T_out_inside-a.T_in)
    elif isinstance(a.provider,GasMixturePropertyProvider):
        qw = a.m_dot*(a.provider.full_at(T=r.T_out_inside,p=a.p).h
                     -a.provider.full_at(T=a.T_in,p=a.p).h)
    else:
        qw = a.m_dot * 1000 * (IAPWS97(T=r.T_out_inside, P=a.p/1e6).h
                               - IAPWS97(T=a.T_in, P=a.p/1e6).h)
    from core.phase_change.water_equilibrium import saturated_water_ratio
    from core.properties.water import water_saturation_pressure
    cap = detect_phase_change_capability(b.provider)
    outlet_limit = (None if water_saturation_pressure(r.T_out_outside)>=b.p else
                    saturated_water_ratio(p_total=b.p,T=r.T_out_outside,M_dry=cap.M_dry))
    return dict(converged=r.converged and p.converged, controls=controls, Q_W=r.q,UA_W_K=r.UA,
                Q_sensible_W=p.Q_sensible, Q_latent_W=p.Q_latent,
                condensate_kg_s=p.m_dot_condensate,
                gas_out_K=r.T_out_outside, water_out_K=r.T_out_inside,
                possible=p.possible, active=p.active, near_onset=p.near_onset,
                onset_margin_K=p.onset_margin_K, wet_fraction=p.wet_surface_fraction,
                surface_min_K=p.wall_temperature_min,
                surface_mean_K=p.wall_temperature_mean,
                surface_max_K=p.wall_temperature_max,
                wet_surface_K=p.wall_temperature_wet_mean,
                W_out=p.W_out, W_sat=p.W_sat_wet_surface,
                W_saturation_outlet=outlet_limit,
                metal_inside_K=e.inside_mean,metal_outside_K=e.outside_mean,
                wet_area_m2=p.wet_area,
                drain_W=drain, Q_gas_W=qg, Q_water_W=qw,
                gas_residual_W=r.q-qg, water_residual_W=r.q-qw,
                mass_residual_kg_s=p.m_dot_dry_carrier*(p.W_in-p.W_out)-p.m_dot_condensate,
                area_m2=hx.bundle.total_outer_area,
                warnings=[str(w) for w in r.warnings])


def record(key, hx, a, b, model, **metadata):
    if any(row["key"] == key for row in output["rows"]):
        return
    row = dict(key=key, **metadata)
    start = time.monotonic()
    try:
        row.update(execute(hx, a, b, model))
        row["status"] = "PASS" if row["converged"] else "NOT_CONVERGED"
    except Exception as exc:
        row.update(status="ERROR", error=type(exc).__name__, message=str(exc),
                   failure_classification=getattr(exc,"reason",type(exc).__name__),
                   diagnostics=getattr(exc,"diagnostics",{}),
                   traceback=traceback.format_exc())
    row["seconds"] = time.monotonic() - start
    output["rows"].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in row.items() if k in
                     ("key", "status", "Q_W", "condensate_kg_s", "message", "seconds")}), flush=True)


if args.mode == "mihara":
    # The IAPWS pure-water evaporation preflight fails before wet dispatch on
    # this low-gas-capacity case on BOTH snapshots (separate pilot receipt).
    # Use the supported CoolProp liquid-water API, preserving all boundaries.
    # Compare with independent IF97 water calorimetry; do not change pressure.
    if args.water_backend == "coolprop":
        water = CoolPropFluidProvider("Water")
    data = json.loads(Path(__file__).with_name("v097_mihara_inputs.json").read_text(encoding="utf-8"))
    for point in data["points"]:
        pid = point["point_id"]
        if args.point and pid != args.point:
            continue
        for diameter in (.0081, .0085):
            # Later journal pitches + effective countercurrent stage circuit.
            # CONTINUOUS keeps the published 10/9 pattern: exactly 380 tubes.
            tube = BareTube(diameter, .0105, .2, .2, 15.,
                            tube_orientation=TubeOrientation.HORIZONTAL)
            bundle = TubeBundle(tube, 40, 9.5, .0205, .0205, "staggered", 40,
                                flow_arrangement="counterflow", n_passes_transverse=1,
                                row_parity_mode=RowParityMode.CONTINUOUS)
            hx = BareTubeHeatExchanger(bundle)
            a = HXSideInput(water, float(point["water_kg_h"])/3600,
                            float(point["water_in_C"])+273.15, 101325.,
                            phase_change_mode=PhaseChangeMode.DISABLED)
            for humidity, s in point["gas_scenarios"].items():
                dry = {"CO2":s["xco2"], "O2":s["xo2"], "N2":s["xn2"]}
                gas = GasMixturePropertyProvider(gas_mixture_from_dry_composition_and_water_ratio(
                    dry, water_ratio=s["w"], dry_basis="mole"))
                b = HXSideInput(gas, s["wet"]/3600, float(point["gas_in_C"])+273.15, 101325.)
                for name, provider in providers.items():
                    key = f"{pid}_ID{diameter}_airW{humidity}_{name}"
                    record(key, hx, a, b, provider, point_id=pid, provider=name,
                           tube_ID_m=diameter, assumed_air_humidity=float(humidity),
                           measured_Q_kW=float(point["Q_process_kW"]),
                           water_backend=args.water_backend,
                           gas_W=s["w"], gas_wet_kg_h=s["wet"])
elif args.mode == "onset":
    # Freeze these points on both snapshots. A narrow bracket around the dry
    # screen's activation threshold is included, alongside partial/full wet.
    for humidity in (.0175,.0178,.01785,.0179,.01792,.01794,.01796,.01798,
                     .0180,.0181,.01819,.01820,.01821,.0183,.04):
        tube = BareTube(.021, .025, 2., 2., 45., tube_orientation=TubeOrientation.HORIZONTAL)
        hx = BareTubeHeatExchanger(TubeBundle(tube, 6, 8, .06, .05, "staggered", 2,
                                            flow_arrangement="counterflow"))
        gas = GasMixturePropertyProvider(gas_mixture_from_dry_composition_and_water_ratio(
            {"N2":.77, "O2":.15, "CO2":.08}, water_ratio=humidity, dry_basis="mole"))
        a = HXSideInput(water, .4, 293.15, 600000., phase_change_mode=PhaseChangeMode.DISABLED)
        b = HXSideInput(gas, .5*(1+humidity), 333.15, 101325.)
        for name, provider in providers.items():
            record(f"W{humidity}_{name}", hx, a, b, provider,
                   humidity=humidity, provider=name, mode="AUTO")
        record(f"W{humidity}_Disabled", hx, a,
               replace(b, phase_change_mode=PhaseChangeMode.DISABLED), providers["BulkMeanAnalogy"],
               humidity=humidity, provider="Disabled", mode="DISABLED")
else:
    from core.tests.legacy_wet_coil_provider_test import historical_case
    for finned in (False,True):
        hx,a,b = historical_case(finned)
        for name,provider in providers.items():
            record(f"synthetic_{'finned' if finned else 'bare'}_{name}",hx,a,b,provider,
                   finned=finned,provider=name)
