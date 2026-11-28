"""Independent synthetic hydraulic comparison; run as a module from repo root.

    python -m examples.extended_surface_hydraulics

The two profiles have equal projected mean thickness. All non-pitch inputs
stay fixed. These are calculated predictions, not experimental measurements.
"""
from dataclasses import replace
import csv
import math
import sys

from core.geometry import BareTube, CircularFinnedTube, TubeBundle
from core.pressure_drop import calculate_finned_tube_pressure_drop
from core.properties.common import FluidTransportProperties


def predictions():
    core = BareTube(D_i=.027, D_o=.031, length_total=1.8,
                    length_effective=1.76, wall_k=38.)
    tube = CircularFinnedTube(core_tube=core, D_root=.031, D_fin=.061,
        fin_pitch=.0036, fin_thickness_root=.00075,
        fin_thickness_tip=.00025, fin_k=170.)
    bank = TubeBundle(tube=tube, n_rows=8, n_tubes_per_row=9,
        pitch_transverse=.074, pitch_longitudinal=.074*math.sqrt(3)/2,
        layout="staggered", n_passes_tube=2, flow_arrangement="counterflow")
    gas = FluidTransportProperties(rho=1.05, mu=2.05e-5, k=.029, cp=1015.)
    for profile in ("constant", "taper"):
        t = tube if profile == "taper" else replace(tube,
            fin_thickness_root=.0005, fin_thickness_tip=.0005)
        for tenth_mm in range(20, 81):
            b = replace(bank, tube=replace(t, fin_pitch=tenth_mm/10000.))
            for model in ("auto", "robinson_briggs_1966", "genic2006_staggered_tapered"):
                result = calculate_finned_tube_pressure_drop(3.7, b, gas, provider=model,
                    allow_engineering_estimates=True)
                yield dict(profile=profile, model=model, pitch_mm=tenth_mm/10.,
                    resolved_model=result.method,
                    selection_reason=result.diagnostics['auto_selection_reason'],
                    engineering_estimate=result.diagnostics['engineering_estimate'],
                    bank_pressure_Pa=result.dp_drag,
                    source_domain_status=result.diagnostics.get(
                        "source_domain_status", "SEE_WARNINGS"),
                    warnings=";".join(w.code for w in result.warnings))


if __name__ == "__main__":
    rows = list(predictions())
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
