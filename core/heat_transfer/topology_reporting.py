"""Derived dry topology diagnostics; terminal LMTD never drives the solver."""
import math

from core.geometry._thermal_topology import CounterCrossflowTopology


def representative_temperature_pairs(topology, ti, ui, to, uo):
    """Terminal pairs for an equivalent 0D topology, with explicit provenance.

    Counter/cocurrent pairs share a physical terminal plane of the equivalent
    flow model. Crossflow and counter-crossflow do not expose a circuit/row
    map: their Cartesian rectangle is only a bulk-temperature bound, never a
    claimed set of co-located states. No longitudinal pass map is inferred.
    """
    if topology == "counterflow":
        return ((ti, uo), (ui, to)), "equivalent_counterflow_terminal_pairs"
    if topology == "cocurrentflow":
        return ((ti, to), (ui, uo)), "equivalent_cocurrent_terminal_pairs"
    if topology == "crossflow" or isinstance(topology, CounterCrossflowTopology):
        return ((ti, to), (ti, uo), (ui, to), (ui, uo)), "unresolved_crossflow_bulk_rectangle_bounds"
    raise ValueError(f"Unsupported thermal topology for temperature envelope: {topology!r}")


def dry_topology_diagnostics(topology, ti, to, ui, uo, emtd):
    """Physical inside/outside inlet/outlet temperatures; EMTD keeps its basis."""
    if isinstance(topology, CounterCrossflowTopology):
        model, n, mixing = topology.model, topology.n_sections, topology.inter_section_mixing
    else:
        model = ("crossflow_outside_mixed_inside_unmixed" if topology == "crossflow" else topology)
        n, mixing = 1, "not_applicable"
    hot_in, hot_out, cold_in, cold_out = (ti,ui,to,uo) if ti >= to else (to,uo,ti,ui)
    a, b = hot_in-cold_out, hot_out-cold_in
    lmtd = None
    if all(math.isfinite(v) for v in (a,b)) and a > 0 and b > 0:
        lo, hi = sorted((a,b))
        d = hi-lo
        if d == 0:
            lmtd = lo
        else:
            ratio = d/lo
            denom = math.log1p(ratio) if math.isfinite(ratio) else math.log(hi)-math.log(lo)
            lmtd = d/denom
    elif a == b == 0:
        lmtd = 0.
    factor = emtd/lmtd if lmtd is not None and lmtd > 0 and math.isfinite(emtd) else None
    return dict(thermal_topology_model=model, n_thermal_sections=n,
                inter_section_mixing=mixing, LMTD_countercurrent=lmtd, F_equivalent=factor)


NO_DRY_TOPOLOGY_DIAGNOSTICS = dict(thermal_topology_model=None,
    n_thermal_sections=None, inter_section_mixing=None,
    LMTD_countercurrent=None, F_equivalent=None)
