"""Internal canonical dry topology; public arrangement strings stay unchanged."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CounterCrossflowTopology:
    """Equal-UA crossflow cells in countercurrent series with complete mixing."""

    n_sections: int
    model: str = "counter_crossflow_equal_ua"
    inter_section_mixing: str = "complete"

    def __post_init__(self):
        if isinstance(self.n_sections, bool) or not isinstance(self.n_sections, int) or self.n_sections < 1:
            raise ValueError("n_sections must be a positive integer.")

    def lower(self):
        """Local arrangement label for legacy physical-side checks only."""
        return "crossflow"


def resolve_thermal_topology(bundle, override=None):
    """Resolve once and retain the descriptor through nested dry calculations."""
    if isinstance(override, CounterCrossflowTopology):
        return override
    arrangement = (bundle.flow_arrangement if override is None else override).lower()
    if arrangement != "auto":
        return arrangement
    if bundle.n_sections_longitudinal == 1:
        return "crossflow"
    if bundle.n_passes_transverse_resolved == 1:
        return "counterflow"
    return CounterCrossflowTopology(bundle.n_sections_longitudinal)


def reject_phase_change_topology(bundle, override=None):
    if isinstance(resolve_thermal_topology(bundle, override), CounterCrossflowTopology):
        raise ValueError(
            "AUTO N-section counter-crossflow supports dry single-phase calculations only; "
            "select an explicit legacy flow_arrangement for a phase-change model."
        )
