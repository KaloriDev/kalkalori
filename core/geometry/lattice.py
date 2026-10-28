# SPDX-License-Identifier: GPL-3.0-only
"""Tube-center topology, independent of tube section, fins and correlations.

x follows outside flow, y is transverse, j indexes downstream rows and i sites.
Unresolved SHIFTED describes an identity, never an invented coordinate phase.
"""
from dataclasses import dataclass
from enum import Enum
import math


class LayoutClass(str, Enum):
    INLINE = 'inline'
    CONVENTIONAL_STAGGERED = 'conventional_staggered'
    SHIFTED = 'shifted'


@dataclass(frozen=True)
class TubeBankLattice:
    layout_class: LayoutClass
    pattern: str = 'UNRESOLVED'
    phase: str | None = None
    nominal_offset: float | None = None
    # SI (dx, dy), indexed [downstream row j][transverse site i].
    site_offsets: tuple[tuple[tuple[float, float], ...], ...] | None = None

    def __post_init__(self):
        kind = self.layout_class
        if kind == 'staggered':
            kind = LayoutClass.CONVENTIONAL_STAGGERED
        object.__setattr__(self, 'layout_class', LayoutClass(kind))
        if self.nominal_offset is not None and (
                isinstance(self.nominal_offset, bool)
                or not math.isfinite(self.nominal_offset) or self.nominal_offset < 0):
            raise ValueError('nominal_offset must be finite and nonnegative [m].')
        if self.layout_class != LayoutClass.SHIFTED:
            if self.site_offsets is not None or self.phase is not None or self.nominal_offset is not None:
                raise ValueError('Custom offsets/phase require SHIFTED identity.')
            if self.pattern not in ('UNRESOLVED', 'REGULAR', 'WHOLE_ROW_HALF_PITCH'):
                raise ValueError('Conventional lattices have a fixed coordinate rule.')
            object.__setattr__(self, 'pattern', 'REGULAR' if self.layout_class == LayoutClass.INLINE else 'WHOLE_ROW_HALF_PITCH')
        elif self.site_offsets is None:
            if self.pattern != 'UNRESOLVED' or self.phase is not None:
                raise ValueError('Resolved SHIFTED requires explicit per-site offsets and phase.')
        else:
            if self.pattern == 'UNRESOLVED' or not isinstance(self.pattern, str) or not self.pattern.strip() or not isinstance(self.phase, str) or not self.phase.strip():
                raise ValueError('Explicit SHIFTED offsets require a recorded pattern and phase.')
            offsets = tuple(tuple(tuple(pair) for pair in row) for row in self.site_offsets)
            if not offsets or any(not row for row in offsets) or any(
                    len(pair) != 2 or any(isinstance(v, bool) or not math.isfinite(v) for v in pair)
                    for row in offsets for pair in row):
                raise ValueError('site_offsets must contain finite (dx, dy) pairs [m].')
            object.__setattr__(self, 'site_offsets', offsets)

    @classmethod
    def from_legacy(cls, layout):
        return cls('conventional_staggered' if layout.lower() == 'staggered' else layout.lower())

    @property
    def legacy_layout(self):
        return 'staggered' if self.layout_class == LayoutClass.CONVENTIONAL_STAGGERED else self.layout_class.value

    def coordinates(self, *, pitch_transverse, pitch_longitudinal, row_counts, row_phases=None):
        """Return row-indexed (x,y) centers; no coordinates for unresolved phase.

        row_phases records physical row parity for legacy repeated modules.
        Custom SHIFTED offsets refer to global physical site indices directly.
        """
        counts = tuple(row_counts)
        if not counts or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts):
            raise ValueError('row_counts must contain positive integers.')
        if any(isinstance(p, bool) or not math.isfinite(p) or p <= 0 for p in (pitch_transverse, pitch_longitudinal)):
            raise ValueError('Lattice pitches must be positive and finite.')
        phases = tuple(range(len(counts))) if row_phases is None else tuple(row_phases)
        if len(phases) != len(counts) or any(isinstance(j, bool) or not isinstance(j, int) or j < 0 for j in phases):
            raise ValueError('row_phases must contain one nonnegative integer per row.')
        if self.layout_class == LayoutClass.SHIFTED:
            if self.site_offsets is None:
                raise ValueError('SHIFTED layout_pattern=UNRESOLVED: coordinates unavailable.')
            if tuple(map(len, self.site_offsets)) != counts:
                raise ValueError('SHIFTED offsets must match the physical row/site counts.')
        result = []
        for j, count in enumerate(counts):
            row = []
            for i in range(count):
                dx, dy = 0., 0.
                if self.layout_class == LayoutClass.CONVENTIONAL_STAGGERED:
                    dy = (phases[j] % 2)*pitch_transverse/2
                elif self.layout_class == LayoutClass.SHIFTED:
                    dx, dy = self.site_offsets[j][i]
                row.append((j*pitch_longitudinal+dx, i*pitch_transverse+dy))
            result.append(tuple(row))
        flat = tuple(p for row in result for p in row)
        if len(set(flat)) != len(flat):
            raise ValueError('Lattice contains coincident tube centers.')
        return tuple(result)
