# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Outside-side specified pressure-drop flow path (v0.5.6 architecture).

An outside-side flow path is:

    inlet assembly -> bare tube bank -> outlet assembly

The bare tube-bank stage itself is not part of this geometry: it is the
existing ``core.heat_transfer.outside_flow.OutsideTubeBankHydraulicResult``
(source of truth for bank geometry: ``core.geometry.bundle.TubeBundle``),
inserted into the aggregated result as the ``tube_bank`` core stage (see
``core.pressure_drop.flow_path``).

This module also defines ``OutsidePressureDropDesignRequest``, the request
type for a future "suggested geometry" sizing mode. That mode is not
implemented in this commit; see ``core.pressure_drop.flow_path`` for how a
design request is rejected explicitly rather than silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.geometry.pressure_drop_stages import PressureDropAssemblyGeometry


@dataclass(frozen=True)
class SpecifiedOutsidePressureDropPath:
    """User-specified outside-side local pressure-drop geometry."""

    inlet: PressureDropAssemblyGeometry
    outlet: PressureDropAssemblyGeometry


@dataclass(frozen=True)
class OutsidePressureDropDesignRequest:
    """Request type for a future outside-side suggested-geometry sizing mode.

    Not implemented in this commit. Passing this request type into the
    current (specified-geometry) calculation path must not silently
    construct geometry or return a successful zero-loss result; see
    ``core.pressure_drop.flow_path`` for the explicit ``not_implemented``
    handling.
    """

    maximum_pressure_drop: float | None = None
    minimum_velocity: float | None = None
    maximum_velocity: float | None = None
