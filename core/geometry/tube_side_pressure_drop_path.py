# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

"""
Tube-side specified pressure-drop flow path (v0.5.6 architecture).

A tube-side flow path is:

    inlet assembly -> straight tube bundle -> return assemblies between
    tube passes -> outlet assembly

The straight tube-bundle stage itself is not part of this geometry: it is
the existing ``core.pressure_drop.internal_pressure_drop.
TubeBundleHydraulicResult`` (source of truth: ``core.geometry.bundle.
TubeBundle``), inserted into the aggregated result as the ``tube_bundle``
core stage (see ``core.pressure_drop.flow_path``).

This module also defines ``TubeSidePressureDropDesignRequest``, the request
type for a future "suggested geometry" sizing mode. That mode is not
implemented in this commit; see ``core.pressure_drop.flow_path`` for how a
design request is rejected explicitly rather than silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.geometry.pressure_drop_stages import PressureDropAssemblyGeometry


@dataclass(frozen=True)
class SpecifiedTubeSidePressureDropPath:
    """User-specified tube-side local pressure-drop geometry.

    ``returns`` holds one assembly per return between consecutive tube
    passes, in flow order. For a bundle with ``n_tube_passes`` passes, the
    physically consistent count is ``n_tube_passes - 1``; use
    ``validate_specified_tube_side_path`` to check a path against a
    specific bundle. Return assemblies are never invented automatically --
    an empty ``returns`` tuple means no return geometry was specified.
    """

    inlet: PressureDropAssemblyGeometry
    returns: tuple[PressureDropAssemblyGeometry, ...]
    outlet: PressureDropAssemblyGeometry


def validate_specified_tube_side_path(
    path: SpecifiedTubeSidePressureDropPath,
    *,
    n_tube_passes: int,
) -> None:
    """Validate that ``path.returns`` matches ``n_tube_passes - 1``.

    Raises:
        ValueError: if ``n_tube_passes`` is not positive, or if the number
            of supplied return assemblies does not equal
            ``n_tube_passes - 1``.
    """
    if n_tube_passes <= 0:
        raise ValueError("n_tube_passes must be a positive integer.")

    expected_returns = n_tube_passes - 1
    actual_returns = len(path.returns)
    if actual_returns != expected_returns:
        raise ValueError(
            "tube_side_pressure_drop_path_return_count_mismatch: "
            f"bundle has n_tube_passes={n_tube_passes}, which requires "
            f"{expected_returns} return assembl{'y' if expected_returns == 1 else 'ies'}, "
            f"but the specified path supplies {actual_returns}."
        )


@dataclass(frozen=True)
class TubeSidePressureDropDesignRequest:
    """Request type for a future tube-side suggested-geometry sizing mode.

    Not implemented in this commit. Passing this request type into the
    current (specified-geometry) calculation path must not silently
    construct geometry or return a successful zero-loss result; see
    ``core.pressure_drop.flow_path`` for the explicit ``not_implemented``
    handling.
    """

    maximum_pressure_drop: float | None = None
    minimum_velocity: float | None = None
    maximum_velocity: float | None = None
