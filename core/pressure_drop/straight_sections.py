# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only

# NOTE ON UNITS
# -------------
# All calculations use SI units:
# - Re [-], relative_roughness [-] (absolute roughness / hydraulic diameter)

"""
Straight circular-section pressure-drop friction factors (v0.5.6).

This module is the single canonical source of truth for the Darcy
friction factor used by the tube-side pressure-drop calculation
(``core.pressure_drop.internal_pressure_drop.calculate_tube_bundle_
hydraulics``). It covers hydraulically smooth tubes (unchanged from
v0.5.5/v0.5.6) and rough tubes via the Colebrook-White equation.

Scope
-----
This module is pressure-drop-side only. The independent smooth-tube
friction factor used by the internal-flow Gnielinski heat-transfer
correlation (``core.heat_transfer.internal_flow.friction_factor_smooth``)
is a separate function and is not modified or replaced by this module;
pressure-drop roughness and heat-transfer roughness effects remain
distinct in this commit (see that module's docstring and the v0.5.6
changelog notes).

Literature references
----------------------
- Colebrook, C. F. (1939), "Turbulent Flow in Pipes, with Particular
  Reference to the Transition Region Between the Smooth and Rough Pipe
  Laws", Journal of the ICE.
- Haaland, S. E. (1983), "Simple and Explicit Formulas for the Friction
  Factor in Turbulent Pipe Flow", ASME J. Fluids Eng. -- used only as an
  explicit initial guess for the iterative Colebrook-White solve below,
  not as a replacement for it.
- White, F. M., "Fluid Mechanics".
- Idelchik, I. E., "Handbook of Hydraulic Resistance".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.geometry.pressure_drop_stages import StraightSectionGeometry
    from core.pressure_drop.flow_path import PressureDropFlowState, PressureDropStageResult

#: Convergence tolerance on the Colebrook-White fixed-point variable
#: ``x = 1/sqrt(f)`` (dimensionless). Iteration stops once successive
#: values differ by less than this.
COLEBROOK_TOLERANCE = 1.0e-12

#: Maximum number of fixed-point iterations before raising. Well-posed
#: Colebrook-White solves with a Haaland initial guess typically converge
#: in well under 10 iterations; this is a generous safety margin.
COLEBROOK_MAX_ITERATIONS = 100


def friction_factor_smooth(Re: float) -> float:
    """
    Darcy friction factor for hydraulically smooth tubes.

    - Laminar (Re < 2300): f = 64 / Re
    - Turbulent: Petukhov explicit approximation (smooth pipe)

    Preserved exactly from the pre-roughness v0.5.6 implementation; no
    numerical result may change here.

    Ref: White (smooth pipe friction), widely used in engineering practice.
    """
    if Re <= 0.0:
        raise ValueError("Re must be positive.")
    if Re < 2300.0:
        return 64.0 / Re
    return 1.0 / (0.79 * math.log(Re) - 1.64) ** 2


def _validate_relative_roughness(relative_roughness: float | None) -> None:
    if relative_roughness is None:
        return
    if not math.isfinite(relative_roughness) or relative_roughness < 0.0:
        raise ValueError(
            "straight_sections_invalid_relative_roughness: relative_roughness "
            "must be None or a finite, non-negative value."
        )


def darcy_friction_factor_method(
    Re: float,
    relative_roughness: float | None = None,
) -> str:
    """
    The stable, deterministic friction-factor method identifier that
    ``darcy_friction_factor`` would use for the given ``Re``/
    ``relative_roughness`` -- the single source of truth for both the
    actual calculation and hydraulic-point diagnostics.

    Returns one of:
        "laminar_64_over_re"
        "petukhov_smooth"
        "colebrook_white"
    """
    if Re <= 0.0:
        raise ValueError("Re must be positive.")
    _validate_relative_roughness(relative_roughness)

    if Re < 2300.0:
        return "laminar_64_over_re"
    if relative_roughness is None or relative_roughness == 0.0:
        return "petukhov_smooth"
    return "colebrook_white"


def _haaland_initial_guess(Re: float, relative_roughness: float) -> float:
    """Haaland (1983) explicit approximation, used only to seed the
    Colebrook-White iteration below."""
    inv_sqrt_f = -1.8 * math.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / Re)
    return 1.0 / (inv_sqrt_f ** 2)


def _colebrook_white(Re: float, relative_roughness: float) -> float:
    """
    Solve the Colebrook-White equation for the Darcy friction factor:

        1/sqrt(f) = -2 * log10(relative_roughness/3.7 + 2.51/(Re*sqrt(f)))

    via fixed-point iteration on ``x = 1/sqrt(f)``:

        x_(n+1) = -2 * log10(relative_roughness/3.7 + 2.51*x_n/Re)

    seeded with the explicit Haaland approximation. This form is
    numerically well-behaved for the turbulent Re/roughness combinations
    relevant here and converges monotonically in a handful of iterations.

    Raises:
        RuntimeError: if convergence is not reached within
            ``COLEBROOK_MAX_ITERATIONS`` iterations to within
            ``COLEBROOK_TOLERANCE``.
    """
    x = 1.0 / math.sqrt(_haaland_initial_guess(Re, relative_roughness))
    for _ in range(COLEBROOK_MAX_ITERATIONS):
        x_new = -2.0 * math.log10(relative_roughness / 3.7 + 2.51 * x / Re)
        if abs(x_new - x) < COLEBROOK_TOLERANCE:
            return 1.0 / (x_new ** 2)
        x = x_new
    raise RuntimeError(
        "colebrook_white_did_not_converge: fixed-point iteration failed to "
        f"converge within {COLEBROOK_MAX_ITERATIONS} iterations "
        f"(tolerance={COLEBROOK_TOLERANCE:.3g}) for "
        f"Re={Re:.6g}, relative_roughness={relative_roughness:.6g}."
    )


def darcy_friction_factor(
    Re: float,
    relative_roughness: float | None = None,
) -> float:
    """
    Darcy friction factor for a straight circular tube section.

    - ``relative_roughness is None`` or ``== 0.0``: hydraulically smooth
      (``friction_factor_smooth``); numerically identical to omitting
      roughness entirely.
    - ``relative_roughness > 0.0`` and turbulent (``Re >= 2300``):
      Colebrook-White (iterative; see ``_colebrook_white``).
    - Laminar (``Re < 2300``): ``f = 64/Re`` regardless of roughness.

    This is the single canonical pressure-drop-side friction-factor
    function; see the module docstring for the (separate, unmodified)
    heat-transfer-side smooth friction factor used by Gnielinski.
    """
    method = darcy_friction_factor_method(Re, relative_roughness)
    if method == "laminar_64_over_re":
        return 64.0 / Re
    if method == "petukhov_smooth":
        return friction_factor_smooth(Re)
    return _colebrook_white(Re, relative_roughness)


def calculate_straight_section_pressure_drop(
    *,
    geometry: "StraightSectionGeometry",
    state: "PressureDropFlowState",
    stage_id: str,
) -> "PressureDropStageResult":
    """
    Explicit local-pressure-drop straight constant-area section
    (nozzle/duct/header run, etc.), Darcy-Weisbach:

        dp_irreversible = f_D * L / D_h * rho * V**2 / 2

    Uses the canonical ``darcy_friction_factor`` above (same friction-factor
    physics as the tube-side straight-tube-bundle calculation, applied here
    to an explicitly invoked local straight section). An equal-area section
    has no dynamic-pressure change (``delta_dynamic_pressure = 0``).

    Local import of ``core.pressure_drop.flow_path`` avoids a circular
    import: ``flow_path`` imports ``internal_pressure_drop``, which imports
    this module, at module load time.
    """
    from core.pressure_drop.flow_path import (
        PressureDropStageResult,
        PressureDropStageStatus,
        evaluate_section_flow,
    )

    flow = evaluate_section_flow(state, geometry)
    relative_roughness = geometry.roughness / geometry.hydraulic_diameter
    method = darcy_friction_factor_method(flow.reynolds, relative_roughness)
    f = darcy_friction_factor(flow.reynolds, relative_roughness)

    dp_irreversible = f * geometry.length / geometry.hydraulic_diameter * flow.dynamic_pressure

    return PressureDropStageResult(
        stage_id=stage_id,
        stage_type="straight_section",
        status=PressureDropStageStatus.CALCULATED,
        dp_irreversible=dp_irreversible,
        delta_dynamic_pressure=0.0,
        method="darcy_weisbach",
        warnings=(),
        reference_area=geometry.flow_area,
        reference_velocity=flow.velocity,
        reference_dynamic_pressure=flow.dynamic_pressure,
        upstream_area=geometry.flow_area,
        downstream_area=geometry.flow_area,
        upstream_velocity=flow.velocity,
        downstream_velocity=flow.velocity,
        reynolds=flow.reynolds,
        friction_factor=f,
        friction_factor_method=method,
        relative_roughness=relative_roughness,
    )
