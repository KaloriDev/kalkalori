# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
#
# -------------------------------------------------------------------------
# OUTSIDE PRESSURE DROP – CROSSFLOW OVER TUBE BANKS
# -------------------------------------------------------------------------
#
# This module contains the pressure-drop / Euler-number layer for outside
# crossflow over tube banks.
#
# Design goals:
#   - keep GPL core fully open,
#   - isolate pressure-drop model selection,
#   - allow future attachment of an external closed-data provider
#     without importing proprietary code into the GPL codebase.
#
# Built-in providers currently supported:
#   - "zukauskas" : open fallback placeholder with correct API shape
#   - "kern"      : open fallback placeholder with correct API shape
#   - "esdu"      : reserved for finned-tube-bank implementation
#
# External provider support:
#   - pass an object implementing:
#         evaluate(request: EulerRequest) -> EulerResult
#
# IMPORTANT:
#   The built-in "zukauskas" / "kern" implementations below are intentionally
#   conservative placeholders until full open correlations are wired in.
#   They are here to stabilize architecture, not to claim final hydraulic
#   fidelity.
#
# Open references relevant to future implementations:
#   - Zukauskas, A. (1972), Heat Transfer from Tubes in Crossflow
#   - Incropera et al., Fundamentals of Heat and Mass Transfer
#   - VDI Heat Atlas (pressure drop / Euler number usage for tube banks)
#   - Kays & London, Compact Heat Exchangers
#   - Idelchik, Handbook of Hydraulic Resistance
#
# -------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from core.common.warnings import ModelWarning, make_warning


Layout = Literal["inline", "staggered"]


# -------------------------------------------------------------------------
# Data contracts
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class EulerRequest:
    """
    Standardized input for outside pressure-drop models.

    Parameters
    ----------
    Re:
        Reynolds number based on selected reference velocity.
    ST_over_D:
        Transverse pitch ratio, S_T / D_o.
    SL_over_D:
        Longitudinal pitch ratio, S_L / D_o.
    layout:
        "inline" or "staggered".
    n_rows:
        Number of rows in flow direction.
    is_finned:
        Geometry flag reserved for models such as ESDU.
    geometry_meta:
        Optional extensible payload for future models:
        sigma, fin spacing, fin height, A_min, etc.
    """
    Re: float
    ST_over_D: float
    SL_over_D: float
    layout: Layout
    n_rows: int
    is_finned: bool = False
    geometry_meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class EulerResult:
    """
    Standardized output from pressure-drop models.
    """
    Eu: float
    source: str
    model: str
    validity_note: str | None = None


@runtime_checkable
class EulerProvider(Protocol):
    """
    Provider protocol for outside pressure-drop / Euler models.

    This protocol can be implemented by:
      - built-in open providers,
      - external CLI adapters,
      - external HTTP adapters,
      - internal project-specific wrappers.

    The GPL core only depends on this protocol, not on closed implementations.
    """
    def evaluate(self, request: EulerRequest) -> EulerResult:
        ...


# -------------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------------

def _validate_request(request: EulerRequest) -> None:
    if request.Re <= 0.0:
        raise ValueError("Re must be positive.")
    if request.ST_over_D <= 1.0:
        raise ValueError("ST_over_D must be > 1.0 for valid crossflow gap.")
    if request.SL_over_D <= 0.0:
        raise ValueError("SL_over_D must be positive.")
    if request.n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    if request.layout not in ("inline", "staggered"):
        raise ValueError("layout must be 'inline' or 'staggered'.")


# -------------------------------------------------------------------------
# Built-in open providers
# -------------------------------------------------------------------------

class ZukauskasEulerProvider:
    """
    Open fallback provider with Zukauskas-style intent.

    Current status:
      - API-stable placeholder
      - trend-aware vs Re
      - includes weak geometry sensitivity
      - NOT a final literature-grade hydraulic implementation

    Behaviour:
      Eu_total = Eu_per_row(Re, geometry, layout) * n_rows
    """

    def evaluate(self, request: EulerRequest) -> EulerResult:
        _validate_request(request)

        Re = request.Re
        a = request.ST_over_D
        b = request.SL_over_D
        n_rows = request.n_rows
        layout = request.layout

        if Re < 1.0e2:
            eu_per_row_base = 2.00
        elif Re < 1.0e3:
            eu_per_row_base = 1.40
        elif Re < 1.0e4:
            eu_per_row_base = 0.95
        elif Re < 1.0e5:
            eu_per_row_base = 0.70
        else:
            eu_per_row_base = 0.55

        geom_factor_t = max(0.70, min(1.80, 1.25 / (a - 1.0)))
        geom_factor_l = max(0.80, min(1.40, 1.15 / max(b - 1.0, 0.20)))
        layout_factor = 1.10 if layout == "staggered" else 1.00

        eu_per_row = eu_per_row_base * geom_factor_t * geom_factor_l * layout_factor
        eu_total = eu_per_row * float(n_rows)

        return EulerResult(
            Eu=eu_total,
            source="open_gpl_builtin",
            model="zukauskas",
            validity_note=(
                "Open placeholder provider with correct Eu(Re, geometry, layout, N_L) "
                "structure; replace with validated literature correlation if higher "
                "hydraulic fidelity is required."
            ),
        )


class KernEulerProvider:
    """
    Open fallback provider with Kern-style intent.

    Current status:
      - simplified engineering placeholder
      - deliberately more conservative / lower-fidelity than target bare-bank model
      - useful as alternative selectable backend and future comparison point
    """

    def evaluate(self, request: EulerRequest) -> EulerResult:
        _validate_request(request)

        Re = request.Re
        a = request.ST_over_D
        n_rows = request.n_rows

        if Re < 1.0e2:
            eu_per_row_base = 1.60
        elif Re < 1.0e3:
            eu_per_row_base = 1.15
        elif Re < 1.0e4:
            eu_per_row_base = 0.85
        elif Re < 1.0e5:
            eu_per_row_base = 0.68
        else:
            eu_per_row_base = 0.58

        geom_factor = max(0.75, min(1.60, 1.15 / (a - 1.0)))
        eu_total = eu_per_row_base * geom_factor * float(n_rows)

        return EulerResult(
            Eu=eu_total,
            source="open_gpl_builtin",
            model="kern",
            validity_note=(
                "Open placeholder provider with Kern-style role; treat as comparative "
                "engineering estimate, not as validated final bare-tube-bank standard."
            ),
        )


class EsduEulerProvider:
    """
    Reserved provider for future ESDU-based finned-tube-bank pressure drop.

    Current status:
      - architecture stub
      - explicit geometry gate
    """

    def evaluate(self, request: EulerRequest) -> EulerResult:
        _validate_request(request)

        if not request.is_finned:
            raise ValueError(
                "ESDU pressure-drop provider is reserved for finned tube banks; "
                "current request is bare-tube geometry."
            )

        raise NotImplementedError(
            "ESDU pressure-drop model is not implemented yet in GPL core. "
            "This provider name is reserved for future finned-bank support."
        )


# -------------------------------------------------------------------------
# Provider factory / dispatcher
# -------------------------------------------------------------------------

def _resolve_builtin_provider(euler_provider: str) -> EulerProvider:
    name = euler_provider.strip().lower()

    if name == "zukauskas":
        return ZukauskasEulerProvider()
    if name == "kern":
        return KernEulerProvider()
    if name == "esdu":
        return EsduEulerProvider()

    raise ValueError(
        f"Unknown euler_provider='{euler_provider}'. "
        "Supported built-in providers: 'zukauskas', 'kern', 'esdu', "
        "or pass a custom provider object implementing EulerProvider."
    )


def evaluate_euler(
    request: EulerRequest,
    *,
    euler_provider: str | EulerProvider = "zukauskas",
) -> EulerResult:
    """
    Public dispatcher for outside pressure-drop calculation.
    """
    if isinstance(euler_provider, str):
        provider = _resolve_builtin_provider(euler_provider)
    else:
        if not isinstance(euler_provider, EulerProvider):
            raise TypeError(
                "Custom euler_provider must implement EulerProvider.evaluate(request)."
            )
        provider = euler_provider

    return provider.evaluate(request)


def pressure_drop_from_euler(
    rho: float,
    v_ref: float,
    eu: float,
) -> float:
    """
    Convert Euler number to pressure drop:

        Δp = Eu * (1/2) * rho * v_ref^2
    """
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if v_ref <= 0.0:
        raise ValueError("v_ref must be positive.")
    if eu < 0.0:
        raise ValueError("eu must be non-negative.")

    return eu * (rho * v_ref * v_ref / 2.0)


# -------------------------------------------------------------------------
# Applicability / diagnostics
# -------------------------------------------------------------------------

def check_outside_dp_applicability(
    request: EulerRequest,
    *,
    euler_provider: str | EulerProvider = "zukauskas",
    use_vmax_for_dp: bool = True,
) -> list[ModelWarning]:
    """
    Applicability / diagnostic checks for outside pressure-drop model.

    This function is intentionally conservative and non-blocking.
    """
    warnings: list[ModelWarning] = []

    if request.Re <= 0.0:
        warnings.append(
            make_warning(
                code="outside_dp_re_nonpositive",
                message="outside_dp: Reynolds number must be positive.",
                source="outside_dp",
                severity="critical",
            )
        )
        return warnings

    if request.ST_over_D <= 1.0:
        warnings.append(
            make_warning(
                code="outside_dp_st_over_d_invalid",
                message="outside_dp: ST/D <= 1.0 is geometrically invalid for crossflow tube banks.",
                source="outside_dp",
                severity="critical",
            )
        )

    if request.SL_over_D <= 0.0:
        warnings.append(
            make_warning(
                code="outside_dp_sl_over_d_nonpositive",
                message="outside_dp: SL/D must be positive.",
                source="outside_dp",
                severity="critical",
            )
        )

    if request.n_rows <= 0:
        warnings.append(
            make_warning(
                code="outside_dp_n_rows_nonpositive",
                message="outside_dp: n_rows must be positive.",
                source="outside_dp",
                severity="critical",
            )
        )

    if request.layout not in ("inline", "staggered"):
        warnings.append(
            make_warning(
                code="outside_dp_layout_invalid",
                message="outside_dp: layout must be 'inline' or 'staggered'.",
                source="outside_dp",
                severity="critical",
            )
        )

    if request.Re < 50.0:
        warnings.append(
            make_warning(
                code="outside_dp_re_very_low",
                message="outside_dp: very low Re for tube-bank hydraulic prediction; pressure-drop confidence is reduced.",
                source="outside_dp",
                severity="warning",
            )
        )
    elif request.Re > 2.0e5:
        warnings.append(
            make_warning(
                code="outside_dp_re_high",
                message="outside_dp: high Re may exceed the intended range of simplified open pressure-drop models.",
                source="outside_dp",
                severity="warning",
            )
        )

    if request.ST_over_D < 1.1:
        warnings.append(
            make_warning(
                code="outside_dp_st_over_d_tight",
                message="outside_dp: ST/D is very tight; pressure-drop sensitivity is high.",
                source="outside_dp",
                severity="warning",
            )
        )
    elif request.ST_over_D > 4.0:
        warnings.append(
            make_warning(
                code="outside_dp_st_over_d_large",
                message="outside_dp: unusually large ST/D; verify applicability of selected pressure-drop model.",
                source="outside_dp",
                severity="warning",
            )
        )

    if request.SL_over_D < 1.1:
        warnings.append(
            make_warning(
                code="outside_dp_sl_over_d_tight",
                message="outside_dp: SL/D is very small; wake interference may be strong and model confidence is reduced.",
                source="outside_dp",
                severity="warning",
            )
        )
    elif request.SL_over_D > 4.0:
        warnings.append(
            make_warning(
                code="outside_dp_sl_over_d_large",
                message="outside_dp: unusually large SL/D; verify applicability of selected pressure-drop model.",
                source="outside_dp",
                severity="warning",
            )
        )

    if request.n_rows == 1:
        warnings.append(
            make_warning(
                code="outside_dp_single_row",
                message="outside_dp: single-row bank; entry/exit effects dominate and hydraulic uncertainty is elevated.",
                source="outside_dp",
                severity="warning",
            )
        )
    elif request.n_rows < 5:
        warnings.append(
            make_warning(
                code="outside_dp_few_rows",
                message="outside_dp: very small number of rows; finite-row effects may dominate pressure drop.",
                source="outside_dp",
                severity="warning",
            )
        )

    if not use_vmax_for_dp:
        warnings.append(
            make_warning(
                code="outside_dp_velocity_reference_nonstandard",
                message="outside_dp: pressure drop is not using V_max as reference velocity; many tube-bank formulations are referenced to maximum gap velocity.",
                source="outside_dp",
                severity="info",
            )
        )

    if isinstance(euler_provider, str):
        provider_name = euler_provider.strip().lower()

        if provider_name == "zukauskas":
            warnings.append(
                make_warning(
                    code="outside_dp_zukauskas_placeholder",
                    message="outside_dp: current built-in Zukauskas provider is an open placeholder with correct structure, not yet a fully validated hydraulic correlation.",
                    source="outside_dp",
                    severity="info",
                )
            )

        elif provider_name == "kern":
            warnings.append(
                make_warning(
                    code="outside_dp_kern_selected",
                    message="outside_dp: Kern provider selected; treat as alternative engineering estimate, not final high-fidelity bare-bank standard.",
                    source="outside_dp",
                    severity="info",
                )
            )

        elif provider_name == "esdu":
            if not request.is_finned:
                warnings.append(
                    make_warning(
                        code="outside_dp_esdu_bare_tube",
                        message="outside_dp: ESDU provider is intended for finned tube banks; current request is bare-tube geometry.",
                        source="outside_dp",
                        severity="critical",
                    )
                )
            else:
                warnings.append(
                    make_warning(
                        code="outside_dp_esdu_not_implemented",
                        message="outside_dp: ESDU provider is reserved for future finned-bank implementation and is not yet available in GPL core.",
                        source="outside_dp",
                        severity="info",
                    )
                )
    else:
        warnings.append(
            make_warning(
                code="outside_dp_external_provider_selected",
                message="outside_dp: external Euler provider selected; verify correlation provenance, validity range, and deployment availability.",
                source="outside_dp",
                severity="info",
            )
        )

    return warnings