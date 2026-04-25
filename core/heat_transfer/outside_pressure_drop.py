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
#   - "zukauskas" : implemented open Zhukauskas/Ulinskas-style provider
#   - "gaddis_gnielinski" : implemented open Gaddis-Gnielinski provider
#   - "esdu"      : reserved for finned-tube-bank implementation
#
# External provider support:
#   - pass an object implementing:
#         evaluate(request: EulerRequest) -> EulerResult
#
# Open references relevant to current implementation:
#   - Zhukauskas, A. (1972), "Heat Transfer from Tubes in Crossflow",
#     Advances in Heat Transfer, Vol. 8
#   - Zhukauskas, A., Ulinskas, R. (1988), Heat Transfer in Tube Banks in Crossflow
#   - Gaddis, E.S., Gnielinski, V. (1985),
#     "Pressure drop in cross flow across tube bundles",
#     Int. Chem. Eng., Vol. 25, No. 1, pp. 1-15.
#   - Open implementation reference:
#       TORCHE (FAST Research Group), dP_Zu / Zhukauskas pressure-drop model
#
# IMPORTANT:
#   - The current ZukauskasEulerProvider is implemented from openly published /
#     openly distributed material and returns Euler number.
#   - pressure_drop_from_euler() must use the SAME reference velocity that was
#     used to build Reynolds number for the provider. For the current provider,
#     this should be V_max.
#
# -------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import math
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
        For the current Zukauskas implementation this should be based on V_max.
    ST_over_D:
        Transverse pitch ratio, S_T / D_o.
    SL_over_D:
        Longitudinal pitch ratio, S_L / D_o.
    layout:
        "inline" or "staggered".
    n_rows:
        Number of tube rows in flow direction.
    is_finned:
        Geometry flag reserved for models such as ESDU.
    geometry_meta:
        Optional extensible payload for future models and advanced corrections.
        Currently recognized optional keys for Zukauskas provider:
        - "mu_wall": dynamic viscosity at wall / surface condition
        - "mu_bulk": dynamic viscosity at bulk condition
        - "alpha_deg": rotated-crossflow correction angle
        - "beta_deg": incidence-angle correction

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


def _lin_interp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _interp_over_grid(x: float, x_grid: list[float], y_grid: list[float]) -> float:
    """
    Clamped linear interpolation over a small monotonic grid.
    """
    if len(x_grid) != len(y_grid) or len(x_grid) < 2:
        raise ValueError("x_grid and y_grid must have same length >= 2.")

    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[-1]:
        return y_grid[-1]

    for i in range(len(x_grid) - 1):
        x0 = x_grid[i]
        x1 = x_grid[i + 1]
        if x0 <= x <= x1:
            return _lin_interp(x, x0, x1, y_grid[i], y_grid[i + 1])

    return y_grid[-1]


# -------------------------------------------------------------------------
# Built-in open providers
# -------------------------------------------------------------------------

class ZukauskasEulerProvider:
    """
    Open Zhukauskas/Ulinskas-style Euler-number provider.

    Implementation basis
    --------------------
    This provider follows the open TORCHE implementation of the Zhukauskas
    pressure-drop method for crossflow tube bundles, which cites:
      - Zhukauskas, A., Ulinskas, R. (1988), Heat Transfer in Tube Banks in Crossflow
      - Zhukauskas, A. (1972), Advances in Heat Transfer, Vol. 8

    Model structure
    ---------------
      Eu_total = Eu_per_row * k1 * k2 * k3 * k4 * k5 * N_L

    where:
      - Eu_per_row is evaluated from:
            c0 + c1/Re + c2/Re^2 + c3/Re^3 + c4/Re^4
      - coefficient set depends on:
            layout, Re regime, and b = SL/D
      - k1: non-rectangular inline correction (a != b)
      - k2: viscosity correction, optional
      - k3: entry / finite-row correction
      - k4: rotated-crossflow correction
      - k5: incidence-angle correction

    Important notes
    ---------------
      - Re should be based on V_max and D_o.
      - pressure_drop_from_euler() must then use the same V_max reference velocity.
      - Validity is strongest near tabulated b values:
            b in {1.25, 1.5, 2.0}
        Here we linearly interpolate across b and clamp outside the grid.
    """

    _B_GRID = [1.25, 1.5, 2.0]

    def evaluate(self, request: EulerRequest) -> EulerResult:
        _validate_request(request)

        if request.is_finned:
            raise ValueError(
                "ZukauskasEulerProvider is intended for bare tube banks, not finned tube banks."
            )

        Re = request.Re
        a = request.ST_over_D
        b = request.SL_over_D
        n_rows = request.n_rows
        layout = request.layout
        meta = request.geometry_meta or {}

        mu_wall = meta.get("mu_wall")
        mu_bulk = meta.get("mu_bulk")
        alpha_deg = float(meta.get("alpha_deg", 0.0))
        beta_deg = float(meta.get("beta_deg", 90.0))

        if layout == "inline":
            eu_per_row = self._eu_per_row_inline(Re, b)
            k1 = self._k1_inline_nonrectangular(Re, a, b)
            k3 = self._k3_inline(Re, n_rows)
        else:
            eu_per_row = self._eu_per_row_staggered(Re, b)
            k1 = 1.0
            k3 = self._k3_staggered(Re, n_rows)

        k2 = self._k2_viscosity(Re, mu_wall=mu_wall, mu_bulk=mu_bulk)
        k4 = math.cos(math.radians(alpha_deg))
        k5 = self._k5_incidence(beta_deg)

        eu_total = eu_per_row * k1 * k2 * k3 * k4 * k5 * float(n_rows)

        # numerical guardrail
        if eu_total < 0.0:
            eu_total = 0.0

        return EulerResult(
            Eu=eu_total,
            source="open_gpl_builtin",
            model="zukauskas",
            validity_note=(
                "Open Zhukauskas/Ulinskas-style pressure-drop provider. "
                "Best consistency is obtained when Re is based on V_max and "
                "pressure drop is evaluated with the same V_max reference."
            ),
        )

    def _eu_per_row_inline(self, Re: float, b: float) -> float:
        if Re > 2.0e3:
            c0 = [0.267, 0.235, 0.247]
            c1 = [0.249e4, 0.197e4, -0.595]
            c2 = [-0.927e7, -0.124e8, 0.15]
            c3 = [0.1e11, 0.312e11, -0.137]
            c4 = [0.0, -0.274e14, 0.396]
        elif Re > 800.0:
            c0 = [0.272, 0.263, 0.247]
            c1 = [0.207e3, 0.867e2, 56.6 - (4.766e-2 * Re)]
            c2 = [0.102e3, -0.202, 0.15]
            c3 = [-0.286e3, 0.0, -0.137]
            c4 = [0.0, 0.0, 0.396]
        else:
            c0 = [0.272, 0.263, 0.188]
            c1 = [0.207e3, 0.867e2, 56.6]
            c2 = [0.102e3, -0.202, -646.0]
            c3 = [-0.286e3, 0.0, 6010.0]
            c4 = [0.0, 0.0, -18300.0]

        coeffs = (
            _interp_over_grid(b, self._B_GRID, c0),
            _interp_over_grid(b, self._B_GRID, c1),
            _interp_over_grid(b, self._B_GRID, c2),
            _interp_over_grid(b, self._B_GRID, c3),
            _interp_over_grid(b, self._B_GRID, c4),
        )
        return self._eu_poly(Re, coeffs)

    def _eu_per_row_staggered(self, Re: float, b: float) -> float:
        if Re < 1.0e3:
            c0 = [0.795, 0.683, 0.343]
            c1 = [0.247e3, 0.111e3, 0.303e3]
            c2 = [0.335e3, -0.973e2, -0.717e5]
            c3 = [-0.155e4, -0.426e3, 0.88e7]
            c4 = [0.241e4, 0.574e3, -0.38e9]
        elif Re < 1.0e4:
            c0 = [0.245, 0.203, 0.343]
            c1 = [0.339e4, 0.248e4, 0.303e3]
            c2 = [-0.984e7, -0.758e7, -0.717e5]
            c3 = [0.132e11, 0.104e11, 0.88e7]
            c4 = [-0.599e13, -0.482e13, -0.38e9]
        else:
            c0 = [0.245, 0.203, 0.162]
            c1 = [0.339e4, 0.248e4, 0.181e4]
            c2 = [-0.984e7, -0.758e7, 0.792e8]
            c3 = [0.132e11, 0.104e11, -0.165e13]
            c4 = [-0.599e13, -0.482e13, 0.872e16]

        coeffs = (
            _interp_over_grid(b, self._B_GRID, c0),
            _interp_over_grid(b, self._B_GRID, c1),
            _interp_over_grid(b, self._B_GRID, c2),
            _interp_over_grid(b, self._B_GRID, c3),
            _interp_over_grid(b, self._B_GRID, c4),
        )
        return self._eu_poly(Re, coeffs)

    @staticmethod
    def _eu_poly(
        Re: float,
        coeffs: tuple[float, float, float, float, float],
    ) -> float:
        c0, c1, c2, c3, c4 = coeffs
        inv = 1.0 / Re
        inv2 = inv * inv
        inv3 = inv2 * inv
        inv4 = inv3 * inv
        return c0 + c1 * inv + c2 * inv2 + c3 * inv3 + c4 * inv4

    @staticmethod
    def _k1_inline_nonrectangular(Re: float, a: float, b: float) -> float:
        if b <= 1.0:
            return 1.0

        x = (a - 1.0) / (b - 1.0)
        if abs(x - 1.0) < 1.0e-12:
            return 1.0

        if 1.0e3 <= Re < 1.0e4:
            return 0.9849 * x ** (-0.8129)
        if 1.0e4 <= Re < 7.0e4:
            return 0.9802 * x ** (-0.7492)
        if 7.0e4 <= Re < 1.5e5:
            return 0.9880 * x ** (-0.6388)
        return 1.0

    @staticmethod
    def _k2_viscosity(
        Re: float,
        *,
        mu_wall: float | None,
        mu_bulk: float | None,
    ) -> float:
        if mu_wall is None or mu_bulk is None:
            return 1.0
        if mu_wall <= 0.0 or mu_bulk <= 0.0:
            return 1.0

        ratio = mu_wall / mu_bulk
        p = 1.0

        if ratio > 1.0:
            p = 0.776 * math.exp(-0.545 * Re ** 0.256)
        elif ratio < 1.0 and Re < 1.0e4:
            p = 0.968 * math.exp(-1.076 * Re ** 0.196)

        return ratio ** p

    @staticmethod
    def _k3_inline(Re: float, n_rows: int) -> float:
        if n_rows < 1:
            return 1.0

        el_1 = [1.9, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0]
        el_2 = [2.7, 1.8, 1.5, 1.4, 1.3, 1.2, 1.2]

        if 0 < n_rows <= 7 and 1.0e2 < Re < 1.0e6:
            return el_1[n_rows - 1]
        if 0 < n_rows <= 7 and Re >= 1.0e6:
            return el_2[n_rows - 1]
        return 1.0

    @staticmethod
    def _k3_staggered(Re: float, n_rows: int) -> float:
        if n_rows < 1:
            return 1.0

        el_1 = [1.4, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0]
        el_2 = [1.1, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0]
        el_3 = [0.25, 0.45, 0.6, 0.65, 0.7, 0.75, 0.8]

        if 0 < n_rows <= 7 and 1.0e2 < Re < 1.0e4:
            return el_1[n_rows - 1]
        if 0 < n_rows <= 7 and 1.0e4 <= Re < 1.0e6:
            return el_2[n_rows - 1]
        if 0 < n_rows <= 7 and Re >= 1.0e6:
            return el_3[n_rows - 1]
        return 1.0

    @staticmethod
    def _k5_incidence(beta_deg: float) -> float:
        if abs(beta_deg - 90.0) < 1.0e-9:
            return 1.0
        return (
            -1.0e-6 * beta_deg ** 3
            + 1.0e-4 * beta_deg ** 2
            + 0.0132 * beta_deg
            - 0.033
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


class GaddisGnielinskiEulerProvider:
    """
    Open Gaddis-Gnielinski pressure-drop provider for bare tube banks in crossflow.

    Literature basis
    ----------------
    - Gaddis, E.S., Gnielinski, V. (1985),
      "Pressure drop in cross flow across tube bundles",
      International Chemical Engineering, Vol. 25, No. 1, pp. 1-15.
    - VDI Heat Atlas, Chapter L1.
    - Open implementation reference:
      TORCHE, function dP_GG.

    Model structure
    ---------------
    The original model computes pressure drop as:

        Δp = 0.5 * D_tot * N_eff * rho * V_max^2

    Therefore, when the core pressure-drop API is Euler-based and uses
    the same V_max reference velocity:

        Eu = Δp / (0.5 * rho * V_max^2)
        Eu = D_tot * N_eff

    Important
    ---------
    - Re must be based on V_max and tube outside diameter.
    - pressure_drop_from_euler() must use the same V_max reference velocity.
    - This provider is for bare tube banks, not finned tube banks.
    """

    def evaluate(self, request: EulerRequest) -> EulerResult:
        _validate_request(request)

        if request.is_finned:
            raise ValueError(
                "GaddisGnielinskiEulerProvider is intended for bare tube banks, not finned tube banks."
            )

        Re = request.Re
        a = request.ST_over_D
        b = request.SL_over_D
        N = request.n_rows
        layout = request.layout

        if Re <= 0.0:
            raise ValueError("Re must be positive.")
        if a <= 1.0:
            raise ValueError("ST_over_D must be > 1.0.")
        if b <= 0.0:
            raise ValueError("SL_over_D must be positive.")
        if 4.0 * a * b <= math.pi:
            raise ValueError(
                "Invalid geometry for Gaddis-Gnielinski: 4*a*b must be greater than pi."
            )

        if layout == "inline":
            d_tot = self._drag_total_inline(Re, a, b, N)
            n_eff = float(N)

        elif layout == "staggered":
            # TORCHE reference implementation reduces the effective number
            # of rows by one for staggered arrangements.
            n_eff_int = N - 1
            if n_eff_int <= 0:
                raise ValueError(
                    "Gaddis-Gnielinski staggered model requires at least 2 physical rows."
                )
            d_tot = self._drag_total_staggered(Re, a, b, n_eff_int)
            n_eff = float(n_eff_int)

        else:
            raise ValueError("layout must be 'inline' or 'staggered'.")

        eu_total = d_tot * n_eff

        if eu_total < 0.0:
            eu_total = 0.0

        return EulerResult(
            Eu=eu_total,
            source="open_gpl_builtin",
            model="gaddis_gnielinski",
            validity_note=(
                "Open Gaddis-Gnielinski pressure-drop provider for bare tube banks. "
                "Use with Re and pressure-drop dynamic pressure both referenced to V_max."
            ),
        )

    @staticmethod
    def _f_nt_inline(a: float, n_rows: int) -> float:
        if 5 < n_rows <= 10:
            return (1.0 / (a * a)) * (1.0 / float(n_rows) - 0.1)
        if n_rows > 10:
            return 0.0
        raise ValueError(
            "Gaddis-Gnielinski inline model is intended for more than 5 rows."
        )

    @staticmethod
    def _f_nt_staggered(a: float, c: float, n_eff: int) -> float:
        if 5 < n_eff <= 10:
            return (2.0 * (c - 1.0) / a * (a - 1.0) ** 2) * (
                1.0 / float(n_eff) - 0.1
            )
        if n_eff > 10:
            return 0.0
        raise ValueError(
            "Gaddis-Gnielinski staggered model is intended for more than 5 effective rows."
        )

    def _drag_total_inline(self, Re: float, a: float, b: float, n_rows: int) -> float:
        d_lam = (
            280.0
            * math.pi
            * ((math.sqrt(b) - 0.6) ** 2 + 0.75)
            / ((a ** 1.6) * (4.0 * a * b - math.pi) * Re)
        )

        f_nt = self._f_nt_inline(a, n_rows)

        f_ti = (
            0.22
            + 1.2
            * (1.0 - (0.94 / b)) ** 0.6
            / ((a - 0.85) ** 1.3)
        ) * (10.0 ** (0.47 * (b / a - 1.5))) + 0.03 * (a - 1.0) * (b - 1.0)

        d_turb = f_ti / (Re ** (0.1 * b / a))

        d_tot = d_lam + (d_turb + f_nt) * (
            1.0 - math.exp(-(Re + 1000.0) / 2000.0)
        )

        return d_tot

    def _drag_total_staggered(self, Re: float, a: float, b: float, n_eff: int) -> float:
        c = math.sqrt((a * 0.5) ** 2 + b * b)

        # Same branch logic as open TORCHE dP_GG implementation:
        # if b is tight enough, the diagonal passage controls; otherwise transverse passage controls.
        if b < 0.5 * math.sqrt(2.0 * a + 1.0):
            characteristic_spacing = c

            d_lam = (
                280.0
                * math.pi
                * ((math.sqrt(b) - 0.6) ** 2 + 0.75)
                / ((characteristic_spacing ** 1.6) * (4.0 * a * b - math.pi) * Re)
            )

            f_nt = self._f_nt_staggered(a, c, n_eff)

        else:
            d_lam = (
                280.0
                * math.pi
                * ((math.sqrt(b) - 0.6) ** 2 + 0.75)
                / ((a ** 1.6) * (4.0 * a * b - math.pi) * Re)
            )

            f_nt = self._f_nt_inline(a, n_eff)

        f_ts = (
            2.5
            + 1.2 / ((a - 0.85) ** 1.08)
            + 0.4 * (b / a - 1.0) ** 3
            - 0.01 * (a / b - 1.0) ** 3
        )

        d_turb = f_ts / (Re ** 0.25)

        d_tot = d_lam + (d_turb + f_nt) * (
            1.0 - math.exp(-(Re + 200.0) / 1000.0)
        )

        return d_tot


# -------------------------------------------------------------------------
# Provider factory / dispatcher
# -------------------------------------------------------------------------

def _resolve_builtin_provider(euler_provider: str) -> EulerProvider:
    name = euler_provider.strip().lower()

    if name == "zukauskas":
        return ZukauskasEulerProvider()
    if name in ("gaddis_gnielinski", "gaddis-gnielinski", "gg"):
        return GaddisGnielinskiEulerProvider()
    if name == "esdu":
        return EsduEulerProvider()

    raise ValueError(
        f"Unknown euler_provider='{euler_provider}'. "
        "Supported built-in providers: 'zukauskas', 'gaddis_gnielinski', 'esdu', "
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

    if request.Re < 10.0:
        warnings.append(
            make_warning(
                code="outside_dp_re_below_zukauskas_lower",
                message="outside_dp: Re is below ~10, which is below the usual lower validity bound for Zhukauskas-style pressure-drop usage.",
                source="outside_dp",
                severity="critical",
            )
        )
    elif request.Re < 100.0:
        warnings.append(
            make_warning(
                code="outside_dp_re_very_low",
                message="outside_dp: very low Re for tube-bank hydraulic prediction; pressure-drop confidence is reduced.",
                source="outside_dp",
                severity="warning",
            )
        )
    elif request.Re > 1.0e6:
        warnings.append(
            make_warning(
                code="outside_dp_re_above_open_reference",
                message="outside_dp: Re exceeds ~1e6; verify applicability of selected open pressure-drop model and row-entry corrections.",
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
                message="outside_dp: pressure drop is not using V_max as reference velocity; the current Zukauskas provider expects Re referenced to V_max.",
                source="outside_dp",
                severity="critical",
            )
        )

    if isinstance(euler_provider, str):
        provider_name = euler_provider.strip().lower()

        if provider_name == "zukauskas":
            if request.layout == "inline":
                if request.ST_over_D not in (1.25, 1.5, 2.0):
                    warnings.append(
                        make_warning(
                            code="outside_dp_zukauskas_inline_st_over_d_offgrid",
                            message="outside_dp: current open Zukauskas provider uses tabulated b-grid interpolation and is most defensible near the published pitch-ratio grids; verify extrapolation for this geometry.",
                            source="outside_dp",
                            severity="info",
                        )
                    )
                if request.SL_over_D not in (1.25, 1.5, 2.0):
                    warnings.append(
                        make_warning(
                            code="outside_dp_zukauskas_inline_sl_over_d_offgrid",
                            message="outside_dp: current open Zukauskas provider is strongest near SL/D = 1.25, 1.5, 2.0 and uses interpolation/clamping outside these points.",
                            source="outside_dp",
                            severity="info",
                        )
                    )
            else:
                if request.SL_over_D not in (1.25, 1.5, 2.0):
                    warnings.append(
                        make_warning(
                            code="outside_dp_zukauskas_staggered_sl_over_d_offgrid",
                            message="outside_dp: current open Zukauskas provider is strongest near staggered SL/D = 1.25, 1.5, 2.0 and uses interpolation/clamping outside these points.",
                            source="outside_dp",
                            severity="info",
                        )
                    )
        elif provider_name in ("gaddis_gnielinski", "gaddis-gnielinski", "gg"):
            if request.is_finned:
                warnings.append(
                    make_warning(
                        code="outside_dp_gg_finned_not_supported",
                        message="outside_dp: Gaddis-Gnielinski provider is intended for bare tube banks, not finned tube banks.",
                        source="outside_dp",
                        severity="critical",
                    )
                )

            if request.Re < 1.0:
                warnings.append(
                    make_warning(
                        code="outside_dp_gg_re_below_range",
                        message="outside_dp: Gaddis-Gnielinski correlation is below its reported lower Reynolds range.",
                        source="outside_dp",
                        severity="critical",
                    )
                )
            elif request.Re > 3.5e5:
                warnings.append(
                    make_warning(
                        code="outside_dp_gg_re_above_nominal_range",
                        message="outside_dp: Gaddis-Gnielinski nominal range is commonly reported up to about Re = 3.5e5; verify extrapolation.",
                        source="outside_dp",
                        severity="warning",
                    )
                )

            if request.n_rows <= 5:
                warnings.append(
                    make_warning(
                        code="outside_dp_gg_few_rows",
                        message="outside_dp: Gaddis-Gnielinski implementation is intended for more than 5 rows; result may be invalid or provider may raise.",
                        source="outside_dp",
                        severity="critical",
                    )
                )

            if not use_vmax_for_dp:
                warnings.append(
                    make_warning(
                        code="outside_dp_gg_velocity_reference_nonstandard",
                        message="outside_dp: Gaddis-Gnielinski provider expects Re and Eu dynamic pressure referenced to V_max.",
                        source="outside_dp",
                        severity="critical",
                    )
                )

            if request.layout == "inline":
                if request.Re < 1.0e4 and (
                    request.ST_over_D not in (1.25, 1.5, 2.0)
                    or request.SL_over_D not in (1.25, 1.5, 2.0)
                ):
                    warnings.append(
                        make_warning(
                            code="outside_dp_gg_inline_low_re_pitch_grid",
                            message="outside_dp: for Re < 1e4, Gaddis-Gnielinski inline data are strongest near a*b = 1.25*1.25, 1.5*1.5, 2.0*2.0.",
                            source="outside_dp",
                            severity="info",
                        )
                    )
            else:
                c = (0.25 * request.ST_over_D * request.ST_over_D + request.SL_over_D * request.SL_over_D) ** 0.5
                if request.Re >= 1.0e4 and c < 1.25:
                    warnings.append(
                        make_warning(
                            code="outside_dp_gg_staggered_c_small",
                            message="outside_dp: Gaddis-Gnielinski staggered high-Re range expects diagonal pitch ratio c >= 1.25.",
                            source="outside_dp",
                            severity="warning",
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