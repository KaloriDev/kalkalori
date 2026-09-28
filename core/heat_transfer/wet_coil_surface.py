# SPDX-License-Identifier: GPL-3.0-only
"""Annular surface response for source-profile wet coils.

Only the fin is discretized radially. Existing annular conduction geometry is
reused; the selectively recovered continuous wet-front quadrature integrates
mass, latent heat and drain enthalpy on identical radial faces. This component
does not choose the whole-coil humidity closure or cap a vapor inventory.
"""
from dataclasses import dataclass
from math import fsum

from core.phase_change.wet_finned_surface import (
    _build_fin_mesh,
    _build_prescribed_base_chain,
)
from core.phase_change.wet_finned_surface import _solve_tridiagonal
from core.heat_transfer.wet_coil_radial import SurfaceProperties, integrate_fin_cells
from core.heat_transfer.wet_coil import WetCoilModelError


@dataclass(frozen=True)
class AnnularResponse:
    heat_liquid: float
    condensate: float
    drain_enthalpy: float
    base_temperature: float
    tip_temperature: float
    radial_temperatures: tuple
    wet_area: float
    energy_residual: float
    iterations: int


def annular_response(
    tube,
    *,
    base_temperature,
    gas_temperature,
    humidity,
    alpha,
    cp_dry,
    pressure,
    M_dry,
    M_water,
    radial_cells=64,
):
    if gas_temperature <= base_temperature or min(alpha, cp_dry, pressure) <= 0:
        raise ValueError("Invalid annular surface boundary")
    chain = _build_prescribed_base_chain(
        _build_fin_mesh(tube, radial_cells), base_temperature
    )
    areas = chain.surface_areas
    count = len(areas)
    props = SurfaceProperties(pressure, M_dry, M_water, humidity)
    dew = props.dew()
    lower = [0.0, *(-g for g in chain.edges)]
    upper = [*(-g for g in chain.edges), 0.0]
    diag = []
    rhs = []
    for i, (area, boundary) in enumerate(zip(areas, chain.boundary_conductances)):
        diag.append(
            alpha * area
            + boundary
            + (chain.edges[i - 1] if i else 0.0)
            + (chain.edges[i] if i < count - 1 else 0.0)
        )
        rhs.append(
            alpha * area * gas_temperature + boundary * chain.boundary_temperatures[i]
        )
    temperatures = _solve_tridiagonal(lower, diag, upper, rhs)

    def evaluate(T, order=4, derivatives=True):
        cells = integrate_fin_cells(
            chain,
            T,
            properties=props,
            dew=dew,
            alpha=alpha,
            cp=cp_dry,
            W=humidity,
            lewis=1.0,
            order=order,
            with_derivatives=derivatives,
        )
        loads = []
        jac = []
        wet = 0.0
        for i, area in enumerate(areas):
            if i in cells:
                v, d, w = cells[i]
                loads.append(v)
                jac.append(d)
                wet += w
            elif dew is not None and T[i] < dew:
                ws, hl, hfg = props.values(T[i])
                m = alpha / cp_dry * (humidity - ws) * area
                loads.append([m, m * hfg, m * hl, m * T[i], m * ws])
                wet += area
                if derivatives:
                    dw, dl, dh = props.derivatives(T[i])
                    dm = -alpha / cp_dry * dw * area
                    jac.append({i: [dm, dm * hfg + m * dh, dm * hl + m * dl]})
                else:
                    jac.append({})
            else:
                loads.append([0.0] * 5)
                jac.append({})
        residual = []
        lo = list(lower)
        dd = list(diag)
        up = list(upper)
        for i in range(count):
            residual.append(
                diag[i] * T[i]
                - rhs[i]
                - loads[i][1]
                + (lower[i] * T[i - 1] if i else 0.0)
                + (upper[i] * T[i + 1] if i < count - 1 else 0.0)
            )
            for column, values in jac[i].items():
                row = lo if column == i - 1 else dd if column == i else up
                row[i] -= values[1]
        return residual, lo, dd, up, loads, wet

    for iteration in range(60):
        residual, lo, dd, up, loads, wet = evaluate(temperatures)
        heat = fsum(
            alpha * A * (gas_temperature - T) + v[1]
            for A, T, v in zip(areas, temperatures, loads)
        )
        norm = max(abs(v) for v in residual)
        if norm < max(2e-10, abs(heat) * 2e-10):
            independent = evaluate(temperatures, order=8, derivatives=False)[4]
            for field, absolute in ((0, 2e-14), (1, 2e-8), (2, 2e-8)):
                error = fsum(
                    abs(a[field] - b[field]) for a, b in zip(loads, independent)
                )
                if error > max(
                    absolute, 2e-8 * fsum(abs(a[field]) for a in independent)
                ):
                    raise WetCoilModelError(
                        "radial source quadrature unresolved", error=error
                    )
            return AnnularResponse(
                heat,
                fsum(v[0] for v in loads),
                fsum(v[2] for v in loads),
                base_temperature,
                temperatures[-1],
                tuple(temperatures),
                wet,
                fsum(residual),
                iteration + 1,
            )
        delta = _solve_tridiagonal(lo, dd, up, [-v for v in residual])
        damping = 1.0
        for _ in range(16):
            trial = [t + damping * d for t, d in zip(temperatures, delta)]
            if (
                min(trial) >= base_temperature - 1e-8
                and max(trial) <= gas_temperature + 1e-8
            ):
                candidate = evaluate(trial, derivatives=False)
                if max(abs(v) for v in candidate[0]) < norm:
                    temperatures = trial
                    break
            damping /= 2
        else:
            raise WetCoilModelError("radial Newton step failed", residual=norm)
    raise WetCoilModelError("radial surface did not converge")
