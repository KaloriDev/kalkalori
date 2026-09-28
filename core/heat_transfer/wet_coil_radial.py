# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
"""Condensation integrals on existing annular-fin control-volume faces.

Interpolate radial temperature between existing nodes. Integrate mass, latent
heat and liquid drainage on the same wet support and positive quadrature.
Conduction and sensible-source discretizations belong to the calling network.
Selectively recovered from d8b38b8. Here cp is explicitly J/(kg DRY carrier K),
so no wet-to-dry mass conversion is applied again.
"""
from __future__ import annotations

import math

from core.phase_change.mass_heat_transfer import mass_transfer_coefficient
from core.phase_change.water_equilibrium import (
    is_frost_regime,
    saturated_water_ratio,
    water_dew_point,
    water_mole_fraction_from_ratio,
)
from core.properties.water import (
    WATER_CRITICAL_TEMPERATURE_K,
    WATER_TRIPLE_POINT_TEMPERATURE_K,
    water_latent_heat_of_vaporization,
    water_saturation_liquid_enthalpy,
)

_GAUSS = {
    4: (
        (
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ),
        (
            0.34785484513745385,
            0.6521451548625461,
            0.6521451548625461,
            0.34785484513745385,
        ),
    ),
    8: (
        (
            -0.9602898564975363,
            -0.7966664774136267,
            -0.525532409916329,
            -0.1834346424956498,
            0.1834346424956498,
            0.525532409916329,
            0.7966664774136267,
            0.9602898564975363,
        ),
        (
            0.10122853629037626,
            0.22238103445337448,
            0.31370664587788727,
            0.36268378337836177,
            0.36268378337836177,
            0.31370664587788727,
            0.22238103445337448,
            0.10122853629037626,
        ),
    ),
}


class SurfaceProperties:
    """One bulk state's common equilibrium and caloric queries."""

    def __init__(self, p, md, mw, W):
        self.p, self.md, self.mw, self.W = p, md, mw, W
        self._values = {}

    def values(self, T):
        if T not in self._values:
            try:
                sigma = saturated_water_ratio(
                    p_total=self.p, T=T, M_dry=self.md, M_h2o=self.mw
                )
            except ValueError as exc:
                if "no saturated gas-phase state exists" not in str(exc):
                    raise
                sigma = None
            self._values[T] = (
                (math.inf, 0.0, 0.0)
                if sigma is None
                else (
                    sigma,
                    water_saturation_liquid_enthalpy(T=T),
                    water_latent_heat_of_vaporization(T=T),
                )
            )
        return self._values[T]

    def derivatives(self, T):
        step = 1e-4
        if T - step < WATER_TRIPLE_POINT_TEMPERATURE_K:
            a, b, c = self.values(T), self.values(T + step), self.values(T + 2 * step)
            return tuple((-3 * x + 4 * y - z) / (2 * step) for x, y, z in zip(a, b, c))
        if T + step >= WATER_CRITICAL_TEMPERATURE_K:
            a, b, c = self.values(T), self.values(T - step), self.values(T - 2 * step)
            return tuple((3 * x - 4 * y + z) / (2 * step) for x, y, z in zip(a, b, c))
        a, b = self.values(T - step), self.values(T + step)
        return tuple((y - x) / (2 * step) for x, y in zip(a, b))

    def dew(self):
        if self.W <= 0:
            return None
        partial = self.p * water_mole_fraction_from_ratio(
            self.W, M_dry=self.md, M_h2o=self.mw
        )
        if is_frost_regime(partial):
            return None
        return water_dew_point(partial, tolerance_K=1e-10)


def integrate_face_piece(
    *,
    r_left,
    r_right,
    T_left,
    T_right,
    a,
    b,
    slope,
    km,
    W,
    properties,
    dew,
    order=4,
    offset=0.0,
    fraction=1.0,
    with_derivatives=True,
):
    """Integrals and derivatives with respect to two radial endpoint nodes.

    Values are mass, latent, drain, mass*temperature and mass*saturation.
    Jacobian rows are mass, latent and drain; columns are the two temperatures.
    ``slope`` includes the actual taper and geometric fin-area replication.
    """
    values = [0.0] * 5
    jacobian = [[0.0, 0.0] for _ in range(3)]
    if dew is None or fraction == 0:
        return values, jacobian, 0.0
    dt, dr = T_right - T_left, r_right - r_left
    ta = T_left + (a - r_left) / dr * dt + offset
    tb = T_left + (b - r_left) / dr * dt + offset
    if min(ta, tb) >= dew:
        return values, jacobian, 0.0
    lo, hi = a, b
    if max(ta, tb) > dew:
        crossing = r_left + (dew - offset - T_left) * dr / dt
        if dt > 0:
            hi = crossing
        else:
            lo = crossing
    if hi <= lo:
        return values, jacobian, 0.0
    points, weights = _GAUSS[order]
    for point, weight in zip(points, weights):
        r = (hi + lo) / 2 + point * (hi - lo) / 2
        right_shape = (r - r_left) / dr
        shapes = (1 - right_shape, right_shape)
        T = T_left + right_shape * dt + offset
        sigma, hl, hfg = properties.values(T)
        driving = W - sigma
        if driving <= 0:
            continue  # Physical positive-part law at the saturation boundary.
        measure = 4 * math.pi * r * slope * weight * (hi - lo) / 2 * fraction
        mass = km * driving
        for k, value in enumerate(
            (mass, mass * hfg, mass * hl, mass * T, mass * sigma)
        ):
            values[k] += measure * value
        if with_derivatives:
            ds, dhl, dhfg = properties.derivatives(T)
            derivatives = (
                -km * ds,
                km * (-ds * hfg + driving * dhfg),
                km * (-ds * hl + driving * dhl),
            )
            for row in range(3):
                for column in range(2):
                    jacobian[row][column] += measure * derivatives[row] * shapes[column]
    return values, jacobian, 2 * math.pi * slope * (hi * hi - lo * lo) * fraction


def integrate_fin_cells(
    chain,
    temperatures,
    *,
    properties,
    dew,
    alpha,
    cp,
    W,
    lewis,
    order=4,
    fraction=1.0,
    offset=0.0,
    with_derivatives=True,
):
    """Map integrated face loads and neighboring derivatives to native cells."""
    if not chain.fin_cell_indices or W <= 0 or fraction <= 0:
        return {}
    km = mass_transfer_coefficient(alpha, cp, lewis_number=lewis)
    mesh = chain.mesh
    indices = (chain.root_index, *chain.fin_cell_indices, chain.fin_tip_index)
    radii = (mesh.r_root, *mesh.cell_centers, mesh.r_tip)
    base = (
        chain.fixed_fin_base_temperature
        if chain.root_index is None
        else temperatures[chain.root_index]
    )
    nodes = (
        base,
        *(temperatures[i] for i in chain.fin_cell_indices),
        temperatures[chain.fin_tip_index],
    )
    result = {}
    for k, index in enumerate(chain.fin_cell_indices):
        west, east = mesh.r_root + k * mesh.dr, mesh.r_root + (k + 1) * mesh.dr
        # Exact native physical face area, including any authoritative override.
        factor = chain.surface_areas[index] / (
            2 * math.pi * (east * east - west * west)
        )
        values = [0.0] * 5
        derivatives = {}
        wet_area = 0.0
        for left, a, b in ((k, west, radii[k + 1]), (k + 1, radii[k + 1], east)):
            vals, jac, wet = integrate_face_piece(
                r_left=radii[left],
                r_right=radii[left + 1],
                T_left=nodes[left],
                T_right=nodes[left + 1],
                a=a,
                b=b,
                slope=factor,
                km=km,
                W=W,
                properties=properties,
                dew=dew,
                order=order,
                offset=offset,
                fraction=fraction,
                with_derivatives=with_derivatives,
            )
            values = [x + y for x, y in zip(values, vals)]
            wet_area += wet
            for side, column in enumerate(indices[left : left + 2]):
                if column is not None:
                    old = derivatives.setdefault(column, [0.0] * 3)
                    for row in range(3):
                        old[row] += jac[row][side]
        result[index] = (values, derivatives, wet_area)
    return result
