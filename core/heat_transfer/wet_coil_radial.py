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
    _saturation_enthalpy_pairs_batch,
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

    def prefetch(self, temperatures, *, with_derivatives):
        """Group only exact, uncached quadrature/derivative temperatures."""
        requested = dict.fromkeys(temperatures)
        if with_derivatives:
            step = 1e-4
            for T in temperatures:
                if T - step < WATER_TRIPLE_POINT_TEMPERATURE_K:
                    points = (T + step, T + 2 * step)
                elif T + step >= WATER_CRITICAL_TEMPERATURE_K:
                    points = (T - step, T - 2 * step)
                else:
                    points = (T - step, T + step)
                requested.update(dict.fromkeys(points))
        ready, humidity = [], []
        for T in requested:
            if T in self._values:
                continue
            try:
                sigma = saturated_water_ratio(
                    p_total=self.p, T=T, M_dry=self.md, M_h2o=self.mw
                )
            except ValueError as exc:
                if "no saturated gas-phase state exists" not in str(exc):
                    raise
                self._values[T] = (math.inf, 0.0, 0.0)
                continue
            ready.append(T)
            humidity.append(sigma)
        if not ready:
            return
        pairs = _saturation_enthalpy_pairs_batch(ready)
        for T, sigma, (hf, hg) in zip(ready, humidity, pairs):
            self._values[T] = (sigma, hf, hg - hf)

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


def _face_quadrature(r_left, r_right, T_left, T_right, a, b, dew, order, offset):
    """Identical wet-support nodes for property prefetch and integration."""
    if dew is None:
        return (), a, a
    dt, dr = T_right - T_left, r_right - r_left
    ta = T_left + (a - r_left) / dr * dt + offset
    tb = T_left + (b - r_left) / dr * dt + offset
    if min(ta, tb) >= dew:
        return (), a, a
    lo, hi = a, b
    if max(ta, tb) > dew:
        crossing = r_left + (dew - offset - T_left) * dr / dt
        if dt > 0:
            hi = crossing
        else:
            lo = crossing
    if hi <= lo:
        return (), a, a
    points, weights = _GAUSS[order]
    samples = []
    for point, weight in zip(points, weights):
        r = (hi + lo) / 2 + point * (hi - lo) / 2
        right_shape = (r - r_left) / dr
        shapes = (1 - right_shape, right_shape)
        T = T_left + right_shape * dt + offset
        samples.append((r, shapes, T, weight))
    return samples, lo, hi


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
    _prefetched=False,
):
    """Integrals and derivatives with respect to two radial endpoint nodes.

    Values are mass, latent, drain, mass*temperature, mass*saturation and wet-area*temperature.
    Jacobian rows are mass, latent and drain; columns are the two temperatures.
    ``slope`` includes the actual taper and geometric fin-area replication.
    """
    values = [0.0] * 6
    jacobian = [[0.0, 0.0] for _ in range(3)]
    if dew is None or fraction == 0:
        return values, jacobian, 0.0
    samples, lo, hi = _face_quadrature(
        r_left, r_right, T_left, T_right, a, b, dew, order, offset
    )
    prefetch = getattr(properties, "prefetch", None)
    if prefetch is not None and not _prefetched:
        prefetch([sample[2] for sample in samples], with_derivatives=with_derivatives)
    for r, shapes, T, weight in samples:
        sigma, hl, hfg = properties.values(T)
        driving = W - sigma
        if driving <= 0:
            continue  # Physical positive-part law at the saturation boundary.
        measure = 4 * math.pi * r * slope * weight * (hi - lo) / 2 * fraction
        mass = km * driving
        for k, value in enumerate(
            (mass, mass * hfg, mass * hl, mass * T, mass * sigma, T)
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
    prefetch = getattr(properties, "prefetch", None)
    if prefetch is not None:
        # Evaluate exact IF97 rows in one fin-wide batch. Nodes, derivative
        # stencils and per-row arithmetic are unchanged; no interpolation.
        pending = []
        for k in range(len(chain.fin_cell_indices)):
            west, east = mesh.r_root + k * mesh.dr, mesh.r_root + (k + 1) * mesh.dr
            for left, a, b in ((k, west, radii[k + 1]), (k + 1, radii[k + 1], east)):
                samples, _, _ = _face_quadrature(
                    radii[left], radii[left + 1], nodes[left], nodes[left + 1],
                    a, b, dew, order, offset,
                )
                pending.extend(sample[2] for sample in samples)
        prefetch(pending, with_derivatives=with_derivatives)
    result = {}
    for k, index in enumerate(chain.fin_cell_indices):
        west, east = mesh.r_root + k * mesh.dr, mesh.r_root + (k + 1) * mesh.dr
        # Exact native physical face area, including any authoritative override.
        factor = chain.surface_areas[index] / (
            2 * math.pi * (east * east - west * west)
        )
        values = [0.0] * 6
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
                _prefetched=prefetch is not None,
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
