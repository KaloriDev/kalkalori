# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only

"""Nonlinear radial wet-fin response for circular-finned outside surfaces.

This module is intentionally a surface component rather than a second heat-
exchanger condensation solver.  The whole-HX integrations own wet-gas outlet
enthalpy/composition and call this component with one representative bulk
state.  Heat is positive from the wet outside gas into the tube/core.

The annular-fin discretisation is the conservative geometry used by
``core.heat_transfer.fin_efficiency``: uniform radial control volumes, exact
areas for both (possibly sloping) faces, linear-taper radial conduction areas,
and a physical convecting tip node.  Local H2O transfer uses the existing
project functions and dry-carrier mass basis.

Wet extended-surface formulation references (SI units throughout):

* Sharqawy, Moinuddin & Zubair (2012), International Journal of
  Refrigeration, DOI 10.1016/j.ijrefrig.2011.11.004.
* Sharqawy & Zubair (2007), International Journal of Refrigeration,
  DOI 10.1016/j.ijrefrig.2006.12.008.
* Rosario & Rahman (1999), International Journal of Heat and Fluid Flow,
  DOI 10.1016/S0142-727X(99)00057-0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import math
from typing import Mapping

from core.common.warnings import ModelWarning
from core.geometry.bundle import TubeBundle
from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.tube import TubeSurfaceType
from core.heat_transfer.outside_dispatch import ThermalResistanceNetwork
from core.phase_change.condensation_solver_helpers import FrostingNotSupportedError
from core.phase_change.mass_heat_transfer import (
    condensation_mass_flux,
    mass_transfer_coefficient,
)
from core.phase_change.water_equilibrium import (
    WATER_MOLAR_MASS_KG_PER_MOL,
    saturated_water_ratio,
)
from core.properties.water import (
    WATER_TRIPLE_POINT_TEMPERATURE_K,
    water_latent_heat_of_vaporization,
    water_saturation_liquid_enthalpy,
)


_METHOD = "damped_newton_tridiagonal_wet_annular_fin"
# Wet properties make each nonlinear cell materially more expensive than the
# verified linear dry FVM.  A 160-cell wet default resolves radial dew-point
# crossings while keeping whole-HX and notebook fixed-point loops practical;
# the dry solver's independent 400-cell default is deliberately untouched.
DEFAULT_WET_FIN_RADIAL_CELLS = 160
_MIN_LINE_SEARCH_DAMPING = 2.0**-14
_DERIVATIVE_STEP_RELATIVE = 1.0e-6
_DERIVATIVE_STEP_MIN_K = 1.0e-4
_WATER_PROPERTY_INTERPOLATION_STEP_K = 0.25


class WetFinState(str, Enum):
    """Radial wet state of one representative annular fin."""

    DRY = "dry"
    PARTIALLY_WET = "partially_wet"
    FULLY_WET = "fully_wet"


class WetFinConvergenceError(RuntimeError):
    """Raised instead of returning an unconverged wet-fin iterate."""

    def __init__(
        self,
        message: str,
        *,
        iterations: int,
        residuals: Mapping[str, float],
    ) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.residuals = dict(residuals)


@dataclass(frozen=True)
class WetAnnularFinResult:
    """Solved heat/mass response of one physical annular fin.

    Areas and rates are for one geometric fin.  ``wet_fin_area`` is the area
    of both faces from the root to a linearly interpolated dew-point crossing;
    the physical tip area is included only when the tip itself is wet.
    ``wet_dry_boundary_radius`` is therefore the radial dew-point crossing on
    the two faces and is ``None`` for dry and fully-wet states.
    """

    fin_wet_state: WetFinState
    fin_wet_fraction: float
    wet_fin_area: float
    dry_fin_area: float

    Q_fin_sensible: float
    Q_fin_latent: float
    Q_fin_total: float
    m_dot_condensate_fin: float
    condensate_enthalpy_rate_fin: float

    fin_base_temperature: float
    fin_tip_temperature: float
    wet_dry_boundary_radius: float | None

    radial_cell_centers: tuple[float, ...]
    radial_cell_temperatures: tuple[float, ...]
    radial_cell_side_areas: tuple[float, ...]
    radial_cell_condensate_fluxes: tuple[float, ...]
    tip_condensate_flux: float

    outside_alpha_physical: float
    mass_transfer_coefficient: float
    radial_cells: int
    iterations: int
    residuals: Mapping[str, float]
    method: str = _METHOD


@dataclass(frozen=True)
class WetFinnedSurfaceResult:
    """Whole-bundle primary-plus-fin wet outside-surface response.

    The fin temperature field always uses physical fin geometry.  Its rates
    and areas are multiplied by ``equivalent_fin_count``, defined from the
    authoritative solver fin area.  This preserves external-area overrides
    without changing fin shape or efficiency.
    """

    fin_wet_state: WetFinState
    fin_wet_fraction: float
    wet_fin_area: float
    dry_fin_area: float
    Q_fin_sensible: float
    Q_fin_latent: float
    Q_fin_total: float
    m_dot_condensate_fin: float
    fin_base_temperature: float
    fin_tip_temperature: float
    wet_dry_boundary_radius: float | None

    Q_primary_sensible: float
    Q_primary_latent: float
    Q_primary_total: float
    m_dot_condensate_primary: float
    primary_surface_temperature: float
    wet_primary_area: float

    Q_sensible: float
    Q_latent: float
    Q_total: float
    m_dot_condensate: float
    condensate_enthalpy_rate: float
    wet_area: float
    wet_surface_fraction: float

    core_wall_temperature: float
    inside_wall_temperature: float
    root_surface_temperature: float
    wall_temperature_wet_mean: float | None
    W_sat_wet_surface: float | None

    outside_total_area: float
    primary_area: float
    fin_area: float
    equivalent_fin_count: float
    water_availability_scale: float

    outside_alpha_physical: float
    outside_alpha_wet_effective_gross_core_basis: float
    outside_alpha_wet_effective_basis: str
    mass_transfer_coefficient: float
    contact_topology: str
    contact_input_mode: str
    contact_resistance_used: float
    resistance_contact: float
    resistance_root: float

    mass_balance_error: float
    energy_balance_error: float
    iterations: int
    residuals: Mapping[str, float]
    assumptions: tuple[str, ...]
    warnings: tuple[ModelWarning, ...]
    annular_fin: WetAnnularFinResult | None = None
    method: str = _METHOD


@dataclass(frozen=True)
class _FinMesh:
    r_root: float
    r_tip: float
    dr: float
    cell_centers: tuple[float, ...]
    side_areas: tuple[float, ...]
    conductance_root: float
    conductances_between_cells: tuple[float, ...]
    conductance_tip: float
    tip_area: float


@dataclass(frozen=True)
class _Chain:
    mesh: _FinMesh
    edges: tuple[float, ...]
    boundary_conductances: tuple[float, ...]
    boundary_temperatures: tuple[float, ...]
    surface_areas: tuple[float, ...]
    fin_surface_indices: tuple[int, ...]
    fin_cell_indices: tuple[int, ...]
    fin_tip_index: int | None
    primary_index: int | None
    core_index: int | None
    root_index: int | None
    fixed_fin_base_temperature: float | None
    fin_area_scale: float


@dataclass(frozen=True)
class _TransferEvaluation:
    raw_mass_flux: float
    mass_flux: float
    sensible_flux: float
    latent_flux: float
    total_flux: float
    condensate_enthalpy_flux: float
    W_sat: float | None


@dataclass(frozen=True)
class _ChainEvaluation:
    residual: tuple[float, ...]
    lower: tuple[float, ...]
    diagonal: tuple[float, ...]
    upper: tuple[float, ...]
    transfers: tuple[_TransferEvaluation, ...]
    availability_scale: float
    Q_sensible: float
    Q_latent: float
    Q_total: float
    m_dot_condensate: float
    condensate_enthalpy_rate: float
    equation_residual_W: float


def solve_wet_annular_fin(
    tube: CircularFinnedTube,
    *,
    fin_base_temperature: float,
    gas_bulk_temperature: float,
    outside_alpha_physical: float,
    cp_gas: float,
    W_bulk: float,
    p_total: float,
    M_dry: float,
    M_h2o: float = WATER_MOLAR_MASS_KG_PER_MOL,
    lewis_number: float = 1.0,
    radial_cells: int = DEFAULT_WET_FIN_RADIAL_CELLS,
    max_iterations: int = 80,
    temperature_tolerance_K: float = 1.0e-7,
    relative_heat_tolerance: float = 1.0e-9,
    condensate_tolerance_kg_s: float = 1.0e-12,
    relaxation_factor: float = 1.0,
) -> WetAnnularFinResult:
    """Solve one wet annular fin with a prescribed physical base temperature."""

    _validate_common_inputs(
        tube=tube,
        gas_bulk_temperature=gas_bulk_temperature,
        outside_alpha_physical=outside_alpha_physical,
        cp_gas=cp_gas,
        W_bulk=W_bulk,
        p_total=p_total,
        M_dry=M_dry,
        M_h2o=M_h2o,
        lewis_number=lewis_number,
        radial_cells=radial_cells,
        max_iterations=max_iterations,
        temperature_tolerance_K=temperature_tolerance_K,
        relative_heat_tolerance=relative_heat_tolerance,
        condensate_tolerance_kg_s=condensate_tolerance_kg_s,
        relaxation_factor=relaxation_factor,
    )
    _validate_temperature(fin_base_temperature, "fin_base_temperature")

    mesh = _build_fin_mesh(tube, radial_cells)
    chain = _build_prescribed_base_chain(mesh, fin_base_temperature)
    temperatures, evaluation, iterations, residuals = _solve_nonlinear_chain(
        chain,
        tube=tube,
        gas_bulk_temperature=gas_bulk_temperature,
        outside_alpha_physical=outside_alpha_physical,
        cp_gas=cp_gas,
        W_bulk=W_bulk,
        p_total=p_total,
        M_dry=M_dry,
        M_h2o=M_h2o,
        lewis_number=lewis_number,
        m_dot_water_vapor_available=math.inf,
        max_iterations=max_iterations,
        temperature_tolerance_K=temperature_tolerance_K,
        relative_heat_tolerance=relative_heat_tolerance,
        condensate_tolerance_kg_s=condensate_tolerance_kg_s,
        relaxation_factor=relaxation_factor,
        temperature_bounds=sorted((fin_base_temperature, gas_bulk_temperature)),
    )

    return _build_annular_fin_result(
        tube=tube,
        chain=chain,
        temperatures=temperatures,
        evaluation=evaluation,
        fin_base_temperature=fin_base_temperature,
        gas_bulk_temperature=gas_bulk_temperature,
        outside_alpha_physical=outside_alpha_physical,
        cp_gas=cp_gas,
        W_bulk=W_bulk,
        p_total=p_total,
        M_dry=M_dry,
        M_h2o=M_h2o,
        lewis_number=lewis_number,
        iterations=iterations,
        residuals=residuals,
    )


def solve_wet_finned_surface(
    bundle: TubeBundle,
    network: ThermalResistanceNetwork,
    *,
    gas_bulk_temperature: float,
    inside_bulk_temperature: float,
    cp_gas: float,
    W_bulk: float,
    p_total: float,
    M_dry: float,
    m_dot_water_vapor_available: float,
    M_h2o: float = WATER_MOLAR_MASS_KG_PER_MOL,
    lewis_number: float = 1.0,
    radial_cells: int = DEFAULT_WET_FIN_RADIAL_CELLS,
    max_iterations: int = 80,
    temperature_tolerance_K: float = 1.0e-7,
    relative_heat_tolerance: float = 1.0e-9,
    condensate_tolerance_kg_s: float = 1.0e-10,
    relaxation_factor: float = 1.0,
) -> WetFinnedSurfaceResult:
    """Solve the whole circular-finned outside surface at one bulk state.

    ``network`` must be the dry operating-point resistance network built with
    the same physical outside HTC.  It supplies the already-resolved contact
    precedence/equivalent resistance and the downstream inside-film plus core-
    wall terms.  The nonlinear chain then applies these topology terms once.
    """

    if not isinstance(bundle, TubeBundle):
        raise TypeError("bundle must be a TubeBundle instance.")
    tube = bundle.tube
    if not isinstance(tube, CircularFinnedTube):
        raise TypeError("solve_wet_finned_surface requires CircularFinnedTube geometry.")
    if not isinstance(network, ThermalResistanceNetwork):
        raise TypeError("network must be a ThermalResistanceNetwork instance.")
    if network.surface_type is not TubeSurfaceType.CIRCULAR_FINNED:
        raise ValueError("network must describe a circular-finned surface.")
    _validate_temperature(inside_bulk_temperature, "inside_bulk_temperature")
    if gas_bulk_temperature <= inside_bulk_temperature:
        raise ValueError(
            "Wet outside condensation requires gas_bulk_temperature greater "
            "than inside_bulk_temperature."
        )
    if (
        not math.isfinite(m_dot_water_vapor_available)
        or m_dot_water_vapor_available < 0.0
    ):
        raise ValueError(
            "m_dot_water_vapor_available must be finite and non-negative."
        )

    _validate_common_inputs(
        tube=tube,
        gas_bulk_temperature=gas_bulk_temperature,
        outside_alpha_physical=network.outside_alpha_physical,
        cp_gas=cp_gas,
        W_bulk=W_bulk,
        p_total=p_total,
        M_dry=M_dry,
        M_h2o=M_h2o,
        lewis_number=lewis_number,
        radial_cells=radial_cells,
        max_iterations=max_iterations,
        temperature_tolerance_K=temperature_tolerance_K,
        relative_heat_tolerance=relative_heat_tolerance,
        condensate_tolerance_kg_s=condensate_tolerance_kg_s,
        relaxation_factor=relaxation_factor,
    )
    _validate_network_areas(bundle, network)

    mesh = _build_fin_mesh(tube, radial_cells)
    equivalent_fin_count = network.area_fin / tube.fin_area_per_fin
    chain = _build_bundle_chain(
        mesh,
        network=network,
        inside_bulk_temperature=inside_bulk_temperature,
        equivalent_fin_count=equivalent_fin_count,
    )
    temperatures, evaluation, iterations, residuals = _solve_nonlinear_chain(
        chain,
        tube=tube,
        gas_bulk_temperature=gas_bulk_temperature,
        outside_alpha_physical=network.outside_alpha_physical,
        cp_gas=cp_gas,
        W_bulk=W_bulk,
        p_total=p_total,
        M_dry=M_dry,
        M_h2o=M_h2o,
        lewis_number=lewis_number,
        m_dot_water_vapor_available=m_dot_water_vapor_available,
        max_iterations=max_iterations,
        temperature_tolerance_K=temperature_tolerance_K,
        relative_heat_tolerance=relative_heat_tolerance,
        condensate_tolerance_kg_s=condensate_tolerance_kg_s,
        relaxation_factor=relaxation_factor,
        temperature_bounds=(inside_bulk_temperature, gas_bulk_temperature),
    )

    core_index = chain.core_index
    root_index = chain.root_index
    primary_index = chain.primary_index
    assert core_index is not None and root_index is not None
    assert primary_index is not None
    T_core = temperatures[core_index]
    T_root = temperatures[root_index]
    T_primary = temperatures[primary_index]
    R_downstream = network.resistance_inside + network.resistance_core_wall
    Q_downstream = (T_core - inside_bulk_temperature) / R_downstream

    primary_transfer = evaluation.transfers[primary_index]
    primary_area = network.area_primary_outside
    Q_primary_sensible = primary_area * primary_transfer.sensible_flux
    Q_primary_latent = primary_area * primary_transfer.latent_flux
    m_primary = primary_area * primary_transfer.mass_flux
    H_primary = primary_area * primary_transfer.condensate_enthalpy_flux

    fin_base_temperature = T_root
    annular_fin: WetAnnularFinResult | None
    if equivalent_fin_count > 0.0:
        annular_fin = _build_annular_fin_result(
            tube=tube,
            chain=chain,
            temperatures=temperatures,
            evaluation=evaluation,
            fin_base_temperature=fin_base_temperature,
            gas_bulk_temperature=gas_bulk_temperature,
            outside_alpha_physical=network.outside_alpha_physical,
            cp_gas=cp_gas,
            W_bulk=W_bulk,
            p_total=p_total,
            M_dry=M_dry,
            M_h2o=M_h2o,
            lewis_number=lewis_number,
            iterations=iterations,
            residuals=residuals,
        )
        Q_fin_sensible = annular_fin.Q_fin_sensible * equivalent_fin_count
        Q_fin_latent = annular_fin.Q_fin_latent * equivalent_fin_count
        m_fin = annular_fin.m_dot_condensate_fin * equivalent_fin_count
        H_fin = annular_fin.condensate_enthalpy_rate_fin * equivalent_fin_count
        wet_fin_area = annular_fin.wet_fin_area * equivalent_fin_count
        dry_fin_area = network.area_fin - wet_fin_area
        fin_state = annular_fin.fin_wet_state
        fin_wet_fraction = annular_fin.fin_wet_fraction
        fin_tip_temperature = annular_fin.fin_tip_temperature
        boundary_radius = annular_fin.wet_dry_boundary_radius
    else:
        annular_fin = None
        Q_fin_sensible = Q_fin_latent = m_fin = H_fin = 0.0
        wet_fin_area = dry_fin_area = 0.0
        fin_state = WetFinState.DRY
        fin_wet_fraction = 0.0
        fin_tip_temperature = fin_base_temperature
        boundary_radius = None

    Q_primary_total = Q_primary_sensible + Q_primary_latent
    Q_fin_total = Q_fin_sensible + Q_fin_latent
    Q_sensible = Q_primary_sensible + Q_fin_sensible
    Q_latent = Q_primary_latent + Q_fin_latent
    Q_total = Q_sensible + Q_latent
    m_total = m_primary + m_fin
    H_total = H_primary + H_fin

    primary_is_wet = primary_transfer.raw_mass_flux > 0.0
    wet_primary_area = primary_area if primary_is_wet else 0.0
    wet_area = wet_primary_area + wet_fin_area
    wet_fraction = wet_area / network.area_outside_gross

    wet_mass_temperature_sum = 0.0
    wet_mass_W_sat_sum = 0.0
    for index, transfer in enumerate(evaluation.transfers):
        node_mass = chain.surface_areas[index] * transfer.mass_flux
        if node_mass > 0.0:
            wet_mass_temperature_sum += node_mass * temperatures[index]
            assert transfer.W_sat is not None
            wet_mass_W_sat_sum += node_mass * transfer.W_sat
    T_wet_mean = (
        None if m_total <= 0.0 else wet_mass_temperature_sum / m_total
    )
    W_sat_wet = None if m_total <= 0.0 else wet_mass_W_sat_sum / m_total

    inside_wall_temperature = (
        inside_bulk_temperature + Q_downstream * network.resistance_inside
    )
    split_mass_error = m_total - (m_primary + m_fin)
    split_energy_error = Q_total - (Q_sensible + Q_latent)
    conduction_error = Q_downstream - Q_total
    final_residuals = dict(residuals)
    final_residuals.update(
        {
            "surface_to_downstream_energy_W": abs(conduction_error),
            "primary_fin_mass_split_kg_s": abs(split_mass_error),
            "sensible_latent_energy_split_W": abs(split_energy_error),
        }
    )
    wet_alpha_denominator = network.area_outside_gross * (
        gas_bulk_temperature - T_core
    )
    if not math.isfinite(wet_alpha_denominator) or wet_alpha_denominator <= 0.0:
        raise ValueError(
            "Cannot reconstruct wet effective gross-area alpha from a non-positive "
            "bulk-gas-to-core-wall temperature difference."
        )
    outside_alpha_wet_effective = Q_total / wet_alpha_denominator

    return WetFinnedSurfaceResult(
        fin_wet_state=fin_state,
        fin_wet_fraction=fin_wet_fraction,
        wet_fin_area=wet_fin_area,
        dry_fin_area=dry_fin_area,
        Q_fin_sensible=Q_fin_sensible,
        Q_fin_latent=Q_fin_latent,
        Q_fin_total=Q_fin_total,
        m_dot_condensate_fin=m_fin,
        fin_base_temperature=fin_base_temperature,
        fin_tip_temperature=fin_tip_temperature,
        wet_dry_boundary_radius=boundary_radius,
        Q_primary_sensible=Q_primary_sensible,
        Q_primary_latent=Q_primary_latent,
        Q_primary_total=Q_primary_total,
        m_dot_condensate_primary=m_primary,
        primary_surface_temperature=T_primary,
        wet_primary_area=wet_primary_area,
        Q_sensible=Q_sensible,
        Q_latent=Q_latent,
        Q_total=Q_total,
        m_dot_condensate=m_total,
        condensate_enthalpy_rate=H_total,
        wet_area=wet_area,
        wet_surface_fraction=wet_fraction,
        core_wall_temperature=T_core,
        inside_wall_temperature=inside_wall_temperature,
        root_surface_temperature=T_root,
        wall_temperature_wet_mean=T_wet_mean,
        W_sat_wet_surface=W_sat_wet,
        outside_total_area=network.area_outside_gross,
        primary_area=primary_area,
        fin_area=network.area_fin,
        equivalent_fin_count=equivalent_fin_count,
        water_availability_scale=evaluation.availability_scale,
        outside_alpha_physical=network.outside_alpha_physical,
        outside_alpha_wet_effective_gross_core_basis=outside_alpha_wet_effective,
        outside_alpha_wet_effective_basis=(
            "gross_outside_area_and_bulk_gas_to_core_wall_temperature_difference"
        ),
        mass_transfer_coefficient=mass_transfer_coefficient(
            network.outside_alpha_physical,
            cp_gas,
            lewis_number=lewis_number,
        ),
        contact_topology=network.contact_topology,
        contact_input_mode=network.contact_input_mode,
        contact_resistance_used=network.contact_resistance_used,
        resistance_contact=network.resistance_contact,
        resistance_root=network.resistance_root,
        mass_balance_error=split_mass_error,
        energy_balance_error=max(abs(split_energy_error), abs(conduction_error)),
        iterations=iterations,
        residuals=final_residuals,
        assumptions=(
            "bulk_mean_property_evaluation_0d",
            "authoritative_water_properties_linear_interpolation_0.25_K",
            "fully_drained_liquid_condensate",
            "local_saturated_liquid_condensate_enthalpy",
            "lewis_number_chilton_colburn_analogy",
            "dry_gas_composition_unchanged_by_condensation",
            "authoritative_fin_area_uniform_equivalent_fin_count",
            "no_condensate_film_resistance_or_wet_pressure_drop_correction",
        ),
        warnings=(),
        annular_fin=annular_fin,
    )


def _build_fin_mesh(tube: CircularFinnedTube, radial_cells: int) -> _FinMesh:
    r_root = 0.5 * tube.D_root
    r_tip = 0.5 * tube.D_fin
    dr = (r_tip - r_root) / radial_cells
    slope = tube.fin_side_slope_factor

    centers: list[float] = []
    side_areas: list[float] = []
    for index in range(radial_cells):
        r_west = r_root + index * dr
        r_east = r_west + dr
        centers.append(0.5 * (r_west + r_east))
        # Both sloping faces are integrated exactly over this radial FVM
        # cell [m2].  Wet annular-fin formulation: Sharqawy & Zubair (2007),
        # DOI 10.1016/j.ijrefrig.2006.12.008; tapered wet-fin treatment:
        # Rosario & Rahman (1999), DOI 10.1016/S0142-727X(99)00057-0.
        side_areas.append(
            2.0 * math.pi * slope * (r_east * r_east - r_west * r_west)
        )

    conductance_root = (
        tube.fin_k * _radial_conduction_area(tube, r_root) / (0.5 * dr)
    )
    conductances_between = tuple(
        tube.fin_k
        * _radial_conduction_area(tube, r_root + index * dr)
        / dr
        for index in range(1, radial_cells)
    )
    conductance_tip = (
        tube.fin_k * tube.fin_tip_area_per_fin / (0.5 * dr)
    )
    area_sum = math.fsum(side_areas) + tube.fin_tip_area_per_fin
    expected_area = tube.fin_area_per_fin
    area_scale = max(abs(area_sum), abs(expected_area))
    # Every cell area contains rounded face radii and a subtraction of their
    # squares.  Permit their accumulated floating-point error in proportion
    # to both physical area scale and cell count; this remains many orders of
    # magnitude tighter than any geometric or thermal tolerance.
    area_closure_tolerance = max(
        32.0 * (radial_cells + 2) * math.ulp(area_scale),
        8.0 * math.ulp(1.0) * area_scale,
    )
    if abs(area_sum - expected_area) > area_closure_tolerance:
        raise ValueError("Wet annular-fin mesh does not close its physical area.")
    return _FinMesh(
        r_root=r_root,
        r_tip=r_tip,
        dr=dr,
        cell_centers=tuple(centers),
        side_areas=tuple(side_areas),
        conductance_root=conductance_root,
        conductances_between_cells=conductances_between,
        conductance_tip=conductance_tip,
        tip_area=tube.fin_tip_area_per_fin,
    )


def _build_prescribed_base_chain(
    mesh: _FinMesh,
    fin_base_temperature: float,
) -> _Chain:
    count = len(mesh.cell_centers) + 1
    edges = [0.0] * (count - 1)
    for index, conductance in enumerate(mesh.conductances_between_cells):
        edges[index] = conductance
    edges[-1] = mesh.conductance_tip
    surface_areas = (*mesh.side_areas, mesh.tip_area)
    boundaries = [0.0] * count
    boundary_temperatures = [0.0] * count
    boundaries[0] = mesh.conductance_root
    boundary_temperatures[0] = fin_base_temperature
    indices = tuple(range(count))
    return _Chain(
        mesh=mesh,
        edges=tuple(edges),
        boundary_conductances=tuple(boundaries),
        boundary_temperatures=tuple(boundary_temperatures),
        surface_areas=tuple(surface_areas),
        fin_surface_indices=indices,
        fin_cell_indices=tuple(range(count - 1)),
        fin_tip_index=count - 1,
        primary_index=None,
        core_index=None,
        root_index=None,
        fixed_fin_base_temperature=fin_base_temperature,
        fin_area_scale=1.0,
    )


def _build_bundle_chain(
    mesh: _FinMesh,
    *,
    network: ThermalResistanceNetwork,
    inside_bulk_temperature: float,
    equivalent_fin_count: float,
) -> _Chain:
    has_fin = equivalent_fin_count > 0.0
    welded = network.contact_topology == "fin_branch_only"
    continuous = network.contact_topology == (
        "series_before_primary_and_fin_parallel_branches"
    )
    if not (welded or continuous):
        raise ValueError(f"Unsupported fin contact topology: {network.contact_topology!r}.")

    separate_root = continuous or (has_fin and network.resistance_contact > 0.0)
    prefix_count = 2 if separate_root else 1
    fin_node_count = len(mesh.cell_centers) + 1 if has_fin else 0
    count = prefix_count + fin_node_count
    core_index = 0
    root_index = 1 if separate_root else 0
    primary_index = core_index if welded else root_index
    first_cell_index = prefix_count if has_fin else None
    tip_index = count - 1 if has_fin else None

    edges = [0.0] * max(count - 1, 0)
    if separate_root:
        common_resistance = (
            network.resistance_root + network.resistance_contact
            if continuous
            else network.resistance_contact
        )
        if not math.isfinite(common_resistance) or common_resistance <= 0.0:
            raise ValueError("Separated fin/root node requires positive finite resistance.")
        edges[0] = 1.0 / common_resistance
    if has_fin:
        assert first_cell_index is not None
        edges[first_cell_index - 1] = (
            equivalent_fin_count * mesh.conductance_root
        )
        for offset, conductance in enumerate(mesh.conductances_between_cells):
            edges[first_cell_index + offset] = equivalent_fin_count * conductance
        assert tip_index is not None
        edges[tip_index - 1] = equivalent_fin_count * mesh.conductance_tip

    boundaries = [0.0] * count
    boundary_temperatures = [0.0] * count
    R_downstream = network.resistance_inside + network.resistance_core_wall
    if not math.isfinite(R_downstream) or R_downstream <= 0.0:
        raise ValueError("Inside-film plus core-wall resistance must be positive.")
    boundaries[core_index] = 1.0 / R_downstream
    boundary_temperatures[core_index] = inside_bulk_temperature

    surface_areas = [0.0] * count
    surface_areas[primary_index] = network.area_primary_outside
    fin_surface_indices: tuple[int, ...] = ()
    fin_cell_indices: tuple[int, ...] = ()
    if has_fin:
        assert first_cell_index is not None and tip_index is not None
        for offset, area in enumerate(mesh.side_areas):
            surface_areas[first_cell_index + offset] = equivalent_fin_count * area
        surface_areas[tip_index] = equivalent_fin_count * mesh.tip_area
        fin_surface_indices = tuple(range(first_cell_index, tip_index + 1))
        fin_cell_indices = tuple(range(first_cell_index, tip_index))

    return _Chain(
        mesh=mesh,
        edges=tuple(edges),
        boundary_conductances=tuple(boundaries),
        boundary_temperatures=tuple(boundary_temperatures),
        surface_areas=tuple(surface_areas),
        fin_surface_indices=fin_surface_indices,
        fin_cell_indices=fin_cell_indices,
        fin_tip_index=tip_index,
        primary_index=primary_index,
        core_index=core_index,
        root_index=root_index,
        fixed_fin_base_temperature=None,
        fin_area_scale=equivalent_fin_count,
    )


def _solve_nonlinear_chain(
    chain: _Chain,
    *,
    tube: CircularFinnedTube,
    gas_bulk_temperature: float,
    outside_alpha_physical: float,
    cp_gas: float,
    W_bulk: float,
    p_total: float,
    M_dry: float,
    M_h2o: float,
    lewis_number: float,
    m_dot_water_vapor_available: float,
    max_iterations: int,
    temperature_tolerance_K: float,
    relative_heat_tolerance: float,
    condensate_tolerance_kg_s: float,
    relaxation_factor: float,
    temperature_bounds: list[float] | tuple[float, float],
) -> tuple[list[float], _ChainEvaluation, int, dict[str, float]]:
    del tube  # geometry is fully represented by ``chain`` at this layer.
    low_temperature, high_temperature = temperature_bounds
    initial_temperature = 0.5 * (low_temperature + high_temperature)
    temperatures = [initial_temperature] * len(chain.surface_areas)

    # One Newton step solves the latent-free linear chain exactly and is a
    # deterministic dry-field initialisation for the nonlinear solve.
    dry = _evaluate_chain(
        chain,
        temperatures,
        gas_bulk_temperature=gas_bulk_temperature,
        outside_alpha_physical=outside_alpha_physical,
        cp_gas=cp_gas,
        W_bulk=W_bulk,
        p_total=p_total,
        M_dry=M_dry,
        M_h2o=M_h2o,
        lewis_number=lewis_number,
        m_dot_water_vapor_available=0.0,
    )
    dry_delta = _solve_tridiagonal(
        list(dry.lower),
        list(dry.diagonal),
        list(dry.upper),
        [-value for value in dry.residual],
    )
    temperatures = [
        temperature + delta
        for temperature, delta in zip(temperatures, dry_delta)
    ]

    previous_mass: float | None = None
    residuals: dict[str, float] = {}
    q_scale = _heat_scale(
        outside_alpha_physical,
        sum(chain.surface_areas),
        high_temperature - low_temperature,
    )
    mass_scale = max(
        mass_transfer_coefficient(
            outside_alpha_physical,
            cp_gas,
            lewis_number=lewis_number,
        )
        * sum(chain.surface_areas)
        * max(W_bulk, 1.0e-12),
        1.0e-15,
    )

    for iteration in range(1, max_iterations + 1):
        current = _evaluate_chain(
            chain,
            temperatures,
            gas_bulk_temperature=gas_bulk_temperature,
            outside_alpha_physical=outside_alpha_physical,
            cp_gas=cp_gas,
            W_bulk=W_bulk,
            p_total=p_total,
            M_dry=M_dry,
            M_h2o=M_h2o,
            lewis_number=lewis_number,
            m_dot_water_vapor_available=m_dot_water_vapor_available,
        )
        heat_tolerance = max(
            relative_heat_tolerance
            * max(q_scale, abs(current.Q_total), 1.0e-12),
            _jacobian_roundoff_floor(current, temperatures),
        )
        mass_step = (
            0.0
            if previous_mass is None
            else abs(current.m_dot_condensate - previous_mass)
        )
        residuals = {
            "temperature_step_K": 0.0,
            "equation_residual_W": current.equation_residual_W,
            "energy_balance_W": _chain_energy_balance_error(chain, temperatures, current),
            "condensate_step_kg_s": mass_step,
            "availability_scale": current.availability_scale,
        }
        if (
            current.equation_residual_W <= heat_tolerance
            and residuals["energy_balance_W"] <= heat_tolerance
            and (previous_mass is None or mass_step <= max(
                condensate_tolerance_kg_s,
                relative_heat_tolerance * mass_scale,
            ))
        ):
            return temperatures, current, iteration, residuals

        delta = _solve_tridiagonal(
            list(current.lower),
            list(current.diagonal),
            list(current.upper),
            [-value for value in current.residual],
        )
        damping = relaxation_factor
        current_norm = current.equation_residual_W
        accepted: tuple[list[float], _ChainEvaluation] | None = None
        while damping >= _MIN_LINE_SEARCH_DAMPING:
            trial_temperatures = [
                temperature + damping * change
                for temperature, change in zip(temperatures, delta)
            ]
            if not all(
                low_temperature - temperature_tolerance_K
                <= value
                <= high_temperature + temperature_tolerance_K
                for value in trial_temperatures
            ):
                damping *= 0.5
                continue
            trial = _evaluate_chain(
                chain,
                trial_temperatures,
                gas_bulk_temperature=gas_bulk_temperature,
                outside_alpha_physical=outside_alpha_physical,
                cp_gas=cp_gas,
                W_bulk=W_bulk,
                p_total=p_total,
                M_dry=M_dry,
                M_h2o=M_h2o,
                lewis_number=lewis_number,
                m_dot_water_vapor_available=m_dot_water_vapor_available,
            )
            if (
                trial.equation_residual_W < current_norm
                or trial.equation_residual_W <= heat_tolerance
            ):
                accepted = trial_temperatures, trial
                break
            damping *= 0.5

        if accepted is None:
            residuals["line_search_damping"] = damping
            raise WetFinConvergenceError(
                "Wet annular-fin Newton line search could not reduce the "
                f"residual at iteration {iteration}; residuals={residuals}.",
                iterations=iteration,
                residuals=residuals,
            )

        new_temperatures, new_evaluation = accepted
        temperature_step = max(
            abs(new - old)
            for new, old in zip(new_temperatures, temperatures)
        )
        mass_step = abs(
            new_evaluation.m_dot_condensate - current.m_dot_condensate
        )
        energy_error = _chain_energy_balance_error(
            chain, new_temperatures, new_evaluation
        )
        residuals = {
            "temperature_step_K": temperature_step,
            "equation_residual_W": new_evaluation.equation_residual_W,
            "energy_balance_W": energy_error,
            "condensate_step_kg_s": mass_step,
            "availability_scale": new_evaluation.availability_scale,
            "line_search_damping": damping,
        }
        temperatures = new_temperatures
        previous_mass = current.m_dot_condensate
        heat_tolerance = max(
            relative_heat_tolerance
            * max(q_scale, abs(new_evaluation.Q_total), 1.0e-12),
            _jacobian_roundoff_floor(new_evaluation, new_temperatures),
        )
        mass_tolerance = max(
            condensate_tolerance_kg_s,
            relative_heat_tolerance * mass_scale,
        )
        if (
            temperature_step <= temperature_tolerance_K
            and new_evaluation.equation_residual_W <= heat_tolerance
            and energy_error <= heat_tolerance
            and mass_step <= mass_tolerance
        ):
            return temperatures, new_evaluation, iteration, residuals

    raise WetFinConvergenceError(
        "Wet annular-fin solver did not converge within "
        f"max_iterations={max_iterations}; residuals={residuals}.",
        iterations=max_iterations,
        residuals=residuals,
    )


def _evaluate_chain(
    chain: _Chain,
    temperatures: list[float],
    *,
    gas_bulk_temperature: float,
    outside_alpha_physical: float,
    cp_gas: float,
    W_bulk: float,
    p_total: float,
    M_dry: float,
    M_h2o: float,
    lewis_number: float,
    m_dot_water_vapor_available: float,
) -> _ChainEvaluation:
    count = len(temperatures)
    raw_fluxes = [
        _raw_condensation_flux(
            temperature,
            outside_alpha_physical=outside_alpha_physical,
            cp_gas=cp_gas,
            W_bulk=W_bulk,
            p_total=p_total,
            M_dry=M_dry,
            M_h2o=M_h2o,
            lewis_number=lewis_number,
        )
        if chain.surface_areas[index] > 0.0
        else 0.0
        for index, temperature in enumerate(temperatures)
    ]
    raw_mass_rate = sum(
        area * flux for area, flux in zip(chain.surface_areas, raw_fluxes)
    )
    if raw_mass_rate <= 0.0:
        availability_scale = 1.0
    elif math.isinf(m_dot_water_vapor_available):
        availability_scale = 1.0
    else:
        availability_scale = min(
            1.0, m_dot_water_vapor_available / raw_mass_rate
        )

    transfers: list[_TransferEvaluation] = []
    derivatives: list[float] = []
    for index, temperature in enumerate(temperatures):
        area = chain.surface_areas[index]
        if area <= 0.0:
            transfers.append(_zero_transfer())
            derivatives.append(0.0)
            continue
        transfer = _transfer_at_temperature(
            temperature,
            gas_bulk_temperature=gas_bulk_temperature,
            outside_alpha_physical=outside_alpha_physical,
            cp_gas=cp_gas,
            W_bulk=W_bulk,
            p_total=p_total,
            M_dry=M_dry,
            M_h2o=M_h2o,
            lewis_number=lewis_number,
            availability_scale=availability_scale,
            raw_mass_flux=raw_fluxes[index],
        )
        transfers.append(transfer)
        derivatives.append(
            _total_flux_temperature_derivative(
                temperature,
                gas_bulk_temperature=gas_bulk_temperature,
                outside_alpha_physical=outside_alpha_physical,
                cp_gas=cp_gas,
                W_bulk=W_bulk,
                p_total=p_total,
                M_dry=M_dry,
                M_h2o=M_h2o,
                lewis_number=lewis_number,
                availability_scale=availability_scale,
            )
        )

    residual = [0.0] * count
    lower = [0.0] * count
    diagonal = [0.0] * count
    upper = [0.0] * count
    for index in range(count):
        area = chain.surface_areas[index]
        residual[index] -= area * transfers[index].total_flux
        diagonal[index] -= area * derivatives[index]
        boundary_conductance = chain.boundary_conductances[index]
        if boundary_conductance > 0.0:
            residual[index] += boundary_conductance * (
                temperatures[index] - chain.boundary_temperatures[index]
            )
            diagonal[index] += boundary_conductance
        if index > 0:
            conductance = chain.edges[index - 1]
            residual[index] += conductance * (
                temperatures[index] - temperatures[index - 1]
            )
            diagonal[index] += conductance
            lower[index] = -conductance
        if index < count - 1:
            conductance = chain.edges[index]
            residual[index] += conductance * (
                temperatures[index] - temperatures[index + 1]
            )
            diagonal[index] += conductance
            upper[index] = -conductance

    if any(
        not math.isfinite(value) or value <= 0.0 for value in diagonal
    ):
        raise ValueError("Wet annular-fin Newton Jacobian has a non-positive diagonal.")
    Q_sensible = sum(
        area * transfer.sensible_flux
        for area, transfer in zip(chain.surface_areas, transfers)
    )
    Q_latent = sum(
        area * transfer.latent_flux
        for area, transfer in zip(chain.surface_areas, transfers)
    )
    m_dot = sum(
        area * transfer.mass_flux
        for area, transfer in zip(chain.surface_areas, transfers)
    )
    H_condensate = sum(
        area * transfer.condensate_enthalpy_flux
        for area, transfer in zip(chain.surface_areas, transfers)
    )
    return _ChainEvaluation(
        residual=tuple(residual),
        lower=tuple(lower),
        diagonal=tuple(diagonal),
        upper=tuple(upper),
        transfers=tuple(transfers),
        availability_scale=availability_scale,
        Q_sensible=Q_sensible,
        Q_latent=Q_latent,
        Q_total=Q_sensible + Q_latent,
        m_dot_condensate=m_dot,
        condensate_enthalpy_rate=H_condensate,
        equation_residual_W=max(abs(value) for value in residual),
    )


def _transfer_at_temperature(
    temperature: float,
    *,
    gas_bulk_temperature: float,
    outside_alpha_physical: float,
    cp_gas: float,
    W_bulk: float,
    p_total: float,
    M_dry: float,
    M_h2o: float,
    lewis_number: float,
    availability_scale: float,
    raw_mass_flux: float | None = None,
) -> _TransferEvaluation:
    if raw_mass_flux is None:
        raw_mass_flux = _raw_condensation_flux(
            temperature,
            outside_alpha_physical=outside_alpha_physical,
            cp_gas=cp_gas,
            W_bulk=W_bulk,
            p_total=p_total,
            M_dry=M_dry,
            M_h2o=M_h2o,
            lewis_number=lewis_number,
        )
    sensible_flux = outside_alpha_physical * (
        gas_bulk_temperature - temperature
    )
    if raw_mass_flux <= 0.0 or availability_scale <= 0.0:
        W_sat = (
            None
            if W_bulk <= 0.0
            else _saturated_ratio_if_gas_phase_exists(
                p_total=p_total,
                temperature=temperature,
                M_dry=M_dry,
                M_h2o=M_h2o,
            )
        )
        return _TransferEvaluation(
            raw_mass_flux=raw_mass_flux,
            mass_flux=0.0,
            sensible_flux=sensible_flux,
            latent_flux=0.0,
            total_flux=sensible_flux,
            condensate_enthalpy_flux=0.0,
            W_sat=W_sat,
        )
    mass_flux = availability_scale * raw_mass_flux
    latent_flux = mass_flux * _latent_heat_at_temperature(temperature)
    condensate_enthalpy_flux = mass_flux * _liquid_enthalpy_at_temperature(
        temperature
    )
    return _TransferEvaluation(
        raw_mass_flux=raw_mass_flux,
        mass_flux=mass_flux,
        sensible_flux=sensible_flux,
        latent_flux=latent_flux,
        total_flux=sensible_flux + latent_flux,
        condensate_enthalpy_flux=condensate_enthalpy_flux,
        W_sat=_saturated_ratio_if_gas_phase_exists(
            p_total=p_total,
            temperature=temperature,
            M_dry=M_dry,
            M_h2o=M_h2o,
        ),
    )


def _raw_condensation_flux(
    temperature: float,
    *,
    outside_alpha_physical: float,
    cp_gas: float,
    W_bulk: float,
    p_total: float,
    M_dry: float,
    M_h2o: float,
    lewis_number: float,
) -> float:
    if W_bulk <= 0.0:
        return 0.0
    if temperature <= WATER_TRIPLE_POINT_TEMPERATURE_K:
        raise FrostingNotSupportedError(
            "wet_finned_surface: an exposed condensing surface reached "
            f"{temperature:.3f} K, at/below the water triple point."
        )
    W_sat = _saturated_ratio_if_gas_phase_exists(
        p_total=p_total,
        temperature=temperature,
        M_dry=M_dry,
        M_h2o=M_h2o,
    )
    # Above water's boiling temperature at this total pressure there is no
    # finite saturated wet-gas ratio with non-condensable headroom.  The
    # local surface is unambiguously dry, rather than a property error.
    if W_sat is None:
        return 0.0
    # SI dry-carrier basis [kg H2O/(m2 s)], using KalKalori's existing
    # Chilton-Colburn convention.  The coupled sensible/latent wet-fin source
    # follows Sharqawy, Moinuddin & Zubair (2012),
    # DOI 10.1016/j.ijrefrig.2011.11.004.
    return condensation_mass_flux(
        alfa_dry=outside_alpha_physical,
        cp_gas=cp_gas,
        W_bulk=W_bulk,
        W_sat_surface=W_sat,
        lewis_number=lewis_number,
    )


def _saturated_ratio_if_gas_phase_exists(
    *,
    p_total: float,
    temperature: float,
    M_dry: float,
    M_h2o: float,
) -> float | None:
    """Interpolate authoritative W_sat values on a deterministic 0.25 K grid."""

    lower, upper, fraction = _property_grid_bracket(temperature)
    lower_value = _saturated_ratio_grid_node(
        p_total, lower, M_dry, M_h2o
    )
    upper_value = _saturated_ratio_grid_node(
        p_total, upper, M_dry, M_h2o
    )
    if lower_value is None or upper_value is None:
        # A bracket that straddles the boiling/no-headroom boundary is kept
        # exact so interpolation never advances the dry boundary by one grid
        # interval.
        return _saturated_ratio_exact(p_total, temperature, M_dry, M_h2o)
    return lower_value + fraction * (upper_value - lower_value)


@lru_cache(maxsize=32768)
def _saturated_ratio_grid_node(
    p_total: float,
    temperature: float,
    M_dry: float,
    M_h2o: float,
) -> float | None:
    return _saturated_ratio_exact(p_total, temperature, M_dry, M_h2o)


def _saturated_ratio_exact(
    p_total: float,
    temperature: float,
    M_dry: float,
    M_h2o: float,
) -> float | None:
    """Return exact project W_sat, or None only for p_sat >= p_total."""

    try:
        return saturated_water_ratio(
            p_total=p_total,
            T=temperature,
            M_dry=M_dry,
            M_h2o=M_h2o,
        )
    except ValueError as exc:
        # Keep unrelated equilibrium/input errors explicit.  This exact
        # helper diagnostic identifies the expected high-temperature case.
        if "no saturated gas-phase state exists" in str(exc):
            return None
        raise


@lru_cache(maxsize=32768)
def _latent_heat_grid_node(temperature: float) -> float:
    return water_latent_heat_of_vaporization(T=temperature)


@lru_cache(maxsize=32768)
def _liquid_enthalpy_grid_node(temperature: float) -> float:
    return water_saturation_liquid_enthalpy(T=temperature)


def _latent_heat_at_temperature(temperature: float) -> float:
    lower, upper, fraction = _property_grid_bracket(temperature)
    lower_value = _latent_heat_grid_node(lower)
    upper_value = _latent_heat_grid_node(upper)
    return lower_value + fraction * (upper_value - lower_value)


def _liquid_enthalpy_at_temperature(temperature: float) -> float:
    lower, upper, fraction = _property_grid_bracket(temperature)
    lower_value = _liquid_enthalpy_grid_node(lower)
    upper_value = _liquid_enthalpy_grid_node(upper)
    return lower_value + fraction * (upper_value - lower_value)


def _property_grid_bracket(temperature: float) -> tuple[float, float, float]:
    step = _WATER_PROPERTY_INTERPOLATION_STEP_K
    lower = step * math.floor(temperature / step)
    if lower < WATER_TRIPLE_POINT_TEMPERATURE_K:
        lower = WATER_TRIPLE_POINT_TEMPERATURE_K
    upper = lower + step
    fraction = (temperature - lower) / (upper - lower)
    return lower, upper, fraction


def _total_flux_temperature_derivative(
    temperature: float,
    *,
    gas_bulk_temperature: float,
    outside_alpha_physical: float,
    cp_gas: float,
    W_bulk: float,
    p_total: float,
    M_dry: float,
    M_h2o: float,
    lewis_number: float,
    availability_scale: float,
) -> float:
    step = max(
        _DERIVATIVE_STEP_MIN_K,
        _DERIVATIVE_STEP_RELATIVE * abs(temperature),
    )
    lower_temperature = temperature - step
    if lower_temperature <= WATER_TRIPLE_POINT_TEMPERATURE_K:
        lower_temperature = temperature
    upper_temperature = temperature + step

    def total_flux(value: float) -> float:
        raw_mass_flux = _raw_condensation_flux(
            value,
            outside_alpha_physical=outside_alpha_physical,
            cp_gas=cp_gas,
            W_bulk=W_bulk,
            p_total=p_total,
            M_dry=M_dry,
            M_h2o=M_h2o,
            lewis_number=lewis_number,
        )
        latent_flux = (
            availability_scale
            * raw_mass_flux
            * _latent_heat_at_temperature(value)
            if raw_mass_flux > 0.0 and availability_scale > 0.0
            else 0.0
        )
        return (
            outside_alpha_physical * (gas_bulk_temperature - value)
            + latent_flux
        )

    if lower_temperature == temperature:
        return (total_flux(upper_temperature) - total_flux(temperature)) / step
    return (
        total_flux(upper_temperature) - total_flux(lower_temperature)
    ) / (upper_temperature - lower_temperature)


def _build_annular_fin_result(
    *,
    tube: CircularFinnedTube,
    chain: _Chain,
    temperatures: list[float],
    evaluation: _ChainEvaluation,
    fin_base_temperature: float,
    gas_bulk_temperature: float,
    outside_alpha_physical: float,
    cp_gas: float,
    W_bulk: float,
    p_total: float,
    M_dry: float,
    M_h2o: float,
    lewis_number: float,
    iterations: int,
    residuals: Mapping[str, float],
) -> WetAnnularFinResult:
    mesh = chain.mesh
    scale = chain.fin_area_scale
    if scale <= 0.0 or chain.fin_tip_index is None:
        raise ValueError("Cannot build an annular-fin result for zero fin area.")
    fin_indices = chain.fin_surface_indices
    per_fin_areas = [chain.surface_areas[index] / scale for index in fin_indices]
    fin_transfers = [evaluation.transfers[index] for index in fin_indices]
    Q_sensible = sum(
        area * transfer.sensible_flux
        for area, transfer in zip(per_fin_areas, fin_transfers)
    )
    Q_latent = sum(
        area * transfer.latent_flux
        for area, transfer in zip(per_fin_areas, fin_transfers)
    )
    m_dot = sum(
        area * transfer.mass_flux
        for area, transfer in zip(per_fin_areas, fin_transfers)
    )
    H_condensate = sum(
        area * transfer.condensate_enthalpy_flux
        for area, transfer in zip(per_fin_areas, fin_transfers)
    )
    cell_temperatures = tuple(
        temperatures[index] for index in chain.fin_cell_indices
    )
    tip_temperature = temperatures[chain.fin_tip_index]
    cell_fluxes = tuple(
        evaluation.transfers[index].mass_flux
        for index in chain.fin_cell_indices
    )
    tip_flux = evaluation.transfers[chain.fin_tip_index].mass_flux
    state, wet_area, boundary = _classify_fin_wet_state(
        tube=tube,
        mesh=mesh,
        fin_base_temperature=fin_base_temperature,
        cell_temperatures=cell_temperatures,
        tip_temperature=tip_temperature,
        outside_alpha_physical=outside_alpha_physical,
        cp_gas=cp_gas,
        W_bulk=W_bulk,
        p_total=p_total,
        M_dry=M_dry,
        M_h2o=M_h2o,
        lewis_number=lewis_number,
    )
    dry_area = tube.fin_area_per_fin - wet_area
    root_heat_rate = mesh.conductance_root * (
        cell_temperatures[0] - fin_base_temperature
    )
    fin_energy_error = abs(root_heat_rate - (Q_sensible + Q_latent))
    fin_residuals = dict(residuals)
    fin_residuals["fin_root_energy_balance_W"] = fin_energy_error
    return WetAnnularFinResult(
        fin_wet_state=state,
        fin_wet_fraction=wet_area / tube.fin_area_per_fin,
        wet_fin_area=wet_area,
        dry_fin_area=dry_area,
        Q_fin_sensible=Q_sensible,
        Q_fin_latent=Q_latent,
        Q_fin_total=Q_sensible + Q_latent,
        m_dot_condensate_fin=m_dot,
        condensate_enthalpy_rate_fin=H_condensate,
        fin_base_temperature=fin_base_temperature,
        fin_tip_temperature=tip_temperature,
        wet_dry_boundary_radius=boundary,
        radial_cell_centers=mesh.cell_centers,
        radial_cell_temperatures=cell_temperatures,
        radial_cell_side_areas=mesh.side_areas,
        radial_cell_condensate_fluxes=cell_fluxes,
        tip_condensate_flux=tip_flux,
        outside_alpha_physical=outside_alpha_physical,
        mass_transfer_coefficient=mass_transfer_coefficient(
            outside_alpha_physical,
            cp_gas,
            lewis_number=lewis_number,
        ),
        radial_cells=len(mesh.cell_centers),
        iterations=iterations,
        residuals=fin_residuals,
    )


def _classify_fin_wet_state(
    *,
    tube: CircularFinnedTube,
    mesh: _FinMesh,
    fin_base_temperature: float,
    cell_temperatures: tuple[float, ...],
    tip_temperature: float,
    outside_alpha_physical: float,
    cp_gas: float,
    W_bulk: float,
    p_total: float,
    M_dry: float,
    M_h2o: float,
    lewis_number: float,
) -> tuple[WetFinState, float, float | None]:
    del outside_alpha_physical, cp_gas, lewis_number
    if W_bulk <= 0.0:
        return WetFinState.DRY, 0.0, None

    radii = (mesh.r_root, *mesh.cell_centers, mesh.r_tip)
    temperatures = (fin_base_temperature, *cell_temperatures, tip_temperature)
    drives = tuple(
        (
            W_bulk - W_sat
            if (
                W_sat := _saturated_ratio_if_gas_phase_exists(
                    p_total=p_total,
                    temperature=temperature,
                    M_dry=M_dry,
                    M_h2o=M_h2o,
                )
            )
            is not None
            else -math.inf
        )
        for temperature in temperatures
    )
    if not any(drive > 0.0 for drive in drives):
        return WetFinState.DRY, 0.0, None
    if drives[-1] > 0.0:
        return WetFinState.FULLY_WET, tube.fin_area_per_fin, None
    if drives[0] <= 0.0:
        # A wet island away from a dry root is non-physical for the supported
        # outside-hotter radial boundary.  Do not report an ambiguous radius.
        raise ValueError("Wet annular-fin solution has a dry base and wet outer cells.")

    boundary: float | None = None
    for left in range(len(drives) - 1):
        if drives[left] > 0.0 and drives[left + 1] <= 0.0:
            # The reported radius is a linear interpolation of the humidity-
            # ratio driving force between adjacent radial temperature nodes;
            # wet face area is then integrated analytically to that radius.
            # See the partial-wet annular-fin boundary convention in
            # Sharqawy & Zubair (2007), DOI 10.1016/j.ijrefrig.2006.12.008.
            denominator = drives[left] - drives[left + 1]
            fraction = drives[left] / denominator if denominator > 0.0 else 0.0
            boundary = radii[left] + fraction * (
                radii[left + 1] - radii[left]
            )
            break
    if boundary is None:
        raise ValueError("Wet annular-fin solution has no unique wet/dry crossing.")
    wet_area = (
        2.0
        * math.pi
        * tube.fin_side_slope_factor
        * (boundary * boundary - mesh.r_root * mesh.r_root)
    )
    wet_area = min(max(wet_area, 0.0), tube.fin_both_sides_area_per_fin)
    return WetFinState.PARTIALLY_WET, wet_area, boundary


def _chain_energy_balance_error(
    chain: _Chain,
    temperatures: list[float],
    evaluation: _ChainEvaluation,
) -> float:
    boundary_heat_removed = sum(
        conductance * (temperature - boundary_temperature)
        for conductance, boundary_temperature, temperature in zip(
            chain.boundary_conductances,
            chain.boundary_temperatures,
            temperatures,
        )
        if conductance > 0.0
    )
    return abs(boundary_heat_removed - evaluation.Q_total)


def _heat_scale(alpha: float, area: float, temperature_span: float) -> float:
    return max(alpha * area * max(abs(temperature_span), 1.0e-6), 1.0e-12)


def _jacobian_roundoff_floor(
    evaluation: _ChainEvaluation,
    temperatures: list[float],
) -> float:
    """Scale the heat residual floor for high-conductivity cancellation."""

    return (
        128.0
        * math.ulp(1.0)
        * max(evaluation.diagonal)
        * max(max(abs(value) for value in temperatures), 1.0)
    )


def _linear_thickness(tube: CircularFinnedTube, radius: float) -> float:
    r_root = 0.5 * tube.D_root
    fraction = (radius - r_root) / tube.fin_height
    return tube.fin_thickness_root + fraction * (
        tube.fin_thickness_tip_effective - tube.fin_thickness_root
    )


def _radial_conduction_area(tube: CircularFinnedTube, radius: float) -> float:
    return 2.0 * math.pi * radius * _linear_thickness(tube, radius)


def _solve_tridiagonal(
    lower: list[float],
    diagonal: list[float],
    upper: list[float],
    rhs: list[float],
) -> list[float]:
    count = len(diagonal)
    if count == 0 or not (
        len(lower) == len(upper) == len(rhs) == count
    ):
        raise ValueError("Invalid wet-fin tridiagonal system dimensions.")
    c_prime = [0.0] * count
    d_prime = [0.0] * count
    pivot = diagonal[0]
    if not math.isfinite(pivot) or pivot <= 0.0:
        raise ValueError("Invalid first pivot in wet-fin tridiagonal solver.")
    c_prime[0] = upper[0] / pivot
    d_prime[0] = rhs[0] / pivot
    for index in range(1, count):
        pivot = diagonal[index] - lower[index] * c_prime[index - 1]
        if not math.isfinite(pivot) or pivot <= 0.0:
            raise ValueError("Non-positive pivot in wet-fin tridiagonal solver.")
        c_prime[index] = upper[index] / pivot if index < count - 1 else 0.0
        d_prime[index] = (
            rhs[index] - lower[index] * d_prime[index - 1]
        ) / pivot
    result = [0.0] * count
    result[-1] = d_prime[-1]
    for index in range(count - 2, -1, -1):
        result[index] = d_prime[index] - c_prime[index] * result[index + 1]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("Wet-fin tridiagonal solution is non-finite.")
    return result


def _zero_transfer() -> _TransferEvaluation:
    return _TransferEvaluation(
        raw_mass_flux=0.0,
        mass_flux=0.0,
        sensible_flux=0.0,
        latent_flux=0.0,
        total_flux=0.0,
        condensate_enthalpy_flux=0.0,
        W_sat=None,
    )


def _validate_common_inputs(
    *,
    tube: CircularFinnedTube,
    gas_bulk_temperature: float,
    outside_alpha_physical: float,
    cp_gas: float,
    W_bulk: float,
    p_total: float,
    M_dry: float,
    M_h2o: float,
    lewis_number: float,
    radial_cells: int,
    max_iterations: int,
    temperature_tolerance_K: float,
    relative_heat_tolerance: float,
    condensate_tolerance_kg_s: float,
    relaxation_factor: float,
) -> None:
    if not isinstance(tube, CircularFinnedTube):
        raise TypeError("tube must be a CircularFinnedTube instance.")
    _validate_temperature(gas_bulk_temperature, "gas_bulk_temperature")
    for name, value in (
        ("outside_alpha_physical", outside_alpha_physical),
        ("cp_gas", cp_gas),
        ("p_total", p_total),
        ("M_dry", M_dry),
        ("M_h2o", M_h2o),
        ("lewis_number", lewis_number),
        ("temperature_tolerance_K", temperature_tolerance_K),
        ("relative_heat_tolerance", relative_heat_tolerance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    if not math.isfinite(W_bulk) or W_bulk < 0.0:
        raise ValueError("W_bulk must be finite and non-negative.")
    if (
        not math.isfinite(condensate_tolerance_kg_s)
        or condensate_tolerance_kg_s < 0.0
    ):
        raise ValueError("condensate_tolerance_kg_s must be finite and non-negative.")
    if (
        isinstance(radial_cells, bool)
        or not isinstance(radial_cells, int)
        or radial_cells < 4
    ):
        raise ValueError("radial_cells must be an integer >= 4.")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if not math.isfinite(relaxation_factor) or not (0.0 < relaxation_factor <= 1.0):
        raise ValueError("relaxation_factor must be in (0, 1].")


def _validate_temperature(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite absolute temperature.")


def _validate_network_areas(
    bundle: TubeBundle,
    network: ThermalResistanceNetwork,
) -> None:
    for name, actual, expected in (
        ("area_primary_outside", network.area_primary_outside, bundle.total_primary_outside_area),
        ("area_fin", network.area_fin, bundle.total_fin_area),
        ("area_outside_gross", network.area_outside_gross, bundle.total_outer_area),
    ):
        if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError(
                f"network.{name} does not match the supplied bundle ({actual!r} != {expected!r})."
            )
    if not math.isclose(
        network.area_primary_outside + network.area_fin,
        network.area_outside_gross,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("Primary and fin areas do not close the gross outside area.")


__all__ = [
    "DEFAULT_WET_FIN_RADIAL_CELLS",
    "WetFinState",
    "WetFinConvergenceError",
    "WetAnnularFinResult",
    "WetFinnedSurfaceResult",
    "solve_wet_annular_fin",
    "solve_wet_finned_surface",
]
