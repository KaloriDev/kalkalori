# KalKalori - Heat Exchanger Open Engine
# GNU GPL v3 only

"""Surface-aware dry outside-side dispatch and resistance bookkeeping.

This module is the integration boundary between exchanger workflows and the
two deliberately separate outside-correlation families.  A plain tube can
only reach the existing Zukauskas/Euler path; a circular-finned tube can only
reach Briggs-Young/Robinson-Briggs.  Keeping that decision in one place
prevents a workflow refresh from silently falling back to a bare-tube model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from core.common.warnings import ModelWarning
from core.geometry.bundle import TubeBundle
from core.geometry.finned_tube import CircularFinnedTube
from core.geometry.tube import TubeSurfaceType
from core.heat_transfer.fin_efficiency import FinEfficiencyResult, annular_fin_efficiency
from core.heat_transfer.finned_tube_outside import (
    BriggsYoung1963Provider,
    FinnedTubeHeatTransferProvider,
    FinnedTubeHeatTransferResult,
    calculate_finned_tube_outside_heat_transfer,
)
from core.heat_transfer.outside_flow import (
    OutsideHydraulicPoint,
    OutsideTubeBankHydraulicResult,
    calculate_outside_tube_bank_hydraulics,
    nusselt_zukauskas,
    outside_flow_from_mass_flow,
)
from core.pressure_drop.finned_tube_pressure_drop import (
    FinnedTubeBankHydraulicResult,
    FinnedTubeHydraulicPoint,
    FinnedTubePressureDropProvider,
    RobinsonBriggs1966Provider,
    calculate_finned_tube_bank_hydraulics,
)
from core.pressure_drop.outside_pressure_drop import EulerProvider


DEFAULT_BARE_EULER_PROVIDER = "zukauskas"
DEFAULT_FINNED_HT_PROVIDER = "briggs_young_1963"
DEFAULT_FINNED_DP_PROVIDER = "robinson_briggs_1966"


class IncompatibleOutsideCorrelationProviderError(TypeError):
    """A provider from one surface family was supplied to the other family."""


@dataclass(frozen=True)
class OutsideThermalDispatchResult:
    """Physical outside-film result selected by the surface dispatcher.

    ``alpha_physical`` is the mean film coefficient returned directly by the
    selected correlation.  It intentionally does *not* include extended
    surface efficiency, contact resistance, area enhancement, or root-layer
    conduction.  Those geometry/topology effects are applied only by
    :func:`calculate_resistance_network`, which produces the separately
    named gross-area effective coefficient.
    """

    surface_type: TubeSurfaceType
    face_velocity: float
    reference_velocity: float
    reynolds_number: float
    prandtl_number: float
    nusselt_number: float
    nusselt_number_base: float
    wall_property_correction: float
    alpha_physical: float
    warnings: tuple[ModelWarning, ...]
    finned_result: FinnedTubeHeatTransferResult | None = None

    @property
    def alpha(self) -> float:
        """Compatibility alias for the physical outside-film coefficient."""
        return self.alpha_physical

    @property
    def outside_alpha_physical(self) -> float:
        """Fully qualified alias for :attr:`alpha_physical`."""
        return self.alpha_physical

    @property
    def outside_htc(self) -> float:
        """Compatibility alias for the physical outside-film coefficient."""
        return self.alpha_physical


@dataclass(frozen=True)
class ThermalResistanceNetwork:
    """Whole-exchanger resistance network on an authoritative gross-area basis.

    ``outside_alpha_physical`` is the film coefficient supplied to the fin
    model.  ``outside_alpha_effective_gross`` is derived *after* the complete
    outside topology is assembled and therefore satisfies
    ``resistance_outside == 1 / (outside_alpha_effective_gross *
    area_outside_gross)``.  In particular, finite fin contact and a continuous
    root layer are part of ``resistance_outside`` rather than hidden in the
    total-resistance sum.

    ``contact_input_mode`` records which of the two mutually exclusive
    contact inputs was resolved (``"explicit_resistance"``,
    ``"contact_efficiency"`` or ``"ideal_default"``; ``"not_applicable"`` for
    a plain tube). ``resistance_contact`` is always the actual absolute
    [K/W] contact resistance used by the network, however it was resolved.
    ``contact_resistance_used`` is the corresponding *areal* [m2 K/W]
    equivalent actually used by this network -- for explicit-resistance mode
    it equals the supplied ``fin_contact_resistance`` exactly;  for
    efficiency mode it is the operating-point-derived equivalent; for the
    ideal default it is ``0.0``. ``contact_resistance_equivalent_areal`` is
    an explicitly-named alias of the same resolved value.
    ``fin_contact_efficiency_effective`` is the actual contact-affected-path
    conductance ratio implied by ``resistance_contact`` (``1.0`` for ideal
    contact); for efficiency mode it reproduces
    ``fin_contact_efficiency_input`` to numerical tolerance.
    """

    surface_type: TubeSurfaceType
    area_inside: float
    area_primary_outside: float
    area_fin: float
    area_outside_gross: float
    area_outside_geometric: float
    area_outside_effective: float
    fin_efficiency: float
    overall_surface_efficiency: float
    outside_alpha_physical: float
    outside_alpha_effective_gross: float
    resistance_inside: float
    resistance_core_wall: float
    resistance_root: float
    contact_input_mode: str
    fin_contact_efficiency_input: float | None
    fin_contact_efficiency_effective: float
    contact_resistance_equivalent_areal: float
    contact_resistance_used: float
    resistance_contact: float
    contact_area: float
    contact_area_basis: str
    contact_topology: str
    conductance_primary_outside: float
    conductance_fin_outside: float
    resistance_outside_branches: float
    resistance_outside: float
    resistance_total: float
    UA: float
    U_gross_outside: float
    fin_efficiency_result: FinEfficiencyResult | None = None


@dataclass(frozen=True)
class FinnedTubeDiagnostics:
    """Public, auditable circular-finned-tube thermal/hydraulic result.

    ``outside_alpha_physical`` is the direct Briggs-Young film HTC.  The
    separately named ``outside_alpha_effective_gross`` is the generic
    gross-area coefficient reconstructed from the full outside resistance,
    including the declared contact/root topology.

    ``contact_input_mode``, ``fin_contact_efficiency_input``,
    ``fin_contact_efficiency_effective`` and
    ``contact_resistance_equivalent_areal`` audit which contact input was
    resolved and its operating-point equivalent; see
    ``ThermalResistanceNetwork`` for their exact definitions.
    """

    tube_surface_type: TubeSurfaceType
    area_inside: float
    area_primary_outside: float
    area_fin: float
    area_outside_gross: float
    area_outside_geometric: float
    area_outside_effective: float
    fin_efficiency: float
    overall_surface_efficiency: float
    outside_alpha_physical: float
    outside_alpha_effective_gross: float
    resistance_inside: float
    resistance_core_wall: float
    resistance_root: float
    contact_input_mode: str
    fin_contact_efficiency_input: float | None
    fin_contact_efficiency_effective: float
    contact_resistance_equivalent_areal: float
    contact_resistance_used: float
    resistance_contact: float
    contact_area: float
    contact_area_basis: str
    contact_topology: str
    conductance_primary_outside: float
    conductance_fin_outside: float
    resistance_outside_branches: float
    resistance_outside: float
    resistance_total: float
    UA: float
    U_gross_outside: float
    fin_tip_temperature_ratio: float
    face_velocity: float
    reference_velocity: float
    reynolds_number: float
    prandtl_number: float
    nusselt_number: float
    colburn_j_factor: float
    pressure_drop_coefficient: float
    pressure_drop_coefficient_definition: str
    outside_dp_drag: float
    outside_dp_acceleration: float
    outside_dp_total: float
    heat_transfer_metadata: object | None
    pressure_drop_metadata: object | None
    warnings: tuple[ModelWarning, ...]

    @property
    def alpha(self) -> float:
        """Compatibility alias for the physical outside-film coefficient."""
        return self.outside_alpha_physical

    @property
    def alpha_physical(self) -> float:
        """Short alias for :attr:`outside_alpha_physical`."""
        return self.outside_alpha_physical

    @property
    def outside_htc(self) -> float:
        """Compatibility alias for the physical outside-film coefficient."""
        return self.outside_alpha_physical

    @property
    def A_i(self) -> float:
        return self.area_inside

    @property
    def A_inside(self) -> float:
        return self.area_inside

    @property
    def A_primary(self) -> float:
        return self.area_primary_outside

    @property
    def A_primary_outside(self) -> float:
        return self.area_primary_outside

    @property
    def A_fin(self) -> float:
        return self.area_fin

    @property
    def A_o(self) -> float:
        return self.area_outside_gross

    @property
    def A_outside_gross(self) -> float:
        return self.area_outside_gross

    @property
    def A_outside_geometric(self) -> float:
        return self.area_outside_geometric

    @property
    def A_effective(self) -> float:
        return self.area_outside_effective

    @property
    def A_outside_effective(self) -> float:
        return self.area_outside_effective

    @property
    def eta_fin(self) -> float:
        return self.fin_efficiency

    @property
    def eta_overall(self) -> float:
        return self.overall_surface_efficiency

    @property
    def U(self) -> float:
        return self.U_gross_outside

    @property
    def Re(self) -> float:
        return self.reynolds_number

    @property
    def Nu(self) -> float:
        return self.nusselt_number

    @property
    def j(self) -> float:
        return self.colburn_j_factor

    @property
    def f(self) -> float:
        return self.pressure_drop_coefficient

    @property
    def dp(self) -> float:
        return self.outside_dp_total


OutsideBankHydraulicResult = OutsideTubeBankHydraulicResult | FinnedTubeBankHydraulicResult
OutsideBankHydraulicPoint = OutsideHydraulicPoint | FinnedTubeHydraulicPoint


def tube_surface_type(bundle: TubeBundle) -> TubeSurfaceType:
    """Resolve and validate the explicit outside-surface discriminator."""

    value = getattr(bundle.tube, "surface_type", None)
    if value is TubeSurfaceType.PLAIN:
        return TubeSurfaceType.PLAIN
    if value is TubeSurfaceType.CIRCULAR_FINNED:
        if not isinstance(bundle.tube, CircularFinnedTube):
            raise TypeError(
                "TubeSurfaceType.CIRCULAR_FINNED requires CircularFinnedTube geometry."
            )
        return TubeSurfaceType.CIRCULAR_FINNED
    raise TypeError(
        "Tube geometry must expose a supported TubeSurfaceType; supported values "
        "are PLAIN and CIRCULAR_FINNED."
    )


def evaluate_outside_thermal(
    *,
    bundle: TubeBundle,
    m_dot: float,
    props: Any,
    Pr_s: float | None = None,
    euler_provider: str | EulerProvider = DEFAULT_BARE_EULER_PROVIDER,
    finned_heat_transfer_provider: (
        str | FinnedTubeHeatTransferProvider
    ) = DEFAULT_FINNED_HT_PROVIDER,
) -> OutsideThermalDispatchResult:
    """Evaluate the dry outside film coefficient for the declared surface."""

    surface = tube_surface_type(bundle)
    if surface is TubeSurfaceType.PLAIN:
        _reject_finned_provider_on_plain(
            finned_heat_transfer_provider,
            expected_default=DEFAULT_FINNED_HT_PROVIDER,
        )
        v, reynolds, prandtl, alpha, _, warnings, _ = outside_flow_from_mass_flow(
            m_dot=m_dot,
            frontal_area=bundle.frontal_flow_area,
            n_tubes_per_row=bundle.n_tubes_per_row,
            tube_outer_diameter=float(getattr(bundle.tube, "D_o")),
            tube_pitch_transverse=bundle.pitch_transverse,
            tube_pitch_longitudinal=bundle.pitch_longitudinal,
            layout=bundle.layout,
            n_rows=bundle.n_rows,
            props=props,
            Pr_s=Pr_s,
            euler_provider=euler_provider,
            calculate_pressure_drop=False,
        )
        nu_base = nusselt_zukauskas(reynolds, prandtl, bundle.n_rows, Pr_s=None)
        nu = (
            nusselt_zukauskas(reynolds, prandtl, bundle.n_rows, Pr_s=Pr_s)
            if Pr_s is not None
            else nu_base
        )
        return OutsideThermalDispatchResult(
            surface_type=surface,
            face_velocity=v,
            reference_velocity=v * _bare_reference_velocity_ratio(bundle),
            reynolds_number=reynolds,
            prandtl_number=prandtl,
            nusselt_number=nu,
            nusselt_number_base=nu_base,
            wall_property_correction=nu / nu_base if nu_base != 0.0 else 1.0,
            alpha_physical=alpha,
            warnings=tuple(warnings),
        )

    _reject_bare_provider_on_finned(euler_provider)
    if isinstance(finned_heat_transfer_provider, RobinsonBriggs1966Provider):
        raise IncompatibleOutsideCorrelationProviderError(
            "RobinsonBriggs1966Provider is a finned pressure-drop provider, "
            "not a finned heat-transfer provider."
        )
    result = calculate_finned_tube_outside_heat_transfer(
        m_dot,
        bundle,
        props,
        provider=finned_heat_transfer_provider,
    )
    return OutsideThermalDispatchResult(
        surface_type=surface,
        face_velocity=result.face_velocity,
        reference_velocity=result.reference_velocity,
        reynolds_number=result.reynolds_number,
        prandtl_number=result.prandtl_number,
        nusselt_number=result.nusselt_number,
        nusselt_number_base=result.nusselt_number,
        wall_property_correction=1.0,
        alpha_physical=result.alpha,
        warnings=result.warnings,
        finned_result=result,
    )


def evaluate_outside_hydraulics(
    *,
    bundle: TubeBundle,
    m_dot: float,
    property_provider: Any | None = None,
    temperature_in: float | None = None,
    temperature_out: float | None = None,
    pressure: float | None = None,
    inlet_props: Any | None = None,
    midpoint_props: Any | None = None,
    outlet_props: Any | None = None,
    euler_provider: str | EulerProvider = DEFAULT_BARE_EULER_PROVIDER,
    finned_pressure_drop_provider: (
        str | FinnedTubePressureDropProvider
    ) = DEFAULT_FINNED_DP_PROVIDER,
) -> OutsideBankHydraulicResult:
    """Evaluate the three-state outside hydraulics for the declared surface."""

    surface = tube_surface_type(bundle)
    if surface is TubeSurfaceType.PLAIN:
        _reject_finned_provider_on_plain(
            finned_pressure_drop_provider,
            expected_default=DEFAULT_FINNED_DP_PROVIDER,
        )
        return calculate_outside_tube_bank_hydraulics(
            m_dot=m_dot,
            face_area=bundle.frontal_flow_area,
            tube_outer_diameter=float(getattr(bundle.tube, "D_o")),
            tube_pitch_transverse=bundle.pitch_transverse,
            tube_pitch_longitudinal=bundle.pitch_longitudinal,
            layout=bundle.layout,
            n_rows=bundle.n_rows,
            n_tubes_per_row=bundle.n_tubes_per_row,
            provider=property_provider,
            temperature_in=temperature_in,
            temperature_out=temperature_out,
            pressure=pressure,
            euler_provider=euler_provider,
            inlet_props=inlet_props if property_provider is None else None,
            midpoint_props=midpoint_props,
            outlet_props=outlet_props,
        )

    _reject_bare_provider_on_finned(euler_provider)
    if isinstance(finned_pressure_drop_provider, BriggsYoung1963Provider):
        raise IncompatibleOutsideCorrelationProviderError(
            "BriggsYoung1963Provider is a finned heat-transfer provider, "
            "not a finned pressure-drop provider."
        )
    return calculate_finned_tube_bank_hydraulics(
        m_dot,
        bundle,
        property_provider=property_provider,
        temperature_in=temperature_in,
        temperature_out=temperature_out,
        pressure=pressure,
        inlet_props=inlet_props,
        midpoint_props=midpoint_props,
        outlet_props=outlet_props,
        pressure_drop_provider=finned_pressure_drop_provider,
    )


def _resolve_contact_input(
    *,
    tube: CircularFinnedTube,
    bundle: TubeBundle,
    contact_area: float,
    ideal_path_resistance: float,
) -> tuple[str, float, float]:
    """Resolve the declared contact input into an absolute resistance.

    Returns ``(contact_input_mode, resistance_contact, contact_resistance_
    equivalent_areal)``. ``resistance_contact`` is the whole-bundle absolute
    [K/W] resistance the caller must fold into the network exactly as the
    legacy explicit-resistance term was folded in. ``ideal_path_resistance``
    is the *contact-affected path's* ideal-contact resistance at the current
    operating point: ``1/G_fin_ideal`` for a welded fin (``fin_branch_
    only``) or ``R_root + R_branches`` for a continuous root layer -- see
    ``calculate_resistance_network``.

    Precedence: explicit ``fin_contact_resistance`` (even ``0.0``) over
    ``fin_contact_efficiency`` over the ideal default (``R_contact == 0``).
    """
    mode = tube.contact_input_mode
    if mode == "explicit_resistance":
        # Unchanged from the pre-v0.7.2 explicit-resistance path: the areal
        # equivalent reported is exactly the supplied value, and the
        # absolute resistance folded into the network is bit-for-bit the
        # same computation as before efficiency mode existed.
        return mode, bundle.fin_contact_thermal_resistance, tube.fin_contact_resistance

    if mode == "contact_efficiency":
        eta_contact = tube.fin_contact_efficiency
        if eta_contact >= 1.0 or not math.isfinite(ideal_path_resistance):
            # eta_contact == 1 is required exactly, not just to numerical
            # tolerance. A non-finite ideal_path_resistance means the
            # contact-affected path has zero ideal conductance (e.g. an
            # external-area override that removes the fin branch entirely);
            # there is then no contact-affected conductance for the
            # efficiency to degrade, so it is inert rather than a 0 * inf
            # indeterminate form.
            resistance_contact = 0.0
        else:
            resistance_contact = (1.0 / eta_contact - 1.0) * ideal_path_resistance
        areal = 0.0 if contact_area == 0.0 else resistance_contact * contact_area
        return mode, resistance_contact, areal

    return mode, 0.0, 0.0


def _effective_contact_efficiency(
    *, resistance_contact: float, ideal_path_resistance: float
) -> float:
    """Conductance ratio of the contact-affected path implied by the network.

    ``eta = G_actual / G_ideal = R_ideal / (R_ideal + R_contact)``. Equals
    ``1.0`` exactly for ideal contact, and reproduces the requested
    ``fin_contact_efficiency`` to numerical tolerance in efficiency mode.
    """
    if resistance_contact <= 0.0 or not math.isfinite(ideal_path_resistance):
        return 1.0
    return ideal_path_resistance / (ideal_path_resistance + resistance_contact)


def calculate_resistance_network(
    *,
    bundle: TubeBundle,
    alpha_inside: float,
    outside_alpha_physical: float,
    resistance_core_wall: float,
) -> ThermalResistanceNetwork:
    """Build core terms and the full topology-correct outside resistance.

    ``outside_alpha_physical`` is always the physical film HTC from the selected
    outside correlation.  The returned ``resistance_outside`` starts at the
    core tube's external wall and ends in the outside bulk fluid.  For a
    continuous root layer it consequently includes the common root/contact
    series terms; ``resistance_outside_branches`` exposes only the downstream
    primary/fin parallel branch component for auditability.
    """

    for name, value in (
        ("alpha_inside", alpha_inside),
        ("outside_alpha_physical", outside_alpha_physical),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    if not math.isfinite(resistance_core_wall) or resistance_core_wall < 0.0:
        raise ValueError("resistance_core_wall must be finite and non-negative.")

    surface = tube_surface_type(bundle)
    area_i = bundle.total_inner_area
    area_gross = bundle.total_outer_area
    resistance_i = 1.0 / (alpha_inside * area_i)
    fin_result: FinEfficiencyResult | None = None

    if surface is TubeSurfaceType.PLAIN:
        area_primary = area_gross
        area_fin = 0.0
        area_geometric = area_gross
        area_effective = area_gross
        eta_fin = 1.0
        eta_overall = 1.0
        resistance_root = 0.0
        contact_input_mode = "not_applicable"
        fin_contact_efficiency_input = None
        fin_contact_efficiency_effective = 1.0
        contact_resistance_equivalent_areal = 0.0
        contact_resistance_used = 0.0
        resistance_contact = 0.0
        contact_area = 0.0
        contact_area_basis = "not_applicable"
        contact_topology = "not_applicable"
        conductance_primary = outside_alpha_physical * area_primary
        conductance_fin = 0.0
        resistance_outside_branches = 1.0 / conductance_primary
        resistance_outside = resistance_outside_branches
        # Preserve the exact historical bare-tube value instead of producing
        # a numerically equivalent reciprocal round trip.
        outside_alpha_effective_gross = outside_alpha_physical
    else:
        tube = bundle.tube
        assert isinstance(tube, CircularFinnedTube)
        fin_result = annular_fin_efficiency(tube, outside_alpha_physical)
        area_primary = bundle.total_primary_outside_area
        area_fin = bundle.total_fin_area
        area_geometric = bundle.total_outer_geometric_area
        area_effective = fin_result.outside_area_effective * bundle.n_tubes_total
        eta_fin = fin_result.fin_efficiency
        eta_overall = area_effective / area_gross
        resistance_root = bundle.root_conduction_resistance
        contact_area = tube.contact_area_per_tube * bundle.n_tubes_total
        contact_area_basis = tube.contact_area_basis
        contact_topology = tube.contact_topology
        fin_contact_efficiency_input = tube.fin_contact_efficiency
        conductance_primary = outside_alpha_physical * area_primary
        ideal_fin_conductance = outside_alpha_physical * eta_fin * area_fin
        if tube.D_root == tube.D_o:
            # Welded fins contact only beneath their roots. The exposed
            # primary cylinder is a genuine parallel bypass and must not be
            # penalised by the fin/root contact resistance. The ideal
            # (contact-free) resistance of that same branch is the basis for
            # both the efficiency-mode conversion and the reported effective
            # efficiency.
            ideal_path_resistance = (
                math.inf if ideal_fin_conductance == 0.0 else 1.0 / ideal_fin_conductance
            )
            contact_input_mode, resistance_contact, contact_resistance_equivalent_areal = (
                _resolve_contact_input(
                    tube=tube,
                    bundle=bundle,
                    contact_area=contact_area,
                    ideal_path_resistance=ideal_path_resistance,
                )
            )
            conductance_fin = (
                0.0
                if ideal_fin_conductance == 0.0
                else 1.0
                / (resistance_contact + 1.0 / ideal_fin_conductance)
            )
            common_root_contact = 0.0
        else:
            # A continuous root layer covers the core/root interface, so its
            # conduction and contact terms are common to both outside
            # branches. Their parallel convection remains represented by the
            # effective-area conductance below. The ideal (contact-free)
            # resistance of the *complete* downstream path -- root
            # conduction plus the parallel branches -- is the basis for both
            # the efficiency-mode conversion and the reported effective
            # efficiency.
            conductance_fin = ideal_fin_conductance
            ideal_path_resistance = resistance_root + 1.0 / (
                conductance_primary + ideal_fin_conductance
            )
            contact_input_mode, resistance_contact, contact_resistance_equivalent_areal = (
                _resolve_contact_input(
                    tube=tube,
                    bundle=bundle,
                    contact_area=contact_area,
                    ideal_path_resistance=ideal_path_resistance,
                )
            )
            common_root_contact = resistance_root + resistance_contact
        fin_contact_efficiency_effective = _effective_contact_efficiency(
            resistance_contact=resistance_contact,
            ideal_path_resistance=ideal_path_resistance,
        )
        # ``contact_resistance_used`` now carries the network's resolved
        # value (identical to ``contact_resistance_equivalent_areal``) so it
        # is never misleading in efficiency mode; explicit-resistance mode
        # keeps it numerically identical to the pre-v0.7.2 field.
        contact_resistance_used = contact_resistance_equivalent_areal

    outside_conductance = conductance_primary + conductance_fin
    if not math.isfinite(outside_conductance) or outside_conductance <= 0.0:
        raise ValueError("Invalid outside thermal conductance.")
    if surface is not TubeSurfaceType.PLAIN:
        resistance_outside_branches = 1.0 / outside_conductance
        resistance_outside = common_root_contact + resistance_outside_branches
        outside_alpha_effective_gross = 1.0 / (
            resistance_outside * area_gross
        )
    if not math.isfinite(resistance_outside) or resistance_outside <= 0.0:
        raise ValueError("Invalid outside thermal resistance.")
    if (
        not math.isfinite(outside_alpha_effective_gross)
        or outside_alpha_effective_gross <= 0.0
    ):
        raise ValueError("Invalid effective gross-area outside heat-transfer coefficient.")
    resistance_total = (
        resistance_i
        + resistance_core_wall
        + resistance_outside
    )
    if not math.isfinite(resistance_total) or resistance_total <= 0.0:
        raise ValueError("Invalid total thermal resistance.")
    ua = 1.0 / resistance_total
    return ThermalResistanceNetwork(
        surface_type=surface,
        area_inside=area_i,
        area_primary_outside=area_primary,
        area_fin=area_fin,
        area_outside_gross=area_gross,
        area_outside_geometric=area_geometric,
        area_outside_effective=area_effective,
        fin_efficiency=eta_fin,
        overall_surface_efficiency=eta_overall,
        outside_alpha_physical=outside_alpha_physical,
        outside_alpha_effective_gross=outside_alpha_effective_gross,
        resistance_inside=resistance_i,
        resistance_core_wall=resistance_core_wall,
        resistance_root=resistance_root,
        contact_input_mode=contact_input_mode,
        fin_contact_efficiency_input=fin_contact_efficiency_input,
        fin_contact_efficiency_effective=fin_contact_efficiency_effective,
        contact_resistance_equivalent_areal=contact_resistance_equivalent_areal,
        contact_resistance_used=contact_resistance_used,
        resistance_contact=resistance_contact,
        contact_area=contact_area,
        contact_area_basis=contact_area_basis,
        contact_topology=contact_topology,
        conductance_primary_outside=conductance_primary,
        conductance_fin_outside=conductance_fin,
        resistance_outside_branches=resistance_outside_branches,
        resistance_outside=resistance_outside,
        resistance_total=resistance_total,
        UA=ua,
        U_gross_outside=ua / area_gross,
        fin_efficiency_result=fin_result,
    )


def build_finned_tube_diagnostics(
    *,
    network: ThermalResistanceNetwork,
    thermal: OutsideThermalDispatchResult,
    hydraulic: OutsideBankHydraulicResult | None = None,
    geometry_warnings: tuple[ModelWarning, ...] = (),
) -> FinnedTubeDiagnostics | None:
    """Materialize the public diagnostics for a circular-finned result."""

    if network.surface_type is TubeSurfaceType.PLAIN:
        return None
    ht = thermal.finned_result
    bank = hydraulic if isinstance(hydraulic, FinnedTubeBankHydraulicResult) else None
    midpoint = None if bank is None else bank.midpoint
    warnings = _deduplicate_warnings(
        (*geometry_warnings, *thermal.warnings, *((bank.warnings) if bank else ()))
    )
    assert network.fin_efficiency_result is not None
    return FinnedTubeDiagnostics(
        tube_surface_type=TubeSurfaceType.CIRCULAR_FINNED,
        area_inside=network.area_inside,
        area_primary_outside=network.area_primary_outside,
        area_fin=network.area_fin,
        area_outside_gross=network.area_outside_gross,
        area_outside_geometric=network.area_outside_geometric,
        area_outside_effective=network.area_outside_effective,
        fin_efficiency=network.fin_efficiency,
        overall_surface_efficiency=network.overall_surface_efficiency,
        outside_alpha_physical=network.outside_alpha_physical,
        outside_alpha_effective_gross=network.outside_alpha_effective_gross,
        resistance_inside=network.resistance_inside,
        resistance_core_wall=network.resistance_core_wall,
        resistance_root=network.resistance_root,
        contact_input_mode=network.contact_input_mode,
        fin_contact_efficiency_input=network.fin_contact_efficiency_input,
        fin_contact_efficiency_effective=network.fin_contact_efficiency_effective,
        contact_resistance_equivalent_areal=network.contact_resistance_equivalent_areal,
        contact_resistance_used=network.contact_resistance_used,
        resistance_contact=network.resistance_contact,
        contact_area=network.contact_area,
        contact_area_basis=network.contact_area_basis,
        contact_topology=network.contact_topology,
        conductance_primary_outside=network.conductance_primary_outside,
        conductance_fin_outside=network.conductance_fin_outside,
        resistance_outside_branches=network.resistance_outside_branches,
        resistance_outside=network.resistance_outside,
        resistance_total=network.resistance_total,
        UA=network.UA,
        U_gross_outside=network.U_gross_outside,
        fin_tip_temperature_ratio=network.fin_efficiency_result.tip_temperature_ratio,
        face_velocity=thermal.face_velocity,
        reference_velocity=thermal.reference_velocity,
        reynolds_number=thermal.reynolds_number,
        prandtl_number=thermal.prandtl_number,
        nusselt_number=thermal.nusselt_number,
        colburn_j_factor=math.nan if ht is None else ht.colburn_j_factor,
        pressure_drop_coefficient=math.nan if midpoint is None else midpoint.coefficient,
        pressure_drop_coefficient_definition=(
            "" if bank is None else bank.coefficient_definition
        ),
        outside_dp_drag=math.nan if bank is None else bank.dp_drag,
        outside_dp_acceleration=math.nan if bank is None else bank.dp_acceleration,
        outside_dp_total=math.nan if bank is None else bank.dp_total,
        heat_transfer_metadata=None if ht is None else ht.metadata,
        pressure_drop_metadata=None if bank is None else bank.metadata,
        warnings=warnings,
    )


def merge_finned_tube_diagnostics(
    thermal: FinnedTubeDiagnostics | None,
    hydraulic: FinnedTubeDiagnostics | None,
) -> FinnedTubeDiagnostics | None:
    """Use converged thermal fields and the final three-state hydraulic fields."""

    if thermal is None:
        return hydraulic
    if hydraulic is None:
        return thermal
    return replace(
        thermal,
        pressure_drop_coefficient=hydraulic.pressure_drop_coefficient,
        pressure_drop_coefficient_definition=hydraulic.pressure_drop_coefficient_definition,
        outside_dp_drag=hydraulic.outside_dp_drag,
        outside_dp_acceleration=hydraulic.outside_dp_acceleration,
        outside_dp_total=hydraulic.outside_dp_total,
        pressure_drop_metadata=hydraulic.pressure_drop_metadata,
        warnings=_deduplicate_warnings((*thermal.warnings, *hydraulic.warnings)),
    )


def point_reynolds(point: Any) -> float:
    value = getattr(point, "reynolds", None)
    if value is None:
        value = getattr(point, "reynolds_number")
    return float(value)


def _reject_bare_provider_on_finned(provider: object) -> None:
    # The historical default string is treated as the public API's automatic
    # selection sentinel. Any other bare-provider choice is an explicit,
    # incompatible request and is rejected rather than silently ignored.
    if isinstance(provider, str) and provider.strip().lower().replace("-", "_") in (
        "zukauskas",
        "default",
        "auto",
    ):
        return
    raise IncompatibleOutsideCorrelationProviderError(
        "CircularFinnedTube cannot use a bare-tube Euler/HTC provider. "
        "Use the automatic Briggs-Young/Robinson-Briggs dispatch (the default) "
        "or pass dedicated finned-tube providers."
    )


def _reject_finned_provider_on_plain(
    provider: object,
    *,
    expected_default: str,
) -> None:
    # Dedicated provider arguments retain their documented defaults even for
    # a plain bundle, where they are inactive. A non-default value means the
    # caller attempted to cross the surface-family boundary.
    if (
        isinstance(provider, str)
        and provider.strip().lower().replace("-", "_") == expected_default
    ):
        return
    raise IncompatibleOutsideCorrelationProviderError(
        "A dedicated circular-finned-tube provider cannot be used with a plain tube."
    )


def _bare_reference_velocity_ratio(bundle: TubeBundle) -> float:
    diameter = float(getattr(bundle.tube, "D_o"))
    transverse = bundle.pitch_transverse
    longitudinal = bundle.pitch_longitudinal
    if bundle.layout.lower() == "inline":
        return transverse / (transverse - diameter)
    diagonal = math.hypot(longitudinal, transverse / 2.0)
    ratio_gap = transverse / (2.0 * (diagonal - diameter))
    ratio_transverse = transverse / (transverse - diameter)
    return ratio_gap if ratio_gap > ratio_transverse else ratio_transverse


def _deduplicate_warnings(warnings: tuple[ModelWarning, ...]) -> tuple[ModelWarning, ...]:
    unique: list[ModelWarning] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        key = (warning.source, warning.code)
        if key not in seen:
            unique.append(warning)
            seen.add(key)
    return tuple(unique)
