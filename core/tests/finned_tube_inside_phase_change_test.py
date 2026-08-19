"""Public dry-finned-outside coverage for the established inside phase paths."""

from __future__ import annotations

import math

import pytest

from core.geometry import BareTube, CircularFinnedTube, TubeBundle, TubeOrientation
from core.heat_transfer.finned_tube_outside import BriggsYoung1963Provider
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.heat_balance import BalanceSideSpec
from core.models.simulation import HXSideInput
from core.phase_change.types import PhaseChangeDirection
from core.pressure_drop.finned_tube_pressure_drop import RobinsonBriggs1966Provider
from core.properties.common import FluidTransportProperties
from core.properties.fluids import ConstantPropertyProvider
from core.properties.gas_mixture import GasMixturePropertyProvider, GasMixtureSpec
from core.properties.water import IAPWS97WaterSteamProvider


P_WATER = 1.0e6
P_AIR = 101_325.0
HOT_AIR = ConstantPropertyProvider(
    FluidTransportProperties(rho=1.2, mu=1.8e-5, k=0.026, cp=1005.0)
)
COLD_AIR = HOT_AIR


class _CountingHeatTransferProvider:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        return BriggsYoung1963Provider().evaluate(request)


class _CountingPressureDropProvider:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        return RobinsonBriggs1966Provider().evaluate(request)


def _hx(*, orientation: TubeOrientation | None) -> BareTubeHeatExchanger:
    core = BareTube(
        D_i=0.022,
        D_o=0.025,
        length_total=2.8,
        length_effective=2.8,
        wall_k=50.0,
        tube_orientation=orientation,
    )
    tube = CircularFinnedTube(
        core_tube=core,
        fin_k=200.0,
        D_fin=0.052,
        D_root=0.025,
        fin_thickness_root=0.0005,
        fin_pitch=0.003,
        fin_contact_resistance=0.0,
    )
    transverse_pitch = 0.060
    return BareTubeHeatExchanger(
        TubeBundle(
            tube=tube,
            n_rows=6,
            n_tubes_per_row=10,
            pitch_transverse=transverse_pitch,
            pitch_longitudinal=math.sqrt(3.0) * transverse_pitch / 2.0,
            layout="staggered",
            n_passes_tube=2,
            flow_arrangement="counterflow",
        )
    )


def _wet_spec() -> GasMixtureSpec:
    y_h2o = 0.08
    remainder = 1.0 - y_h2o
    dry = {"N2": 0.70, "O2": 0.10, "CO2": 0.08}
    total = sum(dry.values())
    return GasMixtureSpec(
        components={name: value * remainder / total for name, value in dry.items()}
        | {"H2O": y_h2o},
        basis="mole",
    )


def _dry_spec() -> GasMixtureSpec:
    return GasMixtureSpec(components={"N2": 0.79, "O2": 0.21}, basis="mole")


def _assert_dry_finned_result(result) -> None:
    diagnostics = result.finned_tube_diagnostics
    assert diagnostics is not None
    assert result.final_result.finned_tube_diagnostics is not None
    assert diagnostics.outside_alpha_physical > 0.0
    assert diagnostics.outside_alpha_effective_gross > 0.0
    assert diagnostics.resistance_outside == pytest.approx(
        1.0
        / (
            diagnostics.outside_alpha_effective_gross
            * diagnostics.area_outside_gross
        )
    )
    assert diagnostics.UA == pytest.approx(1.0 / diagnostics.resistance_total)
    assert diagnostics.U == pytest.approx(
        diagnostics.UA / diagnostics.area_outside_gross
    )
    outside_alpha = (
        result.outside_alfa_mean
        if hasattr(result, "outside_alfa_mean")
        else result.alfa_o
    )
    assert outside_alpha == pytest.approx(
        diagnostics.outside_alpha_effective_gross
    )
    assert result.final_result.outside_side_thermal.alfa == pytest.approx(
        result.final_result.finned_tube_diagnostics.outside_alpha_effective_gross
    )


def _assert_provider_threading(
    simulation,
    rating,
    heat_provider: _CountingHeatTransferProvider,
    pressure_provider: _CountingPressureDropProvider,
    calls_after_simulation: tuple[int, int],
) -> None:
    _assert_dry_finned_result(simulation)
    assert calls_after_simulation[0] > 0
    assert calls_after_simulation[1] > 0
    _assert_dry_finned_result(rating)
    assert heat_provider.calls > calls_after_simulation[0]
    assert pressure_provider.calls > calls_after_simulation[1]


def test_dry_finned_outside_supports_inside_water_evaporation_in_simulation_and_rating() -> None:
    hx = _hx(orientation=TubeOrientation.VERTICAL_UPWARD)
    heat_provider = _CountingHeatTransferProvider()
    pressure_provider = _CountingPressureDropProvider()
    simulation = hx.simulate(
        HXSideInput(
            provider=IAPWS97WaterSteamProvider(),
            m_dot=1.0,
            T_in=350.0,
            p=P_WATER,
        ),
        HXSideInput(provider=HOT_AIR, m_dot=30.0, T_in=700.0, p=P_AIR),
        finned_heat_transfer_provider=heat_provider,
        finned_pressure_drop_provider=pressure_provider,
    )
    assert simulation.inside_phase_change.direction is PhaseChangeDirection.EVAPORATION
    assert simulation.inside_phase_change.active is True
    calls_after_simulation = (heat_provider.calls, pressure_provider.calls)
    rating = hx.rate(
        BalanceSideSpec(
            provider=IAPWS97WaterSteamProvider(),
            p=P_WATER,
            m_dot=1.0,
            T_in=350.0,
            quality_out=0.5,
        ),
        BalanceSideSpec(provider=HOT_AIR, p=P_AIR, m_dot=30.0, T_in=700.0),
        finned_heat_transfer_provider=heat_provider,
        finned_pressure_drop_provider=pressure_provider,
    )
    assert rating.inside_phase_change.direction is PhaseChangeDirection.EVAPORATION
    assert rating.inside_phase_change.active is True
    _assert_provider_threading(
        simulation, rating, heat_provider, pressure_provider, calls_after_simulation
    )


def test_dry_finned_outside_supports_inside_steam_condensation_in_simulation_and_rating() -> None:
    hx = _hx(orientation=TubeOrientation.VERTICAL_DOWNWARD)
    heat_provider = _CountingHeatTransferProvider()
    pressure_provider = _CountingPressureDropProvider()
    simulation = hx.simulate(
        HXSideInput(
            provider=IAPWS97WaterSteamProvider(),
            m_dot=1.0,
            quality_in=1.0,
            p=P_WATER,
        ),
        HXSideInput(provider=COLD_AIR, m_dot=30.0, T_in=300.0, p=P_AIR),
        finned_heat_transfer_provider=heat_provider,
        finned_pressure_drop_provider=pressure_provider,
    )
    assert simulation.inside_phase_change.direction is PhaseChangeDirection.CONDENSATION
    assert simulation.inside_phase_change.active is True
    calls_after_simulation = (heat_provider.calls, pressure_provider.calls)
    rating = hx.rate(
        BalanceSideSpec(
            provider=IAPWS97WaterSteamProvider(),
            p=P_WATER,
            m_dot=1.0,
            quality_in=1.0,
            quality_out=0.5,
        ),
        BalanceSideSpec(provider=COLD_AIR, p=P_AIR, m_dot=30.0, T_in=300.0),
        finned_heat_transfer_provider=heat_provider,
        finned_pressure_drop_provider=pressure_provider,
    )
    assert rating.inside_phase_change.direction is PhaseChangeDirection.CONDENSATION
    assert rating.inside_phase_change.active is True
    _assert_provider_threading(
        simulation, rating, heat_provider, pressure_provider, calls_after_simulation
    )


def test_dry_finned_outside_supports_unknown_steam_mass_flow_rating() -> None:
    """v0.7.1: Rating derives required steam m_dot through the finned path.

    Uses the established finned dry-outside topology/alpha semantics
    unchanged; only the tube-side mass flow is left for Rating to solve.
    """
    hx = _hx(orientation=TubeOrientation.VERTICAL_DOWNWARD)
    heat_provider = _CountingHeatTransferProvider()
    pressure_provider = _CountingPressureDropProvider()
    inside = BalanceSideSpec(
        provider=IAPWS97WaterSteamProvider(),
        p=P_WATER,
        m_dot=None,
        T_in=520.0,
        quality_out=0.0,
    )
    outside = BalanceSideSpec(
        provider=COLD_AIR, p=P_AIR, m_dot=5.0, T_in=290.0, T_out=310.0,
    )
    rating = hx.rate(
        inside,
        outside,
        finned_heat_transfer_provider=heat_provider,
        finned_pressure_drop_provider=pressure_provider,
    )
    assert rating.inside_phase_change.direction is PhaseChangeDirection.CONDENSATION
    assert rating.inside_phase_change.active is True
    assert rating.inside_phase_change.quality_out == pytest.approx(0.0, abs=1e-9)
    assert rating.closed_balance.inside.m_dot == pytest.approx(
        rating.inside_phase_change.mass_flow_total
    )
    assert rating.closed_balance.inside.m_dot > 0.0
    assert inside.m_dot is None
    _assert_dry_finned_result(rating)


def test_dry_finned_outside_supports_inside_wet_gas_condensation_in_simulation_and_rating() -> None:
    hx = _hx(orientation=TubeOrientation.VERTICAL_DOWNWARD)
    heat_provider = _CountingHeatTransferProvider()
    pressure_provider = _CountingPressureDropProvider()
    wet_provider = GasMixturePropertyProvider(_wet_spec())
    dry_provider = GasMixturePropertyProvider(_dry_spec())
    simulation = hx.simulate(
        HXSideInput(provider=wet_provider, m_dot=1.0, T_in=360.0, p=P_AIR),
        HXSideInput(provider=dry_provider, m_dot=5.0, T_in=290.0, p=P_AIR),
        finned_heat_transfer_provider=heat_provider,
        finned_pressure_drop_provider=pressure_provider,
    )
    assert simulation.inside_phase_change.direction is PhaseChangeDirection.CONDENSATION
    assert simulation.inside_phase_change.active is True
    calls_after_simulation = (heat_provider.calls, pressure_provider.calls)
    rating = hx.rate(
        BalanceSideSpec(
            provider=wet_provider,
            p=P_AIR,
            m_dot=1.0,
            T_in=360.0,
            T_out=320.34499765485987,
        ),
        BalanceSideSpec(
            provider=dry_provider,
            p=P_AIR,
            m_dot=5.0,
            T_in=290.0,
            T_out=298.6985136488704,
        ),
        finned_heat_transfer_provider=heat_provider,
        finned_pressure_drop_provider=pressure_provider,
    )
    assert rating.inside_phase_change.direction is PhaseChangeDirection.CONDENSATION
    assert rating.inside_phase_change.active is True
    _assert_provider_threading(
        simulation, rating, heat_provider, pressure_provider, calls_after_simulation
    )
