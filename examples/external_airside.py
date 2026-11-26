"""Synthetic API example, with no empirical correlation or calibration."""
from core.geometry import BareTube, TubeBundle
from core.models.bare_tube import BareTubeHeatExchanger
from core.models.simulation import HXSideInput
from core.models.heat_balance import BalanceSideSpec
from core.properties import ConstantPropertyProvider
from core.properties.common import FluidTransportProperties
from core.heat_transfer import OutsideProviderMetadata, OutsideThermalProviderResult
from core.pressure_drop import OutsideBankPressureDropResult, OutsidePressureScope


class SyntheticFilm:
    def evaluate(self, request):
        return OutsideThermalProviderResult(60., OutsideProviderMetadata(
            'synthetic film', 'API demonstration only', 'uniform physical film on actual surfaces'))


class SyntheticPressure:
    def evaluate(self, request):
        return OutsideBankPressureDropResult(25., OutsideProviderMetadata(
            'synthetic pressure', 'API demonstration only', 'complete bank loss'),
            OutsidePressureScope.BANK_ONLY, 'Bank face to bank face; no entrance, exit or acceleration')


def main():
    tube=BareTube(D_i=.020,D_o=.024,length_total=2.,length_effective=2.,wall_k=45.)
    bundle=TubeBundle(tube,8,10,.050,.050,'inline',2)
    hx=BareTubeHeatExchanger(bundle,outside_thermal_provider=SyntheticFilm(),
                            outside_hydraulic_provider=SyntheticPressure())
    water=ConstantPropertyProvider(FluidTransportProperties(1000.,.001,.6,4200.))
    air=ConstantPropertyProvider(FluidTransportProperties(1.2,1.85e-5,.026,1006.))
    inside=HXSideInput(provider=water,m_dot=1.,T_in=360.,p=200000.)
    outside=HXSideInput(provider=air,m_dot=1.,T_in=300.,p=101325.)
    simulation=hx.simulate(inside,outside)
    rating=hx.rate(
        BalanceSideSpec(provider=water,m_dot=1.,T_in=360.,T_out=simulation.T_out_inside,p=200000.),
        BalanceSideSpec(provider=air,m_dot=1.,T_in=300.,T_out=simulation.T_out_outside,p=101325.))
    print(f'Synthetic: Q={simulation.q:.3f} W; UA={rating.UA_actual:.3f} W/K; dp={rating.outside_dp_total:.3f} Pa')


if __name__=='__main__':
    main()
