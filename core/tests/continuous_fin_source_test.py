# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace
import pytest
from core.tests.continuous_plate_fin_geometry_test import bank
from core.properties.common import FluidTransportProperties
from core.heat_transfer.continuous_fin_source import (
    ContinuousFinRequest, ContinuousFinSourceError, ExperimentalWang2000HTC,
    WANG2000_SOURCE, wang2000_j, wang2000_f,
)
from core.pressure_drop.continuous_fin import ExperimentalWang2000DP, wang_density_change_estimate


# Handover B1-B5: the geometry here is deliberately the original provisional
# face/cylinder convention, NOT our finite-pack geometry or an experiment.
@pytest.mark.parametrize("N,Do,Pt,Pl,Fp,Re,j,f", [
    (1,.00952,.0254,.01905,.0012,800,.0314659,.087068),
    (2,.00952,.0254,.01905,.0024,1200,.0153803,.051720),
    (4,.007,.021,.0127,.0015,2000,.0124556,.052259),
    (6,.0127,.03175,.0275,.003,3500,.0078254,.034547),
    (3,.00952,.0254,.01905,.002,5000,.0066963,.028190),
])
def test_five_provisional_arithmetic_references(N, Do, Pt, Pl, Fp, Re, j, f):
    dc, t, w, nt = Do+.00023, .000115, .36, 8
    nf, height, depth = round(w/Fp), 9*Pt, (N+1)*Pl
    ac = height*w-nf*t*(height-nt*dc)-nt*dc*w
    ao = 2*nf*(height*depth-N*nt*3.141592653589793*dc**2/4)+N*nt*3.141592653589793*dc*(w-nf*t)
    args = dict(Re_Dc=Re, N=N, D_c=dc, F_p=Fp, P_t=Pt, P_l=Pl, D_h=4*ac*depth/ao)
    assert wang2000_j(**args) == pytest.approx(j, rel=1e-5)
    assert wang2000_f(**args) == pytest.approx(f, rel=2e-5)


@pytest.mark.parametrize("N", [0, 7, True, 2.5])
def test_no_row_clipping(N):
    with pytest.raises(ContinuousFinSourceError):
        wang2000_j(Re_Dc=1000, N=N, D_c=.01, F_p=.002, P_t=.025, P_l=.02, D_h=.003)


def test_collar_reynolds_and_conversions():
    b = bank()
    p = FluidTransportProperties(1.18, 1.85e-5, .026, 1007.)
    r = ContinuousFinRequest(b, .005, p)
    ht, dp = ExperimentalWang2000HTC().evaluate(r), ExperimentalWang2000DP().evaluate(r)
    assert ht.reynolds_number == pytest.approx(.005/b.minimum_free_flow_area*.0102/p.mu)
    assert ht.alpha == pytest.approx(ht.colburn_j_factor*r.mass_flux*p.cp/ht.prandtl_number**(2/3))
    assert dp.core_friction_estimate == pytest.approx(dp.coefficient*r.flow.source_face_cylinder_area/r.flow.minimum_free_flow_area*r.mass_flux**2/(2*p.rho))
    assert dp.metadata is ht.metadata is WANG2000_SOURCE
    assert ht.metadata.reynolds_domain_status == "SOURCE_DOMAIN_UNKNOWN"
    assert len(ht.warnings) == 3


@pytest.mark.parametrize("changes", [dict(fluid_family="flue_gas"), dict(surface_condition="wet"), dict(strict_source_domain=True)])
def test_source_guards(changes):
    with pytest.raises(ContinuousFinSourceError):
        ContinuousFinRequest(bank(), .005, FluidTransportProperties(1., 2e-5, .03, 1000.), **changes)


@pytest.mark.parametrize("layout", ["inline", "shifted"])
def test_unsupported_source_lattice(layout):
    from core.geometry import TubeBankLattice
    # Resolved SHIFTED has valid physical geometry but no Wang physics.
    lattice = TubeBankLattice("inline") if layout == "inline" else TubeBankLattice("shifted", pattern="custom", phase="fixed", site_offsets=tuple(tuple((0.,0.) for _ in range(3)) for _ in range(3)))
    b = replace(bank(count=3., height=.09), lattice=lattice)
    with pytest.raises(ContinuousFinSourceError, match="INLINE/SHIFTED"):
        ContinuousFinRequest(b, .005, FluidTransportProperties(1., 2e-5, .03, 1000.))


def test_signed_density_term_and_no_extra_factor_four():
    assert wang_density_change_estimate(mass_flux=2., sigma=.5, density_in=1., density_out=2.) == pytest.approx(-1.25)
    assert wang_density_change_estimate(mass_flux=2., sigma=.5, density_in=1., density_out=1.) == 0.
