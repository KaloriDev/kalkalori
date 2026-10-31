# SPDX-License-Identifier: GPL-3.0-only
"""Continuous-sheet provider contracts and provisional Wang source card.

Independent equation implementation from the 2026-10-31 forensic handover,
section 2.1. No python-hvac or ACHP source code is copied. Arithmetic agreement
is not experimental validation. See docs/continuous_plate_fins.md.
"""
from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

from core.common.warnings import make_warning
from core.geometry.lattice import LayoutClass


class ContinuousFinSourceError(ValueError):
    pass


class ContactPolicy(str, Enum):
    EMPIRICAL_INCLUDED = "empirical_included"
    IDEAL_CONTACT = "ideal_contact"
    EXPLICIT_CONTACT_RESISTANCE = "explicit_contact_resistance"


@dataclass(frozen=True)
class ContinuousFinSourceCard:
    method: str = "experimental_wang_chi_chang_2000"
    doi: str = "10.1016/S0017-9310(99)00333-6"
    source_status: str = "SOURCE_PARTIALLY_CLOSED"
    validation_status: str = "SYNTHETIC_ARITHMETIC_ONLY"
    reynolds_domain_status: str = "SOURCE_DOMAIN_UNKNOWN"
    reynolds_basis: str = "Re_Dc=(m_dot/A_min)*collar_OD/mu"
    area_basis: str = "candidate_fin_faces_plus_exposed_tube_and_collar_cylinders; source closure pending"
    coefficient_basis: str = "compact_HX_f_Ao_over_Ac; no extra factor four"
    contact_policy: ContactPolicy = ContactPolicy.EMPIRICAL_INCLUDED
    pressure_scope: str = "unresolved_source_estimate; entry_exit_scope_unknown"
    open_gates: tuple[str, ...] = ("B01", "B02", "B03", "B04", "B05")
    transcription_findings: tuple[str, ...] = (
        "P2=-0.236+0.126*ln(Re); OhioLINK secondary has conflicting minus",
        "P5=-0.083+0.058*N/ln(Re); OhioLINK secondary has conflicting minus",
        "F3=1.696-15.695/ln(Re); IIR2019 secondary has conflicting plus",
        "P6=-5.735+1.21*ln(Re/N); OhioLINK secondary prints 1.211",
    )


WANG2000_SOURCE = ContinuousFinSourceCard()


def source_warnings():
    return tuple(make_warning(code=code, message=message, source=WANG2000_SOURCE.method, severity="warning")
        for code, message in (
            ("CONTINUOUS_FIN_EXPERIMENTAL", "Wang primary source/experimental validation incomplete (B01-B05); research use only."),
            ("CONTINUOUS_FIN_RE_DOMAIN_UNKNOWN", "Original Wang Reynolds envelope is unresolved; software bounds are not source limits."),
            ("CONTINUOUS_FIN_PRESSURE_SCOPE_UNKNOWN", "Compact-HX pressure estimate has unresolved entry/exit scope; not validated irreversible bank loss."),
        ))


def positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


@dataclass(frozen=True)
class ContinuousFinRequest:
    bundle: object
    mass_flow: float
    props: object
    fluid_family: str = "dry_air"
    surface_condition: str = "dry"
    strict_source_domain: bool = False

    def __post_init__(self):
        positive(self.mass_flow, "mass_flow")
        for name in ("rho", "mu", "k", "cp"):
            positive(getattr(self.props, name), name)
        if self.bundle.continuous_fin_geometry is None:
            raise ContinuousFinSourceError("Continuous sheet geometry required; no individual-fin substitution.")
        if self.bundle.tube_bank_lattice.layout_class is not LayoutClass.CONVENTIONAL_STAGGERED:
            raise ContinuousFinSourceError("Wang requires conventional STAGGERED; INLINE/SHIFTED unsupported.")
        if self.fluid_family != "dry_air" or self.surface_condition != "dry":
            raise ContinuousFinSourceError("Wang supports dry air only; wet/condensing external surfaces unsupported.")
        f = self.bundle.continuous_fin_geometry
        for name, value, lo, hi in (
            ("N", self.bundle.n_rows, 1, 6),
            ("tube_OD", self.bundle.tube.D_o, .00635, .0127),
            ("fin_pitch", f.pitch, .00119, .0087),
            ("pitch_transverse", self.bundle.pitch_transverse, .0177, .03175),
            ("pitch_longitudinal", self.bundle.pitch_longitudinal, .0124, .0275),
        ):
            if not lo-1e-12 <= value <= hi+1e-12:
                raise ContinuousFinSourceError(f"Wang nominal secondary-source domain: {name}={value} outside [{lo}, {hi}]; no clipping or industrial extrapolation.")
        if self.strict_source_domain:
            raise ContinuousFinSourceError("Strict Wang source validation cannot pass: original Reynolds domain and B01-B05 remain unresolved.")

    @property
    def flow(self):
        return self.bundle.continuous_fin_surface

    @property
    def mass_flux(self):
        return self.mass_flow/self.flow.minimum_free_flow_area

    @property
    def reynolds_number(self):
        return self.mass_flux*self.bundle.continuous_fin_geometry.collar_outer_diameter/self.props.mu


def _kernel_inputs(Re, N, Dc, Fp, Pt, Pl, Dh):
    for value, name in ((Re, "Re_Dc"), (Dc, "D_c"), (Fp, "F_p"), (Pt, "P_t"), (Pl, "P_l"), (Dh, "D_h")):
        positive(value, name)
    if isinstance(N, bool) or not isinstance(N, int) or not 1 <= N <= 6:
        raise ContinuousFinSourceError("Wang N must be an integer from 1 to 6; never clipped.")
    if Re <= 1:
        raise ContinuousFinSourceError("Wang logarithmic arithmetic requires Re_Dc > 1; this is not a validated Reynolds lower bound.")
    return math.log(Re)


def wang2000_j(*, Re_Dc, N, D_c, F_p, P_t, P_l, D_h):
    """Provisional scalar j kernel, retaining both row branches and signs."""
    lnr = _kernel_inputs(Re_Dc, N, D_c, F_p, P_t, P_l, D_h)
    if N == 1:
        return .108*Re_Dc**(-.29)*(P_t/P_l)**(1.9-.23*lnr)*(F_p/D_c)**(-1.084)*(F_p/D_h)**(-.786)*(F_p/P_t)**(-.236+.126*lnr)
    p3 = -.361-.042*N/lnr+.158*math.log(N*(F_p/D_c)**.41)
    p4 = -1.224-.076*(P_l/D_h)**1.42/lnr
    p5 = -.083+.058*N/lnr
    p6 = -5.735+1.21*math.log(Re_Dc/N)
    return .086*Re_Dc**p3*N**p4*(F_p/D_c)**p5*(F_p/D_h)**p6*(F_p/P_t)**(-.93)


def wang2000_f(*, Re_Dc, N, D_c, F_p, P_t, P_l, D_h):
    """Provisional compact-HX friction kernel; not a Darcy factor."""
    lnr = _kernel_inputs(Re_Dc, N, D_c, F_p, P_t, P_l, D_h)
    f1 = -.764+.739*(P_t/P_l)+.177*(F_p/D_c)-.00758/N
    f2 = -15.689+64.021/lnr
    f3 = 1.696-15.695/lnr
    return .0267*Re_Dc**f1*(P_t/P_l)**f2*(F_p/D_c)**f3


def kernel_arguments(request):
    b, f = request.bundle, request.bundle.continuous_fin_geometry
    return dict(Re_Dc=request.reynolds_number, N=b.n_rows, D_c=f.collar_outer_diameter,
        F_p=f.pitch, P_t=b.pitch_transverse, P_l=b.pitch_longitudinal,
        D_h=request.flow.compact_hydraulic_diameter)


@dataclass(frozen=True)
class ContinuousFinHTCResult:
    alpha: float
    colburn_j_factor: float
    reynolds_number: float
    prandtl_number: float
    reference_mass_flux: float
    metadata: ContinuousFinSourceCard = WANG2000_SOURCE
    warnings: tuple = ()


class ContinuousFinHTCProvider(Protocol):
    def evaluate(self, request: ContinuousFinRequest) -> ContinuousFinHTCResult: ...


@dataclass(frozen=True)
class ExperimentalWang2000HTC:
    def evaluate(self, request):
        j = positive(wang2000_j(**kernel_arguments(request)), "j")
        pr = request.props.cp*request.props.mu/request.props.k
        h = positive(j*request.mass_flux*request.props.cp/pr**(2/3), "HTC")
        return ContinuousFinHTCResult(h, j, request.reynolds_number, pr, request.mass_flux, warnings=source_warnings())
