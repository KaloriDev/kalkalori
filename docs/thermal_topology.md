# Dry thermal topology — v0.9.1

`TubeBundle` keeps the public `flow_arrangement` values `auto`, `crossflow`,
`counterflow`, and `cocurrentflow`. Explicit geometry values and explicit
Rating/Simulation/solve overrides retain their existing physical laws.

| AUTO circuit | Dry law | Thermal sections | Inter-section mixing |
| --- | --- | --- | --- |
| One longitudinal section | Simple crossflow | 1 | Not applicable |
| Multiple sections, one transverse pass each | Ideal counterflow | 1 global model | Not applicable |
| Multiple sections and multiple transverse passes each | Counter-crossflow equal-UA series | `n_passes_tube / n_passes_transverse` | Complete |

`None` for transverse passes retains the legacy one-section interpretation.
Pass counts must be positive integers, with transverse passes dividing the
total exactly. No fractional section count is inferred. The compatibility
`flow_arrangement_resolved` string describes the local arrangement; use
`thermal_topology_resolved` or result diagnostics to identify section coupling.
An explicit method override `auto` requests circuit resolution even if the
bundle stores an explicit arrangement.

## Restricted physical model

Each of N longitudinal sections receives equal conductance `UA_total/N`.
Both streams retain their full serial heat-capacity rates in each section.
Within a section, the outside stream is transversely mixed and the inside
stream is unmixed. Each section's bulk outlet feeds the adjacent section
in countercurrent progression; complete mixing between sections removes
transverse temperature memory. Capacity rates and UA are representative 0D
values. This model does not resolve individual tubes, row-wise properties,
maldistribution, return/header geometry or a general circuit graph.

Existing nonexact row/pass tube allocations remain the declared effective
geometry approximation. A clean integer pass-group quotient selects the
equal-UA law even if effective rows per section are noninteger. Tube count
and row parity do not generate separate UA weights. Actual unequal section
conductance requires an explicitly defined section-UA map and a future solver.

The model supports dry single-phase bare and circular-finned calculations.
Film, fin, contact, fouling, hydraulic and property-provider physics remain
in their existing paths. Wet-model candidates and pure-water phase-change
zone routes reject this new AUTO topology explicitly. Active gas condensation
also rejects it. Selecting an explicit legacy arrangement retains existing
phase-model applicability; the new law is never used for wet equivalent-UA
reporting. For a known dry calculation with a phase-capable provider, use the
existing `PhaseChangeMode.DISABLED` contract.

## Forward law

Let `r = C_min/C_max`, `x = UA_total/C_min`, and `e` be the local cell
effectiveness evaluated at `x/N` using the physical mixed-side identity:

```text
outside mixed stream is C_min:
    e = 1 - exp(-(1 - exp(-r*x/N))/r)
inside unmixed stream is C_min:
    e = (1 - exp(-r*(1 - exp(-x/N))))/r

K = ((1 - r*e)/(1 - e))**N
E = (K - 1)/(K - r)                    r != 1
E = N*e/(1 + (N - 1)*e)                r == 1
E = 1 - exp(-x)                        r == 0
```

The implementation evaluates the powers through `log1p` and `expm1`, avoiding
overflow and cancellation. It uses the exact balanced-capacity limit rather
than the historical NASA near-unity approximation. N=1 at the shared legacy
NTU entry points dispatches to the unchanged crossflow evaluator, preserving
its original evaluation order and small-ratio threshold. At fixed total UA,
effectiveness grows with N and approaches ideal counterflow.

## Inverse and reachability

The inverse derives the section effectiveness from the target global E:

```text
b = expm1(log1p((1-r)*E/(1-E))/N)
e = b/(b + 1-r)                        r != 1
e = E/(N - (N-1)*E)                    r == 1

outside is C_min: x = -N*log1p(r*log1p(-e))/r
inside is C_min:  x = -N*log1p(log1p(-r*e)/r)
r == 0:          x = -log1p(-E)
```

For finite N and positive r, the local infinite-area ceilings are
`1-exp(-1/r)` (outside C_min) and `(1-exp(-r))/r` (inside C_min). Composing
that ceiling with the same series law gives the global ceiling. Targets at
or above it, negative targets and nonfinite inputs fail explicitly. Targets
numerically indistinguishable from the ceiling can also fail with a binary64
conditioning explanation; arbitrarily large finite NTU is not recoverable
after the forward result rounds to its asymptote. A high-N ceiling may itself
round to 1. Positive finite physical capacity rates must match the hot/cold
rates. The dimensionless kernel separately supports the exact r=0 limit.

## Physical row parity

`RowParityMode` is exported by `core.geometry`; constructor strings are
accepted as well:

```python
bundle = TubeBundle(..., row_parity_mode="continuous")
```

`reset` is the default. At exact row partitions it restarts the odd/even
pattern per physical section module; at nonexact partitions it preserves
the legacy global fallback and its diagnostic. `continuous` numbers the
whole physical bank globally, regardless of circuit partition or thermal
mixing. A 6-row staggered bank with 10/9 tubes and two 3-row modules contains
58 tubes in reset mode and 57 in continuous mode. Equal odd/even counts,
one section and even rows per section give equal totals in both modes.

The chosen total feeds inner/gross outer areas, effective parallel tubes per
pass, tube-side velocity, Reynolds number and hydraulics. Frontal/gap geometry
keeps its existing periodic row average and pitch definitions. No thermal
mixing choice changes row numbering implicitly.

## Result diagnostics

Dry `HXResult`, `HXRatingResult` and `HXSimulationResult` expose defaulted
`thermal_topology_model`, `n_thermal_sections`, `inter_section_mixing`,
`LMTD_countercurrent` and `F_equivalent`. Explicit crossflow is identified as
`crossflow_outside_mixed_inside_unmixed`; sectioned AUTO as
`counter_crossflow_equal_ua`. Legacy explicit/global laws report one thermal
model section, regardless of geometric pass groups.

Terminal countercurrent LMTD is derived from the reported inlet/outlet
temperatures. F is `EMTD/LMTD_countercurrent`, with no input correction factor.
The existing EMTD remains duty divided by required/process UA (Rating), or
by derated process UA (Simulation). Direct solve derives diagnostics from
its own uncorrected UA; the higher-level thermal state remains authoritative.
Ideal counterflow gives F=1 within numerical tolerance. Nonpositive terminal
approaches leave LMTD/F undefined; equal inlet temperatures give LMTD=0 and
undefined F. Active phase reports leave dry topology diagnostics unset.

For the dimensionless prescribed point `r=0.7`, `E=2/3`, outside C_min:

| Law | Required NTU | Equivalent F | EMTD for Q/C_min=40 K |
| --- | ---: | ---: | ---: |
| Simple crossflow / N=1 | 2.093516278001823 | 0.7483480212763305 | 19.10661045262012 K |
| Equal-UA N=2 | 1.672074567194472 | 0.9369670437492145 | 23.92237809532317 K |
| Ideal counterflow | 1.566678764152452 | 1 | 25.53171774281331 K |

These are regression diagnostics, not calibration. Validation also solves
the connected cell port temperatures simultaneously from energy balances,
independently of the production composition formula; checks both physical
branches, r=0/near-zero/near-one/one, forward/inverse recovery, finite-N
ceilings, N progression and the large-N limit; and exercises dry Rating,
Simulation, derating, iteration and the comparison bridge on bare/finned tubes.

## Public implementation provenance and future weights

NASA-CR-159399, *Applications of thermal energy storage in the cement industry*
(1978), appendix ENTU, printed pp.161-162 (PDF pp.169-170), supplies local
mixed crossflow and identical-stage countercurrent composition. The report
is publicly available through [NASA NTRS](https://ntrs.nasa.gov/citations/19780025594).
Delnero's CSU thesis (2000), Eq.4.33, printed p.31 (PDF p.45), independently
supports the global composition law in the
[university-hosted full text](https://projects-web.engr.colostate.edu/nnhvac/papers/chris_thesis.pdf).
Only its global composition equation is used here. The inverse and stable
evaluation are independently derived from those public forward equations.
No licensed reference or private benchmark is implementation authority.

A future weighted / unequal-section counter-crossflow solver should accept
explicit `UA_j = weight_j * UA_total`, with positive weights summing to one,
solve the connected cell port balances, and have Rating numerically invert
that same forward solver. It must retain physical side identity and mixing
assumptions and pass independent matrix/reference checks. Fractional-N,
averaged-F and tube-count-derived UA heuristics are outside this release.
