"""Equal-UA countercurrent series of outside-mixed/inside-unmixed cells.

Public provenance: NASA-CR-159399 (1978), appendix ENTU, printed pp.161-162,
https://ntrs.nasa.gov/citations/19780025594; Delnero, CSU (2000), Eq.4.33,
https://projects-web.engr.colostate.edu/nnhvac/papers/chris_thesis.pdf.
The Cr=1 limit is exact; the NASA near-unity approximation is not used.
No fractional sections or unequal-UA allocation are implied.
"""
import math


def _validate(ntu, cr, n_sections):
    if not math.isfinite(ntu) or ntu < 0:
        raise ValueError("NTU must be finite and nonnegative.")
    if not math.isfinite(cr) or not 0 <= cr <= 1:
        raise ValueError("Cr must be finite and in [0, 1].")
    if isinstance(n_sections, bool) or not isinstance(n_sections, int) or n_sections < 1:
        raise ValueError("n_sections must be a positive integer.")


def _cell(ntu, cr, outside_is_cmin):
    if cr == 0:
        return -math.expm1(-ntu)
    if outside_is_cmin:
        return -math.expm1(math.expm1(-cr*ntu)/cr)
    return -math.expm1(cr*math.expm1(-ntu))/cr


def _compose(ep, cr, n):
    if ep == 0 or ep == 1:
        return ep
    if cr == 1:
        return n*ep/(1+(n-1)*ep)
    delta = 1-cr
    log_k = n*math.log1p(delta*ep/(1-ep))
    t = -math.expm1(-log_k)
    return t/(delta+cr*t)


def effectiveness_counter_crossflow(ntu, cr, n_sections, *, outside_is_cmin):
    """Forward law, total NTU and full serial capacity rates in every cell."""
    _validate(ntu, cr, n_sections)
    if cr == 0:
        return -math.expm1(-ntu)
    return _compose(_cell(ntu/n_sections, cr, outside_is_cmin), cr, n_sections)


def counter_crossflow_ceiling(cr, n_sections, *, outside_is_cmin):
    """Infinite-area effectiveness ceiling (finite-N topology dependent)."""
    _validate(0., cr, n_sections)
    if cr == 0:
        return 1.
    ep = -math.expm1(-1/cr) if outside_is_cmin else -math.expm1(-cr)/cr
    return _compose(ep, cr, n_sections)


def ntu_counter_crossflow(effectiveness, cr, n_sections, *, outside_is_cmin):
    """Algebraic inverse of the same cell/composition law, with reachability guard."""
    limit = counter_crossflow_ceiling(cr, n_sections, outside_is_cmin=outside_is_cmin)
    if not math.isfinite(effectiveness) or not 0 <= effectiveness < limit:
        raise ValueError(f"Requested effectiveness is unreachable: finite-area domain is "
                         f"0 <= effectiveness < {limit:.17g} for N={n_sections}, Cr={cr}.")
    if effectiveness == 0:
        return 0.
    if cr == 0:
        return -math.log1p(-effectiveness)
    if cr == 1:
        ep = effectiveness/(n_sections-(n_sections-1)*effectiveness)
    else:
        delta = 1-cr
        b = math.expm1(math.log1p(delta*effectiveness/(1-effectiveness))/n_sections)
        ep = b/(b+delta)
    try:
        if outside_is_cmin:
            cell_ntu = -math.log1p(cr*math.log1p(-ep))/cr
        else:
            cell_ntu = -math.log1p(math.log1p(-cr*ep)/cr)
    except ValueError as exc:
        raise ValueError("Requested effectiveness is numerically indistinguishable from "
                         "the counter-crossflow ceiling in binary64.") from exc
    return n_sections*cell_ntu
