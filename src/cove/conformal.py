"""Verification decision rules on a calibration set of m identity units.

All scores are exact integers (see ``exact.py``).  A *unit* is one image of one
identity; calibration units come from m distinct identities disjoint from test.

Notation: N = C(m+2, 2) is the number of unordered pairs in the augmented array
{calibration units, probe, reference}.  ``cal_sorted`` holds the C(m, 2)
calibration pair scores in ascending order.

Rules (accept = declare "same identity"):

* ``cove``        transductive pairwise conformal test (paper Thm 2, validity):
                  p = #{augmented pairs with score >= s*} / N, accept iff p <= alpha.
* ``u_conformal`` inductive threshold from calibration pairs only (paper Prop 1):
                  accept iff s* > k-th largest calibration pair,
                  k = floor(alpha * N) - 2m.  Valid; trivial when k < 1.
* ``disjoint``    split conformal on floor(m/2) disjoint calibration pairs.
* ``naive``       empirical (1 - alpha) quantile of all calibration pairs,
                  treating the C(m, 2) dependent pairs as i.i.d.  Not valid.
* ``naive_cp``    Clopper-Pearson upper bound on FAR treating pairs as i.i.d.
                  (the usual "PAC" recipe).  Not valid under dependence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy.stats import beta as _beta


def n_aug_pairs(m: int) -> int:
    return (m + 2) * (m + 1) // 2


@dataclass
class Calibration:
    """Pre-computed calibration state for one draw of m units."""
    units: torch.Tensor          # (m, d) quantized embeddings
    cal_sorted: torch.Tensor     # (C(m,2),) ascending int64 pair scores
    disjoint_sorted: torch.Tensor  # (floor(m/2),) ascending scores of a random matching
    pair_i: torch.Tensor | None = None   # unit indices of each sorted calibration pair
    pair_j: torch.Tensor | None = None

    @property
    def m(self) -> int:
        return int(self.units.shape[0])

    @property
    def n_pairs(self) -> int:
        return int(self.cal_sorted.numel())


def build_calibration(units: torch.Tensor, pair_scores: torch.Tensor,
                      gen: torch.Generator | None = None) -> Calibration:
    """``pair_scores`` is the (m, m) symmetric matrix g(U_i, U_j) (diagonal unused)."""
    m = units.shape[0]
    if m < 2:
        raise ValueError("need at least two calibration units")
    s = pair_scores
    iu = torch.triu_indices(m, m, offset=1, device=s.device)
    srt = torch.sort(s[iu[0], iu[1]], stable=True)
    cal_sorted = srt.values
    perm = torch.randperm(m, generator=gen).to(s.device)
    k = m // 2
    disjoint = s[perm[:k], perm[k:2 * k]]
    return Calibration(units=units, cal_sorted=cal_sorted,
                       disjoint_sorted=torch.sort(disjoint).values,
                       pair_i=iu[0][srt.indices], pair_j=iu[1][srt.indices])


def count_ge(sorted_asc: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """#{x in sorted_asc : x >= v} for each v."""
    return sorted_asc.numel() - torch.searchsorted(sorted_asc, values, side="left")


def rank_budget(alpha: float, n: int) -> int:
    """Exact floor(alpha*n) for the supplied binary floating-point alpha.

    Avoid an epsilon or rounded float multiplication that could cross an integer
    rank boundary and invalidate an otherwise exact finite-sample guarantee.
    """
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    numerator, denominator = alpha.as_integer_ratio()
    return numerator * int(n) // denominator


def _accept_all(sorted_asc: torch.Tensor) -> torch.Tensor:
    """Threshold below every possible score of this dtype (the rule accepts all trials)."""
    if sorted_asc.dtype.is_floating_point:
        return torch.tensor(float("-inf"), dtype=sorted_asc.dtype, device=sorted_asc.device)
    return torch.tensor(torch.iinfo(sorted_asc.dtype).min, dtype=sorted_asc.dtype, device=sorted_asc.device)


def kth_largest(sorted_asc: torch.Tensor, k: int) -> torch.Tensor:
    return sorted_asc[sorted_asc.numel() - k]


def cove_rank(cal: Calibration, s_star: torch.Tensor, cross_a: torch.Tensor,
              cross_b: torch.Tensor) -> torch.Tensor:
    """Rank R = #{augmented pairs with score >= s*}, including the trial itself.

    ``cross_a`` / ``cross_b`` are (T, m) exact scores of probe / reference against
    the calibration units.  The p-value is R / N.
    """
    r = 1 + count_ge(cal.cal_sorted, s_star)
    r = r + (cross_a >= s_star[:, None]).sum(dim=1) + (cross_b >= s_star[:, None]).sum(dim=1)
    return r


def cove_accept(rank: torch.Tensor, m: int, alpha: float) -> torch.Tensor:
    """Accept iff p = R/N <= alpha, i.e. R <= floor(alpha * N)."""
    return rank <= rank_budget(alpha, n_aug_pairs(m))


def threshold_u_conformal(cal: Calibration, alpha: float) -> torch.Tensor | None:
    """Prop 1: valid inductive threshold; None means the rule rejects everything."""
    k = rank_budget(alpha, n_aug_pairs(cal.m)) - 2 * cal.m
    if k < 1:
        return None
    if k > cal.n_pairs:            # only at alpha = 1: every rank is admissible, accept all
        return _accept_all(cal.cal_sorted)
    return kth_largest(cal.cal_sorted, k)


def threshold_disjoint(cal: Calibration, alpha: float) -> torch.Tensor | None:
    n = cal.disjoint_sorted.numel()
    k = rank_budget(alpha, n + 1)
    if k < 1:
        return None
    if k > n:                      # only at alpha = 1: accept all
        return _accept_all(cal.disjoint_sorted)
    return kth_largest(cal.disjoint_sorted, k)


def threshold_naive(cal: Calibration, alpha: float) -> torch.Tensor | None:
    """Accept iff s > t where exactly floor(alpha * n) calibration pairs exceed t."""
    n = cal.n_pairs
    k = rank_budget(alpha, n)
    # threshold is the (k+1)-th largest pair: accepting s > t admits k pairs
    return kth_largest(cal.cal_sorted, min(k + 1, n))


def threshold_disjoint_cp(cal: Calibration, alpha: float, delta: float = 0.05) -> torch.Tensor | None:
    """PAC-valid under i.i.d. identity draws, not exchangeability alone.

    The floor(m/2) disjoint pair scores are then i.i.d., so the Clopper-Pearson
    choice controls P(conditional FAR > alpha) by delta. Sampling identities
    without replacement from a fixed finite pool does not meet this hypothesis.
    """
    n = cal.disjoint_sorted.numel()
    if n == 0 or _cp_upper(0, n, delta) > alpha:
        return None
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _cp_upper(mid, n, delta) <= alpha:
            lo = mid
        else:
            hi = mid - 1
    return kth_largest(cal.disjoint_sorted, min(lo + 1, n))


def _cp_upper(k: int, n: int, delta: float) -> float:
    if k >= n:
        return 1.0
    return float(_beta.ppf(1.0 - delta, k + 1, n - k))


def threshold_naive_cp(cal: Calibration, alpha: float, delta: float = 0.05) -> torch.Tensor | None:
    """Largest k with CP_upper(k exceedances out of C(m,2)) <= alpha."""
    n = cal.n_pairs
    if _cp_upper(0, n, delta) > alpha:
        return None
    lo, hi = 0, n - 1
    while lo < hi:  # largest k with upper <= alpha (monotone in k)
        mid = (lo + hi + 1) // 2
        if _cp_upper(mid, n, delta) <= alpha:
            lo = mid
        else:
            hi = mid - 1
    return kth_largest(cal.cal_sorted, min(lo + 1, n))


def threshold_id_bootstrap(cal: Calibration, alpha: float, delta: float = 0.05, B: int = 200,
                           seed: int = 0, chunk_pairs: int = 1 << 22) -> torch.Tensor | None:
    """Identity-level (subsets) bootstrap threshold, the practitioner's dependence-aware
    recipe (Bolle et al. 2004; Fogliato et al. 2023).  Resample the m calibration identities
    with replacement; a resample is a multinomial weight vector w, under which pair (i, j),
    i != j, has weight w_i w_j (pairs within one identity are genuine and excluded).  For
    each resample take the threshold t_b whose weighted exceedance fraction is <= alpha;
    return the (1 - delta)-quantile of t_b (a percentile-bootstrap upper bound).  Not valid in
    finite samples; asymptotically motivated.  None means the rule rejects everything."""
    m = cal.m
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, m, (B, m), generator=g)
    w = torch.zeros(B, m, dtype=torch.float64).scatter_add_(1, idx, torch.ones(B, m, dtype=torch.float64))
    w = w.to(cal.cal_sorted.device)
    n = cal.n_pairs
    # weights of sorted pairs, processed from the largest score downwards
    tb = torch.empty(B, dtype=cal.cal_sorted.dtype, device=w.device)
    done = torch.zeros(B, dtype=torch.bool, device=w.device)
    total = (w.sum(1) ** 2 - (w * w).sum(1)) / 2                     # sum_{i<j} w_i w_j
    budget = alpha * total
    acc = torch.zeros(B, dtype=torch.float64, device=w.device)
    for hi in range(n, 0, -chunk_pairs):
        lo = max(0, hi - chunk_pairs)
        pi, pj = cal.pair_i[lo:hi].flip(0), cal.pair_j[lo:hi].flip(0)   # descending scores
        pw = w[:, pi] * w[:, pj]                                        # (B, chunk)
        cum = acc[:, None] + torch.cumsum(pw, dim=1)
        # first position where the cumulative weight strictly exceeds the budget
        over = cum > budget[:, None]
        has = over.any(1) & ~done
        first = torch.argmax(over.int(), dim=1)
        vals = cal.cal_sorted[lo:hi].flip(0)
        tb[has] = vals[first[has]]
        done |= has
        acc = cum[:, -1]
        if bool(done.all()):
            break
    if not bool(done.all()):
        return None
    # accept s > t_b admits weight <= budget; conservative (1-delta) quantile over resamples
    q = torch.quantile(tb.double(), 1 - delta, interpolation="higher")
    return q.to(cal.cal_sorted.dtype)


def accept_threshold(s_star: torch.Tensor, t: torch.Tensor | None) -> torch.Tensor:
    if t is None:
        return torch.zeros_like(s_star, dtype=torch.bool)
    return s_star > t


THRESHOLD_RULES = {
    "u_conformal": threshold_u_conformal,
    "disjoint": threshold_disjoint,
    "naive": threshold_naive,
    "naive_cp": threshold_naive_cp,
    "disjoint_cp": threshold_disjoint_cp,
    "id_bootstrap": threshold_id_bootstrap,
}


def cp_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(_beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def resolution(m: int) -> float:
    """Smallest non-trivial level of the transductive test: 1/N."""
    return 1.0 / n_aug_pairs(m)


def inductive_floor(m: int) -> float:
    """Smallest level at which Prop 1's inductive rule is non-trivial."""
    return (2 * m + 1) / n_aug_pairs(m)


def lower_bound_nontrivial(m: int, alpha: float, rho: float = 1.0) -> float:
    """Theorem 1(a) of the paper: an inductive rule valid under every identity-exchangeable
    law with rho-hubs is non-trivial with probability at most (m+2) alpha / (2 rho)."""
    return min(1.0, (m + 2) * alpha / (2 * rho))


def per_probe_far_bound(m: int, alpha: float) -> float:
    """Proposition (per-probe control): P(COVE accepts | probe) <= floor(alpha N)/(m+1)."""
    return rank_budget(alpha, n_aug_pairs(m)) / (m + 1)


__all__ = [
    "Calibration", "build_calibration", "cove_rank", "cove_accept", "count_ge",
    "threshold_u_conformal", "threshold_disjoint", "threshold_naive",
    "threshold_naive_cp", "accept_threshold", "THRESHOLD_RULES", "n_aug_pairs",
    "cp_interval", "resolution", "inductive_floor", "lower_bound_nontrivial", "per_probe_far_bound",
]
