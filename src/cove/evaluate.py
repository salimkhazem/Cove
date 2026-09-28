"""Replicate engine: one calibration draw, many test trials, every rule.

A *pool* holds quantized unit embeddings with identity labels.  The calibration
pool and the test pool contain disjoint identities (A1).  Each replicate:

1. draws m distinct calibration identities and one image of each;
2. draws impostor trials (two test images of different identities) and genuine
   trials (two distinct images of one test identity);
3. evaluates COVE and every threshold rule at every alpha.

Reported per replicate and rule: FAR = accepted impostor trials / impostor
trials, TAR = accepted genuine trials / genuine trials.  The mean of FAR over
replicates estimates the marginal false-accept probability that Thm 2 bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from . import conformal as cf
from .exact import exact_rowdot, exact_sim
from .scores import UnitStats, subset, transform


@dataclass
class Pool:
    q: torch.Tensor                  # (n, d) quantized embeddings (float32 ints) on device
    labels: np.ndarray               # (n,) identity ids
    stats: UnitStats | None = None   # external-cohort statistics, aligned with rows
    by_id: dict = field(default_factory=dict)

    def __post_init__(self):
        order = np.argsort(self.labels, kind="stable")
        ids, starts, counts = np.unique(self.labels[order], return_index=True, return_counts=True)
        self.members = order                 # images grouped by identity
        self.id_list = ids.astype(np.int64)
        self.starts = starts.astype(np.int64)
        self.counts = counts.astype(np.int64)
        self.by_id = {int(i): order[s:s + c] for i, s, c in zip(ids, starts, counts)}

    @property
    def ids(self) -> np.ndarray:
        return self.id_list

    def random_image(self, id_pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One uniformly random image of each identity (given by position in id_list)."""
        off = np.floor(rng.random(len(id_pos)) * self.counts[id_pos]).astype(np.int64)
        return self.members[self.starts[id_pos] + off]


def _allowed_positions(pool: Pool, exclude: np.ndarray | None, min_count: int = 1) -> np.ndarray:
    ok = pool.counts >= min_count
    if exclude is not None and len(exclude):
        ok &= ~np.isin(pool.id_list, exclude)
    return np.flatnonzero(ok)


def sample_units(pool: Pool, m: int, rng: np.random.Generator,
                 exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """m distinct identities uniformly at random (not in ``exclude``), one uniform image each."""
    pos = _allowed_positions(pool, exclude)
    if m > pos.size:
        raise ValueError(f"m={m} exceeds {pos.size} available identities")
    chosen = rng.choice(pos, size=m, replace=False)
    return pool.random_image(chosen, rng), pool.id_list[chosen]


def sample_impostors(pool: Pool, n: int, rng: np.random.Generator,
                     exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """n impostor trials sampled exactly like calibration units (A1): two distinct
    identities uniformly at random (not in ``exclude``), one uniform image each."""
    pos = _allowed_positions(pool, exclude)
    i = rng.integers(0, pos.size, size=n)
    j = rng.integers(0, pos.size - 1, size=n)
    j = j + (j >= i)                     # uniform over identities distinct from i
    return pool.random_image(pos[i], rng), pool.random_image(pos[j], rng)


def sample_genuines(pool: Pool, n: int, rng: np.random.Generator,
                    exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """n genuine trials: a uniformly random identity with >= 2 images, two distinct images."""
    pos = _allowed_positions(pool, exclude, min_count=2)
    if pos.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    p = pos[rng.integers(0, pos.size, size=n)]
    c = pool.counts[p]
    oa = np.floor(rng.random(n) * c).astype(np.int64)
    ob = np.floor(rng.random(n) * (c - 1)).astype(np.int64)
    ob = ob + (ob >= oa)
    return pool.members[pool.starts[p] + oa], pool.members[pool.starts[p] + ob]


def _pair_scores(kind, qa, qb, sa, sb, k):
    return transform(kind, exact_rowdot(qa, qb), sa, sb, k)


def trial_ranks(cal: cf.Calibration, cal_stats, test: Pool, a: np.ndarray, b: np.ndarray,
                kind: str, k: int, chunk: int = 16384) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (s*, R) for trials (a, b): trial score and COVE rank."""
    dev = test.q.device
    s_all, r_all = [], []
    for i in range(0, len(a), chunk):
        ia = torch.as_tensor(a[i:i + chunk], device=dev)
        ib = torch.as_tensor(b[i:i + chunk], device=dev)
        qa, qb = test.q[ia], test.q[ib]
        sa, sb = subset(test.stats, ia), subset(test.stats, ib)
        s = _pair_scores(kind, qa, qb, sa, sb, k)
        ca = transform(kind, exact_sim(qa, cal.units), sa, cal_stats, k)
        cb = transform(kind, exact_sim(qb, cal.units), sb, cal_stats, k)
        s_all.append(s)
        r_all.append(cf.cove_rank(cal, s, ca, cb))
    return torch.cat(s_all), torch.cat(r_all)


@dataclass
class ReplicateResult:
    far: dict      # (rule, alpha) -> float
    tar: dict
    trivial: dict  # (rule, alpha) -> bool
    far_pop: dict = field(default_factory=dict)  # threshold rules: FAR given this calibration draw


def run_replicate(cal_pool: Pool, test: Pool, m: int, alphas: list[float],
                  n_imp: int, n_gen: int, rng: np.random.Generator,
                  kind: str = "raw", k: int = 10,
                  oracle_t: dict | None = None, random_split: bool = False,
                  pop_sorted=None,
                  with_bootstrap: bool = False) -> ReplicateResult:
    """``random_split``: cal_pool and test are the same pool; calibration identities
    are drawn afresh and trials use only the remaining identities (exactly A1).
    Otherwise calibration and test pools are fixed identity-disjoint sets."""
    dev = cal_pool.q.device
    units_idx, cal_ids = sample_units(cal_pool, m, rng)
    idx = torch.as_tensor(units_idx, device=dev)
    units = cal_pool.q[idx]
    ustats = subset(cal_pool.stats, idx)
    pair = transform(kind, exact_sim(units, units), ustats, ustats, k)
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    cal = cf.build_calibration(units, pair, gen)

    excl = cal_ids if random_split else None
    ai, bi = sample_impostors(test, n_imp, rng, excl)
    ag, bg = sample_genuines(test, n_gen, rng, excl)
    s_i, r_i = trial_ranks(cal, ustats, test, ai, bi, kind, k)
    s_g, r_g = trial_ranks(cal, ustats, test, ag, bg, kind, k) if len(ag) else (None, None)

    far, tar, triv, far_pop = {}, {}, {}, {}
    pop_keep = None
    if pop_sorted is not None:
        if isinstance(pop_sorted, tuple):
            ps, ida, idb = pop_sorted
            if random_split:
                cid = torch.as_tensor(np.asarray(cal_ids), dtype=torch.int64, device=ps.device)
                pop_keep = ps[~(torch.isin(ida, cid) | torch.isin(idb, cid))]
            else:
                pop_keep = ps
        else:
            pop_keep = pop_sorted
    for alpha in alphas:
        acc_i = cf.cove_accept(r_i, m, alpha)
        far[("cove", alpha)] = acc_i.double().mean().item()
        tar[("cove", alpha)] = cf.cove_accept(r_g, m, alpha).double().mean().item() if s_g is not None else float("nan")
        triv[("cove", alpha)] = cf.rank_budget(alpha, cf.n_aug_pairs(m)) < 1
        # pairs-as-exchangeable conformal p-value (ignores cross pairs): not valid
        kk = cf.rank_budget(alpha, cal.n_pairs + 1)
        far[("pairs_iid", alpha)] = ((1 + cf.count_ge(cal.cal_sorted, s_i)) <= kk).double().mean().item()
        tar[("pairs_iid", alpha)] = ((1 + cf.count_ge(cal.cal_sorted, s_g)) <= kk).double().mean().item() if s_g is not None else float("nan")
        triv[("pairs_iid", alpha)] = kk < 1
        for name, rule in cf.THRESHOLD_RULES.items():
            if name == "id_bootstrap" and (cal.m > 1000 or not with_bootstrap):
                continue   # O(B m^2) per level: evaluated only in the bootstrap tier, m <= 1000
            t = rule(cal, alpha)
            far[(name, alpha)] = cf.accept_threshold(s_i, t).double().mean().item()
            tar[(name, alpha)] = cf.accept_threshold(s_g, t).double().mean().item() if s_g is not None else float("nan")
            triv[(name, alpha)] = t is None
            if pop_keep is not None:
                # calibration-conditional FAR of a threshold rule: tail, above the
                # threshold, of the population impostor scores whose two identities
                # both remain after this calibration draw
                far_pop[(name, alpha)] = 0.0 if t is None else float(
                    (pop_keep.numel() - torch.searchsorted(pop_keep, t.reshape(1).to(pop_keep.dtype), side="right")).item()
                    / pop_keep.numel())
        if oracle_t is not None and alpha in oracle_t:
            t = oracle_t[alpha]
            far[("oracle", alpha)] = (s_i > t).double().mean().item()
            tar[("oracle", alpha)] = (s_g > t).double().mean().item() if s_g is not None else float("nan")
            triv[("oracle", alpha)] = False
    return ReplicateResult(far, tar, triv, far_pop)


def oracle_thresholds(test: Pool, alphas: list[float], n: int, rng: np.random.Generator,
                      kind: str = "raw", k: int = 10, chunk: int = 1 << 17):
    """Population FAR-alpha thresholds of the score, estimated from n test impostor pairs.

    Returns (thresholds, population) where population = (sorted scores, identity of side a,
    identity of side b), aligned, so that the calibration-conditional FAR of a threshold can
    be computed over the identities that remain after a calibration draw."""
    a, b = sample_impostors(test, n, rng)
    dev = test.q.device
    vals = []
    for i in range(0, n, chunk):
        ia = torch.as_tensor(a[i:i + chunk], device=dev)
        ib = torch.as_tensor(b[i:i + chunk], device=dev)
        vals.append(_pair_scores(kind, test.q[ia], test.q[ib], subset(test.stats, ia),
                                 subset(test.stats, ib), k).cpu())
    v, order = torch.sort(torch.cat(vals), stable=True)
    ida = torch.as_tensor(test.labels[a], dtype=torch.int64)[order]
    idb = torch.as_tensor(test.labels[b], dtype=torch.int64)[order]
    out = {}
    for alpha in alphas:
        j = max(1, math.floor(alpha * n))
        out[alpha] = v[n - j].to(dev)  # accept s > t admits < j pairs
    return out, (v.to(dev), ida.to(dev), idb.to(dev))
