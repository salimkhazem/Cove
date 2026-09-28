"""Pair score functions g(u, v) used inside the augmented array.

Thm 2 requires g to be a *fixed symmetric* function of the two units (it may
also depend on data that is independent of calibration and test, such as an
external cohort Z).  Each transform below is exactly symmetric in floating
point: integer arithmetic where possible, otherwise commutative addition of two
per-direction terms computed by the same expression.

* ``raw``   exact integer cosine (quantized).
* ``csls``  2K*s(u,v) - T_u - T_v, T_u = sum of u's top-K scores against Z
            (cross-domain similarity local scaling, integer form).
* ``snorm`` 0.5*[(s - mu_u)/sd_u + (s - mu_v)/sd_v], cohort statistics of each
            unit against Z (symmetric score normalisation).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .exact import exact_sim


@dataclass
class UnitStats:
    """Per-unit cohort statistics (all exact or deterministic functions of the unit)."""
    topk_sum: torch.Tensor | None = None   # int64
    mu: torch.Tensor | None = None         # float64
    sd: torch.Tensor | None = None         # float64


def cohort_stats(units: torch.Tensor, cohort: torch.Tensor, k: int = 10,
                 chunk: int = 8192) -> UnitStats:
    """Statistics of each unit's exact scores against a fixed external cohort."""
    tops, mus, sds = [], [], []
    n = cohort.shape[0]
    if n >= 2**14:  # keeps the int64 sum of squared scores (< 2^48 each) from overflowing
        raise ValueError("cohort too large for exact int64 second moments")
    for i in range(0, units.shape[0], chunk):
        s = exact_sim(units[i:i + chunk], cohort)            # int64, exact
        tops.append(torch.topk(s, k, dim=1).values.sum(dim=1))
        s1 = s.sum(dim=1)                                     # exact int64
        s2 = (s * s).sum(dim=1)                               # exact int64 (< 2^63)
        mu = s1.double() / n
        var = s2.double() / n - mu * mu
        mus.append(mu)
        sds.append(torch.sqrt(torch.clamp(var, min=1e-12)))
    return UnitStats(torch.cat(tops), torch.cat(mus), torch.cat(sds))


def transform(kind: str, s: torch.Tensor, stats_a: UnitStats | None, stats_b: UnitStats | None,
              k: int = 10) -> torch.Tensor:
    """Apply g to a block of exact scores.

    ``s`` has shape (A, B) (a block) or (T,) (row-aligned pairs).  ``stats_a`` and
    ``stats_b`` are the per-unit statistics aligned with rows / columns (or with
    the two sides of row-aligned pairs).
    """
    if kind == "raw":
        return s
    block = s.dim() == 2
    if kind == "csls":
        ta, tb = stats_a.topk_sum, stats_b.topk_sum
        if block:
            return 2 * k * s - ta[:, None] - tb[None, :]
        return 2 * k * s - ta - tb
    if kind == "snorm":
        sd_ = s.double()
        if block:
            za = (sd_ - stats_a.mu[:, None]) / stats_a.sd[:, None]
            zb = (sd_ - stats_b.mu[None, :]) / stats_b.sd[None, :]
        else:
            za = (sd_ - stats_a.mu) / stats_a.sd
            zb = (sd_ - stats_b.mu) / stats_b.sd
        return 0.5 * (za + zb)
    raise ValueError(f"unknown score transform {kind}")


def subset(stats: UnitStats | None, idx: torch.Tensor) -> UnitStats | None:
    if stats is None:
        return None
    return UnitStats(*(None if t is None else t[idx] for t in
                       (stats.topk_sum, stats.mu, stats.sd)))
