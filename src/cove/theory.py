"""Measurable quantities from the theory: zoo index, hubness, predicted dispersion."""
from __future__ import annotations

import math

import numpy as np
import torch

from .exact import exact_sim


def identity_uniform_indices(labels: np.ndarray, n: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Draw n units by selecting identities uniformly, then one image per identity.

    Sampling is with replacement across draws; it targets the same finite-population
    unit law as the calibration and impostor-trial samplers.
    """
    order = np.argsort(labels, kind="stable")
    _, starts, counts = np.unique(labels[order], return_index=True, return_counts=True)
    id_pos = rng.integers(0, len(starts), size=n)
    offsets = np.floor(rng.random(n) * counts[id_pos]).astype(np.int64)
    return order[starts[id_pos] + offsets]


def per_unit_impostor_rates(q: torch.Tensor, labels: np.ndarray, thresholds: list,
                            n_probe: int, n_ref: int, rng: np.random.Generator,
                            chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Estimate h(u; t) under identity-uniform probe and reference sampling.

    Returns (h (n_probe, T), n_eff (n_probe,)). References with the same identity
    as the probe are excluded, matching impostor trials.
    """
    probes = identity_uniform_indices(labels, n_probe, rng)
    refs = identity_uniform_indices(labels, n_ref, rng)
    lab = torch.as_tensor(labels, device=q.device)
    th = torch.as_tensor(np.asarray([float(t) for t in thresholds]), device=q.device)
    hs, ns = [], []
    ref_q = q[torch.as_tensor(refs, device=q.device)]
    ref_l = lab[torch.as_tensor(refs, device=q.device)]
    for i in range(0, len(probes), chunk):
        pi = torch.as_tensor(probes[i:i + chunk], device=q.device)
        s = exact_sim(q[pi], ref_q).double()                  # (c, n_ref)
        other = (lab[pi][:, None] != ref_l[None, :])
        cnt = other.sum(1)
        exc = torch.stack([((s >= t) & other).sum(1) for t in th], dim=1)
        hs.append((exc.double() / cnt[:, None]).cpu())
        ns.append(cnt.cpu())
    return torch.cat(hs).numpy(), torch.cat(ns).numpy()


def zoo_index(h: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    """kappa = Var(h(U; t)) / Gbar(t)^2, with the binomial noise of each h removed."""
    g = h.mean(axis=0)
    noise = np.mean(h * (1 - h) / n_eff[:, None], axis=0)
    var = np.maximum(h.var(axis=0, ddof=1) - noise, 0.0)
    return var / np.maximum(g, 1e-300) ** 2


def predicted_rel_se(kappa: float, m: int, gbar: float) -> float:
    """Prop 2: relative s.e. of the calibration exceedance fraction at level gbar."""
    return math.sqrt((4 * (m - 2) * kappa * gbar**2 + 2 * gbar * (1 - gbar)) / (m * (m - 1))) / gbar


def k_occurrence_skewness(q: torch.Tensor, n: int, k: int, rng: np.random.Generator) -> float:
    """Hubness: skewness of the k-occurrence distribution among n random units."""
    idx = torch.as_tensor(rng.choice(q.shape[0], size=min(n, q.shape[0]), replace=False), device=q.device)
    s = exact_sim(q[idx], q[idx]).double()
    s.fill_diagonal_(-float("inf"))
    nn = torch.topk(s, k, dim=1).indices
    occ = torch.bincount(nn.flatten(), minlength=len(idx)).double().cpu().numpy()
    mu, sd = occ.mean(), occ.std()
    return float(np.mean((occ - mu) ** 3) / sd**3)
