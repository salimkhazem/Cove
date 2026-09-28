"""Minimal user-facing API.

    from cove.api import Verifier
    v = Verifier(calib_embeddings)            # one L2-normalised embedding per calibration identity
    accept, p = v.decide(probe, reference, alpha=1e-5)

Guarantee (paper, Theorem 2): if the calibration identities, the probe identity and the
reference identity are exchangeable draws from the same population of identities (and the
probe's identity is not among the calibration identities), then for an impostor trial
P(accept) <= floor(alpha * N) / N with N = C(m+2, 2).  The guarantee is marginal over the
calibration draw and the trial.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import conformal as cf
from .exact import FP32_EXACT, exact_sim, quantize


class Verifier:
    def __init__(self, calib: np.ndarray, device: str | None = None):
        f = np.asarray(calib, dtype=np.float64)
        f = f / np.linalg.norm(f, axis=1, keepdims=True)
        # a fixed, data-independent scale, so that the score g is the same function for every
        # calibration set; rows are unit-norm up to rounding, hence ||round(f * scale)|| < 2^12
        scale = (math.sqrt(FP32_EXACT) - 1 - math.sqrt(f.shape[1]) / 2) / (1 + 2**-20)
        q, self.scale, _ = quantize(f, scale=scale)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.units = torch.as_tensor(q, device=self.device)
        self.cal = cf.build_calibration(self.units, exact_sim(self.units, self.units))
        self.m = self.units.shape[0]

    def _q(self, x: np.ndarray) -> torch.Tensor:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        q = np.rint(x * self.scale)
        if np.max(np.einsum("ij,ij->i", q, q)) >= 2**24:
            raise ValueError("embedding norm exceeds the exact-arithmetic range")
        return torch.as_tensor(q.astype(np.float32), device=self.device)

    def _rank(self, probe: np.ndarray, reference: np.ndarray) -> torch.Tensor:
        a, b = self._q(probe), self._q(reference)
        s = torch.round((a * b).sum(1)).to(torch.int64)
        return cf.cove_rank(self.cal, s, exact_sim(a, self.units), exact_sim(b, self.units))

    def p_value(self, probe: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return (self._rank(probe, reference).double() / cf.n_aug_pairs(self.m)).cpu().numpy()

    def decide(self, probe: np.ndarray, reference: np.ndarray, alpha: float):
        r = self._rank(probe, reference)
        accept = cf.cove_accept(r, self.m, alpha).cpu().numpy()   # integer rank budget
        return accept, (r.double() / cf.n_aug_pairs(self.m)).cpu().numpy()
