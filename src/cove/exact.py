"""Exact integer similarity arithmetic.

The validity theorem (Theorem 2 of the paper) is a statement about ranks of a *fixed symmetric
similarity function*.  Floating-point matrix products are not a fixed function:
the same pair can round differently depending on the kernel's blocking, which
would silently break the exchangeability argument at the level of ties.

We therefore quantize every L2-normalised embedding once, with a single scale per
(dataset, encoder), to integer vectors q with ||q||^2 < 2^24.  Every product
q_i r_i and, by Cauchy-Schwarz, every partial sum of a dot product is then an
integer of magnitude below 2^24, which IEEE float32 represents exactly -- so any
float32 matmul (any summation order, blocking or split-K) returns the exact
integer.  TF32 and reduced-precision matmul modes are disabled and asserted off,
because TF32 is exact only for |q_i| <= 2048.  The quantized cosine is the
similarity function the guarantee is about.
"""
from __future__ import annotations

import math

import numpy as np
import torch

FP32_EXACT = 2**24

# exactness requires true float32 products and accumulation
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def _assert_fp32_mode() -> None:
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("exact similarity requires float32 matmul precision 'highest' and TF32 off")


def quantize(feats: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float, int]:
    """Quantize L2-normalised rows to integer vectors (stored as float32).

    Returns (q, scale, qmax) with q = round(f * scale) and max ||q||^2 < 2^24.
    ``scale`` is computed once per (dataset, encoder) over *all* splits so that
    calibration and test units share one similarity function.
    """
    f = np.asarray(feats, dtype=np.float64)
    d = f.shape[1]
    if scale is None:
        # ||round(f s)|| <= s ||f|| + sqrt(d)/2 must stay below 2^12
        scale = (math.sqrt(FP32_EXACT) - 1 - math.sqrt(d) / 2) / float(np.max(np.linalg.norm(f, axis=1)))
    q = np.rint(f * scale)
    sq = np.einsum("ij,ij->i", q, q)
    if sq.max() >= FP32_EXACT:
        raise ValueError("quantized norm too large for exact float32 dot products")
    return q.astype(np.float32), float(scale), int(np.abs(q).max())


def exact_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact integer similarities a @ b.T returned as int64."""
    _assert_fp32_mode()
    out = a.float() @ b.float().T
    return torch.round(out).to(torch.int64)


def exact_rowdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact integer row-wise dot products <a_i, b_i> as int64."""
    return torch.round((a.float() * b.float()).sum(dim=1)).to(torch.int64)


def check_exact(q: torch.Tensor, n: int = 512, m: int = 3000, seed: int = 0) -> None:
    """Canary on production-like shapes: fp32 GEMM and row-wise products must equal an
    int64 elementwise reference, off-diagonal entries included."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ia = torch.randint(0, q.shape[0], (min(n, q.shape[0]),), generator=g).to(q.device)
    ib = torch.randint(0, q.shape[0], (min(m, q.shape[0]),), generator=g).to(q.device)
    A, B = q[ia], q[ib]
    gemm = exact_sim(A, B)
    ref = torch.zeros_like(gemm)
    Al, Bl = A.long(), B.long()
    for j in range(0, B.shape[0], 64):  # int64 elementwise reference, no BLAS
        ref[:, j:j + 64] = (Al[:, None, :] * Bl[None, j:j + 64, :]).sum(-1)
    k = min(A.shape[0], B.shape[0])
    rd = exact_rowdot(A[:k], B[:k])
    if not torch.equal(gemm, ref) or not torch.equal(rd, torch.diagonal(ref[:k, :k])):
        raise RuntimeError("exact integer similarity canary failed")
