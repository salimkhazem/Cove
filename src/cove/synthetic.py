"""Controlled latent-identity models.

``hub_model``: identity centres on the sphere with a shared direction e.  A
fraction ``p_hub`` of identities are hubs (centres pulled toward e), so their
images are similar to everyone -- the "wolves" of the biometric zoo and the
"hubs" of high-dimensional retrieval.  Images add isotropic noise to the centre.

``wolf_scores``: the construction behind Thm 1 at the level of score arrays.
"""
from __future__ import annotations

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def hub_model(n_ids: int, n_img: int, d: int = 64, beta: float = 0.6,
              beta_hub: float = 3.0, p_hub: float = 0.0, sigma: float = 0.6,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features (n_ids*n_img, d) L2-normalised, labels, is_hub per identity)."""
    rng = rng or np.random.default_rng(0)
    e = np.zeros(d)
    e[0] = 1.0
    hub = rng.random(n_ids) < p_hub
    b = np.where(hub, beta_hub, beta)[:, None]
    centres = _normalize(b * e + rng.standard_normal((n_ids, d)) / np.sqrt(d) * 1.0)
    x = np.repeat(centres, n_img, axis=0) + sigma * rng.standard_normal((n_ids * n_img, d)) / np.sqrt(d)
    labels = np.repeat(np.arange(n_ids), n_img)
    return _normalize(x), labels, hub


def wolf_scores(n: int, p_wolf: float, rng: np.random.Generator,
                wolf_value: int = 10**6) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric integer score array for n i.i.d. units; wolves score ``wolf_value``
    against everyone, other pairs are i.i.d. uniform integers in [0, 10^5)."""
    s = rng.integers(0, 10**5, size=(n, n))
    s = np.triu(s, 1)
    s = s + s.T
    wolf = rng.random(n) < p_wolf
    s[wolf, :] = wolf_value
    s[:, wolf] = wolf_value
    np.fill_diagonal(s, 0)
    return s, wolf
