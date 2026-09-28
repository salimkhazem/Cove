"""Measured theory quantities per (dataset, encoder): zoo index kappa(alpha),
hub fraction rho*(alpha) = max_u h(u; tau_alpha), k-occurrence skewness, and the
per-trial decomposition behind Prop 3 (which genuine trials COVE refuses).

Output: results/raw/zoo/<dataset>__<encoder>.json
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cove import conformal as cf  # noqa: E402
from cove.evaluate import (oracle_thresholds, sample_genuines, sample_impostors,  # noqa: E402
                           sample_units, trial_ranks)
from cove.exact import exact_sim  # noqa: E402
from cove.provenance import atomic_json, run_record  # noqa: E402
from cove.theory import k_occurrence_skewness, per_unit_impostor_rates, zoo_index  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from run_cells import cap_memory, make_pools  # noqa: E402

ALPHAS = [1e-2, 1e-3, 1e-4, 1e-5]
PROP4_ALPHAS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]


def hub_direction_strength(pool, taus, rng, n_units=20000, k_list=(10, 50, 200)):
    """Descriptive directional cap occupancies, not bounds on the hub parameter.

    Evaluate approximate cosine thresholds converted from the quantized score scale
    on observed-unit, mean, and k-means candidate directions. Boundary mass and
    quantization prevent interpreting these measurements as rho_Q lower bounds.
    """
    q = pool.q
    idx = torch.as_tensor(rng.choice(q.shape[0], size=min(n_units, q.shape[0]), replace=False), device=q.device)
    x = torch.nn.functional.normalize(q[idx].double(), dim=1)
    scale2 = float((q[idx].double().norm(dim=1) ** 2).mean())   # integer score ~ scale^2 * cosine
    cos_t = [float(t) / scale2 for t in taus]
    cands = {"mean": torch.nn.functional.normalize(x.mean(0, keepdim=True), dim=1)}
    for k in k_list:  # a few Lloyd iterations of spherical k-means
        c = x[torch.as_tensor(rng.choice(x.shape[0], size=k, replace=False), device=x.device)]
        for _ in range(15):
            a = (x @ c.T).argmax(1)
            c = torch.nn.functional.normalize(torch.zeros_like(c).index_add_(0, a, x), dim=1)
        cands[f"kmeans{k}"] = c
    cands["units"] = x[:5000]
    out = {}
    for name, c in cands.items():
        sims = x @ c.T                                   # (n, n_cand)
        out[name] = [float((sims >= ct).double().mean(0).max()) for ct in cos_t]
    out["best"] = [max(out[n][i] for n in cands) for i in range(len(cos_t))]
    return out


def main():
    cap_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--n-probe", type=int, default=5000)
    ap.add_argument("--n-ref", type=int, default=40000)
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--n-gen", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    pool, _, info = make_pools(f"features/{args.dataset}/{args.encoder}.npz", "cuda", 0.2, merged=True)
    rng = np.random.default_rng([args.seed, 11])
    orc, pop = oracle_thresholds(pool, ALPHAS, 50_000_000, rng)
    taus = [orc[a] for a in ALPHAS]
    h, n_eff = per_unit_impostor_rates(pool.q, pool.labels, taus, args.n_probe,
                                       args.n_ref, rng)
    kappa = zoo_index(h, n_eff)
    rho_star = h.max(axis=0)
    rho_dir = hub_direction_strength(pool, taus, rng)
    skew = k_occurrence_skewness(pool.q, 10000, 10, rng)

    # Prop 3: per genuine trial, COVE vs oracle decision and the trial's cross-pair load
    m = min(args.m, info["n_cal_ids"] - 100)
    # smallest grid level at which COVE has at least 10 admissible ranks
    alpha = min(a for a in PROP4_ALPHAS if a * cf.n_aug_pairs(m) >= 10)
    t_orc = orc[alpha] if alpha in orc else oracle_thresholds(pool, [alpha], 20_000_000, rng)[0][alpha]
    rec_trials = {"cross": [], "cal": [], "cove": [], "oracle": []}
    for r in range(args.reps):
        rr = np.random.default_rng([args.seed, 12, r])
        units_idx, cal_ids = sample_units(pool, m, rr)
        units = pool.q[torch.as_tensor(units_idx, device="cuda")]
        cal = cf.build_calibration(units, exact_sim(units, units), torch.Generator().manual_seed(r))
        a, b = sample_genuines(pool, args.n_gen, rr, cal_ids)
        s, R = trial_ranks(cal, None, pool, a, b, "raw", 10)
        calc = cf.count_ge(cal.cal_sorted, s)
        cross = R - 1 - calc
        rec_trials["cross"].append(cross.cpu().numpy())
        rec_trials["cal"].append(calc.cpu().numpy())
        rec_trials["cove"].append(cf.cove_accept(R, m, alpha).cpu().numpy())
        rec_trials["oracle"].append((s > t_orc).cpu().numpy())
    cross = np.concatenate(rec_trials["cross"])
    cov = np.concatenate(rec_trials["cove"])
    ora = np.concatenate(rec_trials["oracle"])
    edges = [0, 1, 2, 3, 5, 10, 20, 50, 10**9]
    strata = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (cross >= lo) & (cross < hi) & ora
        strata.append({"cross_lo": lo, "cross_hi": hi, "n_oracle_accepts": int(sel.sum()),
                       "cove_accept_given_oracle": float(cov[sel].mean()) if sel.any() else None})
    rec = run_record(vars(args))
    rec.update({"dataset": args.dataset, "encoder": args.encoder, "info": info,
                "seconds": time.time() - t0,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                "alphas": ALPHAS, "tau": [float(t) for t in taus],
                "kappa": kappa.tolist(), "rho_star": rho_star.tolist(), "rho_dir": rho_dir,
                "h_quantiles": {str(a): np.quantile(h[:, i] / a, [0.5, 0.9, 0.99, 0.999, 1.0]).tolist()
                                for i, a in enumerate(ALPHAS)},
                "k_occurrence_skewness": skew,
                "prop4": {"m": m, "alpha": alpha, "n_trials": int(len(cross)),
                          "tar_cove": float(cov.mean()), "tar_oracle": float(ora.mean()),
                          "cove_and_not_oracle": float((cov & ~ora).mean()),
                          "oracle_and_not_cove": float((ora & ~cov).mean()), "strata": strata}})
    atomic_json(rec, f"results/raw/zoo/{args.dataset}__{args.encoder}.json")
    print(args.dataset, args.encoder, "kappa", np.round(kappa, 2), "rho*/alpha",
          np.round(rho_star / np.array(ALPHAS), 1), "rho_dir", {k: np.round(v, 4).tolist() for k, v in rho_dir.items()},
          "skew", round(skew, 2), flush=True)


if __name__ == "__main__":
    main()
