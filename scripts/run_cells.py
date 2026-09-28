"""Evaluate every rule on real features: one JSON per (dataset, encoder, score, m).

Output rows: one per (replicate, rule, alpha) with FAR, TAR, trivial.  All rules
share the same calibration draw and the same trials within a replicate, so
comparisons are paired.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cove.evaluate import Pool, oracle_thresholds, run_replicate  # noqa: E402
from cove.exact import check_exact, quantize  # noqa: E402
from cove.provenance import atomic_json, run_record  # noqa: E402
from cove.scores import cohort_stats  # noqa: E402

ALPHAS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]


def make_pools(path, device, cohort_frac, cohort_seed=7, k=10, need_stats=False, merged=False,
               category_split=False):
    z = np.load(path)
    f, y, pool = z["feats"].astype(np.float32), z["labels"], z["pool"]
    q, scale, Q = quantize(f)
    qt = torch.as_tensor(q, device=device)
    check_exact(qt)
    cal_mask = pool == 0
    ids = np.unique(y[cal_mask])
    rng = np.random.default_rng(cohort_seed)
    n_coh = int(round(cohort_frac * len(ids)))
    coh_ids = rng.choice(ids, size=n_coh, replace=False) if n_coh else np.array([], ids.dtype)
    coh_mask = cal_mask & np.isin(y, coh_ids)
    draw_mask = cal_mask & ~coh_mask
    test_mask = pool == 1
    stats_all = None
    if need_stats:
        coh_idx = torch.as_tensor(np.flatnonzero(coh_mask), device=device)
        stats_all = cohort_stats(qt, qt[coh_idx], k=k)
    def sub(mask):
        idx = torch.as_tensor(np.flatnonzero(mask), device=device)
        st = None
        if stats_all is not None:
            from cove.scores import subset
            st = subset(stats_all, idx)
        return Pool(q=qt[idx], labels=y[mask], stats=st)
    if category_split:  # adverse (A1 violated): calibrate on 6 SOP super-categories, test on the other 6
        meta = np.load(os.path.join(os.path.dirname(path), "meta.npz"))
        cat = meta["category"]
        assert np.array_equal(meta["labels"], y)
        draw_mask = (cat < 6) & ~coh_mask
        test_mask = (cat >= 6) & ~coh_mask
    if merged:  # random identity split per replicate: one pool of all non-cohort identities
        draw_mask = draw_mask | test_mask
        test_mask = draw_mask
    info = {"scale": scale, "Q": Q, "d": int(f.shape[1]), "n_cohort_ids": int(n_coh),
            "n_cal_ids": int(len(np.unique(y[draw_mask]))), "n_test_ids": int(len(np.unique(y[test_mask]))),
            "n_test_imgs": int(test_mask.sum())}
    return sub(draw_mask), sub(test_mask), info


def cap_memory(gb: float = 24.0) -> None:
    """Keep every run within a 24 GB budget so it reproduces on an RTX 3090."""
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total))


def main():
    cap_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--score", default="raw", choices=["raw", "csls", "snorm"])
    ap.add_argument("--ms", type=int, nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-imp", type=int, default=200_000)
    ap.add_argument("--n-gen", type=int, default=20_000)
    ap.add_argument("--n-oracle", type=int, default=50_000_000)
    ap.add_argument("--cohort-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--split", default="random", choices=["random", "fixed", "category"])
    ap.add_argument("--with-bootstrap", action="store_true", help="also evaluate the identity bootstrap")
    args = ap.parse_args()
    dev = "cuda"
    cal_pool, test, info = make_pools(f"features/{args.dataset}/{args.encoder}.npz", dev,
                                      args.cohort_frac, need_stats=args.score != "raw",
                                      merged=args.split == "random",
                                      category_split=args.split == "category")
    import hashlib
    key = int(hashlib.sha1(f'{args.dataset}|{args.encoder}|{args.score}'.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([args.seed, key])
    orc, pop = oracle_thresholds(test, ALPHAS, args.n_oracle, rng, kind=args.score)
    for m in args.ms:
        if m > info["n_cal_ids"] - (100 if args.split == "random" else 0):
            print(f"skip m={m} > {info['n_cal_ids']} calibration identities", flush=True)
            continue
        t0 = time.time()
        rows = []
        for r in range(args.reps):
            res = run_replicate(cal_pool, test, m, ALPHAS, args.n_imp, args.n_gen,
                                np.random.default_rng([args.seed, m, r]), kind=args.score,
                                oracle_t=orc, pop_sorted=pop, random_split=args.split == "random",
                                with_bootstrap=args.with_bootstrap)
            for (rule, a), v in res.far.items():
                rows.append({"rep": r, "rule": rule, "alpha": a, "far": v,
                             "tar": res.tar[(rule, a)], "trivial": bool(res.trivial[(rule, a)]),
                             "far_pop": res.far_pop.get((rule, a))})
        rec = run_record(vars(args))
        rec.update({"dataset": args.dataset, "encoder": args.encoder, "score": args.score, "m": m,
                    "split": args.split,
                    "info": info, "seconds": time.time() - t0,
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "oracle_thresholds": {str(a): float(t) for a, t in orc.items()}, "rows": rows})
        out = f"results/raw/{args.tag}/{args.dataset}__{args.encoder}__{args.score}__{args.split}__m{m}.json"
        atomic_json(rec, out)
        # compact console summary: mean FAR/alpha and mean TAR per rule at a few alphas
        import pandas as pd
        df = pd.DataFrame(rows)
        g = df.groupby(["rule", "alpha"]).agg(far=("far", "mean"), tar=("tar", "mean"),
                                              triv=("trivial", "mean")).reset_index()
        g["far/a"] = g["far"] / g["alpha"]
        sel = g[g.alpha.isin([1e-3, 1e-4, 1e-5])].pivot(index="rule", columns="alpha",
                                                        values=["far/a", "tar"])
        print(f"== {args.dataset} {args.encoder} {args.score} m={m} ({time.time()-t0:.0f}s)\n{sel.round(3)}", flush=True)


if __name__ == "__main__":
    main()
