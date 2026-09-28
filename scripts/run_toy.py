"""Controlled validation of Theorems 1-2 and Propositions 1 and 3 on the hub model.

Same engine and rules as the real-data runs; identities are synthetic, so the
random-split protocol satisfies A1 exactly and the hub fraction is controlled.
Output: results/raw/toy/hub<p>__m<m>.json (same schema as run_cells.py).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cove.evaluate import Pool, oracle_thresholds, run_replicate  # noqa: E402
from cove.exact import check_exact, quantize  # noqa: E402
from cove.provenance import atomic_json, run_record  # noqa: E402
from cove.synthetic import hub_model  # noqa: E402

ALPHAS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]


def cap_memory(gb: float = 24.0) -> None:
    """Keep every run within a 24 GB budget so it reproduces on an RTX 3090."""
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total))


def main():
    cap_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-hub", type=float, nargs="+", default=[0.0, 0.01, 0.03])
    ap.add_argument("--ms", type=int, nargs="+", default=[30, 100, 300, 1000])
    ap.add_argument("--n-ids", type=int, default=20000)
    ap.add_argument("--n-img", type=int, default=3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-imp", type=int, default=200_000)
    ap.add_argument("--n-gen", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-tag", default="toy")
    args = ap.parse_args()
    for p_hub in args.p_hub:
        f, y, hub = hub_model(args.n_ids, args.n_img, d=args.d, p_hub=p_hub, sigma=args.sigma,
                              rng=np.random.default_rng([args.seed, int(p_hub * 1e6)]))
        q, scale, Q = quantize(f)
        qt = torch.as_tensor(q, device="cuda")
        check_exact(qt)
        pool = Pool(q=qt, labels=y)
        orc, pop = oracle_thresholds(pool, ALPHAS, 20_000_000, np.random.default_rng([args.seed, 1]))
        for m in args.ms:
            t0 = time.time()
            rows = []
            for r in range(args.reps):
                res = run_replicate(pool, pool, m, ALPHAS, args.n_imp, args.n_gen,
                                    np.random.default_rng([args.seed, m, r, int(p_hub * 1e6)]),
                                    oracle_t=orc, pop_sorted=pop, random_split=True)
                for (rule, a), v in res.far.items():
                    rows.append({"rep": r, "rule": rule, "alpha": a, "far": v,
                                 "tar": res.tar[(rule, a)], "trivial": bool(res.trivial[(rule, a)]),
                                 "far_pop": res.far_pop.get((rule, a))})
            rec = run_record(vars(args))
            rec.update({"dataset": f"toy_hub{p_hub}", "encoder": "synthetic", "score": "raw",
                        "split": "random", "m": m, "p_hub": p_hub, "n_hub_ids": int(hub.sum()),
                        "info": {"scale": scale, "Q": Q, "d": args.d}, "seconds": time.time() - t0,
                        "oracle_thresholds": {str(a): float(t) for a, t in orc.items()}, "rows": rows})
            atomic_json(rec, f"results/raw/{args.out_tag}/hub{p_hub}__m{m}.json")
            print(f"p_hub={p_hub} m={m} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
