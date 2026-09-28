"""Save per-row metadata needed by the adverse experiments (SOP super-category)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cove.data import load  # noqa: E402

if __name__ == "__main__":
    t, _, y, pool, _ = load("sop")
    cat = np.asarray(t.column("category").to_pylist())
    names, codes = np.unique(cat, return_inverse=True)
    os.makedirs("features/sop", exist_ok=True)
    np.savez("features/sop/meta.npz", category=codes, category_names=names, labels=y, pool=pool)
    print({str(n): int((codes == i).sum()) for i, n in enumerate(names)})
