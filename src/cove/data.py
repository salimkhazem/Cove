"""Dataset access with identity-disjoint calibration / test pools.

Every loader returns a list of shards; each shard is a (pyarrow table, image
column, identity column, pool) tuple, where pool is 0 for the calibration side
(calibration draws + external cohort) and 1 for the test side.  Identity-disjoint
splits follow the standard deep-metric-learning protocols where one exists.
"""
from __future__ import annotations

import glob
import hashlib
import io
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

DATA = os.environ.get("COVE_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
HF = os.path.expanduser("~/.cache/huggingface/hub")


def _snap(repo: str) -> str:
    d = os.path.join(HF, "datasets--" + repo.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(d, "*")))
    if not snaps:
        raise FileNotFoundError(f"{repo} not downloaded (run scripts/download_data.py)")
    return snaps[-1]


def _read_parquet(files):
    return pa.concat_tables([pq.read_table(f) for f in sorted(files)])


def _read_arrow(files):
    tabs = []
    for f in sorted(files):
        with pa.memory_map(f) as src:
            try:
                tabs.append(pa.ipc.open_stream(src).read_all())
            except pa.ArrowInvalid:
                tabs.append(pa.ipc.open_file(src).read_all())
    return pa.concat_tables(tabs)


def _col(t, name):
    return np.asarray(t.column(name).to_pylist())


def _inshop_item_key(value: str) -> str:
    """Full category-plus-item key, omitting only the image-specific suffix."""
    import re
    match = re.match(r"^(.+_id_\d+)(?:_|$)", value)
    if match is None:
        raise ValueError(f"unrecognized In-shop item_ID format: {value!r}")
    return match.group(1)


def load(name: str):
    """Return (table, image_column, identities (np.int64), pool (np.int8), meta)."""
    if name == "imagenet":
        root = glob.glob(f"{DATA}/Imagenet/ILSVRC___imagenet-1k/default/0.0.0/*/")[0]
        t = _read_arrow(glob.glob(root + "imagenet-1k-validation-*.arrow"))
        y = _col(t, "label").astype(np.int64)
        perm = np.random.default_rng(20260926).permutation(1000)
        pool = np.isin(y, perm[500:]).astype(np.int8)       # 500 / 500 classes
        return t, "image", y, pool, {"source": root, "protocol": "val, seeded 500/500 class split"}
    if name == "cars":
        files = glob.glob(f"{DATA}/stanford/train-*.parquet") + glob.glob(f"{DATA}/stanford/test-*.parquet")
        t = _read_parquet(files)
        y = _col(t, "label").astype(np.int64)
        return t, "image", y, (y >= 98).astype(np.int8), {"source": f"{DATA}/stanford", "protocol": "classes 0-97 / 98-195"}
    if name == "cub":
        s = _snap("Donghyun99/CUB-200-2011")
        t = _read_parquet(glob.glob(f"{s}/data/*.parquet"))
        y = _col(t, "label").astype(np.int64)
        return t, "image", y, (y >= 100).astype(np.int8), {"source": s, "protocol": "classes 0-99 / 100-199"}
    if name == "sop":
        s = _snap("nyris/stanford-online-products-v1")
        tr = _read_parquet(glob.glob(f"{s}/data/train-*.parquet"))
        te = _read_parquet(glob.glob(f"{s}/data/test-*.parquet"))
        t = pa.concat_tables([tr, te])
        y = _col(t, "item_id").astype(np.int64)
        pool = np.concatenate([np.zeros(tr.num_rows, np.int8), np.ones(te.num_rows, np.int8)])
        return t, "image", y, pool, {"source": s, "protocol": "official train / test items"}
    if name == "inshop":
        s = _snap("Marqo/deepfashion-inshop")
        t = _read_parquet(glob.glob(f"{s}/data/*.parquet"))
        # item_ID is per image ("MEN_Denim_id_00000080_01_1_front").
        # The item identity includes its category prefix: bare id_00000080 values
        # recur across categories and would merge unrelated products.
        raw = [_inshop_item_key(v) for v in t.column("item_ID").to_pylist()]
        ids = {v: i for i, v in enumerate(sorted(set(raw)))}
        y = np.array([ids[v] for v in raw], dtype=np.int64)
        pool = np.array([int(hashlib.sha1(str(v).encode()).hexdigest(), 16) % 2 for v in raw], np.int8)
        return t, "image", y, pool, {"source": s, "protocol": "full category-plus-item keys split by sha1 parity"}
    if name == "gldv2mini":
        s = _snap("zguo0525/google-landmarks-v2-mini")
        t = _read_parquet(glob.glob(f"{s}/data/*.parquet"))
        y = _col(t, "label").astype(np.int64)
        u = np.unique(y)
        perm = np.random.default_rng(20260926).permutation(u)
        pool = np.isin(y, perm[len(u) // 2:]).astype(np.int8)
        return t, "image", y, pool, {"source": s, "protocol": "landmarks, seeded half / half"}
    raise ValueError(name)


def decode(cell) -> Image.Image:
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b is None and cell.get("path"):
            return Image.open(cell["path"]).convert("RGB")
        return Image.open(io.BytesIO(b)).convert("RGB")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    raise TypeError(type(cell))
