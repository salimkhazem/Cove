"""Cache L2-normalised frozen embeddings: features/<dataset>/<encoder>.npz (+ .json)."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cove.data import decode, load  # noqa: E402
from cove.provenance import run_record  # noqa: E402

ENCODERS = {
    "dinov2_s": ("vit_small_patch14_dinov2.lvd142m", {"img_size": 224}),
    "dinov2_b": ("vit_base_patch14_dinov2.lvd142m", {"img_size": 224}),
    "dinov2_l": ("vit_large_patch14_dinov2.lvd142m", {"img_size": 224}),
    "clip_b16": ("vit_base_patch16_clip_224.openai", {}),
    "clip_l14": ("vit_large_patch14_clip_224.openai", {}),
    "siglip_b16": ("vit_base_patch16_siglip_224.webli", {}),
    "resnet50": ("resnet50.a1_in1k", {}),
    "convnext_b": ("convnext_base.fb_in22k_ft_in1k", {}),
}


class ArrowImages(Dataset):
    def __init__(self, table, col, tf):
        self.col = table.column(col)
        self.tf = tf

    def __len__(self):
        return len(self.col)

    def __getitem__(self, i):
        return self.tf(decode(self.col[i].as_py()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    table, col, y, pool, meta = load(args.dataset)
    out_dir = os.path.join("features", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    for enc in args.encoders:
        name, kw = ENCODERS[enc]
        t0 = time.time()
        model = timm.create_model(name, pretrained=True, num_classes=0, **kw).eval().cuda()
        override = {}
        if "img_size" in kw:  # DINOv2 evaluated at 224 (resize 256, centre crop 224)
            override = {"input_size": (3, kw["img_size"], kw["img_size"]), "crop_pct": 0.875}
        cfg = timm.data.resolve_data_config(override, model=model)
        tf = timm.data.create_transform(**cfg)
        dl = DataLoader(ArrowImages(table, col, tf), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=True)
        feats = []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for x in dl:
                f = model(x.cuda(non_blocking=True)).float()
                feats.append(torch.nn.functional.normalize(f, dim=1).cpu())
        f = torch.cat(feats).numpy().astype(np.float32)
        path = os.path.join(out_dir, f"{enc}.npz")
        np.savez(path, feats=f.astype(np.float16), labels=y, pool=pool)
        rec = run_record(vars(args))
        rec.update({"dataset": args.dataset, "encoder": enc, "timm_name": name, "timm_kwargs": kw,
                    "data_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                    "n": int(f.shape[0]), "d": int(f.shape[1]),
                    "n_ids": {"cal": int(len(np.unique(y[pool == 0]))), "test": int(len(np.unique(y[pool == 1])))},
                    "sha256_feats": hashlib.sha256(f.astype(np.float16).tobytes()).hexdigest(),
                    "seconds": time.time() - t0,
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "meta": meta})
        with open(path.replace(".npz", ".json"), "w") as fh:
            json.dump(rec, fh, indent=2)
        print(json.dumps({k: rec[k] for k in ("dataset", "encoder", "n", "d", "n_ids", "seconds", "peak_mem_gb")}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
