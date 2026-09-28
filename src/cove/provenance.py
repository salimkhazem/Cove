"""Run records and atomic result writes."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_record(config: dict) -> dict:
    import numpy as np
    import torch
    rec = {
        "utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config": config,
        "argv": sys.argv,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain", "--untracked-files=no") not in ("", "unknown"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "host_hash": __import__("hashlib").sha1(socket.gethostname().encode()).hexdigest()[:8],
    }
    if torch.cuda.is_available():
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["driver_cuda"] = torch.cuda.get_device_capability(0)
    return rec


def _check_finite(obj, path="root"):
    if isinstance(obj, float) and not math.isfinite(obj) and not math.isnan(obj):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_finite(v, f"{path}[{i}]")


def atomic_json(obj, path: str) -> None:
    _check_finite(obj)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)
