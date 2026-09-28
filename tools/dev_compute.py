"""Total logged GPU time of development runs that are not reported: pilots, runs superseded
by defect fixes, and the complete run replaced by the clean rerun.  Identical records kept
in two archives are counted once.  Writes audits/development_compute.json, which the paper's
compute statement reads (the development records themselves are not shipped).
Usage: python tools/dev_compute.py
"""
import glob
import hashlib
import json

ROOTS = ["results/raw/pilot", "results/raw/pilot2", "results/raw/pilot_random",
         "results/superseded", "results/pre_clean_rerun"]


def main():
    seen, sec, timed, untimed = set(), 0.0, 0, 0
    for root in ROOTS:
        for p in sorted(glob.glob(f"{root}/**/*.json", recursive=True)):
            raw = open(p, "rb").read()
            h = hashlib.sha256(raw).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("seconds") is None:
                untimed += 1
            else:
                timed += 1
                sec += float(rec["seconds"])
    out = {"gpu_hours": sec / 3600, "timed_records": timed, "untimed_records": untimed,
           "roots": ROOTS, "deduplicated_by": "sha256 of file content"}
    json.dump(out, open("audits/development_compute.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
