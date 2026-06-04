#!/usr/bin/env python3
"""Survey data QUALITY and RANGE across candidate scroll volumes.

For each (scroll, volume) we read a cheap downsampled level and compute quality
metrics so we cache only good training data and skip junk (all-air regions,
low-contrast/artifacted volumes). Restoration target = finest available data, so
we prioritize the finest volumes and verify each actually has signal.

Metrics per volume (computed on a downsampled level, fast):
  - occupancy:   fraction of nonzero (papyrus vs air/padding)
  - contrast:    std of occupied voxels (texture richness; near-0 = flat/bad)
  - dyn_range:   p99-p1 of occupied voxels (usable intensity span)
  - p_lo/p_hi:   robust intensity percentiles (for per-volume normalization later)
A volume is "good" if occupancy and contrast clear thresholds.

    python scripts/survey_quality.py --level 4 --out /mnt/data/survey.json \
        --volumes scroll1=path1.zarr scroll2=path2.zarr ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superres.s3zarr import open_s3

BUCKET = "vesuvius-challenge-open-data"


def score_volume(scroll: str, zarr_path: str, level: int) -> dict:
    z = open_s3(BUCKET, scroll, zarr_path, level=level)
    full = z.read_region(0, 0, 0, *z.shape)  # downsampled level: small enough
    occ = float((full > 0).mean())
    nz = full[full > 0]
    if nz.size == 0:
        return {"scroll": scroll, "zarr": zarr_path, "shape_l0": None,
                "occupancy": 0.0, "good": False, "reason": "empty"}
    p1, p50, p99 = (float(x) for x in np.percentile(nz, [1, 50, 99]))
    contrast = float(nz.std())
    # level-0 shape = level shape * 2^level
    l0 = tuple(int(s * (2 ** level)) for s in z.shape)
    good = occ > 0.005 and contrast > 5.0 and (p99 - p1) > 20.0
    return {
        "scroll": scroll, "zarr": zarr_path, "level_scanned": level,
        "shape_l0": l0, "occupancy": round(occ, 4),
        "contrast_std": round(contrast, 2), "dyn_range": round(p99 - p1, 1),
        "p_lo": round(p1, 1), "p_hi": round(p99, 1), "median": round(p50, 1),
        "good": bool(good),
        "reason": "ok" if good else "low occupancy/contrast/range",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=4, help="downsampled level to scan")
    ap.add_argument("--out", default="survey.json")
    ap.add_argument("--volumes", nargs="+", required=True,
                    help="scroll=volumes/...zarr entries")
    args = ap.parse_args()

    results = []
    for entry in args.volumes:
        scroll, zarr_path = entry.split("=", 1)
        try:
            r = score_volume(scroll, zarr_path, args.level)
            tag = "GOOD" if r["good"] else "skip"
            print(f"[{tag}] {scroll:14s} occ={r['occupancy']:.3f} "
                  f"contrast={r.get('contrast_std','-')} range={r.get('dyn_range','-')} "
                  f"shape={r.get('shape_l0')}", flush=True)
            results.append(r)
        except Exception as e:
            print(f"[ERR ] {scroll}: {str(e)[:60]}", flush=True)
            results.append({"scroll": scroll, "zarr": zarr_path, "good": False,
                            "reason": f"error: {e}"})

    Path(args.out).write_text(json.dumps(results, indent=2))
    good = [r for r in results if r.get("good")]
    print(f"\n{len(good)}/{len(results)} volumes usable. wrote {args.out}")


if __name__ == "__main__":
    main()
