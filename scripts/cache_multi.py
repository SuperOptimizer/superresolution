#!/usr/bin/env python3
"""Cache quality-gated ROIs from multiple scrolls to local NVMe for training.

Reads configs/sources.json (curated by the quality survey), and for each volume:
  1. scouts the occupied ROI on a downsampled level (cheap),
  2. downloads a level-0 cube centered on the ROI (parallel chunk fetch),
  3. quality-gates the cube (occupancy/contrast); skips bad ones,
  4. saves <out_dir>/<scroll>.npy and appends to a manifest with voxel_um.

The manifest (cubes.json) drives the multi-cube training dataset. Each cube keeps
its voxel_um for FiLM scale conditioning.

    python scripts/cache_multi.py --sources configs/sources.json \
        --out-dir /mnt/data/cubes --cube 768 --scout-level 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superres.config import load_config
from superres.s3zarr import open_s3


def scout_and_fetch(scroll, zarr_path, bucket, scout_level, cube):
    z0 = open_s3(bucket, scroll, zarr_path, level=0)
    zlow = open_s3(bucket, scroll, zarr_path, level=scout_level)
    scale = 2 ** scout_level
    low = zlow.read_region(0, 0, 0, *zlow.shape, workers=24)
    mask = low > 0
    if not mask.any():
        return None, "empty scout"
    coords = np.where(mask)
    center = [int((c.min() + c.max()) / 2 * scale) for c in coords]
    half = (cube // 128 // 2) * 128
    edge = 2 * half
    origin = []
    for c, dim in zip(center, z0.shape):
        origin.append(max(0, min(c - half, dim - min(edge, dim))))
    ez = min(edge, z0.shape[0]); ey = min(edge, z0.shape[1]); ex = min(edge, z0.shape[2])
    cube_arr = z0.read_region(origin[0], origin[1], origin[2], ez, ey, ex, workers=32)
    return (cube_arr, origin), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--out-dir", default="/mnt/data/cubes")
    ap.add_argument("--cube", type=int, default=768, help="level-0 cube edge (voxels)")
    ap.add_argument("--scout-level", type=int, default=5)
    ap.add_argument("--min-occ", type=float, default=0.02)
    ap.add_argument("--min-contrast", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0, help="cap number of volumes (0=all)")
    args = ap.parse_args()

    src = load_config(args.sources)
    bucket = src["bucket"]
    vols = src["volumes"]
    if args.limit:
        vols = vols[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for v in vols:
        scroll, zarr_path, vum = v["scroll"], v["zarr"], v["voxel_um"]
        t = time.time()
        try:
            res, status = scout_and_fetch(scroll, zarr_path, bucket, args.scout_level, args.cube)
        except Exception as e:
            print(f"[ERR ] {scroll}: {str(e)[:60]}", flush=True)
            continue
        if res is None:
            print(f"[skip] {scroll}: {status}", flush=True)
            continue
        cube_arr, origin = res
        occ = float((cube_arr > 0).mean())
        nz = cube_arr[cube_arr > 0]
        contrast = float(nz.std()) if nz.size else 0.0
        if occ < args.min_occ or contrast < args.min_contrast:
            print(f"[skip] {scroll}: occ={occ:.3f} contrast={contrast:.1f} (below gate)",
                  flush=True)
            continue
        path = out_dir / f"{scroll}.npy"
        np.save(path, cube_arr)
        manifest.append({
            "scroll": scroll, "path": str(path), "voxel_um": vum,
            "shape": list(cube_arr.shape), "occupancy": round(occ, 4),
            "contrast": round(contrast, 2), "origin_zyx": origin,
        })
        print(f"[ok  ] {scroll:12s} vum={vum:5.3f} occ={occ:.3f} contrast={contrast:.1f} "
              f"{cube_arr.nbytes/1e9:.2f}GB in {time.time()-t:.0f}s", flush=True)

    man_path = out_dir / "cubes.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    total_gb = sum(np.prod(m["shape"]) for m in manifest) / 1e9
    print(f"\ncached {len(manifest)} cubes ({total_gb:.1f} GB) -> {man_path}")


if __name__ == "__main__":
    main()
