#!/usr/bin/env python3
"""Scout the occupied ROI of a scroll volume and cache a level-0 cube to local disk.

Random 128^3 patches over the full padded volume mostly hit air, and per-patch S3
fetches starve the GPU. This tool:
  1. Scans a cheap downsampled level (default 5, ~hundreds of MB) to find the
     occupied bounding box (the central ROI) -- no full-volume read.
  2. Maps the bbox up to level 0 and downloads a cube of level-0 chunks within it
     to a local raw .npy on fast NVMe, so training samples patches locally.

The cached cube is chosen around the ROI center with a configurable size (in level-0
voxels, snapped to the 128 chunk grid) so the download stays bounded.

    python scripts/cache_roi.py --config configs/default.yaml \
        --out /mnt/data/roi.npy --cube 1536 --scout-level 5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superres.config import load_config
from superres.s3zarr import open_s3, open_local


def scout_bbox(zlow, occ_thresh: float):
    """Read the whole downsampled level and return the occupied bbox (in its voxels)."""
    full = zlow.read_region(0, 0, 0, *zlow.shape)
    mask = full > 0
    if not mask.any():
        raise RuntimeError("scout level is entirely empty; wrong volume?")
    # robust occupancy: per-axis projection, threshold on fraction occupied
    coords = np.where(mask)
    z0, z1 = coords[0].min(), coords[0].max() + 1
    y0, y1 = coords[1].min(), coords[1].max() + 1
    x0, x1 = coords[2].min(), coords[2].max() + 1
    occ = float(mask.mean())
    return (z0, z1, y0, y1, x0, x1), occ, full.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="output .npy path (local NVMe)")
    ap.add_argument("--scout-level", type=int, default=5)
    ap.add_argument("--cube", type=int, default=1536,
                    help="level-0 cube edge in voxels (snapped to 128); centered on ROI")
    ap.add_argument("--occ-thresh", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]

    def opener(level):
        if d.get("local_path"):
            return open_local(d["local_path"], level=level)
        return open_s3(d["bucket"], d["scroll"], d["zarr"], level=level)

    z0arr = opener(0)
    zlow = opener(args.scout_level)
    scale = 2 ** args.scout_level

    print(f"scouting ROI on level {args.scout_level} (shape {zlow.shape}) ...")
    t0 = time.time()
    (sz0, sz1, sy0, sy1, sx0, sx1), occ, _ = scout_bbox(zlow, args.occ_thresh)
    print(f"  occupied bbox (level {args.scout_level}): "
          f"z[{sz0}:{sz1}] y[{sy0}:{sy1}] x[{sx0}:{sx1}]  occ={occ:.1%}  "
          f"({time.time()-t0:.1f}s)")

    # ROI center in level-0 voxels
    cz = int((sz0 + sz1) / 2 * scale)
    cy = int((sy0 + sy1) / 2 * scale)
    cx = int((sx0 + sx1) / 2 * scale)
    half = (args.cube // 128 // 2) * 128  # snap, half-edge
    edge = 2 * half

    def clamp(c, dim):
        lo = max(0, min(c - half, dim - edge))
        return lo
    gz = clamp(cz, z0arr.shape[0])
    gy = clamp(cy, z0arr.shape[1])
    gx = clamp(cx, z0arr.shape[2])
    edge_z = min(edge, z0arr.shape[0])
    edge_y = min(edge, z0arr.shape[1])
    edge_x = min(edge, z0arr.shape[2])

    nbytes = edge_z * edge_y * edge_x
    print(f"downloading level-0 cube @({gz},{gy},{gx}) size "
          f"({edge_z},{edge_y},{edge_x}) = {nbytes/1e9:.2f} GB ...")
    t0 = time.time()
    cube = z0arr.read_region(gz, gy, gx, edge_z, edge_y, edge_x)
    dt = time.time() - t0
    occ_cube = float((cube > 0).mean())
    print(f"  fetched in {dt:.0f}s ({nbytes/1e6/max(dt,1e-3):.0f} MB/s), "
          f"cube occupancy={occ_cube:.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, cube)
    # sidecar with provenance
    meta = {
        "scroll": d.get("scroll"), "zarr": d.get("zarr"), "level": 0,
        "origin_zyx": [gz, gy, gx], "shape_zyx": list(cube.shape),
        "occupancy": occ_cube, "dtype": str(cube.dtype),
    }
    import json
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out} ({cube.nbytes/1e9:.2f} GB) + {out.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
