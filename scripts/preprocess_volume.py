#!/usr/bin/env python3
"""Standalone physics deblur of a scroll volume (NO ML).

Inverts the known ESRF/nabu reconstruction operators (Paganin + unsharp), derived
from the volume's own metadata.json, to sharpen it. Pure physics -- usable for
segmentation / ink detection / reading / vc3d viewer post-processing.

  # Sharpen a region and write TIFF slices (quick look / viewer-style)
  python scripts/preprocess_volume.py --scroll PHerc1203 \
      --zarr volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr \
      --region 400 400 400 256 256 256 --reg 0.05 --out-tiff /tmp/sharp

  # Batch sharpen a region into a local .npy
  ... --region ... --out-npy sharp.npy

`--reg` is the sharpening-strength dial (lower = sharper + noisier).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superres.s3zarr import open_s3, open_local
from superres.paganin import physics_from_metadata
from superres.preprocess import sharpen_region, kernel_extent_voxels

BUCKET = "vesuvius-challenge-open-data"


def fetch_metadata(scroll, zarr):
    uri = f"s3://{BUCKET}/{scroll}/{zarr}/metadata.json"
    p = subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--quiet", uri, "-"],
                       capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    try:
        return json.loads(p.stdout)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll")
    ap.add_argument("--zarr")
    ap.add_argument("--local", help="local zarr path instead of S3")
    ap.add_argument("--region", nargs=6, type=int, metavar=("Z", "Y", "X", "DZ", "DY", "DX"),
                    required=True)
    ap.add_argument("--reg", type=float, default=0.05, help="sharpening strength (lower=sharper)")
    ap.add_argument("--out-tiff", help="dir for before/after TIFF slices")
    ap.add_argument("--out-npy", help="path for sharpened .npy")
    # optional manual physics override
    ap.add_argument("--delta-beta", type=float)
    ap.add_argument("--energy", type=float)
    ap.add_argument("--distance-mm", type=float)
    ap.add_argument("--pixel-mm", type=float, default=0.0024)
    args = ap.parse_args()

    z, y, x, dz, dy, dx = args.region
    if args.local:
        zr = open_local(args.local, level=0)
        md = None
    else:
        zr = open_s3(BUCKET, args.scroll, args.zarr, level=0)
        md = fetch_metadata(args.scroll, args.zarr)

    phys = physics_from_metadata(md) if md else None
    if phys is None:
        if args.delta_beta and args.energy:
            phys = {"paganin_delta_beta": args.delta_beta, "energy_kev": args.energy,
                    "sample_detector_mm": args.distance_mm or 220.0,
                    "sample_pixel_mm": args.pixel_mm, "unsharp_sigma": 1.2, "unsharp_coeff": 4.0}
            print("using manual physics:", phys)
        else:
            print("ERROR: no metadata physics found and no manual --delta-beta/--energy given")
            sys.exit(1)
    else:
        print(f"physics from metadata: db={phys['paganin_delta_beta']} E={phys['energy_kev']} "
              f"D={phys['sample_detector_mm']}mm px={phys['sample_pixel_mm']}mm")
    print(f"kernel half-extent (halo): {kernel_extent_voxels(phys)} voxels")

    from superres.data import robust_normalize
    def fetch(zz, yy, xx, ddz, ddy, ddx):
        return robust_normalize(zr.read_region(zz, yy, xx, ddz, ddy, ddx, workers=32))

    raw = fetch(z, y, x, dz, dy, dx)
    sharp = sharpen_region(fetch, z, y, x, dz, dy, dx, zr.shape, phys, reg=args.reg)
    print(f"sharpened region {(dz,dy,dx)}  raw_range=[{raw.min():.2f},{raw.max():.2f}] "
          f"sharp_range=[{sharp.min():.2f},{sharp.max():.2f}]")

    if args.out_npy:
        np.save(args.out_npy, sharp)
        print(f"wrote {args.out_npy}")
    if args.out_tiff:
        import tifffile
        out = Path(args.out_tiff); out.mkdir(parents=True, exist_ok=True)
        def to8(a): return (np.clip(a, 0, 1) * 255).astype(np.uint8)
        for sl in [dz // 4, dz // 2, 3 * dz // 4]:
            panel = np.concatenate([to8(raw[sl]), to8(sharp[sl])], axis=1)
            tifffile.imwrite(str(out / f"slice{sl}_raw_vs_sharp.tif"), panel)
        print(f"wrote before/after TIFF panels to {out}")


if __name__ == "__main__":
    main()
