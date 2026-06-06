#!/usr/bin/env python3
"""Full 1024^3 nonzero histogram -> find the two modes (dark vs light) and the valley.
Streams the cube in slabs (constant RAM), accumulates a 256-bin histogram."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr

CUBE = "/home/forrest/paris4_2um_cube"; N = 1024


def main():
    z = s3zarr.open_local(CUBE, 0)
    h = np.zeros(256, np.int64)
    for z0 in range(0, N, 128):                     # stream 128-thick slabs
        slab = z.read_region(z0, 0, 0, 128, N, N, workers=12)
        h += np.bincount(slab.ravel(), minlength=256)
        print(f"  slab z={z0}: cumulative nonzero={int(h[1:].sum()):,}", flush=True)

    tot = int(h.sum()); nzt = int(h[1:].sum())
    print(f"\nFULL CUBE: voxels={tot:,}  u8=0={100*h[0]/tot:.1f}%  nonzero={100*nzt/tot:.1f}%\n")

    hs = np.convolve(h.astype(float), np.ones(5)/5, mode="same")
    hs[0] = 0  # ignore the outside-ROI zero spike
    print("=== nonzero u8 histogram (log-ish bar), every 4 levels ===")
    mx = hs[1:].max()
    for u in range(0, 256, 4):
        bar = "#" * int(70 * hs[u] / mx)
        mark = ""
        print(f"  u8 {u:3d}: {bar}{mark}")

    # find peaks: local maxima of the smoothed histogram (skip noise floor < u8 3)
    peaks = [u for u in range(3, 254)
             if hs[u] >= hs[u-1] and hs[u] >= hs[u+1] and hs[u] > 0.04*mx]
    # collapse near-adjacent peaks
    merged = []
    for p in peaks:
        if not merged or p - merged[-1] > 8:
            merged.append(p)
        elif hs[p] > hs[merged[-1]]:
            merged[-1] = p
    print(f"\n  modes (peaks) at u8 = {merged}")
    # valley between the two biggest peaks
    if len(merged) >= 2:
        top2 = sorted(merged, key=lambda u: hs[u], reverse=True)[:2]
        a, b = sorted(top2)
        valley = a + int(np.argmin(hs[a:b]))
        print(f"  two dominant modes: dark~u8={a}, light~u8={b}")
        print(f"  VALLEY between them at u8={valley}  (the air|papyrus cutoff)")
        print(f"  -> zeroing u8<{valley} removes {100*h[1:valley].sum()/nzt:.2f}% of nonzero voxels")
    else:
        print("  only one mode found over the full cube")


if __name__ == "__main__":
    main()
