#!/usr/bin/env python3
"""Verify the u8~88 valley cutoff: does zeroing u8<88 remove the 'other' (dark mode) and
KEEP the papyrus sheets (light mode)? Render slices in all 3 axes with the cutoff applied,
plus the bimodal histogram with the valley marked, so we can SEE it's the right boundary."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr

CUBE = "/home/forrest/paris4_2um_cube"; OUTDIR = "/home/forrest/paris4_2um_viz"; N = 1024
VALLEY = 88


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    z = s3zarr.open_local(CUBE, 0)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    def norm(a):
        v = a[a > 0]
        if v.size == 0: return a
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

    # mid-plane in each axis (read a thin slab, take center)
    planes = {
        "Z": z.read_region(N//2, 0, 0, 1, N, N)[0],
        "Y": z.read_region(0, N//2, 0, N, 1, N)[:, 0, :],
        "X": z.read_region(0, 0, N//2, N, N, 1)[:, :, 0],
    }
    for axis, sl in planes.items():
        sl = sl.astype(np.float32)
        keep = sl.copy(); keep[sl < VALLEY] = 0
        removed = ((sl > 0) & (sl < VALLEY)).mean() * 100
        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
        ax[0].imshow(norm(sl), cmap="gray"); ax[0].set_title(f"{axis} raw"); ax[0].axis("off")
        ax[1].imshow(norm(keep), cmap="gray")
        ax[1].set_title(f"{axis} u8>={VALLEY} kept (removed {removed:.0f}%)"); ax[1].axis("off")
        # show WHAT was removed (the 'other')
        other = np.where(sl < VALLEY, sl, 0)
        ax[2].imshow(norm(other), cmap="magma"); ax[2].set_title(f"{axis} removed (the 'other')"); ax[2].axis("off")
        p = os.path.join(OUTDIR, f"valley_{axis}.png")
        fig.tight_layout(); fig.savefig(p, dpi=95, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {p}  (removed {removed:.0f}% on this plane)")


if __name__ == "__main__":
    main()
