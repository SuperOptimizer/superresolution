#!/usr/bin/env python3
"""Run the OPTIMIZED pipeline (reconciled calibration + air-zero) on the 1024^3 cube and emit
1024^2 before/after PNGs through all 3 axes. Only processes a halo-padded slab around each
mid-plane (enough for the slice), so it's fast. On-disk only."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fastmetrics as fm
from superres import s3zarr, fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"; OUTDIR = os.path.expanduser("~"); N = 1024


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    z = s3zarr.open_local(CUBE, 0)
    cal = fm.fast_cal(CUBE)                 # reconciled calibration (cached)
    cal.do_air_zero = True; cal.scratch_passes = 5
    halo = max(cal.halo, 4)
    print(f"optimized cal: reg={cal.deconv_reg} eps={cal.guided_eps:.4g} air_zero=ON halo={halo}", flush=True)

    def norm(a):
        v = a[a > 0]
        if v.size == 0: return a
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

    mid = N // 2
    for axis in ("Z", "Y", "X"):
        t = time.time()
        # read a thin halo-padded slab perpendicular to `axis`, full 1024^2 in the other two
        if axis == "Z":
            sl0 = mid - halo
            slab = z.read_region(sl0, 0, 0, 2*halo+1, N, N, workers=16)
            center = halo
        elif axis == "Y":
            sl0 = mid - halo
            slab = z.read_region(0, sl0, 0, N, 2*halo+1, N, workers=16)
            center = halo
        else:
            sl0 = mid - halo
            slab = z.read_region(0, 0, sl0, N, N, 2*halo+1, workers=16)
            center = halo
        before = slab.astype(np.float32) / 255.0
        # process the WHOLE slab through the optimized chain (process_tile handles any shape)
        out = fp.process_tile(np.ascontiguousarray(slab, np.uint8), cal).astype(np.float32) / 255.0
        # extract the center plane in this axis
        if axis == "Z":
            b2, a2 = before[center], out[center]
        elif axis == "Y":
            b2, a2 = before[:, center, :], out[:, center, :]
        else:
            b2, a2 = before[:, :, center], out[:, :, center]
        nb, na = norm(b2), norm(a2)
        # comparison panel
        fig, ax = plt.subplots(1, 2, figsize=(22, 11))
        ax[0].imshow(nb, cmap="gray"); ax[0].set_title(f"{axis} BEFORE (raw 2.4um)"); ax[0].axis("off")
        ax[1].imshow(na, cmap="gray")
        ax[1].set_title(f"{axis} AFTER (deconv reg={cal.deconv_reg}+denoise eps={cal.guided_eps:.4f}+air-zero)")
        ax[1].axis("off")
        pp = os.path.join(OUTDIR, f"opt_{axis}_compare.png")
        fig.tight_layout(); fig.savefig(pp, dpi=110, bbox_inches="tight"); plt.close(fig)
        # separate full-res before/after images
        from PIL import Image
        Image.fromarray((nb*255).astype(np.uint8)).save(os.path.join(OUTDIR, f"opt_{axis}_before.png"))
        Image.fromarray((na*255).astype(np.uint8)).save(os.path.join(OUTDIR, f"opt_{axis}_after.png"))
        print(f"wrote opt_{axis}_compare/before/after.png  ({time.time()-t:.0f}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
