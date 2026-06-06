#!/usr/bin/env python3
"""Characterize the intra-ROI dark-voxel population on the PHercParis4 2.4um cube to find a
CONSERVATIVE zeroing cutoff: bright papyrus (keep) vs dark 'something else' (safe to zero).

We want the boundary that zeros the dark population WITHOUT eating papyrus. Erring toward
zeroing LESS. Outputs: nonzero histogram (find the valley between dark mode and papyrus mode),
local-variance vs intensity (is the dark stuff flat/structureless?), and a slice viz with
candidate thresholds overlaid so we can SEE what each cutoff removes."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"
OUTDIR = "/home/forrest/paris4_2um_viz"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    z = s3zarr.open_local(CUBE, 0)
    read = lambda zz,yy,xx,dz,dy,dx: z.read_region(zz,yy,xx,dz,dy,dx,workers=8)
    md = fp.load_md_phys(CUBE)
    wlo, whi = md["window_f32_min"], md["window_f32_max"]

    big = read(0,0,0,512,512,512)
    nz = big[big > 0]
    print(f"voxels={big.size:,}  nonzero={nz.size:,} ({100*nz.size/big.size:.1f}%)  "
          f"u8=0 (outside-ROI): {100*(big==0).mean():.1f}%\n")

    # ---- 1. nonzero histogram: find the valley between dark-mode and papyrus-mode ----
    h = np.bincount(nz, minlength=256).astype(float)
    hs = np.convolve(h, np.ones(5)/5, mode="same")   # smooth
    print("=== nonzero u8 histogram (smoothed), every 8 levels ===")
    peak = hs.argmax()
    for u in range(0, 256, 8):
        bar = "#" * int(60 * hs[u] / hs.max())
        print(f"  u8 {u:3d}: {bar}")
    print(f"  papyrus mode (peak) at u8={peak}")

    # find a valley: the min of the smoothed histogram BELOW the main peak
    lo_search = hs[2:peak]   # skip the u8=1.. noise floor
    if lo_search.size:
        valley = int(np.argmin(lo_search)) + 2
        print(f"  valley (dark|papyrus boundary candidate) at u8={valley}")
    else:
        valley = None

    # percentile-based safe cutoffs (conservative: low percentiles of the papyrus body)
    p1, p2, p5, p10 = [int(np.percentile(nz, q)) for q in (1, 2, 5, 10)]
    print(f"  nonzero percentiles: p1={p1} p2={p2} p5={p5} p10={p10}")
    print(f"  window-derived: u8=13 (5% of span), Otsu would be ~93 (too high)\n")

    # ---- 2. is the dark population FLAT (air/void) or STRUCTURED (real)? ----
    # local std in 5^3 windows; compare for dark vs bright voxels.
    try:
        from scipy.ndimage import uniform_filter
        sub = big[128:384, 128:384, 128:384].astype(np.float32)
        m = uniform_filter(sub, 5); m2 = uniform_filter(sub*sub, 5)
        lstd = np.sqrt(np.maximum(m2 - m*m, 0))
        for name, mask in [("dark (1-20)", (sub>=1)&(sub<20)),
                           ("mid (20-60)", (sub>=20)&(sub<60)),
                           ("papyrus (80-180)", (sub>=80)&(sub<180))]:
            if mask.sum() > 100:
                print(f"  local-std of {name:18s}: median={np.median(lstd[mask]):.2f}  "
                      f"(low=flat/structureless -> safe to zero)  n={mask.sum():,}")
    except Exception as e:
        print("  (scipy unavailable for variance analysis)", e)

    # ---- 3. how much would each candidate cutoff zero? (want to NOT zero papyrus) ----
    print(f"\n=== fraction of NONZERO voxels removed by each candidate cutoff ===")
    for thr in [5, 8, 10, 13, 16, 20, 25, 30, p5, valley or 20]:
        frac = (nz < thr).mean()
        print(f"  cutoff u8<{thr:3d}: zeros {frac*100:5.2f}% of nonzero voxels")

    # ---- 4. visualize: a slice with candidate thresholds, so we SEE what's removed ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sl = big[256].astype(np.float32)
        def norm(a):
            lo,hi=np.percentile(a[a>0],1),np.percentile(a[a>0],99); return np.clip((a-lo)/(hi-lo),0,1)
        cands = [10, 16, 25]
        fig, ax = plt.subplots(1, 4, figsize=(28, 7))
        ax[0].imshow(norm(sl), cmap="gray"); ax[0].set_title("raw slice"); ax[0].axis("off")
        for i, thr in enumerate(cands):
            kept = sl.copy(); kept[sl < thr] = 0
            ax[i+1].imshow(norm(kept), cmap="gray")
            removed = ((sl>0)&(sl<thr)).mean()*100
            ax[i+1].set_title(f"zero u8<{thr} (removes {removed:.1f}% of nonzero)"); ax[i+1].axis("off")
        p = os.path.join(OUTDIR, "air_cutoff_candidates.png")
        fig.tight_layout(); fig.savefig(p, dpi=100, bbox_inches="tight"); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print("viz skipped:", e)


if __name__ == "__main__":
    main()
