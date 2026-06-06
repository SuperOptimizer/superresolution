#!/usr/bin/env python3
"""On the cached PHercParis4 2.4um 1024^3 cube:
  (1) calibrate once, then sweep several disjoint sub-regions -> legibility spread.
  (2) process slabs through the tuned chain and emit 1024^2 BEFORE/AFTER PNGs in all 3 axes.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr
from superres import fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"
OUTDIR = "/home/forrest/paris4_2um_viz"
N = 1024


def proc01(u8, cal):
    out = fp.process_tile(u8, cal, do_deconv=cal.do_deconv,
                          do_denoise=cal.do_denoise, do_diffusion=False)
    return out.astype(np.float32)/255.0 if out.dtype == np.uint8 else out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    MD_PHYS = fp.load_md_phys(CUBE)   # policy: derive physics from metadata.json
    print(f"physics from metadata.json: energy={MD_PHYS['energy_kev']}keV "
          f"dist={MD_PHYS['distance_mm']}mm pixel={MD_PHYS['pixel_um']}um "
          f"delta_beta={MD_PHYS['delta_beta']} ({MD_PHYS.get('phase_method')})")
    z = s3zarr.open_local(CUBE, 0)
    read = lambda zz,yy,xx,dz,dy,dx: z.read_region(zz,yy,xx,dz,dy,dx, workers=8)
    print(f"cube {z.shape}")

    # calibrate once on smart-sampled tiles
    picked = fp.select_sample_tiles(read, z.shape, n=6, tile=128, candidates=27)
    cal = fp.calibrate_prepass(MD_PHYS, picked, verbose=False)
    print(f"CALIB: deconv={cal.do_deconv}(reg={cal.deconv_reg}) "
          f"denoise={cal.do_denoise}(eps={cal.guided_eps:.4g}) "
          f"rescale=[{cal.deconv_lo:.3f},{cal.deconv_hi:.3f}]\n")

    # ---- (1) SUB-REGION SWEEP: 8 disjoint 128^3 tiles across the cube ----
    print("=== SUB-REGION SWEEP (128^3 tiles, tuned chain) ===")
    coords = []
    for a in (192, 704):
        for b in (192, 704):
            for c in (192, 704):
                coords.append((a, b, c))
    rows = []
    for (zz, yy, xx) in coords:
        u8 = read(zz, yy, xx, 128, 128, 128)
        if (u8 > 0).mean() < 0.5:
            continue
        ref = u8.astype(np.float32)/255.0
        out = proc01(u8, cal)
        m = fp._metric_panel(ref, out)["raw"]
        rows.append((zz, yy, xx, m["legibility"], m["tex_ratio"], m["noise_ratio"], m["midret"], u8.mean()))
        print(f"  ({zz:>4},{yy:>4},{xx:>4}) mean={u8.mean():5.1f}  "
              f"legib={m['legibility']:.2f}x  tex={m['tex_ratio']:.2f}  "
              f"noise={m['noise_ratio']:.2f}x  contrast={m['midret']:.2f}x")
    L = np.array([r[3] for r in rows])
    NO = np.array([r[5] for r in rows]); CT = np.array([r[6] for r in rows])
    print(f"\n  legibility: min={L.min():.2f} median={np.median(L):.2f} max={L.max():.2f}  (n={len(rows)})")
    print(f"  noise ratio: median={np.median(NO):.2f}x   contrast: median={np.median(CT):.2f}x")

    # ---- (2) VISUALIZE: process a slab around each mid-plane, save 1024^2 before/after ----
    print("\n=== VISUALIZE (1024^2 before/after, all 3 axes) ===")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    mid = N // 2
    halo = cal.halo
    # We need the processed mid-plane in each axis. Process a halo-padded SLAB perpendicular
    # to each axis (thin in that axis, full 1024^2 in the other two), then take center plane.
    def save_pair(axis, before2d, after2d, tag):
        # robust display normalization shared so before/after are comparable
        def norm(a):
            lo, hi = np.percentile(a, 1), np.percentile(a, 99)
            return np.clip((a-lo)/(hi-lo+1e-9), 0, 1)
        b = norm(before2d); a = norm(after2d)
        if plt is not None:
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            ax[0].imshow(b, cmap="gray"); ax[0].set_title(f"{tag} BEFORE (raw 2.4um)"); ax[0].axis("off")
            ax[1].imshow(a, cmap="gray"); ax[1].set_title(f"{tag} AFTER (deconv reg={cal.deconv_reg}+denoise)"); ax[1].axis("off")
            p = os.path.join(OUTDIR, f"{tag}_compare.png")
            fig.tight_layout(); fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
            print(f"  wrote {p}")
        # also dump raw u8 PNGs of each panel for full-res viewing
        from PIL import Image
        Image.fromarray((b*255).astype(np.uint8)).save(os.path.join(OUTDIR, f"{tag}_before.png"))
        Image.fromarray((a*255).astype(np.uint8)).save(os.path.join(OUTDIR, f"{tag}_after.png"))

    # AXIS Z: slab [mid-halo : mid+halo+1] x full y,x. process, take center z-plane.
    sz0 = mid - halo
    slabz = read(sz0, 0, 0, 2*halo+1, N, N)
    procz = proc01(slabz, cal)
    save_pair("z", slabz[halo].astype(np.float32)/255.0, procz[halo], "axisZ")

    # AXIS Y: thin in y. process slab [0:N, mid-halo:mid+halo+1, 0:N], center y-plane.
    slaby = read(0, mid-halo, 0, N, 2*halo+1, N)
    procy = proc01(slaby, cal)
    save_pair("y", slaby[:, halo, :].astype(np.float32)/255.0, procy[:, halo, :], "axisY")

    # AXIS X: thin in x.
    slabx = read(0, 0, mid-halo, N, N, 2*halo+1)
    procx = proc01(slabx, cal)
    save_pair("x", slabx[:, :, halo].astype(np.float32)/255.0, procx[:, :, halo], "axisX")

    print(f"\nimages in {OUTDIR}/")


if __name__ == "__main__":
    main()
