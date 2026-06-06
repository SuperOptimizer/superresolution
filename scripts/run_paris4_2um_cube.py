#!/usr/bin/env python3
"""Run the fysics whole-volume pipeline over a 1024^3 region from the MIDDLE of the
real PHercParis4 2.4um scroll (137keV volume), streaming from S3 (never loading the
whole volume), with calibration pre-pass, then report before/after stats.

  python scripts/run_paris4_2um_cube.py
"""
import os, sys, time, resource
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
from superres import s3zarr
from superres import fysics_pipeline as fp

BUCKET = "vesuvius-challenge-open-data"
SCROLL = "PHercParis4"
ZARR = "volumes/20260323153942-2.400um-0.2m-137keV-masked.zarr"

WINDOW = (0.01, 0.16)  # target_window_f32_min/max (from metadata zarr_export)

# 1024^3 region anchored at the fully-occupied center (chunk 25,32,32 = vox 3200,4096,4096)
Z0, Y0, X0 = 3200, 4096, 4096
N = 1024


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    z = s3zarr.open_s3(BUCKET, SCROLL, ZARR, 0)
    print(f"volume shape {z.shape}  region [{Z0}:{Z0+N},{Y0}:{Y0+N},{X0}:{X0+N}]")

    # DERIVE physics from metadata.json over the S3 backend (policy: always use it).
    MD_PHYS = fp.load_md_phys(None, backend=z.backend)
    print(f"physics from metadata.json: energy={MD_PHYS['energy_kev']}keV "
          f"dist={MD_PHYS['distance_mm']}mm pixel={MD_PHYS['pixel_um']}um "
          f"delta_beta={MD_PHYS['delta_beta']} ({MD_PHYS.get('phase_method')})")

    # bounded read_region wrapping the S3 zarr, offset into the cube origin.
    read = lambda zz, yy, xx, dz, dy, dx: z.read_region(Z0+zz, Y0+yy, X0+xx, dz, dy, dx, workers=16)

    # --- pull sample tiles for the calibration pre-pass (textured, from inside cube) ---
    t0 = time.time()
    samp = []
    for (sz, sy, sx) in [(0,0,0),(0,512,512),(512,0,512),(512,512,0)]:
        samp.append(read(sz, sy, sx, 128, 128, 128))
    print(f"pulled {len(samp)} sample tiles in {time.time()-t0:.0f}s, RSS={rss_mb():.0f}MB")

    cal = fp.calibrate_prepass(MD_PHYS, samp, auto_deltabeta=True, verbose=True)
    cal.window = WINDOW
    print(f"CALIBRATION: deconv={cal.do_deconv}(reg={cal.deconv_reg}) "
          f"denoise={cal.do_denoise}(eps={cal.guided_eps:.4g}) "
          f"diffusion={cal.do_diffusion} halo={cal.halo} "
          f"db_scale={cal.db_scale:.3g} deconv_rescale=[{cal.deconv_lo:.3f},{cal.deconv_hi:.3f}]")

    # --- measure a held-out reference slab BEFORE, capture its processed version AFTER ---
    # Use a discard writer that records only one central slab for before/after metrics.
    SLAB = (448, 448, 448, 128, 128, 128)  # central 128^3 we'll score
    captured = {}
    def write(zz, yy, xx, blk):
        bz, by, bx = blk.shape
        sz0, sy0, sx0, sn, _, _ = SLAB
        # if this written inner tile covers the slab, capture the overlap
        if (zz <= sz0 and zz+bz >= sz0+sn and yy <= sy0 and yy+by >= sy0+sn
                and xx <= sx0 and xx+bx >= sx0+sn):
            captured["out"] = blk[sz0-zz:sz0-zz+sn, sy0-yy:sy0-yy+sn, sx0-xx:sx0-xx+sn].copy()

    print("running 2-pass parallel pipeline (streaming)...")
    tstart = time.time()
    peak = [rss_mb()]
    def prog(tag, val=None):
        if tag in ("pass1","pass2"):
            peak[0] = max(peak[0], rss_mb())
            print(f"  {tag} {val:.0f}% RSS={rss_mb():.0f}MB {time.time()-tstart:.0f}s")
        elif tag == "workers":
            print(f"  workers={val:.0f}")
    stats = fp.run_pipeline_2pass_parallel(
        read, write, (N, N, N), cal, tile=128,
        do_deconv=cal.do_deconv, do_denoise=cal.do_denoise, do_diffusion=False,
        progress=prog)
    dt = time.time() - tstart
    print(f"DONE in {dt:.0f}s  peakRSS={max(peak[0], rss_mb()):.0f}MB  stats={stats}")

    # --- before/after on the captured central slab ---
    ref_u8 = read(*SLAB)
    out_u8 = captured.get("out")
    if out_u8 is None:
        print("!! slab not captured"); return
    ref = ref_u8.astype(np.float32) / 255.0
    out = out_u8.astype(np.float32) / 255.0
    panel = fp._metric_panel(ref, out)
    print("\n=== BEFORE/AFTER (central 128^3 slab) ===")
    r = panel["raw"]
    print(f"  legibility (tex/noise gain): {r['legibility']:.3f}x")
    print(f"  texture kept:                {r['tex_ratio']:.3f}x")
    print(f"  noise ratio (lower=denoised):{r['noise_ratio']:.3f}x")
    print(f"  mid-band contrast:           {r['midret']:.3f}x")
    print(f"  clip fraction:               {r['clip']:.4f}")
    print(f"  panel score:                 {panel['score']:.3f}  ok={panel['ok']}")
    print(f"  raw mean u8:  before={ref_u8.mean():.1f}  after={out_u8.mean():.1f}")
    print(f"  raw std  u8:  before={ref_u8.std():.1f}  after={out_u8.std():.1f}")


if __name__ == "__main__":
    main()
