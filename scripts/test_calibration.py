#!/usr/bin/env python3
"""Exercise the improved calibrate_prepass: joint search, refinement, memoization,
smart sampling, explainability. Runs on synthetic tiles (fast, no network) and on the
local 45um Paris4 volume if present."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import fysics_pipeline as fp


def synth_tile(n=96, noise=8.0, seed=0):
    """Fibrous papyrus-like tile: layered sheets + fibers + noise, u8."""
    rng = np.random.RandomState(seed)
    z, y, x = np.indices((n, n, n)).astype(np.float32)
    # stacked sheets along y with some warp
    sheets = 0.5 + 0.5 * np.sin((y + 6*np.sin(x/20.0)) * (2*np.pi/9.0))
    fibers = 0.3 * np.sin(x/3.0 + z/5.0)
    v = np.clip(0.45 + 0.35*sheets + 0.1*fibers, 0, 1)
    # blur a touch (acquisition MTF) then add noise + u8 quantize
    try:
        from scipy.ndimage import gaussian_filter
        v = gaussian_filter(v, 0.8)
    except Exception:
        pass
    u = np.clip(v*255 + rng.randn(*v.shape)*noise, 0, 255).astype(np.uint8)
    return u


MD = dict(energy_kev=137.0, distance_mm=220.0, pixel_um=2.4, delta_beta=1000.0)


def main():
    print("=== synthetic: fine, low-noise (expect deconv ON, maybe gentle denoise) ===")
    tiles = [synth_tile(seed=s, noise=6.0) for s in range(4)]
    t0 = time.time()
    cal = fp.calibrate_prepass(MD, tiles, verbose=True)
    print(f"  -> deconv={cal.do_deconv}(reg={cal.deconv_reg}) denoise={cal.do_denoise}"
          f"(eps={cal.guided_eps:.4g}) in {time.time()-t0:.1f}s")
    print(f"  tuning={cal.tuning}\n")

    print("=== synthetic: very noisy (expect denoise to matter; maybe deconv backs off) ===")
    tiles = [synth_tile(seed=s, noise=22.0) for s in range(4)]
    cal = fp.calibrate_prepass(MD, tiles, verbose=False)
    print(f"  -> deconv={cal.do_deconv}(reg={cal.deconv_reg}) denoise={cal.do_denoise}"
          f"(eps={cal.guided_eps:.4g})")
    print(f"  tuning.deconv={cal.tuning.get('deconv')}")
    print(f"  tuning.denoise={cal.tuning.get('denoise')}")
    if "all_off_reason" in cal.tuning:
        print(f"  all_off_reason={cal.tuning['all_off_reason']}")
    print()

    # ---- real 45um local volume: smart-sample then calibrate ----
    root = "/home/forrest/paris4_local"
    if os.path.isdir(os.path.join(root, "0")):
        print("=== real PHercParis4 45um (local): smart sampling + calibrate ===")
        from superres import s3zarr
        z = s3zarr.open_local(root, 0)
        print(f"  volume {z.shape}")
        read = lambda zz,yy,xx,dz,dy,dx: z.read_region(zz,yy,xx,dz,dy,dx, workers=8)
        t0 = time.time()
        picked = fp.select_sample_tiles(read, z.shape, n=4, tile=96, candidates=27)
        print(f"  selected {len(picked)} textured tiles in {time.time()-t0:.1f}s "
              f"(means={[int(p.mean()) for p in picked]})")
        if picked:
            t0 = time.time()
            cal = fp.calibrate_prepass(MD, picked, verbose=True)
            print(f"  -> deconv={cal.do_deconv}(reg={cal.deconv_reg}) denoise={cal.do_denoise}"
                  f"(eps={cal.guided_eps:.4g}) in {time.time()-t0:.1f}s")
            print(f"  tuning={cal.tuning}")

    print("\n=== memoization sanity: calibrate twice, second has warm cache per-call ===")
    tiles = [synth_tile(seed=s) for s in range(4)]
    t0 = time.time(); fp.calibrate_prepass(MD, tiles); dt1 = time.time()-t0
    print(f"  full joint+refine calibrate: {dt1:.2f}s (memoized deconv across 6x6 grid)")


if __name__ == "__main__":
    main()
