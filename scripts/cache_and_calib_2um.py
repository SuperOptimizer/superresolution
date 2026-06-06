#!/usr/bin/env python3
"""Cache a 1024^3 cube from the MIDDLE of the PHercParis4 2.4um volume to local disk
(raw-zarr, constant RAM), then run the improved calibration pre-pass LOCALLY on it.

  python scripts/cache_and_calib_2um.py
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
from superres import s3zarr
from superres import fysics_pipeline as fp

BUCKET = "vesuvius-challenge-open-data"
SCROLL = "PHercParis4"
ZARR = "volumes/20260323153942-2.400um-0.2m-137keV-masked.zarr"   # [6625,8431,8431]

# 1024^3 anchored at the fully-occupied center (chunk 25,32,32 = vox 3200,4096,4096)
Z0, Y0, X0 = 3200, 4096, 4096
N = 1024
OUT = "/home/forrest/paris4_2um_cube"   # local raw-zarr cache


def cache_cube():
    src = s3zarr.open_s3(BUCKET, SCROLL, ZARR, 0)
    os.makedirs(os.path.join(OUT, "0"), exist_ok=True)
    # ALWAYS grab metadata.json alongside the data so the run derives physics from it.
    meta_raw = src.backend.get("metadata.json")
    if meta_raw is None:
        raise SystemExit("metadata.json missing for this volume -- refusing to run on hardcoded physics")
    with open(os.path.join(OUT, "metadata.json"), "wb") as f:
        f.write(meta_raw)
    print("  saved metadata.json", flush=True)
    json.dump({"chunks": [128,128,128], "compressor": None, "dtype": "|u1",
               "fill_value": 0, "order": "C", "shape": [N,N,N], "zarr_format": 2,
               "dimension_separator": "/"}, open(os.path.join(OUT,"0",".zarray"),"w"))
    cz0, cy0, cx0 = Z0//128, Y0//128, X0//128
    ncc = N//128
    total = ncc**3; done = 0; t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    def fetch(coord):
        dz, dy, dx = coord
        raw = src.backend.get(src._chunk_key(cz0+dz, cy0+dy, cx0+dx))
        # local chunk index is (dz,dy,dx) since cube origin is chunk-aligned
        p = os.path.join(OUT, "0", str(dz), str(dy), str(dx))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if raw is None:
            np.zeros(128**3, np.uint8).tofile(p)
        else:
            with open(p, "wb") as f: f.write(raw)
        return 1
    coords = [(dz,dy,dx) for dz in range(ncc) for dy in range(ncc) for dx in range(ncc)]
    with ThreadPoolExecutor(max_workers=32) as ex:
        for r in ex.map(fetch, coords):
            done += 1
            if done % 32 == 0:
                el = time.time()-t0
                print(f"  cached {done}/{total} chunks  {el:.0f}s  ~{done/max(el,1):.1f} chunk/s", flush=True)
    print(f"cube cached to {OUT} in {time.time()-t0:.0f}s ({total} chunks, ~1GB)", flush=True)


def main():
    if not os.path.exists(os.path.join(OUT, "0", ".zarray")):
        print(f"caching 1024^3 cube from {SCROLL} 2.4um [{Z0}:{Z0+N},{Y0}:{Y0+N},{X0}:{X0+N}]...")
        cache_cube()
    else:
        print(f"cube already cached at {OUT}")
        # backfill metadata.json if an older cache lacks it
        if not os.path.exists(os.path.join(OUT, "metadata.json")):
            src = s3zarr.open_s3(BUCKET, SCROLL, ZARR, 0)
            mr = src.backend.get("metadata.json")
            if mr is None:
                raise SystemExit("metadata.json missing -- refusing to run on hardcoded physics")
            open(os.path.join(OUT, "metadata.json"), "wb").write(mr)
            print("  backfilled metadata.json")

    # DERIVE physics from metadata.json (policy: if it exists, always use it).
    MD_PHYS = fp.load_md_phys(OUT)
    print(f"physics from metadata.json: energy={MD_PHYS['energy_kev']}keV "
          f"dist={MD_PHYS['distance_mm']}mm pixel={MD_PHYS['pixel_um']}um "
          f"delta_beta={MD_PHYS['delta_beta']} ({MD_PHYS.get('phase_method')}) "
          f"unsharp(s={MD_PHYS['unsharp_sigma']},c={MD_PHYS['unsharp_coeff']})")
    print(f"beam current: {MD_PHYS.get('machine_current_start')} -> {MD_PHYS.get('machine_current_stop')} mA")

    z = s3zarr.open_local(OUT, 0)
    print(f"\nlocal cube {z.shape}")
    read = lambda zz,yy,xx,dz,dy,dx: z.read_region(zz,yy,xx,dz,dy,dx, workers=8)

    # occupancy sanity
    mid = read(N//2-64, N//2-64, N//2-64, 128, 128, 128)
    print(f"center tile: mean={mid.mean():.1f} std={mid.std():.1f} occ={(mid>0).mean():.2f}")

    print("\n=== SMART SAMPLING ===")
    t0 = time.time()
    picked = fp.select_sample_tiles(read, z.shape, n=6, tile=128, candidates=27)
    print(f"selected {len(picked)} tiles in {time.time()-t0:.1f}s  means={[int(p.mean()) for p in picked]}")

    print("\n=== CALIBRATION (joint + refine, verbose) ===")
    t0 = time.time()
    cal = fp.calibrate_prepass(MD_PHYS, picked, verbose=True)
    dt = time.time()-t0
    print(f"\nTUNED in {dt:.1f}s:")
    print(f"  deconv  = {cal.do_deconv}  reg={cal.deconv_reg}  db_scale={cal.db_scale:.3f}")
    print(f"  denoise = {cal.do_denoise}  eps={cal.guided_eps:.4g}")
    print(f"  halo={cal.halo}")
    print(f"  tuning.deconv  = {cal.tuning.get('deconv')}")
    print(f"  tuning.denoise = {cal.tuning.get('denoise')}")
    if 'all_off_reason' in cal.tuning:
        print(f"  all_off_reason = {cal.tuning['all_off_reason']}")

    # ---- before/after on one held-out central tile through the tuned chain ----
    print("\n=== BEFORE/AFTER on a held-out central 128^3 tile ===")
    ref_u8 = read(512, 512, 512, 128, 128, 128)
    out = fp.process_tile(ref_u8, cal, do_deconv=cal.do_deconv,
                          do_denoise=cal.do_denoise, do_diffusion=False)
    ref = ref_u8.astype(np.float32)/255.0
    panel = fp._metric_panel(ref, out.astype(np.float32)/255.0 if out.dtype==np.uint8 else out)
    r = panel["raw"]
    print(f"  legibility (tex/noise): {r['legibility']:.3f}x")
    print(f"  texture kept:           {r['tex_ratio']:.3f}x")
    print(f"  noise ratio (lower=better): {r['noise_ratio']:.3f}x")
    print(f"  mid-band contrast:      {r['midret']:.3f}x")
    print(f"  clip fraction:          {r['clip']:.4f}")
    print(f"  panel score: {panel['score']:.3f}  ok={panel['ok']}")


if __name__ == "__main__":
    main()
