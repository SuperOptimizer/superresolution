#!/usr/bin/env python3
"""FAST + PARALLEL valley-lever test. Load 256^3 once, run independent levers concurrently in
threads (C kernels release the GIL -> real parallelism). Reports valley depth per lever."""
import os, sys, time, ctypes as C
import numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")   # parallelize over levers, not within
from superres import s3zarr
import scripts.valley_levers as vl   # reuse guided2, diffuse, gat_denoise, perz_norm, depth_of

CUBE = "/home/forrest/paris4_2um_cube"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    off = (1024 - n)//2
    z = s3zarr.open_local(CUBE, 0)
    t0 = time.time()
    vol = z.read_region(off, off, off, n, n, n, workers=16)
    f01 = vol.astype(np.float32)/255.0
    print(f"loaded {n}^3 in {time.time()-t0:.1f}s, running levers in parallel...\n", flush=True)

    base = vl.guided2(f01)   # baseline shared by diffusion levers (compute once)

    levers = {
        "raw":                 lambda: f01,
        "guided x2 (baseline)": lambda: base,
        "+ diffusion s1":      lambda: vl.diffuse(base, 1),
        "+ diffusion s2":      lambda: vl.diffuse(base, 2),
        "GAT-denoise":         lambda: vl.gat_denoise(f01),
        "per-z + guided x2":   lambda: vl.guided2(vl.perz_norm(f01)),
        "per-z + guided + diff1": lambda: vl.diffuse(vl.guided2(vl.perz_norm(f01)), 1),
    }

    def run(item):
        name, fn = item
        t = time.time()
        try:
            out = fn(); r = vl.depth_of(np.clip(out*255, 0, 255).astype(np.uint8))
            return (name, r, time.time()-t, None)
        except Exception as e:
            return (name, None, time.time()-t, f"{type(e).__name__}: {e}")

    tstart = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(run, levers.items()))
    print(f"all levers done in {time.time()-tstart:.0f}s (wall, parallel)\n")

    base_depth = next((r[3] for n,r,_,_ in results if n=="guided x2 (baseline)"), None)
    print(f"{'lever':26s} dark light valley DEPTH   time")
    for name, r, dt, err in results:
        if err: print(f"{name:26s} ERROR {err}"); continue
        a,b,v,d = r
        tag = "  <- DEEPER" if base_depth and d > base_depth+0.01 else ("  (worse)" if base_depth and d < base_depth-0.01 else "")
        print(f"{name:26s} {a:>4} {b:>5} {v:>5} {d:.3f} ({dt:.0f}s){tag}")


if __name__ == "__main__":
    main()
