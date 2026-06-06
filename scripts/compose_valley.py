#!/usr/bin/env python3
"""PRINCIPLED compositional search for the deepest-valley preprocessing stack.

Principle: each operation removes a DIFFERENT valley-filling component (noise / outliers /
cross-filter residual / slow shading). We GREEDILY compose: from the current best volume, try
every candidate op, keep the one that deepens the valley most, repeat until nothing improves
(the plateau). Ordering falls out automatically; we stop at the measured optimum, not a guess.

All ops are band-selective and NON-saturating (we already know sharpening fills the valley, so
the op set is denoise-family only). Fast: 256^3 in RAM, C kernels, parallel candidate eval."""
import os, sys, ctypes as C, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib, scripts.valley_levers as vl; importlib.reload(vl)
from superres import s3zarr
from scipy.ndimage import median_filter, gaussian_filter

f32p = C.POINTER(C.c_float)
import superres.fysics_pipeline as fp
L = fp.lib()


def guided(v, eps, r=2):
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    L.fy_guided_denoise(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,r,eps); return o

def bilateral(v, ss, sr, r=3):
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    L.fy_bilateral_denoise.argtypes=[f32p,f32p,C.c_int,C.c_int,C.c_int,C.c_double,C.c_double,C.c_int]
    L.fy_bilateral_denoise(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,ss,sr,r); return o

# the candidate op library -- each targets a different valley-filling component.
# (name, fn) -- band-selective, non-saturating denoise-family only.
OPS = {
    "guided(0.004)":      lambda v: guided(v, 0.004),
    "guided(0.008)":      lambda v: guided(v, 0.008),
    "guided_r3(0.006)":   lambda v: guided(v, 0.006, 3),
    "bilateral(2,0.05)":  lambda v: bilateral(v, 2.0, 0.05),
    "bilateral(3,0.08)":  lambda v: bilateral(v, 3.0, 0.08),
    "median3":            lambda v: median_filter(v, 3),
}


def depth(v):
    r = vl.depth_of(np.clip(v*255, 0, 255).astype(np.uint8))
    return r[3] if r else -1.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    off = (1024-n)//2
    z = s3zarr.open_local("/home/forrest/paris4_2um_cube", 0)
    cur = z.read_region(off, off, off, n, n, n, workers=16).astype(np.float32)/255
    print(f"loaded {n}^3\n", flush=True)

    stack = []
    cur_depth = depth(cur)
    print(f"start (raw) depth={cur_depth:.3f}\n", flush=True)
    MAX_STEPS = 5
    for step in range(MAX_STEPS):
        # evaluate every candidate op applied to the CURRENT best, in parallel
        def ev(item):
            name, fn = item
            try: return name, fn(cur)
            except Exception as e: return name, None
        with ThreadPoolExecutor(max_workers=6) as ex:
            outs = list(ex.map(ev, OPS.items()))
        scored = [(name, o, depth(o)) for name, o in outs if o is not None]
        scored.sort(key=lambda r: r[2], reverse=True)
        print(f"step {step+1} candidates:")
        for name, o, d in scored:
            print(f"    {name:20s} -> {d:.3f}  ({'+' if d>cur_depth else ''}{d-cur_depth:+.3f})")
        best_name, best_o, best_d = scored[0]
        if best_d <= cur_depth + 0.003:   # plateau: no op meaningfully improves
            print(f"  -> PLATEAU (best {best_name}={best_d:.3f} <= current {cur_depth:.3f}). Stop.\n", flush=True)
            break
        stack.append(best_name); cur = best_o; cur_depth = best_d
        print(f"  -> ADD {best_name}, depth now {cur_depth:.3f}\n", flush=True)

    print("="*50)
    print(f"BEST STACK: {' -> '.join(stack) if stack else '(none, raw is best)'}")
    print(f"FINAL DEPTH: {cur_depth:.3f}  (raw was {depth(z.read_region(off,off,off,n,n,n,workers=16).astype(np.float32)/255):.3f})")


if __name__ == "__main__":
    main()
