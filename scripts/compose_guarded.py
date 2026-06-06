#!/usr/bin/env python3
"""GUARDED compositional search for the best valley-deepening stack that PRESERVES signal.

The naive objective (valley depth alone) is gameable by over-smoothing: blur everything to two
flat blobs and the histogram 'valley' deepens while the papyrus geometry/contrast/texture is
destroyed. We guard against that. A candidate is scored ONLY if it keeps:
  - texture retention  >= TEX_FLOOR  (high-pass std vs raw -- papyrus fiber detail)
  - edge/contrast      >= EDGE_FLOOR (gradient-magnitude energy vs raw -- sheet boundaries)
  - structure SSIM-ish >= STRUCT_FLOOR (local-corr with raw -- geometry not moved)
Objective = valley_depth * (all guards pass) else -inf. We greedily add the op with the best
GUARDED objective until no op improves it. Fast: 128^3 in RAM, C kernels, parallel candidates."""
import os, sys, ctypes as C, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib, scripts.valley_levers as vl; importlib.reload(vl)
from superres import s3zarr
import superres.fysics_pipeline as fp
from scipy.ndimage import gaussian_filter, median_filter
f32p = C.POINTER(C.c_float); L = fp.lib()

# signal-preservation floors (a stack must keep at least this much of each)
TEX_FLOOR = 0.55     # keep >=55% of papyrus high-freq texture
EDGE_FLOOR = 0.60    # keep >=60% of sheet-boundary gradient energy
STRUCT_FLOOR = 0.97  # geometry barely moves (local correlation with raw)


def guided(v, eps, r=2):
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    L.fy_guided_denoise(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,r,eps); return o

def bilateral(v, ss, sr, r=3):
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    L.fy_bilateral_denoise.argtypes=[f32p,f32p,C.c_int,C.c_int,C.c_int,C.c_double,C.c_double,C.c_int]
    L.fy_bilateral_denoise(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,ss,sr,r); return o

OPS = {
    "bilat(2,0.05)":   lambda v: bilateral(v,2.0,0.05),
    "bilat(3,0.08)":   lambda v: bilateral(v,3.0,0.08),
    "bilat(2,0.03)":   lambda v: bilateral(v,2.0,0.03),
    "guided(0.004)":   lambda v: guided(v,0.004),
    "median3":         lambda v: median_filter(v,3),
}

# --- signal metrics, all relative to the RAW volume ---
def _hp(v): return v - gaussian_filter(v, 1.0)
def _grad(v):
    gz,gy,gx = np.gradient(v); return np.sqrt(gz*gz+gy*gy+gx*gx)

def guards(o, raw, raw_tex, raw_edge):
    tex = _hp(o).std() / raw_tex
    edge = _grad(o).mean() / raw_edge
    # structure: local correlation with raw (geometry preserved). cheap global corr proxy.
    a=(raw-raw.mean()).ravel(); b=(o-o.mean()).ravel()
    struct = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    ok = (tex>=TEX_FLOOR and edge>=EDGE_FLOOR and struct>=STRUCT_FLOOR)
    return ok, tex, edge, struct

def depth(o):
    r = vl.depth_of(np.clip(o*255,0,255).astype(np.uint8)); return r[3] if r else -1


def main():
    n = int(sys.argv[1]) if len(sys.argv)>1 else 128
    off=(1024-n)//2
    z=s3zarr.open_local("/home/forrest/paris4_2um_cube",0)
    raw=z.read_region(off,off,off,n,n,n,workers=16).astype(np.float32)/255
    raw_tex=_hp(raw).std(); raw_edge=_grad(raw).mean()
    print(f"GUARDED search on {n}^3 (floors: tex>={TEX_FLOOR} edge>={EDGE_FLOOR} struct>={STRUCT_FLOOR})\n",flush=True)

    cur=raw; cur_d=depth(cur)
    print(f"raw: depth={cur_d:.3f} (guards trivially 1.0)\n",flush=True)
    stack=[]
    for step in range(6):
        def ev(item):
            name,fn=item
            try:
                o=fn(cur); ok,tex,edge,struct=guards(o,raw,raw_tex,raw_edge); d=depth(o)
                return name,o,d,ok,tex,edge,struct
            except Exception as e:
                return name,None,-1,False,0,0,0
        with ThreadPoolExecutor(max_workers=5) as ex:
            res=list(ex.map(ev,OPS.items()))
        # rank by depth among those that PASS guards
        valid=[r for r in res if r[1] is not None and r[3]]
        valid.sort(key=lambda r:r[2],reverse=True)
        print(f"step {step+1}:")
        for name,o,d,ok,tex,edge,struct in sorted(res,key=lambda r:r[2],reverse=True):
            flag="OK " if ok else "BLOCKED"
            print(f"    {name:14s} depth={d:.3f} tex={tex:.2f} edge={edge:.2f} struct={struct:.3f}  [{flag}]")
        if not valid or valid[0][2] <= cur_d+0.003:
            print(f"  -> STOP (no guarded op improves depth past {cur_d:.3f})\n",flush=True); break
        name,o,d,ok,tex,edge,struct=valid[0]
        stack.append(name); cur=o; cur_d=d
        print(f"  -> ADD {name}: depth={cur_d:.3f} tex={tex:.2f} edge={edge:.2f}\n",flush=True)

    print("="*55)
    print(f"GUARDED BEST STACK: {' -> '.join(stack) if stack else '(raw)'}")
    ok,tex,edge,struct=guards(cur,raw,raw_tex,raw_edge)
    print(f"FINAL: depth={cur_d:.3f}  texture_kept={tex:.2f}  edge_kept={edge:.2f}  struct={struct:.3f}")
    print(f"(unguarded greedy reached depth 0.78 but at tex~0.5-0.7; this respects the signal floors)")


if __name__ == "__main__":
    main()
