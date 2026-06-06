#!/usr/bin/env python3
"""Try methods to DEEPEN the dark-'other'|papyrus valley (better separation -> cleaner air
masking). Measured on RAW on-disk data (deepest valley domain). For each method we report
valley depth = 1 - valley/min(peak); HIGHER = better separated. Also the 2D intensity x
local-variance axis (air is flat, papyrus is textured) which can separate beyond intensity."""
import os, sys, ctypes as C
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"; f32p = C.POINTER(C.c_float)
N = 1024


def depth_of(h):
    hf = np.convolve(h.astype(float), np.ones(5)/5, mode="same"); hf[0]=0; mx=hf.max()
    peaks=[u for u in range(3,254) if hf[u]>=hf[u-1] and hf[u]>=hf[u+1] and hf[u]>0.05*mx]
    merged=[]
    for p in peaks:
        if not merged or p-merged[-1]>10: merged.append(p)
        elif hf[p]>hf[merged[-1]]: merged[-1]=p
    if len(merged)<2: return None,None,None,None
    a,b=sorted(sorted(merged,key=lambda u:hf[u],reverse=True)[:2])
    v=a+int(np.argmin(hf[a:b]))
    return a,b,v,1.0-hf[v]/min(hf[a],hf[b])


def guided(v, eps, r=2):
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    fp.lib().fy_guided_denoise(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,r,eps); return o

def bilateral(v, ss, sr, r=3):
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    L=fp.lib(); L.fy_bilateral_denoise.argtypes=[f32p,f32p,C.c_int,C.c_int,C.c_int,C.c_double,C.c_double,C.c_int]
    L.fy_bilateral_denoise(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,ss,sr,r); return o


def hist_over_cube(fn=None):
    """accumulate u8 histogram over the full cube, optionally transforming each slab via fn(float01)->float01."""
    z=s3zarr.open_local(CUBE,0); h=np.zeros(256,np.int64)
    for z0 in range(0,N,128):
        slab=z.read_region(z0,0,0,128,N,N,workers=12)
        if fn is None:
            h+=np.bincount(slab.ravel(),minlength=256)
        else:
            # process in 128^3 tiles to bound memory
            for y0 in range(0,N,256):
                for x0 in range(0,N,256):
                    sub=slab[:,y0:y0+256,x0:x0+256].astype(np.float32)/255
                    out=fn(sub)
                    h+=np.bincount(np.clip(out*255,0,255).astype(np.uint8).ravel(),minlength=256)
    return h


def main():
    print("baseline (raw)...")
    hb=hist_over_cube()
    a,b,v,d=depth_of(hb)
    print(f"  RAW: dark={a} light={b} valley={v} DEPTH={d:.3f}\n")
    results=[("raw",d,a,b,v)]

    methods=[
        ("guided eps=0.002", lambda s: guided(s,0.002)),
        ("guided eps=0.006", lambda s: guided(s,0.006)),
        ("guided eps=0.02",  lambda s: guided(s,0.02)),
        ("guided r=3 eps=0.01", lambda s: guided(s,0.01,3)),
        ("bilateral ss2 sr0.05", lambda s: bilateral(s,2.0,0.05)),
        ("bilateral ss3 sr0.1",  lambda s: bilateral(s,3.0,0.1)),
    ]
    for name,fn in methods:
        h=hist_over_cube(fn); a,b,v,dd=depth_of(h)
        if dd is None: print(f"  {name}: lost bimodality"); continue
        tag="  <- DEEPER" if dd>d+0.01 else ("  (worse)" if dd<d-0.01 else "")
        print(f"  {name:22s}: dark={a} light={b} valley={v} DEPTH={dd:.3f}{tag}")
        results.append((name,dd,a,b,v))

    results.sort(key=lambda r:r[1],reverse=True)
    print(f"\n=== RANKED by valley depth (deeper=better separation) ===")
    for name,dd,a,b,v in results:
        print(f"  {dd:.3f}  {name}  (cut~u8 {a}+0.5*({b}-{a})/?  valley={v})")
    best=results[0]
    print(f"\nBEST: {best[0]} -> depth {best[1]:.3f} (raw was {d:.3f}, "
          f"{'+' if best[1]>d else ''}{best[1]-d:+.3f})")


if __name__ == "__main__":
    main()
