#!/usr/bin/env python3
"""FAST valley-deepening test: load ONE representative subvolume into RAM once, run every
method on it in-process (no per-method disk reads / re-tiling). 512^3 = 134M voxels is plenty
for stable valley statistics. Reports valley depth (higher = better air|papyrus separation)."""
import os, sys, ctypes as C, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"; f32p = C.POINTER(C.c_float)


def depth_of(u8vol):
    h = np.bincount(u8vol.ravel(), minlength=256).astype(float)
    hf = np.convolve(h, np.ones(5)/5, mode="same"); hf[0]=0; mx=hf.max()
    peaks=[u for u in range(3,254) if hf[u]>=hf[u-1] and hf[u]>=hf[u+1] and hf[u]>0.05*mx]
    merged=[]
    for p in peaks:
        if not merged or p-merged[-1]>10: merged.append(p)
        elif hf[p]>hf[merged[-1]]: merged[-1]=p
    if len(merged)<2: return None
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


def main():
    z = s3zarr.open_local(CUBE, 0)
    t0=time.time()
    # ONE load into RAM (512^3 from the cube center -> representative, both modes present)
    vol = z.read_region(256, 256, 256, 512, 512, 512, workers=16)
    f01 = vol.astype(np.float32)/255.0
    print(f"loaded 512^3 ({vol.size:,} voxels) in {time.time()-t0:.1f}s, reused for all methods\n")

    from scipy.ndimage import uniform_filter, median_filter
    def texture_score(s):
        m=uniform_filter(s,5); m2=uniform_filter(s*s,5); lstd=np.sqrt(np.maximum(m2-m*m,0))
        tg=np.clip(lstd/(np.median(lstd[s>0.1])+1e-6),0,2)*0.5
        return np.clip(s*(0.6+tg),0,1)

    methods = [
        ("raw (intensity)",        lambda s: s),
        ("guided eps=0.006",       lambda s: guided(s,0.006)),
        ("guided x2 light(0.004)", lambda s: guided(guided(s,0.004),0.004)),
        ("median 3",               lambda s: median_filter(s,3)),
        ("bilateral ss2 sr0.05",   lambda s: bilateral(s,2.0,0.05)),
        ("texture-score",          lambda s: texture_score(s)),
        ("guided+texture-score",   lambda s: texture_score(guided(s,0.006))),
        ("median+guided+texture",  lambda s: texture_score(guided(median_filter(s,3),0.006))),
    ]
    base=None
    print(f"{'method':26s} dark light valley  DEPTH    time")
    for name,fn in methods:
        t=time.time()
        try:
            out=fn(f01); u8=np.clip(out*255,0,255).astype(np.uint8)
            r=depth_of(u8)
            dt=time.time()-t
            if r is None: print(f"{name:26s} lost bimodality   ({dt:.1f}s)"); continue
            a,b,v,d=r
            if base is None: base=d
            print(f"{name:26s} {a:>4} {b:>5} {v:>6}  {d:.3f} {d-base:+.3f} ({dt:.1f}s)", flush=True)
        except Exception as e:
            print(f"{name:26s} ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
