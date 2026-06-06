#!/usr/bin/env python3
"""Test PHYSICS levers to deepen the air|papyrus valley, beyond the verified guided-x2 (+44%).
All in-RAM (load 512^3 once), C kernels for the math. Levers:
  A. anisotropic coherence diffusion (edge-preserving flatten -> push valley voxels to a mode)
  B. GAT variance-stabilize -> denoise -> inverse (equalize the two modes' noise widths)
  C. per-z-slab normalization (remove beam-drift smearing that widens both modes)
  D. combinations
Metric: valley depth = 1 - valley/min(peak). Higher = better separated."""
import os, sys, ctypes as C, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"; f32p = C.POINTER(C.c_float)


def depth_of(u8vol):
    h = np.bincount(u8vol.ravel(), minlength=256).astype(float)
    # EXCLUDE the clip rails: u8=0 (masked/below-window) and u8>=254 (saturation pile-up).
    # These are export-clip artifacts, not real modes -- including them contaminates the
    # bimodal fit and the percentile-normalized metric (esp. for hard sharpeners that dump
    # voxels onto 255). Measure the separation of the INTERIOR (real) distribution only.
    h[0] = 0; h[254] = 0; h[255] = 0
    hf = np.convolve(h, np.ones(5)/5, mode="same"); mx=hf.max()
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

def guided2(v): return guided(guided(v,0.004),0.004)

def diffuse(v, s=1):
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    fp.lib().fy_coherence_diffusion_auto(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,int(s)); return o

def gat_denoise(v):
    L=fp.lib()
    nm=fp._NoiseModel(); a=np.ascontiguousarray(v,np.float32)
    L.fy_estimate_noise(a.ctypes.data_as(f32p),*a.shape,5,10.0,0.4,C.byref(nm))
    g,b = max(nm.g,1e-6), nm.b
    L.fy_gat_forward.argtypes=[f32p,f32p,C.c_size_t,C.c_double,C.c_double]
    L.fy_gat_inverse.argtypes=[f32p,f32p,C.c_size_t,C.c_double,C.c_double]
    t=np.empty_like(a); L.fy_gat_forward(a.ctypes.data_as(f32p),t.ctypes.data_as(f32p),a.size,g,b)
    # in stabilized domain noise ~1; denoise with a fixed eps, then invert
    td=guided(t/ (t.max()+1e-6), 0.01)*(t.max()+1e-6)
    o=np.empty_like(a); L.fy_gat_inverse(td.ctypes.data_as(f32p),o.ctypes.data_as(f32p),a.size,g,b)
    return np.clip(o,0,1)

def perz_norm(v):
    # remove per-z brightness drift: subtract each z-slab's median of its NONZERO papyrus body,
    # re-add the global one. Flattens beam-drift that smears the modes across z.
    out=v.copy()
    gmed=np.median(v[v>0.1])
    for z in range(v.shape[0]):
        sl=v[z]; body=sl[sl>0.1]
        if body.size>1000:
            out[z]=np.clip(sl*(gmed/(np.median(body)+1e-6)),0,1)
    return out


def main():
    z=s3zarr.open_local(CUBE,0)
    vol=z.read_region(256,256,256,512,512,512,workers=16); f01=vol.astype(np.float32)/255
    print("loaded 512^3\n")
    base=guided2(f01); rb=depth_of(np.clip(base*255,0,255).astype(np.uint8))
    print(f"{'lever':34s} dark light valley DEPTH   time")
    print(f"{'guided x2 (baseline)':34s} {rb[0]:>4} {rb[1]:>5} {rb[2]:>5} {rb[3]:.3f}")
    levers=[
        ("+ diffusion s1",        lambda: diffuse(base,1)),
        ("+ diffusion s2",        lambda: diffuse(base,2)),
        ("GAT-denoise",           lambda: gat_denoise(f01)),
        ("per-z norm + guided x2", lambda: guided2(perz_norm(f01))),
        ("per-z + guided x2 + diff1", lambda: diffuse(guided2(perz_norm(f01)),1)),
    ]
    for name,fn in levers:
        t=time.time()
        try:
            out=fn(); r=depth_of(np.clip(out*255,0,255).astype(np.uint8)); dt=time.time()-t
            if r is None: print(f"{name:34s} lost bimodality ({dt:.0f}s)"); continue
            tag="  <- DEEPER" if r[3]>rb[3]+0.01 else ("  (worse)" if r[3]<rb[3]-0.01 else "")
            print(f"{name:34s} {r[0]:>4} {r[1]:>5} {r[2]:>5} {r[3]:.3f} ({dt:.0f}s){tag}",flush=True)
        except Exception as e:
            print(f"{name:34s} ERROR {type(e).__name__}: {e}",flush=True)
    print(f"\nbaseline guided-x2 depth = {rb[3]:.3f} (raw was ~0.377)")


if __name__ == "__main__":
    main()
