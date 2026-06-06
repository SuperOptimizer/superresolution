#!/usr/bin/env python3
"""Is the dark|papyrus valley DEEPER or SHALLOWER after de-Paganin (deconv)? Deeper valley =
cleaner separation = more reliable air-cut. Measure valley depth (separability) on RAW vs
DECONVOLVED histograms, over the sample tiles.

Separability metric: valley_count / min(dark_peak, light_peak). Lower ratio = deeper valley
(better separated). Also report the classic bimodality: (peak - valley)/peak for each side."""
import os, sys, ctypes as C
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"; f32p = C.POINTER(C.c_float)


def analyze(h, label):
    hf = np.convolve(h.astype(float), np.ones(5)/5, mode="same"); hf[0] = 0
    mx = hf.max()
    # two dominant peaks
    peaks = [u for u in range(3,254) if hf[u]>=hf[u-1] and hf[u]>=hf[u+1] and hf[u]>0.05*mx]
    merged=[]
    for p in peaks:
        if not merged or p-merged[-1]>10: merged.append(p)
        elif hf[p]>hf[merged[-1]]: merged[-1]=p
    if len(merged)<2:
        print(f"{label}: only {len(merged)} mode(s) -> not bimodal"); return None
    a,b = sorted(sorted(merged,key=lambda u:hf[u],reverse=True)[:2])
    vidx = a+int(np.argmin(hf[a:b])); vval=hf[vidx]
    pdark,plight = hf[a],hf[b]
    sep = vval/min(pdark,plight)         # lower = deeper valley
    depth = 1.0 - sep                    # higher = deeper
    print(f"{label}:")
    print(f"  dark peak u8={a} (h={pdark:.0f})   light peak u8={b} (h={plight:.0f})")
    print(f"  valley u8={vidx} (h={vval:.0f})")
    print(f"  separability: valley/min(peak) = {sep:.3f}  -> valley DEPTH = {depth:.3f} (higher=cleaner)")
    return dict(dark=a, light=b, valley=vidx, depth=depth, sep=sep)


def main():
    z = s3zarr.open_local(CUBE, 0)
    read = lambda zz,yy,xx,dz,dy,dx: z.read_region(zz,yy,xx,dz,dy,dx,workers=8)
    md = fp.load_md_phys(CUBE)
    # use a bigger, representative sample so the histogram is smooth (full slabs)
    print("accumulating RAW + DECONVOLVED histograms over the cube (streaming)...")
    picked = fp.select_sample_tiles(read, z.shape, n=6, tile=128, candidates=27)
    cal = fp.calibrate_prepass(md, picked, verbose=False)
    print(f"deconv reg={cal.deconv_reg} db_scale={cal.db_scale:.3f}\n")

    hraw = np.zeros(256, np.int64); hdec = np.zeros(256, np.int64)
    # stream full 1024 in slabs for a clean histogram, deconv each slab tile-wise (128^3)
    N=1024
    for z0 in range(0, N, 128):
      for y0 in range(0, N, 256):
        for x0 in range(0, N, 256):
            u8 = read(z0,y0,x0,128,256,256)
            hraw += np.bincount(u8.ravel(), minlength=256)
            # deconv in 128^3 sub-tiles
            for yy in range(0,256,128):
              for xx in range(0,256,128):
                sub = np.ascontiguousarray(u8[:,yy:yy+128,xx:xx+128].astype(np.float32)/255)
                o=np.empty_like(sub); ph=cal.scaled_phys()
                fp.lib().fy_deconvolve(sub.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*sub.shape,C.byref(ph),C.c_double(cal.deconv_reg))
                lo,hi=cal.deconv_lo,cal.deconv_hi
                dec=np.clip((o-lo)/(hi-lo),0,1) if hi>lo else np.clip(o,0,1)
                hdec += np.bincount((dec*255).astype(np.uint8).ravel(), minlength=256)
      print(f"  slab z={z0} done", flush=True)

    print()
    r=analyze(hraw,  "RAW (Paganin'd)")
    print()
    d=analyze(hdec, "DECONVOLVED (de-Paganin'd)")
    if r and d:
        print(f"\n=== VERDICT ===")
        print(f"  valley depth RAW={r['depth']:.3f}  DECONV={d['depth']:.3f}")
        if d['depth']>r['depth']+0.02:
            print("  -> de-Paganin makes the valley DEEPER (cleaner separation -> cut AFTER deconv)")
        elif d['depth']<r['depth']-0.02:
            print("  -> de-Paganin makes the valley SHALLOWER (worse separation -> cut BEFORE deconv)")
        else:
            print("  -> valley depth ~unchanged (cut either side; before is safer operationally)")


if __name__ == "__main__":
    main()
