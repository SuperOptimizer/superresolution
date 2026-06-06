#!/usr/bin/env python3
"""Measure how much the export's rail-clipping (u8=255 saturation, u8=0) distorts our
deconvolution. Hypothesis: deconv amplifies high values, so near saturated rails it
invents overshoot/ringing (sharpening a flat-topped clipped plateau as if it were a real
edge). We compare deconv behaviour in NEIGHBORHOODS of clipped voxels vs clean voxels."""
import os, sys, ctypes as C
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"
f32p = C.POINTER(C.c_float)


def deconv01(u8, cal):
    a = np.ascontiguousarray(u8.astype(np.float32)/255.0); o = np.empty_like(a)
    ph = cal.scaled_phys()
    fp.lib().fy_deconvolve(a.ctypes.data_as(f32p), o.ctypes.data_as(f32p),
                           *a.shape, C.byref(ph), C.c_double(cal.deconv_reg))
    return o


def main():
    z = s3zarr.open_local(CUBE, 0)
    read = lambda zz,yy,xx,dz,dy,dx: z.read_region(zz,yy,xx,dz,dy,dx,workers=8)
    md = fp.load_md_phys(CUBE)
    picked = fp.select_sample_tiles(read, z.shape, n=6, tile=128, candidates=27)
    cal = fp.calibrate_prepass(md, picked, verbose=False)
    print(f"deconv reg={cal.deconv_reg} db_scale={cal.db_scale:.3f}\n")

    # find tiles that actually CONTAIN saturated rails
    print("scanning for tiles with clipped rails...")
    rail_tiles = []
    for (zz,yy,xx) in [(a,b,c) for a in (128,384,640,896) for b in (384,640) for c in (384,640)]:
        u8 = read(zz,yy,xx,128,128,128)
        sat = (u8 >= 254).mean(); zero_in = ((u8==0) & False)  # interior zeros rare in dense
        if sat > 1e-5:
            rail_tiles.append((zz,yy,xx,u8,sat))
    rail_tiles.sort(key=lambda r: r[4], reverse=True)
    print(f"  {len(rail_tiles)} tiles with saturation; using top {min(4,len(rail_tiles))}\n")

    # ---- core measurement: deconv overshoot near saturated rails vs elsewhere ----
    from scipy.ndimage import binary_dilation, maximum_filter
    all_near=[]; all_far=[]; sat_frac=[]
    for (zz,yy,xx,u8,sat) in rail_tiles[:4]:
        dec = deconv01(u8, cal)
        # rescale to compare on the same footing as the pipeline output
        lo,hi = cal.deconv_lo, cal.deconv_hi
        decr = (dec-lo)/(hi-lo) if hi>lo else dec
        raw = u8.astype(np.float32)/255.0
        # where did deconv push values ABOVE 1.0 (overshoot beyond the valid range)?
        overshoot = np.maximum(dec - dec.max()*0 - 1.0, 0.0)  # raw overshoot past 1.0 (pre-rescale)
        sat_mask = (u8 >= 254)
        near = binary_dilation(sat_mask, iterations=3) & ~sat_mask  # ring around saturation
        far  = ~binary_dilation(sat_mask, iterations=8)             # away from any saturation
        # metric: |deconv - raw| change magnitude (how much deconv MOVED the value)
        delta = np.abs(decr - raw)
        all_near.append(float(delta[near].mean()) if near.any() else np.nan)
        all_far.append(float(delta[far].mean()) if far.any() else np.nan)
        sat_frac.append(sat)
        print(f"  tile({zz},{yy},{xx}) sat={sat*100:.3f}%: "
              f"deconv |delta| NEAR-rail={np.nanmean(all_near[-1:]):.4f}  FAR={np.nanmean(all_far[-1:]):.4f}  "
              f"ratio={all_near[-1]/max(all_far[-1],1e-9):.2f}x")

    nn, ff = np.nanmean(all_near), np.nanmean(all_far)
    print(f"\n=== VERDICT ===")
    print(f"  mean saturation fraction: {np.mean(sat_frac)*100:.3f}% of voxels")
    print(f"  deconv |delta| near saturated rails: {nn:.4f}")
    print(f"  deconv |delta| far from rails:       {ff:.4f}")
    print(f"  RATIO near/far: {nn/max(ff,1e-9):.2f}x")
    if nn/max(ff,1e-9) > 1.5:
        print("  -> deconv IS distorting MORE near clipped rails (inventing structure). Guard worth it.")
    else:
        print("  -> deconv distortion near rails ~ everywhere else. Clip impact NEGLIGIBLE; skip the guard.")
    print(f"\n  context: only {np.mean(sat_frac)*100:.3f}% of voxels are clipped, so even a high")
    print(f"  per-voxel ratio affects a tiny volume fraction. Both numbers matter.")


if __name__ == "__main__":
    main()
