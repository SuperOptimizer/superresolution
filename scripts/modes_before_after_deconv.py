#!/usr/bin/env python3
"""Where should the air-cut live: BEFORE or AFTER de-Paganin (our deconv)? Fit the dark/
papyrus modes on the RAW u8 vs the DECONVOLVED output, on the same tiles, and report where
the boundary sits in each domain. Decides whether to threshold pre- or post-deconv."""
import os, sys, ctypes as C
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"; f32p = C.POINTER(C.c_float)


def fit_two(h):
    from scipy.optimize import curve_fit
    x = np.arange(len(h)); hf = h.astype(float); hf[0] = 0
    g = lambda x,a,m,s: a*np.exp(-0.5*((x-m)/s)**2)
    two = lambda x,a1,m1,s1,a2,m2,s2: g(x,a1,m1,s1)+g(x,a2,m2,s2)
    try:
        p,_ = curve_fit(two, x, hf, p0=[hf[55],55,15,hf[120],120,25], maxfev=20000)
        a1,m1,s1,a2,m2,s2 = p
        if m1>m2: a1,m1,s1,a2,m2,s2 = a2,m2,s2,a1,m1,s1
        return (m1,abs(s1)),(m2,abs(s2))
    except Exception as e:
        return None,None


def main():
    z = s3zarr.open_local(CUBE, 0)
    read = lambda zz,yy,xx,dz,dy,dx: z.read_region(zz,yy,xx,dz,dy,dx,workers=8)
    md = fp.load_md_phys(CUBE)
    picked = fp.select_sample_tiles(read, z.shape, n=6, tile=128, candidates=27)
    cal = fp.calibrate_prepass(md, picked, verbose=False)
    print(f"deconv reg={cal.deconv_reg} db_scale={cal.db_scale:.3f} rescale=[{cal.deconv_lo:.3f},{cal.deconv_hi:.3f}]\n")

    # accumulate histograms over the sample tiles, RAW vs DECONVOLVED
    hraw = np.zeros(256, np.int64); hdec = np.zeros(256, np.int64)
    for u8 in picked:
        hraw += np.bincount(u8.ravel(), minlength=256)
        a = np.ascontiguousarray(u8.astype(np.float32)/255); o = np.empty_like(a); ph = cal.scaled_phys()
        fp.lib().fy_deconvolve(a.ctypes.data_as(f32p), o.ctypes.data_as(f32p), *a.shape, C.byref(ph), C.c_double(cal.deconv_reg))
        lo,hi = cal.deconv_lo, cal.deconv_hi
        dec = np.clip((o-lo)/(hi-lo),0,1) if hi>lo else np.clip(o,0,1)
        hdec += np.bincount((dec*255).astype(np.uint8).ravel(), minlength=256)

    for name,h in [("RAW (Paganin'd, on-disk)",hraw),("DECONVOLVED (de-Paganin'd)",hdec)]:
        d,p = fit_two(h)
        if d:
            # boundary candidates in this domain
            def wmu(u8): return md['window_f32_min']+(u8/255)*(md['window_f32_max']-md['window_f32_min'])
            print(f"{name}:")
            print(f"  dark 'other' mode mu={d[0]:.0f} (phys {wmu(d[0]):.4f})  sigma={d[1]:.0f}")
            print(f"  papyrus mode      mu={p[0]:.0f} (phys {wmu(p[0]):.4f})  sigma={p[1]:.0f}")
            print(f"  conservative cut (dark_mu+0.5sig) = u8 {d[0]+0.5*d[1]:.0f}\n")
        else:
            print(f"{name}: fit failed (maybe unimodal here)\n")


if __name__ == "__main__":
    main()
