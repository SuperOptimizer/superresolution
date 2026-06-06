#!/usr/bin/env python3
"""A/B the denoise tiers on the real PHercParis4 2.4um cube:
  current   = guided filter (fy_guided_denoise) at the calibrated eps
  quality   = GAT + small-window NLM (fy_denoise_quality, self-calibrating)
Both applied AFTER the tuned deconv (the real in-pipeline order). Scored by the metric
panel across several sub-regions, with timing. Decides if the quality tier is worth wiring in."""
import os, sys, time, ctypes as C
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr
from superres import fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"
f32p = C.POINTER(C.c_float)


def deconv_only(u8, cal):
    """tuned deconv + seam-safe rescale, return float [0,1] (no denoise)."""
    a = np.ascontiguousarray(u8.astype(np.float32)/255.0); o = np.empty_like(a)
    ph = cal.scaled_phys()
    fp.lib().fy_deconvolve(a.ctypes.data_as(f32p), o.ctypes.data_as(f32p),
                           *a.shape, C.byref(ph), C.c_double(cal.deconv_reg))
    lo, hi = cal.deconv_lo, cal.deconv_hi
    return np.clip((o - lo)/(hi - lo), 0, 1) if hi > lo else o


def guided(v, eps):
    a = np.ascontiguousarray(v, np.float32); o = np.empty_like(a)
    fp.lib().fy_guided_denoise(a.ctypes.data_as(f32p), o.ctypes.data_as(f32p), *a.shape, 2, eps)
    return o


def quality(v):
    L = fp.lib()
    L.fy_denoise_quality.argtypes = [f32p, f32p, C.c_int, C.c_int, C.c_int]
    L.fy_denoise_quality.restype = C.c_int
    a = np.ascontiguousarray(v, np.float32); o = np.empty_like(a)
    L.fy_denoise_quality(a.ctypes.data_as(f32p), o.ctypes.data_as(f32p), *a.shape)
    return o


def main():
    z = s3zarr.open_local(CUBE, 0)
    read = lambda zz,yy,xx,dz,dy,dx: z.read_region(zz,yy,xx,dz,dy,dx, workers=8)
    md = fp.load_md_phys(CUBE)
    picked = fp.select_sample_tiles(read, z.shape, n=6, tile=128, candidates=27)
    cal = fp.calibrate_prepass(md, picked, verbose=False)
    print(f"CALIB deconv reg={cal.deconv_reg} guided eps={cal.guided_eps:.4g}\n")

    coords = [(a,b,c) for a in (256,512,768) for b in (256,768) for c in (256,768)]
    g_leg=[]; q_leg=[]; g_tex=[]; q_tex=[]; g_noi=[]; q_noi=[]; tg=0.0; tq=0.0
    print(f"{'region':>18} | {'guided legib/tex/noise':>26} | {'quality legib/tex/noise':>26}")
    for (zz,yy,xx) in coords:
        u8 = read(zz,yy,xx,128,128,128)
        if (u8>0).mean() < 0.5: continue
        ref = u8.astype(np.float32)/255.0
        dec = deconv_only(u8, cal)            # shared deconv base
        t=time.time(); g = guided(dec, cal.guided_eps); tg += time.time()-t
        t=time.time(); q = quality(dec);              tq += time.time()-t
        mg = fp._metric_panel(ref, g)["raw"]; mq = fp._metric_panel(ref, q)["raw"]
        g_leg.append(mg["legibility"]); q_leg.append(mq["legibility"])
        g_tex.append(mg["tex_ratio"]);  q_tex.append(mq["tex_ratio"])
        g_noi.append(mg["noise_ratio"]);q_noi.append(mq["noise_ratio"])
        print(f"({zz:>4},{yy:>4},{xx:>4}) | "
              f"{mg['legibility']:.2f}/{mg['tex_ratio']:.2f}/{mg['noise_ratio']:.2f}      | "
              f"{mq['legibility']:.2f}/{mq['tex_ratio']:.2f}/{mq['noise_ratio']:.2f}")
    n=len(g_leg)
    print(f"\n=== MEDIANS (n={n}) ===")
    print(f"  legibility: guided {np.median(g_leg):.3f}x   quality {np.median(q_leg):.3f}x")
    print(f"  texture:    guided {np.median(g_tex):.3f}    quality {np.median(q_tex):.3f}")
    print(f"  noise:      guided {np.median(g_noi):.3f}x   quality {np.median(q_noi):.3f}x")
    print(f"  time/tile:  guided {tg/n*1000:.0f}ms        quality {tq/n*1000:.0f}ms ({tq/max(tg,1e-6):.0f}x)")
    # verdict
    better_leg = np.median(q_leg) > np.median(g_leg)
    keeps_tex  = np.median(q_tex) >= np.median(g_tex) - 0.05
    print(f"\n  VERDICT: quality {'BEATS' if better_leg and keeps_tex else 'does NOT beat'} guided "
          f"(legib {'+' if better_leg else ''}{(np.median(q_leg)-np.median(g_leg)):.2f}, "
          f"tex {(np.median(q_tex)-np.median(g_tex)):+.2f})")


if __name__ == "__main__":
    main()
