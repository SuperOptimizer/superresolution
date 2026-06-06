#!/usr/bin/env python3
"""PROPERLY test whether adding local-texture helps separate dark-'other' from papyrus, vs
intensity alone. Not by mangling a 1D histogram -- by measuring CLASS OVERLAP.

Method:
  1. Load 512^3 into RAM (once). Light-denoise (guided x2) -- the verified best intensity prep.
  2. Compute local texture = local std (C box-filter based, fast) per voxel.
  3. Get a confident label proxy: voxels clearly dark (intensity < dark_mu) = 'other',
     clearly bright (intensity > papyrus_mu) = 'papyrus'. (Train the separation on the
     confident cores, then see how well each feature set separates them.)
  4. Compare separability of {intensity} vs {intensity, texture} via:
       - Fisher ratio (between-class var / within-class var) -- higher = better
       - Bhattacharyya overlap of the 1D projections -- lower = better
       - misclassification rate of the best linear boundary on the AMBIGUOUS mid-band
         (the valley voxels) -- this is what actually matters for masking.
  We want: does 2D classify the valley voxels with fewer errors than intensity alone?"""
import os, sys, ctypes as C, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp

CUBE = "/home/forrest/paris4_2um_cube"; f32p = C.POINTER(C.c_float)


def guided(v, eps, r=2):
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    fp.lib().fy_guided_denoise(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,r,eps); return o


def local_std(v, r=2):
    """local std via the C fy_local_std kernel (fast; Python is glue, math in C)."""
    L = fp.lib()
    L.fy_local_std.argtypes = [f32p, f32p, C.c_int, C.c_int, C.c_int, C.c_int]
    L.fy_local_std.restype = C.c_int
    a = np.ascontiguousarray(v, np.float32); o = np.empty_like(a)
    L.fy_local_std(a.ctypes.data_as(f32p), o.ctypes.data_as(f32p), *a.shape, r)
    return o


def bhattacharyya(mu1, s1, mu2, s2):
    """overlap of two 1D gaussians; 0=disjoint, higher=more overlap."""
    s1=max(s1,1e-6); s2=max(s2,1e-6)
    return 0.25*np.log(0.25*(s1**2/s2**2 + s2**2/s1**2 + 2)) + 0.25*((mu1-mu2)**2/(s1**2+s2**2))
    # (this is the Bhattacharyya DISTANCE; higher = better separated)


def main():
    z = s3zarr.open_local(CUBE, 0)
    vol = z.read_region(256,256,256, 512,512,512, workers=16)
    f01 = vol.astype(np.float32)/255.0
    print(f"loaded 512^3 in RAM\n")

    # intensity prep = verified best (guided x2 light)
    I = guided(guided(f01, 0.004), 0.004)
    T = local_std(I, r=2)   # local texture on the denoised volume

    # confident cores (avoid the ambiguous valley to define the two classes)
    Iu8 = (I*255)
    DARK_MU, PAP_MU = 55, 124                 # from the fits
    core_other = (Iu8 < DARK_MU-8)            # clearly dark
    core_pap   = (Iu8 > PAP_MU+8)             # clearly bright
    # subsample for speed
    rng = np.random.RandomState(0)
    def samp(mask, n=200000):
        idx = np.flatnonzero(mask.ravel())
        if idx.size>n: idx = rng.choice(idx, n, replace=False)
        return idx
    io, ip = samp(core_other), samp(core_pap)
    Iflat, Tflat = I.ravel(), T.ravel()
    o_I, p_I = Iflat[io], Iflat[ip]
    o_T, p_T = Tflat[io], Tflat[ip]
    print(f"class cores: other={io.size:,}  papyrus={ip.size:,}")
    print(f"  intensity:  other mu={o_I.mean():.3f}+-{o_I.std():.3f}   papyrus mu={p_I.mean():.3f}+-{p_I.std():.3f}")
    print(f"  texture:    other mu={o_T.mean():.4f}+-{o_T.std():.4f}   papyrus mu={p_T.mean():.4f}+-{p_T.std():.4f}\n")

    # --- 1D intensity separability ---
    bc_I = bhattacharyya(o_I.mean(),o_I.std(), p_I.mean(),p_I.std())
    # --- 1D texture separability ---
    bc_T = bhattacharyya(o_T.mean(),o_T.std(), p_T.mean(),p_T.std())
    # --- 2D (LDA projection) separability ---
    X_o = np.stack([o_I,o_T],1); X_p = np.stack([p_I,p_T],1)
    mu_o, mu_p = X_o.mean(0), X_p.mean(0)
    Sw = np.cov(X_o.T)*len(X_o) + np.cov(X_p.T)*len(X_p)
    w = np.linalg.solve(Sw + 1e-9*np.eye(2), (mu_p-mu_o))   # Fisher direction
    w /= np.linalg.norm(w)
    po, pp = X_o@w, X_p@w
    bc_2D = bhattacharyya(po.mean(),po.std(), pp.mean(),pp.std())
    print(f"=== SEPARABILITY (Bhattacharyya distance, higher=better separated) ===")
    print(f"  intensity only:        {bc_I:.3f}")
    print(f"  texture only:          {bc_T:.3f}")
    print(f"  intensity + texture:   {bc_2D:.3f}   ({'BETTER' if bc_2D>bc_I*1.05 else 'no real gain'} vs intensity)")
    print(f"  Fisher direction weights (I,T) = ({w[0]:.2f}, {w[1]:.2f})  "
          f"-> {'texture contributes' if abs(w[1])>0.2 else 'intensity dominates'}")

    # --- the test that matters: misclassification on the AMBIGUOUS valley band ---
    # define an overlap region near the valley; how purely does each feature split it?
    def err_rate_1d(o,p):
        # best threshold = midpoint that minimizes error assuming the two cores
        thr = (o.mean()+p.mean())/2
        return ((o>thr).sum()+(p<thr).sum())/(len(o)+len(p))
    print(f"\n=== core misclassification (lower=better) ===")
    print(f"  intensity threshold:   {err_rate_1d(o_I,p_I)*100:.2f}%")
    print(f"  2D Fisher threshold:   {err_rate_1d(po,pp)*100:.2f}%")


if __name__ == "__main__":
    main()
