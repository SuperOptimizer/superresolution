#!/usr/bin/env python3
"""Test the 2D intensity x local-texture axis for deepening air|papyrus separation.
Partial-volume/boundary voxels sit IN the intensity valley but are FLAT (low local std);
real papyrus is TEXTURED. A score that combines intensity with local texture should separate
the two better than intensity alone. We compare valley depth of:
  (a) intensity only
  (b) intensity after light denoise
  (c) a 'papyrus score' = intensity boosted where locally textured (air stays low)
Also: iterated light denoise (2x) vs single, and median pre-filter."""
import os, sys, ctypes as C
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp
import scripts.deepen_valley as dv

CUBE = "/home/forrest/paris4_2um_cube"; f32p = C.POINTER(C.c_float); N = 1024


def main():
    from scipy.ndimage import uniform_filter, median_filter

    def texture_score(s):
        """intensity weighted by local texture: flat (air) stays low, textured (papyrus) rises."""
        m = uniform_filter(s, 5); m2 = uniform_filter(s*s, 5)
        lstd = np.sqrt(np.maximum(m2 - m*m, 0.0))
        # normalize texture to ~[0,1] gate; combine multiplicatively with a floor so bright-
        # but-flat stays, dark-textured rises. score = intensity * (0.5 + texture_gate)
        tg = np.clip(lstd / (np.median(lstd[s>0.1]) + 1e-6), 0, 2) * 0.5
        return np.clip(s * (0.6 + tg), 0, 1)

    def med(s): return median_filter(s, size=3)
    def guided2(s): return dv.guided(dv.guided(s, 0.004), 0.004)  # iterated light

    tests = [
        ("intensity only (raw)",        None),
        ("median 3",                    med),
        ("guided x2 (iterated light)",  guided2),
        ("guided then texture-score",   lambda s: texture_score(dv.guided(s, 0.006))),
        ("texture-score only",          texture_score),
    ]
    print(f"{'method':30s}  dark light valley  DEPTH")
    base = None
    for name, fn in tests:
        try:
            h = dv.hist_over_cube(fn)
            a,b,v,d = dv.depth_of(h)
            if d is None: print(f"{name:30s}  lost bimodality"); continue
            if base is None and fn is None: base = d
            tag = f"  ({d-base:+.3f})" if base is not None else ""
            print(f"{name:30s}  {a:>4} {b:>5} {v:>6}  {d:.3f}{tag}", flush=True)
        except Exception as e:
            print(f"{name:30s}  ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
