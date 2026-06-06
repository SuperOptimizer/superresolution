#!/usr/bin/env python3
"""Pick the CONSERVATIVE air/other cutoff: fit the two modes (dark='other' ~55, light=
papyrus ~123) and report, per candidate cutoff, what fraction of EACH mode it removes.
We want: remove most of the dark mode, lose ~none of the papyrus mode. Streams full 1024^3."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr

CUBE = "/home/forrest/paris4_2um_cube"; N = 1024


def gauss(x, a, mu, sig): return a*np.exp(-0.5*((x-mu)/sig)**2)


def main():
    z = s3zarr.open_local(CUBE, 0)
    h = np.zeros(256, np.int64)
    for z0 in range(0, N, 128):
        slab = z.read_region(z0, 0, 0, 128, N, N, workers=12)
        h += np.bincount(slab.ravel(), minlength=256)
    h[0] = 0
    x = np.arange(256); hf = h.astype(float)

    # two-Gaussian fit (dark + light) via scipy
    try:
        from scipy.optimize import curve_fit
        def two(x, a1,m1,s1, a2,m2,s2): return gauss(x,a1,m1,s1)+gauss(x,a2,m2,s2)
        p0 = [hf[55], 55, 15, hf[123], 123, 25]
        popt,_ = curve_fit(two, x, hf, p0=p0, maxfev=20000)
        a1,m1,s1,a2,m2,s2 = popt
        if m1 > m2:  # ensure mode1=dark
            a1,m1,s1,a2,m2,s2 = a2,m2,s2,a1,m1,s1
        print(f"DARK 'other' mode:  mu={m1:.0f} sigma={s1:.0f}")
        print(f"LIGHT papyrus mode: mu={m2:.0f} sigma={s2:.0f}")
        dark = gauss(x,a1,m1,s1); light = gauss(x,a2,m2,s2)
        dark_tot, light_tot = dark.sum(), light.sum()
        print(f"\n{'cutoff':>7} | {'% DARK removed':>14} | {'% PAPYRUS lost':>15} | {'% all nonzero':>13}")
        nzt = hf.sum()
        for thr in [25,30,35,40,45,50,55,60,70,88]:
            dr = dark[:thr].sum()/dark_tot*100
            pl = light[:thr].sum()/light_tot*100
            allr = hf[:thr].sum()/nzt*100
            flag = "  <- safe" if pl < 1.0 else ("  <- eats papyrus" if pl>3 else "")
            print(f"  u8<{thr:>3} | {dr:>13.1f}% | {pl:>14.2f}% | {allr:>12.1f}%{flag}")
        # recommend: largest cutoff with papyrus-loss < 1%
        rec = max([t for t in range(15,90) if gauss(np.arange(t),a2,m2,s2).sum()/light_tot < 0.01], default=15)
        print(f"\nRECOMMEND conservative cutoff u8<{rec} "
              f"(papyrus loss <1%, removes {dark[:rec].sum()/dark_tot*100:.0f}% of the dark mode)")
    except Exception as e:
        print("fit failed:", e)


if __name__ == "__main__":
    main()
