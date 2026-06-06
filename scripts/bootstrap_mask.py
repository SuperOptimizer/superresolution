#!/usr/bin/env python3
"""BOOTSTRAP air/papyrus separation, breaking the circularity (we can't measure against a
perfectly-masked volume because building it IS the goal).

Round 0: crude mask (intensity threshold at the histogram valley).
Each round:
  - measure SEPARATELY inside the provisional PAPYRUS region vs AIR region:
      * valley depth (separation) -- want deep
      * papyrus-region texture retention vs raw (real signal) -- must stay HIGH
      * air-region noise removed (junk) -- want gone
  - pick the strongest valley-deepening op whose PAPYRUS texture stays above the floor
  - refine the mask from the cleaned volume (re-fit the two modes, re-threshold)
  - repeat until the mask (papyrus voxel set) stops changing -> converged.

The papyrus floor is measured on the PAPYRUS REGION ONLY (not whole-volume, which air noise
inflates) -- that's the honest 'are we destroying the papyrus we care about' guard.

NOTE: the cleaned/denoised volume is used ONLY to decide the mask + measure separation. The
papyrus floor protects the signal; what you ultimately APPLY the mask to is a separate choice."""
import os, sys, ctypes as C, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib, scripts.valley_levers as vl; importlib.reload(vl)
from superres import s3zarr
import superres.fysics_pipeline as fp
from scipy.ndimage import gaussian_filter, binary_erosion
f32p = C.POINTER(C.c_float); L = fp.lib()

PAP_TEX_FLOOR = 0.80   # keep >=80% of PAPYRUS-REGION texture (the signal we care about)


def bilateral(v, ss, sr, r=3):
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    L.fy_bilateral_denoise.argtypes=[f32p,f32p,C.c_int,C.c_int,C.c_int,C.c_double,C.c_double,C.c_int]
    L.fy_bilateral_denoise(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,ss,sr,r); return o
def guided(v, eps, r=2):
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    L.fy_guided_denoise(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,r,eps); return o

OPS = {
    "bilat(2,0.03)": lambda v: bilateral(v,2.0,0.03),
    "bilat(2,0.05)": lambda v: bilateral(v,2.0,0.05),
    "bilat(3,0.08)": lambda v: bilateral(v,3.0,0.08),
    "guided(0.004)": lambda v: guided(v,0.004),
}

def _hp(v): return v - gaussian_filter(v, 1.0)

def fit_modes(u8vol):
    """return (dark_mu, light_mu, valley) excluding rails, or None."""
    h=np.bincount(u8vol.ravel(),minlength=256).astype(float); h[0]=0;h[254]=0;h[255]=0
    hf=np.convolve(h,np.ones(5)/5,mode="same"); mx=hf.max()
    peaks=[u for u in range(3,253) if hf[u]>=hf[u-1] and hf[u]>=hf[u+1] and hf[u]>0.05*mx]
    mg=[]
    for p in peaks:
        if not mg or p-mg[-1]>10: mg.append(p)
        elif hf[p]>hf[mg[-1]]: mg[-1]=p
    if len(mg)<2: return None
    a,b=sorted(sorted(mg,key=lambda u:hf[u],reverse=True)[:2]); v=a+int(np.argmin(hf[a:b]))
    depth=1-hf[v]/min(hf[a],hf[b])
    return a,b,v,depth


def main():
    n = int(sys.argv[1]) if len(sys.argv)>1 else 128
    off=(1024-n)//2
    z=s3zarr.open_local("/home/forrest/paris4_2um_cube",0)
    raw=z.read_region(off,off,off,n,n,n,workers=16).astype(np.float32)/255
    print(f"BOOTSTRAP on {n}^3 (papyrus-region texture floor = {PAP_TEX_FLOOR})\n",flush=True)

    # ---- round 0: crude mask from raw histogram valley ----
    fm=fit_modes(np.clip(raw*255,0,255).astype(np.uint8))
    if not fm: print("raw not bimodal"); return
    dark,light,valley,depth0=fm
    pap_mask=(raw*255)>=valley         # crude: above-valley = papyrus
    air_mask=((raw*255)>2)&((raw*255)<dark)
    print(f"round 0 (raw): modes dark={dark} light={light} valley={valley} depth={depth0:.3f}")
    print(f"  papyrus voxels={pap_mask.sum():,}  air voxels={air_mask.sum():,}\n")

    cur=raw
    for rnd in range(1,7):
        # erode masks a bit so we measure INTERIORS (away from ambiguous boundary)
        pap_core=binary_erosion(pap_mask,iterations=2)
        air_core=binary_erosion(air_mask,iterations=1)
        raw_pap_tex=_hp(raw)[pap_core].std()+1e-9
        raw_air_noise=_hp(raw)[air_core].std()+1e-9 if air_core.sum()>100 else 1e-9

        # try each op; measure papyrus-region texture (the guard) + valley depth
        cands=[]
        for name,fn in OPS.items():
            o=fn(cur)
            ptex=_hp(o)[pap_core].std()/raw_pap_tex
            anoise=_hp(o)[air_core].std()/raw_air_noise if air_core.sum()>100 else 0
            fmo=fit_modes(np.clip(o*255,0,255).astype(np.uint8))
            d=fmo[3] if fmo else -1
            cands.append((name,o,d,ptex,anoise,fmo))
        # rank by depth among those keeping papyrus texture above floor
        valid=[c for c in cands if c[3]>=PAP_TEX_FLOOR and c[2]>0]
        valid.sort(key=lambda c:c[2],reverse=True)
        print(f"round {rnd}:")
        for name,o,d,ptex,anoise,fmo in sorted(cands,key=lambda c:c[2],reverse=True):
            ok="OK " if ptex>=PAP_TEX_FLOOR else "BLOCK"
            print(f"    {name:14s} depth={d:.3f}  pap_tex={ptex:.2f}  air_noise={anoise:.2f}  [{ok}]")
        if not valid:
            print(f"  -> no op keeps papyrus texture >= {PAP_TEX_FLOOR}; STOP (raw mask is best we can safely do)\n")
            break
        name,o,d,ptex,anoise,fmo=valid[0]
        cur=o
        # refine the mask from the cleaned volume
        new_pap=(o*255)>=fmo[2]; new_air=((o*255)>2)&((o*255)<fmo[0])
        changed=int((new_pap!=pap_mask).sum())
        print(f"  -> ADD {name}: depth={d:.3f} pap_tex={ptex:.2f} air_noise={anoise:.2f}; mask changed {changed:,} voxels\n",flush=True)
        pap_mask,air_mask=new_pap,new_air
        if changed < 0.005*pap_mask.size:   # mask converged
            print(f"  -> mask CONVERGED (changed {changed:,} < 0.5%). Done.\n"); break

    print("="*55)
    fmf=fit_modes(np.clip(cur*255,0,255).astype(np.uint8))
    pc=binary_erosion(pap_mask,iterations=2)
    print(f"FINAL: valley depth={fmf[3]:.3f} (raw {depth0:.3f})  papyrus-region texture kept="
          f"{_hp(cur)[pc].std()/(_hp(raw)[pc].std()+1e-9):.2f}")
    print(f"  mask: {pap_mask.sum():,} papyrus voxels, threshold u8>={fmf[2]}")


if __name__ == "__main__":
    main()
