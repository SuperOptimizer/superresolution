#!/usr/bin/env python3
"""WHOLE-CHAIN validation on 128^3 cubes: run the FULL preprocess chain (deconv -> denoise ->
air-mask) and measure a SOUNDNESS PANEL, not just one metric. Tests across several regions.

Panel:
  SEPARATION:   valley depth (rail-excluded), Haralick-Shapiro J (bias-guarded)
  SIGNAL KEPT:  FRC resolution (res_frac at FSC=0.143) original vs output; papyrus-region
                texture retention; mid-band contrast retention
  MASK QUALITY: keep%, connected components, mask stability (IoU under +-3 u8 threshold jitter)
  SAFETY:       clip fraction, confident-papyrus-core removed %
This is the harness we extend as the air-mask is integrated into process_tile."""
import os, sys, ctypes as C
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib, scripts.valley_levers as vl; importlib.reload(vl)
from superres import s3zarr, fysics_pipeline as fp
from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter, label
f32p = C.POINTER(C.c_float); L = fp.lib()


def hp(v): return v - gaussian_filter(v, 1.0)

def fsc_res(vol, nbins=32):
    L.fy_fsc_self.argtypes=[f32p,C.c_int,C.c_int,C.c_int,C.c_int,f32p,f32p]
    a=np.ascontiguousarray(vol,np.float32); fr=np.zeros(nbins,np.float32); fs=np.zeros(nbins,np.float32)
    L.fy_fsc_self(a.ctypes.data_as(f32p),*a.shape,nbins,fr.ctypes.data_as(f32p),fs.ctypes.data_as(f32p))
    below=np.where(fs<0.143)[0]
    return float(fr[below[0]] if len(below) else fr[-1])

def hsj(vol):
    best=-1
    for t in range(25,140):
        a=vol[vol<t/255]; p=vol[vol>=t/255]
        if a.size<1000 or p.size<1000: continue
        J=(p.mean()-a.mean())**2/(a.var()+p.var()+1e-9)*(min(a.size,p.size)/vol.size/0.5)
        best=max(best,J)
    return best

def midband(vol):
    from numpy.fft import fftn,fftshift
    v=vol-vol.mean(); F=np.abs(fftshift(fftn(v)))**2; n=vol.shape[0]; c=n//2
    zz,yy,xx=np.indices(vol.shape)-c; r=np.sqrt(zz*zz+yy*yy+xx*xx).astype(int)
    rp=(np.bincount(r.ravel(),F.ravel())/np.maximum(np.bincount(r.ravel()),1))[:c]
    nyq=len(rp)-1; return rp[int(0.17*nyq):int(0.5*nyq)].sum()+1e-12


def main():
    regions=[(384,384,384),(640,384,384),(200,200,200),(384,640,640),(800,800,800)]
    md=fp.load_md_phys("/home/forrest/paris4_2um_cube")
    z=s3zarr.open_local("/home/forrest/paris4_2um_cube",0)
    read=lambda zz,yy,xx,dz,dy,dx: z.read_region(zz,yy,xx,dz,dy,dx,workers=12)
    picked=fp.select_sample_tiles(read,z.shape,n=6,tile=128,candidates=27)
    cal=fp.calibrate_prepass(md,picked,verbose=False)
    cal.air_thresh=fp.air_thresh_from_physics(md)
    print(f"cal: deconv reg={cal.deconv_reg} denoise eps={cal.guided_eps:.4g} air_thresh={cal.air_thresh}\n")
    print(f"{'region':>15} {'depth':>6} {'HS-J':>5} {'FRC_o':>6} {'FRC_out':>7} {'pap_tex':>7} {'contrast':>8} {'keep%':>6} {'comp':>4} {'stab_IoU':>8} {'core_lost':>9}")
    for c in regions:
        f=read(*c,128,128,128).astype(np.float32)/255
        if (f>0.02).mean()<0.3: print(f"{str(c):>15}  (air - skip)"); continue
        u8=np.clip(f*255,0,255).astype(np.uint8)
        # FULL CHAIN via process_tile (current integrated pipeline)
        out_u8=fp.process_tile(u8,cal)
        out=out_u8.astype(np.float32)/255
        # metrics
        fm=vl.depth_of(np.clip(out*255,0,255).astype(np.uint8)); depth=fm[3] if fm else 0
        j=hsj(out)
        frc_o=fsc_res(f); frc_out=fsc_res(out)
        pap=binary_erosion(f>(130/255),iterations=2)
        ptex=hp(out)[pap].std()/(hp(f)[pap].std()+1e-9) if pap.sum()>500 else 0
        contrast=midband(out)/midband(f)
        # mask from process (approx: where output != original-zeroed). use air_thresh on output
        mask=out>0.001
        lab,ncomp=label(mask)
        base=binary_fill_holes((np.clip(out*255,0,255).astype(np.uint8))>=int(cal.air_thresh*255+8))
        stabs=[]
        for dt in (-3,3):
            m=binary_fill_holes((np.clip(out*255,0,255).astype(np.uint8))>=int(cal.air_thresh*255+8+dt))
            stabs.append((base&m).sum()/((base|m).sum()+1e-9))
        stab=np.mean(stabs)
        core_lost=(out[pap]<=0.001).mean()*100 if pap.sum()>500 else 0
        print(f"{str(c):>15} {depth:>6.3f} {j:>5.1f} {frc_o:>6.3f} {frc_out:>7.3f} {ptex:>7.2f} {contrast:>8.2f} {mask.mean()*100:>5.0f}% {ncomp:>4d} {stab:>8.3f} {core_lost:>8.2f}%",flush=True)
    print("\\nwant: FRC_out~FRC_o (resolution kept), pap_tex~1 (signal kept), stab~1 (stable mask), core_lost~0")


if __name__ == "__main__":
    main()
