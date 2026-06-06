"""Fast 128^3 validation metrics for the preprocess-chain experiments. All histogram- or
C-kernel-based where possible so we iterate quickly. Import from experiment scripts."""
import ctypes as C
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import fysics_pipeline as fp
f32p = C.POINTER(C.c_float); L = fp.lib()


_i64p = C.POINTER(C.c_long); _ip = C.POINTER(C.c_int)
L.fy_haralick_shapiro.argtypes = [_i64p, C.c_int, C.c_int, C.c_int, _ip]
L.fy_haralick_shapiro.restype = C.c_double
L.fy_valley_depth.argtypes = [_i64p, _ip, _ip, _ip]
L.fy_valley_depth.restype = C.c_double


def _hist(vol):
    u8 = np.clip(vol*255, 0, 255).astype(np.uint8)
    return np.bincount(u8.ravel(), minlength=256).astype(np.int64)


def hsj(vol, lo=25, hi=140, min_count=1000):
    """Haralick-Shapiro bias-guarded Fisher J (C kernel). Returns (best_J, best_threshold).
    Python only builds the histogram; the threshold sweep + stats are in C (O(256))."""
    h = np.ascontiguousarray(_hist(vol))
    bt = C.c_int(0)
    j = L.fy_haralick_shapiro(h.ctypes.data_as(_i64p), lo, hi, min_count, C.byref(bt))
    return float(j), int(bt.value)


def valley_depth(vol):
    """Bimodal valley depth, rails excluded (C kernel). Returns (dark,light,valley,depth) or None."""
    h = np.ascontiguousarray(_hist(vol))
    dk, lt, vl = C.c_int(0), C.c_int(0), C.c_int(0)
    d = L.fy_valley_depth(h.ctypes.data_as(_i64p), C.byref(dk), C.byref(lt), C.byref(vl))
    if d < 0: return None
    return int(dk.value), int(lt.value), int(vl.value), float(d)


def fsc_res(vol, nbins=32, threshold=0.143):
    """FRC resolution as fraction of Nyquist at FSC=threshold (higher=finer). C kernel.
    Signature: fy_fsc_self(vol, nz,ny,nx, nbins, threshold, *res_frac, *freqs, *fsc)."""
    L.fy_fsc_self.argtypes=[f32p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_float,f32p,f32p,f32p]
    a=np.ascontiguousarray(vol,np.float32)
    rf=np.zeros(1,np.float32); fr=np.zeros(nbins,np.float32); fs=np.zeros(nbins,np.float32)
    L.fy_fsc_self(a.ctypes.data_as(f32p),*a.shape,nbins,C.c_float(threshold),
                  rf.ctypes.data_as(f32p),fr.ctypes.data_as(f32p),fs.ctypes.data_as(f32p))
    return float(rf[0])


def local_std(v, r=2):
    """texture = local std via C fy_local_std (fast)."""
    L.fy_local_std.argtypes=[f32p,f32p,C.c_int,C.c_int,C.c_int,C.c_int]
    a=np.ascontiguousarray(v,np.float32); o=np.empty_like(a)
    L.fy_local_std(a.ctypes.data_as(f32p),o.ctypes.data_as(f32p),*a.shape,r)
    return o


def hp_texture(vol, region=None):
    """high-pass texture energy (std), optionally inside a region mask. Uses C local_std as
    a fast high-pass proxy (local std ~ local high-freq energy)."""
    t = local_std(vol, 2)
    return float(t[region].std() if region is not None else t.std())


def fast_cal(cube_dir, region=(448,448,448), force=False):
    """Fast calibration for 128^3 experiments: calibrate on the SINGLE current 128^3 cube
    (13s, gives the correct reg/eps) and CACHE the resolved params to disk so subsequent
    experiment runs load instantly. Pass force=True to recompute. Returns a Calibration."""
    import json, os
    from superres import s3zarr
    cache_path = os.path.join(cube_dir, "calcache.json")
    md = fp.load_md_phys(cube_dir)
    z = s3zarr.open_local(cube_dir, 0)
    f = z.read_region(*region, 128, 128, 128, workers=8)
    if force or not os.path.exists(cache_path):
        cal = fp.calibrate_prepass(md, [f], verbose=False)   # calibrate on THIS cube only
        cal.air_thresh = air_thresh_phys(md)
        json.dump(dict(deconv_reg=float(cal.deconv_reg), guided_eps=float(cal.guided_eps),
                       deconv_lo=float(cal.deconv_lo), deconv_hi=float(cal.deconv_hi),
                       do_deconv=bool(cal.do_deconv), do_denoise=bool(cal.do_denoise),
                       db_scale=float(cal.db_scale), halo=int(cal.halo),
                       air_thresh=float(cal.air_thresh or 0), denoise_radius=int(cal.denoise_radius)),
                  open(cache_path, "w"))
        return cal
    # load cached params into a fresh Calibration
    c = json.load(open(cache_path))
    cal = fp.calibrate(md, [f.astype(np.float32)/255], auto_deltabeta=True)
    cal.deconv_reg=c["deconv_reg"]; cal.guided_eps=c["guided_eps"]
    cal.deconv_lo=c["deconv_lo"]; cal.deconv_hi=c["deconv_hi"]
    cal.do_deconv=c["do_deconv"]; cal.do_denoise=c["do_denoise"]
    cal.air_thresh=c["air_thresh"] or None; cal.denoise_radius=c["denoise_radius"]
    return cal


def air_thresh_phys(md):
    return fp.air_thresh_from_physics(md)


def midband_ratio(out, ref):
    """mid-band (0.17-0.5 Nyquist) power ratio out/ref -- contrast retention."""
    from numpy.fft import fftn, fftshift
    def mb(vol):
        v=vol-vol.mean(); F=np.abs(fftshift(fftn(v)))**2; n=vol.shape[0]; c=n//2
        zz,yy,xx=np.indices(vol.shape)-c; rr=np.sqrt(zz*zz+yy*yy+xx*xx).astype(int)
        rp=(np.bincount(rr.ravel(),F.ravel())/np.maximum(np.bincount(rr.ravel()),1))[:c]
        nyq=len(rp)-1; return rp[int(0.17*nyq):int(0.5*nyq)].sum()+1e-12
    return mb(out)/mb(ref)
